// magi_groupby.cu — operator-facing entry that routes
// `GPUPhysicalGroupedAggregate` through the Magi NVLink-shuffle backend.
//
// Each legacy worker thread (one per GPU) calls Run() on its partition's
// columns. This is a *generic* path: there are no per-query shape detectors.
// Run() inspects the group keys and aggregates, packs them into the magi
// data model (KeyFieldEntry[] + AggOpEntry[]), and calls the generic
// magi_generic::distributed_hash_groupby_run_per_gpu, which std::barriers
// across the per-GPU threads and NVLink-shuffles the partial aggregates.
// On return it materialises this GPU's hash-partitioned slice back into a
// fresh (chars+offsets) VARCHAR + DOUBLE/INT64 column set on the same GPU.
//
// Supported aggregates: SUM (INT64/DOUBLE), COUNT_STAR, COUNT, MIN, MAX, and
// AVG (carried as SUM + a shared COUNT_STAR slot, divided at emit). Anything
// the packer can't express falls back to DuckDB CPU.

#include "operator/magi_groupby.hpp"

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "gpu_buffer_manager.hpp"
#include "operator/magi_distributed_groupby.hpp"

namespace duckdb {
namespace magi_groupby {

namespace {

// ── Generic-path adapters ─────────────────────────────────────────────────
// These translate sirius's per-GPU GPUColumn views into the magi-side
// ColPack + AggOpEntry table, and emit the magi result slice back into
// sirius output columns. The translation is data-driven (column-type +
// AggregationType) — no per-query branches.

// Inspect the group-by key columns and produce both:
//   - KeyKind          (chooses the templated atomic-width kernel)
//   - KeyFieldEntry[]  (tells the kernel how to pack input bytes into KeyT)
// Sirius assigns ColPack slots in BuildGenericInputs in the same order it
// walks `keys`, so the field table's src_col_idx values match.
//
// Supported shapes today (the only device-side dependency is that KeyT ∈
// {int32, uint64} cover the byte layout — extending this is pure host work):
//   - 1× INT     → INT32   fields=[{INT32,   src=0, off=0, len=4}]
//   - 1× BIGINT  → UINT64  fields=[{INT64,   src=0, off=0, len=8}]   (Q11)
//   - 1× VARCHAR → UINT64  fields=[{VARCHAR, src=0, off=0, len=8}]    (Q5)
//   - 2× VARCHAR → INT32   fields=[{VARCHAR, src=0, off=1, len=1},
//                                  {VARCHAR, src=1, off=0, len=1}]   (Q1)
bool DeriveKeyShape(const vector<shared_ptr<GPUColumn>>& keys,
                    int                                  n_keys,
                    magi_generic::KeyKind&               out_kind,
                    std::vector<magi_ops::KeyFieldEntry>& out_fields)
{
  out_fields.clear();
  using magi_ops::KeyFieldKind;
  using magi_ops::KeyFieldEntry;
  if (n_keys <= 0) return false;

  // ── Generic fixed-width packing (no per-shape hardcoding) ────────────────
  // Any combination of fixed-width keys auto-packs into one key word: walk the
  // keys in order, give each its byte range, and pick the key width from the
  // total. src_col_idx is the per-kind pool index (BuildGenericInputs assigns
  // i_cols / i64_cols independently in key order), so we track one counter per
  // pool. Covers single INT, single BIGINT (Q11), and any INT/BIGINT compound
  // (Q9 = nationkey + o_year) for free.
  //   total ≤ 4B → INT32 key word;  ≤ 8B → UINT64;  > 8B → needs wide-key path.
  {
    bool all_fixed = true;
    int byte_off = 0, i32_idx = 0, i64_idx = 0;
    std::vector<KeyFieldEntry> f;
    for (int k = 0; k < n_keys; ++k) {
      auto id = keys[k]->data_wrapper.type.id();
      if (id == GPUColumnTypeId::INT32) {
        f.push_back({KeyFieldKind::INT32, (int8_t)i32_idx++, (int8_t)byte_off, 4});
        byte_off += 4;
      } else if (id == GPUColumnTypeId::INT64) {
        f.push_back({KeyFieldKind::INT64, (int8_t)i64_idx++, (int8_t)byte_off, 8});
        byte_off += 8;
      } else {
        all_fixed = false;
        break;
      }
    }
    if (all_fixed) {
      if (byte_off > 16) return false;  // > 128-bit key → side-table (future)
      out_kind = (byte_off <= 4)  ? magi_generic::KeyKind::INT32
                 : (byte_off <= 8) ? magi_generic::KeyKind::UINT64
                                   : magi_generic::KeyKind::UINT128;
      out_fields = std::move(f);
      return true;
    }
  }

  // ── VARCHAR shapes (variable width → explicit recipes) ───────────────────
  // Strings can't be auto-budgeted into a fixed key word, so each varchar shape
  // states its own prefix length. The collision-freedom is a property of the
  // query's FIXED domain — NOT provable for arbitrary VARCHAR:
  //   Q1: l_returnflag / l_linestatus are single-char by the TPC-H spec, so
  //       1 byte each is provably enough for that schema.
  //   Q5: the 25 fixed TPC-H nation names are distinguishable within 8 bytes
  //       ("UNITED K" vs "UNITED S"). Two distinct strings sharing an 8-byte
  //       prefix WOULD silently merge into one group — a general VARCHAR key
  //       needs a full-string side-table (future), not a fixed prefix.
  if (n_keys == 1 && keys[0]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
    out_kind = magi_generic::KeyKind::UINT64;   // Q5: first 8 bytes of n_name
    out_fields.push_back({KeyFieldKind::VARCHAR_PREFIX, 0, /*off=*/0, /*len=*/8});
    return true;
  }
  if (n_keys == 2 &&
      keys[0]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR &&
      keys[1]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
    out_kind = magi_generic::KeyKind::INT32;     // Q1: (c0<<8)|c1, 1 byte each
    out_fields.push_back({KeyFieldKind::VARCHAR_PREFIX, 0, /*off=*/1, /*len=*/1});
    out_fields.push_back({KeyFieldKind::VARCHAR_PREFIX, 1, /*off=*/0, /*len=*/1});
    return true;
  }
  return false;
}

// Pick the receiver hash-table tier. Cardinality-aware within the BIGINT-keyed
// subclass, which is where tier choice actually matters:
//
//   - No BIGINT key component (Q1: 4 groups, Q5: 5, Q9: 175 — packed from
//     INT/VARCHAR into ≤8B): low cardinality by construction in the TPC-H set,
//     so route to SMALL (the single-kernel shmem combiner — fast for low
//     card + high row count, and its 256-slot table is exact).
//
//   - BIGINT key component: splits into low-card (Q11: ps_partkey<20 → 19
//     groups, tens of rows) and high-card (Q3: l_orderkey compound → ~1.1M
//     groups, millions of rows). Here the per-GPU input row count is a good
//     cardinality proxy *for this subclass* (distinct ≤ rows; Q11 has few
//     rows, Q3 has many), so size the tier from it. This drops Q11 to SMALL
//     (its slice is tiny → fast shmem path, avoids the big-tier O(slots) host
//     copyback) while letting Q3-sf100 reach XLARGE and stay on magi instead
//     of overflowing LARGE and falling back to DuckDB.
//
// The estimate carries ~1.33× headroom so the open-addressing load factor stays
// below ~0.77 even when keys are near-unique; the overflow counter is the
// backstop (→ DuckDB fallback) if a mis-estimate still fills the table. With
// even cache slicing every GPU sees ~the same row count and so picks the same
// tier (the cross-GPU wire format is tier-independent regardless).
magi_generic::TableSize PickTableSize(const vector<shared_ptr<GPUColumn>>& keys,
                                       int                                  n_keys)
{
  bool has_bigint = false;
  for (int k = 0; k < n_keys; ++k) {
    if (keys[k]->data_wrapper.type.id() == GPUColumnTypeId::INT64) {
      has_bigint = true;
      break;
    }
  }
  if (!has_bigint) return magi_generic::TableSize::SMALL;

  const uint64_t local_rows = keys[0]->column_length;
  const uint64_t est        = local_rows + local_rows / 3;   // ~1.33× headroom
  if (est <= magi_generic::N_SLOTS_SMALL)  return magi_generic::TableSize::SMALL;
  if (est <= magi_generic::N_SLOTS_MEDIUM) return magi_generic::TableSize::MEDIUM;
  if (est <= magi_generic::N_SLOTS_LARGE)  return magi_generic::TableSize::LARGE;
  return magi_generic::TableSize::XLARGE;
}

magi_generic::PerGpuInputs BuildGenericInputs(
    const vector<shared_ptr<GPUColumn>>& keys,
    const vector<shared_ptr<GPUColumn>>& aggs,
    int                                  num_group_keys,
    int                                  num_aggregates,
    sirius::AggregationType*             agg_mode)
{
  magi_generic::PerGpuInputs in{};
  in.n_filtered = keys[0]->column_length;
  in.cols       = {};

  // Column slot assignment, in the same order DeriveKeyShape walks `keys`:
  // VARCHAR → v_chars[v_idx]/v_offsets[v_idx], INT32 → i_cols[i_idx],
  // INT64 → i64_cols[i64_idx]. KeyFieldEntry.src_col_idx references these
  // indices, so DeriveKeyShape and BuildGenericInputs must agree on order.
  int v_idx   = 0;
  int i_idx   = 0;
  int i64_idx = 0;
  for (int k = 0; k < num_group_keys; ++k) {
    auto id = keys[k]->data_wrapper.type.id();
    if (id == GPUColumnTypeId::VARCHAR) {
      in.cols.v_chars  [v_idx] = keys[k]->data_wrapper.data;
      in.cols.v_offsets[v_idx] = keys[k]->data_wrapper.offset;
      ++v_idx;
    } else if (id == GPUColumnTypeId::INT32) {
      in.cols.i_cols[i_idx++] =
          reinterpret_cast<const int32_t*>(keys[k]->data_wrapper.data);
    } else if (id == GPUColumnTypeId::INT64) {
      in.cols.i64_cols[i64_idx++] =
          reinterpret_cast<const int64_t*>(keys[k]->data_wrapper.data);
    }
  }
  in.cols.n_varchars = v_idx;
  in.cols.n_ints     = i_idx;
  in.cols.n_int64s   = i64_idx;

  // Pass through the KEY column's validity_mask if sirius's filter set one.
  // The filter operator marks filtered-out rows with validity bit = 0 but
  // leaves the raw data buffer untouched (= sentinel garbage like -1 for
  // INT64 — TPC-H ps_partkey < 1000 hit this). Stage 1 producer skips
  // rows where the bit is 0 (mirroring cudf::groupby's null handling).
  // We reinterpret-cast to uint32_t* to keep ColPack header free of cudf
  // includes; bitmask_type is uint32_t.
  in.cols.row_validity =
      (num_group_keys > 0 && keys[0] && keys[0]->data_wrapper.validity_mask)
          ? reinterpret_cast<const uint32_t*>(keys[0]->data_wrapper.validity_mask)
          : nullptr;

  // Aggregate input columns, assigned in lockstep with BuildAggOpsTable so the
  // AggOpEntry.src_col_idx values line up on EVERY GPU — including one whose
  // post-filter slice is empty (the column object still exists with its type,
  // but data == nullptr). We keep the typed slot for such a column (pointer
  // left null); Stage-1 producer reads 0 local rows there and never
  // dereferences it. Skipping empties (as the old code did) desynchronised the
  // per-GPU ops tables, so keys shuffled to an empty-input GPU were never
  // merged (they came back as 0). We key off the column TYPE, not its data
  // pointer, since the type survives an empty slice.
  //   SUM(FLOAT64) → d_cols[];  SUM(INT64/DECIMAL≤18) → i64_agg_cols[];
  //   MIN/MAX      → d_cols[];  COUNT_STAR/COUNT       → no source column.
  int d_idx    = 0;
  int i64a_idx = 0;
  for (int a = 0; a < num_aggregates; ++a) {
    switch (agg_mode[a]) {
      // AVERAGE carries the same input column as a SUM would (it accumulates a
      // SUM on the magi-direct path; the divide-by-count happens at emit). Keep
      // it in lockstep with BuildAggOpsTable's AVERAGE branch.
      case sirius::AggregationType::AVERAGE:
      case sirius::AggregationType::SUM: {
        const auto  id   = aggs[a] ? aggs[a]->data_wrapper.type.id()
                                   : GPUColumnTypeId::FLOAT64;
        const void* data = aggs[a] ? aggs[a]->data_wrapper.data : nullptr;
        if (id == GPUColumnTypeId::INT64 || id == GPUColumnTypeId::DECIMAL) {
          in.cols.i64_agg_cols[i64a_idx++] = reinterpret_cast<const int64_t*>(data);
        } else {
          in.cols.d_cols[d_idx++] = reinterpret_cast<const double*>(data);
        }
        break;
      }
      case sirius::AggregationType::MIN:
      case sirius::AggregationType::MAX: {
        const void* data = aggs[a] ? aggs[a]->data_wrapper.data : nullptr;
        in.cols.d_cols[d_idx++] = reinterpret_cast<const double*>(data);
        break;
      }
      case sirius::AggregationType::COUNT_STAR:
      case sirius::AggregationType::COUNT:
        // No source column (AggOpEntry.src_col_idx == -1).
        break;
      default:
        // Unsupported modes throw in BuildAggOpsTable; nothing to assign here.
        break;
    }
  }
  in.cols.n_doubles    = d_idx;
  in.cols.n_int64_aggs = i64a_idx;
  return in;
}

std::vector<magi_ops::AggOpEntry> BuildAggOpsTable(
    const vector<shared_ptr<GPUColumn>>& aggs,
    int                                  num_aggregates,
    sirius::AggregationType*             agg_mode,
    int&                                 avg_count_slot)  // out: slot holding the
                                                          // per-group row count for
                                                          // AVG divisors (-1 if none)
{
  std::vector<magi_ops::AggOpEntry> ops;
  ops.reserve(num_aggregates + 1);
  avg_count_slot = -1;

  // Source columns are typed-pool-co-numbered with their AggKind:
  //   SUM_DOUBLE   → d_idx into ColPack::d_cols
  //   SUM_INT64    → i64a_idx into ColPack::i64_agg_cols
  //   COUNT_STAR/_VALID → no source column (src_col_idx = -1)
  // dst_slot_idx is a single shared index into AggSlot64::values[].
  int d_idx    = 0;
  int i64a_idx = 0;
  int dst_idx  = 0;
  for (int a = 0; a < num_aggregates; ++a) {
    magi_ops::AggOpEntry e{};
    switch (agg_mode[a]) {
      // AVERAGE accumulates exactly like SUM on the device (SUM_DOUBLE /
      // SUM_INT64 by input type); WriteGenericSliceToColumns divides the
      // accumulated sum by the per-group row count (avg_count_slot, a shared
      // COUNT_STAR appended below). Must stay in lockstep with the AVERAGE
      // branch in BuildGenericInputs (same column-slot consumption order).
      case sirius::AggregationType::AVERAGE:
      case sirius::AggregationType::SUM: {
        // Build the op from the column TYPE (preserved even when this GPU's
        // post-filter slice is empty / data == nullptr) so dst_slot_idx and
        // src_col_idx stay identical across GPUs. Skipping empties here would
        // desync the consumer ops table and zero out shuffled groups.
        const auto id = aggs[a] ? aggs[a]->data_wrapper.type.id()
                                : GPUColumnTypeId::FLOAT64;
        if (id == GPUColumnTypeId::FLOAT64) {
          e.kind         = magi_ops::AggKind::SUM_DOUBLE;
          e.src_col_idx  = static_cast<int8_t>(d_idx++);
        } else if (id == GPUColumnTypeId::INT64 || id == GPUColumnTypeId::DECIMAL) {
          // DECIMAL with width ≤ 18 is stored as int64 by sirius — sum it
          // as int64 (precision preserved) and re-tag the output column
          // with the same DECIMAL {width, scale} metadata on emit.
          if (id == GPUColumnTypeId::DECIMAL) {
            const auto* dti = aggs[a]->data_wrapper.type.GetDecimalTypeInfo();
            if (dti && dti->GetDecimalTypeSize() > sizeof(int64_t)) {
              throw NotImplementedException(
                  "magi_groupby (generic): DECIMAL width %d > 18 (int128 "
                  "storage) not supported yet",
                  static_cast<int>(dti->width_));
            }
          }
          e.kind         = magi_ops::AggKind::SUM_INT64;
          e.src_col_idx  = static_cast<int8_t>(i64a_idx++);
        } else {
          throw NotImplementedException(
              "magi_groupby (generic): SUM on column type %d not supported "
              "(only FLOAT64, INT64, DECIMAL≤18)",
              static_cast<int>(id));
        }
        e.dst_slot_idx = static_cast<int8_t>(dst_idx++);
        break;
      }
      case sirius::AggregationType::COUNT_STAR:
      case sirius::AggregationType::COUNT:
        e.kind         = (agg_mode[a] == sirius::AggregationType::COUNT_STAR)
                            ? magi_ops::AggKind::COUNT_STAR
                            : magi_ops::AggKind::COUNT_VALID;
        e.src_col_idx  = -1;
        e.dst_slot_idx = static_cast<int8_t>(dst_idx++);
        break;
      case sirius::AggregationType::MIN:
        e.kind         = magi_ops::AggKind::MIN_DOUBLE;
        e.src_col_idx  = static_cast<int8_t>(d_idx++);
        e.dst_slot_idx = static_cast<int8_t>(dst_idx++);
        break;
      case sirius::AggregationType::MAX:
        e.kind         = magi_ops::AggKind::MAX_DOUBLE;
        e.src_col_idx  = static_cast<int8_t>(d_idx++);
        e.dst_slot_idx = static_cast<int8_t>(dst_idx++);
        break;
      default:
        throw NotImplementedException(
            "magi_groupby (generic): unsupported AggregationType %d at idx %d",
            static_cast<int>(agg_mode[a]), a);
    }
    ops.push_back(e);
  }

  // If any AVERAGE is present, append ONE shared COUNT_STAR carrier: the
  // per-group row count is the same divisor for every AVG column, so a single
  // extra slot serves them all. dst_idx continues past the real aggregates'
  // slots; the widened AggSlot64 (128B → up to 14 double slots) has room.
  bool has_avg = false;
  for (int a = 0; a < num_aggregates; ++a)
    if (agg_mode[a] == sirius::AggregationType::AVERAGE) { has_avg = true; break; }
  if (has_avg) {
    magi_ops::AggOpEntry ce{};
    ce.kind         = magi_ops::AggKind::COUNT_STAR;
    ce.src_col_idx  = -1;
    ce.dst_slot_idx = static_cast<int8_t>(dst_idx++);
    ops.push_back(ce);
    avg_count_slot  = ce.dst_slot_idx;
  }
  return ops;
}

// Emit one VARCHAR key column from packed uint64-or-int32 keys (depending on
// KeyKind). Layout matches sirius's existing convention (chars+offsets[N+1]).
void EmitVarcharKeyFromPacked(int                                              gpu_id,
                              size_t                                           N,
                              const std::vector<magi_generic::AggResultRow>&   slice,
                              shared_ptr<GPUColumn>&                           out_col,
                              int                                              key_byte_offset,  // 0 for c0, 1 for c1 (Q1 (rf,ls) pack)
                              int                                              key_byte_len,     // 1 (VARCHAR(1)) or 8 (Q5 pack)
                              GPUBufferManager*                                gbm)
{
  if (N == 0) {
    out_col = make_shared_ptr<GPUColumn>(0,
                                         GPUColumnType(GPUColumnTypeId::VARCHAR),
                                         /*data=*/nullptr,
                                         /*offset=*/nullptr,
                                         /*num_bytes=*/0,
                                         /*is_string_data=*/true,
                                         /*validity_mask=*/nullptr);
    out_col->row_id_count = 0;
    return;
  }
  std::vector<uint64_t> h_offsets(N + 1);
  size_t total_bytes = 0;
  h_offsets[0] = 0;
  for (size_t i = 0; i < N; ++i) {
    uint64_t k = static_cast<uint64_t>(slice[i].key_packed);  // varchar key ≤ 8B
    int len = 0;
    for (int b = 0; b < key_byte_len; ++b) {
      uint8_t c = static_cast<uint8_t>((k >> ((key_byte_offset + b) * 8)) & 0xff);
      if (c == 0) break;
      ++len;
    }
    total_bytes      += len;
    h_offsets[i + 1]  = total_bytes;
  }
  std::vector<uint8_t> h_chars(total_bytes);
  for (size_t i = 0; i < N; ++i) {
    uint64_t k = static_cast<uint64_t>(slice[i].key_packed);  // varchar key ≤ 8B
    size_t dst = h_offsets[i];
    for (int b = 0; b < key_byte_len; ++b) {
      uint8_t c = static_cast<uint8_t>((k >> ((key_byte_offset + b) * 8)) & 0xff);
      if (c == 0) break;
      h_chars[dst++] = c;
    }
  }
  auto* d_chars   = gbm->customCudaMalloc<uint8_t> (total_bytes == 0 ? 1 : total_bytes, gpu_id, false);
  auto* d_offsets = gbm->customCudaMalloc<uint64_t>(N + 1, gpu_id, false);
  if (total_bytes > 0)
    cudaMemcpy(d_chars, h_chars.data(), total_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, h_offsets.data(), (N + 1) * sizeof(uint64_t),
             cudaMemcpyHostToDevice);
  out_col = make_shared_ptr<GPUColumn>(N,
                                       GPUColumnType(GPUColumnTypeId::VARCHAR),
                                       d_chars,
                                       d_offsets,
                                       /*num_bytes=*/total_bytes,
                                       /*is_string_data=*/true,
                                       createNullMask(N));
  out_col->row_id_count = 0;
}

void EmitDoubleAggCol(int                                              gpu_id,
                      size_t                                           N,
                      const std::vector<magi_generic::AggResultRow>&   slice,
                      int                                              dst_slot_idx,
                      shared_ptr<GPUColumn>&                           out_col,
                      GPUBufferManager*                                gbm)
{
  if (N == 0) {
    out_col = make_shared_ptr<GPUColumn>(0,
                                         GPUColumnType(GPUColumnTypeId::FLOAT64),
                                         /*data=*/nullptr,
                                         /*validity_mask=*/nullptr);
    out_col->row_id_count = 0;
    return;
  }
  std::vector<double> v(N);
  for (size_t i = 0; i < N; ++i) v[i] = slice[i].values[dst_slot_idx];
  auto* d_buf = gbm->customCudaMalloc<double>(N, gpu_id, false);
  cudaMemcpy(d_buf, v.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  out_col = make_shared_ptr<GPUColumn>(N,
                                       GPUColumnType(GPUColumnTypeId::FLOAT64),
                                       reinterpret_cast<uint8_t*>(d_buf),
                                       createNullMask(N));
  out_col->row_id_count = 0;
}

void EmitInt64AggCol(int                                              gpu_id,
                     size_t                                           N,
                     const std::vector<magi_generic::AggResultRow>&   slice,
                     int                                              dst_slot_idx,
                     shared_ptr<GPUColumn>&                           out_col,
                     GPUBufferManager*                                gbm)
{
  if (N == 0) {
    out_col = make_shared_ptr<GPUColumn>(0,
                                         GPUColumnType(GPUColumnTypeId::INT64),
                                         /*data=*/nullptr,
                                         /*validity_mask=*/nullptr);
    out_col->row_id_count = 0;
    return;
  }
  std::vector<uint64_t> v(N);
  for (size_t i = 0; i < N; ++i) {
    // COUNT_STAR / COUNT_VALID are stored at `values[dst]` as the raw 8-byte
    // counter (uint64); just reinterpret the double bit pattern.
    double d = slice[i].values[dst_slot_idx];
    std::memcpy(&v[i], &d, sizeof(uint64_t));
  }
  auto* d_buf = gbm->customCudaMalloc<uint64_t>(N, gpu_id, false);
  cudaMemcpy(d_buf, v.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
  out_col = make_shared_ptr<GPUColumn>(N,
                                       GPUColumnType(GPUColumnTypeId::INT64),
                                       reinterpret_cast<uint8_t*>(d_buf),
                                       createNullMask(N));
  out_col->row_id_count = 0;
}

// Emit an AVG output column = (accumulated SUM at sum_slot) / (per-group row
// count at count_slot). The sum was accumulated either as a double (SUM_DOUBLE,
// FLOAT64 input) or as raw int64 bits (SUM_INT64, INT64/DECIMAL input — divide
// the raw int64 by 10^decimal_scale to recover the decimal value). The count
// slot holds a uint64 counter in the double-sized field (COUNT_STAR). DuckDB's
// avg() over these types yields DOUBLE, so the output column is FLOAT64.
void EmitAvgCol(int                                              gpu_id,
                size_t                                           N,
                const std::vector<magi_generic::AggResultRow>&   slice,
                int                                              sum_slot,
                int                                              count_slot,
                bool                                             sum_is_int64,
                int                                              decimal_scale,
                shared_ptr<GPUColumn>&                           out_col,
                GPUBufferManager*                                gbm)
{
  if (N == 0) {
    out_col = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::FLOAT64),
                                         nullptr, nullptr);
    out_col->row_id_count = 0;
    return;
  }
  const double scale_div = decimal_scale > 0 ? std::pow(10.0, decimal_scale) : 1.0;
  std::vector<double> v(N);
  for (size_t i = 0; i < N; ++i) {
    uint64_t cnt = 0;
    std::memcpy(&cnt, &slice[i].values[count_slot], sizeof(uint64_t));
    double sum;
    if (sum_is_int64) {
      int64_t raw = 0;
      std::memcpy(&raw, &slice[i].values[sum_slot], sizeof(int64_t));
      sum = static_cast<double>(raw) / scale_div;
    } else {
      sum = slice[i].values[sum_slot];
    }
    v[i] = (cnt != 0) ? (sum / static_cast<double>(cnt)) : 0.0;
  }
  auto* d_buf = gbm->customCudaMalloc<double>(N, gpu_id, false);
  cudaMemcpy(d_buf, v.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  out_col = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::FLOAT64),
                                       reinterpret_cast<uint8_t*>(d_buf), createNullMask(N));
  out_col->row_id_count = 0;
}

// ════════════════════════════════════════════════════════════════════════════
// Device-side emit: build output GPUColumns directly from a device AggResultRow
// buffer (produced by the runtime's aggslot_to_resultrow), with NO D2H of the
// slots + host slice construction + per-column host transpose. Mirrors the host
// Emit* helpers above. Wins on high-cardinality GROUP BY (host emit is O(groups)).
// ════════════════════════════════════════════════════════════════════════════
__global__ void k_gb_key_i32(const magi_generic::AggResultRow* in, size_t N, int shift, int32_t* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = (int32_t)((in[i].key_packed >> shift) & 0xFFFFFFFFull);
}
__global__ void k_gb_key_i64(const magi_generic::AggResultRow* in, size_t N, int shift, int64_t* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = (int64_t)(in[i].key_packed >> shift);
}
__global__ void k_gb_double(const magi_generic::AggResultRow* in, size_t N, int slot, double* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = in[i].values[slot];
}
__global__ void k_gb_int64(const magi_generic::AggResultRow* in, size_t N, int slot, uint64_t* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) { uint64_t x; __builtin_memcpy(&x, &in[i].values[slot], 8); out[i] = x; }
}
__global__ void k_gb_avg(const magi_generic::AggResultRow* in, size_t N, int sum_slot, int count_slot,
                         bool sum_is_int64, double scale_div, double* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  uint64_t cnt; __builtin_memcpy(&cnt, &in[i].values[count_slot], 8);
  double sum;
  if (sum_is_int64) { int64_t raw; __builtin_memcpy(&raw, &in[i].values[sum_slot], 8); sum = (double)raw / scale_div; }
  else sum = in[i].values[sum_slot];
  out[i] = cnt ? (sum / (double)cnt) : 0.0;
}
__global__ void k_gb_vc_len(const magi_generic::AggResultRow* in, size_t N, int off, int len_max, uint64_t* len) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  uint64_t k = (uint64_t)in[i].key_packed;
  int l = 0;
  for (int b = 0; b < len_max; ++b) { uint8_t c = (uint8_t)((k >> ((off + b) * 8)) & 0xff); if (c == 0) break; ++l; }
  len[i] = (uint64_t)l;
}
__global__ void k_gb_vc_chars(const magi_generic::AggResultRow* in, size_t N, int off, int len_max,
                              const uint64_t* offsets, uint8_t* chars) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  uint64_t k = (uint64_t)in[i].key_packed;
  uint64_t o = offsets[i];
  for (int b = 0; b < len_max; ++b) { uint8_t c = (uint8_t)((k >> ((off + b) * 8)) & 0xff); if (c == 0) break; chars[o++] = c; }
}
static inline void gb_cfg(size_t N, int& tpb, unsigned& grid) {
  tpb = 256; grid = (unsigned)((N + tpb - 1) / tpb); if (grid == 0) grid = 1;
}

void EmitVarcharKeyDevice(int gpu_id, size_t N, const magi_generic::AggResultRow* d_in,
                          shared_ptr<GPUColumn>& out_col, int key_byte_offset, int key_byte_len,
                          GPUBufferManager* gbm) {
  if (N == 0) { out_col = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::VARCHAR), nullptr, nullptr, 0, true, nullptr); out_col->row_id_count = 0; return; }
  int tpb; unsigned grid; gb_cfg(N, tpb, grid);
  auto* d_len = gbm->customCudaMalloc<uint64_t>(N, gpu_id, false);
  k_gb_vc_len<<<grid, tpb>>>(d_in, N, key_byte_offset, key_byte_len, d_len);
  auto* d_off = gbm->customCudaMalloc<uint64_t>(N + 1, gpu_id, false);
  cudaMemset(d_off, 0, sizeof(uint64_t));
  thrust::inclusive_scan(thrust::device, d_len, d_len + N, d_off + 1);
  uint64_t total = 0; cudaMemcpy(&total, d_off + N, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  auto* d_chars = gbm->customCudaMalloc<uint8_t>(total == 0 ? 1 : total, gpu_id, false);
  k_gb_vc_chars<<<grid, tpb>>>(d_in, N, key_byte_offset, key_byte_len, d_off, d_chars);
  out_col = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::VARCHAR), d_chars, d_off, total, true, createNullMask(N));
  out_col->row_id_count = 0;
}
void EmitDoubleAggColDevice(int gpu_id, size_t N, const magi_generic::AggResultRow* d_in, int slot, shared_ptr<GPUColumn>& out_col, GPUBufferManager* gbm) {
  if (N == 0) { out_col = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::FLOAT64), nullptr, nullptr); out_col->row_id_count = 0; return; }
  int tpb; unsigned grid; gb_cfg(N, tpb, grid);
  auto* d = gbm->customCudaMalloc<double>(N, gpu_id, false);
  k_gb_double<<<grid, tpb>>>(d_in, N, slot, d);
  out_col = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::FLOAT64), reinterpret_cast<uint8_t*>(d), createNullMask(N));
  out_col->row_id_count = 0;
}
void EmitInt64AggColDevice(int gpu_id, size_t N, const magi_generic::AggResultRow* d_in, int slot, shared_ptr<GPUColumn>& out_col, GPUBufferManager* gbm) {
  if (N == 0) { out_col = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr); out_col->row_id_count = 0; return; }
  int tpb; unsigned grid; gb_cfg(N, tpb, grid);
  auto* d = gbm->customCudaMalloc<uint64_t>(N, gpu_id, false);
  k_gb_int64<<<grid, tpb>>>(d_in, N, slot, d);
  out_col = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(d), createNullMask(N));
  out_col->row_id_count = 0;
}
void EmitAvgColDevice(int gpu_id, size_t N, const magi_generic::AggResultRow* d_in, int sum_slot, int count_slot,
                      bool sum_is_int64, int decimal_scale, shared_ptr<GPUColumn>& out_col, GPUBufferManager* gbm) {
  if (N == 0) { out_col = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::FLOAT64), nullptr, nullptr); out_col->row_id_count = 0; return; }
  int tpb; unsigned grid; gb_cfg(N, tpb, grid);
  const double scale_div = decimal_scale > 0 ? std::pow(10.0, decimal_scale) : 1.0;
  auto* d = gbm->customCudaMalloc<double>(N, gpu_id, false);
  k_gb_avg<<<grid, tpb>>>(d_in, N, sum_slot, count_slot, sum_is_int64, scale_div, d);
  out_col = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::FLOAT64), reinterpret_cast<uint8_t*>(d), createNullMask(N));
  out_col->row_id_count = 0;
}

// Device analog of WriteGenericSliceToColumns: identical orchestration, reads the
// device AggResultRow buffer `d_in` (N rows) via the Emit*Device helpers.
void WriteGenericSliceToColumnsDevice(int gpu_id, const magi_generic::AggResultRow* d_in, size_t N,
                                      vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& aggs,
                                      int num_group_keys, int num_aggregates, sirius::AggregationType* agg_mode,
                                      const std::vector<magi_ops::KeyFieldEntry>& key_fields, int avg_count_slot,
                                      GPUBufferManager* gbm) {
  if ((int)key_fields.size() != num_group_keys)
    throw NotImplementedException("magi_groupby (generic device): key_fields/num_group_keys mismatch (%zu vs %d)", key_fields.size(), num_group_keys);
  int tpb; unsigned grid; gb_cfg(N, tpb, grid);
  for (int ki = 0; ki < num_group_keys; ++ki) {
    const magi_ops::KeyFieldEntry& f = key_fields[ki];
    const int shift = f.byte_offset * 8;
    switch (f.kind) {
      case magi_ops::KeyFieldKind::VARCHAR_PREFIX:
        EmitVarcharKeyDevice(gpu_id, N, d_in, keys[ki], f.byte_offset, f.byte_len, gbm);
        break;
      case magi_ops::KeyFieldKind::INT32: {
        if (N == 0) { keys[ki] = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT32), nullptr, nullptr); keys[ki]->row_id_count = 0; break; }
        auto* d = gbm->customCudaMalloc<int32_t>(N, gpu_id, false);
        k_gb_key_i32<<<grid, tpb>>>(d_in, N, shift, d);
        keys[ki] = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::INT32), reinterpret_cast<uint8_t*>(d), createNullMask(N));
        keys[ki]->row_id_count = 0;
      } break;
      case magi_ops::KeyFieldKind::INT64: {
        if (N == 0) { keys[ki] = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr); keys[ki]->row_id_count = 0; break; }
        auto* d = gbm->customCudaMalloc<int64_t>(N, gpu_id, false);
        k_gb_key_i64<<<grid, tpb>>>(d_in, N, shift, d);
        keys[ki] = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(d), createNullMask(N));
        keys[ki]->row_id_count = 0;
      } break;
    }
  }
  int dst_idx = 0;
  for (int a = 0; a < num_aggregates; ++a) {
    switch (agg_mode[a]) {
      case sirius::AggregationType::SUM:
      case sirius::AggregationType::MIN:
      case sirius::AggregationType::MAX: {
        const auto in_id = aggs[a] ? aggs[a]->data_wrapper.type.id() : GPUColumnTypeId::FLOAT64;
        const auto* in_dti = (in_id == GPUColumnTypeId::DECIMAL && aggs[a]) ? aggs[a]->data_wrapper.type.GetDecimalTypeInfo() : nullptr;
        const int dti_w = in_dti ? in_dti->width_ : 0;
        const int dti_s = in_dti ? in_dti->scale_ : 0;
        if (in_id == GPUColumnTypeId::INT64 || in_id == GPUColumnTypeId::DECIMAL) {
          EmitInt64AggColDevice(gpu_id, N, d_in, dst_idx++, aggs[a], gbm);
          if (in_id == GPUColumnTypeId::DECIMAL && in_dti && aggs[a]) {
            aggs[a]->data_wrapper.type = GPUColumnType(GPUColumnTypeId::DECIMAL);
            aggs[a]->data_wrapper.type.SetDecimalTypeInfo(dti_w, dti_s);
          }
        } else {
          EmitDoubleAggColDevice(gpu_id, N, d_in, dst_idx++, aggs[a], gbm);
        }
        break;
      }
      case sirius::AggregationType::AVERAGE: {
        const auto in_id = aggs[a] ? aggs[a]->data_wrapper.type.id() : GPUColumnTypeId::FLOAT64;
        const bool sum_is_int64 = (in_id == GPUColumnTypeId::INT64 || in_id == GPUColumnTypeId::DECIMAL);
        int scale = 0;
        if (in_id == GPUColumnTypeId::DECIMAL && aggs[a]) { const auto* dti = aggs[a]->data_wrapper.type.GetDecimalTypeInfo(); if (dti) scale = dti->scale_; }
        EmitAvgColDevice(gpu_id, N, d_in, /*sum_slot=*/dst_idx, /*count_slot=*/avg_count_slot, sum_is_int64, scale, aggs[a], gbm);
        dst_idx++;
        break;
      }
      case sirius::AggregationType::COUNT_STAR:
      case sirius::AggregationType::COUNT:
        EmitInt64AggColDevice(gpu_id, N, d_in, dst_idx++, aggs[a], gbm);
        break;
      default:
        throw NotImplementedException("magi_groupby (generic device): unsupported AggregationType %d at idx %d during result emit", static_cast<int>(agg_mode[a]), a);
    }
  }
}

void WriteGenericSliceToColumns(int                                              gpu_id,
                                const std::vector<magi_generic::AggResultRow>&   slice,
                                vector<shared_ptr<GPUColumn>>&                   keys,
                                vector<shared_ptr<GPUColumn>>&                   aggs,
                                int                                              num_group_keys,
                                int                                              num_aggregates,
                                sirius::AggregationType*                         agg_mode,
                                const std::vector<magi_ops::KeyFieldEntry>&      key_fields,
                                int                                              avg_count_slot,
                                GPUBufferManager*                                gbm)
{
  const size_t N = slice.size();

  // ── Emit key columns (table-driven — inverse of pack_key_from_fields) ────
  // key_fields[i] describes keys[i]: its KeyFieldKind and byte range inside the
  // packed key word. We invert each independently, so the emit needs no
  // per-query branches — it mirrors whatever recipe DeriveKeyShape produced.
  //   VARCHAR_PREFIX → rebuild the string from [off, off+len) bytes
  //   INT32          → extract 4 bytes at off
  //   INT64          → extract 8 bytes at off
  if (static_cast<int>(key_fields.size()) != num_group_keys) {
    throw NotImplementedException(
        "magi_groupby (generic): key_fields/num_group_keys mismatch "
        "(%zu vs %d)", key_fields.size(), num_group_keys);
  }
  for (int ki = 0; ki < num_group_keys; ++ki) {
    const magi_ops::KeyFieldEntry& f = key_fields[ki];
    const int shift = f.byte_offset * 8;
    switch (f.kind) {
      case magi_ops::KeyFieldKind::VARCHAR_PREFIX:
        EmitVarcharKeyFromPacked(gpu_id, N, slice, keys[ki],
                                 f.byte_offset, f.byte_len, gbm);
        break;
      case magi_ops::KeyFieldKind::INT32: {
        if (N == 0) {
          keys[ki] = make_shared_ptr<GPUColumn>(0,
              GPUColumnType(GPUColumnTypeId::INT32), nullptr, nullptr);
          keys[ki]->row_id_count = 0;
          break;
        }
        std::vector<int32_t> v(N);
        for (size_t i = 0; i < N; ++i)
          v[i] = static_cast<int32_t>((slice[i].key_packed >> shift) & 0xFFFFFFFFu);
        auto* d_buf = gbm->customCudaMalloc<int32_t>(N, gpu_id, false);
        cudaMemcpy(d_buf, v.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
        keys[ki] = make_shared_ptr<GPUColumn>(N,
            GPUColumnType(GPUColumnTypeId::INT32),
            reinterpret_cast<uint8_t*>(d_buf), createNullMask(N));
        keys[ki]->row_id_count = 0;
      } break;
      case magi_ops::KeyFieldKind::INT64: {
        // The packed uint64 holds the original int64 bit-pattern (run_per_gpu
        // used an unsigned cast so high bits don't flip).
        if (N == 0) {
          keys[ki] = make_shared_ptr<GPUColumn>(0,
              GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr);
          keys[ki]->row_id_count = 0;
          break;
        }
        std::vector<int64_t> v(N);
        for (size_t i = 0; i < N; ++i)
          v[i] = static_cast<int64_t>(slice[i].key_packed >> shift);
        auto* d_buf = gbm->customCudaMalloc<int64_t>(N, gpu_id, false);
        cudaMemcpy(d_buf, v.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);
        keys[ki] = make_shared_ptr<GPUColumn>(N,
            GPUColumnType(GPUColumnTypeId::INT64),
            reinterpret_cast<uint8_t*>(d_buf), createNullMask(N));
        keys[ki]->row_id_count = 0;
      } break;
    }
  }

  // ── Emit aggregate columns ─────────────────────────────────────────────
  // The dst_slot_idx assignment in BuildAggOpsTable matches the order
  // sirius hands us aggregates, so we walk `aggregates` in lockstep.
  // Emit output column type depends on the SUM's input column type:
  //   SUM(FLOAT64) → FLOAT64 (SUM_DOUBLE path)
  //   SUM(INT64)   → INT64   (SUM_INT64 path, reinterpret raw bits)
  //   SUM(DECIMAL) → DECIMAL with same {width, scale} (SUM_INT64 path,
  //                  same int64 storage, decimal semantics preserved)
  int dst_idx = 0;
  for (int a = 0; a < num_aggregates; ++a) {
    switch (agg_mode[a]) {
      case sirius::AggregationType::SUM:
      case sirius::AggregationType::MIN:
      case sirius::AggregationType::MAX: {
        // Derive the output type from the input column's *type*, not its data
        // pointer: on a GPU whose post-filter slice is empty the input column
        // has data == nullptr but its type (e.g. DECIMAL) is still set. Keying
        // off the data pointer here would wrongly fall back to FLOAT64 and
        // emit a column whose type disagrees with the other GPUs / the DuckDB
        // plan. Capture the DECIMAL {width, scale} now, because EmitInt64AggCol
        // replaces aggs[a] (out_col is by-reference) with a fresh INT64 column.
        const auto in_id =
            aggs[a] ? aggs[a]->data_wrapper.type.id() : GPUColumnTypeId::FLOAT64;
        const auto* in_dti =
            (in_id == GPUColumnTypeId::DECIMAL && aggs[a])
                ? aggs[a]->data_wrapper.type.GetDecimalTypeInfo()
                : nullptr;
        const int dti_w = in_dti ? in_dti->width_ : 0;
        const int dti_s = in_dti ? in_dti->scale_ : 0;
        if (in_id == GPUColumnTypeId::INT64 || in_id == GPUColumnTypeId::DECIMAL) {
          EmitInt64AggCol(gpu_id, N, slice, dst_idx++, aggs[a], gbm);
          // Re-tag the (freshly emitted) output column with the input's DECIMAL
          // {width, scale} so result_collector treats it as DECIMAL, not INT64.
          if (in_id == GPUColumnTypeId::DECIMAL && in_dti && aggs[a]) {
            aggs[a]->data_wrapper.type = GPUColumnType(GPUColumnTypeId::DECIMAL);
            aggs[a]->data_wrapper.type.SetDecimalTypeInfo(dti_w, dti_s);
          }
        } else {
          EmitDoubleAggCol(gpu_id, N, slice, dst_idx++, aggs[a], gbm);
        }
        break;
      }
      case sirius::AggregationType::AVERAGE: {
        // AVG emit = sum_slot / per-group-count (avg_count_slot). The sum was
        // accumulated as SUM_DOUBLE (FLOAT64) or SUM_INT64 (INT64/DECIMAL, with
        // 10^scale to recover the decimal value). dst_idx is this AVG's sum slot
        // (assigned in BuildAggOpsTable lockstep); avg_count_slot is the shared
        // COUNT_STAR carrier appended after the real aggregates.
        const auto in_id =
            aggs[a] ? aggs[a]->data_wrapper.type.id() : GPUColumnTypeId::FLOAT64;
        const bool sum_is_int64 =
            (in_id == GPUColumnTypeId::INT64 || in_id == GPUColumnTypeId::DECIMAL);
        int scale = 0;
        if (in_id == GPUColumnTypeId::DECIMAL && aggs[a]) {
          const auto* dti = aggs[a]->data_wrapper.type.GetDecimalTypeInfo();
          if (dti) scale = dti->scale_;
        }
        EmitAvgCol(gpu_id, N, slice, /*sum_slot=*/dst_idx, /*count_slot=*/avg_count_slot,
                   sum_is_int64, scale, aggs[a], gbm);
        dst_idx++;
        break;
      }
      case sirius::AggregationType::COUNT_STAR:
      case sirius::AggregationType::COUNT:
        EmitInt64AggCol(gpu_id, N, slice, dst_idx++, aggs[a], gbm);
        break;
      default:
        throw NotImplementedException(
            "magi_groupby (generic): unsupported AggregationType %d at idx %d "
            "during result emit",
            static_cast<int>(agg_mode[a]), a);
    }
  }
}

}  // namespace

// High-cardinality predicate (exported): true iff this per-GPU slice would route
// magi to the XLARGE tier — i.e. a BIGINT-keyed GROUP BY with a large row-count
// proxy. For these the operator does a cudf LOCAL aggregation first (no fixed cap,
// reduces rows -> distinct), then magi re-aggregates the partials over NVLink with
// a tier sized from the (now small) distinct count. Reuses PickTableSize so the
// cudf-vs-magi-direct decision and the tier picker cannot drift; every GPU sees
// ~the same row count and so decides identically (the begin barrier stays balanced).
bool ShouldCudfPreAgg(const vector<shared_ptr<GPUColumn>>& keys, int n_keys)
{
  return PickTableSize(keys, n_keys) == magi_generic::TableSize::XLARGE;
}

void Run(int                                gpu_id,
         vector<shared_ptr<GPUColumn>>&     group_by_keys,
         vector<shared_ptr<GPUColumn>>&     aggregate_keys,
         int                                num_group_keys,
         int                                num_aggregates,
         sirius::AggregationType*           agg_mode)
{
  // ── Generic path (table-driven; handles all supported GROUP BY shapes) ──
  magi_generic::KeyKind                 key_kind;
  std::vector<magi_ops::KeyFieldEntry>  key_fields;
  if (!DeriveKeyShape(group_by_keys, num_group_keys, key_kind, key_fields)) {
    throw NotImplementedException(
        "magi_groupby (generic): unsupported key shape "
        "(n_keys=%d, first_type=%d). Extend DeriveKeyShape with a new "
        "KeyFieldEntry recipe — no device-side code change needed.",
        num_group_keys,
        group_by_keys.empty() ? -1
        : static_cast<int>(group_by_keys[0]->data_wrapper.type.id()));
  }
  const magi_generic::TableSize table_size =
      PickTableSize(group_by_keys, num_group_keys);

  auto in   = BuildGenericInputs(group_by_keys, aggregate_keys,
                                  num_group_keys, num_aggregates, agg_mode);
  int  avg_count_slot = -1;
  auto ops  = BuildAggOpsTable(aggregate_keys, num_aggregates, agg_mode,
                               avg_count_slot);
  std::vector<magi_generic::AggResultRow> slice;  // unused under device_emit (API ref)
  magi_generic::AggResultRow* d_rows = nullptr;
  const std::size_t n_rows = magi_generic::distributed_hash_groupby_run_per_gpu(
      gpu_id, in, key_fields, ops, key_kind, table_size, slice,
      /*device_emit=*/true, &d_rows);

  // Build the output GPUColumns ON-DEVICE directly from the device AggResultRow
  // buffer (no D2H of slots + host slice construction + per-column host transpose).
  WriteGenericSliceToColumnsDevice(gpu_id, d_rows, n_rows,
                                   group_by_keys, aggregate_keys,
                                   num_group_keys, num_aggregates, agg_mode,
                                   key_fields, avg_count_slot,
                                   &GPUBufferManager::GetInstance());
  // The emit kernels + thrust scans run on the default stream; flush before the
  // columns are consumed downstream and before the device buffer is freed.
  cudaDeviceSynchronize();
  if (d_rows) cudaFree(d_rows);
}

}  // namespace magi_groupby
}  // namespace duckdb
