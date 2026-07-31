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

#include <array>
#include <barrier>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
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
  // pool. Covers single INT, single BIGINT (Q11), and any INT/BIGINT/DATE
  // compound (Q9 = nationkey + o_year; Q3 = l_orderkey + o_orderdate +
  // o_shippriority → 16B) for free. DATE is physically an int32 day count, so
  // it packs through the INT32 pool unchanged — the writeback re-types the
  // emitted column from the input column, so no device work is DATE-aware.
  //   total ≤ 4B → INT32 key word;  ≤ 8B → UINT64;  ≤ 16B → UINT128.
  {
    bool all_fixed = true;
    int byte_off = 0, i32_idx = 0, i64_idx = 0;
    std::vector<KeyFieldEntry> f;
    for (int k = 0; k < n_keys; ++k) {
      auto id = keys[k]->data_wrapper.type.id();
      if (id == GPUColumnTypeId::INT32 || id == GPUColumnTypeId::DATE) {
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
  if (est <= magi_generic::N_SLOTS_XLARGE) return magi_generic::TableSize::XLARGE;
  if (est <= magi_generic::N_SLOTS_XXLARGE || !magi_generic::GroupbyXxxlEnabled()) {
    // Beyond the arena's largest tier Run()'s cross-GPU-consistent guard
    // throws (fallback) — returning XXLARGE for an over-XXLARGE estimate is
    // deliberate: the guard compares real exchanged counts, not this local
    // estimate, so borderline queries still get their chance.
    return magi_generic::TableSize::XXLARGE;
  }
  return magi_generic::TableSize::XXXLARGE;
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
    } else if (id == GPUColumnTypeId::INT32 || id == GPUColumnTypeId::DATE) {
      // DATE is an int32 day count — same byte layout, same pool.
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
        // Same type-based routing as SUM: INT64/DECIMAL≤18 must be read as
        // int64 (MIN_INT64/MAX_INT64) — the double path would reinterpret
        // the int64 bit pattern as a denormal double. Lockstep with
        // BuildAggOpsTable's MIN/MAX branch.
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
      case sirius::AggregationType::MAX: {
        // Kind by input type, mirroring the SUM branch (and BuildGenericInputs'
        // MIN/MAX routing): FLOAT64 → MIN/MAX_DOUBLE; INT64/DECIMAL≤18 →
        // MIN/MAX_INT64 (raw int64 order == decimal order at fixed scale; the
        // emit re-tags the output with the input's {width, scale}).
        const bool is_min = agg_mode[a] == sirius::AggregationType::MIN;
        const auto id     = aggs[a] ? aggs[a]->data_wrapper.type.id()
                                    : GPUColumnTypeId::FLOAT64;
        if (id == GPUColumnTypeId::FLOAT64) {
          e.kind        = is_min ? magi_ops::AggKind::MIN_DOUBLE
                                 : magi_ops::AggKind::MAX_DOUBLE;
          e.src_col_idx = static_cast<int8_t>(d_idx++);
        } else if (id == GPUColumnTypeId::INT64 || id == GPUColumnTypeId::DECIMAL) {
          if (id == GPUColumnTypeId::DECIMAL) {
            const auto* dti = aggs[a]->data_wrapper.type.GetDecimalTypeInfo();
            if (dti && dti->GetDecimalTypeSize() > sizeof(int64_t)) {
              throw NotImplementedException(
                  "magi_groupby (generic): MIN/MAX on DECIMAL width %d > 18 "
                  "(int128 storage) not supported yet",
                  static_cast<int>(dti->width_));
            }
          }
          e.kind        = is_min ? magi_ops::AggKind::MIN_INT64
                                 : magi_ops::AggKind::MAX_INT64;
          e.src_col_idx = static_cast<int8_t>(i64a_idx++);
        } else {
          throw NotImplementedException(
              "magi_groupby (generic): MIN/MAX on column type %d not supported "
              "(only FLOAT64, INT64, DECIMAL≤18)",
              static_cast<int>(id));
        }
        e.dst_slot_idx = static_cast<int8_t>(dst_idx++);
        break;
      }
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
                      GPUBufferManager*                                gbm,
                      int                                              dec = 0)
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
  for (size_t i = 0; i < N; ++i) {
    if (dec == 0) { v[i] = slice[i].values[dst_slot_idx]; continue; }
    // MIN/MAX slots hold the order-preserving encoding — decode.
    uint64_t u;
    std::memcpy(&u, &slice[i].values[dst_slot_idx], sizeof(u));
    v[i] = magi_ops::dec_f64_asc(dec == 1 ? ~u : u);
  }
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
                     GPUBufferManager*                                gbm,
                     int                                              dec = 0)
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
    // counter (uint64); just reinterpret the double bit pattern. MIN/MAX
    // (dec != 0) hold the order-preserving encoding — decode it.
    double d = slice[i].values[dst_slot_idx];
    std::memcpy(&v[i], &d, sizeof(uint64_t));
    if (dec != 0) v[i] = (uint64_t)magi_ops::dec_i64_asc(dec == 1 ? ~v[i] : v[i]);
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
// `dec`: 0 = raw slot word; 1/2 = the slot holds the order-preserving MIN/MAX
// encoding (agg_slot.cuh) — decode it (MIN stores the bit-inverted encoding).
__global__ void k_gb_double(const magi_generic::AggResultRow* in, size_t N, int slot, int dec, double* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  if (dec == 0) { out[i] = in[i].values[slot]; return; }
  uint64_t u; __builtin_memcpy(&u, &in[i].values[slot], 8);
  out[i] = magi_ops::dec_f64_asc(dec == 1 ? ~u : u);
}
__global__ void k_gb_int64(const magi_generic::AggResultRow* in, size_t N, int slot, int dec, uint64_t* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  uint64_t x; __builtin_memcpy(&x, &in[i].values[slot], 8);
  if (dec != 0) x = (uint64_t)magi_ops::dec_i64_asc(dec == 1 ? ~x : x);
  out[i] = x;
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
void EmitDoubleAggColDevice(int gpu_id, size_t N, const magi_generic::AggResultRow* d_in, int slot, shared_ptr<GPUColumn>& out_col, GPUBufferManager* gbm, int dec = 0) {
  if (N == 0) { out_col = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::FLOAT64), nullptr, nullptr); out_col->row_id_count = 0; return; }
  int tpb; unsigned grid; gb_cfg(N, tpb, grid);
  auto* d = gbm->customCudaMalloc<double>(N, gpu_id, false);
  k_gb_double<<<grid, tpb>>>(d_in, N, slot, dec, d);
  out_col = make_shared_ptr<GPUColumn>(N, GPUColumnType(GPUColumnTypeId::FLOAT64), reinterpret_cast<uint8_t*>(d), createNullMask(N));
  out_col->row_id_count = 0;
}
void EmitInt64AggColDevice(int gpu_id, size_t N, const magi_generic::AggResultRow* d_in, int slot, shared_ptr<GPUColumn>& out_col, GPUBufferManager* gbm, int dec = 0) {
  if (N == 0) { out_col = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr); out_col->row_id_count = 0; return; }
  int tpb; unsigned grid; gb_cfg(N, tpb, grid);
  auto* d = gbm->customCudaMalloc<uint64_t>(N, gpu_id, false);
  k_gb_int64<<<grid, tpb>>>(d_in, N, slot, dec, d);
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
    // 4-byte fields pack DATE and INT32 identically; re-type the emitted
    // column from the input column so DATE keys come back as DATE.
    const auto i32_out_id = (keys[ki] && keys[ki]->data_wrapper.type.id() == GPUColumnTypeId::DATE)
                              ? GPUColumnTypeId::DATE
                              : GPUColumnTypeId::INT32;
    switch (f.kind) {
      case magi_ops::KeyFieldKind::VARCHAR_PREFIX:
        EmitVarcharKeyDevice(gpu_id, N, d_in, keys[ki], f.byte_offset, f.byte_len, gbm);
        break;
      case magi_ops::KeyFieldKind::INT32: {
        if (N == 0) { keys[ki] = make_shared_ptr<GPUColumn>(0, GPUColumnType(i32_out_id), nullptr, nullptr); keys[ki]->row_id_count = 0; break; }
        auto* d = gbm->customCudaMalloc<int32_t>(N, gpu_id, false);
        k_gb_key_i32<<<grid, tpb>>>(d_in, N, shift, d);
        keys[ki] = make_shared_ptr<GPUColumn>(N, GPUColumnType(i32_out_id), reinterpret_cast<uint8_t*>(d), createNullMask(N));
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
        const int mm_dec =
            agg_mode[a] == sirius::AggregationType::MIN   ? 1
            : agg_mode[a] == sirius::AggregationType::MAX ? 2
                                                          : 0;
        if (in_id == GPUColumnTypeId::INT64 || in_id == GPUColumnTypeId::DECIMAL) {
          EmitInt64AggColDevice(gpu_id, N, d_in, dst_idx++, aggs[a], gbm, mm_dec);
          if (in_id == GPUColumnTypeId::DECIMAL && in_dti && aggs[a]) {
            aggs[a]->data_wrapper.type = GPUColumnType(GPUColumnTypeId::DECIMAL);
            aggs[a]->data_wrapper.type.SetDecimalTypeInfo(dti_w, dti_s);
          }
        } else {
          EmitDoubleAggColDevice(gpu_id, N, d_in, dst_idx++, aggs[a], gbm, mm_dec);
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
    // 4-byte fields pack DATE and INT32 identically; re-type the emitted
    // column from the input column so DATE keys come back as DATE.
    const auto i32_out_id = (keys[ki] && keys[ki]->data_wrapper.type.id() == GPUColumnTypeId::DATE)
                              ? GPUColumnTypeId::DATE
                              : GPUColumnTypeId::INT32;
    switch (f.kind) {
      case magi_ops::KeyFieldKind::VARCHAR_PREFIX:
        EmitVarcharKeyFromPacked(gpu_id, N, slice, keys[ki],
                                 f.byte_offset, f.byte_len, gbm);
        break;
      case magi_ops::KeyFieldKind::INT32: {
        if (N == 0) {
          keys[ki] = make_shared_ptr<GPUColumn>(0,
              GPUColumnType(i32_out_id), nullptr, nullptr);
          keys[ki]->row_id_count = 0;
          break;
        }
        std::vector<int32_t> v(N);
        for (size_t i = 0; i < N; ++i)
          v[i] = static_cast<int32_t>((slice[i].key_packed >> shift) & 0xFFFFFFFFu);
        auto* d_buf = gbm->customCudaMalloc<int32_t>(N, gpu_id, false);
        cudaMemcpy(d_buf, v.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
        keys[ki] = make_shared_ptr<GPUColumn>(N,
            GPUColumnType(i32_out_id),
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
        const int mm_dec =
            agg_mode[a] == sirius::AggregationType::MIN   ? 1
            : agg_mode[a] == sirius::AggregationType::MAX ? 2
                                                          : 0;
        if (in_id == GPUColumnTypeId::INT64 || in_id == GPUColumnTypeId::DECIMAL) {
          EmitInt64AggCol(gpu_id, N, slice, dst_idx++, aggs[a], gbm, mm_dec);
          // Re-tag the (freshly emitted) output column with the input's DECIMAL
          // {width, scale} so result_collector treats it as DECIMAL, not INT64.
          if (in_id == GPUColumnTypeId::DECIMAL && in_dti && aggs[a]) {
            aggs[a]->data_wrapper.type = GPUColumnType(GPUColumnTypeId::DECIMAL);
            aggs[a]->data_wrapper.type.SetDecimalTypeInfo(dti_w, dti_s);
          }
        } else {
          EmitDoubleAggCol(gpu_id, N, slice, dst_idx++, aggs[a], gbm, mm_dec);
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

// ══════════════════════════════════════════════════════════════════════════
// Wide-key GROUP BY: hash-routed magi shuffle with the ORIGINAL key bytes
// carried inline in the tuple (user design: "key 走 hash, payload 完整随行").
//
// Handles every key shape DeriveKeyShape cannot pack into a 128-bit word
// (e.g. TPC-H Q10: 7 keys incl. VARCHAR(117) c_comment). Per row we build a
// canonical byte layout of all key columns, hash it into 128 bits (two
// FNV-1a lanes + avalanche) — that hash IS the magi KeyT (routing + receiver
// atomic key; collision odds at 2^-128 are ignorable) — and ship the layout
// bytes as KEEP_I64 value slots in a 320B AggSlot64, merged first-writer-wins
// (identical bytes per group). The receiver merges partials ON ARRIVAL like
// any magi groupby; emit unpacks the inline bytes back into typed columns.
// ══════════════════════════════════════════════════════════════════════════
static constexpr int WIDE_MAX_KEYS  = 8;
static constexpr int WIDE_MAX_BYTES = 248;  // 31 KEEP words; +hash key +1 agg fits 320B

struct WideColDesc {
  const uint8_t*  data;
  const uint64_t* offs;    // VARCHAR only
  int32_t         width;   // fixed byte width; -1 = VARCHAR
  int32_t         off;     // byte offset inside the packed layout
  int32_t         maxlen;  // VARCHAR char budget (excludes the 2B length)
};
struct WideDesc {
  WideColDesc cols[WIDE_MAX_KEYS];
  int32_t     n_cols;
  int32_t     total_bytes;
  int32_t     n_keep;
  uint64_t    n_rows;
};

__global__ void k_wide_maxlen(const uint64_t* offs, uint64_t n, uint32_t* out)
{
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  atomicMax(out, (uint32_t)(offs[r + 1] - offs[r]));
}

// Contention-free "any string longer than the budget?" probe: only violating
// threads store (plain store, idempotent). An atomicMax over 120M rows onto a
// single counter costs ~4ms/column (Q1); this is a pure bandwidth-bound read.
__global__ void k_prefix_exceeds(const uint64_t* offs, uint64_t n, uint32_t budget,
                                 uint32_t* flag)
{
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  if ((uint32_t)(offs[r + 1] - offs[r]) > budget) { *flag = 1u; }
}

// One pass per input row: assemble the canonical key-byte layout, hash it
// (128-bit), and scatter the layout into the column-major KEEP blob the
// producer kernel will read through the i64_agg_cols pool.
__global__ void k_wide_pack(WideDesc d, int64_t* hash_lo, int64_t* hash_hi, int64_t* blob)
{
  const uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= d.n_rows) return;
  uint8_t buf[WIDE_MAX_BYTES];
  for (int i = 0; i < d.total_bytes; ++i) buf[i] = 0;
  for (int c = 0; c < d.n_cols; ++c) {
    const WideColDesc& col = d.cols[c];
    if (col.width >= 0) {
      for (int b = 0; b < col.width; ++b) buf[col.off + b] = col.data[r * col.width + b];
    } else {
      const uint64_t s   = col.offs[r];
      uint32_t       len = (uint32_t)(col.offs[r + 1] - s);
      if ((int32_t)len > col.maxlen) len = col.maxlen;
      buf[col.off]     = (uint8_t)(len & 0xFF);
      buf[col.off + 1] = (uint8_t)(len >> 8);
      for (uint32_t b = 0; b < len; ++b) buf[col.off + 2 + b] = col.data[s + b];
    }
  }
  uint64_t h1 = 1469598103934665603ULL;
  uint64_t h2 = 0x9E3779B97F4A7C15ULL;
  for (int i = 0; i < d.total_bytes; ++i) {
    h1 = (h1 ^ buf[i]) * 1099511628211ULL;
    h2 = (h2 ^ buf[i]) * 0xC2B2AE3D27D4EB4FULL;
  }
  h1 ^= h1 >> 33; h1 *= 0xFF51AFD7ED558CCDULL; h1 ^= h1 >> 33;
  h2 ^= h2 >> 29; h2 *= 0x94D049BB133111EBULL; h2 ^= h2 >> 29;
  hash_lo[r] = (int64_t)h1;
  hash_hi[r] = (int64_t)h2;
  for (int j = 0; j < d.n_keep; ++j) {
    int64_t w;
    memcpy(&w, buf + j * 8, 8);
    blob[(uint64_t)j * d.n_rows + r] = w;
  }
}

__global__ void k_wide_emit_fixed(const magi_generic::AggResultRow* in, size_t N,
                                  int byte_off, int width, uint8_t* out)
{
  size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  const uint8_t* src = reinterpret_cast<const uint8_t*>(in[r].values) + byte_off;
  for (int b = 0; b < width; ++b) out[r * width + b] = src[b];
}
__global__ void k_wide_emit_len(const magi_generic::AggResultRow* in, size_t N,
                                int byte_off, uint64_t* lens)
{
  size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  const uint8_t* src = reinterpret_cast<const uint8_t*>(in[r].values) + byte_off;
  lens[r] = (uint64_t)src[0] | ((uint64_t)src[1] << 8);
}
__global__ void k_wide_emit_chars(const magi_generic::AggResultRow* in, size_t N,
                                  int byte_off, const uint64_t* offs, uint8_t* chars)
{
  size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  const uint8_t* src = reinterpret_cast<const uint8_t*>(in[r].values) + byte_off;
  const uint64_t len = (uint64_t)src[0] | ((uint64_t)src[1] << 8);
  uint8_t* dst = chars + offs[r];
  for (uint64_t b = 0; b < len; ++b) dst[b] = src[2 + b];
}

// Plan-consistent eligibility (identical on every worker: type-driven only).
bool WideKeyShapeSupported(const vector<shared_ptr<GPUColumn>>& keys, int n_keys)
{
  if (n_keys <= 0 || n_keys > WIDE_MAX_KEYS) return false;
  for (int k = 0; k < n_keys; ++k) {
    if (!keys[k]) return false;
    const auto id = keys[k]->data_wrapper.type.id();
    switch (id) {
      case GPUColumnTypeId::INT32:
      case GPUColumnTypeId::DATE:
      case GPUColumnTypeId::INT64:
      case GPUColumnTypeId::VARCHAR:
        break;
      case GPUColumnTypeId::DECIMAL:
        if (keys[k]->data_wrapper.getColumnTypeSize() > 8) return false;
        break;
      default:
        return false;
    }
  }
  return true;
}

void RunWideKey(int                            gpu_id,
                vector<shared_ptr<GPUColumn>>& keys,
                vector<shared_ptr<GPUColumn>>& aggs,
                int                            num_group_keys,
                int                            num_aggregates,
                sirius::AggregationType*       agg_mode,
                const magi_generic::SlotPredicate& having_pred = {})
{
  GPUBufferManager* gbm      = &GPUBufferManager::GetInstance();
  const int         num_gpus = static_cast<int>(gbm->tables_per_gpu.size());
  const uint64_t    n_rows   = keys[0]->column_length;
  static const bool phase_time = std::getenv("MAGI_PHASE_TIME") != nullptr;
  const auto        t0         = std::chrono::steady_clock::now();

  // Per-VARCHAR local max length, then rendezvous so every worker agrees on
  // the SAME byte layout (a worker whose slice lacks the longest string must
  // still reserve room for it).
  std::array<uint32_t, WIDE_MAX_KEYS> local_max{};
  for (int k = 0; k < num_group_keys; ++k) {
    if (keys[k]->data_wrapper.type.id() != GPUColumnTypeId::VARCHAR || n_rows == 0) continue;
    uint32_t* d_max = gbm->customCudaMalloc<uint32_t>(1, gpu_id, 0);
    cudaMemset(d_max, 0, sizeof(uint32_t));
    int tpb; unsigned grid; gb_cfg(n_rows, tpb, grid);
    k_wide_maxlen<<<grid, tpb>>>(keys[k]->data_wrapper.offset, n_rows, d_max);
    cudaMemcpy(&local_max[k], d_max, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      throw InvalidInputException("magi_groupby (wide): maxlen kernel failed: %s",
                                  cudaGetErrorString(e));
    }
  }
  if (num_gpus > 1) {
    static std::array<std::array<uint32_t, WIDE_MAX_KEYS>, 8> slots;
    static std::unique_ptr<std::barrier<>> bar;
    static std::once_flag                  once;
    std::call_once(once, [&] { bar = std::make_unique<std::barrier<>>(num_gpus); });
    slots[gpu_id] = local_max;
    bar->arrive_and_wait();
    for (int g = 0; g < num_gpus; ++g)
      for (int k = 0; k < WIDE_MAX_KEYS; ++k)
        if (slots[g][k] > local_max[k]) local_max[k] = slots[g][k];
    bar->arrive_and_wait();
  }

  // Byte layout (identical across workers: fixed widths + agreed maxlens).
  WideDesc d{};
  d.n_cols = num_group_keys;
  int off  = 0;
  for (int k = 0; k < num_group_keys; ++k) {
    auto& col = keys[k]->data_wrapper;
    if (col.type.id() == GPUColumnTypeId::VARCHAR) {
      d.cols[k] = {col.data, col.offset, -1, off, (int32_t)local_max[k]};
      off += 2 + (int)local_max[k];
    } else {
      const int w = (int)col.getColumnTypeSize();
      d.cols[k]   = {col.data, nullptr, w, off, 0};
      off += w;
    }
  }
  if (off > WIDE_MAX_BYTES) {
    throw NotImplementedException(
        "magi_groupby (wide): key layout %d B exceeds the %d B inline budget",
        off, WIDE_MAX_BYTES);
  }
  d.total_bytes = off;
  d.n_keep      = (off + 7) / 8;
  d.n_rows      = n_rows;
  const auto t_prep = std::chrono::steady_clock::now();

  // hash + KEEP blob
  int64_t* hash_lo = gbm->customCudaMalloc<int64_t>(n_rows > 0 ? n_rows : 1, gpu_id, 0);
  int64_t* hash_hi = gbm->customCudaMalloc<int64_t>(n_rows > 0 ? n_rows : 1, gpu_id, 0);
  int64_t* blob    = gbm->customCudaMalloc<int64_t>(
      n_rows > 0 ? (uint64_t)d.n_keep * n_rows : 1, gpu_id, 0);
  if (n_rows > 0) {
    int tpb; unsigned grid; gb_cfg(n_rows, tpb, grid);
    k_wide_pack<<<grid, tpb>>>(d, hash_lo, hash_hi, blob);
  }
  cudaDeviceSynchronize();
  {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      throw InvalidInputException("magi_groupby (wide): pack kernel failed: %s",
                                  cudaGetErrorString(e));
    }
    if (phase_time) {
      std::fprintf(stderr, "[wide-dbg gpu=%d] pack ok rows=%llu layout=%dB keep=%d\n",
                   gpu_id, (unsigned long long)n_rows, d.total_bytes, d.n_keep);
    }
  }
  const auto t_pack = std::chrono::steady_clock::now();

  // Route through the stock UINT128 machinery with (hash_lo, hash_hi) as the
  // packed key and the real aggregates untouched; append the KEEP columns.
  vector<shared_ptr<GPUColumn>> hkeys(2);
  hkeys[0] = make_shared_ptr<GPUColumn>(n_rows, GPUColumnType(GPUColumnTypeId::INT64),
                                        reinterpret_cast<uint8_t*>(hash_lo), nullptr);
  hkeys[1] = make_shared_ptr<GPUColumn>(n_rows, GPUColumnType(GPUColumnTypeId::INT64),
                                        reinterpret_cast<uint8_t*>(hash_hi), nullptr);
  magi_generic::KeyKind                key_kind;
  std::vector<magi_ops::KeyFieldEntry> key_fields;
  DeriveKeyShape(hkeys, 2, key_kind, key_fields);
  const auto table_size = PickTableSize(hkeys, 2);

  auto in             = BuildGenericInputs(hkeys, aggs, 2, num_aggregates, agg_mode);
  int  avg_count_slot = -1;
  auto ops            = BuildAggOpsTable(aggs, num_aggregates, agg_mode, avg_count_slot);
  int dst_base = 0;
  for (const auto& op : ops)
    if ((int)op.dst_slot_idx + 1 > dst_base) dst_base = (int)op.dst_slot_idx + 1;
  const int keep_col_base = in.cols.n_int64_aggs;
  if (keep_col_base + d.n_keep > magi_ops::MAX_INT64_AGG_COLS ||
      dst_base + d.n_keep > magi_generic::AggResultRow::N_VALUES) {
    throw NotImplementedException(
        "magi_groupby (wide): %d agg slots + %d keep words exceed slot capacity",
        dst_base, d.n_keep);
  }
  // Bisect switch: MAGI_WIDE_NO_KEEP drops the inline-key KEEP slots (keys
  // emit as garbage) to isolate the KEEP device branches from the rest of the
  // wide pipeline when debugging.
  static const bool no_keep = std::getenv("MAGI_WIDE_NO_KEEP") != nullptr;
  if (!no_keep) {
    for (int j = 0; j < d.n_keep; ++j) {
      in.cols.i64_agg_cols[keep_col_base + j] = blob + (uint64_t)j * (n_rows > 0 ? n_rows : 1);
      ops.push_back({magi_ops::AggKind::KEEP_I64, (int8_t)(keep_col_base + j),
                     (int8_t)(dst_base + j), 0});
    }
    in.cols.n_int64_aggs = keep_col_base + d.n_keep;
  }

  std::vector<magi_generic::AggResultRow> slice;
  magi_generic::AggResultRow*             d_rows = nullptr;
  const std::size_t N = magi_generic::distributed_hash_groupby_run_per_gpu(
      gpu_id, in, key_fields, ops, key_kind, table_size, slice,
      /*device_emit=*/true, &d_rows, having_pred);
  const auto t_run = std::chrono::steady_clock::now();
  if (phase_time) {
    cudaError_t e = cudaGetLastError();
    std::fprintf(stderr, "[wide-dbg gpu=%d] shuffle done groups=%zu err=%s\n",
                 gpu_id, N, cudaGetErrorString(e));
  }

  // ── Emit: inline key bytes → typed columns; aggregates via stock helpers ──
  // Two phases so the per-VARCHAR pipelines overlap instead of serializing:
  // phase 1 queues ALL fixed-col emits, lens kernels and prefix scans without
  // a single blocking call (par_nosync — stock thrust::device syncs per scan);
  // phase 2's first total-D2H then waits once for everything, the remaining
  // D2Hs are ~µs, and the chars kernels queue behind them. With 5 VARCHAR keys
  // (Q10) this collapses 5×(scan sync + D2H sync) into one pipeline flush.
  const int byte_base = dst_base * 8;
  int tpb; unsigned grid; gb_cfg(N > 0 ? N : 1, tpb, grid);
  struct VcPending { int k; uint64_t* lens; uint64_t* offs; };
  std::vector<VcPending> vc_pending;
  for (int k = 0; k < num_group_keys; ++k) {
    const GPUColumnType out_type = keys[k]->data_wrapper.type;  // keeps DATE/DECIMAL info
    const int           col_off  = byte_base + d.cols[k].off;
    if (d.cols[k].width >= 0) {
      if (N == 0) {
        keys[k] = make_shared_ptr<GPUColumn>(0, out_type, nullptr, nullptr);
      } else {
        uint8_t* out = gbm->customCudaMalloc<uint8_t>(N * d.cols[k].width, gpu_id, 0);
        k_wide_emit_fixed<<<grid, tpb>>>(d_rows, N, col_off, d.cols[k].width, out);
        keys[k] = make_shared_ptr<GPUColumn>(N, out_type, out, createNullMask(N));
      }
      keys[k]->row_id_count = 0;
    } else {
      if (N == 0) {
        keys[k] = make_shared_ptr<GPUColumn>(0, out_type, nullptr, nullptr, 0, true, nullptr);
        keys[k]->row_id_count = 0;
      } else {
        uint64_t* lens = gbm->customCudaMalloc<uint64_t>(N, gpu_id, 0);
        uint64_t* offs = gbm->customCudaMalloc<uint64_t>(N + 1, gpu_id, 0);
        k_wide_emit_len<<<grid, tpb>>>(d_rows, N, col_off, lens);
        cudaMemsetAsync(offs, 0, sizeof(uint64_t));
        thrust::inclusive_scan(thrust::cuda::par_nosync, lens, lens + N, offs + 1);
        vc_pending.push_back({k, lens, offs});
      }
    }
  }
  for (const auto& vc : vc_pending) {
    const GPUColumnType out_type = keys[vc.k]->data_wrapper.type;
    const int           col_off  = byte_base + d.cols[vc.k].off;
    uint64_t total_chars = 0;
    cudaMemcpy(&total_chars, vc.offs + N, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint8_t* chars = gbm->customCudaMalloc<uint8_t>(total_chars > 0 ? total_chars : 1,
                                                    gpu_id, 0);
    k_wide_emit_chars<<<grid, tpb>>>(d_rows, N, col_off, vc.offs, chars);
    keys[vc.k] = make_shared_ptr<GPUColumn>(N, out_type, chars, vc.offs, total_chars, true,
                                            createNullMask(N));
    keys[vc.k]->row_id_count = 0;
  }
  if (phase_time) cudaDeviceSynchronize();
  const auto t_ekeys = std::chrono::steady_clock::now();
  // Aggregates: same emit sequence as WriteGenericSliceToColumnsDevice.
  int dst_idx = 0;
  for (int a = 0; a < num_aggregates; ++a) {
    switch (agg_mode[a]) {
      case sirius::AggregationType::SUM:
      case sirius::AggregationType::MIN:
      case sirius::AggregationType::MAX: {
        const auto in_id =
            aggs[a] ? aggs[a]->data_wrapper.type.id() : GPUColumnTypeId::FLOAT64;
        const auto* in_dti = (in_id == GPUColumnTypeId::DECIMAL && aggs[a])
                                 ? aggs[a]->data_wrapper.type.GetDecimalTypeInfo()
                                 : nullptr;
        const int dti_w = in_dti ? in_dti->width_ : 0;
        const int dti_s = in_dti ? in_dti->scale_ : 0;
        const int mm_dec =
            agg_mode[a] == sirius::AggregationType::MIN   ? 1
            : agg_mode[a] == sirius::AggregationType::MAX ? 2
                                                          : 0;
        if (in_id == GPUColumnTypeId::INT64 || in_id == GPUColumnTypeId::DECIMAL) {
          EmitInt64AggColDevice(gpu_id, N, d_rows, dst_idx++, aggs[a], gbm, mm_dec);
          if (in_id == GPUColumnTypeId::DECIMAL && in_dti && aggs[a]) {
            aggs[a]->data_wrapper.type = GPUColumnType(GPUColumnTypeId::DECIMAL);
            aggs[a]->data_wrapper.type.SetDecimalTypeInfo(dti_w, dti_s);
          }
        } else {
          EmitDoubleAggColDevice(gpu_id, N, d_rows, dst_idx++, aggs[a], gbm, mm_dec);
        }
        break;
      }
      case sirius::AggregationType::AVERAGE: {
        const auto in_id =
            aggs[a] ? aggs[a]->data_wrapper.type.id() : GPUColumnTypeId::FLOAT64;
        const bool sum_is_int64 =
            (in_id == GPUColumnTypeId::INT64 || in_id == GPUColumnTypeId::DECIMAL);
        int scale = 0;
        if (in_id == GPUColumnTypeId::DECIMAL && aggs[a]) {
          const auto* dti = aggs[a]->data_wrapper.type.GetDecimalTypeInfo();
          if (dti) scale = dti->scale_;
        }
        EmitAvgColDevice(gpu_id, N, d_rows, dst_idx, avg_count_slot, sum_is_int64, scale,
                         aggs[a], gbm);
        dst_idx++;
        break;
      }
      case sirius::AggregationType::COUNT_STAR:
      case sirius::AggregationType::COUNT:
        EmitInt64AggColDevice(gpu_id, N, d_rows, dst_idx++, aggs[a], gbm);
        break;
      default:
        throw NotImplementedException(
            "magi_groupby (wide): unsupported AggregationType %d at emit",
            static_cast<int>(agg_mode[a]));
    }
  }
  cudaDeviceSynchronize();
  const auto t_eaggs = std::chrono::steady_clock::now();
  // d_rows lives in the runtime's g_agg_dev arena (or a runtime-tracked
  // overflow malloc) — the runtime owns it; do not free here.
  if (phase_time) {
    const auto t_end = std::chrono::steady_clock::now();
    auto ms = [](std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
                 "[wide-groupby gpu=%d] rows_in=%llu groups_out=%zu layout=%dB keep=%d "
                 "prep=%.2f pack=%.2f run=%.2f emitk=%.2f emita=%.2f free=%.2f "
                 "total=%.2fms\n",
                 gpu_id, (unsigned long long)n_rows, N, d.total_bytes, d.n_keep,
                 ms(t0, t_prep), ms(t_prep, t_pack), ms(t_pack, t_run),
                 ms(t_run, t_ekeys), ms(t_ekeys, t_eaggs), ms(t_eaggs, t_end),
                 ms(t0, t_end));
  }
}

static bool VarcharPrefixInsufficient(int                                         gpu_id,
                                      vector<shared_ptr<GPUColumn>>&              keys,
                                      const std::vector<magi_ops::KeyFieldEntry>& fields);

// Cross-GPU max of a per-worker row count. The pre-agg decision (and any
// rows-thresholded gate ahead of a rendezvous) must be IDENTICAL on every
// worker: per-GPU partitions differ by a few hundred rows, so a local-rows
// threshold could split workers across a barrier (deadlock) or diverge the
// COUNT→SUM re-aggregation modes (wrong merge). Unconditional rendezvous —
// every worker calls exactly once per decision point.
static uint64_t CrossGpuMaxRows(int gpu_id, uint64_t local_rows)
{
  GPUBufferManager* gbm      = &GPUBufferManager::GetInstance();
  const int         num_gpus = static_cast<int>(gbm->tables_per_gpu.size());
  if (num_gpus <= 1) { return local_rows; }
  static std::array<uint64_t, 8>         slots;
  static std::unique_ptr<std::barrier<>> bar;
  static std::once_flag                  once;
  std::call_once(once, [&] { bar = std::make_unique<std::barrier<>>(num_gpus); });
  slots[gpu_id] = local_rows;
  bar->arrive_and_wait();
  uint64_t m = 0;
  for (int g = 0; g < num_gpus; ++g) {
    if (slots[g] > m) { m = slots[g]; }
  }
  bar->arrive_and_wait();
  return m;
}

bool ShouldCudfPreAgg(int gpu_id, vector<shared_ptr<GPUColumn>>& keys, int n_keys)
{
  // >= : the predicate means "big-tier bound" — adding XXLARGE above XLARGE
  // silently disabled the pre-agg for Q18-class inputs (150M raw rows went
  // straight at the tier guard and fell back).
  if (PickTableSize(keys, n_keys) >= magi_generic::TableSize::XLARGE) { return true; }
  // Inputs headed for the WIDE path benefit from a local pre-agg regardless
  // of the tier estimate: the 320B-slot machinery costs O(input rows) at
  // random-access bandwidth, so collapse duplicates locally first. Q4's
  // `group by o_orderpriority` (1.3M rows -> 4 groups, "4-NOT SPECIFIED"
  // exceeds the 8B prefix) paid 6.5ms in the wide path for a 4-group
  // aggregation without this.
  const uint64_t n_rows = keys.empty() || !keys[0] ? 0 : keys[0]->column_length;
  if (CrossGpuMaxRows(gpu_id, n_rows) <= 65536) { return false; }
  magi_generic::KeyKind                key_kind;
  std::vector<magi_ops::KeyFieldEntry> key_fields;
  bool shape_ok = DeriveKeyShape(keys, n_keys, key_kind, key_fields);
  // Lockstep with Run()'s u128-typed-tier quarantine: compound keys go wide.
  static const bool u128_typed = std::getenv("MAGI_U128_TYPED") != nullptr;
  if (shape_ok && !u128_typed && key_kind == magi_generic::KeyKind::UINT128) { shape_ok = false; }
  if (!shape_ok) { return WideKeyShapeSupported(keys, n_keys); }
  // VarcharPrefixInsufficient rendezvouses across workers behind
  // type-deterministic gates only, and its verdict is a cross-GPU OR —
  // identical on every worker (Run() later repeats it — also symmetric).
  return VarcharPrefixInsufficient(gpu_id, keys, key_fields);
}

// A VARCHAR_PREFIX recipe is only exact when every string in the query's
// actual data fits its prefix budget: the emit rebuilds the string FROM the
// packed prefix (a longer string comes back truncated — Q5's "INDONESIA" →
// "INDONESI"), and two distinct strings sharing a prefix would silently merge
// into one group. Checked against the runtime max string length; the verdict
// is rendezvoused across workers (cross-GPU OR) so every worker takes the
// same DeriveKeyShape-vs-wide branch — a split would deadlock the barriers.
static bool VarcharPrefixInsufficient(int                                        gpu_id,
                                      vector<shared_ptr<GPUColumn>>&             keys,
                                      const std::vector<magi_ops::KeyFieldEntry>& fields)
{
  bool has_prefix = false;
  for (const auto& f : fields) {
    if (f.kind == magi_ops::KeyFieldKind::VARCHAR_PREFIX) { has_prefix = true; break; }
  }
  if (!has_prefix) { return false; }   // identical on every worker (same recipe)

  GPUBufferManager* gbm      = &GPUBufferManager::GetInstance();
  const int         num_gpus = static_cast<int>(gbm->tables_per_gpu.size());
  bool local_bad = false;
  uint32_t* d_flag = gbm->customCudaMalloc<uint32_t>(1, gpu_id, 0);
  cudaMemset(d_flag, 0, sizeof(uint32_t));
  for (const auto& f : fields) {
    if (f.kind != magi_ops::KeyFieldKind::VARCHAR_PREFIX) { continue; }
    const auto& col = keys[f.src_col_idx];
    const uint64_t n = col ? col->column_length : 0;
    if (n == 0 || col->data_wrapper.offset == nullptr) { continue; }
    int tpb; unsigned grid; gb_cfg(n, tpb, grid);
    k_prefix_exceeds<<<grid, tpb>>>(col->data_wrapper.offset, n,
                                    static_cast<uint32_t>(f.byte_len), d_flag);
  }
  uint32_t exceeded = 0;
  cudaMemcpy(&exceeded, d_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  local_bad = exceeded != 0;
  if (num_gpus > 1) {
    static std::array<uint8_t, 8>          slots;
    static std::unique_ptr<std::barrier<>> bar;
    static std::once_flag                  once;
    std::call_once(once, [&] { bar = std::make_unique<std::barrier<>>(num_gpus); });
    slots[gpu_id] = local_bad ? 1 : 0;
    bar->arrive_and_wait();
    bool any = false;
    for (int g = 0; g < num_gpus; ++g) { any = any || slots[g] != 0; }
    bar->arrive_and_wait();
    return any;
  }
  return local_bad;
}

void Run(int                                gpu_id,
         vector<shared_ptr<GPUColumn>>&     group_by_keys,
         vector<shared_ptr<GPUColumn>>&     aggregate_keys,
         int                                num_group_keys,
         int                                num_aggregates,
         sirius::AggregationType*           agg_mode,
         const SlotPredicate&               having_pred)
{
  // ── Widen narrow SUM/AVG inputs up front ────────────────────────────────
  // SUM over INT32 (e.g. Q12's `CASE WHEN .. THEN 1 ELSE 0`) has no device
  // path; widening to INT64 here lets every type-driven stage downstream
  // (input routing, ops table, emit) take the existing INT64 path. DuckDB's
  // sum(int) result is HUGEINT and the collector already converts our INT64
  // output to INT128, so the logical types line up.
  for (int a = 0; a < num_aggregates; ++a) {
    if (agg_mode[a] != sirius::AggregationType::SUM &&
        agg_mode[a] != sirius::AggregationType::AVERAGE) {
      continue;
    }
    auto& col = aggregate_keys[a];
    if (!col || col->data_wrapper.type.id() != GPUColumnTypeId::INT32) { continue; }
    const size_t n   = col->column_length;
    uint8_t*     wide = nullptr;
    if (col->data_wrapper.data != nullptr && n > 0) {
      wide = GPUBufferManager::GetInstance().customCudaMalloc<uint8_t>(
        n * sizeof(int64_t), gpu_id, 0);
      convertInt32ToInt64(col->data_wrapper.data, wide, n);
    }
    auto widened = make_shared_ptr<GPUColumn>(
      n, GPUColumnType(GPUColumnTypeId::INT64), wide, col->data_wrapper.validity_mask);
    widened->row_id_count = 0;
    col = widened;
  }

  // ── Generic path (table-driven; handles all supported GROUP BY shapes) ──
  magi_generic::KeyKind                 key_kind;
  std::vector<magi_ops::KeyFieldEntry>  key_fields;
  bool shape_ok = DeriveKeyShape(group_by_keys, num_group_keys, key_kind, key_fields);
  // QUARANTINE: the u128 TYPED tier (16B compound keys, e.g. Q20's
  // (l_partkey, l_suppkey)) misbehaved on the 2-int64 shape — probe storms
  // (A 2.5s/B 11.5s on a 600k-row input), empty/undercounted results,
  // intermittent kernel-B hangs. Route compound keys through the WIDE
  // hash128 path (proven by Q16 at 3M rows/GPU) unless MAGI_U128_TYPED=1
  // (debug/repair knob). Lockstep with ShouldCudfPreAgg's override.
  static const bool u128_typed = std::getenv("MAGI_U128_TYPED") != nullptr;
  if (shape_ok && !u128_typed && key_kind == magi_generic::KeyKind::UINT128) { shape_ok = false; }
  if (shape_ok && VarcharPrefixInsufficient(gpu_id, group_by_keys, key_fields)) {
    // Strings exceed the prefix budget → the packed-prefix path would emit
    // truncated keys (and could merge distinct groups). Reroute to the wide
    // path, which carries the full key bytes.
    shape_ok = false;
  }
  if (!shape_ok) {
    // Wide-key fallback: hash-routed shuffle with the original key bytes
    // inlined in the tuple (see RunWideKey). Eligibility is type-driven, so
    // every worker takes the same branch.
    if (std::getenv("MAGI_NO_WIDE_GROUPBY") == nullptr &&
        WideKeyShapeSupported(group_by_keys, num_group_keys)) {
      RunWideKey(gpu_id, group_by_keys, aggregate_keys, num_group_keys,
                 num_aggregates, agg_mode, having_pred);
      return;
    }
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
  // MAGI_INPUT_DUMP=1: sample every aggregate input column AT MAGI ENTRY
  // (front/middle/back windows -> mean/min/max). The SF100 rerun corruption
  // signature (avg 4.9e6 where l_quantity should read 25.5) points at the
  // input columns being wrong before any shuffle happens; this pins down
  // whether the damage exists here or is introduced later.
  if (getenv("MAGI_INPUT_DUMP")) {
    auto dump_win = [&](const char* tag, int a, const void* base, size_t n,
                        bool is_i64) {
      if (!base || n == 0) {
        fprintf(stderr, "[magi-in gpu=%d] agg%d %s ptr=null n=%zu\n",
                gpu_id, a, tag, n);
        return;
      }
      constexpr size_t W = 1024;
      const size_t off[3] = {0, n > W ? n / 2 : 0, n > W ? n - W : 0};
      double mn = 1e300, mx = -1e300, sum = 0.0;
      size_t cnt = 0;
      std::vector<double> h(W);
      for (int w = 0; w < 3; ++w) {
        const size_t take = std::min(W, n - off[w]);
        cudaMemcpy(h.data(),
                   reinterpret_cast<const char*>(base) + off[w] * 8,
                   take * 8, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < take; ++i) {
          const double v = is_i64
              ? static_cast<double>(reinterpret_cast<const int64_t*>(h.data())[i])
              : h[i];
          mn = std::min(mn, v); mx = std::max(mx, v); sum += v; ++cnt;
        }
      }
      fprintf(stderr,
              "[magi-in gpu=%d] agg%d %s ptr=%p n=%zu mean=%.6g min=%.6g max=%.6g\n",
              gpu_id, a, tag, base, n, sum / cnt, mn, mx);
    };
    fprintf(stderr, "[magi-in gpu=%d] n_filtered=%zu key0_len=%zu key0_rowids=%zu\n",
            gpu_id, (size_t)in.n_filtered,
            (size_t)group_by_keys[0]->column_length,
            (size_t)group_by_keys[0]->row_id_count);
    for (int a = 0; a < num_aggregates; ++a) {
      if (!aggregate_keys[a]) continue;
      const auto id = aggregate_keys[a]->data_wrapper.type.id();
      const bool i64 = (id == GPUColumnTypeId::INT64 ||
                        id == GPUColumnTypeId::DECIMAL);
      dump_win(i64 ? "i64" : "f64", a, aggregate_keys[a]->data_wrapper.data,
               aggregate_keys[a]->column_length, i64);
      if (aggregate_keys[a]->row_id_count)
        fprintf(stderr, "[magi-in gpu=%d] agg%d HAS row_ids=%zu\n", gpu_id, a,
                (size_t)aggregate_keys[a]->row_id_count);
    }
  }
  // MAGI_INPUT_SCAN=1: full sweep of the first i64 aggregate input for values
  // no TPC-H cents column can hold (negative or > 1e10). The warm-run garbage
  // rows land somewhere in these 296M-row buffers; this finds their indices
  // and prints every column (and the group-key bytes) at those rows.
  if (getenv("MAGI_INPUT_SCAN")) {
    const size_t N = in.n_filtered;
    std::vector<size_t> bad;
    const int64_t* q = nullptr;
    for (int a = 0; a < num_aggregates && !q; ++a) {
      if (!aggregate_keys[a] || !aggregate_keys[a]->data_wrapper.data) continue;
      const auto id = aggregate_keys[a]->data_wrapper.type.id();
      if (id == GPUColumnTypeId::INT64 || id == GPUColumnTypeId::DECIMAL)
        q = reinterpret_cast<const int64_t*>(aggregate_keys[a]->data_wrapper.data);
    }
    if (q && N) {
      constexpr size_t CH = 8u << 20;
      std::vector<int64_t> h(CH);
      for (size_t off = 0; off < N && bad.size() < 64; off += CH) {
        const size_t take = std::min(CH, N - off);
        cudaMemcpy(h.data(), q + off, take * 8, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < take && bad.size() < 64; ++i)
          if (h[i] < 0 || h[i] > 10000000000LL) bad.push_back(off + i);
      }
    }
    fprintf(stderr, "[magi-scan gpu=%d] N=%zu bad_rows=%zu%s\n", gpu_id, N,
            bad.size(), bad.size() >= 64 ? " (capped)" : "");
    for (size_t bi = 0; bi < bad.size() && bi < 8; ++bi) {
      const size_t i = bad[bi];
      fprintf(stderr, "[magi-scan gpu=%d] row=%zu (N-row=%zu)", gpu_id, i, N - i);
      for (int a = 0; a < num_aggregates; ++a) {
        if (!aggregate_keys[a] || !aggregate_keys[a]->data_wrapper.data) continue;
        int64_t v;
        cudaMemcpy(&v,
                   reinterpret_cast<const int64_t*>(
                       aggregate_keys[a]->data_wrapper.data) + i,
                   8, cudaMemcpyDeviceToHost);
        fprintf(stderr, " a%d=%lld", a, (long long)v);
      }
      for (int k = 0; k < num_group_keys; ++k) {
        if (group_by_keys[k]->data_wrapper.type.id() != GPUColumnTypeId::VARCHAR)
          continue;
        uint64_t o2[2] = {0, 0};
        cudaMemcpy(o2,
                   reinterpret_cast<const uint64_t*>(
                       group_by_keys[k]->data_wrapper.offset) + i,
                   16, cudaMemcpyDeviceToHost);
        char c[8] = {0};
        const uint64_t L = std::min<uint64_t>(o2[1] - o2[0], 7);
        if (o2[1] > o2[0] && L <= 7)
          cudaMemcpy(c,
                     reinterpret_cast<const char*>(
                         group_by_keys[k]->data_wrapper.data) + o2[0],
                     L, cudaMemcpyDeviceToHost);
        fprintf(stderr, " k%d='%s'(off=%llu len=%llu)", k, c,
                (unsigned long long)o2[0], (unsigned long long)(o2[1] - o2[0]));
      }
      fprintf(stderr, "\n");
    }
  }
  int  avg_count_slot = -1;
  auto ops  = BuildAggOpsTable(aggregate_keys, num_aggregates, agg_mode,
                               avg_count_slot);
  std::vector<magi_generic::AggResultRow> slice;  // unused under device_emit (API ref)
  magi_generic::AggResultRow* d_rows = nullptr;
  const std::size_t n_rows = magi_generic::distributed_hash_groupby_run_per_gpu(
      gpu_id, in, key_fields, ops, key_kind, table_size, slice,
      /*device_emit=*/true, &d_rows, having_pred);

  // Build the output GPUColumns ON-DEVICE directly from the device AggResultRow
  // buffer (no D2H of slots + host slice construction + per-column host transpose).
  WriteGenericSliceToColumnsDevice(gpu_id, d_rows, n_rows,
                                   group_by_keys, aggregate_keys,
                                   num_group_keys, num_aggregates, agg_mode,
                                   key_fields, avg_count_slot,
                                   &GPUBufferManager::GetInstance());
  // The emit kernels + thrust scans run on the default stream; flush before the
  // columns are consumed downstream. d_rows lives in the runtime's g_agg_dev
  // arena (or a runtime-tracked overflow malloc) — the runtime owns it.
  cudaDeviceSynchronize();
}

}  // namespace magi_groupby
}  // namespace duckdb
