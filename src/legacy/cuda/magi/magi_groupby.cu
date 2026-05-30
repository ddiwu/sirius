// magi_groupby.cu — operator-facing entry that routes
// `GPUPhysicalGroupedAggregate` through the Magi NVLink-shuffle backend.
//
// Each legacy worker thread (one per GPU) calls Run() on its partition's
// columns. The dispatch table inside Run() picks the kernel for the
// current input shape; for M1 only Q1's shape is wired:
//
//     2 group keys, both VARCHAR(1)
//     N aggregates ∈ {SUM(DOUBLE), COUNT_STAR, COUNT(DOUBLE)}
//
// Anything else throws NotImplementedException. Future queries widen the
// dispatch by adding more shape branches. The Q1 path internally calls
// magi_q1::Q1MagiRunPerGpu (which std::barriers across the per-GPU
// threads), then materialises this GPU's hash-partitioned slice as a
// new (chars+offsets) VARCHAR + DOUBLE/INT64 column set on the same GPU.

#include "operator/magi_groupby.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "gpu_buffer_manager.hpp"
#include "operator/magi_q1.hpp"
#include "operator/magi_q5.hpp"
#include "operator/magi_distributed_groupby.hpp"

namespace duckdb {
namespace magi_groupby {

namespace {

// Detect Q1 shape: keys are 2 VARCHAR (rf, ls), aggregates are SUM(DOUBLE)
// + COUNT_STAR / COUNT in any order. We also need 4 SUM aggregates and 1
// COUNT-ish for the Q1 result row, but we leave that count-check to the
// Q1 mapping below — Run() is happy to dispatch as long as the columns
// look Q1-shaped.
bool IsQ1Shape(const vector<shared_ptr<GPUColumn>>& keys,
               const vector<shared_ptr<GPUColumn>>& aggs,
               int                                  num_group_keys,
               int                                  num_aggregates,
               sirius::AggregationType*                     agg_mode)
{
  if (num_group_keys != 2) return false;
  for (int i = 0; i < 2; ++i) {
    if (!keys[i]) return false;
    if (keys[i]->data_wrapper.type.id() != GPUColumnTypeId::VARCHAR) return false;
  }
  for (int i = 0; i < num_aggregates; ++i) {
    if (agg_mode[i] == sirius::AggregationType::COUNT_STAR) continue;
    if (!aggs[i]) return false;
    auto tid = aggs[i]->data_wrapper.type.id();
    bool is_sum = (agg_mode[i] == sirius::AggregationType::SUM)   && tid == GPUColumnTypeId::FLOAT64;
    bool is_cnt = (agg_mode[i] == sirius::AggregationType::COUNT) && tid == GPUColumnTypeId::FLOAT64;
    if (!is_sum && !is_cnt) return false;
  }
  return true;
}

// Pull the column pointers Q1's kernel needs out of sirius's GPUColumns.
// The kernel's input layout is fixed: l_quantity / l_extendedprice /
// l_discount / l_tax + (rf chars+offsets) + (ls chars+offsets). The
// 4 SUM aggregates from the operator are passed in some order; for M1
// we trust that the operator hands us {sum_qty, sum_ep, sum_disc, sum_tax}
// in that index order — caller (gpu_physical_grouped_aggregate) ensures
// it because that's the order the planner emits them in for Q1.
//
// FUTURE: when we widen past Q1, replace this with a generic
// (column_index → role) mapping derived from the BoundAggregateExpression
// children, so we don't depend on planner emit order.
magi_q1::PerGpuInputs BuildPerGpuInputs(
    const vector<shared_ptr<GPUColumn>>& keys,
    const vector<shared_ptr<GPUColumn>>& aggs,
    int                                  num_aggregates)
{
  magi_q1::PerGpuInputs in{};
  in.n_filtered = keys[0]->column_length;

  // Aggregate columns 0..3 are SUM(double); skip COUNT_STAR (no data ptr).
  // We need exactly 4 SUM doubles in some prefix; M1's q1.cuh doesn't read
  // tax separately, but the planner still passes it as a SUM input.
  int sum_idx = 0;
  const double* sum_ptrs[4] = { nullptr, nullptr, nullptr, nullptr };
  for (int i = 0; i < num_aggregates && sum_idx < 4; ++i) {
    if (!aggs[i]) continue;  // COUNT_STAR has no data
    if (aggs[i]->data_wrapper.type.id() != GPUColumnTypeId::FLOAT64) continue;
    sum_ptrs[sum_idx++] = reinterpret_cast<const double*>(aggs[i]->data_wrapper.data);
  }
  in.d_quantity = sum_ptrs[0];
  in.d_ep       = sum_ptrs[1];
  in.d_disc     = sum_ptrs[2];
  in.d_tax      = sum_ptrs[3];

  in.rf_chars   = keys[0]->data_wrapper.data;
  in.rf_offsets = keys[0]->data_wrapper.offset;
  in.ls_chars   = keys[1]->data_wrapper.data;
  in.ls_offsets = keys[1]->data_wrapper.offset;
  return in;
}

// Materialise this GPU's slice into freshly-allocated device columns.
// VARCHAR(1) layout matches sirius's existing convention: chars buffer
// of length N (one byte per row) + offsets buffer of length N+1
// (offsets[i] = i, offsets[N] = N).
void WriteSliceToColumns(int                                gpu_id,
                         const std::vector<magi_q1::AggResultRow>& slice,
                         vector<shared_ptr<GPUColumn>>&     keys,
                         vector<shared_ptr<GPUColumn>>&     aggs,
                         int                                num_aggregates,
                         sirius::AggregationType*                   agg_mode,
                         GPUBufferManager*                  gbm)
{
  const size_t N = slice.size();

  // ── Group keys (2 × VARCHAR(1)) ────────────────────────────────────────
  for (int kcol = 0; kcol < 2; ++kcol) {
    if (N == 0) {
      // Empty slice: hand sirius a column with data=nullptr so the operator's
      // combine branch picks the peer thread's non-empty column (instead of
      // falling into combineMasks with garbage on N=0). At 4 GPUs with Q1's 4
      // hash buckets some workers regularly get zero rows.
      keys[kcol] = make_shared_ptr<GPUColumn>(0,
                                              GPUColumnType(GPUColumnTypeId::VARCHAR),
                                              /*data=*/nullptr,
                                              /*offset=*/nullptr,
                                              /*num_bytes=*/0,
                                              /*is_string_data=*/true,
                                              /*validity_mask=*/nullptr);
      keys[kcol]->row_id_count = 0;
      continue;
    }
    auto* d_chars   = gbm->customCudaMalloc<uint8_t> (N,     gpu_id, false);
    auto* d_offsets = gbm->customCudaMalloc<uint64_t>(N + 1, gpu_id, false);
    std::vector<uint8_t>  h_chars(N);
    std::vector<uint64_t> h_offsets(N + 1);
    for (size_t i = 0; i < N; ++i) {
      h_chars[i]   = static_cast<uint8_t>(kcol == 0 ? slice[i].rf : slice[i].ls);
      h_offsets[i] = i;
    }
    h_offsets[N] = N;
    cudaMemcpy(d_chars,   h_chars.data(),   N,                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets.data(), (N + 1) * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    auto* mask = createNullMask(N);  // ALL_VALID
    keys[kcol] = make_shared_ptr<GPUColumn>(N,
                                            GPUColumnType(GPUColumnTypeId::VARCHAR),
                                            d_chars,
                                            d_offsets,
                                            /*num_bytes=*/N,
                                            /*is_string_data=*/true,
                                            mask);
    keys[kcol]->row_id_count = 0;
  }

  // ── Aggregates: 4 SUM doubles (in planner-emit order) + COUNT_STAR ─────
  // Helper: write a vector of doubles to device and replace the column.
  auto write_double_col = [&](int agg_idx, const std::vector<double>& host) {
    if (N == 0) {
      aggs[agg_idx] = make_shared_ptr<GPUColumn>(0,
                                                 GPUColumnType(GPUColumnTypeId::FLOAT64),
                                                 /*data=*/nullptr,
                                                 /*validity_mask=*/nullptr);
      aggs[agg_idx]->row_id_count = 0;
      return;
    }
    auto* d_buf = gbm->customCudaMalloc<double>(N, gpu_id, false);
    cudaMemcpy(d_buf, host.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    aggs[agg_idx] = make_shared_ptr<GPUColumn>(N,
                                               GPUColumnType(GPUColumnTypeId::FLOAT64),
                                               reinterpret_cast<uint8_t*>(d_buf),
                                               createNullMask(N));
    aggs[agg_idx]->row_id_count = 0;
  };
  auto write_int64_col = [&](int agg_idx, const std::vector<uint64_t>& host) {
    if (N == 0) {
      aggs[agg_idx] = make_shared_ptr<GPUColumn>(0,
                                                 GPUColumnType(GPUColumnTypeId::INT64),
                                                 /*data=*/nullptr,
                                                 /*validity_mask=*/nullptr);
      aggs[agg_idx]->row_id_count = 0;
      return;
    }
    auto* d_buf = gbm->customCudaMalloc<uint64_t>(N, gpu_id, false);
    cudaMemcpy(d_buf, host.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    aggs[agg_idx] = make_shared_ptr<GPUColumn>(N,
                                               GPUColumnType(GPUColumnTypeId::INT64),
                                               reinterpret_cast<uint8_t*>(d_buf),
                                               createNullMask(N));
    aggs[agg_idx]->row_id_count = 0;
  };

  // Walk the agg list in planner order; for each SUM/COUNT slot, emit the
  // matching field from the slice. SUM order matches BuildPerGpuInputs.
  std::vector<double>  v_qty(N), v_ep(N), v_disc(N), v_charge(N);
  std::vector<uint64_t> v_count(N);
  // q1.cuh's slot encoding doesn't track sum_tax; the planner doesn't need
  // it as a final agg either (TPC-H Q1 uses tax only inside the disc_price
  // / charge expression, which Q1Tuple precomputes). But the planner does
  // pass an l_tax SUM aggregate column. Map that slot to slice.sum_disc
  // (= partial sum of l_discount, also unused downstream). It's a
  // placeholder only — kill this once we generalise past Q1's hardcoded
  // schema, but keeps the column count + types matching what the operator
  // expects to find on output.
  std::vector<double> v_disc_placeholder(N);
  for (size_t i = 0; i < N; ++i) {
    v_qty[i]               = slice[i].sum_qty;
    v_ep[i]                = slice[i].sum_ep;
    v_disc[i]              = slice[i].sum_disc_price;
    v_charge[i]            = slice[i].sum_charge;
    v_disc_placeholder[i]  = slice[i].sum_disc;
    v_count[i]             = slice[i].count;
  }
  int sum_emit = 0;
  for (int i = 0; i < num_aggregates; ++i) {
    if (agg_mode[i] == sirius::AggregationType::COUNT_STAR ||
        agg_mode[i] == sirius::AggregationType::COUNT) {
      write_int64_col(i, v_count);
    } else if (agg_mode[i] == sirius::AggregationType::SUM) {
      // Map by emit order. The planner emits SUM(qty), SUM(ep),
      // SUM(ep*(1-disc)), SUM(ep*(1-disc)*(1+tax)) for TPC-H Q1.
      // Q1Tuple precomputed the multiplications producer-side and stored:
      //   slice.sum_qty        ← SUM(l_quantity)
      //   slice.sum_ep         ← SUM(l_extendedprice)
      //   slice.sum_disc_price ← SUM(l_extendedprice*(1-l_discount))
      //   slice.sum_charge     ← SUM(l_extendedprice*(1-l_discount)*(1+l_tax))
      const std::vector<double>* src = nullptr;
      switch (sum_emit++) {
        case 0: src = &v_qty;    break;
        case 1: src = &v_ep;     break;
        case 2: src = &v_disc;   break;  // sum_disc_price
        case 3: src = &v_charge; break;
        default: src = &v_disc_placeholder; break;
      }
      write_double_col(i, *src);
    } else {
      throw NotImplementedException(
          "magi_groupby M1: unsupported aggregate at index %d", i);
    }
  }
}

// ── Q5 dispatch (1 VARCHAR key + 1 SUM(DOUBLE)) ────────────────────────
bool IsQ5Shape(const vector<shared_ptr<GPUColumn>>& keys,
               const vector<shared_ptr<GPUColumn>>& aggs,
               int                                  num_group_keys,
               int                                  num_aggregates,
               sirius::AggregationType*             agg_mode)
{
  if (num_group_keys != 1) return false;
  if (num_aggregates  != 1) return false;
  if (!keys[0] || keys[0]->data_wrapper.type.id() != GPUColumnTypeId::VARCHAR) return false;
  if (!aggs[0] || aggs[0]->data_wrapper.type.id() != GPUColumnTypeId::FLOAT64) return false;
  if (agg_mode[0] != sirius::AggregationType::SUM) return false;
  return true;
}

magi_q5::PerGpuInputs BuildPerGpuInputs_Q5(
    const vector<shared_ptr<GPUColumn>>& keys,
    const vector<shared_ptr<GPUColumn>>& aggs)
{
  magi_q5::PerGpuInputs in{};
  in.n_filtered     = keys[0]->column_length;
  in.n_name_chars   = keys[0]->data_wrapper.data;
  in.n_name_offsets = keys[0]->data_wrapper.offset;
  in.d_revenue      = reinterpret_cast<const double*>(aggs[0]->data_wrapper.data);
  return in;
}

void WriteSliceToColumns_Q5(int                                gpu_id,
                            const std::vector<magi_q5::AggResultRow>& slice,
                            vector<shared_ptr<GPUColumn>>&     keys,
                            vector<shared_ptr<GPUColumn>>&     aggs,
                            GPUBufferManager*                  gbm)
{
  const size_t N = slice.size();

  // Empty-slice path: hand sirius nullptr-data columns so the operator's
  // combine branch picks a peer's non-empty column (same pattern as Q1).
  if (N == 0) {
    keys[0] = make_shared_ptr<GPUColumn>(0,
                                          GPUColumnType(GPUColumnTypeId::VARCHAR),
                                          /*data=*/nullptr,
                                          /*offset=*/nullptr,
                                          /*num_bytes=*/0,
                                          /*is_string_data=*/true,
                                          /*validity_mask=*/nullptr);
    keys[0]->row_id_count = 0;
    aggs[0] = make_shared_ptr<GPUColumn>(0,
                                          GPUColumnType(GPUColumnTypeId::FLOAT64),
                                          /*data=*/nullptr,
                                          /*validity_mask=*/nullptr);
    aggs[0]->row_id_count = 0;
    return;
  }

  // Q5 key column (VARCHAR n_name): unpack the packed first-8-chars back
  // into a variable-length char buffer + N+1 offsets. Names with len > 8
  // get truncated; for TPC-H 25 nations this only loses bytes 9.. of
  // INDONESIA, which shares no prefix with peers so SUM stays correct.
  // TODO: pass full name through wire tuple once we tackle longer keys.
  std::vector<uint64_t> h_offsets(N + 1);
  size_t total_bytes = 0;
  h_offsets[0] = 0;
  for (size_t i = 0; i < N; ++i) {
    uint64_t k = slice[i].n_name_packed;
    int len = 0;
    for (int b = 0; b < 8; ++b) {
      if (((k >> (b * 8)) & 0xff) == 0) break;
      ++len;
    }
    total_bytes += len;
    h_offsets[i + 1] = total_bytes;
  }
  std::vector<uint8_t> h_chars(total_bytes);
  for (size_t i = 0; i < N; ++i) {
    uint64_t k = slice[i].n_name_packed;
    size_t dst = h_offsets[i];
    for (int b = 0; b < 8; ++b) {
      uint8_t c = static_cast<uint8_t>((k >> (b * 8)) & 0xff);
      if (c == 0) break;
      h_chars[dst++] = c;
    }
  }

  auto* d_chars   = gbm->customCudaMalloc<uint8_t>(total_bytes == 0 ? 1 : total_bytes, gpu_id, false);
  auto* d_offsets = gbm->customCudaMalloc<uint64_t>(N + 1, gpu_id, false);
  if (total_bytes > 0)
    cudaMemcpy(d_chars, h_chars.data(), total_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, h_offsets.data(), (N + 1) * sizeof(uint64_t),
             cudaMemcpyHostToDevice);
  keys[0] = make_shared_ptr<GPUColumn>(N,
                                        GPUColumnType(GPUColumnTypeId::VARCHAR),
                                        d_chars,
                                        d_offsets,
                                        /*num_bytes=*/total_bytes,
                                        /*is_string_data=*/true,
                                        createNullMask(N));
  keys[0]->row_id_count = 0;

  // Q5 sum column.
  std::vector<double> v_sum(N);
  for (size_t i = 0; i < N; ++i) v_sum[i] = slice[i].sum_revenue;
  auto* d_buf = gbm->customCudaMalloc<double>(N, gpu_id, false);
  cudaMemcpy(d_buf, v_sum.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  aggs[0] = make_shared_ptr<GPUColumn>(N,
                                        GPUColumnType(GPUColumnTypeId::FLOAT64),
                                        reinterpret_cast<uint8_t*>(d_buf),
                                        createNullMask(N));
  aggs[0]->row_id_count = 0;
}

// ── Generic-path adapters ─────────────────────────────────────────────────
// These translate sirius's per-GPU GPUColumn views into the magi-side
// ColPack + AggOpEntry table, and emit the magi result slice back into
// sirius output columns. The translation is data-driven (column-type +
// AggregationType) — no per-query branches.

// Decide which `magi::distributed_hash_groupby_kernel<KeyT,…>` instantiation
// to invoke from the key column shape. Supported today:
//   - 1 INT      → KeyKind::INT32
//   - 1 VARCHAR  → KeyKind::UINT64 (first 8 bytes packed)
//   - 2 VARCHAR  → KeyKind::INT32  (Q1 (rf,ls) → (c0<<8)|c1 pack)
bool TryDeriveKeyKind(const vector<shared_ptr<GPUColumn>>& keys,
                      int                                  n_keys,
                      magi_generic::KeyKind&               out_kind)
{
  if (n_keys == 1) {
    auto id = keys[0]->data_wrapper.type.id();
    if (id == GPUColumnTypeId::INT32) { out_kind = magi_generic::KeyKind::INT32;  return true; }
    if (id == GPUColumnTypeId::VARCHAR){ out_kind = magi_generic::KeyKind::UINT64; return true; }
    return false;
  }
  if (n_keys == 2) {
    if (keys[0]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR &&
        keys[1]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
      out_kind = magi_generic::KeyKind::INT32;
      return true;
    }
  }
  return false;
}

magi_generic::PerGpuInputs BuildGenericInputs(
    const vector<shared_ptr<GPUColumn>>& keys,
    const vector<shared_ptr<GPUColumn>>& aggs,
    int                                  num_group_keys,
    int                                  num_aggregates)
{
  magi_generic::PerGpuInputs in{};
  in.n_filtered = keys[0]->column_length;
  in.cols       = {};

  // VARCHAR keys go to v_chars[0..]; INT keys go to i_cols[0..]. pack_key<KeyT>
  // specialisations (col_pack.cuh) read from these by fixed index, so the
  // ordering here must match the convention there.
  int v_idx = 0;
  int i_idx = 0;
  for (int k = 0; k < num_group_keys; ++k) {
    auto id = keys[k]->data_wrapper.type.id();
    if (id == GPUColumnTypeId::VARCHAR) {
      in.cols.v_chars  [v_idx] = keys[k]->data_wrapper.data;
      in.cols.v_offsets[v_idx] = keys[k]->data_wrapper.offset;
      ++v_idx;
    } else if (id == GPUColumnTypeId::INT32) {
      in.cols.i_cols[i_idx++] =
          reinterpret_cast<const int32_t*>(keys[k]->data_wrapper.data);
    }
  }
  in.cols.n_varchars = v_idx;
  in.cols.n_ints     = i_idx;

  // Aggregate input columns (sirius's projection has already materialised
  // the SUM(expr) inputs as plain DOUBLE columns; the BoundRef assertion in
  // gpu_physical_grouped_aggregate.cpp guarantees this). COUNT_STAR has no
  // backing column — it shows up as a nullptr in `aggs[]`.
  int d_idx = 0;
  for (int a = 0; a < num_aggregates; ++a) {
    if (!aggs[a] || aggs[a]->data_wrapper.data == nullptr) continue;
    if (aggs[a]->data_wrapper.type.id() != GPUColumnTypeId::FLOAT64) continue;
    in.cols.d_cols[d_idx++] =
        reinterpret_cast<const double*>(aggs[a]->data_wrapper.data);
  }
  in.cols.n_doubles = d_idx;
  return in;
}

std::vector<magi_ops::AggOpEntry> BuildAggOpsTable(
    const vector<shared_ptr<GPUColumn>>& aggs,
    int                                  num_aggregates,
    sirius::AggregationType*             agg_mode)
{
  std::vector<magi_ops::AggOpEntry> ops;
  ops.reserve(num_aggregates);

  // d_cols and dst_slot_idx are co-numbered: each surviving SUM/MIN/MAX
  // op gets one d_col slot and one AggSlot64::values[] slot. COUNT_STAR
  // gets its own dst slot but no src column (src_col_idx = -1).
  int d_idx   = 0;
  int dst_idx = 0;
  for (int a = 0; a < num_aggregates; ++a) {
    magi_ops::AggOpEntry e{};
    switch (agg_mode[a]) {
      case sirius::AggregationType::SUM:
        if (!aggs[a] || aggs[a]->data_wrapper.data == nullptr) continue;
        e.kind         = magi_ops::AggKind::SUM_DOUBLE;
        e.src_col_idx  = static_cast<int8_t>(d_idx++);
        e.dst_slot_idx = static_cast<int8_t>(dst_idx++);
        break;
      case sirius::AggregationType::COUNT_STAR:
      case sirius::AggregationType::COUNT:
        e.kind         = (agg_mode[a] == sirius::AggregationType::COUNT_STAR)
                            ? magi_ops::AggKind::COUNT_STAR
                            : magi_ops::AggKind::COUNT_VALID;
        e.src_col_idx  = -1;
        e.dst_slot_idx = static_cast<int8_t>(dst_idx++);
        break;
      case sirius::AggregationType::MIN:
        if (!aggs[a] || aggs[a]->data_wrapper.data == nullptr) continue;
        e.kind         = magi_ops::AggKind::MIN_DOUBLE;
        e.src_col_idx  = static_cast<int8_t>(d_idx++);
        e.dst_slot_idx = static_cast<int8_t>(dst_idx++);
        break;
      case sirius::AggregationType::MAX:
        if (!aggs[a] || aggs[a]->data_wrapper.data == nullptr) continue;
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
    uint64_t k = slice[i].key_as_u64;
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
    uint64_t k = slice[i].key_as_u64;
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

void WriteGenericSliceToColumns(int                                              gpu_id,
                                const std::vector<magi_generic::AggResultRow>&   slice,
                                vector<shared_ptr<GPUColumn>>&                   keys,
                                vector<shared_ptr<GPUColumn>>&                   aggs,
                                int                                              num_group_keys,
                                int                                              num_aggregates,
                                sirius::AggregationType*                         agg_mode,
                                magi_generic::KeyKind                            key_kind,
                                GPUBufferManager*                                gbm)
{
  const size_t N = slice.size();

  // ── Emit key columns ───────────────────────────────────────────────────
  // KeyKind selects how to unpack the 64-bit key into one or more output
  // columns:
  //   INT32 + 2 VARCHAR keys → bytes [0] and [1] of key as VARCHAR(1) each
  //   INT32 + 1 INT key      → cast directly to INT32
  //   UINT64 + 1 VARCHAR     → bytes [0..8) of key as VARCHAR up to 8
  if (key_kind == magi_generic::KeyKind::INT32 && num_group_keys == 2 &&
      keys[0]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR &&
      keys[1]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
    // Q1-shape: int32 key = (c0<<8)|c1 — but pack_key<int32_t> reads c0 from
    // v_chars[0] (i.e., the FIRST input varchar) and shifts c0 << 8. So:
    //   bit 8..15 = c0 (= keys[0])
    //   bit 0..7  = c1 (= keys[1])
    EmitVarcharKeyFromPacked(gpu_id, N, slice, keys[0],
                              /*byte_offset=*/1, /*byte_len=*/1, gbm);
    EmitVarcharKeyFromPacked(gpu_id, N, slice, keys[1],
                              /*byte_offset=*/0, /*byte_len=*/1, gbm);
  } else if (key_kind == magi_generic::KeyKind::UINT64 && num_group_keys == 1 &&
             keys[0]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
    EmitVarcharKeyFromPacked(gpu_id, N, slice, keys[0],
                              /*byte_offset=*/0, /*byte_len=*/8, gbm);
  } else if (key_kind == magi_generic::KeyKind::INT32 && num_group_keys == 1 &&
             keys[0]->data_wrapper.type.id() == GPUColumnTypeId::INT32) {
    // 1 INT key — emit as int32 column directly.
    if (N == 0) {
      keys[0] = make_shared_ptr<GPUColumn>(0,
                                           GPUColumnType(GPUColumnTypeId::INT32),
                                           /*data=*/nullptr,
                                           /*validity_mask=*/nullptr);
      keys[0]->row_id_count = 0;
    } else {
      std::vector<int32_t> v(N);
      for (size_t i = 0; i < N; ++i)
        v[i] = static_cast<int32_t>(slice[i].key_as_u64);
      auto* d_buf = gbm->customCudaMalloc<int32_t>(N, gpu_id, false);
      cudaMemcpy(d_buf, v.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
      keys[0] = make_shared_ptr<GPUColumn>(N,
                                           GPUColumnType(GPUColumnTypeId::INT32),
                                           reinterpret_cast<uint8_t*>(d_buf),
                                           createNullMask(N));
      keys[0]->row_id_count = 0;
    }
  } else {
    throw NotImplementedException(
        "magi_groupby (generic): unsupported key emit shape "
        "(KeyKind=%d, n_keys=%d, type0=%d)",
        static_cast<int>(key_kind), num_group_keys,
        static_cast<int>(keys[0]->data_wrapper.type.id()));
  }

  // ── Emit aggregate columns ─────────────────────────────────────────────
  // The dst_slot_idx assignment in BuildAggOpsTable matches the order
  // sirius hands us aggregates, so we walk `aggregates` in lockstep.
  int dst_idx = 0;
  for (int a = 0; a < num_aggregates; ++a) {
    switch (agg_mode[a]) {
      case sirius::AggregationType::SUM:
      case sirius::AggregationType::MIN:
      case sirius::AggregationType::MAX:
        EmitDoubleAggCol(gpu_id, N, slice, dst_idx++, aggs[a], gbm);
        break;
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

void Run(int                                gpu_id,
         vector<shared_ptr<GPUColumn>>&     group_by_keys,
         vector<shared_ptr<GPUColumn>>&     aggregate_keys,
         int                                num_group_keys,
         int                                num_aggregates,
         sirius::AggregationType*           agg_mode)
{
  // ── Legacy A/B knob ────────────────────────────────────────────────────
  // MAGI_LEGACY=1 falls back to the per-query Q1/Q5 paths so we can
  // bit-compare them against the generic library. Production runs leave
  // it unset.
  static const bool legacy = std::getenv("MAGI_LEGACY") != nullptr;
  if (legacy) {
    if (IsQ5Shape(group_by_keys, aggregate_keys,
                  num_group_keys, num_aggregates, agg_mode)) {
      auto in = BuildPerGpuInputs_Q5(group_by_keys, aggregate_keys);
      std::vector<magi_q5::AggResultRow> slice;
      magi_q5::Q5MagiRunPerGpu(gpu_id, in, slice);
      WriteSliceToColumns_Q5(gpu_id, slice,
                             group_by_keys, aggregate_keys,
                             &GPUBufferManager::GetInstance());
      return;
    }
    if (IsQ1Shape(group_by_keys, aggregate_keys,
                  num_group_keys, num_aggregates, agg_mode)) {
      auto in = BuildPerGpuInputs(group_by_keys, aggregate_keys, num_aggregates);
      std::vector<magi_q1::AggResultRow> slice;
      magi_q1::Q1MagiRunPerGpu(gpu_id, in, slice);
      WriteSliceToColumns(gpu_id, slice,
                          group_by_keys, aggregate_keys,
                          num_aggregates, agg_mode,
                          &GPUBufferManager::GetInstance());
      return;
    }
    throw NotImplementedException(
        "MAGI_LEGACY=1 only covers Q1/Q5 shapes — unset MAGI_LEGACY to use "
        "the generic distributed_hash_groupby path.");
  }

  // ── Generic path (default) ─────────────────────────────────────────────
  magi_generic::KeyKind key_kind;
  if (!TryDeriveKeyKind(group_by_keys, num_group_keys, key_kind)) {
    throw NotImplementedException(
        "magi_groupby (generic): unsupported key shape "
        "(n_keys=%d, first_type=%d). Add a pack_key<KeyT> specialisation + "
        "KeyKind value to extend.",
        num_group_keys,
        group_by_keys.empty() ? -1
        : static_cast<int>(group_by_keys[0]->data_wrapper.type.id()));
  }

  auto in   = BuildGenericInputs(group_by_keys, aggregate_keys,
                                  num_group_keys, num_aggregates);
  auto ops  = BuildAggOpsTable(aggregate_keys, num_aggregates, agg_mode);
  std::vector<magi_generic::AggResultRow> slice;
  magi_generic::distributed_hash_groupby_run_per_gpu(
      gpu_id, in, ops, key_kind, slice);

  WriteGenericSliceToColumns(gpu_id, slice,
                             group_by_keys, aggregate_keys,
                             num_group_keys, num_aggregates, agg_mode,
                             key_kind,
                             &GPUBufferManager::GetInstance());
}

}  // namespace magi_groupby
}  // namespace duckdb
