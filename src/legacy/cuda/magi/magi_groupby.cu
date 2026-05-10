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
    auto* d_chars   = gbm->customCudaMalloc<uint8_t> (N == 0 ? 1 : N,     gpu_id, false);
    auto* d_offsets = gbm->customCudaMalloc<uint64_t>(N + 1,              gpu_id, false);
    std::vector<uint8_t>  h_chars(N == 0 ? 1 : N);
    std::vector<uint64_t> h_offsets(N + 1);
    for (size_t i = 0; i < N; ++i) {
      h_chars[i]   = static_cast<uint8_t>(kcol == 0 ? slice[i].rf : slice[i].ls);
      h_offsets[i] = i;
    }
    h_offsets[N] = N;
    if (N > 0) {
      cudaMemcpy(d_chars,   h_chars.data(),   N,                 cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_offsets, h_offsets.data(), (N + 1) * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    auto* mask = createNullMask(N == 0 ? 1 : N);  // ALL_VALID
    keys[kcol] = make_shared_ptr<GPUColumn>(N,
                                            GPUColumnType(GPUColumnTypeId::VARCHAR),
                                            d_chars,
                                            d_offsets,
                                            /*num_bytes=*/N,
                                            /*is_string_data=*/true,
                                            mask);
    // GPUColumn ctor leaves row_id_count uninitialised; sirius downstream
    // (PROJECTION's HandleMaterializeExpression) reads it. Force-zero here.
    keys[kcol]->row_id_count = 0;
  }

  // ── Aggregates: 4 SUM doubles (in planner-emit order) + COUNT_STAR ─────
  // Helper: write a vector of doubles to device and replace the column.
  auto write_double_col = [&](int agg_idx, const std::vector<double>& host) {
    auto* d_buf = gbm->customCudaMalloc<double>(N == 0 ? 1 : N, gpu_id, false);
    if (N > 0) {
      cudaMemcpy(d_buf, host.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    }
    aggs[agg_idx] = make_shared_ptr<GPUColumn>(N,
                                               GPUColumnType(GPUColumnTypeId::FLOAT64),
                                               reinterpret_cast<uint8_t*>(d_buf),
                                               createNullMask(N == 0 ? 1 : N));
    aggs[agg_idx]->row_id_count = 0;
  };
  auto write_int64_col = [&](int agg_idx, const std::vector<uint64_t>& host) {
    auto* d_buf = gbm->customCudaMalloc<uint64_t>(N == 0 ? 1 : N, gpu_id, false);
    if (N > 0) {
      cudaMemcpy(d_buf, host.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    aggs[agg_idx] = make_shared_ptr<GPUColumn>(N,
                                               GPUColumnType(GPUColumnTypeId::INT64),
                                               reinterpret_cast<uint8_t*>(d_buf),
                                               createNullMask(N == 0 ? 1 : N));
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

}  // namespace

void Run(int                                gpu_id,
         vector<shared_ptr<GPUColumn>>&     group_by_keys,
         vector<shared_ptr<GPUColumn>>&     aggregate_keys,
         int                                num_group_keys,
         int                                num_aggregates,
         sirius::AggregationType*                   agg_mode)
{
  if (!IsQ1Shape(group_by_keys, aggregate_keys,
                 num_group_keys, num_aggregates, agg_mode)) {
    throw NotImplementedException(
        "magi_groupby M1: only TPC-H Q1 input shape "
        "(2 VARCHAR(1) keys + SUM(DOUBLE)/COUNT aggregates) is wired up; "
        "got %d keys, %d aggregates with first key type id=%d",
        num_group_keys, num_aggregates,
        group_by_keys.empty() ? -1
        : static_cast<int>(group_by_keys[0]->data_wrapper.type.id()));
  }

  auto in = BuildPerGpuInputs(group_by_keys, aggregate_keys, num_aggregates);
  std::vector<magi_q1::AggResultRow> slice;
  magi_q1::Q1MagiRunPerGpu(gpu_id, in, slice);

  WriteSliceToColumns(gpu_id, slice,
                      group_by_keys, aggregate_keys,
                      num_aggregates, agg_mode,
                      &GPUBufferManager::GetInstance());
}

}  // namespace magi_groupby
}  // namespace duckdb
