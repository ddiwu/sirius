/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "operator/gpu_physical_order.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"
#include "operator/gpu_materialize.hpp"

#include <cuda_runtime.h>

#include <array>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace duckdb {
using sirius::OrderByType;

void HandleOrderBy(vector<shared_ptr<GPUColumn>>& order_by_keys,
                   vector<shared_ptr<GPUColumn>>& projection_columns,
                   const vector<BoundOrderByNode>& orders,
                   uint64_t num_projections)
{
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  OrderByType* order_by_type = gpuBufferManager->customCudaHostAlloc<OrderByType>(orders.size());
  for (int order_idx = 0; order_idx < orders.size(); order_idx++) {
    if (orders[order_idx].type == OrderType::ASCENDING) {
      order_by_type[order_idx] = OrderByType::ASCENDING;
    } else {
      order_by_type[order_idx] = OrderByType::DESCENDING;
    }
  }

  cudf_orderby(order_by_keys, projection_columns, orders.size(), num_projections, order_by_type);
}

GPUPhysicalOrder::GPUPhysicalOrder(vector<LogicalType> types,
                                   vector<BoundOrderByNode> orders,
                                   vector<idx_t> projections_p,
                                   idx_t estimated_cardinality,
                                   bool is_index_sort_p)
  : GPUPhysicalOperator(PhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

SourceResultType GPUPhysicalOrder::GetData(GPUIntermediateRelation& output_relation) const
{
  auto start        = std::chrono::high_resolution_clock::now();
  auto& sort_result = runtime_state<OrderRuntimeState>(sirius_current_gpu).sort_result;
  for (int col = 0; col < sort_result->columns.size(); col++) {
    SIRIUS_LOG_DEBUG("Writing order by result to column {}", col);
    output_relation.columns[col] = sort_result->columns[col];
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Order by GetData time: {:.2f} ms", duration.count() / 1000.0);
  return SourceResultType::FINISHED;
}

SinkResultType GPUPhysicalOrder::Sink(GPUIntermediateRelation& input_relation) const
{
  auto start                         = std::chrono::high_resolution_clock::now();
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

  vector<shared_ptr<GPUColumn>> order_by_keys(orders.size());
  vector<shared_ptr<GPUColumn>> projection_columns(projections.size());

  for (int projection_idx = 0; projection_idx < projections.size(); projection_idx++) {
    auto input_idx = projections[projection_idx];
    projection_columns[projection_idx] =
      HandleMaterializeExpression(input_relation.columns[input_idx], gpuBufferManager);
    input_relation.columns[input_idx] = projection_columns[projection_idx];
  }

  for (int order_idx = 0; order_idx < orders.size(); order_idx++) {
    auto& expr = *orders[order_idx].expression;
    if (expr.expression_class != ExpressionClass::BOUND_REF) {
      throw NotImplementedException("Order by expression not supported");
    }
    auto input_idx = expr.Cast<BoundReferenceExpression>().index;
    order_by_keys[order_idx] =
      HandleMaterializeExpression(input_relation.columns[input_idx], gpuBufferManager);
  }

  if (order_by_keys[0]->column_length > INT32_MAX) {
    throw NotImplementedException(
      "Order by with column length greater than INT32_MAX is not supported");
  }

  HandleOrderBy(order_by_keys, projection_columns, orders, projections.size());

  // Per-GPU slot: this worker sorted only its own partition. Lazy-init here so
  // each GPU's sort_result is independent (no cross-worker clobber).
  auto& rstate = runtime_state<OrderRuntimeState>(sirius_current_gpu);
  if (!rstate.sort_result) {
    rstate.sort_result = make_shared_ptr<GPUIntermediateRelation>(projections.size());
  }
  auto& sort_result = rstate.sort_result;
  for (int col = 0; col < projections.size(); col++) {
    if (sort_result->columns[col] == nullptr || sort_result->columns[col]->column_length == 0 ||
        sort_result->columns[col]->data_wrapper.data == nullptr) {
      sort_result->columns[col]               = projection_columns[col];
      sort_result->columns[col]->row_ids      = nullptr;
      sort_result->columns[col]->row_id_count = 0;
    } else if (sort_result->columns[col] != nullptr && projection_columns[col]->column_length > 0 &&
               projection_columns[col]->data_wrapper.data != nullptr) {
      throw NotImplementedException("Order by with partially NULL values is not supported");
    }
  }

  MaybeGpuMergeAcrossGpus(rstate);

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Order by Sink time: {:.2f} ms", duration.count() / 1000.0);
  return SinkResultType::FINISHED;
}

// ── Cross-GPU merge of the per-GPU sorted runs, ON a GPU ─────────────────────
// The serial host k-way merge dominates large ORDER BY (381K rows: host merge
// ~24ms vs per-GPU cudf sort 0.45ms). When the TOTAL row count is large, every
// worker publishes its sorted run; ONE worker (lowest gpu id) copies the peer
// runs over NVLink to its device (a few MB → sub-ms) and cudf::merge's them;
// the peers replace their result with an empty run. The collector then sees a
// single already-sorted run and just concatenates. Small totals keep the host
// merge (rendezvous cost not worth it; TOP_N always host-merges: its per-GPU
// output is ≤ limit+offset rows).
//
// Worker-consistency: eligibility is plan-derived (identical on every worker)
// and the size decision is made from the SAME exchanged totals, so all workers
// take the same branch (divergence would strand the barrier).
void GPUPhysicalOrder::MaybeGpuMergeAcrossGpus(OrderRuntimeState& rstate) const
{
  GPUBufferManager* gbm = &(GPUBufferManager::GetInstance());
  const int num_gpus    = static_cast<int>(gbm->tables_per_gpu.size());
  if (num_gpus <= 1) { return; }
  static const bool disabled = std::getenv("MAGI_NO_GPU_MERGE") != nullptr;
  if (disabled) { return; }

  // Plan-derived eligibility — identical on every worker.
  //  - all output columns fixed-width (VARCHAR gather over P2P not done yet)
  //  - every order key is a BOUND_REF that maps to an output column (so the
  //    merged table itself carries the keys for cudf::merge)
  vector<idx_t> key_cols;
  for (auto& ord : orders) {
    if (ord.expression->GetExpressionClass() != ExpressionClass::BOUND_REF) { return; }
    const idx_t in_idx = ord.expression->Cast<BoundReferenceExpression>().index;
    idx_t out_col = DConstants::INVALID_INDEX;
    for (idx_t j = 0; j < projections.size(); j++) {
      if (projections[j] == in_idx) { out_col = j; break; }
    }
    if (out_col == DConstants::INVALID_INDEX) { return; }
    key_cols.push_back(out_col);
  }
  for (auto& col : rstate.sort_result->columns) {
    if (col && col->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) { return; }
  }

  // Rendezvous: publish run + row count, decide from the exchanged totals.
  struct Slot {
    GPUIntermediateRelation* run;
    uint64_t                 rows;
  };
  static std::array<Slot, 8>             slots;
  static std::unique_ptr<std::barrier<>> bar;
  static std::once_flag                  bar_once;
  std::call_once(bar_once, [&] { bar = std::make_unique<std::barrier<>>(num_gpus); });

  const int g = sirius_current_gpu;
  // Publish AFTER our sort kernels landed (peers read these buffers over UVA).
  cudaDeviceSynchronize();
  const uint64_t my_rows =
    rstate.sort_result->columns.empty() || !rstate.sort_result->columns[0]
      ? 0
      : rstate.sort_result->columns[0]->column_length;
  slots[g] = {rstate.sort_result.get(), my_rows};
  bar->arrive_and_wait();  // every run published

  uint64_t total = 0;
  for (int i = 0; i < num_gpus; ++i) { total += slots[i].rows; }
  static const uint64_t min_rows = [] {
    const char* e = std::getenv("MAGI_GPU_MERGE_MIN");
    return e ? std::strtoull(e, nullptr, 10) : uint64_t(65536);
  }();
  static const bool phase_time = std::getenv("MAGI_PHASE_TIME") != nullptr;

  if (total >= min_rows) {
    const auto t0 = std::chrono::steady_clock::now();
    // Lowest-id worker with rows merges; everyone else empties its run.
    int merger = 0;
    if (g == merger) {
      // P2P so the D2D pulls ride NVLink instead of host staging.
      static std::array<std::once_flag, 64> peer_once;
      for (int i = 0; i < num_gpus; ++i) {
        if (i == g) { continue; }
        std::call_once(peer_once[g * 8 + i], [&] {
          cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
          if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
            SIRIUS_LOG_DEBUG("gpu-merge: peer access {} -> {} unavailable ({})", g, i, (int)e);
          }
          (void)cudaGetLastError();
        });
      }
      const idx_t ncols = rstate.sort_result->columns.size();
      vector<vector<shared_ptr<GPUColumn>>> runs;
      for (int i = 0; i < num_gpus; ++i) {
        if (slots[i].rows == 0) { continue; }
        vector<shared_ptr<GPUColumn>> cols(ncols);
        for (idx_t c = 0; c < ncols; ++c) {
          auto& src = slots[i].run->columns[c];
          if (i == g) {
            cols[c] = src;  // local run: zero copy
            continue;
          }
          const size_t esz   = src->data_wrapper.getColumnTypeSize();
          const size_t bytes = src->column_length * esz;
          uint8_t*     data  = gbm->customCudaMalloc<uint8_t>(bytes, g, 0);
          cudaMemcpy(data, src->data_wrapper.data, bytes, cudaMemcpyDeviceToDevice);
          cudf::bitmask_type* mask = nullptr;
          if (src->data_wrapper.validity_mask != nullptr) {
            const size_t mb = getMaskBytesSize(src->column_length);
            mask            = reinterpret_cast<cudf::bitmask_type*>(
              gbm->customCudaMalloc<uint8_t>(mb, g, 0));
            cudaMemcpy(mask, src->data_wrapper.validity_mask, mb, cudaMemcpyDeviceToDevice);
          }
          cols[c] = make_shared_ptr<GPUColumn>(
            src->column_length, src->data_wrapper.type, data, mask);
        }
        runs.push_back(std::move(cols));
      }
      if (runs.size() > 1) {
        OrderByType* obt = gbm->customCudaHostAlloc<OrderByType>(orders.size());
        for (idx_t i = 0; i < orders.size(); i++) {
          obt[i] = orders[i].type == OrderType::ASCENDING ? OrderByType::ASCENDING
                                                          : OrderByType::DESCENDING;
        }
        vector<shared_ptr<GPUColumn>> merged = rstate.sort_result->columns;
        cudf_merge_sorted(runs, key_cols, obt, orders.size(), ncols, merged);
        rstate.sort_result->columns = std::move(merged);
      } else if (runs.size() == 1 && slots[g].rows == 0) {
        // All rows live on ONE peer: pull its run over as ours (already sorted).
        rstate.sort_result->columns = std::move(runs[0]);
      }
    }
    bar->arrive_and_wait();  // merger done reading peer buffers
    if (g != merger) {
      // Peer: hand everything to the merger; emit an empty run downstream.
      // MUST happen only after the barrier above — replacing these shared_ptrs
      // while the merger is still reading the published relation raced it into
      // copying 0-length columns (rmm alloc(0) -> nullptr -> worker throw).
      for (auto& col : rstate.sort_result->columns) {
        if (col) {
          col = make_shared_ptr<GPUColumn>(0, col->data_wrapper.type, nullptr, nullptr);
        }
      }
    }
    if (phase_time && g == merger) {
      std::fprintf(stderr, "[gpu-merge gpu=%d] rows=%llu merge=%.2fms\n", g,
                   (unsigned long long)total,
                   std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count());
    }
  } else {
    bar->arrive_and_wait();  // keep barrier phases aligned across workers
  }
}

}  // namespace duckdb
