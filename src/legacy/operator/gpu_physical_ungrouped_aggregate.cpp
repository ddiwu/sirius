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

#include "operator/gpu_physical_ungrouped_aggregate.hpp"

#include "operator/gpu_physical_grouped_aggregate.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"
#include "operator/gpu_materialize.hpp"
#include "operator/gpu_physical_result_collector.hpp"
#include "operator/gpu_physical_table_scan.hpp"

#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

namespace duckdb {
using sirius::AggregationType;

void HandleAggregateExpressionCuDF(vector<shared_ptr<GPUColumn>>& aggregate_keys,
                                   GPUBufferManager* gpuBufferManager,
                                   const vector<unique_ptr<Expression>>& aggregates)
{
  AggregationType* agg_mode =
    gpuBufferManager->customCudaHostAlloc<AggregationType>(aggregates.size());
  SIRIUS_LOG_DEBUG("Handling ungrouped aggregate expression");
  for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
    auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
    if (expr.IsDistinct()) {
      if (expr.function.name.compare("count") == 0) {
        agg_mode[agg_idx] = AggregationType::COUNT_DISTINCT;
      } else {
        SIRIUS_LOG_DEBUG("Aggregate function (distinct)  not supported: {}", expr.function.name);
        throw NotImplementedException("Aggregate function (distinct) not supported");
      }
    } else {
      if (expr.function.name.compare("count") == 0 &&
          aggregate_keys[agg_idx]->data_wrapper.data == nullptr &&
          aggregate_keys[agg_idx]->column_length == 0) {
        agg_mode[agg_idx] = AggregationType::COUNT;
      } else if (expr.function.name.compare("sum") == 0 &&
                 aggregate_keys[agg_idx]->data_wrapper.data == nullptr &&
                 aggregate_keys[agg_idx]->column_length == 0) {
        agg_mode[agg_idx] = AggregationType::SUM;
      } else if (expr.function.name.compare("sum") == 0 &&
                 aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
        agg_mode[agg_idx] = AggregationType::SUM;
      } else if (expr.function.name.compare("sum_no_overflow") == 0 &&
                 aggregate_keys[agg_idx]->data_wrapper.data == nullptr &&
                 aggregate_keys[agg_idx]->column_length == 0) {
        agg_mode[agg_idx] = AggregationType::SUM;
      } else if (expr.function.name.compare("sum_no_overflow") == 0 &&
                 aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
        agg_mode[agg_idx] = AggregationType::SUM;
        if (aggregate_keys[agg_idx]->data_wrapper.type.id() == GPUColumnTypeId::INT32) {
          SIRIUS_LOG_DEBUG("Converting INT32 to INT64 for sum_no_overflow");
          uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(
            aggregate_keys[agg_idx]->column_length, 0, 0);
          convertInt32ToInt64(aggregate_keys[agg_idx]->data_wrapper.data,
                              reinterpret_cast<uint8_t*>(temp),
                              aggregate_keys[agg_idx]->column_length);
          aggregate_keys[agg_idx]->data_wrapper.data = reinterpret_cast<uint8_t*>(temp);
          aggregate_keys[agg_idx]->data_wrapper.type = GPUColumnType(GPUColumnTypeId::INT64);
          aggregate_keys[agg_idx]->data_wrapper.num_bytes =
            aggregate_keys[agg_idx]->data_wrapper.num_bytes * 2;
        }
      } else if (expr.function.name.compare("avg") == 0 &&
                 (aggregate_keys[agg_idx]->data_wrapper.data != nullptr ||
                  aggregate_keys[agg_idx]->column_length == 0)) {
        agg_mode[agg_idx] = AggregationType::AVERAGE;
      } else if (expr.function.name.compare("max") == 0 &&
                 (aggregate_keys[agg_idx]->data_wrapper.data != nullptr ||
                  aggregate_keys[agg_idx]->column_length == 0)) {
        agg_mode[agg_idx] = AggregationType::MAX;
      } else if (expr.function.name.compare("min") == 0 &&
                 (aggregate_keys[agg_idx]->data_wrapper.data != nullptr ||
                  aggregate_keys[agg_idx]->column_length == 0)) {
        agg_mode[agg_idx] = AggregationType::MIN;
      } else if (expr.function.name.compare("count_star") == 0 &&
                 aggregate_keys[agg_idx]->data_wrapper.data == nullptr) {
        agg_mode[agg_idx] = AggregationType::COUNT_STAR;
      } else if (expr.function.name.compare("count") == 0 &&
                 aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
        agg_mode[agg_idx] = AggregationType::COUNT;
      } else if (expr.function.name.compare("first") == 0) {
        agg_mode[agg_idx] = AggregationType::FIRST;
      } else {
        SIRIUS_LOG_DEBUG("Aggregate function (not distinct) not supported: {}", expr.function.name);
        throw NotImplementedException("Aggregate function (not distinct) not supported");
      }
    }
  }

  cudf_aggregate(aggregate_keys, aggregates.size(), agg_mode);

  // Duckdb requires count(distinct) returns int64
  for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
    if (agg_mode[agg_idx] == AggregationType::COUNT_DISTINCT &&
        aggregate_keys[agg_idx]->data_wrapper.type.id() != GPUColumnTypeId::INT64) {
      auto from_cudf_column_view = aggregate_keys[agg_idx]->convertToCudfColumn();
      auto to_cudf_type          = cudf::data_type(cudf::type_id::INT64);
      auto to_cudf_column        = cudf::cast(from_cudf_column_view,
                                       to_cudf_type,
                                       rmm::cuda_stream_default,
                                       GPUBufferManager::GetInstance().get_mr_ref());
      aggregate_keys[agg_idx]->setFromCudfColumn(
        *to_cudf_column, false, nullptr, 0, gpuBufferManager);
    }
  }
}

GPUPhysicalUngroupedAggregate::GPUPhysicalUngroupedAggregate(
  vector<LogicalType> types,
  vector<unique_ptr<Expression>> expressions,
  idx_t estimated_cardinality,
  TupleDataValidityType distinct_validity)
  : GPUPhysicalOperator(
      PhysicalOperatorType::UNGROUPED_AGGREGATE, std::move(types), estimated_cardinality),
    aggregates(std::move(expressions))
{
  distinct_collection_info = DistinctAggregateCollectionInfo::Create(aggregates);
  // Per-GPU aggregation_result lives in UngroupedAggregateRuntimeState and is
  // lazy-initialised on first Sink for each GPU.
  if (!distinct_collection_info) { return; }
  distinct_data = make_uniq<DistinctAggregateData>(*distinct_collection_info, distinct_validity);
}

SinkResultType GPUPhysicalUngroupedAggregate::Sink(GPUIntermediateRelation& input_relation) const
{
  SIRIUS_LOG_DEBUG("Performing ungrouped aggregation");
  auto start = std::chrono::high_resolution_clock::now();

  // Replicated-input guard (see GPUPhysicalGroupedAggregate::Sink): all GPUs
  // holding identical full input would each contribute the whole aggregate
  // -> values xN after the cross-GPU merge. Fail loudly -> DuckDB fallback.
  if (GPUBufferManager::GetMaxGpus() > 1 && !children.empty() &&
      SubtreeAllReplicated(*children[0])) {
    throw NotImplementedException(
      "Multi-GPU ungrouped aggregate over fully-replicated input (small tables only) is not "
      "supported; falling back to DuckDB");
  }
  vector<shared_ptr<GPUColumn>> aggregate_column(aggregates.size());
  for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
    aggregate_column[aggr_idx] = nullptr;
  }

  if (distinct_data) { MaterializeDistinctInput(input_relation, aggregate_column); }

  uint64_t column_size = 0;
  for (int i = 0; i < input_relation.columns.size(); i++) {
    if (input_relation.columns[i] != nullptr) {
      if (input_relation.columns[i]->row_ids != nullptr) {
        column_size = input_relation.columns[i]->row_id_count;
      } else if (input_relation.columns[i]->data_wrapper.data != nullptr) {
        column_size = input_relation.columns[i]->column_length;
      }
      break;
    } else {
      throw NotImplementedException("Input relation is null");
    }
  }

  // Diagnostic (debug level): per-GPU aggregate-input fingerprint. Two workers
  // logging IDENTICAL row counts here (for a partitioned input) points at an
  // upstream partition-routing problem, not at the aggregate itself.
  if (interior_cross_gpu_merge && GPUBufferManager::GetMaxGpus() > 1) {
    SIRIUS_LOG_DEBUG(
      "Interior agg SINK-INPUT gpu={} rows={} col0_data={} col0_rowids={}",
      sirius_current_gpu, column_size,
      (void*)(input_relation.columns[0] ? input_relation.columns[0]->data_wrapper.data
                                        : nullptr),
      (void*)(input_relation.columns[0] ? input_relation.columns[0]->row_ids : nullptr));
  }

  idx_t payload_idx                  = 0;
  idx_t next_payload_idx             = 0;
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

  for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
    D_ASSERT(aggregates[aggr_idx]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
    auto& aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();

    payload_idx      = next_payload_idx;
    next_payload_idx = payload_idx + aggregate.children.size();

    if (aggregate.IsDistinct()) { continue; }

    if (aggregate.filter) {
      auto& bound_ref_expr = aggregate.filter->Cast<BoundReferenceExpression>();
      SIRIUS_LOG_DEBUG("Reading filter column from index {}", bound_ref_expr.index);
    }

    idx_t payload_cnt = 0;

    SIRIUS_LOG_DEBUG("Aggregate type: {}", aggregate.function.name);
    if (aggregate.children.size() > 1)
      throw NotImplementedException("Aggregates with multiple children not supported yet");
    for (idx_t i = 0; i < aggregate.children.size(); ++i) {
      for (auto& child_expr : aggregate.children) {
        D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
        SIRIUS_LOG_DEBUG(
          "Reading aggregation column from index {} and passing it to index {} in aggregation "
          "result",
          payload_idx + payload_cnt,
          aggr_idx);
        aggregate_column[aggr_idx] = HandleMaterializeExpression(
          input_relation.columns[payload_idx + payload_cnt], gpuBufferManager);
        payload_cnt++;
      }
    }
  }

  for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
    auto& aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();
    // here we probably have count(*) or sum(*) or something like that
    if (aggregate.children.size() == 0) {
      SIRIUS_LOG_DEBUG("Passing * aggregate to index {} in aggregation result", aggr_idx);
      aggregate_column[aggr_idx] = make_shared_ptr<GPUColumn>(
        column_size, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr);
    }
  }

  // Stash this batch; FinalizeSink (called by the executor at this op's LAST
  // sinking pipeline) concatenates the batches and aggregates once. A
  // RIGHT/OUTER join upstream legally sinks twice — the old single-batch
  // Sink threw here and the whole query fell back (Q17's root sum sits above
  // the delim's RIGHT-family join).
  {
    auto& stash_state = runtime_state<UngroupedAggregateRuntimeState>(sirius_current_gpu);
    stash_state.pending.push_back(std::move(aggregate_column));
    stash_state.pending_sizes.push_back(column_size);
  }

  auto sink_end      = std::chrono::high_resolution_clock::now();
  auto sink_duration = std::chrono::duration_cast<std::chrono::microseconds>(sink_end - start);
  SIRIUS_LOG_DEBUG("Ungrouped aggregate Sink time: {:.2f} ms", sink_duration.count() / 1000.0);
  return SinkResultType::FINISHED;
}

void GPUPhysicalUngroupedAggregate::FinalizeSink() const
{
  auto start = std::chrono::high_resolution_clock::now();
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  auto& rstate = runtime_state<UngroupedAggregateRuntimeState>(sirius_current_gpu);
  if (rstate.pending.empty()) { return; }

  // Drop empty batches (a RIGHT-family probe pipeline legally emits a 0-row
  // batch); keep one batch if all are empty so the aggregate still produces
  // its empty-input value (NULL for SUM, 0 for COUNT).
  vector<size_t> keep;
  for (size_t b = 0; b < rstate.pending.size(); ++b) {
    if (rstate.pending_sizes[b] > 0) { keep.push_back(b); }
  }
  if (keep.empty()) { keep.push_back(0); }

  if (keep.size() > 1 && interior_cross_gpu_merge) {
    throw NotImplementedException(
      "Multi-batch INTERIOR ungrouped aggregate not supported yet (falls back)");
  }

  vector<shared_ptr<GPUColumn>> aggregate_column(aggregates.size());
  uint64_t total_size = 0;
  for (size_t k : keep) { total_size += rstate.pending_sizes[k]; }
  for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); ++aggr_idx) {
    auto& first = rstate.pending[keep[0]][aggr_idx];
    const bool placeholder = !first || first->data_wrapper.data == nullptr;
    if (placeholder) {
      // count(*)-style placeholder: no data, the LENGTH is the value.
      aggregate_column[aggr_idx] = make_shared_ptr<GPUColumn>(
        total_size, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr);
    } else {
      auto acc = first;
      for (size_t i = 1; i < keep.size(); ++i) {
        acc = CombineColumns(acc, rstate.pending[keep[i]][aggr_idx], gpuBufferManager);
      }
      aggregate_column[aggr_idx] = acc;
    }
  }
  rstate.pending.clear();
  rstate.pending_sizes.clear();

  if (aggregate_column[0]->column_length > INT32_MAX) {
    throw NotImplementedException("Column length greater than INT32_MAX is not supported");
  }

  // Stage the partial 1-row aggregate into per-GPU runtime state. Cross-GPU
  // reduce of partials happens later in GPUPhysicalMaterializedCollector::GetResult.
  if (!rstate.aggregation_result) {
    rstate.aggregation_result = make_shared_ptr<GPUIntermediateRelation>(aggregates.size());
  }
  auto& aggregation_result = rstate.aggregation_result;

  // AVG cross-GPU merge needs each partition's non-null row count: cuDF's
  // MEAN reducer (called inside HandleAggregateExpressionCuDF below) returns
  // only the mean and discards the count. Snapshot it now while we still
  // have the input column.
  rstate.avg_valid_counts.assign(aggregates.size(), 0);
  for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); ++aggr_idx) {
    auto& aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();
    if (aggregate.function.name != "avg") continue;
    auto& col = aggregate_column[aggr_idx];
    if (!col || col->data_wrapper.data == nullptr) continue;
    auto cudf_col = col->convertToCudfColumn();
    rstate.avg_valid_counts[aggr_idx] = col->column_length - cudf_col.null_count();
  }

  HandleAggregateExpressionCuDF(aggregate_column, gpuBufferManager, aggregates);

  for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
    // TODO: has to fix this for columns with partially NULL values
    if (aggregation_result->columns[aggr_idx] == nullptr) {
      SIRIUS_LOG_DEBUG(
        "Passing aggregate column {} to aggregation result column {}", aggr_idx, aggr_idx);
      aggregation_result->columns[aggr_idx]               = aggregate_column[aggr_idx];
      aggregation_result->columns[aggr_idx]->row_ids      = nullptr;
      aggregation_result->columns[aggr_idx]->row_id_count = 0;
    } else if (aggregation_result->columns[aggr_idx] != nullptr) {
      if (aggregate_column[aggr_idx]->data_wrapper.data != nullptr &&
          aggregation_result->columns[aggr_idx]->data_wrapper.data != nullptr) {
        throw NotImplementedException("Combine not implemented yet for ungrouped aggregate");
      } else if (aggregate_column[aggr_idx]->data_wrapper.data != nullptr &&
                 aggregation_result->columns[aggr_idx]->data_wrapper.data == nullptr) {
        SIRIUS_LOG_DEBUG(
          "Passing aggregate column {} to aggregation result column {}", aggr_idx, aggr_idx);
        aggregation_result->columns[aggr_idx]               = aggregate_column[aggr_idx];
        aggregation_result->columns[aggr_idx]->row_ids      = nullptr;
        aggregation_result->columns[aggr_idx]->row_id_count = 0;
      } else {
        SIRIUS_LOG_DEBUG("Aggregate column {} is null, skipping", aggr_idx);
      }
    }
  }

  // Interior multi-GPU merge: signal this GPU's partial is staged so peer
  // GPUs' GetDataInteriorMerged can proceed once every partial exists.
  if (interior_cross_gpu_merge && GPUBufferManager::GetMaxGpus() > 1) {
    // Diagnostic (debug level): staged device pointer + value readback for
    // the single-DOUBLE-aggregate case.
    if (types.size() == 1 && types[0].InternalType() == PhysicalType::DOUBLE &&
        aggregation_result->columns[0] &&
        aggregation_result->columns[0]->data_wrapper.data) {
      double staged_val = 0;
      cudaMemcpy(&staged_val, aggregation_result->columns[0]->data_wrapper.data,
                 sizeof(double), cudaMemcpyDeviceToHost);
      SIRIUS_LOG_DEBUG("Interior agg STAGED gpu={} ptr={} val={:.6f}",
                       sirius_current_gpu,
                       (void*)aggregation_result->columns[0]->data_wrapper.data,
                       staged_val);
    }
    std::lock_guard<std::mutex> lk(sink_mutex);
    ++gpus_sunk;
    sink_cv.notify_all();
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Ungrouped aggregate Finalize time: {:.2f} ms", duration.count() / 1000.0);
}

// Interior multi-GPU path: this aggregate's output feeds another operator
// (e.g. a scalar-subquery threshold consumed by a join), so the per-GPU
// PARTIAL staged by Sink must NOT go downstream — every GPU has to see the
// GLOBAL value. Wait until all execution GPUs staged their partial, merge
// them host-side with the same helpers the collector uses for top-level
// aggregates, and upload the merged row onto the calling GPU. Each GPU
// worker merges independently (the merge is deterministic, inputs are
// read-only by then, and sirius_current_gpu is thread_local).
SourceResultType GPUPhysicalUngroupedAggregate::GetDataInteriorMerged(
  GPUIntermediateRelation& output_relation) const
{
  auto& rstate = runtime_state<UngroupedAggregateRuntimeState>(sirius_current_gpu);
  if (!rstate.merged_result) {
    // The sequential per-GPU executor (SIRIUS_LEGACY_PARALLEL=0) runs GPU g's
    // whole plan before GPU g+1 starts, so peer partials cannot exist yet at
    // this point. Refuse loudly -> DuckDB CPU fallback stays correct.
    const char* parallel_env = std::getenv("SIRIUS_LEGACY_PARALLEL");
    if (parallel_env && std::string(parallel_env) == "0") {
      throw NotImplementedException(
        "Interior multi-GPU ungrouped aggregate requires the parallel executor");
    }
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    const int expected = static_cast<int>(gpuBufferManager->tables_per_gpu.size());
    {
      std::unique_lock<std::mutex> lk(sink_mutex);
      if (!sink_cv.wait_for(lk, std::chrono::seconds(60),
                            [&] { return gpus_sunk >= expected; })) {
        // A peer GPU never staged its partial (its pipeline failed or stalled);
        // don't hang the query -- fail over to DuckDB.
        throw NotImplementedException(
          "Timed out waiting for peer GPUs' ungrouped-aggregate partials");
      }
    }

    // Make every GPU's staged partial coherent before reading it from this
    // thread: the peer worker's cudf reduction ran on ITS stream and may not
    // have landed in device memory yet when it signalled Sink completion.
    // Same fix as the cross-GPU input sync in magi_groupby_runtime.cu (stale
    // cross-device reads there produced wrong, run-to-run-varying sums).
    {
      int prev_device = 0;
      cudaGetDevice(&prev_device);
      for (int g = 0; g < GPUBufferManager::GetMaxGpus(); ++g) {
        if (!per_gpu_state[g]) continue;
        cudaSetDevice(g);
        cudaDeviceSynchronize();
      }
      cudaSetDevice(prev_device);
    }

    // Diagnostic (debug level): re-read every GPU's staged pointer + value at
    // merge time, to compare against the STAGED logs from Sink.
    if (types.size() == 1 && types[0].InternalType() == PhysicalType::DOUBLE) {
      for (int g = 0; g < GPUBufferManager::GetMaxGpus(); ++g) {
        if (!per_gpu_state[g]) continue;
        auto& rs = static_cast<UngroupedAggregateRuntimeState&>(*per_gpu_state[g]);
        if (!rs.aggregation_result || !rs.aggregation_result->columns[0] ||
            !rs.aggregation_result->columns[0]->data_wrapper.data)
          continue;
        double mv = 0;
        cudaMemcpy(&mv, rs.aggregation_result->columns[0]->data_wrapper.data,
                   sizeof(double), cudaMemcpyDeviceToHost);
        SIRIUS_LOG_DEBUG("Interior agg MERGE-READ reader_gpu={} src_gpu={} ptr={} val={:.6f}",
                         sirius_current_gpu, g,
                         (void*)rs.aggregation_result->columns[0]->data_wrapper.data, mv);
      }
    }

    // Re-materialise every GPU's 1-row raw partial host-side and reduce them
    // to the single global row (COUNT/SUM add, MIN/MAX, AVG count-weighted).
    auto raw = CollectUngroupedRawPartials(*this, types, gpuBufferManager);
    unique_ptr<GPUResultCollection> merged_coll;
    if (raw->write_idx > 1) {
      // Diagnostic (debug level only): per-GPU partials for the common
      // single-DOUBLE-aggregate case (e.g. a SUM threshold subquery).
      if (types.size() == 1 && types[0].InternalType() == PhysicalType::DOUBLE) {
        for (size_t g = 0; g < raw->write_idx; ++g) {
          SIRIUS_LOG_DEBUG(
            "Interior ungrouped-agg partial[{}] = {:.6f}", g,
            FlatVector::GetData<double>(raw->data_chunks[g].data[0])[0]);
        }
      }
      merged_coll = ReduceUngroupedAcrossGpus(*raw, *this, types, gpuBufferManager);
      if (types.size() == 1 && types[0].InternalType() == PhysicalType::DOUBLE) {
        SIRIUS_LOG_DEBUG(
          "Interior ungrouped-agg merged = {:.6f}",
          FlatVector::GetData<double>(merged_coll->data_chunks[0].data[0])[0]);
      }
    } else {
      merged_coll = std::move(raw);  // single partial -> already global
    }
    if (merged_coll->write_idx == 0) {
      throw NotImplementedException(
        "Interior multi-GPU ungrouped aggregate produced no partials");
    }

    // Upload the merged host row onto THIS GPU as 1-row device columns
    // (same pattern as cudf_aggregate's COUNT_STAR result construction).
    // Fixed-width types only: a VARCHAR row would upload a dangling string_t.
    auto& chunk  = merged_coll->data_chunks[0];
    auto merged  = make_shared_ptr<GPUIntermediateRelation>(types.size());
    for (size_t col = 0; col < types.size(); ++col) {
      if (types[col].InternalType() == PhysicalType::VARCHAR) {
        throw NotImplementedException(
          "Interior multi-GPU ungrouped aggregate over VARCHAR not supported");
      }
      const PhysicalType pt   = types[col].InternalType();
      const size_t value_size = GetTypeIdSize(pt);
      auto& v                 = chunk.data[col];
      const bool valid        = FlatVector::Validity(v).RowIsValid(0);
      uint8_t* dev = gpuBufferManager->customCudaMalloc<uint8_t>(value_size, 0, 0);
      // Untyped data pointer: the typed FlatVector::GetData<T> asserts the
      // vector's physical type matches T and would throw here (v is e.g.
      // DOUBLE). We copy raw bytes, so the untyped accessor is the right one.
      cudaMemcpy(dev, FlatVector::GetData(v), value_size, cudaMemcpyHostToDevice);
      auto mask = createNullMask(
        1, valid ? cudf::mask_state::ALL_VALID : cudf::mask_state::ALL_NULL);
      merged->columns[col] = make_shared_ptr<GPUColumn>(
        1, convertLogicalTypeToColumnType(types[col]), dev, mask);
    }
    rstate.merged_result = std::move(merged);
  }

  for (int col = 0; col < rstate.merged_result->columns.size(); col++) {
    output_relation.columns[col] =
      make_shared_ptr<GPUColumn>(rstate.merged_result->columns[col]);
  }
  return SourceResultType::FINISHED;
}

SourceResultType GPUPhysicalUngroupedAggregate::GetData(
  GPUIntermediateRelation& output_relation) const
{
  auto start = std::chrono::high_resolution_clock::now();
  if (interior_cross_gpu_merge && GPUBufferManager::GetMaxGpus() > 1) {
    return GetDataInteriorMerged(output_relation);
  }
  // Read this GPU's partial aggregate from per-GPU runtime state.
  auto& rstate = runtime_state<UngroupedAggregateRuntimeState>(sirius_current_gpu);
  if (!rstate.aggregation_result) {
    throw NotImplementedException("UngroupedAggregate::GetData called before Sink");
  }
  auto& aggregation_result = rstate.aggregation_result;
  for (int col = 0; col < aggregation_result->columns.size(); col++) {
    SIRIUS_LOG_DEBUG("Writing aggregation result to column {}", col);
    output_relation.columns[col] = make_shared_ptr<GPUColumn>(aggregation_result->columns[col]);
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Ungrouped aggregate GetData time: {:.2f} ms", duration.count() / 1000.0);
  return SourceResultType::FINISHED;
}

void GPUPhysicalUngroupedAggregate::MaterializeDistinctInput(
  GPUIntermediateRelation& input_relation, vector<shared_ptr<GPUColumn>>& aggregate_column) const
{
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  auto& distinct_info                = *distinct_collection_info;
  auto& distinct_indices             = distinct_info.Indices();
  auto& distinct_filter              = distinct_info.Indices();

  for (auto& idx : distinct_indices) {
    auto& aggregate = aggregates[idx]->Cast<BoundAggregateExpression>();

    D_ASSERT(distinct_info.table_map.count(idx));

    if (aggregate.filter) {
      auto& bound_ref_expr = aggregate.filter->Cast<BoundReferenceExpression>();
      SIRIUS_LOG_DEBUG("Reading filter column from index {}", bound_ref_expr.index);
    }

    for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
      auto& child     = aggregate.children[child_idx];
      auto& bound_ref = child->Cast<BoundReferenceExpression>();
      SIRIUS_LOG_DEBUG(
        "Reading aggregation column from index {} and passing it to index {} in groupby result",
        bound_ref.index,
        bound_ref.index);
      aggregate_column[bound_ref.index] =
        HandleMaterializeExpression(input_relation.columns[bound_ref.index], gpuBufferManager);
    }
  }
}

}  // namespace duckdb
