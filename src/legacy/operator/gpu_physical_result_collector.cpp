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

#include "operator/gpu_physical_result_collector.hpp"

#include "cudf/cudf_utils.hpp"
#include "duckdb/common/hugeint.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_context.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "gpu_pipeline.hpp"
#include "log/logging.hpp"
#include "operator/gpu_materialize.hpp"
#include "operator/gpu_physical_projection.hpp"
#include "legacy/operator/magi_fused_agg.hpp"   // Option 3 fused join->agg side channel
#include "operator/gpu_physical_table_scan.hpp"
#include "operator/gpu_physical_ungrouped_aggregate.hpp"
#include "utils.hpp"

namespace duckdb {

// Multi-GPU: compute whether `node`'s per-GPU output is REPLICATED (identical
// rows on every GPU) as opposed to PARTITIONED (each GPU holds a slice), and
// mark interior UNGROUPED_AGGREGATEs that need the cross-GPU merge along the
// way. The collector-side reduce in GetResult only pattern-matches the top of
// the plan; an interior aggregate (e.g. a scalar-subquery threshold feeding a
// join) would otherwise hand its per-GPU PARTIAL downstream and every GPU
// would filter against a wrong, partition-local scalar.
//   - A scan is replicated iff the cached table is replicated (small tables).
//   - An interior ungrouped aggregate over PARTITIONED input is marked for the
//     GetData merge; its output is then the global row on every GPU, i.e.
//     REPLICATED.
//   - An ungrouped aggregate over REPLICATED input already holds the global
//     value on every GPU — it must NOT be merged (e.g. the cardinality-check
//     COUNT DuckDB plans around scalar subqueries would double-count).
//   - Any other node is replicated iff all its children are; unknown leaves
//     are assumed partitioned (matches SubtreeAllReplicated).
// `reduce_visible` is the one aggregate (if any) that GetResult WILL reduce.
static bool ComputeReplicationAndMark(GPUPhysicalOperator& node,
                                      const GPUPhysicalOperator* reduce_visible)
{
  if (node.type == PhysicalOperatorType::TABLE_SCAN) {
    auto& scan      = node.Cast<GPUPhysicalTableScan>();
    auto table_name = scan.GetCatalogTableName();
    auto& tables    = GPUBufferManager::GetInstance().tables_per_gpu[0];
    auto it         = tables.find(table_name);
    return it != tables.end() && it->second->is_replicated;
  }
  if (node.type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
    bool child_replicated = true;
    for (auto& child : node.children) {
      child_replicated = ComputeReplicationAndMark(*child, reduce_visible) && child_replicated;
    }
    if (&node != reduce_visible && !child_replicated) {
      node.Cast<GPUPhysicalUngroupedAggregate>().interior_cross_gpu_merge = true;
    }
    // Whether merged here or already fed replicated input, every GPU ends up
    // holding the same single row. (For reduce_visible the collector merges at
    // GetResult; nothing inside the plan consumes it, so the value is moot.)
    return true;
  }
  if (node.children.empty()) {
    // Unknown leaf (DELIM_SCAN, CTE, ...): assume partitioned.
    return false;
  }
  bool replicated = true;
  for (auto& child : node.children) {
    replicated = ComputeReplicationAndMark(*child, reduce_visible) && replicated;
  }
  return replicated;
}

GPUPhysicalResultCollector::GPUPhysicalResultCollector(GPUPreparedStatementData& data)
  : GPUPhysicalOperator(PhysicalOperatorType::RESULT_COLLECTOR, {LogicalType::BOOLEAN}, 0),
    statement_type(data.prepared->statement_type),
    properties(data.prepared->properties),
    plan(*data.gpu_physical_plan),
    names(data.prepared->names)
{
  this->types      = data.prepared->types;
  gpuBufferManager = &(GPUBufferManager::GetInstance());

  if (GPUBufferManager::GetMaxGpus() > 1) {
    const GPUPhysicalOperator* reduce_visible = nullptr;
    if (plan.type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
      reduce_visible = &plan;
    } else if (plan.type == PhysicalOperatorType::PROJECTION && !plan.children.empty() &&
               plan.children[0]->type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
      reduce_visible = plan.children[0].get();
    }
    ComputeReplicationAndMark(plan, reduce_visible);
  }
}

vector<const_reference<GPUPhysicalOperator>> GPUPhysicalResultCollector::GetChildren() const
{
  return {plan};
}

void GPUPhysicalResultCollector::BuildPipelines(GPUPipeline& current,
                                                GPUMetaPipeline& meta_pipeline)
{
  // operator is a sink, build a pipeline
  sink_state.reset();

  D_ASSERT(children.empty());

  // single operator: the operator becomes the data source of the current pipeline
  auto& state = meta_pipeline.GetState();
  state.SetPipelineSource(current, *this);

  // we create a new pipeline starting from the child
  auto& child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
  child_meta_pipeline.Build(plan);
}

GPUPhysicalMaterializedCollector::GPUPhysicalMaterializedCollector(GPUPreparedStatementData& data)
  : GPUPhysicalResultCollector(data)
{
  // Per-GPU result_collection now lives in ResultCollectorRuntimeState; it's
  // lazy-initialised on first Sink for each GPU. No work needed here.
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class GPUMaterializedCollectorGlobalState : public GlobalSinkState {
 public:
  mutex glock;
  shared_ptr<ClientContext> context;
};

class GPUMaterializedCollectorLocalState : public LocalSinkState {
 public:
  ColumnDataAppendState append_state;
};

template <typename T>
void GPUPhysicalMaterializedCollector::FinalMaterializeInternal(
  GPUIntermediateRelation input_relation, GPUIntermediateRelation& output_relation, size_t col)
{
  if (input_relation.checkLateMaterialization(col)) {
    T* data                  = reinterpret_cast<T*>(input_relation.columns[col]->data_wrapper.data);
    uint64_t* row_ids        = reinterpret_cast<uint64_t*>(input_relation.columns[col]->row_ids);
    cudf::bitmask_type* mask = input_relation.columns[col]->data_wrapper.validity_mask;
    T* materialized;
    cudf::bitmask_type* out_mask = nullptr;
    materializeExpression<T>(
      data, materialized, row_ids, input_relation.columns[col]->row_id_count, mask, out_mask);
    output_relation.columns[col] =
      make_shared_ptr<GPUColumn>(input_relation.columns[col]->row_id_count,
                                 input_relation.columns[col]->data_wrapper.type,
                                 reinterpret_cast<uint8_t*>(materialized),
                                 out_mask);
    output_relation.columns[col]->row_id_count = 0;
    output_relation.columns[col]->row_ids      = nullptr;
    output_relation.columns[col]->is_unique    = input_relation.columns[col]->is_unique;
  } else {
    output_relation.columns[col] =
      make_shared_ptr<GPUColumn>(input_relation.columns[col]->column_length,
                                 input_relation.columns[col]->data_wrapper.type,
                                 input_relation.columns[col]->data_wrapper.data,
                                 input_relation.columns[col]->data_wrapper.validity_mask);
    output_relation.columns[col]->is_unique = input_relation.columns[col]->is_unique;
  }
}

void GPUPhysicalMaterializedCollector::FinalMaterializeString(
  GPUIntermediateRelation input_relation, GPUIntermediateRelation& output_relation, size_t col)
{
  if (input_relation.checkLateMaterialization(col)) {
    // Late materalize the input relationship
    uint8_t* data     = input_relation.columns[col]->data_wrapper.data;
    uint64_t* offset  = input_relation.columns[col]->data_wrapper.offset;
    uint64_t* row_ids = input_relation.columns[col]->row_ids;
    size_t num_rows   = input_relation.columns[col]->row_id_count;
    uint8_t* result;
    uint64_t* result_offset;
    uint64_t* new_num_bytes;
    cudf::bitmask_type* out_mask = nullptr;
    cudf::bitmask_type* mask     = input_relation.columns[col]->data_wrapper.validity_mask;

    SIRIUS_LOG_DEBUG("Running string late materalization with {} rows", num_rows);

    materializeString(
      data, offset, result, result_offset, row_ids, new_num_bytes, num_rows, mask, out_mask);

    output_relation.columns[col] =
      make_shared_ptr<GPUColumn>(num_rows,
                                 GPUColumnType(GPUColumnTypeId::VARCHAR),
                                 reinterpret_cast<uint8_t*>(result),
                                 result_offset,
                                 new_num_bytes[0],
                                 true,
                                 out_mask);
    output_relation.columns[col]->row_id_count = 0;
    output_relation.columns[col]->row_ids      = nullptr;
    output_relation.columns[col]->is_unique    = input_relation.columns[col]->is_unique;
  } else {
    // output_relation.columns[col] = make_shared_ptr<GPUColumn>(*input_relation.columns[col]);
    output_relation.columns[col] =
      make_shared_ptr<GPUColumn>(input_relation.columns[col]->column_length,
                                 input_relation.columns[col]->data_wrapper.type,
                                 input_relation.columns[col]->data_wrapper.data,
                                 input_relation.columns[col]->data_wrapper.offset,
                                 input_relation.columns[col]->data_wrapper.num_bytes,
                                 true,
                                 input_relation.columns[col]->data_wrapper.validity_mask);
    output_relation.columns[col]->is_unique = input_relation.columns[col]->is_unique;
  }
}

size_t GPUPhysicalMaterializedCollector::FinalMaterialize(GPUIntermediateRelation input_relation,
                                                          GPUIntermediateRelation& output_relation,
                                                          size_t col)
{
  size_t size_bytes;

  switch (input_relation.columns[col]->data_wrapper.type.id()) {
    case GPUColumnTypeId::INT64:
    case GPUColumnTypeId::TIMESTAMP_SEC:
    case GPUColumnTypeId::TIMESTAMP_MS:
    case GPUColumnTypeId::TIMESTAMP_US:
    case GPUColumnTypeId::TIMESTAMP_NS:
      FinalMaterializeInternal<int64_t>(input_relation, output_relation, col);
      size_bytes = output_relation.columns[col]->column_length * sizeof(int64_t);
      break;
    case GPUColumnTypeId::INT32:
    case GPUColumnTypeId::DATE:
      FinalMaterializeInternal<int>(input_relation, output_relation, col);
      size_bytes = output_relation.columns[col]->column_length * sizeof(int);
      break;
    case GPUColumnTypeId::INT16:
      FinalMaterializeInternal<int16_t>(input_relation, output_relation, col);
      size_bytes = output_relation.columns[col]->column_length * sizeof(int16_t);
      break;
    case GPUColumnTypeId::FLOAT64:
      FinalMaterializeInternal<double>(input_relation, output_relation, col);
      size_bytes = output_relation.columns[col]->column_length * sizeof(double);
      break;
    case GPUColumnTypeId::FLOAT32:
      FinalMaterializeInternal<float>(input_relation, output_relation, col);
      size_bytes = output_relation.columns[col]->column_length * sizeof(float);
      break;
    case GPUColumnTypeId::BOOLEAN:
      FinalMaterializeInternal<uint8_t>(input_relation, output_relation, col);
      size_bytes = output_relation.columns[col]->column_length * sizeof(uint8_t);
      break;
    case GPUColumnTypeId::VARCHAR:
      FinalMaterializeString(input_relation, output_relation, col);
      break;
    case GPUColumnTypeId::DECIMAL: {
      switch (input_relation.columns[col]->data_wrapper.getColumnTypeSize()) {
        case sizeof(int32_t): {
          FinalMaterializeInternal<int32_t>(input_relation, output_relation, col);
          size_bytes = output_relation.columns[col]->column_length * sizeof(int32_t);
          break;
        }
        case sizeof(int64_t): {
          FinalMaterializeInternal<int64_t>(input_relation, output_relation, col);
          size_bytes = output_relation.columns[col]->column_length * sizeof(int64_t);
          break;
        }
          throw NotImplementedException(
            "Unsupported sirius DECIMAL column type size in `FinalMaterialize`: %zu",
            input_relation.columns[col]->data_wrapper.getColumnTypeSize());
      }
      break;
    }
    default:
      throw NotImplementedException(
        "Unsupported sirius column type in `FinalMaterialize`: %d",
        static_cast<int>(input_relation.columns[col]->data_wrapper.type.id()));
  }
  // output_relation.length = output_relation.columns[col]->column_length;
  // SIRIUS_LOG_DEBUG("Final materialize size {} bytes", size_bytes);
  return size_bytes;
}

SinkResultType GPUPhysicalMaterializedCollector::ConvertGPUTableToCPUCollection(
  GPUIntermediateRelation& input_relation,
  const vector<LogicalType>& types,
  GPUResultCollection* result_collection,
  GPUBufferManager* gpuBufferManager)
{
  // TODO: Don't forget to check the if input relation is already materialized or not, if not then
  // materialize it
  if (types.size() != input_relation.columns.size()) {
    throw InvalidInputException("Column count mismatch");
  }

  // measure time
  auto start = std::chrono::high_resolution_clock::now();
  // auto &gstate = GetGlobalSinkState(input_relation.context);

  auto materialize_start_time = std::chrono::high_resolution_clock::now();

  // First figure out the total number of strings and chars in all of the columns
  size_t all_columns_num_strings = 0;
  size_t all_columns_total_chars = 0;
  for (int col = 0; col < input_relation.columns.size(); col++) {
    DataWrapper column_data_wrapper = input_relation.columns[col]->data_wrapper;
    if (column_data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
      all_columns_num_strings += column_data_wrapper.size;
      all_columns_total_chars += column_data_wrapper.num_bytes;
    }
  }
  size_t all_columns_strings_buffer_size = all_columns_num_strings * sizeof(string_t);
  size_t all_columns_chars_buffer_size   = all_columns_total_chars * sizeof(char);

  // Now allocate the buffers for the columns
  size_t total_buffer_size     = all_columns_strings_buffer_size + all_columns_chars_buffer_size;
  uint8_t* combined_buffer     = gpuBufferManager->customCudaHostAlloc<uint8_t>(total_buffer_size);
  string_t* all_columns_string = reinterpret_cast<string_t*>(combined_buffer);
  char* all_columns_chars =
    reinterpret_cast<char*>(combined_buffer + all_columns_strings_buffer_size);

  size_t size_bytes = 0;
  uint8_t** host_data =
    gpuBufferManager->customCudaHostAlloc<uint8_t*>(input_relation.columns.size());
  uint8_t** host_mask_data =
    gpuBufferManager->customCudaHostAlloc<uint8_t*>(input_relation.columns.size());

  GPUIntermediateRelation materialized_relation(input_relation.columns.size());
  string_t** duckdb_strings =
    gpuBufferManager->customCudaHostAlloc<string_t*>(input_relation.columns.size());
  string_t* curr_column_string_buffer = all_columns_string;
  char* curr_column_chars_buffer      = all_columns_chars;
  for (int col = 0; col < input_relation.columns.size(); col++) {
    auto col_materialize_start_time = std::chrono::high_resolution_clock::now();

    // Just return when there is an empty column
    size_t actual_column_len = input_relation.columns[col]->row_ids
                                 ? input_relation.columns[col]->row_id_count
                                 : input_relation.columns[col]->column_length;
    if (actual_column_len == 0) { return SinkResultType::FINISHED; }
    // Final materialization
    size_bytes = FinalMaterialize(input_relation, materialized_relation, col);

    const GPUColumnType& col_type = input_relation.columns[col]->data_wrapper.type;
    bool is_string                = false;
    if (col_type.id() != GPUColumnTypeId::VARCHAR) {
      if (types[col].InternalType() == PhysicalType::INT128) {
        if (materialized_relation.columns[col]->data_wrapper.type.id() == GPUColumnTypeId::INT64) {
          SIRIUS_LOG_DEBUG("Converting INT64 to INT128 for column {}", col);
          uint8_t* temp_int128 = gpuBufferManager->customCudaMalloc<uint8_t>(size_bytes * 2, 0, 0);
          convertInt64ToInt128(materialized_relation.columns[col]->data_wrapper.data,
                               temp_int128,
                               materialized_relation.columns[col]->column_length);
          host_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(size_bytes * 2);
          callCudaMemcpyDeviceToHost<uint8_t>(host_data[col], temp_int128, size_bytes * 2, 0);
        } else if (materialized_relation.columns[col]->data_wrapper.type.id() ==
                   GPUColumnTypeId::INT32) {
          SIRIUS_LOG_DEBUG("Converting INT32 to INT128 for column {}", col);
          uint8_t* temp_int128 = gpuBufferManager->customCudaMalloc<uint8_t>(size_bytes * 4, 0, 0);
          convertInt32ToInt128(materialized_relation.columns[col]->data_wrapper.data,
                               temp_int128,
                               materialized_relation.columns[col]->column_length);
          host_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(size_bytes * 4);
          callCudaMemcpyDeviceToHost<uint8_t>(host_data[col], temp_int128, size_bytes * 4, 0);
        } else if (materialized_relation.columns[col]->data_wrapper.type.id() ==
                   GPUColumnTypeId::INT16) {
          SIRIUS_LOG_DEBUG("Converting INT16 to INT128 for column {}", col);
          uint8_t* temp_int128 = gpuBufferManager->customCudaMalloc<uint8_t>(size_bytes * 8, 0, 0);
          convertInt16ToInt128(materialized_relation.columns[col]->data_wrapper.data,
                               temp_int128,
                               materialized_relation.columns[col]->column_length);
          host_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(size_bytes * 8);
          callCudaMemcpyDeviceToHost<uint8_t>(host_data[col], temp_int128, size_bytes * 8, 0);
        } else if (materialized_relation.columns[col]->data_wrapper.type.id() ==
                   GPUColumnTypeId::DECIMAL) {
          if (types[col].id() != LogicalTypeId::DECIMAL) {
            throw InternalException(
              "Destination type is not decimal when performing INT128 (physical type) conversion "
              "for decimal,"
              " destination type: %d",
              static_cast<int>(types[col].id()));
          }
          int from_decimal_size =
            materialized_relation.columns[col]->data_wrapper.getColumnTypeSize();
          int from_scale =
            materialized_relation.columns[col]->data_wrapper.type.GetDecimalTypeInfo()->scale_;
          int to_width = DecimalType::GetWidth(types[col]);
          int to_scale = DecimalType::GetScale(types[col]);
          if (from_decimal_size != sizeof(__int128_t) || from_scale != to_scale) {
            // `from` and `to` decimal types are different, need to cast
            auto from_cudf_column_view = materialized_relation.columns[col]->convertToCudfColumn();
            auto to_cudf_type          = GetCudfType(types[col]);
            auto to_cudf_column        = cudf::cast(from_cudf_column_view,
                                             to_cudf_type,
                                             rmm::cuda_stream_default,
                                             GPUBufferManager::GetInstance().get_mr_ref());
            size_bytes     = materialized_relation.columns[col]->column_length * sizeof(__int128_t);
            host_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(size_bytes);
            uint8_t* to_cudf_data = const_cast<uint8_t*>(to_cudf_column->view().data<uint8_t>());
            callCudaMemcpyDeviceToHost<uint8_t>(host_data[col], to_cudf_data, size_bytes, 0);
          } else {
            // `from` and `to` decimal types are the same
            host_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(size_bytes);
            callCudaMemcpyDeviceToHost<uint8_t>(
              host_data[col], materialized_relation.columns[col]->data_wrapper.data, size_bytes, 0);
          }
          materialized_relation.columns[col]->data_wrapper.type.SetDecimalTypeInfo(to_width,
                                                                                   to_scale);
        } else {
          throw NotImplementedException(
            "Unsupported siris column type for INT128 conversion: %d",
            static_cast<int>(materialized_relation.columns[col]->data_wrapper.type.id()));
        }
      } else {
        SIRIUS_LOG_DEBUG("DBG: about to host-alloc {} bytes for col {}", size_bytes, col);
        host_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(size_bytes);
        SIRIUS_LOG_DEBUG("DBG: host-alloc done, host_data[{}]={} ; src device ptr={}",
                         col,
                         (void*)host_data[col],
                         (void*)materialized_relation.columns[col]->data_wrapper.data);
        callCudaMemcpyDeviceToHost<uint8_t>(host_data[col],
                                            materialized_relation.columns[col]->data_wrapper.data,
                                            size_bytes,
                                            sirius_current_gpu);
        SIRIUS_LOG_DEBUG("DBG: device-to-host memcpy done for col {}", col);
      }

      if (materialized_relation.columns[col]->data_wrapper.validity_mask == nullptr) {
        SIRIUS_LOG_DEBUG("Column {} has no validity mask, creating a mask with all valid values\n",
                         col);
        uint64_t padded_bytes = getMaskBytesSize(materialized_relation.columns[col]->column_length);
        // If the validity mask is null, we create a mask with all valid values
        host_mask_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(padded_bytes);
        memset(host_mask_data[col], 0xFF, padded_bytes);  // All bits set to 1 (valid)
      } else {
        // Copy the existing validity mask
        SIRIUS_LOG_DEBUG("DBG: about to host-alloc mask {} bytes for col {}",
                         materialized_relation.columns[col]->data_wrapper.mask_bytes,
                         col);
        host_mask_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(
          materialized_relation.columns[col]->data_wrapper.mask_bytes);
        SIRIUS_LOG_DEBUG("DBG: mask host-alloc done {}; src ptr={}",
                         (void*)host_mask_data[col],
                         (void*)materialized_relation.columns[col]->data_wrapper.validity_mask);
        callCudaMemcpyDeviceToHost<uint8_t>(
          host_mask_data[col],
          reinterpret_cast<uint8_t*>(
            materialized_relation.columns[col]->data_wrapper.validity_mask),
          materialized_relation.columns[col]->data_wrapper.mask_bytes,
          sirius_current_gpu);
        SIRIUS_LOG_DEBUG("DBG: mask memcpy done for col {}", col);
      }
    } else {
      // Use the helper method to materialize the string on the GPU
      shared_ptr<GPUColumn> str_column = materialized_relation.columns[col];
      materializeStringColumnToDuckdbFormat(
        str_column, curr_column_chars_buffer, curr_column_string_buffer);
      duckdb_strings[col]                = curr_column_string_buffer;
      materialized_relation.columns[col] = str_column;
      is_string                          = true;
      if (str_column->data_wrapper.validity_mask == nullptr) {
        SIRIUS_LOG_DEBUG("Column {} has no validity mask, creating a mask with all valid values\n",
                         col);
        // printf("Column %d has no validity mask, creating a mask with all valid values\n", col);
        uint64_t padded_bytes = getMaskBytesSize(str_column->column_length);
        // If the validity mask is null, we create a mask with all valid values
        host_mask_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(padded_bytes);
        memset(host_mask_data[col], 0xFF, padded_bytes);  // All bits set to 1 (valid)
      } else {
        // Copy the existing validity mask
        SIRIUS_LOG_DEBUG("Copying validity mask for column {}\n", col);
        host_mask_data[col] =
          gpuBufferManager->customCudaHostAlloc<uint8_t>(str_column->data_wrapper.mask_bytes);
        callCudaMemcpyDeviceToHost<uint8_t>(
          host_mask_data[col],
          reinterpret_cast<uint8_t*>(str_column->data_wrapper.validity_mask),
          str_column->data_wrapper.mask_bytes,
          0);
      }

      // Advance the buffer pointers based on this column's details
      DataWrapper str_column_data = str_column->data_wrapper;
      curr_column_chars_buffer += str_column_data.num_bytes;
      curr_column_string_buffer += str_column_data.size;
    }
  }
  auto materialize_end_time    = std::chrono::high_resolution_clock::now();
  auto materialize_duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                   materialize_end_time - materialize_start_time)
                                   .count() /
                                 1000.0;
  SIRIUS_LOG_DEBUG("Result Collector CPU Materialize Time: {:.2f} ms", materialize_duration_ms);

  auto chunk_start_time = std::chrono::high_resolution_clock::now();
  size_t num_records    = materialized_relation.columns[0]->column_length;
  size_t total_vector   = (num_records + STANDARD_VECTOR_SIZE - 1) / STANDARD_VECTOR_SIZE;
  // Phase 2: SetCapacity now appends to existing capacity (see
  // GPUResultCollection::SetCapacity); each per-GPU iteration asks for room
  // for its own chunks without discarding earlier writes.
  result_collection->SetCapacity(total_vector);
  SIRIUS_LOG_DEBUG(
    "Result Collector: Num Records - {}, Total vectors - {}", num_records, total_vector);

  size_t remaining    = num_records;
  uint64_t read_index = 0;
  for (uint64_t vec = 0; vec < total_vector; vec++) {
    size_t chunk_cardinality = std::min(remaining, (size_t)STANDARD_VECTOR_SIZE);
    DataChunk chunk;
    chunk.InitializeEmpty(types);
    for (int col = 0; col < materialized_relation.columns.size(); col++) {
      if (materialized_relation.columns[col]->data_wrapper.type.id() != GPUColumnTypeId::VARCHAR) {
        uint8_t* data =
          host_data[col] + vec * STANDARD_VECTOR_SIZE * GetTypeIdSize(types[col].InternalType());
        Vector vector(types[col], data);
        ValidityMask validity_mask(reinterpret_cast<validity_t*>(host_mask_data[col]),
                                   chunk_cardinality);
        FlatVector::SetValidity(vector, validity_mask);
        chunk.data[col].Reference(vector);
      } else {
        // Add the strings to the vector
        Vector str_vector(LogicalType::VARCHAR,
                          reinterpret_cast<data_ptr_t>(duckdb_strings[col] + read_index));
        ValidityMask validity_mask(reinterpret_cast<validity_t*>(host_mask_data[col]),
                                   chunk_cardinality);
        FlatVector::SetValidity(str_vector, validity_mask);
        chunk.data[col].Reference(str_vector);
      }
    }

    // Record this chunk
    chunk.SetCardinality(chunk_cardinality);
    result_collection->AddChunk(chunk);

    // Move to the next chunk
    remaining -= chunk_cardinality;
    read_index += chunk_cardinality;
  }
  auto chunk_end_time = std::chrono::high_resolution_clock::now();
  auto chunking_duration_ms =
    std::chrono::duration_cast<std::chrono::microseconds>(chunk_end_time - chunk_start_time)
      .count() /
    1000.0;
  SIRIUS_LOG_DEBUG("Result Collector Chunking Time: {:.2f} ms", chunking_duration_ms);

  // measure time
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Result collector time: {:.2f} ms", duration.count() / 1000.0);
  return SinkResultType::FINISHED;
}

SinkResultType GPUPhysicalMaterializedCollector::Sink(GPUIntermediateRelation& input_relation) const
{
  // Replicated-input guard: a plan whose every base table is REPLICATED
  // (small-tables-only query) produces the identical full result on every
  // GPU worker — merging the per-GPU collections would return N copies of
  // each row. Fail loudly -> DuckDB fallback.
  if (GPUBufferManager::GetMaxGpus() > 1 && SubtreeAllReplicated(plan)) {
    throw NotImplementedException(
      "Multi-GPU query over only replicated (small) tables is not supported; "
      "falling back to DuckDB");
  }

  // Each GPU worker thread accumulates into its own per-GPU collection. They
  // are merged at GetResult time. Lazy-init on first Sink for this GPU.
  auto& rstate = runtime_state<ResultCollectorRuntimeState>(sirius_current_gpu);
  if (!rstate.result_collection) {
    rstate.result_collection = make_uniq<GPUResultCollection>();
  }
  return ConvertGPUTableToCPUCollection(
    input_relation, types, rstate.result_collection.get(), gpuBufferManager);
}

unique_ptr<GlobalSinkState> GPUPhysicalMaterializedCollector::GetGlobalSinkState(
  ClientContext& context) const
{
  auto state     = make_uniq<GPUMaterializedCollectorGlobalState>();
  state->context = context.shared_from_this();
  return std::move(state);
}

unique_ptr<LocalSinkState> GPUPhysicalMaterializedCollector::GetLocalSinkState(
  ExecutionContext& context) const
{
  auto state = make_uniq<GPUMaterializedCollectorLocalState>();
  return std::move(state);
}

namespace {

// Add a single source value at row `row` of `vsrc` into `dst` (which is the
// running accumulator for this column). Used for COUNT/SUM cross-GPU merge.
void AddValueInPlace(PhysicalType pt, uint8_t* dst, Vector& vsrc, idx_t row)
{
  switch (pt) {
    case PhysicalType::INT32:
      *reinterpret_cast<int32_t*>(dst) += FlatVector::GetData<int32_t>(vsrc)[row];
      break;
    case PhysicalType::INT64:
      *reinterpret_cast<int64_t*>(dst) += FlatVector::GetData<int64_t>(vsrc)[row];
      break;
    case PhysicalType::INT128:
      *reinterpret_cast<hugeint_t*>(dst) += FlatVector::GetData<hugeint_t>(vsrc)[row];
      break;
    case PhysicalType::FLOAT:
      *reinterpret_cast<float*>(dst) += FlatVector::GetData<float>(vsrc)[row];
      break;
    case PhysicalType::DOUBLE:
      *reinterpret_cast<double*>(dst) += FlatVector::GetData<double>(vsrc)[row];
      break;
    default:
      throw NotImplementedException(
        "Cross-GPU SUM/COUNT not supported for PhysicalType %s",
        TypeIdToString(pt));
  }
}

// Replace `dst` with min(dst, src) when take_min, else max(dst, src).
// On is_first the destination is uninitialised; just copy the source.
template <typename T>
inline void min_or_max_into(uint8_t* dst, T src, bool is_first, bool take_min)
{
  T* d = reinterpret_cast<T*>(dst);
  if (is_first) { *d = src; return; }
  if (take_min ? src < *d : src > *d) *d = src;
}

void MinOrMaxValueInPlace(
  PhysicalType pt, uint8_t* dst, Vector& vsrc, idx_t row, bool is_first, bool take_min)
{
  switch (pt) {
    case PhysicalType::INT32:
      min_or_max_into(dst, FlatVector::GetData<int32_t>(vsrc)[row], is_first, take_min);
      break;
    case PhysicalType::INT64:
      min_or_max_into(dst, FlatVector::GetData<int64_t>(vsrc)[row], is_first, take_min);
      break;
    case PhysicalType::INT128:
      min_or_max_into(dst, FlatVector::GetData<hugeint_t>(vsrc)[row], is_first, take_min);
      break;
    case PhysicalType::FLOAT:
      min_or_max_into(dst, FlatVector::GetData<float>(vsrc)[row], is_first, take_min);
      break;
    case PhysicalType::DOUBLE:
      min_or_max_into(dst, FlatVector::GetData<double>(vsrc)[row], is_first, take_min);
      break;
    default:
      throw NotImplementedException(
        "Cross-GPU MIN/MAX not supported for PhysicalType %s",
        TypeIdToString(pt));
  }
}

}  // namespace

// Cross-GPU final reduce of an UNGROUPED_AGGREGATE. Each per-GPU pipeline
// produced a 1-row partial. Collapse those N rows into a single row by
// applying each aggregate's reducer:
//   COUNT/COUNT_STAR/SUM/SUM_NO_OVERFLOW -> sum across GPUs
//   MIN/MAX                              -> min/max across GPUs (skip NULL)
//   AVG                                  -> count-weighted average; uses
//                                           UngroupedAggregateRuntimeState::
//                                           avg_valid_counts populated by
//                                           the upstream Sink
// FIRST and COUNT_DISTINCT still throw NotImplementedException; FIRST is
// order-dependent (and partition order isn't preserved) and COUNT_DISTINCT
// can't be merged from per-partition counts without rehashing the values.
unique_ptr<GPUResultCollection> ReduceUngroupedAcrossGpus(
  GPUResultCollection& combined,
  const GPUPhysicalUngroupedAggregate& agg,
  const vector<LogicalType>& types,
  GPUBufferManager* gbm)
{
  const size_t num_gpus = combined.write_idx;
  D_ASSERT(num_gpus > 1);
  // All chunks are 1 row each (ungrouped agg per GPU). Sanity check.
  for (size_t g = 0; g < num_gpus; ++g) {
    if (combined.data_chunks[g].size() != 1) {
      throw NotImplementedException(
        "Cross-GPU ungrouped aggregate expected 1 row per GPU, got %zu",
        combined.data_chunks[g].size());
    }
  }

  auto reduced = make_uniq<GPUResultCollection>();
  reduced->SetCapacity(1);

  DataChunk merged;
  merged.InitializeEmpty(types);

  // Persistent host buffers for the merged row's columns. We Reference these
  // into the merged chunk via Vector, matching the pattern used in
  // ConvertGPUTableToCPUCollection.
  for (size_t col = 0; col < types.size(); ++col) {
    auto& expr        = agg.aggregates[col]->Cast<BoundAggregateExpression>();
    const auto fname  = expr.function.name;
    PhysicalType pt   = types[col].InternalType();
    size_t value_size = GetTypeIdSize(pt);

    // COUNT(DISTINCT x) / SUM(DISTINCT x) etc. cannot be merged from per-GPU
    // partial counts -- the partials don't carry the underlying distinct
    // values, only the per-partition distinct count -- and our SUM-merge
    // would silently double-count overlaps. Refuse and let sirius fall back
    // to CPU.
    if (expr.IsDistinct()) {
      throw NotImplementedException(
        "Cross-GPU DISTINCT aggregate not supported (function: %s)", fname);
    }

    uint8_t* dst_data = gbm->customCudaHostAlloc<uint8_t>(value_size);
    uint8_t* dst_mask = gbm->customCudaHostAlloc<uint8_t>(getMaskBytesSize(1));
    memset(dst_data, 0, value_size);
    *dst_mask = 0xFF;  // assume valid; flip if all partials are NULL below

    if (fname == "count" || fname == "count_star" || fname == "sum" ||
        fname == "sum_no_overflow") {
      bool any_valid = false;
      for (size_t g = 0; g < num_gpus; ++g) {
        auto& vsrc = combined.data_chunks[g].data[col];
        if (!FlatVector::Validity(vsrc).RowIsValid(0)) continue;
        AddValueInPlace(pt, dst_data, vsrc, 0);
        any_valid = true;
      }
      // SUM over an empty input is NULL; COUNT is 0 (always valid).
      if (!any_valid && (fname == "sum" || fname == "sum_no_overflow")) {
        *dst_mask = 0x00;
      }
    } else if (fname == "min" || fname == "max") {
      const bool take_min = (fname == "min");
      bool any_valid      = false;
      for (size_t g = 0; g < num_gpus; ++g) {
        auto& vsrc = combined.data_chunks[g].data[col];
        if (!FlatVector::Validity(vsrc).RowIsValid(0)) continue;
        MinOrMaxValueInPlace(pt, dst_data, vsrc, 0, /*is_first=*/!any_valid, take_min);
        any_valid = true;
      }
      if (!any_valid) *dst_mask = 0x00;
    } else if (fname == "first" || fname == "arbitrary") {
      // DuckDB plans an uncorrelated scalar subquery as FIRST() over the
      // subquery's 1-row result, so on the interior-merge path every GPU's
      // partial carries the SAME already-merged scalar — any valid partial is
      // the answer. (FIRST without ORDER BY is order-arbitrary by SQL
      // semantics, so taking the first valid partial is a valid instance in
      // the general case too.)
      if (pt == PhysicalType::VARCHAR) {
        throw NotImplementedException(
          "Cross-GPU FIRST not supported for VARCHAR");
      }
      bool any_valid = false;
      for (size_t g = 0; g < num_gpus && !any_valid; ++g) {
        auto& vsrc = combined.data_chunks[g].data[col];
        if (!FlatVector::Validity(vsrc).RowIsValid(0)) continue;
        memcpy(dst_data, FlatVector::GetData(vsrc), value_size);
        any_valid = true;
      }
      if (!any_valid) *dst_mask = 0x00;
    } else if (fname == "avg") {
      // Weighted average across GPUs: total = SUM(mean_g * count_g);
      // result  = total / SUM(count_g). cuDF's MEAN reducer returns the
      // mean cast to FLOAT64 for int/decimal inputs and preserves the
      // input float type otherwise. Handle DOUBLE and FLOAT result types.
      if (pt != PhysicalType::DOUBLE && pt != PhysicalType::FLOAT) {
        throw NotImplementedException(
          "Cross-GPU AVG result type %s not supported", TypeIdToString(pt));
      }
      double total          = 0.0;
      uint64_t grand_count  = 0;
      for (size_t g = 0; g < num_gpus; ++g) {
        auto& vsrc = combined.data_chunks[g].data[col];
        if (!FlatVector::Validity(vsrc).RowIsValid(0)) continue;
        if (!agg.per_gpu_state[g]) continue;
        const auto& rstate =
          static_cast<const UngroupedAggregateRuntimeState&>(*agg.per_gpu_state[g]);
        if (col >= rstate.avg_valid_counts.size()) continue;
        const uint64_t cnt = rstate.avg_valid_counts[col];
        if (cnt == 0) continue;
        const double mean = (pt == PhysicalType::DOUBLE)
                              ? FlatVector::GetData<double>(vsrc)[0]
                              : static_cast<double>(FlatVector::GetData<float>(vsrc)[0]);
        total += mean * static_cast<double>(cnt);
        grand_count += cnt;
      }
      if (grand_count > 0) {
        const double result = total / static_cast<double>(grand_count);
        if (pt == PhysicalType::DOUBLE) {
          *reinterpret_cast<double*>(dst_data) = result;
        } else {
          *reinterpret_cast<float*>(dst_data) = static_cast<float>(result);
        }
      } else {
        *dst_mask = 0x00;
      }
    } else {
      throw NotImplementedException(
        "Cross-GPU ungrouped aggregate not supported for function: %s", fname);
    }

    Vector v(types[col], dst_data);
    ValidityMask vmask(reinterpret_cast<validity_t*>(dst_mask), 1);
    FlatVector::SetValidity(v, vmask);
    merged.data[col].Reference(v);
  }
  merged.SetCardinality(1);
  reduced->AddChunk(merged);
  return reduced;
}

// Build a GPUResultCollection holding one 1-row HOST DataChunk per GPU from the
// UNGROUPED_AGGREGATE node's retained per-GPU RAW partials (the pre-projection
// aggregate columns staged in UngroupedAggregateRuntimeState::aggregation_result
// by GPUPhysicalUngroupedAggregate::Sink). Used by the PROJECTION-over-
// UNGROUPED_AGGREGATE multi-GPU path: the per-GPU result_collection chunks hold
// the (per-GPU, hence wrong) projected ratios, so we cannot reduce those; we
// re-materialise the raw partials here and reduce THOSE instead.
//
// `agg_types` are the aggregate node's output types (NOT the collector's
// projected types). The returned collection's write_idx equals the number of
// GPUs that produced a partial.
unique_ptr<GPUResultCollection> CollectUngroupedRawPartials(
  const GPUPhysicalUngroupedAggregate& agg,
  const vector<LogicalType>& agg_types,
  GPUBufferManager* gbm)
{
  auto partials = make_uniq<GPUResultCollection>();
  const int saved_gpu = sirius_current_gpu;
  int prev_device     = 0;
  cudaGetDevice(&prev_device);
  for (int g = 0; g < GPUBufferManager::GetMaxGpus(); ++g) {
    if (!agg.per_gpu_state[g]) continue;
    auto& rstate = static_cast<UngroupedAggregateRuntimeState&>(*agg.per_gpu_state[g]);
    if (!rstate.aggregation_result) continue;
    // The conversion (and the cudf ops it may invoke) must target GPU g, and
    // ConvertGPUTableToCPUCollection reads `sirius_current_gpu` for its
    // device->host memcpy. Pin both for the duration of this GPU's partial.
    sirius_current_gpu = g;
    cudaSetDevice(g);
    GPUPhysicalMaterializedCollector::ConvertGPUTableToCPUCollection(
      *rstate.aggregation_result, agg_types, partials.get(), gbm);
  }
  cudaSetDevice(prev_device);
  sirius_current_gpu = saved_gpu;
  return partials;
}

unique_ptr<QueryResult> GPUPhysicalMaterializedCollector::GetResult(GlobalSinkState& state)
{
  auto& gstate = state.Cast<GPUMaterializedCollectorGlobalState>();
  if (!gstate.context)
    throw InvalidInputException("No context set in GPUMaterializedCollectorState");
  auto prop = gstate.context->GetClientProperties();

  // Concatenate the per-GPU result_collections (in GPU id order) into one
  // combined collection that GPUQueryResult will hand out chunk-by-chunk.
  auto combined = make_uniq<GPUResultCollection>();
  for (int g = 0; g < GPUBufferManager::GetMaxGpus(); ++g) {
    if (!per_gpu_state[g]) continue;
    auto& rstate = static_cast<ResultCollectorRuntimeState&>(*per_gpu_state[g]);
    if (!rstate.result_collection) continue;
    auto& src = *rstate.result_collection;
    if (src.write_idx == 0) continue;  // empty per-GPU collection
    combined->SetCapacity(src.write_idx);
    for (size_t i = 0; i < src.write_idx; ++i) {
      // Move chunks across; src is destructed when its RuntimeState is freed.
      combined->AddChunk(src.data_chunks[i]);
    }
    // Release the per-GPU collection now that we've moved its content.
    rstate.result_collection.reset();
  }

  // Option 3 fused join->aggregate: the magi join evaluated the aggregate-input
  // expression(s) per match and accumulated per-GPU partial sums into the side
  // channel (no materialized join output, no separate projection/aggregate pass).
  // Reduce those partials across GPUs and apply the query's top projection here,
  // overriding the sentinel-row output the trivial downstream pipeline produced.
  {
    auto& ch = duckdb::magi_fused::fused_channel();
    if (ch.active && plan.type == PhysicalOperatorType::PROJECTION && !plan.children.empty()) {
      const auto& agg_types = plan.children[0]->types;
      DataChunk agg_chunk;
      agg_chunk.Initialize(*gstate.context, agg_types);
      for (int a = 0; a < ch.n_aggs; ++a) {
        double g = 0.0;
        for (int gpu = 0; gpu < duckdb::magi_fused::FUSED_MAX_GPUS; ++gpu)
          g += ch.partials[gpu][a];
        agg_chunk.SetValue(a, 0, Value::DOUBLE(g));
      }
      agg_chunk.SetCardinality(1);
      ExpressionExecutor executor(*gstate.context,
                                  plan.Cast<GPUPhysicalProjection>().select_list);
      DataChunk projected;
      projected.Initialize(*gstate.context, types);
      executor.Execute(agg_chunk, projected);
      auto final_coll = make_uniq<GPUResultCollection>();
      final_coll->SetCapacity(1);
      final_coll->AddChunk(projected);
      combined = std::move(final_coll);
      for (int gpu = 0; gpu < duckdb::magi_fused::FUSED_MAX_GPUS; ++gpu)
        for (int a = 0; a < duckdb::magi_fused::MAX_FUSED_AGGS; ++a) ch.partials[gpu][a] = 0.0;
      ch.active = false;
    }
  }

  // Cross-GPU final reduce: each per-GPU run of an UNGROUPED_AGGREGATE
  // produced a 1-row partial. Without this, COUNT(*) on an N-GPU partition
  // would return N rows instead of one.
  if (combined->write_idx > 1 && plan.type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
    combined = ReduceUngroupedAcrossGpus(
      *combined, plan.Cast<GPUPhysicalUngroupedAggregate>(), types, gpuBufferManager);
  } else if (combined->write_idx > 1 && plan.type == PhysicalOperatorType::PROJECTION &&
             !plan.children.empty() &&
             plan.children[0]->type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
    // PROJECTION over an UNGROUPED_AGGREGATE on multi-GPU: the per-GPU pipeline
    // ran the projection on each GPU's PARTIAL aggregate, so the per-GPU
    // result_collection chunks in `combined` are per-GPU ratios (e.g. Q14's
    // 100*p_g/t_g) that must NOT be concatenated. Instead: (1) gather the RAW
    // pre-projection aggregate partials retained per-GPU, (2) reduce those raw
    // partials across GPUs into a single row, (3) apply the projection on the
    // HOST over that single reduced row. See gpu_physical_result_collector.cpp
    // history / MEMORY for the bug analysis.
    auto& agg            = plan.children[0]->Cast<GPUPhysicalUngroupedAggregate>();
    const auto& agg_types = plan.children[0]->types;

    // (1) per-GPU raw aggregate partials, re-materialised host-side.
    auto raw_partials = CollectUngroupedRawPartials(agg, agg_types, gpuBufferManager);

    if (raw_partials->write_idx > 1) {
      // (2) cross-GPU reduce of the RAW partials (sum/min/max/avg/count).
      auto reduced =
        ReduceUngroupedAcrossGpus(*raw_partials, agg, agg_types, gpuBufferManager);

      // (3) apply the projection on the host over the single reduced row,
      // producing the final 1-row chunk in the collector's projected `types`.
      // `reduced->data_chunks[0]` is already a valid 1-row DataChunk in
      // `agg_types`; feed it straight to the host ExpressionExecutor.
      ExpressionExecutor executor(*gstate.context, plan.Cast<GPUPhysicalProjection>().select_list);
      DataChunk projected;
      projected.Initialize(*gstate.context, types);
      executor.Execute(reduced->data_chunks[0], projected);

      auto final_coll = make_uniq<GPUResultCollection>();
      final_coll->SetCapacity(1);
      final_coll->AddChunk(projected);
      combined = std::move(final_coll);
    }
    // If raw_partials->write_idx <= 1 (only one GPU actually produced a
    // partial), the existing per-GPU projected row in `combined` is already
    // correct -- leave it untouched.
  }

  auto result = make_uniq<GPUQueryResult>(
    statement_type, properties, names, types, prop, std::move(combined));
  return std::move(result);
}

// bool PhysicalMaterializedCollector::ParallelSink() const {
// 	return parallel;
// }

// bool PhysicalMaterializedCollector::SinkOrderDependent() const {
// 	return true;
// }

}  // namespace duckdb
