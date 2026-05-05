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
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_context.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "gpu_pipeline.hpp"
#include "log/logging.hpp"
#include "operator/gpu_materialize.hpp"
#include "operator/gpu_physical_ungrouped_aggregate.hpp"
#include "utils.hpp"

namespace duckdb {

GPUPhysicalResultCollector::GPUPhysicalResultCollector(GPUPreparedStatementData& data)
  : GPUPhysicalOperator(PhysicalOperatorType::RESULT_COLLECTOR, {LogicalType::BOOLEAN}, 0),
    statement_type(data.prepared->statement_type),
    properties(data.prepared->properties),
    plan(*data.gpu_physical_plan),
    names(data.prepared->names)
{
  this->types      = data.prepared->types;
  gpuBufferManager = &(GPUBufferManager::GetInstance());
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

// Cross-GPU final reduce of an UNGROUPED_AGGREGATE. Each per-GPU pipeline
// produced a 1-row partial. Collapse those N rows into a single row by
// applying each aggregate's reducer (SUM-merge for COUNT/COUNT_STAR/SUM/
// SUM_NO_OVERFLOW). Other functions throw — re-evaluating MIN/MAX/AVG across
// partials needs more thought (AVG is not associative, MIN/MAX would work
// trivially but aren't wired up yet).
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

}  // namespace

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

  // Cross-GPU final reduce: each per-GPU run of an UNGROUPED_AGGREGATE
  // produced a 1-row partial. Without this, COUNT(*) on an N-GPU partition
  // would return N rows instead of one.
  if (combined->write_idx > 1 && plan.type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
    combined = ReduceUngroupedAcrossGpus(
      *combined, plan.Cast<GPUPhysicalUngroupedAggregate>(), types, gpuBufferManager);
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
