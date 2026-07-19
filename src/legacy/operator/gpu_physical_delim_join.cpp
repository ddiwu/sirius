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

#include "operator/gpu_physical_delim_join.hpp"

#include "gpu_meta_pipeline.hpp"
#include "gpu_pipeline.hpp"
#include "log/logging.hpp"
#include "gpu_buffer_manager.hpp"
#include "operator/gpu_physical_column_data_scan.hpp"
#include "operator/gpu_physical_dummy_scan.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "operator/gpu_physical_hash_join.hpp"

namespace duckdb {

class GPULeftDelimJoinLocalState : public LocalSinkState {
 public:
  unique_ptr<LocalSinkState> distinct_state;
  shared_ptr<GPUIntermediateRelation> lhs_data;
  ColumnDataAppendState append_state;

  // void Append(DataChunk &input) {
  // 	lhs_data.Append(input);
  // }
};

class GPURightDelimJoinLocalState : public LocalSinkState {
 public:
  unique_ptr<LocalSinkState> join_state;
  unique_ptr<LocalSinkState> distinct_state;
};

GPUPhysicalDelimJoin::GPUPhysicalDelimJoin(PhysicalOperatorType type,
                                           vector<LogicalType> types,
                                           unique_ptr<GPUPhysicalOperator> original_join,
                                           vector<const_reference<GPUPhysicalOperator>> delim_scans,
                                           idx_t estimated_cardinality,
                                           optional_idx delim_idx)
  : GPUPhysicalOperator(type, std::move(types), estimated_cardinality),
    join(std::move(original_join)),
    delim_scans(std::move(delim_scans))
{
  D_ASSERT(type == PhysicalOperatorType::LEFT_DELIM_JOIN ||
           type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
}

GPUPhysicalRightDelimJoin::GPUPhysicalRightDelimJoin(
  vector<LogicalType> types,
  unique_ptr<GPUPhysicalOperator> original_join,
  vector<const_reference<GPUPhysicalOperator>> delim_scans,
  idx_t estimated_cardinality,
  optional_idx delim_idx)
  : GPUPhysicalDelimJoin(PhysicalOperatorType::RIGHT_DELIM_JOIN,
                         std::move(types),
                         std::move(original_join),
                         std::move(delim_scans),
                         estimated_cardinality,
                         delim_idx)
{
  D_ASSERT(join->children.size() == 2);
  children.push_back(std::move(join->children[1]));

  // we replace it with a PhysicalDummyScan, which contains no data, just the types, it won't be
  // scanned anyway
  join->children[1] =
    make_uniq<GPUPhysicalDummyScan>(children[0]->GetTypes(), estimated_cardinality);
}

GPUPhysicalLeftDelimJoin::GPUPhysicalLeftDelimJoin(
  vector<LogicalType> types,
  unique_ptr<GPUPhysicalOperator> original_join,
  vector<const_reference<GPUPhysicalOperator>> delim_scans,
  idx_t estimated_cardinality,
  optional_idx delim_idx)
  : GPUPhysicalDelimJoin(PhysicalOperatorType::LEFT_DELIM_JOIN,
                         std::move(types),
                         std::move(original_join),
                         std::move(delim_scans),
                         estimated_cardinality,
                         delim_idx)
{
  D_ASSERT(join->children.size() == 2);
  // now for the original join
  // we take its left child, this is the side that we will duplicate eliminate
  children.push_back(std::move(join->children[0]));

  // we replace it with a PhysicalColumnDataScan, that scans the ColumnDataCollection that we keep
  // cached the actual chunk collection to scan will be created in the LeftDelimJoinGlobalState
  auto cached_chunk_scan =
    make_uniq<GPUPhysicalColumnDataScan>(children[0]->GetTypes(),
                                         PhysicalOperatorType::COLUMN_DATA_SCAN,
                                         estimated_cardinality,
                                         nullptr);
  if (delim_idx.IsValid()) { cached_chunk_scan->cte_index = delim_idx.GetIndex(); }
  join->children[0] = std::move(cached_chunk_scan);
}

SinkResultType GPUPhysicalRightDelimJoin::Sink(GPUIntermediateRelation& input_relation) const
{
  // auto &lstate = input.local_state.Cast<GPURightDelimJoinLocalState>();

  // OperatorSinkInput join_sink_input {*join->sink_state, *lstate.join_state,
  // input.interrupt_state}; join->Sink(context, input_relation, join_sink_input);
  SIRIUS_LOG_DEBUG("Sinking input relation to join");
  join->Sink(input_relation);

  // OperatorSinkInput distinct_sink_input {*distinct->sink_state, *lstate.distinct_state,
  // input.interrupt_state}; distinct->Sink(context, input_relation, distinct_sink_input);
  SIRIUS_LOG_DEBUG("Sinking input relation to distinct group by");
  distinct->Sink(input_relation);
  // The distinct groupby is driven INTERNALLY (not a scheduled pipeline
  // sink), so the executor's finalize-point pass can't see it — finalize
  // here. The delim Sink runs once per worker, so this stays aligned.
  distinct->FinalizeSink();

  return SinkResultType::FINISHED;
}

SinkResultType GPUPhysicalLeftDelimJoin::Sink(GPUIntermediateRelation& input_relation) const
{
  // auto &lstate = input.local_state.Cast<GPULeftDelimJoinLocalState>();
  // lstate.lhs_data.Append(lstate.append_state, chunk);
  auto& cached_chunk_scan = join->children[0]->Cast<GPUPhysicalColumnDataScan>();
  // Each worker caches ITS partition into its own slot (see BuildPipelines);
  // writing the shared intermediate_relation raced across workers.
  auto slot = make_shared_ptr<GPUIntermediateRelation>(input_relation.columns.size());
  for (int i = 0; i < input_relation.columns.size(); i++) {
    SIRIUS_LOG_DEBUG("Passing input relation idx {} to column data scan idx {}", i, i);
    slot->columns[i]      = input_relation.columns[i];
    slot->column_names[i] = input_relation.column_names[i];
    slot->names           = input_relation.names;
  }
  (*cached_chunk_scan.per_gpu_relations)[sirius_current_gpu] = std::move(slot);

  distinct->Sink(input_relation);
  // Internally-driven distinct (not a scheduled pipeline sink) — finalize
  // here; the delim Sink runs once per worker, so this stays aligned.
  distinct->FinalizeSink();
  return SinkResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void GPUPhysicalLeftDelimJoin::BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline)
{
  op_state.reset();
  sink_state.reset();

  // Per-GPU slots for the cached delim input (single-threaded plan phase).
  // Sink used to write the SHARED intermediate_relation -- last worker won and
  // every join probe then read ONE GPU's partition on BOTH GPUs (the same
  // race the materialized-CTE fix addressed; first hit by Q2's correlated
  // min subquery).
  auto& cached_chunk_scan = join->children[0]->Cast<GPUPhysicalColumnDataScan>();
  cached_chunk_scan.per_gpu_relations =
    make_shared_ptr<vector<shared_ptr<GPUIntermediateRelation>>>(8);

  auto& child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
  child_meta_pipeline.Build(*children[0]);

  D_ASSERT(type == PhysicalOperatorType::LEFT_DELIM_JOIN);
  // recurse into the actual join
  // any pipelines in there depend on the main pipeline
  // any scan of the duplicate eliminated data on the RHS depends on this pipeline
  // we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
  auto& state = meta_pipeline.GetState();
  for (auto& delim_scan : delim_scans) {
    state.delim_join_dependencies.insert(
      make_pair(delim_scan, reference<GPUPipeline>(*child_meta_pipeline.GetBasePipeline())));
  }
  join->BuildPipelines(current, meta_pipeline);
}

void GPUPhysicalRightDelimJoin::BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline)
{
  op_state.reset();
  sink_state.reset();

  auto& child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
  child_meta_pipeline.Build(*children[0]);

  D_ASSERT(type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
  // recurse into the actual join
  // any pipelines in there depend on the main pipeline
  // any scan of the duplicate eliminated data on the LHS depends on this pipeline
  // we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
  auto& state = meta_pipeline.GetState();
  for (auto& delim_scan : delim_scans) {
    state.delim_join_dependencies.insert(
      make_pair(delim_scan, reference<GPUPipeline>(*child_meta_pipeline.GetBasePipeline())));
  }

  // Build join pipelines without building the RHS (already built in the Sink of this op)
  GPUPhysicalHashJoin::BuildJoinPipelines(current, meta_pipeline, *join, false);
}

}  // namespace duckdb
