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

#pragma once

#include "duckdb/common/enums/tuple_data_layout_enums.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "gpu_physical_operator.hpp"

#include <condition_variable>
#include <mutex>

namespace duckdb {
using sirius::AggregationType;

void cudf_aggregate(vector<shared_ptr<GPUColumn>>& column,
                    uint64_t num_aggregates,
                    AggregationType* agg_mode);

// Per-GPU runtime state for ungrouped aggregate. Each GPU worker thread runs
// the same UNGROUPED_AGGREGATE op on its own partition; the partial 1-row
// result is staged here. The cross-GPU final reduce happens later in
// GPUPhysicalMaterializedCollector::GetResult.
struct UngroupedAggregateRuntimeState : OpRuntimeState {
  shared_ptr<GPUIntermediateRelation> aggregation_result;
  //! Multi-batch stash: a RIGHT/OUTER join upstream legally sinks twice
  //! (matched + unmatched batches, see the grouped aggregate's FinalizeSink
  //! contract). Sink materializes and stashes each batch's aggregate-input
  //! columns; FinalizeSink concatenates and aggregates ONCE. Single batch =
  //! the same work, just moved.
  vector<vector<shared_ptr<GPUColumn>>> pending;
  vector<uint64_t>                      pending_sizes;
  // For AVG aggregates: per-aggregate non-null row count on this GPU's
  // partition. Sized aggregates.size(); 0 for non-AVG indices. Cross-GPU
  // reduce uses this to compute a count-weighted average from the per-GPU
  // means (cuDF MEAN reduction discards the count after producing the mean).
  vector<uint64_t> avg_valid_counts;
  // Interior multi-GPU merge (see interior_cross_gpu_merge below): the merged
  // GLOBAL 1-row result uploaded onto this GPU, built lazily by GetData once
  // every GPU's Sink has staged its partial. Cached so repeated GetData calls
  // re-emit without re-merging.
  shared_ptr<GPUIntermediateRelation> merged_result;
};

class GPUPhysicalUngroupedAggregate : public GPUPhysicalOperator {
 public:
  static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::UNGROUPED_AGGREGATE;

 public:
  GPUPhysicalUngroupedAggregate(vector<LogicalType> types,
                                vector<unique_ptr<Expression>> select_list,
                                idx_t estimated_cardinality,
                                TupleDataValidityType distinct_validity);

  //! The aggregates that have to be computed
  vector<unique_ptr<Expression>> aggregates;
  unique_ptr<DistinctAggregateData> distinct_data;
  unique_ptr<DistinctAggregateCollectionInfo> distinct_collection_info;

  //! Set at result-collector construction when this aggregate is an INTERIOR
  //! node of a multi-GPU plan (its output feeds another operator rather than
  //! the result collector). GetData then performs the cross-GPU merge of the
  //! per-GPU partials itself; top-level aggregates keep the collector-side
  //! reduce in GPUPhysicalMaterializedCollector::GetResult.
  bool interior_cross_gpu_merge = false;
  //! Sink-completion tracking for the interior merge: how many GPUs have
  //! staged their 1-row partial. Guarded by sink_mutex; GetData waits on
  //! sink_cv until all execution GPUs have sunk (or times out -> fallback).
  mutable std::mutex sink_mutex;
  mutable std::condition_variable sink_cv;
  mutable int gpus_sunk = 0;

  SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

  //! Interior multi-GPU path of GetData: wait for every GPU's partial, merge
  //! them into the global 1-row result (reusing the collector's helpers), and
  //! emit that merged row on the calling GPU.
  SourceResultType GetDataInteriorMerged(GPUIntermediateRelation& output_relation) const;

  bool IsSource() const override { return true; }

 public:
  SinkResultType Sink(GPUIntermediateRelation& input_relation) const override;
  void FinalizeSink() const override;

  bool IsSink() const override { return true; }

  bool ParallelSink() const override { return true; }

  void MaterializeDistinctInput(GPUIntermediateRelation& input_relation,
                                vector<shared_ptr<GPUColumn>>& aggregate_column) const;
};
}  // namespace duckdb
