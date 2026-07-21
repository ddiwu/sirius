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

#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "operator/magi_groupby.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/radix_partitioned_hashtable.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/storage/data_table.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {
using sirius::AggregationType;

uint64_t* createFixedSizeOffsets(size_t record_size, uint64_t num_rows);

void cudf_groupby(vector<shared_ptr<GPUColumn>>& keys,
                  vector<shared_ptr<GPUColumn>>& aggregate_keys,
                  uint64_t num_keys,
                  uint64_t num_aggregates,
                  AggregationType* agg_mode,
                  idx_t estimated_output_groups = 0);

void cudf_duplicate_elimination(vector<shared_ptr<GPUColumn>>& keys, uint64_t num_keys);

template <typename T>
void combineColumns(T* a, T* b, T*& c, uint64_t N_a, uint64_t N_b);

void combineStrings(uint8_t* a,
                    uint8_t* b,
                    uint8_t*& c,
                    uint64_t* offset_a,
                    uint64_t* offset_b,
                    uint64_t*& offset_c,
                    uint64_t num_bytes_a,
                    uint64_t num_bytes_b,
                    uint64_t N_a,
                    uint64_t N_b);

void combineMasks(
  cudf::bitmask_type* a, cudf::bitmask_type* b, cudf::bitmask_type*& c, uint64_t N_a, uint64_t N_b);

class ClientContext;

// Per-GPU runtime state for grouped aggregate. Magi shuffles input rows
// into per-GPU hash-disjoint partitions, so each worker thread's Sink
// produces a disjoint slice of (key, agg) output rows. Storing the slice
// per-worker (instead of merging into a shared `group_by_result`) lets
// GetData emit each worker's slice independently — no Sink-side mutex,
// no shared-state read race, full downstream pipeline parallelism across
// workers.
struct GroupedAggregateRuntimeState : OpRuntimeState {
  shared_ptr<GPUIntermediateRelation> slice;
  //! Materialized input batches accumulated across Sink calls. A RIGHT/OUTER
  //! join upstream legally sinks TWICE (matched pairs from the probe pipeline
  //! + NULL-padded unmatched rows from its unmatched-scan child pipeline);
  //! the aggregation itself runs once in FinalizeSink over the union.
  //! Layout per batch: [group cols..., agg cols...]; rows[] holds each
  //! batch's logical row count (columns may be 0-length NULL placeholders).
  vector<shared_ptr<GPUIntermediateRelation>> pending;
  vector<uint64_t>                            pending_rows;
};

// Concatenate two dense GPU columns of the same type (defined in
// gpu_physical_grouped_aggregate.cpp; also used by the ungrouped aggregate's
// multi-batch FinalizeSink).
shared_ptr<GPUColumn> CombineColumns(shared_ptr<GPUColumn> column1,
                                     shared_ptr<GPUColumn> column2,
                                     GPUBufferManager* gpuBufferManager);

class GPUPhysicalGroupedAggregate : public GPUPhysicalOperator {
 public:
  GPUPhysicalGroupedAggregate(ClientContext& context,
                              vector<LogicalType> types,
                              vector<unique_ptr<Expression>> expressions,
                              idx_t estimated_cardinality);

  GPUPhysicalGroupedAggregate(ClientContext& context,
                              vector<LogicalType> types,
                              vector<unique_ptr<Expression>> expressions,
                              vector<unique_ptr<Expression>> groups,
                              idx_t estimated_cardinality);

  GPUPhysicalGroupedAggregate(ClientContext& context,
                              vector<LogicalType> types,
                              vector<unique_ptr<Expression>> expressions,
                              vector<unique_ptr<Expression>> groups,
                              vector<GroupingSet> grouping_sets,
                              vector<unsafe_vector<idx_t>> grouping_functions,
                              idx_t estimated_cardinality,
                              TupleDataValidityType group_validity,
                              TupleDataValidityType distinct_validity);

  //! The grouping sets
  GroupedAggregateData grouped_aggregate_data;

  // HAVING pushdown (set at plan time when a FILTER(agg cmp constant) sits
  // directly above this aggregate): applied inside the magi flush compaction
  // so only qualifying groups are extracted. The plan FILTER stays as the
  // exact residual check. cmp==0 -> disabled.
  magi_groupby::SlotPredicate having_pushdown;

  vector<GroupingSet> grouping_sets;
  //! The radix partitioned hash tables (one per grouping set)
  vector<HashAggregateGroupingData> groupings;
  unique_ptr<DistinctAggregateCollectionInfo> distinct_collection_info;
  //! A recreation of the input chunk, with nulls for everything that isn't a group
  vector<LogicalType> input_group_types;

  // Filters given to Sink and friends
  unsafe_vector<idx_t> non_distinct_filter;
  unsafe_vector<idx_t> distinct_filter;

  unordered_map<Expression*, size_t> filter_indexes;

  shared_ptr<GPUIntermediateRelation> group_by_result;

 public:
  // Source interface
  SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

  // Source interface
  bool IsSource() const override { return true; }
  bool ParallelSource() const override { return true; }

  OrderPreservationType SourceOrder() const override { return OrderPreservationType::NO_ORDER; }

 public:
  // Sink interface. Sink materializes and STASHES each input batch;
  // FinalizeSink (once per worker, after the last sinking pipeline) runs the
  // actual aggregation over the union — see GroupedAggregateRuntimeState.
  SinkResultType Sink(GPUIntermediateRelation& input_relation) const override;
  void FinalizeSink() const override;

  // Sink interface
  bool IsSink() const override { return true; }

  bool ParallelSink() const override { return true; }

  bool SinkOrderDependent() const override { return false; }

 private:
  static bool CheckGroupKeyTypesForSiriusImpl(const vector<shared_ptr<GPUColumn>>& columns);
  //! The actual aggregation (cudf pre-agg decision + magi shuffle or the
  //! distinct-only path) over already-materialized columns; fills the
  //! per-worker slice. Extracted from the old single-batch Sink.
  void RunAggregation(vector<shared_ptr<GPUColumn>>& group_by_column,
                      vector<shared_ptr<GPUColumn>>& aggregate_column,
                      uint64_t column_size) const;
};
}  // namespace duckdb
