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

#include "duckdb/common/optionally_owned_ptr.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {

//! The PhysicalColumnDataScan scans a ColumnDataCollection
class GPUPhysicalColumnDataScan : public GPUPhysicalOperator {
 public:
  static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

 public:
  GPUPhysicalColumnDataScan(vector<LogicalType> types,
                            PhysicalOperatorType op_type,
                            idx_t estimated_cardinality,
                            optionally_owned_ptr<ColumnDataCollection> collection);

  GPUPhysicalColumnDataScan(vector<LogicalType> types,
                            PhysicalOperatorType op_type,
                            idx_t estimated_cardinality,
                            idx_t cte_index);

  //! (optionally owned) column data collection to scan
  optionally_owned_ptr<ColumnDataCollection> collection;

  idx_t cte_index;
  optional_idx delim_index;

  shared_ptr<GPUIntermediateRelation> intermediate_relation;

  //! Per-GPU relations for CTE_SCAN under multi-GPU, indexed by
  //! sirius_current_gpu (shared with GPUPhysicalCTE::working_table_gpu).
  //! When set it takes precedence over intermediate_relation.
  shared_ptr<vector<shared_ptr<GPUIntermediateRelation>>> per_gpu_relations;

  // CTE_SCAN only: the GPUPhysicalCTE this scan reads (set in BuildPipelines).
  // Lets SubtreeOutputReplicated see through the CTE into its definition —
  // a CTE materialized from a replicated-output subtree (e.g. Q2's europe
  // supplier chain over replicated-cached tables) holds a FULL copy per GPU;
  // treating the scan as partitioned made downstream joins pick strategies
  // that emitted every result row once per GPU.
  const GPUPhysicalOperator* cte_op = nullptr;

 public:
  SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

  bool IsSource() const override { return true; }

  void BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline) override;
};

}  // namespace duckdb
