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

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/enums/operator_result_type.hpp"
#include "duckdb/common/enums/order_preservation_type.hpp"
#include "duckdb/common/enums/physical_operator_type.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_operator_states.hpp"
#include "duckdb/optimizer/join_order/join_node.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_columns.hpp"
#include "helper/types.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <array>
#include <memory>
#include <mutex>

namespace duckdb {
class GPUExecutor;
class GPUPhysicalOperator;
class GPUPipeline;
}  // namespace duckdb

namespace duckdb {
class GPUPipelineBuildState;
class GPUMetaPipeline;

enum class MemoryBarrierType { PIPELINE, PARTIAL, FULL };

//! Base class for per-GPU runtime state attached to a GPUPhysicalOperator.
//!
//! Phase B (concurrent multi-GPU execute) requires that operator instances be
//! shared between worker threads (one thread per GPU) but that any *mutable*
//! per-query / per-execution state be isolated per thread. We carry that
//! isolated state in subclasses of OpRuntimeState held in the operator's
//! per_gpu_state vector, indexed by `sirius_current_gpu`.
//!
//! Each concrete operator that has runtime-mutable members defines its own
//! `<Op>RuntimeState : OpRuntimeState` subclass and accesses it through
//! `runtime_state<...>(gpu)`.
class OpRuntimeState {
 public:
  virtual ~OpRuntimeState() = default;
};

//! GPUPhysicalOperator is the base class of the physical operators present in the
//! execution plan
class GPUPhysicalOperator {
 public:
  static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

 public:
  GPUPhysicalOperator(PhysicalOperatorType type,
                      vector<LogicalType> types,
                      idx_t estimated_cardinality)
    : type(type), types(std::move(types)), estimated_cardinality(estimated_cardinality)
  {
    per_gpu_state.resize(GPUBufferManager::GetMaxGpus());
  }
  GPUPhysicalOperator() { per_gpu_state.resize(GPUBufferManager::GetMaxGpus()); }

  virtual ~GPUPhysicalOperator() {}

  //! The physical operator type
  PhysicalOperatorType type;
  //! The set of children of the operator
  vector<unique_ptr<GPUPhysicalOperator>> children;
  //! The types returned by this physical operator
  vector<LogicalType> types;
  //! The estimated cardinality of this physical operator
  idx_t estimated_cardinality;

  //! The global sink state of this operator
  unique_ptr<GlobalSinkState> sink_state;
  //! The global state of this operator
  unique_ptr<GlobalOperatorState> op_state;
  //! Lock for (re)setting any of the operator states
  mutex lock;

  //! Per-GPU runtime state. Lazy-initialised by `runtime_state<T>(gpu)`.
  //! `mutable` so const member functions (e.g. GetData) can lazily initialise.
  mutable vector<unique_ptr<OpRuntimeState>> per_gpu_state;
  //! Guards the lazy creation of per_gpu_state slots.
  mutable std::mutex per_gpu_state_mutex;

  //! Returns (and lazy-creates) this operator's per-GPU runtime state for
  //! `gpu`. T must derive from OpRuntimeState. Each concrete operator that
  //! holds runtime state declares a single `T` subclass and uses this helper.
  template <typename T>
  T& runtime_state(int gpu) const
  {
    std::lock_guard<std::mutex> lk(per_gpu_state_mutex);
    auto& slot = per_gpu_state[gpu];
    if (!slot) { slot = std::make_unique<T>(); }
    return static_cast<T&>(*slot);
  }

 public:
  virtual string GetName() const;
  // virtual string ParamsToString() const {
  // 	return "";
  // }
  // virtual string ToString() const;
  // void Print() const;
  virtual vector<const_reference<GPUPhysicalOperator>> GetChildren() const;

  //! Return a vector of the types that will be returned by this operator
  const vector<LogicalType>& GetTypes() const { return types; }

  virtual bool Equals(const GPUPhysicalOperator& other) const { return false; }

  virtual void Verify();

 public:
  // Operator interface
  virtual unique_ptr<OperatorState> GetOperatorState(ExecutionContext& context) const;
  virtual unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext& context) const;

  virtual OperatorResultType Execute(GPUIntermediateRelation& input_relation,
                                     GPUIntermediateRelation& output_relation) const;

  virtual bool ParallelOperator() const { return false; }

  virtual bool RequiresFinalExecute() const { return false; }

  //! The influence the operator has on order (insertion order means no influence)
  virtual OrderPreservationType OperatorOrder() const
  {
    return OrderPreservationType::INSERTION_ORDER;
  }

 public:
  // Source Interface
  virtual SourceResultType GetData(GPUIntermediateRelation& output_relation) const;
  // TODO: Implement SourceExecute if required in future.

  virtual unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext& context,
                                                           GlobalSourceState& gstate) const;
  virtual unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext& context) const;

  virtual bool IsSource() const { return false; }

  virtual bool ParallelSource() const { return false; }

  //! The type of order emitted by the operator (as a source)
  virtual OrderPreservationType SourceOrder() const
  {
    return OrderPreservationType::INSERTION_ORDER;
  }

 public:
  // Sink interface
  virtual SinkResultType Sink(GPUIntermediateRelation& input_relation) const;
  //! Called once per worker after the LAST pipeline that sinks into this
  //! operator completes (the executor precomputes that pipeline from the
  //! static schedule, so every worker finalizes at the same point). Sinks
  //! that can legally receive MULTIPLE batches per query — a RIGHT/OUTER
  //! join feeds its sink from both the probe pipeline and the unmatched-scan
  //! child pipeline — accumulate in Sink and do their real work here.
  //! Default: no-op (single-batch sinks keep working in Sink).
  virtual void FinalizeSink() const {}

  virtual SinkFinalizeType CombineFinalize(vector<shared_ptr<GPUIntermediateRelation>>& input,
                                           GPUIntermediateRelation& output) const;
  virtual unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext& context) const;
  virtual unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext& context) const;

  virtual bool IsSink() const { return false; }

  virtual bool ParallelSink() const { return false; }

  virtual bool RequiresBatchIndex() const { return false; }

  //! Whether or not the sink operator depends on the order of the input chunks
  //! If this is set to true, we cannot do things like caching intermediate vectors
  virtual bool SinkOrderDependent() const { return false; }

 public:
  // Pipeline construction
  virtual vector<const_reference<GPUPhysicalOperator>> GetSources() const;

  virtual void BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline);

 public:
  template <class TARGET>
  TARGET& Cast()
  {
    if (TARGET::TYPE != PhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw InternalException(
        "Failed to cast physical operator to type - physical operator type mismatch");
    }
    return reinterpret_cast<TARGET&>(*this);
  }

  template <class TARGET>
  const TARGET& Cast() const
  {
    if (TARGET::TYPE != PhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw InternalException(
        "Failed to cast physical operator to type - physical operator type mismatch");
    }
    return reinterpret_cast<const TARGET&>(*this);
  }
};

}  // namespace duckdb
