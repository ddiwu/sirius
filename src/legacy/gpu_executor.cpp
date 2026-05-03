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

#include "gpu_executor.hpp"

#include "config.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "fallback.hpp"
#include "gpu_physical_operator.hpp"
#include "log/logging.hpp"
#include "operator/gpu_physical_hash_join.hpp"
#include "operator/gpu_physical_result_collector.hpp"
#include "operator/gpu_physical_table_scan.hpp"

#include <nvtx3/nvtx3.hpp>

#include <stdio.h>

#include <atomic>
#include <exception>
#include <mutex>
#include <thread>

namespace duckdb {

namespace {
// Serializes the cache-load section (GetDataDuckDB) across per-GPU worker
// threads in parallel Execute(). The first thread to enter this section
// loads the cache for ALL GPUs (Phase 1's per-GPU partitioning runs inside
// GetDataDuckDB); subsequent threads observe `already_cached=true` on every
// column and short-circuit. Without this lock, multiple threads would race
// on TableScan's shared mutable members (column_size, num_rows, ...).
std::mutex g_cache_load_mutex;
}  // namespace

void GPUExecutor::Reset()
{
  // lock_guard<mutex> elock(executor_lock);
  gpu_physical_plan = nullptr;
  // cancelled = false;
  gpu_owned_plan.reset();
  // root_executor.reset();
  root_pipelines.clear();
  root_pipeline_idx   = 0;
  completed_pipelines = 0;
  total_pipelines     = 0;
  // error_manager.Reset();
  pipelines.clear();
  next_operator_id.store(0);
  // events.clear();
  // to_be_rescheduled_tasks.clear();
  // execution_result = PendingExecutionResult::RESULT_NOT_READY;
}

void GPUExecutor::Initialize(unique_ptr<GPUPhysicalOperator> plan)
{
  SIRIUS_LOG_DEBUG("Initializing GPUExecutor");
  Reset();
  gpu_owned_plan = std::move(plan);
  InitializeInternal(*gpu_owned_plan);
}

void GPUExecutor::Execute()
{
  nvtx3::scoped_range nvtx_range{"sirius::legacy_query"};

  // Check if we should fall back to duckdb execution.
  if (Config::ENABLE_FALLBACK_CHECK) {
    FallbackChecker fallback_checker(scheduled);
    fallback_checker.Check();
  }

  // Execution starts here.
  int initial_idx = 0;

  SIRIUS_LOG_DEBUG("Total meta pipelines {}", scheduled.size());

  // Phase B: run the entire pipeline graph concurrently across N GPUs.
  // Each worker thread is bound to one GPU via set_gpu_for_thread() and
  // runs the same plan against its GPU's partition (tables_per_gpu[g]).
  //
  //   - The scan's cache-LOAD section (GetDataDuckDB) is serialized via
  //     g_cache_load_mutex: the first thread loads partitions for all GPUs
  //     (Phase 1's per-GPU split runs inside GetDataDuckDB); subsequent
  //     threads see already_cached=true on every column and short-circuit.
  //   - The READ side (GetData) is reentrant: each thread reads only its
  //     own GPU's tables_per_gpu[g] entries.
  //   - Each per-GPU ResultCollector accumulates into its own
  //     ResultCollectorRuntimeState::result_collection (Phase B2).
  //     GetResult concatenates them at the end.
  //   - cuda_streams on TableScan are touched only inside the
  //     mutex-protected cache load, so they don't need to be per-thread.
  const int num_gpus_for_exec = static_cast<int>(gpuBufferManager->tables_per_gpu.size());

  // Toggle: SIRIUS_LEGACY_PARALLEL=0 forces the old sequential per-GPU loop
  // (Phase 2). Default (unset or =1) uses the parallel N-thread executor
  // (Phase B4). Useful for A/B timing and debugging.
  const char* parallel_env = std::getenv("SIRIUS_LEGACY_PARALLEL");
  const bool run_parallel  = !parallel_env || std::string(parallel_env) != "0";

  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> errors(num_gpus_for_exec);
  workers.reserve(num_gpus_for_exec);

  // Capture state needed by worker lambdas.
  auto& scheduled_ref = this->scheduled;
  auto& executor_ref  = this->executor;
  auto& context_ref   = this->context;
  int initial_idx_w   = initial_idx;
  auto* gpu_bm        = gpuBufferManager;

  auto worker_body = [&scheduled_ref, &executor_ref, &context_ref, initial_idx_w, gpu_bm](
                      int gpu_iter, std::exception_ptr* err_slot) {
    try {
        gpu_bm->set_gpu_for_thread(gpu_iter);
        SIRIUS_LOG_DEBUG("Per-GPU worker thread: GPU {}", gpu_iter);

        for (const auto& pipeline : scheduled_ref) {
    // TODO: This is temporary solution
    // if (pipeline->source->type == PhysicalOperatorType::HASH_JOIN || pipeline->source->type ==
    // PhysicalOperatorType::RESULT_COLLECTOR) { 	continue;
    // }

    vector<shared_ptr<GPUIntermediateRelation>> intermediate_relations;
    shared_ptr<GPUIntermediateRelation> final_relation;
    // vector<unique_ptr<OperatorState>> intermediate_states;
    intermediate_relations.reserve(pipeline->operators.size());
    // intermediate_states.reserve(pipeline->operators.size());

    // SIRIUS_LOG_DEBUG("Executing pipeline op size {}", pipeline->operators.size());
    for (idx_t i = 0; i < pipeline->operators.size(); i++) {
      auto& prev_operator    = i == 0 ? *(pipeline->source) : pipeline->operators[i - 1].get();
      auto& current_operator = pipeline->operators[i].get();

      // auto chunk = make_uniq<DataChunk>();
      // chunk->Initialize(Allocator::Get(context.client), prev_operator.GetTypes());
      shared_ptr<GPUIntermediateRelation> inter_rel =
        make_shared_ptr<GPUIntermediateRelation>(prev_operator.GetTypes().size());
      intermediate_relations.push_back(std::move(inter_rel));

      // auto op_state = current_operator.GetOperatorState(context);
      // intermediate_states.push_back(std::move(op_state));

      // if (current_operator.IsSink() && current_operator.sink_state->state ==
      // SinkFinalizeType::NO_OUTPUT_POSSIBLE) {
      // 	// one of the operators has already figured out no output is possible
      // 	// we can skip executing the pipeline
      // 	FinishProcessing();
      // }
    }
    // InitializeChunk(final_chunk);
    auto& last_op =
      pipeline->operators.empty() ? *pipeline->source : pipeline->operators.back().get();
    final_relation = make_shared_ptr<GPUIntermediateRelation>(last_op.GetTypes().size());

    // auto thread_context = ThreadContext(context);
    // auto exec_context = GPUExecutionContext(context, thread_context, pipeline.get());

    // pipeline->Reset();
    // auto prop = pipeline->executor.context.GetClientProperties();
    // SIRIUS_LOG_DEBUG("Properties: {}", prop.time_zone);
    auto& source_relation =
      pipeline->operators.empty() ? final_relation : intermediate_relations[0];
    // auto source_result = FetchFromSource(source_chunk);

    // StartOperator(*pipeline.source);
    // auto interrupt_state = InterruptState();
    // auto local_source_state = pipeline.source->GetLocalSourceState(exec_context,
    // *pipeline.source_state); OperatorSourceInput source_input = {*pipeline.source_state,
    // *local_source_state, interrupt_state}; pipeline->source->GetData(exec_context,
    // source_relation, source_input);
    auto source_type = pipeline->source.get()->type;
    SIRIUS_LOG_DEBUG("pipeline source type {}", PhysicalOperatorToString(source_type));
    if (source_type == PhysicalOperatorType::TABLE_SCAN) {
      // Cache load is serialized across worker threads. Phase 1's per-GPU
      // partitioning runs INSIDE GetDataDuckDB, so the first thread to grab
      // this mutex loads partitions for all GPUs. Subsequent threads see
      // already_cached=true on every column and short-circuit cheaply.
      std::lock_guard<std::mutex> lk(g_cache_load_mutex);
      Pipeline duckdb_pipeline(*executor_ref);
      ThreadContext thread_context(context_ref);
      ExecutionContext exec_context(context_ref, thread_context, &duckdb_pipeline);
      auto& table_scan = pipeline->source->Cast<GPUPhysicalTableScan>();
      table_scan.GetDataDuckDB(exec_context);
    }
    pipeline->source->GetData(*source_relation);
    // SIRIUS_LOG_DEBUG("source relation size {}", source_relation->columns.size());
    // for (auto col : source_relation->columns) {
    // 	SIRIUS_LOG_DEBUG("source relation column size {} column name {}", col->column_length,
    // col->name);
    // }
    // EndOperator(*pipeline.source, &result);

    // call source
    //  SIRIUS_LOG_DEBUG("{}", pipeline->source.get()->GetName());
    for (int current_idx = 1; current_idx <= pipeline->operators.size(); current_idx++) {
      auto op      = pipeline->operators[current_idx - 1];
      auto op_type = op.get().type;
      SIRIUS_LOG_DEBUG("pipeline operator type {}", PhysicalOperatorToString(op_type));
      // SIRIUS_LOG_DEBUG("{}", op.get().GetName());
      // call operator

      auto current_intermediate = current_idx;
      auto& current_relation    = current_intermediate >= intermediate_relations.size()
                                    ? final_relation
                                    : intermediate_relations[current_intermediate];
      // current_chunk.Reset();

      auto& prev_relation    = current_intermediate == initial_idx_w + 1
                                 ? source_relation
                                 : intermediate_relations[current_intermediate - 1];
      auto operator_idx      = current_idx - 1;
      auto& current_operator = pipeline->operators[operator_idx];

      // auto op_state = current_operator.GetOperatorState(context);
      // intermediate_states.push_back(std::move(op_state));

      // StartOperator(current_operator);
      // auto result = current_operator.get().Execute(exec_context, prev_relation, current_relation,
      // *current_operator.op_state,
      //                                        *intermediate_states[current_intermediate - 1]);

      auto result = current_operator.get().Execute(*prev_relation, *current_relation);
      // EndOperator(current_operator, &current_chunk);
    }
    if (pipeline->sink) {
      auto sink_type = pipeline->sink.get()->type;
      SIRIUS_LOG_DEBUG("pipeline sink type {}", PhysicalOperatorToString(sink_type));
      // SIRIUS_LOG_DEBUG("{}", pipeline->sink.get()->GetName());
      // call sink
      auto& sink_relation = final_relation;
      // SIRIUS_LOG_DEBUG("sink relation size {}", final_relation->columns.size());
      // int i = 0;
      // for (auto col : final_relation->columns) {
      // 	if (col == nullptr) SIRIUS_LOG_DEBUG("{}", i);
      // 	i++;
      // 	// SIRIUS_LOG_DEBUG("sink relation column size {}", col->column_length);
      // }
      // auto interrupt_state = InterruptState();
      // auto local_sink_state = pipeline->sink->GetLocalSinkState(exec_context);
      // OperatorSinkInput sink_input {*pipeline->sink->sink_state, *local_sink_state,
      // interrupt_state}; pipeline->sink->Sink(exec_context, *sink_relation, sink_input);
      pipeline->sink->Sink(*sink_relation);
    }
  }  // end for-pipeline
    } catch (...) {
      *err_slot = std::current_exception();
    }
  };  // end worker_body lambda

  if (run_parallel) {
    SIRIUS_LOG_DEBUG("Executor mode: parallel ({} workers)", num_gpus_for_exec);
    for (int gpu_iter = 0; gpu_iter < num_gpus_for_exec; ++gpu_iter) {
      workers.emplace_back(worker_body, gpu_iter, &errors[gpu_iter]);
    }
    for (auto& w : workers) { w.join(); }
  } else {
    SIRIUS_LOG_DEBUG("Executor mode: sequential ({} iters)", num_gpus_for_exec);
    for (int gpu_iter = 0; gpu_iter < num_gpus_for_exec; ++gpu_iter) {
      worker_body(gpu_iter, &errors[gpu_iter]);
    }
  }

  for (auto& e : errors) {
    if (e) std::rethrow_exception(e);
  }

  // Restore the main thread's GPU binding so any subsequent code on this
  // thread (e.g., the duckdb shell formatting the result) runs on GPU 0.
  gpuBufferManager->set_gpu_for_thread(0);
}

void GPUExecutor::InitializeInternal(GPUPhysicalOperator& plan)
{
  // auto &scheduler = TaskScheduler::GetScheduler(context);
  {
    // lock_guard<mutex> elock(executor_lock);
    gpu_physical_plan = &plan;

    // this->profiler = ClientData::Get(context).profiler;
    // profiler->Initialize(plan);
    // this->producer = scheduler.CreateProducer();

    // build and ready the pipelines
    GPUPipelineBuildState state;
    auto root_pipeline = make_shared_ptr<GPUMetaPipeline>(*this, state, nullptr);
    root_pipeline->Build(*gpu_physical_plan);
    root_pipeline->Ready();

    // ready recursive cte pipelines too
    // TODO: SUPPORT RECURSIVE CTE FOR GPU
    // for (auto &rec_cte_ref : recursive_ctes) {
    // 	auto &rec_cte = rec_cte_ref.get().Cast<PhysicalRecursiveCTE>();
    // 	// rec_cte.recursive_meta_pipeline->Ready();
    // }

    // set root pipelines, i.e., all pipelines that end in the final sink
    root_pipeline->GetPipelines(root_pipelines, false);
    root_pipeline_idx = 0;

    // collect all meta-pipelines from the root pipeline
    vector<shared_ptr<GPUMetaPipeline>> to_schedule;
    scheduled.clear();
    root_pipeline->GetMetaPipelines(to_schedule, true, true);

    // number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to
    // set it here
    total_pipelines = to_schedule.size();

    SIRIUS_LOG_DEBUG("Total meta pipelines {}", to_schedule.size());
    int schedule_count = 0;
    int meta           = 0;
    while (schedule_count < to_schedule.size()) {
      vector<shared_ptr<GPUMetaPipeline>> children;
      to_schedule[to_schedule.size() - 1 - meta]->GetMetaPipelines(children, false, true);
      auto base_pipeline   = to_schedule[to_schedule.size() - 1 - meta]->GetBasePipeline();
      bool should_schedule = true;

      // already scheduled
      if (find(scheduled.begin(), scheduled.end(), base_pipeline) != scheduled.end()) {
        should_schedule = false;
      } else {
        // check if all children are scheduled
        for (auto& child : children) {
          if (find(scheduled.begin(), scheduled.end(), child->GetBasePipeline()) ==
              scheduled.end()) {
            should_schedule = false;
            break;
          }
        }
        // check if all dependencies are scheduled
        for (int dep = 0; dep < base_pipeline->dependencies.size(); dep++) {
          if (find(scheduled.begin(), scheduled.end(), base_pipeline->dependencies[dep]) ==
              scheduled.end()) {
            should_schedule = false;
            break;
          }
        }
      }
      if (should_schedule) {
        vector<shared_ptr<GPUPipeline>> pipeline_inside;
        to_schedule[to_schedule.size() - 1 - meta]->GetPipelines(pipeline_inside, false);
        for (int pipeline_idx = 0; pipeline_idx < pipeline_inside.size(); pipeline_idx++) {
          auto& pipeline = pipeline_inside[pipeline_idx];
          if (pipeline_inside[pipeline_idx]->source->type == PhysicalOperatorType::HASH_JOIN) {
            auto& temp = pipeline_inside[pipeline_idx]->source.get()->Cast<GPUPhysicalHashJoin>();
            if (temp.join_type == JoinType::RIGHT || temp.join_type == JoinType::RIGHT_SEMI ||
                temp.join_type == JoinType::RIGHT_ANTI) {
              if (!Config::MODIFIED_PIPELINE) scheduled.push_back(pipeline);
            }
            continue;
          } else {
            scheduled.push_back(pipeline);
          }
        }
        schedule_count++;
      }
      meta = (meta + 1) % to_schedule.size();
    }

    // collect all pipelines from the root pipelines (recursively) for the progress bar and verify
    // them
    root_pipeline->GetPipelines(pipelines, true);
    SIRIUS_LOG_DEBUG("total_pipelines = {}", pipelines.size());
  }
}

void GPUExecutor::CancelTasks()
{
  pipelines.clear();
  root_pipelines.clear();
}

shared_ptr<GPUPipeline> GPUExecutor::CreateChildPipeline(GPUPipeline& current,
                                                         GPUPhysicalOperator& op)
{
  D_ASSERT(!current.operators.empty());
  D_ASSERT(op.IsSource());
  // found another operator that is a source, schedule a child pipeline
  // 'op' is the source, and the sink is the same
  auto child_pipeline    = make_shared_ptr<GPUPipeline>(*this);
  child_pipeline->sink   = current.sink;
  child_pipeline->source = &op;

  // the child pipeline has the same operators up until 'op'
  for (auto current_op : current.operators) {
    if (&current_op.get() == &op) { break; }
    child_pipeline->operators.push_back(current_op);
  }

  return child_pipeline;
}

bool GPUExecutor::HasResultCollector()
{
  return gpu_physical_plan->type == PhysicalOperatorType::RESULT_COLLECTOR;
}

unique_ptr<QueryResult> GPUExecutor::GetResult()
{
  D_ASSERT(HasResultCollector());
  if (!gpu_physical_plan) throw InvalidInputException("gpu_physical_plan is NULL");
  if (gpu_physical_plan.get() == NULL) throw InvalidInputException("gpu_physical_plan is NULL");
  auto& result_collector = gpu_physical_plan.get()->Cast<GPUPhysicalMaterializedCollector>();
  D_ASSERT(result_collector.sink_state);
  result_collector.sink_state = result_collector.GetGlobalSinkState(context);
  unique_ptr<QueryResult> res = result_collector.GetResult(*(result_collector.sink_state));
  return res;
}

}  // namespace duckdb
