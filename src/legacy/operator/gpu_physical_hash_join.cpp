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

#include "operator/gpu_physical_hash_join.hpp"

#include <array>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <vector>

#include "duckdb/common/enums/physical_operator_type.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_pipeline.hpp"
#include "log/logging.hpp"
#include "operator/gpu_materialize.hpp"
#include "operator/magi_join.hpp"
#include "operator/gpu_physical_table_scan.hpp"

namespace duckdb {

// Allgather one dense fixed-width probe column across the per-GPU workers for
// the broadcast-probe join strategy: every worker publishes its partition's
// device pointer + row count, waits, then copies every partition (UVA
// device-to-device) into a full-length local buffer. Both workers call this
// the same number of times in the same order (identical plans), so the
// static rendezvous slots + barrier stay in lockstep.

static shared_ptr<GPUColumn> AllgatherProbeColumn(const shared_ptr<GPUColumn>& col,
                                                  GPUBufferManager* gpuBufferManager)
{
  const int num_gpus = static_cast<int>(gpuBufferManager->tables_per_gpu.size());
  if (num_gpus <= 1) { return col; }
  // Without peer access a UVA device-to-device cudaMemcpy is staged through
  // host memory (~25 GB/s); over NVLink it's ~10x that. magi enables its own
  // peer access, but a broadcast-only query never initializes magi — so
  // enable it lazily here, once per (current device, peer) pair.
  {
    static std::array<std::once_flag, 64> peer_once;
    const int g0 = sirius_current_gpu;
    for (int i = 0; i < num_gpus; ++i) {
      if (i == g0) { continue; }
      std::call_once(peer_once[g0 * 8 + i], [&] {
        cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
          SIRIUS_LOG_DEBUG("broadcast-probe: peer access {} -> {} unavailable ({}), "
                           "falling back to staged copies", g0, i, (int)e);
        }
        (void)cudaGetLastError();  // clear a sticky already-enabled error
      });
    }
  }
  // A validity mask may still be attached, but the broadcast eligibility
  // exchange only picks this path after verifying every partition's mask is
  // ALL-VALID (MaskAllValid) — so the mask carries no information and the
  // gathered output is emitted dense (mask dropped).
  const bool is_varchar = col->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR;

  struct Slot {
    const uint8_t*  ptr;
    const uint64_t* offs;   // VARCHAR only: offsets array (rows+1, 0-based)
    uint64_t        rows;
    uint64_t        bytes;  // VARCHAR only: chars blob size
  };
  static std::array<Slot, 8>            slots;
  static std::unique_ptr<std::barrier<>> bar;
  static std::once_flag                  bar_once;
  std::call_once(bar_once, [&] { bar = std::make_unique<std::barrier<>>(num_gpus); });

  const int g = sirius_current_gpu;
  // Per-GPU non-blocking stream pool: the gather copies + offset-rebase kernels
  // are ENQUEUED here and overlap across partitions AND across successive
  // column calls (the serial-cudaMemcpy version ran the ~300MB Q10 build
  // allgather at ~57GB/s; NVLink does ~4x that). Non-blocking → the legacy
  // null stream neither orders after nor blocks on them; the call-site
  // flushes with one cudaDeviceSynchronize after its last column (sources are
  // query-lifetime pool allocations, so in-flight reads of a peer's published
  // buffer stay valid after the slot barrier).
  constexpr int AG_STREAMS = 4;
  static std::array<std::array<cudaStream_t, AG_STREAMS>, 8> ag_streams;
  static std::array<std::once_flag, 8>                       ag_streams_once;
  std::call_once(ag_streams_once[g], [&] {
    for (int s = 0; s < AG_STREAMS; ++s) {
      cudaStreamCreateWithFlags(&ag_streams[g][s], cudaStreamNonBlocking);
    }
  });
  int next_stream = 0;
  auto pick = [&]() { return ag_streams[g][next_stream++ % AG_STREAMS]; };

  // Our own materialization kernels (null stream) must have landed before
  // peers read the buffer. Null-stream sync — NOT device sync — so our own
  // in-flight allgather copies from the previous column don't serialize the
  // pipeline.
  cudaStreamSynchronize(nullptr);
  slots[g] = {col->data_wrapper.data, col->data_wrapper.offset, col->column_length,
              col->data_wrapper.num_bytes};
  bar->arrive_and_wait();  // every partition published

  uint64_t total = 0;
  for (int i = 0; i < num_gpus; ++i) { total += slots[i].rows; }

  shared_ptr<GPUColumn> out;
  if (is_varchar) {
    // Concatenate chars blobs; copy each partition's offsets segment and
    // rebase it by the cumulative char count (partition offsets are 0-based
    // after materialization).
    uint64_t total_bytes = 0;
    for (int i = 0; i < num_gpus; ++i) { total_bytes += slots[i].bytes; }
    uint8_t* chars = gpuBufferManager->customCudaMalloc<uint8_t>(
      total_bytes > 0 ? total_bytes : 1, g, 0);
    uint64_t* offs = gpuBufferManager->customCudaMalloc<uint64_t>(total + 1, g, 0);
    uint64_t row_off = 0, char_off = 0;
    for (int i = 0; i < num_gpus; ++i) {
      if (slots[i].rows > 0) {
        if (slots[i].bytes > 0) {
          cudaMemcpyAsync(chars + char_off, slots[i].ptr, slots[i].bytes,
                          cudaMemcpyDeviceToDevice, pick());
        }
        cudaStream_t so = pick();
        cudaMemcpyAsync(offs + row_off, slots[i].offs, slots[i].rows * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, so);
        if (char_off > 0) { addToEachAsync<uint64_t>(offs + row_off, char_off, slots[i].rows, so); }
        row_off += slots[i].rows;
        char_off += slots[i].bytes;
      }
    }
    // Tail offset from pinned scratch (async H2D from a stack variable is not
    // safe once this frame returns).
    uint64_t* h_tail = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    *h_tail = char_off;
    cudaMemcpyAsync(offs + total, h_tail, sizeof(uint64_t), cudaMemcpyHostToDevice, pick());
    out = make_shared_ptr<GPUColumn>(
      total, col->data_wrapper.type, chars, offs, total_bytes, true, nullptr);
  } else {
    const size_t esz = col->data_wrapper.getColumnTypeSize();
    uint8_t*     dst = gpuBufferManager->customCudaMalloc<uint8_t>(total * esz, g, 0);
    uint64_t     off = 0;
    for (int i = 0; i < num_gpus; ++i) {
      if (slots[i].rows > 0) {
        // UVA resolves the source device; goes over NVLink when P2P is enabled.
        cudaMemcpyAsync(dst + off * esz, slots[i].ptr, slots[i].rows * esz,
                        cudaMemcpyDeviceToDevice, pick());
      }
      off += slots[i].rows;
    }
    out = make_shared_ptr<GPUColumn>(total, col->data_wrapper.type, dst, nullptr);
  }
  bar->arrive_and_wait();  // slots may be republished; in-flight reads stay valid
  return out;
}

// TPC-H-style data routinely carries an ALL-VALID validity mask (no actual
// nulls; e.g. masks attached at cache load for nullable-typed storage).
// Broadcasting drops the mask, which is exact iff no bit is unset — verified
// host-side (the mask for a ≤64M-row probe is ≤8MB, a one-off sync D2H).
static bool MaskAllValid(const shared_ptr<GPUColumn>& col)
{
  if (col->data_wrapper.validity_mask == nullptr) { return true; }
  const uint64_t rows = col->column_length;
  if (rows == 0) { return true; }
  const size_t words = (rows + 31) / 32;
  std::vector<uint32_t> host(words);
  if (cudaMemcpy(host.data(), col->data_wrapper.validity_mask, words * sizeof(uint32_t),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return false;
  }
  for (size_t w = 0; w + 1 < words; ++w) {
    if (host[w] != 0xffffffffu) { return false; }
  }
  const uint32_t tail_bits = rows % 32 == 0 ? 32u : static_cast<uint32_t>(rows % 32);
  const uint32_t tail_mask = tail_bits == 32 ? 0xffffffffu : ((1u << tail_bits) - 1u);
  return (host[words - 1] & tail_mask) == tail_mask;
}

// Runtime shuffle-vs-broadcast decision for candidate joins. DuckDB's plan
// estimates are wildly off for filtered probe sides (Q14: 60M estimated vs
// 3.6M actual date-filtered lineitem), so the choice is made at Execute time
// from ACTUAL row counts. Every worker publishes its partition's sizes plus a
// local eligibility bit (nullable/VARCHAR columns can differ per partition),
// then all workers compute the same totals and the same verdict — the branch
// stays worker-consistent, which the collective paths require (divergence
// would strand one worker on a barrier).
struct BroadcastDecision {
  uint64_t probe    = 0;
  uint64_t build    = 0;
  bool     eligible = true;
};

enum class BcastStrategy : uint8_t { SHUFFLE = 0, PROBE = 1, BUILD = 2 };

// Size-based strategy pick, computed from EXCHANGED totals plus schema-derived
// per-row wire widths (identical on every worker), so every worker picks the
// same strategy. Broadcasting a side moves rows*(N-1)*width bytes and
// duplicates that side's join work; the shuffle moves ~(P+B)*(N-1)/N of both
// sides plus two magi session fixed costs, but is the only option when both
// sides are big.
static BcastStrategy PickBcastStrategy(uint64_t probe_total,
                                       uint64_t build_total,
                                       uint64_t probe_row_bytes,
                                       uint64_t build_row_bytes)
{
  // PROJECT RULE: broadcast is ONLY for a genuinely SMALL side. It moves
  // rows*(N-1)*width bytes AND duplicates that side's join work on every GPU,
  // so a join whose both sides are tens of millions of rows must shuffle.
  // The caps bound the broadcast side in absolute terms; the bytes/ratio
  // rules below additionally keep it small RELATIVE to the other side.
  // TPC-H SF50 reference: Q14 probe 3.7M, Q3 builds 1.5M/7.3M, Q12 build
  // 1.6M — all far under the cap.
  static const uint64_t probe_max = [] {
    const char* e = std::getenv("MAGI_BCAST_PROBE_MAX");
    return e ? std::strtoull(e, nullptr, 10) : uint64_t(16'000'000);
  }();
  static const uint64_t build_max = [] {
    const char* e = std::getenv("MAGI_BCAST_BUILD_MAX");
    return e ? std::strtoull(e, nullptr, 10) : uint64_t(16'000'000);
  }();
  const bool probe_fits = probe_total > 0 && probe_total <= probe_max && build_total > 0;
  const bool build_fits = build_total > 0 && build_total <= build_max && probe_total > 0;
  if (probe_fits && build_fits) {
    return probe_total * probe_row_bytes <= build_total * build_row_bytes
             ? BcastStrategy::PROBE
             : BcastStrategy::BUILD;
  }
  // Build too big to broadcast: probe broadcast duplicates the probe-side join
  // work on every GPU, so only take it when the probe is not much bigger than
  // the build (same guard as the original probe-only rule).
  if (probe_fits && probe_total * 2 <= build_total * 3) { return BcastStrategy::PROBE; }
  if (build_fits) { return BcastStrategy::BUILD; }
  return BcastStrategy::SHUFFLE;
}
static BroadcastDecision ExchangeBroadcastDecision(GPUBufferManager* gpuBufferManager,
                                                   uint64_t probe_rows,
                                                   uint64_t build_rows,
                                                   bool local_eligible)
{
  const int num_gpus = static_cast<int>(gpuBufferManager->tables_per_gpu.size());
  BroadcastDecision d;
  struct Slot {
    uint64_t probe;
    uint64_t build;
    bool     ok;
  };
  static std::array<Slot, 8>             slots;
  static std::unique_ptr<std::barrier<>> bar;
  static std::once_flag                  bar_once;
  if (num_gpus > 1) {
    std::call_once(bar_once, [&] { bar = std::make_unique<std::barrier<>>(num_gpus); });
    slots[sirius_current_gpu] = {probe_rows, build_rows, local_eligible};
    bar->arrive_and_wait();  // every partition published
    for (int i = 0; i < num_gpus; ++i) {
      d.probe += slots[i].probe;
      d.build += slots[i].build;
      d.eligible = d.eligible && slots[i].ok;
    }
    bar->arrive_and_wait();  // all reads done before the slots are reused
  } else {
    d = {probe_rows, build_rows, local_eligible};
  }
  return d;
}


template <typename T>
void ResolveTypeProbeExpression(vector<shared_ptr<GPUColumn>>& probe_keys,
                                uint64_t*& count,
                                uint64_t*& row_ids_left,
                                uint64_t*& row_ids_right,
                                unsigned long long* ht,
                                uint64_t ht_len,
                                const vector<JoinCondition>& conditions,
                                JoinType join_type,
                                bool unique_build_keys,
                                GPUBufferManager* gpuBufferManager)
{
  int num_keys         = conditions.size();
  uint8_t** probe_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_keys);

  for (int key = 0; key < num_keys; key++) {
    probe_data[key] = probe_keys[key]->data_wrapper.data;
  }
  size_t size = probe_keys[0]->column_length;

  int* condition_mode = gpuBufferManager->customCudaHostAlloc<int>(num_keys);
  for (int key = 0; key < num_keys; key++) {
    if (conditions[key].comparison == ExpressionType::COMPARE_EQUAL ||
        conditions[key].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      condition_mode[key] = 0;
    } else if (conditions[key].comparison == ExpressionType::COMPARE_NOTEQUAL ||
               conditions[key].comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
      condition_mode[key] = 1;
    } else {
      throw NotImplementedException("Unsupported comparison type");
    }
  }

  // TODO: Need to handle special case for unique keys for better performance
  if (join_type == JoinType::INNER) {
    // if (unique_build_keys) {
    // 	probeHashTableSingleMatch<T>(probe_data, ht, ht_len, row_ids_left, row_ids_right, count,
    // size, condition_mode, num_keys, 0); } else { 	probeHashTable<T>(probe_data, ht, ht_len,
    // row_ids_left, row_ids_right, count, size, condition_mode, num_keys, false);
    // }
    throw NotImplementedException("Unsupported join type: INNER");
  } else if (join_type == JoinType::SEMI) {
    probeHashTableSingleMatch<T>(probe_data,
                                 ht,
                                 ht_len,
                                 row_ids_left,
                                 row_ids_right,
                                 count,
                                 size,
                                 condition_mode,
                                 num_keys,
                                 1);
  } else if (join_type == JoinType::ANTI) {
    probeHashTableSingleMatch<T>(probe_data,
                                 ht,
                                 ht_len,
                                 row_ids_left,
                                 row_ids_right,
                                 count,
                                 size,
                                 condition_mode,
                                 num_keys,
                                 2);
  } else if (join_type == JoinType::RIGHT) {
    if (unique_build_keys) {
      probeHashTableSingleMatch<T>(probe_data,
                                   ht,
                                   ht_len,
                                   row_ids_left,
                                   row_ids_right,
                                   count,
                                   size,
                                   condition_mode,
                                   num_keys,
                                   3);
    } else {
      probeHashTable<T>(probe_data,
                        ht,
                        ht_len,
                        row_ids_left,
                        row_ids_right,
                        count,
                        size,
                        condition_mode,
                        num_keys,
                        true);
    }
  } else if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
    if (unique_build_keys) {
      probeHashTableRightSemiAntiSingleMatch<T>(
        probe_data, ht, ht_len, size, condition_mode, num_keys);
    } else {
      probeHashTableRightSemiAnti<T>(probe_data, ht, ht_len, size, condition_mode, num_keys);
    }
  } else {
    throw NotImplementedException("Unsupported join type");
  }
}

void HandleProbeExpression(vector<shared_ptr<GPUColumn>>& probe_keys,
                           uint64_t*& count,
                           uint64_t*& row_ids_left,
                           uint64_t*& row_ids_right,
                           unsigned long long* ht,
                           uint64_t ht_len,
                           const vector<JoinCondition>& conditions,
                           JoinType join_type,
                           bool unique_build_keys,
                           GPUBufferManager* gpuBufferManager)
{
  switch (probe_keys[0]->data_wrapper.type.id()) {
    case GPUColumnTypeId::INT32:
      ResolveTypeProbeExpression<int32_t>(probe_keys,
                                          count,
                                          row_ids_left,
                                          row_ids_right,
                                          ht,
                                          ht_len,
                                          conditions,
                                          join_type,
                                          unique_build_keys,
                                          gpuBufferManager);
      break;
    case GPUColumnTypeId::INT64:
    case GPUColumnTypeId::FLOAT64:
      ResolveTypeProbeExpression<int64_t>(probe_keys,
                                          count,
                                          row_ids_left,
                                          row_ids_right,
                                          ht,
                                          ht_len,
                                          conditions,
                                          join_type,
                                          unique_build_keys,
                                          gpuBufferManager);
      break;
    default:
      throw NotImplementedException("Unsupported sirius column type in `HandleProbeExpression`: %d",
                                    static_cast<int>(probe_keys[0]->data_wrapper.type.id()));
  }
}

template <typename T>
void ResolveTypeMarkExpression(vector<shared_ptr<GPUColumn>>& probe_keys,
                               uint8_t*& output,
                               unsigned long long* ht,
                               uint64_t ht_len,
                               const vector<JoinCondition>& conditions,
                               GPUBufferManager* gpuBufferManager)
{
  int num_keys         = conditions.size();
  uint8_t** probe_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_keys);

  for (int key = 0; key < num_keys; key++) {
    probe_data[key] = probe_keys[key]->data_wrapper.data;
  }
  size_t size = probe_keys[0]->column_length;

  int* condition_mode = gpuBufferManager->customCudaHostAlloc<int>(num_keys);
  for (int key = 0; key < num_keys; key++) {
    if (conditions[key].comparison == ExpressionType::COMPARE_EQUAL ||
        conditions[key].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      condition_mode[key] = 0;
    } else if (conditions[key].comparison == ExpressionType::COMPARE_NOTEQUAL ||
               conditions[key].comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
      // TODO: Currently only support TPC-H Q21: l2.l_orderkey = l1.l_orderkey and l2.l_suppkey !=
      // l1.l_suppkey
      if (key != 1 || num_keys != 2) throw NotImplementedException("Unsupported comparison type");
      condition_mode[key] = 1;
    } else {
      throw NotImplementedException("Unsupported comparison type");
    }
  }

  probeHashTableMark<T>(probe_data, ht, ht_len, output, size, condition_mode, num_keys);
}

void HandleMarkExpression(vector<shared_ptr<GPUColumn>>& probe_keys,
                          uint8_t*& output,
                          unsigned long long* ht,
                          uint64_t ht_len,
                          const vector<JoinCondition>& conditions,
                          GPUBufferManager* gpuBufferManager)
{
  switch (probe_keys[0]->data_wrapper.type.id()) {
    case GPUColumnTypeId::INT32:
      ResolveTypeMarkExpression<int32_t>(
        probe_keys, output, ht, ht_len, conditions, gpuBufferManager);
      break;
    case GPUColumnTypeId::INT64:
    case GPUColumnTypeId::FLOAT64:
      ResolveTypeMarkExpression<int64_t>(
        probe_keys, output, ht, ht_len, conditions, gpuBufferManager);
      break;
    default:
      throw NotImplementedException("Unsupported sirius column type in `HandleMarkExpression`: %d",
                                    static_cast<int>(probe_keys[0]->data_wrapper.type.id()));
  }
}

template <typename T>
void ResolveTypeBuildExpression(vector<shared_ptr<GPUColumn>>& build_keys,
                                unsigned long long* ht,
                                uint64_t ht_len,
                                const vector<JoinCondition>& conditions,
                                JoinType join_type,
                                GPUBufferManager* gpuBufferManager)
{
  int num_keys         = conditions.size();
  uint8_t** build_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_keys);

  for (int key = 0; key < num_keys; key++) {
    build_data[key] = build_keys[key]->data_wrapper.data;
  }
  size_t size = build_keys[0]->column_length;

  int* condition_mode = gpuBufferManager->customCudaHostAlloc<int>(num_keys);
  for (int key = 0; key < num_keys; key++) {
    if (conditions[key].comparison == ExpressionType::COMPARE_EQUAL ||
        conditions[key].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      condition_mode[key] = 0;
    } else if (conditions[key].comparison == ExpressionType::COMPARE_NOTEQUAL ||
               conditions[key].comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
      // TODO: Currently only support TPC-H Q21: l2.l_orderkey = l1.l_orderkey and l2.l_suppkey !=
      // l1.l_suppkey
      if (key != 1 || num_keys != 2) throw NotImplementedException("Unsupported comparison type");
      condition_mode[key] = 1;
    } else {
      throw NotImplementedException("Unsupported comparison type");
    }
  }

  if (join_type == JoinType::INNER || join_type == JoinType::SEMI || join_type == JoinType::MARK ||
      join_type == JoinType::ANTI) {
    buildHashTable<T>(build_data, ht, ht_len, size, condition_mode, num_keys, 0);
  } else if (join_type == JoinType::RIGHT || join_type == JoinType::RIGHT_SEMI ||
             join_type == JoinType::RIGHT_ANTI) {
    buildHashTable<T>(build_data, ht, ht_len, size, condition_mode, num_keys, 1);
  } else {
    throw NotImplementedException("Unsupported join type");
  }
}

void HandleBuildExpression(vector<shared_ptr<GPUColumn>>& build_keys,
                           unsigned long long* ht,
                           uint64_t ht_len,
                           const vector<JoinCondition>& conditions,
                           JoinType join_type,
                           GPUBufferManager* gpuBufferManager)
{
  switch (build_keys[0]->data_wrapper.type.id()) {
    case GPUColumnTypeId::INT32:
      ResolveTypeBuildExpression<int32_t>(
        build_keys, ht, ht_len, conditions, join_type, gpuBufferManager);
      break;
    case GPUColumnTypeId::INT64:
    case GPUColumnTypeId::FLOAT64:
      ResolveTypeBuildExpression<int64_t>(
        build_keys, ht, ht_len, conditions, join_type, gpuBufferManager);
      break;
    default:
      throw NotImplementedException("Unsupported sirius column type in `HandleBuildExpression`: %d",
                                    static_cast<int>(build_keys[0]->data_wrapper.type.id()));
  }
}

void HandleScanHTExpression(unsigned long long* ht,
                            uint64_t ht_len,
                            uint64_t*& row_ids,
                            uint64_t*& count,
                            JoinType join_type,
                            const vector<JoinCondition>& conditions)
{
  int num_keys = conditions.size();
  if (join_type == JoinType::RIGHT_SEMI) {
    scanHashTableRight(ht, ht_len, row_ids, count, 0, num_keys);
  } else if (join_type == JoinType::RIGHT || join_type == JoinType::RIGHT_ANTI) {
    scanHashTableRight(ht, ht_len, row_ids, count, 1, num_keys);
  } else {
    throw NotImplementedException("Unsupported join type");
  }
}

void ReorderJoinConditions(vector<JoinCondition>& conditions)
{
  // we reorder conditions so the ones with COMPARE_EQUAL occur first
  // check if this is already the case
  bool is_ordered     = true;
  bool seen_non_equal = false;
  for (auto& cond : conditions) {
    if (cond.comparison == ExpressionType::COMPARE_EQUAL ||
        cond.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      if (seen_non_equal) {
        is_ordered = false;
        break;
      }
    } else {
      seen_non_equal = true;
    }
  }
  if (is_ordered) {
    // no need to re-order
    return;
  }
  // gather lists of equal/other conditions
  vector<JoinCondition> equal_conditions;
  vector<JoinCondition> other_conditions;
  for (auto& cond : conditions) {
    if (cond.comparison == ExpressionType::COMPARE_EQUAL ||
        cond.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      equal_conditions.push_back(std::move(cond));
    } else {
      other_conditions.push_back(std::move(cond));
    }
  }
  conditions.clear();
  // reconstruct the sorted conditions
  for (auto& cond : equal_conditions) {
    conditions.push_back(std::move(cond));
  }
  for (auto& cond : other_conditions) {
    conditions.push_back(std::move(cond));
  }
}

GPUPhysicalHashJoin::GPUPhysicalHashJoin(LogicalOperator& op,
                                         unique_ptr<GPUPhysicalOperator> left,
                                         unique_ptr<GPUPhysicalOperator> right,
                                         vector<JoinCondition> cond,
                                         JoinType join_type,
                                         const vector<idx_t>& left_projection_map,
                                         const vector<idx_t>& right_projection_map,
                                         vector<LogicalType> delim_types,
                                         idx_t estimated_cardinality,
                                         unique_ptr<JoinFilterPushdownInfo> pushdown_info_p)
  : GPUPhysicalOperator(PhysicalOperatorType::HASH_JOIN, op.types, estimated_cardinality),
    join_type(join_type),
    conditions(std::move(cond)),
    delim_types(std::move(delim_types))
{
  ReorderJoinConditions(conditions);

  filter_pushdown = std::move(pushdown_info_p);

  children.push_back(std::move(left));
  children.push_back(std::move(right));

  // Collect condition types, and which conditions are just references (so we won't duplicate them
  // in the payload)
  unordered_map<idx_t, idx_t> build_columns_in_conditions;
  for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
    auto& condition = conditions[cond_idx];
    condition_types.push_back(condition.left->return_type);
    if (condition.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
      build_columns_in_conditions.emplace(condition.right->Cast<BoundReferenceExpression>().index,
                                          cond_idx);
    }
  }

  auto& lhs_input_types = children[0]->GetTypes();

  // Create a projection map for the LHS (if it was empty), for convenience
  lhs_output_columns.col_idxs = left_projection_map;
  if (lhs_output_columns.col_idxs.empty()) {
    lhs_output_columns.col_idxs.reserve(lhs_input_types.size());
    for (idx_t i = 0; i < lhs_input_types.size(); i++) {
      lhs_output_columns.col_idxs.emplace_back(i);
    }
  }

  for (auto& lhs_col : lhs_output_columns.col_idxs) {
    auto& lhs_col_type = lhs_input_types[lhs_col];
    lhs_output_columns.col_types.push_back(lhs_col_type);
  }

  // For ANTI, SEMI and MARK join, we only need to store the keys, so for these the payload/RHS
  // types are empty
  if (join_type == JoinType::ANTI || join_type == JoinType::SEMI || join_type == JoinType::MARK) {
    materialized_build_key =
      make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
    hash_table_result =
      make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
    return;
  }

  auto& rhs_input_types = children[1]->GetTypes();

  // Create a projection map for the RHS (if it was empty), for convenience
  auto right_projection_map_copy = right_projection_map;
  if (right_projection_map_copy.empty()) {
    right_projection_map_copy.reserve(rhs_input_types.size());
    for (idx_t i = 0; i < rhs_input_types.size(); i++) {
      right_projection_map_copy.emplace_back(i);
    }
  }

  // Now fill payload expressions/types and RHS columns/types
  for (auto& rhs_col : right_projection_map_copy) {
    auto& rhs_col_type = rhs_input_types[rhs_col];

    auto it = build_columns_in_conditions.find(rhs_col);
    if (it == build_columns_in_conditions.end()) {
      // This rhs column is not a join key
      payload_columns.col_idxs.push_back(rhs_col);
      payload_columns.col_types.push_back(rhs_col_type);
      rhs_output_columns.col_idxs.push_back(condition_types.size() +
                                            payload_columns.col_types.size() - 1);
    } else {
      // This rhs column is a join key
      rhs_output_columns.col_idxs.push_back(it->second);
    }
    rhs_output_columns.col_types.push_back(rhs_col_type);
  }

  hash_table_result = make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size() +
                                                               payload_columns.col_idxs.size());
  materialized_build_key =
    make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
};

SourceResultType GPUPhysicalHashJoin::GetData(GPUIntermediateRelation& output_relation) const
{
  auto start = std::chrono::high_resolution_clock::now();

  idx_t left_column_count = output_relation.columns.size() - rhs_output_columns.col_idxs.size();
  if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
    SIRIUS_LOG_DEBUG("Right semi or right anti join so there will be no columns from LHS");
    left_column_count = 0;
  } else if (join_type == JoinType::RIGHT || join_type == JoinType::OUTER) {
    for (idx_t col = 0; col < left_column_count; col++) {
      // pretend this to be NUll column from the left table (it should be NULL for the RIGHT join)
      SIRIUS_LOG_DEBUG("Right join so columns from LHS will be null");
      output_relation.columns[col] =
        make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), nullptr, nullptr);
    }
  } else {
    throw InvalidInputException("Get data not supported for this join type");
  }

  auto& rstate = runtime_state<HashJoinRuntimeState>(sirius_current_gpu);
  if (join_type == JoinType::OUTER && rstate.outer_join_handled_in_execute) {
    // cudf::full_join already emitted all rows (matched + unmatched on both sides)
    // in Execute(), so GetData has nothing more to produce.
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_DEBUG("Full outer join GetData (no-op, handled in Execute): {} us",
                     duration.count());
    return SourceResultType::FINISHED;
  }

  uint64_t* row_ids                  = nullptr;
  uint64_t* count                    = nullptr;
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  const int num_gpus = static_cast<int>(gpuBufferManager->tables_per_gpu.size());
  if (rstate.semi_bcast && num_gpus > 1) {
    // Multi-GPU RIGHT_SEMI/RIGHT_ANTI: every GPU probed its LOCAL probe
    // partition against the identical (gathered) build table, so the match
    // marks are partial per GPU. Convert marks → flags indexed by build ROW
    // ID (identical space across GPUs; slot layout is not), OR-merge across
    // GPUs, then emit a disjoint row-id range per GPU.
    const uint64_t total = rstate.semi_build_total;
    uint8_t* flags =
      gpuBufferManager->customCudaMalloc<uint8_t>(total > 0 ? total : 1, sirius_current_gpu, 0);
    cudaMemset(flags, 0, total > 0 ? total : 1);
    if (rstate.ht_len > 0) {
      scanHTMatchedToFlags(rstate.gpu_hash_table, rstate.ht_len,
                           static_cast<int>(conditions.size()), flags);
      gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(rstate.gpu_hash_table),
                                       sirius_current_gpu);
      rstate.gpu_hash_table = nullptr;
    }
    static std::array<uint8_t*, 8>         flag_slots;
    static std::unique_ptr<std::barrier<>> fbar;
    static std::once_flag                  fonce;
    std::call_once(fonce, [&] { fbar = std::make_unique<std::barrier<>>(num_gpus); });
    flag_slots[sirius_current_gpu] = flags;
    fbar->arrive_and_wait();  // every GPU's flags fully written & published
    if (total > 0) {
      uint8_t* scratch =
        gpuBufferManager->customCudaMalloc<uint8_t>(total, sirius_current_gpu, 0);
      for (int g = 0; g < num_gpus; ++g) {
        if (g == sirius_current_gpu) { continue; }
        // UVA resolves the peer device; NVLink when P2P is enabled.
        cudaMemcpy(scratch, flag_slots[g], total, cudaMemcpyDeviceToDevice);
        orFlagsInPlace(flags, scratch, total);
      }
    }
    fbar->arrive_and_wait();  // all reads done before the slots are reused
    const uint64_t lo = total * static_cast<uint64_t>(sirius_current_gpu) / num_gpus;
    const uint64_t hi = total * (static_cast<uint64_t>(sirius_current_gpu) + 1) / num_gpus;
    selectFlaggedRange(flags, lo, hi, join_type == JoinType::RIGHT_SEMI, row_ids, count);
  } else {
    HandleScanHTExpression(
      rstate.gpu_hash_table, rstate.ht_len, row_ids, count, join_type, conditions);
  }

  for (idx_t i = 0; i < rhs_output_columns.col_idxs.size(); i++) {
    const auto rhs_col = rhs_output_columns.col_idxs[i];
    SIRIUS_LOG_DEBUG("Writing hash_table column {} to column {}", rhs_col, i);
  }
  // TODO: Check if we need to maintain unique for the RHS columns
  if (rstate.unique_probe_keys) {
    HandleMaterializeRowIDsRHS(*rstate.hash_table_result,
                               output_relation,
                               rhs_output_columns.col_idxs,
                               left_column_count,
                               count[0],
                               row_ids,
                               gpuBufferManager,
                               true);
  } else {
    HandleMaterializeRowIDsRHS(*rstate.hash_table_result,
                               output_relation,
                               rhs_output_columns.col_idxs,
                               left_column_count,
                               count[0],
                               row_ids,
                               gpuBufferManager,
                               false);
  }
  // for (idx_t i = 0; i < hash_table_result->columns.size(); i++) {
  // 	if (find(rhs_output_columns.col_idxs.begin(), rhs_output_columns.col_idxs.end(), i) ==
  // rhs_output_columns.col_idxs.end()) {
  // 		gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(hash_table_result->columns[i]->data_wrapper.data),
  // 0); 		if (hash_table_result->columns[i]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
  // 			gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(hash_table_result->columns[i]->data_wrapper.offset),
  // 0);
  // 		}
  // 	}
  // }

  // check if all output columns has the same size
  //  for (idx_t i = 1; i < output_relation.columns.size(); i++) {
  //  	if (output_relation.columns[i]->column_length != output_relation.columns[0]->column_length)
  //  { 		printf("Column %d has length %zu, while column 0 has length %zu\n", i,
  //  output_relation.columns[i]->column_length, output_relation.columns[0]->column_length); throw
  //  InvalidInputException("Output columns have different sizes");
  //  	}
  //  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Hash Join GetData time: {:.2f} ms", duration.count() / 1000.0);

  return SourceResultType::FINISHED;
}

OperatorResultType GPUPhysicalHashJoin::Execute(GPUIntermediateRelation& input_relation,
                                                GPUIntermediateRelation& output_relation) const
{
  auto start = std::chrono::high_resolution_clock::now();

  if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
    // for RIGHT SEMI and RIGHT ANTI joins, the output is the RHS
    // we only need to output the RHS columns
    // the LHS columns are NULL
    if (output_relation.columns.size() != rhs_output_columns.col_idxs.size()) {
      throw InvalidInputException("Wrong input size");
    }
  } else if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
    // for SEMI and ANTI join, the output is the LHS
    // we only need to output the LHS columns
    // the RHS columns are NULL
    if (output_relation.columns.size() != lhs_output_columns.col_idxs.size()) {
      throw InvalidInputException("Wrong input size");
    }
  } else if (join_type == JoinType::RIGHT || join_type == JoinType::LEFT ||
             join_type == JoinType::INNER || join_type == JoinType::OUTER) {
    // for INNER and OUTER join, we output all columns
    if (output_relation.columns.size() !=
        lhs_output_columns.col_idxs.size() + rhs_output_columns.col_idxs.size()) {
      throw InvalidInputException("Wrong input size");
    }
  } else if (join_type == JoinType::MARK) {
    // for MARK join, we output all columns from the LHS and one extra boolean column
    if (output_relation.columns.size() != lhs_output_columns.col_idxs.size() + 1) {
      throw InvalidInputException("Wrong input size");
    }
  } else {
    throw InvalidInputException("Unsupported join type");
  }

  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  auto& rstate = runtime_state<HashJoinRuntimeState>(sirius_current_gpu);
  vector<shared_ptr<GPUColumn>> probe_key(conditions.size());
  for (int i = 0; i < conditions.size(); i++) {
    probe_key[i] = nullptr;
  }
  uint64_t* count;
  uint64_t* row_ids_left  = nullptr;
  uint64_t* row_ids_right = nullptr;
  uint8_t* output         = nullptr;  // for MARK JOIN
  // if (conditions.size() > 1) throw NotImplementedException("Multiple conditions not supported
  // yet");

  for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
    auto& condition     = conditions[cond_idx];
    auto join_key_index = condition.left->Cast<BoundReferenceExpression>().index;
    if (input_relation.columns[join_key_index]->is_unique) { rstate.unique_probe_keys = true; }
    SIRIUS_LOG_DEBUG("Materializing join key for probing hash table from index {}", join_key_index);
    probe_key[cond_idx] =
      HandleMaterializeExpression(input_relation.columns[join_key_index], gpuBufferManager);
  }

  // ── broadcast-probe runtime decision (candidates only) ────────────────────
  // Workers exchange ACTUAL per-partition probe/build sizes plus a local
  // eligibility bit (nullability can differ per partition) and all apply the
  // same rule to the same totals — the collective path stays worker-consistent.
  vector<shared_ptr<GPUColumn>> bcast_lhs_dense;
  vector<shared_ptr<GPUColumn>> bcast_build_dense;
  static const bool bcast_phase_time = std::getenv("MAGI_PHASE_TIME") != nullptr;
  // Cold-cache blind spot of the Sink-time replicated-probe veto: the probe
  // pipeline runs AFTER the build sink, so on the very first query the probe
  // subtree's tables were not cached yet and SubtreeOutputReplicated couldn't
  // see they are replicated. By Execute time they are — re-check and force
  // the ordinary local path (the candidate Sink also staged the local build).
  // Plan+catalog-deterministic → every worker flips identically, no exchange.
  if (rstate.bcast_candidate && !children.empty() && SubtreeOutputReplicated(*children[0])) {
    rstate.bcast_candidate  = false;
    rstate.use_shuffle_join = false;
  }
  // Same cold-cache re-check for the SEMI/LEFT/RIGHT family: a replicated
  // probe invisible at Sink time would make every GPU emit the same rows.
  if (rstate.semi_bcast && !children.empty() && SubtreeOutputReplicated(*children[0])) {
    throw NotImplementedException(
      "Multi-GPU SEMI/ANTI/LEFT/RIGHT join with a replicated probe side is not supported yet "
      "(falls back)");
  }
  if (rstate.bcast_candidate) {
    const auto bt0 = std::chrono::steady_clock::now();
    const uint64_t local_probe = probe_key.empty() ? 0 : probe_key[0]->column_length;
    const uint64_t local_build =
      (rstate.materialized_build_key && !rstate.materialized_build_key->columns.empty() &&
       rstate.materialized_build_key->columns[0])
        ? rstate.materialized_build_key->columns[0]->column_length
        : 0;
    // Round 1: sizes only (~µs). The strategy pick is a pure function of the
    // exchanged totals plus schema-derived widths, so all workers agree and
    // consistently run (or skip) round 2's rendezvous.
    const auto d1 = ExchangeBroadcastDecision(gpuBufferManager, local_probe, local_build, true);
    auto wire_width = [](const shared_ptr<GPUColumn>& col) -> uint64_t {
      if (!col) { return 8; }
      if (col->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) { return 16; }
      return col->data_wrapper.getColumnTypeSize();
    };
    uint64_t probe_row_bytes = 0, build_row_bytes = 0;
    for (auto& pk : probe_key) { probe_row_bytes += wire_width(pk); }
    for (auto lhs_idx : lhs_output_columns.col_idxs) {
      probe_row_bytes += wire_width(input_relation.columns[lhs_idx]);
    }
    for (auto& bk : rstate.materialized_build_key->columns) { build_row_bytes += wire_width(bk); }
    for (idx_t i = conditions.size(); i < rstate.hash_table_result->columns.size(); i++) {
      build_row_bytes += wire_width(rstate.hash_table_result->columns[i]);
    }
    const auto strategy =
      PickBcastStrategy(d1.probe, d1.build, probe_row_bytes, build_row_bytes);
    const auto bt1 = std::chrono::steady_clock::now();
    BroadcastDecision d = d1;
    d.eligible = false;
    double elig_ms = 0.0;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    if (strategy == BcastStrategy::PROBE) {
      // Round 2: per-partition eligibility (nullability can differ across
      // partitions, so it must be exchanged, not decided locally).
      bool eligible = true;
      for (auto& pk : probe_key) {
        if (!MaskAllValid(pk)) {
          eligible = false;
        }
      }
      bcast_lhs_dense.reserve(lhs_output_columns.col_idxs.size());
      for (auto lhs_idx : lhs_output_columns.col_idxs) {
        auto dense = HandleMaterializeExpression(input_relation.columns[lhs_idx], gpuBufferManager);
        if (!MaskAllValid(dense)) {
          eligible = false;
        }
        bcast_lhs_dense.push_back(std::move(dense));
      }
      elig_ms = ms(bt1, std::chrono::steady_clock::now());
      d = ExchangeBroadcastDecision(gpuBufferManager, local_probe, local_build, eligible);
      if (d.eligible) {
        rstate.broadcast_probe  = true;
        rstate.use_shuffle_join = false;
      }
    } else if (strategy == BcastStrategy::BUILD) {
      // Round 2 for the build side: keys are already materialized; the build
      // output (payload) columns may still be late-materialized refs, so
      // densify them here — they must be gathered anyway if we broadcast.
      bool eligible = true;
      for (auto& bk : rstate.materialized_build_key->columns) {
        if (!MaskAllValid(bk)) {
          eligible = false;
        }
      }
      bcast_build_dense.reserve(rstate.hash_table_result->columns.size() - conditions.size());
      for (idx_t i = conditions.size(); i < rstate.hash_table_result->columns.size(); i++) {
        auto dense = HandleMaterializeExpression(rstate.hash_table_result->columns[i],
                                                 gpuBufferManager);
        if (!MaskAllValid(dense)) {
          eligible = false;
        }
        bcast_build_dense.push_back(std::move(dense));
      }
      elig_ms = ms(bt1, std::chrono::steady_clock::now());
      d = ExchangeBroadcastDecision(gpuBufferManager, local_probe, local_build, eligible);
      if (d.eligible) {
        rstate.broadcast_build  = true;
        rstate.use_shuffle_join = false;
      }
    }
    if (bcast_phase_time) {
      const char* verdict = rstate.broadcast_probe   ? "broadcast-probe"
                            : rstate.broadcast_build ? "broadcast-build"
                                                     : "shuffle";
      const char* primary = strategy == BcastStrategy::PROBE   ? "probe"
                            : strategy == BcastStrategy::BUILD ? "build"
                                                               : "none";
      std::fprintf(stderr,
                   "[bcast-decision gpu=%d] probe_total=%llu(%lluB/row) build_total=%llu(%lluB/row) "
                   "primary=%s eligible=%d exch1=%.2fms elig=%.2fms -> %s\n",
                   sirius_current_gpu, (unsigned long long)d.probe,
                   (unsigned long long)probe_row_bytes, (unsigned long long)d.build,
                   (unsigned long long)build_row_bytes, primary, (int)d.eligible, ms(bt0, bt1),
                   elig_ms, verdict);
    }
  }

  // ── broadcast-probe: allgather the probe side, then use the local path ────
  // The build stayed local (disjoint per-GPU partitions), so probing with the
  // FULL probe set on every GPU yields each match exactly once; the collector
  // concatenation of the per-GPU outputs is the exact INNER join.
  if (rstate.broadcast_probe) {
    const auto at0 = std::chrono::steady_clock::now();
    for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
      probe_key[cond_idx] = AllgatherProbeColumn(probe_key[cond_idx], gpuBufferManager);
    }
    const auto at1 = std::chrono::steady_clock::now();
    for (idx_t i = 0; i < lhs_output_columns.col_idxs.size(); i++) {
      auto lhs_idx = lhs_output_columns.col_idxs[i];
      auto dense   = i < bcast_lhs_dense.size()
                       ? bcast_lhs_dense[i]
                       : HandleMaterializeExpression(input_relation.columns[lhs_idx], gpuBufferManager);
      input_relation.columns[lhs_idx] = AllgatherProbeColumn(dense, gpuBufferManager);
    }
    // Flush the allgather stream pool before anything consumes the gathered
    // columns (copies are enqueued on non-blocking streams; ag_key above is
    // enqueue time, the full cost lands here).
    cudaDeviceSynchronize();
    const auto at2 = std::chrono::steady_clock::now();
    if (bcast_phase_time) {
      std::fprintf(stderr, "[bcast-phase gpu=%d] ag_key=%.2fms ag_lhs=%.2fms\n",
                   sirius_current_gpu,
                   std::chrono::duration<double, std::milli>(at1 - at0).count(),
                   std::chrono::duration<double, std::milli>(at2 - at1).count());
    }
  }

  // ── broadcast-build: allgather the build side, then use the local path ────
  // The probe partition stays untouched; every GPU joins it against the FULL
  // build set (keys + build output columns), so the output remains
  // probe-partitioned and the per-GPU concatenation is the exact join. The
  // cudf hash build for INNER happens inside the probe call below anyway, so
  // swapping the build references here is all it takes.
  if (rstate.broadcast_build) {
    const auto at0 = std::chrono::steady_clock::now();
    for (idx_t c = 0; c < conditions.size(); c++) {
      auto full = AllgatherProbeColumn(rstate.materialized_build_key->columns[c],
                                       gpuBufferManager);
      rstate.materialized_build_key->columns[c] = full;
      if (c < rstate.hash_table_result->columns.size()) {
        rstate.hash_table_result->columns[c] = full;
      }
    }
    for (idx_t i = 0; i < bcast_build_dense.size(); i++) {
      rstate.hash_table_result->columns[conditions.size() + i] =
        AllgatherProbeColumn(bcast_build_dense[i], gpuBufferManager);
    }
    // Flush the allgather stream pool before the cudf build consumes the
    // gathered columns (copies are enqueued on non-blocking streams).
    cudaDeviceSynchronize();
    // Per-partition uniqueness does not imply global uniqueness across the
    // gathered build; drop the fast-path hint (cudf handles duplicates).
    rstate.unique_build_keys = false;
    if (bcast_phase_time) {
      std::fprintf(stderr, "[bcast-phase gpu=%d] ag_build=%.2fms (%llu rows full)\n",
                   sirius_current_gpu,
                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                             at0)
                     .count(),
                   (unsigned long long)rstate.materialized_build_key->columns[0]->column_length);
    }
  }

  // ── magi NVLink shuffle join (build side not replicated / forced) ─────────
  // Runs the full build+probe shuffle over NVLink and emits the join output
  // columns directly, bypassing the cudf probe + row_id materialization (after
  // the shuffle the rows are relocated to owner GPUs, so original row_ids don't
  // apply). v1: INNER, unique build key. LHS (probe) output columns flow as
  // probe payload, RHS (build) output columns as build payload; both ≤ 8-byte.
  if (rstate.use_shuffle_join) {
    // v1 contract: build key is UNIQUE (single-slot H_build → ≤1 match per probe
    // row). sirius does not propagate table-PK uniqueness to is_unique, so we do
    // NOT gate on rstate.unique_build_keys (it would reject Q11's supplier/nation
    // PK joins). A genuinely non-unique build key would undercount — a documented
    // v1 limitation (chained multi-value table is the follow-up).
    // Payload carried as 8-byte wire slots (≤ JOIN_MAX_PAYLOAD=4 per side). Reject
    // wider projections / non-8-byte columns loudly → DuckDB fallback (avoids the
    // int64 force-fit OOB read for INT32/DATE payloads).
    auto is_supported = [](GPUColumnTypeId tid) {
      return tid == GPUColumnTypeId::INT32 || tid == GPUColumnTypeId::INT64 ||
             tid == GPUColumnTypeId::FLOAT64 || tid == GPUColumnTypeId::DECIMAL ||
             tid == GPUColumnTypeId::VARCHAR;
    };
    // Wire-slot budget per side: a numeric payload is 1 slot, a VARCHAR is
    // inlined into 4 (length-prefixed, ≤31 chars). Each side's total slots must
    // fit JOIN_MAX_PAYLOAD = 4 — so a VARCHAR consumes a side's whole budget.
    auto slot_width = [](GPUColumnTypeId tid) {
      return tid == GPUColumnTypeId::VARCHAR ? 4 : 1;
    };
    int lhs_slots = 0, rhs_slots = 0;
    for (auto col_idx : lhs_output_columns.col_idxs) {
      auto tid = input_relation.columns[col_idx]->data_wrapper.type.id();
      if (!is_supported(tid))
        throw NotImplementedException(
          "magi shuffle join v1: LHS payload column type unsupported (INT32/INT64/DECIMAL/FLOAT64/VARCHAR)");
      lhs_slots += slot_width(tid);
    }
    for (idx_t i = 0; i < rhs_output_columns.col_idxs.size(); i++) {
      auto tid = rstate.shuffle_build_payload->columns[i]->data_wrapper.type.id();
      if (!is_supported(tid))
        throw NotImplementedException(
          "magi shuffle join v1: RHS payload column type unsupported (INT32/INT64/DECIMAL/FLOAT64/VARCHAR)");
      rhs_slots += slot_width(tid);
    }
    if (lhs_slots > 4 || rhs_slots > 4)
      throw NotImplementedException(
        "magi shuffle join v1: payload exceeds 4 wire slots/side (a VARCHAR uses all 4)");

    vector<shared_ptr<GPUColumn>> build_key_cols;
    for (idx_t c = 0; c < conditions.size(); c++)
      build_key_cols.push_back(rstate.materialized_build_key->columns[c]);
    vector<shared_ptr<GPUColumn>> build_payload;
    for (idx_t i = 0; i < rhs_output_columns.col_idxs.size(); i++)
      build_payload.push_back(rstate.shuffle_build_payload->columns[i]);
    vector<shared_ptr<GPUColumn>> probe_payload;
    for (auto col_idx : lhs_output_columns.col_idxs)
      probe_payload.push_back(
        HandleMaterializeExpression(input_relation.columns[col_idx], gpuBufferManager));

    vector<shared_ptr<GPUColumn>> out_key, out_build_payload, out_payload;
    magi_join_op::Run(sirius_current_gpu, build_key_cols, build_payload, probe_key,
                      probe_payload, out_key, out_build_payload, out_payload);
    // Output layout = LHS columns first, then RHS columns (matches the cudf path's
    // HandleMaterializeRowIDsLHS then ...RHS).
    const idx_t n_lhs = lhs_output_columns.col_idxs.size();
    for (idx_t i = 0; i < n_lhs; i++)                          output_relation.columns[i]         = out_payload[i];
    for (idx_t i = 0; i < rhs_output_columns.col_idxs.size(); i++) output_relation.columns[n_lhs + i] = out_build_payload[i];
    return OperatorResultType::FINISHED;
  }

  // probing hash table
  SIRIUS_LOG_DEBUG("Probing hash table");
  if (join_type == JoinType::INNER || join_type == JoinType::LEFT || join_type == JoinType::OUTER) {
    // check if there is a non-equality condition
    vector<shared_ptr<GPUColumn>> build_key(conditions.size());
    for (int cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
      build_key[cond_idx] = rstate.materialized_build_key->columns[cond_idx];
    }
    if (build_key[0]->column_length > INT32_MAX || probe_key[0]->column_length > INT32_MAX) {
      throw NotImplementedException("Column length greater than INT32_MAX is not supported");
    } else {
      bool has_non_equality_condition = false;
      for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
        if (conditions[cond_idx].comparison != ExpressionType::COMPARE_EQUAL &&
            conditions[cond_idx].comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
          has_non_equality_condition = true;
          break;
        }
      }
      if (!has_non_equality_condition) {
        if (join_type == JoinType::OUTER) {
          cudf_hash_full_join(
            probe_key, build_key, conditions.size(), row_ids_left, row_ids_right, count);
          rstate.outer_join_handled_in_execute = true;
        } else if (join_type == JoinType::LEFT) {
          cudf_hash_left_join(probe_key,
                              build_key,
                              conditions.size(),
                              row_ids_left,
                              row_ids_right,
                              count,
                              rstate.unique_build_keys);
        } else {
          cudf_hash_inner_join(probe_key,
                               build_key,
                               conditions.size(),
                               row_ids_left,
                               row_ids_right,
                               count,
                               rstate.unique_build_keys);
        }
      } else {
        if (join_type == JoinType::LEFT || join_type == JoinType::OUTER) {
          throw NotImplementedException(
            "Left/full outer join with non-equality condition is not supported yet");
        }
        cudf_mixed_or_conditional_inner_join(
          probe_key, build_key, conditions, join_type, row_ids_left, row_ids_right, count);
      }
    }
    // Eager-aggregation join-back verification: the rewrite grouped by the
    // narrow key only, which is exact iff every aggregate row matches EXACTLY
    // one probe row globally (probe-side key unique among matches). Each
    // aggregate key is guaranteed ≥1 match (its values came through this very
    // join's tables), so sum(outputs) == global aggregate rows ⟺ uniqueness.
    // Mismatch throws → gpu_processing falls back to DuckDB (never a wrong
    // answer, only a slower one).
    if (eager_agg_verify && join_type == JoinType::INNER) {
      const int      vgpus       = static_cast<int>(gpuBufferManager->tables_per_gpu.size());
      const uint64_t local_out   = count[0];
      // Under broadcast-build every GPU holds the FULL aggregate (global rows
      // == local build rows); under a partitioned build they sum.
      const uint64_t local_build = build_key[0]->column_length;
      struct VSlot { uint64_t out; uint64_t build; };
      static std::array<VSlot, 8>            vslots;
      static std::unique_ptr<std::barrier<>> vbar;
      static std::once_flag                  vbar_once;
      uint64_t total_out = local_out, total_build = local_build;
      if (vgpus > 1) {
        std::call_once(vbar_once, [&] { vbar = std::make_unique<std::barrier<>>(vgpus); });
        vslots[sirius_current_gpu] = {local_out, local_build};
        vbar->arrive_and_wait();
        total_out = 0;
        total_build = 0;
        for (int i = 0; i < vgpus; ++i) {
          total_out += vslots[i].out;
          total_build += vslots[i].build;
        }
        vbar->arrive_and_wait();
      }
      const uint64_t expected = rstate.broadcast_build ? local_build : total_build;
      if (total_out != expected) {
        throw NotImplementedException(
          "eager-agg rewrite verification failed: join-back output %llu != aggregate rows %llu "
          "(probe-side key not unique); falling back to DuckDB",
          total_out, expected);
      }
      SIRIUS_LOG_DEBUG("eager-agg join-back verified: {} output rows == {} aggregate rows",
                       total_out, expected);
    }
  } else if (join_type == JoinType::SEMI || join_type == JoinType::RIGHT ||
             join_type == JoinType::ANTI) {
    HandleProbeExpression(probe_key,
                          count,
                          row_ids_left,
                          row_ids_right,
                          rstate.gpu_hash_table,
                          rstate.ht_len,
                          conditions,
                          join_type,
                          rstate.unique_build_keys,
                          gpuBufferManager);
    // if (count[0] == 0) throw NotImplementedException("No match found");
  } else if (join_type == JoinType::MARK) {
    SIRIUS_LOG_DEBUG("Writing boolean column to output relation");
    HandleMarkExpression(
      probe_key, output, rstate.gpu_hash_table, rstate.ht_len, conditions, gpuBufferManager);
  } else if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
    HandleProbeExpression(probe_key,
                          count,
                          row_ids_left,
                          row_ids_right,
                          rstate.gpu_hash_table,
                          rstate.ht_len,
                          conditions,
                          join_type,
                          rstate.unique_build_keys,
                          gpuBufferManager);
  } else {
    throw NotImplementedException("Unsupported join type");
  }

  // materialize columns from the left table
  if (join_type == JoinType::SEMI || join_type == JoinType::ANTI || join_type == JoinType::INNER ||
      join_type == JoinType::RIGHT || join_type == JoinType::LEFT || join_type == JoinType::OUTER) {
    SIRIUS_LOG_DEBUG("Writing LHS columns to output relation");

    if (join_type == JoinType::SEMI || join_type == JoinType::ANTI || rstate.unique_build_keys) {
      HandleMaterializeRowIDsLHS(input_relation,
                                 output_relation,
                                 lhs_output_columns.col_idxs,
                                 count[0],
                                 row_ids_left,
                                 gpuBufferManager,
                                 true);
    } else {
      HandleMaterializeRowIDsLHS(input_relation,
                                 output_relation,
                                 lhs_output_columns.col_idxs,
                                 count[0],
                                 row_ids_left,
                                 gpuBufferManager,
                                 false);
    }
    // free all the columns in the input relation that are not in the lhs_output_columns
    // for (idx_t i = 0; i < input_relation.columns.size(); i++) {
    // 	if (find(lhs_output_columns.col_idxs.begin(), lhs_output_columns.col_idxs.end(), i) ==
    // lhs_output_columns.col_idxs.end()) {
    // 		gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[i]->data_wrapper.data),
    // 0); 		if (input_relation.columns[i]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
    // 			gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[i]->data_wrapper.offset),
    // 0);
    // 		}
    // 	}
    // }
  } else if (join_type == JoinType::MARK) {
    SIRIUS_LOG_DEBUG("Writing LHS columns to output relation");
    for (idx_t i = 0; i < lhs_output_columns.col_idxs.size(); i++) {
      auto lhs_col = lhs_output_columns.col_idxs[i];
      SIRIUS_LOG_DEBUG("Passing column idx {} from LHS to idx {} in output relation", lhs_col, i);
      // output_relation.columns[i] =
      // make_shared_ptr<GPUColumn>(input_relation.columns[lhs_col]->column_length,
      // input_relation.columns[lhs_col]->data_wrapper.type,
      // input_relation.columns[lhs_col]->data_wrapper.data,
      // 				input_relation.columns[lhs_col]->data_wrapper.offset,
      // input_relation.columns[lhs_col]->data_wrapper.num_bytes,
      // input_relation.columns[lhs_col]->data_wrapper.is_string_data,
      // 				input_relation.columns[lhs_col]->data_wrapper.validity_mask);
      output_relation.columns[i] = make_shared_ptr<GPUColumn>(input_relation.columns[lhs_col]);
      output_relation.columns[i]->row_ids      = input_relation.columns[lhs_col]->row_ids;
      output_relation.columns[i]->row_id_count = input_relation.columns[lhs_col]->row_id_count;
      if (rstate.unique_build_keys) {
        output_relation.columns[i]->is_unique = input_relation.columns[lhs_col]->is_unique;
      } else {
        output_relation.columns[i]->is_unique = false;
      }
    }
    // output_relation.columns[lhs_output_columns.col_idxs.size()] =
    // make_shared_ptr<GPUColumn>(probe_key[0]->column_length,
    // GPUColumnType(GPUColumnTypeId::BOOLEAN), output);
    auto validity_mask = createNullMask(probe_key[0]->column_length);
    output_relation.columns[lhs_output_columns.col_idxs.size()] = make_shared_ptr<GPUColumn>(
      probe_key[0]->column_length, GPUColumnType(GPUColumnTypeId::BOOLEAN), output, validity_mask);
    output_relation.columns[lhs_output_columns.col_idxs.size()]->row_ids = probe_key[0]->row_ids;
    output_relation.columns[lhs_output_columns.col_idxs.size()]->row_id_count =
      probe_key[0]->row_id_count;
    // free all the columns in the input relation that are not in the lhs_output_columns
    // for (idx_t i = 0; i < input_relation.columns.size(); i++) {
    // 	if (find(lhs_output_columns.col_idxs.begin(), lhs_output_columns.col_idxs.end(), i) ==
    // lhs_output_columns.col_idxs.end()) {
    // 		gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[i]->data_wrapper.data),
    // 0); 		if (input_relation.columns[i]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
    // 			gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[i]->data_wrapper.offset),
    // 0);
    // 		}
    // 	}
    // }
  } else if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
    // WE SHOULD NOT NEED TO DO ANYTHING HERE
  } else {
    throw NotImplementedException("Unsupported join type");
  }

  // materialize columns from the right tables
  if (join_type == JoinType::INNER || join_type == JoinType::RIGHT || join_type == JoinType::LEFT ||
      join_type == JoinType::OUTER) {
    SIRIUS_LOG_DEBUG("Writing row IDs from RHS to output relation");
    auto& ht_result = *rstate.hash_table_result;
    for (int col = 0; col < ht_result.columns.size(); col++) {
      gpuBufferManager->lockAllocation(
        reinterpret_cast<uint8_t*>(ht_result.columns[col]->data_wrapper.data),
        sirius_current_gpu);
      gpuBufferManager->lockAllocation(
        reinterpret_cast<uint8_t*>(ht_result.columns[col]->row_ids), sirius_current_gpu);
      // If the column type is VARCHAR, also lock the offset allocation
      if (ht_result.columns[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
        gpuBufferManager->lockAllocation(
          reinterpret_cast<uint8_t*>(ht_result.columns[col]->data_wrapper.offset),
          sirius_current_gpu);
      }
    }
    if (rstate.unique_probe_keys) {
      HandleMaterializeRowIDsRHS(ht_result,
                                 output_relation,
                                 rhs_output_columns.col_idxs,
                                 lhs_output_columns.col_idxs.size(),
                                 count[0],
                                 row_ids_right,
                                 gpuBufferManager,
                                 true);
    } else {
      HandleMaterializeRowIDsRHS(ht_result,
                                 output_relation,
                                 rhs_output_columns.col_idxs,
                                 lhs_output_columns.col_idxs.size(),
                                 count[0],
                                 row_ids_right,
                                 gpuBufferManager,
                                 false);
    }
    if (join_type == JoinType::INNER) {
      // free all the columns in the hash table result that are not in the rhs_output_columns
      // for (idx_t i = 0; i < hash_table_result->columns.size(); i++) {
      // 	if (find(rhs_output_columns.col_idxs.begin(), rhs_output_columns.col_idxs.end(), i) ==
      // rhs_output_columns.col_idxs.end()) {
      // 		gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(hash_table_result->columns[i]->data_wrapper.data),
      // 0); 		if (hash_table_result->columns[i]->data_wrapper.type.id() ==
      // GPUColumnTypeId::VARCHAR)
      // {
      // 			gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(hash_table_result->columns[i]->data_wrapper.offset),
      // 0);
      // 		}
      // 	}
      // }
    }
  } else if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
    SIRIUS_LOG_DEBUG("Writing row IDs from RHS to output relation");
    for (idx_t i = 0; i < rhs_output_columns.col_idxs.size(); i++) {
      const auto rhs_col = rhs_output_columns.col_idxs[i];
      SIRIUS_LOG_DEBUG(
        "Passing column idx {} from RHS (late materialized) to idx {} in output relation",
        rhs_col,
        i);
      output_relation.columns[i] = make_shared_ptr<GPUColumn>(
        0, rstate.hash_table_result->columns[rhs_col]->data_wrapper.type, nullptr, nullptr);
    }
  }

  if (join_type == JoinType::INNER || join_type == JoinType::SEMI || join_type == JoinType::MARK) {
    if (rstate.gpu_hash_table != nullptr) {
      gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(rstate.gpu_hash_table),
                                       sirius_current_gpu);
      rstate.gpu_hash_table = nullptr;
    }
  }
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Hash Join Execute time: {:.2f} ms", duration.count() / 1000.0);

  return OperatorResultType::FINISHED;
};

SinkResultType GPUPhysicalHashJoin::Sink(GPUIntermediateRelation& input_relation) const
{
  auto start = std::chrono::high_resolution_clock::now();

  if (!delim_types.empty() && join_type == JoinType::MARK) {
    // correlated MARK join
    throw NotImplementedException("Correlated MARK join not supported yet");
  }

  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  const int num_gpus                 = GPUBufferManager::GetMaxGpus();
  const int gpu                      = sirius_current_gpu;

  // Multi-GPU broadcast join. The build side must be REPLICATED — i.e. its
  // subtree reads only tables the scan cached in full on every GPU — so each
  // GPU builds an identical full hash table locally with zero cross-GPU
  // traffic, and probes it with its own partition of the probe side. Two
  // gates, both loud (-> DuckDB fallback):
  //   1. Join type: only probe-side-semantics joins (each probe row handled
  //      exactly once on its own GPU). Build-side-emitting types (RIGHT
  //      family, FULL OUTER) would emit unmatched build rows once per GPU
  //      and keep per-replica match flags — wrong without a cross-GPU flag
  //      merge; they need the magi shuffle join.
  //   2. Build layout: a partitioned (large) build side can't broadcast;
  //      large-large joins also need the magi shuffle join.
  bool semi_bcast_gather = false;
  if (num_gpus > 1) {
    // Multi-GPU coverage: the cudf probe path (INNER/LEFT) plus the SEMI
    // family via the replicated-build scheme below. MARK/RIGHT/OUTER still
    // fail loud → DuckDB fallback instead of crashing.
    // RIGHT rides the same replicated-build + match-flags scheme as the SEMI
    // family: matched pairs are emitted per GPU against the LOCAL probe
    // partition (disjoint by construction), and the NULL-padded unmatched
    // build rows are the flags COMPLEMENT (identical to RIGHT_ANTI's GetData),
    // merged across GPUs and emitted in disjoint row-id ranges.
    // LEFT is the easy sibling: its unmatched side is the PROBE — a LOCAL
    // property — so with the build replicated, each GPU's ordinary cudf left
    // join over its probe partition is already exact (no flags, no GetData).
    const bool semi_family = join_type == JoinType::SEMI || join_type == JoinType::ANTI ||
                             join_type == JoinType::RIGHT_SEMI ||
                             join_type == JoinType::RIGHT_ANTI ||
                             join_type == JoinType::RIGHT || join_type == JoinType::LEFT;
    if (join_type != JoinType::INNER && !semi_family) {
      throw NotImplementedException(
        "Multi-GPU broadcast hash join only supports INNER/LEFT/RIGHT/SEMI/ANTI yet (this join "
        "type falls back)");
    }
    // SubtreeOutputReplicated (not the scan-only SubtreeAllReplicated): an
    // interior ungrouped aggregate's output (e.g. Q15's max-of-revenue0
    // scalar) is already identical on every GPU after its cross-GPU merge —
    // allgathering such a build would concatenate N identical copies and
    // duplicate every probe match ×N.
    const bool build_replicated =
        (children.size() >= 2 && SubtreeOutputReplicated(*children[1]));
    // A REPLICATED probe side (small cached table, e.g. Q15's supplier) must
    // take the ordinary LOCAL join: every GPU already holds the full probe, so
    // joining it against the local build partition yields the exact join as
    // the union of disjoint outputs. Broadcasting the build (or shuffling)
    // would make BOTH sides present on every GPU and each GPU would emit the
    // same matches — global row duplication (Q15 emitted every result row
    // once per GPU).
    const bool probe_replicated =
        (!children.empty() && SubtreeOutputReplicated(*children[0]));
    static const bool force_shuffle = std::getenv("MAGI_FORCE_SHUFFLE_JOIN") != nullptr;

    if (semi_family) {
      // SEMI-family scheme: every GPU holds the FULL build side (allgather it
      // below unless the cache already replicated it) and runs the unchanged
      // single-GPU custom build + probe over its LOCAL probe partition.
      //  - SEMI/ANTI/LEFT emit local probe rows → outputs are naturally
      //    disjoint (LEFT's NULL-padding is a local property of the probe).
      //  - RIGHT_SEMI/RIGHT_ANTI/RIGHT emit BUILD rows: each GPU marks the
      //    subset its probe partition hits; GetData converts marks to row-id
      //    flags, OR-merges them across GPUs, and emits a disjoint range.
      // A REPLICATED probe breaks the disjointness argument (every GPU would
      // emit the same rows) — fail loud → DuckDB fallback. Execute re-checks
      // (probe tables may be uncached at Sink time on the first query).
      if (probe_replicated) {
        throw NotImplementedException(
          "Multi-GPU SEMI/ANTI/LEFT/RIGHT join with a replicated probe side is not supported "
          "yet (falls back)");
      }
      auto& srstate      = runtime_state<HashJoinRuntimeState>(gpu);
      srstate.semi_bcast = true;
      semi_bcast_gather  = !build_replicated;
    } else {
    // Broadcast CANDIDATE gate — plan-deterministic checks only, so every
    // worker takes the same Sink branch. The actual shuffle / broadcast-probe
    // / broadcast-build choice happens at Execute from real row counts (plan
    // estimates are useless for filtered probes). For a candidate this Sink
    // stages BOTH paths: the shuffle stash below AND the ordinary local
    // per-partition build (for INNER that's just materialize-reuse +
    // reference stores). Type/nullability eligibility of whichever side would
    // be broadcast is checked and exchanged at Execute.
    const bool bcast_candidate = !build_replicated && !probe_replicated && !force_shuffle &&
                                 join_type == JoinType::INNER && children.size() >= 2 &&
                                 std::getenv("MAGI_NO_BCAST_JOIN") == nullptr;
    if ((!build_replicated && !probe_replicated) || force_shuffle) {
      // Build side not replicated (or forced for testing) → magi NVLink shuffle
      // join. v1: INNER only. Stash the materialized build join key; the
      // probe-time Execute runs the whole build+probe shuffle and emits the LHS
      // output columns directly (no replicated hash table built here).
      if (join_type != JoinType::INNER) {
        throw NotImplementedException(
          "magi shuffle join: only INNER supported in v1 (this join type falls back)");
      }
      auto& rstate = runtime_state<HashJoinRuntimeState>(gpu);
      rstate.use_shuffle_join  = true;
      rstate.unique_build_keys = true;   // require unique build key (cleared below)
      rstate.materialized_build_key =
        make_shared_ptr<GPUIntermediateRelation>(conditions.size());
      for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
        auto& condition = conditions[cond_idx];
        if (condition.right->GetExpressionClass() != ExpressionClass::BOUND_REF) {
          throw InvalidInputException("magi shuffle join: unsupported build-side join condition");
        }
        auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
        // v1's H_build is single-slot — a non-unique build key would fan out
        // (one match per key), which we can't emit. Track it so Execute falls back.
        if (!input_relation.columns[join_key_index]->is_unique) rstate.unique_build_keys = false;
        rstate.materialized_build_key->columns[cond_idx] =
          HandleMaterializeExpression(input_relation.columns[join_key_index], gpuBufferManager);
      }
      // Build-side (RHS) output columns → build payload carried through the shuffle.
      rstate.shuffle_build_payload =
        make_shared_ptr<GPUIntermediateRelation>(rhs_output_columns.col_idxs.size());
      for (idx_t i = 0; i < rhs_output_columns.col_idxs.size(); i++)
        rstate.shuffle_build_payload->columns[i] =
          HandleMaterializeExpression(input_relation.columns[rhs_output_columns.col_idxs[i]],
                                      gpuBufferManager);
      if (!bcast_candidate) { return SinkResultType::FINISHED; }
      rstate.bcast_candidate = true;
      // fall through: ALSO stage the ordinary local per-partition build (cheap
      // — re-materialize is zero-copy on dense columns, the rest is reference
      // stores) so Execute can take the broadcast-probe local path without a
      // second Sink pass if the runtime decision picks broadcast.
    }
    }  // !semi_family
  }

  auto& rstate = runtime_state<HashJoinRuntimeState>(gpu);

  vector<shared_ptr<GPUColumn>> build_keys(conditions.size());
  for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
    auto& condition = conditions[cond_idx];
    if (condition.right->GetExpressionClass() != ExpressionClass::BOUND_REF) {
      throw InvalidInputException("Unsupported join condition");
    }
    auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
    if (input_relation.columns[join_key_index]->is_unique) { rstate.unique_build_keys = true; }
    SIRIUS_LOG_DEBUG("Materializing join key for building hash table from index {}",
                     join_key_index);
    build_keys[cond_idx] =
      HandleMaterializeExpression(input_relation.columns[join_key_index], gpuBufferManager);
    if (semi_bcast_gather) {
      // SEMI family: every GPU needs the FULL build. Concatenation order is
      // GPU-index order on every worker, so the build ROW-ID space is
      // identical across GPUs (the flags merge in GetData relies on this).
      build_keys[cond_idx] = AllgatherProbeColumn(build_keys[cond_idx], gpuBufferManager);
    }
  }
  if (semi_bcast_gather) {
    // AllgatherProbeColumn enqueues on non-blocking streams; the build kernel
    // below reads the gathered columns on the null stream.
    cudaDeviceSynchronize();
    // Gathered build keys lost the per-partition uniqueness guarantee.
    rstate.unique_build_keys = false;
  }
  if (rstate.semi_bcast) { rstate.semi_build_total = build_keys[0]->column_length; }

  // Per-GPU hash table over the (locally complete, replicated) build side.
  SIRIUS_LOG_DEBUG("Building hash table");
  rstate.ht_len = build_keys[0]->column_length * 2;
  if (join_type == JoinType::INNER || join_type == JoinType::SEMI || join_type == JoinType::MARK ||
      join_type == JoinType::ANTI) {
    if (rstate.ht_len == 0)
      rstate.gpu_hash_table = nullptr;
    else
      rstate.gpu_hash_table = (unsigned long long*)gpuBufferManager->customCudaMalloc<uint64_t>(
        rstate.ht_len * (conditions.size() + 1), gpu, 0);
  } else if (join_type == JoinType::RIGHT || join_type == JoinType::RIGHT_SEMI ||
             join_type == JoinType::RIGHT_ANTI) {
    if (rstate.ht_len == 0)
      rstate.gpu_hash_table = nullptr;
    else
      rstate.gpu_hash_table = (unsigned long long*)gpuBufferManager->customCudaMalloc<uint64_t>(
        rstate.ht_len * (conditions.size() + 2), gpu, 0);
  }

  if (join_type == JoinType::INNER || join_type == JoinType::LEFT || join_type == JoinType::OUTER) {
    // INNER/LEFT/OUTER probe through cudf at Execute time using
    // materialized_build_key; no custom hash table build here.
  } else {
    HandleBuildExpression(
      build_keys, rstate.gpu_hash_table, rstate.ht_len, conditions, join_type, gpuBufferManager);
  }

  // Per-GPU build-side result columns (sizes from the plan-time templates).
  rstate.hash_table_result =
    make_shared_ptr<GPUIntermediateRelation>(hash_table_result->columns.size());
  rstate.materialized_build_key =
    make_shared_ptr<GPUIntermediateRelation>(materialized_build_key->columns.size());
  // RIGHT_SEMI/RIGHT_ANTI under the multi-GPU scheme emit BUILD rows by
  // row-ids in the GATHERED space — the emission columns must be the gathered
  // full build, not local partition references.
  const bool semi_gather_rhs =
    semi_bcast_gather &&
    (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI ||
     join_type == JoinType::RIGHT || join_type == JoinType::LEFT);
  int right_idx = 0;
  for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
    auto& condition     = conditions[cond_idx];
    auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
    SIRIUS_LOG_DEBUG("Passing column idx {} from input relation to index {} in RHS hash table",
                     join_key_index,
                     cond_idx);
    rstate.hash_table_result->columns[cond_idx] =
      semi_gather_rhs ? build_keys[cond_idx] : input_relation.columns[join_key_index];
    rstate.materialized_build_key->columns[cond_idx] = build_keys[cond_idx];
    right_idx++;
  }
  for (idx_t i = 0; i < payload_columns.col_idxs.size(); i++) {
    auto payload_idx = payload_columns.col_idxs[i];
    SIRIUS_LOG_DEBUG("Passing column idx {} from input relation to index {} in RHS hash table",
                     payload_idx,
                     right_idx + i);
    if (semi_gather_rhs) {
      rstate.hash_table_result->columns[right_idx + i] = AllgatherProbeColumn(
        HandleMaterializeExpression(input_relation.columns[payload_idx], gpuBufferManager),
        gpuBufferManager);
    } else {
      rstate.hash_table_result->columns[right_idx + i] = input_relation.columns[payload_idx];
    }
  }
  if (semi_gather_rhs && !payload_columns.col_idxs.empty()) {
    // Flush the allgather stream pool before anything consumes the gathered
    // payload columns.
    cudaDeviceSynchronize();
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Hash Join Sink time: {:.2f} ms", duration.count() / 1000.0);

  return SinkResultType::FINISHED;
};

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void GPUPhysicalHashJoin::BuildJoinPipelines(GPUPipeline& current,
                                             GPUMetaPipeline& meta_pipeline,
                                             GPUPhysicalOperator& op,
                                             bool build_rhs)
{
  op.op_state.reset();
  op.sink_state.reset();

  // 'current' is the probe pipeline: add this operator
  auto& state = meta_pipeline.GetState();
  state.AddPipelineOperator(current, op);

  // save the last added pipeline to set up dependencies later (in case we need to add a child
  // pipeline)
  vector<shared_ptr<GPUPipeline>> pipelines_so_far;
  meta_pipeline.GetPipelines(pipelines_so_far, false);
  auto& last_pipeline = *pipelines_so_far.back();

  vector<shared_ptr<GPUPipeline>> dependencies;
  optional_ptr<GPUMetaPipeline> last_child_ptr;
  if (build_rhs) {
    // on the RHS (build side), we construct a child MetaPipeline with this operator as its sink
    auto& child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, op);
    child_meta_pipeline.Build(*op.children[1]);
    // if (op.children[1].get().CanSaturateThreads(current.GetClientContext())) {
    // 	// if the build side can saturate all available threads,
    // 	// we don't just make the LHS pipeline depend on the RHS, but recursively all LHS children
    // too.
    // 	// this prevents breadth-first plan evaluation
    // 	child_meta_pipeline.GetPipelines(dependencies, false);
    // 	last_child_ptr = meta_pipeline.GetLastChild();
    // }
  }

  // continue building the current pipeline on the LHS (probe side)
  op.children[0]->BuildPipelines(current, meta_pipeline);

  // if (last_child_ptr) {
  // 	// the pointer was set, set up the dependencies
  // 	meta_pipeline.AddRecursiveDependencies(dependencies, *last_child_ptr);
  // }

  switch (op.type) {
    case PhysicalOperatorType::POSITIONAL_JOIN:
      throw NotImplementedException("POSITIONAL_JOIN is not implemented yet");
      // Positional joins are always outer
      meta_pipeline.CreateChildPipeline(current, op, last_pipeline);
      return;
    case PhysicalOperatorType::CROSS_PRODUCT:
      throw NotImplementedException("CROSS_PRODUCT is not implemented yet");
      return;
    default: break;
  }

  // Join can become a source operator if it's RIGHT/OUTER, or if the hash join goes out-of-core
  bool add_child_pipeline = false;
  auto& join_op           = op.Cast<GPUPhysicalHashJoin>();
  if (join_op.IsSource()) { add_child_pipeline = true; }

  if (add_child_pipeline) { meta_pipeline.CreateChildPipeline(current, op, last_pipeline); }
}

void GPUPhysicalHashJoin::BuildPipelines(GPUPipeline& current, GPUMetaPipeline& meta_pipeline)
{
  GPUPhysicalHashJoin::BuildJoinPipelines(current, meta_pipeline, *this);
}

}  // namespace duckdb
