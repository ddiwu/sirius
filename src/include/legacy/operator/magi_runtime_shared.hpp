// magi_runtime_shared.hpp — accessor surface to the shared magi runtime
// singleton owned by magi_runtime.cu.
//
// The per-query dispatchers (magi_groupby_runtime.cu, magi_join_runtime.cu)
// pull these declarations in to reach into the magi runtime (Endpoints,
// ChannelRuntime, P2P streams, identity row_ids cache) without seeing the
// MagiState struct definition or the heavy magi template headers.
//
// All definitions live in magi_runtime.cu; this header just declares the
// function entry points.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "legacy/gpu_buffer_manager.hpp"  // for kSiriusLegacyNumGpus

namespace duckdb {
namespace magi_runtime {

// Number of GPUs the legacy magi runtime targets, set at cmake configure
// time via -DSIRIUS_LEGACY_NUM_GPUS=N (mirrored into kSiriusLegacyNumGpus).
// The shared runtime and every per-query dispatcher size their per-GPU
// arrays from this constant.
constexpr int NUM_GPUS = kSiriusLegacyNumGpus;

// One-shot init: builds Endpoints, ChannelRuntimes, P2P paths. Idempotent;
// safe to call from any dispatcher's per-GPU entry.
void MagiInitOnce();

// Physical CUDA device id for a logical gpu_id ∈ [0, NUM_GPUS).
int magi_phys_gpu(int gpu_id);

// Stream associated with the given GPU's channel.
cudaStream_t magi_stream(int gpu_id);

// Tell the channel runtime the wire-tuple size for the next session. Must
// be called by every per-GPU worker before launching its kernel — the
// channel uses this to compute partition byte offsets.
void magi_set_tuple_size(int gpu_id, std::size_t tuple_bytes);

// Bump the session counter and return the new session_id. Should be called
// by exactly one worker (typically gpu_id==0) per query session; the result
// is then broadcast to all workers via the dispatcher's barrier.
std::uint64_t magi_bump_session();

// Block until the host worker has drained / acknowledged the session.
void magi_sync_after_session(int gpu_id, std::uint64_t session_id);

// Identity row_ids[0..n) on the given physical GPU, cached and grown on
// demand. Sirius's per-GPU TableScan output is dense, so row_ids[i] = i
// works for every query — we just need a device buffer to feed magi.
std::uint64_t* GetIdentityRowIdsShared(int phys_gpu, std::size_t n);

// ── Unified-pool allocation (implemented in magi_runtime.cu, which can see
// GPUBufferManager). magi's device memory comes out of sirius's reserved
// pools instead of raw cudaMalloc, so it never competes for the residual
// free memory outside the pools (at SF100 the pools left ~0.3GB free and
// magi's unchecked lazy-init mallocs handed out garbage pointers → illegal
// memory access on 16 of 22 queries).
//   persistent=true  → CACHE pool (explicit gpu index; survives the per-query
//                      ResetBuffer, freed only by ResetCache — arena-class).
//   persistent=false → PROCESSING pool (routes to the CALLING thread's GPU;
//                      reclaimed automatically by end-of-query ResetBuffer —
//                      per-query wire buffers / counters).
// Throws duckdb InvalidInputException when the pool is exhausted.
std::uint8_t* magi_pool_alloc(std::size_t bytes, int gpu_id, bool persistent);

// Eagerly carve magi's persistent arenas (groupby agg/stage + join H_build +
// the small ops/fields tables) out of the CACHE pool. Called from
// gpu_buffer_init right after the pools are reserved — the cache bump pointer
// is still 0, so the arenas always fit and table caching gets what remains
// (its own cache-size check reports overflow cleanly).
void magi_prealloc_arenas();

}  // namespace magi_runtime
}  // namespace duckdb
