// magi_runtime_shared.hpp — accessor surface to the shared magi runtime
// singleton owned by q1_dispatcher.cu.
//
// Sibling query dispatchers (q5_dispatcher.cu, future q9_dispatcher.cu, ...)
// pull these declarations in to reach into the magi runtime (Endpoints,
// ChannelRuntime, P2P streams, identity row_ids cache) without seeing the
// MagiState struct definition or the heavy magi template headers.
//
// All definitions live in q1_dispatcher.cu; this header just declares the
// function entry points.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "legacy/operator/magi_q1.hpp"  // for NUM_GPUS

namespace duckdb {
namespace magi_q1 {

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

}  // namespace magi_q1
}  // namespace duckdb
