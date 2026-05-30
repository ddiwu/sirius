// magi_groupby_runtime.cu — single TU hosting the generic magi groupby path.
//
// What lives here:
//   1. Explicit instantiations of `distributed_hash_groupby_kernel<KeyT,…>`
//      for KeyT ∈ {int32_t, uint64_t}. These end up in libsirius_extension
//      so no per-query .cu / .hpp is needed for new queries.
//   2. Per-GPU `AggSlot64` device buffer (one byte arena per GPU, reused
//      across queries; cleared via cudaMemset at session start).
//   3. Per-GPU AggOpEntry device buffer (the ops table is small, copied
//      per query).
//   4. The host entry `distributed_hash_groupby_run_per_gpu` that drives
//      one worker thread through the magi runtime barrier exchange and
//      launches the right KeyT kernel.
//
// This is the analog of cudf's groupby_hash.cu — one file, no per-query
// dispatcher. Q1's and Q5's special-cased .cu files remain in tree under
// MAGI_LEGACY=1 only.
//
// Build is gated by ENABLE_MAGI_TPCH; this TU is not compiled when off.

#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#include "data_plane/api/distributed_hash_groupby.cuh"
#include "data_plane/config.cuh"

#include "legacy/operator/magi_distributed_groupby.hpp"
#include "legacy/operator/magi_runtime_shared.hpp"

namespace duckdb { namespace magi_generic {

// ── Kernel-launch constants ──────────────────────────────────────────────
constexpr int    BLOCK_SIZE      = 1024;
constexpr int    N_LOCAL_SLOTS   = magi_ops::DEFAULT_N_LOCAL_SLOTS;
constexpr int    N_GLOBAL_SLOTS  = magi_ops::DEFAULT_N_GLOBAL_SLOTS;
constexpr size_t AGG_BUF_BYTES   = 64 * N_GLOBAL_SLOTS;
constexpr int    MAX_AGG_OPS     = 16;  // ample for any TPC-H GROUP BY

}}  // namespace duckdb::magi_generic

// ── Per-KeyT init kernel ──────────────────────────────────────────────────
// `cudaMemset(agg, 0, …)` works for uint64 keys (empty_key_v<uint64_t> = 0)
// but not for int32 keys (empty_key_v<int32_t> = -1). One small kernel
// writes the correct empty-key sentinel into every slot at session start.
namespace duckdb { namespace magi_generic {
template <typename KeyT, int N_GLOBAL_SLOTS>
__global__ void init_global_agg_slots(magi_ops::AggSlot64<KeyT>* agg)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N_GLOBAL_SLOTS) {
    agg[i] = magi_ops::AggSlot64<KeyT>{};   // zero values + count + index
    agg[i].key = magi_ops::empty_key_v<KeyT>;
  }
}
}}  // namespace duckdb::magi_generic

// ── Explicit kernel instantiations (KeyT ∈ {int32_t, uint64_t}) ───────────
//
// Outside namespace duckdb so the symbols match what the kernel template
// expects in `namespace magi`. This is the analog of q1_dispatcher.cu's
// `template __global__ void q1_kernel<...>` instantiation.
namespace magi {

template __global__ void
distributed_hash_groupby_kernel<int32_t,
                                 duckdb::magi_generic::BLOCK_SIZE,
                                 KBUFFERING_INTRA_PARTITION_SIZE,
                                 KBUFFERING_INTER_PARTITION_SIZE,
                                 duckdb::magi_generic::N_LOCAL_SLOTS,
                                 duckdb::magi_generic::N_GLOBAL_SLOTS>(
    const std::uint64_t*, std::uint64_t,
    magi_ops::ColPack,
    const magi_ops::AggOpEntry*, int,
    magi_ops::AggSlot64<int32_t>*, bool);

template __global__ void
distributed_hash_groupby_kernel<std::uint64_t,
                                 duckdb::magi_generic::BLOCK_SIZE,
                                 KBUFFERING_INTRA_PARTITION_SIZE,
                                 KBUFFERING_INTER_PARTITION_SIZE,
                                 duckdb::magi_generic::N_LOCAL_SLOTS,
                                 duckdb::magi_generic::N_GLOBAL_SLOTS>(
    const std::uint64_t*, std::uint64_t,
    magi_ops::ColPack,
    const magi_ops::AggOpEntry*, int,
    magi_ops::AggSlot64<std::uint64_t>*, bool);

}  // namespace magi

namespace duckdb { namespace magi_generic {

// ── Per-GPU device buffers (lazy-init) ────────────────────────────────────
// `g_agg_dev[i]`  : N_GLOBAL_SLOTS × 64B = 16KB per GPU, both KeyT
//                    instantiations cast in-place (same byte layout for
//                    the values region).
// `g_ops_dev[i]`  : MAX_AGG_OPS × 4B = 64B per GPU; AggOpEntry table
//                    copied here per query.
static std::byte*               g_agg_dev[NUM_GPUS] = { nullptr };
static magi_ops::AggOpEntry*    g_ops_dev[NUM_GPUS] = { nullptr };
static std::mutex               g_init_mu;

static void EnsureDeviceBuffers()
{
  std::lock_guard<std::mutex> lk(g_init_mu);
  if (g_agg_dev[0] != nullptr) return;
  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = magi_q1::magi_phys_gpu(i);
    cudaSetDevice(gpu);
    cudaMalloc(reinterpret_cast<void**>(&g_agg_dev[i]), AGG_BUF_BYTES);
    cudaMalloc(reinterpret_cast<void**>(&g_ops_dev[i]),
               sizeof(magi_ops::AggOpEntry) * MAX_AGG_OPS);
  }
}

// ── Per-thread barrier exchange ──────────────────────────────────────────
// Same shape as Q5Exchange: stash inputs, sync, run, sync, return slice.
// `key_kind_for_this_query` is written by gpu_id==0 (all workers see the
// same shape since sirius's plan-gen is deterministic) before
// `after_session_start.arrive_and_wait()`, then read after the barrier.
struct GenericExchange {
  std::barrier<>                             begin{NUM_GPUS};
  std::barrier<>                             after_session_start{NUM_GPUS};
  std::barrier<>                             end{NUM_GPUS};
  std::array<PerGpuInputs, NUM_GPUS>         inputs{};
  std::array<const magi_ops::AggOpEntry*,
             NUM_GPUS>                       ops_host_ptrs{};
  std::array<int, NUM_GPUS>                  n_ops{};
  KeyKind                                    key_kind_for_this_query = KeyKind::INT32;
  std::uint64_t                              session_id = 0;
};
static GenericExchange& exchange() { static GenericExchange e; return e; }

// ── Per-GPU launch (templated on KeyT) ────────────────────────────────────
// Mirrors Q5MagiRunPerGpu's flow exactly; the only differences are the
// kernel symbol and the slice-extraction loop.
template <typename KeyT>
static std::size_t run_per_gpu_typed(int                         gpu_id,
                                      std::vector<AggResultRow>&  my_slice)
{
  auto& xc = exchange();
  const int gpu = magi_q1::magi_phys_gpu(gpu_id);
  cudaSetDevice(gpu);

  // Clear receiver global agg (writes empty_key_v<KeyT> per slot — a plain
  // memset to 0 would mis-mark int32 slots as "occupied" since
  // empty_key_v<int32_t> = -1).
  auto* agg_typed_for_init =
      reinterpret_cast<magi_ops::AggSlot64<KeyT>*>(g_agg_dev[gpu_id]);
  constexpr int INIT_BLOCK = 64;
  constexpr int INIT_GRID  = (N_GLOBAL_SLOTS + INIT_BLOCK - 1) / INIT_BLOCK;
  init_global_agg_slots<KeyT, N_GLOBAL_SLOTS>
      <<<INIT_GRID, INIT_BLOCK, 0, magi_q1::magi_stream(gpu_id)>>>(
          agg_typed_for_init);
  cudaMemcpyAsync(g_ops_dev[gpu_id],
                   xc.ops_host_ptrs[gpu_id],
                   sizeof(magi_ops::AggOpEntry) * xc.n_ops[gpu_id],
                   cudaMemcpyHostToDevice,
                   magi_q1::magi_stream(gpu_id));
  magi_q1::magi_set_tuple_size(gpu_id, sizeof(magi_ops::AggSlot64<KeyT>));

  // Session bump (thread 0) + sync.
  if (gpu_id == 0) xc.session_id = magi_q1::magi_bump_session();
  xc.after_session_start.arrive_and_wait();

  // Launch the generic kernel.
  const auto&         in      = xc.inputs[gpu_id];
  std::uint64_t*      row_ids = magi_q1::GetIdentityRowIdsShared(gpu, in.n_filtered);
  auto* global_agg_typed =
      reinterpret_cast<magi_ops::AggSlot64<KeyT>*>(g_agg_dev[gpu_id]);

  magi::distributed_hash_groupby_kernel<KeyT,
                                         BLOCK_SIZE,
                                         KBUFFERING_INTRA_PARTITION_SIZE,
                                         KBUFFERING_INTER_PARTITION_SIZE,
                                         N_LOCAL_SLOTS,
                                         N_GLOBAL_SLOTS>
      <<<USER_KERNEL_GRID_SIZE, BLOCK_SIZE, 0,
         magi_q1::magi_stream(gpu_id)>>>(row_ids,
                                          in.n_filtered,
                                          in.cols,
                                          g_ops_dev[gpu_id],
                                          xc.n_ops[gpu_id],
                                          global_agg_typed,
                                          /*just_load=*/false);
  cudaStreamSynchronize(magi_q1::magi_stream(gpu_id));
  magi_q1::magi_sync_after_session(gpu_id, xc.session_id);

  // D2H the per-GPU agg + filter to non-empty slots.
  std::vector<magi_ops::AggSlot64<KeyT>> host(N_GLOBAL_SLOTS);
  cudaMemcpy(host.data(), g_agg_dev[gpu_id], AGG_BUF_BYTES,
             cudaMemcpyDeviceToHost);

  my_slice.clear();
  for (int j = 0; j < N_GLOBAL_SLOTS; ++j) {
    if (host[j].key == magi_ops::empty_key_v<KeyT>) continue;
    AggResultRow row{};
    // Widen KeyT to u64 for the result row. int32 sign-extends — caller
    // knows the original column type and re-narrows on emit.
    row.key_as_u64 = static_cast<std::uint64_t>(
        static_cast<std::make_unsigned_t<KeyT>>(host[j].key));
    for (int v = 0; v < magi_ops::AggSlot64<KeyT>::N_DOUBLES; ++v) {
      row.values[v] = host[j].values[v];
    }
    row.partial_count = host[j].partial_count;
    my_slice.push_back(row);
  }

  xc.end.arrive_and_wait();
  return my_slice.size();
}

// ── Public entry ─────────────────────────────────────────────────────────
std::size_t distributed_hash_groupby_run_per_gpu(
    int                                      gpu_id,
    const PerGpuInputs&                      inputs,
    const std::vector<magi_ops::AggOpEntry>& ops,
    KeyKind                                  key_kind,
    std::vector<AggResultRow>&               my_slice)
{
  if (gpu_id < 0 || gpu_id >= NUM_GPUS) {
    std::fprintf(stderr,
                 "[magi-generic] bad gpu_id=%d (NUM_GPUS=%d)\n",
                 gpu_id, NUM_GPUS);
    return 0;
  }
  if (static_cast<int>(ops.size()) > MAX_AGG_OPS) {
    std::fprintf(stderr,
                 "[magi-generic] %zu agg ops > MAX_AGG_OPS=%d; bump and "
                 "rebuild\n",
                 ops.size(), MAX_AGG_OPS);
    return 0;
  }

  magi_q1::MagiInitOnce();
  EnsureDeviceBuffers();

  auto& xc = exchange();
  xc.inputs[gpu_id]        = inputs;
  xc.ops_host_ptrs[gpu_id] = ops.data();
  xc.n_ops[gpu_id]         = static_cast<int>(ops.size());
  if (gpu_id == 0) xc.key_kind_for_this_query = key_kind;
  xc.begin.arrive_and_wait();

  // Per-worker dispatch on KeyT. All workers see the same plan, so the
  // KeyKind written by gpu_id==0 is the same value every worker computes
  // independently from its inputs — but reading from the exchange is safe
  // post-barrier and avoids re-deriving the type.
  switch (xc.key_kind_for_this_query) {
    case KeyKind::INT32:
      return run_per_gpu_typed<std::int32_t> (gpu_id, my_slice);
    case KeyKind::UINT64:
      return run_per_gpu_typed<std::uint64_t>(gpu_id, my_slice);
  }
  return 0;
}

}}  // namespace duckdb::magi_generic
