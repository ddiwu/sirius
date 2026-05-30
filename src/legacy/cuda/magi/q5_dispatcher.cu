// q5_dispatcher.cu — sirius-side per-GPU entry for the Magi-backed
// TPC-H Q5 GROUP BY. Mirrors q1_dispatcher.cu's Q1MagiRunPerGpu pattern,
// shares the runtime singleton (Endpoint / ChannelRuntime / KBuffering)
// via accessors in magi_runtime_shared.hpp. Owns only Q5's per-GPU
// AggSlot device buffer.
//
// Build is gated by ENABLE_MAGI_TPCH; this TU is not compiled when off.

#include <array>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "gpu_buffer_manager.hpp"
#include "data_plane/queries/q5.cuh"
#include "data_plane/config.cuh"  // KBUFFERING_INTRA/INTER_PARTITION_SIZE, USER_KERNEL_GRID_SIZE

#include "legacy/operator/magi_q5.hpp"
#include "legacy/operator/magi_runtime_shared.hpp"

namespace duckdb { namespace magi_q5 {

constexpr int Q5_BLOCK_SIZE = 1024;

}}  // namespace duckdb::magi_q5

// ── Explicit instantiation of q5_kernel ────────────────────────────────────
namespace q5 {
template __global__ void q5_kernel<duckdb::magi_q5::Q5_BLOCK_SIZE,
                                    KBUFFERING_INTRA_PARTITION_SIZE,
                                    KBUFFERING_INTER_PARTITION_SIZE>(
    const std::uint64_t*, std::size_t,
    const std::uint8_t*, const std::uint64_t*,
    const double*,
    Q5AggSlot*, bool);
}  // namespace q5

namespace duckdb {
namespace magi_q5 {

// ── Q5-specific per-GPU state ─────────────────────────────────────────────
// File-static; one entry per logical GPU. Allocated on the first call to
// Q5MagiRunPerGpu (under g_q5_init_mu).
static ::q5::Q5AggSlot* g_q5_agg_dev[NUM_GPUS] = { nullptr };
static std::mutex g_q5_init_mu;

static void EnsureQ5AggDev()
{
  std::lock_guard<std::mutex> lk(g_q5_init_mu);
  if (g_q5_agg_dev[0] != nullptr) return;  // already allocated
  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = magi_q1::magi_phys_gpu(i);
    cudaSetDevice(gpu);
    cudaMalloc(&g_q5_agg_dev[i],
               ::q5::Q5_AGG_SLOTS * sizeof(::q5::Q5AggSlot));
  }
}

// ── Per-thread barrier exchange ──────────────────────────────────────────
struct Q5Exchange {
  std::barrier<>                begin{NUM_GPUS};
  std::barrier<>                after_session_start{NUM_GPUS};
  std::barrier<>                end{NUM_GPUS};
  std::array<PerGpuInputs, NUM_GPUS> inputs{};
  std::uint64_t                 session_id = 0;
};
static Q5Exchange& exchange()
{
  static Q5Exchange e;
  return e;
}

std::size_t Q5MagiRunPerGpu(int                                gpu_id,
                            const PerGpuInputs&                my_inputs,
                            std::vector<AggResultRow>&         my_slice)
{
  // Shared init: builds Endpoints/Channels on first call from ANY query;
  // Q1 may have already done this — idempotent.
  magi_q1::MagiInitOnce();
  EnsureQ5AggDev();

  if (gpu_id < 0 || gpu_id >= NUM_GPUS) {
    std::fprintf(stderr,
                 "[magi-q5] Q5MagiRunPerGpu: bad gpu_id=%d (NUM_GPUS=%d)\n",
                 gpu_id, NUM_GPUS);
    return 0;
  }
  auto& xc = exchange();

  // Phase timing instrumentation (SIRIUS_MAGI_PROFILE=1 to enable).
  const bool profile = std::getenv("SIRIUS_MAGI_PROFILE") != nullptr;
  auto T0 = std::chrono::high_resolution_clock::now();
  auto tick = [&](const char* tag) {
    if (!profile) return;
    auto now = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(now - T0).count();
    std::printf("[magi-q5-prof gpu=%d] %-32s +%8.1f us\n", gpu_id, tag, us);
    T0 = now;
  };

  // Stash inputs + barrier so every worker thread has reached this point.
  xc.inputs[gpu_id] = my_inputs;
  xc.begin.arrive_and_wait();
  tick("phase B/begin barrier");

  int gpu = magi_q1::magi_phys_gpu(gpu_id);
  cudaSetDevice(gpu);
  cudaMemset(g_q5_agg_dev[gpu_id], 0,
             ::q5::Q5_AGG_SLOTS * sizeof(::q5::Q5AggSlot));
  magi_q1::magi_set_tuple_size(gpu_id, sizeof(::q5::Q5Tuple));
  tick("phase C/cudaMemset+set_tuple");

  // One thread bumps the session counter; the rest pick it up after barrier.
  if (gpu_id == 0) xc.session_id = magi_q1::magi_bump_session();
  xc.after_session_start.arrive_and_wait();
  tick("phase C/session barrier");

  // Launch this GPU's q5_kernel on its own input slice.
  const auto&    in      = xc.inputs[gpu_id];
  std::uint64_t* row_ids = magi_q1::GetIdentityRowIdsShared(gpu, in.n_filtered);
  ::q5::q5_kernel<Q5_BLOCK_SIZE,
                  KBUFFERING_INTRA_PARTITION_SIZE,
                  KBUFFERING_INTER_PARTITION_SIZE>
      <<<USER_KERNEL_GRID_SIZE, Q5_BLOCK_SIZE, 0,
         magi_q1::magi_stream(gpu_id)>>>(row_ids,
                                          in.n_filtered,
                                          in.n_name_chars,
                                          in.n_name_offsets,
                                          in.d_revenue,
                                          g_q5_agg_dev[gpu_id],
                                          /*just_load=*/false);
  cudaStreamSynchronize(magi_q1::magi_stream(gpu_id));
  tick("phase D/q5_kernel + streamSync");

  magi_q1::magi_sync_after_session(gpu_id, xc.session_id);
  tick("phase E/sync_after_session");

  // D2H this GPU's slice (256-slot sparse table). Filter non-empty.
  std::vector<::q5::Q5AggSlot> host(::q5::Q5_AGG_SLOTS);
  cudaMemcpy(host.data(), g_q5_agg_dev[gpu_id],
             ::q5::Q5_AGG_SLOTS * sizeof(::q5::Q5AggSlot),
             cudaMemcpyDeviceToHost);
  tick("phase F/agg_dev D2H");

  my_slice.clear();
  for (int j = 0; j < ::q5::Q5_AGG_SLOTS; ++j) {
    if (host[j].key == 0ULL && host[j].count == 0) continue;
    my_slice.push_back(AggResultRow{
        /*n_name_packed=*/ host[j].key,
        /*sum_revenue=*/   host[j].sum_revenue,
        /*count=*/         host[j].count,
    });
  }
  tick("phase F/scan non-empty slots");

  xc.end.arrive_and_wait();
  tick("phase G/end barrier");
  return my_slice.size();
}

}  // namespace magi_q5
}  // namespace duckdb
