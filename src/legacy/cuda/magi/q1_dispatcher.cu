// q1_dispatcher.cu — sirius-side entry point for the Magi-backed
// cross-GPU TPC-H Q1 implementation.
//
// Phase 2: lazy-initialise the Magi runtime inside sirius's process,
// expose a function the sirius MagiQ1 table function can call to launch
// q1_kernel across NUM_GPUS partitions and read the per-(rf,ls) agg
// results back out.
//
// All Magi state is held in a translation-unit-local singleton MagiState.
// The runtime is started once (host worker threads, Endpoints,
// ChannelRuntime per GPU, NVLink P2P paths between every pair) on the
// first invocation and reused across queries; per-query work is just
// agg slot reset + kernel launch + sync_after_session.
//
// Build is gated by ENABLE_MAGI_TPCH; this TU is not compiled when off.

#include <array>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <gdrapi.h>

#include "gpu_buffer_manager.hpp"  // for kSiriusLegacyNumGpus
#include "legacy/operator/magi_runtime_shared.hpp"  // ensures decl/def of magi_phys_gpu etc. match
#include "data_plane/queries/q1.cuh"
#include "data_plane/channel/channel_runtime.cuh"
#include "data_plane/infra/endpoint.cuh"
#include "data_plane/infra/semaphore.cuh"
#include "data_plane/transport/nvlink_path.cuh"
#include "data_plane/config.cuh"

// ── Magi runtime + kernel sizing constants ────────────────────────────────
// NUM_GPUS matches sirius's `kSiriusLegacyNumGpus`. PARTITIONS_COUNT is a
// compile-time template parameter to Endpoint.
//
// q1_kernel uses more registers per thread than q3_kernel because of the
// 6 column pointers + 2 offset arrays (vs q3's single tuple-array pointer)
// — naive register layout pushes ~70 regs/thread, so 1024 threads/block
// would exceed the H100 SM register budget (cudaErrorLaunchOutOfResources,
// "too many resources requested for launch").
//
// q1.cuh sidesteps this by caching all 9 input pointers (+ partitions_count)
// in __shared__ CachedInputs at producer entry — those are uniform across
// the block, so per-thread regs go down by ~16. With that fix, 1024 fits.
namespace duckdb { namespace magi_q1 {
// NUM_GPUS already declared in magi_q1.hpp (= kSiriusLegacyNumGpus); just
// add the dispatcher-private partition + block size here.
constexpr int PARTITIONS_COUNT = NUM_GPUS;
constexpr int Q1_BLOCK_SIZE    = 1024;
}}  // namespace duckdb::magi_q1

// ── Explicit instantiation of q1_kernel ────────────────────────────────────
// Kernel template params must match the Endpoint instantiation below.
namespace q1 {
template __global__ void q1_kernel<duckdb::magi_q1::Q1_BLOCK_SIZE,
                                    KBUFFERING_INTRA_PARTITION_SIZE,
                                    KBUFFERING_INTER_PARTITION_SIZE>(
    const uint64_t*, size_t,
    const double*, const double*, const double*, const double*,
    const uint8_t*, const uint64_t*,
    const uint8_t*, const uint64_t*,
    Q1AggSlot*, bool);
}  // namespace q1

namespace duckdb {
namespace magi_q1 {

using EndpointT = ::Endpoint<1, 1, PARTITIONS_COUNT,
                             USER_KERNEL_GRID_SIZE, USER_KERNEL_GRID_SIZE,
                             K_BUFFERING_K,
                             KBUFFERING_INTRA_PARTITION_SIZE,
                             KBUFFERING_INTER_PARTITION_SIZE,
                             ::q1::Q1Tuple>;
using ChannelT = channel::ChannelRuntime<USER_KERNEL_GRID_SIZE>;

// ── Singleton runtime state ────────────────────────────────────────────────
struct MagiState {
  bool                              initialised = false;
  gdr_t                             gdr{};
  std::vector<int>                  gpu_ids;
  std::unique_ptr<magi::P2PMemcpyOp> nvlink;
  std::vector<EndpointT*>           endpoints;            // per-GPU
  std::unique_ptr<ChannelT[]>       channels;             // per-GPU, non-moveable
  channel::SessionBarrier           barrier;
  std::map<int, KBuffering*>        recv_stagings;
  std::map<int, Semaphore*>         recv_connections;
  std::vector<std::map<int, int>>   fwd_tables;
  std::map<int, int>                lock_table;
  std::vector<cudaStream_t>         streams;
  std::vector<::q1::Q1AggSlot*>     agg_dev;              // [NUM_GPUS], 256 slots each
  std::vector<uint64_t*>            row_ids_dev;          // identity row_ids cached per GPU
  std::vector<size_t>               row_ids_capacity;
  uint64_t                          sessions_done = 0;
};

// Process-wide singleton. Exposed (non-static) so Q5/Q9/... dispatchers in
// sibling .cu files share the same magi runtime (Endpoint, ChannelRuntime,
// P2P, streams) instead of each query allocating its own ~18GB-per-GPU
// staging — at 4-GPU SF=100 the second per-query runtime wouldn't fit.
// Each query's own agg_dev still lives in that query's dispatcher TU.
MagiState& magi_state()
{
  static MagiState s;
  return s;
}

// ── One-shot init: builds Endpoints, paths, ChannelRuntimes ────────────────
// Shared one-shot magi runtime init. Idempotent; first call from any
// query's dispatcher builds Endpoints/Channels/P2P, subsequent calls
// are no-ops. Per-query state (agg_dev) is allocated separately in the
// caller's dispatcher TU.
void MagiInitOnce()
{
  auto& s = magi_state();
  if (s.initialised) return;

  s.gdr = gdr_open_safe();  // GDR_COPY=0 in our build → returns nullptr (ok)
  s.gpu_ids.resize(NUM_GPUS);
  for (int i = 0; i < NUM_GPUS; ++i) s.gpu_ids[i] = i;

  // Bump device printf buffer so all per-block diagnostic prints survive
  // (default is 1 MB, easy to overrun with 64 blocks × multi-iter prints).
  for (int i = 0; i < NUM_GPUS; ++i) {
    cudaSetDevice(s.gpu_ids[i]);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64ull * 1024 * 1024);
  }

  // Enable P2P peer access between every pair — P2PMemcpyOp's
  // cudaMemcpyPeerAsync silently no-ops (async) when peer access is not
  // enabled, so we'd see transfers_done go up but recv_staging stay
  // empty. Magi's group_by_direct test driver doesn't call this
  // explicitly, presumably because its containing process auto-enables
  // somewhere we don't run inside sirius.
  for (int i = 0; i < NUM_GPUS; ++i) {
    int src = s.gpu_ids[i];
    cudaSetDevice(src);
    for (int j = 0; j < NUM_GPUS; ++j) {
      int dst = s.gpu_ids[j];
      if (src == dst) continue;
      int can = 0;
      cudaDeviceCanAccessPeer(&can, src, dst);
      if (!can) {
        std::printf("[magi-q1] P2P from GPU %d to GPU %d NOT supported\n",
                    src, dst);
        continue;
      }
      cudaError_t er = cudaDeviceEnablePeerAccess(dst, 0);
      if (er == cudaSuccess) {
        std::printf("[magi-q1] P2P enabled: GPU %d -> GPU %d\n", src, dst);
      } else if (er == cudaErrorPeerAccessAlreadyEnabled) {
        std::printf("[magi-q1] P2P already enabled: GPU %d -> GPU %d\n",
                    src, dst);
        cudaGetLastError();  // clear sticky error
      } else {
        std::printf("[magi-q1] P2P enable FAIL %d->%d: %s\n",
                    src, dst, cudaGetErrorString(er));
      }
    }
  }

  // Diagnostic: free / total VRAM per GPU before we start eating ~10 GB
  // of staging buffers. Lets us tell "magi config too big" from "sirius's
  // RMM pool already grabbed the world".
  for (int i = 0; i < NUM_GPUS; ++i) {
    cudaSetDevice(s.gpu_ids[i]);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::printf("[magi-q1] GPU %d before init: free=%.2f GB / total=%.2f GB\n",
                s.gpu_ids[i],
                free_b  / (1024.0 * 1024 * 1024),
                total_b / (1024.0 * 1024 * 1024));
  }

  s.nvlink = std::make_unique<magi::P2PMemcpyOp>();

  s.endpoints.assign(NUM_GPUS, nullptr);
  s.fwd_tables.resize(NUM_GPUS);
  s.streams.resize(NUM_GPUS);
  s.agg_dev.assign(NUM_GPUS, nullptr);
  s.row_ids_dev.assign(NUM_GPUS, nullptr);
  s.row_ids_capacity.assign(NUM_GPUS, 0);

  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = s.gpu_ids[i];
    CHECK_CUDA_ERR(cudaSetDevice(gpu));

    int* fwd_tmp = new int[NUM_GPUS];
    for (int p = 0; p < NUM_GPUS; ++p) {
      s.fwd_tables[gpu][p] = s.gpu_ids[p];
      fwd_tmp[p]           = s.gpu_ids[p];
    }
    s.endpoints[gpu] = new EndpointT(gpu, s.gdr, fwd_tmp, MAX_LOCAL_GPUS);
    delete[] fwd_tmp;

    s.recv_stagings[gpu]    = s.endpoints[gpu]->recv_staging;
    s.recv_connections[gpu] = new Semaphore(MAX_RECV_CONNECTIONS);
    s.lock_table[gpu]       = 0;

    CHECK_CUDA_ERR(
        cudaStreamCreateWithFlags(&s.streams[gpu], cudaStreamNonBlocking));
    CHECK_CUDA_ERR(
        cudaMalloc(&s.agg_dev[gpu],
                   ::q1::Q1_AGG_SLOTS * sizeof(::q1::Q1AggSlot)));
  }

  // Warmup launch (just_load=true) BEFORE the channel runtime starts.
  // q3.cuh / group_by_direct.cuh do this — gets the kernel binary loaded
  // and ptx→sass JIT compiled per-GPU so the first real launch is fast
  // and (more importantly) host_worker_runtime threads aren't blocked
  // waiting on a kernel that's still compiling.
  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = s.gpu_ids[i];
    CHECK_CUDA_ERR(cudaSetDevice(gpu));
    std::printf("[magi-q1] GPU %d warmup launch...\n", gpu);
    fflush(stdout);
    ::q1::q1_kernel<Q1_BLOCK_SIZE,
                    KBUFFERING_INTRA_PARTITION_SIZE,
                    KBUFFERING_INTER_PARTITION_SIZE>
        <<<1, Q1_BLOCK_SIZE, 0, s.streams[gpu]>>>(
            nullptr, 0,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr,
            nullptr, nullptr,
            s.agg_dev[gpu],
            /*just_load=*/true);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
      std::printf("[magi-q1] GPU %d warmup launch FAIL: %s\n",
                  gpu, cudaGetErrorString(le));
      fflush(stdout);
    }
    CHECK_CUDA_ERR(cudaStreamSynchronize(s.streams[gpu]));
    std::printf("[magi-q1] GPU %d warmup done\n", gpu);
    fflush(stdout);
  }

  // ChannelRuntime is non-copyable + non-moveable, so vector<> won't work.
  // Plain heap array of default-constructed ChannelTs.
  s.channels.reset(new ChannelT[NUM_GPUS]);
  s.barrier.num_workers = NUM_GPUS;

  for (int i = 0; i < NUM_GPUS; ++i) {
    for (int dst : s.gpu_ids) {
      constexpr uint64_t BW_NVLINK = 200ULL * 1024 * 1024 * 1024;
      s.channels[i].add_path(
          dst,
          std::make_shared<channel::CopyOpChannel>(
              "nvlink", s.nvlink.get(), BW_NVLINK),
          /*weight=*/10);
    }
  }

  channel::ChannelConfig ch_cfg;
  // Magi's default 8 s timeout is too tight when the runtime lives inside
  // sirius's process: between channel.init() (which starts the host worker
  // and arms the deadlock monitor) and the first q1_kernel launch we still
  // have to do the per-GPU agg reset, build identity row_ids, etc., and
  // when the binary cache is cold the first launch itself takes seconds.
  // 60 s is big enough that only real deadlocks trip it.
  ch_cfg.deadlock_timeout_ms = 60'000;
  ch_cfg.check_interval_ms   = 500;
  ch_cfg.enable_state_dump   = true;
  ch_cfg.abort_on_deadlock   = true;

  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = s.gpu_ids[i];
    s.channels[i].init(gpu,
                       /*core_id=*/10 + gpu,
                       sizeof(::q1::Q1Tuple),
                       s.endpoints[gpu]->send_partitioning,
                       s.recv_stagings,
                       s.recv_connections,
                       s.fwd_tables[gpu],
                       s.lock_table,
                       s.barrier,
                       ch_cfg);
  }

  // Give host worker threads time to enter session 0 before the first
  // real launch (q3.cuh/group_by_direct.cuh both sleep ~1s here).
  std::this_thread::sleep_for(std::chrono::seconds(1));

  s.initialised = true;
  std::printf("[magi-q1] runtime initialised: %d GPUs\n", NUM_GPUS);
}

// ── Identity row_ids[0..n) per GPU, cached and grown on demand ────────────
static uint64_t* GetIdentityRowIds(MagiState& s, int gpu, size_t n)
{
  if (s.row_ids_capacity[gpu] >= n) return s.row_ids_dev[gpu];
  CHECK_CUDA_ERR(cudaSetDevice(gpu));
  if (s.row_ids_dev[gpu]) cudaFree(s.row_ids_dev[gpu]);
  CHECK_CUDA_ERR(cudaMalloc(&s.row_ids_dev[gpu], n * sizeof(uint64_t)));
  std::vector<uint64_t> host_ids(n);
  for (size_t i = 0; i < n; ++i) host_ids[i] = i;
  CHECK_CUDA_ERR(cudaMemcpy(s.row_ids_dev[gpu],
                            host_ids.data(),
                            n * sizeof(uint64_t),
                            cudaMemcpyHostToDevice));
  s.row_ids_capacity[gpu] = n;
  return s.row_ids_dev[gpu];
}

// Non-static wrapper for Q5/Q9/... dispatchers that need the same
// identity row_ids on each GPU (sirius's per-GPU TableScan output is
// already dense, so row_ids[i] = i works for every query).
uint64_t* GetIdentityRowIdsShared(int gpu, size_t n)
{
  return GetIdentityRowIds(magi_state(), gpu, n);
}

// PerGpuInputs / AggResultRow are now declared in magi_q1.hpp (pulled in
// via magi_runtime_shared.hpp at the top). Removed local duplicates.

// ── Run one Q1 query on the given per-GPU partitions ────────────────────────
size_t Q1MagiRun(const std::vector<PerGpuInputs>& inputs,
                 std::vector<AggResultRow>&        out)
{
  MagiInitOnce();
  auto& s = magi_state();

  if (static_cast<int>(inputs.size()) != NUM_GPUS) {
    std::fprintf(stderr,
                 "[magi-q1] inputs.size=%zu, expected %d\n",
                 inputs.size(), NUM_GPUS);
    return 0;
  }

  // Reset per-GPU agg state and bind tuple size for this session.
  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = s.gpu_ids[i];
    CHECK_CUDA_ERR(cudaSetDevice(gpu));
    CHECK_CUDA_ERR(cudaMemset(s.agg_dev[gpu], 0,
                              ::q1::Q1_AGG_SLOTS * sizeof(::q1::Q1AggSlot)));
    s.channels[i].set_tuple_size(sizeof(::q1::Q1Tuple));
  }

  // Per-GPU kernel launch.
  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = s.gpu_ids[i];
    CHECK_CUDA_ERR(cudaSetDevice(gpu));
    const auto& in = inputs[i];
    auto* row_ids  = GetIdentityRowIds(s, gpu, in.n_filtered);
    ::q1::q1_kernel<Q1_BLOCK_SIZE,
                    KBUFFERING_INTRA_PARTITION_SIZE,
                    KBUFFERING_INTER_PARTITION_SIZE>
        <<<USER_KERNEL_GRID_SIZE, Q1_BLOCK_SIZE, 0,
           s.streams[gpu]>>>(row_ids,
                             in.n_filtered,
                             in.d_quantity, in.d_ep, in.d_disc, in.d_tax,
                             in.rf_chars,   in.rf_offsets,
                             in.ls_chars,   in.ls_offsets,
                             s.agg_dev[gpu],
                             /*just_load=*/false);
  }
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA_ERR(cudaSetDevice(s.gpu_ids[i]));
    CHECK_CUDA_ERR(cudaStreamSynchronize(s.streams[s.gpu_ids[i]]));
  }

  // Drain the magi session — wait for the host worker on each GPU to finish
  // flushing/EOF and reset its slot bookkeeping.
  s.sessions_done++;
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA_ERR(cudaSetDevice(s.gpu_ids[i]));
    s.channels[i].sync_after_session(s.endpoints[s.gpu_ids[i]],
                                     s.sessions_done);
  }

  // Copy each GPU's 256-slot agg buffer back, sum across GPUs (a single
  // (rf,ls) tuple should land on exactly one GPU, but summing is safe and
  // avoids encoding any assumption about hash collisions).
  std::vector<::q1::Q1AggSlot> merged(::q1::Q1_AGG_SLOTS, ::q1::Q1AggSlot{});
  std::vector<::q1::Q1AggSlot> host(::q1::Q1_AGG_SLOTS);
  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = s.gpu_ids[i];
    CHECK_CUDA_ERR(cudaSetDevice(gpu));
    CHECK_CUDA_ERR(cudaMemcpy(host.data(),
                              s.agg_dev[gpu],
                              ::q1::Q1_AGG_SLOTS * sizeof(::q1::Q1AggSlot),
                              cudaMemcpyDeviceToHost));
    for (int j = 0; j < ::q1::Q1_AGG_SLOTS; ++j) {
      merged[j].sum_qty        += host[j].sum_qty;
      merged[j].sum_ep         += host[j].sum_ep;
      merged[j].sum_disc       += host[j].sum_disc;
      merged[j].sum_disc_price += host[j].sum_disc_price;
      merged[j].sum_charge     += host[j].sum_charge;
      merged[j].count          += host[j].count;
    }
  }

  out.clear();
  for (int j = 0; j < ::q1::Q1_AGG_SLOTS; ++j) {
    if (merged[j].count == 0) continue;
    out.push_back(AggResultRow{
        /*rf=*/             j >> 8,
        /*ls=*/             j & 0xff,
        /*sum_qty=*/        merged[j].sum_qty,
        /*sum_ep=*/         merged[j].sum_ep,
        /*sum_disc=*/       merged[j].sum_disc,
        /*sum_disc_price=*/ merged[j].sum_disc_price,
        /*sum_charge=*/     merged[j].sum_charge,
        /*count=*/          merged[j].count,
    });
  }
  return out.size();
}

// ── Per-GPU entry: each worker thread runs its own GPU's part ─────────────
// Sirius's legacy executor spawns NUM_GPUS worker threads, one per GPU.
// They all enter Sink in parallel, so the existing serial-from-one-thread
// `Q1MagiRun` can't be reused as-is. We coordinate via a singleton
// std::barrier that gathers inputs, lets all threads launch their own
// GPU's kernel concurrently, and joins on session sync. After return,
// each thread reads only its own GPU's agg_dev — those are already
// hash-partitioned by (rf<<8)|ls % NUM_GPUS, so no cross-GPU merge is
// needed; sirius concatenates each thread's slice for the final output.
struct PerGpuExchange {
  std::barrier<>             begin{NUM_GPUS};
  std::barrier<>             after_session_start{NUM_GPUS};
  std::barrier<>             end{NUM_GPUS};
  std::array<PerGpuInputs, NUM_GPUS> inputs{};
  uint64_t                   session_id = 0;
};
static PerGpuExchange& exchange()
{
  static PerGpuExchange e;
  return e;
}
static std::mutex g_init_mu;

size_t Q1MagiRunPerGpu(int                                gpu_id,
                       const PerGpuInputs&                my_inputs,
                       std::vector<AggResultRow>&         my_slice)
{
  // Init under a mutex so only one thread runs the lazy setup; the rest
  // wait for it to finish before proceeding.
  {
    std::lock_guard<std::mutex> lk(g_init_mu);
    MagiInitOnce();
  }
  auto& s  = magi_state();
  auto& xc = exchange();

  if (gpu_id < 0 || gpu_id >= NUM_GPUS) {
    std::fprintf(stderr,
                 "[magi-q1] Q1MagiRunPerGpu: bad gpu_id=%d (NUM_GPUS=%d)\n",
                 gpu_id, NUM_GPUS);
    return 0;
  }

  // ── Phase timing instrumentation (SIRIUS_MAGI_PROFILE=1 to enable) ─────
  const bool profile = std::getenv("SIRIUS_MAGI_PROFILE") != nullptr;
  auto T0 = std::chrono::high_resolution_clock::now();
  auto tick = [&](const char* tag) {
    if (!profile) return;
    auto now = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(now - T0).count();
    std::printf("[magi-prof gpu=%d] %-32s +%8.1f us\n", gpu_id, tag, us);
    T0 = now;
  };

  // Stash my input + decide the new session id.
  xc.inputs[gpu_id] = my_inputs;
  xc.begin.arrive_and_wait();
  tick("phase B/begin barrier");

  int gpu = s.gpu_ids[gpu_id];
  CHECK_CUDA_ERR(cudaSetDevice(gpu));
  CHECK_CUDA_ERR(cudaMemset(s.agg_dev[gpu], 0,
                            ::q1::Q1_AGG_SLOTS * sizeof(::q1::Q1AggSlot)));
  s.channels[gpu_id].set_tuple_size(sizeof(::q1::Q1Tuple));
  tick("phase C/cudaMemset+set_tuple");

  // Bump the session counter once (thread 0 only) so all threads share
  // the same session_id when calling sync_after_session below.
  if (gpu_id == 0) {
    s.sessions_done++;
    xc.session_id = s.sessions_done;
  }
  xc.after_session_start.arrive_and_wait();
  tick("phase C/session barrier");

  const auto& in = xc.inputs[gpu_id];
  auto*       row_ids = GetIdentityRowIds(s, gpu, in.n_filtered);
  ::q1::q1_kernel<Q1_BLOCK_SIZE,
                  KBUFFERING_INTRA_PARTITION_SIZE,
                  KBUFFERING_INTER_PARTITION_SIZE>
      <<<USER_KERNEL_GRID_SIZE, Q1_BLOCK_SIZE, 0,
         s.streams[gpu]>>>(row_ids,
                           in.n_filtered,
                           in.d_quantity, in.d_ep, in.d_disc, in.d_tax,
                           in.rf_chars,   in.rf_offsets,
                           in.ls_chars,   in.ls_offsets,
                           s.agg_dev[gpu],
                           /*just_load=*/false);
  CHECK_CUDA_ERR(cudaStreamSynchronize(s.streams[gpu]));
  tick("phase D/q1_kernel + streamSync");
  s.channels[gpu_id].sync_after_session(s.endpoints[gpu], xc.session_id);
  tick("phase E/sync_after_session");

  // Read my GPU's agg slice (hash-partitioned, no overlap with peers).
  std::vector<::q1::Q1AggSlot> host(::q1::Q1_AGG_SLOTS);
  CHECK_CUDA_ERR(cudaMemcpy(host.data(),
                            s.agg_dev[gpu],
                            ::q1::Q1_AGG_SLOTS * sizeof(::q1::Q1AggSlot),
                            cudaMemcpyDeviceToHost));
  tick("phase F/agg_dev D2H");
  my_slice.clear();
  for (int j = 0; j < ::q1::Q1_AGG_SLOTS; ++j) {
    if (host[j].count == 0) continue;
    my_slice.push_back(AggResultRow{
        /*rf=*/             j >> 8,
        /*ls=*/             j & 0xff,
        /*sum_qty=*/        host[j].sum_qty,
        /*sum_ep=*/         host[j].sum_ep,
        /*sum_disc=*/       host[j].sum_disc,
        /*sum_disc_price=*/ host[j].sum_disc_price,
        /*sum_charge=*/     host[j].sum_charge,
        /*count=*/          host[j].count,
    });
  }
  tick("phase F/scan non-empty slots");
  xc.end.arrive_and_wait();
  tick("phase G/end barrier");
  return my_slice.size();
}

// ── Shared-runtime accessors (declared in magi_runtime_shared.hpp) ────────
// These let q5_dispatcher.cu / future qN_dispatcher.cu reach into the
// magi runtime singleton without seeing MagiState's full definition or
// the magi template headers.

int magi_phys_gpu(int gpu_id)
{
  auto& s = magi_state();
  return s.gpu_ids[gpu_id];
}

cudaStream_t magi_stream(int gpu_id)
{
  auto& s = magi_state();
  return s.streams[s.gpu_ids[gpu_id]];
}

void magi_set_tuple_size(int gpu_id, std::size_t tuple_bytes)
{
  auto& s = magi_state();
  s.channels[gpu_id].set_tuple_size(tuple_bytes);
}

std::uint64_t magi_bump_session()
{
  auto& s = magi_state();
  return ++s.sessions_done;
}

void magi_sync_after_session(int gpu_id, std::uint64_t session_id)
{
  auto& s = magi_state();
  int gpu = s.gpu_ids[gpu_id];
  s.channels[gpu_id].sync_after_session(s.endpoints[gpu], session_id);
}

}  // namespace magi_q1
}  // namespace duckdb
