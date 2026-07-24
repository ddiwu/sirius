// magi_runtime.cu — sirius-side shared Magi runtime for the cross-GPU
// (NVLink-shuffle) TPC-H query backend.
//
// Lazily initialises the Magi runtime inside sirius's process: per-GPU host
// worker threads, Endpoints, a ChannelRuntime per GPU, and NVLink P2P paths
// between every pair of GPUs. The runtime is started once on the first magi
// query and reused across queries; per-query agg state lives in each query's
// own dispatcher TU (magi_groupby_runtime.cu / magi_join_runtime.cu).
//
// All Magi state is held in a translation-unit-local singleton MagiState,
// exposed through the accessors declared in magi_runtime_shared.hpp so the
// per-query dispatchers can drive the shared Endpoints/Channels without
// seeing MagiState's full definition or the heavy magi template headers.
//
// This runtime is query-agnostic: it carries no Q1/Q5/... specific kernels or
// tuple layouts. Data moves over the channel in fixed CELL_SIZE (64 B) units
// via the cell API; the per-session tuple size is set at runtime by each
// dispatcher (magi_set_tuple_size).
//
// Build is gated by ENABLE_MAGI_TPCH; this TU is not compiled when off.

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

// Magi data-plane headers. core.cuh + kbuffering.cuh must be included before
// channel_runtime.cuh: the channel runtime references KBuffering / Semaphore /
// SessionBarrier / WorkerProgress / the host-worker runtime / MAGI_EOF etc.
// without including their definitions itself — q1.cuh used to satisfy this
// transitively before the runtime was made query-agnostic.
#include "data_plane/api/core.cuh"
#include "data_plane/infra/endpoint.cuh"
#include "data_plane/infra/kbuffering.cuh"
#include "data_plane/channel/channel_runtime.cuh"
#include "data_plane/infra/semaphore.cuh"
#include "data_plane/transport/nvlink_path.cuh"
#include "data_plane/config.cuh"

// ── Magi runtime sizing constants + wire-cell type ─────────────────────────
// NUM_GPUS matches sirius's `kSiriusLegacyNumGpus` (declared in
// magi_runtime_shared.hpp). PARTITIONS_COUNT is a compile-time template
// parameter to Endpoint.
// Arena preallocators implemented in the per-query dispatcher TUs (their
// arena statics are file-local); forward-declared to avoid the heavy headers.
namespace duckdb { namespace magi_generic {
void magi_groupby_prealloc_arenas();
void magi_join_prealloc_arenas();
}}  // namespace duckdb::magi_generic

namespace duckdb {
namespace magi_runtime {

constexpr int PARTITIONS_COUNT = NUM_GPUS;

// Neutral 64-byte wire-cell type for the Endpoint/Channel. The shared magi
// runtime moves data in CELL_SIZE (= 64 B) units via the cell API; the
// actual per-session tuple size is set at runtime via magi_set_tuple_size.
// Endpoint's tuple template parameter is only used for compile-time buffer
// sizing (sizeof must be a multiple of CELL_SIZE) and is otherwise unused —
// so a plain 64-byte POD decouples the shared runtime from any particular
// query's tuple layout.
struct MagiWireCell {
  unsigned char bytes[64];
};
static_assert(sizeof(MagiWireCell) == 64,
              "MagiWireCell must equal the magi cell size (64 bytes)");

using EndpointT = ::Endpoint<1, 1, PARTITIONS_COUNT,
                             USER_KERNEL_GRID_SIZE, USER_KERNEL_GRID_SIZE,
                             K_BUFFERING_K,
                             KBUFFERING_INTRA_PARTITION_SIZE,
                             KBUFFERING_INTER_PARTITION_SIZE,
                             MagiWireCell>;
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
  std::vector<uint64_t*>            row_ids_dev;          // identity row_ids cached per GPU
  std::vector<size_t>               row_ids_capacity;
  uint64_t                          sessions_done = 0;
};

// Process-wide singleton. Shared by every query dispatcher (groupby/join) so
// they reuse the same magi runtime (Endpoint, ChannelRuntime, P2P, streams)
// instead of each allocating its own ~18 GB-per-GPU staging — at 4-GPU
// SF=100 a second per-query runtime wouldn't fit. Each query's own agg
// buffers live in that query's dispatcher TU.
MagiState& magi_state()
{
  static MagiState s;
  return s;
}

// ── One-shot init: builds Endpoints, paths, ChannelRuntimes ────────────────
// Idempotent; the first call from any query's dispatcher builds the
// Endpoints/Channels/P2P paths, subsequent calls are no-ops. Per-query state
// is allocated separately in the caller's dispatcher TU.
//
// Thread-safe: the per-GPU sirius worker threads all call this on the first
// magi query; internal locking lets the groupby/join dispatchers call it
// without each constructing its own Endpoint set under a race.
void MagiInitOnce()
{
  static std::mutex s_init_mu;
  std::lock_guard<std::mutex> lk(s_init_mu);
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
  // enabled, so we'd see transfers_done go up but recv_staging stay empty.
  for (int i = 0; i < NUM_GPUS; ++i) {
    int src = s.gpu_ids[i];
    cudaSetDevice(src);
    for (int j = 0; j < NUM_GPUS; ++j) {
      int dst = s.gpu_ids[j];
      if (src == dst) continue;
      int can = 0;
      cudaDeviceCanAccessPeer(&can, src, dst);
      if (!can) {
        std::printf("[magi] P2P from GPU %d to GPU %d NOT supported\n",
                    src, dst);
        continue;
      }
      cudaError_t er = cudaDeviceEnablePeerAccess(dst, 0);
      if (er == cudaSuccess) {
        std::printf("[magi] P2P enabled: GPU %d -> GPU %d\n", src, dst);
      } else if (er == cudaErrorPeerAccessAlreadyEnabled) {
        std::printf("[magi] P2P already enabled: GPU %d -> GPU %d\n",
                    src, dst);
        cudaGetLastError();  // clear sticky error
      } else {
        std::printf("[magi] P2P enable FAIL %d->%d: %s\n",
                    src, dst, cudaGetErrorString(er));
      }
    }
  }

  // Diagnostic: free / total VRAM per GPU before we start eating ~10 GB of
  // staging buffers. Lets us tell "magi config too big" from "sirius's RMM
  // pool already grabbed the world".
  for (int i = 0; i < NUM_GPUS; ++i) {
    cudaSetDevice(s.gpu_ids[i]);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::printf("[magi] GPU %d before init: free=%.2f GB / total=%.2f GB\n",
                s.gpu_ids[i],
                free_b  / (1024.0 * 1024 * 1024),
                total_b / (1024.0 * 1024 * 1024));
  }

  s.nvlink = std::make_unique<magi::P2PMemcpyOp>();

  s.endpoints.assign(NUM_GPUS, nullptr);
  s.fwd_tables.resize(NUM_GPUS);
  s.streams.resize(NUM_GPUS);
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
  // and arms the deadlock monitor) and the first real kernel launch we still
  // have to do the per-GPU agg reset, build identity row_ids, etc., and when
  // the binary cache is cold the first launch itself takes seconds. 60 s is
  // big enough that only real deadlocks trip it.
  ch_cfg.deadlock_timeout_ms = 60'000;
  ch_cfg.check_interval_ms   = 500;
  ch_cfg.enable_state_dump   = true;
  ch_cfg.abort_on_deadlock   = true;

  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = s.gpu_ids[i];
    s.channels[i].init(gpu,
                       /*core_id=*/10 + gpu,
                       sizeof(MagiWireCell),
                       s.endpoints[gpu]->send_partitioning,
                       s.recv_stagings,
                       s.recv_connections,
                       s.fwd_tables[gpu],
                       s.lock_table,
                       s.barrier,
                       ch_cfg);
  }

  // Give host worker threads time to enter session 0 before the first real
  // launch (q3.cuh/group_by_direct.cuh both sleep ~1 s here).
  std::this_thread::sleep_for(std::chrono::seconds(1));

  s.initialised = true;
  std::printf("[magi] runtime initialised: %d GPUs\n", NUM_GPUS);
}

// ── Identity row_ids[0..n) per GPU, cached and grown on demand ────────────
__global__ static void k_iota_u64(uint64_t* p, size_t n)
{
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = i;
}

static uint64_t* GetIdentityRowIds(MagiState& s, int gpu, size_t n)
{
  if (s.row_ids_capacity[gpu] >= n) return s.row_ids_dev[gpu];
  CHECK_CUDA_ERR(cudaSetDevice(gpu));
  if (s.row_ids_dev[gpu]) cudaFree(s.row_ids_dev[gpu]);
  CHECK_CUDA_ERR(cudaMalloc(&s.row_ids_dev[gpu], n * sizeof(uint64_t)));
  // Device-side fill. The old host loop + pageable H2D cost ~74ms for a
  // 37.5M-row first touch (Q18); the kernel is ~0.2ms.
  constexpr int TPB = 256;
  const unsigned grid = static_cast<unsigned>((n + TPB - 1) / TPB);
  k_iota_u64<<<grid, TPB>>>(s.row_ids_dev[gpu], n);
  CHECK_CUDA_ERR(cudaDeviceSynchronize());
  s.row_ids_capacity[gpu] = n;
  return s.row_ids_dev[gpu];
}

// Non-static wrapper for the per-query dispatchers that need identity
// row_ids on each GPU (sirius's per-GPU TableScan output is already dense,
// so row_ids[i] = i works for every query).
uint64_t* GetIdentityRowIdsShared(int gpu, size_t n)
{
  return GetIdentityRowIds(magi_state(), gpu, n);
}

// ── Shared-runtime accessors (declared in magi_runtime_shared.hpp) ────────
// These let the groupby/join dispatchers reach into the magi runtime
// singleton without seeing MagiState's full definition or the magi template
// headers.

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

std::uint8_t* magi_pool_alloc(std::size_t bytes, int gpu_id, bool persistent)
{
  auto& mgr = GPUBufferManager::GetInstance();
  return mgr.customCudaMalloc<std::uint8_t>(bytes, gpu_id, persistent);
}

void magi_prealloc_arenas()
{
  duckdb::magi_generic::magi_groupby_prealloc_arenas();
  duckdb::magi_generic::magi_join_prealloc_arenas();
}

}  // namespace magi_runtime
}  // namespace duckdb
