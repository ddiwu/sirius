// magi_join_runtime.cu — generic magi shuffle-join runtime (host entry +
// explicit kernel instantiations). The analog of magi_groupby_runtime.cu.
//
// Two sessions per query, both over the magi NVLink channel:
//   BUILD : pack build rows -> JoinBuildWire[] -> join_build_kernel
//           (send_direct by join key -> owner global_find_or_insert into H_build)
//   PROBE : pack probe rows -> JoinProbeWire[] -> join_probe_kernel
//           (send_direct by join key -> owner global_find + emit JoinResultRow)
// A cross-GPU barrier between the two sessions guarantees every owner's H_build
// is fully built before any GPU probes (hash-join build-before-probe semantics).
//
// v1 scope: INNER join, UNIQUE build key, key kinds {int32,uint64}, tiers
// {SMALL,MEDIUM,LARGE}. H_build is sized to the LARGE tier.

#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "data_plane/config.cuh"   // KBUFFERING_INTRA/INTER_PARTITION_SIZE, USER_KERNEL_GRID_SIZE

#include "legacy/operator/magi_join_stages.cuh"
#include "legacy/operator/magi_distributed_join.hpp"
#include "legacy/operator/magi_distributed_groupby.hpp"  // N_SLOTS_* tiers
#include "legacy/operator/magi_runtime_shared.hpp"

// ── Explicit kernel instantiations (build + probe) ─────────────────────────
namespace magi {

#define MAGI_JOIN_INST_PACK_BUILD(KEY_TYPE)                                     \
  template __global__ void join_pack_build_kernel<KEY_TYPE>(                    \
      const std::uint64_t*, std::uint64_t, magi_ops::ColPack,                  \
      const magi_ops::KeyFieldEntry*, int,                                      \
      const duckdb::magi_generic::JoinPayloadEntry*, int,                       \
      duckdb::magi_join::JoinBuildWire*, unsigned int*)

#define MAGI_JOIN_INST_PACK_PROBE(KEY_TYPE)                                     \
  template __global__ void join_pack_probe_kernel<KEY_TYPE>(                    \
      const std::uint64_t*, std::uint64_t, magi_ops::ColPack,                  \
      const magi_ops::KeyFieldEntry*, int,                                      \
      const duckdb::magi_generic::JoinPayloadEntry*, int,                       \
      duckdb::magi_join::JoinProbeWire*, unsigned int*)

#define MAGI_JOIN_INST_BUILD(KEY_TYPE, N_SLOTS)                                 \
  template __global__ void                                                      \
  join_build_kernel<KEY_TYPE, N_SLOTS,                                          \
                    KBUFFERING_INTRA_PARTITION_SIZE,                            \
                    KBUFFERING_INTER_PARTITION_SIZE>(                           \
      const duckdb::magi_join::JoinBuildWire*, std::uint64_t,                   \
      magi_ops::AggSlot64<KEY_TYPE>*, int, bool, unsigned int*,                 \
      duckdb::magi_join::WirePackSrc)

#define MAGI_JOIN_INST_PROBE(KEY_TYPE, N_SLOTS)                                 \
  template __global__ void                                                      \
  join_probe_kernel<KEY_TYPE, N_SLOTS,                                          \
                    KBUFFERING_INTRA_PARTITION_SIZE,                            \
                    KBUFFERING_INTER_PARTITION_SIZE>(                           \
      const duckdb::magi_join::JoinProbeWire*, std::uint64_t,                   \
      magi_ops::AggSlot64<KEY_TYPE>*,                                           \
      duckdb::magi_generic::JoinResultRow*, unsigned int*, unsigned int, int,   \
      bool, unsigned int*, unsigned int*,                                       \
      double*, const duckdb::magi_fused::FusedAggProgram*,                      \
      const duckdb::magi_join::CucoFindRef*,                                    \
      magi_ops::AggSlot64<std::uint64_t>*, unsigned int*, int,                  \
      duckdb::magi_join::WirePackSrc)

#define MAGI_JOIN_INST_ALL(KEY_TYPE)                                            \
  MAGI_JOIN_INST_PACK_BUILD(KEY_TYPE);                                          \
  MAGI_JOIN_INST_PACK_PROBE(KEY_TYPE);                                          \
  MAGI_JOIN_INST_BUILD(KEY_TYPE, duckdb::magi_generic::N_SLOTS_SMALL);          \
  MAGI_JOIN_INST_BUILD(KEY_TYPE, duckdb::magi_generic::N_SLOTS_MEDIUM);         \
  MAGI_JOIN_INST_BUILD(KEY_TYPE, duckdb::magi_generic::N_SLOTS_LARGE);          \
  MAGI_JOIN_INST_BUILD(KEY_TYPE, duckdb::magi_generic::N_SLOTS_XLARGE);         \
  MAGI_JOIN_INST_BUILD(KEY_TYPE, duckdb::magi_generic::N_SLOTS_XXLARGE);        \
  MAGI_JOIN_INST_PROBE(KEY_TYPE, duckdb::magi_generic::N_SLOTS_SMALL);          \
  MAGI_JOIN_INST_PROBE(KEY_TYPE, duckdb::magi_generic::N_SLOTS_MEDIUM);         \
  MAGI_JOIN_INST_PROBE(KEY_TYPE, duckdb::magi_generic::N_SLOTS_LARGE);          \
  MAGI_JOIN_INST_PROBE(KEY_TYPE, duckdb::magi_generic::N_SLOTS_XLARGE);         \
  MAGI_JOIN_INST_PROBE(KEY_TYPE, duckdb::magi_generic::N_SLOTS_XXLARGE)

MAGI_JOIN_INST_ALL(std::int32_t);
MAGI_JOIN_INST_ALL(std::uint64_t);
MAGI_JOIN_INST_ALL(unsigned __int128);

// Clear an H_build tier prefix to the empty-key sentinel (memset to 0 mis-marks
// int32 slots, whose empty key is -1).
template <typename KeyT>
__global__ void join_clear_hbuild(magi_ops::AggSlot64<KeyT>* H, int n_slots)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_slots) { H[i] = magi_ops::AggSlot64<KeyT>{}; H[i].key = magi_ops::empty_key_v<KeyT>; }
}
template __global__ void join_clear_hbuild<std::int32_t>(magi_ops::AggSlot64<std::int32_t>*, int);
template __global__ void join_clear_hbuild<std::uint64_t>(magi_ops::AggSlot64<std::uint64_t>*, int);
template __global__ void join_clear_hbuild<unsigned __int128>(magi_ops::AggSlot64<unsigned __int128>*, int);

}  // namespace magi

namespace duckdb {

namespace magi_fused {
// Singleton side channel (Option 3): the fused join publishes per-GPU partial
// sums here; the result collector reduces them + applies the top projection.
FusedResultChannel& fused_channel() { static FusedResultChannel c{}; return c; }
}  // namespace magi_fused

namespace magi_generic {

// ── Per-GPU persistent buffers ──────────────────────────────────────────────
// H_build sized to the XLARGE tier (16M slots × 64B = 1 GB) so realistic
// large-large joins (e.g. lineitem⋈part, ~10M distinct partkey) fit. Wire/output
// buffers are per-query (sizes vary widely) — allocated/freed inside the run.
constexpr int    JOIN_MAX_TIER_SLOTS = N_SLOTS_XXLARGE;  // 64M slots × 64B = 4GB/GPU arena
constexpr size_t JOIN_HBUILD_BYTES =
    sizeof(magi_ops::AggSlot64<std::uint64_t>) * JOIN_MAX_TIER_SLOTS;
constexpr int    JOIN_BLOCK = 1024;

// Grid size for the join's user kernels. MAGI_USER_GRID can only lower it:
// USER_KERNEL_GRID_SIZE is baked into the channel's data structures — kbuffering
// keeps `size_t write_slots[USER_KERNEL_GRID_SIZE]` / `read_slots[...]` indexed
// by blockIdx, and endpoint allocates KBUFFERING_K * USER_KERNEL_GRID_SIZE
// staging buffers. Launching MORE blocks than the constant makes blocks past the
// end scribble out of bounds (measured: grid 96/128 segfault, 132 hangs). To
// raise the grid, raise the constant in Magi-Dev/include/data_plane/config.cuh
// so the structures grow with it.
static inline int magi_user_grid()
{
  static const int g = [] {
    const char* e = std::getenv("MAGI_USER_GRID");
    const int   v = e ? std::atoi(e) : 0;
    const int   c = static_cast<int>(USER_KERNEL_GRID_SIZE);
    return (v > 0 && v < c) ? v : c;
  }();
  return g;
}

static std::byte*               g_hbuild_dev   [NUM_GPUS] = { nullptr };
static magi_ops::KeyFieldEntry* g_jkfields_dev [NUM_GPUS] = { nullptr };
static JoinPayloadEntry*        g_jppl_dev     [NUM_GPUS] = { nullptr };  // probe payload table
static JoinPayloadEntry*        g_jbpl_dev     [NUM_GPUS] = { nullptr };  // build payload table
static unsigned int*            g_joverflow_dev[NUM_GPUS] = { nullptr };
static std::mutex               g_jinit_mu;

static constexpr int JOIN_MAX_KEY_FIELDS = 8;
static constexpr int JOIN_MAX_PAYLOAD_ENTRIES = 8;

static void EnsureJoinBuffers()
{
  std::lock_guard<std::mutex> lk(g_jinit_mu);
  if (g_hbuild_dev[0] != nullptr) return;
  // Persistent arenas from sirius's CACHE pool (see EnsureDeviceBuffers in
  // magi_groupby_runtime.cu for the rationale — raw cudaMalloc here competed
  // for the sliver of memory left outside the pools and crashed at SF100).
  // gpu_buffer_init preallocates via magi_join_prealloc_arenas() while the
  // cache bump pointer is still 0.
  for (int i = 0; i < NUM_GPUS; ++i) {
    g_hbuild_dev[i] = reinterpret_cast<std::byte*>(
      magi_runtime::magi_pool_alloc(JOIN_HBUILD_BYTES, i, /*persistent=*/true));
    g_jkfields_dev[i] = reinterpret_cast<magi_ops::KeyFieldEntry*>(
      magi_runtime::magi_pool_alloc(sizeof(magi_ops::KeyFieldEntry) * JOIN_MAX_KEY_FIELDS, i, true));
    g_jppl_dev[i] = reinterpret_cast<JoinPayloadEntry*>(
      magi_runtime::magi_pool_alloc(sizeof(JoinPayloadEntry) * JOIN_MAX_PAYLOAD_ENTRIES, i, true));
    g_jbpl_dev[i] = reinterpret_cast<JoinPayloadEntry*>(
      magi_runtime::magi_pool_alloc(sizeof(JoinPayloadEntry) * JOIN_MAX_PAYLOAD_ENTRIES, i, true));
    g_joverflow_dev[i] = reinterpret_cast<unsigned int*>(
      magi_runtime::magi_pool_alloc(sizeof(unsigned int), i, true));
  }
}

// Eager arena carve-out, called from gpu_buffer_init (via
// magi_runtime::magi_prealloc_arenas) before any table is cached.
void magi_join_prealloc_arenas() { EnsureJoinBuffers(); }

// ── Per-GPU barrier exchange ────────────────────────────────────────────────
struct JoinExchange {
  std::barrier<> begin{NUM_GPUS};
  std::barrier<> alloc_check{NUM_GPUS};
  std::barrier<> build_start{NUM_GPUS};
  std::barrier<> after_build{NUM_GPUS};
  std::barrier<> probe_start{NUM_GPUS};
  std::barrier<> end{NUM_GPUS};

  // Per-GPU verdict of the pre-flight wire-buffer memory check. Published
  // before build_start so every worker throws symmetrically (no stranded peer).
  std::array<int, NUM_GPUS>                             alloc_failed{};

  std::array<PerGpuJoinInputs, NUM_GPUS>                 build_in{};
  std::array<PerGpuJoinInputs, NUM_GPUS>                 probe_in{};
  std::array<const magi_ops::KeyFieldEntry*, NUM_GPUS>   kf_ptr{};
  std::array<int, NUM_GPUS>                              n_kf{};
  std::array<const JoinPayloadEntry*, NUM_GPUS>          ppl_ptr{};
  std::array<int, NUM_GPUS>                              n_ppl{};
  std::array<const JoinPayloadEntry*, NUM_GPUS>          bpl_ptr{};
  std::array<int, NUM_GPUS>                              n_bpl{};
  std::array<JoinResultRow*, NUM_GPUS>                   out_buf{};        // wrapper-allocated (pool)
  std::array<unsigned int*, NUM_GPUS>                    out_count_buf{};
  std::array<std::uint64_t, NUM_GPUS>                    out_cap{};
  KeyKind        key_kind   = KeyKind::INT32;
  TableSize      table_size = TableSize::SMALL;
  std::uint64_t  build_session = 0;
  std::uint64_t  probe_session = 0;
  std::uint64_t  total_probe   = 0;   // sum of probe n_rows → output cap
};
static JoinExchange& jexchange() { static JoinExchange e; return e; }

// ── Per-GPU run (templated on KeyT and tier) ────────────────────────────────
template <typename KeyT, int N_SLOTS>
static std::size_t join_run_typed_tier(int gpu_id, std::vector<JoinResultRow>& my_slice,
                                       bool device_emit,
                                       const duckdb::magi_fused::FusedAggProgram* agg_prog_host)
{
  auto&             xc  = jexchange();
  const int         gpu = magi_runtime::magi_phys_gpu(gpu_id);
  cudaSetDevice(gpu);

  // Coarse per-phase wall timing, enabled by MAGI_PHASE_TIME=1 (one fprintf
  // per GPU per query at the end; no hot-loop instrumentation).
  const bool phase_time = std::getenv("MAGI_PHASE_TIME") != nullptr;
  using pt_clock = std::chrono::steady_clock;
  auto pt_t0 = pt_clock::now();
  auto pt_ms = [](pt_clock::time_point a, pt_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  cudaStream_t      st  = magi_runtime::magi_stream(gpu_id);
  auto*             H   = reinterpret_cast<magi_ops::AggSlot64<KeyT>*>(g_hbuild_dev[gpu_id]);

  const bool DBG = std::getenv("MAGI_JOIN_DEBUG") != nullptr;
  auto chk = [&](const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
      std::fprintf(stderr, "[magi-join g%d tier=%d] CUDA ERR after %s: %s\n",
                   gpu_id, N_SLOTS, what, cudaGetErrorString(e));
  };

  // Wire buffers for BOTH phases, from sirius's PROCESSING pool (unified
  // accounting; reclaimed automatically by the end-of-query ResetBuffer, so
  // no explicit frees). Allocation happens up front and the per-GPU verdict
  // is exchanged BEFORE any collective work: pool exhaustion on one GPU must
  // throw on EVERY GPU (a one-sided throw strands the peer at build_start).
  // The old raw cudaMalloc was unchecked and at SF100 handed the pack kernel
  // a garbage pointer -> illegal memory access -> process crash.
  magi_join::JoinBuildWire* bw = nullptr;
  unsigned int*             bw_count = nullptr;
  magi_join::JoinProbeWire* pw = nullptr;
  unsigned int*             pw_count = nullptr;
  unsigned int*             out_count_dev = nullptr;
  // Fused packing: the send loop packs each chunk itself, so the full-size wire
  // arrays (n_rows * 64B — 9.6 GB/GPU for SF50's probe side) collapse to a
  // per-block staging area of grid * FUSE_PACK_CHUNK tuples, and the pack
  // kernels' ~17 ms of dead time before the shuffle disappears.
  static const bool fuse_pack = std::getenv("MAGI_FUSE_PACK") != nullptr;
  constexpr int     FUSE_PACK_CHUNK = 64 * JOIN_BLOCK;  // matches the old send chunk
  std::uint8_t*     bpack_scratch = nullptr;
  std::uint8_t*     ppack_scratch = nullptr;
  {
    bool ok = true;
    try {
      const std::uint64_t bn0 = std::max<std::uint64_t>(xc.build_in[gpu_id].n_rows, 1);
      const std::uint64_t pn0 = std::max<std::uint64_t>(xc.probe_in[gpu_id].n_rows, 1);
      if (fuse_pack) {
        const std::size_t bsz = sizeof(magi_join::JoinBuildWire) *
                                static_cast<std::size_t>(magi_user_grid()) * FUSE_PACK_CHUNK;
        const std::size_t psz = sizeof(magi_join::JoinProbeWire) *
                                static_cast<std::size_t>(magi_user_grid()) * FUSE_PACK_CHUNK;
        bpack_scratch = magi_runtime::magi_pool_alloc(bsz, gpu_id, false);
        ppack_scratch = magi_runtime::magi_pool_alloc(psz, gpu_id, false);
      } else {
        bw = reinterpret_cast<magi_join::JoinBuildWire*>(
          magi_runtime::magi_pool_alloc(sizeof(magi_join::JoinBuildWire) * bn0, gpu_id, false));
        pw = reinterpret_cast<magi_join::JoinProbeWire*>(
          magi_runtime::magi_pool_alloc(sizeof(magi_join::JoinProbeWire) * pn0, gpu_id, false));
      }
      bw_count = reinterpret_cast<unsigned int*>(
        magi_runtime::magi_pool_alloc(sizeof(unsigned int), gpu_id, false));
      pw_count = reinterpret_cast<unsigned int*>(
        magi_runtime::magi_pool_alloc(sizeof(unsigned int), gpu_id, false));
      out_count_dev = reinterpret_cast<unsigned int*>(
        magi_runtime::magi_pool_alloc(sizeof(unsigned int), gpu_id, false));
    } catch (...) { ok = false; }
    xc.alloc_failed[gpu_id] = ok ? 0 : 1;
    xc.alloc_check.arrive_and_wait();
    bool any = false;
    for (int i = 0; i < NUM_GPUS; ++i) any = any || xc.alloc_failed[i];
    if (any)
      throw std::runtime_error(
          "magi shuffle join: wire buffers exceed the processing pool "
          "(falls back to DuckDB)");
  }

  // Clear H_build prefix + push the per-query key_fields / probe-payload tables.
  const int clear_grid = (N_SLOTS + 255) / 256;
  magi::join_clear_hbuild<KeyT><<<clear_grid, 256, 0, st>>>(H, N_SLOTS);
  cudaMemcpyAsync(g_jkfields_dev[gpu_id], xc.kf_ptr[gpu_id],
                  sizeof(magi_ops::KeyFieldEntry) * xc.n_kf[gpu_id],
                  cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(g_jppl_dev[gpu_id], xc.ppl_ptr[gpu_id],
                  sizeof(JoinPayloadEntry) * xc.n_ppl[gpu_id],
                  cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(g_jbpl_dev[gpu_id], xc.bpl_ptr[gpu_id],
                  sizeof(JoinPayloadEntry) * xc.n_bpl[gpu_id],
                  cudaMemcpyHostToDevice, st);

  // BUILD payload SLOT count (a VARCHAR spans VARCHAR_PAYLOAD_SLOTS): the pack
  // kernel iterates ENTRIES (xc.n_bpl), but BuildInsert / ProbeEmit copy SLOTS
  // into H_build->values[] / JoinResultRow::build_values[].
  int n_bpl_slots = 0;
  for (int i = 0; i < xc.n_bpl[gpu_id]; ++i)
    n_bpl_slots += (xc.bpl_ptr[gpu_id][i].src_kind == JoinPayloadEntry::Src::VARCHAR)
                       ? VARCHAR_PAYLOAD_SLOTS : 1;

  // ── BUILD: pack rows → wire tuples ────────────────────────────────────────
  const std::uint64_t bn = xc.build_in[gpu_id].n_rows;
  cudaMemsetAsync(bw_count, 0, sizeof(unsigned int), st);
  cudaMemsetAsync(g_joverflow_dev[gpu_id], 0, sizeof(unsigned int), st);
  std::uint64_t* brow = magi_runtime::GetIdentityRowIdsShared(gpu, bn);
  cudaDeviceSynchronize();  // make this GPU's cached input coherent (cf. groupby)
  magi_join::WirePackSrc bsrc{};
  if (fuse_pack) {
    bsrc.row_ids      = brow;
    bsrc.n_rows       = bn;
    bsrc.cols         = xc.build_in[gpu_id].cols;
    bsrc.key_fields   = g_jkfields_dev[gpu_id];
    bsrc.pl           = g_jbpl_dev[gpu_id];
    bsrc.scratch      = bpack_scratch;
    bsrc.n_key_fields = xc.n_kf[gpu_id];
    bsrc.n_pl         = xc.n_bpl[gpu_id];
    bsrc.chunk        = FUSE_PACK_CHUNK;
  } else {
    magi::join_pack_build_kernel<KeyT><<<magi_user_grid(), JOIN_BLOCK, 0, st>>>(
        brow, bn, xc.build_in[gpu_id].cols, g_jkfields_dev[gpu_id], xc.n_kf[gpu_id],
        g_jbpl_dev[gpu_id], xc.n_bpl[gpu_id], bw, bw_count);
    cudaStreamSynchronize(st);
    chk("pack_build");
  }
  auto pt_t1 = pt_clock::now();
  unsigned int bw_n = 0;
  if (!fuse_pack) cudaMemcpy(&bw_n, bw_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  if (DBG) std::fprintf(stderr, "[magi-join g%d] bn=%lu bw_n=%u n_bpl=%d n_ppl=%d\n",
                        gpu_id, (unsigned long)bn, bw_n, xc.n_bpl[gpu_id], xc.n_ppl[gpu_id]);

  // ── BUILD session: shuffle wire tuples → owners insert into H_build ───────
  magi_runtime::magi_set_tuple_size(gpu_id, sizeof(magi_join::JoinBuildWire));
  if (gpu_id == 0) xc.build_session = magi_runtime::magi_bump_session();
  xc.build_start.arrive_and_wait();
  auto pt_t2 = pt_clock::now();
  magi::join_build_kernel<KeyT, N_SLOTS,
                          KBUFFERING_INTRA_PARTITION_SIZE,
                          KBUFFERING_INTER_PARTITION_SIZE>
      <<<magi_user_grid(), JOIN_BLOCK, 0, st>>>(
          bw, bw_n, H, n_bpl_slots, /*just_load=*/false, g_joverflow_dev[gpu_id], bsrc);
  cudaStreamSynchronize(st);
  chk("build_kernel");
  auto pt_t3 = pt_clock::now();
  magi_runtime::magi_sync_after_session(gpu_id, xc.build_session);
  xc.after_build.arrive_and_wait();   // every owner's H_build is complete
  auto pt_t4 = pt_clock::now();

  // ── Option A: build a cuco map (packed uint64 key -> H_build slot index) so
  // the probe does a cudf-quality find instead of the hand-rolled open-address
  // global_find. Only the uint64 key path (cuco caps keys at 8B). Gated by
  // MAGI_CUCO. The map holds every owned build key; the value indexes back into
  // H_build so the existing emit path (slot->values) is unchanged.
  const bool use_cuco = std::getenv("MAGI_CUCO") != nullptr &&
                        (std::is_same<KeyT, std::uint64_t>::value ||
                         std::is_same<KeyT, unsigned __int128>::value);  // u128 packs to uint64
  std::unique_ptr<magi_join::CucoMap> cuco_map;
  magi_join::CucoFindRef*             d_find_ref = nullptr;
  double                              cuco_build_ms = 0.0;
  if (use_cuco) {
    auto cb0 = pt_clock::now();
    std::uint64_t total_build = 0;
    for (int i = 0; i < NUM_GPUS; ++i) total_build += xc.build_in[i].n_rows;
    const std::size_t cap = static_cast<std::size_t>(total_build) * 3 / 2 + 1024;
    cuco_map = std::make_unique<magi_join::CucoMap>(
        cap, cuco::empty_key<std::uint64_t>{0xFFFFFFFFFFFFFFFFULL},
        cuco::empty_value<std::int32_t>{-1});
    cudaDeviceSynchronize();  // map storage alloc (default stream) before insert on st
    auto ins_ref = cuco_map->ref(cuco::insert);
    const int g = (N_SLOTS + 255) / 256;
    magi::cuco_insert_from_hbuild<KeyT, N_SLOTS><<<g, 256, 0, st>>>(ins_ref, H);
    cudaStreamSynchronize(st);
    chk("cuco_insert");
    auto find_ref = cuco_map->ref(cuco::find);
    cudaMalloc(reinterpret_cast<void**>(&d_find_ref), sizeof(find_ref));
    cudaMemcpy(d_find_ref, &find_ref, sizeof(find_ref), cudaMemcpyHostToDevice);
    cuco_build_ms = pt_ms(cb0, pt_clock::now());
    if (DBG) std::fprintf(stderr, "[magi-join g%d] cuco build=%.2fms cap=%zu\n",
                          gpu_id, cuco_build_ms, cap);
  }

  // ── PROBE: pack rows → wire tuples ────────────────────────────────────────
  const std::uint64_t pn = xc.probe_in[gpu_id].n_rows;
  cudaMemsetAsync(pw_count, 0, sizeof(unsigned int), st);
  std::uint64_t* prow = magi_runtime::GetIdentityRowIdsShared(gpu, pn);
  // Make this GPU's cross-device-uploaded probe slice coherent before the pack
  // kernel reads it — same fix the build phase (above) and the groupby runtime
  // apply. Without it, a consuming GPU != 0 reads stale probe keys/payload →
  // nondeterministic wrong join results.
  cudaDeviceSynchronize();
  magi_join::WirePackSrc psrc{};
  if (fuse_pack) {
    psrc.row_ids      = prow;
    psrc.n_rows       = pn;
    psrc.cols         = xc.probe_in[gpu_id].cols;
    psrc.key_fields   = g_jkfields_dev[gpu_id];
    psrc.pl           = g_jppl_dev[gpu_id];
    psrc.scratch      = ppack_scratch;
    psrc.n_key_fields = xc.n_kf[gpu_id];
    psrc.n_pl         = xc.n_ppl[gpu_id];
    psrc.chunk        = FUSE_PACK_CHUNK;
  } else {
    magi::join_pack_probe_kernel<KeyT><<<magi_user_grid(), JOIN_BLOCK, 0, st>>>(
        prow, pn, xc.probe_in[gpu_id].cols, g_jkfields_dev[gpu_id], xc.n_kf[gpu_id],
        g_jppl_dev[gpu_id], xc.n_ppl[gpu_id], pw, pw_count);
    cudaStreamSynchronize(st);
    chk("pack_probe");
  }
  auto pt_t5 = pt_clock::now();
  unsigned int pw_n = 0;
  if (!fuse_pack) cudaMemcpy(&pw_n, pw_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  // Output buffer is pre-allocated by the wrapper from sirius's processing pool
  // (customCudaMalloc, sized ≈ this GPU's probe count = received + headroom).
  // Using the pre-reserved pool — not raw cudaMalloc — avoids OOM against the
  // pool + RMM (raw cudaMalloc(4.5GB) failed → nullptr write → illegal access).
  JoinResultRow*      out       = xc.out_buf[gpu_id];
  unsigned int*       out_count = xc.out_count_buf[gpu_id];
  const std::uint64_t out_cap   = xc.out_cap[gpu_id];
  if (DBG)
    std::fprintf(stderr, "[magi-join g%d] pn=%lu pw_n=%u out_cap=%lu out_bytes=%.2fGB\n",
                 gpu_id, (unsigned long)pn, pw_n, (unsigned long)out_cap,
                 sizeof(JoinResultRow) * (double)out_cap / 1e9);
  out_count = out_count_dev;  // pooled; the wrapper-passed buffer stays unused
  cudaMemsetAsync(out_count, 0, sizeof(unsigned int), st);
  cudaMemsetAsync(g_joverflow_dev[gpu_id], 0, sizeof(unsigned int), st);
  cudaMemsetAsync(pw_count, 0, sizeof(unsigned int), st);  // reuse as DRAINED counter

  // ── PROBE session: shuffle probe tuples → owners find + emit ──────────────
  magi_runtime::magi_set_tuple_size(gpu_id, sizeof(magi_join::JoinProbeWire));
  if (gpu_id == 0) xc.probe_session = magi_runtime::magi_bump_session();
  xc.probe_start.arrive_and_wait();
  auto pt_t6 = pt_clock::now();
  // Option 3: fused join->aggregate. When a program is supplied, allocate a small
  // per-GPU accumulator + copy the opcode program to device; the probe kernel
  // evaluates each agg-input expression per match and atomicAdds into the accum
  // (overlapping the agg with the shuffle, no out_buf materialization needed).
  double*                                    d_accum = nullptr;
  duckdb::magi_fused::FusedAggProgram*       d_prog  = nullptr;
  if (agg_prog_host) {
    cudaMalloc(reinterpret_cast<void**>(&d_accum),
               sizeof(double) * agg_prog_host->n_aggs);
    cudaMemsetAsync(d_accum, 0, sizeof(double) * agg_prog_host->n_aggs, st);
    cudaMalloc(reinterpret_cast<void**>(&d_prog),
               sizeof(duckdb::magi_fused::FusedAggProgram));
    cudaMemcpyAsync(d_prog, agg_prog_host,
                    sizeof(duckdb::magi_fused::FusedAggProgram),
                    cudaMemcpyHostToDevice, st);
  }
  // Fused grouped-agg (scheme ① POC): a per-owner group table the probe
  // aggregates matches into (group key = high-64 of the compound join key =
  // l_suppkey), instead of materializing JoinResultRows. Env-gated + u128 only.
  magi_ops::AggSlot64<std::uint64_t>* group_tbl = nullptr;
  unsigned int*                       group_of  = nullptr;
  static const bool use_fuse_group =
      std::getenv("MAGI_FUSE_GROUPED") != nullptr;
  // Fused-agg column wiring (POC: env-driven; the planner will supply these once
  // the join->groupby detection lands). fuse_sum_idx = wire/result slot of the
  // summed probe column; fuse_key_dst = result slot of the group-key build column.
  const char* e_sum = std::getenv("MAGI_FUSE_SUM_IDX");
  const char* e_key = std::getenv("MAGI_FUSE_KEY_DST");
  const int fuse_sum_idx = e_sum ? std::atoi(e_sum) : 0;
  const int fuse_key_dst = e_key ? std::atoi(e_key) : 0;
  if (DBG && use_fuse_group) {
    for (int i = 0; i < xc.n_ppl[gpu_id]; ++i)
      std::fprintf(stderr, "[fuse-cols g%d] probe_pl[%d] kind=%d src=%d dst=%d\n",
                   gpu_id, i, (int)xc.ppl_ptr[gpu_id][i].src_kind,
                   (int)xc.ppl_ptr[gpu_id][i].src_col_idx,
                   (int)xc.ppl_ptr[gpu_id][i].dst_idx);
    for (int i = 0; i < xc.n_bpl[gpu_id]; ++i)
      std::fprintf(stderr, "[fuse-cols g%d] build_pl[%d] kind=%d src=%d dst=%d\n",
                   gpu_id, i, (int)xc.bpl_ptr[gpu_id][i].src_kind,
                   (int)xc.bpl_ptr[gpu_id][i].src_col_idx,
                   (int)xc.bpl_ptr[gpu_id][i].dst_idx);
  }
  if (use_fuse_group && std::is_same<KeyT, unsigned __int128>::value) {
    // group_tbl from the SAME pool allocator as out_count_dev (which works with
    // zero loss). Raw cudaMalloc during a live magi session returned memory that
    // overlapped the channel k-buffers → scattered group writes were clobbered
    // by incoming wire traffic (single-slot survived, scatter lost 93-99%).
    const size_t gt_bytes =
        sizeof(magi_ops::AggSlot64<std::uint64_t>) * magi_join::GROUP_N_SLOTS;
    group_tbl = reinterpret_cast<magi_ops::AggSlot64<std::uint64_t>*>(
        magi_runtime::magi_pool_alloc(gt_bytes, gpu_id, false));
    group_of = reinterpret_cast<unsigned int*>(
        magi_runtime::magi_pool_alloc(sizeof(unsigned int), gpu_id, false));
    cudaMemsetAsync(group_of, 0, sizeof(unsigned int), st);
    if (DBG) {
      int cur_dev = -1; cudaGetDevice(&cur_dev);
      std::fprintf(stderr, "[fuse-alloc g%d] pool ptr=%p bytes=%zu cur_dev=%d\n",
          gpu_id, (void*)group_tbl, gt_bytes, cur_dev);
    }
    // Clear and probe are both issued on `st`, so the clear is ordered before
    // the probe's group writes without an explicit sync.
    const int gclear = (magi_join::GROUP_N_SLOTS + 255) / 256;
    magi::join_clear_hbuild<std::uint64_t>
        <<<gclear, 256, 0, st>>>(group_tbl, magi_join::GROUP_N_SLOTS);
  }
  magi::join_probe_kernel<KeyT, N_SLOTS,
                          KBUFFERING_INTRA_PARTITION_SIZE,
                          KBUFFERING_INTER_PARTITION_SIZE>
      <<<magi_user_grid(), JOIN_BLOCK, 0, st>>>(
          pw, pw_n, H, out, out_count, static_cast<unsigned int>(out_cap),
          n_bpl_slots, /*just_load=*/false, g_joverflow_dev[gpu_id],
          // Drained-tuple counter is diagnostic only, and it costs one global
          // atomic on a single address per drained tuple (150M/GPU here) right
          // in the probe's hot path — pass it only when debugging.
          /*n_drained=*/(DBG ? pw_count : nullptr),
          d_accum, d_prog, d_find_ref, group_tbl, group_of, fuse_sum_idx, psrc);
  cudaStreamSynchronize(st);
  chk("probe_kernel");
  if (d_find_ref) cudaFree(d_find_ref);
  if (group_tbl != nullptr) {          // fused: groups -> output rows
    if (DBG) {
      // Host-side verification only. A device-side stats kernel over this table
      // reports impossible "lost writes" (two reads of one slot key inside the
      // same kernel disagree) and cost a full session chasing a phantom race —
      // the host copy has always shown the table to be correct. Never reinstate
      // a device-side verifier here.
      std::vector<magi_ops::AggSlot64<std::uint64_t>> h_slots(magi_join::GROUP_N_SLOTS);
      cudaMemcpy(h_slots.data(), group_tbl,
                 sizeof(magi_ops::AggSlot64<std::uint64_t>) * magi_join::GROUP_N_SLOTS,
                 cudaMemcpyDeviceToHost);
      long long groups = 0; double sumc = 0.0; long long sumv = 0;
      for (int i = 0; i < magi_join::GROUP_N_SLOTS; ++i) {
        if (h_slots[i].key == magi_ops::empty_key_v<std::uint64_t>) continue;
        ++groups; sumc += h_slots[i].values[0];
        long long v; __builtin_memcpy(&v, &h_slots[i].values[1], sizeof(v));
        sumv += v;
      }
      unsigned int h_gof = 0;
      cudaMemcpy(&h_gof, group_of, sizeof(unsigned int), cudaMemcpyDeviceToHost);
      std::fprintf(stderr, "[fuse-host g%d] groups=%lld matches=%.0f sum_val=%lld overflow=%u\n",
                   gpu_id, groups, sumc, sumv, h_gof);
    }
    // Emit one result row per group (the join's output for the fused path).
    cudaMemsetAsync(out_count_dev, 0, sizeof(unsigned int), st);
    const int cg = (magi_join::GROUP_N_SLOTS + 255) / 256;
    magi::fused_group_compact<<<cg, 256, 0, st>>>(
        group_tbl, magi_join::GROUP_N_SLOTS, out, out_count_dev,
        static_cast<unsigned int>(out_cap), fuse_key_dst, fuse_sum_idx);
    cudaStreamSynchronize(st);
    chk("fused_group_compact");
  }
  auto pt_t7 = pt_clock::now();
  if (agg_prog_host) {
    double h_accum[duckdb::magi_fused::MAX_FUSED_AGGS] = {0};
    cudaMemcpy(h_accum, d_accum, sizeof(double) * agg_prog_host->n_aggs,
               cudaMemcpyDeviceToHost);
    // Publish this GPU's partial sums to the side channel for the result collector.
    auto& ch = duckdb::magi_fused::fused_channel();
    for (int a = 0; a < agg_prog_host->n_aggs; ++a) ch.partials[gpu_id][a] = h_accum[a];
    ch.n_aggs = agg_prog_host->n_aggs;
    ch.active = true;
    if (DBG) {
      std::fprintf(stderr, "[fused-agg g%d]", gpu_id);
      for (int a = 0; a < agg_prog_host->n_aggs; ++a)
        std::fprintf(stderr, " a%d=%.4f", a, h_accum[a]);
      std::fprintf(stderr, "\n");
    }
    cudaFree(d_accum);
    cudaFree(d_prog);
  }
  if (DBG) {
    unsigned int dr = 0;
    cudaMemcpy(&dr, pw_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::fprintf(stderr, "[magi-join g%d] DRAINED=%u (sent pw_n=%u)\n", gpu_id, dr, pw_n);
  }
  magi_runtime::magi_sync_after_session(gpu_id, xc.probe_session);
  auto pt_t8 = pt_clock::now();

  // ── Collect this GPU's emitted join rows ─────────────────────────────────
  unsigned int emitted = 0;
  cudaMemcpy(&emitted, out_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  if (DBG) std::fprintf(stderr, "[magi-join g%d] emitted=%u (cap=%lu)\n",
                        gpu_id, emitted, (unsigned long)out_cap);
  if (emitted > out_cap) emitted = static_cast<unsigned int>(out_cap);
  // device_emit: leave the matched rows in the device `out` buffer and let the
  // caller build the output GPUColumns on-device (EmitKeyColumnDevice etc.),
  // skipping the O(rows) D2H + host transpose that dominates join->agg latency.
  if (!device_emit) {
    my_slice.resize(emitted);
    if (emitted > 0)
      cudaMemcpy(my_slice.data(), out, sizeof(JoinResultRow) * emitted,
                 cudaMemcpyDeviceToHost);
  }

  unsigned int overflow_count = 0;
  cudaMemcpy(&overflow_count, g_joverflow_dev[gpu_id], sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  // bw/pw/counters live in the processing pool — reclaimed by ResetBuffer.
  // out / out_count are wrapper-owned (sirius processing pool) — not freed here.

  if (phase_time) {
    auto pt_t9 = pt_clock::now();
    std::fprintf(stderr,
      "[magi-join-phase gpu=%d] pack_build=%.2f bar1=%.2f build_kern=%.2f "
      "build_sess=%.2f pack_probe=%.2f bar2=%.2f probe_kern=%.2f "
      "probe_sess=%.2f extract=%.2f total=%.2f ms\n",
      gpu_id, pt_ms(pt_t0, pt_t1), pt_ms(pt_t1, pt_t2), pt_ms(pt_t2, pt_t3),
      pt_ms(pt_t3, pt_t4), pt_ms(pt_t4, pt_t5), pt_ms(pt_t5, pt_t6),
      pt_ms(pt_t6, pt_t7), pt_ms(pt_t7, pt_t8), pt_ms(pt_t8, pt_t9),
      pt_ms(pt_t0, pt_t9));
  }

  xc.end.arrive_and_wait();

  if (overflow_count != 0) {
    throw std::runtime_error(
        "magi_join: build hash capacity / output cap exceeded (" +
        std::to_string(overflow_count) + " on GPU " + std::to_string(gpu_id) +
        "); falling back to DuckDB.");
  }
  return emitted;
}

// ── Dispatch over (KeyKind, TableSize) — v1: int32/uint64 × S/M/L ──────────
static std::size_t join_dispatch(int gpu_id, KeyKind kk, TableSize ts,
                                 std::vector<JoinResultRow>& my_slice, bool device_emit,
                                 const duckdb::magi_fused::FusedAggProgram* agg_prog_host)
{
  switch (kk) {
    case KeyKind::INT32:
      switch (ts) {
        case TableSize::SMALL:  return join_run_typed_tier<std::int32_t, N_SLOTS_SMALL >(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::MEDIUM: return join_run_typed_tier<std::int32_t, N_SLOTS_MEDIUM>(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::LARGE:  return join_run_typed_tier<std::int32_t, N_SLOTS_LARGE >(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::XLARGE: return join_run_typed_tier<std::int32_t, N_SLOTS_XLARGE >(gpu_id, my_slice, device_emit, agg_prog_host);
        default:                return join_run_typed_tier<std::int32_t, N_SLOTS_XXLARGE>(gpu_id, my_slice, device_emit, agg_prog_host);
      }
    case KeyKind::UINT64:
      switch (ts) {
        case TableSize::SMALL:  return join_run_typed_tier<std::uint64_t, N_SLOTS_SMALL >(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::MEDIUM: return join_run_typed_tier<std::uint64_t, N_SLOTS_MEDIUM>(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::LARGE:  return join_run_typed_tier<std::uint64_t, N_SLOTS_LARGE >(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::XLARGE: return join_run_typed_tier<std::uint64_t, N_SLOTS_XLARGE >(gpu_id, my_slice, device_emit, agg_prog_host);
        default:                return join_run_typed_tier<std::uint64_t, N_SLOTS_XXLARGE>(gpu_id, my_slice, device_emit, agg_prog_host);
      }
    case KeyKind::UINT128:
      switch (ts) {
        case TableSize::SMALL:  return join_run_typed_tier<unsigned __int128, N_SLOTS_SMALL >(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::MEDIUM: return join_run_typed_tier<unsigned __int128, N_SLOTS_MEDIUM>(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::LARGE:  return join_run_typed_tier<unsigned __int128, N_SLOTS_LARGE >(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::XLARGE: return join_run_typed_tier<unsigned __int128, N_SLOTS_XLARGE >(gpu_id, my_slice, device_emit, agg_prog_host);
        default:                return join_run_typed_tier<unsigned __int128, N_SLOTS_XXLARGE>(gpu_id, my_slice, device_emit, agg_prog_host);
      }
    default:
      throw std::runtime_error("magi_join: unsupported key kind");
  }
}

// ── Public entry ────────────────────────────────────────────────────────────
std::size_t distributed_hash_join_run_per_gpu(
    int                                         gpu_id,
    const PerGpuJoinInputs&                     build_inputs,
    const PerGpuJoinInputs&                     probe_inputs,
    const std::vector<magi_ops::KeyFieldEntry>& key_fields,
    const std::vector<JoinPayloadEntry>&        build_payload,
    const std::vector<JoinPayloadEntry>&        probe_payload,
    KeyKind                                     key_kind,
    TableSize                                   table_size,
    JoinResultRow*                              out_buf,
    unsigned int*                               out_count_buf,
    std::uint64_t                               out_cap,
    std::vector<JoinResultRow>&                 my_slice,
    bool                                        device_emit,
    const duckdb::magi_fused::FusedAggProgram*  agg_prog_host)
{
  if (gpu_id < 0 || gpu_id >= NUM_GPUS) {
    std::fprintf(stderr, "[magi-join] bad gpu_id=%d (NUM_GPUS=%d)\n", gpu_id, NUM_GPUS);
    return 0;
  }
  if (static_cast<int>(build_payload.size()) > JOIN_MAX_PAYLOAD_ENTRIES ||
      static_cast<int>(probe_payload.size()) > JOIN_MAX_PAYLOAD_ENTRIES) {
    throw std::runtime_error("magi_join: payload column count exceeds table capacity");
  }
  magi_runtime::MagiInitOnce();
  EnsureJoinBuffers();

  auto& xc = jexchange();
  xc.build_in[gpu_id] = build_inputs;
  xc.probe_in[gpu_id] = probe_inputs;
  xc.kf_ptr[gpu_id]   = key_fields.data();
  xc.n_kf[gpu_id]     = static_cast<int>(key_fields.size());
  xc.ppl_ptr[gpu_id]  = probe_payload.data();
  xc.n_ppl[gpu_id]    = static_cast<int>(probe_payload.size());
  xc.bpl_ptr[gpu_id]  = build_payload.data();
  xc.n_bpl[gpu_id]    = static_cast<int>(build_payload.size());
  xc.out_buf[gpu_id]       = out_buf;
  xc.out_count_buf[gpu_id] = out_count_buf;
  xc.out_cap[gpu_id]       = out_cap;
  if (gpu_id == 0) {
    xc.key_kind   = key_kind;
    xc.table_size = table_size;
  }
  xc.begin.arrive_and_wait();
  // After `begin`, every GPU's probe_in is stashed and visible — each GPU
  // computes the same output cap (Σ probe n_rows) itself, no extra barrier.
  return join_dispatch(gpu_id, xc.key_kind, xc.table_size, my_slice, device_emit,
                       agg_prog_host);
}

}  // namespace magi_generic
}  // namespace duckdb
