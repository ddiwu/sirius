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
#include <cstdio>
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
      magi_ops::AggSlot64<KEY_TYPE>*, int, bool, unsigned int*)

#define MAGI_JOIN_INST_PROBE(KEY_TYPE, N_SLOTS)                                 \
  template __global__ void                                                      \
  join_probe_kernel<KEY_TYPE, N_SLOTS,                                          \
                    KBUFFERING_INTRA_PARTITION_SIZE,                            \
                    KBUFFERING_INTER_PARTITION_SIZE>(                           \
      const duckdb::magi_join::JoinProbeWire*, std::uint64_t,                   \
      magi_ops::AggSlot64<KEY_TYPE>*,                                           \
      duckdb::magi_generic::JoinResultRow*, unsigned int*, unsigned int, int,   \
      bool, unsigned int*, unsigned int*,                                       \
      double*, const duckdb::magi_fused::FusedAggProgram*)

#define MAGI_JOIN_INST_ALL(KEY_TYPE)                                            \
  MAGI_JOIN_INST_PACK_BUILD(KEY_TYPE);                                          \
  MAGI_JOIN_INST_PACK_PROBE(KEY_TYPE);                                          \
  MAGI_JOIN_INST_BUILD(KEY_TYPE, duckdb::magi_generic::N_SLOTS_SMALL);          \
  MAGI_JOIN_INST_BUILD(KEY_TYPE, duckdb::magi_generic::N_SLOTS_MEDIUM);         \
  MAGI_JOIN_INST_BUILD(KEY_TYPE, duckdb::magi_generic::N_SLOTS_LARGE);          \
  MAGI_JOIN_INST_BUILD(KEY_TYPE, duckdb::magi_generic::N_SLOTS_XLARGE);         \
  MAGI_JOIN_INST_PROBE(KEY_TYPE, duckdb::magi_generic::N_SLOTS_SMALL);          \
  MAGI_JOIN_INST_PROBE(KEY_TYPE, duckdb::magi_generic::N_SLOTS_MEDIUM);         \
  MAGI_JOIN_INST_PROBE(KEY_TYPE, duckdb::magi_generic::N_SLOTS_LARGE);          \
  MAGI_JOIN_INST_PROBE(KEY_TYPE, duckdb::magi_generic::N_SLOTS_XLARGE)

MAGI_JOIN_INST_ALL(std::int32_t);
MAGI_JOIN_INST_ALL(std::uint64_t);

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
constexpr int    JOIN_MAX_TIER_SLOTS = N_SLOTS_XLARGE;
constexpr size_t JOIN_HBUILD_BYTES =
    sizeof(magi_ops::AggSlot64<std::uint64_t>) * JOIN_MAX_TIER_SLOTS;
constexpr int    JOIN_BLOCK = 1024;

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
  for (int i = 0; i < NUM_GPUS; ++i) {
    int gpu = magi_runtime::magi_phys_gpu(i);
    cudaSetDevice(gpu);
    cudaMalloc(reinterpret_cast<void**>(&g_hbuild_dev[i]), JOIN_HBUILD_BYTES);
    cudaMalloc(reinterpret_cast<void**>(&g_jkfields_dev[i]),
               sizeof(magi_ops::KeyFieldEntry) * JOIN_MAX_KEY_FIELDS);
    cudaMalloc(reinterpret_cast<void**>(&g_jppl_dev[i]),
               sizeof(JoinPayloadEntry) * JOIN_MAX_PAYLOAD_ENTRIES);
    cudaMalloc(reinterpret_cast<void**>(&g_jbpl_dev[i]),
               sizeof(JoinPayloadEntry) * JOIN_MAX_PAYLOAD_ENTRIES);
    cudaMalloc(reinterpret_cast<void**>(&g_joverflow_dev[i]), sizeof(unsigned int));
  }
}

// ── Per-GPU barrier exchange ────────────────────────────────────────────────
struct JoinExchange {
  std::barrier<> begin{NUM_GPUS};
  std::barrier<> build_start{NUM_GPUS};
  std::barrier<> after_build{NUM_GPUS};
  std::barrier<> probe_start{NUM_GPUS};
  std::barrier<> end{NUM_GPUS};

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
  cudaStream_t      st  = magi_runtime::magi_stream(gpu_id);
  auto*             H   = reinterpret_cast<magi_ops::AggSlot64<KeyT>*>(g_hbuild_dev[gpu_id]);

  const bool DBG = std::getenv("MAGI_JOIN_DEBUG") != nullptr;
  auto chk = [&](const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
      std::fprintf(stderr, "[magi-join g%d tier=%d] CUDA ERR after %s: %s\n",
                   gpu_id, N_SLOTS, what, cudaGetErrorString(e));
  };

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
  magi_join::JoinBuildWire* bw = nullptr;
  unsigned int*             bw_count = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&bw),
             sizeof(magi_join::JoinBuildWire) * std::max<std::uint64_t>(bn, 1));
  cudaMalloc(reinterpret_cast<void**>(&bw_count), sizeof(unsigned int));
  cudaMemsetAsync(bw_count, 0, sizeof(unsigned int), st);
  cudaMemsetAsync(g_joverflow_dev[gpu_id], 0, sizeof(unsigned int), st);
  std::uint64_t* brow = magi_runtime::GetIdentityRowIdsShared(gpu, bn);
  cudaDeviceSynchronize();  // make this GPU's cached input coherent (cf. groupby)
  magi::join_pack_build_kernel<KeyT><<<USER_KERNEL_GRID_SIZE, JOIN_BLOCK, 0, st>>>(
      brow, bn, xc.build_in[gpu_id].cols, g_jkfields_dev[gpu_id], xc.n_kf[gpu_id],
      g_jbpl_dev[gpu_id], xc.n_bpl[gpu_id], bw, bw_count);
  cudaStreamSynchronize(st);
  chk("pack_build");
  unsigned int bw_n = 0;
  cudaMemcpy(&bw_n, bw_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  if (DBG) std::fprintf(stderr, "[magi-join g%d] bn=%lu bw_n=%u n_bpl=%d n_ppl=%d\n",
                        gpu_id, (unsigned long)bn, bw_n, xc.n_bpl[gpu_id], xc.n_ppl[gpu_id]);

  // ── BUILD session: shuffle wire tuples → owners insert into H_build ───────
  magi_runtime::magi_set_tuple_size(gpu_id, sizeof(magi_join::JoinBuildWire));
  if (gpu_id == 0) xc.build_session = magi_runtime::magi_bump_session();
  xc.build_start.arrive_and_wait();
  magi::join_build_kernel<KeyT, N_SLOTS,
                          KBUFFERING_INTRA_PARTITION_SIZE,
                          KBUFFERING_INTER_PARTITION_SIZE>
      <<<USER_KERNEL_GRID_SIZE, JOIN_BLOCK, 0, st>>>(
          bw, bw_n, H, n_bpl_slots, /*just_load=*/false, g_joverflow_dev[gpu_id]);
  cudaStreamSynchronize(st);
  chk("build_kernel");
  magi_runtime::magi_sync_after_session(gpu_id, xc.build_session);
  xc.after_build.arrive_and_wait();   // every owner's H_build is complete

  // ── PROBE: pack rows → wire tuples ────────────────────────────────────────
  const std::uint64_t pn = xc.probe_in[gpu_id].n_rows;
  magi_join::JoinProbeWire* pw = nullptr;
  unsigned int*             pw_count = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&pw),
             sizeof(magi_join::JoinProbeWire) * std::max<std::uint64_t>(pn, 1));
  cudaMalloc(reinterpret_cast<void**>(&pw_count), sizeof(unsigned int));
  cudaMemsetAsync(pw_count, 0, sizeof(unsigned int), st);
  std::uint64_t* prow = magi_runtime::GetIdentityRowIdsShared(gpu, pn);
  // Make this GPU's cross-device-uploaded probe slice coherent before the pack
  // kernel reads it — same fix the build phase (above) and the groupby runtime
  // apply. Without it, a consuming GPU != 0 reads stale probe keys/payload →
  // nondeterministic wrong join results.
  cudaDeviceSynchronize();
  magi::join_pack_probe_kernel<KeyT><<<USER_KERNEL_GRID_SIZE, JOIN_BLOCK, 0, st>>>(
      prow, pn, xc.probe_in[gpu_id].cols, g_jkfields_dev[gpu_id], xc.n_kf[gpu_id],
      g_jppl_dev[gpu_id], xc.n_ppl[gpu_id], pw, pw_count);
  cudaStreamSynchronize(st);
  chk("pack_probe");
  unsigned int pw_n = 0;
  cudaMemcpy(&pw_n, pw_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

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
  cudaMalloc(reinterpret_cast<void**>(&out_count), sizeof(unsigned int));
  cudaMemsetAsync(out_count, 0, sizeof(unsigned int), st);
  cudaMemsetAsync(g_joverflow_dev[gpu_id], 0, sizeof(unsigned int), st);
  cudaMemsetAsync(pw_count, 0, sizeof(unsigned int), st);  // reuse as DRAINED counter

  // ── PROBE session: shuffle probe tuples → owners find + emit ──────────────
  magi_runtime::magi_set_tuple_size(gpu_id, sizeof(magi_join::JoinProbeWire));
  if (gpu_id == 0) xc.probe_session = magi_runtime::magi_bump_session();
  xc.probe_start.arrive_and_wait();
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
  magi::join_probe_kernel<KeyT, N_SLOTS,
                          KBUFFERING_INTRA_PARTITION_SIZE,
                          KBUFFERING_INTER_PARTITION_SIZE>
      <<<USER_KERNEL_GRID_SIZE, JOIN_BLOCK, 0, st>>>(
          pw, pw_n, H, out, out_count, static_cast<unsigned int>(out_cap),
          n_bpl_slots, /*just_load=*/false, g_joverflow_dev[gpu_id],
          /*n_drained=*/pw_count,     // reuse pw_count (pw_n already read) as drained counter
          d_accum, d_prog);
  cudaStreamSynchronize(st);
  chk("probe_kernel");
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

  cudaFree(bw); cudaFree(bw_count);
  cudaFree(pw); cudaFree(pw_count);
  // out / out_count are wrapper-owned (sirius processing pool) — not freed here.

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
        default:                return join_run_typed_tier<std::int32_t, N_SLOTS_XLARGE>(gpu_id, my_slice, device_emit, agg_prog_host);
      }
    case KeyKind::UINT64:
      switch (ts) {
        case TableSize::SMALL:  return join_run_typed_tier<std::uint64_t, N_SLOTS_SMALL >(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::MEDIUM: return join_run_typed_tier<std::uint64_t, N_SLOTS_MEDIUM>(gpu_id, my_slice, device_emit, agg_prog_host);
        case TableSize::LARGE:  return join_run_typed_tier<std::uint64_t, N_SLOTS_LARGE >(gpu_id, my_slice, device_emit, agg_prog_host);
        default:                return join_run_typed_tier<std::uint64_t, N_SLOTS_XLARGE>(gpu_id, my_slice, device_emit, agg_prog_host);
      }
    default:
      throw std::runtime_error("magi_join: unsupported key kind for v1 (int32/uint64 only)");
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
