// magi_groupby_runtime.cu — single TU hosting the generic magi groupby path.
//
// What lives here:
//   1. Explicit instantiations of `distributed_hash_groupby_kernel<KeyT,…>`
//      for KeyT ∈ {int32_t, uint64_t} × N_GLOBAL_SLOTS ∈ {SMALL, MEDIUM, LARGE}.
//      All combinations end up in libsirius_extension so a new query never
//      forces a recompile — pick (KeyKind, TableSize) at runtime.
//   2. Per-GPU device buffers (lazy-init, reused across queries):
//        - g_agg_dev[]    AggSlot64 arena, sized for LARGE tier (largest)
//        - g_ops_dev[]    AggOpEntry table, copied H2D per query
//        - g_kfields_dev[] KeyFieldEntry table, copied H2D per query
//   3. The host entry `distributed_hash_groupby_run_per_gpu` that drives
//      one worker thread through the magi runtime barrier exchange and
//      launches the right (KeyT, tier) kernel.
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
constexpr int    MAX_AGG_OPS     = 16;  // ample for any TPC-H GROUP BY
constexpr int    MAX_KEY_FIELDS  = 8;   // 2-VARCHAR (Q1) or compound INT+VARCHAR

// Tier sizes. SMALL covers Q1/Q5/Q9; MEDIUM covers Q16/Q18-class; LARGE
// covers Q3/Q11. Slot is 64B, so MAX_TIER_SLOTS * 64 is the per-GPU agg
// buffer footprint (we keep it allocated at the largest tier and just
// memset/memcpy the prefix the active tier needs).
constexpr int N_SLOTS_SMALL    = 256;
constexpr int N_SLOTS_MEDIUM   = 16 * 1024;
constexpr int N_SLOTS_LARGE    = 1024 * 1024;
constexpr int MAX_TIER_SLOTS   = N_SLOTS_LARGE;

// Producer per-block local pre-aggregation hash (shared memory). It MUST be at
// least as large as the query's group cardinality: producer_local_agg drops any
// row whose key can't claim a local slot (BlockHashAgg returns nullptr when the
// table is full). With the old value of 16 that silently undercounted every
// query with >16 groups (Q9 has 175) — keys that overflowed the 16-slot table
// lost all their rows. Size it to the SMALL tier (256): a 256-slot, 64B-state
// hash is ~18KB of shared memory (well under the 48KB static limit) and blockDim
// (1024) ≥ 256 so the one-thread-per-slot flush covers every slot. This makes
// SMALL-tier groupby (cardinality ≤ 256) exact. MEDIUM/LARGE high-cardinality
// queries still need a spill/pass-through path instead of dropping — tracked for
// the Q3 cuco-style global hashagg work.
constexpr int N_LOCAL_SLOTS    = N_SLOTS_SMALL;
constexpr size_t AGG_BUF_BYTES = static_cast<size_t>(64) * MAX_TIER_SLOTS;

// Convert TableSize → slot count.
constexpr int slots_for(TableSize t) {
  switch (t) {
    case TableSize::SMALL:  return N_SLOTS_SMALL;
    case TableSize::MEDIUM: return N_SLOTS_MEDIUM;
    case TableSize::LARGE:  return N_SLOTS_LARGE;
  }
  return N_SLOTS_SMALL;
}

}}  // namespace duckdb::magi_generic

// ── Per-KeyT init kernel ──────────────────────────────────────────────────
// `cudaMemset(agg, 0, …)` works for uint64 keys (empty_key_v<uint64_t> = 0)
// but not for int32 keys (empty_key_v<int32_t> = -1). One small kernel
// writes the correct empty-key sentinel into every slot at session start.
// `n_slots` is the *active* tier's slot count — we only touch that prefix.
namespace duckdb { namespace magi_generic {
template <typename KeyT>
__global__ void init_global_agg_slots(magi_ops::AggSlot64<KeyT>* agg, int n_slots)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_slots) {
    agg[i] = magi_ops::AggSlot64<KeyT>{};   // zero values + count + index
    agg[i].key = magi_ops::empty_key_v<KeyT>;
  }
}
}}  // namespace duckdb::magi_generic

// ── Explicit kernel instantiations (KeyT × N_GLOBAL_SLOTS tier) ───────────
//
// Outside namespace duckdb so the symbols match what the kernel template
// expects in `namespace magi`. 6 instantiations: {int32_t, uint64_t} × {S,M,L}.
namespace magi {

#define MAGI_INSTANTIATE_KERNEL(KEY_TYPE, N_SLOTS)                              \
  template __global__ void                                                       \
  distributed_hash_groupby_kernel<KEY_TYPE,                                      \
                                   duckdb::magi_generic::BLOCK_SIZE,             \
                                   KBUFFERING_INTRA_PARTITION_SIZE,              \
                                   KBUFFERING_INTER_PARTITION_SIZE,              \
                                   duckdb::magi_generic::N_LOCAL_SLOTS,          \
                                   N_SLOTS>(                                     \
      const std::uint64_t*, std::uint64_t,                                       \
      magi_ops::ColPack,                                                         \
      const magi_ops::KeyFieldEntry*, int,                                       \
      const magi_ops::AggOpEntry*, int,                                          \
      magi_ops::AggSlot64<KEY_TYPE>*, bool)

MAGI_INSTANTIATE_KERNEL(std::int32_t,  duckdb::magi_generic::N_SLOTS_SMALL);
MAGI_INSTANTIATE_KERNEL(std::int32_t,  duckdb::magi_generic::N_SLOTS_MEDIUM);
MAGI_INSTANTIATE_KERNEL(std::int32_t,  duckdb::magi_generic::N_SLOTS_LARGE);
MAGI_INSTANTIATE_KERNEL(std::uint64_t, duckdb::magi_generic::N_SLOTS_SMALL);
MAGI_INSTANTIATE_KERNEL(std::uint64_t, duckdb::magi_generic::N_SLOTS_MEDIUM);
MAGI_INSTANTIATE_KERNEL(std::uint64_t, duckdb::magi_generic::N_SLOTS_LARGE);

#undef MAGI_INSTANTIATE_KERNEL

}  // namespace magi

namespace duckdb { namespace magi_generic {

// ── Per-GPU device buffers (lazy-init) ────────────────────────────────────
// `g_agg_dev[i]`     : MAX_TIER_SLOTS × 64B per GPU (LARGE-tier sized;
//                       smaller tiers just use the prefix).
// `g_ops_dev[i]`     : MAX_AGG_OPS × 4B per GPU; AggOpEntry table copied
//                       per query.
// `g_kfields_dev[i]` : MAX_KEY_FIELDS × 4B per GPU; KeyFieldEntry table
//                       copied per query.
static std::byte*               g_agg_dev    [NUM_GPUS] = { nullptr };
static magi_ops::AggOpEntry*    g_ops_dev    [NUM_GPUS] = { nullptr };
static magi_ops::KeyFieldEntry* g_kfields_dev[NUM_GPUS] = { nullptr };
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
    cudaMalloc(reinterpret_cast<void**>(&g_kfields_dev[i]),
               sizeof(magi_ops::KeyFieldEntry) * MAX_KEY_FIELDS);
  }
}

// ── Per-thread barrier exchange ──────────────────────────────────────────
// Same shape as Q5Exchange: stash inputs, sync, run, sync, return slice.
// All shape fields (key_kind, table_size, n_ops, n_key_fields) are written
// by gpu_id==0 before `after_session_start.arrive_and_wait()` (sirius's
// plan-gen is deterministic so all workers compute the same values, but
// only one source-of-truth keeps the launch path branch-free).
struct GenericExchange {
  std::barrier<>                                  begin{NUM_GPUS};
  std::barrier<>                                  after_session_start{NUM_GPUS};
  std::barrier<>                                  end{NUM_GPUS};
  std::array<PerGpuInputs, NUM_GPUS>              inputs{};
  std::array<const magi_ops::AggOpEntry*,
             NUM_GPUS>                            ops_host_ptrs{};
  std::array<int, NUM_GPUS>                       n_ops{};
  std::array<const magi_ops::KeyFieldEntry*,
             NUM_GPUS>                            kfields_host_ptrs{};
  std::array<int, NUM_GPUS>                       n_key_fields{};
  KeyKind                                         key_kind_for_this_query = KeyKind::INT32;
  TableSize                                       table_size_for_this_query = TableSize::SMALL;
  std::uint64_t                                   session_id = 0;
};
static GenericExchange& exchange() { static GenericExchange e; return e; }

// ── Per-GPU launch (templated on KeyT and N_SLOTS tier) ───────────────────
// Mirrors Q5MagiRunPerGpu's flow exactly; the only differences are the
// kernel symbol (now selected by both KeyT and tier) and the slice
// extraction loop.
template <typename KeyT, int N_SLOTS>
static std::size_t run_per_gpu_typed_tier(int                        gpu_id,
                                           std::vector<AggResultRow>& my_slice)
{
  auto& xc = exchange();
  const int gpu = magi_q1::magi_phys_gpu(gpu_id);
  cudaSetDevice(gpu);

  // Clear receiver global agg (writes empty_key_v<KeyT> per slot — a plain
  // memset to 0 would mis-mark int32 slots as "occupied" since
  // empty_key_v<int32_t> = -1). Only touch the prefix this tier uses.
  auto* agg_typed_for_init =
      reinterpret_cast<magi_ops::AggSlot64<KeyT>*>(g_agg_dev[gpu_id]);
  constexpr int INIT_BLOCK = 256;
  const     int init_grid  = (N_SLOTS + INIT_BLOCK - 1) / INIT_BLOCK;
  init_global_agg_slots<KeyT>
      <<<init_grid, INIT_BLOCK, 0, magi_q1::magi_stream(gpu_id)>>>(
          agg_typed_for_init, N_SLOTS);

  // Push per-query tables: ops + key_fields.
  cudaMemcpyAsync(g_ops_dev[gpu_id],
                   xc.ops_host_ptrs[gpu_id],
                   sizeof(magi_ops::AggOpEntry) * xc.n_ops[gpu_id],
                   cudaMemcpyHostToDevice,
                   magi_q1::magi_stream(gpu_id));
  cudaMemcpyAsync(g_kfields_dev[gpu_id],
                   xc.kfields_host_ptrs[gpu_id],
                   sizeof(magi_ops::KeyFieldEntry) * xc.n_key_fields[gpu_id],
                   cudaMemcpyHostToDevice,
                   magi_q1::magi_stream(gpu_id));
  magi_q1::magi_set_tuple_size(gpu_id, sizeof(magi_ops::AggSlot64<KeyT>));

  // Session bump (thread 0) + sync.
  if (gpu_id == 0) xc.session_id = magi_q1::magi_bump_session();
  xc.after_session_start.arrive_and_wait();

  // Launch the generic kernel for (KeyT, N_SLOTS) tier.
  const auto&         in      = xc.inputs[gpu_id];
  std::uint64_t*      row_ids = magi_q1::GetIdentityRowIdsShared(gpu, in.n_filtered);

  // Make this GPU's cached aggregate input coherent before the kernel reads it.
  // sirius uploads each GPU's cache slice with a cudaMemcpyAsync issued on a
  // GPU-0 stream (the scan's stream pool lives on device 0), so for a consuming
  // GPU != 0 the slice lands in its DRAM via a cross-device copy whose result is
  // not guaranteed visible to a kernel on that device without a device-side
  // sync. Without this, the first cross-GPU GROUP BY read stale data and
  // produced wrong, run-to-run-varying sums (e.g. Q5 was correct only ~2/8
  // runs; with this sync it is 10/10).
  cudaDeviceSynchronize();
  auto* global_agg_typed =
      reinterpret_cast<magi_ops::AggSlot64<KeyT>*>(g_agg_dev[gpu_id]);

  magi::distributed_hash_groupby_kernel<KeyT,
                                         BLOCK_SIZE,
                                         KBUFFERING_INTRA_PARTITION_SIZE,
                                         KBUFFERING_INTER_PARTITION_SIZE,
                                         N_LOCAL_SLOTS,
                                         N_SLOTS>
      <<<USER_KERNEL_GRID_SIZE, BLOCK_SIZE, 0,
         magi_q1::magi_stream(gpu_id)>>>(row_ids,
                                          in.n_filtered,
                                          in.cols,
                                          g_kfields_dev[gpu_id],
                                          xc.n_key_fields[gpu_id],
                                          g_ops_dev[gpu_id],
                                          xc.n_ops[gpu_id],
                                          global_agg_typed,
                                          /*just_load=*/false);
  cudaStreamSynchronize(magi_q1::magi_stream(gpu_id));
  magi_q1::magi_sync_after_session(gpu_id, xc.session_id);

  // D2H only the active tier's prefix (saves up to 64 MB at SMALL).
  const size_t tier_bytes = static_cast<size_t>(N_SLOTS)
                          * sizeof(magi_ops::AggSlot64<KeyT>);
  std::vector<magi_ops::AggSlot64<KeyT>> host(N_SLOTS);
  cudaMemcpy(host.data(), g_agg_dev[gpu_id], tier_bytes, cudaMemcpyDeviceToHost);

  my_slice.clear();
  for (int j = 0; j < N_SLOTS; ++j) {
    if (host[j].key == magi_ops::empty_key_v<KeyT>) continue;
    AggResultRow row{};
    // Widen KeyT to u64 for the result row. For int32 we zero-extend via
    // unsigned cast (caller knows the original column type and re-narrows
    // on emit; sign-extension would corrupt high bits for negative keys).
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

// ── 6-way dispatch over (KeyKind, TableSize) ─────────────────────────────
static std::size_t dispatch_by_kind_and_tier(int                        gpu_id,
                                              KeyKind                    kk,
                                              TableSize                  ts,
                                              std::vector<AggResultRow>& my_slice)
{
  switch (kk) {
    case KeyKind::INT32:
      switch (ts) {
        case TableSize::SMALL:  return run_per_gpu_typed_tier<std::int32_t, N_SLOTS_SMALL >(gpu_id, my_slice);
        case TableSize::MEDIUM: return run_per_gpu_typed_tier<std::int32_t, N_SLOTS_MEDIUM>(gpu_id, my_slice);
        case TableSize::LARGE:  return run_per_gpu_typed_tier<std::int32_t, N_SLOTS_LARGE >(gpu_id, my_slice);
      }
      break;
    case KeyKind::UINT64:
      switch (ts) {
        case TableSize::SMALL:  return run_per_gpu_typed_tier<std::uint64_t, N_SLOTS_SMALL >(gpu_id, my_slice);
        case TableSize::MEDIUM: return run_per_gpu_typed_tier<std::uint64_t, N_SLOTS_MEDIUM>(gpu_id, my_slice);
        case TableSize::LARGE:  return run_per_gpu_typed_tier<std::uint64_t, N_SLOTS_LARGE >(gpu_id, my_slice);
      }
      break;
  }
  return 0;
}

// ── Public entry ─────────────────────────────────────────────────────────
std::size_t distributed_hash_groupby_run_per_gpu(
    int                                              gpu_id,
    const PerGpuInputs&                              inputs,
    const std::vector<magi_ops::KeyFieldEntry>&      key_fields,
    const std::vector<magi_ops::AggOpEntry>&         ops,
    KeyKind                                          key_kind,
    TableSize                                        table_size,
    std::vector<AggResultRow>&                       my_slice)
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
  if (static_cast<int>(key_fields.size()) > MAX_KEY_FIELDS) {
    std::fprintf(stderr,
                 "[magi-generic] %zu key fields > MAX_KEY_FIELDS=%d; bump and "
                 "rebuild\n",
                 key_fields.size(), MAX_KEY_FIELDS);
    return 0;
  }

  magi_q1::MagiInitOnce();
  EnsureDeviceBuffers();

  auto& xc = exchange();
  xc.inputs[gpu_id]            = inputs;
  xc.ops_host_ptrs[gpu_id]     = ops.data();
  xc.n_ops[gpu_id]             = static_cast<int>(ops.size());
  xc.kfields_host_ptrs[gpu_id] = key_fields.data();
  xc.n_key_fields[gpu_id]      = static_cast<int>(key_fields.size());
  if (gpu_id == 0) {
    xc.key_kind_for_this_query   = key_kind;
    xc.table_size_for_this_query = table_size;
  }
  xc.begin.arrive_and_wait();

  return dispatch_by_kind_and_tier(gpu_id,
                                    xc.key_kind_for_this_query,
                                    xc.table_size_for_this_query,
                                    my_slice);
}

}}  // namespace duckdb::magi_generic
