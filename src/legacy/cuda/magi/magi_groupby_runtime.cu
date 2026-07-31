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
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "data_plane/api/distributed_hash_groupby.cuh"
#include "data_plane/config.cuh"

#include "legacy/operator/magi_distributed_groupby.hpp"
#include "legacy/operator/magi_runtime_shared.hpp"

namespace duckdb { namespace magi_generic {

// ── Kernel-launch constants ──────────────────────────────────────────────
constexpr int    BLOCK_SIZE      = 1024;
constexpr int    MAX_AGG_OPS     = 48;  // TPC-H aggs + wide-key inline key-blob KEEP ops
constexpr int    MAX_KEY_FIELDS  = 8;   // 2-VARCHAR (Q1) or compound INT+VARCHAR

// Tier sizes (N_SLOTS_SMALL/MEDIUM/LARGE/XLARGE) live in the public header so
// PickTableSize's routing policy and these instantiations share one source of
// truth. The per-GPU agg buffer footprint is sizeof(AggSlot64) * MAX_TIER_SLOTS
// (we keep it allocated at the largest tier and just memset/memcpy the prefix
// the active tier needs). MUST use sizeof(slot), NOT a hardcoded 64 — the slot
// width is now 128B (widened for fat GROUP BYs like Q1); a literal 64 here
// under-allocates the arena to half size and corrupts memory on write.
constexpr int MAX_TIER_SLOTS   = N_SLOTS_XLARGE;

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
// Arena byte sizes. BASE is the pre-XXXLARGE footprint (4 GB: 64M 64B slots);
// XXXL doubles it for the 128M-slot groupby tier. Which one the two resident
// arenas actually get is decided at prealloc time by GroupbyXxxlEnabled() —
// every capacity check on the query path MUST use agg_buf_bytes(), never the
// constants, or an XXXL-tier query would write past a BASE-sized arena.
constexpr size_t AGG_BUF_BYTES_BASE =
    std::max(sizeof(magi_ops::AggSlot64<std::uint64_t, 128>) * MAX_TIER_SLOTS,
             sizeof(magi_ops::AggSlot64<std::uint64_t, 64>) *
                 (size_t)N_SLOTS_XXLARGE);
constexpr size_t AGG_BUF_BYTES_XXXL =
    sizeof(magi_ops::AggSlot64<std::uint64_t, 64>) * (size_t)N_SLOTS_XXXLARGE;
inline size_t agg_buf_bytes() {
  return GroupbyXxxlEnabled() ? AGG_BUF_BYTES_XXXL : AGG_BUF_BYTES_BASE;
}
// Largest tier the groupby path may pick under the current arena size.
inline int max_groupby_slots() {
  return GroupbyXxxlEnabled() ? N_SLOTS_XXXLARGE : N_SLOTS_XXLARGE;
}

// Wide-key (320B-slot) table sizes. Two tiers, both fitting the 2GB arena:
//  - WIDE_M (2M slots, 640MB): picked when this GPU's input rows fit at ≤~60%
//    load. Table init + compact scan cost is O(table bytes), so right-sizing
//    matters: at Q10 scale (1.08M rows) the 6M tier spends 13ms/query just
//    initializing 2×1.92GB of slots.
//  - WIDE (6M slots, 1.92GB): everything bigger; ~4.6M groups per GPU at the
//    0.77 load bound; overflow counter falls back to DuckDB beyond that.
// Tier choice is PER-GPU LOCAL (by own n_filtered): GPUs may legally disagree —
// the wire format depends only on SB (320B), and each GPU's stage/final tables
// are private. Extreme skew (peer sends me far more groups than my own rows)
// can overflow the small tier → overflow counter throws → CPU fallback, never
// a wrong result.
constexpr int N_SLOTS_WIDE   = 6 * 1024 * 1024;
constexpr int N_SLOTS_WIDE_M = 2 * 1024 * 1024;
static_assert((size_t)N_SLOTS_WIDE *
                  sizeof(magi_ops::AggSlot64<unsigned __int128, 320>) <=
              AGG_BUF_BYTES_BASE,
              "wide-tier table must fit even the BASE-sized AggSlot64 arena");

// Convert TableSize → slot count.
constexpr int slots_for(TableSize t) {
  switch (t) {
    case TableSize::SMALL:  return N_SLOTS_SMALL;
    case TableSize::MEDIUM: return N_SLOTS_MEDIUM;
    case TableSize::LARGE:  return N_SLOTS_LARGE;
    case TableSize::XLARGE: return N_SLOTS_XLARGE;
    case TableSize::XXLARGE: return N_SLOTS_XXLARGE;
    case TableSize::XXXLARGE: return N_SLOTS_XXXLARGE;
  }
  return N_SLOTS_SMALL;
}

}}  // namespace duckdb::magi_generic

// ── Per-KeyT init kernel ──────────────────────────────────────────────────
// Writes the empty-key sentinel into every slot at session start. Only the
// all-zero-sentinel kinds (__int128 hash keys) can use plain cudaMemset
// instead (see the constexpr branch at the call site); int32/uint64 use
// non-zero sentinels (INT32_MIN / all-ones — real key values of 0 or -1 must
// not collide with "slot free", see empty_key_v) and need this kernel.
// `n_slots` is the *active* tier's slot count — we only touch that prefix.
namespace duckdb { namespace magi_generic {
template <typename KeyT, int SB>
__global__ void init_global_agg_slots(magi_ops::AggSlot64<KeyT, SB>* agg, int n_slots)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_slots) {
    agg[i] = magi_ops::AggSlot64<KeyT, SB>{};   // zero values + count + index
    agg[i].key = magi_ops::empty_key_v<KeyT>;
  }
}

// Flush helper: GPU-compact the LIVE slots of a tier-sized hash table into a dense
// prefix of `out`, counting them via an atomic. Lets the flush D2H + host loop run
// in O(live) instead of O(tier) (XLARGE = 16M slots = 1 GB scanned per GPU).
template <typename KeyT, int SB>
__global__ void compact_live_slots_kernel(const magi_ops::AggSlot64<KeyT, SB>* __restrict__ in,
                                          magi_ops::AggSlot64<KeyT, SB>* __restrict__       out,
                                          unsigned int* __restrict__                    count,
                                          int                                           n_slots,
                                          SlotPredicate                                 pred)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_slots) return;
  if (in[i].key == magi_ops::empty_key_v<KeyT>) return;
  // HAVING pushdown: this table is the FINAL (post-shuffle, fully merged)
  // per-GPU partition, so slot values are GLOBAL aggregates — filtering here
  // is exact. Thresholds carry conservative slack (may keep extra rows, never
  // drops a qualifying one); the residual plan FILTER trims the rest.
  if (pred.cmp != 0) {
    bool keep;
    if (pred.value_is_int64) {
      long long v;
      memcpy(&v, &in[i].values[pred.slot], sizeof(v));
      keep = (pred.cmp == 1) ? (v > pred.i_threshold)
           : (pred.cmp == 2) ? (v >= pred.i_threshold)
           : (pred.cmp == 3) ? (v < pred.i_threshold)
                             : (v <= pred.i_threshold);
    } else {
      const double v = in[i].values[pred.slot];
      keep = (pred.cmp == 1) ? (v > pred.d_threshold)
           : (pred.cmp == 2) ? (v >= pred.d_threshold)
           : (pred.cmp == 3) ? (v < pred.d_threshold)
                             : (v <= pred.d_threshold);
    }
    if (!keep) return;
  }
  out[atomicAdd(count, 1u)] = in[i];
}
}}  // namespace duckdb::magi_generic

// ── Explicit kernel instantiations (KeyT × N_GLOBAL_SLOTS tier) ───────────
//
// Outside namespace duckdb so the symbols match what the kernel template
// expects in `namespace magi`. 6 instantiations: {int32_t, uint64_t} × {S,M,L}.
namespace magi {

// Two-kernel global-memory hashagg: Kernel A (rows -> per-GPU global hash H)
// and Kernel B (scan H -> shuffle -> per-GPU final). Replaces the single
// shmem-pre-agg kernel, which capped each block at N_LOCAL_SLOTS distinct keys.
#define MAGI_INSTANTIATE_PREAGG(KEY_TYPE, N_SLOTS, SB)                          \
  template __global__ void                                                       \
  global_preagg_kernel<KEY_TYPE, N_SLOTS, SB>(                                   \
      const std::uint64_t*, std::uint64_t,                                       \
      magi_ops::ColPack,                                                         \
      const magi_ops::KeyFieldEntry*, int,                                       \
      const magi_ops::AggOpEntry*, int,                                          \
      magi_ops::AggSlot64<KEY_TYPE, SB>*, bool, unsigned int*)

#define MAGI_INSTANTIATE_SHUFFLE(KEY_TYPE, N_SLOTS, SB)                         \
  template __global__ void                                                       \
  shuffle_global_kernel<KEY_TYPE,                                                \
                        KBUFFERING_INTRA_PARTITION_SIZE,                         \
                        KBUFFERING_INTER_PARTITION_SIZE,                         \
                        N_SLOTS, SB, false>(                                     \
      const magi_ops::AggSlot64<KEY_TYPE, SB>*,                                  \
      const magi_ops::AggOpEntry*, int,                                          \
      magi_ops::AggSlot64<KEY_TYPE, SB>*, bool, unsigned int*)

// send_direct-batch benchmark variant (A/B vs SendDirectStream); 64B slots
// only — send_direct is tuple-granular.
#define MAGI_INSTANTIATE_SHUFFLE_DIRECT(KEY_TYPE, N_SLOTS)                       \
  template __global__ void                                                       \
  shuffle_global_kernel<KEY_TYPE,                                                \
                        KBUFFERING_INTRA_PARTITION_SIZE,                         \
                        KBUFFERING_INTER_PARTITION_SIZE,                         \
                        N_SLOTS, 64, true>(                                      \
      const magi_ops::AggSlot64<KEY_TYPE, 64>*,                                  \
      const magi_ops::AggOpEntry*, int,                                          \
      magi_ops::AggSlot64<KEY_TYPE, 64>*, bool, unsigned int*)

#define MAGI_INSTANTIATE_BOTH(KEY_TYPE, N_SLOTS, SB)                            \
  MAGI_INSTANTIATE_PREAGG(KEY_TYPE, N_SLOTS, SB);                                \
  MAGI_INSTANTIATE_SHUFFLE(KEY_TYPE, N_SLOTS, SB)

// Single-kernel shmem-combiner path — used for the SMALL tier (≤256 groups)
// with ≤8B keys. For low cardinality + high row count (Q1: 4 groups, 120M
// rows) the per-block shmem combiner avoids the two-kernel global path's
// global-atomic contention (~600M atomicAdds onto 4 slots). SMALL guarantees
// ≤256 distinct keys so the 256-slot shmem table doesn't drop; if a query is
// mis-routed and exceeds it, the overflow counter (threaded through
// producer_local_agg) trips → throw → DuckDB fallback (never a silent drop).
#define MAGI_INSTANTIATE_SINGLE(KEY_TYPE, SB)                                   \
  template __global__ void                                                       \
  distributed_hash_groupby_kernel<KEY_TYPE,                                      \
                                   duckdb::magi_generic::BLOCK_SIZE,             \
                                   KBUFFERING_INTRA_PARTITION_SIZE,              \
                                   KBUFFERING_INTER_PARTITION_SIZE,             \
                                   duckdb::magi_generic::N_LOCAL_SLOTS,          \
                                   duckdb::magi_generic::N_SLOTS_SMALL, SB>(     \
      const std::uint64_t*, std::uint64_t,                                       \
      magi_ops::ColPack,                                                         \
      const magi_ops::KeyFieldEntry*, int,                                       \
      const magi_ops::AggOpEntry*, int,                                          \
      magi_ops::AggSlot64<KEY_TYPE, SB>*, bool, unsigned int*)

// Single-kernel SMALL path: ≤8B keys only; both slot widths (64B for ≤6-slot
// GROUP BYs, 128B for Q1-class).
MAGI_INSTANTIATE_SINGLE(std::int32_t,  64);
MAGI_INSTANTIATE_SINGLE(std::int32_t,  128);
MAGI_INSTANTIATE_SINGLE(std::uint64_t, 64);
MAGI_INSTANTIATE_SINGLE(std::uint64_t, 128);
#undef MAGI_INSTANTIATE_SINGLE

// Two-kernel path: all 3 key kinds × 4 tiers × {64B, 128B} slot.
#define MAGI_INST_ALLTIERS(KEY_TYPE, SB)                                        \
  MAGI_INSTANTIATE_BOTH(KEY_TYPE, duckdb::magi_generic::N_SLOTS_SMALL,  SB);     \
  MAGI_INSTANTIATE_BOTH(KEY_TYPE, duckdb::magi_generic::N_SLOTS_MEDIUM, SB);     \
  MAGI_INSTANTIATE_BOTH(KEY_TYPE, duckdb::magi_generic::N_SLOTS_LARGE,  SB);     \
  MAGI_INSTANTIATE_BOTH(KEY_TYPE, duckdb::magi_generic::N_SLOTS_XLARGE, SB)

MAGI_INST_ALLTIERS(std::int32_t,      64);
MAGI_INSTANTIATE_BOTH(std::int32_t,  duckdb::magi_generic::N_SLOTS_XXLARGE, 64);
MAGI_INSTANTIATE_BOTH(std::uint64_t, duckdb::magi_generic::N_SLOTS_XXLARGE, 64);
MAGI_INSTANTIATE_BOTH(unsigned __int128, duckdb::magi_generic::N_SLOTS_XXLARGE, 64);
MAGI_INSTANTIATE_BOTH(std::int32_t,  duckdb::magi_generic::N_SLOTS_XXXLARGE, 64);
MAGI_INSTANTIATE_SHUFFLE_DIRECT(std::uint64_t, duckdb::magi_generic::N_SLOTS_XXLARGE);
MAGI_INSTANTIATE_SHUFFLE_DIRECT(std::uint64_t, duckdb::magi_generic::N_SLOTS_XXXLARGE);
MAGI_INSTANTIATE_BOTH(std::uint64_t, duckdb::magi_generic::N_SLOTS_XXXLARGE, 64);
MAGI_INSTANTIATE_BOTH(unsigned __int128, duckdb::magi_generic::N_SLOTS_XXXLARGE, 64);
MAGI_INST_ALLTIERS(std::int32_t,      128);
MAGI_INST_ALLTIERS(std::uint64_t,     64);
MAGI_INST_ALLTIERS(std::uint64_t,     128);
MAGI_INST_ALLTIERS(unsigned __int128, 64);
MAGI_INST_ALLTIERS(unsigned __int128, 128);
#undef MAGI_INST_ALLTIERS

#undef MAGI_INSTANTIATE_PREAGG
#undef MAGI_INSTANTIATE_SHUFFLE
#undef MAGI_INSTANTIATE_BOTH

}  // namespace magi

namespace duckdb { namespace magi_generic {

// ── Per-GPU device buffers (lazy-init) ────────────────────────────────────
// `g_agg_dev[i]`     : MAX_TIER_SLOTS × 64B per GPU (XLARGE-tier sized = 256 MB;
//                       smaller tiers just use the prefix).
// `g_ops_dev[i]`     : MAX_AGG_OPS × 4B per GPU; AggOpEntry table copied
//                       per query.
// `g_kfields_dev[i]` : MAX_KEY_FIELDS × 4B per GPU; KeyFieldEntry table
//                       copied per query.
static std::byte*               g_agg_dev    [NUM_GPUS] = { nullptr };  // final (post-shuffle, owned keys)
static std::byte*               g_stage_dev  [NUM_GPUS] = { nullptr };  // H (pre-shuffle per-GPU pre-agg)
static magi_ops::AggOpEntry*    g_ops_dev    [NUM_GPUS] = { nullptr };
static magi_ops::KeyFieldEntry* g_kfields_dev[NUM_GPUS] = { nullptr };
static unsigned int*            g_overflow_dev[NUM_GPUS] = { nullptr };  // 1×u32: rows the hash had to drop
// Overflow d_rows allocation (device_emit results too big for the arena);
// owned by the runtime, freed lazily at the next run's entry.
static AggResultRow*            g_rows_malloc[NUM_GPUS] = { nullptr };
static std::mutex               g_init_mu;

// MAGI_FORCE_GLOBAL=1 forces the two-kernel global-memory path even for the
// SMALL tier — only for A/B perf comparison against the shmem-combiner path.
static const bool g_force_global = [] {
  const char* e = std::getenv("MAGI_FORCE_GLOBAL");
  return e && e[0] == '1';
}();

static void EnsureDeviceBuffers()
{
  std::lock_guard<std::mutex> lk(g_init_mu);
  if (g_agg_dev[0] != nullptr) return;
  // Arenas come out of sirius's CACHE pool (persistent across queries — the
  // per-query ResetBuffer clears only the processing pool). Raw cudaMalloc
  // here used to fight table caching for the residual free memory OUTSIDE
  // the pools: at SF100 that residual was ~0.3GB, the mallocs failed
  // unchecked, and the first kernel through the garbage pointers took the
  // whole process down. gpu_buffer_init calls magi_groupby_prealloc_arenas()
  // right after the pools are reserved, so this normally runs while the
  // cache bump pointer is still 0 and can never fail.
  for (int i = 0; i < NUM_GPUS; ++i) {
    g_agg_dev[i]   = reinterpret_cast<std::byte*>(
      magi_runtime::magi_pool_alloc(agg_buf_bytes(), i, /*persistent=*/true));
    g_stage_dev[i] = reinterpret_cast<std::byte*>(
      magi_runtime::magi_pool_alloc(agg_buf_bytes(), i, true));
    g_ops_dev[i]   = reinterpret_cast<magi_ops::AggOpEntry*>(
      magi_runtime::magi_pool_alloc(sizeof(magi_ops::AggOpEntry) * MAX_AGG_OPS, i, true));
    g_kfields_dev[i] = reinterpret_cast<magi_ops::KeyFieldEntry*>(
      magi_runtime::magi_pool_alloc(sizeof(magi_ops::KeyFieldEntry) * MAX_KEY_FIELDS, i, true));
    g_overflow_dev[i] = reinterpret_cast<unsigned int*>(
      magi_runtime::magi_pool_alloc(sizeof(unsigned int), i, true));
  }
}

// Eager arena carve-out, called from gpu_buffer_init (via
// magi_runtime::magi_prealloc_arenas) before any table is cached.
void magi_groupby_prealloc_arenas() { EnsureDeviceBuffers(); }

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
  // HAVING pushdown predicate, applied in the flush compaction (per-GPU copy;
  // plan-derived so every worker publishes the identical value).
  std::array<SlotPredicate, NUM_GPUS>             having_pred{};
  TableSize                                       table_size_for_this_query = TableSize::SMALL;
  std::uint64_t                                   session_id = 0;
};
static GenericExchange& exchange() { static GenericExchange e; return e; }

// Device-side conversion of the compacted slot buffer (AggSlot64<KeyT,SB>) into a
// flat, non-templated AggResultRow buffer — the on-device analog of the host loop
// below. Lets the caller build output GPUColumns directly on-device (no D2H of the
// slots + host AggResultRow construction), which dominates high-cardinality GROUP BY.
template <typename KeyT, int SB>
__global__ void aggslot_to_resultrow(const magi_ops::AggSlot64<KeyT, SB>* dense,
                                     unsigned live, AggResultRow* out)
{
  unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= live) return;
  if constexpr (sizeof(KeyT) > 8) {
    out[j].key_packed = static_cast<unsigned __int128>(dense[j].key);
  } else {
    out[j].key_packed = static_cast<unsigned __int128>(
        static_cast<typename std::make_unsigned<KeyT>::type>(dense[j].key));
  }
  constexpr int ND = magi_ops::AggSlot64<KeyT, SB>::N_DOUBLES;
  for (int v = 0; v < ND; ++v) out[j].values[v] = dense[j].values[v];
  for (int v = ND; v < AggResultRow::N_VALUES; ++v) out[j].values[v] = 0.0;
  out[j].partial_count = dense[j].partial_count;
}

// ── Per-GPU launch (templated on KeyT and N_SLOTS tier) ───────────────────
// Each per-GPU worker resets its agg slots, launches the generic
// distributed_hash_groupby_kernel (selected by both KeyT and slot tier),
// drains the magi session, and extracts its hash-partitioned slice.
template <typename KeyT, int N_SLOTS, int SB>
static std::size_t run_per_gpu_typed_tier(int                        gpu_id,
                                           std::vector<AggResultRow>& my_slice,
                                           bool                       device_emit,
                                           AggResultRow**             d_rows_out)
{
  auto& xc = exchange();
  const int gpu = magi_runtime::magi_phys_gpu(gpu_id);
  cudaSetDevice(gpu);

  // Coarse per-phase wall timing, enabled by MAGI_PHASE_TIME=1 (one fprintf
  // per GPU per query at the end; no hot-loop instrumentation).
  const bool phase_time = std::getenv("MAGI_PHASE_TIME") != nullptr;
  using pt_clock = std::chrono::steady_clock;
  auto pt_t0 = pt_clock::now();
  auto pt_ms = [](pt_clock::time_point a, pt_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  // Clear receiver global agg (writes empty_key_v<KeyT> per slot — a plain
  // memset to 0 would mis-mark int32 slots as "occupied" since
  // empty_key_v<int32_t> = -1). Only touch the prefix this tier uses.
  if constexpr (magi_ops::empty_key_v<KeyT> == KeyT{}) {
    // empty key is all-zero bits (the __int128 hash kinds), so a zero-filled
    // slot is bit-identical to what init_global_agg_slots writes (AggSlot64{}
    // + empty key). cudaMemset saturates DRAM (~1.5TB/s) where the strided
    // init kernel manages ~290GB/s — on the 320B wide tier (2 × 6M × 320B =
    // 3.8GB) that is the difference between 13ms and ~1ms of per-query setup.
    // int32/uint64 kinds use non-zero sentinels (see empty_key_v) and take
    // the kernel below; their tables are 64/128B-slot tiers, far cheaper.
    const size_t bytes = sizeof(magi_ops::AggSlot64<KeyT, SB>) * (size_t)N_SLOTS;
    cudaMemsetAsync(g_agg_dev[gpu_id],   0, bytes, magi_runtime::magi_stream(gpu_id));
    cudaMemsetAsync(g_stage_dev[gpu_id], 0, bytes, magi_runtime::magi_stream(gpu_id));
  } else {
    auto* agg_typed_for_init =
        reinterpret_cast<magi_ops::AggSlot64<KeyT, SB>*>(g_agg_dev[gpu_id]);
    auto* stage_typed_for_init =
        reinterpret_cast<magi_ops::AggSlot64<KeyT, SB>*>(g_stage_dev[gpu_id]);
    constexpr int INIT_BLOCK = 256;
    const     int init_grid  = (N_SLOTS + INIT_BLOCK - 1) / INIT_BLOCK;
    init_global_agg_slots<KeyT, SB>
        <<<init_grid, INIT_BLOCK, 0, magi_runtime::magi_stream(gpu_id)>>>(
            agg_typed_for_init, N_SLOTS);
    // Stage hash H (producer pre-agg) — same empty-key init.
    init_global_agg_slots<KeyT, SB>
        <<<init_grid, INIT_BLOCK, 0, magi_runtime::magi_stream(gpu_id)>>>(
            stage_typed_for_init, N_SLOTS);
  }
  auto pt_tI = pt_t0;
  if (phase_time) {
    cudaStreamSynchronize(magi_runtime::magi_stream(gpu_id));
    pt_tI = pt_clock::now();
  }

  // Push per-query tables: ops + key_fields.
  cudaMemcpyAsync(g_ops_dev[gpu_id],
                   xc.ops_host_ptrs[gpu_id],
                   sizeof(magi_ops::AggOpEntry) * xc.n_ops[gpu_id],
                   cudaMemcpyHostToDevice,
                   magi_runtime::magi_stream(gpu_id));
  cudaMemcpyAsync(g_kfields_dev[gpu_id],
                   xc.kfields_host_ptrs[gpu_id],
                   sizeof(magi_ops::KeyFieldEntry) * xc.n_key_fields[gpu_id],
                   cudaMemcpyHostToDevice,
                   magi_runtime::magi_stream(gpu_id));
  // CELL mode: the groupby shuffle sends AggSlot64 as CELL_SIZE-byte cells
  // (recv_direct[_self]_cell reassembles them), so the host worker must move
  // CELL_SIZE-sized units. The slot is sizeof(AggSlot64) = N×CELL_SIZE; setting
  // tuple_size = CELL_SIZE lets nbytes = tail_cells × CELL_SIZE come out right
  // for slots wider than one cell (128B). For 64B slots CELL_SIZE == sizeof,
  // so this is unchanged behaviour.
  magi_runtime::magi_set_tuple_size(gpu_id, magi::CELL_SIZE);
  if (phase_time) {
    cudaStreamSynchronize(magi_runtime::magi_stream(gpu_id));
    cudaError_t e = cudaGetLastError();
    std::fprintf(stderr, "[magi-dbg gpu=%d] init+upload: %s\n", gpu_id,
                 cudaGetErrorString(e));
  }
  auto pt_t1 = pt_clock::now();

  // Session bump (thread 0) + sync.
  if (gpu_id == 0) xc.session_id = magi_runtime::magi_bump_session();
  xc.after_session_start.arrive_and_wait();
  auto pt_t2 = pt_clock::now();

  // Launch the generic kernel for (KeyT, N_SLOTS) tier.
  const auto&         in      = xc.inputs[gpu_id];
  std::uint64_t*      row_ids = magi_runtime::GetIdentityRowIdsShared(gpu, in.n_filtered);
  auto pt_t3 = pt_clock::now();

  // Make this GPU's cached aggregate input coherent before the kernel reads it.
  // sirius uploads each GPU's cache slice with a cudaMemcpyAsync issued on a
  // GPU-0 stream (the scan's stream pool lives on device 0), so for a consuming
  // GPU != 0 the slice lands in its DRAM via a cross-device copy whose result is
  // not guaranteed visible to a kernel on that device without a device-side
  // sync. Without this, the first cross-GPU GROUP BY read stale data and
  // produced wrong, run-to-run-varying sums (e.g. Q5 was correct only ~2/8
  // runs; with this sync it is 10/10).
  cudaDeviceSynchronize();
  auto pt_t4 = pt_clock::now();
  auto* global_agg_typed =
      reinterpret_cast<magi_ops::AggSlot64<KeyT, SB>*>(g_agg_dev[gpu_id]);
  auto* stage_typed =
      reinterpret_cast<magi_ops::AggSlot64<KeyT, SB>*>(g_stage_dev[gpu_id]);

  static const bool recv_check_on = std::getenv("MAGI_RECV_CHECK") != nullptr;
  if (recv_check_on) {
    magi_ops::RecvCheckState st{ gpu_id, NUM_GPUS, 0u, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} };
    cudaMemcpyToSymbol(magi_ops::g_recv_check, &st, sizeof(st));
  }

  // Zero the overflow counter; the kernel bumps it whenever a row's key can't
  // claim a global hash slot (i.e. cardinality exceeds the tier). On the same
  // stream as the kernels so ordering is guaranteed.
  cudaMemsetAsync(g_overflow_dev[gpu_id], 0, sizeof(unsigned int),
                  magi_runtime::magi_stream(gpu_id));

  // Path selection:
  //  - SMALL tier (≤256 groups) + ≤8B key → single shmem-combiner kernel. Low
  //    cardinality + high row count (Q1) would otherwise hammer a few global
  //    slots; the per-block shmem combiner makes it ~50× faster.
  //  - everything else (MEDIUM/LARGE high-card, or wide __int128 keys) → the
  //    two-kernel global-memory hashagg.
  // MAGI_FORCE_GLOBAL=1 forces the two-kernel path even for SMALL (A/B perf).
  bool used_single = false;
  auto pt_tA       = pt_t4;
  if constexpr (sizeof(KeyT) <= 8 && N_SLOTS == N_SLOTS_SMALL) {
    if (!g_force_global) {
      used_single = true;
      // Single kernel: producer shmem-combiner → shuffle → final, all at once.
      // Overflow still counted in producer_local_agg → no silent drop.
      magi::distributed_hash_groupby_kernel<KeyT,
                                            BLOCK_SIZE,
                                            KBUFFERING_INTRA_PARTITION_SIZE,
                                            KBUFFERING_INTER_PARTITION_SIZE,
                                            N_LOCAL_SLOTS,
                                            N_SLOTS, SB>
          <<<USER_KERNEL_GRID_SIZE, BLOCK_SIZE, 0,
             magi_runtime::magi_stream(gpu_id)>>>(row_ids,
                                              in.n_filtered,
                                              in.cols,
                                              g_kfields_dev[gpu_id],
                                              xc.n_key_fields[gpu_id],
                                              g_ops_dev[gpu_id],
                                              xc.n_ops[gpu_id],
                                              global_agg_typed,
                                              /*just_load=*/false,
                                              g_overflow_dev[gpu_id]);
    }
  }
  if (!used_single) {
    // Two-kernel global-memory hashagg (see distributed_hash_groupby.cuh). Both
    // Small-session grid: the shuffle/EOF protocol cost scales with the
    // BLOCK COUNT (per-block send rings + system-fenced header handshakes on
    // flush/EOF), not with payload — on a ~25k-row input the fixed 64-block
    // session costs ~2ms while the actual aggregation is <0.1ms (Q2's min
    // groupby). Shrink BOTH kernels' grids for small inputs. The ring
    // geometry is block-symmetric (sender block b fills the peer's ring b),
    // so sender and receiver grids must match ACROSS GPUs: decide from the
    // cross-GPU MAX of n_filtered — xc.inputs[] is published before
    // xc.begin, so every peer computes the same answer. A mismatched grid
    // would strand cells in rings the receiver never polls (hang).
    unsigned session_grid = USER_KERNEL_GRID_SIZE;
    {
      // DEFAULT OFF: shrinking the grid deadlocked the session (the recv
      // side evidently waits on ring state beyond the block-symmetric
      // mapping assumed here — needs a real look at the KBuffering reset /
      // host-worker geometry before this can be enabled). Opt in with
      // MAGI_SMALL_SESSION_GRID=<blocks> for experiments.
      static const long sg_grid = [] {
        const char* e = std::getenv("MAGI_SMALL_SESSION_GRID");
        return e ? std::strtol(e, nullptr, 10) : 0L;
      }();
      static const long sg_rows = [] {
        const char* e = std::getenv("MAGI_SMALL_SESSION_ROWS");
        return e ? std::strtol(e, nullptr, 10) : 200000L;
      }();
      if (sg_grid > 0 && sg_rows > 0) {
        std::uint64_t max_rows = 0;
        for (int i = 0; i < NUM_GPUS; ++i) {
          max_rows = std::max<std::uint64_t>(max_rows, xc.inputs[i].n_filtered);
        }
        if (max_rows <= static_cast<std::uint64_t>(sg_rows)) {
          session_grid = static_cast<unsigned>(sg_grid);
        }
      }
    }
    // run on this GPU's magi stream, so kernel B observes a fully-built H.
    //   A: rows -> per-GPU global hash H (no per-block shmem cap → no drop)
    magi::global_preagg_kernel<KeyT, N_SLOTS, SB>
        <<<session_grid, BLOCK_SIZE, 0,
           magi_runtime::magi_stream(gpu_id)>>>(row_ids,
                                            in.n_filtered,
                                            in.cols,
                                            g_kfields_dev[gpu_id],
                                            xc.n_key_fields[gpu_id],
                                            g_ops_dev[gpu_id],
                                            xc.n_ops[gpu_id],
                                            stage_typed,
                                            /*just_load=*/false,
                                            g_overflow_dev[gpu_id]);
    if (phase_time) {
      cudaStreamSynchronize(magi_runtime::magi_stream(gpu_id));
      pt_tA         = pt_clock::now();
      cudaError_t e = cudaGetLastError();
      std::fprintf(stderr, "[magi-dbg gpu=%d] stageA(global_preagg): %s\n", gpu_id,
                   cudaGetErrorString(e));
    }
    // DIAG(MAGI_STAGE_DUMP): dump the occupied stage slots right after the local
    // pre-agg, BEFORE the shuffle. Splits "pre-agg already wrong" from
    // "shuffle/merge corrupts it" — the SF100 128B-slot groupby is correct on a
    // cold run and garbage on warm ones, so we need to see which stage flips.
    if (std::getenv("MAGI_STAGE_DUMP")) {
      cudaStreamSynchronize(magi_runtime::magi_stream(gpu_id));
      std::vector<magi_ops::AggSlot64<KeyT, SB>> h(N_SLOTS);
      cudaMemcpy(h.data(), stage_typed,
                 sizeof(magi_ops::AggSlot64<KeyT, SB>) * (size_t)N_SLOTS,
                 cudaMemcpyDeviceToHost);
      int shown = 0;
      for (int i = 0; i < N_SLOTS && shown < 6; ++i) {
        if (h[i].key == magi_ops::empty_key_v<KeyT>) continue;
        std::fprintf(stderr,
            "[stage gpu=%d] slot=%d key=%llu v0=%.2f v1=%.2f v2=%.2f pc=%d\n",
            gpu_id, i, (unsigned long long)(std::uint64_t)h[i].key,
            h[i].values[0], h[i].values[1], h[i].values[2], h[i].partial_count);
        ++shown;
      }
    }
    //   B: scan H -> shuffle to owners -> merge into final
    // MAGI_SEND_DIRECT_BATCH=1: benchmark variant driving the send through
    // bounded send_direct batches (q5-style) instead of SendDirectStream —
    // wired only for uint64/64B tiers that have the direct instantiation.
    static const bool send_direct_batch =
        std::getenv("MAGI_SEND_DIRECT_BATCH") != nullptr;
    bool launched_direct = false;
    if constexpr (SB == 64 && std::is_same_v<KeyT, std::uint64_t> &&
                  (N_SLOTS == N_SLOTS_XXLARGE || N_SLOTS == N_SLOTS_XXXLARGE)) {
      if (send_direct_batch) {
        magi::shuffle_global_kernel<KeyT,
                                    KBUFFERING_INTRA_PARTITION_SIZE,
                                    KBUFFERING_INTER_PARTITION_SIZE,
                                    N_SLOTS, SB, true>
            <<<session_grid, BLOCK_SIZE, 0,
               magi_runtime::magi_stream(gpu_id)>>>(stage_typed,
                                                g_ops_dev[gpu_id],
                                                xc.n_ops[gpu_id],
                                                global_agg_typed,
                                                /*just_load=*/false,
                                                g_overflow_dev[gpu_id]);
        launched_direct = true;
      }
    }
    if (!launched_direct) {
      magi::shuffle_global_kernel<KeyT,
                                  KBUFFERING_INTRA_PARTITION_SIZE,
                                  KBUFFERING_INTER_PARTITION_SIZE,
                                  N_SLOTS, SB>
          <<<session_grid, BLOCK_SIZE, 0,
             magi_runtime::magi_stream(gpu_id)>>>(stage_typed,
                                              g_ops_dev[gpu_id],
                                              xc.n_ops[gpu_id],
                                              global_agg_typed,
                                              /*just_load=*/false,
                                              g_overflow_dev[gpu_id]);
    }
  }
  {
    cudaError_t e = cudaGetLastError();
    if (phase_time)
      std::fprintf(stderr, "[magi-dbg gpu=%d] post-launch: %s\n", gpu_id,
                   cudaGetErrorString(e));
  }
  cudaStreamSynchronize(magi_runtime::magi_stream(gpu_id));
  {
    cudaError_t e = cudaGetLastError();
    if (phase_time)
      std::fprintf(stderr, "[magi-dbg gpu=%d] post-sync: %s\n", gpu_id,
                   cudaGetErrorString(e));
  }
  if (std::getenv("MAGI_EOF_PROFILE")) {
    // Per-block EOF-protocol leg durations recorded by receiver_merge_until_eof
    // (clock64 deltas within each block; see g_eof_prof layout).
    unsigned long long prof[64 * 8];
    if (cudaMemcpyFromSymbol(prof, magi_ops::g_eof_prof, sizeof(prof)) ==
        cudaSuccess) {
      static thread_local double us_per_cyc = 0.0;
      if (us_per_cyc == 0.0) {
        int dev = 0, khz = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev);
        us_per_cyc = khz > 0 ? 1000.0 / static_cast<double>(khz) : 0.0;
      }
      double mx[4] = {0, 0, 0, 0}, sum[4] = {0, 0, 0, 0};
      unsigned long long iters = 0, it_flush = 0, it_eof = 0;
      int n = 0;
      for (int b = 0; b < 64; ++b) {
        const unsigned long long* r = &prof[b * 8];
        if (r[0] == 0 || r[4] == 0) continue;
        ++n;
        const double d[4] = {
          r[1] ? (r[1] - r[0]) * us_per_cyc : 0.0,
          r[2] ? (r[2] - r[0]) * us_per_cyc : 0.0,
          r[3] ? (r[3] - r[0]) * us_per_cyc : 0.0,
          (r[4] - r[0]) * us_per_cyc};
        for (int i = 0; i < 4; ++i) {
          if (d[i] > mx[i]) mx[i] = d[i];
          sum[i] += d[i];
        }
        iters += r[5];
        it_flush += r[6];
        it_eof += r[7];
      }
      if (n > 0) {
        std::fprintf(stderr,
                     "[eof-prof gpu=%d] blocks=%d avg/max us: flushed=%.0f/%.0f "
                     "eof_sent=%.0f/%.0f recv_eof=%.0f/%.0f exit=%.0f/%.0f "
                     "iters(avg)=%.0f flush_at=%.0f eof_at=%.0f\n",
                     gpu_id, n, sum[0] / n, mx[0], sum[1] / n, mx[1], sum[2] / n,
                     mx[2], sum[3] / n, mx[3], (double)iters / n,
                     (double)it_flush / n, (double)it_eof / n);
      }
    } else {
      (void)cudaGetLastError();
    }
  }
  auto pt_t5 = pt_clock::now();
  magi_runtime::magi_sync_after_session(gpu_id, xc.session_id);
  auto pt_t6 = pt_clock::now();

  // Read the overflow counter back. A non-zero value means the hash dropped
  // rows (cardinality > N_SLOTS, i.e. beyond the LARGE tier) and this GPU's
  // result is an undercount — defer the throw until after the `end` barrier so
  // peers don't deadlock waiting on a worker that bailed early.
  unsigned int overflow_count = 0;
  cudaMemcpy(&overflow_count, g_overflow_dev[gpu_id], sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  if (recv_check_on) {
    magi_ops::RecvCheckState st{};
    cudaMemcpyFromSymbol(&st, magi_ops::g_recv_check, sizeof(st));
    std::fprintf(stderr,
                 "[recv-check gpu=%d] violations=%u by_site self=%u peer=%u other=%u "
                 "sample_keys=%llu,%llu,%llu,%llu\n",
                 gpu_id, st.violations, st.by_site[0], st.by_site[1],
                 st.by_site[2] + st.by_site[3], st.sample_key[0], st.sample_key[1],
                 st.sample_key[2], st.sample_key[3]);
  }

  // Flush-compaction: GPU-compact the live slots into a dense buffer (reuse
  // g_stage_dev — free once the producer pre-agg is done; counter reuses
  // g_overflow_dev, already read into overflow_count above) so the D2H + host loop
  // are O(live) not O(tier). At XLARGE this avoids a 1 GB D2H + a 16M-slot host
  // scan — the dominant cost of high-cardinality GROUP BY (nsys: ~430ms host-side).
  auto*        agg_d = reinterpret_cast<magi_ops::AggSlot64<KeyT, SB>*>(g_agg_dev[gpu_id]);
  auto*        dense = reinterpret_cast<magi_ops::AggSlot64<KeyT, SB>*>(g_stage_dev[gpu_id]);
  cudaStream_t st    = magi_runtime::magi_stream(gpu_id);
  cudaMemsetAsync(g_overflow_dev[gpu_id], 0, sizeof(unsigned int), st);
  constexpr int CB = 256;
  // DIAG(MAGI_STAGE_DUMP): the FINAL table after the cross-GPU merge, before
  // compaction — paired with the [stage] dump above so a wrong value can be
  // pinned to either the local pre-agg or the shuffle/merge.
  if (std::getenv("MAGI_STAGE_DUMP")) {
    cudaStreamSynchronize(st);
    std::vector<magi_ops::AggSlot64<KeyT, SB>> h(N_SLOTS);
    cudaMemcpy(h.data(), agg_d,
               sizeof(magi_ops::AggSlot64<KeyT, SB>) * (size_t)N_SLOTS,
               cudaMemcpyDeviceToHost);
    int shown = 0;
    for (int i = 0; i < N_SLOTS && shown < 6; ++i) {
      if (h[i].key == magi_ops::empty_key_v<KeyT>) continue;
      std::fprintf(stderr,
          "[final gpu=%d] slot=%d key=%llu v0=%.2f v1=%.2f v2=%.2f pc=%d\n",
          gpu_id, i, (unsigned long long)(std::uint64_t)h[i].key,
          h[i].values[0], h[i].values[1], h[i].values[2], h[i].partial_count);
      ++shown;
    }
  }
  const int     cg = (N_SLOTS + CB - 1) / CB;
  compact_live_slots_kernel<KeyT, SB><<<cg, CB, 0, st>>>(agg_d, dense,
                                                     g_overflow_dev[gpu_id], N_SLOTS,
                                                     exchange().having_pred[gpu_id]);
  unsigned int live = 0;
  cudaMemcpyAsync(&live, g_overflow_dev[gpu_id], sizeof(unsigned int),
                  cudaMemcpyDeviceToHost, st);
  cudaStreamSynchronize(st);
  auto pt_tC = pt_clock::now();
  bool rows_alloc_failed = false;
  if (device_emit) {
    // On-device: convert the compacted slots to a flat device AggResultRow
    // buffer and hand it to the caller (no D2H of slots + host slice
    // construction). The buffer is the FINAL-table arena g_agg_dev — dead once
    // compact copied the live slots into `dense` (g_stage_dev) — so no
    // cudaMalloc/cudaFree on the query path (the malloc alone cost ~2.5ms at
    // Q10's 292MB). The pointer stays valid until the next groupby run
    // re-initializes the arena; the caller consumes it within this query and
    // MUST NOT free it. Falls back to a tracked cudaMalloc only if live rows
    // exceed the arena (>~6.7M live), freed on the next run's entry.
    AggResultRow* d_rows = nullptr;
    if (live > 0) {
      if (static_cast<size_t>(live) * sizeof(AggResultRow) <= agg_buf_bytes()) {
        d_rows = reinterpret_cast<AggResultRow*>(g_agg_dev[gpu_id]);
      } else {
        // Overflow allocation can be huge (AggResultRow is sized for the
        // widest slot: live 37.5M × ~312B ≈ 11.7GB on Q18) and CAN FAIL —
        // unchecked, the conversion kernel then writes through garbage
        // (cudaErrorIllegalAddress poisoning the whole context). Check, and
        // defer the throw past xc.end like the overflow counter — a
        // one-sided throw here would strand the peer at the barrier.
        cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&d_rows),
                   static_cast<size_t>(live) * sizeof(AggResultRow));
        if (e != cudaSuccess || d_rows == nullptr) {
          (void)cudaGetLastError();
          d_rows            = nullptr;
          rows_alloc_failed = true;
        } else {
          g_rows_malloc[gpu_id] = d_rows;
        }
      }
      if (d_rows != nullptr) {
        constexpr int RB = 256;
        const unsigned rg = static_cast<unsigned>((live + RB - 1) / RB);
        aggslot_to_resultrow<KeyT, SB><<<rg, RB, 0, st>>>(dense, live, d_rows);
        cudaStreamSynchronize(st);
      }
    }
    if (d_rows_out) *d_rows_out = d_rows;
  } else {
    std::vector<magi_ops::AggSlot64<KeyT, SB>> host(live);
    cudaMemcpy(host.data(), dense,
               static_cast<size_t>(live) * sizeof(magi_ops::AggSlot64<KeyT, SB>),
               cudaMemcpyDeviceToHost);

    my_slice.clear();
    my_slice.reserve(live);
    for (unsigned int j = 0; j < live; ++j) {
      AggResultRow row{};
      // Widen KeyT to the 128-bit result key. For int32/uint64 zero-extend via
      // the unsigned cast (the caller re-narrows on emit; sign-extension would
      // corrupt high bits for negative keys). 16-byte keys pass through as-is.
      if constexpr (sizeof(KeyT) > 8) {
        row.key_packed = static_cast<unsigned __int128>(host[j].key);
      } else {
        row.key_packed = static_cast<unsigned __int128>(
            static_cast<std::make_unsigned_t<KeyT>>(host[j].key));
      }
      for (int v = 0; v < magi_ops::AggSlot64<KeyT, SB>::N_DOUBLES; ++v) {
        row.values[v] = host[j].values[v];
      }
      row.partial_count = host[j].partial_count;
      my_slice.push_back(row);
    }
  }

  if (phase_time) {
    auto pt_t7 = pt_clock::now();
    std::fprintf(stderr,
      "[magi-phase gpu=%d] setup=%.2f(init=%.2f up=%.2f) barrier=%.2f rowids=%.2f "
      "devsync=%.2f kernels=%.2f(A=%.2f B=%.2f) session=%.2f "
      "extract=%.2f(compact=%.2f rows=%.2f) total=%.2f ms live=%u\n",
      gpu_id, pt_ms(pt_t0, pt_t1), pt_ms(pt_t0, pt_tI), pt_ms(pt_tI, pt_t1),
      pt_ms(pt_t1, pt_t2), pt_ms(pt_t2, pt_t3),
      pt_ms(pt_t3, pt_t4), pt_ms(pt_t4, pt_t5),
      used_single ? 0.0 : pt_ms(pt_t4, pt_tA),
      used_single ? pt_ms(pt_t4, pt_t5) : pt_ms(pt_tA, pt_t5),
      pt_ms(pt_t5, pt_t6),
      pt_ms(pt_t6, pt_t7), pt_ms(pt_t6, pt_tC), pt_ms(pt_tC, pt_t7),
      pt_ms(pt_t0, pt_t7), live);
  }

  xc.end.arrive_and_wait();

  // All collective barriers are done; safe to throw now. A non-zero overflow
  // means this query's group cardinality exceeded the magi hash tables, so the
  // GPU result silently undercounts. Throwing makes gpu_processing fall back to
  // DuckDB (correct) instead of returning wrong numbers. Native high-cardinality
  // support is the Q3 cuco-style hashagg work.
  if (rows_alloc_failed) {
    throw std::runtime_error(
        "magi_groupby: device-emit row buffer allocation failed (live rows too "
        "large); falling back to DuckDB.");
  }
  if (overflow_count != 0) {
    // std::runtime_error (not a duckdb type) keeps this .cu free of duckdb
    // headers — GPUContext::GPUExecuteQuery catches std::exception and falls
    // back to DuckDB, which is exactly the behaviour we want.
    throw std::runtime_error(
        "magi_groupby: group cardinality exceeds hash capacity (dropped " +
        std::to_string(overflow_count) + " rows on GPU " +
        std::to_string(gpu_id) +
        "); falling back to DuckDB. A high-cardinality (cuco-style) global "
        "hashagg is needed to run this on the GPU.");
  }
  return device_emit ? static_cast<std::size_t>(live) : my_slice.size();
}

// Pick the narrowest slot that fits the query's agg-slot count, then run.
// Each output aggregate uses one 8B value slot; a 64B slot holds
// AggSlot64<KeyT,64>::N_DOUBLES of them (6 for ≤8B keys, 5 for the 16B compound
// key). Wider GROUP BYs (e.g. TPC-H Q1 = 9 slots: 4 SUM + 3 AVG + COUNT(*) +
// the shared AVG-COUNT) spill to the 128B slot. The byte budget is derived from
// the key type — no magic constant. SB must be identical on every GPU (they
// shuffle slots to each other); guaranteed since `n_slots` comes from the
// identical per-query ops table.
template <typename KeyT, int N_SLOTS>
static std::size_t run_tier(int gpu_id, int n_slots,
                            std::vector<AggResultRow>& my_slice,
                            bool device_emit, AggResultRow** d_rows_out)
{
  constexpr int N64  = magi_ops::AggSlot64<KeyT, 64>::N_DOUBLES;
  constexpr int N128 = magi_ops::AggSlot64<KeyT, 128>::N_DOUBLES;
  const int sb = (n_slots <= N64) ? 64 : (n_slots <= N128 ? 128 : 320);
  if (std::getenv("MAGI_SB_DEBUG"))
    std::fprintf(stderr, "[magi-sb] gpu=%d n_slots=%d N64=%d -> SB=%dB\n",
                 gpu_id, n_slots, N64, sb);
  if constexpr (N_SLOTS >= duckdb::magi_generic::N_SLOTS_XXLARGE) {
    // XXLARGE is instantiated for 64B slots only (arena cost). n_slots is
    // identical on every GPU (same ops table) so this throw is symmetric.
    if (n_slots > N64) {
      throw std::runtime_error(
        "magi groupby: XXLARGE tier supports at most " + std::to_string(N64) +
        " aggregate slots (falls back)");
    }
    return run_per_gpu_typed_tier<KeyT, N_SLOTS, 64>(gpu_id, my_slice, device_emit, d_rows_out);
  } else {
  if (n_slots <= N64)  return run_per_gpu_typed_tier<KeyT, N_SLOTS, 64 >(gpu_id, my_slice, device_emit, d_rows_out);
  if (n_slots <= N128) return run_per_gpu_typed_tier<KeyT, N_SLOTS, 128>(gpu_id, my_slice, device_emit, d_rows_out);
  // 320B slots (5 cells): wide-key GROUP BY — 128-bit hash key + inline
  // original-key bytes as KEEP_I64 slots. Only instantiated for the 128-bit
  // key kind to keep template bloat down. Uses the fixed WIDE tiers instead of
  // the caller's: XLARGE x 320B would overrun the 2GB slot arena.
  if constexpr (std::is_same_v<KeyT, unsigned __int128>) {
    // Right-size by this GPU's own row count (see N_SLOTS_WIDE_M comment for
    // why per-GPU local choice is legal): stage table needs ≥ n_filtered slots
    // (every local row may be a distinct group), final table needs ≥ owned
    // groups (≈ n_filtered under balanced hash routing). ≤60% worst-case load.
    const std::uint64_t rows = exchange().inputs[gpu_id].n_filtered;
    if (rows <= (std::uint64_t)N_SLOTS_WIDE_M * 3 / 5) {
      return run_per_gpu_typed_tier<KeyT, N_SLOTS_WIDE_M, 320>(gpu_id, my_slice, device_emit, d_rows_out);
    }
    return run_per_gpu_typed_tier<KeyT, N_SLOTS_WIDE, 320>(gpu_id, my_slice, device_emit, d_rows_out);
  } else {
    std::fprintf(stderr, "[magi-generic] %d agg slots need 320B slots, only "
                 "supported for 128-bit keys\n", n_slots);
    return 0;
  }
  }
}

// ── dispatch over (KeyKind, TableSize) × runtime SlotSize ────────────────
static std::size_t dispatch_by_kind_and_tier(int                        gpu_id,
                                              KeyKind                    kk,
                                              TableSize                  ts,
                                              int                        n_slots,
                                              std::vector<AggResultRow>& my_slice,
                                              bool                       device_emit,
                                              AggResultRow**             d_rows_out)
{
  switch (kk) {
    case KeyKind::INT32:
      switch (ts) {
        case TableSize::SMALL:  return run_tier<std::int32_t, N_SLOTS_SMALL >(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::MEDIUM: return run_tier<std::int32_t, N_SLOTS_MEDIUM>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::LARGE:  return run_tier<std::int32_t, N_SLOTS_LARGE >(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::XLARGE: return run_tier<std::int32_t, N_SLOTS_XLARGE>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::XXLARGE: return run_tier<std::int32_t, N_SLOTS_XXLARGE>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::XXXLARGE: return run_tier<std::int32_t, N_SLOTS_XXXLARGE>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
      }
      break;
    case KeyKind::UINT64:
      switch (ts) {
        case TableSize::SMALL:  return run_tier<std::uint64_t, N_SLOTS_SMALL >(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::MEDIUM: return run_tier<std::uint64_t, N_SLOTS_MEDIUM>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::LARGE:  return run_tier<std::uint64_t, N_SLOTS_LARGE >(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::XLARGE: return run_tier<std::uint64_t, N_SLOTS_XLARGE>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::XXLARGE: return run_tier<std::uint64_t, N_SLOTS_XXLARGE>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::XXXLARGE: return run_tier<std::uint64_t, N_SLOTS_XXXLARGE>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
      }
      break;
    case KeyKind::UINT128:
      switch (ts) {
        case TableSize::SMALL:  return run_tier<unsigned __int128, N_SLOTS_SMALL >(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::MEDIUM: return run_tier<unsigned __int128, N_SLOTS_MEDIUM>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::LARGE:  return run_tier<unsigned __int128, N_SLOTS_LARGE >(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::XLARGE: return run_tier<unsigned __int128, N_SLOTS_XLARGE>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::XXLARGE: return run_tier<unsigned __int128, N_SLOTS_XXLARGE>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
        case TableSize::XXXLARGE: return run_tier<unsigned __int128, N_SLOTS_XXXLARGE>(gpu_id, n_slots, my_slice, device_emit, d_rows_out);
      }
      break;
  }
  // Every (KeyKind, TableSize) pair must dispatch above. This used to be a
  // silent `return 0`: a UINT128 XXLARGE query (missing case) returned an
  // EMPTY result with no error — Q20's first standalone repro reported
  // "live=0" through exactly this hole. Throw is cross-GPU consistent (every
  // worker computes the same (kind, tier)) so nobody strands at a barrier.
  throw std::runtime_error(
    "magi groupby: no kernel instantiation for this (key kind, table tier) "
    "(falls back to DuckDB)");
}

// ── Public entry ─────────────────────────────────────────────────────────
std::size_t distributed_hash_groupby_run_per_gpu(
    int                                              gpu_id,
    const PerGpuInputs&                              inputs,
    const std::vector<magi_ops::KeyFieldEntry>&      key_fields,
    const std::vector<magi_ops::AggOpEntry>&         ops,
    KeyKind                                          key_kind,
    TableSize                                        table_size,
    std::vector<AggResultRow>&                       my_slice,
    bool                                             device_emit,
    AggResultRow**                                   d_rows_out,
    const SlotPredicate&                             having_pred)
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

  magi_runtime::MagiInitOnce();
  EnsureDeviceBuffers();

  // Free a leftover oversized d_rows buffer from a previous run (the arena
  // path leaves this null; see device_emit block).
  if (g_rows_malloc[gpu_id] != nullptr) {
    cudaFree(g_rows_malloc[gpu_id]);
    g_rows_malloc[gpu_id] = nullptr;
  }

  auto& xc = exchange();
  xc.inputs[gpu_id]            = inputs;
  xc.having_pred[gpu_id]       = having_pred;
  xc.ops_host_ptrs[gpu_id]     = ops.data();
  xc.n_ops[gpu_id]             = static_cast<int>(ops.size());
  xc.kfields_host_ptrs[gpu_id] = key_fields.data();
  xc.n_key_fields[gpu_id]      = static_cast<int>(key_fields.size());
  if (gpu_id == 0) {
    xc.key_kind_for_this_query   = key_kind;
    xc.table_size_for_this_query = table_size;
  }
  xc.begin.arrive_and_wait();

  // Over-tier guard. Past this barrier every worker sees all xc.inputs, so
  // the cross-GPU MAX gives an identical verdict on every GPU — a one-sided
  // throw would strand the peer at the next barrier. Cardinality beyond the
  // largest tier used to overflow the global hash and DEADLOCK kernel B
  // (Q18 first hit this at ~37.5M partials/GPU before XXLARGE existed).
  //
  // n_filtered is only a cardinality proxy when the caller cudf-pre-agged the
  // input (ShouldCudfPreAgg gates that at tier >= XLARGE, so post-preagg rows
  // ≈ per-GPU distinct groups). Below XLARGE the input is RAW rows — Q1 feeds
  // 148M rows for 4 groups — and the guard must not fire (it silently sent
  // Q1 back to CPU for weeks: every battery showed 1 OVERRIDE-FAIL unread).
  if (table_size >= magi_generic::TableSize::XLARGE) {
    std::uint64_t mx = 0;
    for (int i = 0; i < NUM_GPUS; ++i) {
      mx = std::max<std::uint64_t>(mx, xc.inputs[i].n_filtered);
    }
    if (mx + mx / 3 > (std::uint64_t)max_groupby_slots()) {
      throw std::runtime_error(
        "magi groupby: cardinality estimate exceeds the largest tier "
        "(falls back to DuckDB)");
    }
  }

  // Number of agg value-slots this query uses = max dst_slot_idx + 1 over the
  // ops table (includes the hidden AVG COUNT carrier). Drives the 64B-vs-128B
  // slot choice in run_tier. Identical on every GPU (same ops table) so the
  // shuffle slot width matches across peers.
  int n_slots = 0;
  for (const auto& op : ops) {
    const int s = static_cast<int>(op.dst_slot_idx) + 1;
    if (s > n_slots) n_slots = s;
  }

  return dispatch_by_kind_and_tier(gpu_id,
                                    xc.key_kind_for_this_query,
                                    xc.table_size_for_this_query,
                                    n_slots,
                                    my_slice,
                                    device_emit,
                                    d_rows_out);
}

}}  // namespace duckdb::magi_generic
