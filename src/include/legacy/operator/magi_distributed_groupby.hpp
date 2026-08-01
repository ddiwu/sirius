// magi_distributed_groupby.hpp — public surface of the generic magi groupby
// runtime (`src/legacy/cuda/magi/magi_groupby_runtime.cu`).
//
// One header, one entry point. Sirius's GROUP BY operator calls
// `distributed_hash_groupby_run_per_gpu` for every grouped aggregate that
// reaches the magi path — there are no per-query Q1/Q5/Q3 entries, and no
// shape predicates. The caller supplies:
//   - KeyKind  (which atomic-width KeyT to template the kernel on)
//   - KeyFieldEntry[] (how to pack input columns into that KeyT)
//   - AggOpEntry[]    (which input cols → which slot fields)
//   - TableSize       (receiver hash table capacity tier)
// Adding a query = dispatcher in magi_groupby.cu picks these four; no
// device-side code change needed.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "operator/magi_groupby.hpp"
#include "data_plane/ops/agg_kinds.cuh"
#include "data_plane/ops/agg_slot.cuh"
#include "data_plane/ops/col_pack.cuh"

#include "legacy/operator/magi_runtime_shared.hpp"  // for magi_runtime::NUM_GPUS

namespace duckdb {
namespace magi_generic {

// Which key-type instantiation of `distributed_hash_groupby_kernel<KeyT,…>`
// the runtime should launch. The KeyT decides the shmem hash atomicCAS
// width; the packing of bytes into that KeyT is data (KeyFieldEntry[]),
// not code.
enum class KeyKind : std::int8_t {
  INT32   = 0,  // 32-bit packed key (Q1: 2×VARCHAR(1), or 1 INT)
  UINT64  = 1,  // 64-bit packed key (Q5: VARCHAR prefix, or BIGINT, or compound)
  UINT128 = 2,  // 128-bit packed key (Q3: bigint+date+int 16B compound)
};

// Receiver global-hash table capacity tier. Pre-instantiated per (KeyT,size)
// pair so the dispatcher can pick at runtime without recompilation.
//
//   SMALL  — 256    slots, fits Q1/Q5/Q9-class   (cardinality ≤ ~200)
//   MEDIUM — 16 K   slots, fits Q16/Q18-class    (cardinality ≤ ~12 K)
//   LARGE  — 1 M    slots, fits Q3-sf50 / Q11    (cardinality ≤ ~800 K)
//   XLARGE — 4 M    slots, fits Q3-sf100         (cardinality ≤ ~3 M)
//
// XLARGE costs 256 MB / GPU of always-allocated buffer (we keep the buffer at
// the largest tier and only cudaMemcpy as many slots as the chosen tier needs).
// PickTableSize routes low-cardinality BIGINT-keyed queries (Q11) down to the
// small tiers, so only genuinely high-card queries pay the big-tier copyback.
enum class TableSize : std::int8_t {
  SMALL   = 0,
  MEDIUM  = 1,
  LARGE   = 2,
  XLARGE  = 3,
  XXLARGE = 4,
  XXXLARGE = 5,   // JOIN only (narrow JoinSlot); the groupby path stops at XXLARGE
};

// Tier slot counts. Single source of truth shared by the cardinality→tier
// routing policy (PickTableSize, magi_groupby.cu) and the kernel
// instantiations / dispatch (magi_groupby_runtime.cu) so they cannot drift.
using SlotPredicate = ::duckdb::magi_groupby::SlotPredicate;

constexpr int N_SLOTS_SMALL  = 256;
constexpr int N_SLOTS_MEDIUM = 16 * 1024;
constexpr int N_SLOTS_LARGE  = 1024 * 1024;
// XLARGE bumped 4M->16M so high-cardinality GROUP BYs (e.g. l_partkey ~10M
// distinct) fit BOTH the per-GPU local hash H (~10M) AND the post-shuffle
// receiver merge table (~card/NUM_GPUS) without overflowing -> previously these
// threw "cardinality exceeds hash capacity" and fell back to DuckDB CPU. Cost:
// g_agg_dev/g_stage_dev grow to 16M*64B = 1 GB each per GPU (always allocated).
// Must stay a power of two (open-addressing uses `& (N_SLOTS-1)`); 16M = 2^24.
constexpr int N_SLOTS_XLARGE = 16 * 1024 * 1024;
// XXLARGE (64M = 2^26): Q18-class cardinality — group by l_orderkey leaves
// ~37.5M partials PER GPU after the cudf local pre-agg (est ×1.33 ≈ 50M).
// Beyond the tier the global hash overflowed and kernel B DEADLOCKED (the
// overflow drain never reached EOF); Run() now throws a cross-GPU-consistent
// NotImplemented beyond XXLARGE instead. 64B slots only (≤6 agg values) —
// a 128B XXLARGE table would double the always-allocated arena again.
constexpr int N_SLOTS_XXLARGE = 64 * 1024 * 1024;
// XXXLARGE (128M = 2^27): only reachable by the JOIN path, whose table is the
// narrow JoinSlot (key + a 4-byte payload index) rather than a 64B AggSlot64.
// Sized so the ALWAYS-RESIDENT arena stays at the 4 GB it was before the slot
// shrank (128M * 32B for the widest key); 256M slots would double it to 8 GB and
// the SF100 pools cannot spare that — the arena allocation itself then stalls
// long before any join runs. Doubling the slot count while keeping the arena
// constant is exactly what the narrow slot bought.
// Needed for build sides like TPC-H SF100 Q9's orders (150M rows -> ~100M keys
// per GPU with headroom), which no AggSlot64 tier could hold.
constexpr int N_SLOTS_XXXLARGE = 128 * 1024 * 1024;

// The GROUPBY path may also use XXXLARGE (Q18@SF100 groups by l_orderkey:
// ~75M partials/GPU, over the 64M XXLARGE ceiling), but its slots stay 64B,
// so enabling it doubles the two always-resident agg arenas (4 GB -> 8 GB
// each). That is affordable inside the SF100 protocol pools (52 GB cache)
// but eats cache headroom at smaller inits — this switch is the escape
// hatch: MAGI_GROUPBY_XXXL=0 restores the XXLARGE ceiling and the 4 GB
// arenas without a rebuild.
// DEFAULT OFF since the narrow XXXLARGE receiver landed: pre-aggregated
// ≤8B-key queries (the only shapes that reach XXXLARGE in practice) use
// per-query pool memory instead, and the resident arenas stay at BASE
// (4 GB). MAGI_GROUPBY_XXXL=1 re-grows them for the remaining fat-XXXLARGE
// shapes (u128 keys / forced producer-hash path).
inline bool GroupbyXxxlEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("MAGI_GROUPBY_XXXL");
    return e && e[0] == '1';
  }();
  return on;
}

// One worker thread's input: how many rows + the packed column views the
// kernel will read. ColPack column ordering must match the KeyFieldEntry[]
// src_col_idx indices and the AggOpEntry[] src_col_idx values the caller
// passes.
struct PerGpuInputs {
  std::uint64_t      n_filtered;
  magi_ops::ColPack  cols;
  // Input is already one row per key (cudf pre-agg ran): the runtime skips
  // the producer hash and streams rows directly (shuffle_direct).
  bool               preagged = false;
};

// One output row per surviving open-addressing slot on this GPU's receiver.
// `key_packed` carries the packed KeyT widened to 128 bits (int32/uint64 zero-
// extended via unsigned cast; uint64 identity in the low 64; 16-byte compound
// keys use the full width). The host unpacks each group-by column out of it by
// the same byte offsets the kernel packed with. `values[]` are the raw 8-byte
// slots — the caller knows which AggKind each holds (it built the AggOpEntry
// table) so it can reinterpret per kind.
struct AggResultRow {
  // Sized for the WIDEST slot (128B → 14 doubles) so this host-side row holds the
  // result of whichever (64B/128B) GPU slot the dynamic dispatch picked; the
  // runtime copies only the active slot's N_DOUBLES prefix.
  static constexpr int N_VALUES = magi_ops::AggSlot64<unsigned __int128, 320>::N_DOUBLES;

  unsigned __int128 key_packed;
  double            values[N_VALUES];
  std::uint64_t     partial_count;
};

// Per-GPU dispatcher entry. Called by every sirius worker thread (one per
// legacy GPU) — threads synchronise internally via a singleton barrier,
// just like `Q5MagiRunPerGpu`. Lazily initialises the magi runtime on the
// first call from any query. `my_slice` is cleared and filled with this
// GPU's hash-partitioned surviving rows on return.
std::size_t distributed_hash_groupby_run_per_gpu(
    int                                              gpu_id,
    const PerGpuInputs&                              inputs,
    const std::vector<magi_ops::KeyFieldEntry>&      key_fields,
    const std::vector<magi_ops::AggOpEntry>&         ops,
    KeyKind                                          key_kind,
    TableSize                                        table_size,
    std::vector<AggResultRow>&                       my_slice,
    // device_emit=true: skip the host slice build and instead return (via
    // d_rows_out) a device AggResultRow buffer for on-device column emit.
    bool                                             device_emit = false,
    AggResultRow**                                   d_rows_out  = nullptr,
    // HAVING pushdown predicate applied in the flush compaction (default off).
    const SlotPredicate&                             having_pred = {});

// Number of GPUs the runtime expects, mirrored from
// `magi_runtime::NUM_GPUS` (= `kSiriusLegacyNumGpus`). Exposed so sirius
// callers can size per-GPU vectors without instantiating the runtime.
constexpr int NUM_GPUS = magi_runtime::NUM_GPUS;

}  // namespace magi_generic
}  // namespace duckdb
