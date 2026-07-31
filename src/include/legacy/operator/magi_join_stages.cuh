// magi_join_stages.cuh — device-side stages for the generic magi shuffle join.
//
// Structure mirrors Magi-Dev's own join reference
// (src/data_plane/tests/scenarios/multisession_join.cuh): the generic
// `send_direct` ships wire tuples, and the receiver action is written inline as
// a lambda passed to the lambda-based drains `recv_direct_self_drain` /
// `recv_direct_drain`. We extended send_direct to take a partition functor as a
// trailing argument (default preserves the old `tuple.suppkey` routing), so this
// join routes by the join key via a lambda — no field-aliasing hack (suppkey
// stays a real, meaningful column in sirius).
//
// Difference from the multisession_join benchmark: it uses a direct-mapped
// membership byte table `build_table[k & (BUILD_SIZE-1)]`, which collides
// (false positives) — fine for a benchmark, not for SQL correctness. We instead
// insert/probe an OPEN-ADDRESSING hash `AggSlot64<KeyT>` via
// global_find_or_insert (linear-probe, key stored → exact).
//
// Phase 1 (BUILD) lives here:
//   join_pack_build_kernel  build rows (ColPack)         -> build wire tuples[]
//   join_build_kernel       send wire tuples (by key) ->| owner global_find_or_insert
//                           + drain incoming during back-pressure + flush/EOF   -> H_build
// The probe phase is added next.
//
// v1 scope: INNER join, UNIQUE build key on the join key (each H_build slot
// claimed once; "store" needs no merge). Q11's supplier/nation are unique on
// their join keys, and Q11's build side is a pure semi-join set (no payload).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "data_plane/config.cuh"
#include "data_plane/api/core.cuh"
#include "data_plane/ops/agg_slot.cuh"
#include "data_plane/ops/col_pack.cuh"
#include "data_plane/ops/groupby_stages.cuh"  // global_find_or_insert

#include "legacy/operator/magi_distributed_join.hpp"  // JoinResultRow
#include "legacy/operator/magi_fused_agg.hpp"          // FusedAggProgram + device VM

#include <utility>              // std::declval (cuco ref type deduction)
#include <cuco/static_map.cuh>  // Option A: cudf-quality find inside the persistent kernel

namespace duckdb {
namespace magi_join {

// ── cuco hash map for the shuffle-join probe (Option A) ─────────────────────
// Replaces the hand-rolled open-addressing global_find (DRAM-latency-bound at
// ~1M probe/s) with cuco's bucketized find — the exact map cudf's hash join
// uses (~1e9 probe/s). Key is packed to <=8B uint64 (cuco caps keys at 8B);
// value = the H_build slot index, so the existing emit path reuses
// H_build[slot].values for the build payload. scalar probing (cg_size=1) so it
// drops straight into the 1-thread-per-tuple recv drain — no magi-core change.
using CucoMap = cuco::static_map<
    std::uint64_t, std::int32_t, cuco::extent<std::size_t>,
    cuda::thread_scope_device, cuda::std::equal_to<std::uint64_t>,
    cuco::linear_probing<1, cuco::default_hash_function<std::uint64_t>>,
    cuco::cuda_allocator<cuco::pair<std::uint64_t, std::int32_t>>,
    cuco::storage<1>>;
using CucoFindRef   = decltype(std::declval<CucoMap&>().ref(cuco::find));
using CucoInsertRef = decltype(std::declval<CucoMap&>().ref(cuco::insert));

// Max payload columns carried inline on a wire tuple (and in JoinResultRow).
// Shared by the build and probe wires. Real max usage is 5 (Q9's probe side:
// l_extendedprice/discount/quantity, ps_supplycost, s_nationkey). The wire key
// is a fixed 16-byte `unsigned __int128` so a single wire layout serves every
// key kind (INT32/UINT64 use the low bytes; compound 16B keys like Q-custom's
// (l_partkey, l_suppkey) use the full width) — that plus the 24B header (key16
// + index4 + pad4) leaves exactly 5 payload slots at 64B (= CELL_SIZE).
constexpr int JOIN_MAX_PAYLOAD = 5;

// Fused join+grouped-agg (scheme ①): when the join feeds a GROUP BY whose key is
// carried in the probe (e.g. custom-Q's ps_suppkey == l_suppkey, a component of
// the compound join key), the probe aggregates each match directly into this
// per-owner group table INSTEAD of materializing a JoinResultRow — killing the
// 16.8GB out_buf write + the slot->values read, and overlapping the aggregate
// with the shuffle (it runs inside the persistent recv drain). 1M slots handles
// ~700K groups (SF50 has 500K suppliers).
constexpr int GROUP_N_SLOTS = 1 << 20;

// Build-side wire tuple, 64B. `key` FIRST so the 16-byte compound key is
// naturally 16-aligned with no interior padding (same layout discipline as
// AggSlot64). `index` is magi's EOF/validity marker (receiver checks
// index != -1). `payload` carries the build-side output columns (the join's
// RHS output) as raw 8-byte values; the receiver stores them into the H_build
// slot so the probe can emit them on a match.
struct alignas(16) JoinBuildWire {
  unsigned __int128 key;   // full join key (routed by the partition functor, hashed into H_build)
  int32_t index;           // 1 = valid; magi EOF protocol uses -1
  int32_t _pad;
  int64_t payload[JOIN_MAX_PAYLOAD];   // build payload (RHS output cols), raw 8B
};
static_assert(sizeof(JoinBuildWire) == 64, "JoinBuildWire must be 64B");

// Receiver/partition actions are functor structs (not lambdas): the legacy
// build does not enable nvcc --extended-lambda, and functors are the idiom the
// data_plane already uses (cf. magi::PartBySuppkey). The drains
// (recv_direct_*_drain) and send_direct accept any callable ProcessFn/PartFn.

// ── Narrow build slot: key + an index into a dense payload array ────────────
// The build table used to be AggSlot64, i.e. the key plus room for JOIN_MAX_PAYLOAD
// doubles — 64 bytes per slot, reserved in EVERY slot including the empty ones.
// That capped the table at 64M slots for a 4 GB arena, which is why a 150M-row
// build side (TPC-H SF100 Q9's orders) could not fit: 75M keys per GPU against
// 64M slots cannot be held even at 100% load.
//
// The payload does not have to live in the slot. It arrives in the build wire
// tuple, and the wire buffers are recycled by the channel, so the receiver must
// copy it somewhere — but "somewhere" can be a DENSE array appended in arrival
// order, sized by the number of rows received rather than by the number of
// slots. The slot then only needs the key and a 4-byte index into that array.
//
//   uint64 key : 8 + 4 + 4  = 16 B  (4x more capacity per byte than AggSlot64)
//   u128 key   : 16 + 4 + 4 = 24 B, padded to 32 B by the 16-byte alignment
//
// Everything stays on the owner GPU: the shuffle already moved the row here, so
// the probe reads a local array, never a peer's memory.
template <typename KeyT>
struct alignas(sizeof(KeyT) > 8 ? 16 : 8) JoinSlot {
  KeyT    key;
  int32_t index;        // wide-key claim state: 0=empty, -1=claiming, 1=valid
  int32_t payload_idx;  // -1 = no payload stored
};

// Claim/insert in the narrow table. Deliberately a copy of magi_ops::
// global_find_or_insert's protocol rather than a call into it: that one is
// typed on AggSlot64 and shared with the groupby path, whose slot layout is
// constrained by the channel's cell size. The HASH below must stay
// byte-identical to global_find (and to groupby_stages.cuh's), or build and
// probe map the same key to different slots and every match is missed.
template <typename KeyT, int N_SLOTS>
__device__ __forceinline__ JoinSlot<KeyT>*
join_find_or_insert(JoinSlot<KeyT>* __restrict__ H, KeyT key,
                    unsigned int* __restrict__ overflow)
{
  if (overflow != nullptr && *overflow != 0u) return nullptr;  // table already full
  unsigned int h;
  if constexpr (sizeof(KeyT) > 8) {
    const unsigned __int128 ku = static_cast<unsigned __int128>(key);
    const std::uint64_t lo = static_cast<std::uint64_t>(ku);
    const std::uint64_t hi = static_cast<std::uint64_t>(ku >> 64);
    std::uint64_t z = lo * 0x9E3779B97F4A7C15ull ^ hi;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z =  z ^ (z >> 31);
    h = static_cast<unsigned int>(z);
  } else {
    const std::uint64_t k64 =
        static_cast<std::uint64_t>(static_cast<std::make_unsigned_t<KeyT>>(key));
    h = static_cast<unsigned int>(k64 ^ (k64 >> 32)) * 2654435761u;
  }
  const int start = static_cast<int>(h) & (N_SLOTS - 1);
  // Bounded probe: scanning all N_SLOTS on a full table turns "does not fit"
  // into a hang instead of an error (see the SF100 Q9 note above).
  const int max_probe = N_SLOTS < 4096 ? N_SLOTS : 4096;
  for (int probe = 0; probe < max_probe; ++probe) {
    const int idx = (start + probe) & (N_SLOTS - 1);
    if constexpr (sizeof(KeyT) > 8) {
      const int prev = atomicCAS(&H[idx].index, 0, -1);   // claim empty?
      if (prev == 0) {
        H[idx].key = key;                                 // 16B plain store
        __threadfence();
        atomicExch(&H[idx].index, 1);                     // publish
        return &H[idx];
      }
      for (unsigned ns = 8; atomicAdd(&H[idx].index, 0) != 1; ) {
        __nanosleep(ns);                                  // backoff, else the
        if (ns < 256) ns <<= 1;                           // claimer starves
      }
      __threadfence();
      if (H[idx].key == key) return &H[idx];
    } else if constexpr (sizeof(KeyT) == 4) {
      const int existing = atomicCAS(reinterpret_cast<int*>(&H[idx].key),
                                     static_cast<int>(magi_ops::empty_key_v<KeyT>),
                                     static_cast<int>(key));
      if (existing == static_cast<int>(magi_ops::empty_key_v<KeyT>) ||
          existing == static_cast<int>(key)) return &H[idx];
    } else {
      const unsigned long long existing = atomicCAS(
          reinterpret_cast<unsigned long long*>(&H[idx].key),
          static_cast<unsigned long long>(magi_ops::empty_key_v<KeyT>),
          static_cast<unsigned long long>(key));
      if (existing == static_cast<unsigned long long>(magi_ops::empty_key_v<KeyT>) ||
          existing == static_cast<unsigned long long>(key)) return &H[idx];
    }
  }
  if (overflow != nullptr) atomicAdd(overflow, 1u);
  return nullptr;
}

// Receiver action (build): hash the incoming key into H_build, append this row's
// payload to the dense array, and record its index in the claimed slot.
// Unique build key → each slot is claimed once (first writer stores).
template <typename KeyT, int N_SLOTS>
struct BuildInsert {
  JoinSlot<KeyT>* H_build;
  double*         payload_arr;   // n_rows * n_build_pl doubles, dense
  unsigned int*   payload_count;  // bump allocator over payload_arr
  unsigned int    payload_cap;    // rows the array can hold
  int             n_build_pl;
  unsigned int*   overflow;
  __device__ __forceinline__ void operator()(JoinBuildWire& t) const {
    if (t.index == -1) return;
    JoinSlot<KeyT>* slot =
        join_find_or_insert<KeyT, N_SLOTS>(H_build, static_cast<KeyT>(t.key), overflow);
    if (slot == nullptr) return;
    if (n_build_pl <= 0 || payload_arr == nullptr) { slot->payload_idx = -1; return; }
    const unsigned pos = atomicAdd(payload_count, 1u);
    if (pos >= payload_cap) {                 // never expected: cap = total build rows
      if (overflow != nullptr) atomicAdd(overflow, 1u);
      slot->payload_idx = -1;
      return;
    }
    double* dst = payload_arr + static_cast<std::size_t>(pos) * n_build_pl;
    for (int i = 0; i < n_build_pl && i < JOIN_MAX_PAYLOAD; ++i) {
      double d; __builtin_memcpy(&d, &t.payload[i], sizeof(double));
      dst[i] = d;
    }
    slot->payload_idx = static_cast<int32_t>(pos);
  }
};

// Send-side partition: route each wire tuple by its join key. Build and probe
// MUST use the identical function so equal keys land on the same owner GPU.
// Templated on the wire type (both JoinBuildWire and JoinProbeWire have `.key`).
template <typename WireT>
struct JoinKeyPartition {
  __device__ __forceinline__ int operator()(const WireT& t,
                                             int partitions_count) const {
    return static_cast<int>(static_cast<std::uint64_t>(t.key) %
                            static_cast<std::uint64_t>(partitions_count));
  }
};

// ── Probe-side wire tuple ───────────────────────────────────────────────────
// Carries the join key (routing + lookup) plus the probe payload columns that
// flow to the join output, packed as raw 8-byte values (host re-narrows by the
// JoinPayloadEntry table). 48B, 16-aligned.
struct alignas(16) JoinProbeWire {
  unsigned __int128 key;   // full join key (16B; low bytes for narrow kinds)
  int32_t index;           // 1 = valid; magi EOF protocol uses -1
  int32_t _pad;
  int64_t payload[JOIN_MAX_PAYLOAD];
};
static_assert(sizeof(JoinProbeWire) == 64, "JoinProbeWire must be 64B");

// Inputs the send loop needs to pack rows itself, instead of reading a wire
// array that a separate kernel materialized first. `scratch` is a per-block
// staging area of `chunk` wire tuples (grid * chunk * sizeof(WireT) total);
// scratch == nullptr selects the old pre-packed path.
struct WirePackSrc {
  const std::uint64_t*                          row_ids;
  std::uint64_t                                 n_rows;
  magi_ops::ColPack                             cols;
  const magi_ops::KeyFieldEntry*                key_fields;
  const duckdb::magi_generic::JoinPayloadEntry* pl;
  void*                                         scratch;
  int                                           n_key_fields;
  int                                           n_pl;
  int                                           chunk;
};


// Find-only open-addressing probe of H_build (never claims a slot). Returns the
// matching slot or nullptr. v1 supports <=8B keys (Q11's join key is int32).
template <typename KeyT, int N_SLOTS>
__device__ __forceinline__ JoinSlot<KeyT>*
global_find(JoinSlot<KeyT>* __restrict__ H, KeyT key)
{
  // Hash MUST match magi_ops::global_find_or_insert (build side) or the probe
  // lands in a different slot and every match is missed. For 16B compound keys
  // that means folding BOTH halves (lo^hi), exactly as the build path does —
  // a low64-only hash would ignore the high 8 bytes of e.g. (partkey,suppkey).
  unsigned int h;
  if constexpr (sizeof(KeyT) > 8) {
    const unsigned __int128 ku = static_cast<unsigned __int128>(key);
    const std::uint64_t lo = static_cast<std::uint64_t>(ku);
    const std::uint64_t hi = static_cast<std::uint64_t>(ku >> 64);
    // Mix lo and hi INDEPENDENTLY (splitmix64 finalizer over lo*ODD ^ hi), not a
    // plain lo^hi: correlated compound keys like TPC-H (partkey,suppkey) collapse
    // under xor into few buckets → O(n) probe chains (the 45s custom-join case).
    // MUST stay byte-identical to global_find_or_insert (groupby_stages.cuh) or
    // build and probe hash a key to different slots and every match is missed.
    std::uint64_t z = lo * 0x9E3779B97F4A7C15ull ^ hi;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z =  z ^ (z >> 31);
    h = static_cast<unsigned int>(z);
  } else {
    const std::uint64_t k64 =
        static_cast<std::uint64_t>(static_cast<std::make_unsigned_t<KeyT>>(key));
    h = static_cast<unsigned int>(k64 ^ (k64 >> 32)) * 2654435761u;
  }
  const int start = static_cast<int>(h) & (N_SLOTS - 1);
  // Same bound as the insert side: a miss must not scan the whole table.
  const int max_probe = N_SLOTS < 4096 ? N_SLOTS : 4096;
  for (int probe = 0; probe < max_probe; ++probe) {
    const int idx = (start + probe) & (N_SLOTS - 1);
    if constexpr (sizeof(KeyT) > 8) {
      // Wide keys are published via `index`; the key field of an unclaimed slot
      // is whatever join_clear_slots wrote, so test the claim state, not the key.
      if (H[idx].index == 0) return nullptr;                     // never claimed → miss
      if (H[idx].key == key) return &H[idx];                     // hit
    } else {
      const KeyT slot_key = H[idx].key;
      if (slot_key == magi_ops::empty_key_v<KeyT>) return nullptr;
      if (slot_key == key)                          return &H[idx];
    }
  }
  return nullptr;
}

// Receiver action (probe): look up H_build; on a hit emit one JoinResultRow
// (build payload from the matched slot + this probe tuple's payload). `out_cap`
// guards under-allocation (drops + bumps overflow once cap is reached).
template <typename KeyT, int N_SLOTS>
struct ProbeEmit {
  JoinSlot<KeyT>*                      H_build;
  const double*                        build_payload;  // 密集数组,按 slot->payload_idx 索引
  duckdb::magi_generic::JoinResultRow* out;
  unsigned int*                        out_count;
  unsigned int                         out_cap;
  int                                  n_build_pl;
  unsigned int*                        overflow;
  unsigned int*                        n_drained;   // debug: count every drained probe tuple
  // Fused join->aggregate (Option 3): when agg_prog != nullptr, evaluate each
  // aggregate-input expression on this match and accumulate directly into the
  // per-GPU `agg_accum` — overlapping the agg with the shuffle, no out_buf needed.
  double*                                       agg_accum;
  const duckdb::magi_fused::FusedAggProgram*    agg_prog;
  // Option A: when non-null, look the key up in the cuco map (cudf-quality) and
  // read the build payload from H_build at the returned slot index; else use
  // the hand-rolled open-addressing find. Only wired for the uint64 key path.
  const CucoFindRef*                            cuco_ref;
  // Fused grouped-agg (scheme ①): when group_tbl != nullptr, a match aggregates
  // into the group table (group key = high 64 bits of the compound join key =
  // l_suppkey; aggs = COUNT + SUM(probe payload[1])) instead of emitting to
  // out_buf. group_key_hi selects which half of the 16B key is the group key.
  magi_ops::AggSlot64<std::uint64_t>*           group_tbl;
  unsigned int*                                 group_overflow;
  int                                           fuse_sum_idx;
  __device__ __forceinline__ void operator()(JoinProbeWire& t) const {
    // No early returns before the warp ballot below: every active lane must
    // reach it (non-matching lanes vote 0). We compute a `do_emit` flag + slot,
    // then one atomicAdd(out_count) per WARP reserves a contiguous run so the
    // 112B JoinResultRow stores land coalesced (base+rank) instead of scattering
    // — the 150M-row full-join emit was atomic-serialized + DRAM-scattered
    // (~5.3s); warp aggregation cuts atomics 32× and coalesces the writes.
    const unsigned active = __activemask();
    bool                        do_emit = false;
    JoinSlot<KeyT>*             slot    = nullptr;
    if (t.index != -1) {
      if (n_drained) atomicAdd(n_drained, 1u);
      if (cuco_ref != nullptr) {
        // cuco caps keys at 8B; a 16B compound (partkey,suppkey) whose halves
        // fit 32 bits packs losslessly into one uint64 (partkey lo32 | suppkey
        // lo32<<32). MUST match cuco_insert_from_hbuild's packing.
        std::uint64_t ck;
        if constexpr (sizeof(KeyT) > 8) {
          const unsigned __int128 ku = static_cast<unsigned __int128>(t.key);
          ck = static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<std::uint64_t>(ku))) |
               (static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<std::uint64_t>(ku >> 64))) << 32);
        } else {
          ck = static_cast<std::uint64_t>(t.key);
        }
        auto it = cuco_ref->find(ck);
        if (it != cuco_ref->end()) slot = &H_build[(*it).second];
      } else {
        slot = global_find<KeyT, N_SLOTS>(H_build, static_cast<KeyT>(t.key));
      }
      if (slot != nullptr) {
        if (group_tbl != nullptr) {
          // FUSED GROUPED AGG (scheme ①): group by the high 64 bits of the
          // compound join key (l_suppkey), aggregate inline into the per-owner
          // group table. Runs inside the recv drain so it overlaps the shuffle;
          // no slot->values read, no out_buf write. This lane votes 0 below.
          const std::uint64_t gkey =
              static_cast<std::uint64_t>(static_cast<unsigned __int128>(t.key) >> 64);
          auto* g = magi_ops::global_find_or_insert<std::uint64_t, GROUP_N_SLOTS>(
              group_tbl, gkey, group_overflow);
          if (g != nullptr) {
            atomicAdd(&g->values[0], 1.0);                              // COUNT(*)
            // The payload carries RAW 8-byte column bits; a DECIMAL column is an
            // int64, so sum it as int64 (exact) — reading those bits as a double
            // gives denormals that sum to 0.0, which is what the first cut did.
            const long long sv = static_cast<long long>(t.payload[fuse_sum_idx]);
            atomicAdd(reinterpret_cast<unsigned long long*>(&g->values[1]),
                      static_cast<unsigned long long>(sv));             // SUM(col)
          } else if (group_overflow != nullptr) {
            atomicAdd(group_overflow, 1u);                              // table full
          }
        } else if (agg_prog != nullptr) {
          // Fused ungrouped agg (Option 3): consume the match into agg_accum,
          // no out_buf emit (this lane votes 0 in the ballot below).
          double probe_d[JOIN_MAX_PAYLOAD];
#pragma unroll
          for (int i = 0; i < JOIN_MAX_PAYLOAD; ++i)
            __builtin_memcpy(&probe_d[i], &t.payload[i], sizeof(double));
          for (int a = 0; a < agg_prog->n_aggs; ++a) {
            const double* bvals =
                (build_payload != nullptr && slot->payload_idx >= 0)
                    ? build_payload + (std::size_t)slot->payload_idx * n_build_pl
                    : nullptr;
            double v = (agg_prog->kind[a] == duckdb::magi_fused::AGG_COUNT)
                           ? 1.0
                           : duckdb::magi_fused::magi_vm_eval(*agg_prog, a,
                                                              bvals, probe_d);
            atomicAdd(&agg_accum[a], v);
          }
        } else {
          do_emit = true;
        }
      }
    }
    // Fused path never emits to out_buf — skip the warp-aggregated emit entirely.
    // group_tbl is warp-uniform (same pointer for all lanes) so this branch is
    // divergence-free: all lanes return, or none do → __syncwarp(active) below is
    // only reached by all-non-fused warps (no stale-mask UB).
    if (group_tbl != nullptr) return;
    __syncwarp(active);
    const unsigned emask = __ballot_sync(active, do_emit);
    if (emask == 0u) return;                          // warp-uniform: all or none
    const int      lane   = static_cast<int>(threadIdx.x & 31u);
    const int      leader = __ffs(emask) - 1;
    const int      rank   = __popc(emask & ((1u << lane) - 1u));
    const int      nmatch = __popc(emask);
    unsigned       base   = 0u;
    if (lane == leader) base = atomicAdd(out_count, static_cast<unsigned>(nmatch));
    base = __shfl_sync(active, base, leader);
    if (!do_emit) return;
    const unsigned pos = base + static_cast<unsigned>(rank);
    if (pos >= out_cap) { if (overflow) atomicAdd(overflow, 1u); return; }
    duckdb::magi_generic::JoinResultRow& r = out[pos];
    r.key_packed =
        static_cast<unsigned __int128>(static_cast<std::uint64_t>(t.key));
    const double* bvals = (build_payload != nullptr && slot->payload_idx >= 0)
                              ? build_payload + (std::size_t)slot->payload_idx * n_build_pl
                              : nullptr;
    for (int i = 0; i < duckdb::magi_generic::JoinResultRow::N_VALUES; ++i)
      r.build_values[i] = (bvals != nullptr && i < n_build_pl) ? bvals[i] : 0.0;
    for (int i = 0; i < duckdb::magi_generic::JoinResultRow::N_VALUES; ++i) {
      if (i < JOIN_MAX_PAYLOAD) {
        double d; __builtin_memcpy(&d, &t.payload[i], sizeof(double));
        r.probe_values[i] = d;
      } else {
        r.probe_values[i] = 0.0;
      }
    }
  }
};

}  // namespace magi_join
}  // namespace duckdb

namespace magi {

// ── Pack ONE row into a wire tuple ─────────────────────────────────────────
// Shared by the standalone pack kernels and by the fused pack-inside-send path
// (see WirePackSrc). Build and probe wires are structurally identical (key,
// index, _pad, payload[JOIN_MAX_PAYLOAD]), so one template covers both — they
// MUST stay identical, or the two sides disagree on the wire layout.
// Returns false for a row filtered out by the validity mask.
template <typename KeyT, typename WireT>
__device__ __forceinline__ bool
pack_wire_row(std::uint64_t r, const magi_ops::ColPack& cols,
              const magi_ops::KeyFieldEntry* __restrict__ key_fields,
              int n_key_fields,
              const duckdb::magi_generic::JoinPayloadEntry* __restrict__ pl,
              int n_pl, WireT& w)
{
  using duckdb::magi_generic::JoinPayloadEntry;
  if (!magi_ops::row_is_valid(cols.row_validity, r)) return false;
  const KeyT key =
      magi_ops::pack_key_from_fields<KeyT>(cols, r, key_fields, n_key_fields);
  w.index = 1;
  w._pad  = 0;
  if constexpr (sizeof(KeyT) > 8) {
    w.key = static_cast<unsigned __int128>(key);  // already u128, no make_unsigned
  } else {
    w.key = static_cast<unsigned __int128>(static_cast<std::make_unsigned_t<KeyT>>(key));
  }
  #pragma unroll
  for (int i = 0; i < duckdb::magi_join::JOIN_MAX_PAYLOAD; ++i) w.payload[i] = 0;
  for (int i = 0; i < n_pl; ++i) {
    const int dst = pl[i].dst_idx;
    const int src = pl[i].src_col_idx;
    if (dst < 0 || dst >= duckdb::magi_join::JOIN_MAX_PAYLOAD) continue;
    if (pl[i].src_kind == JoinPayloadEntry::Src::VARCHAR) {
      // Inline the local string into VARCHAR_PAYLOAD_SLOTS slots: [len][chars…].
      // Done before the key-partition shuffle, so the bytes ride the fixed-width
      // tuple to the owner GPU like any other payload.
      const std::uint8_t*  vch = cols.v_chars[src];
      const std::uint64_t* vof = cols.v_offsets[src];
      const std::uint64_t  s0  = vof[r];
      std::uint32_t len = static_cast<std::uint32_t>(vof[r + 1] - s0);
      if (len > duckdb::magi_generic::VARCHAR_PAYLOAD_MAXLEN)
        len = duckdb::magi_generic::VARCHAR_PAYLOAD_MAXLEN;
      unsigned char buf[duckdb::magi_generic::VARCHAR_PAYLOAD_SLOTS * 8];
      #pragma unroll
      for (int b = 0; b < duckdb::magi_generic::VARCHAR_PAYLOAD_SLOTS * 8; ++b) buf[b] = 0;
      buf[0] = static_cast<unsigned char>(len);
      for (std::uint32_t b = 0; b < len; ++b) buf[1 + b] = vch[s0 + b];
      #pragma unroll
      for (int ws = 0; ws < duckdb::magi_generic::VARCHAR_PAYLOAD_SLOTS; ++ws)
        if (dst + ws < duckdb::magi_join::JOIN_MAX_PAYLOAD)
          __builtin_memcpy(&w.payload[dst + ws], &buf[ws * 8], sizeof(std::int64_t));
    } else if (pl[i].src_kind == JoinPayloadEntry::Src::INT64) {
      w.payload[dst] = cols.i64_agg_cols[src][r];
    } else if (pl[i].src_kind == JoinPayloadEntry::Src::INT32) {
      w.payload[dst] = static_cast<int64_t>(cols.i_cols[src][r]);  // widen INT32 → int64
    } else {  // DOUBLE — bit-cast into the int64 wire slot
      const double d = cols.d_cols[src][r];
      __builtin_memcpy(&w.payload[dst], &d, sizeof(int64_t));
    }
  }
  return true;
}

// ── Pack build rows (ColPack) into wire tuples ──────────────────────────────
// Grid-strided over local build rows; packs the join key (via the table-driven
// pack_key_from_fields) into a JoinBuildWire. suppkey = low32(key) is the
// send_direct routing field; joinkey = full key for the hash table.
template <typename KeyT>
__global__ __launch_bounds__(1024, 1) void
join_pack_build_kernel(const std::uint64_t* __restrict__       row_ids,
                       std::uint64_t                            n_rows,
                       magi_ops::ColPack                        cols,
                       const magi_ops::KeyFieldEntry* __restrict__ key_fields,
                       int                                      n_key_fields,
                       const duckdb::magi_generic::JoinPayloadEntry* __restrict__ build_pl,
                       int                                      n_build_pl,
                       duckdb::magi_join::JoinBuildWire* __restrict__ out,
                       unsigned int* __restrict__               out_count)
{
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t k =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < n_rows; k += stride) {
    duckdb::magi_join::JoinBuildWire w;
    if (!pack_wire_row<KeyT>(row_ids[k], cols, key_fields, n_key_fields,
                             build_pl, n_build_pl, w))
      continue;
    const unsigned pos = atomicAdd(out_count, 1u);
    out[pos] = w;
  }
}

// ── Build kernel: send wire tuples to owners, insert into H_build ───────────
// Verbatim structure of multisession_join.cuh's join_build_kernel: chunked
// send_direct + recv_direct_self_drain / recv_direct_drain with the receiver
// action as a lambda, then the flush + EOF + drain tail. Receiver action =
// global_find_or_insert into H_build (open-addressing, exact). `overflow` bumps
// when build cardinality exceeds the tier.
template <typename KeyT, int N_SLOTS, size_t K_INTRA, size_t K_INTER>
__global__ __launch_bounds__(1024, 1) void
join_build_kernel(const duckdb::magi_join::JoinBuildWire* __restrict__ tuples,
                  std::uint64_t                            n,
                  duckdb::magi_join::JoinSlot<KeyT>* __restrict__ H_build,
                  double* __restrict__                     build_payload,
                  unsigned int* __restrict__               payload_count,
                  unsigned int                             payload_cap,
                  int                                      n_build_pl,
                  bool                                     just_load,
                  unsigned int* __restrict__               overflow,
                  duckdb::magi_join::WirePackSrc           pack_src)
{
  if (just_load) return;
  using Wire = duckdb::magi_join::JoinBuildWire;

  // Same fused-packing scheme as the probe kernel — see the comment there.
  const bool          fused_pack = (pack_src.scratch != nullptr);
  const std::uint64_t total      = fused_pack ? pack_src.n_rows : n;

  const std::uint64_t rows_per_block = (total + gridDim.x - 1) / gridDim.x;
  const std::uint64_t start_row      = static_cast<std::uint64_t>(blockIdx.x) * rows_per_block;
  const std::uint64_t end_row        = min(start_row + rows_per_block, total);
  const std::uint64_t rows           = (start_row >= total) ? 0 : (end_row - start_row);

  Wire* const my_scratch =
      fused_pack ? static_cast<Wire*>(pack_src.scratch) +
                       static_cast<std::size_t>(blockIdx.x) * pack_src.chunk
                 : nullptr;
  __shared__ unsigned int s_packed;
  __shared__ int          s_taken;

  bool recv_eof = false;

  duckdb::magi_join::BuildInsert<KeyT, N_SLOTS>      insert{
      H_build, build_payload, payload_count, payload_cap, n_build_pl, overflow};
  duckdb::magi_join::JoinKeyPartition<Wire>          part;

  std::uint64_t offset  = 0;
  bool          pending = false;
  while (offset < rows || pending) {
    while (true) {
      const Wire* send_ptr;
      int         items;
      if (fused_pack) {
        if (!pending) {
          if (offset >= rows) break;
          const int want = static_cast<int>(
              min(rows - offset, static_cast<std::uint64_t>(pack_src.chunk)));
          if (threadIdx.x == 0) { s_packed = 0u; s_taken = want; }
          __syncthreads();
          for (int i = threadIdx.x; i < want; i += blockDim.x) {
            Wire w;
            if (pack_wire_row<KeyT>(pack_src.row_ids[start_row + offset + i],
                                    pack_src.cols, pack_src.key_fields,
                                    pack_src.n_key_fields, pack_src.pl,
                                    pack_src.n_pl, w)) {
              my_scratch[atomicAdd(&s_packed, 1u)] = w;
            }
          }
          __syncthreads();
          pending = true;
        }
        send_ptr = my_scratch;
        items    = static_cast<int>(s_packed);
      } else {
        if (offset >= rows) break;
        items = static_cast<int>(rows - offset);
        if (items > 64 * static_cast<int>(blockDim.x)) items = 64 * static_cast<int>(blockDim.x);
        send_ptr = &tuples[offset + start_row];
      }
      const int status = (items > 0)
                             ? magi::send_direct<Wire, K_INTRA, K_INTER>(send_ptr, items, part)
                             : MAGI_STATUS_SUCCESS;
      if (status == MAGI_STATUS_SUCCESS) {
        if (fused_pack) { offset += static_cast<std::uint64_t>(s_taken); pending = false; }
        else            { offset += items; }
      } else {
        break;
      }
    }
    while (true) {
      const int s = magi::recv_direct_self_drain<Wire>(insert);
      if (s != MAGI_STATUS_SUCCESS) break;
    }
    while (!recv_eof) {
      const int s = magi::recv_direct_drain<Wire>(insert);
      if (s == MAGI_STATUS_EOF) { recv_eof = true; break; }
      else if (s != MAGI_STATUS_SUCCESS) break;
    }
  }

  // Flush + EOF + drain remainder (multisession_join build tail).
  bool flushed = false, eof_sent = false;
  while (true) {
    if (!flushed && magi::flush_direct_nb()) flushed = true;
    __syncthreads();
    while (true) {
      const int s = magi::recv_direct_self_drain<Wire>(insert);
      if (s != MAGI_STATUS_SUCCESS) break;
    }
    __syncthreads();
    if (flushed && !eof_sent && magi::eof_send_direct_nb()) eof_sent = true;
    __syncthreads();
    while (!recv_eof) {
      const int s = magi::recv_direct_drain<Wire>(insert);
      if (s == MAGI_STATUS_EOF) { recv_eof = true; break; }
      else if (s != MAGI_STATUS_SUCCESS) break;
    }
    __syncthreads();
    if (flushed && eof_sent && recv_eof) break;
  }
}

// ── Pack probe rows (ColPack) into probe wire tuples ────────────────────────
// Packs the join key (pack_key_from_fields) + the probe payload columns (raw
// 8-byte values, per the JoinPayloadEntry table) into a JoinProbeWire.
template <typename KeyT>
__global__ __launch_bounds__(1024, 1) void
join_pack_probe_kernel(const std::uint64_t* __restrict__       row_ids,
                       std::uint64_t                            n_rows,
                       magi_ops::ColPack                        cols,
                       const magi_ops::KeyFieldEntry* __restrict__ key_fields,
                       int                                      n_key_fields,
                       const duckdb::magi_generic::JoinPayloadEntry* __restrict__ probe_pl,
                       int                                      n_probe_pl,
                       duckdb::magi_join::JoinProbeWire* __restrict__ out,
                       unsigned int* __restrict__               out_count)
{
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t k =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < n_rows; k += stride) {
    duckdb::magi_join::JoinProbeWire w;
    if (!pack_wire_row<KeyT>(row_ids[k], cols, key_fields, n_key_fields,
                             probe_pl, n_probe_pl, w))
      continue;
    const unsigned pos = atomicAdd(out_count, 1u);
    out[pos] = w;
  }
}

// Option A: bulk-insert (key -> H_build slot index) into the cuco map after the
// build session. One thread per H_build slot; occupied slots (key != empty)
// insert their packed uint64 key with the slot index as value. Scalar insert
// (cg_size=1). Only instantiated/launched for the uint64 key path.
template <typename KeyT, int N_SLOTS>
__global__ void cuco_insert_from_hbuild(duckdb::magi_join::CucoInsertRef ins,
                                        const duckdb::magi_join::JoinSlot<KeyT>* __restrict__ H)
{
  const std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
  if (i >= (std::uint64_t)N_SLOTS) return;
  // Occupancy test matches global_find: wide keys publish via `index`.
  if constexpr (sizeof(KeyT) > 8) { if (H[i].index == 0) return; }
  const KeyT k = H[i].key;
  if constexpr (sizeof(KeyT) <= 8) { if (k == magi_ops::empty_key_v<KeyT>) return; }
  std::uint64_t ck;
  if constexpr (sizeof(KeyT) > 8) {  // pack 16B compound key -> uint64 (see ProbeEmit)
    const unsigned __int128 ku = static_cast<unsigned __int128>(k);
    ck = static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<std::uint64_t>(ku))) |
         (static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<std::uint64_t>(ku >> 64))) << 32);
  } else {
    ck = static_cast<std::uint64_t>(k);
  }
  ins.insert(cuco::pair<std::uint64_t, std::int32_t>{ck, static_cast<std::int32_t>(i)});
}

// Clear the narrow join table: index=0 marks "never claimed" for wide keys, and
// the key sentinel does the same for narrow ones. payload_idx = -1 = no payload.
template <typename KeyT>
__global__ void join_clear_slots(duckdb::magi_join::JoinSlot<KeyT>* H, int n_slots)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_slots) {
    H[i].key         = magi_ops::empty_key_v<KeyT>;
    H[i].index       = 0;
    H[i].payload_idx = -1;
  }
}

// NOTE: a device-side stats kernel over the fused group table used to live here.
// It reported impossible results (90% "lost writes", two reads of the same slot
// key disagreeing within one kernel) and burned a session on a phantom race; the
// host copy in join_run_typed_tier's DBG block is the ground truth. Don't add one.

// Compact the fused group table into join output rows — one row per non-empty
// group instead of one per match (150M -> 500K here). The per-GPU rows are
// PARTIAL aggregates; the downstream GROUP BY merges them (SUM of partial SUMs
// is the true SUM), i.e. this is classic two-phase aggregation with phase 1
// pushed into the shuffle-join probe.
static __global__ void fused_group_compact(
    const magi_ops::AggSlot64<std::uint64_t>* __restrict__ g, int n_slots,
    duckdb::magi_generic::JoinResultRow* __restrict__ out,
    unsigned int* __restrict__ out_count, unsigned int out_cap,
    int key_dst, int sum_dst)
{
  const std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
  if (i >= (std::uint64_t)n_slots) return;
  const std::uint64_t k = g[i].key;
  if (k == magi_ops::empty_key_v<std::uint64_t>) return;
  const unsigned pos = atomicAdd(out_count, 1u);
  if (pos >= out_cap) return;
  duckdb::magi_generic::JoinResultRow& r = out[pos];
  // The group key is the HIGH half of the compound join key (l_suppkey), so put
  // it back where the key layout expects it — the caller rebuilds the join-key
  // columns by splitting key_packed (low = partkey, high = suppkey).
  r.key_packed = static_cast<unsigned __int128>(k) << 64;
  for (int j = 0; j < duckdb::magi_generic::JoinResultRow::N_VALUES; ++j) {
    r.build_values[j] = 0.0;
    r.probe_values[j] = 0.0;
  }
  {
    const long long kk = static_cast<long long>(k);      // group key column
    double d; __builtin_memcpy(&d, &kk, sizeof(d));      // raw bits; host narrows
    if (key_dst >= 0) {
      if (key_dst < duckdb::magi_generic::JoinResultRow::N_VALUES)
        r.build_values[key_dst] = d;
    } else {
      // key_dst < 0: POC probe mode — put the group key in EVERY slot on both
      // sides (except the aggregate's own slot) so the downstream GROUP BY
      // finds it whichever column the plan reads.
      for (int j = 0; j < duckdb::magi_generic::JoinResultRow::N_VALUES; ++j) {
        r.build_values[j] = d;
        if (j != sum_dst) r.probe_values[j] = d;
      }
    }
  }
  if (sum_dst >= 0 && sum_dst < duckdb::magi_generic::JoinResultRow::N_VALUES)
    r.probe_values[sum_dst] = g[i].values[1];            // int64 sum bits
}

// ── Probe kernel: send probe tuples to owners, look up H_build, emit matches ─
// Same chunked send_direct + drain/flush/EOF structure as join_build_kernel,
// with the receiver action = ProbeEmit (find in H_build + emit JoinResultRow).
// This is the tuple-pipeline probe: each probe tuple, the moment it is drained
// off a ready k-buffer, is looked up and emitted — NVLink transfer overlaps the
// hash probe at tuple granularity.
template <typename KeyT, int N_SLOTS, size_t K_INTRA, size_t K_INTER>
__global__ __launch_bounds__(1024, 1) void
join_probe_kernel(const duckdb::magi_join::JoinProbeWire* __restrict__ tuples,
                  std::uint64_t                            n,
                  duckdb::magi_join::JoinSlot<KeyT>* __restrict__ H_build,
                  const double* __restrict__               build_payload,
                  duckdb::magi_generic::JoinResultRow* __restrict__ out,
                  unsigned int* __restrict__               out_count,
                  unsigned int                             out_cap,
                  int                                      n_build_pl,
                  bool                                     just_load,
                  unsigned int* __restrict__               overflow,
                  unsigned int* __restrict__               n_drained,
                  double* __restrict__                     agg_accum,
                  const duckdb::magi_fused::FusedAggProgram* __restrict__ agg_prog,
                  const duckdb::magi_join::CucoFindRef* __restrict__ cuco_ref,
                  magi_ops::AggSlot64<std::uint64_t>* __restrict__ group_tbl,
                  unsigned int* __restrict__ group_overflow,
                  int fuse_sum_idx,
                  duckdb::magi_join::WirePackSrc pack_src)
{
  if (just_load) return;
  using Wire = duckdb::magi_join::JoinProbeWire;

  // Fused packing: pack each chunk right before sending it, instead of reading a
  // wire array a separate kernel materialized up front. That array is
  // n_probe_rows * 64B (9.6 GB/GPU at SF50) and its pack kernel is dead time on
  // the link — the shuffle cannot start until the last row is packed.
  const bool          fused_pack = (pack_src.scratch != nullptr);
  const std::uint64_t total      = fused_pack ? pack_src.n_rows : n;

  const std::uint64_t rows_per_block = (total + gridDim.x - 1) / gridDim.x;
  const std::uint64_t start_row      = static_cast<std::uint64_t>(blockIdx.x) * rows_per_block;
  const std::uint64_t end_row        = min(start_row + rows_per_block, total);
  const std::uint64_t rows           = (start_row >= total) ? 0 : (end_row - start_row);

  Wire* const my_scratch =
      fused_pack ? static_cast<Wire*>(pack_src.scratch) +
                       static_cast<std::size_t>(blockIdx.x) * pack_src.chunk
                 : nullptr;
  __shared__ unsigned int s_packed;  // tuples that survived the validity mask
  __shared__ int          s_taken;   // source rows the packed chunk consumed

  bool recv_eof = false;
  duckdb::magi_join::ProbeEmit<KeyT, N_SLOTS> emit{H_build, build_payload,
                                                   out,      out_count,
                                                   out_cap,  n_build_pl, overflow,
                                                   n_drained, agg_accum, agg_prog, cuco_ref,
                                                   group_tbl, group_overflow,
                                                   fuse_sum_idx};
  duckdb::magi_join::JoinKeyPartition<Wire>   part;

  std::uint64_t offset  = 0;
  bool          pending = false;  // scratch holds a packed chunk not yet sent
  while (offset < rows || pending) {
    while (true) {
      const Wire* send_ptr;
      int         items;
      if (fused_pack) {
        if (!pending) {
          if (offset >= rows) break;
          const int want = static_cast<int>(
              min(rows - offset, static_cast<std::uint64_t>(pack_src.chunk)));
          if (threadIdx.x == 0) { s_packed = 0u; s_taken = want; }
          __syncthreads();
          for (int i = threadIdx.x; i < want; i += blockDim.x) {
            Wire w;
            if (pack_wire_row<KeyT>(pack_src.row_ids[start_row + offset + i],
                                    pack_src.cols, pack_src.key_fields,
                                    pack_src.n_key_fields, pack_src.pl,
                                    pack_src.n_pl, w)) {
              // Order within a chunk is irrelevant: send_direct routes each
              // tuple by key, and the receiver hashes it.
              my_scratch[atomicAdd(&s_packed, 1u)] = w;
            }
          }
          __syncthreads();
          pending = true;
        }
        send_ptr = my_scratch;
        items    = static_cast<int>(s_packed);
      } else {
        if (offset >= rows) break;
        items = static_cast<int>(rows - offset);
        if (items > 64 * static_cast<int>(blockDim.x)) items = 64 * static_cast<int>(blockDim.x);
        send_ptr = &tuples[offset + start_row];
      }
      // An all-filtered chunk sends nothing but still consumes its source rows.
      const int status = (items > 0)
                             ? magi::send_direct<Wire, K_INTRA, K_INTER>(send_ptr, items, part)
                             : MAGI_STATUS_SUCCESS;
      if (status == MAGI_STATUS_SUCCESS) {
        if (fused_pack) { offset += static_cast<std::uint64_t>(s_taken); pending = false; }
        else            { offset += items; }
      } else {
        break;  // back-pressure: keep the packed chunk, drain, retry
      }
    }
    while (true) {
      const int s = magi::recv_direct_self_drain<Wire>(emit);
      if (s != MAGI_STATUS_SUCCESS) break;
    }
    while (!recv_eof) {
      const int s = magi::recv_direct_drain<Wire>(emit);
      if (s == MAGI_STATUS_EOF) { recv_eof = true; break; }
      else if (s != MAGI_STATUS_SUCCESS) break;
    }
  }

  bool flushed = false, eof_sent = false;
  while (true) {
    if (!flushed && magi::flush_direct_nb()) flushed = true;
    __syncthreads();
    while (true) {
      const int s = magi::recv_direct_self_drain<Wire>(emit);
      if (s != MAGI_STATUS_SUCCESS) break;
    }
    __syncthreads();
    if (flushed && !eof_sent && magi::eof_send_direct_nb()) eof_sent = true;
    __syncthreads();
    while (!recv_eof) {
      const int s = magi::recv_direct_drain<Wire>(emit);
      if (s == MAGI_STATUS_EOF) { recv_eof = true; break; }
      else if (s != MAGI_STATUS_SUCCESS) break;
    }
    __syncthreads();
    if (flushed && eof_sent && recv_eof) break;
  }
}

}  // namespace magi
