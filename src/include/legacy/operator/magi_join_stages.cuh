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

namespace duckdb {
namespace magi_join {

// Max payload columns carried inline on a wire tuple (and in JoinResultRow).
// Shared by the build and probe wires.
constexpr int JOIN_MAX_PAYLOAD = 4;

// Build-side wire tuple. 48B. `index` is magi's EOF/validity marker (receiver
// checks index != -1). `key` is the full join key (partition + hash). `payload`
// carries the build-side output columns (the join's RHS output) as raw 8-byte
// values; the receiver stores them into the H_build slot so the probe can emit
// them on a match.
struct alignas(16) JoinBuildWire {
  int32_t index;      // 1 = valid; magi EOF protocol uses -1
  int32_t _pad;
  int64_t key;        // join key — routed by the partition functor, hashed into H_build
  int64_t payload[JOIN_MAX_PAYLOAD];   // build payload (RHS output cols), raw 8B
};
static_assert(sizeof(JoinBuildWire) == 48, "JoinBuildWire must be 48B");

// Receiver/partition actions are functor structs (not lambdas): the legacy
// build does not enable nvcc --extended-lambda, and functors are the idiom the
// data_plane already uses (cf. magi::PartBySuppkey). The drains
// (recv_direct_*_drain) and send_direct accept any callable ProcessFn/PartFn.

// Receiver action (build): hash the incoming key into H_build (open-addressing)
// and store the build payload into the claimed slot's values[] so the probe can
// emit it. Unique build key → each slot is claimed once (first writer stores).
template <typename KeyT, int N_SLOTS>
struct BuildInsert {
  magi_ops::AggSlot64<KeyT>* H_build;
  int                        n_build_pl;
  unsigned int*              overflow;
  __device__ __forceinline__ void operator()(JoinBuildWire& t) const {
    if (t.index == -1) return;
    magi_ops::AggSlot64<KeyT>* slot =
        magi_ops::global_find_or_insert<KeyT, N_SLOTS>(
            H_build, static_cast<KeyT>(t.key), overflow);
    if (slot == nullptr) return;
    for (int i = 0; i < n_build_pl && i < JOIN_MAX_PAYLOAD; ++i) {
      double d; __builtin_memcpy(&d, &t.payload[i], sizeof(double));
      slot->values[i] = d;
    }
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
  int32_t index;     // 1 = valid; magi EOF protocol uses -1
  int32_t _pad;
  int64_t key;       // join key
  int64_t payload[JOIN_MAX_PAYLOAD];
};
static_assert(sizeof(JoinProbeWire) == 48, "JoinProbeWire must be 48B");

// Find-only open-addressing probe of H_build (never claims a slot). Returns the
// matching slot or nullptr. v1 supports <=8B keys (Q11's join key is int32).
template <typename KeyT, int N_SLOTS>
__device__ __forceinline__ magi_ops::AggSlot64<KeyT>*
global_find(magi_ops::AggSlot64<KeyT>* __restrict__ H, KeyT key)
{
  const std::uint64_t k64 =
      static_cast<std::uint64_t>(static_cast<std::make_unsigned_t<KeyT>>(key));
  const unsigned int h =
      static_cast<unsigned int>(k64 ^ (k64 >> 32)) * 2654435761u;
  const int start = static_cast<int>(h) & (N_SLOTS - 1);
  for (int probe = 0; probe < N_SLOTS; ++probe) {
    const int  idx      = (start + probe) & (N_SLOTS - 1);
    const KeyT slot_key = H[idx].key;
    if (slot_key == magi_ops::empty_key_v<KeyT>) return nullptr;  // empty → miss
    if (slot_key == key)                          return &H[idx]; // hit
  }
  return nullptr;
}

// Receiver action (probe): look up H_build; on a hit emit one JoinResultRow
// (build payload from the matched slot + this probe tuple's payload). `out_cap`
// guards under-allocation (drops + bumps overflow once cap is reached).
template <typename KeyT, int N_SLOTS>
struct ProbeEmit {
  magi_ops::AggSlot64<KeyT>*           H_build;
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
  __device__ __forceinline__ void operator()(JoinProbeWire& t) const {
    if (t.index == -1) return;
    if (n_drained) atomicAdd(n_drained, 1u);
    magi_ops::AggSlot64<KeyT>* slot =
        global_find<KeyT, N_SLOTS>(H_build, static_cast<KeyT>(t.key));
    if (slot == nullptr) return;                 // no match (INNER → drop)
    if (agg_prog != nullptr) {
      double probe_d[JOIN_MAX_PAYLOAD];
#pragma unroll
      for (int i = 0; i < JOIN_MAX_PAYLOAD; ++i)
        __builtin_memcpy(&probe_d[i], &t.payload[i], sizeof(double));
      for (int a = 0; a < agg_prog->n_aggs; ++a) {
        double v = (agg_prog->kind[a] == duckdb::magi_fused::AGG_COUNT)
                       ? 1.0
                       : duckdb::magi_fused::magi_vm_eval(*agg_prog, a,
                                                          slot->values, probe_d);
        atomicAdd(&agg_accum[a], v);
      }
      return;  // fused: the match is consumed into agg_accum — skip out_buf entirely
    }
    const unsigned pos = atomicAdd(out_count, 1u);
    if (pos >= out_cap) { if (overflow) atomicAdd(overflow, 1u); return; }
    duckdb::magi_generic::JoinResultRow& r = out[pos];
    r.key_packed =
        static_cast<unsigned __int128>(static_cast<std::uint64_t>(t.key));
    for (int i = 0; i < duckdb::magi_generic::JoinResultRow::N_VALUES; ++i)
      r.build_values[i] = (i < n_build_pl) ? slot->values[i] : 0.0;
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
  using duckdb::magi_generic::JoinPayloadEntry;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t k =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < n_rows; k += stride) {
    const std::uint64_t r = row_ids[k];
    if (!magi_ops::row_is_valid(cols.row_validity, r)) continue;
    const KeyT key =
        magi_ops::pack_key_from_fields<KeyT>(cols, r, key_fields, n_key_fields);
    duckdb::magi_join::JoinBuildWire w;
    w.index = 1;
    w._pad  = 0;
    w.key   = static_cast<int64_t>(static_cast<std::make_unsigned_t<KeyT>>(key));
    #pragma unroll
    for (int i = 0; i < duckdb::magi_join::JOIN_MAX_PAYLOAD; ++i) w.payload[i] = 0;
    for (int i = 0; i < n_build_pl; ++i) {
      const int dst = build_pl[i].dst_idx;
      const int src = build_pl[i].src_col_idx;
      if (dst < 0 || dst >= duckdb::magi_join::JOIN_MAX_PAYLOAD) continue;
      if (build_pl[i].src_kind == JoinPayloadEntry::Src::VARCHAR) {
        // Inline the local string into VARCHAR_PAYLOAD_SLOTS slots: [len][chars…].
        // Done here (before the key-partition shuffle), so the bytes ride the
        // fixed-width tuple to the owner GPU like any other payload.
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
      } else if (build_pl[i].src_kind == JoinPayloadEntry::Src::INT64) {
        w.payload[dst] = cols.i64_agg_cols[src][r];
      } else if (build_pl[i].src_kind == JoinPayloadEntry::Src::INT32) {
        w.payload[dst] = static_cast<int64_t>(cols.i_cols[src][r]);  // widen INT32 → int64
      } else {  // DOUBLE — bit-cast into the int64 wire slot
        const double d = cols.d_cols[src][r];
        __builtin_memcpy(&w.payload[dst], &d, sizeof(int64_t));
      }
    }
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
                  magi_ops::AggSlot64<KeyT>* __restrict__  H_build,
                  int                                      n_build_pl,
                  bool                                     just_load,
                  unsigned int* __restrict__               overflow)
{
  if (just_load) return;
  using Wire = duckdb::magi_join::JoinBuildWire;

  const std::uint64_t rows_per_block = (n + gridDim.x - 1) / gridDim.x;
  const std::uint64_t start_row      = static_cast<std::uint64_t>(blockIdx.x) * rows_per_block;
  const std::uint64_t end_row        = min(start_row + rows_per_block, n);
  const std::uint64_t rows           = (start_row >= n) ? 0 : (end_row - start_row);

  bool recv_eof = false;

  duckdb::magi_join::BuildInsert<KeyT, N_SLOTS>      insert{H_build, n_build_pl, overflow};
  duckdb::magi_join::JoinKeyPartition<Wire>          part;

  std::uint64_t offset = 0;
  while (offset < rows) {
    while (true) {
      int items = static_cast<int>(rows - offset);
      if (items > 64 * static_cast<int>(blockDim.x)) items = 64 * static_cast<int>(blockDim.x);
      const int status = magi::send_direct<Wire, K_INTRA, K_INTER>(
          &tuples[offset + start_row], items, part);
      if (status == MAGI_STATUS_SUCCESS) offset += items;
      else break;
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
  using duckdb::magi_generic::JoinPayloadEntry;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t k =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < n_rows; k += stride) {
    const std::uint64_t r = row_ids[k];
    if (!magi_ops::row_is_valid(cols.row_validity, r)) continue;
    const KeyT key =
        magi_ops::pack_key_from_fields<KeyT>(cols, r, key_fields, n_key_fields);
    duckdb::magi_join::JoinProbeWire w;
    w.index = 1;
    w._pad  = 0;
    w.key   = static_cast<int64_t>(static_cast<std::make_unsigned_t<KeyT>>(key));
    #pragma unroll
    for (int i = 0; i < duckdb::magi_join::JOIN_MAX_PAYLOAD; ++i) w.payload[i] = 0;
    for (int i = 0; i < n_probe_pl; ++i) {
      const int dst = probe_pl[i].dst_idx;
      const int src = probe_pl[i].src_col_idx;
      if (dst < 0 || dst >= duckdb::magi_join::JOIN_MAX_PAYLOAD) continue;
      if (probe_pl[i].src_kind == JoinPayloadEntry::Src::VARCHAR) {
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
      } else if (probe_pl[i].src_kind == JoinPayloadEntry::Src::INT64) {
        w.payload[dst] = cols.i64_agg_cols[src][r];
      } else if (probe_pl[i].src_kind == JoinPayloadEntry::Src::INT32) {
        w.payload[dst] = static_cast<int64_t>(cols.i_cols[src][r]);  // widen INT32 → int64
      } else {  // DOUBLE — bit-cast into the int64 wire slot
        const double d = cols.d_cols[src][r];
        __builtin_memcpy(&w.payload[dst], &d, sizeof(int64_t));
      }
    }
    const unsigned pos = atomicAdd(out_count, 1u);
    out[pos] = w;
  }
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
                  magi_ops::AggSlot64<KeyT>* __restrict__  H_build,
                  duckdb::magi_generic::JoinResultRow* __restrict__ out,
                  unsigned int* __restrict__               out_count,
                  unsigned int                             out_cap,
                  int                                      n_build_pl,
                  bool                                     just_load,
                  unsigned int* __restrict__               overflow,
                  unsigned int* __restrict__               n_drained,
                  double* __restrict__                     agg_accum,
                  const duckdb::magi_fused::FusedAggProgram* __restrict__ agg_prog)
{
  if (just_load) return;
  using Wire = duckdb::magi_join::JoinProbeWire;

  const std::uint64_t rows_per_block = (n + gridDim.x - 1) / gridDim.x;
  const std::uint64_t start_row      = static_cast<std::uint64_t>(blockIdx.x) * rows_per_block;
  const std::uint64_t end_row        = min(start_row + rows_per_block, n);
  const std::uint64_t rows           = (start_row >= n) ? 0 : (end_row - start_row);

  bool recv_eof = false;
  duckdb::magi_join::ProbeEmit<KeyT, N_SLOTS> emit{H_build,  out,       out_count,
                                                   out_cap,  n_build_pl, overflow,
                                                   n_drained, agg_accum, agg_prog};
  duckdb::magi_join::JoinKeyPartition<Wire>   part;

  std::uint64_t offset = 0;
  while (offset < rows) {
    while (true) {
      int items = static_cast<int>(rows - offset);
      if (items > 64 * static_cast<int>(blockDim.x)) items = 64 * static_cast<int>(blockDim.x);
      const int status = magi::send_direct<Wire, K_INTRA, K_INTER>(
          &tuples[offset + start_row], items, part);
      if (status == MAGI_STATUS_SUCCESS) offset += items;
      else break;
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
