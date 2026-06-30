// magi_distributed_join.hpp — public surface of the generic magi shuffle-join
// runtime (`src/legacy/cuda/magi/magi_join_runtime.cu`).
//
// The analog of magi_distributed_groupby.hpp, but for a large–large equi-join
// that cannot be served by the broadcast join (build side is NOT replicated).
// Sirius's hash-join operator calls `distributed_hash_join_run_per_gpu` from the
// `num_gpus>1 && !SubtreeAllReplicated(build)` branch in gpu_physical_hash_join.cpp
// (today that branch throws "needs the magi shuffle join").
//
// Model — magi tuple pipeline, NOT cudf host funcs (build/probe are kernels):
//   Phase 1 (build, BARRIER): partition build rows by join key, k-buffering
//     NVLink shuffle to owners, receiver `global_find_or_insert`s each build
//     tuple into a per-owner open-addressing hash table H_build.
//   Phase 2 (probe, PIPELINE): partition probe rows by join key, k-buffering
//     shuffle to owners; the receiver drains + probes H_build + emits matched
//     (build_payload, probe_payload) in one fused send+drain interleave loop
//     (reuses the groupby shuffle_global_kernel pattern). The k-deep ring hides
//     NVLink latency behind the probe lookup — tuple granularity, no host tiles.
//
// Reuses the generic magi infra: KBuffering ring, Endpoints/ChannelRuntime
// (magi_runtime::MagiInitOnce / magi_stream / magi_bump_session), send_direct /
// recv_direct, global_find_or_insert, and the GenericExchange per-GPU barrier.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "data_plane/ops/agg_slot.cuh"   // AggSlot64 (reused for the receiver slot width)
#include "data_plane/ops/col_pack.cuh"    // ColPack, KeyFieldEntry
#include "data_plane/ops/agg_kinds.cuh"

#include "legacy/operator/magi_distributed_groupby.hpp"  // KeyKind, TableSize, NUM_GPUS

namespace duckdb {
namespace magi_generic {

// One join side's per-GPU input: how many rows + the packed column views.
// Same shape as the groupby's PerGpuInputs; used for BOTH the build and the
// probe side (each GPU holds 1/N rows of each side, row-partitioned by scan).
struct PerGpuJoinInputs {
  std::uint64_t     n_rows;
  magi_ops::ColPack cols;
};

// How to pack one output payload field out of a side's ColPack into the wire
// tuple / result row. `src_kind` selects which ColPack array the column lives
// in; `src_col_idx` indexes that array; `dst_idx` is the slot in the result
// row's value vector. Mirrors AggOpEntry but with no reduction — the join
// carries raw column values through to the output.
struct JoinPayloadEntry {
  // INT64  → cols.i64_agg_cols[src] (8B);  DOUBLE → cols.d_cols[src] (8B);
  // INT32  → cols.i_cols[src] (4B, widened to int64 on the wire, narrowed on emit).
  // VARCHAR→ cols.v_chars[src]/v_offsets[src]; the string is inlined into
  //          VARCHAR_PAYLOAD_SLOTS consecutive 8B wire slots (length-prefixed),
  //          so it rides the fixed-width tuple through the key-partition shuffle
  //          like any other payload — no separate string buffer / cross-GPU dict.
  enum class Src : std::int8_t { INT64 = 0, DOUBLE = 1, INT32 = 2, VARCHAR = 3 };
  Src           src_kind;
  std::int16_t  src_col_idx;   // INT/DOUBLE: pool index; VARCHAR: v_chars/v_offsets index
  std::int16_t  dst_idx;       // BASE wire/result slot (VARCHAR spans 4 from here)
};

// A VARCHAR payload is inlined into this many consecutive 8-byte wire slots:
// byte 0 = length, bytes 1..31 = chars (zero-padded). v1 cap: strings longer
// than VARCHAR_PAYLOAD_MAXLEN are truncated, and one VARCHAR payload consumes a
// side's entire 4-slot wire payload budget (JOIN_MAX_PAYLOAD). A future widening
// can make this a dynamic slot like the groupby's AggSlot64<64/128>.
static constexpr int VARCHAR_PAYLOAD_SLOTS  = 4;
static constexpr int VARCHAR_PAYLOAD_MAXLEN = VARCHAR_PAYLOAD_SLOTS * 8 - 1;  // 31

// One emitted matched row. `key_packed` is the join key widened to 128 bits
// (same convention as AggResultRow). `build_values` / `probe_values` carry the
// payload columns each side contributes to the output, packed by the
// JoinPayloadEntry tables the caller supplies. For a pure semi-join filter
// (Q11: build = German-supplier set, contributes no output column) n_build
// payload entries is 0 and `build_values` is unused.
struct JoinResultRow {
  static constexpr int N_VALUES = magi_ops::AggSlot64<std::uint64_t>::N_DOUBLES;

  unsigned __int128 key_packed;
  double            build_values[N_VALUES];
  double            probe_values[N_VALUES];
};

// Per-GPU dispatcher entry. Called by every sirius worker thread (one per
// legacy GPU); workers sync internally via a singleton barrier, exactly like
// distributed_hash_groupby_run_per_gpu. Lazily inits the magi runtime.
//
//   build_inputs / probe_inputs : this GPU's 1/N slice of each side.
//   key_fields                  : how to pack the equi-join key from each side
//                                 (build and probe must pack an IDENTICAL KeyT
//                                 from their respective columns — same KeyKind,
//                                 same byte layout — so equal keys hash equal).
//   build_payload / probe_payload : which columns to carry to the output.
//   key_kind / table_size       : KeyT instantiation + H_build capacity tier
//                                 (sized to the BUILD-side cardinality / NUM_GPUS).
//
// `my_slice` is cleared and filled with the join rows this GPU owns (i.e. whose
// key hashes to this GPU). Returns my_slice.size().
//
// v1 scope: INNER join, UNIQUE build key on the join key (single-slot hash
// table — Q11's supplier/nation are unique on their join keys). Non-unique
// build keys (one key → many build rows, needs a chained multi-value table)
// and LEFT/semi-emit-unmatched are follow-ups.
std::size_t distributed_hash_join_run_per_gpu(
    int                                         gpu_id,
    const PerGpuJoinInputs&                     build_inputs,
    const PerGpuJoinInputs&                     probe_inputs,
    const std::vector<magi_ops::KeyFieldEntry>& key_fields,
    const std::vector<JoinPayloadEntry>&        build_payload,
    const std::vector<JoinPayloadEntry>&        probe_payload,
    KeyKind                                     key_kind,
    TableSize                                   table_size,
    JoinResultRow*                              out_buf,        // device, wrapper-allocated (pool)
    unsigned int*                               out_count_buf,  // device, wrapper-allocated
    std::uint64_t                               out_cap,        // capacity of out_buf (rows)
    std::vector<JoinResultRow>&                 my_slice,
    // device_emit=true: skip the host-slice D2H/transpose and leave the matched
    // rows in `out_buf` for the caller to columnize on-device (EmitKeyColumnDevice).
    bool                                        device_emit = false);

}  // namespace magi_generic
}  // namespace duckdb
