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
#include <vector>

#include "data_plane/ops/agg_kinds.cuh"
#include "data_plane/ops/agg_slot.cuh"
#include "data_plane/ops/col_pack.cuh"

#include "legacy/operator/magi_q1.hpp"  // for magi_q1::NUM_GPUS

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
//   SMALL  — 256    slots, fits Q1/Q5/Q9-class (cardinality ≤ ~100)
//   MEDIUM — 16 K   slots, fits Q16/Q18-class  (cardinality ≤ ~10 K)
//   LARGE  — 1 M    slots, fits Q3/Q11-class   (cardinality ≤ ~500 K)
//
// LARGE costs 64 MB / GPU of always-allocated buffer (we keep the buffer at
// largest tier and only cudaMemcpy as many slots as the chosen tier needs).
enum class TableSize : std::int8_t {
  SMALL  = 0,
  MEDIUM = 1,
  LARGE  = 2,
};

// One worker thread's input: how many rows + the packed column views the
// kernel will read. ColPack column ordering must match the KeyFieldEntry[]
// src_col_idx indices and the AggOpEntry[] src_col_idx values the caller
// passes.
struct PerGpuInputs {
  std::uint64_t      n_filtered;
  magi_ops::ColPack  cols;
};

// One output row per surviving open-addressing slot on this GPU's receiver.
// `key_packed` carries the packed KeyT widened to 128 bits (int32/uint64 zero-
// extended via unsigned cast; uint64 identity in the low 64; 16-byte compound
// keys use the full width). The host unpacks each group-by column out of it by
// the same byte offsets the kernel packed with. `values[]` are the raw 8-byte
// slots — the caller knows which AggKind each holds (it built the AggOpEntry
// table) so it can reinterpret per kind.
struct AggResultRow {
  static constexpr int N_VALUES = magi_ops::AggSlot64<std::uint64_t>::N_DOUBLES;

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
    std::vector<AggResultRow>&                       my_slice);

// Number of GPUs the runtime expects, mirrored from
// `magi_q1::NUM_GPUS` (= `kSiriusLegacyNumGpus`). Exposed so sirius
// callers can size per-GPU vectors without instantiating the runtime.
constexpr int NUM_GPUS = magi_q1::NUM_GPUS;

}  // namespace magi_generic
}  // namespace duckdb
