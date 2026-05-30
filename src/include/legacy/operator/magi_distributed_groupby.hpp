// magi_distributed_groupby.hpp — public surface of the generic magi groupby
// runtime (`src/legacy/cuda/magi/magi_groupby_runtime.cu`).
//
// One header, one entry point. Sirius's GROUP BY operator calls
// `distributed_hash_groupby_run_per_gpu` for every grouped aggregate that
// reaches the magi path — there are no per-query Q1/Q5/Q3 entries, and no
// shape predicates. The KeyT branch is chosen at runtime via `KeyKind`.

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
// the runtime should launch. KeyKind is the only piece of "shape" the host
// has to decide; everything else is data in the AggOpEntry table.
enum class KeyKind : std::int8_t {
  INT32  = 0,   // packed VARCHAR(1)×N or a single INT column
  UINT64 = 1,   // first-8-chars of a VARCHAR, or compound INT pack
};

// One worker thread's input: how many rows + the packed column views the
// kernel will read. ColPack column ordering is by convention agreed with
// the host caller (pack_key<KeyT> reads from fixed indices in ColPack —
// see col_pack.cuh).
struct PerGpuInputs {
  std::uint64_t      n_filtered;
  magi_ops::ColPack  cols;
};

// One output row per surviving open-addressing slot on this GPU's
// receiver. `key_as_u64` carries the packed KeyT widened to 64 bits
// (int32 sign-extended, uint64 identity); `values[]` are the raw 8-byte
// slots — the caller knows which AggKind each slot holds (it built the
// AggOpEntry table) so it can reinterpret per kind. `N_VALUES` matches
// `AggSlot64<…>::N_DOUBLES` for both supported KeyT (both are 6).
struct AggResultRow {
  static constexpr int N_VALUES = magi_ops::AggSlot64<std::uint64_t>::N_DOUBLES;

  std::uint64_t key_as_u64;
  double        values[N_VALUES];
  std::uint64_t partial_count;
};

// Per-GPU dispatcher entry. Called by every sirius worker thread (one per
// legacy GPU) — threads synchronise internally via a singleton barrier,
// just like `Q5MagiRunPerGpu`. Lazily initialises the magi runtime on the
// first call from any query. `my_slice` is cleared and filled with this
// GPU's hash-partitioned surviving rows on return.
std::size_t distributed_hash_groupby_run_per_gpu(
    int                                      gpu_id,
    const PerGpuInputs&                      inputs,
    const std::vector<magi_ops::AggOpEntry>& ops,
    KeyKind                                  key_kind,
    std::vector<AggResultRow>&               my_slice);

// Number of GPUs the runtime expects, mirrored from
// `magi_q1::NUM_GPUS` (= `kSiriusLegacyNumGpus`). Exposed so sirius
// callers can size per-GPU vectors without instantiating the runtime.
constexpr int NUM_GPUS = magi_q1::NUM_GPUS;

}  // namespace magi_generic
}  // namespace duckdb
