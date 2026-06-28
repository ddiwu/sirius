// magi_q5.hpp — Public interface to the Magi-backed Q5 dispatcher
// (legacy/cuda/magi/q5_dispatcher.cu).
//
// Mirrors magi_q1.hpp's shape — same NUM_GPUS contract, same per-thread
// Run entry. Q5 input shape: 1 VARCHAR key (n_name) + 1 SUM(DOUBLE)
// (revenue_per_row). After magi shuffle, each GPU's slice holds a
// disjoint set of (n_name, sum_revenue) groups.
//
// Only used when ENABLE_MAGI_TPCH is defined.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "legacy/operator/magi_q1.hpp"  // share NUM_GPUS constant

namespace duckdb {
namespace magi_q5 {

// One partition's worth of column pointers + row count, as already cached
// by sirius's per-GPU TableScan. Pointers are device pointers on the
// matching GPU. n_filtered is the number of rows the kernel iterates over.
struct PerGpuInputs {
  std::size_t          n_filtered;
  const std::uint8_t*  n_name_chars;     // VARCHAR chars buffer
  const std::uint64_t* n_name_offsets;   // VARCHAR offsets[n_filtered+1]
  const double*        d_revenue;        // revenue_per_row column
};

// One output row per (n_name) group that this GPU's partition received.
// n_name_packed holds the first 8 bytes of n_name as a uint64 (Q5's 5
// ASIA nations are unique on first 8 chars, so this is collision-free).
// Caller's WriteSliceToColumns_Q5 unpacks back into a VARCHAR column.
struct AggResultRow {
  std::uint64_t n_name_packed;
  double        sum_revenue;
  std::uint64_t count;
};

// Per-GPU entry. Each sirius worker thread (one per legacy GPU) calls
// this with its own partition's input pointers. Threads coordinate via a
// singleton barrier; all NUM_GPUS threads must enter for any to make
// progress. On return, `my_slice` holds only the (n_name) groups that
// hashed to this GPU's partition. Lazy-initialises the magi runtime
// (shared with q1/...) on first call from any query.
std::size_t Q5MagiRunPerGpu(int                         gpu_id,
                            const PerGpuInputs&         my_inputs,
                            std::vector<AggResultRow>&  my_slice);

// Mirrored from magi_runtime::NUM_GPUS for clarity at use sites; same value.
constexpr int NUM_GPUS = magi_runtime::NUM_GPUS;

}  // namespace magi_q5
}  // namespace duckdb
