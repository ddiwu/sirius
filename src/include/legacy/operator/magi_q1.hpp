// Public interface to the Magi-backed Q1 dispatcher (legacy/cuda/magi/q1_dispatcher.cu).
// Only used when ENABLE_MAGI_TPCH is defined.

#pragma once

#include <cstdint>
#include <vector>

namespace duckdb {
namespace magi_q1 {

// One partition's worth of column pointers + row count, as already cached
// by sirius's per-GPU TableScan. Pointers are device pointers on the
// matching GPU. n_filtered is the number of rows the kernel will iterate
// (sirius's TableScan filter must already have produced row_ids[0..n)
// dense — currently we feed an identity 0..n-1 inside the dispatcher,
// so callers can pass the unfiltered partition size).
struct PerGpuInputs {
  size_t          n_filtered;
  const double*   d_quantity;
  const double*   d_ep;
  const double*   d_disc;
  const double*   d_tax;
  const uint8_t*  rf_chars;
  const uint64_t* rf_offsets;
  const uint8_t*  ls_chars;
  const uint64_t* ls_offsets;
};

struct AggResultRow {
  int      rf;        // l_returnflag char value
  int      ls;        // l_linestatus char value
  double   sum_qty;
  double   sum_ep;
  double   sum_disc;
  double   sum_disc_price;
  double   sum_charge;
  uint64_t count;
};

// NUM_GPUS partitions are expected (must match dispatcher's compile-time
// constant). Lazy-initialises the magi runtime on first call. Fills `out`
// with one row per (rf,ls) combination that received any tuples; returns
// out.size().
size_t Q1MagiRun(const std::vector<PerGpuInputs>& inputs,
                 std::vector<AggResultRow>&        out);

// Diagnostic: prints sizeof(Q1Tuple), sizeof(Q1AggSlot), Q1_AGG_SLOTS
// to stdout. Used by the original "smoke-test" path of CALL magi_q1().
void Q1MagiHello();

// Number of GPUs the dispatcher expects, exposed so the sirius-side
// driver can size its PerGpuInputs vector. Mirrored from q1_dispatcher.cu.
constexpr int NUM_GPUS = 2;

}  // namespace magi_q1
}  // namespace duckdb
