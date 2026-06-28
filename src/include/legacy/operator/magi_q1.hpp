// Public interface to the Magi-backed Q1 dispatcher (legacy/cuda/magi/q1_dispatcher.cu).
// Only used when ENABLE_MAGI_TPCH is defined.

#pragma once

#include <cstdint>
#include <vector>

#include "legacy/gpu_buffer_manager.hpp"  // kSiriusLegacyNumGpus

namespace duckdb {
namespace magi_runtime {

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

// Per-GPU entry. Each worker thread (one per legacy GPU) calls this with
// its own partition's input pointers. Threads coordinate via a singleton
// barrier; all NUM_GPUS threads must enter for any to make progress. On
// return, `my_slice` holds only the (rf,ls) groups that hashed to this
// GPU's partition — sirius's per-thread group_by_result concat then yields
// the full result. Lazy-initialises the magi runtime on first call.
size_t Q1MagiRunPerGpu(int                                gpu_id,
                       const PerGpuInputs&                my_inputs,
                       std::vector<AggResultRow>&         my_slice);

// Number of GPUs the dispatcher expects, exposed so the sirius-side
// driver can size its PerGpuInputs vector. Mirrored from q1_dispatcher.cu
// which uses kSiriusLegacyNumGpus (set at cmake configure time via
// -DSIRIUS_LEGACY_NUM_GPUS=N).
constexpr int NUM_GPUS = kSiriusLegacyNumGpus;

}  // namespace magi_runtime
}  // namespace duckdb
