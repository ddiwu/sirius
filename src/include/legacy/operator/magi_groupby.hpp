// Operator-facing entry to the Magi-backed cross-GPU GROUP BY backend.
// Each legacy worker thread calls Run() on its own GPU's input columns;
// the threads coordinate internally so all NUM_GPUS GPUs participate in
// one magi shuffle session. On return, the per-GPU `group_by_keys` /
// `aggregate_keys` are mutated to hold this GPU's slice of the result —
// hash-partitioned by group key, so concatenating slices across threads
// (sirius's existing `CombineColumns` pattern) yields the full result.
//
// Only used when ENABLE_MAGI_TPCH is defined.

#pragma once

#include "duckdb.hpp"
#include "helper/types.hpp"
#include "gpu_columns.hpp"

namespace duckdb {
namespace magi_groupby {

// gpu_id: this thread's GPU index in [0, kSiriusLegacyNumGpus).
//
// On entry: group_by_keys[i] / aggregate_keys[i] hold this GPU's input
// partition's columns (the same shape sirius normally feeds to
// `cudf_groupby`).
//
// On return: same vectors, mutated to hold the per-GPU result slice
// (column_length = number of distinct groups whose hash lands on this
// GPU's partition; all aggregate columns are aligned 1:1 with the keys).
//
// M1 supports only Q1's input shape:
//   - num_group_keys=2, both VARCHAR(1)
//   - num_aggregates ∈ {SUM(DOUBLE), COUNT_STAR, COUNT(DOUBLE)}
// Any other shape throws NotImplementedException; future queries widen
// the dispatch table inside Run().
void Run(int                                gpu_id,
         vector<shared_ptr<GPUColumn>>&     group_by_keys,
         vector<shared_ptr<GPUColumn>>&     aggregate_keys,
         int                                num_group_keys,
         int                                num_aggregates,
         sirius::AggregationType*           agg_mode);

}  // namespace magi_groupby
}  // namespace duckdb
