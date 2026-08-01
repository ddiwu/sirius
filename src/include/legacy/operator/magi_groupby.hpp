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

// HAVING pushdown: keep a group iff `slot-value cmp threshold`, applied inside
// the magi flush compaction on the FINAL (post-shuffle, fully merged) table —
// so the predicate sees GLOBAL aggregates. The comparison runs on the SLOT
// REPRESENTATION: raw int64 for SUM over INT/DECIMAL and COUNT (the caller
// pre-scales DECIMAL thresholds by 10^scale, with 1-unit CONSERVATIVE slack so
// rounding can only keep extra rows — the residual plan FILTER trims those),
// double for SUM over FLOAT64. cmp: 0=off, 1 '>', 2 '>=', 3 '<', 4 '<='.
struct SlotPredicate {
  int       cmp            = 0;
  int       slot           = 0;
  bool      value_is_int64 = false;
  double    d_threshold    = 0.0;
  long long i_threshold    = 0;
};

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
// input_preagged: the caller already collapsed this GPU's slice to one row
// per key (cudf pre-agg) — magi may then skip its producer hash and stream
// rows straight to their owner GPUs (shuffle_direct).
void Run(int                                gpu_id,
         vector<shared_ptr<GPUColumn>>&     group_by_keys,
         vector<shared_ptr<GPUColumn>>&     aggregate_keys,
         int                                num_group_keys,
         int                                num_aggregates,
         sirius::AggregationType*           agg_mode,
         const SlotPredicate&               having_pred = {},
         bool                               input_preagged = false);

// True iff this per-GPU slice is high-cardinality (would route magi to XLARGE).
// The operator runs a cudf local groupby first for these, then calls Run() on the
// reduced partials with re-aggregation agg modes (COUNT*->SUM, etc.). Also fires
// for large inputs headed to the WIDE path (DeriveKeyShape miss or a VARCHAR key
// exceeding its prefix budget): the 320B-slot machinery costs O(input rows) at
// random-access bandwidth, so collapse duplicates locally first. May launch a
// max-strlen probe and rendezvous across workers — call exactly once per grouped
// aggregate on every worker.
bool ShouldCudfPreAgg(int gpu_id, vector<shared_ptr<GPUColumn>>& keys, int n_keys);

}  // namespace magi_groupby
}  // namespace duckdb
