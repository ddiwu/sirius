// magi_join.hpp — operator-facing entry for the magi shuffle join.
//
// The analog of magi_groupby.hpp's Run: the multi-GPU hash-join operator
// (gpu_physical_hash_join.cpp), on the non-replicated-build branch, calls this
// instead of throwing "needs the magi shuffle join". This wrapper builds the
// ColPack inputs / key recipe / payload table from sirius GPUColumns, drives
// magi_generic::distributed_hash_join_run_per_gpu, and emits the joined rows
// back as GPUColumns.
//
// v1 scope: INNER join, UNIQUE build key, fixed-width join key (INT32/INT64),
// probe payload columns INT64 / FLOAT64 / DECIMAL(≤18). The build side is a
// pure key set (semi-join filter — Q11's supplier/nation); build payload is a
// follow-up.

#pragma once

#include <vector>

#include "legacy/gpu_columns.hpp"

namespace duckdb {
namespace magi_join_op {

// Run the shuffle join on this GPU's slice. `build_keys` / `probe_keys` are the
// equi-join key column(s) of each side (must derive an identical key recipe).
// `probe_payload` are the probe (LHS) columns that flow to the output; build_payload
// are the build (RHS) columns that flow to the output (e.g. Q11's partsupp⋈supplier
// carries s_nationkey for the downstream ⋈nation). On return: `out_key` = emitted
// join-key column(s); `out_payload` = emitted probe/LHS columns; `out_build_payload`
// = emitted build/RHS columns — all in input order. The operator assembles its
// output relation from these (LHS slots, then RHS slots).
void Run(int                                       gpu_id,
         const std::vector<shared_ptr<GPUColumn>>& build_keys,
         const std::vector<shared_ptr<GPUColumn>>& build_payload,
         const std::vector<shared_ptr<GPUColumn>>& probe_keys,
         const std::vector<shared_ptr<GPUColumn>>& probe_payload,
         std::vector<shared_ptr<GPUColumn>>&       out_key,
         std::vector<shared_ptr<GPUColumn>>&       out_build_payload,
         std::vector<shared_ptr<GPUColumn>>&       out_payload);

}  // namespace magi_join_op
}  // namespace duckdb
