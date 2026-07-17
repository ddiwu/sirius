/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "log/logging.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "operator/gpu_physical_hash_join.hpp"
#include "operator/gpu_physical_projection.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include "operator/gpu_physical_ungrouped_aggregate.hpp"

namespace duckdb {

static uint32_t RequiredBitsForValue(uint32_t n)
{
  idx_t required_bits = 0;
  while (n > 0) {
    n >>= 1;
    required_bits++;
  }
  return UnsafeNumericCast<uint32_t>(required_bits);
}

template <class T>
hugeint_t GetRangeHugeint(const BaseStatistics& nstats)
{
  return Hugeint::Convert(NumericStats::GetMax<T>(nstats)) -
         Hugeint::Convert(NumericStats::GetMin<T>(nstats));
}

static bool CanUsePartitionedAggregate(ClientContext& context,
                                       LogicalAggregate& op,
                                       GPUPhysicalOperator& child,
                                       vector<column_t>& partition_columns)
{
  if (op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) { return false; }
  for (auto& expression : op.expressions) {
    auto& aggregate = expression->Cast<BoundAggregateExpression>();
    if (aggregate.IsDistinct()) {
      // distinct aggregates are not supported in partitioned hash aggregates
      return false;
    }
  }
  // check if the source is partitioned by the aggregate columns
  // figure out the columns we are grouping by
  for (auto& group_expr : op.groups) {
    // only support bound reference here
    if (group_expr->GetExpressionType() != ExpressionType::BOUND_REF) { return false; }
    auto& ref = group_expr->Cast<BoundReferenceExpression>();
    partition_columns.push_back(ref.index);
  }
  // traverse the children of the aggregate to find the source operator
  reference<GPUPhysicalOperator> child_ref(child);
  while (child_ref.get().type != PhysicalOperatorType::TABLE_SCAN) {
    auto& child_op = child_ref.get();
    switch (child_op.type) {
      case PhysicalOperatorType::PROJECTION: {
        // recompute partition columns
        auto& projection = child_op.Cast<GPUPhysicalProjection>();
        vector<column_t> new_columns;
        for (auto& partition_col : partition_columns) {
          // we only support bound reference here
          auto& expr = projection.select_list[partition_col];
          if (expr->GetExpressionType() != ExpressionType::BOUND_REF) { return false; }
          auto& ref = expr->Cast<BoundReferenceExpression>();
          new_columns.push_back(ref.index);
        }
        // continue into child node with new columns
        partition_columns = std::move(new_columns);
        child_ref         = *child_op.children[0];
        break;
      }
      case PhysicalOperatorType::FILTER:
        // continue into child operators
        child_ref = *child_op.children[0];
        break;
      default:
        // unsupported operator for partition pass-through
        return false;
    }
  }
  auto& table_scan = child_ref.get().Cast<GPUPhysicalTableScan>();
  if (!table_scan.function.get_partition_info) {
    // this source does not expose partition information - skip
    return false;
  }
  // get the base columns by projecting over the projection_ids/column_ids
  if (!table_scan.projection_ids.empty()) {
    for (auto& partition_col : partition_columns) {
      partition_col = table_scan.projection_ids[partition_col];
    }
  }
  vector<column_t> base_columns;
  for (const auto& partition_idx : partition_columns) {
    auto col_idx = partition_idx;
    col_idx      = table_scan.column_ids[col_idx].GetPrimaryIndex();
    base_columns.push_back(col_idx);
  }
  // check if the source operator is partitioned by the grouping columns
  TableFunctionPartitionInput input(table_scan.bind_data.get(), base_columns);
  auto partition_info = table_scan.function.get_partition_info(context, input);
  if (partition_info != TablePartitionInfo::SINGLE_VALUE_PARTITIONS) {
    // we only support single-value partitions currently
    return false;
  }
  // we have single value partitions!
  return true;
}

static bool CanUsePerfectHashAggregate(ClientContext& context,
                                       LogicalAggregate& op,
                                       vector<idx_t>& bits_per_group)
{
  if (op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) { return false; }
  idx_t perfect_hash_bits = 0;
  for (idx_t group_idx = 0; group_idx < op.groups.size(); group_idx++) {
    auto& group = op.groups[group_idx];
    auto& stats = op.group_stats[group_idx];

    switch (group->return_type.InternalType()) {
      case PhysicalType::INT8:
      case PhysicalType::INT16:
      case PhysicalType::INT32:
      case PhysicalType::INT64:
      case PhysicalType::UINT8:
      case PhysicalType::UINT16:
      case PhysicalType::UINT32:
      case PhysicalType::UINT64: break;
      default:
        // we only support simple integer types for perfect hashing
        return false;
    }
    // check if the group has stats available
    auto& group_type = group->return_type;
    if (!stats) {
      // no stats, but we might still be able to use perfect hashing if the type is small enough
      // for small types we can just set the stats to [type_min, type_max]
      switch (group_type.InternalType()) {
        case PhysicalType::INT8:
        case PhysicalType::INT16:
        case PhysicalType::UINT8:
        case PhysicalType::UINT16: break;
        default:
          // type is too large and there are no stats: skip perfect hashing
          return false;
      }
      // construct stats with the min and max value of the type
      stats = NumericStats::CreateUnknown(group_type).ToUnique();
      NumericStats::SetMin(*stats, Value::MinimumValue(group_type));
      NumericStats::SetMax(*stats, Value::MaximumValue(group_type));
    }
    auto& nstats = *stats;

    if (!NumericStats::HasMinMax(nstats)) { return false; }

    if (NumericStats::Max(*stats) < NumericStats::Min(*stats)) {
      // May result in underflow
      return false;
    }

    // we have a min and a max value for the stats: use that to figure out how many bits we have
    // we add two here, one for the NULL value, and one to make the computation one-indexed
    // (e.g. if min and max are the same, we still need one entry in total)
    hugeint_t range_h;
    switch (group_type.InternalType()) {
      case PhysicalType::INT8: range_h = GetRangeHugeint<int8_t>(nstats); break;
      case PhysicalType::INT16: range_h = GetRangeHugeint<int16_t>(nstats); break;
      case PhysicalType::INT32: range_h = GetRangeHugeint<int32_t>(nstats); break;
      case PhysicalType::INT64: range_h = GetRangeHugeint<int64_t>(nstats); break;
      case PhysicalType::UINT8: range_h = GetRangeHugeint<uint8_t>(nstats); break;
      case PhysicalType::UINT16: range_h = GetRangeHugeint<uint16_t>(nstats); break;
      case PhysicalType::UINT32: range_h = GetRangeHugeint<uint32_t>(nstats); break;
      case PhysicalType::UINT64: range_h = GetRangeHugeint<uint64_t>(nstats); break;
      default:
        throw InternalException("Unsupported type for perfect hash (should be caught before)");
    }

    uint64_t range;
    if (!Hugeint::TryCast(range_h, range)) { return false; }

    // bail out on any range bigger than 2^32
    if (range >= NumericLimits<int32_t>::Maximum()) { return false; }

    range += 2;
    // figure out how many bits we need
    idx_t required_bits = RequiredBitsForValue(UnsafeNumericCast<uint32_t>(range));
    bits_per_group.push_back(required_bits);
    perfect_hash_bits += required_bits;
    // check if we have exceeded the bits for the hash
    if (perfect_hash_bits > Settings::Get<PerfectHtThresholdSetting>(context)) {
      // too many bits for perfect hash
      return false;
    }
  }
  for (auto& expression : op.expressions) {
    auto& aggregate = expression->Cast<BoundAggregateExpression>();
    if (aggregate.IsDistinct() || !aggregate.function.combine) {
      // distinct aggregates are not supported in perfect hash aggregates
      return false;
    }
  }
  return true;
}

// unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalAggregate& op)
// {
//   unique_ptr<GPUPhysicalOperator> groupby;
//   D_ASSERT(op.children.size() == 1);

//   auto plan = CreatePlan(*op.children[0]);

//   plan = ExtractAggregateExpressions(std::move(plan), op.expressions, op.groups);

//   bool can_use_simple_aggregation = true;
//   for (auto& expression : op.expressions) {
//     auto& aggregate = expression->Cast<BoundAggregateExpression>();
//     if (!aggregate.function.simple_update) {
//       // unsupported aggregate for simple aggregation: use hash aggregation
//       can_use_simple_aggregation = false;
//       break;
//     }
//   }
//   if (op.groups.empty() && op.grouping_sets.size() <= 1) {
//     // no groups, check if we can use a simple aggregation
//     // special case: aggregate entire columns together
//     if (can_use_simple_aggregation) {
//       groupby = make_uniq_base<GPUPhysicalOperator, GPUPhysicalUngroupedAggregate>(
//         op.types, std::move(op.expressions), op.estimated_cardinality);
//     } else {
//       throw NotImplementedException("Non simple aggregation is not supported");
//       // groupby = make_uniq_base<GPUPhysicalOperator, GPUPhysicalGroupedAggregate>(
//       //     context, op.types, std::move(op.expressions), op.estimated_cardinality);
//     }
//   } else {
//     // groups! create a GROUP BY aggregator
//     // use a partitioned or perfect hash aggregate if possible
//     vector<column_t> partition_columns;
//     vector<idx_t> required_bits;
//     if (can_use_simple_aggregation &&
//         CanUsePartitionedAggregate(context, op, *plan, partition_columns)) {
//       // groupby = make_uniq_base<PhysicalOperator, PhysicalPartitionedAggregate>(
//       //     context, op.types, std::move(op.expressions), std::move(op.groups),
//       //     std::move(partition_columns), op.estimated_cardinality);
//       groupby = make_uniq_base<GPUPhysicalOperator, GPUPhysicalGroupedAggregate>(
//         context,
//         op.types,
//         std::move(op.expressions),
//         std::move(op.groups),
//         std::move(op.grouping_sets),
//         std::move(op.grouping_functions),
//         op.estimated_cardinality);
//     } else if (CanUsePerfectHashAggregate(context, op, required_bits)) {
//       // groupby = make_uniq_base<PhysicalOperator, PhysicalPerfectHashAggregate>(
//       //     context, op.types, std::move(op.expressions), std::move(op.groups),
//       //     std::move(op.group_stats), std::move(required_bits), op.estimated_cardinality);
//       groupby = make_uniq_base<GPUPhysicalOperator, GPUPhysicalGroupedAggregate>(
//         context,
//         op.types,
//         std::move(op.expressions),
//         std::move(op.groups),
//         std::move(op.grouping_sets),
//         std::move(op.grouping_functions),
//         op.estimated_cardinality);
//     } else {
//       // groupby = make_uniq_base<PhysicalOperator, PhysicalHashAggregate>(
//       //     context, op.types, std::move(op.expressions), std::move(op.groups),
//       //     std::move(op.grouping_sets), std::move(op.grouping_functions),
//       //     op.estimated_cardinality);
//       groupby = make_uniq_base<GPUPhysicalOperator, GPUPhysicalGroupedAggregate>(
//         context,
//         op.types,
//         std::move(op.expressions),
//         std::move(op.groups),
//         std::move(op.grouping_sets),
//         std::move(op.grouping_functions),
//         op.estimated_cardinality);
//     }
//   }
//   groupby->children.push_back(std::move(plan));
//   return groupby;
// }

namespace {

// Collect every BoundReferenceExpression under `expr` (including itself).
void CollectBoundRefs(Expression& expr, vector<BoundReferenceExpression*>& out)
{
  if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
    out.push_back(&expr.Cast<BoundReferenceExpression>());
    return;
  }
  ExpressionIterator::EnumerateChildren(expr,
                                        [&](Expression& child) { CollectBoundRefs(child, out); });
}

// Resolve a join-OUTPUT position to (side, child-output index), honouring the
// join's projection maps (output layout = [left(map), right(map)]).
bool JoinOutputToChild(const LogicalComparisonJoin& j, idx_t out_pos, int& side, idx_t& child_idx)
{
  const idx_t n_left_out =
    j.left_projection_map.empty() ? j.children[0]->types.size() : j.left_projection_map.size();
  if (out_pos < n_left_out) {
    side      = 0;
    child_idx = j.left_projection_map.empty() ? out_pos : j.left_projection_map[out_pos];
    return true;
  }
  const idx_t rpos = out_pos - n_left_out;
  const idx_t n_right_out =
    j.right_projection_map.empty() ? j.children[1]->types.size() : j.right_projection_map.size();
  if (rpos >= n_right_out) { return false; }
  side      = 1;
  child_idx = j.right_projection_map.empty() ? rpos : j.right_projection_map[rpos];
  return true;
}

}  // namespace

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::TryEagerAggRewrite(LogicalAggregate& op)
{
  if (std::getenv("MAGI_NO_EAGER_AGG") != nullptr) { return nullptr; }
  // Shape: AGG(groups g1..gn, aggs) ← PROJ ← T=JOIN(L, J=JOIN(P, S)); every
  // group column from P, one group g* = P's equi-join key in J, every
  // aggregate input from L. Rewrites to
  //   PROJ' ← JOIN_back(P, AGG(group = S's key) ← PROJ ← T=JOIN(L, S)).
  // Exact iff g* is unique within P (attribute columns are then functionally
  // dependent on it) — verified at runtime by the join-back row-count check
  // (mismatch throws → DuckDB CPU fallback), so a false positive here can
  // never produce a wrong answer.
  if (op.groups.size() < 2 || op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) {
    return nullptr;
  }
  if (op.expressions.empty()) { return nullptr; }
  if (op.children.size() != 1 ||
      op.children[0]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    return nullptr;
  }
  auto& t = op.children[0]->Cast<LogicalComparisonJoin>();
  if (t.join_type != JoinType::INNER || t.predicate) { return nullptr; }
  // v1 scope: the group-supplying join J sits on T's RHS (build side).
  if (t.children[1]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return nullptr; }
  auto& j = t.children[1]->Cast<LogicalComparisonJoin>();
  if (j.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN || j.join_type != JoinType::INNER ||
      j.conditions.size() != 1 || j.predicate) {
    return nullptr;
  }
  auto& jcond = j.conditions[0];
  if (jcond.comparison != ExpressionType::COMPARE_EQUAL ||
      jcond.left->GetExpressionClass() != ExpressionClass::BOUND_REF ||
      jcond.right->GetExpressionClass() != ExpressionClass::BOUND_REF) {
    return nullptr;
  }
  const idx_t n_t_left_out =
    t.left_projection_map.empty() ? t.children[0]->types.size() : t.left_projection_map.size();

  // Try P = each side of J; S = the other. S must carry T's join key(s).
  for (int p_side = 0; p_side < 2; ++p_side) {
    const int s_side = 1 - p_side;
    auto& p_key_ref =
      (p_side == 0 ? jcond.left : jcond.right)->Cast<BoundReferenceExpression>();
    auto& s_key_ref =
      (p_side == 0 ? jcond.right : jcond.left)->Cast<BoundReferenceExpression>();
    const auto key_id = s_key_ref.return_type.id();
    if (key_id != LogicalTypeId::INTEGER && key_id != LogicalTypeId::BIGINT) { continue; }

    // Every T condition's RHS must resolve into S (survives J's removal).
    bool t_conds_ok = true;
    for (auto& tc : t.conditions) {
      vector<BoundReferenceExpression*> refs;
      CollectBoundRefs(*tc.right, refs);
      for (auto* r : refs) {
        int side; idx_t cidx;
        if (!JoinOutputToChild(j, r->index, side, cidx) || side != s_side) {
          t_conds_ok = false;
          break;
        }
      }
      if (!t_conds_ok) { break; }
    }
    if (!t_conds_ok) { continue; }

    // Trace all groups (plain column refs into T's output) to P columns; one
    // must be P's join key.
    vector<idx_t> group_p_idx(op.groups.size());
    bool groups_ok  = true;
    bool found_gkey = false;
    for (idx_t gi = 0; gi < op.groups.size(); ++gi) {
      if (op.groups[gi]->GetExpressionClass() != ExpressionClass::BOUND_REF) {
        groups_ok = false;
        break;
      }
      const idx_t t_pos = op.groups[gi]->Cast<BoundReferenceExpression>().index;
      if (t_pos < n_t_left_out) { groups_ok = false; break; }   // group from L side
      // T's RHS output position → J output position (T's right map).
      idx_t j_out = t_pos - n_t_left_out;
      if (!t.right_projection_map.empty()) {
        if (j_out >= t.right_projection_map.size()) { groups_ok = false; break; }
        j_out = t.right_projection_map[j_out];
      }
      int side; idx_t cidx;
      if (!JoinOutputToChild(j, j_out, side, cidx) || side != p_side) {
        groups_ok = false;
        break;
      }
      group_p_idx[gi] = cidx;
      if (cidx == p_key_ref.index &&
          op.groups[gi]->return_type == p_key_ref.return_type) {
        found_gkey = true;
      }
    }
    if (!groups_ok || !found_gkey) { continue; }

    // Aggregate inputs (arbitrary expressions, e.g. l_ext*(1-l_disc)) must
    // reference only T's L side; no filters.
    bool aggs_ok = true;
    for (auto& aexpr : op.expressions) {
      auto& agg = aexpr->Cast<BoundAggregateExpression>();
      if (agg.filter) { aggs_ok = false; break; }
      for (auto& child : agg.children) {
        vector<BoundReferenceExpression*> refs;
        CollectBoundRefs(*child, refs);
        for (auto* r : refs) {
          if (r->index >= n_t_left_out) { aggs_ok = false; break; }
        }
        if (!aggs_ok) { break; }
      }
      if (!aggs_ok) { break; }
    }
    if (!aggs_ok) { continue; }

    // ── Pattern matched: mutate the tree ──────────────────────────────────
    SIRIUS_LOG_DEBUG("eager-agg rewrite fires: {} groups -> 1 narrow key", op.groups.size());
    const idx_t pk = p_key_ref.index;
    const idx_t sk = s_key_ref.index;
    const auto  p_key_type = p_key_ref.return_type;
    const auto  s_key_type = s_key_ref.return_type;

    // T's RHS condition refs indexed J's output; remap them to S's output
    // (must happen while J's children are still attached).
    for (auto& tc : t.conditions) {
      vector<BoundReferenceExpression*> refs;
      CollectBoundRefs(*tc.right, refs);
      for (auto* r : refs) {
        int side; idx_t cidx;
        JoinOutputToChild(j, r->index, side, cidx);   // verified side == s_side above
        r->index = cidx;
      }
    }
    auto j_owned = std::move(t.children[1]);
    auto& j2     = j_owned->Cast<LogicalComparisonJoin>();
    auto p_tree  = std::move(j2.children[p_side]);
    auto s_tree  = std::move(j2.children[s_side]);
    const idx_t n_p = p_tree->types.size();
    t.children[1] = std::move(s_tree);
    // T now outputs all of S (drop the stale RHS map — late materialization
    // keeps unreferenced columns free). L-side positions are unchanged, so
    // the aggregate input expressions stay valid as-is.
    t.right_projection_map.clear();

    // Narrow aggregate directly over the rewired T, grouped by S's key.
    auto new_agg = make_uniq<LogicalAggregate>(op.group_index, op.aggregate_index,
                                               std::move(op.expressions));
    new_agg->groupings_index = op.groupings_index;
    new_agg->groups.push_back(make_uniq<BoundReferenceExpression>(s_key_type, n_t_left_out + sk));
    GroupingSet gs;
    gs.insert(0);
    new_agg->grouping_sets.push_back(std::move(gs));
    new_agg->distinct_validity = op.distinct_validity;
    const idx_t n_aggs         = new_agg->expressions.size();
    vector<LogicalType> agg_out_types;
    for (auto& e : new_agg->expressions) { agg_out_types.push_back(e->return_type); }
    new_agg->children.push_back(std::move(op.children[0]));

    // Join P back above the aggregate: probe = P (attributes stay local),
    // build = the narrow aggregate (broadcast-eligible at runtime).
    auto join_back = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
    JoinCondition jb;
    jb.left       = make_uniq<BoundReferenceExpression>(p_key_type, pk);
    jb.right      = make_uniq<BoundReferenceExpression>(s_key_type, 0);
    jb.comparison = ExpressionType::COMPARE_EQUAL;
    join_back->conditions.push_back(std::move(jb));
    join_back->children.push_back(std::move(p_tree));
    join_back->children.push_back(std::move(new_agg));

    // Restore the original aggregate's output layout [groups..., aggs...].
    vector<unique_ptr<Expression>> final_exprs;
    for (idx_t gi = 0; gi < group_p_idx.size(); ++gi) {
      final_exprs.push_back(
        make_uniq<BoundReferenceExpression>(op.groups[gi]->return_type, group_p_idx[gi]));
    }
    for (idx_t a = 0; a < n_aggs; ++a) {
      final_exprs.push_back(make_uniq<BoundReferenceExpression>(agg_out_types[a], n_p + 1 + a));
    }
    auto final_proj = make_uniq<LogicalProjection>(op.group_index, std::move(final_exprs));
    final_proj->children.push_back(std::move(join_back));
    final_proj->ResolveOperatorTypes();

    auto gpu = CreatePlan(*final_proj);
    // Flag the join-back for the runtime row-count verification (see
    // GPUPhysicalHashJoin::eager_agg_verify). CreatePlan may omit the final
    // projection, so search the top of the returned tree.
    GPUPhysicalOperator* node = gpu.get();
    while (node != nullptr && node->type != PhysicalOperatorType::HASH_JOIN &&
           !node->children.empty()) {
      node = node->children[0].get();
    }
    if (node != nullptr && node->type == PhysicalOperatorType::HASH_JOIN) {
      node->Cast<GPUPhysicalHashJoin>().eager_agg_verify = true;
    }
    return gpu;
  }
  return nullptr;
}

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalAggregate& op)
{
  D_ASSERT(op.children.size() == 1);

  const bool dump_logical = std::getenv("MAGI_DUMP_LOGICAL") != nullptr;
  if (dump_logical) {
    fprintf(stderr, "[eager-agg-dump] ===== LogicalAggregate subtree =====\n%s\n",
            op.ToString().c_str());
    for (auto& g : op.groups) {
      fprintf(stderr, "[eager-agg-dump] group: %s (type %s)\n", g->ToString().c_str(),
              g->return_type.ToString().c_str());
    }
    for (auto& a : op.expressions) {
      fprintf(stderr, "[eager-agg-dump] aggregate: %s\n", a->ToString().c_str());
    }
    const LogicalOperator* c = op.children[0].get();
    while (c != nullptr) {
      fprintf(stderr, "[eager-agg-dump] child: type=%s n_types=%zu\n",
              LogicalOperatorToString(c->type).c_str(), c->types.size());
      if (c->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
        auto& j = c->Cast<LogicalComparisonJoin>();
        for (auto& cond : j.conditions) {
          fprintf(stderr, "[eager-agg-dump]   cond: %s %s %s | L n_types=%zu R n_types=%zu\n",
                  cond.left->ToString().c_str(),
                  ExpressionTypeToString(cond.comparison).c_str(),
                  cond.right->ToString().c_str(),
                  j.children[0]->types.size(), j.children[1]->types.size());
        }
        break;
      }
      c = c->children.empty() ? nullptr : c->children[0].get();
    }
  }

  auto rewritten = TryEagerAggRewrite(op);
  if (rewritten) { return rewritten; }

  auto plan = CreatePlan(*op.children[0]);

  plan = ExtractAggregateExpressions(std::move(plan), op.expressions, op.groups, op.grouping_sets);
  bool can_use_simple_aggregation = true;
  for (auto& expression : op.expressions) {
    auto& aggregate = expression->Cast<BoundAggregateExpression>();
    if (!aggregate.function.simple_update) {
      // unsupported aggregate for simple aggregation: use hash aggregation
      can_use_simple_aggregation = false;
      break;
    }
  }

  // Check if all groups are valid
  if (op.group_stats.empty()) { op.group_stats.resize(op.groups.size()); }
  auto group_validity = TupleDataValidityType::CANNOT_HAVE_NULL_VALUES;
  for (const auto& stats : op.group_stats) {
    if (stats && !stats->CanHaveNull()) { continue; }
    group_validity = TupleDataValidityType::CAN_HAVE_NULL_VALUES;
    break;
  }

  if (op.groups.empty() && op.grouping_sets.size() <= 1) {
    // no groups, check if we can use a simple aggregation
    // special case: aggregate entire columns together
    if (can_use_simple_aggregation) {
      auto group_by = make_uniq_base<GPUPhysicalOperator, GPUPhysicalUngroupedAggregate>(
        op.types, std::move(op.expressions), op.estimated_cardinality, op.distinct_validity);
      group_by->children.push_back(std::move(plan));
      return group_by;
    }
    throw NotImplementedException("Non simple aggregation is not supported");
    // auto &group_by =
    //     Make<GPUPhysicalGroupedAggregate>(context, op.types, std::move(op.expressions),
    //     op.estimated_cardinality);
    // group_by.children.push_back(plan);
    // return group_by;
  }

  // groups! create a GROUP BY aggregator
  // use a partitioned or perfect hash aggregate if possible
  vector<column_t> partition_columns;
  vector<idx_t> required_bits;
  if (can_use_simple_aggregation &&
      CanUsePartitionedAggregate(context, op, *plan, partition_columns)) {
    // auto &group_by =
    //     Make<PhysicalPartitionedAggregate>(context, op.types, std::move(op.expressions),
    //     std::move(op.groups),
    //                                        std::move(partition_columns),
    //                                        op.estimated_cardinality);
    auto group_by = make_uniq_base<GPUPhysicalOperator, GPUPhysicalGroupedAggregate>(
      context,
      op.types,
      std::move(op.expressions),
      std::move(op.groups),
      std::move(op.grouping_sets),
      std::move(op.grouping_functions),
      op.estimated_cardinality,
      group_validity,
      op.distinct_validity);
    group_by->children.push_back(std::move(plan));
    return group_by;
  }

  if (CanUsePerfectHashAggregate(context, op, required_bits)) {
    // auto &group_by = Make<PhysicalPerfectHashAggregate>(context, op.types,
    // std::move(op.expressions),
    //                                                     std::move(op.groups),
    //                                                     std::move(op.group_stats),
    //                                                     std::move(required_bits),
    //                                                     op.estimated_cardinality);
    auto group_by = make_uniq_base<GPUPhysicalOperator, GPUPhysicalGroupedAggregate>(
      context,
      op.types,
      std::move(op.expressions),
      std::move(op.groups),
      std::move(op.grouping_sets),
      std::move(op.grouping_functions),
      op.estimated_cardinality,
      group_validity,
      op.distinct_validity);
    group_by->children.push_back(std::move(plan));
    return group_by;
  }

  auto group_by = make_uniq_base<GPUPhysicalOperator, GPUPhysicalGroupedAggregate>(
    context,
    op.types,
    std::move(op.expressions),
    std::move(op.groups),
    std::move(op.grouping_sets),
    std::move(op.grouping_functions),
    op.estimated_cardinality,
    group_validity,
    op.distinct_validity);
  group_by->children.push_back(std::move(plan));
  return group_by;
}

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::ExtractAggregateExpressions(
  unique_ptr<GPUPhysicalOperator> child,
  vector<unique_ptr<Expression>>& aggregates,
  vector<unique_ptr<Expression>>& groups,
  optional_ptr<vector<GroupingSet>> grouping_sets)
{
  vector<unique_ptr<Expression>> expressions;
  vector<LogicalType> types;

  // bind sorted aggregates
  for (auto& aggr : aggregates) {
    auto& bound_aggr = aggr->Cast<BoundAggregateExpression>();
    if (bound_aggr.order_bys) {
      // sorted aggregate!
      FunctionBinder::BindSortedAggregate(context, bound_aggr, groups, grouping_sets);
    }
  }
  for (auto& group : groups) {
    auto ref = make_uniq<BoundReferenceExpression>(group->return_type, expressions.size());
    types.push_back(group->return_type);
    expressions.push_back(std::move(group));
    group = std::move(ref);
  }
  for (auto& aggr : aggregates) {
    auto& bound_aggr = aggr->Cast<BoundAggregateExpression>();
    for (auto& child_expr : bound_aggr.children) {
      auto ref = make_uniq<BoundReferenceExpression>(child_expr->return_type, expressions.size());
      types.push_back(child_expr->return_type);
      expressions.push_back(std::move(child_expr));
      child_expr = std::move(ref);
    }
    if (bound_aggr.filter) {
      auto& filter = bound_aggr.filter;
      auto ref     = make_uniq<BoundReferenceExpression>(filter->return_type, expressions.size());
      types.push_back(filter->return_type);
      expressions.push_back(std::move(filter));
      bound_aggr.filter = std::move(ref);
    }
  }
  if (expressions.empty()) { return child; }
  auto projection = make_uniq<GPUPhysicalProjection>(
    std::move(types), std::move(expressions), child->estimated_cardinality);
  projection->children.push_back(std::move(child));
  return std::move(projection);
}

}  // namespace duckdb
