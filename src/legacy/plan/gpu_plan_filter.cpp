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

#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "operator/gpu_physical_filter.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "operator/gpu_physical_projection.hpp"

#include <cmath>

namespace duckdb {

// HAVING pushdown: when a FILTER(agg_output cmp constant) sits DIRECTLY above
// a grouped aggregate, mirror the predicate onto the aggregate so the magi
// flush compaction emits only qualifying groups (Q18: 75M groups -> a few
// thousand survive sum(l_quantity) > 300, which also keeps the device-emit
// row buffer within the arena). The FILTER stays in the plan as the exact
// residual check, so the pushdown may be CONSERVATIVE (thresholds get one
// raw unit of slack toward keeping) but must never drop a qualifying group.
static void TryHavingPushdown(GPUPhysicalOperator& child, Expression& fexpr)
{
  if (child.type != PhysicalOperatorType::HASH_GROUP_BY) { return; }
  if (fexpr.GetExpressionClass() != ExpressionClass::BOUND_COMPARISON) { return; }
  auto& cmp = fexpr.Cast<BoundComparisonExpression>();

  Expression* ref_side = cmp.left.get();
  Expression* cst_side = cmp.right.get();
  bool        flipped  = false;
  if (ref_side->GetExpressionClass() != ExpressionClass::BOUND_REF) {
    std::swap(ref_side, cst_side);
    flipped = true;
  }
  if (ref_side->GetExpressionClass() != ExpressionClass::BOUND_REF ||
      cst_side->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
    return;
  }

  // cmp code: 1 '>', 2 '>=', 3 '<', 4 '<=' (equality not pushed — the slack
  // rule below has no keep-superset form for '=').
  int code = 0;
  switch (fexpr.GetExpressionType()) {
    case ExpressionType::COMPARE_GREATERTHAN:          code = flipped ? 3 : 1; break;
    case ExpressionType::COMPARE_GREATERTHANOREQUALTO: code = flipped ? 4 : 2; break;
    case ExpressionType::COMPARE_LESSTHAN:             code = flipped ? 1 : 3; break;
    case ExpressionType::COMPARE_LESSTHANOREQUALTO:    code = flipped ? 2 : 4; break;
    default: return;
  }

  auto& agg_op   = child.Cast<GPUPhysicalGroupedAggregate>();
  auto& agg_data = agg_op.grouped_aggregate_data;
  const idx_t n_groups = agg_data.groups.size();
  const idx_t ref_idx  = ref_side->Cast<BoundReferenceExpression>().index;
  if (ref_idx < n_groups) { return; }  // predicate on a group column, not an agg
  const idx_t agg_idx = ref_idx - n_groups;
  if (agg_idx >= agg_data.aggregates.size()) { return; }

  auto& aexpr = agg_data.aggregates[agg_idx]->Cast<BoundAggregateExpression>();
  if (aexpr.IsDistinct()) { return; }
  const string& fname = aexpr.function.name;
  const bool is_sum   = fname == "sum" || fname == "sum_no_overflow";
  const bool is_count = fname == "count" || fname == "count_star";
  if (!is_sum && !is_count) { return; }  // MIN/MAX slots are order-encoded; AVG divides at emit

  // Slot representation: SUM over INT/DECIMAL and COUNT accumulate raw int64
  // (DECIMAL at 10^scale); SUM over FLOAT/DOUBLE accumulates double.
  bool val_is_i64 = true;
  int  scale      = 0;
  if (is_sum) {
    if (aexpr.children.empty()) { return; }
    const auto& in_type = aexpr.children[0]->return_type;
    switch (in_type.id()) {
      case LogicalTypeId::DECIMAL:
        scale = DecimalType::GetScale(in_type);
        break;
      case LogicalTypeId::TINYINT:
      case LogicalTypeId::SMALLINT:
      case LogicalTypeId::INTEGER:
      case LogicalTypeId::BIGINT:
        break;
      case LogicalTypeId::FLOAT:
      case LogicalTypeId::DOUBLE:
        val_is_i64 = false;
        break;
      default: return;
    }
  }

  auto& cval = cst_side->Cast<BoundConstantExpression>().value;
  if (cval.IsNull()) { return; }
  double d = 0.0;
  try {
    d = cval.GetValue<double>();
  } catch (...) {
    return;
  }

  magi_groupby::SlotPredicate pred;
  pred.cmp            = code;
  pred.slot           = static_cast<int>(agg_idx);
  pred.value_is_int64 = val_is_i64;
  if (val_is_i64) {
    const double scaled = d * std::pow(10.0, scale);
    // One raw unit of slack toward KEEPING: rounding may only admit extra
    // groups (the residual FILTER removes them), never drop a qualifying one.
    pred.i_threshold = (code <= 2) ? static_cast<long long>(std::floor(scaled)) - 1
                                   : static_cast<long long>(std::ceil(scaled)) + 1;
  } else {
    pred.d_threshold = (code <= 2) ? std::nextafter(d, -1e308)
                                   : std::nextafter(d, 1e308);
  }
  agg_op.having_pushdown = pred;
}

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalFilter& op)
{
  D_ASSERT(op.children.size() == 1);
  unique_ptr<GPUPhysicalOperator> plan = CreatePlan(*op.children[0]);
  if (op.expressions.size() == 1) { TryHavingPushdown(*plan, *op.expressions[0]); }
  if (!op.expressions.empty()) {
    D_ASSERT(plan->types.size() > 0);
    // create a filter if there is anything to filter
    auto filter = make_uniq<GPUPhysicalFilter>(
      plan->types, std::move(op.expressions), op.estimated_cardinality);
    filter->children.push_back(std::move(plan));
    plan = std::move(filter);
  }
  if (op.HasProjectionMap()) {
    // there is a projection map, generate a physical projection
    vector<unique_ptr<Expression>> select_list;
    for (idx_t i = 0; i < op.projection_map.size(); i++) {
      select_list.push_back(make_uniq<BoundReferenceExpression>(op.types[i], op.projection_map[i]));
    }
    auto proj =
      make_uniq<GPUPhysicalProjection>(op.types, std::move(select_list), op.estimated_cardinality);
    proj->children.push_back(std::move(plan));
    plan = std::move(proj);
  }
  return plan;
}

}  // namespace duckdb
