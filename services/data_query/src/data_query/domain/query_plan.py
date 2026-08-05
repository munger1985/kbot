"""不依赖 SQL 方言的 Query Plan 语义模型校验。"""

from __future__ import annotations

from data_query.contracts import DataQueryPlanV1, SemanticModelDefinition


class QueryPlanValidationError(ValueError):
    """Query Plan 的稳定拒绝码。"""


def validate_query_plan(*, plan: DataQueryPlanV1, model: SemanticModelDefinition, policy_max_limit: int) -> None:
    """验证所有名称均来自已发布 Semantic Model，绝不接触物理 SQL 名称。"""
    if plan.limit > policy_max_limit:
        raise QueryPlanValidationError("POLICY_LIMIT_EXCEEDED")
    datasets = {item.name for item in model.datasets}
    if plan.dataset not in datasets:
        raise QueryPlanValidationError("DATASET_NOT_FOUND")
    dimensions = {item.name: item for item in model.dimensions if item.dataset == plan.dataset}
    measures = {item.name: item for item in model.measures if item.dataset == plan.dataset}
    for name in plan.dimensions:
        dimension = dimensions.get(name)
        if dimension is None:
            raise QueryPlanValidationError("DIMENSION_NOT_FOUND")
        if not dimension.groupable:
            raise QueryPlanValidationError("DIMENSION_NOT_GROUPABLE")
    for measure in plan.measures:
        definition = measures.get(measure.name)
        if definition is None:
            raise QueryPlanValidationError("MEASURE_NOT_FOUND")
        if definition.aggregation != measure.aggregation:
            raise QueryPlanValidationError("MEASURE_AGGREGATION_DENIED")
    for filter_ in plan.filters:
        dimension = dimensions.get(filter_.field)
        if dimension is None:
            raise QueryPlanValidationError("FILTER_FIELD_NOT_FOUND")
        if not dimension.filterable:
            raise QueryPlanValidationError("FILTER_NOT_ALLOWED")
    allowed_orders = set(dimensions) | set(measures)
    if any(item.field not in allowed_orders for item in plan.order_by):
        raise QueryPlanValidationError("ORDER_FIELD_NOT_FOUND")
