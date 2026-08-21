"""把统一 Asset 搜索计划确定性编译为受控 DataQueryPlan。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from platform_core.contracts import (
    AssetBooleanExpression,
    AssetSearchPlanV1,
)
from platform_core.contracts.data_query import (
    DataQueryPlanV1,
    PlanFilter,
    PlanFilterExpression,
    PlanMeasure,
    PlanOrderBy,
)


@dataclass(frozen=True)
class _ProjectedExpression:
    node_type: str
    criterion_id: str | None = None
    children: tuple["_ProjectedExpression", ...] = ()
    child: "_ProjectedExpression | None" = None

    def references(self) -> tuple[str, ...]:
        if self.node_type == "REF":
            return (str(self.criterion_id),)
        if self.node_type == "NOT":
            return self.child.references() if self.child else ()
        return tuple(
            value for item in self.children for value in item.references()
        )


class AssetSearchDataQueryCompiler:
    """只编译可由受管元数据目录表达的条件。"""

    @classmethod
    def compile(
        cls,
        *,
        search_plan: AssetSearchPlanV1,
        models: list[dict[str, Any]],
    ) -> DataQueryPlanV1:
        selected = models[0] if len(models) == 1 else None
        if not isinstance(selected, dict):
            raise ValueError("KM Asset 必须且只能绑定一个受管语义模型")
        datasets = [
            item for item in selected.get("datasets") or []
            if isinstance(item, dict) and item.get("name")
        ]
        if len(datasets) != 1:
            raise ValueError("KM Asset 受管模型必须且只能包含一个数据集")
        dataset = str(datasets[0]["name"])
        dimensions = {
            str(item.get("name")): item
            for item in selected.get("dimensions") or []
            if isinstance(item, dict) and item.get("name")
        }
        measures = {
            str(item.get("name")): item
            for item in selected.get("measures") or []
            if isinstance(item, dict) and item.get("name")
        }
        criterion_by_id = {
            item.criterion_id: item for item in search_plan.criteria
        }
        projected = cls._project_metadata_expression(
            search_plan.eligibility_expression,
            criterion_by_id=criterion_by_id,
        )
        referenced = set(projected.references()) if projected else set()
        filter_criteria = [
            item for item in search_plan.criteria
            if item.criterion_id in referenced
        ]
        filters: list[PlanFilter] = []
        filter_index_by_criterion: dict[str, int] = {}
        for criterion in filter_criteria:
            if len(criterion.field_scope) != 1:
                raise ValueError("元数据条件必须且只能引用一个逻辑字段")
            field = criterion.field_scope[0].casefold()
            if field not in dimensions:
                raise ValueError(f"KM Asset 受管模型缺少逻辑字段：{field}")
            allowed = tuple(
                str(item).upper()
                for item in dimensions[field].get(
                    "allowed_filter_operators", ()
                )
            )
            if allowed and criterion.operator not in allowed:
                raise ValueError(
                    f"字段 {field} 不允许操作符 {criterion.operator}"
                )
            filter_index_by_criterion[criterion.criterion_id] = len(filters)
            filters.append(PlanFilter(
                field=field,
                operator=criterion.operator,
                values=criterion.values,
            ))
        filter_expression = cls._compile_filter_expression(
            projected,
            filter_index_by_criterion=filter_index_by_criterion,
        )

        operation = search_plan.operation
        semantic = search_plan.has_semantic_eligibility or bool(
            search_plan.preferences
        )
        retrieval_scope = (
            operation in {"ANSWER", "COMPARE"}
            and search_plan.target == "CONTENT"
        )
        max_rows = selected.get("max_rows")
        if not isinstance(max_rows, int) or max_rows < 1:
            max_rows = 1000
        if operation == "LIST" or retrieval_scope:
            projected_dimensions = [
                field.casefold()
                for field in search_plan.projection
                if field.casefold() in dimensions
            ]
            for criterion in search_plan.criteria:
                if criterion.kind not in {"METADATA", "IDENTIFIER"}:
                    continue
                for field_name in criterion.field_scope:
                    field = field_name.casefold()
                    if field in dimensions and field not in projected_dimensions:
                        projected_dimensions.append(field)
            for required in (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution", "asset_date",
            ):
                if required in dimensions and required not in projected_dimensions:
                    projected_dimensions.append(required)
            selected_measures = cls._measures(
                names=("asset_count",), catalog=measures
            )
            limit = (
                max_rows
                if semantic or retrieval_scope
                else min(int(search_plan.display_limit or 10), max_rows)
            )
        elif operation == "GROUP":
            projected_dimensions = [
                field.casefold() for field in search_plan.group_by
            ]
            selected_measures = cls._measures(
                names=tuple(item.name for item in search_plan.measures),
                catalog=measures,
            )
            limit = max_rows
        else:
            projected_dimensions = []
            selected_measures = cls._measures(
                names=tuple(item.name for item in search_plan.measures)
                or ("asset_count",),
                catalog=measures,
            )
            limit = max_rows
        unknown_dimensions = sorted(
            set(projected_dimensions) - set(dimensions)
        )
        if unknown_dimensions:
            raise ValueError(
                f"KM Asset 受管模型缺少投影字段：{unknown_dimensions}"
            )

        order_by = []
        for item in search_plan.order_by:
            field = item.field.casefold()
            if field not in dimensions and field not in measures:
                raise ValueError(f"KM Asset 受管模型缺少排序字段：{field}")
            order_by.append(PlanOrderBy(
                field=field,
                direction=item.direction,
            ))
            if field in dimensions and field not in projected_dimensions:
                projected_dimensions.append(field)
        if (
            (operation == "LIST" or retrieval_scope)
            and not order_by
            and "asset_date" in dimensions
        ):
            order_by.append(PlanOrderBy(field="asset_date", direction="DESC"))

        return DataQueryPlanV1(
            semantic_model_id=selected["semantic_model_id"],
            semantic_model_version=selected["semantic_model_version"],
            dataset=dataset,
            measures=selected_measures,
            dimensions=tuple(projected_dimensions),
            filters=tuple(filters),
            filter_expression=filter_expression,
            order_by=tuple(order_by),
            limit=min(limit, 10_000),
            time_zone=search_plan.time_zone,
        )

    @staticmethod
    def _measures(
        *, names: tuple[str, ...], catalog: dict[str, dict[str, Any]]
    ) -> tuple[PlanMeasure, ...]:
        result: list[PlanMeasure] = []
        for name in names:
            item = catalog.get(name)
            if item is None:
                raise ValueError(f"KM Asset 受管模型缺少指标：{name}")
            result.append(PlanMeasure(
                name=name,
                aggregation=str(item.get("aggregation") or "COUNT"),
            ))
        return tuple(result)

    @classmethod
    def _project_metadata_expression(
        cls,
        expression: AssetBooleanExpression | None,
        *,
        criterion_by_id,
        negated: bool = False,
    ) -> _ProjectedExpression | None:
        """生成语义条件的元数据必要条件，None 表示不限制。"""
        if expression is None:
            return None
        if expression.node_type == "REF":
            criterion = criterion_by_id[str(expression.criterion_id)]
            if criterion.kind not in {"METADATA", "IDENTIFIER"}:
                return None
            leaf = _ProjectedExpression(
                node_type="REF",
                criterion_id=criterion.criterion_id,
            )
            return (
                _ProjectedExpression(node_type="NOT", child=leaf)
                if negated
                else leaf
            )
        if expression.node_type == "NOT":
            return cls._project_metadata_expression(
                expression.child,
                criterion_by_id=criterion_by_id,
                negated=not negated,
            )
        effective_type = expression.node_type
        if negated:
            effective_type = "ANY" if effective_type == "ALL" else "ALL"
        projected_children = [
            cls._project_metadata_expression(
                item,
                criterion_by_id=criterion_by_id,
                negated=negated,
            )
            for item in expression.children
        ]
        if effective_type == "ANY" and any(
            item is None for item in projected_children
        ):
            return None
        present = tuple(item for item in projected_children if item is not None)
        if not present:
            return None
        if len(present) == 1:
            return present[0]
        return _ProjectedExpression(
            node_type=effective_type,
            children=present,
        )

    @classmethod
    def _compile_filter_expression(
        cls,
        expression: _ProjectedExpression | None,
        *,
        filter_index_by_criterion: dict[str, int],
    ) -> PlanFilterExpression | None:
        if expression is None:
            return None
        if expression.node_type == "REF":
            return PlanFilterExpression(
                node_type="FILTER",
                filter_index=filter_index_by_criterion[
                    str(expression.criterion_id)
                ],
            )
        if expression.node_type == "NOT":
            return PlanFilterExpression(
                node_type="NOT",
                child=cls._compile_filter_expression(
                    expression.child,
                    filter_index_by_criterion=filter_index_by_criterion,
                ),
            )
        return PlanFilterExpression(
            node_type=expression.node_type,
            children=tuple(
                cls._compile_filter_expression(
                    item,
                    filter_index_by_criterion=filter_index_by_criterion,
                )
                for item in expression.children
            ),
        )
