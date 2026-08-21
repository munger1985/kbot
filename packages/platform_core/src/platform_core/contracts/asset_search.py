"""KM Asset 统一搜索的版本化逻辑合同。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


AssetSearchOperation = Literal["ANSWER", "LIST", "COUNT", "GROUP", "COMPARE"]
AssetSearchTarget = Literal["ASSET", "CONTENT"]
CriterionKind = Literal[
    "METADATA",
    "SEMANTIC_CONCEPT",
    "EXACT_PHRASE",
    "IDENTIFIER",
    "CONTENT_TYPE",
]
CriterionOccurrence = Literal["MUST", "MUST_NOT"]
CriterionEvidenceRequirement = Literal[
    "QUERY_RESULT",
    "CONTENT",
    "METADATA_OR_CONTENT",
]
AssetResultMode = Literal["PRIMARY", "SUPPORTING"]
AssetResultSelection = Literal[
    "REQUESTED_ORDER",
    "RECENT_WITHIN_RESULT",
    "EVIDENCE_COVERAGE_THEN_RECENT",
    "PREFERENCES_THEN_RELEVANCE",
    "RECENT_RELEVANT",
]


class ResolvedAssetConcept(_Contract):
    concept_id: str = Field(min_length=1, max_length=256)
    canonical_name: str = Field(min_length=1, max_length=256)
    equivalents: tuple[str, ...] = Field(default=(), max_length=32)
    vocabulary_version: str = Field(min_length=1, max_length=128)


class AssetSearchCriterion(_Contract):
    criterion_id: str = Field(pattern=r"^c[1-9][0-9]{0,2}$")
    kind: CriterionKind
    field_scope: tuple[str, ...] = Field(min_length=1, max_length=16)
    operator: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    values: tuple[str | int | float | bool, ...] = Field(
        default=(), max_length=100
    )
    occurrence: CriterionOccurrence = "MUST"
    evidence_requirement: CriterionEvidenceRequirement
    resolved_concept: ResolvedAssetConcept | None = None

    @model_validator(mode="after")
    def validate_kind_semantics(self) -> "AssetSearchCriterion":
        metadata_operators = {
            "EQ", "NE", "IN", "NOT_IN", "BETWEEN", "GT", "GTE",
            "LT", "LTE", "CONTAINS", "STARTS_WITH", "IS_NULL",
            "IS_NOT_NULL",
        }
        if self.kind == "METADATA" and self.operator not in metadata_operators:
            raise ValueError("METADATA 条件使用了不受支持的操作符")
        if self.operator in {"IS_NULL", "IS_NOT_NULL"} and self.values:
            raise ValueError(f"{self.operator} 不允许 values")
        if self.operator not in {"IS_NULL", "IS_NOT_NULL"} and not self.values:
            raise ValueError(f"{self.operator} 必须提供 values")
        if self.kind == "SEMANTIC_CONCEPT" and self.operator != "RELATED_TO":
            raise ValueError("SEMANTIC_CONCEPT 必须使用 RELATED_TO")
        if self.kind == "EXACT_PHRASE" and self.operator != "CONTAINS":
            raise ValueError("EXACT_PHRASE 必须使用 CONTAINS")
        if self.kind == "IDENTIFIER" and self.operator not in {"EQ", "IN"}:
            raise ValueError("IDENTIFIER 只能使用 EQ 或 IN")
        if self.kind == "CONTENT_TYPE" and self.operator not in {
            "EQ", "IN", "EQ_OR_RELATED",
        }:
            raise ValueError("CONTENT_TYPE 使用了不受支持的操作符")
        if self.kind != "METADATA" and self.evidence_requirement == "QUERY_RESULT":
            raise ValueError("非元数据条件不能只要求 QUERY_RESULT 证据")
        return self


class AssetSearchPreference(_Contract):
    preference_id: str = Field(pattern=r"^p[1-9][0-9]{0,2}$")
    criterion: AssetSearchCriterion
    priority: int = Field(ge=1, le=100)
    evidence_requirement: CriterionEvidenceRequirement

    @model_validator(mode="after")
    def validate_preference(self) -> "AssetSearchPreference":
        if self.criterion.occurrence != "MUST":
            raise ValueError("软偏好条件不能使用 MUST_NOT")
        return self


class AssetBooleanExpression(_Contract):
    node_type: Literal["REF", "ALL", "ANY", "NOT"]
    criterion_id: str | None = Field(
        default=None,
        pattern=r"^c[1-9][0-9]{0,2}$",
    )
    children: tuple["AssetBooleanExpression", ...] = Field(
        default=(),
        max_length=64,
    )
    child: "AssetBooleanExpression | None" = None

    @model_validator(mode="after")
    def validate_shape(self) -> "AssetBooleanExpression":
        if self.node_type == "REF":
            if self.criterion_id is None or self.children or self.child is not None:
                raise ValueError("REF 只能包含 criterion_id")
            return self
        if self.node_type in {"ALL", "ANY"}:
            if len(self.children) < 2 or self.criterion_id is not None or self.child is not None:
                raise ValueError("ALL/ANY 必须且只能包含至少两个 children")
            return self
        if self.child is None or self.criterion_id is not None or self.children:
            raise ValueError("NOT 必须且只能包含一个 child")
        return self

    def references(self, *, negated: bool = False) -> tuple[tuple[str, bool], ...]:
        """返回条件引用及其规范化负极性。"""
        if self.node_type == "REF":
            return ((str(self.criterion_id), negated),)
        if self.node_type == "NOT":
            return self.child.references(negated=not negated) if self.child else ()
        result: list[tuple[str, bool]] = []
        for item in self.children:
            result.extend(item.references(negated=negated))
        return tuple(result)

    def depth(self) -> int:
        """计算表达式深度，限制异常递归计划。"""
        nested = self.children or ((self.child,) if self.child is not None else ())
        return 1 + max((item.depth() for item in nested), default=0)


class AssetSearchMeasure(_Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    aggregation: Literal[
        "COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"
    ]


class AssetSearchOrder(_Contract):
    field: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    direction: Literal["ASC", "DESC"]


class AssetEvidencePolicy(_Contract):
    coverage: Literal["BREADTH", "DEPTH", "BALANCED"] = "BALANCED"
    required_support: Literal[
        "DIRECT_SUPPORT", "METADATA_SUPPORT", "METADATA_OR_DIRECT"
    ] = "METADATA_OR_DIRECT"
    minimum_distinct_bundles: int = Field(default=1, ge=1, le=10)


class AssetResultPolicy(_Contract):
    mode: AssetResultMode
    target_count: int = Field(ge=1, le=10)
    selection: AssetResultSelection


class AssetSearchAmbiguity(_Contract):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    question: str = Field(min_length=1, max_length=512)


class AssetSearchPlanV1(_Contract):
    contract_version: Literal["AssetSearchPlan.v1"] = "AssetSearchPlan.v1"
    query_text: str = Field(min_length=1, max_length=4000)
    language: str = Field(min_length=2, max_length=16)
    operation: AssetSearchOperation
    target: AssetSearchTarget
    answer_detail: Literal["NONE", "BRIEF", "DETAILED"] = "BRIEF"
    criteria: tuple[AssetSearchCriterion, ...] = Field(default=(), max_length=64)
    eligibility_expression: AssetBooleanExpression | None = None
    preferences: tuple[AssetSearchPreference, ...] = Field(default=(), max_length=32)
    measures: tuple[AssetSearchMeasure, ...] = Field(default=(), max_length=32)
    group_by: tuple[str, ...] = Field(default=(), max_length=32)
    projection: tuple[str, ...] = Field(default=(), max_length=32)
    order_by: tuple[AssetSearchOrder, ...] = Field(default=(), max_length=8)
    include_total_count: bool = False
    display_limit: int | None = Field(default=None, ge=1, le=10)
    result_assets: AssetResultPolicy
    evidence_policy: AssetEvidencePolicy = Field(default_factory=AssetEvidencePolicy)
    unsupported_requests: tuple[
        Literal["SEMANTIC_TOTAL_COUNT"], ...
    ] = ()
    ambiguities: tuple[AssetSearchAmbiguity, ...] = ()
    time_zone: str = Field(default="Asia/Shanghai", min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_plan(self) -> "AssetSearchPlanV1":
        criterion_ids = [item.criterion_id for item in self.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("criteria 的 criterion_id 不能重复")
        preference_ids = [item.preference_id for item in self.preferences]
        if len(preference_ids) != len(set(preference_ids)):
            raise ValueError("preferences 的 preference_id 不能重复")
        priorities = [item.priority for item in self.preferences]
        if len(priorities) != len(set(priorities)):
            raise ValueError("preferences 的 priority 不能重复")
        if self.criteria and self.eligibility_expression is None:
            raise ValueError("存在硬条件时必须提供 eligibility_expression")
        if not self.criteria and self.eligibility_expression is not None:
            raise ValueError("没有硬条件时不得提供 eligibility_expression")
        if self.eligibility_expression is not None:
            if self.eligibility_expression.depth() > 8:
                raise ValueError("eligibility_expression 深度不能超过 8")
            references = self.eligibility_expression.references()
            referenced = [item[0] for item in references]
            unknown = sorted(set(referenced) - set(criterion_ids))
            missing = sorted(set(criterion_ids) - set(referenced))
            if unknown:
                raise ValueError(f"eligibility_expression 引用了未知条件：{unknown}")
            if missing:
                raise ValueError(f"存在未被引用的硬条件：{missing}")
            occurrence_by_id = {
                item.criterion_id: item.occurrence for item in self.criteria
            }
            for criterion_id, negated in references:
                expected_negative = occurrence_by_id[criterion_id] == "MUST_NOT"
                if negated != expected_negative:
                    raise ValueError("条件 occurrence 与布尔表达式极性不一致")
        if self.operation == "LIST":
            if self.display_limit is None:
                raise ValueError("LIST 必须提供 display_limit")
            if self.result_assets.mode != "PRIMARY":
                raise ValueError("LIST 的 result_assets.mode 必须是 PRIMARY")
        else:
            if self.display_limit is not None:
                raise ValueError("非 LIST 不得提供 display_limit")
            if self.result_assets.mode != "SUPPORTING":
                raise ValueError("非 LIST 的 result_assets.mode 必须是 SUPPORTING")
            if not 3 <= self.result_assets.target_count <= 5:
                raise ValueError("支撑 Asset 目标数量必须在 3 到 5 之间")
        if self.operation == "COUNT" and not self.measures:
            raise ValueError("COUNT 必须提供 measure")
        if self.operation == "GROUP" and (not self.measures or not self.group_by):
            raise ValueError("GROUP 必须提供 measure 和 group_by")
        if self.operation in {"COUNT", "GROUP"} and self.target != "ASSET":
            raise ValueError("COUNT/GROUP 的 target 必须是 ASSET")
        if self.unsupported_requests and self.include_total_count:
            raise ValueError("不支持请求不得同时要求完整总数")
        return self

    @property
    def has_semantic_eligibility(self) -> bool:
        return any(
            item.kind in {"SEMANTIC_CONCEPT", "EXACT_PHRASE", "CONTENT_TYPE"}
            for item in self.criteria
        )

    def model_payload(self) -> dict[str, Any]:
        """返回可持久化的 JSON 载荷。"""
        return self.model_dump(mode="json")


AssetBooleanExpression.model_rebuild()
