"""把 KM Asset 问题规划为单一、可校验的搜索合同。"""

from __future__ import annotations

import asyncio
from datetime import date
import json
from typing import Any

from loguru import logger
from platform_core.contracts import AssetSearchPlanV1


_DEFAULT_ASSET_PROJECTION = (
    "asset_id",
    "title",
    "bundle_id",
    "bundle_revision_id",
    "product",
    "solution",
    "asset_date",
)

_EVIDENCE_REQUIREMENTS = {
    "QUERY_RESULT", "CONTENT", "METADATA_OR_CONTENT",
}

_METADATA_FIELD_ALIASES = {
    "domain": "product",
    "domains": "product",
    "products": "product",
    "solutions": "solution",
    "authors": "author",
    "creator": "author",
    "creators": "author",
    "industries": "industry",
    "categories": "category",
    "date": "asset_date",
    "dates": "asset_date",
    "publish_date": "asset_date",
    "published_date": "asset_date",
    "created_date": "asset_date",
    "creation_date": "asset_date",
    "status": "ingestion_status",
    "statuses": "ingestion_status",
    "state": "ingestion_status",
}

_SYSTEM_SCOPE_FIELDS = frozenset({"ingestion_status"})


def _items(value: Any) -> list[Any]:
    """只接受 JSON Array，避免把字符串拆成字段列表。"""
    return list(value) if isinstance(value, (list, tuple)) else []


def _canonical_metadata_field(value: Any) -> str:
    """把用户和规划模型使用的字段别名收敛为受管逻辑字段。"""
    normalized = "_".join(
        str(value).strip().casefold().replace("-", " ").split()
    )
    return _METADATA_FIELD_ALIASES.get(normalized, normalized)


def _canonical_metadata_fields(value: Any) -> list[str]:
    """规范化字段列表并删除别名收敛后产生的重复项。"""
    return list(dict.fromkeys(
        field
        for item in _items(value)
        if (field := _canonical_metadata_field(item))
        and field not in _SYSTEM_SCOPE_FIELDS
    ))


def _canonical_order_by(value: Any) -> list[dict[str, Any]]:
    """规范化排序字段；同一逻辑字段只保留第一次声明。"""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _items(value):
        if not isinstance(item, dict):
            continue
        field = _canonical_metadata_field(item.get("field"))
        if not field or field in _SYSTEM_SCOPE_FIELDS or field in seen:
            continue
        result.append({**item, "field": field})
        seen.add(field)
    return result


def _is_ready_scope_criterion(criterion: dict[str, Any]) -> bool:
    """识别已由 searchable view 强制满足的 READY 状态条件。"""
    return (
        criterion.get("kind") in {"METADATA", "IDENTIFIER"}
        and criterion.get("field_scope") == ["ingestion_status"]
        and criterion.get("operator") in {"EQ", "IN"}
        and bool(criterion.get("values"))
        and all(
            str(value).strip().upper() == "READY"
            for value in criterion.get("values") or ()
        )
    )


def _normalize_criterion(raw: Any, *, sequence: int) -> dict[str, Any] | None:
    """把可无歧义识别的条件近似字段转换为正式合同字段。"""
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or "").upper()
    field_scope = raw.get("field_scope")
    if isinstance(field_scope, str):
        field_scope = [field_scope]
    else:
        field_scope = _items(field_scope)
    field_scope = [
        (
            _canonical_metadata_field(item)
            if kind in {"METADATA", "IDENTIFIER"}
            else str(item).upper()
        )
        for item in field_scope
    ]
    values = raw.get("values")
    if values is None and "value" in raw:
        values = raw.get("value")
    if not isinstance(values, (list, tuple)):
        values = [] if values is None else [values]
    evidence_requirement = str(
        raw.get("evidence_requirement") or ""
    ).upper()
    if evidence_requirement not in _EVIDENCE_REQUIREMENTS:
        evidence_requirement = (
            "QUERY_RESULT" if kind == "METADATA" else "CONTENT"
        )
    if kind not in {"METADATA", "IDENTIFIER"} and any(
        field in {"TITLE", "PRODUCT", "SOLUTION"}
        for field in field_scope
    ):
        evidence_requirement = "METADATA_OR_CONTENT"
    result: dict[str, Any] = {
        "criterion_id": f"c{sequence}",
        "kind": kind,
        "field_scope": field_scope,
        "operator": str(raw.get("operator") or "").upper(),
        "values": list(values),
        "occurrence": str(raw.get("occurrence") or "MUST").upper(),
        "evidence_requirement": evidence_requirement,
        "resolved_concept": None,
    }
    concept = raw.get("resolved_concept")
    if isinstance(concept, dict):
        result["resolved_concept"] = {
            key: concept[key]
            for key in (
                "concept_id", "canonical_name", "equivalents",
                "vocabulary_version",
            )
            if key in concept
        }
    return result


def _criterion_signature(item: dict[str, Any]) -> str:
    """生成条件语义签名，用于识别被模型重复放入硬条件的软偏好。"""
    return json.dumps(
        {
            "kind": item.get("kind"),
            "field_scope": item.get("field_scope"),
            "operator": item.get("operator"),
            "values": item.get("values"),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _normalize_expression(
    raw: Any, *, criterion_ids: dict[str, str]
) -> dict[str, Any] | None:
    """规范化布尔表达式别名，并删除已迁移为软偏好的引用。"""
    if not isinstance(raw, dict):
        return None
    node_type = str(raw.get("node_type") or "").upper()
    if node_type == "REF":
        reference = raw.get("criterion_id", raw.get("ref"))
        criterion_id = criterion_ids.get(str(reference))
        return (
            {"node_type": "REF", "criterion_id": criterion_id}
            if criterion_id else None
        )
    if node_type in {"ALL", "ANY"}:
        raw_children = raw.get("children", raw.get("conditions"))
        children = [
            child
            for item in _items(raw_children)
            if (child := _normalize_expression(
                item, criterion_ids=criterion_ids
            )) is not None
        ]
        if not children:
            return None
        if len(children) == 1:
            return children[0]
        return {"node_type": node_type, "children": children}
    if node_type == "NOT":
        raw_child = raw.get("child")
        if raw_child is None:
            candidates = _items(raw.get("conditions"))
            raw_child = candidates[0] if candidates else None
        child = _normalize_expression(raw_child, criterion_ids=criterion_ids)
        return {"node_type": "NOT", "child": child} if child else None
    return None


def _default_expression(criteria: list[dict[str, Any]]) -> dict[str, Any] | None:
    """在模型未生成可用表达式时按全部硬条件构造确定性表达式。"""
    children: list[dict[str, Any]] = []
    for item in criteria:
        reference: dict[str, Any] = {
            "node_type": "REF",
            "criterion_id": item["criterion_id"],
        }
        if item.get("occurrence") == "MUST_NOT":
            reference = {"node_type": "NOT", "child": reference}
        children.append(reference)
    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return {"node_type": "ALL", "children": children}


def _apply_asset_semantic_scope(criterion: dict[str, Any]) -> None:
    """Asset 主题搜索统一覆盖可搜索元数据和正文。"""
    if criterion.get("kind") != "SEMANTIC_CONCEPT":
        return
    scopes = list(criterion.get("field_scope") or [])
    for field in ("TITLE", "PRODUCT", "SOLUTION", "CONTENT"):
        if field not in scopes:
            scopes.append(field)
    criterion["field_scope"] = scopes
    criterion["evidence_requirement"] = "METADATA_OR_CONTENT"


class AssetSearchPlanner:
    """调用冻结模型生成搜索计划，并执行确定性产品边界规范化。"""

    def __init__(
        self,
        *,
        model_client,
        prompt_resolver,
        timeout_seconds: float = 30,
    ) -> None:
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver
        self._timeout_seconds = timeout_seconds

    async def plan(
        self,
        *,
        model_name: str,
        question: str,
        language: str,
        conversation_context: dict[str, Any] | None,
    ) -> tuple[AssetSearchPlanV1, str]:
        if not model_name or self._model_client is None or self._prompt_resolver is None:
            raise ValueError("KM Agent 未配置可用的 Asset Search Planner 模型")
        prompt = await self._prompt_resolver.resolve(
            "agent_runtime.km_asset_search_plan"
        )
        messages = [
            {"role": "system", "content": prompt.content},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_date": date.today().isoformat(),
                        "time_zone": "Asia/Shanghai",
                        "language": language,
                        "current_input": question,
                        "managed_metadata": {
                            "dimensions": [
                                "asset_id", "title", "author", "product",
                                "solution", "industry", "category",
                                "asset_date",
                            ],
                            "measures": ["asset_count", "author_count"],
                        },
                        "contract_schema": AssetSearchPlanV1.model_json_schema(),
                        "planning_rules": {
                            "broad_semantic_asset_scope": [
                                "TITLE", "PRODUCT", "SOLUTION", "CONTENT"
                            ],
                            "content_only_scope_requires_explicit_request": True,
                            "metadata_or_content_for_searchable_metadata": True,
                        },
                        "conversation_summary": (
                            (conversation_context or {}).get("summary") or {}
                        ),
                        "recent_items": (
                            (conversation_context or {}).get("recent_items") or []
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]
        last_error = ""
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    self._model_client.get_llm_json(
                        served_model_name=model_name,
                        prompt=messages,
                    ),
                    timeout=self._timeout_seconds,
                )
            except Exception as exc:
                logger.warning(
                    "Asset Search Planner 模型调用失败，降级为合同化语义检索："
                    "model={} timeout_seconds={} error_type={} error={}",
                    model_name,
                    self._timeout_seconds,
                    type(exc).__name__,
                    str(exc),
                )
                return (
                    self.semantic_fallback_plan(
                        question=question,
                        language=language,
                    ),
                    f"{prompt.version}-semantic-fallback",
                )
            try:
                normalized = self.normalize_response(
                    response=response,
                    question=question,
                    language=language,
                )
                return AssetSearchPlanV1.model_validate(normalized), prompt.version
            except (TypeError, ValueError) as exc:
                last_error = str(exc)
                if attempt == 0:
                    messages.extend([
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                response,
                                ensure_ascii=False,
                                default=str,
                            ),
                        },
                        {
                            "role": "system",
                            "content": (
                                "上一份输出不符合 AssetSearchPlan.v1："
                                f"{last_error}。请仅输出修正后的 JSON。"
                            ),
                        },
                    ])
        logger.warning(
            "Asset Search Planner 连续输出无效合同，降级为合同化语义检索：{}",
            last_error,
        )
        return (
            self.semantic_fallback_plan(
                question=question,
                language=language,
            ),
            f"{prompt.version}-semantic-fallback",
        )

    @staticmethod
    def semantic_fallback_plan(
        *, question: str, language: str
    ) -> AssetSearchPlanV1:
        """在规划模型不可用时保留 READY Asset 候选约束与引用链路。"""
        normalized = AssetSearchPlanner.normalize_response(
            question=question,
            language=language,
            response={
                "operation": "LIST",
                "target": "ASSET",
                "answer_detail": "BRIEF",
                "criteria": [
                    {
                        "criterion_id": "c1",
                        "kind": "SEMANTIC_CONCEPT",
                        "field_scope": [
                            "TITLE", "PRODUCT", "SOLUTION", "CONTENT"
                        ],
                        "operator": "RELATED_TO",
                        "values": [question],
                        "occurrence": "MUST",
                        "evidence_requirement": "METADATA_OR_CONTENT",
                    }
                ],
                "eligibility_expression": {
                    "node_type": "REF",
                    "criterion_id": "c1",
                },
                "projection": list(_DEFAULT_ASSET_PROJECTION),
                "order_by": [
                    {"field": "asset_date", "direction": "DESC"}
                ],
                "display_limit": 5,
                "result_assets": {
                    "mode": "PRIMARY",
                    "target_count": 5,
                    "selection": "RECENT_RELEVANT",
                },
            },
        )
        return AssetSearchPlanV1.model_validate(normalized)

    @staticmethod
    def normalize_response(
        *, response: Any, question: str, language: str
    ) -> dict[str, Any]:
        """补齐系统控制字段，并把语义总数请求转为参考列表。"""
        if not isinstance(response, dict):
            raise TypeError("Asset Search Planner 输出必须是 JSON Object")
        normalized = dict(response)
        normalized["contract_version"] = "AssetSearchPlan.v1"
        normalized["query_text"] = question
        normalized["language"] = language
        normalized["time_zone"] = "Asia/Shanghai"
        normalized["answer_detail"] = str(
            normalized.get("answer_detail") or "BRIEF"
        ).upper()
        normalized["operation"] = str(
            normalized.get("operation") or ""
        ).upper()
        normalized["target"] = str(normalized.get("target") or "").upper()
        normalized["measures"] = [
            item for item in _items(normalized.get("measures"))
            if isinstance(item, dict)
        ]
        normalized["group_by"] = _canonical_metadata_fields(
            normalized.get("group_by")
        )
        normalized["projection"] = _canonical_metadata_fields(
            normalized.get("projection")
        )
        normalized["order_by"] = _canonical_order_by(
            normalized.get("order_by")
        )
        normalized.setdefault("include_total_count", False)
        normalized.setdefault("unsupported_requests", [])
        normalized.setdefault("ambiguities", [])
        normalized["evidence_policy"] = {}

        preferences: list[dict[str, Any]] = []
        preference_signatures: set[str] = set()
        for position, raw in enumerate(
            _items(normalized.get("preferences")), start=1
        ):
            if not isinstance(raw, dict):
                continue
            criterion = _normalize_criterion(
                raw.get("criterion"), sequence=position
            )
            if criterion is None or _is_ready_scope_criterion(criterion):
                continue
            preference_signatures.add(_criterion_signature(criterion))
            preferences.append({
                "preference_id": f"p{position}",
                "criterion": criterion,
                "priority": position,
                "evidence_requirement": criterion["evidence_requirement"],
            })
        normalized["preferences"] = preferences

        criteria: list[dict[str, Any]] = []
        criterion_ids: dict[str, str] = {}
        for position, raw in enumerate(_items(normalized.get("criteria"))):
            criterion = _normalize_criterion(raw, sequence=len(criteria) + 1)
            if criterion is None or _is_ready_scope_criterion(criterion):
                continue
            if _criterion_signature(criterion) in preference_signatures:
                continue
            criteria.append(criterion)
            if isinstance(raw, dict):
                raw_id = raw.get("criterion_id", position)
                criterion_ids[str(raw_id)] = criterion["criterion_id"]
            criterion_ids.setdefault(str(position), criterion["criterion_id"])
        normalized["criteria"] = criteria
        normalized["eligibility_expression"] = (
            _normalize_expression(
                normalized.get("eligibility_expression"),
                criterion_ids=criterion_ids,
            )
            or _default_expression(criteria)
        )
        semantic = any(
            str(item.get("kind") or "") in {
                "SEMANTIC_CONCEPT", "EXACT_PHRASE", "CONTENT_TYPE"
            }
            for item in criteria
        )
        requested_operation = str(normalized.get("operation") or "").upper()
        if normalized.get("target") == "ASSET":
            for criterion in criteria:
                _apply_asset_semantic_scope(criterion)
            for preference in preferences:
                _apply_asset_semantic_scope(preference["criterion"])
                preference["evidence_requirement"] = preference[
                    "criterion"
                ]["evidence_requirement"]
        if semantic and (
            requested_operation == "COUNT"
            or bool(normalized.get("include_total_count"))
        ):
            normalized.update({
                "operation": "LIST",
                "target": "ASSET",
                "measures": [],
                "group_by": [],
                "include_total_count": False,
                "display_limit": 5,
                "result_assets": {
                    "mode": "PRIMARY",
                    "target_count": 5,
                    "selection": "RECENT_RELEVANT",
                },
                "unsupported_requests": ["SEMANTIC_TOTAL_COUNT"],
            })
            normalized["order_by"] = [{
                "field": "asset_date",
                "direction": "DESC",
            }]
        elif requested_operation == "LIST":
            raw_limit = normalized.get("display_limit", 10)
            if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
                raw_limit = 10
            display_limit = max(1, min(raw_limit, 10))
            raw_result_assets = normalized.get("result_assets")
            if not isinstance(raw_result_assets, dict):
                raw_result_assets = {}
            normalized["display_limit"] = display_limit
            normalized["result_assets"] = {
                "mode": "PRIMARY",
                "target_count": display_limit,
                "selection": str(
                    raw_result_assets.get("selection")
                    or "REQUESTED_ORDER"
                ),
            }
        else:
            normalized["display_limit"] = None
            normalized["result_assets"] = {
                "mode": "SUPPORTING",
                "target_count": 5,
                "selection": (
                    "EVIDENCE_COVERAGE_THEN_RECENT"
                    if requested_operation in {"ANSWER", "COMPARE"}
                    else "RECENT_WITHIN_RESULT"
                ),
            }
        if str(normalized.get("operation") or "").upper() in {
            "LIST", "COUNT", "GROUP", "COMPARE"
        }:
            projection = list(normalized["projection"])
            for field in _DEFAULT_ASSET_PROJECTION:
                if field not in projection:
                    projection.append(field)
            normalized["projection"] = projection
        allowed_fields = set(AssetSearchPlanV1.model_fields)
        return {
            key: value for key, value in normalized.items()
            if key in allowed_fields
        }
