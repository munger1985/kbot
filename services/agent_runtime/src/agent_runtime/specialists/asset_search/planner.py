"""把 KM Asset 问题规划为单一、可校验的搜索合同。"""

from __future__ import annotations

from datetime import date
import json
from typing import Any

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


class AssetSearchPlanner:
    """调用冻结模型生成搜索计划，并执行确定性产品边界规范化。"""

    def __init__(self, *, model_client, prompt_resolver) -> None:
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

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
            response = await self._model_client.get_llm_json(
                served_model_name=model_name,
                prompt=messages,
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
        raise ValueError(f"Asset Search Planner 输出不符合契约：{last_error}")

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
        normalized.setdefault("answer_detail", "BRIEF")
        normalized.setdefault("criteria", [])
        normalized.setdefault("eligibility_expression", None)
        normalized.setdefault("preferences", [])
        normalized.setdefault("measures", [])
        normalized.setdefault("group_by", [])
        normalized.setdefault("projection", [])
        normalized.setdefault("order_by", [])
        normalized.setdefault("include_total_count", False)
        normalized.setdefault("unsupported_requests", [])
        normalized.setdefault("ambiguities", [])
        normalized.setdefault("evidence_policy", {})

        raw_criteria = normalized.get("criteria")
        criteria = [
            item for item in raw_criteria
            if isinstance(item, dict)
        ] if isinstance(raw_criteria, (list, tuple)) else []
        semantic = any(
            str(item.get("kind") or "") in {
                "SEMANTIC_CONCEPT", "EXACT_PHRASE", "CONTENT_TYPE"
            }
            for item in criteria
        )
        requested_operation = str(normalized.get("operation") or "").upper()
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
            raw_projection = normalized.get("projection")
            projection = (
                [str(item) for item in raw_projection]
                if isinstance(raw_projection, (list, tuple))
                else []
            )
            for field in _DEFAULT_ASSET_PROJECTION:
                if field not in projection:
                    projection.append(field)
            normalized["projection"] = projection
        return normalized
