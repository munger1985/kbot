"""KM Asset Agent 的统一搜索规划与确定性路由。"""

from enum import StrEnum
from typing import Any, Callable, Literal

from platform_core.contracts import AssetSearchPlanV1

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.specialists.root import RouteDecision, RouteType

from .search import AssetSearchPlanner


class KmAssetAnswerBasis(StrEnum):
    """KM Asset 回答证据口径。"""

    DOCUMENT_CONTENT = "DOCUMENT_CONTENT"
    SEMANTIC_RELEVANCE_BREADTH = "SEMANTIC_RELEVANCE_BREADTH"
    SEMANTIC_RELEVANCE_BALANCED = "SEMANTIC_RELEVANCE_BALANCED"
    SEMANTIC_RELEVANCE_ENUMERATION = "SEMANTIC_RELEVANCE_ENUMERATION"
    SEMANTIC_RELEVANCE_AGGREGATE = "SEMANTIC_RELEVANCE_AGGREGATE"
    EXACT_METADATA_ENUMERATION = "EXACT_METADATA_ENUMERATION"
    EXACT_METADATA = "EXACT_METADATA"
    UNSCOPED_AGGREGATE = "UNSCOPED_AGGREGATE"
    AMBIGUOUS = "AMBIGUOUS"


class KmAssetRouteDecision(RouteDecision):
    """KM Asset 路由携带已校验的统一搜索计划。"""

    asset_search_plan: AssetSearchPlanV1


class KmAssetRoutePlanner:
    """只处理 KM Asset Agent，不参与其他 Agent 的通用路由。"""

    def __init__(
        self,
        *,
        model_client,
        prompt_resolver,
        timeout_seconds: float,
    ) -> None:
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver
        self._search_planner = AssetSearchPlanner(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
            timeout_seconds=timeout_seconds,
        )

    async def decide_for_input(
        self,
        *,
        agent_snapshot: dict[str, Any],
        objective: str,
        conversation_context: dict[str, Any] | None,
        language: str,
        requests_chart: Callable[[str], bool],
    ) -> RouteDecision:
        capabilities = set(agent_snapshot.get("enabled_capabilities") or [])
        if not {"document", "data_query"}.issubset(capabilities):
            raise ValueError("KM Asset Agent 必须启用 document 与 data_query")
        model_name = str(
            agent_model_name(agent_snapshot, "router_llm") or ""
        ).strip()
        if (
            not model_name
            or self._model_client is None
            or self._prompt_resolver is None
        ):
            raise ValueError("KM Agent 未配置可用的 Router 模型")
        plan, prompt_version = await self._search_planner.plan(
            model_name=model_name,
            question=objective,
            language=language,
            conversation_context=conversation_context,
        )
        if plan.ambiguities:
            return KmAssetRouteDecision(
                route_type=RouteType.CLARIFY,
                confidence=0,
                reason="统一搜索计划存在会改变结果集合的歧义",
                clarification_question=plan.ambiguities[0].question,
                requires_chart=False,
                context_required=False,
                coverage_mode="BALANCED",
                answer_basis=KmAssetAnswerBasis.AMBIGUOUS,
                asset_search_plan=plan,
                classifier_version=f"asset-search-plan-v1:{prompt_version}",
            )
        route_type, answer_basis, coverage_mode = self._route_for_plan(plan)
        return KmAssetRouteDecision(
            route_type=route_type,
            confidence=1,
            reason="统一 Asset Search Plan 已通过合同校验",
            clarification_question=None,
            requires_chart=(
                requests_chart(objective)
                and route_type == RouteType.DATA_QUERY
            ),
            context_required=False,
            coverage_mode=coverage_mode,
            answer_basis=answer_basis,
            asset_search_plan=plan,
            classifier_version=f"asset-search-plan-v1:{prompt_version}",
        )

    @staticmethod
    def _route_for_plan(
        plan: AssetSearchPlanV1,
    ) -> tuple[
        RouteType,
        KmAssetAnswerBasis,
        Literal["BREADTH", "BALANCED"],
    ]:
        semantic = plan.has_semantic_eligibility or any(
            item.criterion.kind in {
                "SEMANTIC_CONCEPT", "EXACT_PHRASE", "CONTENT_TYPE"
            }
            for item in plan.preferences
        )
        if semantic or plan.target == "CONTENT":
            return (
                RouteType.HYBRID_DATA_FIRST,
                KmAssetAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
                "BALANCED",
            )
        if plan.operation == "LIST":
            return (
                RouteType.HYBRID_DATA_FIRST,
                KmAssetAnswerBasis.EXACT_METADATA_ENUMERATION,
                "BALANCED",
            )
        if not plan.criteria and plan.operation == "COUNT":
            return (
                RouteType.DATA_QUERY,
                KmAssetAnswerBasis.UNSCOPED_AGGREGATE,
                "BALANCED",
            )
        return (
            RouteType.DATA_QUERY,
            KmAssetAnswerBasis.EXACT_METADATA,
            "BALANCED",
        )
