"""KM Asset 对通用 Agent Skill 的专属实现入口。"""

from __future__ import annotations

from agent_runtime.specialists.data_query import DataQuerySkill
from agent_runtime.specialists.document import KnowledgeRetrievalSkill
from agent_runtime.specialists.response_composer import ResponseComposerSkill

from .document_scope import KmAssetDocumentScopeExtractSkill
from .composer import KmAssetComposerMixin
from .retrieval import KmAssetRetrievalMixin


class KmAssetKnowledgeRetrievalSkill(
    KmAssetRetrievalMixin, KnowledgeRetrievalSkill
):
    """执行 KM Asset Search Plan 与同 Bundle KC 取证。"""

    @staticmethod
    def _prefer_evidence_order(context):
        route = context.config_snapshot.get("route") or {}
        plan = route.get("asset_search_plan") if isinstance(route, dict) else None
        if not isinstance(plan, dict):
            return False
        unsupported = set(plan.get("unsupported_requests") or ())
        return "SEMANTIC_TOTAL_COUNT" not in unsupported

    async def _retrieve_evidence(
        self,
        *,
        context,
        query,
        candidates,
        scoped,
        retrieval_config,
        coverage_mode,
    ):
        search_plan = self._asset_search_plan(context)
        if search_plan is not None and scoped:
            return await self._retrieve_asset_plan_evidence(
                context=context,
                plan=search_plan,
                candidates=candidates,
                retrieval_config=retrieval_config,
                coverage_mode=coverage_mode,
            )
        return await super()._retrieve_evidence(
            context=context,
            query=query,
            candidates=candidates,
            scoped=scoped,
            retrieval_config=retrieval_config,
            coverage_mode=coverage_mode,
        )


class KmAssetResponseComposerSkill(
    KmAssetComposerMixin, ResponseComposerSkill
):
    """组合 KM Asset 问文、问数和引用结果。"""

    async def _compose_specialized(self, context):
        query_result = self._query_result(context)
        retrieval = self._document_result(context)
        search_plan = self._asset_search_plan(context)
        if search_plan is None or query_result is None:
            return None
        if retrieval is not None:
            if search_plan.operation in {"COUNT", "GROUP"}:
                return await self._compose_asset_aggregate_with_evidence(
                    context, query_result, retrieval, search_plan
                )
            return await self._compose_km_asset_enumeration(
                context,
                query_result,
                retrieval,
                search_plan=search_plan,
            )
        return await self._compose_asset_query_result(
            context, query_result, search_plan
        )


class KmAssetDataQuerySkill(DataQuerySkill):
    """执行 KM Asset 托管语义模型问数。"""
