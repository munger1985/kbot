"""KM Asset 的语义问数执行器。"""

from agent_runtime.specialists.data_query import SemanticDataQueryExecutor

from .data_query_support import KmAssetDataQuerySupportMixin
from .search import AssetSearchDataQueryCompiler


class KmAssetSemanticDataQueryExecutor(
    KmAssetDataQuerySupportMixin, SemanticDataQueryExecutor
):
    """执行 Asset Search Plan 与 KM 专属多语言枚举。"""

    def _validate_specialized_plan(
        self, *, context, consumer_app_id, plan
    ) -> None:
        self._validate_km_topic_plan(
            context=context,
            consumer_app_id=consumer_app_id,
            plan=plan,
        )

    @staticmethod
    def _compile_asset_plan(search_plan, models):
        return AssetSearchDataQueryCompiler.compile(
            search_plan=search_plan,
            models=models,
        )

    async def _create_plan(self, *, context, question, models):
        search_plan = self._asset_search_plan(context)
        if search_plan is not None:
            return AssetSearchDataQueryCompiler.compile(
                search_plan=search_plan,
                models=models,
            )
        return await super()._create_plan(
            context=context,
            question=question,
            models=models,
        )

    async def _execute_specialized(
        self,
        *,
        context,
        question,
        consumer_app_id,
        agent_version_id,
        auth_context,
        plan,
        models,
    ):
        search_plan = self._asset_search_plan(context)
        if search_plan is not None:
            return await self._execute_asset_search_plan(
                context=context,
                question=question,
                consumer_app_id=consumer_app_id,
                agent_version_id=agent_version_id,
                auth_context=auth_context,
                search_plan=search_plan,
                query_plan=plan,
                models=models,
            )
        answer_basis = self._answer_basis(context)
        topic_terms: tuple[str, ...] = ()
        expansion_warnings: tuple[str, ...] = ()
        if (
            answer_basis == "SEMANTIC_RELEVANCE_ENUMERATION"
            or (
                answer_basis == "SEMANTIC_RELEVANCE_AGGREGATE"
                and self._is_asset_count_plan(plan)
            )
        ):
            topic_terms, expansion_warnings = await self._km_topic_terms(
                context=context,
                question=question,
                plan=plan,
            )
        if answer_basis in self._enumeration_answer_bases():
            return await self._execute_km_asset_enumeration(
                context=context,
                question=question,
                consumer_app_id=consumer_app_id,
                agent_version_id=agent_version_id,
                auth_context=auth_context,
                list_plan=plan,
                topic_terms=topic_terms,
                expansion_warnings=expansion_warnings,
            )
        if (
            answer_basis == "SEMANTIC_RELEVANCE_AGGREGATE"
            and len(topic_terms) >= 3
        ):
            return await self._execute_km_asset_multilingual_count(
                context=context,
                question=question,
                consumer_app_id=consumer_app_id,
                agent_version_id=agent_version_id,
                auth_context=auth_context,
                plan=plan,
                topic_terms=topic_terms,
                expansion_warnings=expansion_warnings,
            )
        return None

    @staticmethod
    def _enumeration_answer_bases() -> frozenset[str]:
        return frozenset({
            "SEMANTIC_RELEVANCE_ENUMERATION",
            "EXACT_METADATA_ENUMERATION",
        })
