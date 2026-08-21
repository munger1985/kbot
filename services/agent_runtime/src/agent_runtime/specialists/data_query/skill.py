"""由冻结 Agent 配置选择 MCP 或语义问数 Provider。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from uuid import UUID

from loguru import logger

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.language import detect_unicode_language
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from platform_core.contracts import AssetSearchPlanV1, AuthContext
from platform_core.contracts.data_query import DataQueryPlanV1, PlanFilter
from platform_core.identity import uuid7

from .contracts import KMTopicExpansion, QueryResult


_PLANNER_INPUT_ECHO_FIELDS = frozenset(
    {"question", "models", "document_constraints"}
)
_KM_TOPIC_EXPANSION_LANGUAGES = frozenset({"zh-CN", "ja-JP", "ko-KR"})
_KM_ENUMERATION_ANSWER_BASES = frozenset({
    "SEMANTIC_RELEVANCE_ENUMERATION",
    "EXACT_METADATA_ENUMERATION",
})


class MCPDataQueryExecutor:
    def __init__(self, *, client) -> None:
        self._client = client

    async def execute(self, *, context: ExecutionContext, question: str) -> QueryResult:
        if self._client is None:
            raise RuntimeError("DATA_QUERY_MCP_PROVIDER_UNAVAILABLE")
        agent = dict(context.config_snapshot.get("agent") or {})
        agent_config = dict(agent.get("config") or {})
        profile = str(agent_config.get("data_profile_name") or "").strip()
        if not profile:
            raise ValueError("MCP 问数模式未配置 data_profile_name")
        result = await self._client.query(
            profile=profile,
            user=context.actor_id,
            question=question,
        )
        raw_rows = result.get("rows")
        if not isinstance(raw_rows, list) or any(not isinstance(item, dict) for item in raw_rows):
            raise ValueError("MCP 问数服务返回的结果行不是对象")
        rows = tuple(dict(item) for item in raw_rows)
        columns = tuple({"name": str(name)} for name in (rows[0].keys() if rows else ()))
        truncated = bool(result.get("truncated"))
        warnings = ("问数结果超过行数上限，已截断",) if truncated else ()
        return QueryResult(
            query_result_id=uuid7(),
            provider="MCP",
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            warnings=warnings,
            provenance={
                "profile": profile,
                "external_request_id": result.get("external_request_id"),
            },
        )


class SemanticDataQueryExecutor:
    def __init__(self, *, client, model_client, prompt_resolver) -> None:
        self._client = client
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, *, context: ExecutionContext, question: str) -> QueryResult:
        if self._client is None:
            raise RuntimeError("SEMANTIC_DATA_QUERY_UNAVAILABLE")
        auth_context = self._auth_context(context)
        agent_snapshot = context.config_snapshot.get("agent", {})
        consumer_app_id = str(agent_snapshot.get("owner_app_id") or "")
        agent_version_id = UUID(str(agent_snapshot.get("agent_version_id")))
        planning = await self._client.get_planning_context(
            consumer_app_id=consumer_app_id,
            agent_id=context.agent_id,
            agent_version_id=agent_version_id,
            auth_context=auth_context,
        )
        models = planning.get("models") if isinstance(planning, dict) else None
        if not isinstance(models, list) or not models:
            raise RuntimeError("SEMANTIC_DATA_QUERY_NOT_CONFIGURED")
        plan = await self._create_plan(
            context=context,
            question=question,
            models=models,
        )
        specialized = await self._execute_specialized(
            context=context,
            question=question,
            consumer_app_id=consumer_app_id,
            agent_version_id=agent_version_id,
            auth_context=auth_context,
            plan=plan,
            models=models,
        )
        if specialized is not None:
            return specialized
        run_id, result = await self._run_plan(
            context=context,
            question=question,
            consumer_app_id=consumer_app_id,
            agent_version_id=agent_version_id,
            auth_context=auth_context,
            plan=plan,
            idempotency_suffix="query",
        )
        query_result = self._query_result_from_response(
            run_id=run_id, result=result, plan=plan
        )
        return query_result

    async def _execute_specialized(
        self,
        *,
        context: ExecutionContext,
        question: str,
        consumer_app_id: str,
        agent_version_id: UUID,
        auth_context: AuthContext,
        plan: DataQueryPlanV1,
        models: list[dict],
    ) -> QueryResult | None:
        """供 App 专属语义问数执行器扩展。"""
        del (
            context,
            question,
            consumer_app_id,
            agent_version_id,
            auth_context,
            plan,
            models,
        )
        return None

    async def _run_plan(
        self,
        *,
        context: ExecutionContext,
        question: str,
        consumer_app_id: str,
        agent_version_id: UUID,
        auth_context: AuthContext,
        plan: DataQueryPlanV1,
        idempotency_suffix: str,
    ) -> tuple[UUID, dict]:
        receipt = await self._client.create_run(
            payload={
                "idempotency_key": (
                    f"{context.run_id}:{context.task_id}:{idempotency_suffix}"
                ),
                "original_question": context.original_input,
                "standalone_query": question,
                "plan": plan.model_dump(mode="json"),
                "agent_id": str(context.agent_id),
                "consumer_app_id": consumer_app_id,
                "agent_version_id": str(agent_version_id),
                "parent_agent_run_id": str(context.run_id),
                "parent_agent_task_id": str(context.task_id),
                "deadline_at": (
                    context.deadline_at.isoformat()
                    if context.deadline_at is not None
                    else None
                ),
            },
            auth_context=auth_context,
        )
        run_id = UUID(str(receipt.get("data_query_run_id")))
        result = await self._wait_result(
            run_id=run_id,
            auth_context=auth_context,
            deadline_at=context.deadline_at,
        )
        return run_id, result

    def _query_result_from_response(
        self, *, run_id: UUID, result: dict, plan: DataQueryPlanV1
    ) -> QueryResult:
        raw_rows = result.get("preview_rows")
        if not isinstance(raw_rows, list) or any(not isinstance(item, dict) for item in raw_rows):
            raise RuntimeError("SEMANTIC_DATA_QUERY_INVALID_RESULT")
        truncated = bool(result.get("truncated"))
        warnings = ("问数结果超过策略上限，已截断",) if truncated else ()
        provenance = dict(result.get("provenance") or {})
        provenance["data_query_run_id"] = str(run_id)
        provenance.setdefault("query_plan_hash", self._plan_hash(plan))
        return QueryResult(
            query_result_id=uuid7(),
            provider="SEMANTIC",
            columns=tuple(dict(item) for item in result.get("columns") or ()),
            rows=tuple(dict(item) for item in raw_rows),
            row_count=int(result.get("row_count") or 0),
            truncated=truncated,
            warnings=warnings,
            provenance=provenance,
        )

    async def _execute_km_asset_enumeration(
        self,
        *,
        context: ExecutionContext,
        question: str,
        consumer_app_id: str,
        agent_version_id: UUID,
        auth_context: AuthContext,
        list_plan: DataQueryPlanV1,
        topic_terms: tuple[str, ...],
        expansion_warnings: tuple[str, ...],
    ) -> QueryResult:
        """按展示上限多取一条判断截断，避免列表请求先做全量计数。"""
        if len(topic_terms) >= 3:
            return await self._execute_km_asset_multilingual_enumeration(
                context=context,
                question=question,
                consumer_app_id=consumer_app_id,
                agent_version_id=agent_version_id,
                auth_context=auth_context,
                list_plan=list_plan,
                topic_terms=topic_terms,
                expansion_warnings=expansion_warnings,
            )
        probe_plan = self._enumeration_probe_plan(list_plan)
        list_run_id, list_response = await self._run_plan(
            context=context,
            question=question,
            consumer_app_id=consumer_app_id,
            agent_version_id=agent_version_id,
            auth_context=auth_context,
            plan=probe_plan,
            idempotency_suffix="list",
        )
        probe_rows = self._asset_rows_up_to(
            list_response, query_limit=probe_plan.limit
        )
        rows = probe_rows[:list_plan.limit]
        source_truncated = bool(list_response.get("truncated"))
        truncated = source_truncated or len(probe_rows) > len(rows)
        provenance = dict(list_response.get("provenance") or {})
        provenance.update({
            "data_query_run_id": str(list_run_id),
            "data_query_run_ids": [str(list_run_id)],
            "list_query_plan_hash": self._plan_hash(probe_plan),
            "count_exact": not truncated,
            "display_limit": list_plan.limit,
        })
        return QueryResult(
            query_result_id=uuid7(),
            provider="SEMANTIC",
            columns=tuple(
                dict(item) for item in list_response.get("columns") or ()
            ),
            rows=tuple(dict(item) for item in rows),
            row_count=len(probe_rows),
            truncated=truncated,
            warnings=(
                expansion_warnings
                + (
                    (self._enumeration_truncation_warning(list_plan.limit),)
                    if truncated else ()
                )
            ),
            provenance=provenance,
        )

    async def _execute_km_asset_multilingual_enumeration(
        self,
        *,
        context: ExecutionContext,
        question: str,
        consumer_app_id: str,
        agent_version_id: UUID,
        auth_context: AuthContext,
        list_plan: DataQueryPlanV1,
        topic_terms: tuple[str, ...],
        expansion_warnings: tuple[str, ...],
    ) -> QueryResult:
        """并发检索双语清单，多取一条判断截断并按 Asset 去重。"""
        original_term, *english_terms = topic_terms
        original_list_plan = self._enumeration_probe_plan(
            self._replace_topic_groups(list_plan, ((original_term,),))
        )
        english_list_plan = self._enumeration_probe_plan(
            self._replace_topic_groups(list_plan, (tuple(english_terms),))
        )

        results = await self._run_plan_variants(
            context=context,
            question=question,
            consumer_app_id=consumer_app_id,
            agent_version_id=agent_version_id,
            auth_context=auth_context,
            variants=(
                ("topic-original-list", original_list_plan),
                ("topic-english-list", english_list_plan),
            ),
        )
        original_list_run_id, original_list_response = results[
            "topic-original-list"
        ]
        english_list_run_id, english_list_response = results[
            "topic-english-list"
        ]
        original_rows = self._asset_rows_up_to(
            original_list_response, query_limit=original_list_plan.limit,
        )
        english_rows = self._asset_rows_up_to(
            english_list_response, query_limit=english_list_plan.limit,
        )
        rows_by_id: dict[str, dict] = {}
        for row in (*original_rows, *english_rows):
            asset_id = str(row.get("asset_id") or "").strip().casefold()
            if not asset_id:
                raise RuntimeError("KM_ASSET_ENUMERATION_ASSET_ID_MISSING")
            rows_by_id.setdefault(asset_id, row)
        merged_probe_rows = sorted(
            rows_by_id.values(),
            key=lambda item: str(item.get("asset_date") or ""),
            reverse=True,
        )
        merged_rows = merged_probe_rows[:list_plan.limit]
        source_truncated = any(
            bool(item.get("truncated"))
            for item in (original_list_response, english_list_response)
        )
        probe_exhausted = any(
            len(rows) >= plan.limit
            for rows, plan in (
                (original_rows, original_list_plan),
                (english_rows, english_list_plan),
            )
        )
        truncated = (
            source_truncated
            or probe_exhausted
            or len(merged_probe_rows) > len(merged_rows)
        )

        run_ids = (
            original_list_run_id,
            english_list_run_id,
        )
        provenance = dict(original_list_response.get("provenance") or {})
        provenance.update({
            "data_query_run_id": str(original_list_run_id),
            "data_query_run_ids": [str(item) for item in run_ids],
            "topic_search_mode": "ORIGINAL_AND_ENGLISH_PARALLEL",
            "topic_terms": list(topic_terms),
            "query_plan_hashes": [
                self._plan_hash(item)
                for item in (
                    original_list_plan,
                    english_list_plan,
                )
            ],
            "count_exact": not truncated,
            "display_limit": list_plan.limit,
        })
        return QueryResult(
            query_result_id=uuid7(),
            provider="SEMANTIC",
            columns=tuple(
                dict(item)
                for item in original_list_response.get("columns") or ()
            ),
            rows=tuple(dict(item) for item in merged_rows),
            row_count=len(merged_probe_rows),
            truncated=truncated,
            warnings=(
                expansion_warnings
                + (
                    (self._enumeration_truncation_warning(list_plan.limit),)
                    if truncated else ()
                )
            ),
            provenance=provenance,
        )

    async def _execute_km_asset_multilingual_count(
        self,
        *,
        context: ExecutionContext,
        question: str,
        consumer_app_id: str,
        agent_version_id: UUID,
        auth_context: AuthContext,
        plan: DataQueryPlanV1,
        topic_terms: tuple[str, ...],
        expansion_warnings: tuple[str, ...],
    ) -> QueryResult:
        """并发统计原语言与英文词组及交集，返回准确去重数量。"""
        original_term, *english_terms = topic_terms
        original_plan = self._count_plan(
            self._replace_topic_groups(plan, ((original_term,),))
        )
        english_plan = self._count_plan(
            self._replace_topic_groups(plan, (tuple(english_terms),))
        )
        overlap_plan = self._count_plan(
            self._replace_topic_groups(
                plan, ((original_term,), tuple(english_terms))
            )
        )
        results = await self._run_plan_variants(
            context=context,
            question=question,
            consumer_app_id=consumer_app_id,
            agent_version_id=agent_version_id,
            auth_context=auth_context,
            variants=(
                ("topic-original-count", original_plan),
                ("topic-english-count", english_plan),
                ("topic-overlap-count", overlap_plan),
            ),
        )
        original_run_id, original_response = results["topic-original-count"]
        english_run_id, english_response = results["topic-english-count"]
        overlap_run_id, overlap_response = results["topic-overlap-count"]
        total_count = (
            self._asset_count(original_response)
            + self._asset_count(english_response)
            - self._asset_count(overlap_response)
        )
        if total_count < 0:
            raise RuntimeError("KM_ASSET_MULTILINGUAL_COUNT_INVALID")
        run_ids = (original_run_id, english_run_id, overlap_run_id)
        provenance = dict(original_response.get("provenance") or {})
        provenance.update({
            "data_query_run_id": str(original_run_id),
            "data_query_run_ids": [str(item) for item in run_ids],
            "topic_search_mode": "ORIGINAL_AND_ENGLISH_PARALLEL",
            "topic_terms": list(topic_terms),
            "query_plan_hashes": [
                self._plan_hash(item)
                for item in (original_plan, english_plan, overlap_plan)
            ],
            "count_exact": not any(
                bool(item.get("truncated"))
                for item in (
                    original_response,
                    english_response,
                    overlap_response,
                )
            ),
        })
        return QueryResult(
            query_result_id=uuid7(),
            provider="SEMANTIC",
            columns=tuple(
                dict(item) for item in original_response.get("columns") or ()
            ),
            rows=({"asset_count": total_count},),
            row_count=1,
            truncated=False,
            warnings=expansion_warnings,
            provenance=provenance,
        )

    async def _run_plan_variants(
        self,
        *,
        context: ExecutionContext,
        question: str,
        consumer_app_id: str,
        agent_version_id: UUID,
        auth_context: AuthContext,
        variants: tuple[tuple[str, DataQueryPlanV1], ...],
    ) -> dict[str, tuple[UUID, dict]]:
        """并发执行具备独立幂等键的问数计划变体。"""
        responses = await asyncio.gather(*(
            self._run_plan(
                context=context,
                question=question,
                consumer_app_id=consumer_app_id,
                agent_version_id=agent_version_id,
                auth_context=auth_context,
                plan=plan,
                idempotency_suffix=suffix,
            )
            for suffix, plan in variants
        ))
        return {
            suffix: response
            for (suffix, _), response in zip(
                variants, responses, strict=True
            )
        }

    async def _km_topic_terms(
        self,
        *,
        context: ExecutionContext,
        question: str,
        plan: DataQueryPlanV1,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """为中日韩主题生成两到三个英文补充词；英文及其他语言保持单路。"""
        topic_filter = next(
            (item for item in plan.filters if item.field == "topic"),
            None,
        )
        if topic_filter is None or len(topic_filter.values) != 1:
            raise ValueError("KM topic 筛选缺少单一原语言主题")
        original_topic = str(topic_filter.values[0]).strip()
        language = detect_unicode_language(question)
        if (
            language not in _KM_TOPIC_EXPANSION_LANGUAGES
            or detect_unicode_language(original_topic) == "en-US"
        ):
            return (original_topic,), ()

        agent = context.config_snapshot.get("agent", {})
        model_name = str(
            agent_model_name(agent, "data_planner_llm")
            or agent_model_name(agent, "composer_llm")
            or ""
        ).strip()
        if not model_name:
            return (original_topic,), (
                "主题英文补充检索不可用，已仅使用原语言检索",
            )
        prompt = await self._prompt_resolver.resolve(
            "agent_runtime.km_topic_english_expand"
        )
        messages = [
            {"role": "system", "content": prompt.content},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "source_language": language,
                        "question": question,
                        "original_topic": original_topic,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        last_error = ""
        for attempt in range(2):
            try:
                response = await self._model_client.get_llm_json(
                    served_model_name=model_name,
                    prompt=messages,
                )
                expansion = KMTopicExpansion.model_validate(response)
                if expansion.source_language != language:
                    raise ValueError("source_language 与输入不一致")
                if expansion.original_topic.strip() != original_topic:
                    raise ValueError("original_topic 不得改写")
                english_topics = tuple(
                    item.strip() for item in expansion.english_topics
                )
                return (original_topic, *english_topics), ()
            except (TypeError, ValueError) as exc:
                last_error = str(exc)
                if attempt == 0:
                    messages.append({
                        "role": "system",
                        "content": (
                            "上一个结果不符合 KMTopicExpansion.v1，"
                            f"请只输出修正后的 JSON。错误：{last_error}"
                        ),
                    })
        logger.warning(
            "KM 主题英文补充提取失败，退化为原语言单路：language={} error={}",
            language,
            last_error,
        )
        return (original_topic,), (
            "主题英文补充检索失败，已仅使用原语言检索",
        )

    @staticmethod
    def _replace_topic_groups(
        plan: DataQueryPlanV1, groups: tuple[tuple[str, ...], ...]
    ) -> DataQueryPlanV1:
        """复制计划；组内 topic 按 OR，组之间按 AND 组合。"""
        filters = tuple(
            item for item in plan.filters if item.field != "topic"
        ) + tuple(
            PlanFilter(
                field="topic",
                operator="CONTAINS",
                values=group,
            )
            for group in groups
        )
        return plan.model_copy(update={"filters": filters})

    @staticmethod
    def _count_plan(plan: DataQueryPlanV1) -> DataQueryPlanV1:
        return plan.model_copy(update={
            "dimensions": (),
            "order_by": (),
            "limit": 1,
        })

    @staticmethod
    def _enumeration_probe_plan(plan: DataQueryPlanV1) -> DataQueryPlanV1:
        """列表只多取一条作为截断探针，展示边界仍由原计划决定。"""
        return plan.model_copy(update={"limit": plan.limit + 1})

    @staticmethod
    def _enumeration_truncation_warning(display_limit: int) -> str:
        return f"相关 Asset 超过 {display_limit} 个，清单已截断"

    @staticmethod
    def _is_asset_count_plan(plan: DataQueryPlanV1) -> bool:
        """只对无分组的 Asset 数量执行双路集合容斥。"""
        return (
            not plan.dimensions
            and len(plan.measures) == 1
            and plan.measures[0].name == "asset_count"
            and plan.measures[0].aggregation == "COUNT"
        )

    @staticmethod
    def _asset_rows_up_to(
        result: dict, *, query_limit: int
    ) -> tuple[dict, ...]:
        rows = result.get("preview_rows")
        if not isinstance(rows, list) or any(
            not isinstance(item, dict) for item in rows
        ):
            raise RuntimeError("SEMANTIC_DATA_QUERY_INVALID_RESULT")
        if len(rows) > query_limit:
            raise RuntimeError(
                "KM_ASSET_ENUMERATION_RESULT_INCONSISTENT: "
                f"list={len(rows)}, query_limit={query_limit}"
            )
        return tuple(dict(item) for item in rows)

    @staticmethod
    def _asset_count(result: dict) -> int:
        rows = result.get("preview_rows")
        if not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError("KM_ASSET_COUNT_RESULT_INVALID")
        row = rows[0]
        if not isinstance(row, dict):
            raise RuntimeError("KM_ASSET_COUNT_RESULT_INVALID")
        value = next(
            (
                item for key, item in row.items()
                if str(key).casefold() == "asset_count"
            ),
            None,
        )
        if isinstance(value, bool):
            raise RuntimeError("KM_ASSET_COUNT_RESULT_INVALID")
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("KM_ASSET_COUNT_RESULT_INVALID") from exc
        if count < 0:
            raise RuntimeError("KM_ASSET_COUNT_RESULT_INVALID")
        return count

    async def _create_plan(self, *, context, question, models) -> DataQueryPlanV1:
        agent = context.config_snapshot.get("agent", {})
        model_name = str(
            agent_model_name(agent, "data_planner_llm")
            or agent_model_name(agent, "composer_llm")
            or ""
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 data_planner_llm 或 composer_llm")
        prompt = await self._prompt_resolver.resolve("agent_runtime.data_query_plan")
        constraints = next(
            (
                item.payload
                for item in reversed(context.input_artifacts)
                if item.artifact_type == "DATA_QUERY_CONSTRAINTS"
            ),
            None,
        )
        messages = [
            {"role": "system", "content": prompt.content},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "models": models,
                        "document_constraints": constraints,
                        "answer_basis": self._answer_basis(context),
                    },
                    ensure_ascii=False,
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
                normalized = self._normalize_plan_response(
                    response=response,
                    models=models,
                    question=question,
                    consumer_app_id=str(agent.get("owner_app_id") or ""),
                    answer_basis=self._answer_basis(context),
                )
                plan = DataQueryPlanV1.model_validate(normalized)
                self._validate_km_topic_plan(
                    context=context,
                    consumer_app_id=str(agent.get("owner_app_id") or ""),
                    plan=plan,
                )
                return plan
            except (TypeError, ValueError) as exc:
                last_error = str(exc)
                if attempt == 0:
                    messages.extend([
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                response, ensure_ascii=False, default=str
                            ),
                        },
                        {
                            "role": "system",
                            "content": (
                                "上一个计划不符合问数契约，请仅输出修正后的 "
                                f"DataQueryPlan.v1 JSON。错误：{last_error}"
                            ),
                        },
                    ])
        raise ValueError(f"问数 Planner 模型输出不符合契约：{last_error}")

    async def _execute_asset_search_plan(
        self,
        *,
        context: ExecutionContext,
        question: str,
        consumer_app_id: str,
        agent_version_id: UUID,
        auth_context: AuthContext,
        search_plan: AssetSearchPlanV1,
        query_plan: DataQueryPlanV1,
        models: list[dict],
    ) -> QueryResult:
        """执行统一计划；KM 问数阶段不再二次调用规划模型。"""
        semantic = search_plan.has_semantic_eligibility or bool(
            search_plan.preferences
        )
        if search_plan.operation == "LIST" and not semantic:
            result = await self._execute_km_asset_enumeration(
                context=context,
                question=question,
                consumer_app_id=consumer_app_id,
                agent_version_id=agent_version_id,
                auth_context=auth_context,
                list_plan=query_plan,
                topic_terms=(),
                expansion_warnings=(),
            )
            return result.model_copy(update={
                "provenance": {
                    **result.provenance,
                    "asset_search_plan": search_plan.model_payload(),
                    "planning_mode": "ASSET_SEARCH_DETERMINISTIC",
                },
            })
        aggregate_run_id, aggregate_response = await self._run_plan(
            context=context,
            question=question,
            consumer_app_id=consumer_app_id,
            agent_version_id=agent_version_id,
            auth_context=auth_context,
            plan=query_plan,
            idempotency_suffix="asset-search-primary",
        )
        primary = self._query_result_from_response(
            run_id=aggregate_run_id,
            result=aggregate_response,
            plan=query_plan,
        )
        provenance = {
            **primary.provenance,
            "asset_search_plan": search_plan.model_payload(),
            "planning_mode": "ASSET_SEARCH_DETERMINISTIC",
            "unsupported_requests": list(search_plan.unsupported_requests),
        }
        if search_plan.operation == "LIST":
            return primary.model_copy(update={"provenance": provenance})
        if "asset_id" in query_plan.dimensions:
            return primary.model_copy(update={"provenance": provenance})

        sample_payload = search_plan.model_dump(mode="json")
        sample_payload.update({
            "operation": "LIST",
            "target": "ASSET",
            "measures": [],
            "group_by": [],
            "include_total_count": False,
            "display_limit": search_plan.result_assets.target_count,
            "result_assets": {
                "mode": "PRIMARY",
                "target_count": search_plan.result_assets.target_count,
                "selection": "RECENT_WITHIN_RESULT",
            },
            "order_by": [{
                "field": "asset_date",
                "direction": "DESC",
            }],
        })
        sample_search_plan = AssetSearchPlanV1.model_validate(sample_payload)
        sample_plan = self._compile_asset_plan(sample_search_plan, models)
        sample = await self._execute_km_asset_enumeration(
            context=context,
            question=question,
            consumer_app_id=consumer_app_id,
            agent_version_id=agent_version_id,
            auth_context=auth_context,
            list_plan=sample_plan,
            topic_terms=(),
            expansion_warnings=(),
        )
        provenance.update({
            "data_query_run_ids": [
                str(aggregate_run_id),
                *list(sample.provenance.get("data_query_run_ids") or []),
            ],
            "supporting_query_result_id": str(sample.query_result_id),
        })
        return primary.model_copy(update={
            "supporting_columns": sample.columns,
            "supporting_rows": sample.rows,
            "supporting_query_result_id": sample.query_result_id,
            "provenance": provenance,
            "warnings": primary.warnings + sample.warnings,
        })

    def _compile_asset_plan(
        self, search_plan: AssetSearchPlanV1, models: list[dict]
    ) -> DataQueryPlanV1:
        """Asset 专属执行器必须显式提供搜索计划编译器。"""
        del search_plan, models
        raise RuntimeError("ASSET_SEARCH_COMPILER_NOT_REGISTERED")

    @staticmethod
    def _validate_km_topic_plan(
        *, context, consumer_app_id: str, plan: DataQueryPlanV1
    ) -> None:
        """确保 KM 主题计数与列举共享同一 topic 过滤口径。"""
        answer_basis = SemanticDataQueryExecutor._answer_basis(context)
        if (
            consumer_app_id != "km_asset"
            or answer_basis not in {
                "SEMANTIC_RELEVANCE_AGGREGATE",
                "SEMANTIC_RELEVANCE_ENUMERATION",
            }
        ):
            return
        topic_filters = [
            item for item in plan.filters if item.field == "topic"
        ]
        if len(topic_filters) != 1:
            raise ValueError("KM 主题问数必须且只能包含一个 topic 筛选")
        topic_filter = topic_filters[0]
        if (
            topic_filter.operator != "CONTAINS"
            or len(topic_filter.values) != 1
            or not str(topic_filter.values[0]).strip()
        ):
            raise ValueError("KM topic 筛选必须使用单值 CONTAINS")
        if "topic" in plan.dimensions or any(
            item.field == "topic" for item in plan.order_by
        ):
            raise ValueError("KM topic 是仅筛选维度，不能分组或排序")
        if answer_basis == "SEMANTIC_RELEVANCE_ENUMERATION" and not {
            "asset_id", "title", "bundle_id", "bundle_revision_id"
        }.issubset(plan.dimensions):
            raise ValueError("KM 主题列举缺少 Asset 或 Bundle 标识维度")

    @staticmethod
    def _answer_basis(context) -> str | None:
        """从冻结路由快照读取结构化答案依据。"""
        route = context.config_snapshot.get("route")
        value = (
            route.get("answer_basis")
            if isinstance(route, Mapping)
            else getattr(route, "answer_basis", None)
        )
        return str(value) if value is not None else None

    @staticmethod
    def _asset_search_plan(context) -> AssetSearchPlanV1 | None:
        route = context.config_snapshot.get("route") or {}
        payload = (
            route.get("asset_search_plan")
            if isinstance(route, Mapping)
            else getattr(route, "asset_search_plan", None)
        )
        if not payload:
            return None
        return AssetSearchPlanV1.model_validate(payload)

    @staticmethod
    def _normalize_plan_response(
        *, response, models, question: str, consumer_app_id: str,
        answer_basis: str | None = None,
    ) -> dict:
        """只使用规划目录中的值修复常见 LLM JSON 类型与缺省字段。"""
        if not isinstance(response, dict):
            raise ValueError("问数 Planner 返回的计划不是 JSON Object")
        # 部分模型会在合法计划旁回显 Planner 输入；只清理这三个已知
        # 输入字段，其他未知字段仍由 DataQueryPlanV1 严格拒绝。
        normalized = {
            key: value
            for key, value in response.items()
            if key not in _PLANNER_INPUT_ECHO_FIELDS
        }
        selected = next(
            (
                item for item in models
                if str(item.get("semantic_model_id"))
                == str(normalized.get("semantic_model_id"))
            ),
            models[0] if len(models) == 1 else None,
        )
        if not isinstance(selected, dict):
            return normalized
        normalized.setdefault(
            "semantic_model_id", selected.get("semantic_model_id")
        )
        normalized.setdefault(
            "semantic_model_version",
            selected.get("semantic_model_version"),
        )
        datasets = {
            str(item.get("name")): item
            for item in selected.get("datasets") or ()
            if isinstance(item, dict) and item.get("name")
        }
        if normalized.get("dataset") not in datasets and len(datasets) == 1:
            normalized["dataset"] = next(iter(datasets))
        catalog_measures = {
            str(item.get("name")): item
            for item in selected.get("measures") or ()
            if isinstance(item, dict) and item.get("name")
        }
        measures = []
        for raw in normalized.get("measures") or ():
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            catalog = catalog_measures.get(str(item.get("name")))
            if catalog is not None and not item.get("aggregation"):
                item["aggregation"] = catalog.get("aggregation")
            measures.append(item)
        if not measures and consumer_app_id == "km_asset":
            measure_name = "author_count" if any(
                phrase in question.casefold()
                for phrase in (
                    "作者数量", "用户数量", "多少个作者",
                    "多少位作者", "多少名作者",
                )
            ) else "asset_count"
            catalog = catalog_measures.get(measure_name)
            if catalog is not None:
                measures = [{
                    "name": measure_name,
                    "aggregation": catalog.get("aggregation"),
                }]
        normalized["measures"] = measures
        catalog_dimensions = {
            str(item.get("name")): item
            for item in selected.get("dimensions") or ()
            if isinstance(item, dict) and item.get("name")
        }
        filters = []
        for raw in normalized.get("filters") or ():
            if not isinstance(raw, dict):
                continue
            field = str(raw.get("field") or raw.get("dimension") or "")
            if field not in catalog_dimensions:
                field = str(raw.get("field") or "")
            operator = str(raw.get("operator") or "").upper()
            if "values" in raw:
                raw_values = raw.get("values")
                values = (
                    list(raw_values)
                    if isinstance(raw_values, (list, tuple))
                    else [raw_values]
                )
            elif "value" in raw:
                values = [raw.get("value")]
            else:
                values = []
            catalog = catalog_dimensions.get(field)
            allowed = (
                tuple(catalog.get("allowed_filter_operators") or ())
                if isinstance(catalog, dict)
                else ()
            )
            if allowed and operator not in allowed:
                if len(values) > 1 and "IN" in allowed:
                    operator = "IN"
                elif "EQ" in allowed:
                    operator = "EQ"
            filters.append({
                "field": field,
                "operator": operator,
                "values": values,
            })
        normalized["filters"] = filters
        if (
            consumer_app_id == "km_asset"
            and "ingestion_status" in catalog_dimensions
        ):
            normalized["filters"] = [
                item for item in normalized["filters"]
                if item.get("field") != "ingestion_status"
            ]
            normalized["filters"].append({
                "field": "ingestion_status",
                "operator": "EQ",
                "values": ["READY"],
            })
        if answer_basis == "SEMANTIC_RELEVANCE_ENUMERATION":
            primary_topic_seen = False
            scoped_filters = []
            for item in normalized["filters"]:
                if item.get("field") != "topic":
                    scoped_filters.append(item)
                    continue
                if not primary_topic_seen:
                    scoped_filters.append(item)
                    primary_topic_seen = True
            normalized["filters"] = scoped_filters
        if consumer_app_id == "km_asset" and (
            answer_basis in _KM_ENUMERATION_ANSWER_BASES
        ):
            required = (
                "asset_id", "title", "bundle_id", "bundle_revision_id"
            )
            missing = [
                name for name in required if name not in catalog_dimensions
            ]
            if missing:
                raise ValueError(f"KM 托管模型缺少列举维度：{missing}")
            normalized["dimensions"] = [
                name
                for name in (
                    *required, "product", "solution", "ingestion_status",
                    "asset_date",
                )
                if name in catalog_dimensions
            ]
            asset_count = catalog_measures.get("asset_count")
            if asset_count is None:
                raise ValueError("KM 托管模型缺少 asset_count 指标")
            normalized["measures"] = [{
                "name": "asset_count",
                "aggregation": asset_count.get("aggregation"),
            }]
            if not normalized.get("order_by"):
                normalized["order_by"] = [{
                    "field": (
                        "asset_date"
                        if "asset_date" in catalog_dimensions
                        else "title"
                    ),
                    "direction": (
                        "DESC"
                        if "asset_date" in catalog_dimensions
                        else "ASC"
                    ),
                }]
        raw_dimensions = normalized.get("dimensions")
        if raw_dimensions is None or isinstance(
            raw_dimensions, (list, tuple)
        ):
            projected_dimensions = list(raw_dimensions or ())
            projected_measures = {
                str(item.get("name"))
                for item in normalized["measures"]
                if isinstance(item, dict) and item.get("name")
            }
            for order in normalized.get("order_by") or ():
                if not isinstance(order, dict):
                    continue
                field = str(order.get("field") or "")
                if (
                    field in catalog_dimensions
                    and field not in projected_dimensions
                ):
                    projected_dimensions.append(field)
                    continue
                if field in catalog_measures and field not in projected_measures:
                    catalog = catalog_measures[field]
                    normalized["measures"].append({
                        "name": field,
                        "aggregation": catalog.get("aggregation"),
                    })
                    projected_measures.add(field)
            normalized["dimensions"] = projected_dimensions
        raw_limit = normalized.get("limit")
        if isinstance(raw_limit, str) and raw_limit.strip().isdigit():
            raw_limit = int(raw_limit.strip())
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
            raw_limit = 100
        max_rows = selected.get("max_rows")
        if isinstance(max_rows, int):
            raw_limit = (
                min(raw_limit, 10, max_rows)
                if (
                    consumer_app_id == "km_asset"
                    and answer_basis in _KM_ENUMERATION_ANSWER_BASES
                )
                else min(raw_limit, max_rows)
            )
        normalized["limit"] = max(1, min(raw_limit, 10_000))
        for field in ("dimensions", "order_by"):
            normalized.setdefault(field, [])
        normalized.setdefault("time_zone", "Asia/Shanghai")
        return normalized

    async def _wait_result(self, *, run_id, auth_context, deadline_at):
        while True:
            if deadline_at is not None and datetime.now(UTC) >= deadline_at.astimezone(UTC):
                await self._client.cancel_run(data_query_run_id=run_id, auth_context=auth_context)
                raise RuntimeError("SEMANTIC_DATA_QUERY_TIMEOUT")
            view = await self._client.get_run(data_query_run_id=run_id, auth_context=auth_context)
            status = str(view.get("status") or "")
            if status in {"COMPLETED", "COMPLETED_EMPTY"}:
                return await self._client.get_result(data_query_run_id=run_id, auth_context=auth_context)
            if status in {"REJECTED", "FAILED", "TIMED_OUT", "CANCELLED"}:
                raise RuntimeError(str(view.get("error_code") or "SEMANTIC_DATA_QUERY_FAILED"))
            await asyncio.sleep(1)

    @staticmethod
    def _auth_context(context: ExecutionContext) -> AuthContext:
        try:
            auth_context = AuthContext.model_validate(context.policy_snapshot["auth_context"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("AUTH_CONTEXT_INVALID") from exc
        if auth_context.domain_id != str(context.domain_id):
            raise RuntimeError("DOMAIN_CONTEXT_MISMATCH")
        return auth_context

    @staticmethod
    def _plan_hash(plan: DataQueryPlanV1) -> str:
        return hashlib.sha256(
            json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class DataQuerySkill:
    """Provider 只能由 Agent 快照决定，模型不能选择。"""

    def __init__(self, *, mcp_executor: MCPDataQueryExecutor, semantic_executor: SemanticDataQueryExecutor) -> None:
        self._executors = {"MCP": mcp_executor, "SEMANTIC": semantic_executor}

    async def execute(self, context: ExecutionContext) -> SkillResult:
        question = self._standalone_query(context)
        agent = dict(context.config_snapshot.get("agent") or {})
        agent_config = dict(agent.get("config") or {})
        mode = str(agent_config.get("data_query_mode") or "")
        executor = self._executors.get(mode)
        if executor is None:
            raise ValueError("Agent data_query_mode 无效")
        output = await executor.execute(context=context, question=question)
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="QUERY_RESULT",
                schema_version="QUERY_RESULT.v1",
                payload=output.model_dump(mode="json"),
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                    "provider": output.provider,
                    **output.provenance,
                },
            ),
            warnings=output.warnings,
        )

    async def execute_stream(self, context: ExecutionContext):
        """最终事件由 complete_task 与 QUERY_RESULT Artifact 原子发布。"""
        result = await self.execute(context)
        yield result

    @staticmethod
    def _standalone_query(context: ExecutionContext) -> str:
        for artifact in reversed(context.input_artifacts):
            if artifact.artifact_type == "CONTEXT_REWRITE":
                value = str((artifact.payload or {}).get("standalone_query") or "").strip()
                if value:
                    return value
        return context.original_input
