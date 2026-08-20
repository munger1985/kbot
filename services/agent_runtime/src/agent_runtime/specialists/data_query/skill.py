"""由冻结 Agent 配置选择 MCP 或语义问数 Provider。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from platform_core.contracts import AuthContext
from platform_core.contracts.data_query import DataQueryPlanV1
from platform_core.identity import uuid7

from .contracts import QueryResult


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
        plan = await self._create_plan(context=context, question=question, models=models)
        receipt = await self._client.create_run(
            payload={
                "idempotency_key": f"{context.run_id}:{context.task_id}",
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
        response = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=[
                {"role": "system", "content": prompt.content},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "models": models,
                            "document_constraints": constraints,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        normalized = self._normalize_plan_response(
            response=response,
            models=models,
            question=question,
            consumer_app_id=str(agent.get("owner_app_id") or ""),
        )
        return DataQueryPlanV1.model_validate(normalized)

    @staticmethod
    def _normalize_plan_response(
        *, response, models, question: str, consumer_app_id: str
    ) -> dict:
        """只使用规划目录中的值修复常见 LLM JSON 类型与缺省字段。"""
        if not isinstance(response, dict):
            raise ValueError("问数 Planner 返回的计划不是 JSON Object")
        normalized = dict(response)
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
        raw_limit = normalized.get("limit")
        if isinstance(raw_limit, str) and raw_limit.strip().isdigit():
            raw_limit = int(raw_limit.strip())
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
            raw_limit = 100
        max_rows = selected.get("max_rows")
        if isinstance(max_rows, int):
            raw_limit = min(raw_limit, max_rows)
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
