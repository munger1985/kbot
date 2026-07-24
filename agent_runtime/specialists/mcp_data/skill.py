"""调用既有 SelectAI/AIReport 服务并生成受控 ECharts 配置。"""

from __future__ import annotations

from typing import Any

from agent_runtime.runtime import (
    ExecutionContext,
    SkillArtifact,
    SkillProgress,
    SkillResult,
)
from platform_core.identity import uuid7

from .contracts import EChartsResult, QueryResult


class MCPDataQuerySkill:
    def __init__(self, *, data_client):
        self._client = data_client

    async def execute(self, context: ExecutionContext) -> SkillResult:
        if self._client is None:
            raise RuntimeError("问数服务未配置或暂不可用")
        profile = str(
            context.config_snapshot.get("agent", {}).get(
                "data_profile_name"
            )
            or ""
        ).strip()
        if not profile:
            raise ValueError("问数 Agent 未配置 data_profile_name")
        question = self._standalone_query(context)
        result = await self._client.query(
            profile=profile,
            user=context.actor_id,
            question=question,
        )
        if any(not isinstance(item, dict) for item in result["rows"]):
            raise ValueError("问数服务返回的结果行不是对象")
        rows = tuple(dict(item) for item in result["rows"])
        output = QueryResult(
            query_result_id=uuid7(),
            profile=profile,
            question=question,
            rows=rows,
            row_count=len(rows),
            upstream_row_count=int(result["upstream_row_count"]),
            truncated=bool(result["truncated"]),
        )
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="QUERY_RESULT",
                schema_version="QueryResult.v1",
                payload=output.model_dump(mode="json"),
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                    "provider": "SelectAI/AIReport",
                },
            ),
            warnings=(
                ("问数结果超过行数上限，已截断",)
                if output.truncated
                else ()
            ),
        )

    @staticmethod
    def _standalone_query(context: ExecutionContext) -> str:
        for artifact in reversed(context.input_artifacts):
            if artifact.artifact_type == "CONTEXT_REWRITE":
                value = str(
                    (artifact.payload or {}).get("standalone_query") or ""
                ).strip()
                if value:
                    return value
        return context.original_input


class EChartsSkill:
    def __init__(self, *, model_client, prompt_resolver):
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, context: ExecutionContext) -> SkillResult:
        query = self._query_result(context)
        agent = context.config_snapshot.get("agent", {})
        model_name = str(
            agent.get("chart_llm_model_name")
            or agent.get("composer_llm_model_name")
            or ""
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 ECharts 生成模型")
        prompt = await self._prompt_resolver.resolve(
            "agent_runtime.generate_echarts"
        )
        messages = [
            {"role": "system", "content": prompt.content},
            {
                "role": "user",
                "content": (
                    f"用户要求：{context.original_input}\n"
                    f"查询结果：{query.model_dump_json()}"
                ),
            },
        ]
        response = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=messages,
            max_tokens=4096,
        )
        output = EChartsResult(
            chart_type=str(response.get("chart_type") or "custom"),
            title=(
                str(response["title"])
                if response.get("title") is not None
                else None
            ),
            option=self._validate_option(response.get("option")),
            query_result_id=query.query_result_id,
        )
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="ECHARTS_CONFIG",
                schema_version="EChartsResult.v1",
                payload=output.model_dump(mode="json"),
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                    "prompt": prompt.ref(),
                },
            )
        )

    @staticmethod
    def _query_result(context: ExecutionContext) -> QueryResult:
        for artifact in reversed(context.input_artifacts):
            if artifact.artifact_type == "QUERY_RESULT":
                return QueryResult.model_validate(artifact.payload)
        raise ValueError("ECharts Skill 缺少 QUERY_RESULT")

    @staticmethod
    def _validate_option(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("模型返回的 ECharts option 无效")
        EChartsSkill._validate_json_node(value)
        return value

    @staticmethod
    def _validate_json_node(value: Any) -> None:
        forbidden = {"__proto__", "prototype", "constructor"}
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError("ECharts option 包含禁止字段")
            for item in value.values():
                EChartsSkill._validate_json_node(item)
        elif isinstance(value, list):
            for item in value:
                EChartsSkill._validate_json_node(item)
        elif isinstance(value, str):
            normalized = value.casefold()
            if any(
                token in normalized
                for token in ("javascript:", "function(", "function (", "=>")
            ):
                raise ValueError("ECharts option 包含可执行脚本文本")
