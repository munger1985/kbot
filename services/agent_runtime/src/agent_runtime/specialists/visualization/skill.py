"""将统一 QueryResult 转换为受控 ECharts 配置。"""

from typing import Any

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.language import language_instruction, response_language
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from agent_runtime.specialists.data_query.contracts import QueryResult

from .contracts import EChartsResult


class EChartsSkill:
    def __init__(self, *, model_client, prompt_resolver):
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, context: ExecutionContext) -> SkillResult:
        query = self._query_result(context)
        agent = context.config_snapshot.get("agent", {})
        model_name = str(
            agent_model_name(agent, "chart_llm")
            or agent_model_name(agent, "composer_llm")
            or ""
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 ECharts 生成模型")
        prompt = await self._prompt_resolver.resolve(
            "agent_runtime.generate_echarts"
        )
        language = response_language(
            context.config_snapshot, context.original_input
        )
        response = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=[
                {"role": "system", "content": prompt.content},
                {
                    "role": "system",
                    "content": language_instruction(language),
                },
                {
                    "role": "user",
                    "content": (
                        f"response_language={language}\n"
                        f"用户要求：{context.original_input}\n"
                        f"查询结果：{query.model_dump_json()}"
                    ),
                },
            ],
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
