"""将统一 QueryResult 转换为渲染器无关的受控图表。"""

from agent_runtime.language import response_language
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from agent_runtime.specialists.data_query.contracts import QueryResult
from platform_core.visualization import ChartSkill as SharedChartSkill


class ChartSkill:
    """把 Runtime Artifact 适配到平台共用 Chart Skill。"""

    async def execute(self, context: ExecutionContext) -> SkillResult:
        query = self._query_result(context)
        language = response_language(
            context.config_snapshot, context.original_input
        )
        title = {
            "zh": "查询结果图表",
            "ja": "クエリ結果チャート",
            "ko": "쿼리 결과 차트",
        }.get(language.split("-", 1)[0], "Query result chart")
        output = SharedChartSkill.tabular(
            title=title,
            columns=query.columns,
            rows=query.rows,
            source_ids=(str(query.query_result_id),),
        )
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type="CHART_SPEC",
                schema_version="CHART_SPEC.v1",
                payload=output.model_dump(mode="json"),
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                },
            )
        )

    @staticmethod
    def _query_result(context: ExecutionContext) -> QueryResult:
        for artifact in reversed(context.input_artifacts):
            if artifact.artifact_type == "QUERY_RESULT":
                return QueryResult.model_validate(artifact.payload)
        raise ValueError("Chart Skill 缺少 QUERY_RESULT")
