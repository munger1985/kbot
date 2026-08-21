"""Hybrid 中间结果只允许结构化约束或受限检索范围。"""

import json

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult


class _HybridExtractSkill:
    prompt_key: str
    artifact_type: str
    schema_version: str
    allowed_fields: frozenset[str]

    def __init__(self, *, model_client, prompt_resolver):
        self._model_client = model_client
        self._prompt_resolver = prompt_resolver

    async def execute(self, context: ExecutionContext) -> SkillResult:
        model_name = str(
            agent_model_name(
                context.config_snapshot.get("agent", {}),
                "composer_llm",
            )
            or ""
        ).strip()
        if not model_name:
            raise ValueError("Agent 未配置 composer_llm")
        prompt = await self._prompt_resolver.resolve(self.prompt_key)
        inputs = [
            {
                "artifact_type": item.artifact_type,
                "payload": item.payload,
            }
            for item in context.input_artifacts
        ]
        response = await self._model_client.get_llm_json(
            served_model_name=model_name,
            prompt=[
                {"role": "system", "content": prompt.content},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": context.original_input,
                            "inputs": inputs,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
        )
        if not isinstance(response, dict):
            raise ValueError("HYBRID_EXTRACT_INVALID")
        unexpected = set(response) - self.allowed_fields
        if unexpected:
            raise ValueError("HYBRID_EXTRACT_UNEXPECTED_FIELDS")
        return SkillResult(
            artifact=SkillArtifact(
                artifact_type=self.artifact_type,
                schema_version=self.schema_version,
                payload=response,
                provenance={
                    "run_id": str(context.run_id),
                    "task_id": str(context.task_id),
                    "prompt": prompt.ref(),
                },
            )
        )


class DataConstraintExtractSkill(_HybridExtractSkill):
    prompt_key = "agent_runtime.data_constraint_extract"
    artifact_type = "DATA_QUERY_CONSTRAINTS"
    schema_version = "DataQueryConstraints.v1"
    allowed_fields = frozenset(
        {"metrics", "thresholds", "time_range", "entities"}
    )


class DocumentScopeExtractSkill(_HybridExtractSkill):
    prompt_key = "agent_runtime.document_scope_extract"
    artifact_type = "DOCUMENT_SCOPE"
    schema_version = "DocumentScope.v1"
    allowed_fields = frozenset(
        {
            "query", "entity_ids", "time_range", "keywords",
            "bundle_targets", "assets", "total_count", "display_limit",
            "truncated",
        }
    )

    async def execute(self, context: ExecutionContext) -> SkillResult:
        result = await super().execute(context)
        query = str((result.artifact.payload or {}).get("query") or "").strip()
        if not query or len(query) > 512:
            raise ValueError("DOCUMENT_SCOPE_QUERY_INVALID")
        return result
