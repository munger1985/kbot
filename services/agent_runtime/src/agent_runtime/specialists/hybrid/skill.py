"""Hybrid 中间结果只允许结构化约束或受限检索范围。"""

import json
from collections.abc import Mapping

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from agent_runtime.specialists.data_query.contracts import QueryResult


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
        if self._answer_basis(context) == "SEMANTIC_RELEVANCE_ENUMERATION":
            return self._km_asset_enumeration_scope(context)
        result = await super().execute(context)
        query = str((result.artifact.payload or {}).get("query") or "").strip()
        if not query or len(query) > 512:
            raise ValueError("DOCUMENT_SCOPE_QUERY_INVALID")
        return result

    @staticmethod
    def _answer_basis(context: ExecutionContext) -> str:
        route = context.config_snapshot.get("route") or {}
        value = (
            route.get("answer_basis")
            if isinstance(route, Mapping)
            else getattr(route, "answer_basis", "")
        )
        return str(value or "")

    @classmethod
    def _km_asset_enumeration_scope(
        cls, context: ExecutionContext
    ) -> SkillResult:
        """用问数结果冻结最多十个 Asset，再定向获取对应正文。"""
        artifact = next(
            (
                item for item in reversed(context.input_artifacts)
                if item.artifact_type == "QUERY_RESULT"
            ),
            None,
        )
        if artifact is None:
            raise ValueError("KM_ASSET_ENUMERATION_QUERY_RESULT_MISSING")
        query = QueryResult.model_validate(artifact.payload)
        selected = list(query.rows[:10])
        assets = []
        targets = []
        for row in selected:
            asset = {
                key: cls._row_value(row, key)
                for key in (
                    "asset_id", "title", "product", "solution",
                    "bundle_id", "bundle_revision_id",
                )
            }
            assets.append(asset)
            if asset["bundle_id"] and asset["bundle_revision_id"]:
                targets.append({
                    "bundle_id": asset["bundle_id"],
                    "bundle_revision_id": asset["bundle_revision_id"],
                    "title": asset["title"],
                })
        payload = {
            "query": context.original_input[:512],
            "entity_ids": [
                item["asset_id"] for item in assets if item["asset_id"]
            ],
            "bundle_targets": targets,
            "assets": assets,
            "total_count": query.row_count,
            "display_limit": 10,
            "truncated": query.truncated or query.row_count > 10,
        }
        return SkillResult(artifact=SkillArtifact(
            artifact_type="DOCUMENT_SCOPE",
            schema_version="DocumentScope.v1",
            payload=payload,
            provenance={
                "run_id": str(context.run_id),
                "task_id": str(context.task_id),
                "source": "QUERY_RESULT.v1",
            },
        ))

    @staticmethod
    def _row_value(row: Mapping, name: str):
        for key, value in row.items():
            if str(key).casefold() == name.casefold():
                return value
        return None
