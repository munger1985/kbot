"""Hybrid 中间结果只允许结构化约束或受限检索范围。"""

import base64
import binascii
import json
from collections.abc import Mapping
from uuid import UUID

from agent_runtime.domain.model_bindings import agent_model_name
from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from agent_runtime.specialists.data_query.contracts import QueryResult
from platform_core.contracts import AssetSearchPlanV1


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
        """用元数据资格集合冻结 Bundle 范围，不按展示数量提前截断。"""
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
        route = context.config_snapshot.get("route") or {}
        raw_search_plan = (
            route.get("asset_search_plan")
            if isinstance(route, Mapping)
            else getattr(route, "asset_search_plan", None)
        )
        search_plan = (
            AssetSearchPlanV1.model_validate(raw_search_plan)
            if raw_search_plan
            else None
        )
        source_rows = (
            query.supporting_rows
            if search_plan is not None
            and search_plan.operation in {"COUNT", "GROUP"}
            and query.supporting_rows
            else query.rows
        )
        selected = list(
            source_rows if search_plan is not None else source_rows[:10]
        )
        assets = []
        targets = []
        for row in selected:
            asset = {
                str(key).casefold(): value for key, value in row.items()
            }
            asset.update({
                "bundle_id": cls._uuid_row_value(row, "bundle_id"),
                "bundle_revision_id": cls._uuid_row_value(
                    row, "bundle_revision_id"
                ),
            })
            assets.append(asset)
            if not asset["bundle_id"] or not asset["bundle_revision_id"]:
                if search_plan is not None:
                    raise ValueError("KM_ASSET_SEARCHABLE_MAPPING_INVALID")
                continue
            targets.append({
                "bundle_id": asset["bundle_id"],
                "bundle_revision_id": asset["bundle_revision_id"],
                "title": asset["title"],
                "asset_id": asset["asset_id"],
            })
        result_limit = (
            search_plan.result_assets.target_count
            if search_plan is not None
            else len(selected)
        )
        payload = {
            "query": context.original_input[:512],
            "entity_ids": [
                item["asset_id"] for item in assets if item["asset_id"]
            ],
            "bundle_targets": targets,
            "assets": assets,
            "total_count": (
                query.row_count
                if search_plan is None or not (
                    search_plan.has_semantic_eligibility
                    or search_plan.preferences
                )
                else None
            ),
            "display_limit": result_limit,
            "truncated": query.truncated or len(query.rows) > len(selected),
            "asset_search_plan": (
                search_plan.model_payload() if search_plan is not None else None
            ),
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

    @classmethod
    def _uuid_row_value(cls, row: Mapping, name: str) -> str | None:
        """将问数结果中的 UUID 字符串或 RAW(16) 传输值统一为 UUID。"""
        value = cls._row_value(row, name)
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, str):
            try:
                return str(UUID(value))
            except ValueError:
                return None
        if not isinstance(value, Mapping):
            return None
        if value.get("encoding") != "base64" or not isinstance(
            value.get("value"), str
        ):
            return None
        try:
            raw = base64.b64decode(value["value"], validate=True)
            if len(raw) != 16:
                return None
            return str(UUID(bytes=raw))
        except (binascii.Error, ValueError):
            return None
