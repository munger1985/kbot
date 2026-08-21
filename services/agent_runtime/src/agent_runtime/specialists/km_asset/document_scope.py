"""KM Asset 的结构化候选范围提取。"""

import base64
import binascii
from collections.abc import Mapping
from uuid import UUID

from platform_core.contracts import AssetSearchPlanV1

from agent_runtime.runtime import ExecutionContext, SkillArtifact, SkillResult
from agent_runtime.specialists.data_query.contracts import QueryResult
from agent_runtime.specialists.hybrid import DocumentScopeExtractSkill


class KmAssetDocumentScopeExtractSkill(DocumentScopeExtractSkill):
    """把问数资格集合冻结为 KM Asset Bundle 检索范围。"""

    async def execute(self, context: ExecutionContext) -> SkillResult:
        if self._answer_basis(context) in {
            "SEMANTIC_RELEVANCE_ENUMERATION",
            "EXACT_METADATA_ENUMERATION",
        }:
            return self._enumeration_scope(context)
        return await super().execute(context)

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
    def _enumeration_scope(cls, context: ExecutionContext) -> SkillResult:
        """保留全部候选供 KC 取证，展示数量由最终组合阶段决定。"""
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
        raw_plan = (
            route.get("asset_search_plan")
            if isinstance(route, Mapping)
            else getattr(route, "asset_search_plan", None)
        )
        search_plan = (
            AssetSearchPlanV1.model_validate(raw_plan) if raw_plan else None
        )
        source_rows = (
            query.supporting_rows
            if search_plan is not None
            and search_plan.operation in {"COUNT", "GROUP"}
            and query.supporting_rows
            else query.rows
        )
        selected = list(source_rows if search_plan is not None else source_rows[:10])
        assets: list[dict] = []
        targets: list[dict] = []
        for row in selected:
            asset = {str(key).casefold(): value for key, value in row.items()}
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
                "title": asset.get("title"),
                "asset_id": asset.get("asset_id"),
            })
        result_limit = (
            search_plan.result_assets.target_count
            if search_plan is not None
            else len(selected)
        )
        payload = {
            "query": context.original_input[:512],
            "entity_ids": [
                item.get("asset_id") for item in assets if item.get("asset_id")
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
        """将 UUID 字符串或 RAW(16) 传输值统一成 UUID 字符串。"""
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
            return str(UUID(bytes=raw)) if len(raw) == 16 else None
        except (binascii.Error, ValueError):
            return None
