"""AIOps 运维知识资产、提炼、审核和两阶段检索服务。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, Callable
from uuid import UUID

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.contracts.knowledge import (
    DiagnosisCaseProfile,
    KnowledgeSearchRequest,
    KnowledgeTagAnalysisOutput,
    KnowledgeTagWrite,
    ManualProcedureCard,
    ManualProcedureStep,
    ManualProfile,
    ManualUploadMetadata,
)
from aiops_agent.entities import (
    OperationsKnowledgeAssetEntity,
    OperationsKnowledgeIndexEntity,
    OperationsKnowledgeReviewEntity,
    OperationsKnowledgeScopeEntity,
    OperationsKnowledgeSourceEntity,
    OperationsKnowledgeTagEntity,
    OperationsKnowledgeVersionEntity,
)
from platform_core.contracts import (
    AgentExecutionSpec,
    AuthContext,
    KnowledgeScopeSnapshotV1,
)
from platform_core.identity import uuid7
from platform_core.security import create_service_auth_context
from platform_clients.knowledge_core import KnowledgeCoreClient, KnowledgeCoreClientError
from platform_clients.model import AIModelConfigClient


MANUAL_COLLECTION = "operations-manuals"
CASE_COLLECTION = "diagnosis-cases"
PUBLISHABLE_STATUSES = frozenset({"DRAFT", "REVIEW_REQUIRED"})
TAG_CATEGORIES = frozenset({"RESOURCE_CATEGORY", "TECHNOLOGY_STACK", "VENDOR"})


def _now() -> datetime:
    return datetime.now(UTC)


def _normalize(value: str) -> str:
    return " ".join(value.strip().upper().split())


def _bundle_source_hash(documents: list[dict[str, Any]]) -> str:
    members = [
        {
            "ordinal": int(document["ordinal"]),
            "file_name": str(document["file_name"]),
            "content_sha256": str(document["content_sha256"]),
            "role": str(document["role"]),
            "required_flag": bool(document["required_flag"]),
        }
        for document in documents
    ]
    value = json.dumps(
        members,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _source_commands(text: str) -> list[str]:
    """只从原文代码块或明确命令行提取内容，不补写任何命令。"""
    commands: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in re.finditer(
        r"```(?:sql|bash|shell|sh|text|rman|powershell|cmd)?\s*\n(.*?)```",
        text,
        re.I | re.S,
    ):
        command = match.group(1).strip()
        if command and command not in commands:
            commands.append(command)
        spans.append(match.span())
    visible = list(text)
    for start, end in spans:
        visible[start:end] = " " * (end - start)
    for line in "".join(visible).splitlines():
        command = line.strip()
        if re.match(
            r"^(SELECT|ALTER|CREATE|DROP|BEGIN|DECLARE|RMAN>|srvctl|crsctl|"
            r"sqlplus|expdp|impdp|pg_dump|pg_restore|psql|mysql|mysqldump|"
            r"systemctl|dnf|yum|apt(?:-get)?|kubectl|docker)\b",
            command,
            re.I,
        ) and command not in commands:
            commands.append(command)
    return commands


def _asset_view(entity: OperationsKnowledgeAssetEntity) -> dict[str, Any]:
    return {
        "asset_id": str(entity.asset_id),
        "domain_id": int(entity.domain_id),
        "asset_kind": entity.asset_kind,
        "display_name": entity.display_name,
        "status": entity.status,
        "current_version_id": (
            str(entity.current_version_id) if entity.current_version_id else None
        ),
        "security_level": int(entity.security_level),
        "row_version": int(entity.row_version),
        "created_by": entity.created_by,
        "updated_by": entity.updated_by,
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


def _version_view(entity: OperationsKnowledgeVersionEntity) -> dict[str, Any]:
    return {
        "asset_version_id": str(entity.asset_version_id),
        "asset_id": str(entity.asset_id),
        "version_no": int(entity.version_no),
        "status": entity.status,
        "source_hash": entity.source_hash,
        "profile_schema_version": entity.profile_schema_version,
        "profile": entity.profile_json,
        "extraction_warnings": entity.extraction_warnings_json or [],
        "tag_analysis": entity.tag_analysis_json,
        "published_at": entity.published_at,
        "retired_at": entity.retired_at,
        "created_by": entity.created_by,
        "created_at": entity.created_at,
        "row_version": int(entity.row_version),
    }


def _scope_view(entity: OperationsKnowledgeScopeEntity) -> dict[str, Any]:
    return {
        "scope_id": str(entity.scope_id),
        "tag_id": str(entity.tag_id) if entity.tag_id else None,
        "kind": entity.scope_kind,
        "value": entity.scope_value,
        "normalized_value": entity.normalized_value,
        "source_kind": entity.source_kind,
        "source_locator": entity.source_locator_json,
    }


def _tag_view(entity: OperationsKnowledgeTagEntity) -> dict[str, Any]:
    return {
        "tag_id": str(entity.tag_id),
        "category": entity.category,
        "display_name": entity.display_name,
        "description": entity.description,
        "status": entity.status,
        "sort_order": int(entity.sort_order),
        "row_version": int(entity.row_version),
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


def _index_view(entity: OperationsKnowledgeIndexEntity) -> dict[str, Any]:
    return {
        "index_ref_id": str(entity.index_ref_id),
        "collection_id": str(entity.collection_id),
        "bundle_id": str(entity.bundle_id),
        "bundle_revision_id": str(entity.bundle_revision_id),
        "status": entity.index_status,
        "expected_row_version": (
            int(entity.expected_row_version)
            if entity.expected_row_version is not None else None
        ),
        "last_checked_at": entity.last_checked_at,
        "error_code": entity.error_code,
        "error_summary": entity.error_summary,
    }


def _profile_dimensions(version: OperationsKnowledgeVersionEntity) -> dict[str, set[str]]:
    """把两类提炼合同投影为统一、可解释的匹配维度。"""
    profile = dict(version.profile_json or {})
    dimensions: dict[str, set[str]] = {}
    if profile.get("schema_version") == "OPS_MANUAL_PROFILE.v1":
        for key, values in dict(profile.get("scope") or {}).items():
            dimensions[str(key).upper()] = {
                _normalize(str(value)) for value in values or () if str(value).strip()
            }
        return dimensions
    environment = dict(profile.get("environment") or {})
    signature = dict(profile.get("problem_signature") or {})
    mappings = {
        "DATABASE_TYPE": (environment.get("database_type"),),
        "DATABASE_VERSION": (environment.get("database_version"),),
        "TOPOLOGY": (environment.get("topology"),),
        "PROBLEM_CLASS": (signature.get("problem_class"),),
        "COMPONENT": (signature.get("component"),),
        "ERROR_CODE": tuple(signature.get("error_codes") or ()),
        "SIGNAL_NAME": tuple(signature.get("signal_names") or ()),
    }
    for key, values in mappings.items():
        normalized = {
            _normalize(str(value)) for value in values
            if value is not None and str(value).strip()
        }
        if normalized:
            dimensions[key] = normalized
    return dimensions


class OperationsKnowledgeService:
    """Registry 是业务真相；KC 只提供文档解析、索引和引用证据。"""

    def __init__(
        self,
        *,
        uow_factory: Callable,
        knowledge_core: KnowledgeCoreClient,
        knowledge_assistant_id: UUID,
        knowledge_assistant_version_id: UUID,
        knowledge_assistant_models: dict[str, UUID],
        knowledge_assistant_instruction: str,
        model_catalog: AIModelConfigClient | None = None,
        structured_model=None,
        prompt_registry=None,
    ) -> None:
        self._uow_factory = uow_factory
        self._kc = knowledge_core
        self._knowledge_assistant_id = knowledge_assistant_id
        self._knowledge_assistant_version_id = knowledge_assistant_version_id
        self._knowledge_assistant_models = dict(knowledge_assistant_models)
        self._knowledge_assistant_instruction = knowledge_assistant_instruction
        self._model_catalog = model_catalog
        self._structured_model = structured_model
        self._prompt_registry = prompt_registry

    @staticmethod
    def _actor(context: AuthContext) -> str:
        return context.asserted_user_id or context.client_id

    async def _collection(
        self, *, domain_id: int, name: str, context: AuthContext
    ) -> dict[str, Any]:
        try:
            catalog = await self._kc.list_collections(
                domain_id=domain_id, auth_context=context
            )
        except KnowledgeCoreClientError as exc:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE",
                message="运维知识索引服务暂时不可用",
                status_code=503,
                retryable=True,
            ) from exc
        matches = [
            item for item in catalog.get("collections", [])
            if item.get("display_name") == name
        ]
        if len(matches) != 1:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_INDEX_STALE",
                message=f"固定知识集合 {name} 未正确初始化",
                status_code=503,
                retryable=True,
            )
        collection = matches[0]
        metadata = collection.get("metadata") or {}
        if (
            metadata.get("owner_app_id") != "aiops"
            or metadata.get("fixed_resource") is not True
            or collection.get("status") != "ACTIVE"
        ):
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_INDEX_STALE",
                message=f"固定知识集合 {name} 的资源标识或状态无效",
                status_code=503,
                retryable=True,
            )
        return collection

    async def overview(self, *, domain_id: int) -> dict[str, int]:
        async with self._uow_factory() as uow:
            return await uow.operations_knowledge.overview(domain_id=domain_id)

    async def list_tags(
        self, *, domain_id: int, include_disabled: bool = True,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await uow.operations_knowledge.list_tags(
                domain_id=domain_id,
                status=None if include_disabled else "ACTIVE",
            )
            return [_tag_view(row) for row in rows]

    async def create_tag(
        self, *, domain_id: int, actor_id: str, command: KnowledgeTagWrite,
    ) -> dict[str, Any]:
        display_name = " ".join(command.display_name.split())
        normalized_name = _normalize(display_name)
        async with self._uow_factory() as uow:
            duplicate = await uow.operations_knowledge.find_tag_name(
                domain_id=domain_id, category=command.category,
                normalized_name=normalized_name,
            )
            if duplicate is not None:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_TAG_DUPLICATE",
                    message="同一标签分类中已存在同名标签", status_code=409,
                )
            entity = OperationsKnowledgeTagEntity(
                domain_id=domain_id, category=command.category,
                display_name=display_name, normalized_name=normalized_name,
                description=command.description.strip() if command.description else None,
                status="ACTIVE", sort_order=command.sort_order,
                created_by=actor_id, updated_by=actor_id,
            )
            await uow.operations_knowledge.add_tag(entity)
            await uow.commit()
            return _tag_view(entity)

    async def update_tag(
        self, *, domain_id: int, tag_id: UUID, actor_id: str,
        command: KnowledgeTagWrite,
    ) -> dict[str, Any]:
        if command.expected_row_version is None:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_TAG_VERSION_REQUIRED",
                message="修改标签必须提供当前版本号", status_code=422,
            )
        display_name = " ".join(command.display_name.split())
        normalized_name = _normalize(display_name)
        async with self._uow_factory() as uow:
            entity = await uow.operations_knowledge.get_tag(
                domain_id=domain_id, tag_id=tag_id, for_update=True
            )
            if entity is None:
                raise self._tag_not_found()
            if int(entity.row_version) != command.expected_row_version:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_TAG_VERSION_CONFLICT",
                    message="标签已被其他请求修改，请刷新后重试", status_code=412,
                )
            duplicate = await uow.operations_knowledge.find_tag_name(
                domain_id=domain_id, category=command.category,
                normalized_name=normalized_name, exclude_tag_id=tag_id,
            )
            if duplicate is not None:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_TAG_DUPLICATE",
                    message="同一标签分类中已存在同名标签", status_code=409,
                )
            entity.category = command.category
            entity.display_name = display_name
            entity.normalized_name = normalized_name
            entity.description = command.description.strip() if command.description else None
            entity.sort_order = command.sort_order
            entity.updated_by = actor_id
            await uow.commit()
            return _tag_view(entity)

    async def set_tag_status(
        self, *, domain_id: int, tag_id: UUID, actor_id: str,
        expected_row_version: int, status: str,
    ) -> dict[str, Any]:
        if status not in {"ACTIVE", "DISABLED"}:
            raise ValueError("status must be ACTIVE or DISABLED")
        async with self._uow_factory() as uow:
            entity = await uow.operations_knowledge.get_tag(
                domain_id=domain_id, tag_id=tag_id, for_update=True
            )
            if entity is None:
                raise self._tag_not_found()
            if int(entity.row_version) != expected_row_version:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_TAG_VERSION_CONFLICT",
                    message="标签已被其他请求修改，请刷新后重试", status_code=412,
                )
            entity.status = status
            entity.updated_by = actor_id
            await uow.commit()
            return _tag_view(entity)

    async def analyze_tags(
        self, *, domain_id: int, asset_version_id: UUID, context: AuthContext,
    ) -> dict[str, Any]:
        """对待审批手册做有界预分析，并让 LLM 只能从有效词表中选择。"""
        if self._structured_model is None or self._prompt_registry is None or self._model_catalog is None:
            return await self._mark_tagging_failed(
                domain_id=domain_id, asset_version_id=asset_version_id,
                code="TAG_MODEL_UNAVAILABLE", message="标签识别模型或 Prompt Registry 未初始化",
            )
        async with self._uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id
            )
            if version is None:
                raise self._not_found()
            asset = await uow.operations_knowledge.get_asset(domain_id=domain_id, asset_id=version.asset_id)
            relations = await uow.operations_knowledge.get_version_relations(asset_version_id=asset_version_id)
            tags = await uow.operations_knowledge.list_tags(domain_id=domain_id, status="ACTIVE")
            asset_kind = str(asset.asset_kind)
            version_status = str(version.status)
            index = next(iter(relations["indexes"]), None)
            index_locator = (
                (
                    index.collection_id,
                    index.bundle_id,
                    index.bundle_revision_id,
                )
                if index is not None
                else None
            )
            vocabulary = [{
                "tag_id": str(tag.tag_id), "category": str(tag.category),
                "display_name": str(tag.display_name),
                "description": tag.description,
            } for tag in tags]
        if asset_kind != "MANUAL" or version_status != "TAGGING":
            return await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
        if not vocabulary:
            return await self._mark_tagging_failed(
                domain_id=domain_id, asset_version_id=asset_version_id,
                code="TAG_VOCABULARY_EMPTY", message="请先在配置与治理中维护至少一个有效知识标签",
            )
        if index_locator is None:
            return await self._mark_tagging_failed(
                domain_id=domain_id, asset_version_id=asset_version_id,
                code="TAG_SOURCE_MISSING", message="知识版本缺少 KC 源文件引用",
            )
        try:
            collection_id, bundle_id, bundle_revision_id = index_locator
            projection = await self._kc.extract_review_text(
                domain_id=domain_id, collection_id=collection_id,
                bundle_id=bundle_id, bundle_revision_id=bundle_revision_id,
                auth_context=context,
            )
            excerpts = list(projection.get("excerpts") or [])
            if not excerpts:
                raise ValueError("审批预分析没有抽取到可用正文")
            model_id = self._knowledge_assistant_models.get("composer_llm")
            if model_id is None:
                raise ValueError("运维知识助手尚未配置 composer_llm")
            definition = await self._model_catalog.get_model(UUID(str(model_id)))
            technical_name = str(definition.get("served_model_name") or "").strip()
            if not technical_name:
                raise ValueError("标签识别模型缺少 served_model_name")
            prompt = await self._prompt_registry.resolve("knowledge_tag_classify")
            result = await self._structured_model.generate_structured(
                purpose="operations_knowledge.tag_classify",
                output_model=KnowledgeTagAnalysisOutput,
                model_snapshot={
                    "model_id": str(model_id), "technical_name": technical_name,
                    "revision": str(model_id),
                },
                prompt_ref={**prompt.ref(), "content": prompt.content},
                input_payload={"vocabulary": vocabulary, "documents": excerpts},
                deadline=None,
                idempotency_key=f"knowledge-tags:{asset_version_id}",
            )
            output = KnowledgeTagAnalysisOutput.model_validate(result.output)
            allowed = {item["tag_id"]: item for item in vocabulary}
            source_by_id = {str(item["document_version_id"]): item for item in excerpts}
            suggestions: list[dict[str, Any]] = []
            seen: set[str] = set()
            for suggestion in output.suggestions:
                tag = allowed.get(suggestion.tag_id)
                if tag is None or suggestion.tag_id in seen:
                    continue
                evidence = []
                for item in suggestion.evidence:
                    source = source_by_id.get(item.document_version_id)
                    if source is None or item.quote not in str(source.get("content") or ""):
                        continue
                    evidence.append({
                        "document_version_id": str(source["document_version_id"]),
                        "file_name": str(source.get("file_name") or ""),
                        "quote": item.quote,
                        "locator": dict(source.get("locator") or {}),
                    })
                if not evidence:
                    continue
                seen.add(suggestion.tag_id)
                suggestions.append({
                    "tag_id": suggestion.tag_id, "category": tag["category"],
                    "display_name": tag["display_name"],
                    "confidence": suggestion.confidence, "evidence": evidence,
                })
            analysis = {
                "schema_version": "OPS_KNOWLEDGE_TAG_ANALYSIS.v1",
                "status": "READY", "suggestions": suggestions,
                "warnings": list(projection.get("warnings") or []),
                "vocabulary": vocabulary,
                "vocabulary_hash": hashlib.sha256(json.dumps(vocabulary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "model_receipt": result.receipt.model_dump(mode="json"),
            }
        except KnowledgeCoreClientError as exc:
            if exc.status_code in {408, 429} or exc.status_code >= 500:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE",
                    message="运维知识标签识别服务暂时不可用",
                    status_code=503,
                    retryable=True,
                ) from exc
            return await self._mark_tagging_failed(
                domain_id=domain_id,
                asset_version_id=asset_version_id,
                code=exc.code,
                message=str(exc),
            )
        except (AIOpsModelError, LookupError, RuntimeError, ValueError) as exc:
            return await self._mark_tagging_failed(
                domain_id=domain_id, asset_version_id=asset_version_id,
                code=getattr(exc, "code", "TAG_ANALYSIS_FAILED"), message=str(exc),
            )
        async with self._uow_factory() as uow:
            locked = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id, for_update=True
            )
            if locked is None:
                raise self._not_found()
            locked_asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=locked.asset_id, for_update=True
            )
            if locked.status == "TAGGING":
                locked.tag_analysis_json = analysis
                locked.status = "REVIEW_REQUIRED"
                if locked_asset.current_version_id is None:
                    locked_asset.status = "REVIEW_REQUIRED"
                await uow.commit()
        return await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)

    async def _mark_tagging_failed(
        self, *, domain_id: int, asset_version_id: UUID, code: str, message: str,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id, for_update=True
            )
            if version is None:
                raise self._not_found()
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=version.asset_id, for_update=True
            )
            version.status = "TAGGING_FAILED"
            version.tag_analysis_json = {
                "schema_version": "OPS_KNOWLEDGE_TAG_ANALYSIS.v1", "status": "FAILED",
                "error_code": str(code), "error_summary": str(message)[:2000], "suggestions": [],
            }
            if asset.current_version_id is None:
                asset.status = "TAGGING_FAILED"
            await uow.commit()
        return await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)

    async def list_assets(
        self, *, domain_id: int, asset_kind: str | None, status: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await uow.operations_knowledge.list_assets(
                domain_id=domain_id, asset_kind=asset_kind, status=status, limit=limit
            )
            return [_asset_view(row) for row in rows]

    async def get_asset(self, *, domain_id: int, asset_id: UUID) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=asset_id
            )
            if asset is None:
                raise self._not_found()
            versions = await uow.operations_knowledge.list_versions(
                domain_id=domain_id, asset_id=asset_id
            )
            return {"asset": _asset_view(asset), "versions": [_version_view(v) for v in versions]}

    async def get_version(
        self, *, domain_id: int, asset_version_id: UUID
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id
            )
            if version is None:
                raise self._not_found()
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=version.asset_id
            )
            relations = await uow.operations_knowledge.get_version_relations(
                asset_version_id=asset_version_id
            )
            return {
                "asset": _asset_view(asset),
                "version": _version_view(version),
                "scopes": [_scope_view(item) for item in relations["scopes"]],
                "sources": [{
                    "source_id": str(item.source_id),
                    "source_kind": item.source_kind,
                    "report_id": str(item.source_report_id) if item.source_report_id else None,
                    "run_id": str(item.source_run_id) if item.source_run_id else None,
                    "artifact_id": str(item.source_artifact_id) if item.source_artifact_id else None,
                    "external_id": item.source_external_id,
                    "content_hash": item.content_hash,
                    "locator": item.source_locator_json,
                } for item in relations["sources"]],
                "indexes": [_index_view(item) for item in relations["indexes"]],
                "reviews": [{
                    "review_id": str(item.review_id),
                    "decision": item.decision,
                    "reviewer_id": item.reviewer_id,
                    "comment": item.comment_text,
                    "before_status": item.before_status,
                    "after_status": item.after_status,
                    "created_at": item.created_at,
                } for item in relations["reviews"]],
            }

    async def stream_source(
        self, *, domain_id: int, asset_version_id: UUID,
        context: AuthContext, range_header: str | None,
    ):
        detail = await self.get_version(
            domain_id=domain_id, asset_version_id=asset_version_id
        )
        index = next(iter(detail["indexes"]), None)
        if index is None:
            raise self._not_found()
        revision = await self._kc.get_revision_status(
            domain_id=domain_id,
            bundle_id=UUID(index["bundle_id"]),
            bundle_revision_id=UUID(index["bundle_revision_id"]),
            include_members=True,
            auth_context=context,
        )
        members = revision.get("members") or revision.get("items") or []
        member = next(
            (item for item in members if item.get("document_role") == "CONTENT"),
            next(iter(members), None),
        )
        if not member or not member.get("document_version_id"):
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_INDEX_PENDING",
                message="原始知识文件尚不可下载",
                status_code=409,
            )
        return await self._kc.stream_source_file(
            domain_id=domain_id,
            collection_id=UUID(index["collection_id"]),
            bundle_id=UUID(index["bundle_id"]),
            bundle_revision_id=UUID(index["bundle_revision_id"]),
            document_version_id=UUID(str(member["document_version_id"])),
            range_header=range_header,
            auth_context=context,
        )

    async def ingest_upload(
        self,
        *,
        domain_id: int,
        context: AuthContext,
        grouping_mode: str,
        bundle_title: str,
        documents: list[dict[str, Any]],
        metadata: ManualUploadMetadata,
        idempotency_key: str,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if grouping_mode == "EACH_FILE":
            for document in documents:
                try:
                    result = await self.ingest_manual(
                        domain_id=domain_id,
                        context=context,
                        file_name=str(document["file_name"]),
                        media_type=str(document["media_type"]),
                        body=bytes(document["body"]),
                        metadata=metadata.model_copy(update={
                            "display_name": (
                                metadata.display_name
                                if len(documents) == 1
                                else str(document["display_name"])
                            ),
                        }),
                        idempotency_key=(
                            f"{idempotency_key}:{document['client_file_id']}"
                        ),
                        declared_sha256=str(document["content_sha256"]),
                    )
                except AIOpsApplicationError as exc:
                    items.append({
                        "status": "REJECTED",
                        "client_file_id": document["client_file_id"],
                        "message": exc.message,
                    })
                    continue
                items.append({
                    "status": "ACCEPTED",
                    "client_file_id": document["client_file_id"],
                    "asset_id": result["asset"]["asset_id"],
                    "asset_version_id": result["version"]["asset_version_id"],
                    "replayed": bool(result.get("replayed")),
                })
        else:
            try:
                result = await self.ingest_manual_bundle(
                    domain_id=domain_id,
                    context=context,
                    title=bundle_title,
                    documents=documents,
                    metadata=metadata.model_copy(update={"display_name": bundle_title}),
                    idempotency_key=idempotency_key,
                )
            except AIOpsApplicationError as exc:
                items = [{
                    "status": "REJECTED",
                    "client_file_id": document["client_file_id"],
                    "message": exc.message,
                } for document in documents]
            else:
                items = [{
                    "status": "ACCEPTED",
                    "client_file_id": document["client_file_id"],
                    "asset_id": result["asset"]["asset_id"],
                    "asset_version_id": result["version"]["asset_version_id"],
                    "bundle_id": result.get("bundle_id"),
                    "bundle_revision_id": result.get("bundle_revision_id"),
                    "replayed": bool(result.get("replayed")),
                } for document in documents]
        return {
            "grouping_mode": grouping_mode,
            "items": items,
            "request_id": context.request_id,
        }

    async def ingest_manual_bundle(
        self,
        *,
        domain_id: int,
        context: AuthContext,
        title: str,
        documents: list[dict[str, Any]],
        metadata: ManualUploadMetadata,
        idempotency_key: str,
    ) -> dict[str, Any]:
        digest = _bundle_source_hash(documents)
        async with self._uow_factory() as uow:
            existing = await uow.operations_knowledge.find_by_source_hash(
                domain_id=domain_id,
                asset_kind="MANUAL",
                source_hash=digest,
            )
            if existing is not None:
                relations = await uow.operations_knowledge.get_version_relations(
                    asset_version_id=existing[1].asset_version_id
                )
                existing_index = next(iter(relations["indexes"]), None)
                return {
                    "asset": _asset_view(existing[0]),
                    "version": _version_view(existing[1]),
                    "bundle_id": (
                        str(existing_index.bundle_id) if existing_index else None
                    ),
                    "bundle_revision_id": (
                        str(existing_index.bundle_revision_id)
                        if existing_index else None
                    ),
                    "replayed": True,
                }
        actor = self._actor(context)
        asset_id = uuid7()
        version_id = uuid7()
        collection = await self._collection(
            domain_id=domain_id,
            name=MANUAL_COLLECTION,
            context=context,
        )
        response = await self._kc.ingest_user_bundle(
            domain_id=domain_id,
            collection_id=UUID(str(collection["collection_id"])),
            client_bundle_id=str(version_id),
            source_revision=str(version_id),
            title=title,
            security_level=metadata.security_level,
            documents=documents,
            idempotency_key=idempotency_key,
            auth_context=context,
        )
        intake_items = list((response.payload or {}).get("items", []))
        if not intake_items or any(
            item.get("status") == "REJECTED" for item in intake_items
        ):
            rejected = next((
                item for item in intake_items if item.get("status") == "REJECTED"
            ), {})
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                message=str(rejected.get("message") or "KC 未受理该运维资料 Bundle"),
                status_code=422,
            )
        bundle_ids = {str(item.get("bundle_id")) for item in intake_items}
        revision_ids = {
            str(item.get("bundle_revision_id")) for item in intake_items
        }
        if len(bundle_ids) != 1 or len(revision_ids) != 1 or "None" in bundle_ids | revision_ids:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                message="KC 返回的 Bundle 身份不一致",
                status_code=502,
            )
        bundle_id = UUID(next(iter(bundle_ids)))
        revision_id = UUID(next(iter(revision_ids)))
        asset = OperationsKnowledgeAssetEntity(
            asset_id=asset_id,
            domain_id=domain_id,
            asset_kind="MANUAL",
            display_name=title,
            status="TAGGING",
            security_level=metadata.security_level,
            created_by=actor,
            updated_by=actor,
        )
        version = OperationsKnowledgeVersionEntity(
            asset_version_id=version_id,
            asset_id=asset_id,
            version_no=1,
            status="TAGGING",
            source_hash=digest,
            extraction_warnings_json=[],
            created_by=actor,
        )
        scopes: list[OperationsKnowledgeScopeEntity] = []
        sources = [OperationsKnowledgeSourceEntity(
            asset_version_id=version_id,
            source_kind="MANUAL_FILE",
            source_external_id=str(document["file_name"]),
            content_hash=str(document["content_sha256"]),
            source_locator_json={
                "metadata": metadata.model_dump(mode="json"),
                "ordinal": int(document["ordinal"]),
                "role": str(document["role"]),
                "required_flag": bool(document["required_flag"]),
            },
        ) for document in documents]
        index = OperationsKnowledgeIndexEntity(
            asset_version_id=version_id,
            collection_id=UUID(str(collection["collection_id"])),
            bundle_id=bundle_id,
            bundle_revision_id=revision_id,
            index_status="PENDING_APPROVAL",
        )
        async with self._uow_factory() as uow:
            await uow.operations_knowledge.add_asset_version(
                asset=asset,
                version=version,
                scopes=scopes,
                sources=sources,
                indexes=[index],
            )
            await uow.commit()
        return {
            "asset": _asset_view(asset),
            "version": _version_view(version),
            "bundle_id": str(bundle_id),
            "bundle_revision_id": str(revision_id),
            "replayed": False,
        }

    async def ingest_manual(
        self,
        *,
        domain_id: int,
        context: AuthContext,
        file_name: str,
        media_type: str,
        body: bytes,
        metadata: ManualUploadMetadata,
        idempotency_key: str,
        declared_sha256: str | None,
    ) -> dict[str, Any]:
        digest = hashlib.sha256(body).hexdigest()
        if declared_sha256 and declared_sha256.lower() != digest:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                message="文件摘要与声明值不一致",
                status_code=422,
            )
        async with self._uow_factory() as uow:
            existing = await uow.operations_knowledge.find_by_source_hash(
                domain_id=domain_id, asset_kind="MANUAL", source_hash=digest
            )
            if existing is not None:
                return {
                    "asset": _asset_view(existing[0]),
                    "version": _version_view(existing[1]),
                    "replayed": True,
                }
        actor = self._actor(context)
        asset_id = uuid7()
        version_id = uuid7()
        collection = await self._collection(
            domain_id=domain_id, name=MANUAL_COLLECTION, context=context
        )
        response = await self._kc.ingest_user_file(
            domain_id=domain_id,
            collection_id=UUID(str(collection["collection_id"])),
            file_name=file_name,
            display_name=metadata.display_name,
            media_type=media_type,
            body=body,
            content_sha256=digest,
            idempotency_key=idempotency_key,
            auth_context=context,
            client_bundle_id=str(asset_id),
            source_revision=str(version_id),
            security_level=metadata.security_level,
        )
        item = next(iter((response.payload or {}).get("items", [])), None)
        if not item or item.get("status") == "REJECTED":
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                message=str((item or {}).get("message") or "KC 未受理该运维手册"),
                status_code=422,
            )
        bundle_id = UUID(str(item["bundle_id"]))
        revision_id = UUID(str(item["bundle_revision_id"]))
        asset = OperationsKnowledgeAssetEntity(
            asset_id=asset_id, domain_id=domain_id, asset_kind="MANUAL",
            display_name=metadata.display_name, status="TAGGING",
            security_level=metadata.security_level, created_by=actor, updated_by=actor,
        )
        version = OperationsKnowledgeVersionEntity(
            asset_version_id=version_id, asset_id=asset_id, version_no=1,
            status="TAGGING", source_hash=digest,
            extraction_warnings_json=[], created_by=actor,
        )
        scopes: list[OperationsKnowledgeScopeEntity] = []
        source = OperationsKnowledgeSourceEntity(
            asset_version_id=version_id,
            source_kind="MANUAL_FILE",
            source_external_id=file_name,
            content_hash=digest,
            source_locator_json={"metadata": metadata.model_dump(mode="json")},
        )
        index = OperationsKnowledgeIndexEntity(
            asset_version_id=version_id,
            collection_id=UUID(str(collection["collection_id"])),
            bundle_id=bundle_id,
            bundle_revision_id=revision_id,
            index_status="PENDING_APPROVAL",
        )
        async with self._uow_factory() as uow:
            await uow.operations_knowledge.add_asset_version(
                asset=asset, version=version, scopes=scopes,
                sources=[source], indexes=[index],
            )
            await uow.commit()
        return {"asset": _asset_view(asset), "version": _version_view(version), "replayed": False}

    async def ingest_manual_version(
        self,
        *,
        domain_id: int,
        asset_id: UUID,
        expected_asset_row_version: int,
        context: AuthContext,
        file_name: str,
        media_type: str,
        body: bytes,
        metadata: ManualUploadMetadata,
        idempotency_key: str,
        declared_sha256: str | None,
    ) -> dict[str, Any]:
        """为稳定手册资产建立新版本，旧发布版本在切换前继续可检索。"""
        digest = hashlib.sha256(body).hexdigest()
        if declared_sha256 and declared_sha256.lower() != digest:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                message="文件摘要与声明值不一致",
                status_code=422,
            )
        async with self._uow_factory() as uow:
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=asset_id
            )
            if asset is None or asset.asset_kind != "MANUAL":
                raise self._not_found()
            if int(asset.row_version) != expected_asset_row_version:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_VERSION_CONFLICT",
                    message="手册资产已被其他请求更新，请刷新后重试",
                    status_code=412,
                )
            existing = await uow.operations_knowledge.find_by_source_hash(
                domain_id=domain_id, asset_kind="MANUAL", source_hash=digest
            )
            if existing is not None:
                if existing[0].asset_id != asset_id:
                    raise AIOpsApplicationError(
                        code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                        message="相同内容已经登记到另一个手册资产",
                        status_code=409,
                    )
                return {
                    "asset": _asset_view(existing[0]),
                    "version": _version_view(existing[1]),
                    "replayed": True,
                }

        version_id = uuid7()
        collection = await self._collection(
            domain_id=domain_id, name=MANUAL_COLLECTION, context=context
        )
        response = await self._kc.ingest_user_file(
            domain_id=domain_id,
            collection_id=UUID(str(collection["collection_id"])),
            file_name=file_name,
            display_name=metadata.display_name,
            media_type=media_type,
            body=body,
            content_sha256=digest,
            idempotency_key=idempotency_key,
            auth_context=context,
            client_bundle_id=str(asset_id),
            source_revision=str(version_id),
            security_level=metadata.security_level,
        )
        item = next(iter((response.payload or {}).get("items", [])), None)
        if not item or item.get("status") == "REJECTED":
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                message=str((item or {}).get("message") or "KC 未受理该手册版本"),
                status_code=422,
            )
        revision_id = UUID(str(item["bundle_revision_id"]))
        actor = self._actor(context)
        async with self._uow_factory() as uow:
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=asset_id, for_update=True
            )
            if asset is None or asset.asset_kind != "MANUAL":
                raise self._not_found()
            if int(asset.row_version) != expected_asset_row_version:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_VERSION_CONFLICT",
                    message="手册资产在上传过程中已更新，请刷新后重试",
                    status_code=412,
                )
            version = OperationsKnowledgeVersionEntity(
                asset_version_id=version_id,
                asset_id=asset_id,
                version_no=await uow.operations_knowledge.next_version_no(
                    asset_id=asset_id
                ),
                status="TAGGING",
                source_hash=digest,
                extraction_warnings_json=[],
                created_by=actor,
            )
            scopes: list[OperationsKnowledgeScopeEntity] = []
            source = OperationsKnowledgeSourceEntity(
                asset_version_id=version_id,
                source_kind="MANUAL_FILE",
                source_external_id=file_name,
                content_hash=digest,
                source_locator_json={"metadata": metadata.model_dump(mode="json")},
            )
            index = OperationsKnowledgeIndexEntity(
                asset_version_id=version_id,
                collection_id=UUID(str(collection["collection_id"])),
                bundle_id=UUID(str(item["bundle_id"])),
                bundle_revision_id=revision_id,
                index_status="PENDING_APPROVAL",
            )
            await uow.operations_knowledge.add_version(
                version=version,
                scopes=scopes,
                sources=[source],
                indexes=[index],
            )
            asset.display_name = metadata.display_name
            asset.security_level = metadata.security_level
            asset.updated_by = actor
            if asset.current_version_id is None:
                asset.status = "TAGGING"
            await uow.commit()
        return {
            "asset": _asset_view(asset),
            "version": _version_view(version),
            "replayed": False,
        }

    async def extract_case(
        self, *, domain_id: int, report_id: UUID, context: AuthContext
    ) -> dict[str, Any]:
        actor = self._actor(context)
        async with self._uow_factory() as uow:
            report = await uow.inspections.get_report_scoped(
                report_id=report_id, domain_id=domain_id
            )
            if report is None:
                raise self._not_found()
            if (
                report.status in {"FAILED"}
                or not bool(report.is_current)
                or not report.content_hash
                or not (report.summary or "").strip()
                or report.content_artifact_id is None
            ):
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                    message="该报告不满足诊断案例提炼条件",
                    status_code=422,
                )
            existing = await uow.operations_knowledge.find_by_source_hash(
                domain_id=domain_id,
                asset_kind="DIAGNOSIS_CASE",
                source_hash=report.content_hash,
            )
            if existing is not None:
                return {"asset": _asset_view(existing[0]), "version": _version_view(existing[1]), "replayed": True}
            report_sources = await uow.inspections.list_report_sources(report_id=report_id)
            artifact = await uow.runs.get_artifact(
                artifact_id=report.content_artifact_id
            )
            if (
                artifact is None
                or artifact.schema_version != "REPORT_CONTENT.v1"
                or artifact.content_hash != report.content_hash
            ):
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                    message="正式报告内容引用不完整，不能提炼案例",
                    status_code=422,
                )
            report_content = dict(artifact.payload_json or {})
            evidence_refs = [
                str(item.get("artifact_id"))
                for item in report_content.get("evidence_refs") or ()
                if isinstance(item, dict) and item.get("artifact_id")
            ] or [str(row.source_artifact_id) for row in report_sources]
            facts = [
                dict(item) for item in report_content.get("facts") or ()
                if isinstance(item, dict)
            ]
            scope = dict(report_content.get("scope") or {})
            root_cause_summary = str(
                scope.get("diagnosis_rationale")
                or next((
                    item.get("summary") for item in facts
                    if item.get("summary")
                ), "")
                or report.summary
            )
            if not evidence_refs or not root_cause_summary.strip():
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                    message="报告没有可追溯的发现或根因证据",
                    status_code=422,
                )
            target = await uow.targets.get_scoped(
                target_id=report.target_id, domain_id=domain_id
            )
            if target is None:
                raise self._not_found()
            recorded_actions = tuple(
                {"summary": str(item), "evidence_refs": evidence_refs}
                for item in scope.get("actions") or ()
                if str(item).strip()
            )
            case_kind = (
                "VERIFIED_RESOLUTION"
                if report.result in {"RESOLVED", "IMPROVED"} and recorded_actions
                else "DIAGNOSTIC_REFERENCE"
            )
            profile = DiagnosisCaseProfile(
                case_kind=case_kind,
                problem_signature={
                    "problem_class": report.report_type,
                    "component": None,
                    "error_codes": [],
                    "normalized_symptoms": [
                        str(item.get("summary"))
                        for item in facts if item.get("summary")
                    ] or [report.title],
                    "signal_names": [],
                },
                environment={
                    "database_type": target.db_type,
                    "database_version": target.version_code,
                    "platform": target.environment,
                    "topology": (target.capabilities_json or {}).get("topology"),
                },
                root_cause={
                    "summary": root_cause_summary,
                    "confidence": (
                        "HIGH"
                        if str(scope.get("root_cause_grade") or "").upper()
                        in {"CONFIRMED", "HIGH"}
                        else "MEDIUM"
                    ),
                    "evidence_refs": evidence_refs,
                },
                actions=recorded_actions,
                verification={
                    "result": report.result or "NOT_VERIFIED",
                    "before_refs": evidence_refs,
                    "after_refs": evidence_refs if case_kind == "VERIFIED_RESOLUTION" else [],
                    "guardrail_refs": [],
                },
                source={
                    "report_id": str(report.report_id),
                    "run_ids": sorted({str(row.ops_run_id) for row in report_sources}),
                    "content_hash": report.content_hash,
                },
            )
            markdown = (
                f"# {report.title}\n\n"
                f"## 问题特征\n\n"
                + "\n".join(
                    f"- {item.get('summary')}" for item in facts
                    if item.get("summary")
                )
                + f"\n\n## 根因\n\n{root_cause_summary}\n\n"
                + "## 已执行动作\n\n"
                + ("\n".join(
                    f"- {item['summary']}" for item in recorded_actions
                ) or "未记录已执行动作，仅作为诊断参考。")
                + f"\n\n## 验证结果\n\n{report.result or 'NOT_VERIFIED'}\n"
            ).encode("utf-8")
        collection = await self._collection(
            domain_id=domain_id, name=CASE_COLLECTION, context=context
        )
        response = await self._kc.ingest_user_file(
            domain_id=domain_id,
            collection_id=UUID(str(collection["collection_id"])),
            file_name=f"diagnosis-case-{report_id}.md",
            display_name=report.title,
            media_type="text/markdown",
            body=markdown,
            content_sha256=hashlib.sha256(markdown).hexdigest(),
            idempotency_key=f"diagnosis-case:{report.content_hash}",
            auth_context=context,
        )
        item = next(iter((response.payload or {}).get("items", [])), None)
        if not item or item.get("status") == "REJECTED":
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_EXTRACTION_FAILED",
                message="诊断案例索引未被受理",
                status_code=502,
                retryable=True,
            )
        revision_id = UUID(str(item["bundle_revision_id"]))
        await self._kc.review_user_intake(
            domain_id=domain_id,
            collection_id=UUID(str(collection["collection_id"])),
            bundle_revision_id=revision_id,
            decision="APPROVE",
            comment="AIOps 诊断案例候选进入 KC 索引，仍需业务审核后发布",
            auth_context=context,
        )
        asset_id, version_id = uuid7(), uuid7()
        asset = OperationsKnowledgeAssetEntity(
            asset_id=asset_id, domain_id=domain_id, asset_kind="DIAGNOSIS_CASE",
            display_name=report.title, status="PROCESSING",
            security_level=int(report.security_level), created_by=actor, updated_by=actor,
        )
        version = OperationsKnowledgeVersionEntity(
            asset_version_id=version_id, asset_id=asset_id, version_no=1,
            status="PROCESSING", source_hash=report.content_hash,
            profile_schema_version=profile.schema_version,
            profile_json=profile.model_dump(mode="json"),
            extraction_warnings_json=[], created_by=actor,
        )
        sources = [OperationsKnowledgeSourceEntity(
            asset_version_id=version_id, source_kind="REPORT",
            source_report_id=report.report_id, source_run_id=report.ops_run_id,
            source_artifact_id=report.content_artifact_id,
            content_hash=report.content_hash,
            source_locator_json={"report_version": int(report.report_version)},
        ), *[
            OperationsKnowledgeSourceEntity(
                asset_version_id=version_id,
                source_kind="REPORT_EVIDENCE",
                source_report_id=report.report_id,
                source_run_id=row.ops_run_id,
                source_artifact_id=row.source_artifact_id,
                content_hash=row.content_hash,
                source_locator_json={"source_kind": row.source_kind},
            )
            for row in report_sources
        ]]
        scope_values = [
            ("DATABASE_TYPE", target.db_type),
            ("DATABASE_VERSION", target.version_code),
            ("PLATFORM", target.environment),
            ("PROBLEM_CLASS", report.report_type),
            ("TOPOLOGY", (target.capabilities_json or {}).get("topology")),
        ]
        scopes = [OperationsKnowledgeScopeEntity(
            asset_version_id=version_id,
            scope_kind=kind,
            scope_value=str(value),
            normalized_value=_normalize(str(value)),
            source_kind="REPORT_FACT",
            source_locator_json={"report_id": str(report.report_id)},
        ) for kind, value in scope_values if value]
        index = OperationsKnowledgeIndexEntity(
            asset_version_id=version_id,
            collection_id=UUID(str(collection["collection_id"])),
            bundle_id=UUID(str(item["bundle_id"])),
            bundle_revision_id=revision_id,
            index_status="PROCESSING",
        )
        async with self._uow_factory() as uow:
            await uow.operations_knowledge.add_asset_version(
                asset=asset, version=version, scopes=scopes, sources=sources, indexes=[index]
            )
            await uow.commit()
        return {"asset": _asset_view(asset), "version": _version_view(version), "replayed": False}

    async def reconcile_version(
        self, *, domain_id: int, asset_version_id: UUID, context: AuthContext
    ) -> dict[str, Any]:
        detail = await self.get_version(
            domain_id=domain_id, asset_version_id=asset_version_id
        )
        if detail["version"]["status"] == "TAGGING":
            return await self.analyze_tags(
                domain_id=domain_id,
                asset_version_id=asset_version_id,
                context=context,
            )
        index_payload = next(iter(detail["indexes"]), None)
        if index_payload is None:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_INDEX_STALE",
                message="知识版本缺少索引引用",
                status_code=409,
            )
        bundle_id = UUID(index_payload["bundle_id"])
        revision_id = UUID(index_payload["bundle_revision_id"])
        try:
            bundle = await self._kc.get_bundle_status(
                domain_id=domain_id, bundle_id=bundle_id, auth_context=context
            )
            revision = await self._kc.get_revision_status(
                domain_id=domain_id, bundle_id=bundle_id,
                bundle_revision_id=revision_id, include_members=True,
                auth_context=context,
            )
        except KnowledgeCoreClientError as exc:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE",
                message="无法核对知识索引状态",
                status_code=503,
                retryable=True,
            ) from exc
        current_id = str(bundle.get("current_revision_id") or bundle.get("bundle", {}).get("current_revision_id") or "")
        expected_profile_revision = f"{asset_version_id}:profile-v1"
        adopted_revision = False
        if (
            detail["asset"]["asset_kind"] == "MANUAL"
            and current_id
            and current_id != str(revision_id)
        ):
            current_summary = next((
                item for item in bundle.get("revisions") or ()
                if str(item.get("bundle_revision_id") or "") == current_id
            ), None)
            if (
                current_summary is not None
                and str(current_summary.get("source_revision") or "")
                == expected_profile_revision
            ):
                revision_id = UUID(current_id)
                revision = await self._kc.get_revision_status(
                    domain_id=domain_id,
                    bundle_id=bundle_id,
                    bundle_revision_id=revision_id,
                    include_members=True,
                    auth_context=context,
                )
                adopted_revision = True
        availability = str(
            bundle.get("availability")
            or bundle.get("availability_status")
            or bundle.get("bundle", {}).get("availability")
            or ""
        ).upper()
        revision_status = str(
            revision.get("status")
            or revision.get("revision", {}).get("status")
            or ""
        ).upper()
        publication_status = str(revision.get("publication_status") or "").upper()
        members = revision.get("members") or revision.get("items") or []
        required_content_ready = any(
            str(item.get("document_role") or "").upper() == "CONTENT"
            and bool(item.get("document_version_id"))
            and str(item.get("member_status") or "").upper()
            in {"READY", "COMPLETED"}
            for item in members
        )
        ready = current_id == str(revision_id) and required_content_ready and (
            availability in {"READY", "PARTIAL"}
            or revision_status in {"READY", "COMPLETED"}
        )
        failed = (
            availability in {"FAILED", "ERROR", "UNAVAILABLE"}
            or revision_status in {"FAILED", "REJECTED", "ERROR"}
            or publication_status in {"FAILED", "ERROR"}
            or any(
                str(item.get("member_status") or "").upper()
                in {
                    "FAILED", "REJECTED", "ERROR",
                    "SOURCE_UNAVAILABLE", "CANCELLED",
                }
                for item in members
            )
        )
        if (
            ready
            and detail["asset"]["asset_kind"] == "MANUAL"
            and detail["version"]["profile"] is None
        ):
            evidence_items: list[dict[str, Any]] = []
            for member in members:
                document_version_id = member.get("document_version_id")
                if not document_version_id:
                    continue
                page = 1
                while page <= 20:
                    evidence_page = await self._kc.list_file_evidence(
                        domain_id=domain_id,
                        collection_id=UUID(index_payload["collection_id"]),
                        document_version_id=UUID(str(document_version_id)),
                        page=page,
                        page_size=200,
                        auth_context=context,
                    )
                    items = evidence_page.get("items") or []
                    evidence_items.extend(items)
                    if page * 200 >= int(evidence_page.get("total") or len(items)):
                        break
                    page += 1
            revision = {**revision, "extraction_evidence": evidence_items}
        replacement_revision_id: UUID | None = None
        if (
            ready
            and detail["asset"]["asset_kind"] == "MANUAL"
            and detail["version"]["profile"] is not None
            and str(revision.get("source_revision") or "")
            != expected_profile_revision
        ):
            source_member = next(
                (
                    item for item in members
                    if str(item.get("document_role") or "").upper() == "CONTENT"
                    and item.get("document_version_id")
                ),
                None,
            )
            if source_member is None:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_INDEX_STALE",
                    message="手册原文索引成员缺少可下载版本",
                    status_code=409,
                )
            streamed = await self._kc.stream_source_file(
                domain_id=domain_id,
                collection_id=UUID(index_payload["collection_id"]),
                bundle_id=bundle_id,
                bundle_revision_id=revision_id,
                document_version_id=UUID(str(source_member["document_version_id"])),
                range_header=None,
                auth_context=context,
            )
            source_chunks: list[bytes] = []
            source_size = 0
            async for chunk in streamed.body:
                source_size += len(chunk)
                if source_size > 64 * 1024 * 1024:
                    raise AIOpsApplicationError(
                        code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
                        message="手册原文超过提炼版本重建上限",
                        status_code=422,
                    )
                source_chunks.append(bytes(chunk))
            source_body = b"".join(source_chunks)
            if hashlib.sha256(source_body).hexdigest() != detail["version"]["source_hash"]:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_INDEX_STALE",
                    message="手册原文摘要与 Registry 不一致",
                    status_code=409,
                )
            profile_body = (
                "# 运维手册结构化导航\n\n"
                "以下内容只用于候选定位；命令和结论仍以原始手册引用为准。\n\n"
                "```json\n"
                + json.dumps(
                    detail["version"]["profile"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n```\n"
            ).encode("utf-8")
            response = await self._kc.ingest_user_bundle(
                domain_id=domain_id,
                collection_id=UUID(index_payload["collection_id"]),
                client_bundle_id=detail["asset"]["asset_id"],
                source_revision=expected_profile_revision,
                title=detail["asset"]["display_name"],
                security_level=int(detail["asset"]["security_level"]),
                documents=(
                    {
                        "file_name": str(source_member.get("declared_name") or "manual.bin"),
                        "media_type": streamed.headers.get(
                            "content-type", "application/octet-stream"
                        ),
                        "body": source_body,
                        "content_sha256": detail["version"]["source_hash"],
                        "role": "CONTENT",
                    },
                    {
                        "file_name": "operations-manual-profile.md",
                        "media_type": "text/markdown",
                        "body": profile_body,
                        "role": "SUPPLEMENT",
                    },
                ),
                idempotency_key=f"manual-profile:{asset_version_id}",
                auth_context=context,
            )
            replacement = next(iter((response.payload or {}).get("items", [])), None)
            if not replacement or replacement.get("status") == "REJECTED":
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_EXTRACTION_FAILED",
                    message="手册结构化导航未被索引服务受理",
                    status_code=502,
                    retryable=True,
                )
            replacement_revision_id = UUID(str(replacement["bundle_revision_id"]))
            await self._kc.review_user_intake(
                domain_id=domain_id,
                collection_id=UUID(index_payload["collection_id"]),
                bundle_revision_id=replacement_revision_id,
                decision="APPROVE",
                comment="AIOps 手册原文与结构化导航形成同一待发布 Revision",
                auth_context=context,
            )
        async with self._uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id,
                for_update=True,
            )
            if version is None:
                raise self._not_found()
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=version.asset_id, for_update=True
            )
            relations = await uow.operations_knowledge.get_version_relations(
                asset_version_id=asset_version_id
            )
            index = relations["indexes"][0]
            index.last_checked_at = _now()
            index.expected_row_version = int(bundle.get("row_version") or 0) or None
            processing_only = False
            if replacement_revision_id is not None:
                index.bundle_revision_id = replacement_revision_id
                index.index_status = "PROCESSING"
                index.expected_row_version = None
                index.error_code = None
                index.error_summary = None
                version.status = "PROCESSING"
                if asset.current_version_id is None:
                    asset.status = "PROCESSING"
                processing_only = True
            if adopted_revision and not processing_only:
                index.bundle_revision_id = revision_id
            if ready and not processing_only:
                index.index_status = "READY"
                index.error_code = None
                index.error_summary = None
                if version.profile_json is None:
                    version.profile_json = self._manual_profile(
                        asset=asset, version=version,
                        scopes=relations["scopes"], sources=relations["sources"],
                        revision=revision,
                    ).model_dump(mode="json")
                    version.profile_schema_version = "OPS_MANUAL_PROFILE.v1"
                    index.index_status = "PROCESSING"
                    version.status = "PROCESSING"
                    if asset.current_version_id is None:
                        asset.status = "PROCESSING"
                    processing_only = True
                if not processing_only:
                    warnings = list(version.extraction_warnings_json or [])
                    profile_warnings = list((version.profile_json or {}).get("warnings") or [])
                    version.extraction_warnings_json = sorted(set([*warnings, *profile_warnings]))
                    if asset.asset_kind == "MANUAL":
                        if asset.current_version_id and asset.current_version_id != version.asset_version_id:
                            previous = await uow.operations_knowledge.get_version(
                                domain_id=domain_id,
                                asset_version_id=asset.current_version_id,
                                for_update=True,
                            )
                            if previous is not None:
                                previous.status = "RETIRED"
                                previous.retired_at = _now()
                        version.status = "PUBLISHED"
                        version.published_at = _now()
                        asset.status = "PUBLISHED"
                        asset.current_version_id = version.asset_version_id
                    else:
                        version.status = (
                            "REVIEW_REQUIRED"
                            if version.extraction_warnings_json else "DRAFT"
                        )
                        if asset.current_version_id is None:
                            asset.status = version.status
            elif failed and not processing_only:
                failure_code = str(
                    revision.get("publication_failure_code")
                    or next((
                        item.get("failure_code") for item in members
                        if item.get("failure_code")
                    ), None)
                    or "KNOWLEDGE_INDEX_FAILED"
                )
                failure_summary = str(
                    revision.get("publication_failure_message")
                    or next((
                        item.get("failure_message") for item in members
                        if item.get("failure_message")
                    ), None)
                    or "KC 未能完成知识版本解析或索引"
                )
                index.index_status = "FAILED"
                index.error_code = failure_code[:128]
                index.error_summary = failure_summary[:2000]
                version.status = "FAILED"
                if asset.current_version_id is None:
                    asset.status = "FAILED"
            elif not processing_only:
                index.index_status = "PROCESSING"
            await uow.commit()
        return await self.get_version(
            domain_id=domain_id, asset_version_id=asset_version_id
        )

    @staticmethod
    def _manual_profile(
        *, asset, version, scopes, sources, revision: dict[str, Any]
    ) -> ManualProfile:
        metadata = (sources[0].source_locator_json or {}).get("metadata", {}) if sources else {}
        by_kind: dict[str, list[str]] = {}
        for scope in scopes:
            by_kind.setdefault(scope.scope_kind.lower(), []).append(scope.scope_value)
        evidence_ids: list[str] = []
        procedures: list[ManualProcedureCard] = []
        members = revision.get("extraction_evidence") or []
        for member in members:
            evidence_id = member.get("evidence_id")
            if evidence_id:
                evidence_ids.append(str(evidence_id))
            text = str(member.get("content_text") or "")
            commands = _source_commands(text)
            if commands:
                procedures.append(ManualProcedureCard(
                    procedure_key=f"source-fragment-{len(procedures) + 1}",
                    title=str(member.get("section_key") or member.get("heading") or "原文操作步骤"),
                    intent="CHANGE",
                    steps=tuple(ManualProcedureStep(
                        ordinal=index + 1,
                        description="执行原文命令",
                        command_text=command,
                        source_evidence_ids=(str(evidence_id),) if evidence_id else (),
                    ) for index, command in enumerate(commands)),
                    source_locator=member.get("locator") or {},
                ))
        missing = [
            name for name in ("publisher", "document_version", "published_date")
            if not metadata.get(name)
        ]
        return ManualProfile(
            title=asset.display_name,
            source={
                "asset_id": str(asset.asset_id),
                "asset_version_id": str(version.asset_version_id),
                "document_version": metadata.get("document_version"),
                "publisher": metadata.get("publisher"),
            },
            scope={key: tuple(values) for key, values in by_kind.items()},
            topics=tuple(by_kind.get("topic", [])),
            procedures=tuple(procedures),
            warnings=(),
            missing_fields=tuple(missing),
            source_evidence_ids=tuple(evidence_ids),
        )

    async def approve_manual(
        self, *, domain_id: int, asset_version_id: UUID,
        expected_row_version: int, tag_ids: tuple[str, ...],
        comment: str | None, context: AuthContext,
    ) -> dict[str, Any]:
        """确认标签后批准 KC 创建正式解析任务。"""
        try:
            selected_ids = tuple(dict.fromkeys(UUID(value) for value in tag_ids))
        except ValueError as exc:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_TAG_INVALID",
                message="审批标签包含无效标识", status_code=422,
            ) from exc
        if not selected_ids:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_TAG_REQUIRED",
                message="至少选择一个受控标签后才能批准解析", status_code=422,
            )
        detail = await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
        if detail["asset"]["asset_kind"] != "MANUAL":
            raise self._state_conflict("只有技术手册使用标签审批流程")
        if detail["version"]["status"] != "REVIEW_REQUIRED":
            raise self._state_conflict("当前手册不处于待审批状态")
        if int(detail["version"]["row_version"]) != expected_row_version:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_VERSION_CONFLICT",
                message="知识版本已被其他请求更新，请刷新后重试", status_code=412,
            )
        index_payload = next(iter(detail["indexes"]), None)
        if index_payload is None or index_payload["status"] != "PENDING_APPROVAL":
            raise self._state_conflict("当前手册的 KC 入库状态不能批准")
        async with self._uow_factory() as uow:
            active_tags = await uow.operations_knowledge.list_tags(domain_id=domain_id, status="ACTIVE")
        tags_by_id = {tag.tag_id: tag for tag in active_tags}
        if any(tag_id not in tags_by_id for tag_id in selected_ids):
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_TAG_STALE",
                message="所选标签已停用或不属于当前 Domain，请刷新后重选", status_code=409,
            )
        try:
            await self._kc.review_user_intake(
                domain_id=domain_id,
                collection_id=UUID(index_payload["collection_id"]),
                bundle_revision_id=UUID(index_payload["bundle_revision_id"]),
                decision="APPROVE",
                comment="AIOps 标签已人工确认，批准 KC 执行正式解析与索引",
                auth_context=context,
            )
        except KnowledgeCoreClientError as exc:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE",
                message="KC 未能受理解析审批", status_code=503, retryable=True,
            ) from exc
        actor = self._actor(context)
        async with self._uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id, for_update=True
            )
            if version is None:
                raise self._not_found()
            if version.status != "REVIEW_REQUIRED" or int(version.row_version) != expected_row_version:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_VERSION_CONFLICT",
                    message="知识版本已被其他请求更新，请刷新后重试", status_code=412,
                )
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=version.asset_id, for_update=True
            )
            relations = await uow.operations_knowledge.get_version_relations(asset_version_id=asset_version_id)
            await uow.operations_knowledge.replace_tag_scopes(
                version=version, tags=[tags_by_id[tag_id] for tag_id in selected_ids]
            )
            analysis = dict(version.tag_analysis_json or {})
            analysis["approved_tag_ids"] = [str(tag_id) for tag_id in selected_ids]
            analysis["approved_at"] = _now().isoformat()
            analysis["approved_by"] = actor
            version.tag_analysis_json = analysis
            version.status = "PROCESSING"
            for index in relations["indexes"]:
                index.index_status = "PROCESSING"
                index.error_code = None
                index.error_summary = None
            if asset.current_version_id is None:
                asset.status = "PROCESSING"
            asset.updated_by = actor
            await uow.operations_knowledge.add_review(OperationsKnowledgeReviewEntity(
                asset_version_id=asset_version_id, decision="APPROVE",
                reviewer_id=actor, comment_text=comment,
                before_status="REVIEW_REQUIRED", after_status="PROCESSING",
            ))
            await uow.commit()
        return await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)

    async def review(
        self,
        *,
        domain_id: int,
        asset_version_id: UUID,
        decision: str,
        expected_row_version: int,
        comment: str | None,
        context: AuthContext,
    ) -> dict[str, Any]:
        actor = self._actor(context)
        now = _now()
        if decision == "PUBLISH":
            detail = await self.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id
            )
            if detail["asset"]["asset_kind"] == "MANUAL":
                raise self._state_conflict(
                    "技术手册应先确认标签并批准解析，解析完成后会自动发布"
                )
            index_payload = next(iter(detail["indexes"]), None)
            if index_payload is None or index_payload["status"] != "READY":
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_INDEX_PENDING",
                    message="知识正文索引尚未就绪，不能发布",
                    status_code=409,
                )
            try:
                bundle = await self._kc.get_bundle_status(
                    domain_id=domain_id,
                    bundle_id=UUID(index_payload["bundle_id"]),
                    auth_context=context,
                )
            except KnowledgeCoreClientError as exc:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE",
                    message="发布前无法复核知识索引状态",
                    status_code=503,
                    retryable=True,
                ) from exc
            if (
                str(bundle.get("current_revision_id") or "")
                != index_payload["bundle_revision_id"]
                or str(bundle.get("availability_status") or "").upper()
                not in {"READY", "PARTIAL"}
                or (
                    index_payload.get("expected_row_version") is not None
                    and int(bundle.get("row_version") or 0)
                    != int(index_payload["expected_row_version"])
                )
            ):
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_INDEX_STALE",
                    message="知识索引已变化，请重新对账后发布",
                    status_code=409,
                )
        elif decision == "REJECT":
            detail = await self.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id
            )
            index_payload = next(iter(detail["indexes"]), None)
            if (
                detail["asset"]["asset_kind"] == "MANUAL"
                and index_payload is not None
                and index_payload["status"] == "PENDING_APPROVAL"
            ):
                try:
                    await self._kc.review_user_intake(
                        domain_id=domain_id,
                        collection_id=UUID(index_payload["collection_id"]),
                        bundle_revision_id=UUID(index_payload["bundle_revision_id"]),
                        decision="REJECT",
                        comment=comment or "AIOps 运维知识审批拒绝",
                        auth_context=context,
                    )
                except KnowledgeCoreClientError as exc:
                    raise AIOpsApplicationError(
                        code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE",
                        message="KC 未能受理拒绝决定", status_code=503, retryable=True,
                    ) from exc
        retry_index: dict[str, Any] | None = None
        async with self._uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id,
                for_update=True,
            )
            if version is None:
                raise self._not_found()
            if int(version.row_version) != expected_row_version:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_VERSION_CONFLICT",
                    message="知识版本已被其他请求更新，请刷新后重试",
                    status_code=412,
                )
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=version.asset_id, for_update=True
            )
            before = version.status
            if decision == "PUBLISH":
                if before not in PUBLISHABLE_STATUSES:
                    raise self._state_conflict("当前知识版本不能发布")
                relations = await uow.operations_knowledge.get_version_relations(
                    asset_version_id=asset_version_id
                )
                if asset.asset_kind == "DIAGNOSIS_CASE":
                    report_ids = {
                        item.source_report_id for item in relations["sources"]
                        if item.source_report_id is not None
                    }
                    for report_id in report_ids:
                        source_report = await uow.inspections.get_report_scoped(
                            report_id=report_id, domain_id=domain_id
                        )
                        if source_report is None or not bool(source_report.is_current):
                            raise self._state_conflict(
                                "来源报告已经更正，必须从当前报告重新提炼案例"
                            )
                if not relations["indexes"] or any(
                    item.index_status != "READY" for item in relations["indexes"]
                ):
                    raise AIOpsApplicationError(
                        code="AIOPS_KNOWLEDGE_INDEX_PENDING",
                        message="知识正文索引尚未就绪，不能发布",
                        status_code=409,
                    )
                if asset.current_version_id and asset.current_version_id != version.asset_version_id:
                    previous = await uow.operations_knowledge.get_version(
                        domain_id=domain_id,
                        asset_version_id=asset.current_version_id,
                        for_update=True,
                    )
                    if previous is not None:
                        previous.status = "RETIRED"
                        previous.retired_at = now
                version.status = "PUBLISHED"
                version.published_at = now
                asset.status = "PUBLISHED"
                asset.current_version_id = version.asset_version_id
            elif decision == "REJECT":
                if before not in {"DRAFT", "REVIEW_REQUIRED", "TAGGING_FAILED"}:
                    raise self._state_conflict("当前知识版本不能拒绝")
                version.status = "REJECTED"
                asset.status = (
                    "PUBLISHED" if asset.current_version_id else "REJECTED"
                )
            elif decision == "RETIRE":
                if before != "PUBLISHED":
                    raise self._state_conflict("只有已发布知识可以退役")
                version.status = "RETIRED"
                version.retired_at = now
                asset.status = "RETIRED"
                if asset.current_version_id == version.asset_version_id:
                    asset.current_version_id = None
            elif decision == "RETRY":
                if before not in {"FAILED", "PROCESSING", "TAGGING_FAILED"}:
                    raise self._state_conflict("当前知识版本不需要重试")
                relations = await uow.operations_knowledge.get_version_relations(
                    asset_version_id=asset_version_id
                )
                if before == "TAGGING_FAILED":
                    version.status = "TAGGING"
                    version.tag_analysis_json = None
                    if asset.current_version_id is None:
                        asset.status = "TAGGING"
                    for index in relations["indexes"]:
                        index.index_status = "PENDING_APPROVAL"
                        index.error_code = None
                        index.error_summary = None
                else:
                    version.status = "PROCESSING"
                    if asset.current_version_id is None:
                        asset.status = "PROCESSING"
                    for index in relations["indexes"]:
                        index.index_status = "PROCESSING"
                        index.error_code = None
                        index.error_summary = None
                    if relations["indexes"]:
                        retry_index = _index_view(relations["indexes"][0])
            else:
                raise self._state_conflict("未知审核决定")
            asset.updated_by = actor
            await uow.operations_knowledge.add_review(
                OperationsKnowledgeReviewEntity(
                    asset_version_id=asset_version_id,
                    decision=decision, reviewer_id=actor, comment_text=comment,
                    before_status=before, after_status=version.status,
                )
            )
            await uow.commit()
        if decision == "RETRY":
            if before == "TAGGING_FAILED":
                return await self.analyze_tags(
                    domain_id=domain_id,
                    asset_version_id=asset_version_id,
                    context=context,
                )
            if retry_index is None:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_INDEX_STALE",
                    message="知识版本缺少可重试的索引引用",
                    status_code=409,
                )
            try:
                await self._kc.reprocess_revision(
                    domain_id=domain_id,
                    collection_id=UUID(retry_index["collection_id"]),
                    bundle_id=UUID(retry_index["bundle_id"]),
                    bundle_revision_id=UUID(retry_index["bundle_revision_id"]),
                    document_version_id=None,
                    auth_context=context,
                )
            except KnowledgeCoreClientError as exc:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE",
                    message="知识版本重处理请求未被索引服务受理",
                    status_code=503,
                    retryable=True,
                ) from exc
            return await self.reconcile_version(
                domain_id=domain_id, asset_version_id=asset_version_id,
                context=context,
            )
        return await self.get_version(
            domain_id=domain_id, asset_version_id=asset_version_id
        )

    async def search(
        self, *, domain_id: int, agent_id: str, request: KnowledgeSearchRequest,
        context: AuthContext, target_id: UUID | None = None,
    ) -> dict[str, Any]:
        filters = {
            "DATABASE_TYPE": (request.database_type,) if request.database_type else (),
            "DATABASE_VERSION": (request.database_major_version,) if request.database_major_version else (),
            "TOPOLOGY": (request.topology,) if request.topology else (),
            "PROBLEM_CLASS": (request.problem_class,) if request.problem_class else (),
            "COMPONENT": request.components,
            "ERROR_CODE": request.error_codes,
            "SIGNAL_NAME": request.signal_names,
            "TAG_ID": request.tag_ids,
        }
        filters = {key: tuple(_normalize(v) for v in values) for key, values in filters.items() if values}
        async with self._uow_factory() as uow:
            candidates = await uow.operations_knowledge.list_published_candidates(
                domain_id=domain_id,
                target_id=target_id,
                source_kinds=request.source_kinds,
                max_security_level=request.max_security_level,
                scope_filters=filters,
                limit=request.max_results,
            )
        if not candidates:
            return {"status": "NO_APPLICABLE_KNOWLEDGE", "results": [], "warnings": []}
        collection_ids = sorted({row[2].collection_id for row in candidates}, key=str)
        revision_ids = [row[2].bundle_revision_id for row in candidates]
        try:
            discovery = await self._kc.discover(
                query=request.query,
                collection_ids=collection_ids,
                bundle_revision_ids=revision_ids,
                domain_id=domain_id,
                agent_id=agent_id,
                auth_context=context,
                max_security_level=request.max_security_level,
                per_collection_limit=request.max_results,
            )
            discovered = discovery.get("candidates") or discovery.get("items") or []
            evidence = await self._kc.retrieve_evidence(
                query=request.query,
                candidates=discovered,
                domain_id=domain_id,
                agent_id=agent_id,
                auth_context=context,
                max_security_level=request.max_security_level,
                max_evidence=request.max_results * 2,
            )
        except KnowledgeCoreClientError as exc:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE",
                message="运维知识正文检索暂时不可用",
                status_code=503,
                retryable=True,
            ) from exc
        by_revision = {str(row[2].bundle_revision_id): row for row in candidates}
        evidence_items = evidence.get("citations") or []
        results = []
        for item in evidence_items:
            revision_id = str(item.get("bundle_revision_id") or "")
            row = by_revision.get(revision_id)
            if row is None:
                continue
            asset, version, _ = row
            known_dimensions = _profile_dimensions(version)
            matched_dimensions = sorted(
                key for key, values in filters.items()
                if known_dimensions.get(key)
                and not known_dimensions[key].isdisjoint(values)
            )
            unknown_dimensions = sorted(
                key for key in filters if not known_dimensions.get(key)
            )
            results.append({
                "asset_id": str(asset.asset_id),
                "asset_version_id": str(version.asset_version_id),
                "asset_kind": asset.asset_kind,
                "title": asset.display_name,
                "version": int(version.version_no),
                "authority": (
                    "APPROVED_MANUAL" if asset.asset_kind == "MANUAL"
                    else str((version.profile_json or {}).get("case_kind") or "DIAGNOSTIC_REFERENCE")
                ),
                "applicability": (
                    "PARTIAL" if unknown_dimensions else "MATCHED"
                ),
                "matched_dimensions": matched_dimensions,
                "mismatched_dimensions": [],
                "unknown_dimensions": unknown_dimensions,
                "citation_pack": [item],
            })
            if len(results) >= request.max_results:
                break
        return {
            "status": "READY" if results else "NO_EVIDENCE_MATCH",
            "results": results,
            "warnings": [] if results else ["有适用候选，但 KC 正文没有形成可引用命中"],
        }

    async def knowledge_execution_spec(
        self,
        *,
        domain_id: int,
        max_security_level: int,
        business_context: dict[str, Any],
        context: AuthContext,
    ) -> dict[str, Any]:
        """签发 AIOps 内部知识助手的 Execution Spec 与精确发布范围。"""
        if "composer_llm" not in self._knowledge_assistant_models:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_ASSISTANT_MODEL_REQUIRED",
                message="运维知识助手尚未配置 composer_llm",
                status_code=503,
                retryable=False,
            )
        mappings = {
            "database_type": "DATABASE_TYPE",
            "database_major_version": "DATABASE_VERSION",
            "topology": "TOPOLOGY",
            "problem_class": "PROBLEM_CLASS",
            "component": "COMPONENT",
        }
        filters = {
            scope_kind: (_normalize(str(business_context[key])),)
            for key, scope_kind in mappings.items()
            if business_context.get(key) is not None
            and str(business_context[key]).strip()
        }
        tag_ids = business_context.get("tag_ids") or ()
        if isinstance(tag_ids, (list, tuple)):
            normalized_tag_ids = tuple(
                str(value).strip().upper() for value in tag_ids if str(value).strip()
            )
            if normalized_tag_ids:
                filters["TAG_ID"] = normalized_tag_ids
        async with self._uow_factory() as uow:
            candidates = await uow.operations_knowledge.list_published_candidates(
                domain_id=domain_id,
                target_id=None,
                source_kinds=("MANUAL", "DIAGNOSIS_CASE"),
                max_security_level=max_security_level,
                scope_filters=filters,
                limit=None,
            )
        if not candidates:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_SCOPE_EMPTY",
                message="当前 Domain 没有可用于问答的已发布运维知识",
                status_code=422,
                retryable=False,
            )
        collection_ids = tuple(sorted(
            {row[2].collection_id for row in candidates},
            key=lambda item: item.bytes,
        ))
        revision_ids = tuple(sorted(
            {row[2].bundle_revision_id for row in candidates},
            key=lambda item: item.bytes,
        ))
        for collection_id in collection_ids:
            await self._kc.bind_collection(
                domain_id=domain_id,
                agent_id=self._knowledge_assistant_id,
                collection_id=collection_id,
                note="AIOps 内部运维知识助手检索授权",
                auth_context=context,
            )
        scope_payload = {
            "schema_version": "1.0",
            "owner_app_id": "aiops",
            "domain_id": domain_id,
            "consumer_agent_id": str(self._knowledge_assistant_id),
            "consumer_agent_version_id": str(
                self._knowledge_assistant_version_id
            ),
            "collection_ids": [str(item) for item in collection_ids],
            "bundle_revision_ids": [str(item) for item in revision_ids],
            "security_level": max_security_level,
            "business_context": dict(business_context),
            "business_resource_refs": [
                {
                    "kind": "OPERATIONS_KNOWLEDGE_VERSION",
                    "id": str(row[1].asset_version_id),
                }
                for row in candidates
            ],
            "policy_version": "aiops-knowledge-scope.v1",
            "issued_at": _now().isoformat(),
        }
        scope_payload["scope_hash"] = KnowledgeScopeSnapshotV1.calculate_hash(
            scope_payload
        )
        scope = KnowledgeScopeSnapshotV1.model_validate(scope_payload)
        spec = AgentExecutionSpec(
            schema_version="1.0",
            owner_app_id="aiops",
            domain_id=domain_id,
            consumer_agent_id=self._knowledge_assistant_id,
            consumer_agent_version_id=self._knowledge_assistant_version_id,
            agent_kind="KNOWLEDGE_RETRIEVAL",
            display_name="AIOps 运维知识助手",
            enabled_capabilities=("document",),
            models=self._knowledge_assistant_models,
            instruction=self._knowledge_assistant_instruction,
            resource_context={
                "resource_mode": "managed_resources",
                "default_route": "DOCUMENT",
                "collection_ids": [str(item) for item in collection_ids],
            },
            runtime_policy={
                "routing": "document_only",
                "allow_general_conversation": False,
            },
            knowledge_scope=scope,
        )
        return spec.model_dump(mode="json")

    async def list_reviews(self, *, domain_id: int, limit: int) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await uow.operations_knowledge.list_reviews(
                domain_id=domain_id, limit=limit
            )
            return [{
                "review_id": str(row.review_id),
                "asset_version_id": str(row.asset_version_id),
                "decision": row.decision,
                "reviewer_id": row.reviewer_id,
                "comment": row.comment_text,
                "before_status": row.before_status,
                "after_status": row.after_status,
                "created_at": row.created_at,
            } for row in rows]

    async def reconcile_next(self, *, caller_service: str) -> bool:
        """由 Worker 小步对账一个 KC 索引，未就绪时按正常空闲处理。"""
        async with self._uow_factory() as uow:
            pending = await uow.operations_knowledge.next_processing_version()
        if pending is None:
            return False
        domain_id, version_id = pending
        detail = await self.reconcile_version(
            domain_id=domain_id,
            asset_version_id=version_id,
            context=create_service_auth_context(caller_service=caller_service),
        )
        return detail["version"]["status"] != "PROCESSING"

    @staticmethod
    def _not_found() -> AIOpsApplicationError:
        return AIOpsApplicationError(
            code="AIOPS_KNOWLEDGE_ASSET_NOT_FOUND",
            message="运维知识资产或版本不存在",
            status_code=404,
        )

    @staticmethod
    def _tag_not_found() -> AIOpsApplicationError:
        return AIOpsApplicationError(
            code="AIOPS_KNOWLEDGE_TAG_NOT_FOUND",
            message="知识标签不存在或不属于当前 Domain",
            status_code=404,
        )

    @staticmethod
    def _state_conflict(message: str) -> AIOpsApplicationError:
        return AIOpsApplicationError(
            code="AIOPS_KNOWLEDGE_REVIEW_REQUIRED",
            message=message,
            status_code=409,
        )
