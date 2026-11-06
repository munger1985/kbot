"""AIOps 运维知识资产、提炼、审核和两阶段检索服务。"""

from __future__ import annotations

from loguru import logger

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from uuid import UUID

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.case_redaction import CaseRedactor
from aiops_agent.application.case_semantic_redaction import redact_case_for_index, shared_content_hash
from aiops_agent.application.case_source_validation import validate_case_sources, validate_case_effect
from aiops_agent.application.case_report import load_case_report, rejected_source
from aiops_agent.application.knowledge_withdrawal import (
    CaseWithdrawalCommand, assert_case_not_withdrawn, invalid_withdrawal,
    queue_case_withdrawals, validate_command_source, validate_receipt, withdrawal_views,
)
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.contracts.knowledge import (
    DiagnosisClosureCommand,
    DiagnosisCaseDraft,
    DiagnosisCaseProfile,
    KnowledgeBaseWrite,
    KnowledgeSearchRequest,
    KnowledgeTagAnalysisOutput,
    KnowledgeTagScopeSnapshot,
    KnowledgeTagWrite,
    ManualProcedureCard,
    ManualProcedureStep,
    ManualProfile,
    ManualUploadMetadata,
)
from aiops_agent.entities import (
    OpsArtifactEntity,
    OperationsKnowledgeAssetEntity,
    OperationsKnowledgeBaseEntity,
    OperationsKnowledgeIndexEntity,
    OperationsKnowledgeReviewEntity,
    OperationsKnowledgeScopeEntity,
    OperationsKnowledgeSourceEntity,
    OperationsKnowledgeTagEntity,
    OperationsKnowledgeVersionEntity,
)
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops.case_collection import CASE_MODEL_ROLES, CaseCollectionModels
from platform_core.dictionary import ModelCategory
from platform_core.identity import uuid7
from platform_core.security import create_service_auth_context
from platform_clients.model import AIModelConfigClient


CASE_COLLECTION = "diagnosis-cases"
PUBLISHABLE_STATUSES = frozenset({"REVIEW_REQUIRED"})
TAG_CATEGORIES = frozenset({"RESOURCE_CATEGORY", "TECHNOLOGY_STACK", "VENDOR"})
DELETABLE_MANUAL_STATUSES = frozenset({
    "DRAFT", "REVIEW_REQUIRED", "TAGGING_FAILED", "FAILED", "REJECTED",
})
STOPPABLE_MANUAL_STATUSES = frozenset({"TAGGING", "PROCESSING"})
REQUESTABLE_MANUAL_DELETE_STATUSES = (
    DELETABLE_MANUAL_STATUSES | STOPPABLE_MANUAL_STATUSES
)


def _now() -> datetime:
    return datetime.now(UTC)


def _normalize(value: str) -> str:
    return " ".join(value.strip().upper().split())


def _freeze_case_version_identity(case: tuple[Any, Any]) -> tuple[UUID, int]:
    """Copy ORM identity values while their unit of work is still active."""
    return UUID(str(case[0].asset_id)), int(case[1].version_no)




def _validate_closure_evidence(
    *, command: DiagnosisClosureCommand, allowed_ids: set[UUID],
    allowed_action_ids: set[UUID],
) -> tuple[list[str], set[str], set[str]]:
    """把收口证据限定在当前诊断，并保证实施前后样本互斥。"""
    selected_ids = {
        *command.action_evidence_ids,
        *command.before_evidence_ids,
        *command.after_evidence_ids,
    }
    if not selected_ids.issubset(allowed_ids):
        raise AIOpsApplicationError(
            code="AIOPS_DIAGNOSIS_CLOSURE_EVIDENCE_INVALID",
            message="诊断收口引用了当前会话之外的证据",
            status_code=422,
        )
    if not set(command.action_evidence_ids).issubset(allowed_action_ids):
        raise AIOpsApplicationError(
            code="AIOPS_DIAGNOSIS_CLOSURE_EVIDENCE_INVALID",
            message="动作证据必须来自实际执行结果，计划或诊断结论不能证明动作已实施",
            status_code=422,
        )
    before_refs = {str(item) for item in command.before_evidence_ids}
    after_refs = {str(item) for item in command.after_evidence_ids}
    if before_refs & after_refs:
        raise AIOpsApplicationError(
            code="AIOPS_DIAGNOSIS_CLOSURE_EVIDENCE_INVALID",
            message="实施前证据与实施后证据必须相互独立",
            status_code=422,
        )
    return (
        [str(item) for item in command.action_evidence_ids],
        before_refs,
        after_refs,
    )


def _diagnosis_case_eligibility(
    *, report, report_content: dict[str, Any], evidence_refs: list[str]
) -> dict[str, Any]:
    """使用冻结报告的结构化语义判断能否沉淀历史案例。"""
    scope = dict(report_content.get("scope") or {})
    reasons: list[dict[str, str]] = []
    # 当前性与纯展示修订链已由 load_case_report 重验，不能在此只按版本标记误拒。
    if str(report.status or "").upper() not in {"READY", "PARTIAL"}:
        reasons.append({"code": "REPORT_NOT_READY", "message": "来源报告尚未形成完整诊断结论"})
    if str(scope.get("source_kind") or "").upper() not in {"CHAT", "ALERT", "INSPECTION"}:
        reasons.append({"code": "SOURCE_NOT_DIAGNOSIS", "message": "来源不是可追溯调查"})
    if not str(scope.get("diagnosis_rationale") or "").strip():
        reasons.append({"code": "ROOT_CAUSE_SUMMARY_MISSING", "message": "缺少可发布的根因说明"})
    if not evidence_refs:
        reasons.append({"code": "EVIDENCE_MISSING", "message": "缺少可追溯诊断证据"})
    if not any(
        isinstance(item, dict) and item.get("record_id") and item.get("claim_id")
        and item.get("evidence_refs") and item.get("level") in {"OBSERVED", "SUPPORTED"}
        for item in report_content.get("facts", ())
    ):
        reasons.append({"code": "INVESTIGATION_RECORD_MISSING", "message": "缺少带来源记录的有效判断；自由正文不能自动升级为案例事实"})
    return {"eligible": not reasons, "reasons": reasons}


def _prepare_case_revision(
    *, command: DiagnosisCaseDraft, base_profile: DiagnosisCaseProfile
) -> DiagnosisCaseProfile:
    """继承不可编辑的证据链，并按证据重新判定案例等级。"""
    submitted = command.profile
    signature = dict(submitted.problem_signature)
    symptoms = [
        str(item).strip()
        for item in signature.get("normalized_symptoms") or ()
        if str(item).strip()
    ]
    if not symptoms:
        raise AIOpsApplicationError(
            code="AIOPS_DIAGNOSIS_CASE_INVALID",
            message="案例至少需要一条问题现象",
            status_code=422,
        )
    signature["normalized_symptoms"] = symptoms
    root_cause = dict(submitted.root_cause)
    root_summary = str(root_cause.get("summary") or "").strip()
    if not root_summary:
        raise AIOpsApplicationError(
            code="AIOPS_DIAGNOSIS_CASE_INVALID",
            message="案例必须保留可追溯的调查结论",
            status_code=422,
        )
    base_root = dict(base_profile.root_cause)
    root_cause.update({
        "summary": root_summary,
        "confidence": base_root.get("confidence", "UNKNOWN"),
        "effective_level": base_root.get("effective_level", "INCONCLUSIVE"),
        "evidence_refs": list(base_root.get("evidence_refs") or ()),
    })
    if root_summary != str(base_root.get("summary") or "").strip():
        root_cause.update({"confidence": "UNKNOWN", "effective_level": "INCONCLUSIVE", "trust_level": "USER_PROVIDED"})
    base_actions = list(base_profile.actions)
    actions: list[dict[str, Any]] = []
    for index, value in enumerate(submitted.actions):
        summary = str(value.get("summary") or "").strip()
        if not summary:
            continue
        inherited = {"summary": summary, "evidence_refs": [], "source_kind": "USER_PROVIDED"}
        if index < len(base_actions):
            base_summary = str(base_actions[index].get("summary") or "").strip()
            if summary == base_summary:
                inherited = dict(base_actions[index])
        actions.append(inherited)
    verification = dict(base_profile.verification)
    before_refs = {str(item) for item in verification.get("before_refs") or ()}
    after_refs = {str(item) for item in verification.get("after_refs") or ()}
    # 引用不同与人工勾选恢复不能证明可比效果；只继承已冻结的受控验证。
    verified = bool(
        base_profile.case_kind in {"VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION"}
        and verification.get("controlled_verification_ref")
        and root_summary == str(base_root.get("summary") or "").strip()
        and [item["summary"] for item in actions] == [str(item.get("summary") or "").strip() for item in base_actions]
        and actions and all(item["evidence_refs"] for item in actions)
        and before_refs and after_refs and before_refs.isdisjoint(after_refs)
    )
    return DiagnosisCaseProfile(
        case_kind=base_profile.case_kind if verified or (
            base_profile.case_kind == "NEGATIVE_EXPERIENCE"
            and actions and all(item["evidence_refs"] for item in actions)
            and [item["summary"] for item in actions] == [str(item.get("summary") or "").strip() for item in base_actions]
        ) else "DIAGNOSTIC_REFERENCE",
        problem_signature=signature,
        environment=dict(submitted.environment),
        root_cause=root_cause,
        investigation_steps=base_profile.investigation_steps,
        actions=tuple(actions),
        action_effects=tuple(
            effect for effect in base_profile.action_effects
            if effect.action_ordinal <= len(actions)
            and actions[effect.action_ordinal - 1] == base_actions[effect.action_ordinal - 1]
        ),
        unresolved_questions=base_profile.unresolved_questions,
        verification=verification,
        applicability=tuple(
            str(item).strip() for item in submitted.applicability if str(item).strip()
        ),
        contraindications=tuple(
            str(item).strip() for item in submitted.contraindications if str(item).strip()
        ),
        source=dict(base_profile.source),
    )


def _diagnosis_case_markdown(*, title: str, profile: DiagnosisCaseProfile) -> bytes:
    """案例的全部生成与修订入口使用同一确定性知识投影。"""
    from aiops_agent.application.case_knowledge_projection import case_knowledge_markdown
    return case_knowledge_markdown(title=title, profile=profile)


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
        "knowledge_base_id": (
            str(entity.knowledge_base_id) if entity.knowledge_base_id else None
        ),
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


def _knowledge_base_view(entity: OperationsKnowledgeBaseEntity) -> dict[str, Any]:
    return {
        "knowledge_base_id": str(entity.knowledge_base_id),
        "domain_id": int(entity.domain_id),
        "display_name": entity.display_name,
        "description": entity.description,
        "configuration": dict(entity.configuration_json or {}),
        "status": entity.status,
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


def _case_item_view(detail: dict[str, Any]) -> dict[str, Any]:
    asset = detail["asset"]
    version = detail["version"]
    profile = dict(version.get("profile") or {})
    signature = dict(profile.get("problem_signature") or {})
    environment = dict(profile.get("environment") or {})
    root_cause = dict(profile.get("root_cause") or {})
    source = dict(profile.get("source") or {})
    report_source = next(
        (item for item in detail.get("sources", ()) if item.get("source_kind") == "REPORT"),
        {},
    )
    return {
        "asset_id": asset["asset_id"],
        "asset_version_id": version["asset_version_id"],
        "display_name": asset["display_name"],
        "status": version["status"],
        "case_kind": profile.get("case_kind"),
        "problem_class": signature.get("problem_class"),
        "component": signature.get("component"),
        "database_type": environment.get("database_type"),
        "database_version": environment.get("database_version"),
        "topology": environment.get("topology"),
        "root_cause_summary": root_cause.get("summary"),
        "source_report_id": report_source.get("report_id") or source.get("report_id"),
        "source_run_ids": source.get("run_ids") or (),
        "version_no": version["version_no"],
        "security_level": asset["security_level"],
        "published_at": version.get("published_at"),
        "updated_at": asset["updated_at"],
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
        **{key: str(getattr(entity, key)) if getattr(entity, key) else None
           for key in ("document_id", "document_version_id", "ingestion_job_id", "parse_run_id", "index_generation_id")},
        "status": entity.index_status,
        "expected_row_version": (
            int(entity.expected_row_version)
            if entity.expected_row_version is not None else None
        ),
        "last_checked_at": entity.last_checked_at,
        "error_code": entity.error_code,
        "error_summary": entity.error_summary,
    }


def _profile_dimensions_from_profile(
    profile: dict[str, Any],
) -> dict[str, set[str]]:
    """把两类提炼合同投影为统一、可解释的匹配维度。"""
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


def _catalog_topics_from_profile(profile: dict[str, Any]) -> tuple[str, ...]:
    """从已发布 Registry 提炼结果中读取可展示的提问主题。"""
    topics: list[str] = []
    if profile.get("schema_version") == "OPS_MANUAL_PROFILE.v1":
        topics.extend(str(item) for item in profile.get("topics") or ())
        topics.extend(
            str(item.get("title"))
            for item in profile.get("procedures") or ()
            if isinstance(item, dict) and item.get("title")
        )
    else:
        signature = dict(profile.get("problem_signature") or {})
        for key in ("problem_class", "component"):
            if signature.get(key):
                topics.append(str(signature[key]))
        for key in ("error_codes", "normalized_symptoms", "signal_names"):
            topics.extend(str(item) for item in signature.get(key) or ())
    return tuple(dict.fromkeys(item.strip() for item in topics if item.strip()))[:8]


def _profile_dimensions(
    version: OperationsKnowledgeVersionEntity,
) -> dict[str, set[str]]:
    return _profile_dimensions_from_profile(dict(version.profile_json or {}))


class OperationsKnowledgeService:
    """Registry 是业务真相；解析和召回只访问 AIOps 自有存储。"""

    def __init__(
        self,
        *,
        uow_factory: Callable,
        documents,
        model_catalogs: dict[int, AIModelConfigClient] | None = None,
        structured_model=None,
        prompt_registry=None,
        plan_capture_reader=None,
        knowledge_search_budget=None,
    ) -> None:
        self._uow_factory = uow_factory
        self._documents = documents
        documents.retrieval.authorization.service = self
        self._model_catalogs = dict(model_catalogs or {})
        self._llm_model_catalog = self._model_catalogs.get(int(ModelCategory.LLM))
        self._structured_model = structured_model
        self._prompt_registry = prompt_registry
        self._case_model_catalog = self._llm_model_catalog
        self._case_model_name_field = "served_model_name"
        self._plan_capture_reader = plan_capture_reader
        self._knowledge_search_budget = knowledge_search_budget

    @staticmethod
    def _actor(context: AuthContext) -> str:
        return context.asserted_user_id or context.client_id

    async def create_knowledge_base(self, *, domain_id, command, context):
        await self._validate_knowledge_base_models(context=context, models=command.models)
        configuration = self._documents.configuration(command.models)
        async with self._uow_factory() as uow:
            duplicate = await uow.operations_knowledge.find_knowledge_base_name(
                domain_id=domain_id, normalized_name=" ".join(command.display_name.casefold().split()))
            if duplicate is not None:
                raise self._state_conflict("当前 Domain 已存在同名运维知识库")
            entity = OperationsKnowledgeBaseEntity(domain_id=domain_id,
                knowledge_base_id=uuid7(), display_name=command.display_name.strip(),
                description=command.description, configuration_json=configuration, status="ACTIVE",
                created_by=self._actor(context), updated_by=self._actor(context))
            await uow.operations_knowledge.add_knowledge_base(entity)
            await uow.commit()
            return _knowledge_base_view(entity)

    async def list_knowledge_bases(
        self, *, domain_id: int, include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await uow.operations_knowledge.list_knowledge_bases(
                domain_id=domain_id, include_archived=include_archived,
            )
            result = []
            for row in rows:
                view = _knowledge_base_view(row)
                view["usage"] = await uow.operations_knowledge.knowledge_base_usage(
                    knowledge_base_id=row.knowledge_base_id
                )
                result.append(view)
            return result

    async def get_knowledge_base(
        self, *, domain_id: int, knowledge_base_id: UUID,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            row = await uow.operations_knowledge.get_knowledge_base(
                domain_id=domain_id, knowledge_base_id=knowledge_base_id,
            )
            if row is None or row.status == "ARCHIVED":
                raise self._knowledge_base_not_found()
            view = _knowledge_base_view(row)
            view["usage"] = await uow.operations_knowledge.knowledge_base_usage(
                knowledge_base_id=knowledge_base_id
            )
            return view

    async def update_knowledge_base(
        self, *, domain_id: int, knowledge_base_id: UUID,
        command: KnowledgeBaseWrite, context: AuthContext,
    ) -> dict[str, Any]:
        if command.expected_row_version is None:
            raise AIOpsApplicationError(
                code="PRECONDITION_REQUIRED",
                message="修改运维知识库必须提供当前版本",
                status_code=428,
            )
        await self._validate_knowledge_base_models(context=context, models=command.models)
        configuration = self._documents.configuration(command.models)
        display_name = command.display_name.strip()
        async with self._uow_factory() as uow:
            row = await uow.operations_knowledge.get_knowledge_base(
                domain_id=domain_id,
                knowledge_base_id=knowledge_base_id,
                for_update=True,
            )
            if row is None or row.status == "ARCHIVED":
                raise self._knowledge_base_not_found()
            if int(row.row_version) != command.expected_row_version:
                raise AIOpsApplicationError(
                    code="STATE_VERSION_CONFLICT",
                    message="运维知识库已被其他请求修改，请刷新后重试",
                    status_code=412,
                )
            duplicate = await uow.operations_knowledge.find_knowledge_base_name(
                domain_id=domain_id,
                normalized_name=" ".join(display_name.casefold().split()),
                exclude_knowledge_base_id=knowledge_base_id,
            )
            if duplicate is not None:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_BASE_NAME_CONFLICT",
                    message="当前 Domain 已存在同名运维知识库",
                    status_code=409,
                )
            row.configuration_json = configuration
            row.display_name = display_name
            row.description = command.description
            row.updated_by = self._actor(context)
            row.updated_at = _now()
            row.row_version = int(row.row_version) + 1
            await uow.commit()
        return await self.get_knowledge_base(
            domain_id=domain_id, knowledge_base_id=knowledge_base_id
        )

    async def delete_knowledge_base(self, *, domain_id, knowledge_base_id, expected_row_version, context):
        async with self._uow_factory() as uow:
            row = await uow.operations_knowledge.get_knowledge_base(
                domain_id=domain_id, knowledge_base_id=knowledge_base_id, for_update=True)
            if row is None or row.status == "ARCHIVED":
                raise self._knowledge_base_not_found()
            if int(row.row_version) != expected_row_version:
                raise self._state_conflict("运维知识库版本已经变化")
            usage = await uow.operations_knowledge.knowledge_base_usage(knowledge_base_id=knowledge_base_id)
            if any(usage.values()):
                raise self._state_conflict("知识库仍被资料或 Agent 引用，不能删除")
            await uow.operations_knowledge.delete_knowledge_base(row)
            await uow.commit()





    async def get_collection_models(self, *, domain_id, knowledge_base_id, context):
        async with self._uow_factory() as uow:
            row = await uow.operations_knowledge.get_knowledge_base(
                domain_id=domain_id, knowledge_base_id=knowledge_base_id)
            if row is None or row.status != "ACTIVE":
                raise self._knowledge_base_not_found()
            return {"knowledge_base_id": str(knowledge_base_id),
                **dict(row.configuration_json or {})}

    async def configure_collections(self, *, domain_id, knowledge_base_id, models, context):
        await self._validate_knowledge_base_models(context=context, models=models)
        configuration = self._documents.configuration(models)
        async with self._uow_factory() as uow:
            row = await uow.operations_knowledge.get_knowledge_base(
                domain_id=domain_id, knowledge_base_id=knowledge_base_id, for_update=True)
            if row is None or row.status != "ACTIVE":
                raise self._knowledge_base_not_found()
            row.configuration_json, row.updated_by = configuration, self._actor(context)
            await uow.commit()
        return {"knowledge_base_id": str(knowledge_base_id), **configuration}

    async def _validate_knowledge_base_models(
        self, *, models: dict[str, UUID], context: AuthContext, required_roles: frozenset[str] | None = None,
    ) -> None:
        required = ({"parser_ocr", "parser_llm", "retrieval_llm", "embedding"}
            if required_roles is None else set(required_roles))
        missing = required - set(models)
        if missing:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_COLLECTION_MODELS_REQUIRED",
                message=f"运维知识库缺少模型角色：{', '.join(sorted(missing))}",
                status_code=422,
            )
        expected = {
            "parser_ocr": int(ModelCategory.OCR),
            "parser_vlm": int(ModelCategory.VLM),
            "parser_llm": int(ModelCategory.LLM),
            "retrieval_llm": int(ModelCategory.LLM),
            "embedding": int(ModelCategory.TXT_EMBEDDING),
            "visual_embedding": int(ModelCategory.IMG_EMBEDDING),
        }
        for role, model_id in models.items():
            if role not in expected:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_MODEL_ROLE_INVALID",
                    message=f"不支持的运维知识库模型角色：{role}",
                    status_code=422,
                )
            model_catalog = self._model_catalogs.get(expected[role])
            if model_catalog is None:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_MODEL_UNAVAILABLE",
                    message=f"模型角色 {role} 的目录服务暂时不可用",
                    status_code=503,
                    retryable=True,
                )
            try:
                definition = await model_catalog.get_model(
                    model_id, auth_context=context,
                )
            except (LookupError, RuntimeError) as exc:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_MODEL_UNAVAILABLE",
                    message=f"模型角色 {role} 当前不可用",
                    status_code=409,
                ) from exc
            if (
                int(definition.get("category") or 0) != expected[role]
                or definition.get("status") != "ACTIVE"
            ):
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_MODEL_UNAVAILABLE",
                    message=f"模型角色 {role} 未绑定到已启用的正确类别模型",
                    status_code=409,
                )

    async def _case_configuration(self, *, domain_id, context):
        access = self._documents.access(context)
        if str(access.domain_id) != str(domain_id):
            raise AIOpsApplicationError(code="DOMAIN_CONTEXT_MISMATCH", message="案例配置必须属于当前空间", status_code=403)
        async with self._uow_factory() as uow:
            row = await uow.knowledge_documents.case_configuration(access)
            if row is None:
                raise self._state_conflict("请先配置 AIOps 本域案例模型和解析检索 Profile")
            return {"models": dict(row.models_json), "profile_ref": row.profile_ref,
                "row_version": int(row.row_version)}

    async def get_case_collection_models(self, *, domain_id, context):
        access = self._documents.access(context)
        if str(access.domain_id) != str(domain_id):
            raise AIOpsApplicationError(code="DOMAIN_CONTEXT_MISMATCH", message="案例配置必须属于当前空间", status_code=403)
        async with self._uow_factory() as uow:
            row = await uow.knowledge_documents.case_configuration(access)
            models = dict(row.models_json) if row else {}
            missing = sorted(set(CASE_MODEL_ROLES) - set(models))
            return {"owner_app_id": "aiops", "models": models,
                "profile_ref": row.profile_ref if row else None,
                "row_version": int(row.row_version) if row else None,
                "configuration_status": "MODELS_REQUIRED" if missing else "CONFIGURED",
                "missing_model_roles": missing, "required_model_roles": sorted(CASE_MODEL_ROLES),
                "retrieval_requires_publication": True}

    async def configure_case_collection_models(self, *, domain_id, models, context):
        command = CaseCollectionModels(models=models)
        await self._validate_knowledge_base_models(context=context, models=command.models, required_roles=CASE_MODEL_ROLES)
        configuration = self._documents.configuration(models)
        access = self._documents.access(context)
        if str(access.domain_id) != str(domain_id):
            raise AIOpsApplicationError(code="DOMAIN_CONTEXT_MISMATCH", message="案例配置必须属于当前空间", status_code=403)
        async with self._uow_factory() as uow:
            row = await uow.knowledge_documents.configure_cases(access,
                models=configuration["models"], profile_ref=configuration["profile_ref"])
            await uow.commit()
        return await self.get_case_collection_models(domain_id=domain_id, context=context)

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
        """对已审批手册识别受控标签，成功后批准 本 App 开始正式解析。"""
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
            index_locator = tuple(row.document_version_id for row in relations["indexes"])
            base_id = asset.knowledge_base_id
            vocabulary = [{
                "tag_id": str(tag.tag_id), "category": str(tag.category),
                "display_name": str(tag.display_name),
                "description": tag.description,
            } for tag in tags]
            approval = dict(
                getattr(version, "tag_analysis_json", None) or {}
            ).get("approval") or {}
        if asset_kind != "MANUAL" or version_status != "TAGGING":
            return await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
        if approval.get("status") != "APPROVED":
            async with self._uow_factory() as uow:
                locked = await uow.operations_knowledge.get_version(
                    domain_id=domain_id,
                    asset_version_id=asset_version_id,
                    for_update=True,
                )
                if locked is None:
                    raise self._not_found()
                locked_asset = await uow.operations_knowledge.get_asset(
                    domain_id=domain_id,
                    asset_id=locked.asset_id,
                    for_update=True,
                )
                if locked.status == "TAGGING":
                    locked.status = "REVIEW_REQUIRED"
                    if locked_asset.current_version_id is None:
                        locked_asset.status = "REVIEW_REQUIRED"
                    await uow.commit()
            return await self.get_version(
                domain_id=domain_id, asset_version_id=asset_version_id
            )
        if self._structured_model is None or self._prompt_registry is None or self._llm_model_catalog is None:
            return await self._mark_tagging_failed(
                domain_id=domain_id, asset_version_id=asset_version_id,
                code="TAG_MODEL_UNAVAILABLE", message="标签识别模型或 Prompt Registry 未初始化",
            )
        if not vocabulary:
            return await self._mark_tagging_failed(
                domain_id=domain_id, asset_version_id=asset_version_id,
                code="TAG_VOCABULARY_EMPTY", message="请先在配置与治理中维护至少一个有效知识标签",
            )
        if not index_locator:
            return await self._mark_tagging_failed(
                domain_id=domain_id, asset_version_id=asset_version_id,
                code="TAG_SOURCE_MISSING", message="知识版本缺少 本 App 源文件引用",
            )
        try:
            access = self._documents.access(context)
            for document_version_id in index_locator:
                processed = await self._documents.process(access, document_version_id)
                if processed["processing_status"] != "PARSED" or not processed["generation_id"]:
                    return await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
            projection = await self._documents.excerpts(access, index_locator)
            excerpts = list(projection.get("excerpts") or [])
            if not excerpts:
                raise ValueError("规范解析结果中没有可用正文")
            configuration = await self.get_collection_models(domain_id=domain_id,
                knowledge_base_id=base_id, context=context)
            model_id = configuration.get("models", {}).get("parser_llm")
            if model_id is None:
                raise ValueError("运维知识库尚未配置 parser_llm")
            definition = await self._llm_model_catalog.get_model(UUID(str(model_id)))
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
            source_by_id = {}
            for item in excerpts:
                identity = str(item["document_version_id"])
                source_by_id.setdefault(identity, []).append(item)
            suggestions: list[dict[str, Any]] = []
            seen: set[str] = set()
            for suggestion in output.suggestions:
                tag = allowed.get(suggestion.tag_id)
                if tag is None or suggestion.tag_id in seen:
                    continue
                evidence = []
                for item in suggestion.evidence:
                    source = next((row for row in source_by_id.get(item.document_version_id, ())
                        if item.quote and item.quote in str(row.get("content") or "")), None)
                    if source is None:
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
            if not suggestions:
                raise ValueError("AI 未识别出具有原文证据的有效受控标签")
            analysis = {
                "schema_version": "OPS_KNOWLEDGE_TAG_ANALYSIS.v1",
                "status": "READY", "approval": approval,
                "suggestions": suggestions,
                "warnings": list(projection.get("warnings") or []),
                "vocabulary": vocabulary,
                "vocabulary_hash": hashlib.sha256(json.dumps(vocabulary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "model_receipt": result.receipt.model_dump(mode="json"),
            }
            selected_tags = [KnowledgeTagScopeSnapshot(
                tag_id=UUID(item["tag_id"]),
                category=item["category"],
                display_name=item["display_name"],
            ) for item in suggestions]
            analysis["assigned_tag_ids"] = [
                str(tag.tag_id) for tag in selected_tags
            ]
            analysis["assigned_at"] = _now().isoformat()
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
                relations = await uow.operations_knowledge.get_version_relations(
                    asset_version_id=asset_version_id
                )
                await uow.operations_knowledge.replace_tag_scopes(
                    version=locked,
                    tags=selected_tags,
                )
                locked.tag_analysis_json = analysis
                locked.status = "PROCESSING"
                for locked_index in relations["indexes"]:
                    locked_index.index_status = "PROCESSING"
                    locked_index.error_code = None
                    locked_index.error_summary = None
                if locked_asset.current_version_id is None:
                    locked_asset.status = "PROCESSING"
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
            previous_analysis = dict(version.tag_analysis_json or {})
            version.tag_analysis_json = {
                "schema_version": "OPS_KNOWLEDGE_TAG_ANALYSIS.v1", "status": "FAILED",
                "approval": previous_analysis.get("approval"),
                "error_code": str(code), "error_summary": str(message)[:2000], "suggestions": [],
            }
            if asset.current_version_id is None:
                asset.status = "TAGGING_FAILED"
            await uow.commit()
        return await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)

    async def list_assets(
        self, *, domain_id: int, asset_kind: str | None, status: str | None,
        knowledge_base_id: UUID | None, limit: int,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await uow.operations_knowledge.list_assets(
                domain_id=domain_id, asset_kind=asset_kind, status=status,
                knowledge_base_id=knowledge_base_id, limit=limit,
            )
            return [_asset_view(row) for row in rows]

    async def _conversation_case_context(
        self, *, domain_id, conversation_id, actor_id, included_turn_ids=None
    ) -> dict[str, Any]:
        """按用户选择冻结多轮有效判断，每个来源均重新核权。"""
        from aiops_agent.application.case_closure import conversation_case_context
        async with self._uow_factory() as uow:
            result = await conversation_case_context(uow=uow, domain_id=domain_id,
                conversation_id=conversation_id, actor_id=actor_id, included_turn_ids=included_turn_ids,
                plan_capture_reader=getattr(self, "_plan_capture_reader", None))
            existing = await uow.operations_knowledge.find_case_by_external_id(
                domain_id=domain_id, external_id=str(conversation_id))
            if (existing is not None and result["eligible"]
                and dict(existing[1].profile_json or {}).get("source", {}).get("source_fingerprint") == result["source_fingerprint"]):
                try:
                    await self._validated_case_replay(uow=uow, domain_id=domain_id, case=existing)
                    result["existing_asset_id"] = str(existing[0].asset_id)
                except AIOpsApplicationError:
                    pass
            return result

    async def conversation_case_preview(
        self, *, domain_id: int, conversation_id: UUID, context: AuthContext,
        included_turn_ids: tuple[UUID, ...] | None = None,
    ) -> dict[str, Any]:
        result = await self._conversation_case_context(
            domain_id=domain_id, conversation_id=conversation_id, actor_id=self._actor(context),
            included_turn_ids=included_turn_ids,
        )
        return {key: value for key, value in result.items() if key != "_context"}

    async def close_diagnosis(
        self, *, domain_id: int, conversation_id: UUID,
        command: DiagnosisClosureCommand, context: AuthContext,
    ) -> dict[str, Any]:
        """多轮收口共用同一来源、脱敏与索引审批围栏，不自动发布。"""
        from aiops_agent.application.case_closure import close_conversation_case
        return await close_conversation_case(self, domain_id=domain_id, conversation_id=conversation_id,
            command=command, context=context,
            entity_scope={}, kc_scope={"domain_id": domain_id},
            model_catalog=self._llm_model_catalog, model_scope={},
            technical_name_field="served_model_name")

    async def list_cases(
        self, *, domain_id: int, status: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """返回面向案例治理的业务投影，不暴露 KC 内部对象。"""
        async with self._uow_factory() as uow:
            assets = await uow.operations_knowledge.list_assets(
                domain_id=domain_id, asset_kind="DIAGNOSIS_CASE", status=None,
                knowledge_base_id=None, limit=500,
            )
            result: list[dict[str, Any]] = []
            for asset in assets:
                versions = await uow.operations_knowledge.list_versions(
                    domain_id=domain_id, asset_id=asset.asset_id,
                )
                if not versions:
                    continue
                version = versions[0]
                if status is not None and version.status != status:
                    continue
                relations = await uow.operations_knowledge.get_version_relations(
                    asset_version_id=version.asset_version_id,
                )
                result.append(_case_item_view({
                    "asset": _asset_view(asset),
                    "version": _version_view(version),
                    "sources": [{
                        "source_kind": item.source_kind,
                        "report_id": str(item.source_report_id) if item.source_report_id else None,
                    } for item in relations["sources"]],
                }))
            return result[:limit]

    async def cases_overview(self, *, domain_id: int) -> dict[str, int]:
        rows = await self.list_cases(domain_id=domain_id, status=None, limit=500)
        return {
            "draft": sum(item["status"] == "DRAFT" for item in rows),
            "review_required": sum(item["status"] == "REVIEW_REQUIRED" for item in rows),
            "published": sum(item["status"] == "PUBLISHED" for item in rows),
            "processing": sum(item["status"] in {"PROCESSING", "FAILED"} for item in rows),
            "retired": sum(item["status"] == "RETIRED" for item in rows),
        }

    async def get_case(self, *, domain_id: int, asset_id: UUID,
        asset_version_id: UUID | None = None) -> dict[str, Any]:
        asset_page = await self.get_asset(domain_id=domain_id, asset_id=asset_id)
        if asset_page["asset"]["asset_kind"] != "DIAGNOSIS_CASE":
            raise self._not_found()
        versions = asset_page["versions"]
        if not versions:
            raise self._not_found()
        detail = await self.get_version(
            domain_id=domain_id,
            asset_version_id=asset_version_id or UUID(versions[0]["asset_version_id"]),
        )
        if str(detail["asset"]["asset_id"]) != str(asset_id):
            raise self._not_found()
        detail["versions"] = versions
        return detail

    async def create_case_version(
        self, *, domain_id: int, asset_id: UUID,
        command: DiagnosisCaseDraft, context: AuthContext,
    ) -> dict[str, Any]:
        """基于最新案例生成不可变的新版本，证据链始终继承自基线版本。"""
        actor = self._actor(context)
        async with self._uow_factory() as uow:
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=asset_id
            )
            if asset is None or asset.asset_kind != "DIAGNOSIS_CASE":
                raise self._not_found()
            if int(asset.row_version) != command.expected_asset_row_version:
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_ASSET_CONFLICT",
                    message="案例已被其他请求更新，请刷新后重试",
                    status_code=412,
                )
            versions = await uow.operations_knowledge.list_versions(
                domain_id=domain_id, asset_id=asset_id
            )
            if not versions or versions[0].asset_version_id != command.base_version_id:
                raise self._state_conflict("只能基于案例最新版本创建修订")
            base = versions[0]
            if base.status in {"PROCESSING", "REVIEW_REQUIRED"}:
                raise self._state_conflict("当前案例正在处理或审核，不能创建修订")
            base_profile = DiagnosisCaseProfile.model_validate(base.profile_json)
            profile = _prepare_case_revision(
                command=command, base_profile=base_profile
            )
            profile = CaseRedactor().profile(profile)
            relations = await uow.operations_knowledge.get_version_relations(
                asset_version_id=base.asset_version_id
            )
            source_snapshots = [{
                "source_kind": item.source_kind,
                "source_report_id": item.source_report_id,
                "source_run_id": item.source_run_id,
                "source_artifact_id": item.source_artifact_id,
                "source_external_id": item.source_external_id,
                "content_hash": item.content_hash,
                "source_locator_json": dict(item.source_locator_json or {}),
            } for item in relations["sources"]]
            target_scope = next((
                str(item.scope_value) for item in relations["scopes"]
                if item.scope_kind == "TARGET_ID"
            ), None)
            next_version_no = int(base.version_no) + 1
            base_source_hash = str(base.source_hash)
            base_status = str(base.status)
            current_version_id = asset.current_version_id

        collection = await self._case_configuration(domain_id=domain_id, context=context)
        display_name, profile = await redact_case_for_index(
            title=command.display_name, profile=profile, collection=collection,
            structured_model=self._structured_model, prompt_registry=self._prompt_registry,
            model_catalog=self._llm_model_catalog, model_scope={}, technical_name_field="served_model_name",
        )
        markdown = _diagnosis_case_markdown(title=display_name, profile=profile)
        source_hash = hashlib.sha256(json.dumps(
            {"display_name": command.display_name, "profile": profile.model_dump(mode="json")},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if source_hash == base_source_hash:
            raise self._state_conflict("案例内容没有变化，无需创建新版本")
        staged = await self._documents.stage(file_name=f"diagnosis-case-{asset_id}-v{next_version_no}.md",
            media_type="text/markdown", body=markdown, expected_hash=hashlib.sha256(markdown).hexdigest())
        version_id = uuid7()
        version = OperationsKnowledgeVersionEntity(
            asset_version_id=version_id,
            asset_id=asset_id,
            version_no=next_version_no,
            status="PROCESSING",
            source_hash=source_hash,
            profile_schema_version=profile.schema_version,
            profile_json=profile.model_dump(mode="json"),
            extraction_warnings_json=[],
            created_by=actor,
        )
        sources = [OperationsKnowledgeSourceEntity(
            asset_version_id=version_id,
            source_kind=value["source_kind"],
            source_report_id=value["source_report_id"],
            source_run_id=value["source_run_id"],
            source_artifact_id=value["source_artifact_id"],
            source_external_id=value["source_external_id"],
            content_hash=value["content_hash"],
            source_locator_json={
                **value["source_locator_json"],
                "revised_from_version_id": str(command.base_version_id),
            },
        ) for value in source_snapshots]
        scope_values = [
            ("TARGET_ID", target_scope),
            ("DATABASE_TYPE", profile.environment.get("database_type")),
            ("DATABASE_VERSION", profile.environment.get("database_version")),
            ("PLATFORM", profile.environment.get("platform")),
            ("PROBLEM_CLASS", profile.problem_signature.get("problem_class")),
            ("TOPOLOGY", profile.environment.get("topology")),
        ]
        scopes = [OperationsKnowledgeScopeEntity(
            asset_version_id=version_id,
            scope_kind=kind,
            scope_value=str(value),
            normalized_value=_normalize(str(value)),
            source_kind="CASE_REVISION",
            source_locator_json={"base_version_id": str(command.base_version_id)},
        ) for kind, value in scope_values if value]
        async with self._uow_factory() as uow:
            locked_asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=asset_id, for_update=True
            )
            latest = await uow.operations_knowledge.list_versions(
                domain_id=domain_id, asset_id=asset_id
            )
            if (
                locked_asset is None
                or int(locked_asset.row_version) != command.expected_asset_row_version
                or not latest
                or latest[0].asset_version_id != command.base_version_id
            ):
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_ASSET_CONFLICT",
                    message="案例已被其他请求更新，请刷新后重试",
                    status_code=412,
                )
            locked_asset.display_name = display_name
            locked_asset.updated_by = actor
            if current_version_id is None:
                locked_asset.status = "PROCESSING"
            identity = await self._documents.register(uow, self._documents.access(context),
                staged=staged, asset_id=asset_id, asset_version_id=version_id)
            index = OperationsKnowledgeIndexEntity(
                asset_version_id=version_id, **identity, index_status="PROCESSING")
            await uow.operations_knowledge.add_version(
                version=version, scopes=scopes, sources=sources, indexes=[index]
            )
            await uow.operations_knowledge.add_review(
                OperationsKnowledgeReviewEntity(
                    asset_version_id=version_id,
                    decision="REVISE",
                    reviewer_id=actor,
                    comment_text="基于案例现有版本创建人工修订",
                    before_status=base_status,
                    after_status="PROCESSING",
                )
            )
            await uow.commit()
        return await self.get_case(domain_id=domain_id, asset_id=asset_id)

    async def case_eligibility(
        self, *, domain_id: int, report_id: UUID
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            report = await uow.inspections.get_report_scoped(
                report_id=report_id, domain_id=domain_id,
            )
            if report is None:
                raise self._not_found()
            try:
                report, content, report_sources = await load_case_report(
                    uow=uow, domain_id=domain_id, report_id=report_id,
                    plan_capture_reader=getattr(self, "_plan_capture_reader", None),
                )
            except AIOpsApplicationError as exc:
                return {"eligible": False, "reasons": [{
                    "code": exc.code, "message": exc.message,
                }]}
            artifact = await uow.runs.get_artifact(artifact_id=report.content_artifact_id)
            evidence_refs = [
                str(item.get("artifact_id"))
                for item in content.get("evidence_refs") or ()
                if isinstance(item, dict) and item.get("artifact_id")
            ] or [str(row.source_artifact_id) for row in report_sources]
            result = _diagnosis_case_eligibility(
                report=report, report_content=content, evidence_refs=evidence_refs,
            )
            if report.report_type == "COMPARISON" and content.get("scope", {}).get("comparison_result_artifact_id"):
                from aiops_agent.application.configuration.common import sha256_json
                from aiops_agent.application.effect_case import profile_from_effect_report
                try:
                    if artifact.content_hash != report.content_hash or sha256_json(content) != artifact.content_hash:
                        raise self._state_conflict("效果报告内容哈希不一致")
                    profile = await profile_from_effect_report(
                        uow=uow, domain_id=domain_id, report=report, content=content, sources=report_sources,
                        plan_capture_reader=getattr(self, "_plan_capture_reader", None),
                    )
                    if profile is not None:
                        result = {"eligible": True, "reasons": [], "case_kind": profile.case_kind}
                except AIOpsApplicationError:
                    result = {"eligible": False, "reasons": [{
                        "code": "EFFECT_SOURCE_INVALID", "message": "效果来源复核未通过，不能生成案例候选",
                    }]}
            closure_artifact_id = dict(content.get("scope") or {}).get(
                "closure_artifact_id"
            )
            try:
                closure_uuid = UUID(str(closure_artifact_id))
            except (TypeError, ValueError):
                closure_uuid = None
            existing = (
                await uow.operations_knowledge.find_case_by_artifact_id(
                    domain_id=domain_id, artifact_id=closure_uuid,
                )
                if closure_uuid is not None else None
            )
            if existing is None:
                existing = await uow.operations_knowledge.find_by_source_hash(
                    domain_id=domain_id, asset_kind="DIAGNOSIS_CASE",
                    source_hash=str(report.content_hash or ""),
                )
            return {
                **result,
                "existing_asset_id": str(existing[0].asset_id) if existing is not None else None,
                "existing_version_id": str(existing[1].asset_version_id) if existing is not None else None,
            }

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

    async def delete_asset(self, *, domain_id, asset_id,
            expected_row_version, purge_source_objects=False, context):
        """撤回误上传文档并保留审计；保留期内不物理删除共享内容地址对象。"""
        if purge_source_objects:
            raise self._state_conflict("源文件受证据保留策略保护；请使用普通撤回删除")
        access = self._documents.access(context)
        cancelled = []
        async with self._uow_factory() as uow:
            asset = await uow.operations_knowledge.get_asset(domain_id=domain_id, asset_id=asset_id, for_update=True)
            if asset is None:
                raise self._not_found()
            if int(asset.row_version) != expected_row_version or asset.asset_kind != "MANUAL":
                raise self._state_conflict("只能删除版本未变化的上传手册")
            versions = await uow.operations_knowledge.list_versions(domain_id=domain_id, asset_id=asset_id)
            if not versions or any(row.status not in REQUESTABLE_MANUAL_DELETE_STATUSES for row in versions):
                raise self._state_conflict("已发布或退役资料不能直接删除")
            for version in versions:
                relations = await uow.operations_knowledge.get_version_relations(asset_version_id=version.asset_version_id)
                cancelled.extend(await self._documents.withdraw(uow, access, relations["indexes"], cancel=True))
            await uow.operations_knowledge.delete_asset(asset=asset, versions=versions)
            await uow.commit()
        for lease in cancelled:
            try:
                await self._documents.ingestion.compute.cancel(lease)
            except Exception:
                # App 已提交取消栅栏；计算服务故障不能使迟到数据重新落库。
                logger.warning("运维知识计算取消暂未送达，等待任务截止回收")

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
                "external_withdrawals": await withdrawal_views(
                    uow=uow, indexes=relations["indexes"], reviews=relations["reviews"],
                ) if asset.asset_kind == "DIAGNOSIS_CASE" else [],
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

    async def stream_source(self, *, domain_id, asset_version_id, context, range_header, document_version_id=None):
        from types import SimpleNamespace
        from urllib.parse import quote
        detail = await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
        if not detail["indexes"]:
            raise self._not_found()
        if range_header:
            raise AIOpsApplicationError(code="RANGE_NOT_SUPPORTED", message="原文件下载暂不支持分段请求", status_code=416)
        matches = [row for row in detail["indexes"]
            if document_version_id is None or row["document_version_id"] == str(document_version_id)]
        if len(matches) != 1:
            raise AIOpsApplicationError(code="DOCUMENT_SELECTION_REQUIRED",
                message="请选择该版本中的一个原始文档", status_code=422)
        index = matches[0]
        access = self._documents.access(context)
        async def authorize():
            current = await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
            if current["asset"]["row_version"] != detail["asset"]["row_version"] or current["version"]["row_version"] != detail["version"]["row_version"]:
                raise self._state_conflict("原件读取期间知识版本已变化")
        record, body = await self._documents.source(access, UUID(index["document_version_id"]), authorize=authorize)
        return SimpleNamespace(body=body, status_code=200, headers={
            "content-length": str(record["byte_count"]), "content-type": record["mime_type"],
            "content-disposition": "attachment; filename*=UTF-8''" + quote(record["title"], safe="")})

    async def ingest_upload(
        self,
        *,
        domain_id: int,
        knowledge_base_id: UUID,
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
                        knowledge_base_id=knowledge_base_id,
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
                    knowledge_base_id=knowledge_base_id,
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
                    "documents": result.get("documents", []),
                    "replayed": bool(result.get("replayed")),
                } for document in documents]
        return {
            "grouping_mode": grouping_mode,
            "items": items,
            "request_id": context.request_id,
        }

    async def ingest_manual_bundle(self, *, domain_id, knowledge_base_id, context,
            title, documents, metadata, idempotency_key):
        return await self._register_manual_documents(domain_id=domain_id,
            knowledge_base_id=knowledge_base_id, context=context, title=title,
            documents=documents, metadata=metadata, idempotency_key=idempotency_key)

    async def ingest_manual(self, *, domain_id, knowledge_base_id, context,
            file_name, media_type, body, metadata, idempotency_key, declared_sha256):
        document = {"file_name": file_name, "media_type": media_type, "body": body,
            "content_sha256": declared_sha256 or hashlib.sha256(body).hexdigest(),
            "ordinal": 0, "role": "CONTENT", "required_flag": True}
        return await self._register_manual_documents(domain_id=domain_id,
            knowledge_base_id=knowledge_base_id, context=context, title=metadata.display_name,
            documents=[document], metadata=metadata, idempotency_key=idempotency_key)

    async def ingest_manual_version(self, *, domain_id, asset_id, expected_asset_row_version,
            context, file_name, media_type, body, metadata, idempotency_key, declared_sha256):
        detail = await self.get_asset(domain_id=domain_id, asset_id=asset_id)
        asset = detail.get("asset", detail)
        return await self._register_manual_documents(domain_id=domain_id,
            knowledge_base_id=UUID(asset["knowledge_base_id"]), context=context,
            title=metadata.display_name, documents=[{"file_name": file_name, "media_type": media_type,
                "body": body, "content_sha256": declared_sha256 or hashlib.sha256(body).hexdigest(),
                "ordinal": 0, "role": "CONTENT", "required_flag": True}],
            metadata=metadata, idempotency_key=idempotency_key,
            asset_id=asset_id, expected_asset_row_version=expected_asset_row_version)

    async def _register_manual_documents(self, *, domain_id, knowledge_base_id, context,
            title, documents, metadata, idempotency_key, asset_id=None, expected_asset_row_version=None):
        """对象写入在事务外；同库行锁保护幂等、业务登记和摄取任务的原子性。"""
        from knowledge_algorithms.contracts.identity import fingerprint
        if not idempotency_key or len(idempotency_key) > 256 or not 1 <= len(documents) <= 20:
            raise self._state_conflict("上传幂等键或文件数量无效")
        access = self._documents.access(context)
        if str(access.domain_id) != str(domain_id):
            raise self._state_conflict("上传 Domain 与可信上下文不一致")
        staged = []
        for document in documents:
            staged.append(await self._documents.stage(file_name=document["file_name"],
                media_type=document["media_type"], body=document["body"],
                expected_hash=document["content_sha256"].lower()))
        digest = fingerprint({"title": title, "metadata": metadata.model_dump(mode="json"),
            "asset_id": str(asset_id) if asset_id else None,
            "files": [{key: value for key, value in doc.items() if key != "body"} for doc in documents]})
        actor = self._actor(context)
        scope = {}
        async with self._uow_factory() as uow:
            base = await uow.operations_knowledge.get_knowledge_base(domain_id=domain_id,
                knowledge_base_id=knowledge_base_id, for_update=True)
            if base is None or base.status != "ACTIVE":
                raise self._knowledge_base_not_found()
            existing = await uow.operations_knowledge.find_upload_request(domain_id=domain_id,
                knowledge_base_id=knowledge_base_id, request_key=idempotency_key)
            if existing is not None:
                if existing[2].content_hash != digest:
                    raise self._state_conflict("同一幂等键不能用于不同文件或业务元数据")
                return {"asset": _asset_view(existing[0]), "version": _version_view(existing[1]), "replayed": True}
            configuration = self._documents.configuration(dict(base.configuration_json["models"]))
            if configuration != base.configuration_json:
                raise self._state_conflict("知识库 Profile 已变化，请显式更新配置后上传")
            asset = None
            if asset_id is not None:
                asset = await uow.operations_knowledge.get_asset(domain_id=domain_id, asset_id=asset_id, for_update=True)
                if (asset is None or asset.asset_kind != "MANUAL" or asset.knowledge_base_id != knowledge_base_id
                    or int(asset.row_version) != expected_asset_row_version):
                    raise self._state_conflict("手册资产或版本已变化")
            is_new = asset is None
            if is_new:
                asset = OperationsKnowledgeAssetEntity(domain_id=domain_id, asset_id=uuid7(),
                    knowledge_base_id=knowledge_base_id, asset_kind="MANUAL", display_name=title,
                    status="REVIEW_REQUIRED", security_level=metadata.security_level, created_by=actor, updated_by=actor)
            version_id = uuid7()
            version = OperationsKnowledgeVersionEntity(**scope, asset_version_id=version_id,
                asset_id=asset.asset_id, version_no=1 if is_new else await uow.operations_knowledge.next_version_no(asset_id=asset.asset_id),
                status="REVIEW_REQUIRED", source_hash=digest, extraction_warnings_json=[], created_by=actor)
            sources, indexes = [], []
            for doc, item in zip(documents, staged, strict=True):
                identity = await self._documents.register(uow, access, staged=item,
                    asset_id=asset.asset_id, asset_version_id=version_id)
                sources.append(OperationsKnowledgeSourceEntity(**scope, asset_version_id=version_id,
                    source_kind="MANUAL_FILE", source_external_id=doc["file_name"],
                    content_hash=item["source"].sha256,
                    source_locator_json={"metadata": metadata.model_dump(mode="json"),
                        "document_version_id": str(item["document_version_id"]),
                        "ordinal": doc["ordinal"], "role": doc["role"], "required_flag": doc["required_flag"]}))
                indexes.append(OperationsKnowledgeIndexEntity(**scope, asset_version_id=version_id,
                    **identity, index_status="PENDING_APPROVAL"))
            sources.append(OperationsKnowledgeSourceEntity(**scope, asset_version_id=version_id,
                source_kind="UPLOAD_REQUEST", source_external_id=idempotency_key, content_hash=digest,
                source_locator_json={"configuration": configuration}))
            if is_new:
                await uow.operations_knowledge.add_asset_version(asset=asset, version=version,
                    scopes=[], sources=sources, indexes=indexes)
            else:
                await uow.operations_knowledge.add_version(version=version, scopes=[], sources=sources, indexes=indexes)
                asset.display_name, asset.updated_by = title, actor
                asset.security_level = max(int(asset.security_level), metadata.security_level)
                if asset.current_version_id is None:
                    asset.status = "REVIEW_REQUIRED"
            await uow.commit()
            return {"asset": _asset_view(asset), "version": _version_view(version),
                "documents": [_index_view(row) for row in indexes], "replayed": False}

    async def _validated_case_replay(self, *, uow, domain_id, case):
        """返回已提炼版本也必须复核其自身来源，而不只看新请求的报告。"""
        if not await validate_case_sources(
            uow=uow, domain_id=domain_id, version_id=case[1].asset_version_id,
            plan_capture_reader=getattr(self, "_plan_capture_reader", None),
        ) or not await validate_case_effect(uow=uow, domain_id=domain_id, profile=case[1].profile_json):
            raise rejected_source()
        return {"asset": _asset_view(case[0]), "version": _version_view(case[1]), "replayed": True}

    async def _case_index_source_fence(self, *, domain_id, version_id, actor, indexed=False):
        """索引审批前后再验来源；变化时保留待审核版本并持久化精确撤回命令。"""
        async with self._uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=version_id, for_update=True,
            )
            if version is None:
                raise self._not_found()
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=version.asset_id, for_update=True,
            )
            if asset is None:
                raise self._not_found()
            try:
                report_id = version.profile_json.get("source", {}).get("report_id")
                if report_id:
                    await load_case_report(
                        uow=uow, domain_id=domain_id, report_id=UUID(report_id),
                        expected_hash=version.source_hash,
                        plan_capture_reader=getattr(self, "_plan_capture_reader", None),
                    )
                valid = await validate_case_sources(
                    uow=uow, domain_id=domain_id, version_id=version_id,
                    plan_capture_reader=getattr(self, "_plan_capture_reader", None),
                ) and await validate_case_effect(uow=uow, domain_id=domain_id, profile=version.profile_json)
            except AIOpsApplicationError:
                valid = False
            if not valid:
                before_status = version.status
                version.status = "REVIEW_REQUIRED"
                if asset.current_version_id in {None, version_id}:
                    asset.current_version_id = None
                    asset.status = "REVIEW_REQUIRED"
                await queue_case_withdrawals(uow=uow, asset=asset, version=version)
                await uow.operations_knowledge.add_review(OperationsKnowledgeReviewEntity(
                    asset_version_id=version_id, decision="SOURCE_INVALIDATED",
                    reviewer_id=actor, before_status=before_status, after_status="REVIEW_REQUIRED",
                    comment_text="提炼期间报告、证据或授权变化；停止索引审批并申请撤回确切版本。",
                ))
            if valid and indexed and version.status == "PROCESSING":
                relations = await uow.operations_knowledge.get_version_relations(asset_version_id=version_id)
                for index in relations.get("indexes", ()):
                    if index.index_status == "PENDING_APPROVAL":
                        index.index_status = "PROCESSING"
            await uow.commit()
            return valid

    async def reconcile_version(self, *,  domain_id, asset_version_id, context, actor_id=None):
        """解析、投影在事务外；最后核对精确索引身份并原子提交业务发布。"""
        detail = await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
        if detail["version"]["status"] == "TAGGING":
            return await self.analyze_tags( domain_id=domain_id, asset_version_id=asset_version_id, context=context)
        if detail["version"]["status"] != "PROCESSING":
            return detail
        indexes = detail["indexes"]
        if not indexes:
            raise self._state_conflict("知识版本缺少本 App 文档引用")
        actor = actor_id if actor_id is not None else self._actor(context)
        is_case = detail["asset"]["asset_kind"] == "DIAGNOSIS_CASE"
        if is_case and not await self._case_index_source_fence(domain_id=domain_id, version_id=asset_version_id, actor=actor):
            raise rejected_source()
        access = self._documents.access(context)
        processed = {}
        for index in indexes:
            processed[index["index_ref_id"]] = await self._documents.process(access, UUID(index["document_version_id"]))
        if any(row["processing_status"] != "PARSED" or not row["generation_id"] for row in processed.values()):
            return detail
        excerpts = await self._documents.excerpts(access, tuple(UUID(index["document_version_id"]) for index in indexes))
        if is_case:
            profile = DiagnosisCaseProfile.model_validate(detail["version"]["profile"])
            expected = profile.knowledge_projection.body_sha256 if profile.knowledge_projection else None
            if not expected or len(processed) != 1 or next(iter(processed.values()))["source_hash"] != expected:
                raise self._state_conflict("案例规范正文哈希与已审阅投影不一致")
            if not await self._case_index_source_fence(domain_id=domain_id, version_id=asset_version_id, actor=actor, indexed=True):
                raise rejected_source()
        async with self._uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(domain_id=domain_id, asset_version_id=asset_version_id, for_update=True)
            if (version is None or version.status != "PROCESSING"
                or int(version.row_version) != int(detail["version"]["row_version"])):
                raise self._state_conflict("解析期间知识版本状态已变化")
            asset = await uow.operations_knowledge.get_asset(domain_id=domain_id, asset_id=version.asset_id, for_update=True)
            if asset is None or int(asset.row_version) != int(detail["asset"]["row_version"]):
                raise self._state_conflict("解析期间业务资料已变化")
            relations = await uow.operations_knowledge.get_version_relations(asset_version_id=asset_version_id)
            if {str(row.index_ref_id) for row in relations["indexes"]} != set(processed):
                raise self._state_conflict("解析期间文档集合已变化")
            for row in relations["indexes"]:
                record = processed[str(row.index_ref_id)]
                if str(row.document_version_id) != record["document_version_id"] or str(row.ingestion_job_id) != record["ingestion_job_id"]:
                    raise self._state_conflict("解析任务身份已变化")
                row.parse_run_id, row.index_generation_id = UUID(record["parse_run_id"]), UUID(record["generation_id"])
                row.index_status, row.last_checked_at = "READY", _now()
                row.error_code = row.error_summary = None
            if is_case:
                await assert_case_not_withdrawn(uow=uow, indexes=relations["indexes"])
                version.status = "DRAFT"
                if asset.current_version_id is None:
                    asset.status = "DRAFT"
            else:
                analysis = dict(version.tag_analysis_json or {})
                if analysis.get("status") != "READY" or (analysis.get("approval") or {}).get("status") != "APPROVED":
                    raise self._state_conflict("手册未完成人工审批与受控标签识别")
                profile = self._manual_profile(asset=asset, version=version, scopes=relations["scopes"],
                    sources=[row for row in relations["sources"] if row.source_kind == "MANUAL_FILE"],
                    revision={"extraction_evidence": [{"evidence_id": row["locator"]["key"],
                        "content_text": row["content"], "locator": row["locator"]} for row in excerpts["excerpts"]]})
                version.profile_json, version.profile_schema_version = profile.model_dump(mode="json"), profile.schema_version
                if asset.current_version_id and asset.current_version_id != asset_version_id:
                    previous = await uow.operations_knowledge.get_version(domain_id=domain_id, asset_version_id=asset.current_version_id, for_update=True)
                    previous_relations = await uow.operations_knowledge.get_version_relations(asset_version_id=previous.asset_version_id)
                    await self._documents.withdraw(uow, access, previous_relations["indexes"])
                    previous.status, previous.retired_at = "RETIRED", _now()
                await self._documents.publish(uow, access, relations["indexes"])
                version.status, version.published_at = "PUBLISHED", _now()
                asset.status, asset.current_version_id = "PUBLISHED", asset_version_id
            asset.updated_by = actor
            await uow.commit()
        return await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)

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
        expected_row_version: int,
        comment: str | None, context: AuthContext,
    ) -> dict[str, Any]:
        """人工批准上传资料；批准后才允许执行 AI 标签识别。"""
        detail = await self.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
        if detail["asset"]["asset_kind"] != "MANUAL":
            raise self._state_conflict("只有技术手册使用资料审批流程")
        if detail["version"]["status"] != "REVIEW_REQUIRED":
            raise self._state_conflict("当前手册不处于待审批状态")
        if int(detail["version"]["row_version"]) != expected_row_version:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_VERSION_CONFLICT",
                message="知识版本已被其他请求更新，请刷新后重试", status_code=412,
            )
        index_payload = next(iter(detail["indexes"]), None)
        if index_payload is None or index_payload["status"] != "PENDING_APPROVAL":
            raise self._state_conflict("当前手册的文档入库状态不能批准")
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
            analysis = dict(getattr(version, "tag_analysis_json", None) or {})
            analysis["approval"] = {
                "status": "APPROVED",
                "approved_at": _now().isoformat(),
                "approved_by": actor,
            }
            version.tag_analysis_json = analysis
            version.status = "TAGGING"
            if asset.current_version_id is None:
                asset.status = "TAGGING"
            asset.updated_by = actor
            await uow.operations_knowledge.add_review(OperationsKnowledgeReviewEntity(
                asset_version_id=asset_version_id, decision="APPROVE",
                reviewer_id=actor, comment_text=comment,
                before_status="REVIEW_REQUIRED", after_status="TAGGING",
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
            if decision == "SUBMIT_REVIEW":
                if asset.asset_kind != "DIAGNOSIS_CASE" or before != "DRAFT":
                    raise self._state_conflict("只有案例草稿可以提交审核")
                relations = await uow.operations_knowledge.get_version_relations(
                    asset_version_id=asset_version_id,
                )
                if not relations["indexes"] or any(
                    item.index_status != "READY" for item in relations["indexes"]
                ):
                    raise AIOpsApplicationError(
                        code="AIOPS_KNOWLEDGE_INDEX_PENDING",
                        message="案例正文索引尚未就绪，不能提交审核",
                        status_code=409,
                    )
                profile = DiagnosisCaseProfile.model_validate(version.profile_json)
                if not profile.root_cause.get("evidence_refs"):
                    raise self._state_conflict("案例缺少可追溯调查证据")
                version.status = "REVIEW_REQUIRED"
                if asset.current_version_id is None:
                    asset.status = "REVIEW_REQUIRED"
            elif decision == "PUBLISH":
                if before not in PUBLISHABLE_STATUSES:
                    raise self._state_conflict("当前知识版本不能发布")
                relations = await uow.operations_knowledge.get_version_relations(
                    asset_version_id=asset_version_id
                )
                if asset.asset_kind == "DIAGNOSIS_CASE":
                    await assert_case_not_withdrawn(uow=uow, indexes=relations["indexes"])
                    if not await validate_case_sources(uow=uow, domain_id=domain_id, version_id=asset_version_id,
                        plan_capture_reader=getattr(self, "_plan_capture_reader", None)):
                        raise self._state_conflict("案例来源已更正、撤权或无法核验，必须重新审核")
                    profile = DiagnosisCaseProfile.model_validate(version.profile_json)
                    if profile.redaction is None:
                        raise self._state_conflict("案例缺少脱敏审阅版本，请创建新版本后再发布")
                    scan = profile.redaction.semantic_provenance
                    frozen_title = profile.knowledge_projection.title if profile.knowledge_projection else asset.display_name
                    if scan.get("status") != "SCANNED" or scan.get("output_hash") != shared_content_hash(frozen_title, profile):
                        raise self._state_conflict("案例语义脱敏未完成或正文已变化，需创建新版本并重新审阅")
                    from aiops_agent.application.case_knowledge_projection import case_projection_is_current
                    if not case_projection_is_current(
                        title=profile.knowledge_projection.title if profile.knowledge_projection else asset.display_name,
                        profile=profile,
                    ):
                        raise self._state_conflict("案例知识正文或定位清单未就绪，需创建新版本并重新索引")
                    if not await validate_case_effect(uow=uow, domain_id=domain_id, profile=profile):
                        raise self._state_conflict("案例效果来源、执行回执或冻结测量无法核验，必须重新审核")
                    profile.redaction = profile.redaction.model_copy(update={"status": "REVIEWED"})
                    version.profile_json = profile.model_dump(mode="json")
                    # validate_case_sources 已核对每条展示修订及当前授权，不再仅按 is_current 拒绝。
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
                        await queue_case_withdrawals(uow=uow, asset=asset, version=previous)
                await self._documents.publish(uow, self._documents.access(context), relations["indexes"])
                version.status = "PUBLISHED"
                version.published_at = now
                asset.status = "PUBLISHED"
                asset.current_version_id = version.asset_version_id
            elif decision == "REJECT":
                if before not in {"DRAFT", "REVIEW_REQUIRED", "TAGGING_FAILED"}:
                    raise self._state_conflict("当前知识版本不能拒绝")
                relations = await uow.operations_knowledge.get_version_relations(asset_version_id=asset_version_id)
                await self._documents.withdraw(uow, self._documents.access(context), relations["indexes"])
                version.status = "REJECTED"
                asset.status = (
                    "PUBLISHED" if asset.current_version_id else "REJECTED"
                )
            elif decision == "RETIRE":
                if before != "PUBLISHED":
                    raise self._state_conflict("只有已发布知识可以退役")
                relations = await uow.operations_knowledge.get_version_relations(asset_version_id=asset_version_id)
                await self._documents.withdraw(uow, self._documents.access(context), relations["indexes"])
                version.status = "RETIRED"
                version.retired_at = now
                asset.status = "RETIRED"
                if asset.current_version_id == version.asset_version_id:
                    asset.current_version_id = None
                await queue_case_withdrawals(uow=uow, asset=asset, version=version)
            elif decision == "RETRY":
                if before not in {"FAILED", "PROCESSING", "TAGGING_FAILED"}:
                    raise self._state_conflict("当前知识版本不需要重试")
                relations = await uow.operations_knowledge.get_version_relations(
                    asset_version_id=asset_version_id
                )
                if before == "TAGGING_FAILED":
                    version.status = "TAGGING"
                    version.tag_analysis_json = {"approval": dict(version.tag_analysis_json or {}).get("approval", {})}
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
                await self._documents.retry(self._documents.access(context), asset_version_id)
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
            await self._documents.retry(self._documents.access(context), asset_version_id)
            return await self.reconcile_version(
                domain_id=domain_id, asset_version_id=asset_version_id,
                context=context,
            )
        return await self.get_version(
            domain_id=domain_id, asset_version_id=asset_version_id
        )

    async def preview_version(
        self, *, domain_id, asset_version_id: UUID, context: AuthContext,
    ) -> dict[str, Any]:
        from aiops_agent.application.knowledge_preview import preview_knowledge_version
        return await preview_knowledge_version(
            self, domain_id=domain_id, asset_version_id=asset_version_id, context=context,
            kc_scope={"domain_id": domain_id},
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
        from aiops_agent.application.knowledge_search import search_published_knowledge
        return await search_published_knowledge(
            self, domain_id=domain_id, agent_id=agent_id, request=request,
            context=context, target_id=target_id,
            filters=filters, kc_scope={"domain_id": domain_id},
        )

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

    async def record_reuse_feedback(self, *, domain_id, asset_id, command, context):
        """保留精确版本及人工反馈；不自动发布或提升案例的效果背书。"""
        from platform_core.contracts.aiops.knowledge import KnowledgeReuseFeedback
        command = KnowledgeReuseFeedback.model_validate(command)
        actor = self._actor(context)
        # 脱敏后持久化；原始诊断或执行证据仍由原 Artifact 保存。
        feedback = command.model_dump(mode="json")
        feedback["notes"] = CaseRedactor().text(command.notes)[:1000]
        feedback["trust_level"] = "USER_PROVIDED"
        feedback["verification_method"] = "USER_REPORTED"
        async with self._uow_factory() as uow:
            asset = await uow.operations_knowledge.get_asset(domain_id=domain_id, asset_id=asset_id, for_update=True)
            version = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=command.asset_version_id,
            )
            if (asset is None or asset.asset_kind != "DIAGNOSIS_CASE" or version is None
                or str(version.asset_id) != str(asset_id)):
                raise self._not_found()
            if command.source_run_id is not None:
                run = await uow.runs.get_run_scoped(ops_run_id=command.source_run_id, domain_id=domain_id)
                if run is None or str(run.actor_id) != str(actor):
                    raise self._not_found()
            encoded = json.dumps(feedback, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            previous = await uow.operations_knowledge.get_review(domain_id=domain_id, review_id=command.feedback_id)
            if previous is not None:
                if (str(previous.reviewer_id) != str(actor) or previous.decision != "REUSE_FEEDBACK"
                    or str(previous.asset_version_id) != str(version.asset_version_id)
                    or previous.comment_text != encoded):
                    raise self._state_conflict("反馈标识已用于不同内容")
                return {"feedback_id": str(previous.review_id), "replayed": True, "trust_level": "USER_PROVIDED"}
            review = OperationsKnowledgeReviewEntity(
                review_id=command.feedback_id, asset_version_id=version.asset_version_id,
                decision="REUSE_FEEDBACK", reviewer_id=actor, comment_text=encoded,
                before_status=version.status, after_status=version.status,
            )
            await uow.operations_knowledge.add_review(review)
            await uow.commit()
            return {"feedback_id": str(command.feedback_id), "replayed": False, "trust_level": "USER_PROVIDED"}

    async def recover_case_withdrawal(self) -> bool:
        """暂时性外部失败每五分钟恢复一项，永久失败仍保留待处理。"""
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            changed = await uow.outbox.rearm_case_withdrawal(
                before=now - timedelta(minutes=5), now=now,
            )
            if changed:
                await uow.commit()
            return changed

    async def deliver_case_withdrawal(self, payload: dict, *, caller_service: str):
        """本 App 内部永久撤回与确认回执同事务；重放不再调用外部 KC。"""
        from .knowledge_withdrawal import KnowledgeWithdrawalReceipt
        from ..domain.knowledge_retrieval import RetrievalAccess
        try:
            command = CaseWithdrawalCommand.model_validate(payload)
        except (ValueError, TypeError) as exc:
            raise invalid_withdrawal() from exc
        domain_id = int(command.domain_id)
        access = RetrievalAccess(domain_id, command.tenant_id, caller_service)
        async with self._uow_factory() as uow:
            asset, version = await validate_command_source(uow=uow, command=command, domain_id=domain_id)
            previous = await uow.operations_knowledge.get_review(domain_id=domain_id, review_id=command.request_id)
            if previous is not None:
                if previous.decision != "EXTERNAL_WITHDRAWAL_ACK" or previous.asset_version_id != command.asset_version_id:
                    raise invalid_withdrawal()
                validate_receipt(command, json.loads(previous.comment_text))
                return
            relations = await uow.operations_knowledge.get_version_relations(asset_version_id=command.asset_version_id)
            indexes = [row for row in relations["indexes"] if row.index_ref_id == command.index_ref_id]
            await self._documents.withdraw(uow, access, indexes)
            receipt = KnowledgeWithdrawalReceipt(request_id=command.request_id, withdrawal_id=command.request_id,
                document_id=command.document_id, document_version_id=command.document_version_id,
                ingestion_job_id=command.ingestion_job_id, status="WITHDRAWN", withdrawn_at=_now())
            await uow.operations_knowledge.add_review(OperationsKnowledgeReviewEntity(

                review_id=command.request_id, asset_version_id=command.asset_version_id,
                decision="EXTERNAL_WITHDRAWAL_ACK", reviewer_id=asset.created_by,
                comment_text=receipt.model_dump_json(), before_status=version.status, after_status=version.status))
            await uow.commit()

    async def revalidate_next_case(self) -> bool:
        """后台轮转复核来源；撤回和审计同一事务完成，不改写历史正文。"""
        async with self._uow_factory() as uow:
            now = _now()
            candidate = await uow.operations_knowledge.next_case_index_for_validation(
                before=now - timedelta(minutes=5),
            )
            if candidate is None:
                return False
            index, domain_id = candidate
            # 先用索引所指版本定位其域，再通过 scoped 仓储重新加锁。
            version = await uow.operations_knowledge.get_version(
                domain_id=domain_id, asset_version_id=index.asset_version_id, for_update=True,
            )
            index.last_checked_at = now
            if version is None or version.status != "PUBLISHED":
                await uow.commit()
                return True
            asset = await uow.operations_knowledge.get_asset(
                domain_id=domain_id, asset_id=version.asset_id, for_update=True,
            )
            if asset is None or asset.current_version_id != version.asset_version_id:
                await uow.commit()
                return True
            valid = await validate_case_sources(
                uow=uow, domain_id=domain_id, version_id=version.asset_version_id,
                plan_capture_reader=getattr(self, "_plan_capture_reader", None),
            )
            if valid:
                valid = await validate_case_effect(uow=uow, domain_id=domain_id, profile=version.profile_json)
            if not valid:
                version.status = "REVIEW_REQUIRED"
                asset.status = "REVIEW_REQUIRED"
                asset.current_version_id = None
                await queue_case_withdrawals(uow=uow, asset=asset, version=version)
                await uow.operations_knowledge.add_review(OperationsKnowledgeReviewEntity(
                    asset_version_id=version.asset_version_id, decision="SOURCE_INVALIDATED",
                    reviewer_id=asset.created_by,
                    comment_text="来源、权限、当前判断或验证链已变化；已暂停新检索，须重新核验后审核。",
                    before_status="PUBLISHED", after_status="REVIEW_REQUIRED",
                ))
                relations = await uow.operations_knowledge.get_version_relations(asset_version_id=version.asset_version_id)
                for source in relations.get("sources", ()):
                    if source.source_run_id is None:
                        continue
                    run = await uow.runs.get_run_scoped(ops_run_id=source.source_run_id, domain_id=domain_id)
                    if run is not None:
                        await uow.platform_notifications.emit_case_review_required(asset=asset, version=version, run=run)
                        break
            await uow.commit()
            return True

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
            context=create_service_auth_context(caller_service=caller_service).model_copy(update={
                "app_id": "aiops", "domain_id": str(domain_id)}),
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
    def _knowledge_base_not_found() -> AIOpsApplicationError:
        return AIOpsApplicationError(
            code="AIOPS_KNOWLEDGE_BASE_NOT_FOUND",
            message="运维知识库不存在或当前不可用",
            status_code=404,
        )

    @staticmethod
    def _state_conflict(message: str) -> AIOpsApplicationError:
        return AIOpsApplicationError(
            code="AIOPS_KNOWLEDGE_REVIEW_REQUIRED",
            message=message,
            status_code=409,
        )
