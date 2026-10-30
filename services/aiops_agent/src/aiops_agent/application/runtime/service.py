"""AIOps Run/Task/Artifact/Event 的确定性事务内核。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from aiops_agent.domain.operations.budget import (
    ExecutionBudgetError, ensure_run_resumable, pause_task_budget, resume_task_budget,
    sync_run_budget, task_deadline,
)

from aiops_agent.application.changes.effect_followup import schedule_effect_followup
from aiops_agent.application.knowledge_withdrawal import queue_case_withdrawals
from aiops_agent.application.investigation.future_observation import (
    FOLLOWUP_WORKFLOW, followup_access_error, future_observation_payload, preserve_future_observation, schedule_future_observation,
)

from sqlalchemy.exc import IntegrityError

from aiops_agent.application.display_dictionary import (
    dictionary_for_payload,
    display_metric_dimensions,
    display_metric_name,
    format_metric_value,
)
from aiops_agent.application.errors import (
    AIOpsApplicationError,
    dependency_unavailable,
    resource_not_found,
    state_conflict,
    validation_failed,
)
from aiops_agent.application.investigation.discovery_binding import (
    prior_plan_has_deferred,
)
from aiops_agent.application.investigation.projection import direct_list_result_kind
from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    SignedCursorCodec,
)
from aiops_agent.application.runtime.dashboard import (
    DashboardRunSnapshot,
    DashboardSituationSnapshot,
    DashboardTargetSnapshot,
    parse_finding_payload,
    project_ops_dashboard,
)
from aiops_agent.application.monitoring_snapshot import MonitoringSnapshotBuilder
from aiops_agent.application.target_identity import target_identity_snapshot
from aiops_agent.application.work_items import route_completed_run_work_items
from aiops_agent.application.diagnosis.findings import compile_findings
from aiops_agent.domain.operations import (
    ERROR_CATALOG,
    TASK_TYPE_TO_RUN_PHASE,
    TERMINAL_RUN_STATUSES,
    ensure_run_transition,
    ensure_task_transition,
    normalize_task_type,
)
from aiops_agent.adapters.diagnostic_sources.catalog import (
    MetricCatalog,
    load_metric_catalog,
)
from aiops_agent.domain.states import (
    DomainOpsRunStatus,
    DomainOpsTaskStatus,
)
from aiops_agent.entities import (
    ChangeProposalEntity,
    HitlEntity,
    OpsAnswerBlockEntity,
    OpsAnswerCitationEntity,
    OpsArtifactEntity,
    OpsConversationMessageEntity,
    OpsRunEntity,
    OpsTaskEntity,
    OpsTurnEventEntity,
    OpsTurnEvidenceEntity,
    OpsTurnRunEntity,
    OutboxEntity,
    ReportEntity,
    ReportSourceEntity,
)
from aiops_agent.contracts.evidence import ObservationSet
from aiops_agent.domain.evidence import (
    METRIC_ROLE_AVAILABILITY,
    extract_metric_trend_rows,
)
from aiops_agent.contracts.tool_execution import (
    DbaToolResult,
    is_turn_evidence_outcome,
)
from aiops_agent.contracts.turn_answer import (
    AIOpsTurnResult,
    DbaSufficiencyAssessment,
)
from aiops_agent.contracts.change import (
    ActionPlan,
    ActionVerification,
    ExecutionResultArtifact,
    ProposalOutcome,
)
from aiops_agent.application.changes.proposal_snapshot import (
    build_proposal_snapshot,
    proposal_summary_payload,
)
from aiops_agent.contracts.report import (
    ComparisonPlan,
    ComparisonResult,
    ReportContent,
)
from aiops_agent.application.report_lifecycle import report_input_hash, report_template_for_source
from aiops_agent.application.reporting import (
    ReportTemplate,
    closed_period_window,
    compare_monthly_inspection_summaries,
    is_monthly_backup_recovery_item,
    monthly_inspection_summary,
    normalize_report_source,
    render_pdf,
    report_presentation,
    resolve_historical_report_template,
    resolve_report_type_template,
    template_snapshot,
    validate_template_definition,
)
from aiops_agent.orchestration import (
    BlueprintRegistry,
    build_advisory_verification_blueprint,
    build_database_diagnostic_blueprint,
    build_multi_round_diagnosis_blueprint,
    build_monitor_observe_blueprint,
)
from aiops_agent.orchestration.diagnosis import (
    DIAGNOSIS_PROMPT_IDS,
    AIOpsPromptRegistry,
)
from aiops_agent.orchestration.hitl import normalize_raw_response
from aiops_agent.contracts.hitl import (
    DiagnosticQueryApprovalRequest,
    HitlOutcome,
    ManualSqlRequest,
    UserDiagnosticSubmission,
)
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.workers.handlers import HandlerRegistry
from platform_core.contracts.aiops import (
    AppendOpsTaskProgressCommand,
    ArtifactInput,
    ClaimOpsTaskCommand,
    CompleteOpsTaskCommand,
    CreateOpsRunCommand,
    FailOpsTaskCommand,
    HeartbeatOpsTaskCommand,
    LeasedArtifact,
    OpsRunEventPage,
    OpsRunEventView,
    SuspendOpsTaskCommand,
    TaskLease,
    TaskMutationReceipt,
)
from platform_core.contracts.aiops.events import UnknownEvent
from platform_core.contracts.aiops.internal import (
    DelegationEventPage,
    FinalDiagnosisRef,
    OpsRunReceipt,
    RootDelegationReceipt,
    RootDelegationRequest,
    RootDelegationResult,
)
from platform_core.contracts.aiops.public import (
    DiagnosticQueryApprovalDecision,
    OpsDashboard,
    HitlResponse,
    HitlResult,
    InspectionFirePage,
    InspectionRunSummary,
    InspectionFireSummary,
    InspectionFireView,
    OpsRunResult,
    OpsRunPage,
    OpsRunSummary,
    PendingInputView,
    ReportPage,
    ReportSectionEdit,
    ReportSummary,
    ReportVersionPage,
    ReportVersionSummary,
    ReportView,
    SignalEventSummary,
    SituationMonitoringSourceSummary,
    SituationPage,
    SituationSummary,
    SituationView,
)
from platform_core.contracts.aiops.types import ArtifactRef
from platform_core.identity import uuid7


_AGENT_TURN_WORKFLOWS = frozenset(
    {"CHAT_TURN", "ALERT_DIAGNOSIS", "INSPECTION"}
)


@dataclass(frozen=True, slots=True)
class GeneratedReportReceipt:
    """跨事务边界返回的正式报告生成结果。"""

    report_id: UUID
    status: str
    template_id: str
    report_version: int

    @classmethod
    def from_entity(cls, report: ReportEntity) -> "GeneratedReportReceipt":
        """在 ORM 实体仍绑定会话时冻结响应所需字段。"""
        return cls(
            report_id=report.report_id,
            status=report.status,
            template_id=report.template_id,
            report_version=int(report.report_version),
        )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _human_target_display_name(
    *candidates: object, target_id: object,
) -> str:
    """从候选值中选择可读 Target 名称，拒绝把 UUID 当作名称。"""

    normalized_id = str(target_id or "").strip()
    for candidate in candidates:
        normalized = str(candidate or "").strip()
        if not normalized or (normalized_id and normalized == normalized_id):
            continue
        try:
            UUID(normalized)
        except (ValueError, AttributeError):
            return normalized
    return "目标已删除或无权访问"


def report_content_reference_is_complete(
    report: ReportEntity,
    artifact: OpsArtifactEntity | None,
) -> bool:
    """校验报告与冻结产物的持久化引用，不重算数据库读回的 JSON。"""
    return bool(
        artifact is not None
        and artifact.schema_version == "REPORT_CONTENT.v1"
        and artifact.content_hash
        and artifact.content_hash == report.content_hash
    )


_LATER_PROPOSAL_BLOCK_TYPES = frozenset(
    {
        "REPORT_ARTIFACT_LINKS",
        "HTML_REPORT_LINKS",
        "TABLE",
        "CHART",
        "EVIDENCE_REQUEST",
    }
)


def proposal_insert_block_no(blocks) -> int:
    """新 Proposal 插在方案或已有 Proposal 之后、报告和数据块之前。"""
    later = [
        int(item.block_no)
        for item in blocks
        if str(getattr(item, "block_type", "")) in _LATER_PROPOSAL_BLOCK_TYPES
    ]
    if later:
        return min(later)
    return max(int(item.block_no) for item in blocks) + 1


def _diagnosis_answer_markdown(payload: dict[str, Any]) -> str:
    """把结构化诊断投影为适合聊天的自然 Markdown。"""
    direct = dict(payload.get("direct_answer") or {})
    solution = dict(payload.get("solution") or {})
    answer = str(
        direct.get("answer_text")
        or payload.get("diagnosis_rationale")
        or "现有证据还不足以回答这个问题。"
    )[:6000]
    limitations = [
        str(item)[:1000]
        for item in list(direct.get("limitations") or ())[:5]
        if item and str(item) not in answer
    ]
    if limitations:
        answer += "\n\n" + "\n".join(f"> {item}" for item in limitations)
    if not direct:
        recommendations = [
            str(item)[:1000]
            for item in (
                *list(solution.get("immediate_mitigations") or ())[:5],
                *list(solution.get("long_term_remediations") or ())[:3],
            )
            if item
        ]
        if recommendations:
            answer += "\n\n接下来可以这样处理：\n\n" + "\n".join(
                f"- {item}" for item in recommendations
            )
    return answer[:16000]


def _runtime_error(
    code: str,
    message: str,
    *,
    status_code: int = 409,
    retryable: bool = False,
) -> AIOpsApplicationError:
    return AIOpsApplicationError(
        code=code,
        message=message,
        status_code=status_code,
        retryable=retryable,
    )


def _replan_budget_exhausted(turn: object, *, now: datetime, max_replans: int, max_seconds: int,
    excluded_wait_milliseconds: int = 0) -> bool:
    started_at = getattr(turn, "started_at", None) or getattr(turn, "created_at", None)
    if started_at is not None and started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    return (
        int(getattr(turn, "current_plan_revision", 1) or 1) > max_replans
        or (started_at is not None and now >= started_at + timedelta(seconds=max_seconds,
            milliseconds=max(0, int(excluded_wait_milliseconds))))
    )


def _assessment_progress(previous: dict[str, Any], current: DbaSufficiencyAssessment) -> bool:
    """比较证据形态和已解决缺口，不能把新 artifact 引用当成新信息。"""
    def evidence_keys(items):
        result = set()
        for item in items:
            value = item if isinstance(item, dict) else item.model_dump(mode="json")
            columns = tuple(
                str(column.get("name") or "").lower()
                for column in value.get("columns", ())
                if isinstance(column, dict)
            )
            identity_columns = {
                "query_id", "sql_id", "digest", "database_oid", "user_oid",
                "schema_name", "table_name", "relation_name", "relation_oid",
                "index_name", "instance_id", "session_id", "pid",
                "snapshot_id", "begin_snapshot_id", "end_snapshot_id",
                "period_start", "period_end", "window_start", "window_end",
            }
            positions = tuple(
                index for index, column in enumerate(columns)
                if column in identity_columns
            )
            identities = tuple(sorted({
                tuple(str(row[index]) for index in positions if index < len(row))
                for row in (value.get("rows") or ())[:500]
                if positions and isinstance(row, (list, tuple))
            }))
            semantics = str(value.get("measurement_semantics") or "")
            # 时点采样允许同一工具在不同时间提供新的观测。
            sampled_at = value.get("captured_at") if semantics == "CURRENT_ACTIVITY" else None
            result.add((
                str(value.get("source_id") or ""),
                str(value.get("tool_id") or ""),
                semantics, columns, identities, bool(value.get("row_count")),
                bool(value.get("truncated")), sampled_at,
            ))
        return result

    def gap_keys(items):
        return {
            (str(value.get("source_id") or ""), str(value.get("code") or ""))
            for item in items
            for value in [item if isinstance(item, dict) else item.model_dump(mode="json")]
        }

    before_facts = evidence_keys(previous.get("evidence") or ())
    after_facts = evidence_keys(current.evidence)
    before_gaps = gap_keys(previous.get("gaps") or ())
    after_gaps = gap_keys(current.gaps)
    return bool(after_facts - before_facts or before_gaps - after_gaps)


class AIOpsRuntimeService:
    """所有运行状态变化均在一个显式 UoW 中提交。"""

    def __init__(
        self,
        *,
        uow_factory,
        blueprint_registry: BlueprintRegistry,
        handler_registry: HandlerRegistry,
        max_tasks_per_run: int = 64,
        default_run_timeout_seconds: int = 3600,
        metric_catalog: MetricCatalog | None = None,
        default_observation_window_seconds: int = 3600,
        max_monitor_response_bytes: int = 5 * 1024 * 1024,
        diagnostic_registry: DiagnosticRegistry | None = None,
        diagnosis_config=None,
        diagnosis_prompt_registry: AIOpsPromptRegistry | None = None,
        agent_catalog=None,
        cursor_codec: SignedCursorCodec | None = None,
        monitoring_snapshot_builder: MonitoringSnapshotBuilder | None = None,
        plan_capture_reader=None,
    ):
        self._uow_factory = uow_factory
        self._plan_capture_reader = plan_capture_reader
        self._blueprints = blueprint_registry
        self._handlers = handler_registry
        self._max_tasks = max_tasks_per_run
        self._default_run_timeout = default_run_timeout_seconds
        self._metric_catalog = metric_catalog or load_metric_catalog()
        self._default_observation_window = (
            default_observation_window_seconds
        )
        self._max_monitor_response_bytes = max_monitor_response_bytes
        self._monitoring_snapshot_builder = (
            monitoring_snapshot_builder
            or MonitoringSnapshotBuilder(
                metric_catalog=self._metric_catalog,
                default_window_seconds=self._default_observation_window,
                max_response_bytes=self._max_monitor_response_bytes,
            )
        )
        self._diagnostic_registry = diagnostic_registry
        self._diagnosis_config = diagnosis_config
        self._diagnosis_prompts = diagnosis_prompt_registry
        self._agent_catalog = agent_catalog
        self._cursor_codec = cursor_codec

    async def create_run(
        self, command: CreateOpsRunCommand
    ) -> OpsRunReceipt:
        trace_id = str(
            command.client_metadata.get("trace_id", command.command_id)
        )
        diagnosis_model = None
        private_agent_binding = None
        if command.blueprint_id == "diagnosis.root-cause":
            if self._agent_catalog is None:
                raise dependency_unavailable(
                    "Agent Runtime 模型解析器尚未配置"
                )
            diagnosis_model = (
                await self._agent_catalog.resolve_diagnosis_model(
                    agent_id=command.agent_id,
                    domain_id=command.domain_id,
                    trace_id=trace_id,
                )
            )
            private_agent_binding = (
                await self._agent_catalog.resolve_runtime_binding(
                    agent_id=command.agent_id,
                    domain_id=command.domain_id,
                    target_id=command.target_id,
                )
            )
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            deadline = command.deadline or (
                now + timedelta(seconds=self._default_run_timeout)
            )
            if deadline <= now:
                raise validation_failed("Run 截止时间必须晚于当前时间")
            existing = await uow.runs.get_by_idempotency(
                target_id=command.target_id,
                trigger_type=str(command.trigger_type),
                actor_id=command.actor_id,
                idempotency_key=command.idempotency_key,
            )
            if existing is not None:
                if (
                    existing.agent_id != command.agent_id
                    or existing.original_request != command.input
                ):
                    raise _runtime_error(
                        "OPS_IDEMPOTENCY_CONFLICT",
                        "相同 Idempotency-Key 对应的 Run 请求不同",
                    )
                cursor = await uow.runs.latest_event_sequence(
                    ops_run_id=existing.ops_run_id
                )
                return self._run_receipt(existing, cursor)

            target = await uow.targets.get_scoped(
                target_id=command.target_id,
                domain_id=command.domain_id,
                lock=True,
            )
            if target is None:
                raise resource_not_found("Target")
            agent = await uow.agents.get(
                domain_id=command.domain_id,
                agent_id=command.agent_id,
            )
            if (
                agent is None
                or agent.status != "ACTIVE"
                or agent.current_version_id is None
            ):
                raise resource_not_found("Active AIOps Agent")
            selected_agent_version_id = (
                command.expected_agent_version_id or agent.current_version_id
            )
            agent_version = await uow.agents.version(
                agent_id=agent.agent_id,
                agent_version_id=selected_agent_version_id,
            )
            if agent_version is None:
                raise resource_not_found("Agent Version")
            if command.expected_agent_version_id is not None:
                private_agent_binding = await uow.agents.get_version_binding(
                    domain_id=command.domain_id,
                    agent_id=command.agent_id,
                    agent_version_id=command.expected_agent_version_id,
                    target_id=command.target_id,
                )
                if private_agent_binding is None:
                    raise resource_not_found("Frozen AIOps Agent Binding")
            source_context = None
            requested_source_run_id = command.client_metadata.get(
                "source_run_id"
            )
            if requested_source_run_id:
                try:
                    source_run_id = UUID(str(requested_source_run_id))
                except (TypeError, ValueError):
                    raise validation_failed("续聊来源 Run ID 无效") from None
                source_run = await uow.runs.get_run_scoped(
                    ops_run_id=source_run_id,
                    domain_id=command.domain_id,
                )
                source_artifact = (
                    await uow.runs.get_artifact(
                        artifact_id=source_run.final_artifact_id
                    )
                    if source_run is not None
                    and source_run.final_artifact_id is not None
                    else None
                )
                if (
                    str(command.trigger_type) != "CHAT"
                    or source_run is None
                    or source_run.trigger_type not in {"ALERT", "SCHEDULE"}
                    or source_run.target_id != target.target_id
                    or source_artifact is None
                ):
                    raise validation_failed("告警或巡检续聊来源关系无效")
                public_keys = (
                    "summary",
                    "root_cause",
                    "root_cause_grade",
                    "diagnosis_rationale",
                    "facts",
                    "findings",
                    "solution",
                    "recommendations",
                    "gaps",
                )
                public_result = {
                    key: source_artifact.payload_json[key]
                    for key in public_keys
                    if key in (source_artifact.payload_json or {})
                }
                source_context = {
                    "source_run_id": str(source_run.ops_run_id),
                    "source_trigger_type": source_run.trigger_type,
                    "source_root_cause_grade": self._artifact_root_cause_grade(
                        source_artifact
                    ),
                    "source_artifact_schema": source_artifact.schema_version,
                    "source_result": json.dumps(
                        public_result,
                        ensure_ascii=False,
                        default=str,
                    )[:1000],
                }
            binding = private_agent_binding
            if binding is None:
                binding = await uow.targets.get_agent_binding(
                    target_id=command.target_id,
                    agent_id=command.agent_id,
                    domain_id=command.domain_id,
                    lock=True,
                )
                if binding is None or binding.status != "ACTIVE":
                    raise resource_not_found("Active Agent Binding")
            elif (
                binding.target_id is not None
                and binding.target_id != command.target_id
            ):
                raise validation_failed("Agent 仅允许诊断其已选择的数据库直连 Target")
            policy = None
            configured_policy_status = None
            if binding.policy_id is not None:
                configured_policy = await uow.policies.get_scoped(
                    policy_id=binding.policy_id,
                    domain_id=command.domain_id,
                    lock=True,
                )
                if configured_policy is None:
                    raise state_conflict("Agent Binding 引用的策略不存在")
                configured_policy_status = configured_policy.status
                if configured_policy.status == "ACTIVE":
                    policy = configured_policy

            target_snapshot = {
                **target_identity_snapshot(target),
                "domain_id": int(target.domain_id),
                "display_name": target.display_name,
                "version_code": target.version_code,
                "environment": target.environment,
                "db_role": target.db_role,
                "deployment_type": target.deployment_type or "SINGLE_INSTANCE",
                "rac_node_number": target.rac_node_number,
                "status": getattr(target, "status", "ENABLED"),
                "connectivity_status": target.connectivity_status,
                "database_endpoint_configured": bool(target.endpoint_json),
                "diagnostic_secret_configured": bool(
                    target.diagnostic_credential_id
                ),
                "execution_secret_configured": bool(
                    target.execution_credential_id
                ),
                "readonly_connection_enabled": bool(
                    target.readonly_connection_enabled
                ),
                "controlled_change_enabled": bool(
                    target.controlled_change_enabled
                ),
                "capabilities": dict(target.capabilities_json or {}),
            }
            binding_snapshot = {
                "binding_id": str(binding.binding_id),
                "agent_id": str(binding.agent_id),
                "allow_mutation": bool(binding.allow_mutation),
                "policy_id": (
                    str(binding.policy_id)
                    if binding.policy_id is not None
                    else None
                ),
                "policy_status": configured_policy_status,
                "allowed_actions": list(
                    binding.allowed_actions_json or []
                ),
                "object_scopes": dict(
                    getattr(binding, "object_scopes_json", {}) or {}
                ),
                "max_daily_executions": getattr(
                    binding, "max_daily_executions", None
                ),
                "row_version": int(binding.row_version),
            }
            policy_snapshot = (
                {
                    "policy_id": str(policy.policy_id),
                    "policy_key": policy.policy_key,
                    "version_no": int(policy.version_no),
                    "policy_hash": policy.policy_hash,
                    "rules": dict(policy.rules_json),
                    "row_version": int(policy.row_version),
                }
                if policy is not None
                else {}
            )
            monitoring_snapshot: dict[str, Any] | None = None
            database_diagnostic_snapshot: dict[str, Any] | None = None
            if command.blueprint_id == "monitor.observe-report":
                (
                    blueprint,
                    monitoring_snapshot,
                ) = await self._monitor_blueprint_snapshot(
                    uow=uow,
                    command=command,
                    target=target,
                    now=now,
                    allowed_source_ids=getattr(
                        binding, "diagnostic_source_ids", None
                    ),
                )
            elif command.blueprint_id == "database.diagnostic-baseline":
                (
                    blueprint,
                    database_diagnostic_snapshot,
                ) = self._database_diagnostic_blueprint_snapshot(
                    command=command,
                    target=target,
                    binding=binding,
                    policy=policy,
                )
            elif command.blueprint_id == "diagnosis.root-cause":
                (
                    _,
                    monitoring_snapshot,
                ) = await self._monitor_blueprint_snapshot(
                    uow=uow,
                    command=command,
                    target=target,
                    now=now,
                    allowed_source_ids=getattr(
                        binding, "diagnostic_source_ids", None
                    ),
                )
                (
                    _,
                    database_diagnostic_snapshot,
                ) = self._database_diagnostic_blueprint_snapshot(
                    command=command,
                    target=target,
                    binding=binding,
                    policy=policy,
                    requested_tool_ids=(
                        "db.instance.identity",
                        "db.session.active",
                        "db.session.blocking_chain",
                        "db.storage.capacity",
                        "db.transaction.long_running",
                        "db.replication.status",
                    ),
                )
                baseline_tools = {
                    "db.instance.identity",
                    "db.session.active",
                    "db.session.blocking_chain",
                    "db.storage.capacity",
                }
                if self._diagnosis_config is None:
                    raise validation_failed("诊断编排尚未配置")
                blueprint = build_multi_round_diagnosis_blueprint(
                    binding_ids=tuple(
                        monitoring_snapshot["observation_binding_ids"]
                    ),
                    log_binding_ids=tuple(
                        monitoring_snapshot["log_binding_ids"]
                    ),
                    tool_ids=tuple(
                        item["tool_id"]
                        for item in database_diagnostic_snapshot["tools"]
                        if item["tool_id"] in baseline_tools
                    ),
                    max_rounds=int(self._diagnosis_config.max_rounds),
                )
            elif command.blueprint_id == "change.advisory-verify":
                requested_verification = dict(
                    command.client_metadata.get(
                        "advisory_verification", {}
                    )
                )
                required = {
                    "proposal_id",
                    "source_run_id",
                    "result_artifact_id",
                }
                if not required.issubset(requested_verification):
                    raise validation_failed("Advisory 验证上下文不完整")
                try:
                    proposal_id = UUID(
                        str(requested_verification["proposal_id"])
                    )
                    result_artifact_id = UUID(
                        str(
                            requested_verification[
                                "result_artifact_id"
                            ]
                        )
                    )
                except (TypeError, ValueError):
                    raise validation_failed(
                        "Advisory 验证来源标识无效"
                    ) from None
                source_proposal = await uow.changes.get_proposal(
                    proposal_id=proposal_id
                )
                source_result = await uow.runs.get_artifact(
                    artifact_id=result_artifact_id
                )
                source_run = (
                    await uow.runs.get_run(
                        ops_run_id=source_proposal.ops_run_id
                    )
                    if source_proposal is not None
                    else None
                )
                snapshot_artifact = (
                    await uow.runs.get_artifact(
                        artifact_id=source_proposal.snapshot_artifact_id
                    )
                    if source_proposal is not None
                    else None
                )
                if (
                    source_proposal is None
                    or source_result is None
                    or source_run is None
                    or snapshot_artifact is None
                    or source_proposal.target_id != target.target_id
                    or source_result.ops_run_id != source_run.ops_run_id
                    or source_result.schema_version
                    not in {
                        "USER_PROVIDED_ACTION_RESULT.v1",
                        "EXECUTION_RESULT.v1",
                    }
                    or str(source_run.ops_run_id)
                    != str(requested_verification["source_run_id"])
                    or source_run.agent_id != command.agent_id
                    or source_run.actor_id != command.actor_id
                ):
                    raise validation_failed(
                        "Advisory 验证来源关系不可信"
                    )
                proposal_outcome = ProposalOutcome.model_validate(
                    snapshot_artifact.payload_json
                )
                proposal_snapshot = proposal_outcome.proposal
                result_payload = dict(source_result.payload_json or {})
                source_status = result_payload.get("status")
                if (
                    proposal_snapshot is None
                    or proposal_snapshot.proposal_id
                    != str(proposal_id)
                    or result_payload.get("proposal_id")
                    != str(proposal_id)
                    or source_status
                    not in {"EXECUTED", "SUCCEEDED", "UNKNOWN"}
                ):
                    raise validation_failed(
                        "Advisory 验证来源内容无效"
                    )
                verification = {
                    "proposal_id": str(proposal_id),
                    "source_run_id": str(source_run.ops_run_id),
                    "result_artifact_id": str(result_artifact_id),
                    "action_template_id": (
                        proposal_snapshot.action_template_id
                    ),
                    "canonical_parameters": dict(
                        proposal_snapshot.canonical_parameters
                    ),
                    "verification_tool_refs": tuple(
                        proposal_snapshot.verification_plan
                    ),
                    "source_result_status": source_status,
                }
                requested_tools = tuple(
                    dict.fromkeys(
                        (
                            "db.instance.identity",
                            *verification["verification_tool_refs"],
                        )
                    )
                )
                (
                    _,
                    database_diagnostic_snapshot,
                ) = self._database_diagnostic_blueprint_snapshot(
                    command=command,
                    target=target,
                    binding=binding,
                    policy=policy,
                    requested_tool_ids=requested_tools,
                )
                if proposal_snapshot.action_template_id in {
                    "db.index.rebuild",
                    "db.index.partition.rebuild",
                    "db.index.coalesce",
                }:
                    object_ref = dict(
                        proposal_snapshot.canonical_parameters["index_ref"]
                    )
                    for tool in database_diagnostic_snapshot["tools"]:
                        if tool["tool_id"] == "db.index.health":
                            tool["parameters"] = {
                                "schema_name": object_ref["schema"],
                                "index_name": object_ref["object_name"],
                            }
                        elif tool["tool_id"] == "db.index.coalesce_candidate":
                            tool["parameters"] = {
                                "schema_name": object_ref["schema"],
                                "index_name": object_ref["object_name"],
                            }
                        elif tool["tool_id"] == "db.index.partition.health":
                            partition_name = (
                                proposal_snapshot.canonical_parameters[
                                    "partition_name"
                                ]
                            )
                            tool["parameters"] = {
                                "schema_name": object_ref["schema"],
                                "index_name": object_ref["object_name"],
                                "partition_name": partition_name,
                            }
                elif proposal_snapshot.action_template_id in {
                    "db.storage.datafile.resize",
                    "db.storage.tempfile.resize",
                    "db.storage.datafile.autoextend",
                    "db.storage.tempfile.autoextend",
                }:
                    parameters = proposal_snapshot.canonical_parameters
                    resize = proposal_snapshot.action_template_id.endswith(
                        ".resize"
                    )
                    for tool in database_diagnostic_snapshot["tools"]:
                        if str(tool["tool_id"]).endswith(".action_state"):
                            tool["parameters"] = {
                                "file_name": parameters["file_name"],
                                "new_size_mb": (
                                    parameters["new_size_mb"] if resize else 0
                                ),
                                "next_mb": (
                                    0 if resize else parameters["next_mb"]
                                ),
                                "max_size_mb": (
                                    0 if resize else parameters["max_size_mb"]
                                ),
                            }
                elif proposal_snapshot.action_template_id == "db.parameter.set":
                    parameters = proposal_snapshot.canonical_parameters
                    for tool in database_diagnostic_snapshot["tools"]:
                        if tool["tool_id"] == "db.parameter.dynamic_state":
                            tool["parameters"] = {
                                "parameter_name": parameters["parameter_name"],
                                "parameter_value": parameters["parameter_value"],
                            }
                elif (
                    proposal_snapshot.action_template_id
                    == "db.resource_manager.plan.switch"
                ):
                    parameters = proposal_snapshot.canonical_parameters
                    for tool in database_diagnostic_snapshot["tools"]:
                        if tool["tool_id"] == "db.resource_manager.plan_state":
                            tool["parameters"] = {
                                "resource_plan_name": parameters[
                                    "resource_plan_name"
                                ]
                            }
                elif proposal_snapshot.action_template_id in {
                    "db.user.privilege.grant",
                    "db.user.privilege.revoke",
                }:
                    parameters = proposal_snapshot.canonical_parameters
                    for tool in database_diagnostic_snapshot["tools"]:
                        if tool["tool_id"] == "db.user.system_privilege_state":
                            tool["parameters"] = {
                                "grantee_name": parameters["grantee_name"],
                                "privilege": parameters["privilege"],
                            }
                        elif tool["tool_id"] == "db.user.object_privilege_state":
                            object_ref = dict(parameters["object_ref"])
                            tool["parameters"] = {
                                "schema_name": object_ref["schema"],
                                "object_name": object_ref["object_name"],
                                "object_type": object_ref["object_type"],
                                "grantee_name": parameters["grantee_name"],
                                "privilege": parameters["privilege"],
                            }
                elif (
                    proposal_snapshot.action_template_id
                    == "db.session.cancel_sql"
                ):
                    parameters = proposal_snapshot.canonical_parameters
                    for tool in database_diagnostic_snapshot["tools"]:
                        if tool["tool_id"] == "db.session.current_sql":
                            tool["parameters"] = {
                                "instance_id": parameters["instance_id"],
                                "session_id": parameters["session_id"],
                            }
                elif proposal_snapshot.action_template_id == "db.object.compile":
                    parameters = proposal_snapshot.canonical_parameters
                    object_ref = dict(parameters["object_ref"])
                    for tool in database_diagnostic_snapshot["tools"]:
                        if tool["tool_id"] == "db.object.status":
                            tool["parameters"] = {
                                "schema_name": object_ref["schema"],
                                "object_name": object_ref["object_name"],
                                "object_type": parameters["object_type"],
                            }
                elif proposal_snapshot.action_template_id in {
                    "db.statistics.gather",
                    "db.statistics.lock",
                    "db.statistics.unlock",
                }:
                    table_ref = dict(
                        proposal_snapshot.canonical_parameters["table_ref"]
                    )
                    for tool in database_diagnostic_snapshot["tools"]:
                        if str(tool["tool_id"]).startswith(
                            "db.table.statistics"
                        ):
                            tool["parameters"] = {
                                "schema_name": table_ref["schema"],
                                "table_name": table_ref["object_name"],
                            }
                elif proposal_snapshot.action_template_id in {
                    "db.scheduler.job.run",
                    "db.scheduler.job.enable",
                    "db.scheduler.job.disable",
                    "db.scheduler.job.stop",
                }:
                    job_ref = dict(
                        proposal_snapshot.canonical_parameters["job_ref"]
                    )
                    for tool in database_diagnostic_snapshot["tools"]:
                        if str(tool["tool_id"]).startswith(
                            "db.scheduler.job."
                        ):
                            tool["parameters"] = {
                                "schema_name": job_ref["schema"],
                                "job_name": job_ref["object_name"],
                            }
                elif proposal_snapshot.action_template_id in {
                    "db.user.lock",
                    "db.user.unlock",
                    "db.user.password.expire",
                }:
                    user_ref = dict(
                        proposal_snapshot.canonical_parameters["user_ref"]
                    )
                    for tool in database_diagnostic_snapshot["tools"]:
                        if str(tool["tool_id"]).startswith("db.user."):
                            tool["parameters"] = {
                                "username": user_ref["object_name"]
                            }
                verification["initial_gap_codes"] = tuple(
                    item["code"]
                    for item in database_diagnostic_snapshot[
                        "initial_gaps"
                    ]
                )
                blueprint = build_advisory_verification_blueprint(
                    tuple(
                        item["tool_id"]
                        for item in database_diagnostic_snapshot["tools"]
                    )
                )
            else:
                blueprint = self._blueprints.resolve(
                    command.blueprint_id, command.blueprint_version
                )
            self._blueprints.validate(
                blueprint, max_tasks=self._max_tasks
            )
            run_id = uuid7()
            mutation_unavailable_reasons = (
                ("BINDING_MUTATION_DISABLED",)
                if not bool(binding.allow_mutation)
                else tuple(
                    reason
                    for unavailable, reason in (
                        (
                            str(command.trigger_type) != "CHAT",
                            "AUTONOMOUS_RUN_ADVISORY_ONLY",
                        ),
                        (
                            not self._management.agent_execution_enabled,
                            "DEPLOYMENT_MUTATION_DISABLED",
                        ),
                        (
                            policy is not None
                            and dict(policy.rules_json or {}).get("allow_agent_execution") is False,
                            "AGENT_MUTATION_DISABLED",
                        ),
                        (
                            not bool(target.controlled_change_enabled),
                            "TARGET_CONTROLLED_CHANGE_DISABLED",
                        ),
                        (
                            not bool(target.execution_credential_id),
                            "EXECUTION_SECRET_MISSING",
                        ),
                        (
                            policy is None,
                            (
                                "POLICY_NOT_ACTIVE"
                                if configured_policy_status is not None
                                else "POLICY_MISSING"
                            ),
                        ),
                        (
                            not bool(binding.allowed_actions_json),
                            "ALLOWED_ACTIONS_EMPTY",
                        ),
                    )
                    if unavailable
                )
            )
            plan_snapshot = {
                "blueprint": {
                    "id": blueprint.blueprint_id,
                    "version": blueprint.version,
                    "final_task_key": blueprint.final_task_key,
                },
                "target": target_snapshot,
                "binding": binding_snapshot,
                "trigger": {
                    "type": str(command.trigger_type),
                    "session_id": command.session_id,
                },
                "client_metadata": dict(command.client_metadata),
                "effective_capabilities": {
                    "monitor_read": bool(
                        monitoring_snapshot
                        and monitoring_snapshot.get("bindings")
                    ),
                    "database_read": bool(
                        database_diagnostic_snapshot
                        and database_diagnostic_snapshot.get(
                            "automatic_access_enabled"
                        )
                    ),
                    "manual_database_read": bool(
                        str(command.trigger_type) == "CHAT"
                        and database_diagnostic_snapshot
                        and database_diagnostic_snapshot.get("tools")
                    ),
                    "mutation_requested": bool(binding.allow_mutation),
                    "mutation_execute": not mutation_unavailable_reasons,
                    "mutation_unavailable_reasons": list(
                        mutation_unavailable_reasons
                    ),
                },
            }
            if monitoring_snapshot is not None:
                plan_snapshot["monitoring"] = monitoring_snapshot
            if command.blueprint_id == "database.diagnostic-baseline":
                plan_snapshot["database_diagnostics"] = (
                    database_diagnostic_snapshot
                )
            if command.blueprint_id == "diagnosis.root-cause":
                plan_snapshot["database_diagnostics"] = (
                    database_diagnostic_snapshot
                )
                plan_snapshot["diagnosis"] = await self._diagnosis_snapshot(
                    command=command,
                    target=target,
                    policy_snapshot=policy_snapshot,
                    monitoring_snapshot=monitoring_snapshot,
                    model_snapshot=diagnosis_model,
                    source_context=source_context,
                )
            if command.blueprint_id == "change.advisory-verify":
                plan_snapshot["database_diagnostics"] = (
                    database_diagnostic_snapshot
                )
                plan_snapshot["advisory_verification"] = verification
            try:
                run = await uow.runs.add_run(
                    OpsRunEntity(
                        ops_run_id=run_id,
                        domain_id=command.domain_id,
                        target_id=target.target_id,
                        agent_id=command.agent_id,
                        agent_version_id=agent_version.agent_version_id,
                        parent_agent_run_id=command.parent_agent_run_id,
                        parent_delegation_id=command.parent_delegation_id,
                        trigger_type=str(command.trigger_type),
                        trigger_signal_event_id=command.trigger_signal_event_id,
                        situation_id=command.situation_id,
                        interaction_mode=(
                            "INTERACTIVE"
                            if str(command.trigger_type) == "CHAT"
                            else "AUTONOMOUS"
                        ),
                        workflow_kind=self._workflow_kind(command),
                        inspection_fire_id=command.inspection_fire_id,
                        source_proposal_id=(
                            UUID(verification["proposal_id"])
                            if command.blueprint_id
                            == "change.advisory-verify"
                            else None
                        ),
                        source_result_artifact_id=(
                            UUID(verification["result_artifact_id"])
                            if command.blueprint_id
                            == "change.advisory-verify"
                            else None
                        ),
                        actor_id=command.actor_id,
                        original_request=command.input,
                        idempotency_key=command.idempotency_key,
                        status=DomainOpsRunStatus.CREATED.value,
                        plan_snapshot_json=plan_snapshot,
                        policy_snapshot_json=policy_snapshot,
                        deadline_at=deadline,
                        trace_id=trace_id,
                    )
                )
            except IntegrityError:
                await uow.rollback()
                return await self._replay_concurrent_create(command)
            if command.client_metadata.get("multi_resource_child") is True:
                try:
                    parent_turn_id = UUID(
                        str(command.client_metadata["parent_turn_id"])
                    )
                    resource_sequence = int(
                        command.client_metadata["resource_sequence"]
                    )
                except (KeyError, TypeError, ValueError):
                    raise validation_failed(
                        "多资源诊断子 Run 缺少有效父 Turn 或资源序号"
                    ) from None
                if resource_sequence < 1 or resource_sequence > 999:
                    raise validation_failed("多资源诊断资源序号无效")
                await uow.turns.add_run(
                    OpsTurnRunEntity(
                        turn_run_id=uuid7(),
                        turn_id=parent_turn_id,
                        ops_run_id=run_id,
                        purpose=f"RESOURCE_{resource_sequence:03d}",
                        sequence_no=resource_sequence,
                    )
                )
            task_ids = {
                spec.task_key: uuid7() for spec in blueprint.tasks
            }
            tasks = [
                OpsTaskEntity(
                    ops_task_id=task_ids[spec.task_key],
                    ops_run_id=run_id,
                    parent_task_id=(
                        task_ids[spec.depends_on[0]]
                        if spec.depends_on
                        else None
                    ),
                    task_key=spec.task_key,
                    task_type=normalize_task_type(spec.task_type),
                    handler_id=spec.handler_id,
                    handler_version=spec.handler_version,
                    input_schema_version=spec.input_schema_version,
                    output_schema_version=spec.output_schema_version,
                    depends_on_json=list(spec.depends_on),
                    input_artifacts_json=list(spec.input_artifact_keys),
                    status=(
                        DomainOpsTaskStatus.READY.value
                        if not spec.depends_on
                        else DomainOpsTaskStatus.PENDING.value
                    ),
                    priority=spec.priority,
                    available_at=now,
                    max_attempts=spec.max_attempts,
                    timeout_seconds=spec.timeout_seconds,
                )
                for spec in blueprint.tasks
            ]
            await uow.runs.add_tasks(tasks)
            event = await uow.runs.append_event(
                ops_run_id=run_id,
                event_type="run.status",
                event_key=f"run:{run_id}:created",
                visibility="USER",
                payload_json={
                    "status": DomainOpsRunStatus.CREATED.value,
                    "trace_id": trace_id,
                },
            )
            await self._add_outbox(
                uow,
                aggregate_id=run_id,
                event_type="OPS_RUN_CREATED",
                idempotency_key=f"run:{run_id}:created",
                payload={
                    "ops_run_id": str(run_id),
                    "ready_task_count": 1,
                },
                trace_id=trace_id,
                now=now,
            )
            if str(command.trigger_type) == "ALERT":
                assert uow.platform_notifications is not None
                await uow.platform_notifications.emit_run_started(
                    run=run,
                    target_name=target.display_name,
                )
            await uow.commit()
            return self._run_receipt(run, int(event.sequence_no))

    async def create_delegated_run(
        self,
        *,
        request: RootDelegationRequest,
        domain_id: int,
        actor_id: str,
        trace_id: str,
    ) -> RootDelegationReceipt:
        """用稳定 Delegation ID 幂等创建 Root 的 AIOps 子 Run。"""
        if str(domain_id) != request.domain_id:
            raise validation_failed("Delegation Domain 与可信上下文不一致")
        receipt = await self.create_run(
            CreateOpsRunCommand(
                command_id=uuid7(),
                idempotency_key=f"delegation:{request.delegation_id}",
                domain_id=domain_id,
                actor_id=actor_id,
                agent_id=request.agent_id,
                target_id=request.target_id,
                trigger_type="CHAT",
                input=request.user_intent,
                parent_agent_run_id=request.parent_agent_run_id,
                parent_delegation_id=request.delegation_id,
                deadline=request.deadline,
                blueprint_id="diagnosis.root-cause",
                blueprint_version="1",
                client_metadata={
                    "trace_id": trace_id,
                    "caller_mode": "ROOT_DELEGATION",
                    "delegation_id": str(request.delegation_id),
                },
            )
        )
        return RootDelegationReceipt(
            delegation_id=request.delegation_id,
            ops_run_id=receipt.ops_run_id,
            status=receipt.status,
            child_event_cursor=receipt.event_cursor,
        )

    async def list_delegation_events(
        self,
        *,
        delegation_id: UUID,
        domain_id: int,
        after_sequence: int,
        limit: int,
    ) -> DelegationEventPage:
        """只返回 Root 可安全投影的 USER 事件，不暴露内部 Task 内容。"""
        async with self._uow_factory() as uow:
            run = await uow.runs.get_by_parent_delegation_scoped(
                parent_delegation_id=delegation_id,
                domain_id=domain_id,
            )
            if run is None:
                raise resource_not_found("Delegation")
            latest = await uow.runs.latest_event_sequence(
                ops_run_id=run.ops_run_id
            )
            if after_sequence > latest:
                raise _runtime_error(
                    "OPS_EVENT_CURSOR_INVALID",
                    "Delegation 事件游标大于当前最新序号",
                )
            events = await uow.runs.list_events_after(
                ops_run_id=run.ops_run_id,
                after_sequence=after_sequence,
                visibility="USER",
                limit=limit,
            )
            safe_events = tuple(
                self._delegation_event(run, item) for item in events
            )
            next_sequence = (
                int(events[-1].sequence_no)
                if events
                else after_sequence
            )
            return DelegationEventPage(
                delegation_id=delegation_id,
                events=safe_events,
                next_sequence=next_sequence,
                terminal=DomainOpsRunStatus(run.status)
                in TERMINAL_RUN_STATUSES,
            )

    async def get_delegation_result(
        self,
        *,
        delegation_id: UUID,
        domain_id: int,
    ) -> RootDelegationResult:
        """读取终态子 Run，并生成不含命令和原始 SQL 的受限结果。"""
        async with self._uow_factory() as uow:
            run = await uow.runs.get_by_parent_delegation_scoped(
                parent_delegation_id=delegation_id,
                domain_id=domain_id,
            )
            if run is None:
                raise resource_not_found("Delegation")
            if DomainOpsRunStatus(run.status) not in TERMINAL_RUN_STATUSES:
                raise _runtime_error(
                    "OPS_DELEGATION_RESULT_NOT_READY",
                    "Delegation 子 Run 尚未进入终态",
                )
            artifact = (
                await uow.runs.get_artifact(
                    artifact_id=run.final_artifact_id
                )
                if run.final_artifact_id is not None
                else None
            )
            diagnosis = None
            safe_summary = self._delegation_safe_summary(run, artifact)
            if artifact is not None:
                diagnosis = FinalDiagnosisRef(
                    artifact=ArtifactRef(
                        artifact_id=artifact.artifact_id,
                        artifact_type=artifact.artifact_type,
                        schema_version=artifact.schema_version,
                        content_hash=artifact.content_hash,
                    ),
                    root_cause_grade=self._artifact_root_cause_grade(
                        artifact
                    ),
                )
            return RootDelegationResult(
                delegation_id=delegation_id,
                ops_run_id=run.ops_run_id,
                status=run.status,
                diagnosis=diagnosis,
                safe_summary=safe_summary,
            )

    async def cancel_delegation(
        self,
        *,
        delegation_id: UUID,
        domain_id: int,
        actor_id: str,
        idempotency_key: str,
        trace_id: str,
    ) -> OpsRunReceipt:
        """按精确父子关联请求取消，保持 AIOps Run 为权威状态。"""
        async with self._uow_factory() as uow:
            run = await uow.runs.get_by_parent_delegation_scoped(
                parent_delegation_id=delegation_id,
                domain_id=domain_id,
            )
            if run is None:
                raise resource_not_found("Delegation")
            run_id = run.ops_run_id
            row_version = int(run.row_version)
        return await self.request_cancel(
            ops_run_id=run_id,
            domain_id=domain_id,
            actor_id=actor_id,
            expected_row_version=row_version,
            idempotency_key=idempotency_key,
            trace_id=trace_id,
        )

    async def _diagnosis_snapshot(
        self,
        *,
        command,
        target,
        policy_snapshot: dict[str, Any],
        monitoring_snapshot: dict[str, Any],
        model_snapshot: dict[str, str] | None,
        source_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """冻结诊断模型、Prompt、窗口、权限范围和预算。"""
        if self._diagnosis_config is None or self._diagnosis_prompts is None:
            raise validation_failed("诊断编排尚未配置")
        config = self._diagnosis_config
        window = monitoring_snapshot["window"]
        raw_capabilities = dict(target.capabilities_json or {})
        capability_names = sorted(
            str(name)
            for name, enabled in raw_capabilities.items()
            if enabled is True
        )
        rules = dict(policy_snapshot.get("rules", {}))
        question = command.input.strip()
        if source_context is not None:
            question = (
                f"{question}\n\n"
                "以下是同一 Target 已完成的自动诊断上下文，仅作为历史证据继续核验：\n"
                f"{source_context['source_result']}"
            )
        return {
            "window": dict(window),
            "symptom_codes": tuple(
                str(item)
                for item in command.client_metadata.get(
                    "symptom_codes", ()
                )
                if isinstance(item, str) and item
            )[:32],
            "question_summary": question[:2000],
            "source_context": dict(source_context or {}),
            "target_capabilities": capability_names,
            "allowed_collection_ids": (),
            "policy_snapshot_hash": sha256_json(policy_snapshot),
            "model": {
                "enabled": bool(model_snapshot),
                "technical_name": (
                    model_snapshot["technical_name"]
                    if model_snapshot
                    else ""
                ),
                "revision": (
                    model_snapshot["revision"]
                    if model_snapshot
                    else ""
                ),
            },
            "prompts": await self._diagnosis_prompts.snapshot(
                DIAGNOSIS_PROMPT_IDS
            ),
            "budget": {
                "max_rounds": int(config.max_rounds),
                "max_tool_calls": int(config.max_tool_calls),
                "max_evidence_facts": int(config.max_evidence_facts),
            },
        }

    def _database_diagnostic_blueprint_snapshot(
        self,
        *,
        command,
        target,
        binding,
        policy,
        requested_tool_ids: tuple[str, ...] | None = None,
    ):
        """冻结 Target、目录选择、能力声明和执行上限。"""
        if command.blueprint_version != "1":
            raise validation_failed("数据库诊断 Blueprint 版本不受支持")
        if self._diagnostic_registry is None:
            raise validation_failed("数据库诊断目录尚未配置")
        raw_capabilities = dict(target.capabilities_json or {})
        capability_names = {
            str(name)
            for name, enabled in raw_capabilities.items()
            if enabled is True
        }
        configured_features = raw_capabilities.get("features", [])
        if isinstance(configured_features, list):
            capability_names.update(
                str(item) for item in configured_features if item
            )
        raw_entitlements = raw_capabilities.get("entitlements", [])
        entitlements = (
            {str(item) for item in raw_entitlements if item}
            if isinstance(raw_entitlements, list)
            else set()
        )
        capability_snapshot = {
            "db_type": target.db_type,
            "configured_version": target.version_code,
            "capabilities": sorted(capability_names),
            "entitlements": sorted(entitlements),
            "target_row_version": int(target.row_version),
        }
        capability_hash = sha256_json(capability_snapshot)
        initial_gaps: list[dict[str, Any]] = []
        selected = []
        target_enabled = getattr(target, "status", "ENABLED") == "ENABLED"
        policy_rules = dict(policy.rules_json) if policy is not None else {}
        policy_allowed = policy_rules.get(
            "readonly_database_enabled", True
        )
        readonly_enabled = bool(target.readonly_connection_enabled)
        if not policy_allowed:
            initial_gaps.append(
                {
                    "code": "DIAGNOSTIC_POLICY_DENIED",
                    "detail": "当前策略禁止数据库直连诊断",
                    "retryable": False,
                }
            )
        if not readonly_enabled:
            initial_gaps.append(
                {
                    "code": "DB_DIRECT_NOT_CONFIGURED",
                    "detail": "Target 未启用只读数据库连接",
                    "retryable": False,
                }
            )
        if not target_enabled:
            initial_gaps.append(
                {
                    "code": "TARGET_INACTIVE",
                    "detail": "Target 当前未启用，跳过数据库直连诊断",
                    "retryable": False,
                }
            )
        if not target.diagnostic_credential_id:
            initial_gaps.append(
                {
                    "code": "DIAGNOSTIC_SECRET_MISSING",
                    "detail": "Target 未配置只读诊断凭据",
                    "retryable": False,
                }
            )
        if not target.endpoint_json:
            initial_gaps.append(
                {
                    "code": "TARGET_ENDPOINT_MISSING",
                    "detail": "Target 未配置数据库地址",
                    "retryable": False,
                }
            )
        if target.connectivity_status not in {"CONNECTED", "DEGRADED"}:
            initial_gaps.append(
                {
                    "code": "TARGET_CONNECTIVITY_UNAVAILABLE",
                    "detail": "Target 当前不可连接，跳过数据库直连诊断",
                    "retryable": True,
                }
            )
        if not target.version_code:
            initial_gaps.append(
                {
                    "code": "VERSION_UNSUPPORTED",
                    "detail": "Target 未声明可用于目录选择的数据库版本",
                    "retryable": False,
                }
            )
        automatic_access_enabled = (
            target_enabled
            and policy_allowed
            and readonly_enabled
            and bool(target.version_code)
            and bool(target.diagnostic_credential_id)
            and bool(target.endpoint_json)
            and target.connectivity_status in {"CONNECTED", "DEGRADED"}
        )
        requested = requested_tool_ids or (
            "db.instance.identity",
            "db.session.active",
            "db.session.blocking_chain",
            "db.storage.capacity",
        )
        # 即使数据库不可直连，也要冻结目录工具，供 CHAT HITL 生成受控只读 SQL。
        if target.version_code:
            for tool_id in requested:
                try:
                    tool = self._diagnostic_registry.resolve(
                        tool_id=tool_id,
                        tool_version="1.0.0",
                        db_type=target.db_type,
                        db_version=target.version_code,
                        capabilities=capability_names,
                        entitlements=entitlements,
                    )
                except LookupError:
                    initial_gaps.append(
                        {
                            "code": "CAPABILITY_UNAVAILABLE",
                            "tool_id": tool_id,
                            "detail": "工具版本、能力或许可条件不满足",
                            "retryable": False,
                        }
                    )
                    continue
                definition = tool.definition
                selected.append(
                    {
                        "tool_id": definition.tool_id,
                        "version": definition.version,
                        "variant": definition.variant,
                        "template_sha256": definition.template_sha256,
                        "parameters": {
                            parameter.name: parameter.default
                            for parameter in definition.parameters
                            if not parameter.required
                        },
                        "parameter_definitions": [
                            parameter.model_dump(mode="json")
                            for parameter in definition.parameters
                        ],
                        "output_columns": [
                            column.model_dump(mode="json")
                            for column in definition.output_columns
                        ],
                        "cost_level": definition.cost_level,
                        "supported_version_min": (
                            definition.supported_version_min
                        ),
                        "supported_version_max_exclusive": (
                            definition.supported_version_max_exclusive
                        ),
                        "limits": {
                            "statement_timeout_seconds": (
                                definition.timeout_seconds
                            ),
                            "max_result_rows": definition.max_rows,
                            "max_result_bytes": definition.max_bytes,
                            "max_columns": 128,
                            "max_cell_chars": 32768,
                        },
                    }
                )
        selected.sort(
            key=lambda item: (
                item["tool_id"] != "db.instance.identity",
                item["tool_id"],
            )
        )
        snapshot = {
            "domain_id": int(target.domain_id),
            "db_type": target.db_type,
            "configured_version": target.version_code or "UNKNOWN",
            "target_row_version": int(target.row_version),
            "connection_profile": dict(target.endpoint_json or {}),
            "diagnostic_credential_id": str(target.diagnostic_credential_id),
            "automatic_access_enabled": automatic_access_enabled,
            "catalog_hash": self._diagnostic_registry.catalog_hash,
            "capability_snapshot": capability_snapshot,
            "capability_snapshot_hash": capability_hash,
            "tools": selected,
            "initial_gaps": initial_gaps,
        }
        blueprint = build_database_diagnostic_blueprint(
            tuple(item["tool_id"] for item in selected)
        )
        return blueprint, snapshot

    async def _monitor_blueprint_snapshot(
        self,
        *,
        uow,
        command: CreateOpsRunCommand,
        target,
        now: datetime,
        allowed_source_ids: tuple[UUID, ...] | None = None,
    ):
        """在 Run 创建事务内冻结监控绑定、目录与查询窗口。"""
        if command.blueprint_version != "1":
            raise validation_failed("监控 Blueprint 版本不受支持")
        snapshot = await self._monitoring_snapshot_builder.build(
            uow=uow,
            domain_id=command.domain_id,
            target=target,
            now=now,
            allowed_source_ids=allowed_source_ids,
            window_start=command.observation_start,
            window_end=command.observation_end,
        )
        blueprint = build_monitor_observe_blueprint(
            tuple(snapshot["observation_binding_ids"])
        )
        return blueprint, snapshot

    async def claim_task(
        self, command: ClaimOpsTaskCommand
    ) -> TaskLease | None:
        lease_token = uuid7()
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            task = await uow.runs.claim_task(
                now=now,
                lease_owner=command.worker_id,
                lease_token=lease_token,
                lease_until=now
                + timedelta(seconds=command.lease_seconds),
            )
            if task is None:
                return None
            run = await uow.runs.get_run(
                ops_run_id=task.ops_run_id
            )
            assert run is not None
            lease_limits = [
                now + timedelta(seconds=command.lease_seconds),
                task_deadline(task),
            ]
            if run.deadline_at is not None:
                lease_limits.append(run.deadline_at)
            task.lease_until = min(lease_limits)
            phase = TASK_TYPE_TO_RUN_PHASE.get(task.task_type)
            if phase is not None and run.status != phase.value:
                ensure_run_transition(
                    DomainOpsRunStatus(run.status), phase
                )
                run.status = phase.value
                run.started_at = run.started_at or now
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    event_type="run.status",
                    event_key=(
                        f"run:{run.ops_run_id}:phase:{phase.value}"
                    ),
                    visibility="USER",
                    payload_json={
                        "status": phase.value,
                        "trace_id": command.trace_id,
                    },
                )
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="task.status",
                event_key=(
                    f"task:{task.ops_task_id}:claimed:"
                    f"{int(task.attempt_count)}:{hashlib.sha256(str(lease_token).encode()).hexdigest()}"
                ),
                visibility="USER",
                payload_json={
                    "status": DomainOpsTaskStatus.RUNNING.value,
                    "task_id": str(task.ops_task_id),
                    "task_type": task.task_type,
                    "task_key": task.task_key,
                    "trace_id": command.trace_id,
                },
            )
            if run.workflow_kind in _AGENT_TURN_WORKFLOWS:
                await self._project_turn_task_started(
                    uow=uow,
                    run=run,
                    task=task,
                )
            artifacts = await self._input_artifacts(
                uow, run_id=run.ops_run_id, task=task
            )
            future_error = (
                await followup_access_error(uow=uow, run=run, now=now)
                if run.workflow_kind == FOLLOWUP_WORKFLOW else None
            )
            await uow.commit()
            lease = self._task_lease(
                run, task, artifacts, lease_token=lease_token
            )
            if run.workflow_kind == FOLLOWUP_WORKFLOW:
                lease = lease.model_copy(update={"plan_snapshot": {
                    **lease.plan_snapshot,
                    "future_observation_access_error": future_error,
                    "future_observation_due_at": task.available_at.isoformat(),
                }})
            return lease

    async def _project_turn_task_started(self, *, uow, run, task) -> None:
        """把后台 Task 领取投影为用户可见阶段，并同步 Tool 运行状态。"""
        link = await uow.turns.get_run_link_by_ops_run_id(
            ops_run_id=run.ops_run_id
        )
        if link is None:
            raise state_conflict("Agent Turn Run 缺少 Primary Turn 关联")
        turn = await uow.turns.get_turn(
            domain_id=int(run.domain_id),
            turn_id=link.turn_id,
            lock=True,
        )
        if turn is None:
            raise state_conflict("Agent Turn Task 缺少有效 Turn")
        invocation = await uow.turns.get_tool_invocation_by_task(
            ops_task_id=task.ops_task_id,
            lock=True,
        )
        if invocation is not None:
            invocation.status = "RUNNING"
            plan = dict(
                dict(run.plan_snapshot_json or {}).get("answer_context") or {}
            ).get("investigation_plan") or {}
            action = next(
                (
                    item
                    for item in plan.get("actions", ())
                    if str(item.get("action_id")) == str(invocation.action_id)
                ),
                {},
            )
            question = str(action.get("question") or "执行只读诊断步骤")
            await self._append_turn_event(
                uow,
                turn,
                event_type="tool.started",
                payload={
                    "action_id": invocation.action_id,
                    "tool_id": invocation.tool_id,
                    "question": question,
                    "expected_evidence_kind": str(
                        action.get("expected_evidence_kind") or "OBSERVATION"
                    ),
                    "public_sections": [
                        {
                            "title": "本步要回答",
                            "items": [question],
                        },
                        {
                            "title": "执行方式",
                            "items": [
                                f"调用只读工具 {invocation.tool_id}",
                                "结果会先登记为证据，再参与后续判断",
                            ],
                        },
                        {
                            "title": "预期产出",
                            "items": [
                                str(
                                    action.get("expected_evidence_kind")
                                    or "数据库诊断观测"
                                )
                            ],
                        },
                    ],
                    "public_summary": f"正在执行：{question}",
                },
            )
            return
        if str(task.task_key).startswith("evidence:assess"):
            await self._append_turn_event(
                uow,
                turn,
                event_type="assessment.started",
                payload={"public_summary": "正在评估证据充分性和下一步动作"},
            )
        elif task.task_key == "answer:compose":
            await self._append_turn_event(
                uow,
                turn,
                event_type="thinking.delta",
                payload={"public_summary": "正在依据已验证证据生成回答"},
            )

    async def heartbeat_task(
        self, command: HeartbeatOpsTaskCommand
    ) -> TaskLease:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            preliminary = await uow.runs.get_task(
                ops_task_id=command.task_id
            )
            if preliminary is None:
                raise self._stale_lease()
            run = await uow.runs.get_run(
                ops_run_id=preliminary.ops_run_id, lock=True
            )
            task = await uow.runs.get_task(
                ops_task_id=command.task_id, lock=True
            )
            if run is None or task is None:
                raise self._stale_lease()
            self._ensure_lease(
                run=run,
                task=task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            requested_until = now + timedelta(
                seconds=command.lease_seconds
            )
            timeout_until = task_deadline(task)
            limits = [requested_until, timeout_until]
            if run.deadline_at is not None:
                limits.append(run.deadline_at)
            task.lease_until = min(limits)
            task.heartbeat_at = now
            artifacts = await self._input_artifacts(
                uow, run_id=run.ops_run_id, task=task
            )
            await uow.commit()
            return self._task_lease(
                run,
                task,
                artifacts,
                lease_token=command.lease_token,
            )

    async def complete_task(
        self, command: CompleteOpsTaskCommand
    ) -> TaskMutationReceipt:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run, task = await self._lock_run_task(
                uow, command.task_id
            )
            event_key = f"task:{task.ops_task_id}:completed"
            prior = await uow.runs.get_event_by_key(
                ops_run_id=run.ops_run_id, event_key=event_key
            )
            artifact_key = self._artifact_key(task)
            existing = await uow.runs.get_artifact_by_key(
                ops_run_id=run.ops_run_id,
                artifact_key=artifact_key,
            )
            artifact_payload = command.artifact.payload
            generated_runbook_artifacts: tuple[dict[str, Any], ...] = ()
            if (
                command.artifact.schema_version == "AIOPS_TURN_RESULT.v1"
                and isinstance(artifact_payload, dict)
            ):
                artifact_payload = json.loads(json.dumps(artifact_payload))
                blocks = list(artifact_payload.get("blocks") or [])
                generated: list[dict[str, Any]] = []
                for block_value in blocks:
                    block = dict(block_value or {})
                    payload_value = dict(block.get("payload") or {})
                    generated.extend(
                        dict(item)
                        for item in tuple(
                            payload_value.pop("_generated_artifacts", ()) or ()
                        )
                    )
                    block["payload"] = payload_value
                    block_value.clear()
                    block_value.update(block)
                generated_runbook_artifacts = tuple(generated)
            content = (
                artifact_payload
                if artifact_payload is not None
                else {"payload_uri": command.artifact.payload_uri}
            )
            content_hash = sha256_json(content)
            if prior is not None and existing is not None:
                self._ensure_same_artifact(
                    existing, command.artifact, content_hash
                )
                return self._mutation_receipt(
                    run, task, int(prior.sequence_no), existing.artifact_id
                )
            self._ensure_lease(
                run=run,
                task=task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            self._validate_artifact(task, command.artifact)
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            if existing is not None:
                self._ensure_same_artifact(
                    existing, command.artifact, content_hash
                )
                raise _runtime_error(
                    "OPS_ARTIFACT_STATE_CONFLICT",
                    "Artifact 已存在但 Task 尚未完成",
                )
            artifact = await uow.runs.add_artifact(
                OpsArtifactEntity(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    artifact_key=artifact_key,
                    artifact_type=command.artifact.artifact_type,
                    schema_version=command.artifact.schema_version,
                    payload_json=artifact_payload,
                    payload_uri=command.artifact.payload_uri,
                    content_hash=content_hash,
                    byte_size=len(canonical_bytes(content)),
                    provenance_json={
                        **command.artifact.provenance,
                        "producer": command.artifact.producer,
                        "producer_version": (
                            command.artifact.producer_version
                        ),
                    },
                    trust_level=command.artifact.trust_level,
                    security_level=command.artifact.security_level,
                )
            )
            await schedule_future_observation(uow=uow, run=run, task=task, artifact=artifact, now=now)
            for generated_payload in generated_runbook_artifacts:
                artifact_id = str(generated_payload["artifact_id"])
                generated_content = str(generated_payload["content"])
                generated_hash = hashlib.sha256(
                    generated_content.encode("utf-8")
                ).hexdigest()
                if generated_hash != str(generated_payload["sha256"]):
                    raise state_conflict(
                        f"实施脚本摘要不一致：{artifact_id}"
                    )
                await uow.runs.add_artifact(
                    OpsArtifactEntity(
                        ops_run_id=run.ops_run_id,
                        ops_task_id=task.ops_task_id,
                        artifact_key=f"implementation-runbook:{artifact_id}",
                        artifact_type="IMPLEMENTATION_SCRIPT",
                        schema_version="AIOPS_RUNBOOK_ARTIFACT.v1",
                        payload_json={
                            "artifact_id": artifact_id,
                            "relative_path": generated_payload["relative_path"],
                            "media_type": generated_payload["media_type"],
                            "content": generated_content,
                        },
                        payload_uri=None,
                        content_hash=generated_hash,
                        byte_size=len(generated_content.encode("utf-8")),
                        provenance_json={
                            "producer": "aiops.implementation-compiler",
                            "answer_artifact_id": str(artifact.artifact_id),
                        },
                        trust_level="SYSTEM",
                        security_level=1,
                    )
                )
            if command.artifact.schema_version == "OBSERVATION_SET.v1":
                await self._reduce_observation_health(
                    uow=uow,
                    run=run,
                    payload=command.artifact.payload or {},
                    now=now,
                )
            if (
                command.artifact.schema_version
                == "DATABASE_DIAGNOSTIC_RESULT.v1"
            ):
                await self._reduce_database_health(
                    uow=uow,
                    run=run,
                    payload=command.artifact.payload or {},
                    now=now,
                )
            if command.artifact.schema_version == "PROPOSAL_OUTCOME.v1":
                await self._materialize_advisory_proposal(
                    uow=uow,
                    run=run,
                    task=task,
                    artifact=artifact,
                    payload=command.artifact.payload or {},
                    trace_id=command.trace_id,
                    now=now,
                )
            ensure_task_transition(
                DomainOpsTaskStatus(task.status),
                DomainOpsTaskStatus.SUCCEEDED,
            )
            task.status = DomainOpsTaskStatus.SUCCEEDED.value
            task.output_artifact_id = artifact.artifact_id
            task.completed_at = now
            self._clear_lease(task)
            if run.workflow_kind in _AGENT_TURN_WORKFLOWS:
                await self._project_turn_task(
                    uow=uow,
                    run=run,
                    task=task,
                    artifact=artifact,
                    now=now,
                )
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="artifact.created",
                event_key=f"artifact:{artifact_key}",
                visibility="INTERNAL",
                payload_json={
                    "artifact_id": str(artifact.artifact_id),
                    "artifact_type": artifact.artifact_type,
                    "schema_version": artifact.schema_version,
                    "content_hash": artifact.content_hash,
                    "trace_id": command.trace_id,
                },
            )
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="task.status",
                event_key=event_key,
                visibility="USER",
                payload_json={
                    "status": DomainOpsTaskStatus.SUCCEEDED.value,
                    "task_id": str(task.ops_task_id),
                    "task_type": task.task_type,
                    "task_key": task.task_key,
                    "trace_id": command.trace_id,
                },
            )
            if (run.workflow_kind == FOLLOWUP_WORKFLOW and task.handler_id == "dba.future-point.invoke"
                and (artifact.payload_json.get("status") != "SUCCEEDED"
                    or any(dict(item.get("observation") or {}).get("truncated")
                        for item in artifact.payload_json.get("tool_outcomes", ())))):
                self._block_unreachable(tasks, failed_key=task.task_key)
                await preserve_future_observation(uow=uow, run=run, now=now, reason="FOLLOWUP_STOPPED")
                ensure_run_transition(DomainOpsRunStatus(run.status), DomainOpsRunStatus.PARTIAL)
                run.status, run.completed_at = DomainOpsRunStatus.PARTIAL.value, now
                event = await uow.runs.append_event(
                    ops_run_id=run.ops_run_id, event_type="run.partial",
                    event_key=f"run:{run.ops_run_id}:terminal", visibility="USER",
                    payload_json={"status": "PARTIAL", "final_artifact_id": str(run.final_artifact_id),
                        "public_summary": "未来观察出现缺口，已停止后续采样并保留实际时点", "trace_id": command.trace_id},
                )
                await self._add_outbox(
                    uow, aggregate_id=run.ops_run_id, event_type="OPS_RUN_COMPLETED",
                    idempotency_key=f"run:{run.ops_run_id}:completed",
                    payload={"ops_run_id": str(run.ops_run_id), "artifact_id": str(run.final_artifact_id)},
                    trace_id=command.trace_id, now=now,
                )
                await uow.commit()
                return self._mutation_receipt(run, task, int(event.sequence_no), artifact.artifact_id)
            released = self._release_successors(tasks, now=now)
            await self._sync_run_budget(uow=uow, run=run, tasks=tasks, now=now)
            if run.workflow_kind == FOLLOWUP_WORKFLOW:
                interval = int(run.plan_snapshot_json["future_observation"]["requested"]["interval_seconds"])
                for successor in released:
                    if successor.handler_id == "dba.future-point.invoke":
                        successor.available_at = max(successor.available_at, now + timedelta(seconds=interval))
            for successor in released:
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=successor.ops_task_id,
                    event_type="task.status",
                    event_key=(
                        f"task:{successor.ops_task_id}:ready:"
                        f"{int(successor.attempt_count)}"
                    ),
                    visibility="USER",
                    payload_json={
                        "status": "READY",
                        "task_id": str(successor.ops_task_id),
                        "task_type": successor.task_type,
                        "task_key": successor.task_key,
                        "trace_id": command.trace_id,
                    },
                )
            if all(
                item.status == DomainOpsTaskStatus.SUCCEEDED.value
                for item in tasks
            ):
                completed_status = (
                    DomainOpsRunStatus.PARTIAL
                    if run.workflow_kind == FOLLOWUP_WORKFLOW and artifact.payload_json.get("status") != "SUCCEEDED"
                    else DomainOpsRunStatus.COMPLETED
                )
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    completed_status,
                )
                run.status = completed_status.value
                final_artifact = artifact
                if artifact.schema_version == "ACTION_VERIFICATION.v1":
                    final_artifact = (
                        await self._publish_comparison_report(
                            uow=uow,
                            run=run,
                            task=task,
                            verification_artifact=artifact,
                            now=now,
                            trace_id=command.trace_id,
                        )
                    )
                elif self._should_publish_turn_inspection_report(
                    run=run,
                    artifact=artifact,
                ):
                    # 定时巡检的交付物是报告而非对话回答。保留原始 Turn
                    # 产物作为报告来源，并将模板化报告设为 Run 的展示终态。
                    final_artifact = await self._publish_turn_inspection_report(
                        uow=uow,
                        run=run,
                        task=task,
                        source_artifact=artifact,
                        now=now,
                        trace_id=command.trace_id,
                    )
                run.final_artifact_id = final_artifact.artifact_id
                run.completed_at = now
                await route_completed_run_work_items(
                    uow=uow,
                    run=run,
                    now=now,
                    actor_id="system:work-item-router",
                )
                if (
                    run.trigger_type == "CHAT"
                    and final_artifact.schema_version
                    == "DIAGNOSIS_REPORT_DRAFT.v1"
                ):
                    answer = _diagnosis_answer_markdown(
                        dict(final_artifact.payload_json or {})
                    )
                    chunks = tuple(
                        answer[index:index + 120]
                        for index in range(0, len(answer), 120)
                    )
                    for index, delta in enumerate(chunks, start=1):
                        await uow.runs.append_event(
                            ops_run_id=run.ops_run_id,
                            event_type="answer.delta",
                            event_key=(
                                f"run:{run.ops_run_id}:answer:{index}"
                            ),
                            visibility="USER",
                            payload_json={
                                "delta": delta,
                                "trace_id": command.trace_id,
                            },
                        )
                    await uow.runs.append_event(
                        ops_run_id=run.ops_run_id,
                        event_type="answer.completed",
                        event_key=f"run:{run.ops_run_id}:answer:completed",
                        visibility="USER",
                        payload_json={
                            "chunk_count": len(chunks),
                            "trace_id": command.trace_id,
                        },
                    )
                event = await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    event_type="run.partial" if completed_status == DomainOpsRunStatus.PARTIAL else "run.completed",
                    event_key=f"run:{run.ops_run_id}:terminal",
                    visibility="USER",
                    payload_json={
                        "status": completed_status.value,
                        "final_artifact_id": str(
                            final_artifact.artifact_id
                        ),
                        "trace_id": command.trace_id,
                    },
                )
                await self._add_outbox(
                    uow,
                    aggregate_id=run.ops_run_id,
                    event_type="OPS_RUN_COMPLETED",
                    idempotency_key=(
                        f"run:{run.ops_run_id}:completed"
                    ),
                    payload={
                        "ops_run_id": str(run.ops_run_id),
                        "artifact_id": str(final_artifact.artifact_id),
                    },
                    trace_id=command.trace_id,
                    now=now,
                )
            await uow.commit()
            return self._mutation_receipt(
                run, task, int(event.sequence_no), artifact.artifact_id
            )

    async def append_task_progress(
        self, command: AppendOpsTaskProgressCommand
    ) -> TaskMutationReceipt:
        """在有效Task租约内提交用户可见增量，并同步投影到Turn事件流。"""
        if command.event_type not in {
            "answer.delta",
            "thinking.delta",
            "tool.progress",
        }:
            raise validation_failed("不允许写入该类型的Task增量事件")
        if len(
            json.dumps(command.payload, ensure_ascii=False, default=str)
        ) > 16000:
            raise validation_failed("Task增量事件内容超过限制")
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run, task = await self._lock_run_task(uow, command.task_id)
            self._ensure_lease(
                run=run,
                task=task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            event_key = (
                f"task:{task.ops_task_id}:progress:{command.event_key}"
            )
            prior = await uow.runs.get_event_by_key(
                ops_run_id=run.ops_run_id,
                event_key=event_key,
            )
            if prior is not None:
                prior_payload = dict(prior.payload_json or {})
                prior_payload.pop("trace_id", None)
                if (
                    prior.event_type != command.event_type
                    or prior_payload != dict(command.payload)
                ):
                    raise state_conflict(
                        "Task增量事件幂等键对应的内容不一致"
                    )
                return self._mutation_receipt(
                    run, task, int(prior.sequence_no), None
                )
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type=command.event_type,
                event_key=event_key,
                visibility="USER",
                payload_json={
                    **dict(command.payload),
                    "trace_id": command.trace_id,
                },
            )
            if run.workflow_kind in _AGENT_TURN_WORKFLOWS:
                link = await uow.turns.get_run_link_by_ops_run_id(
                    ops_run_id=run.ops_run_id
                )
                turn = (
                    await uow.turns.get_turn(
                        domain_id=int(run.domain_id),
                        turn_id=link.turn_id,
                        lock=True,
                    )
                    if link is not None
                    else None
                )
                if turn is None:
                    raise state_conflict(
                        "Agent Turn增量事件缺少有效Turn关联"
                    )
                await self._append_turn_event(
                    uow,
                    turn,
                    event_type=command.event_type,
                    payload=dict(command.payload),
                )
            await uow.commit()
            return self._mutation_receipt(
                run, task, int(event.sequence_no), None
            )

    async def _project_turn_task(
        self,
        *,
        uow,
        run,
        task,
        artifact,
        now: datetime,
    ) -> None:
        """在 Task 完成事务内同步维护 Turn 的权威业务投影。"""
        link = await uow.turns.get_run_link_by_ops_run_id(
            ops_run_id=run.ops_run_id
        )
        if link is None:
            raise state_conflict("Agent Turn Run 缺少 Primary Turn 关联")
        turn = await uow.turns.get_turn(
            domain_id=int(run.domain_id),
            turn_id=link.turn_id,
            lock=True,
        )
        if turn is None:
            raise resource_not_found("Conversation Turn")
        payload = dict(artifact.payload_json or {})
        if artifact.schema_version == "DBA_TOOL_RESULT.v1":
            await self._project_tool_result(
                uow=uow,
                turn=turn,
                task=task,
                artifact=artifact,
                payload=payload,
                now=now,
            )
        elif artifact.schema_version == "TOP_SQL_WINDOW_RESULT.v1":
            await self._project_top_sql_window_result(
                uow=uow, turn=turn, task=task, artifact=artifact, payload=payload, now=now,
            )
        elif artifact.schema_version == "OBSERVATION_SET.v1":
            await self._project_monitoring_result(
                uow=uow,
                turn=turn,
                artifact=artifact,
                payload=payload,
            )
        elif artifact.schema_version == "LOG_EVIDENCE_SET.v1":
            await self._project_log_result(
                uow=uow,
                turn=turn,
                artifact=artifact,
                payload=payload,
            )
        elif artifact.schema_version == "ATTACHMENT_EVIDENCE_SET.v1":
            await self._project_attachment_search_result(
                uow=uow,
                turn=turn,
                task=task,
                artifact=artifact,
                payload=payload,
                now=now,
            )
        elif artifact.schema_version == "DBA_SUFFICIENCY.v1":
            assessment = DbaSufficiencyAssessment.model_validate(payload)
            gained_new_evidence = _assessment_progress(
                dict(turn.sufficiency_json or {}), assessment
            )
            turn.sufficiency_status = str(assessment.status)
            turn.sufficiency_json = assessment.model_dump(mode="json")
            turn.sufficiency_artifact_id = artifact.artifact_id
            turn.assessment_artifact_id = artifact.artifact_id
            if not gained_new_evidence:
                turn.no_progress_count = int(turn.no_progress_count or 0) + 1
            else:
                turn.no_progress_count = 0
            revisions = await uow.turns.list_investigation_revisions(
                turn_id=turn.turn_id
            )
            if revisions:
                revisions[-1].assessment_artifact_id = artifact.artifact_id
            await self._append_turn_event(
                uow,
                turn,
                event_type="assessment.completed",
                payload={
                    "sufficiency_status": str(assessment.status),
                    "evidence_count": len(assessment.evidence),
                    "gap_count": len(assessment.gaps),
                    "public_summary": (
                        f"证据评估完成：{len(assessment.evidence)} 项有效证据，"
                        f"{len(assessment.gaps)} 项缺口"
                    ),
                },
            )
            deterministic_replan = (
                str(assessment.status) in {"NEEDS_EVIDENCE", "PARTIAL"}
                and any(gap.retryable for gap in assessment.gaps)
            )
            investigation_plan = dict(
                dict(run.plan_snapshot_json or {}).get("answer_context")
                or {}
            ).get("investigation_plan")
            replan_limit = int(
                getattr(self._diagnosis_config, "max_investigation_replans", 2)
            )
            replan_seconds = int(
                getattr(self._diagnosis_config, "max_investigation_replan_seconds", 180)
            )
            replan_budget_exhausted = _replan_budget_exhausted(
                turn, now=now, max_replans=replan_limit, max_seconds=replan_seconds,
                excluded_wait_milliseconds=run.budget_waited_ms or 0,
            )
            task_frame = dict(
                dict(run.plan_snapshot_json or {}).get("answer_context") or {}
            ).get("task_frame") or {}
            should_replan = self._should_replan_investigation(
                assessment=assessment,
                task_frame=task_frame,
                deterministic_replan=deterministic_replan,
                no_progress_count=int(turn.no_progress_count or 0),
                has_deferred=prior_plan_has_deferred(investigation_plan),
                deadline_reached=(
                    replan_budget_exhausted
                    or (run.deadline_at is not None and run.deadline_at <= now + timedelta(seconds=int(getattr(self._diagnosis_config, "answer_reserve_seconds", 20))))
                ),
                tool_budget_remaining=(
                    int(turn.tool_call_count or 0)
                    < int(
                        getattr(self._diagnosis_config, "max_tool_calls", 12)
                    )
                ),
            )
            if should_replan and run.workflow_kind != "INSPECTION":
                await self._schedule_turn_replan(
                    uow=uow,
                    run=run,
                    turn=turn,
                    assessment_artifact=artifact,
                )
                return
            turn.status = "ANSWERING"
            await self._append_turn_event(
                uow,
                turn,
                event_type="turn.status",
                payload={
                    "status": "ANSWERING",
                    "public_summary": (
                        "补证预算已用完，正在基于已有证据形成回答"
                        if replan_budget_exhausted else "证据已整理，正在形成回答"
                    ),
                },
            )
        elif artifact.schema_version == "AIOPS_TURN_RESULT.v1":
            await self._project_turn_answer(
                uow=uow,
                turn=turn,
                artifact=artifact,
                payload=payload,
                now=now,
            )

    @staticmethod
    def _should_replan_investigation(
        *,
        assessment: DbaSufficiencyAssessment,
        task_frame: dict[str, Any] | None = None,
        deterministic_replan: bool,
        no_progress_count: int,
        has_deferred: bool = False,
        deadline_reached: bool = False,
        tool_budget_remaining: bool = True,
    ) -> bool:
        """存在可执行缺口时按截止时间、工具预算和连续进展决定补证。"""
        next_action = (
            assessment.investigation.next_action
            if assessment.investigation is not None
            else None
        )
        if (
            direct_list_result_kind(task_frame or {})
            or next_action == "STOP_UNSAFE"
            or no_progress_count >= 2
            or deadline_reached
            or not tool_budget_remaining
        ):
            return False
        if task_frame and task_frame.get("diagnostic_profile") in {"SINGLE_SQL_PERFORMANCE", "SQL_COMPARISON"} and (
            assessment.sql_checkpoint is not None or assessment.sql_diagnosis_error is not None
        ):
            # 已有 SQL 阶段判断仅为关键补证继续，不能因无关窗口/监控缺口重跑。
            return next_action == "REPLAN"
        if has_deferred:
            return True
        requested = (
            next_action == "REPLAN"
            if assessment.investigation is not None
            else deterministic_replan
        )
        blocked_by_model = next_action == "ASK_USER"
        return (requested or deterministic_replan) and not blocked_by_model

    async def _schedule_turn_replan(
        self,
        *,
        uow,
        run,
        turn,
        assessment_artifact,
    ) -> None:
        """冻结回答Task并可靠投递下一轮调查规划。"""
        revision_no = int(turn.current_plan_revision or 1) + 1
        tasks = await uow.runs.list_tasks(
            ops_run_id=run.ops_run_id,
            lock=True,
        )
        answer = next(
            (item for item in tasks if item.task_key == "answer:compose"),
            None,
        )
        if answer is None or answer.status != "PENDING":
            raise state_conflict("重规划时回答Task状态无效")
        assessment_key = f"evidence:assess:r{revision_no}"
        action_plan = next(
            (item for item in tasks if item.task_key == "change:action-plan"),
            None,
        )
        if action_plan is not None:
            if action_plan.status != "PENDING":
                raise state_conflict("重规划时动作计划Task状态无效")
            action_plan.depends_on_json = [assessment_key]
            action_plan.input_artifacts_json = [assessment_key]
            answer.input_artifacts_json = [
                assessment_key,
                "change:proposal",
            ]
        else:
            answer.depends_on_json = [assessment_key]
            answer.input_artifacts_json = [assessment_key]
        payload = {
            "schema_version": "aiops.turn-replan-command.v1",
            "domain_id": int(run.domain_id),
            "turn_id": str(turn.turn_id),
            "ops_run_id": str(run.ops_run_id),
            "assessment_artifact_id": str(
                assessment_artifact.artifact_id
            ),
            "revision_no": revision_no,
            "trace_id": run.trace_id,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        await uow.outbox.add(
            OutboxEntity(
                aggregate_type="CONVERSATION_TURN",
                aggregate_id=turn.turn_id,
                event_type="aiops.turn.replanning_requested",
                idempotency_key=(
                    f"turn-replan:{turn.turn_id}:{revision_no}"
                ),
                payload_json=payload,
                payload_hash=hashlib.sha256(encoded).hexdigest(),
                trace_id=run.trace_id,
            )
        )
        turn.status = "REPLANNING"
        await self._append_turn_event(
            uow,
            turn,
            event_type="turn.status",
            payload={
                "status": "REPLANNING",
                "public_summary": "当前证据仍有可补齐的关键缺口，正在调整调查计划",
            },
        )

    async def _project_deadline_partial(self, *, uow, run, now: datetime) -> bool:
        """有已持久化事实时在原事务内产生唯一部分结果，不再消耗外部执行预算。"""
        from aiops_agent.application.diagnosis.deadline_result import deadline_result

        if run.workflow_kind == FOLLOWUP_WORKFLOW:
            return await preserve_future_observation(
                uow=uow, run=run, now=now, reason="FOLLOWUP_DEADLINE_EXCEEDED",
            )
        if run.workflow_kind != "CHAT_TURN":
            return False
        link = await uow.turns.get_run_link_by_ops_run_id(ops_run_id=run.ops_run_id)
        if link is None:
            return False
        turn = await uow.turns.get_turn(domain_id=run.domain_id, turn_id=link.turn_id, lock=True)
        if turn is None or turn.status in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "WAITING_USER"}:
            return False
        artifacts = await uow.runs.list_artifacts(ops_run_id=run.ops_run_id)
        projected_ids = {str(item.artifact_id) for item in await uow.turns.list_evidence(turn_id=turn.turn_id)}
        from aiops_agent.application.investigation.prior_sql_replay import validate_reused_prior_sql
        from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources

        artifacts = [item for item in artifacts if str(item.artifact_id) in projected_ids and sha256_json(item.payload_json) == item.content_hash]
        projected_ids.intersection_update(str(item.artifact_id) for item in artifacts)
        validated_prior_ids = set()
        validated_assessment_ids = set()
        for item in artifacts:
            if item.schema_version == "PRIOR_TURN_SQL_EVIDENCE.v1" and await validate_reused_prior_sql(uow=uow, run=run, artifact=item, turn=turn):
                validated_prior_ids.add(str(item.artifact_id))
            if item.schema_version == "DBA_SUFFICIENCY.v1" and item.trust_level == "SOURCE_VERIFIED" and await validate_investigation_sources(
                uow=uow, run=run, payload={"evidence": list(item.payload_json.get("evidence") or ())},
                plan_capture_reader=getattr(self, "_plan_capture_reader", None),
            ):
                validated_assessment_ids.add(str(item.artifact_id))
        plan = dict(run.plan_snapshot_json or {})
        result = deadline_result(
            artifacts=artifacts,
            task_frame=dict(dict(plan.get("answer_context") or {}).get("task_frame") or {}),
            database_type=str(dict(plan.get("capability_snapshot") or {}).get("database_type") or "").upper(),
            projected_ids=projected_ids,
            validated_prior_ids=frozenset(validated_prior_ids),
            validated_assessment_ids=frozenset(validated_assessment_ids),
        )
        if result is None:
            return False
        payload = result.model_dump(mode="json")
        artifact = await uow.runs.add_artifact(OpsArtifactEntity(
            ops_run_id=run.ops_run_id, ops_task_id=None,
            artifact_key="turn:deadline-result", artifact_type="AIOPS_TURN_RESULT",
            schema_version="AIOPS_TURN_RESULT.v1", payload_json=payload,
            content_hash=sha256_json(payload), byte_size=len(canonical_bytes(payload)),
            provenance_json={"producer": "execution.deadline", "producer_version": "1"},
            trust_level="SOURCE_VERIFIED", security_level=max(
                item.security_level for item in artifacts
                if str(item.artifact_id) in {str(fact.artifact_id) for fact in result.evidence}
            ),
        ))
        await self._project_turn_answer(uow=uow, turn=turn, artifact=artifact, payload=payload, now=now)
        turn.sufficiency_json = DbaSufficiencyAssessment(
            status="PARTIAL", evidence=result.evidence, gaps=result.evidence_gaps,
            reasons=("执行预算已用完，保留已核验事实。",),
        ).model_dump(mode="json")
        run.final_artifact_id = artifact.artifact_id
        return True

    async def _project_turn_failure(
        self,
        *,
        uow,
        run,
        error_code: str,
        public_summary: str,
        now: datetime,
    ) -> None:
        """在 Task 终态失败事务内同步结束 Agent Turn。"""
        await self._project_turn_terminal(
            uow=uow,
            run=run,
            status="FAILED",
            error_code=error_code,
            public_summary=public_summary,
            now=now,
        )

    async def _project_turn_terminal(
        self,
        *,
        uow,
        run,
        status: str,
        error_code: str | None,
        public_summary: str,
        now: datetime,
    ) -> None:
        """把 Run 的失败、取消或过期终态同步到 Agent Turn。"""
        if status not in {"FAILED", "CANCELLED"}:
            raise ValueError("Agent Turn 终态投影仅支持失败或取消")
        link = await uow.turns.get_run_link_by_ops_run_id(
            ops_run_id=run.ops_run_id
        )
        if link is None:
            raise state_conflict("Agent Turn Run 缺少 Primary Turn 关联")
        turn = await uow.turns.get_turn(
            domain_id=int(run.domain_id),
            turn_id=link.turn_id,
            lock=True,
        )
        if turn is None:
            raise resource_not_found("Conversation Turn")
        if turn.status in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}:
            return
        turn.status = status
        turn.error_domain = "EXECUTION" if error_code else None
        turn.error_code = error_code
        turn.error_message = public_summary
        turn.completed_at = now
        await self._append_turn_event(
            uow,
            turn,
            event_type="turn.status",
            payload={
                "status": status,
                "error_domain": "EXECUTION" if error_code else None,
                "error_code": error_code,
                "public_summary": public_summary,
            },
        )

    async def _project_top_sql_window_result(
        self, *, uow, turn, task, artifact, payload: dict[str, Any], now: datetime,
    ) -> None:
        """登记派生快照结果和其真实 Artifact，不把数据缺口登记为空排行。"""
        result = dict(payload.get("result") or {})
        intervals = tuple(result.get("covered_intervals") or ())
        tool_row = next(
            (item for item in await uow.turns.list_tool_invocations(turn_id=turn.turn_id, lock=True)
             if item.ops_task_id == task.ops_task_id and item.tool_id == "workload.sql.top_window"),
            None,
        )
        if tool_row is not None:
            tool_row.status = "SUCCEEDED" if intervals else "NO_DATA"
            tool_row.output_artifact_id = artifact.artifact_id
            tool_row.attempt_count = int(task.attempt_count)
            tool_row.completed_at = now
        existing = await uow.turns.get_evidence_by_artifact(
            turn_id=turn.turn_id, artifact_id=artifact.artifact_id,
        )
        if intervals and existing is None:
            starts = [datetime.fromisoformat(str(item["start"]).replace("Z", "+00:00")) for item in intervals]
            ends = [datetime.fromisoformat(str(item["end"]).replace("Z", "+00:00")) for item in intervals]
            await uow.turns.add_evidence(
                OpsTurnEvidenceEntity(
                    turn_evidence_id=uuid7(), turn_id=turn.turn_id,
                    artifact_id=artifact.artifact_id,
                    tool_invocation_id=tool_row.tool_invocation_id if tool_row is not None else None,
                    source_kind="DATABASE", evidence_kind="OBSERVATION", confidence=1,
                    evidence_role="SUPPORTS", measurement_semantics="SNAPSHOT_DELTA",
                    observed_at=max(ends), window_start_at=min(starts), window_end_at=max(ends),
                    freshness_status="FRESH", usage_reason="由目标历史快照计算，实际覆盖区间及缺口见证据",
                    linked_by=turn.created_by,
                )
            )
            await self._append_turn_event(
                uow, turn, event_type="evidence.added",
                payload={"source_kind": "DATABASE", "evidence_kind": "OBSERVATION",
                         "public_summary": "历史 SQL 窗口快照依据已加入本轮调查"},
            )
        await self._append_turn_event(
            uow, turn, event_type="tool.completed" if intervals else "tool.gap",
            payload={"tool_id": "workload.sql.top_window", "status": "SUCCEEDED" if intervals else "NO_DATA",
                     "public_summary": "历史 SQL 窗口计算已完成" if intervals else "历史 SQL 快照不足"},
        )

    async def _project_tool_result(
        self,
        *,
        uow,
        turn,
        task,
        artifact,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        result = DbaToolResult.model_validate(payload)
        future = dict(getattr(artifact, "provenance_json", None) or {}).get("future_observation") or {}
        if future.get("status") == "SCHEDULED":
            await self._append_turn_event(
                uow, turn, event_type="evidence.future_observation.scheduled",
                payload={"background_run_id": future["background_run_id"], "requested": future["requested"],
                    "public_summary": "已安排后台未来观察；可查看进度、已采集结果或取消，不覆盖过去窗口"},
            )
        invocation = await uow.turns.get_playbook_invocation_by_task(
            ops_task_id=task.ops_task_id,
            lock=True,
        )
        if invocation is not None:
            if invocation.turn_id != turn.turn_id:
                raise state_conflict("Playbook Task 不属于当前 Turn")
            invocation.status = result.status
            invocation.output_artifact_id = artifact.artifact_id
            invocation.attempt_count = int(task.attempt_count)
            invocation.completed_at = now
        tool_rows = await uow.turns.list_tool_invocations(
            turn_id=turn.turn_id,
            lock=True,
        )
        outcomes_by_tool = {item.tool_id: item for item in result.tool_outcomes}
        if result.sampling is not None:
            # 追加采样后续缺口不能抹去前面已取得的真实观测。
            for item in result.tool_outcomes:
                if item.observation is not None:
                    outcomes_by_tool[item.tool_id] = item
        for tool_row in tool_rows:
            if tool_row.ops_task_id != task.ops_task_id:
                continue
            outcome = outcomes_by_tool.get(tool_row.tool_id)
            if outcome is None:
                continue
            tool_row.status = (
                "PARTIAL" if result.sampling is not None and result.status == "PARTIAL"
                else "SKIPPED" if outcome.status == "SKIPPED"
                else "SUCCEEDED" if outcome.observation is not None
                else "FAILED" if outcome.gap is not None
                else "NO_DATA"
            )
            tool_row.output_artifact_id = artifact.artifact_id
            tool_row.attempt_count = int(task.attempt_count)
            tool_row.completed_at = now
            await self._append_turn_event(
                uow,
                turn,
                event_type=(
                    "tool.completed"
                    if outcome.observation is not None or outcome.status == "SKIPPED"
                    else "tool.gap"
                ),
                payload={
                    "tool_invocation_id": str(tool_row.tool_invocation_id),
                    "tool_id": tool_row.tool_id,
                    "status": tool_row.status,
                    "public_summary": (
                        f"按需采样完成 {result.sampling.completed_count}/{result.sampling.requested.count} 次；仅代表实际采集时点"
                        if result.sampling is not None
                        else "本步骤按计划跳过"
                        if outcome.status == "SKIPPED"
                        else "数据库只读观测已经完成"
                        if outcome.observation is not None
                        else "数据库只读观测未取得有效证据"
                    ),
                },
            )
        observations = tuple(
            item.observation
            for item in result.tool_outcomes
            if is_turn_evidence_outcome(result, item)
        )
        existing = await uow.turns.get_evidence_by_artifact(
            turn_id=turn.turn_id,
            artifact_id=artifact.artifact_id,
        )
        if observations and existing is None:
            observed_at = max(item.captured_at for item in observations)
            evidence_tool_ids = {
                item.tool_id
                for item in result.tool_outcomes
                if is_turn_evidence_outcome(result, item)
            }
            evidence_tool_row = next(
                (row for row in tool_rows if row.tool_id in evidence_tool_ids),
                None,
            )
            await uow.turns.add_evidence(
                OpsTurnEvidenceEntity(
                    turn_evidence_id=uuid7(),
                    turn_id=turn.turn_id,
                    artifact_id=artifact.artifact_id,
                    tool_invocation_id=(
                        evidence_tool_row.tool_invocation_id
                        if evidence_tool_row is not None
                        else None
                    ),
                    source_kind="DATABASE",
                    evidence_kind="OBSERVATION",
                    confidence=1,
                    evidence_role="SUPPORTS",
                    measurement_semantics=str(
                        result.measurement_semantics
                    ),
                    observed_at=observed_at,
                    freshness_status="FRESH",
                    usage_reason=(
                        f"用于回答本轮问题的 {result.source_id} 受控观测"
                    ),
                    linked_by="aiops.turn-projector",
                )
            )
            await self._append_turn_event(
                uow,
                turn,
                event_type="evidence.added",
                payload={
                    "source_kind": "DATABASE",
                    "evidence_kind": "OBSERVATION",
                    "public_summary": "新的数据库诊断依据已加入本轮调查",
                },
            )
        if invocation is not None:
            await self._append_turn_event(
                uow,
                turn,
                event_type="playbook.completed",
                payload={
                    "playbook_invocation_id": str(
                        invocation.playbook_invocation_id
                    ),
                    "playbook_id": result.source_id,
                    "status": result.status,
                    "public_summary": "一组数据库只读观测已经完成",
                },
            )

    async def _project_monitoring_result(
        self,
        *,
        uow,
        turn,
        artifact,
        payload: dict[str, Any],
    ) -> None:
        """把本轮 Prometheus 观测登记为可引用的 Turn 证据。"""
        result = ObservationSet.model_validate(payload)
        monitoring_tool_row = None
        for tool_row in await uow.turns.list_tool_invocations(
            turn_id=turn.turn_id,
            lock=True,
        ):
            if tool_row.tool_id == "monitor.query_range":
                monitoring_tool_row = tool_row
                tool_row.status = (
                    "SUCCEEDED" if result.observations else "NO_DATA"
                )
                tool_row.output_artifact_id = artifact.artifact_id
                tool_row.completed_at = result.collected_at
        existing = await uow.turns.get_evidence_by_artifact(
            turn_id=turn.turn_id,
            artifact_id=artifact.artifact_id,
        )
        if result.observations and existing is None:
            window_start = min(
                item.window_start for item in result.observations
            )
            window_end = max(
                item.window_end for item in result.observations
            )
            await uow.turns.add_evidence(
                OpsTurnEvidenceEntity(
                    turn_evidence_id=uuid7(),
                    turn_id=turn.turn_id,
                    artifact_id=artifact.artifact_id,
                    tool_invocation_id=(
                        monitoring_tool_row.tool_invocation_id
                        if monitoring_tool_row is not None
                        else None
                    ),
                    source_kind="MONITORING",
                    evidence_kind="TIME_SERIES",
                    confidence=1,
                    evidence_role="SUPPORTS",
                    measurement_semantics="HISTORICAL_SAMPLES",
                    observed_at=result.collected_at,
                    window_start_at=window_start,
                    window_end_at=window_end,
                    freshness_status="FRESH",
                    usage_reason="用于回答本轮问题的监控时间序列",
                    linked_by="aiops.turn-projector",
                )
            )
            await self._append_turn_event(
                uow,
                turn,
                event_type="evidence.added",
                payload={
                    "source_kind": "MONITORING",
                    "evidence_kind": "TIME_SERIES",
                    "public_summary": "新的监控诊断依据已加入本轮调查",
                },
            )
        await self._append_turn_event(
            uow,
            turn,
            event_type="tool.completed",
            payload={
                "source_id": result.source_id,
                "observation_count": len(result.observations),
                "gap_count": len(result.gaps),
                "public_summary": "监控时间序列查询已经完成",
            },
        )

    async def _project_log_result(
        self,
        *,
        uow,
        turn,
        artifact,
        payload: dict[str, Any],
    ) -> None:
        """把 Loki 日志查询登记为可引用的 Turn 证据。"""
        from aiops_agent.contracts.evidence import LogEvidenceSet

        result = LogEvidenceSet.model_validate(payload)
        log_tool_row = None
        for tool_row in await uow.turns.list_tool_invocations(
            turn_id=turn.turn_id,
            lock=True,
        ):
            if tool_row.tool_id == "loki.query_range":
                log_tool_row = tool_row
                tool_row.status = "SUCCEEDED" if result.entries else "NO_DATA"
                tool_row.output_artifact_id = artifact.artifact_id
                tool_row.completed_at = result.collected_at
        existing = await uow.turns.get_evidence_by_artifact(
            turn_id=turn.turn_id,
            artifact_id=artifact.artifact_id,
        )
        if result.entries and existing is None:
            await uow.turns.add_evidence(
                OpsTurnEvidenceEntity(
                    turn_evidence_id=uuid7(),
                    turn_id=turn.turn_id,
                    artifact_id=artifact.artifact_id,
                    tool_invocation_id=(
                        log_tool_row.tool_invocation_id
                        if log_tool_row is not None
                        else None
                    ),
                    source_kind="LOG",
                    evidence_kind="LOG_ENTRY",
                    confidence=1,
                    evidence_role="SUPPORTS",
                    measurement_semantics="HISTORICAL_SAMPLES",
                    observed_at=result.collected_at,
                    window_start_at=result.window_start,
                    window_end_at=result.window_end,
                    freshness_status="FRESH",
                    usage_reason="用于回答本轮问题的日志证据",
                    linked_by="aiops.turn-projector",
                )
            )
            await self._append_turn_event(
                uow,
                turn,
                event_type="evidence.added",
                payload={
                    "source_kind": "LOG",
                    "evidence_kind": "LOG_ENTRY",
                    "public_summary": "新的日志依据已加入本轮调查",
                },
            )
        await self._append_turn_event(
            uow,
            turn,
            event_type="tool.completed",
            payload={
                "source_id": result.source_id,
                "entry_count": len(result.entries),
                "gap_count": len(result.gaps),
                "public_summary": "日志查询已经完成",
            },
        )

    async def _project_attachment_search_result(
        self,
        *,
        uow,
        turn,
        task,
        artifact,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        """把受控附件检索登记为用户提供的可引用证据。"""
        from aiops_agent.workers.attachment_handlers import AttachmentEvidenceSet

        result = AttachmentEvidenceSet.model_validate(payload)
        tool_row = next(
            (
                item
                for item in await uow.turns.list_tool_invocations(
                    turn_id=turn.turn_id, lock=True
                )
                if item.tool_id == "artifact.search"
                and str(item.action_id)
                == str(task.task_key).removeprefix("attachment:").split(":", 1)[0]
                and str(dict(item.input_json or {}).get("upload_id"))
                == result.upload_id
            ),
            None,
        )
        if tool_row is not None:
            tool_row.status = "SUCCEEDED" if result.matches else "NO_DATA"
            tool_row.output_artifact_id = artifact.artifact_id
            tool_row.completed_at = now
        existing = await uow.turns.get_evidence_by_artifact(
            turn_id=turn.turn_id,
            artifact_id=artifact.artifact_id,
        )
        if result.matches and existing is None:
            await uow.turns.add_evidence(
                OpsTurnEvidenceEntity(
                    turn_evidence_id=uuid7(),
                    turn_id=turn.turn_id,
                    artifact_id=artifact.artifact_id,
                    tool_invocation_id=(
                        tool_row.tool_invocation_id if tool_row is not None else None
                    ),
                    source_kind="USER",
                    evidence_kind="USER_FILE",
                    confidence=1,
                    evidence_role="SUPPORTS",
                    measurement_semantics="NOT_APPLICABLE",
                    observed_at=now,
                    freshness_status="UNKNOWN",
                    usage_reason=(
                        f"用户上传的 {result.file_name} 已通过受控检索"
                    ),
                    linked_by="aiops.attachment-search",
                )
            )
        await self._append_turn_event(
            uow,
            turn,
            event_type="tool.completed",
            payload={
                "source_id": "user.uploaded-diagnostic-material",
                "match_count": len(result.matches),
                "public_summary": "上传诊断材料检索已经完成",
            },
        )

    async def _project_turn_answer(
        self,
        *,
        uow,
        turn,
        artifact,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        result = AIOpsTurnResult.model_validate(payload)
        prior = await uow.turns.get_message_by_artifact(
            turn_id=turn.turn_id,
            artifact_id=artifact.artifact_id,
        )
        if prior is not None:
            return
        conversation = await uow.conversations.get_conversation(
            domain_id=int(turn.domain_id),
            conversation_id=turn.conversation_id,
            lock=True,
        )
        if conversation is None:
            raise resource_not_found("Conversation")
        conversation.last_message_no = int(conversation.last_message_no) + 1
        conversation.updated_by = turn.created_by
        conversation.updated_at = now
        markdown = "\n\n".join(
            str(block.payload.get("markdown", ""))
            for block in result.blocks
            if str(block.block_type)
            in {
                "MARKDOWN",
                "ANALYSIS_MARKDOWN",
                "SOLUTION_MARKDOWN",
            }
        ).strip()
        message_id = uuid7()
        await uow.turns.add_message(
            OpsConversationMessageEntity(
                message_id=message_id,
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
                sequence_no=conversation.last_message_no,
                role="AGENT",
                message_type="ASSISTANT_MESSAGE",
                payload_schema="AIOPS_ASSISTANT_MESSAGE.v1",
                payload_json={"text": markdown},
                artifact_id=artifact.artifact_id,
                created_by="aiops.answer-composer",
            )
        )
        evidence_rows = await uow.turns.list_evidence(turn_id=turn.turn_id)
        evidence_by_artifact = {
            str(row.artifact_id): row for row in evidence_rows
        }
        if not result.answer_streamed:
            for offset in range(0, len(markdown), 120):
                await self._append_turn_event(
                    uow,
                    turn,
                    event_type="answer.delta",
                    payload={"delta": markdown[offset:offset + 120]},
                )
        for block_no, block in enumerate(result.blocks, start=1):
            block_payload = dict(block.payload)
            answer_block_id = uuid7()
            await uow.turns.add_answer_block(
                OpsAnswerBlockEntity(
                    answer_block_id=answer_block_id,
                    turn_id=turn.turn_id,
                    message_id=message_id,
                    block_no=block_no,
                    block_type=str(block.block_type),
                    schema_version=block.schema_version,
                    payload_json=block_payload,
                    content_hash=sha256_json(block_payload),
                )
            )
            cited: set[UUID] = set()
            for reference in block.evidence_refs:
                artifact_id = reference.removeprefix("artifact:").split(
                    "#", 1
                )[0]
                evidence = evidence_by_artifact.get(artifact_id)
                if evidence is None:
                    raise state_conflict(
                        "回答引用的证据尚未投影到当前 Turn"
                    )
                if evidence.turn_evidence_id in cited:
                    continue
                cited.add(evidence.turn_evidence_id)
                reference_label = (
                    reference.split("#", 1)[1]
                    if "#" in reference
                    else f"证据 {len(cited)}"
                )
                await uow.turns.add_answer_citation(
                    OpsAnswerCitationEntity(
                        answer_block_id=answer_block_id,
                        citation_no=len(cited),
                        turn_evidence_id=evidence.turn_evidence_id,
                        label=reference_label,
                    )
                )
            await self._append_turn_event(
                uow,
                turn,
                event_type="answer.block",
                payload={
                    "answer_block_id": str(answer_block_id),
                    "block_no": block_no,
                    "block_type": str(block.block_type),
                    "schema_version": block.schema_version,
                    "payload": block_payload,
                    "citation_count": len(cited),
                },
                answer_block_id=answer_block_id,
            )
        turn.status = result.status
        turn.sufficiency_status = str(result.sufficiency_status)
        if result.status in {"COMPLETED", "PARTIAL"}:
            turn.completed_at = now
        await self._append_turn_event(
            uow,
            turn,
            event_type="answer.completed",
            payload={"answer_block_count": len(result.blocks)},
        )
        await self._append_turn_event(
            uow,
            turn,
            event_type="turn.status",
            payload={"status": result.status},
        )

    @staticmethod
    async def _append_turn_event(
        uow,
        turn,
        *,
        event_type: str,
        payload: dict[str, Any],
        answer_block_id: UUID | None = None,
    ) -> None:
        turn.event_cursor = int(turn.event_cursor) + 1
        await uow.turns.add_event(
            OpsTurnEventEntity(
                turn_id=turn.turn_id,
                sequence_no=turn.event_cursor,
                event_type=event_type,
                event_key=(
                    f"{event_type}:{turn.turn_id}:{turn.event_cursor}"
                ),
                visibility="USER",
                answer_block_id=answer_block_id,
                payload_json=payload,
            )
        )

    async def _publish_turn_inspection_report(
        self,
        *,
        uow,
        run,
        task,
        source_artifact,
        now: datetime,
        trace_id: str,
    ):
        """把标准 Agent Turn 的最终回答发布为巡检报告。"""
        assert uow.inspections is not None
        plan = dict(run.plan_snapshot_json or {})
        inspection = dict(
            plan.get("client_metadata", {}).get("inspection", {})
        )
        schedule_type = str(inspection.get("schedule_type") or "CRON")
        report_type = {
            "DAILY": "INSPECTION_DAILY",
            "WEEKLY": "INSPECTION_WEEKLY",
        }.get(schedule_type, "INSPECTION_CUSTOM")
        report_key = {
            "DAILY": "inspection.daily",
            "WEEKLY": "inspection.weekly",
        }.get(schedule_type, "inspection.custom")
        title = {
            "DAILY": "数据库日常巡检报告",
            "WEEKLY": "数据库周度巡检报告",
        }.get(schedule_type, "数据库定期巡检报告")
        report_template = resolve_report_type_template(report_type)
        if report_template is None:
            raise state_conflict("巡检报告缺少可重现的系统模板")
        period_start = datetime.fromisoformat(
            str(inspection["period_start"])
        )
        period_end = datetime.fromisoformat(str(inspection["period_end"]))
        source = AIOpsTurnResult.model_validate(
            dict(source_artifact.payload_json or {})
        )
        markdown = "\n\n".join(
            str(block.payload.get("markdown") or "").strip()
            for block in source.blocks
            if str(block.block_type)
            in {"MARKDOWN", "ANALYSIS_MARKDOWN", "SOLUTION_MARKDOWN"}
        ).strip()
        facts, gaps, coverage_summary = self._inspection_report_projection(
            inspection=inspection,
            source=source,
            database_type=str(
                dict(plan.get("target") or {}).get("db_type") or "ORACLE"
            ),
            action_tool_ids={
                str(item.get("action_id") or ""): str(
                    item.get("tool_id") or ""
                )
                for item in dict(
                    plan.get("answer_context", {})
                ).get("investigation_plan", {}).get("actions", ())
                if isinstance(item, dict)
                and str(item.get("action_id") or "")
                and str(item.get("tool_id") or "")
            },
        )
        if markdown:
            facts = (
                *facts,
                {
                    "kind": "agent_health_inspection",
                    "summary": "巡检结论",
                    "markdown": markdown,
                    "sufficiency_status": str(source.sufficiency_status),
                },
            )
        inspection_checks = tuple(
            item for item in facts
            if str(item.get("kind") or "") == "inspection_check"
        )
        check_status_counts: dict[str, int] = {}
        for item in inspection_checks:
            check_status = str(item.get("check_status") or "UNKNOWN")
            check_status_counts[check_status] = (
                check_status_counts.get(check_status, 0) + 1
            )
        tool_execution_covered_count = sum(
            bool(item.get("evidence_refs"))
            for item in inspection_checks
        )
        tool_execution_total_count = len(inspection_checks)
        tool_execution_coverage_percent = round(
            tool_execution_covered_count * 100 / tool_execution_total_count,
            2,
        ) if tool_execution_total_count else 0.0
        semantic_check_ids: set[str] = set()
        unsatisfied_check_ids: set[str] = set()
        check_ids_by_tool: dict[str, set[str]] = {}
        for item in inspection_checks:
            tool_id = str(item.get("tool_id") or "")
            check_ids = {
                str(value)
                for value in item.get("check_ids", ())
                if str(value)
            } or ({tool_id} if tool_id else set())
            semantic_check_ids.update(check_ids)
            check_ids_by_tool.setdefault(tool_id, set()).update(check_ids)
            if not item.get("evidence_refs"):
                unsatisfied_check_ids.update(check_ids)
        for item in gaps:
            unsatisfied_check_ids.update(
                check_ids_by_tool.get(str(item.get("step_id") or ""), set())
            )
        evidence_total_count = len(semantic_check_ids)
        evidence_covered_count = len(
            semantic_check_ids.difference(unsatisfied_check_ids)
        )
        evidence_coverage_percent = round(
            evidence_covered_count * 100 / evidence_total_count,
            2,
        ) if evidence_total_count else 0.0
        monitoring_quality = self._inspection_monitoring_quality(source)
        incomplete_statuses = {"PARTIAL", "GAP", "MISSING"}
        incomplete_check_count = sum(
            count for check_status, count in check_status_counts.items()
            if check_status in incomplete_statuses
        )
        status = (
            "READY"
            if (
                incomplete_check_count == 0
                and not gaps
            )
            else "PARTIAL"
        )
        severity_order = {
            "CRITICAL": 5, "HIGH": 4, "MEDIUM": 3,
            "LOW": 2, "INFO": 1,
        }
        finding_severities = [
            str(finding.get("severity") or "")
            for item in inspection_checks
            for finding in item.get("findings") or ()
        ]
        health_status = (
            max(
                finding_severities,
                key=lambda value: severity_order.get(value, 0),
            )
            if finding_severities
            else ("UNKNOWN" if status == "PARTIAL" else "NORMAL")
        )
        summary = (
            f"共执行 {len(inspection_checks)} 项检查："
            f"正常 {check_status_counts.get('CLEAR', 0)} 项，"
            f"需关注 {check_status_counts.get('FINDING', 0)} 项，"
            f"仅观测 {check_status_counts.get('OBSERVED', 0)} 项，"
            f"不适用 {check_status_counts.get('NOT_APPLICABLE', 0)} 项，"
            f"不完整 {incomplete_check_count} 项；"
            f"数据缺口 {len(gaps)} 项。"
        )
        recommendations = (
            self._inspection_capacity_recommendations(source)
            if schedule_type == "WEEKLY"
            else ()
        )
        if not gaps and not recommendations:
            recommendations = (
                "继续按既定周期执行该巡检模板并关注趋势变化。",
            )
        gap_categories = {
            str(item.get("category") or "DATA_SOURCE") for item in gaps
        }
        if "PRODUCT_CONTRACT" in gap_categories:
            recommendations = (
                *recommendations,
                "报告证据契约存在字段不一致，应由产品修复 Tool 输出与规则目录后重跑，不能由用户补值。",
            )
        if "MONITORING" in gap_categories:
            recommendations = (
                *recommendations,
                "监控时序覆盖不足时应检查采集目标、抓取间隔和保留期；修复采集后等待形成完整巡检窗口。",
            )
        if "SOURCE_ACCESS" in gap_categories:
            recommendations = (
                *recommendations,
                "数据库只读访问失败时，仅核对报告中列明的连接或最小查询权限，不应扩大账号授权范围。",
            )
        if "SOURCE_CAPABILITY" in gap_categories:
            recommendations = (
                *recommendations,
                "目标版本或数据库能力不支持该检查时，应调整模板适用范围或选用对应版本的固定查询。",
            )
        if gap_categories & {"DATA_SOURCE", "REPORT_CONTRACT"}:
            recommendations = (
                *recommendations,
                "数据库查询无数据时，应先核对查询窗口、数据库角色和功能是否实际启用；确认读取方式正确后再判定为真实缺失。",
            )
        recommendations = tuple(dict.fromkeys(recommendations))
        security_level = await self._report_security_level(
            uow=uow,
            run=run,
            plan=plan,
        )
        monitoring_profile = dict(
            dict(plan.get("monitoring") or {}).get("monitoring_profile") or {}
        )
        monitoring_quality = self._inspection_monitoring_quality(source)
        target_snapshot = dict(plan.get("target") or {})
        current_target = None
        target_repository = getattr(uow, "targets", None)
        run_domain_id = getattr(run, "domain_id", None)
        if target_repository is not None and run_domain_id is not None:
            current_target = await target_repository.get_scoped(
                target_id=run.target_id,
                domain_id=run_domain_id,
            )
        target_display_name = _human_target_display_name(
            getattr(current_target, "display_name", None),
            target_snapshot.get("display_name"),
            target_id=run.target_id,
        )
        content = ReportContent(
            report_key=report_key,
            report_type=report_type,
            ops_run_id=str(run.ops_run_id),
            target_id=str(run.target_id),
            title=title,
            status=status,
            summary=summary,
            period_start=period_start,
            period_end=period_end,
            scope={
                "inspection_fire_id": str(run.inspection_fire_id),
                "template_id": inspection["template_id"],
                "template_version": inspection["template_version"],
                "schedule_type": schedule_type,
                "timezone": inspection["timezone"],
                "inspection_coverage": coverage_summary,
                "database_type": str(
                    dict(plan.get("target") or {}).get("db_type") or "ORACLE"
                ),
                "health_status": health_status,
                "check_status_counts": check_status_counts,
                "evidence_coverage_percent": evidence_coverage_percent,
                "evidence_covered_count": evidence_covered_count,
                "evidence_total_count": evidence_total_count,
                "tool_execution_coverage_percent": (
                    tool_execution_coverage_percent
                ),
                "tool_execution_covered_count": (
                    tool_execution_covered_count
                ),
                "tool_execution_total_count": tool_execution_total_count,
                "monitoring_sample_coverage_percent": monitoring_quality[
                    "coverage_percent"
                ],
                "monitoring_metric_count": monitoring_quality["metric_count"],
                "monitoring_low_coverage_metric_count": monitoring_quality[
                    "low_coverage_metric_count"
                ],
                "evidence_gap_count": len(gaps),
                "target": {
                    "target_id": str(run.target_id),
                    "display_name": target_display_name,
                    "database_type": str(
                        target_snapshot.get("db_type") or "ORACLE"
                    ),
                    "version_code": target_snapshot.get("version_code"),
                    "environment": target_snapshot.get("environment"),
                    "db_role": target_snapshot.get("db_role"),
                },
                **(
                    {"monitoring_profile": monitoring_profile}
                    if monitoring_profile
                    else {}
                ),
            },
            facts=facts,
            gaps=gaps,
            evidence_refs=(
                {
                    "artifact_id": str(source_artifact.artifact_id),
                    "content_hash": source_artifact.content_hash,
                    "schema_version": source_artifact.schema_version,
                },
            ),
            provenance={
                "producer": "aiops.agent-turn",
                "llm_used": True,
                "source_turn_result_hash": source_artifact.content_hash,
                "template": template_snapshot(report_template),
            },
            recommendations=recommendations,
        )
        payload = content.model_dump(mode="json")
        content_hash = sha256_json(payload)
        report_artifact = await uow.runs.add_artifact(
            OpsArtifactEntity(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                artifact_key=f"report:{report_key}:v1",
                artifact_type="REPORT_CONTENT",
                schema_version="REPORT_CONTENT.v1",
                payload_json=payload,
                content_hash=content_hash,
                byte_size=len(canonical_bytes(payload)),
                provenance_json={
                    "producer": "aiops.agent-turn-report-publisher",
                    "producer_version": "1",
                    "source_artifact_id": str(source_artifact.artifact_id),
                },
                trust_level="MODEL_INFERENCE",
                security_level=security_level,
            )
        )
        report = await uow.inspections.publish_report(
            ReportEntity(
                report_id=uuid7(),
                ops_run_id=run.ops_run_id,
                target_id=run.target_id,
                report_key=report_key,
                report_version=1,
                is_current=0,
                report_type=report_type,
                title=title,
                status=status,
                period_start=period_start,
                period_end=period_end,
                template_id=report_template.template_ref,
                template_version=report_template.version,
                generated_by_task_id=task.ops_task_id,
                content_artifact_id=report_artifact.artifact_id,
                content_hash=content_hash,
                summary=summary,
                security_level=security_level,
                schema_version="REPORT_CONTENT.v1",
            )
        )
        await uow.runs.append_event(
            ops_run_id=run.ops_run_id,
            ops_task_id=task.ops_task_id,
            event_type="report.ready",
            event_key=f"report:{report.report_id}:ready",
            visibility="USER",
            payload_json={
                "report_id": str(report.report_id),
                "report_key": report_key,
                "report_type": report_type,
                "report_version": 1,
                "status": status,
                "summary": summary,
                "trace_id": trace_id,
            },
        )
        await self._add_outbox(
            uow,
            aggregate_id=report.report_id,
            event_type="OPS_REPORT_READY",
            idempotency_key=f"report:{report.report_id}:ready",
            payload={
                "report_id": str(report.report_id),
                "ops_run_id": str(run.ops_run_id),
                "report_key": report_key,
                "report_type": report_type,
                "report_version": 1,
                "status": status,
            },
            trace_id=trace_id,
            now=now,
        )
        assert uow.platform_notifications is not None
        await uow.platform_notifications.emit_report_ready(
            run=run,
            report=report,
            actor_id=run.actor_id,
        )
        return report_artifact

    @staticmethod
    def _should_publish_turn_inspection_report(*, run, artifact) -> bool:
        """所有定时巡检 Turn 完成时都发布报告，不按周期类型省略。"""
        return (
            getattr(run, "trigger_type", None) == "SCHEDULE"
            and getattr(run, "workflow_kind", None) == "INSPECTION"
            and getattr(artifact, "schema_version", None)
            == "AIOPS_TURN_RESULT.v1"
        )

    @staticmethod
    def _inspection_observed_summary(
        *,
        title: str,
        row_count: int,
        truncated: bool,
    ) -> str:
        """把已完成检查写成明确结论；有行或零行都不能直接判为正常。"""
        suffix = "（结果已截断）" if truncated else ""
        if row_count <= 0:
            return (
                f"{title}：检查已完成，本期没有需要报告的记录。"
                f"{suffix}"
            )
        return (
            f"{title}：检查已完成，采集 {row_count} 条可验证观测，"
            f"未生成需关注项。{suffix}"
        )

    @staticmethod
    def _inspection_finding_summary(
        *, title: str, findings: tuple[Any, ...]
    ) -> str:
        """把确定性 Finding 压缩成巡检检查项结论。"""
        severity_order = {
            "CRITICAL": 5,
            "HIGH": 4,
            "MEDIUM": 3,
            "LOW": 2,
            "INFO": 1,
        }
        severity_labels = {
            "CRITICAL": "严重",
            "HIGH": "高",
            "MEDIUM": "中",
            "LOW": "低",
            "INFO": "提示",
        }
        highest = max(
            (str(item.severity) for item in findings),
            key=lambda value: severity_order.get(value, 0),
        )
        impacts = tuple(
            dict.fromkeys(
                str(item.impact).strip()
                for item in findings
                if str(item.impact).strip()
            )
        )
        detail = "；".join(impacts[:3])
        suffix = f"：{detail}" if detail else "。"
        return (
            f"{title}：发现 {len(findings)} 项需关注项，最高级别为"
            f"{severity_labels.get(highest, highest)}{suffix}"
        )

    @staticmethod
    def _inspection_gap_summary(
        gap: dict[str, Any], *, title_by_tool: dict[str, str]
    ) -> str:
        """将内部步骤标识替换为巡检检查项名称。"""
        detail = str(gap.get("detail") or gap.get("code") or "证据不完整")
        for tool_id, title in sorted(
            title_by_tool.items(), key=lambda item: len(item[0]), reverse=True
        ):
            detail = detail.replace(tool_id, title)
        step_id = str(gap.get("step_id") or "")
        title = title_by_tool.get(step_id)
        if title and not detail.startswith(title):
            return f"{title}：{detail}"
        return detail

    @staticmethod
    def _inspection_monitoring_facts(source: AIOpsTurnResult) -> tuple[dict[str, Any], ...]:
        """把实时监控 Profile 的确定性指标行写入正式报告。"""
        facts: list[dict[str, Any]] = []
        for evidence in source.evidence:
            if evidence.tool_id != "metric.query_range":
                continue
            indexes = {
                str(column.get("name") or ""): index
                for index, column in enumerate(evidence.columns)
            }
            if not {"metric_code", "latest"} <= set(indexes):
                continue
            for row in evidence.rows:
                metric_code = str(row[indexes["metric_code"]] or "")
                if not metric_code:
                    continue
                dimensions = (
                    str(row[indexes["dimensions"]] or "-")
                    if "dimensions" in indexes
                    else "-"
                )
                latest = row[indexes["latest"]]
                unit = str(row[indexes["unit"]] or "") if "unit" in indexes else ""
                average = row[indexes["average"]] if "average" in indexes else None
                maximum = row[indexes["maximum"]] if "maximum" in indexes else None
                sample_count = row[indexes["sample_count"]] if "sample_count" in indexes else None
                coverage_ratio = row[indexes["coverage_ratio"]] if "coverage_ratio" in indexes else None
                window_start = row[indexes["window_start"]] if "window_start" in indexes else None
                window_end = row[indexes["window_end"]] if "window_end" in indexes else None
                history_elapsed_days = row[indexes["history_elapsed_days"]] if "history_elapsed_days" in indexes else None
                semantic_role = str(row[indexes["semantic_role"]] or "") if "semantic_role" in indexes else ""
                value_kind = str(row[indexes["value_kind"]] or "") if "value_kind" in indexes else ""
                daily_points: list[dict[str, Any]] = []
                if "daily_points" in indexes:
                    try:
                        parsed_points = json.loads(
                            str(row[indexes["daily_points"]] or "[]")
                        )
                    except (TypeError, ValueError):
                        parsed_points = []
                    if isinstance(parsed_points, list):
                        daily_points = [
                            dict(item) for item in parsed_points
                            if isinstance(item, dict)
                        ]
                metric_name = display_metric_name(metric_code)
                dimension_display = display_metric_dimensions(dimensions)
                details = [f"当前值 {format_metric_value(latest, unit)}"]
                if average is not None:
                    details.append(f"平均值 {format_metric_value(average, unit)}")
                if maximum is not None:
                    details.append(f"最大值 {format_metric_value(maximum, unit)}")
                if sample_count is not None:
                    details.append(f"{sample_count} 个样本")
                facts.append({
                    "kind": "inspection_monitoring_metric",
                    "title": metric_name,
                    "summary": f"{metric_name}{dimension_display}：" + "，".join(details) + "。",
                    "check_status": "OBSERVED",
                    "tool_id": "metric.query_range",
                    "metric_code": metric_code,
                    "dimensions": dimensions,
                    "latest": latest,
                    "average": average,
                    "maximum": maximum,
                    "unit": unit,
                    "sample_count": sample_count,
                    "coverage_ratio": coverage_ratio,
                    "window_start": window_start,
                    "window_end": window_end,
                    "history_elapsed_days": history_elapsed_days,
                    "semantic_role": semantic_role,
                    "value_kind": value_kind,
                    "daily_points": daily_points,
                    "evidence_ref": evidence.evidence_ref,
                })
        return tuple(facts)

    @staticmethod
    def _inspection_observation_rows(
        observations: list[Any],
        *,
        report_columns: tuple[str, ...],
        limit: int = 5,
    ) -> tuple[dict[str, Any], ...]:
        """只按检查目录白名单冻结少量可审计观测，不泄露 SQL 文本等字段。"""
        if not report_columns:
            return ()
        rows: list[dict[str, Any]] = []
        for observation in observations:
            indexes = {
                str(column.get("name") or ""): index
                for index, column in enumerate(observation.columns)
            }
            selected = [
                column for column in report_columns if column in indexes
            ]
            for raw_row in observation.rows:
                rows.append({
                    column: raw_row[indexes[column]] for column in selected
                })
                if len(rows) >= limit:
                    return tuple(rows)
        return tuple(rows)

    @staticmethod
    def _inspection_gap_category(gap: dict[str, Any]) -> str:
        """区分读取错误、访问限制、能力不支持与真实数据缺口。"""
        source_id = str(gap.get("source_id") or "")
        code = str(gap.get("code") or "").upper()
        product_contract_codes = {
            "FINDING_COLUMN_MISSING",
            "FINDING_VALUE_UNPARSEABLE",
            "FINDING_RATIO_UNDEFINED",
            "OUTPUT_COLUMNS_MISMATCH",
            "OUTPUT_COLUMN_LIMIT_EXCEEDED",
            "OUTPUT_VALUE_TYPE_UNSUPPORTED",
            "OUTPUT_COLUMN_TYPE_MISMATCH",
            "OUTPUT_SCHEMA_INVALID",
            "QUERY_COLUMN_INVALID",
            "QUERY_COLUMN_AMBIGUOUS",
            "QUERY_SYNTAX_INVALID",
        }
        source_access_codes = {
            "AUTH_FAILED",
            "PRIVILEGE_MISSING",
            "PRIVILEGE_PROBE_CACHED_DENIAL",
            "CONNECTION_FAILED",
            "SOURCE_UNREACHABLE",
            "TIMEOUT",
        }
        source_capability_codes = {
            "VERSION_UNSUPPORTED",
            "QUERY_INCOMPATIBLE",
            "QUERY_OBJECT_UNAVAILABLE",
            "SOURCE_QUERY_UNSUPPORTED",
            "INSPECTION_FIXED_TOOL_UNAVAILABLE",
        }
        if source_id == "finding.compiler" or code in product_contract_codes:
            return "PRODUCT_CONTRACT"
        if source_id.startswith("monitoring."):
            return "MONITORING"
        if code in source_access_codes:
            return "SOURCE_ACCESS"
        if code in source_capability_codes:
            return "SOURCE_CAPABILITY"
        if source_id.startswith("inspection.report"):
            return "REPORT_CONTRACT"
        return "DATA_SOURCE"

    @staticmethod
    def _inspection_monitoring_quality(
        source: AIOpsTurnResult,
    ) -> dict[str, Any]:
        """把数据库检查覆盖和监控采样覆盖分开统计。"""
        coverage_values: list[float] = []
        for evidence in source.evidence:
            if evidence.tool_id != "metric.query_range":
                continue
            indexes = {
                str(column.get("name") or ""): index
                for index, column in enumerate(evidence.columns)
            }
            coverage_index = indexes.get("coverage_ratio")
            if coverage_index is None:
                continue
            for row in evidence.rows:
                raw_value = row[coverage_index]
                if raw_value is None:
                    continue
                coverage_values.append(
                    min(1.0, max(0.0, float(raw_value)))
                )
        return {
            "coverage_percent": (
                round(min(coverage_values) * 100, 2)
                if coverage_values
                else None
            ),
            "metric_count": len(coverage_values),
            "low_coverage_metric_count": sum(
                value < 0.8 for value in coverage_values
            ),
        }

    @staticmethod
    def _inspection_report_projection(
        *,
        inspection: dict[str, Any],
        source: AIOpsTurnResult,
        action_tool_ids: dict[str, str],
        database_type: str | None = None,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], str]:
        """把冻结模板的每项检查投影为报告条目，不能因正常零行而省略。"""
        database_type = str(
            database_type or inspection.get("database_type") or "ORACLE"
        ).upper()
        steps = tuple(
            item
            for item in inspection.get("evidence_steps", ())
            if isinstance(item, dict) and str(item.get("tool_id") or "")
        )
        evidence_by_tool: dict[str, list] = {}
        for item in source.evidence:
            evidence_by_tool.setdefault(item.tool_id, []).append(item)
        title_by_tool = {
            str(item["tool_id"]): str(item.get("title") or item["tool_id"])
            for item in steps
        }
        gaps = []
        for evidence_gap in source.evidence_gaps:
            item = evidence_gap.model_dump(mode="json")
            if item.get("source_id") == "investigation.assessment":
                continue
            item["category"] = AIOpsRuntimeService._inspection_gap_category(
                item
            )
            gaps.append(item)
        gaps_by_step: dict[str, list[dict[str, Any]]] = {}
        for item in gaps:
            step_id = str(item.get("step_id") or "")
            tool_id = action_tool_ids.get(step_id, step_id)
            item["step_id"] = tool_id
            item["summary"] = AIOpsRuntimeService._inspection_gap_summary(
                item,
                title_by_tool=title_by_tool,
            )
            gaps_by_step.setdefault(tool_id, []).append(item)

        facts: list[dict[str, Any]] = []
        for identity in evidence_by_tool.get("db.instance.identity", ()):
            identity_rows = AIOpsRuntimeService._inspection_observation_rows(
                [identity],
                report_columns=(
                    "product", "version", "instance_name",
                    "database_role", "server_time",
                ),
                limit=1,
            )
            if identity_rows:
                facts.append({
                    "kind": "inspection_identity",
                    "title": "巡检目标",
                    "summary": "已确认数据库实例身份和服务端时间。",
                    "check_status": "OBSERVED",
                    "tool_id": "db.instance.identity",
                    "observations": identity_rows,
                    "evidence_refs": [identity.evidence_ref],
                })
        covered_count = 0
        for step in steps:
            tool_id = str(step["tool_id"])
            title = str(step.get("title") or tool_id)
            observations = evidence_by_tool.get(tool_id, [])
            step_gaps = gaps_by_step.get(tool_id, [])
            if observations:
                covered_count += 1
                row_count = sum(item.row_count for item in observations)
                truncated = any(item.truncated for item in observations)
                report_rows = AIOpsRuntimeService._inspection_observation_rows(
                    observations,
                    report_columns=tuple(step.get("report_columns") or ()),
                )
                compilation = compile_findings(
                    tuple(observations),
                    database_type=database_type,
                )
                for finding_gap in compilation.gaps:
                    gap = {
                        "source_id": "finding.compiler",
                        "step_id": tool_id,
                        "code": finding_gap.code,
                        "detail": finding_gap.detail,
                        "retryable": False,
                        "category": "PRODUCT_CONTRACT",
                    }
                    gap["summary"] = (
                        AIOpsRuntimeService._inspection_gap_summary(
                            gap,
                            title_by_tool=title_by_tool,
                        )
                    )
                    if not any(
                        item.get("code") == gap["code"]
                        and item.get("step_id") == tool_id
                        and item.get("detail") == gap["detail"]
                        for item in gaps
                    ):
                        gaps.append(gap)
                        step_gaps.append(gap)
                zero_row_status = str(
                    step.get("zero_row_status") or "CLEAR"
                )
                if row_count == 0 and zero_row_status == "GAP":
                    zero_gap = {
                        "source_id": "inspection.report",
                        "step_id": tool_id,
                        "code": "INSPECTION_EXPECTED_ROWS_MISSING",
                        "detail": f"{title} 查询成功但未返回预期状态行",
                        "retryable": True,
                        "category": "DATA_SOURCE",
                    }
                    gaps.append(zero_gap)
                    step_gaps.append(zero_gap)
                if step_gaps:
                    gap_summary = str(step_gaps[0]["summary"])
                    if gap_summary.startswith(f"{title}："):
                        gap_summary = gap_summary[len(title) + 1:]
                    summary = (
                        f"{title}：已取得 {row_count} 条观测，但证据仍不完整："
                        f"{gap_summary}"
                    )
                    check_status = "PARTIAL"
                elif compilation.findings:
                    summary = AIOpsRuntimeService._inspection_finding_summary(
                        title=title,
                        findings=compilation.findings,
                    )
                    check_status = "FINDING"
                elif row_count == 0 and zero_row_status == "NOT_APPLICABLE":
                    summary = f"{title}：查询成功，当前未配置或不适用。"
                    check_status = "NOT_APPLICABLE"
                elif row_count == 0 and zero_row_status == "OBSERVED":
                    summary = f"{title}：查询成功，本期返回 0 条记录。"
                    check_status = "OBSERVED"
                elif compilation.empty_reasons:
                    summary = f"{title}：{compilation.empty_reasons[0]}"
                    check_status = "CLEAR"
                elif row_count == 0 and zero_row_status == "CLEAR":
                    summary = f"{title}：查询成功，未发现达到报告条件的记录。"
                    check_status = "CLEAR"
                else:
                    summary = AIOpsRuntimeService._inspection_observed_summary(
                        title=title,
                        row_count=row_count,
                        truncated=truncated,
                    )
                    check_status = "OBSERVED"
                facts.append(
                    {
                        "kind": "inspection_check",
                        "title": title,
                        "summary": summary,
                        "check_status": check_status,
                        "tool_id": tool_id,
                        "check_ids": list(step.get("check_ids") or ()),
                        "expected_evidence_kind": str(
                            step.get("expected_evidence_kind") or ""
                        ),
                        "evidence_refs": [
                            item.evidence_ref for item in observations
                        ],
                        "observation_count": row_count,
                        "observations": report_rows,
                        "truncated": truncated,
                        "findings": [
                            item.model_dump(mode="json")
                            for item in compilation.findings
                        ],
                    }
                )
                continue
            if step_gaps:
                detail = str(
                    step_gaps[0].get("summary") or "未取得有效观测"
                )
                if detail.startswith(f"{title}："):
                    detail = detail[len(title) + 1:]
                facts.append(
                    {
                        "kind": "inspection_check",
                        "title": title,
                        "summary": f"{title}：检查未完成，{detail}",
                        "check_status": "GAP",
                        "tool_id": tool_id,
                        "check_ids": list(step.get("check_ids") or ()),
                    }
                )
                continue
            missing_gap = {
                "source_id": "inspection.report",
                "step_id": tool_id,
                "code": "INSPECTION_OBSERVATION_MISSING",
                "detail": f"模板检查项未形成可验证观测：{title}",
                "retryable": False,
                "category": "REPORT_CONTRACT",
            }
            missing_gap["summary"] = missing_gap["detail"]
            gaps.append(missing_gap)
            facts.append(
                {
                    "kind": "inspection_check",
                    "title": title,
                    "summary": f"{title}：未形成可验证观测，已列入数据缺口",
                    "check_status": "MISSING",
                    "tool_id": tool_id,
                    "check_ids": list(step.get("check_ids") or ()),
                }
            )

        backup_policy = inspection.get("backup_policy")
        if isinstance(backup_policy, dict) and backup_policy.get("enabled"):
            backup_tool_id = {
                "ORACLE": "db.backup.recent_jobs",
                "POSTGRESQL": "db.postgresql.archiver.status",
                "MYSQL": "db.mysql.binlog.status",
            }.get(database_type)
            backup_observations = evidence_by_tool.get(backup_tool_id or "", [])
            if database_type == "ORACLE":
                successful_end_times: list[datetime] = []
                for observation in backup_observations:
                    indexes = {
                        str(column.get("name") or "").lower(): index
                        for index, column in enumerate(observation.columns)
                    }
                    status_index = indexes.get("status")
                    end_time_index = indexes.get("end_time")
                    if status_index is None or end_time_index is None:
                        continue
                    for row in observation.rows:
                        if not str(row[status_index] or "").upper().startswith("COMPLETED"):
                            continue
                        raw_end_time = row[end_time_index]
                        if isinstance(raw_end_time, datetime):
                            successful_end_times.append(raw_end_time)
                        elif raw_end_time:
                            try:
                                successful_end_times.append(datetime.fromisoformat(str(raw_end_time).replace("Z", "+00:00")))
                            except ValueError:
                                continue
                if successful_end_times:
                    latest_success = max(successful_end_times)
                    if latest_success.tzinfo is None:
                        latest_success = latest_success.replace(tzinfo=UTC)
                    reference_time = datetime.fromisoformat(str(inspection.get("period_end")))
                    compliant = reference_time - latest_success <= timedelta(
                        hours=int(backup_policy["success_max_age_hours"])
                    )
                    facts.append({
                        "kind": "backup_policy_check",
                        "title": "备份策略校验",
                        "summary": f"最近成功 RMAN 备份时间为 {latest_success.isoformat()}，策略要求 {backup_policy['success_max_age_hours']} 小时内成功。",
                        "check_status": "COMPLIANT" if compliant else "BREACHED",
                        "tool_id": backup_tool_id,
                        "policy": backup_policy,
                    })
                    if not compliant:
                        gaps.append({
                            "source_id": "inspection.backup-policy",
                            "step_id": backup_tool_id,
                            "code": "BACKUP_SUCCESS_STALE",
                            "detail": "最近成功 RMAN 备份已超过 Target 备份策略允许时效",
                            "retryable": False,
                        })
                else:
                    facts.append({
                        "kind": "backup_policy_check",
                        "title": "备份策略校验",
                        "summary": "未取得可解析的成功 RMAN 备份记录，无法证明满足备份策略。",
                        "check_status": "NOT_OBSERVED",
                        "tool_id": backup_tool_id,
                        "policy": backup_policy,
                    })
                    gaps.append({
                        "source_id": "inspection.backup-policy",
                        "step_id": backup_tool_id,
                        "code": "BACKUP_SUCCESS_NOT_FOUND",
                        "detail": "RMAN 观测中没有可解析的成功备份记录",
                        "retryable": False,
                        "category": "DATA_SOURCE",
                    })
            else:
                facts.append({
                    "kind": "backup_policy_check",
                    "title": "备份策略校验",
                    "summary": "已检查数据库归档链，但当前证据不能证明第三方全量备份在策略时效内成功。",
                    "check_status": "NOT_OBSERVABLE",
                    "tool_id": backup_tool_id,
                    "policy": backup_policy,
                })
                gaps.append({
                    "source_id": "inspection.backup-policy",
                    "step_id": backup_tool_id,
                    "code": "BACKUP_SUCCESS_NOT_OBSERVABLE",
                    "detail": "归档链观测不能替代备份平台成功记录，需要接入对应备份来源证据",
                    "retryable": False,
                    "category": "DATA_SOURCE",
                })

        if str(inspection.get("schedule_type") or "") == "WEEKLY":
            for trend in extract_metric_trend_rows(source.evidence):
                metric_name = display_metric_name(trend["metric_code"])
                dimension_display = display_metric_dimensions(trend["dimensions"])
                unit = trend.get("unit", "")
                facts.append(
                    {
                        "kind": "inspection_trend",
                        "title": f"{metric_name}趋势",
                        "summary": (
                            f"{metric_name}{dimension_display}："
                            f"期初值 {format_metric_value(trend['first'], unit)}，"
                            f"当前值 {format_metric_value(trend['latest'], unit)}，"
                            f"日均变化量 {format_metric_value(trend['change_per_day'], unit)}。"
                        ),
                        "check_status": "OBSERVED",
                        "tool_id": "metric.query_range",
                        **trend,
                    }
                )

        facts.extend(AIOpsRuntimeService._inspection_monitoring_facts(source))

        if not steps:
            facts.append(
                {
                    "kind": "inspection_coverage",
                    "summary": "本期巡检未记录冻结的勾选检查项。",
                    "check_status": "MISSING",
                }
            )
            gaps.append(
                {
                    "source_id": "inspection.report",
                    "step_id": "template",
                    "code": "INSPECTION_TEMPLATE_STEPS_MISSING",
                    "detail": "巡检 Run 未保存冻结的勾选检查项",
                    "retryable": False,
                    "category": "REPORT_CONTRACT",
                }
            )
        coverage_summary = f"本期按模板取得 {covered_count}/{len(steps)} 个取证步骤的观测"
        if steps and covered_count == len(steps) and not gaps:
            coverage_summary += "，所有模板检查均已形成完整证据。"
        elif covered_count == len(steps):
            coverage_summary += f"；模板检查均有观测，另记录 {len(gaps)} 条报告证据缺口。"
        else:
            coverage_summary += (
                f"；{len(steps) - covered_count} 个步骤未形成可追溯观测，"
                f"共记录 {len(gaps)} 条报告证据缺口。"
            )
        return tuple(facts), tuple(gaps), coverage_summary

    @staticmethod
    def _inspection_capacity_recommendations(
        source: AIOpsTurnResult,
    ) -> tuple[str, ...]:
        """依据表空间历史趋势预测生成可执行的巡检容量建议。"""
        recommendations: list[str] = []
        forecast_found = False
        metric_evidence_found = False
        for evidence in source.evidence:
            if evidence.tool_id != "metric.query_range":
                continue
            metric_evidence_found = True
            indexes = {
                str(column.get("name")): index
                for index, column in enumerate(evidence.columns)
            }
            required = {
                "metric_code",
                "dimensions",
                "forecast_horizon_days",
                "forecast_utilization_percent",
                "estimated_days_to_limit",
            }
            if not required <= set(indexes):
                continue
            confidence_index = indexes.get("forecast_confidence")
            history_days_index = indexes.get("history_elapsed_days")
            for row in evidence.rows:
                if row[indexes["metric_code"]] != "db.storage.used_bytes":
                    continue
                projected = row[indexes["forecast_utilization_percent"]]
                horizon = row[indexes["forecast_horizon_days"]]
                remaining_days = row[indexes["estimated_days_to_limit"]]
                if not isinstance(projected, (int, float)) or not isinstance(
                    horizon, (int, float)
                ):
                    continue
                forecast_found = True
                dimensions = dict(
                    part.split("=", 1)
                    for part in str(row[indexes["dimensions"]]).split(", ")
                    if "=" in part
                )
                tablespace = dimensions.get("tablespace", "未标识表空间")
                projected_percent = float(projected)
                horizon_days = float(horizon)
                remaining = (
                    float(remaining_days)
                    if isinstance(remaining_days, (int, float))
                    else None
                )
                confidence = (
                    str(row[confidence_index])
                    if confidence_index is not None
                    and row[confidence_index] is not None
                    else "MEDIUM"
                )
                history_days = (
                    float(row[history_days_index])
                    if history_days_index is not None
                    and isinstance(row[history_days_index], (int, float))
                    else None
                )
                confidence_prefix = (
                    "低置信度预测" if confidence == "LOW" else "预测"
                )
                confidence_suffix = (
                    (
                        f"该结果仅基于约{history_days:.2f}天历史数据，"
                        "应补齐更长时间序列后复核。"
                    )
                    if confidence == "LOW" and history_days is not None
                    else (
                        "该结果历史证据较短，应补齐更长时间序列后复核。"
                        if confidence == "LOW"
                        else ""
                    )
                )
                if projected_percent >= 95 or (
                    remaining is not None and remaining <= horizon_days
                ):
                    if confidence == "LOW":
                        recommendations.append(
                            f"表空间 {tablespace} 按当前历史增速低置信度预测未来"
                            f"{horizon_days:g}天使用率约为{projected_percent:.2f}%，"
                            "建议立即核对当前容量和自动扩展上限，并准备 AUTOEXTEND 或 RESIZE。"
                            f"{confidence_suffix}"
                        )
                    else:
                        recommendations.append(
                            f"表空间 {tablespace} 按当前历史增速预测未来"
                            f"{horizon_days:g}天使用率约为{projected_percent:.2f}%，"
                            "建议立即核对自动扩展上限，并在预计耗尽日前完成 AUTOEXTEND 或 RESIZE。"
                        )
                elif projected_percent >= 85 or (
                    remaining is not None and remaining <= horizon_days * 2
                ):
                    recommendations.append(
                        f"表空间 {tablespace} {confidence_prefix}未来{horizon_days:g}天"
                        "使用率约为"
                        f"{projected_percent:.2f}%，建议本巡检周期内制定 AUTOEXTEND 或"
                        f" RESIZE 计划并提高复核频率。{confidence_suffix}"
                    )
        if metric_evidence_found and not forecast_found:
            recommendations.append(
                "本次巡检未形成可靠的表空间未来容量预测，不能只依据当前使用率"
                "判断容量方案；应补齐历史监控采样后重新评估。"
            )
        return tuple(dict.fromkeys(recommendations))

    async def _publish_inspection_report(
        self,
        *,
        uow,
        run,
        task,
        source_artifact,
        now: datetime,
        trace_id: str,
    ):
        """把 Schedule Run 终态转换为不可变报告并发布投影。"""
        assert uow.inspections is not None
        plan = dict(run.plan_snapshot_json or {})
        inspection = dict(
            plan.get("client_metadata", {}).get("inspection", {})
        )
        schedule_type = inspection.get("schedule_type")
        if schedule_type not in {"DAILY", "WEEKLY"}:
            raise validation_failed("巡检报告类型无效")
        weekly = schedule_type == "WEEKLY"
        report_type = (
            "INSPECTION_WEEKLY" if weekly else "INSPECTION_DAILY"
        )
        report_key = "inspection.weekly" if weekly else "inspection.daily"
        period_start = datetime.fromisoformat(
            str(inspection["period_start"])
        )
        period_end = datetime.fromisoformat(
            str(inspection["period_end"])
        )
        source = dict(source_artifact.payload_json or {})
        status = (
            "PARTIAL" if source.get("status") == "PARTIAL" else "READY"
        )
        title = (
            "数据库周度巡检报告" if weekly else "数据库日常巡检报告"
        )
        summary = (
            f"已完成 {int(source.get('observation_count', 0))} 项观测，"
            f"发现 {int(source.get('gap_count', 0))} 个数据缺口"
        )
        report_template = resolve_report_type_template(report_type)
        if report_template is None:
            raise state_conflict("巡检报告缺少可重现的系统模板")
        content = ReportContent(
            report_key=report_key,
            report_type=report_type,
            ops_run_id=str(run.ops_run_id),
            target_id=str(run.target_id),
            title=title,
            status=status,
            summary=summary,
            period_start=period_start,
            period_end=period_end,
            scope={
                "inspection_fire_id": str(run.inspection_fire_id),
                "template_id": inspection["template_id"],
                "template_version": inspection["template_version"],
                "schedule_type": schedule_type,
                "timezone": inspection["timezone"],
            },
            facts=(
                {
                    "kind": "database_diagnostic_coverage",
                    "observation_count": int(
                        source.get("observation_count", 0)
                    ),
                    "tools": list(source.get("tools", ())),
                },
            ),
            gaps=tuple(dict(item) for item in source.get("gaps", ())),
            evidence_refs=(
                {
                    "artifact_id": str(source_artifact.artifact_id),
                    "content_hash": source_artifact.content_hash,
                    "schema_version": source_artifact.schema_version,
                },
            ),
            provenance={
                "deterministic": True,
                "llm_used": False,
                "source_report_hash": source_artifact.content_hash,
                "source_provenance": dict(
                    source.get("provenance", {})
                ),
                "template": template_snapshot(report_template),
            },
        )
        payload = content.model_dump(mode="json")
        content_hash = sha256_json(payload)
        security_level = await self._report_security_level(
            uow=uow,
            run=run,
            plan=plan,
        )
        report_artifact = await uow.runs.add_artifact(
            OpsArtifactEntity(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                artifact_key=f"report:{report_key}:v1",
                artifact_type="REPORT_CONTENT",
                schema_version="REPORT_CONTENT.v1",
                payload_json=payload,
                content_hash=content_hash,
                byte_size=len(canonical_bytes(payload)),
                provenance_json={
                    "producer": "aiops.report-publisher",
                    "producer_version": "1",
                    "source_artifact_id": str(
                        source_artifact.artifact_id
                    ),
                },
                trust_level="SOURCE_VERIFIED",
                security_level=security_level,
            )
        )
        report = await uow.inspections.publish_report(
            ReportEntity(
                report_id=uuid7(),
                ops_run_id=run.ops_run_id,
                target_id=run.target_id,
                report_key=report_key,
                report_version=1,
                is_current=0,
                report_type=report_type,
                title=title,
                status=status,
                period_start=period_start,
                period_end=period_end,
                template_id=report_template.template_ref,
                template_version=report_template.version,
                generated_by_task_id=task.ops_task_id,
                content_artifact_id=report_artifact.artifact_id,
                content_hash=content_hash,
                summary=summary,
                security_level=security_level,
                schema_version="REPORT_CONTENT.v1",
            )
        )
        await uow.runs.append_event(
            ops_run_id=run.ops_run_id,
            ops_task_id=task.ops_task_id,
            event_type="report.ready",
            event_key=f"report:{report.report_id}:ready",
            visibility="USER",
            payload_json={
                "report_id": str(report.report_id),
                "report_key": report_key,
                "report_type": report_type,
                "report_version": 1,
                "status": status,
                "summary": summary,
                "trace_id": trace_id,
            },
        )
        await self._add_outbox(
            uow,
            aggregate_id=report.report_id,
            event_type="OPS_REPORT_READY",
            idempotency_key=f"report:{report.report_id}:ready",
            payload={
                "report_id": str(report.report_id),
                "ops_run_id": str(run.ops_run_id),
                "report_key": report_key,
                "report_type": report_type,
                "report_version": 1,
                "status": status,
            },
            trace_id=trace_id,
            now=now,
        )
        assert uow.platform_notifications is not None
        await uow.platform_notifications.emit_report_ready(
            run=run,
            report=report,
            actor_id=run.actor_id,
        )
        return report_artifact

    async def _report_security_level(
        self,
        *,
        uow,
        run,
        plan: dict[str, Any],
    ) -> int:
        """诊断产物使用自身的默认分类，不从运维目标继承知识等级。"""
        return 1

    async def _publish_diagnosis_report(
        self,
        *,
        uow,
        run,
        task,
        source_artifact,
        now: datetime,
        trace_id: str,
        template: ReportTemplate,
        actor_id: str,
        source_override: dict[str, Any] | None = None,
        period_start_override: datetime | None = None,
        period_end_override: datetime | None = None,
        period_kind: str = "AD_HOC",
    ) -> ReportEntity:
        """由用户显式请求，把终态诊断冻结为正式报告。"""
        from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
        if not await validate_investigation_sources(uow=uow, run=run, payload=dict(source_artifact.payload_json or {}),
            plan_capture_reader=getattr(self, "_plan_capture_reader", None)):
            raise state_conflict("调查计算或对象关系与原始证据不一致，不能生成正式报告")
        assert uow.inspections is not None
        plan = dict(run.plan_snapshot_json or {})
        question = str(
            plan.get("diagnosis", {}).get("question_summary") or ""
        )
        trigger_type = getattr(run, "trigger_type", "CHAT")
        source_kind = (
            "CHAT" if trigger_type == "CHAT"
            else "INSPECTION" if trigger_type == "SCHEDULE"
            else "ALERT"
        )
        source = source_override or normalize_report_source(
            schema_version=source_artifact.schema_version,
            payload=dict(source_artifact.payload_json or {}),
            source_kind=source_kind,
        )
        report_type = (
            {
                "DAILY": "INSPECTION_DAILY",
                "MONTHLY": "INSPECTION_MONTHLY",
                "QUARTERLY": "INSPECTION_QUARTERLY",
                "ANNUAL": "INSPECTION_ANNUAL",
            }.get(period_kind, "INSPECTION_CUSTOM")
            if source_kind == "INSPECTION"
            else "INCIDENT"
        )
        template = report_template_for_source(template=template, plan=plan, source_kind=source_kind)
        report_key = template.template_ref.removeprefix("system:")
        subject_ref = dict(
            dict(plan.get("answer_context") or {}).get("task_frame", {}).get("subject_ref", {})
        )
        database_prefix = {"POSTGRESQL_QUERY_ID": "postgresql", "MYSQL_DIGEST": "mysql"}.get(
            subject_ref.get("subject_type")
        )
        if source_kind == "CHAT" and database_prefix:
            report_type = f"{database_prefix.upper()}_SQL_HEALTHCHECK"
            report_key = f"{database_prefix}:sql_healthcheck"
        conversation_snapshot = dict(source.get("conversation") or {})
        source_conversation_id = None
        if conversation_snapshot.get("conversation_id"):
            try:
                source_conversation_id = UUID(
                    str(conversation_snapshot["conversation_id"])
                )
            except ValueError as exc:
                raise state_conflict("会话报告来源引用无效") from exc
        get_conversation_report = getattr(
            uow.inspections,
            "get_current_report_for_conversation_template",
            None,
        )
        get_run_report = getattr(
            uow.inspections, "get_current_report_for_run_template", None
        )
        current_report = (
            await get_conversation_report(
                conversation_id=source_conversation_id,
                template_id=template.template_ref,
                lock=True,
            )
            if source_conversation_id is not None and get_conversation_report
            else await get_run_report(
                ops_run_id=run.ops_run_id,
                template_id=template.template_ref,
                lock=True,
            )
            if get_run_report else None
        )
        report_version = (
            int(current_report.report_version) + 1
            if current_report is not None else 1
        )
        root = dict(source.get("root_cause") or {})
        grade = str(root.get("effective_level") or "INCONCLUSIVE")
        status = (
            "READY"
            if source.get("status") == "READY"
            else "PARTIAL"
        )
        rationale = str(
            source.get("diagnosis_rationale")
            or (
                "已形成可追溯诊断结论"
                if grade != "INCONCLUSIVE"
                else "当前证据不足，尚未确认根因"
            )
        )
        summary = rationale[:2000]
        solution = dict(source.get("solution") or {})
        recommendations = tuple(
            dict.fromkeys(
                str(item)
                for key in (
                    "immediate_mitigations",
                    "long_term_remediations",
                )
                for item in solution.get(key, ())
                if item
            )
        )
        inspection = dict(
            plan.get("client_metadata", {}).get("inspection", {})
        )
        period_start = period_start_override or run.created_at
        period_end = period_end_override or now
        if source_kind == "INSPECTION" and period_start_override is None:
            try:
                period_start = datetime.fromisoformat(
                    str(inspection["period_start"])
                )
                period_end = datetime.fromisoformat(
                    str(inspection["period_end"])
                )
            except (KeyError, ValueError):
                pass
        content = ReportContent(
            report_key=report_key,
            report_type=report_type,
            ops_run_id=str(run.ops_run_id),
            target_id=str(run.target_id),
            title=template.display_name,
            status=status,
            summary=summary,
            period_start=period_start,
            period_end=period_end,
            scope={
                "question_summary": question,
                "root_cause_grade": grade,
                "report_decision_reasons": list(
                    source.get("report_decision_reasons") or ()
                ),
                "effective_capabilities": dict(
                    plan.get("effective_capabilities") or {}
                ),
                "diagnosis_rationale": rationale,
                "source_kind": source_kind,
                "source_situation_id": plan.get("source_situation_id"),
                "conversation": dict(source.get("conversation") or {}),
                "inspection_coverage": source.get(
                    "inspection_coverage",
                    rationale if source_kind == "INSPECTION" else None,
                ),
                "report_period_kind": period_kind,
                "subject_ref": subject_ref,
                "report_origin": "AIOPS_GENERATED" if report_type.endswith("_SQL_HEALTHCHECK") else None,
                **({"sql_plan_assessments": list(dict(source.get("investigation_record") or {}).get("sql_plan_assessments") or ())}
                    if dict(source.get("investigation_record") or {}).get("sql_plan_assessments") else {}),
                "closure_artifact_id": (
                    str(source_artifact.artifact_id)
                    if source_artifact.schema_version == "DIAGNOSIS_CLOSURE.v1"
                    else source.get("closure_artifact_id")
                ),
                "executed_actions": list(source.get("executed_actions") or ()),
                "action_source": source.get("action_source"),
                "verification": dict(source.get("verification") or {}),
                "investigation_goals": list(source.get("investigation_goals") or ()),
                **({
                    key: source.get(key)
                    for key in (
                        "target", "scope_type", "scope_name", "target_count",
                        "database_types", "monthly_summary",
                        "previous_month_summary", "previous_month_comparison",
                        "monthly_report_extensions", "business_system_id",
                        "member_targets",
                    )
                    if source.get(key) is not None
                } if source_kind == "INSPECTION" else {}),
            },
            facts=tuple(
                dict(item) if source_kind == "INSPECTION" else {
                    **dict(item),
                    "summary": item.get("fact_summary") or item.get("summary"),
                }
                for item in source.get("facts", ())
                if isinstance(item, dict)
            ),
            gaps=tuple(
                dict(item) if isinstance(item, dict) else {"code": str(item)}
                for item in source.get("gaps", ())
            ),
            evidence_refs=(
                {
                    "artifact_id": str(source_artifact.artifact_id),
                    "content_hash": source_artifact.content_hash,
                    "schema_version": source_artifact.schema_version,
                },
                *tuple(
                    dict(item)
                    for item in source.get("evidence_refs", ())
                    if (
                        isinstance(item, dict)
                        and str(item.get("artifact_id") or "")
                        != str(source_artifact.artifact_id)
                    )
                ),
            ),
            recommendations=recommendations,
            provenance={
                "deterministic_report_decision": True,
                "report_input_hash": report_input_hash(
                    source_artifact=source_artifact, source=source, template=template,
                    period_kind=period_kind, period_start=period_start_override, period_end=period_end_override,
                ),
                "source_artifact_hash": source_artifact.content_hash,
                "model_receipt_hashes": list(
                    source.get("model_receipt_hashes") or ()
                ),
                "template": template_snapshot(template),
            },
        )
        payload = content.model_dump(mode="json")
        content_hash = sha256_json(payload)
        security_level = await self._report_security_level(
            uow=uow,
            run=run,
            plan=plan,
        )
        report_artifact = await uow.runs.add_artifact(
            OpsArtifactEntity(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                artifact_key=f"report:{report_key}:v{report_version}",
                artifact_type="REPORT_CONTENT",
                schema_version="REPORT_CONTENT.v1",
                payload_json=payload,
                content_hash=content_hash,
                byte_size=len(canonical_bytes(payload)),
                provenance_json={
                    "producer": "aiops.report-publisher",
                    "producer_version": "1",
                    "source_artifact_id": str(source_artifact.artifact_id),
                },
                trust_level="SOURCE_VERIFIED",
                security_level=security_level,
            )
        )
        report = await uow.inspections.publish_report(
            ReportEntity(
                report_id=uuid7(),
                ops_run_id=run.ops_run_id,
                source_conversation_id=source_conversation_id,
                target_id=run.target_id,
                report_key=report_key,
                report_version=report_version,
                supersedes_report_id=(
                    current_report.report_id
                    if current_report is not None else None
                ),
                is_current=0,
                report_type=report_type,
                title=content.title,
                status=status,
                period_start=period_start,
                period_end=period_end,
                result=(
                    str(dict(source.get("verification") or {}).get("result"))
                    if dict(source.get("verification") or {}).get("result")
                    in {"RESOLVED", "IMPROVED", "NOT_ACHIEVED", "ADVERSE"}
                    else None
                ),
                template_id=template.template_ref,
                template_version=template.version,
                generated_by_task_id=task.ops_task_id,
                content_artifact_id=report_artifact.artifact_id,
                content_hash=content_hash,
                summary=summary,
                security_level=security_level,
                schema_version="REPORT_CONTENT.v1",
            )
        )
        source_rows = [
            ReportSourceEntity(
                report_id=report.report_id,
                ops_run_id=run.ops_run_id,
                source_artifact_id=source_artifact.artifact_id,
                source_kind=source_kind,
                content_hash=source_artifact.content_hash,
                observed_at=getattr(run, "completed_at", None) or now,
            )
        ]
        recorded_sources = {(run.ops_run_id, source_artifact.artifact_id)}
        for evidence in source.get("evidence_refs", ()):
            if not isinstance(evidence, dict) or not evidence.get("source_run_id"):
                continue
            try:
                evidence_run_id = UUID(str(evidence["source_run_id"]))
                evidence_artifact_id = UUID(str(evidence["artifact_id"]))
                source_identity = (evidence_run_id, evidence_artifact_id)
                if source_identity in recorded_sources:
                    continue
                recorded_sources.add(source_identity)
                source_rows.append(ReportSourceEntity(
                    report_id=report.report_id,
                    ops_run_id=evidence_run_id,
                    source_artifact_id=evidence_artifact_id,
                    source_kind=str(evidence.get("source_kind") or source_kind),
                    content_hash=str(evidence["content_hash"]),
                    observed_at=datetime.fromisoformat(str(evidence["observed_at"])),
                ))
            except (KeyError, TypeError, ValueError):
                raise state_conflict("周期报告来源引用不完整") from None
        await uow.inspections.add_report_sources(source_rows)
        await uow.runs.append_event(
            ops_run_id=run.ops_run_id,
            ops_task_id=task.ops_task_id,
            event_type="report.ready",
            event_key=f"report:{report.report_id}:ready",
            visibility="USER",
            payload_json={
                "report_id": str(report.report_id),
                "report_key": report_key,
                "report_type": report_type,
                "report_version": report_version,
                "status": status,
                "summary": summary,
                "trace_id": trace_id,
            },
        )
        await self._add_outbox(
            uow,
            aggregate_id=report.report_id,
            event_type="OPS_REPORT_READY",
            idempotency_key=f"report:{report.report_id}:ready",
            payload={
                "report_id": str(report.report_id),
                "ops_run_id": str(run.ops_run_id),
                "report_key": report_key,
                "report_type": report_type,
                "report_version": report_version,
                "status": status,
            },
            trace_id=trace_id,
            now=now,
        )
        assert uow.platform_notifications is not None
        await uow.platform_notifications.emit_report_ready(
            run=run,
            report=report,
            actor_id=actor_id,
        )
        return report

    async def _publish_comparison_report(
        self,
        *,
        uow,
        run,
        task,
        verification_artifact,
        now: datetime,
        trace_id: str,
    ):
        """根据 Verification 事实确定性发布动作级处理前后对比报告。"""
        assert uow.inspections is not None
        if run.source_proposal_id is None:
            raise validation_failed("对比报告缺少来源 Proposal")
        proposal = await uow.changes.get_proposal(
            proposal_id=run.source_proposal_id
        )
        if proposal is None:
            raise validation_failed("对比报告来源 Proposal 不存在")
        source_run = await uow.runs.get_run(
            ops_run_id=proposal.ops_run_id
        )
        source_result = (
            await uow.runs.get_artifact(
                artifact_id=run.source_result_artifact_id
            )
            if run.source_result_artifact_id is not None
            else None
        )
        plan_artifact = await uow.runs.get_artifact_by_key(
            ops_run_id=proposal.ops_run_id,
            artifact_key=(
                f"comparison:proposal:{proposal.proposal_id}:plan:v1"
            ),
        )
        if (
            source_run is None
            or source_result is None
            or plan_artifact is None
            or plan_artifact.schema_version != "COMPARISON_PLAN.v1"
        ):
            raise validation_failed("对比报告来源事实链不完整")
        comparison_plan = ComparisonPlan.model_validate(
            plan_artifact.payload_json
        )
        verification = ActionVerification.model_validate(
            verification_artifact.payload_json
        )
        if (
            comparison_plan.proposal_id != str(proposal.proposal_id)
            or verification.proposal_id != str(proposal.proposal_id)
            or verification.source_run_id != str(source_run.ops_run_id)
            or verification.result_artifact_id
            != str(source_result.artifact_id)
        ):
            raise validation_failed("对比报告来源引用不匹配")
        result, rationale = self._comparison_result(verification)
        after_start = run.created_at or now
        after_end = now
        comparison = ComparisonResult(
            comparison_plan_artifact_id=str(plan_artifact.artifact_id),
            verification_artifact_id=str(
                verification_artifact.artifact_id
            ),
            proposal_id=str(proposal.proposal_id),
            source_run_id=str(source_run.ops_run_id),
            source_result_artifact_id=str(source_result.artifact_id),
            baseline_start=comparison_plan.baseline_start,
            baseline_end=comparison_plan.baseline_end,
            after_start=after_start,
            after_end=after_end,
            primary_signals={
                "target_absent": (
                    None
                    if verification.target_still_present is None
                    else not verification.target_still_present
                ),
                "blocking_absent": (
                    None
                    if verification.blocking_still_present is None
                    else not verification.blocking_still_present
                ),
            },
            gap_codes=verification.gap_codes,
            evidence_hashes=verification.evidence_hashes,
            result=result,
            rationale_codes=rationale,
            direct_effect={"VERIFIED": "ACHIEVED", "NOT_ACHIEVED": "NOT_ACHIEVED", "ADVERSE": "ADVERSE"}.get(verification.status, "UNKNOWN"),
            measured_effect="DEGRADED" if verification.status == "ADVERSE" else "INCONCLUSIVE",
        )
        comparison_payload = comparison.model_dump(mode="json")
        comparison_hash = sha256_json(comparison_payload)
        comparison_artifact = await uow.runs.add_artifact(
            OpsArtifactEntity(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                artifact_key=(
                    f"comparison:proposal:{proposal.proposal_id}:result:v1"
                ),
                artifact_type="COMPARISON_RESULT",
                schema_version="COMPARISON_RESULT.v1",
                payload_json=comparison_payload,
                content_hash=comparison_hash,
                byte_size=len(canonical_bytes(comparison_payload)),
                provenance_json={
                    "producer": "aiops.comparison-engine",
                    "producer_version": "action-effect.v1",
                    "deterministic": True,
                    "llm_used": False,
                },
                trust_level="SOURCE_VERIFIED",
                security_level=1,
            )
        )
        source_payload = dict(source_result.payload_json or {})
        execution_id = source_payload.get("execution_id")
        suffix = execution_id or str(proposal.proposal_id)
        report_key = f"comparison.action.{suffix}"
        report_status = (
            "PARTIAL" if result == "INCONCLUSIVE" else "READY"
        )
        report_template = resolve_report_type_template("COMPARISON")
        if report_template is None:
            raise state_conflict("对比报告缺少可重现的系统模板")
        summary = {
            "RESOLVED": "处理后的验证证据表明目标问题已经解决",
            "IMPROVED": "处理后的直接效果指标已改善",
            "UNCHANGED": "处理后的直接效果指标未发生预期变化",
            "DEGRADED": "处理后发现直接效果或护栏指标退化",
            "INCONCLUSIVE": "处理后证据不足，无法形成可靠对比结论",
        }[result]
        if comparison.direct_effect == "ACHIEVED" and result == "INCONCLUSIVE":
            summary = "动作后置条件已核验满足；业务恢复、可比性能改善及护栏尚未完成测量"
        report_content = ReportContent(
            report_key=report_key,
            report_type="COMPARISON",
            ops_run_id=str(source_run.ops_run_id),
            target_id=str(source_run.target_id),
            title="数据库处理前后对比报告",
            status=report_status,
            summary=summary,
            period_start=comparison.baseline_start,
            period_end=comparison.after_end,
            scope={
                "proposal_id": str(proposal.proposal_id),
                "execution_id": execution_id,
                "solution_group_key": proposal.solution_group_key,
                "action_template_id": proposal.action_template_id,
                "result_rule_version": (
                    comparison_plan.result_rule_version
                ),
                "verification": {
                    "method": "ACTION_POSTCONDITION", "result": comparison.result,
                    "direct_effect": comparison.direct_effect, "measured_effect": comparison.measured_effect,
                    "symptom_recovered": comparison.symptom_recovered, "guardrails_passed": comparison.guardrails_passed,
                    "attribution_strength": comparison.attribution_strength,
                },
            },
            facts=(
                {
                    "kind": "comparison_result",
                    "summary": summary,
                    "result": result,
                    "primary_signals": comparison.primary_signals,
                    "guardrail_signals": comparison.guardrail_signals,
                    "rationale_codes": list(rationale),
                    "causal_limitations": list(
                        comparison.causal_limitations
                    ),
                    "direct_effect": comparison.direct_effect,
                    "measured_effect": comparison.measured_effect,
                    "symptom_recovered": comparison.symptom_recovered,
                    "guardrails_passed": comparison.guardrails_passed,
                    "attribution_strength": comparison.attribution_strength,
                },
            ),
            gaps=tuple(
                {"code": code} for code in verification.gap_codes
            ),
            evidence_refs=(
                {
                    "artifact_id": str(plan_artifact.artifact_id),
                    "content_hash": plan_artifact.content_hash,
                    "schema_version": plan_artifact.schema_version,
                },
                {
                    "artifact_id": str(
                        verification_artifact.artifact_id
                    ),
                    "content_hash": verification_artifact.content_hash,
                    "schema_version": (
                        verification_artifact.schema_version
                    ),
                },
                {
                    "artifact_id": str(comparison_artifact.artifact_id),
                    "content_hash": comparison_hash,
                    "schema_version": "COMPARISON_RESULT.v1",
                },
                {
                    "artifact_id": str(source_result.artifact_id),
                    "content_hash": source_result.content_hash,
                    "schema_version": source_result.schema_version,
                },
            ),
            provenance={
                "deterministic": True,
                "llm_used": False,
                "comparison_result_hash": comparison_hash,
                "template": template_snapshot(report_template),
            },
        )
        report_payload = report_content.model_dump(mode="json")
        report_hash = sha256_json(report_payload)
        report_artifact = await uow.runs.add_artifact(
            OpsArtifactEntity(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                artifact_key=f"report:{report_key}:v1",
                artifact_type="REPORT_CONTENT",
                schema_version="REPORT_CONTENT.v1",
                payload_json=report_payload,
                content_hash=report_hash,
                byte_size=len(canonical_bytes(report_payload)),
                provenance_json={
                    "producer": "aiops.report-publisher",
                    "producer_version": "1",
                    "comparison_result_artifact_id": str(
                        comparison_artifact.artifact_id
                    ),
                },
                trust_level="SOURCE_VERIFIED",
                security_level=int(comparison_artifact.security_level),
            )
        )
        report = await uow.inspections.publish_report(
            ReportEntity(
                report_id=uuid7(),
                ops_run_id=source_run.ops_run_id,
                target_id=source_run.target_id,
                report_key=report_key,
                report_version=1,
                is_current=0,
                report_type="COMPARISON",
                title=report_content.title,
                status=report_status,
                period_start=comparison.baseline_start,
                period_end=comparison.after_end,
                baseline_start=comparison.baseline_start,
                baseline_end=comparison.baseline_end,
                after_start=comparison.after_start,
                after_end=comparison.after_end,
                result=result,
                template_id=report_template.template_ref,
                template_version=report_template.version,
                generated_by_task_id=task.ops_task_id,
                content_artifact_id=report_artifact.artifact_id,
                content_hash=report_hash,
                summary=summary,
                security_level=int(comparison_artifact.security_level),
                schema_version="REPORT_CONTENT.v1",
            )
        )
        await uow.runs.append_event(
            ops_run_id=run.ops_run_id,
            ops_task_id=task.ops_task_id,
            event_type="report.ready",
            event_key=f"report:{report.report_id}:ready",
            visibility="USER",
            payload_json={
                "report_id": str(report.report_id),
                "report_key": report_key,
                "report_type": "COMPARISON",
                "report_version": 1,
                "status": report_status,
                "result": result,
                "summary": summary,
                "trace_id": trace_id,
            },
        )
        await self._add_outbox(
            uow,
            aggregate_id=report.report_id,
            event_type="OPS_REPORT_READY",
            idempotency_key=f"report:{report.report_id}:ready",
            payload={
                "report_id": str(report.report_id),
                "ops_run_id": str(source_run.ops_run_id),
                "report_key": report_key,
                "report_type": "COMPARISON",
                "report_version": 1,
                "status": report_status,
                "result": result,
            },
            trace_id=trace_id,
            now=now,
        )
        assert uow.platform_notifications is not None
        await uow.platform_notifications.emit_report_ready(
            run=source_run,
            report=report,
            actor_id=source_run.actor_id,
        )
        await uow.inspections.add_report_sources([
            ReportSourceEntity(
                report_id=report.report_id, ops_run_id=source.ops_run_id,
                source_artifact_id=source.artifact_id, source_kind="COMPARISON",
                content_hash=source.content_hash, observed_at=now,
            ) for source in (plan_artifact, verification_artifact, comparison_artifact, source_result)
        ])
        await self._release_next_action_proposal(
            uow=uow,
            source_run=source_run,
            source_proposal=proposal,
            verification=verification,
            now=now,
            trace_id=trace_id,
        )
        await schedule_effect_followup(
            uow=uow, run=source_run, proposal=proposal,
            verification_artifact=verification_artifact, receipt=source_result,
            now=now, trace_id=trace_id,
        )
        return report_artifact

    async def _release_next_action_proposal(
        self,
        *,
        uow,
        source_run,
        source_proposal,
        verification: ActionVerification,
        now: datetime,
        trace_id: str,
    ) -> None:
        """仅在当前动作验证成功后释放动作组中的下一条 Proposal。"""
        if verification.status != "VERIFIED":
            return
        tasks = await uow.runs.list_tasks(
            ops_run_id=source_run.ops_run_id
        )
        plan_task = next(
            (item for item in tasks if item.task_key == "change:action-plan"),
            None,
        )
        proposal_task = next(
            (
                item
                for item in tasks
                if item.ops_task_id == source_proposal.ops_task_id
            ),
            None,
        )
        if (
            plan_task is None
            or plan_task.output_artifact_id is None
            or proposal_task is None
        ):
            return
        plan_artifact = await uow.runs.get_artifact(
            artifact_id=plan_task.output_artifact_id
        )
        if (
            plan_artifact is None
            or plan_artifact.schema_version != "ACTION_PLAN.v1"
        ):
            raise validation_failed("后续受控动作缺少冻结 Action Plan")
        plan = ActionPlan.model_validate(plan_artifact.payload_json)
        next_ordinal = int(source_proposal.command_ordinal) + 1
        action = next(
            (item for item in plan.actions if item.ordinal == next_ordinal),
            None,
        )
        if action is None:
            return
        existing = await uow.changes.get_proposal_by_ordinal(
            ops_run_id=source_run.ops_run_id,
            solution_group_key=plan.solution_group_key,
            command_ordinal=next_ordinal,
        )
        if existing is not None:
            return
        target_snapshot = dict(
            (source_run.plan_snapshot_json or {}).get("target")
            or dict(
                (source_run.plan_snapshot_json or {}).get(
                    "change_context", {}
                )
            ).get("target", {})
        )
        snapshot = build_proposal_snapshot(
            plan=plan,
            action=action,
            run_id=str(source_run.ops_run_id),
            task_id=str(source_proposal.ops_task_id),
            target_id=str(source_run.target_id),
            target_version=int(target_snapshot["row_version"]),
            now=now,
        )
        outcome = ProposalOutcome(status="CREATED", proposal=snapshot)
        payload = outcome.model_dump(mode="json")
        snapshot_artifact = await uow.runs.add_artifact(
            OpsArtifactEntity(
                ops_run_id=source_run.ops_run_id,
                ops_task_id=source_proposal.ops_task_id,
                artifact_key=(
                    f"proposal:{plan.solution_group_key}:"
                    f"ordinal:{next_ordinal}:v1"
                ),
                artifact_type="PROPOSAL_OUTCOME",
                schema_version="PROPOSAL_OUTCOME.v1",
                payload_json=payload,
                content_hash=sha256_json(payload),
                byte_size=len(canonical_bytes(payload)),
                provenance_json={
                    "producer": "aiops.action-sequencer",
                    "producer_version": "1",
                    "source_proposal_id": str(
                        source_proposal.proposal_id
                    ),
                    "verification_status": verification.status,
                },
                trust_level="SOURCE_VERIFIED",
                security_level=1,
            )
        )
        await self._materialize_advisory_proposal(
            uow=uow,
            run=source_run,
            task=proposal_task,
            artifact=snapshot_artifact,
            payload=payload,
            trace_id=trace_id,
            now=now,
        )
        await self._append_sequenced_proposal_block(
            uow=uow,
            source_run=source_run,
            snapshot=snapshot,
        )

    async def _append_sequenced_proposal_block(
        self, *, uow, source_run, snapshot
    ) -> None:
        """把新释放的单条 Proposal 追加到原诊断回答中。"""
        link = await uow.turns.get_run_link_by_ops_run_id(
            ops_run_id=source_run.ops_run_id
        )
        if link is None:
            raise validation_failed("后续 Proposal 缺少所属诊断 Turn")
        turn = await uow.turns.get_turn(
            domain_id=int(source_run.domain_id),
            turn_id=link.turn_id,
            lock=True,
        )
        blocks = await uow.turns.list_answer_blocks(turn_id=link.turn_id)
        if turn is None or not blocks:
            raise validation_failed("后续 Proposal 缺少原诊断回答")
        message_id = blocks[0].message_id
        insert_at = proposal_insert_block_no(blocks)
        later_blocks = [
            item for item in blocks if int(item.block_no) >= insert_at
        ]
        if later_blocks:
            offset = max(int(item.block_no) for item in blocks) + len(
                later_blocks
            )
            for item in later_blocks:
                item.block_no = int(item.block_no) + offset
            await uow.session.flush()
            for index, item in enumerate(
                sorted(later_blocks, key=lambda item: int(item.block_no))
            ):
                item.block_no = insert_at + 1 + index
            await uow.session.flush()
        block_payload = proposal_summary_payload(snapshot)
        answer_block_id = uuid7()
        await uow.turns.add_answer_block(
            OpsAnswerBlockEntity(
                answer_block_id=answer_block_id,
                turn_id=turn.turn_id,
                message_id=message_id,
                block_no=insert_at,
                block_type="PROPOSAL_SUMMARY",
                schema_version="AIOPS_PROPOSAL_SUMMARY_BLOCK.v1",
                payload_json=block_payload,
                content_hash=sha256_json(block_payload),
            )
        )
        await self._append_turn_event(
            uow,
            turn,
            event_type="answer.block",
            payload={
                "answer_block_id": str(answer_block_id),
                "block_no": insert_at,
                "block_type": "PROPOSAL_SUMMARY",
                "schema_version": "AIOPS_PROPOSAL_SUMMARY_BLOCK.v1",
                "payload": block_payload,
                "citation_count": 0,
            },
            answer_block_id=answer_block_id,
        )

    @staticmethod
    def _comparison_result(
        verification: ActionVerification,
    ) -> tuple[str, tuple[str, ...]]:
        """把只读 Verification 结果映射为稳定、可审计的对比结论。"""
        if verification.status == "ADVERSE":
            return "DEGRADED", ("VERIFICATION_ADVERSE",)
        if verification.status == "INCONCLUSIVE" or verification.gap_codes:
            return "INCONCLUSIVE", ("EVIDENCE_NOT_COMPARABLE",)
        if verification.status == "VERIFIED":
            return "INCONCLUSIVE", ("DIRECT_ACTION_POSTCONDITION_VERIFIED", "BUSINESS_EFFECT_NOT_MEASURED")
        if verification.status == "NOT_ACHIEVED":
            return "UNCHANGED", ("EXPECTED_DIRECT_EFFECT_NOT_OBSERVED",)
        return "INCONCLUSIVE", ("VERIFICATION_STATE_UNSUPPORTED",)

    async def suspend_task_for_input(
        self, command: SuspendOpsTaskCommand
    ) -> TaskMutationReceipt:
        """在同一事务中持久化请求 Artifact、HITL，并挂起 Run/Task。"""
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run, task = await self._lock_run_task(uow, command.task_id)
            event_key = f"hitl:{command.hitl_id}:requested"
            artifact_key = f"hitl:{command.hitl_id}:request"
            prior = await uow.runs.get_event_by_key(
                ops_run_id=run.ops_run_id,
                event_key=event_key,
            )
            existing = await uow.runs.get_artifact_by_key(
                ops_run_id=run.ops_run_id,
                artifact_key=artifact_key,
            )
            if prior is not None and existing is not None:
                return self._mutation_receipt(
                    run,
                    task,
                    int(prior.sequence_no),
                    existing.artifact_id,
                )
            prior_hitl = await uow.changes.get_hitl_by_idempotency(
                ops_run_id=run.ops_run_id,
                idempotency_key=command.idempotency_key,
            )
            if prior_hitl is not None:
                prior_artifacts = list(
                    prior_hitl.input_artifacts_json or []
                )
                prior_artifact = (
                    await uow.runs.get_artifact(
                        artifact_id=UUID(str(prior_artifacts[0]))
                    )
                    if len(prior_artifacts) == 1
                    else None
                )
                prior_event = await uow.runs.get_event_by_key(
                    ops_run_id=run.ops_run_id,
                    event_key=f"hitl:{prior_hitl.hitl_id}:requested",
                )
                if prior_artifact is None or prior_event is None:
                    raise state_conflict("HITL 幂等记录不完整")
                if (
                    prior_hitl.request_type != command.request_type
                    or prior_hitl.assignee_user_id
                    != command.assignee_user_id
                    or prior_artifact.content_hash
                    != sha256_json(command.request_artifact.payload)
                ):
                    raise _runtime_error(
                        "OPS_IDEMPOTENCY_CONFLICT",
                        "相同 HITL 幂等键对应的请求内容不同",
                    )
                return self._mutation_receipt(
                    run,
                    task,
                    int(prior_event.sequence_no),
                    prior_artifact.artifact_id,
                )
            self._ensure_lease(
                run=run,
                task=task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
            )
            if run.trigger_type != "CHAT":
                raise _runtime_error(
                    "OPS_HITL_TRIGGER_FORBIDDEN",
                    "只有聊天触发的诊断 Run 可以等待用户补充数据",
                    status_code=422,
                )
            if (
                command.request_type
                not in {
                    "DATA_REQUIRED",
                    "MANUAL_DIAGNOSTIC_SQL",
                    "DIAGNOSTIC_QUERY_APPROVAL",
                }
                or command.assignee_user_id != run.actor_id
            ):
                raise _runtime_error(
                    "OPS_HITL_REQUEST_INVALID",
                    "人工补证类型或受理用户无效",
                    status_code=422,
                )
            if command.expires_at <= now:
                raise _runtime_error(
                    "OPS_HITL_EXPIRED",
                    "人工补证请求的有效期已结束",
                    status_code=422,
                )
            human_wait_until = run.human_wait_until or now + timedelta(days=1)
            if command.expires_at > human_wait_until:
                raise validation_failed("人工等待不能超过本 Run 首次人工请求后一天的期限")
            run.human_wait_until = human_wait_until
            content = command.request_artifact.payload
            if content is None:
                raise _runtime_error(
                    "OPS_HITL_REQUEST_INVALID",
                    "人工补证请求必须使用内联 Artifact",
                    status_code=422,
                )
            content_hash = sha256_json(content)
            artifact = await uow.runs.add_artifact(
                OpsArtifactEntity(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    artifact_key=artifact_key,
                    artifact_type=command.request_artifact.artifact_type,
                    schema_version=command.request_artifact.schema_version,
                    payload_json=content,
                    content_hash=content_hash,
                    byte_size=len(canonical_bytes(content)),
                    provenance_json={
                        **command.request_artifact.provenance,
                        "producer": command.request_artifact.producer,
                        "producer_version": (
                            command.request_artifact.producer_version
                        ),
                    },
                    trust_level=command.request_artifact.trust_level,
                    security_level=(
                        command.request_artifact.security_level
                    ),
                )
            )
            await uow.changes.add_hitl(
                HitlEntity(
                    hitl_id=command.hitl_id,
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    request_type=command.request_type,
                    assignee_user_id=command.assignee_user_id,
                    prompt_text=command.prompt_text,
                    response_schema_json=command.response_schema,
                    input_artifacts_json=[str(artifact.artifact_id)],
                    status="PENDING",
                    idempotency_key=command.idempotency_key,
                    requested_by=command.request_artifact.producer,
                    requested_at=now,
                    expires_at=command.expires_at,
                )
            )
            waiting_for_approval = (
                command.request_type == "DIAGNOSTIC_QUERY_APPROVAL"
            )
            task_waiting_status = (
                DomainOpsTaskStatus.WAITING_APPROVAL
                if waiting_for_approval
                else DomainOpsTaskStatus.WAITING_INPUT
            )
            try:
                pause_task_budget(task, now=now)
            except ExecutionBudgetError as exc:
                raise state_conflict(str(exc)) from exc
            ensure_task_transition(
                DomainOpsTaskStatus(task.status), task_waiting_status
            )
            task.status = task_waiting_status.value
            self._clear_lease(task)
            tasks = await uow.runs.list_tasks(ops_run_id=run.ops_run_id, lock=True)
            await self._sync_run_budget(uow=uow, run=run, tasks=tasks, now=now)
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="artifact.created",
                event_key=f"artifact:{command.hitl_id}:request",
                visibility="INTERNAL",
                payload_json={
                    "artifact_id": str(artifact.artifact_id),
                    "artifact_type": artifact.artifact_type,
                    "schema_version": artifact.schema_version,
                    "content_hash": artifact.content_hash,
                    "trace_id": command.trace_id,
                },
            )
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type=(
                    "diagnostic.query_approval_required"
                    if waiting_for_approval
                    else "diagnostic.input_required"
                ),
                event_key=event_key,
                visibility="USER",
                payload_json={
                    "hitl_id": str(command.hitl_id),
                    "hitl_type": command.request_type,
                    "request_artifact_id": str(artifact.artifact_id),
                    "expires_at": command.expires_at.isoformat(),
                    "trace_id": command.trace_id,
                },
            )
            assert uow.platform_notifications is not None
            await uow.platform_notifications.emit_run_event(
                run=run,
                event_type="aiops.diagnostic.input_required",
                summary=(
                    "诊断查询等待用户审批"
                    if waiting_for_approval
                    else "诊断需要补充输入"
                ),
                actor_id=run.actor_id,
            )
            await self._project_turn_interaction_status(
                uow=uow,
                run=run,
                status="WAITING_USER" if run.budget_paused_at is not None else "COLLECTING",
                event_type=(
                    "diagnostic.query_approval_required"
                    if waiting_for_approval
                    else "diagnostic.input_required"
                ),
                payload={
                    "hitl_id": str(command.hitl_id),
                    "hitl_type": command.request_type,
                    "expires_at": command.expires_at.isoformat(),
                },
            )
            await uow.commit()
            return self._mutation_receipt(
                run,
                task,
                int(event.sequence_no),
                artifact.artifact_id,
            )

    @staticmethod
    async def _reduce_database_health(
        *, uow, run, payload: dict[str, Any], now: datetime
    ) -> None:
        """连通性由独立探测任务维护，诊断结果不得覆盖该状态。"""
        del uow, run, payload, now

    @staticmethod
    async def _materialize_advisory_proposal(
        *,
        uow,
        run,
        task,
        artifact,
        payload: dict[str, Any],
        trace_id: str,
        now: datetime,
    ) -> None:
        outcome = ProposalOutcome.model_validate(payload)
        if outcome.status == "NOT_REQUIRED":
            return
        snapshot = outcome.proposal
        if snapshot is None:
            raise validation_failed("Proposal Outcome 缺少权威快照")
        if (
            snapshot.run_id != str(run.ops_run_id)
            or snapshot.task_id != str(task.ops_task_id)
            or snapshot.target_id != str(run.target_id)
        ):
            raise validation_failed("Proposal Snapshot 资源标识不匹配")
        if snapshot.mode not in {"ADVISORY", "AGENT_EXECUTE"}:
            raise validation_failed("Proposal 执行模式无效")
        pending_approval = (
            snapshot.mode == "AGENT_EXECUTE"
            and snapshot.execution_mode == "EXECUTABLE_AFTER_APPROVAL"
        )
        if snapshot.execution_mode == "MANUAL_ONLY" and (
            snapshot.mode != "ADVISORY" or snapshot.executor_kind != "NONE"
        ):
            raise validation_failed("人工动作的 Proposal 执行语义无效")
        status = "PENDING_APPROVAL" if pending_approval else "ADVISORY_READY"
        baseline_start = run.created_at or now
        baseline_seconds = max(
            60, int((now - baseline_start).total_seconds())
        )
        comparison_plan = ComparisonPlan(
            proposal_id=snapshot.proposal_id,
            source_run_id=str(run.ops_run_id),
            solution_group_key=snapshot.solution_group_key,
            action_template_id=snapshot.action_template_id,
            action_template_version=snapshot.action_template_version,
            baseline_start=baseline_start,
            baseline_end=now,
            settle_delay_seconds=0,
            after_window_seconds=baseline_seconds,
            primary_signals=("target_absent", "blocking_absent"),
            required_tool_refs=tuple(snapshot.verification_plan),
            baseline_evidence_refs=tuple(snapshot.evidence_refs),
        )
        comparison_plan_payload = comparison_plan.model_dump(mode="json")
        target_snapshot = dict(
            (run.plan_snapshot_json or {}).get("target")
            or dict(
                (run.plan_snapshot_json or {}).get("change_context", {})
            ).get("target", {})
        )
        turn_link = await uow.turns.get_run_link_by_ops_run_id(
            ops_run_id=run.ops_run_id
        )
        if turn_link is None:
            raise validation_failed("Proposal 缺少所属诊断 Turn")
        comparison_plan_artifact = await uow.runs.add_artifact(
            OpsArtifactEntity(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                artifact_key=(
                    f"comparison:proposal:{snapshot.proposal_id}:plan:v1"
                ),
                artifact_type="COMPARISON_PLAN",
                schema_version="COMPARISON_PLAN.v1",
                payload_json=comparison_plan_payload,
                content_hash=sha256_json(comparison_plan_payload),
                byte_size=len(canonical_bytes(comparison_plan_payload)),
                provenance_json={
                    "producer": "aiops.comparison-planner",
                    "producer_version": "action-effect.v1",
                    "deterministic": True,
                    "llm_used": False,
                },
                trust_level="SOURCE_VERIFIED",
                security_level=1,
            )
        )
        proposal = ChangeProposalEntity(
            proposal_id=UUID(snapshot.proposal_id),
            turn_id=turn_link.turn_id,
            ops_run_id=run.ops_run_id,
            ops_task_id=task.ops_task_id,
            target_id=run.target_id,
            solution_group_key=snapshot.solution_group_key,
            command_ordinal=snapshot.command_ordinal,
            proposal_version=snapshot.proposal_version,
            action_type=snapshot.mode,
            action_family=snapshot.action_family,
            effect_class=snapshot.effect_class,
            execution_mode=snapshot.execution_mode,
            executor_kind=snapshot.executor_kind,
            canonical_object_ref_json=snapshot.canonical_object_ref,
            action_template_id=snapshot.action_template_id,
            action_template_version=snapshot.action_template_version,
            action_template_hash=snapshot.action_template_hash,
            renderer_version=snapshot.renderer_version,
            command_hash=snapshot.command_hash,
            parameters_json=snapshot.canonical_parameters,
            parameters_hash=snapshot.parameters_hash,
            rationale=snapshot.rationale,
            impact_scope_json={"summary": snapshot.impact},
            risk_level=snapshot.risk_level,
            lock_impact=snapshot.lock_impact,
            estimated_duration_seconds=(
                snapshot.estimated_duration_seconds
            ),
            preconditions_json=[
                {"tool_id": item} for item in snapshot.preconditions
            ],
            rollback_plan_json={"description": snapshot.rollback_plan},
            verification_plan_json={
                "tool_refs": list(snapshot.verification_plan)
            },
            evidence_artifacts_json=list(snapshot.evidence_refs),
            policy_decision_hash=snapshot.policy_decision_hash,
            proposal_hash=snapshot.proposal_hash,
            snapshot_artifact_id=artifact.artifact_id,
            status=status,
            expires_at=snapshot.expires_at,
            created_by_task_id=task.ops_task_id,
            created_at=now,
            updated_at=now,
        )
        await uow.changes.add_proposal(proposal)
        await uow.runs.append_event(
            ops_run_id=run.ops_run_id,
            ops_task_id=task.ops_task_id,
            event_type="comparison.plan.created",
            event_key=(
                f"comparison:proposal:{proposal.proposal_id}:plan:v1"
            ),
            visibility="INTERNAL",
            payload_json={
                "proposal_id": str(proposal.proposal_id),
                "artifact_id": str(comparison_plan_artifact.artifact_id),
                "schema_version": "COMPARISON_PLAN.v1",
                "trace_id": trace_id,
            },
        )
        if status == "PENDING_APPROVAL":
            assert uow.platform_notifications is not None
            await uow.platform_notifications.emit_proposal_event(
                run=run,
                proposal=proposal,
                event_type="aiops.proposal.review_required",
                summary="变更方案等待审核",
                actor_id=run.actor_id,
            )
        if status == "PENDING_APPROVAL":
            await uow.changes.add_hitl(
                HitlEntity(
                    hitl_id=uuid7(),
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    proposal_id=proposal.proposal_id,
                    request_type="CHANGE_APPROVAL",
                    assignee_user_id=run.actor_id,
                    prompt_text="请查看权威 Proposal 后逐条确认是否执行",
                    response_schema_json={
                        "schema_version": "APPROVAL_DECISION.v1"
                    },
                    input_artifacts_json=[
                        str(artifact.artifact_id)
                    ],
                    status="PENDING",
                    idempotency_key=(
                        f"proposal:{proposal.proposal_id}:approval"
                    ),
                    requested_by="aiops.change-service",
                    requested_at=now,
                    expires_at=snapshot.expires_at,
                )
            )
        await uow.runs.append_event(
            ops_run_id=run.ops_run_id,
            ops_task_id=task.ops_task_id,
            event_type=(
                "proposal.pending_approval"
                if status == "PENDING_APPROVAL"
                else "proposal.advisory_ready"
            ),
            event_key=f"proposal:{proposal.proposal_id}:ready",
            visibility="USER",
            payload_json={
                "proposal_id": str(proposal.proposal_id),
                "risk_level": proposal.risk_level,
                "expires_at": (
                    proposal.expires_at.isoformat()
                    if proposal.expires_at is not None
                    else None
                ),
                "proposal_hash": proposal.proposal_hash,
                "status": status,
                "trace_id": trace_id,
            },
        )

    @staticmethod
    async def _reduce_observation_health(
        *, uow, run, payload: dict[str, Any], now: datetime
    ) -> None:
        """迟到结果必须同时匹配冻结的配置版本与当前 Health Version。"""
        monitoring = (run.plan_snapshot_json or {}).get("monitoring")
        if not monitoring:
            return
        binding_id = str(payload.get("binding_id", ""))
        snapshot = next(
            (
                item
                for item in monitoring.get("bindings", [])
                if item["binding_id"] == binding_id
            ),
            None,
        )
        if snapshot is None:
            return
        source_snapshot = snapshot["source"]
        source = await uow.diagnostic_sources.get_scoped(
            diagnostic_source_id=UUID(source_snapshot["source_id"]),
            domain_id=int(run.domain_id),
        )
        if source is None:
            return
        gaps = list(payload.get("gaps", []))
        has_observations = bool(payload.get("observations"))
        gap_codes = {str(item.get("code")) for item in gaps}
        if "SOURCE_AUTH_FAILED" in gap_codes:
            source_status, source_error = (
                "DEGRADED",
                "SOURCE_AUTH_FAILED",
            )
        elif "SOURCE_UNREACHABLE" in gap_codes:
            source_status, source_error = (
                "DEGRADED" if has_observations else "UNREACHABLE",
                "SOURCE_UNREACHABLE",
            )
        else:
            source_status, source_error = (
                (
                    "DEGRADED"
                    if source.connectivity_status == "UNREACHABLE"
                    else "CONNECTED"
                ),
                None,
            )
        await uow.diagnostic_sources.reduce_connectivity(
            diagnostic_source_id=source.diagnostic_source_id,
            expected_config_version=int(source_snapshot["config_version"]),
            expected_connectivity_version=int(source.connectivity_version),
            connectivity_status=source_status,
            checked_at=now,
            last_error_code=source_error,
        )
        monitor = await uow.targets.get_source_binding_scoped(
            target_source_binding_id=UUID(binding_id),
            target_id=run.target_id,
            domain_id=int(source.domain_id),
        )
        if monitor is None:
            return
        binding_status = (
            "HEALTHY"
            if payload.get("observations") and not gaps
            else "DEGRADED"
            if payload.get("observations")
            else "UNREACHABLE"
        )
        await uow.targets.reduce_source_binding_health(
            target_source_binding_id=monitor.target_source_binding_id,
            expected_config_version=int(snapshot["binding_version"]),
            expected_health_version=int(monitor.health_version),
            health_status=binding_status,
            checked_at=now,
            last_error_code=(
                None if not gaps else str(gaps[0].get("code"))
            ),
        )
        artifacts = await uow.runs.list_artifacts(
            ops_run_id=run.ops_run_id
        )
        observation_payloads = [
            item.payload_json or {}
            for item in artifacts
            if item.schema_version == "OBSERVATION_SET.v1"
        ]
        availability_values = [
            point.get("value")
            for observation_payload in observation_payloads
            for observation in observation_payload.get("observations", [])
            if observation.get("semantic_role") == METRIC_ROLE_AVAILABILITY
            for series in observation.get("series", [])
            for point in series.get("points", [])[-1:]
            if point.get("quality") == "GOOD"
        ]
        if availability_values:
            normalized = {
                AIOpsRuntimeService._availability_bool(value)
                for value in availability_values
            }
            normalized.discard(None)
        else:
            normalized = set()
        if normalized:
            target_status = (
                "DEGRADED"
                if len(normalized) > 1
                else "UP"
                if True in normalized
                else "DOWN"
            )
            target = await uow.targets.get_scoped(
                target_id=run.target_id,
                domain_id=int(source.domain_id),
            )
            if target is not None:
                await uow.targets.update_observed_status(
                    target_id=target.target_id,
                    observed_status=target_status,
                    checked_at=now,
                    last_error_code=None,
                )

    @staticmethod
    def _availability_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(float(value))
        normalized = str(value).strip().upper()
        if normalized in {"UP", "AVAILABLE", "ONLINE", "TRUE", "1"}:
            return True
        if normalized in {"DOWN", "UNAVAILABLE", "OFFLINE", "FALSE", "0"}:
            return False
        return None

    async def fail_task(
        self, command: FailOpsTaskCommand
    ) -> TaskMutationReceipt:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run, task = await self._lock_run_task(
                uow, command.task_id
            )
            event_key = (
                f"task:{task.ops_task_id}:failed:"
                f"{command.idempotency_key}"
            )
            prior = await uow.runs.get_event_by_key(
                ops_run_id=run.ops_run_id, event_key=event_key
            )
            if prior is not None:
                return self._mutation_receipt(
                    run, task, int(prior.sequence_no), None
                )
            self._ensure_lease(
                run=run,
                task=task,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                now=now,
                # Task 超时与租约截止时间相同；失败回写仍须校验状态、Owner 和 Token。
                allow_expired=True,
            )
            try:
                manifest = self._handlers.resolve(
                    task.handler_id, task.handler_version
                )
                handler_idempotent = manifest.idempotent
            except LookupError:
                handler_idempotent = False
            policy = ERROR_CATALOG.get(
                command.error_code,
                ERROR_CATALOG["HANDLER_TERMINAL_FAILURE"],
            )
            retry = (
                handler_idempotent
                and policy.retryable
                and int(task.attempt_count) < int(task.max_attempts)
            )
            target = (
                DomainOpsTaskStatus.RETRY_WAIT
                if retry
                else DomainOpsTaskStatus.FAILED
            )
            ensure_task_transition(
                DomainOpsTaskStatus(task.status), target
            )
            task.status = target.value
            task.error_code = command.error_code
            task.error_message = policy.safe_message
            task.available_at = (
                now + self._retry_delay(task)
                if retry
                else task.available_at
            )
            task.completed_at = None if retry else now
            self._clear_lease(task)
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="task.status",
                event_key=event_key,
                visibility="USER",
                payload_json={
                    "status": target.value,
                    "task_id": str(task.ops_task_id),
                    "task_type": task.task_type,
                    "task_key": task.task_key,
                    "error_code": command.error_code,
                    "trace_id": command.trace_id,
                },
            )
            if not retry:
                self._block_unreachable(tasks, failed_key=task.task_key)
                await preserve_future_observation(uow=uow, run=run, now=now, reason=command.error_code)
                if command.error_code == "RUN_DEADLINE_EXCEEDED" and DomainOpsRunStatus(run.status) not in TERMINAL_RUN_STATUSES:
                    if await self._project_deadline_partial(uow=uow, run=run, now=now):
                        ensure_run_transition(DomainOpsRunStatus(run.status), DomainOpsRunStatus.PARTIAL)
                        run.status = DomainOpsRunStatus.PARTIAL.value
                        run.error_code = command.error_code
                        run.completed_at = now
                        await uow.runs.append_event(
                            ops_run_id=run.ops_run_id, event_type="run.partial",
                            event_key=f"run:{run.ops_run_id}:terminal", visibility="USER",
                            payload_json={"status": "PARTIAL", "error_code": command.error_code, "trace_id": command.trace_id},
                        )
                        await self._add_multi_resource_terminal_outbox(uow, run=run, trace_id=command.trace_id, now=now)
                        await uow.commit()
                        return self._mutation_receipt(run, task, int(event.sequence_no), None)
                if DomainOpsRunStatus(run.status) not in TERMINAL_RUN_STATUSES:
                    ensure_run_transition(
                        DomainOpsRunStatus(run.status),
                        DomainOpsRunStatus.FAILED,
                    )
                    run.status = DomainOpsRunStatus.FAILED.value
                    run.error_code = command.error_code
                    run.error_message = policy.safe_message
                    run.completed_at = now
                    if run.workflow_kind in _AGENT_TURN_WORKFLOWS:
                        await self._project_turn_failure(
                            uow=uow,
                            run=run,
                            error_code=command.error_code,
                            public_summary=policy.safe_message,
                            now=now,
                        )
                    await uow.runs.append_event(
                        ops_run_id=run.ops_run_id,
                        event_type="run.failed",
                        event_key=f"run:{run.ops_run_id}:terminal",
                        visibility="USER",
                        payload_json={
                            "status": DomainOpsRunStatus.FAILED.value,
                            "error_code": command.error_code,
                            "trace_id": command.trace_id,
                        },
                    )
                    await self._add_multi_resource_terminal_outbox(
                        uow, run=run, trace_id=command.trace_id, now=now
                    )
                    assert uow.platform_notifications is not None
                    await uow.platform_notifications.emit_run_event(
                        run=run,
                        event_type="aiops.run.failed",
                        summary=policy.safe_message,
                        actor_id=run.actor_id,
                    )
            await self._sync_run_budget(uow=uow, run=run, tasks=tasks, now=now)
            await uow.commit()
            return self._mutation_receipt(
                run, task, int(event.sequence_no), None
            )

    async def request_cancel(
        self,
        *,
        ops_run_id: UUID,
        domain_id: int,
        actor_id: str,
        expected_row_version: int,
        idempotency_key: str,
        trace_id: str,
    ) -> OpsRunReceipt:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run = await uow.runs.get_run_scoped(
                ops_run_id=ops_run_id,
                domain_id=domain_id,
                lock=True,
            )
            if run is None:
                raise resource_not_found("Ops Run")
            event_key = f"run:{run.ops_run_id}:cancel:{idempotency_key}"
            prior = await uow.runs.get_event_by_key(
                ops_run_id=run.ops_run_id, event_key=event_key
            )
            if prior is not None:
                return self._run_receipt(run, int(prior.sequence_no))
            if int(run.row_version) != expected_row_version:
                raise _runtime_error(
                    "OPS_ROW_VERSION_CHANGED",
                    "Run 已被其他请求更新",
                    status_code=412,
                )
            if DomainOpsRunStatus(run.status) in TERMINAL_RUN_STATUSES:
                raise state_conflict("终态 Run 不能取消")
            run.cancel_requested_at = now
            run.cancel_requested_by = actor_id
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            running = False
            for task in tasks:
                current = DomainOpsTaskStatus(task.status)
                if current == DomainOpsTaskStatus.RUNNING:
                    running = True
                    continue
                if current in {
                    DomainOpsTaskStatus.PENDING,
                    DomainOpsTaskStatus.READY,
                    DomainOpsTaskStatus.RETRY_WAIT,
                    DomainOpsTaskStatus.WAITING_INPUT,
                    DomainOpsTaskStatus.WAITING_APPROVAL,
                }:
                    ensure_task_transition(
                        current, DomainOpsTaskStatus.CANCELLED
                    )
                    task.status = DomainOpsTaskStatus.CANCELLED.value
                    task.completed_at = now
                    self._clear_lease(task)
            await self._close_pending_hitls(uow=uow, run=run, now=now, status="CANCELLED")
            event = await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                event_type="run.cancel_requested",
                event_key=event_key,
                visibility="USER",
                payload_json={
                    "status": run.status,
                    "trace_id": trace_id,
                },
            )
            if not running:
                await preserve_future_observation(uow=uow, run=run, now=now, reason="FOLLOWUP_CANCELLED")
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    DomainOpsRunStatus.CANCELLED,
                )
                run.status = DomainOpsRunStatus.CANCELLED.value
                run.completed_at = now
                if run.workflow_kind in _AGENT_TURN_WORKFLOWS:
                    await self._project_turn_terminal(
                        uow=uow,
                        run=run,
                        status="CANCELLED",
                        error_code=None,
                        public_summary="诊断已取消",
                        now=now,
                    )
                event = await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    event_type="run.cancelled",
                    event_key=f"run:{run.ops_run_id}:terminal",
                    visibility="USER",
                    payload_json={
                        "status": DomainOpsRunStatus.CANCELLED.value,
                        "trace_id": trace_id,
                    },
                )
                await self._add_multi_resource_terminal_outbox(
                    uow, run=run, trace_id=trace_id, now=now
                )
            await uow.commit()
            return self._run_receipt(run, int(event.sequence_no))

    async def reconcile_once(self, *, trace_id: str) -> bool:
        """依次收敛 Deadline、过期租约和到期重试。"""
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            run = await uow.runs.lock_due_run(now=now)
            if run is not None:
                tasks = await uow.runs.list_tasks(
                    ops_run_id=run.ops_run_id, lock=True
                )
                await self._close_pending_hitls(uow=uow, run=run, now=now, status="EXPIRED")
                for task in tasks:
                    current = DomainOpsTaskStatus(task.status)
                    if current not in {
                        DomainOpsTaskStatus.SUCCEEDED,
                        DomainOpsTaskStatus.FAILED,
                        DomainOpsTaskStatus.BLOCKED,
                        DomainOpsTaskStatus.CANCELLED,
                        DomainOpsTaskStatus.EXPIRED,
                    }:
                        ensure_task_transition(
                            current, DomainOpsTaskStatus.EXPIRED
                        )
                        task.status = DomainOpsTaskStatus.EXPIRED.value
                        task.completed_at = now
                        self._clear_lease(task)
                partial = await self._project_deadline_partial(uow=uow, run=run, now=now)
                terminal_status = DomainOpsRunStatus.PARTIAL if partial else DomainOpsRunStatus.EXPIRED
                ensure_run_transition(DomainOpsRunStatus(run.status), terminal_status)
                run.status = terminal_status.value
                run.completed_at = now
                if not partial and run.workflow_kind in _AGENT_TURN_WORKFLOWS:
                    await self._project_turn_terminal(
                        uow=uow,
                        run=run,
                        status="FAILED",
                        error_code="OPS_RUN_EXPIRED",
                        public_summary="诊断运行超过截止时间",
                        now=now,
                    )
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    event_type="run.partial" if partial else "run.expired",
                    event_key=f"run:{run.ops_run_id}:terminal",
                    visibility="USER",
                    payload_json={
                        "status": terminal_status.value,
                        "trace_id": trace_id,
                    },
                )
                await self._add_multi_resource_terminal_outbox(
                    uow, run=run, trace_id=trace_id, now=now
                )
                await uow.commit()
                return True
            expired_proposal = await uow.changes.find_expired_proposal(
                now=now
            )
            if expired_proposal is not None:
                run = await uow.runs.get_run(
                    ops_run_id=expired_proposal.ops_run_id,
                    lock=True,
                )
                proposal = await uow.changes.get_proposal(
                    proposal_id=expired_proposal.proposal_id,
                    lock=True,
                )
                if (
                    run is None
                    or proposal is None
                    or proposal.status
                    not in {"ADVISORY_READY", "PENDING_APPROVAL"}
                    or proposal.expires_at is None
                    or proposal.expires_at > now
                ):
                    return False
                proposal.status = "EXPIRED"
                proposal.updated_at = now
                approval_hitl = await uow.changes.get_pending_hitl(
                    ops_task_id=proposal.ops_task_id,
                    request_type="CHANGE_APPROVAL",
                    lock=True,
                )
                if (
                    approval_hitl is not None
                    and approval_hitl.proposal_id
                    == proposal.proposal_id
                ):
                    approval_hitl.status = "EXPIRED"
                    approval_hitl.responded_by = "aiops.reconciler"
                    approval_hitl.responded_at = now
                    approval_hitl.response_json = {
                        "reason": "PROPOSAL_EXPIRED"
                    }
                    approval_hitl.response_hash = sha256_json(
                        approval_hitl.response_json
                    )
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=proposal.ops_task_id,
                    event_type="proposal.expired",
                    event_key=(
                        f"proposal:{proposal.proposal_id}:expired"
                    ),
                    visibility="USER",
                    payload_json={
                        "proposal_id": str(proposal.proposal_id),
                        "status": "EXPIRED",
                        "trace_id": trace_id,
                    },
                )
                await uow.commit()
                return True
            expired_hitl = await uow.changes.find_expired_hitl()
            if expired_hitl is not None:
                preliminary = await uow.runs.get_task(
                    ops_task_id=expired_hitl.ops_task_id
                )
                if preliminary is None:
                    raise state_conflict("过期 HITL 对应的 Task 不存在")
                run = await uow.runs.get_run(
                    ops_run_id=preliminary.ops_run_id, lock=True
                )
                task = await uow.runs.get_task(
                    ops_task_id=preliminary.ops_task_id, lock=True
                )
                hitl = await uow.changes.get_hitl(
                    hitl_id=expired_hitl.hitl_id, lock=True
                )
                if (
                    run is None
                    or task is None
                    or hitl is None
                    or hitl.status != "PENDING"
                    or hitl.expires_at > now
                ):
                    return False
                task_status = DomainOpsTaskStatus(task.status)
                run_status = DomainOpsRunStatus(run.status)
                approval_request = (
                    getattr(hitl, "request_type", None)
                    == "DIAGNOSTIC_QUERY_APPROVAL"
                )
                expected_task_status = (
                    DomainOpsTaskStatus.WAITING_APPROVAL
                    if approval_request
                    else DomainOpsTaskStatus.WAITING_INPUT
                )
                if (
                    task_status
                    in {
                        DomainOpsTaskStatus.SUCCEEDED,
                        DomainOpsTaskStatus.FAILED,
                        DomainOpsTaskStatus.BLOCKED,
                        DomainOpsTaskStatus.CANCELLED,
                        DomainOpsTaskStatus.EXPIRED,
                    }
                    or run_status in TERMINAL_RUN_STATUSES
                    or task_status != expected_task_status
                    or run.cancel_requested_at is not None
                ):
                    # Reconciler 必须能够清理旧版本或并发遗留的孤儿 HITL，
                    # 不能尝试把终态 Task/Run 重新推进到诊断态。
                    hitl.status = "EXPIRED"
                    hitl.responded_by = "aiops.reconciler"
                    hitl.responded_at = now
                    hitl.response_json = {
                        "reason": "PARENT_STATE_NOT_WAITING_INPUT",
                        "task_status": task.status,
                        "run_status": run.status,
                    }
                    hitl.response_hash = sha256_json(
                        hitl.response_json
                    )
                    await uow.runs.append_event(
                        ops_run_id=run.ops_run_id,
                        ops_task_id=task.ops_task_id,
                        event_type="diagnostic.input_expired",
                        event_key=f"hitl:{hitl.hitl_id}:expired",
                        visibility="USER",
                        payload_json={
                            "hitl_id": str(hitl.hitl_id),
                            "status": "EXPIRED",
                            "reason": "PARENT_STATE_NOT_WAITING_INPUT",
                            "trace_id": trace_id,
                        },
                    )
                    await uow.commit()
                    return True
                payload = HitlOutcome(
                    hitl_id=str(hitl.hitl_id),
                    status="EXPIRED",
                    gap_code=(
                        "DIAGNOSTIC_QUERY_APPROVAL_EXPIRED"
                        if approval_request
                        else "MANUAL_DIAGNOSTIC_EXPIRED"
                    ),
                ).model_dump(mode="json")
                artifact = await uow.runs.add_artifact(
                    OpsArtifactEntity(
                        ops_run_id=run.ops_run_id,
                        ops_task_id=task.ops_task_id,
                        artifact_key=self._artifact_key(task),
                        artifact_type="HITL_OUTCOME",
                        schema_version="HITL_OUTCOME.v1",
                        payload_json=payload,
                        content_hash=sha256_json(payload),
                        byte_size=len(canonical_bytes(payload)),
                        provenance_json={
                            "producer": "aiops.reconciler",
                            "producer_version": "1",
                        },
                        trust_level="SOURCE_VERIFIED",
                        security_level=1,
                    )
                )
                hitl.status = "EXPIRED"
                hitl.responded_by = "aiops.reconciler"
                hitl.responded_at = now
                hitl.response_json = {
                    "reason": (
                        "DIAGNOSTIC_QUERY_APPROVAL_EXPIRED"
                        if approval_request
                        else "MANUAL_DIAGNOSTIC_EXPIRED"
                    )
                }
                hitl.response_hash = sha256_json(hitl.response_json)
                ensure_task_transition(
                    task_status,
                    DomainOpsTaskStatus.SUCCEEDED,
                )
                task.status = DomainOpsTaskStatus.SUCCEEDED.value
                task.output_artifact_id = artifact.artifact_id
                task.completed_at = now
                tasks = await uow.runs.list_tasks(
                    ops_run_id=run.ops_run_id, lock=True
                )
                self._release_successors(tasks, now=now)
                await self._sync_run_budget(uow=uow, run=run, tasks=tasks, now=now)
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    event_type="diagnostic.input_expired",
                    event_key=f"hitl:{hitl.hitl_id}:expired",
                    visibility="USER",
                    payload_json={
                        "hitl_id": str(hitl.hitl_id),
                        "status": "EXPIRED",
                        "trace_id": trace_id,
                    },
                )
                await self._project_turn_interaction_status(
                    uow=uow,
                    run=run,
                    status="WAITING_USER" if run.budget_paused_at is not None else "COLLECTING",
                    event_type="diagnostic.input_expired",
                    payload={
                        "hitl_id": str(hitl.hitl_id),
                        "status": "EXPIRED",
                    },
                )
                await uow.commit()
                return True
            due_execution = await uow.changes.find_due_execution(now=now)
            if due_execution is not None:
                run = await uow.runs.get_run(
                    ops_run_id=due_execution.ops_run_id, lock=True
                )
                proposal = await uow.changes.get_proposal(
                    proposal_id=due_execution.proposal_id, lock=True
                )
                execution = await uow.changes.get_execution(
                    execution_id=due_execution.execution_id, lock=True
                )
                if (
                    run is None
                    or proposal is None
                    or execution is None
                    or execution.status
                    not in {"CREATED", "SUBMITTED", "RUNNING"}
                    or execution.deadline_at > now
                ):
                    return False
                if execution.status in {"CREATED", "SUBMITTED"}:
                    execution.status = "TIMED_OUT"
                    execution.status_version = (
                        int(execution.status_version) + 1
                    )
                    execution.completed_at = now
                    execution.error_code = "EXECUTION_NOT_STARTED"
                    execution.error_message = (
                        "执行授权在数据库动作开始前已超时"
                    )
                    execution.updated_at = now
                else:
                    result_body = {
                        "accepted": False,
                        "action_template_id": (
                            execution.action_template_id
                        ),
                        "outcome_unknown": True,
                        "reason": "EXECUTOR_CALLBACK_TIMEOUT",
                    }
                    result_hash = sha256_json(result_body)
                    assert execution.executor_instance_id is not None
                    assert execution.grant_jti_hash is not None
                    result = ExecutionResultArtifact(
                        execution_id=str(execution.execution_id),
                        proposal_id=str(proposal.proposal_id),
                        executor_request_id=(
                            execution.executor_request_id
                        ),
                        executor_instance_id=(
                            execution.executor_instance_id
                        ),
                        status="UNKNOWN",
                        status_version=4,
                        occurred_at=now,
                        bounded_result=result_body,
                        result_hash=result_hash,
                        error_code="EXECUTOR_CALLBACK_TIMEOUT",
                        proposal_hash=execution.proposal_hash,
                        command_hash=execution.command_hash,
                        grant_jti_hash=execution.grant_jti_hash,
                    )
                    result_payload = result.model_dump(mode="json")
                    artifact = await uow.runs.add_artifact(
                        OpsArtifactEntity(
                            ops_run_id=run.ops_run_id,
                            ops_task_id=proposal.ops_task_id,
                            artifact_key=(
                                f"execution:{execution.execution_id}:"
                                "result:v1"
                            ),
                            artifact_type="EXECUTION_RESULT",
                            schema_version="EXECUTION_RESULT.v1",
                            payload_json=result_payload,
                            content_hash=sha256_json(result_payload),
                            byte_size=len(
                                canonical_bytes(result_payload)
                            ),
                            provenance_json={
                                "producer": "aiops.reconciler",
                                "producer_version": "mutation.v1",
                                "reason": "CALLBACK_TIMEOUT",
                            },
                            trust_level="SOURCE_VERIFIED",
                            security_level=1,
                        )
                    )
                    execution.status = "UNKNOWN"
                    execution.status_version = 4
                    execution.result_artifact_id = artifact.artifact_id
                    execution.result_hash = result_hash
                    execution.completed_at = now
                    execution.error_code = "EXECUTOR_CALLBACK_TIMEOUT"
                    execution.error_message = (
                        "数据库动作已开始，但未收到可信终态"
                    )
                    execution.updated_at = now
                    target_snapshot = (
                        run.plan_snapshot_json or {}
                    )["target"]
                    verification_payload = {
                        "execution_id": str(execution.execution_id),
                        "proposal_id": str(proposal.proposal_id),
                        "source_run_id": str(run.ops_run_id),
                        "result_artifact_id": str(artifact.artifact_id),
                                                "domain_id": int(
                            target_snapshot["domain_id"]
                        ),
                        "actor_id": run.actor_id,
                        "agent_id": str(run.agent_id),
                        "target_id": str(run.target_id),
                        "trace_id": trace_id,
                    }
                    await uow.outbox.add(
                        OutboxEntity(
                            aggregate_type="OPS_EXECUTION",
                            aggregate_id=execution.execution_id,
                            event_type=(
                                "OPS_EXECUTION_VERIFY_REQUESTED"
                            ),
                            idempotency_key=(
                                f"execution:{execution.execution_id}:verify"
                            ),
                            payload_json=verification_payload,
                            payload_hash=sha256_json(
                                verification_payload
                            ),
                            status="PENDING",
                            available_at=now,
                            max_attempts=5,
                            trace_id=trace_id,
                        )
                    )
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=proposal.ops_task_id,
                    event_type="execution.status",
                    event_key=(
                        f"execution:{execution.execution_id}:"
                        f"{execution.status.lower()}"
                    ),
                    visibility="USER",
                    payload_json={
                        "execution_id": str(execution.execution_id),
                        "status": execution.status,
                        "status_version": int(
                            execution.status_version
                        ),
                        "error_code": execution.error_code,
                        "trace_id": trace_id,
                    },
                )
                await uow.commit()
                return True
            task = await uow.runs.lock_expired_task(now=now)
            if task is not None:
                run = await uow.runs.get_run(
                    ops_run_id=task.ops_run_id
                )
                assert run is not None
                if run.cancel_requested_at is not None:
                    task.status = DomainOpsTaskStatus.CANCELLED.value
                    task.completed_at = now
                    self._clear_lease(task)
                    tasks = await uow.runs.list_tasks(
                        ops_run_id=run.ops_run_id, lock=True
                    )
                    if not any(
                        item.status == DomainOpsTaskStatus.RUNNING.value
                        for item in tasks
                    ):
                        # Turn 取消可能已将 Run 标记为 CANCELLED，而租约
                        # 到期的执行中 Task 仍需在此处收敛。终态 Run 不可
                        # 再迁移；否则事务会回滚并永久重复处理同一 Task。
                        if run.status != DomainOpsRunStatus.CANCELLED.value:
                            await preserve_future_observation(uow=uow, run=run, now=now, reason="FOLLOWUP_CANCELLED")
                            ensure_run_transition(
                                DomainOpsRunStatus(run.status),
                                DomainOpsRunStatus.CANCELLED,
                            )
                            run.status = DomainOpsRunStatus.CANCELLED.value
                            run.completed_at = now
                            if run.workflow_kind in _AGENT_TURN_WORKFLOWS:
                                await self._project_turn_terminal(
                                    uow=uow,
                                    run=run,
                                    status="CANCELLED",
                                    error_code=None,
                                    public_summary="诊断已取消",
                                    now=now,
                                )
                            await uow.runs.append_event(
                                ops_run_id=run.ops_run_id,
                                event_type="run.cancelled",
                                event_key=(
                                    f"run:{run.ops_run_id}:terminal"
                                ),
                                visibility="USER",
                                payload_json={
                                    "status": "CANCELLED",
                                    "trace_id": trace_id,
                                },
                            )
                            await self._add_multi_resource_terminal_outbox(
                                uow, run=run, trace_id=trace_id, now=now
                            )
                else:
                    retry = int(task.attempt_count) < int(
                        task.max_attempts
                    )
                    target = (
                        DomainOpsTaskStatus.RETRY_WAIT
                        if retry
                        else DomainOpsTaskStatus.FAILED
                    )
                    ensure_task_transition(
                        DomainOpsTaskStatus(task.status), target
                    )
                    task.status = target.value
                    task.error_code = "WORKER_LEASE_EXPIRED"
                    task.error_message = ERROR_CATALOG[
                        "WORKER_LEASE_EXPIRED"
                    ].safe_message
                    task.available_at = (
                        now + self._retry_delay(task)
                        if retry
                        else task.available_at
                    )
                    task.completed_at = None if retry else now
                    self._clear_lease(task)
                    if not retry:
                        tasks = await uow.runs.list_tasks(
                            ops_run_id=run.ops_run_id, lock=True
                        )
                        self._block_unreachable(
                            tasks, failed_key=task.task_key
                        )
                        await preserve_future_observation(uow=uow, run=run, now=now, reason="WORKER_LEASE_EXPIRED")
                        ensure_run_transition(
                            DomainOpsRunStatus(run.status),
                            DomainOpsRunStatus.FAILED,
                        )
                        run.status = DomainOpsRunStatus.FAILED.value
                        run.error_code = "WORKER_LEASE_EXPIRED"
                        run.error_message = ERROR_CATALOG[
                            "WORKER_LEASE_EXPIRED"
                        ].safe_message
                        run.completed_at = now
                        if run.workflow_kind in _AGENT_TURN_WORKFLOWS:
                            await self._project_turn_failure(
                                uow=uow,
                                run=run,
                                error_code="WORKER_LEASE_EXPIRED",
                                public_summary=run.error_message,
                                now=now,
                            )
                event = await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    event_type="task.status",
                    event_key=(
                        f"task:{task.ops_task_id}:lease-expired:"
                        f"{int(task.attempt_count)}"
                    ),
                    visibility="USER",
                    payload_json={
                        "status": task.status,
                        "task_id": str(task.ops_task_id),
                        "task_type": task.task_type,
                        "task_key": task.task_key,
                        "error_code": task.error_code,
                        "trace_id": trace_id,
                    },
                )
                if (
                    run.status == DomainOpsRunStatus.FAILED.value
                    and task.error_code == "WORKER_LEASE_EXPIRED"
                ):
                    event = await uow.runs.append_event(
                        ops_run_id=run.ops_run_id,
                        event_type="run.failed",
                        event_key=f"run:{run.ops_run_id}:terminal",
                        visibility="USER",
                        payload_json={
                            "status": "FAILED",
                            "error_code": "WORKER_LEASE_EXPIRED",
                            "trace_id": trace_id,
                        },
                    )
                    await self._add_multi_resource_terminal_outbox(
                        uow, run=run, trace_id=trace_id, now=now
                    )
                    assert uow.platform_notifications is not None
                    await uow.platform_notifications.emit_run_event(
                        run=run,
                        event_type="aiops.run.failed",
                        summary=run.error_message or "诊断运行失败",
                        actor_id=run.actor_id,
                    )
                await uow.commit()
                return True
            task = await uow.runs.lock_due_retry_task(now=now)
            if task is not None:
                ensure_task_transition(
                    DomainOpsTaskStatus(task.status),
                    DomainOpsTaskStatus.READY,
                )
                task.status = DomainOpsTaskStatus.READY.value
                await uow.runs.append_event(
                    ops_run_id=task.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    event_type="task.status",
                    event_key=(
                        f"task:{task.ops_task_id}:retry-ready:"
                        f"{int(task.attempt_count)}"
                    ),
                    visibility="USER",
                    payload_json={
                        "status": "READY",
                        "task_id": str(task.ops_task_id),
                        "task_type": task.task_type,
                        "task_key": task.task_key,
                        "trace_id": trace_id,
                    },
                )
                await uow.commit()
                return True
            return False

    async def get_dashboard(
        self, *, scope: ConfigurationScope
    ) -> OpsDashboard:
        """投影域内全部 Target 的 DBA Dashboard，不过滤当前 Agent。"""
        now = datetime.now(UTC)
        since = now - timedelta(hours=24)
        async with self._uow_factory() as uow:
            targets = await uow.targets.list_scoped(domain_id=scope.domain_id)
            situations = await uow.situations.list_open_for_domain(
                domain_id=scope.domain_id
            )
            latest_runs = await uow.runs.list_latest_completed_by_target(
                domain_id=scope.domain_id
            )
            recent_runs = await uow.runs.list_recent_for_dashboard(
                domain_id=scope.domain_id,
                since=since,
            )
            failed_runs = await uow.runs.list_latest_failed_by_target_since(
                domain_id=scope.domain_id,
                since=since,
            )
            run_status_counts = await uow.runs.count_statuses_since(
                domain_id=scope.domain_id,
                since=since,
            )
            inspection_status_counts = (
                await uow.inspections.count_fire_statuses_since(
                    domain_id=scope.domain_id,
                    since=since,
                )
            )
            payloads = await uow.turns.list_finding_blocks_for_runs(
                ops_run_ids=tuple(item.ops_run_id for item in latest_runs)
            )
            return project_ops_dashboard(
                targets=tuple(
                    DashboardTargetSnapshot(
                        target_id=item.target_id,
                        display_name=item.display_name,
                        db_type=item.db_type,
                        environment=item.environment,
                        db_role=item.db_role,
                        importance_level=int(item.importance_level),
                        status=item.status,
                        connectivity_status=item.connectivity_status,
                        observed_status=item.observed_status,
                        readonly_connection_enabled=bool(
                            item.readonly_connection_enabled
                        ),
                        last_observed_at=item.last_observed_at,
                        last_error_code=item.last_error_code,
                    )
                    for item in targets
                ),
                situations=tuple(
                    DashboardSituationSnapshot(
                        situation_id=item.situation_id,
                        target_id=item.target_id,
                        status=item.status,
                        severity=item.severity,
                        title=item.title,
                        first_observed_at=item.first_observed_at,
                        last_observed_at=item.last_observed_at,
                    )
                    for item in situations
                ),
                latest_runs=tuple(
                    DashboardRunSnapshot(
                        target_id=item.target_id,
                        ops_run_id=item.ops_run_id,
                        status=item.status,
                        created_at=item.created_at,
                        completed_at=item.completed_at,
                        error_code=item.error_code,
                        findings=parse_finding_payload(
                            payloads.get(item.ops_run_id)
                        ),
                    )
                    for item in latest_runs
                ),
                recent_runs=tuple(
                    DashboardRunSnapshot(
                        target_id=item.target_id,
                        ops_run_id=item.ops_run_id,
                        status=item.status,
                        created_at=item.created_at,
                        completed_at=item.completed_at,
                        error_code=item.error_code,
                    )
                    for item in recent_runs
                ),
                failed_runs=tuple(
                    DashboardRunSnapshot(
                        target_id=item.target_id,
                        ops_run_id=item.ops_run_id,
                        status=item.status,
                        created_at=item.created_at,
                        completed_at=item.completed_at,
                        error_code=item.error_code,
                    )
                    for item in failed_runs
                ),
                run_status_counts=run_status_counts,
                inspection_status_counts=inspection_status_counts,
                now=now,
            )

    async def list_runs(
        self, *, scope: ConfigurationScope, target_id: UUID | None,
        status: str | None, agent_ids: tuple[UUID, ...],
        cursor: str | None, limit: int,
    ) -> OpsRunPage:
        filters = {"target_id": str(target_id) if target_id else None, "status": status}
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._decode_cursor(token=cursor, scope=scope, filters=filters)
        async with self._uow_factory() as uow:
            entities = await uow.runs.page_runs(
                domain_id=scope.domain_id, target_id=target_id, status=status,
                agent_ids=agent_ids, before_created_at=before_at,
                before_id=before_id, limit=limit + 1,
            )
            page = entities[:limit]
            return OpsRunPage(
                items=tuple(self._run_summary(item) for item in page),
                next_cursor=self._next_cursor(scope=scope, filters=filters,
                    entities=entities, page_entities=page, limit=limit,
                    id_attribute="ops_run_id"),
                has_more=len(entities) > limit,
            )

    async def list_situations(
        self, *, scope: ConfigurationScope, target_id: UUID | None,
        agent_id: UUID | None, status: str | None, severity: str | None,
        cursor: str | None, limit: int,
    ) -> SituationPage:
        filters = {"target_id": str(target_id) if target_id else None,
                   "agent_id": str(agent_id) if agent_id else None,
                   "status": status, "severity": severity}
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._decode_cursor(token=cursor, scope=scope, filters=filters)
        async with self._uow_factory() as uow:
            entities = await uow.situations.page_situations(
                domain_id=scope.domain_id, target_id=target_id,
                agent_id=agent_id, status=status, severity=severity,
                before_created_at=before_at,
                before_id=before_id, limit=limit + 1,
            )
            page = entities[:limit]
            return SituationPage(
                items=tuple(self._situation_summary(item) for item in page),
                next_cursor=self._next_cursor(scope=scope, filters=filters,
                    entities=entities, page_entities=page, limit=limit,
                    id_attribute="situation_id"),
                has_more=len(entities) > limit,
            )

    async def get_situation(self, *, situation_id: UUID, domain_id: int) -> SituationView:
        async with self._uow_factory() as uow:
            situation = await uow.situations.get_situation_scoped(
                situation_id=situation_id, domain_id=domain_id
            )
            if situation is None:
                raise resource_not_found("Situation")
            source_summaries = (
                await uow.situations.summarize_sources_for_situation(
                    situation_id=situation_id
                )
            )
            events = await uow.situations.list_events_for_situation(
                situation_id=situation_id,
                limit=20,
            )
            runs = await uow.runs.list_by_situation(situation_id=situation_id)
            return SituationView(
                **self._situation_summary(situation).model_dump(),
                monitoring_sources=tuple(
                    SituationMonitoringSourceSummary(**item)
                    for item in source_summaries
                ),
                signal_events=tuple(SignalEventSummary(
                    signal_event_id=item.signal_event_id,
                    diagnostic_source_id=item.diagnostic_source_id,
                    source_event_key=item.source_event_key,
                    signal_kind=item.signal_kind, event_class=item.event_class,
                    severity=item.severity, normalized_status=item.normalized_status,
                    summary=item.summary, occurred_at=item.occurred_at,
                ) for item in events),
                run_ids=tuple(item.ops_run_id for item in runs),
            )

    async def get_run(
        self, *, ops_run_id: UUID, domain_id: int
    ) -> OpsRunSummary:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_run_scoped(
                ops_run_id=ops_run_id,
                domain_id=domain_id,
            )
            if run is None:
                raise resource_not_found("Ops Run")
            final = None
            if run.final_artifact_id is not None:
                artifacts = await uow.runs.list_artifacts(
                    ops_run_id=run.ops_run_id
                )
                artifact = next(
                    (
                        item
                        for item in artifacts
                        if item.artifact_id == run.final_artifact_id
                    ),
                    None,
                )
                if artifact is not None:
                    final = ArtifactRef(
                        artifact_id=artifact.artifact_id,
                        artifact_type=artifact.artifact_type,
                        schema_version=artifact.schema_version,
                        content_hash=artifact.content_hash,
                    )
            from aiops_agent.application.readonly_lifecycle import read_lifecycle
            return self._run_summary(run, final_artifact=final).model_copy(
                update={"readonly_lifecycle": await read_lifecycle(uow=uow, run=run)})

    async def get_run_result(
        self, *, ops_run_id: UUID, domain_id: int
    ) -> OpsRunResult:
        """在校验 Domain 边界后读取 Run 的最终可展示产物。"""
        async with self._uow_factory() as uow:
            run = await uow.runs.get_run_scoped(
                ops_run_id=ops_run_id,
                domain_id=domain_id,
            )
            if run is None:
                raise resource_not_found("Ops Run")

            artifact = None
            if run.final_artifact_id is not None:
                candidate = await uow.runs.get_artifact(
                    artifact_id=run.final_artifact_id
                )
                if (
                    candidate is not None
                    and candidate.ops_run_id == run.ops_run_id
                ):
                    artifact = candidate

            payload = (
                dict(artifact.payload_json or {})
                if artifact is not None
                else None
            )
            if run.workflow_kind == FOLLOWUP_WORKFLOW:
                payload = await future_observation_payload(
                    uow=uow, run=run, now=await uow.runs.database_now(), artifact=artifact,
                )
                if payload is None:
                    raise state_conflict("未来观察的来源或当前授权无法重放，未返回原始查询结果")
            if payload is not None:
                payload["display_dictionary"] = dictionary_for_payload(payload)

            return OpsRunResult(
                ops_run_id=run.ops_run_id,
                status=run.status,
                root_cause_grade=(
                    self._artifact_root_cause_grade(artifact)
                    if artifact is not None
                    else None
                ),
                final_artifact=(
                    self._artifact_ref(artifact)
                    if artifact is not None
                    else None
                ),
                payload=payload,
                completed_at=run.completed_at,
            )

    async def list_inspection_fires(
        self,
        *,
        scope: ConfigurationScope,
        plan_id: UUID | None,
        status: str | None,
        cursor: str | None,
        limit: int,
    ) -> InspectionFirePage:
        allowed = {
            "QUEUED",
            "RUNNING",
            "COMPLETED",
            "PARTIAL",
            "FAILED",
            "SKIPPED",
            "CANCELLED",
        }
        if status is not None and status not in allowed:
            raise validation_failed("Inspection Fire status 过滤条件无效")
        filters = {
            "plan_id": str(plan_id) if plan_id is not None else None,
            "status": status,
        }
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._decode_cursor(
                token=cursor,
                scope=scope,
                filters=filters,
            )
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            entities = await uow.inspections.page_fires(
                domain_id=scope.domain_id,
                plan_id=plan_id,
                statuses=(status,) if status else None,
                before_created_at=before_at,
                before_id=before_id,
                limit=limit + 1,
            )
            page_entities = entities[:limit]
            next_cursor = self._next_cursor(
                scope=scope,
                filters=filters,
                entities=entities,
                page_entities=page_entities,
                limit=limit,
                id_attribute="inspection_fire_id",
            )
            return InspectionFirePage(
                items=tuple(
                    self._fire_summary(item) for item in page_entities
                ),
                next_cursor=next_cursor,
                has_more=len(entities) > limit,
            )

    async def get_inspection_fire(
        self,
        *,
        inspection_fire_id: UUID,
        domain_id: int,
    ) -> InspectionFireView:
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            fire = await uow.inspections.get_fire_scoped(
                inspection_fire_id=inspection_fire_id,
                domain_id=domain_id,
            )
            if fire is None:
                raise resource_not_found("Inspection Fire")
            runs = await uow.inspections.list_runs_for_fire(
                inspection_fire_id=inspection_fire_id
            )
            assert uow.runs is not None
            run_summaries = []
            for item in runs:
                final_artifact = (
                    await uow.runs.get_artifact(
                        artifact_id=item.final_artifact_id
                    )
                    if item.final_artifact_id is not None
                    else None
                )
                report_status = None
                evidence_coverage_percent = None
                evidence_covered_count = None
                evidence_total_count = None
                quality_scope: dict[str, Any] = {}
                if final_artifact is not None:
                    artifact_payload = dict(final_artifact.payload_json or {})
                    artifact_status = str(artifact_payload.get("status") or "")
                    report_status = {
                        "COMPLETED": "READY",
                        "READY": "READY",
                        "PARTIAL": "PARTIAL",
                        "FAILED": "FAILED",
                        "GENERATING": "GENERATING",
                    }.get(artifact_status)
                    artifact_scope = dict(artifact_payload.get("scope") or {})
                    quality_scope = artifact_scope
                    evidence_coverage_percent = artifact_scope.get(
                        "evidence_coverage_percent"
                    )
                    evidence_covered_count = artifact_scope.get(
                        "evidence_covered_count"
                    )
                    evidence_total_count = artifact_scope.get(
                        "evidence_total_count"
                    )
                run_summaries.append(
                    InspectionRunSummary(
                        run_id=item.ops_run_id,
                        target_id=item.target_id,
                        status=item.status,
                        report_status=report_status,
                        evidence_coverage_percent=evidence_coverage_percent,
                        evidence_covered_count=evidence_covered_count,
                        evidence_total_count=evidence_total_count,
                        tool_execution_coverage_percent=quality_scope.get(
                            "tool_execution_coverage_percent"
                        ),
                        tool_execution_covered_count=quality_scope.get(
                            "tool_execution_covered_count"
                        ),
                        tool_execution_total_count=quality_scope.get(
                            "tool_execution_total_count"
                        ),
                        monitoring_sample_coverage_percent=quality_scope.get(
                            "monitoring_sample_coverage_percent"
                        ),
                        monitoring_metric_count=quality_scope.get(
                            "monitoring_metric_count"
                        ),
                        monitoring_low_coverage_metric_count=quality_scope.get(
                            "monitoring_low_coverage_metric_count"
                        ),
                        evidence_gap_count=quality_scope.get(
                            "evidence_gap_count"
                        ),
                        error_code=item.error_code,
                        error_message=item.error_message,
                        completed_at=item.completed_at,
                    )
                )
            summary = self._fire_summary(fire)
            return InspectionFireView(
                **summary.model_dump(),
                runs=tuple(run_summaries),
                created_at=fire.created_at,
            )

    async def list_reports(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID | None,
        report_type: str | None,
        cursor: str | None,
        limit: int,
    ) -> ReportPage:
        allowed = {
            "INCIDENT",
            "PERFORMANCE",
            "INSPECTION_DAILY",
            "INSPECTION_WEEKLY",
            "INSPECTION_CUSTOM",
            "COMPARISON",
        }
        if report_type is not None and report_type not in allowed:
            raise validation_failed("Report type 过滤条件无效")
        filters = {
            "target_id": (
                str(target_id) if target_id is not None else None
            ),
            "report_type": report_type,
        }
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._decode_cursor(
                token=cursor,
                scope=scope,
                filters=filters,
            )
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            entities = await uow.inspections.page_current_reports(
                domain_id=scope.domain_id,
                target_id=target_id,
                report_type=report_type,
                before_created_at=before_at,
                before_id=before_id,
                limit=limit + 1,
            )
            page_entities = entities[:limit]
            next_cursor = self._next_cursor(
                scope=scope,
                filters=filters,
                entities=entities,
                page_entities=page_entities,
                limit=limit,
                id_attribute="report_id",
            )
            return ReportPage(
                items=tuple(
                    self._report_summary(item) for item in page_entities
                ),
                next_cursor=next_cursor,
                has_more=len(entities) > limit,
            )

    async def list_report_versions(
        self,
        *,
        scope: ConfigurationScope,
        report_id: UUID,
        cursor: str | None,
        limit: int,
    ) -> ReportVersionPage:
        filters = {"report_id": str(report_id)}
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._decode_cursor(
                token=cursor,
                scope=scope,
                filters=filters,
            )
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            anchor = await uow.inspections.get_report_scoped(
                report_id=report_id,
                domain_id=scope.domain_id,
            )
            if anchor is None:
                raise resource_not_found("Report")
            entities = await uow.inspections.page_report_versions(
                ops_run_id=anchor.ops_run_id,
                source_conversation_id=getattr(
                    anchor, "source_conversation_id", None
                ),
                report_key=anchor.report_key,
                before_created_at=before_at,
                before_id=before_id,
                limit=limit + 1,
            )
            page_entities = entities[:limit]
            next_cursor = self._next_cursor(
                scope=scope,
                filters=filters,
                entities=entities,
                page_entities=page_entities,
                limit=limit,
                id_attribute="report_id",
            )
            return ReportVersionPage(
                items=tuple(
                    ReportVersionSummary(
                        report_id=item.report_id,
                        report_version=int(item.report_version),
                        status=item.status,
                        published_at=item.created_at,
                    )
                    for item in page_entities
                ),
                next_cursor=next_cursor,
                has_more=len(entities) > limit,
            )

    async def get_report(
        self,
        *,
        report_id: UUID,
        domain_id: int,
    ) -> ReportView:
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            report = await uow.inspections.get_report_scoped(
                report_id=report_id,
                domain_id=domain_id,
            )
            if report is None or report.content_artifact_id is None:
                raise resource_not_found("Report")
            artifact = await uow.runs.get_artifact(
                artifact_id=report.content_artifact_id
            )
            if not report_content_reference_is_complete(report, artifact):
                raise state_conflict("Report 内容引用不完整")
            return self._report_view(report, artifact)

    async def edit_report(
        self,
        *,
        report_id: UUID,
        domain_id: int,
        actor_id: str,
        expected_report_version: int,
        title: str,
        sections: tuple[ReportSectionEdit, ...],
        trace_id: str,
    ) -> ReportView:
        """冻结人工展示编辑为下一版报告，不改写已有事实和证据。"""
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            report = await uow.inspections.get_report_scoped(
                report_id=report_id, domain_id=domain_id, lock=True
            )
            if report is None:
                raise resource_not_found("Report")
            if not report.is_current:
                raise state_conflict("报告已更新，请刷新后再编辑")
            if int(report.report_version) != expected_report_version:
                raise state_conflict("报告版本已变化，请刷新后再编辑")
            if report.content_artifact_id is None:
                raise state_conflict("Report 内容引用不完整")
            artifact = await uow.runs.get_artifact(
                artifact_id=report.content_artifact_id
            )
            if not report_content_reference_is_complete(report, artifact):
                raise state_conflict("Report 内容引用不完整")
            content = ReportContent.model_validate(
                dict(artifact.payload_json or {})
            )
            snapshot = dict(content.provenance.get("template") or {})
            definition = snapshot.get("definition")
            if isinstance(definition, dict):
                template = validate_template_definition(definition)
            else:
                template = resolve_historical_report_template(
                    template_ref=str(report.template_id), report_type=str(report.report_type),
                )
            if template is None:
                raise state_conflict("历史报告缺少可重现的模板快照")
            protected = {"EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX"}
            overrides = dict(content.presentation_overrides)
            edited_kinds: set[str] = set()
            for section in sections:
                if section.kind in protected:
                    raise validation_failed("证据边界和证据索引不能人工编辑")
                if section.kind not in template.sections:
                    raise validation_failed("报告包含不属于模板的章节")
                if section.kind in edited_kinds:
                    raise validation_failed("同一报告章节只能编辑一次")
                edited_kinds.add(section.kind)
                overrides[section.kind] = tuple(
                    item.strip() for item in section.items if item.strip()
                )
                if not overrides[section.kind]:
                    raise validation_failed("报告章节不能保存为空")
            clean_title = title.strip()
            if not clean_title:
                raise validation_failed("报告标题不能为空")
            now = await uow.runs.database_now()
            version = int(report.report_version) + 1
            provenance = dict(content.provenance)
            provenance["human_presentation_edit"] = {
                "actor_id": actor_id,
                "edited_at": now.isoformat(),
                "base_report_id": str(report.report_id),
                "base_content_hash": report.content_hash,
            }
            updated_content = content.model_copy(update={
                "title": clean_title,
                "presentation_overrides": overrides,
                "provenance": provenance,
            })
            payload = updated_content.model_dump(mode="json")
            from aiops_agent.application.report_source_revision import report_ancestors, same_report_diagnosis
            presentation_only = same_report_diagnosis(content.model_dump(mode="json"), payload)
            content_hash = sha256_json(payload)
            report_artifact = await uow.runs.add_artifact(
                OpsArtifactEntity(
                    ops_run_id=report.ops_run_id,
                    ops_task_id=report.generated_by_task_id,
                    artifact_key=f"report:{report.report_key}:v{version}",
                    artifact_type="REPORT_CONTENT",
                    schema_version="REPORT_CONTENT.v1",
                    payload_json=payload,
                    content_hash=content_hash,
                    byte_size=len(canonical_bytes(payload)),
                    provenance_json={
                        "producer": "aiops.report-editor",
                        "producer_version": "1",
                        "base_artifact_id": str(artifact.artifact_id),
                        "actor_id": actor_id,
                    },
                    trust_level="USER_PROVIDED",
                    security_level=int(report.security_level),
                )
            )
            summary = overrides.get("EXECUTIVE_SUMMARY", (content.summary,))[0]
            saved = await uow.inspections.publish_report(
                ReportEntity(
                    report_id=uuid7(),
                    ops_run_id=report.ops_run_id,
                    source_conversation_id=getattr(
                        report, "source_conversation_id", None
                    ),
                    target_id=report.target_id,
                    report_key=report.report_key,
                    report_version=version,
                    supersedes_report_id=report.report_id,
                    is_current=0,
                    report_type=report.report_type,
                    title=updated_content.title,
                    status=report.status,
                    period_start=report.period_start,
                    period_end=report.period_end,
                    baseline_start=report.baseline_start,
                    baseline_end=report.baseline_end,
                    after_start=report.after_start,
                    after_end=report.after_end,
                    result=report.result,
                    template_id=report.template_id,
                    template_version=report.template_version,
                    generated_by_task_id=report.generated_by_task_id,
                    content_artifact_id=report_artifact.artifact_id,
                    content_hash=content_hash,
                    summary=summary,
                    security_level=int(report.security_level),
                    schema_version="REPORT_CONTENT.v1",
                )
            )
            sources = await uow.inspections.list_report_sources(
                report_id=report.report_id
            )
            if sources:
                await uow.inspections.add_report_sources([
                    ReportSourceEntity(
                        report_id=saved.report_id,
                        ops_run_id=source.ops_run_id,
                        source_artifact_id=source.source_artifact_id,
                        source_kind=source.source_kind,
                        content_hash=source.content_hash,
                        observed_at=source.observed_at,
                    )
                    for source in sources
                ])
            await uow.runs.append_event(
                ops_run_id=saved.ops_run_id,
                ops_task_id=saved.generated_by_task_id,
                event_type="report.edited",
                event_key=f"report:{saved.report_id}:edited",
                visibility="USER",
                payload_json={
                    "report_id": str(saved.report_id),
                    "supersedes_report_id": str(report.report_id),
                    "correction_kind": "PRESENTATION_ONLY" if presentation_only else "SUBSTANTIVE",
                    "report_version": version,
                    "trace_id": trace_id,
                },
            )
            await self._add_outbox(
                uow,
                aggregate_id=saved.report_id,
                event_type="OPS_REPORT_EDITED",
                idempotency_key=f"report:{saved.report_id}:edited",
                payload={
                    "report_id": str(saved.report_id),
                    "ops_run_id": str(saved.ops_run_id),
                    "report_version": version,
                    "correction_kind": "PRESENTATION_ONLY" if presentation_only else "SUBSTANTIVE",
                },
                trace_id=trace_id,
                now=now,
            )
            knowledge_repository = getattr(uow, "operations_knowledge", None)
            if knowledge_repository is not None and not presentation_only:
                for affected_report in [report, *await report_ancestors(uow=uow, domain_id=domain_id, report=report)]:
                    affected_cases = await knowledge_repository.mark_cases_for_report_review(
                        report_id=affected_report.report_id, reviewer_id=actor_id,
                    )
                    for case_asset, case_version in affected_cases:
                        await queue_case_withdrawals(uow=uow, asset=case_asset, version=case_version)
            await uow.commit()
            return self._report_view(saved, report_artifact)

    async def generate_user_report(
        self,
        *,
        domain_id: int,
        actor_id: str,
        ops_run_id: UUID,
        template: ReportTemplate,
        period_kind: str,
        trace_id: str,
    ) -> GeneratedReportReceipt:
        """从完成的聊天或告警诊断显式创建正式报告。"""
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            run = await uow.runs.get_run_scoped(
                ops_run_id=ops_run_id, domain_id=domain_id, lock=True
            )
            if run is None:
                raise resource_not_found("OpsRun")
            source_kind = (
                "CHAT" if run.trigger_type == "CHAT"
                else "INSPECTION" if run.trigger_type == "SCHEDULE"
                else "ALERT"
            )
            if source_kind not in template.applicable_source_kinds:
                raise validation_failed("所选报告模板不适用于当前诊断入口")
            if source_kind != "INSPECTION":
                period_kind = "AD_HOC"
            elif period_kind not in {"DAILY", "MONTHLY", "QUARTERLY", "ANNUAL"}:
                raise validation_failed("巡检报告周期无效")
            if period_kind not in template.allowed_period_kinds:
                raise validation_failed("所选报告模板不适用于当前报告周期")
            if run.status not in {"COMPLETED", "PARTIAL"}:
                raise state_conflict("诊断尚未结束，暂不能生成正式报告")
            if run.final_artifact_id is None:
                raise state_conflict("诊断未形成可报告的最终结果")
            source_artifact = await uow.runs.get_artifact(
                artifact_id=run.final_artifact_id
            )
            if source_artifact is None:
                raise state_conflict("诊断最终结果引用不完整")
            if source_artifact.schema_version not in {
                "DIAGNOSIS_REPORT_DRAFT.v1", "AIOPS_TURN_RESULT.v1",
                "DB_DIAGNOSTIC_REPORT.v1", "REPORT_CONTENT.v1",
            }:
                raise validation_failed("当前诊断结果不支持生成正式报告")
            task_artifact_id = source_artifact.artifact_id
            if source_artifact.schema_version == "REPORT_CONTENT.v1":
                source_artifact_id = str(
                    dict(source_artifact.provenance_json or {}).get(
                        "source_artifact_id"
                    )
                    or ""
                )
                if source_artifact_id:
                    try:
                        parent_artifact = await uow.runs.get_artifact(
                            artifact_id=UUID(source_artifact_id)
                        )
                    except ValueError:
                        parent_artifact = None
                    if parent_artifact is not None:
                        task_artifact_id = parent_artifact.artifact_id
            task = next(
                (
                    item for item in await uow.runs.list_tasks(
                        ops_run_id=run.ops_run_id
                    )
                    if item.output_artifact_id == task_artifact_id
                ),
                None,
            )
            if task is None:
                raise state_conflict("诊断最终结果缺少生成任务")
            now = await uow.runs.database_now()
            source_override = None
            period_start = None
            period_end = None
            if source_kind == "INSPECTION" and period_kind != "DAILY":
                inspection = dict(
                    dict(run.plan_snapshot_json or {}).get(
                        "client_metadata", {}
                    ).get("inspection", {})
                )
                timezone = str(inspection.get("timezone") or "UTC")
                period_start, period_end = closed_period_window(
                    period_kind=period_kind, timezone=timezone, now=now
                )
                period_runs = await uow.runs.list_completed_inspection_runs(
                    domain_id=domain_id,
                    target_id=run.target_id,
                    period_start=period_start,
                    period_end=period_end,
                )
                if not period_runs:
                    raise state_conflict("当前完整报告周期内没有可汇总的巡检结果")
                if run.ops_run_id not in {
                    item.ops_run_id for item in period_runs
                }:
                    raise validation_failed(
                        "请从该完整报告周期内的巡检结果发起周期报告"
                    )
                source_override = await self._aggregate_inspection_sources(
                    uow=uow, runs=period_runs, period_kind=period_kind,
                    period_start=period_start, period_end=period_end,
                )
            template = report_template_for_source(
                template=template, plan=dict(run.plan_snapshot_json or {}), source_kind=source_kind,
            )
            current = await uow.inspections.get_current_report_for_run_template(
                ops_run_id=run.ops_run_id,
                template_id=template.template_ref,
            )
            if current is not None and (
                period_start is None or current.period_start == period_start
            ):
                current_artifact = await uow.runs.get_artifact(artifact_id=current.content_artifact_id)
                normalized_source = source_override or normalize_report_source(
                    schema_version=source_artifact.schema_version,
                    payload=dict(source_artifact.payload_json or {}), source_kind=source_kind,
                )
                if (
                    report_content_reference_is_complete(current, current_artifact)
                    and dict(dict(current_artifact.payload_json or {}).get("provenance") or {}).get("report_input_hash")
                    == report_input_hash(
                        source_artifact=source_artifact, source=normalized_source, template=template,
                        period_kind=period_kind, period_start=period_start, period_end=period_end,
                    )
                ):
                    from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
                    if not await validate_investigation_sources(uow=uow, run=run,
                        payload=dict(source_artifact.payload_json or {}),
                        plan_capture_reader=getattr(self, "_plan_capture_reader", None)):
                        raise state_conflict("报告原始调查来源失效，不能复用旧报告")
                    return GeneratedReportReceipt.from_entity(current)
            report = await self._publish_diagnosis_report(
                uow=uow,
                run=run,
                task=task,
                source_artifact=source_artifact,
                now=now,
                trace_id=trace_id,
                template=template,
                actor_id=actor_id,
                source_override=source_override,
                period_start_override=period_start,
                period_end_override=period_end,
                period_kind=period_kind,
            )
            receipt = GeneratedReportReceipt.from_entity(report)
            await uow.commit()
            return receipt

    async def get_conversation_source_agent_id(
        self,
        *,
        conversation_id: UUID,
        domain_id: int,
        actor_id: str,
    ) -> UUID:
        """在生成前确认会话归属，并返回来源 Agent。"""
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(
                conversation_id=conversation_id,
                domain_id=domain_id,
            )
            if conversation is None or conversation.created_by != actor_id:
                raise resource_not_found("Conversation")
            return conversation.agent_id

    async def generate_conversation_report(
        self,
        *,
        domain_id: int,
        actor_id: str,
        conversation_id: UUID,
        template: ReportTemplate,
        trace_id: str,
    ) -> GeneratedReportReceipt:
        """冻结一个智能运维 Session 内全部已终态 Turn 的报告上下文。"""
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            conversation = await uow.conversations.get_conversation(
                conversation_id=conversation_id,
                domain_id=domain_id,
                lock=True,
            )
            if conversation is None or conversation.created_by != actor_id:
                raise resource_not_found("Conversation")
            if "CHAT" not in template.applicable_source_kinds:
                raise validation_failed("所选报告模板不适用于智能运维会话")
            if "AD_HOC" not in template.allowed_period_kinds:
                raise validation_failed("所选报告模板不适用于会话诊断报告")
            turns = await uow.turns.list_all_turns(
                conversation_id=conversation_id
            )
            if not turns:
                raise state_conflict("当前会话尚无可生成报告的诊断内容")
            active_turns = [
                turn for turn in turns
                if turn.status not in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}
            ]
            if active_turns:
                raise state_conflict("会话仍有进行中的诊断，暂不能生成正式报告")

            source_rows: list[tuple[Any, Any, Any]] = []
            missing_turns: list[dict[str, Any]] = []
            for turn in turns:
                link = await uow.turns.get_run_link(
                    turn_id=turn.turn_id, purpose="PRIMARY"
                )
                if link is None:
                    missing_turns.append({
                        "code": "MISSING_PRIMARY_RUN",
                        "turn_id": str(turn.turn_id),
                        "turn_no": int(turn.turn_no),
                    })
                    continue
                run = await uow.runs.get_run_scoped(
                    ops_run_id=link.ops_run_id,
                    domain_id=domain_id,
                    lock=True,
                )
                if (
                    run is None
                    or run.status not in {"COMPLETED", "PARTIAL"}
                    or run.final_artifact_id is None
                ):
                    missing_turns.append({
                        "code": "MISSING_FINAL_RESULT",
                        "turn_id": str(turn.turn_id),
                        "turn_no": int(turn.turn_no),
                    })
                    continue
                artifact = await uow.runs.get_artifact(
                    artifact_id=run.final_artifact_id
                )
                if artifact is None or artifact.schema_version not in {
                    "DIAGNOSIS_REPORT_DRAFT.v1", "AIOPS_TURN_RESULT.v1",
                    "DB_DIAGNOSTIC_REPORT.v1", "REPORT_CONTENT.v1",
                }:
                    missing_turns.append({
                        "code": "UNREPORTABLE_FINAL_RESULT",
                        "turn_id": str(turn.turn_id),
                        "turn_no": int(turn.turn_no),
                    })
                    continue
                from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
                if not await validate_investigation_sources(uow=uow, run=run, payload=dict(artifact.payload_json or {}),
                    plan_capture_reader=getattr(self, "_plan_capture_reader", None)):
                    raise state_conflict("调查计算或对象关系来源失效，不能汇入会话报告")
                source_rows.append((turn, run, artifact))
            if not source_rows:
                raise state_conflict("当前会话尚未形成可报告的诊断结果")

            _, anchor_run, anchor_artifact = source_rows[-1]
            task_artifact_id = anchor_artifact.artifact_id
            if anchor_artifact.schema_version == "REPORT_CONTENT.v1":
                source_artifact_id = str(
                    dict(anchor_artifact.provenance_json or {}).get(
                        "source_artifact_id"
                    ) or ""
                )
                if source_artifact_id:
                    try:
                        parent_artifact = await uow.runs.get_artifact(
                            artifact_id=UUID(source_artifact_id)
                        )
                    except ValueError:
                        parent_artifact = None
                    if parent_artifact is not None:
                        task_artifact_id = parent_artifact.artifact_id
            task = next(
                (
                    item for item in await uow.runs.list_tasks(
                        ops_run_id=anchor_run.ops_run_id
                    )
                    if item.output_artifact_id == task_artifact_id
                ),
                None,
            )
            if task is None:
                raise state_conflict("会话最终诊断结果缺少生成任务")
            now = await uow.runs.database_now()
            source_override = self._aggregate_conversation_sources(
                conversation=conversation,
                source_rows=source_rows,
                missing_turns=missing_turns,
            )
            closure_artifact = None
            report_run = anchor_run
            report_task = task
            closure_candidates = []
            for _, candidate_run, _ in reversed(source_rows):
                candidate = await uow.runs.get_latest_artifact_by_prefix(
                    ops_run_id=candidate_run.ops_run_id,
                    artifact_key_prefix=f"diagnosis-closure:{conversation_id}:",
                )
                if (
                    candidate is None
                    or candidate.schema_version != "DIAGNOSIS_CLOSURE.v1"
                ):
                    continue
                closure_candidates.append((candidate_run, candidate))
            # 收口先后以不可变原件登记时间为准，不由所选最后一轮决定。
            # 最新收口失效时保留缺口，不能退回更早人工判断而掩盖更正。
            closure_candidates.sort(key=lambda row: (row[1].created_at, str(row[1].artifact_id)), reverse=True)
            for candidate_run, candidate in closure_candidates[:1]:
                from aiops_agent.application.case_closure import validate_closure_sources
                if not await validate_closure_sources(uow=uow, domain_id=domain_id, run=candidate_run,
                    artifact=candidate, plan_capture_reader=getattr(self, "_plan_capture_reader", None)):
                    source_override["gaps"] = (*source_override.get("gaps", ()), {
                        "code": "CLOSURE_SOURCE_INVALIDATED",
                        "message": "人工收口的来源或授权已变化，未作为报告事实附录。",
                    })
                    source_override["status"] = "PARTIAL"
                    continue
                candidate_tasks = await uow.runs.list_tasks(
                    ops_run_id=candidate_run.ops_run_id
                )
                candidate_task = next((
                    item for item in candidate_tasks
                    if item.ops_task_id == candidate.ops_task_id
                ), None)
                if candidate_task is None:
                    source_override["gaps"] = (*source_override.get("gaps", ()), {
                        "code": "CLOSURE_TASK_MISSING", "message": "最新人工收口缺少生成任务，未作为报告事实附录。",
                    })
                    source_override["status"] = "PARTIAL"
                    continue
                closure_artifact = candidate
                report_run = candidate_run
                report_task = candidate_task
                break
            if closure_artifact is not None:
                closure_source = normalize_report_source(
                    schema_version=closure_artifact.schema_version,
                    payload=dict(closure_artifact.payload_json or {}),
                    source_kind="CHAT",
                )
                # 人工收口作为附录，不覆盖多轮调查正文和后续反证。
                source_override["facts"] = (*source_override.get("facts", ()), *closure_source.get("facts", ()))
                source_override["closure_artifact_id"] = str(closure_artifact.artifact_id)
                source_override["verification"] = closure_source.get("verification", {})
                source_override["executed_actions"] = closure_source.get("executed_actions", ())
                source_override["action_source"] = closure_source.get("action_source")
                source_override["evidence_refs"] = (*source_override.get("evidence_refs", ()), *closure_source.get("evidence_refs", ()))
                source_override["conversation"] = {
                    **dict(source_override.get("conversation") or {}),
                    "title": conversation.title,
                    "question_summaries": list(dict.fromkeys(
                        str(dict(run.plan_snapshot_json or {}).get(
                            "diagnosis", {}
                        ).get("question_summary") or "").strip()
                        for _, run, _ in source_rows
                        if str(dict(run.plan_snapshot_json or {}).get(
                            "diagnosis", {}
                        ).get("question_summary") or "").strip()
                    )),
                }
            template = report_template_for_source(
                template=template, plan=dict(report_run.plan_snapshot_json or {}), source_kind="CHAT",
            )
            get_current_report = getattr(
                uow.inspections,
                "get_current_report_for_conversation_template",
                None,
            )
            current_report = (
                await get_current_report(
                    conversation_id=conversation_id,
                    template_id=template.template_ref,
                )
                if get_current_report else None
            )
            if (
                current_report is not None
                and current_report.template_version == template.version
                and current_report.content_artifact_id is not None
            ):
                current_artifact = await uow.runs.get_artifact(
                    artifact_id=current_report.content_artifact_id
                )
                if (
                    report_content_reference_is_complete(current_report, current_artifact)
                    and dict(dict(current_artifact.payload_json or {}).get("provenance") or {}).get("report_input_hash")
                    == report_input_hash(
                        source_artifact=closure_artifact or anchor_artifact,
                        source=source_override, template=template,
                    )
                ):
                    return GeneratedReportReceipt.from_entity(current_report)
            report = await self._publish_diagnosis_report(
                uow=uow,
                run=report_run,
                task=report_task,
                source_artifact=closure_artifact or anchor_artifact,
                now=now,
                trace_id=trace_id,
                template=template,
                actor_id=actor_id,
                source_override=source_override,
                period_start_override=(
                    getattr(source_rows[0][1], "started_at", None)
                    or source_rows[0][1].created_at
                ),
                period_end_override=now,
            )
            receipt = GeneratedReportReceipt.from_entity(report)
            await uow.commit()
            return receipt

    def _aggregate_conversation_sources(
        self,
        *,
        conversation,
        source_rows: list[tuple[Any, Any, Any]],
        missing_turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """将会话内各 Turn 的冻结结果聚合为单份可追溯 ReportContext。"""
        facts: list[dict[str, Any]] = []
        gaps = list(missing_turns)
        evidence_refs: list[dict[str, str]] = []
        recommendations: list[str] = []
        reasons: list[str] = []
        questions: list[str] = []
        root_assessments: list[dict[str, str]] = []
        investigation_goals: list[dict[str, Any]] = []
        effective_grade = "INCONCLUSIVE"
        ready = not missing_turns
        from aiops_agent.application.diagnosis.revisions import merge_investigation_sources
        normalized_rows = [
            (run, artifact, normalize_report_source(
                schema_version=artifact.schema_version,
                payload=dict(artifact.payload_json or {}), source_kind="CHAT",
            ))
            for turn, run, artifact in sorted(source_rows, key=lambda row: int(row[0].turn_no))
        ]
        current_sources, claim_history = merge_investigation_sources(normalized_rows)
        for turn, run, artifact in sorted(source_rows, key=lambda row: int(row[0].turn_no)):
            source = current_sources[str(artifact.artifact_id)]
            investigation_goals.extend({
                **item, "turn_id": str(turn.turn_id), "turn_no": int(turn.turn_no),
            } for item in source.get("investigation_goals", ()))
            facts.extend(source.get("facts") or ())
            turn_rationale = str(
                source.get("diagnosis_rationale") or ""
            ).strip()
            if turn_rationale:
                facts.append({
                    "summary": (
                        f"第 {int(turn.turn_no)} 轮诊断结论：{turn_rationale}"
                    ),
                    "turn_id": str(turn.turn_id),
                })
            gaps.extend(source.get("gaps") or ())
            ready = ready and source.get("status") == "READY"
            root = dict(source.get("root_cause") or {})
            grade = str(root.get("effective_level") or "INCONCLUSIVE")
            root_assessments.append({"turn_id": str(turn.turn_id), "effective_level": grade})
            # 多轮不能取最高等级掩盖反证；没有显式修订关联时保留各轮判断。
            effective_grade = grade if len(root_assessments) == 1 else "INCONCLUSIVE"
            solution = dict(source.get("solution") or {})
            recommendations.extend(
                str(item)
                for key in ("immediate_mitigations", "long_term_remediations")
                for item in solution.get(key, ())
                if item
            )
            reasons.extend(str(item) for item in source.get("report_decision_reasons") or ())
            question = str(
                dict(run.plan_snapshot_json or {}).get("diagnosis", {}).get(
                    "question_summary"
                ) or ""
            ).strip()
            if question:
                questions.append(question)
            evidence_refs.append({
                "artifact_id": str(artifact.artifact_id),
                "content_hash": artifact.content_hash,
                "schema_version": artifact.schema_version,
                "source_run_id": str(run.ops_run_id),
                "source_kind": "CHAT",
                "observed_at": (
                    (run.completed_at or run.created_at).isoformat()
                ),
            })
            for reference in source.get("evidence_refs") or ():
                if isinstance(reference, dict):
                    evidence_refs.append({
                        key: str(value) for key, value in reference.items()
                        if value is not None
                    })
        rationale = (
            f"本报告冻结会话“{conversation.title or '未命名诊断'}”的 "
            f"{len(source_rows)} 个已完成 Turn"
        )
        if missing_turns:
            rationale += f"；另有 {len(missing_turns)} 个 Turn 未形成可报告终态结果。"
        else:
            rationale += "。"
        return {
            "status": "READY" if ready else "PARTIAL",
            "root_cause": {"effective_level": effective_grade, "turn_assessments": root_assessments},
            "investigation_goals": tuple(investigation_goals),
            "diagnosis_rationale": rationale,
            "facts": tuple(facts),
            "gaps": tuple(gaps),
            "solution": {
                "long_term_remediations": tuple(dict.fromkeys(recommendations)),
            },
            "evidence_refs": tuple(evidence_refs),
            "report_decision_reasons": tuple(dict.fromkeys(reasons)),
            "conversation": {
                "conversation_id": str(conversation.conversation_id),
                "title": conversation.title,
                "turn_count": len(source_rows) + len(missing_turns),
                "included_turn_ids": [str(turn.turn_id) for turn, _, _ in source_rows],
                "question_summaries": list(dict.fromkeys(questions)),
                "claim_revision_history": list(claim_history),
            },
        }

    async def _aggregate_inspection_sources(
        self, *, uow, runs, period_kind: str, period_start: datetime,
        period_end: datetime,
    ) -> dict[str, Any]:
        """优先汇总已发布巡检报告，并冻结自然周期统计。"""
        facts, gaps, evidence_refs, recommendations = [], [], [], []
        partial_count = 0
        failed_count = 0
        completed_count = 0
        target_snapshot: dict[str, Any] = {}
        inspection_repository = getattr(uow, "inspections", None)
        get_run_report = getattr(
            inspection_repository, "get_current_report_for_run_template", None
        )
        ordered_runs = sorted(
            runs,
            key=lambda item: getattr(item, "completed_at", None)
            or getattr(item, "created_at", period_start),
        )
        for item in ordered_runs:
            failed_count += getattr(item, "status", "COMPLETED") in {
                "FAILED", "CANCELLED"
            }
            artifact = None
            if get_run_report is not None:
                for template_id in (
                    "system:inspection.daily",
                    "system:inspection.weekly",
                    "system:inspection.custom",
                ):
                    source_report = await get_run_report(
                        ops_run_id=item.ops_run_id,
                        template_id=template_id,
                    )
                    if source_report is not None and source_report.content_artifact_id:
                        artifact = await uow.runs.get_artifact(
                            artifact_id=source_report.content_artifact_id
                        )
                        if artifact is not None:
                            break
            if artifact is None and item.final_artifact_id is not None:
                artifact = await uow.runs.get_artifact(
                    artifact_id=item.final_artifact_id
                )
            if artifact is None:
                partial_count += 1
                gaps.append({
                    "code": "MISSING_FINAL_ARTIFACT",
                    "source_run_id": str(item.ops_run_id),
                })
                continue
            completed_count += 1
            payload = dict(artifact.payload_json or {})
            artifact_scope = dict(payload.get("scope") or {})
            if not target_snapshot and isinstance(artifact_scope.get("target"), dict):
                target_snapshot = dict(artifact_scope["target"])
            source = normalize_report_source(
                schema_version=artifact.schema_version,
                payload=payload,
                source_kind="INSPECTION",
            )
            partial_count += source["status"] != "READY"
            observed_at = (
                getattr(item, "completed_at", None)
                or getattr(item, "created_at", period_end)
            )
            for fact in source.get("facts") or ():
                if isinstance(fact, dict):
                    if period_kind == "MONTHLY" and is_monthly_backup_recovery_item(fact):
                        continue
                    facts.append({
                        **fact,
                        "_source_run_id": str(item.ops_run_id),
                        "_source_observed_at": observed_at.isoformat(),
                        "_source_target_id": str(getattr(item, "target_id", "")),
                        "_source_target_name": str(
                            dict(artifact_scope.get("target") or {}).get(
                                "display_name"
                            ) or ""
                        ),
                    })
            gaps.extend(
                value for value in source.get("gaps") or ()
                if period_kind != "MONTHLY"
                or not is_monthly_backup_recovery_item(value)
            )
            recommendations.extend(
                str(value)
                for value in dict(source.get("solution") or {}).get(
                    "long_term_remediations", ()
                )
                if value and (
                    period_kind != "MONTHLY"
                    or not is_monthly_backup_recovery_item(value)
                )
            )
            evidence_refs.append({
                "artifact_id": str(artifact.artifact_id),
                "content_hash": artifact.content_hash,
                "schema_version": artifact.schema_version,
                "source_run_id": str(item.ops_run_id),
                "source_kind": "INSPECTION",
                "observed_at": observed_at.isoformat(),
            })
        run_summary = {
            "scheduled_run_count": len(ordered_runs),
            "completed_run_count": completed_count,
            "partial_run_count": partial_count,
            "failed_run_count": failed_count,
        }
        monthly_summary = None
        previous_summary = None
        previous_comparison = None
        if period_kind == "MONTHLY":
            timezone = "UTC"
            if ordered_runs:
                inspection = dict(
                    dict(getattr(ordered_runs[-1], "plan_snapshot_json", {}) or {})
                    .get("client_metadata", {}).get("inspection", {})
                )
                timezone = str(inspection.get("timezone") or "UTC")
            monthly_summary = monthly_inspection_summary(
                facts=facts,
                period_start=period_start,
                period_end=period_end,
                timezone=timezone,
                run_summary=run_summary,
            )
            previous_getter = getattr(
                inspection_repository, "get_previous_current_period_report", None
            )
            target_id = getattr(ordered_runs[0], "target_id", None) if ordered_runs else None
            if previous_getter is not None and target_id is not None:
                previous_report = await previous_getter(
                    target_id=target_id,
                    report_type="INSPECTION_MONTHLY",
                    before_period_start=period_start,
                )
                if previous_report is not None and previous_report.content_artifact_id:
                    previous_artifact = await uow.runs.get_artifact(
                        artifact_id=previous_report.content_artifact_id
                    )
                    if previous_artifact is not None:
                        previous_payload = dict(previous_artifact.payload_json or {})
                        previous_summary = dict(
                            dict(previous_payload.get("scope") or {}).get(
                                "monthly_summary"
                            ) or {}
                        ) or None
            previous_comparison = compare_monthly_inspection_summaries(
                monthly_summary, previous_summary
            )
        result = {
            "status": "PARTIAL" if partial_count or gaps else "READY",
            "root_cause": {"effective_level": "INCONCLUSIVE"},
            "diagnosis_rationale": (
                f"{period_kind} 周期覆盖 {len(runs)} 次巡检，"
                f"其中 {partial_count} 次结果不完整，{failed_count} 次执行失败。"
            ),
            "facts": tuple(facts), "gaps": tuple(gaps),
            "solution": {
                "long_term_remediations": tuple(dict.fromkeys(recommendations)),
            },
            "evidence_refs": tuple(evidence_refs),
            "inspection_coverage": (
                f"报告时间窗内完成 {len(runs)} 次巡检，"
                f"不完整结果 {partial_count} 次，执行失败 {failed_count} 次。"
            ),
            "period_start": period_start, "period_end": period_end,
            "target": target_snapshot,
            "scope_type": "TARGET",
            "scope_name": _human_target_display_name(
                target_snapshot.get("display_name"),
                target_id=getattr(ordered_runs[0], "target_id", None),
            ) if ordered_runs else "未记录",
            "target_count": 1,
            "database_types": tuple(filter(None, (
                target_snapshot.get("database_type"),
            ))),
            "monthly_report_extensions": ({
                "schema_version": "MONTHLY_REPORT_EXTENSION.v1",
                "extension_key": "backup_recovery",
                "status": "RESERVED",
                "items": [],
            },),
        }
        if monthly_summary is not None:
            result["monthly_summary"] = monthly_summary
            result["previous_month_summary"] = previous_summary
            result["previous_month_comparison"] = previous_comparison
        return result

    async def get_report_presentation(
        self,
        *,
        report_id: UUID,
        domain_id: int,
    ) -> dict[str, Any]:
        """读取冻结内容和模板快照，供预览和 PDF 共用。"""
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            report = await uow.inspections.get_report_scoped(
                report_id=report_id, domain_id=domain_id
            )
            if report is None or report.content_artifact_id is None:
                raise resource_not_found("Report")
            artifact = await uow.runs.get_artifact(
                artifact_id=report.content_artifact_id
            )
            if not report_content_reference_is_complete(report, artifact):
                raise state_conflict("Report 内容引用不完整")
            # Oracle 原生 JSON 会把 100.0 等价归一为 100；读回后重算哈希
            # 会误报内容损坏。冻结完整性由 Report 与 Artifact 的双向哈希引用保证。
            payload = dict(artifact.payload_json or {})
            snapshot = dict(
                dict(payload.get("provenance") or {}).get("template") or {}
            )
            definition = snapshot.get("definition")
            if isinstance(definition, dict):
                checked = validate_template_definition(definition)
                template = ReportTemplate(
                    template_ref=str(snapshot.get("template_ref") or report.template_id),
                    version=str(snapshot.get("version") or report.template_version),
                    display_name=checked.display_name,
                    applicable_source_kinds=checked.applicable_source_kinds,
                    allowed_period_kinds=checked.allowed_period_kinds,
                    sections=checked.sections,
                    definition=definition,
                )
            else:
                template = resolve_historical_report_template(
                    template_ref=str(report.template_id), report_type=str(report.report_type),
                )
                if template is None:
                    raise state_conflict("历史报告缺少可重现的模板快照")
            findings = ()
            turns = getattr(uow, "turns", None)
            ops_run_id = getattr(report, "ops_run_id", None)
            if turns is not None and ops_run_id is not None:
                payloads = await turns.list_finding_blocks_for_runs(
                    ops_run_ids=(ops_run_id,)
                )
                findings = parse_finding_payload(payloads.get(ops_run_id))
            target_display_name = None
            target_repository = getattr(uow, "targets", None)
            report_target_id = getattr(report, "target_id", None)
            if target_repository is not None and report_target_id is not None:
                current_target = await target_repository.get_scoped(
                    target_id=report_target_id,
                    domain_id=domain_id,
                )
                target_display_name = getattr(
                    current_target, "display_name", None
                )
            return report_presentation(
                payload=payload,
                template=template,
                findings=findings,
                report_context={
                    "report_id": str(report.report_id),
                    "report_key": str(
                        getattr(report, "report_key", payload.get("report_key") or "")
                    ),
                    "report_version": int(
                        getattr(report, "report_version", 1)
                    ),
                    "created_at": (
                        getattr(report, "created_at").isoformat()
                        if getattr(report, "created_at", None) is not None
                        else None
                    ),
                    "domain_id": int(domain_id),
                    "target_display_name": target_display_name,
                },
            )

    async def get_report_source_agent_id(
        self, *, report_id: UUID, domain_id: int,
    ) -> UUID:
        """在展示或下载前复核报告与来源 Agent 的 Domain 归属。"""
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            report = await uow.inspections.get_report_scoped(
                report_id=report_id, domain_id=domain_id
            )
            if report is None:
                raise resource_not_found("Report")
            run = await uow.runs.get_run_scoped(
                ops_run_id=report.ops_run_id, domain_id=domain_id
            )
            if run is None:
                raise state_conflict("Report 来源 Run 不完整")
            return run.agent_id

    async def render_report_pdf(
        self,
        *,
        report_id: UUID,
        domain_id: int,
    ) -> bytes:
        return render_pdf(await self.get_report_presentation(
            report_id=report_id, domain_id=domain_id
        ))

    def _decode_cursor(
        self,
        *,
        token: str,
        scope: ConfigurationScope,
        filters: dict[str, Any],
    ) -> tuple[datetime, UUID]:
        if self._cursor_codec is None:
            raise RuntimeError("AIOps 查询 Cursor Codec 尚未配置")
        return self._cursor_codec.decode(
            token=token,
            scope=scope,
            filters=filters,
        )

    @staticmethod
    def _delegation_event(run, event) -> UnknownEvent:
        payload = dict(event.payload_json or {})
        source_type = event.event_type
        projected_type = "delegation.status"
        safe: dict[str, Any] = {}
        if source_type == "task.status":
            projected_type = "delegation.progress"
            safe = {
                "stage": payload.get("task_type"),
                "task_status": payload.get("status"),
            }
        elif source_type in {
            "diagnostic.input_required",
            "diagnostic.query_approval_required",
        }:
            projected_type = "interaction.required"
            safe = {
                "hitl_id": payload.get("hitl_id"),
                "hitl_type": payload.get("hitl_type"),
                "expires_at": payload.get("expires_at"),
            }
        elif source_type == "proposal.pending_approval":
            projected_type = "approval.required"
            safe = {
                "proposal_id": payload.get("proposal_id"),
                "risk_level": payload.get("risk_level"),
                "expires_at": payload.get("expires_at"),
            }
        elif source_type == "report.ready":
            projected_type = "report.ready"
            safe = {
                key: payload.get(key)
                for key in (
                    "report_id",
                    "report_key",
                    "report_type",
                    "report_version",
                    "summary",
                    "result",
                )
            }
        elif source_type in {
            "run.completed",
            "run.failed",
            "run.cancelled",
            "run.expired",
        }:
            projected_type = {
                "run.completed": "delegation.completed",
                "run.failed": "delegation.failed",
                "run.cancelled": "delegation.cancelled",
                "run.expired": "delegation.expired",
            }[source_type]
            safe = {"error_code": payload.get("error_code")}
        return UnknownEvent(
            ops_run_id=run.ops_run_id,
            sequence_no=int(event.sequence_no),
            occurred_at=event.created_at,
            trace_id=str(payload.get("trace_id") or run.trace_id),
            event_type=projected_type,
            status=payload.get("status") or run.status,
            **{key: value for key, value in safe.items() if value is not None},
        )

    @staticmethod
    def _delegation_safe_summary(run, artifact) -> str:
        if artifact is None:
            status = str(run.status)
            code = f"，错误码 {run.error_code}" if run.error_code else ""
            return f"AIOps 子任务已结束，状态为 {status}{code}。"
        payload = dict(artifact.payload_json or {})
        if artifact.schema_version == "DIAGNOSIS_REPORT_DRAFT.v1":
            root = dict(payload.get("root_cause") or {})
            grade = str(
                root.get("effective_level")
                or "INCONCLUSIVE"
            )
            supporting = set(root.get("supporting_fact_refs") or ())
            summaries = [
                str(item.get("fact_summary"))
                for item in payload.get("facts", ())
                if item.get("fact_id") in supporting
                and item.get("fact_summary")
            ][:5]
            facts = (
                "；关键事实：" + "；".join(summaries)
                if summaries
                else ""
            )
            gaps = len(payload.get("gaps") or ())
            return (
                f"根因等级为 {grade}{facts}；"
                f"仍有 {gaps} 个数据缺口。"
            )[:8000]
        if artifact.schema_version == "REPORT_CONTENT.v1":
            return str(payload.get("summary") or "AIOps 报告已生成")[:8000]
        return (
            f"AIOps 子任务已结束，状态为 {run.status}，"
            f"最终产物类型为 {artifact.schema_version}。"
        )[:8000]

    def _next_cursor(
        self,
        *,
        scope: ConfigurationScope,
        filters: dict[str, Any],
        entities: list,
        page_entities: list,
        limit: int,
        id_attribute: str,
    ) -> str | None:
        if len(entities) <= limit or not page_entities:
            return None
        if self._cursor_codec is None:
            raise RuntimeError("AIOps 查询 Cursor Codec 尚未配置")
        last = page_entities[-1]
        return self._cursor_codec.encode(
            scope=scope,
            sort_at=last.created_at,
            resource_id=getattr(last, id_attribute),
            filters=filters,
        )

    @staticmethod
    def _run_summary(run, *, final_artifact=None) -> OpsRunSummary:
        return OpsRunSummary(
            ops_run_id=run.ops_run_id, agent_id=run.agent_id,
            target_id=run.target_id, trigger_type=run.trigger_type,
            interaction_mode=run.interaction_mode,
            workflow_kind=run.workflow_kind, status=run.status,
            source_proposal_id=getattr(run, "source_proposal_id", None),
            source_result_artifact_id=getattr(run, "source_result_artifact_id", None),
            final_artifact=final_artifact,
            error_code=getattr(run, "error_code", None),
            error_message=getattr(run, "error_message", None),
            row_version=int(run.row_version),
            created_at=run.created_at, completed_at=run.completed_at,
        )

    @staticmethod
    def _artifact_root_cause_grade(artifact) -> str:
        payload = dict(artifact.payload_json or {})
        root = dict(payload.get("root_cause") or {})
        return str(
            root.get("effective_level")
            or payload.get("root_cause_grade")
            or "INCONCLUSIVE"
        )

    @staticmethod
    def _workflow_kind(command: CreateOpsRunCommand) -> str:
        trigger_type = str(command.trigger_type)
        if command.client_metadata.get("multi_resource_child") is True:
            return "RESOURCE_DIAGNOSIS_CHILD"
        if command.blueprint_id == "change.advisory-verify":
            return "VERIFICATION"
        if trigger_type == "SCHEDULE":
            return "INSPECTION"
        if trigger_type == "CHAT":
            return "CHAT_TURN"
        if command.blueprint_id.startswith("change."):
            return "CHANGE"
        return "ALERT_DIAGNOSIS"

    @staticmethod
    def _situation_summary(item) -> SituationSummary:
        return SituationSummary(
            situation_id=item.situation_id, target_id=item.target_id,
            situation_type=item.situation_type, title=item.title,
            summary=item.summary, status=item.status, severity=item.severity,
            event_count=int(item.event_count), row_version=int(item.row_version),
            first_observed_at=item.first_observed_at,
            last_observed_at=item.last_observed_at, resolved_at=item.resolved_at,
        )

    @staticmethod
    def _fire_summary(fire) -> InspectionFireSummary:
        snapshot = dict(fire.plan_snapshot_json or {})
        return InspectionFireSummary(
            fire_id=fire.inspection_fire_id,
            plan_id=fire.inspection_plan_id,
            plan_display_name=str(
                snapshot.get("display_name") or "未命名巡检计划"
            ),
            scheduled_at=fire.scheduled_for,
            status=fire.status,
            target_count=int(fire.target_count),
            completed_count=int(fire.completed_count),
            failed_count=int(fire.failed_count),
            error_code=fire.error_code,
            error_message=fire.error_message,
            completed_at=fire.completed_at,
        )

    @staticmethod
    def _report_view(report, artifact) -> ReportView:
        """将已校验的报告及其内容产物映射为公开视图。"""
        return ReportView(
            report_id=report.report_id,
            report_key=report.report_key,
            report_type=report.report_type,
            report_version=int(report.report_version),
            title=report.title,
            status=report.status,
            target_id=report.target_id,
            template_ref=getattr(
                report, "template_id", "system:diagnosis.standard"
            ),
            template_version=getattr(report, "template_version", "1"),
            source_conversation_id=getattr(
                report, "source_conversation_id", None
            ),
            period_start=report.period_start,
            period_end=report.period_end,
            summary=report.summary,
            content_artifact=ArtifactRef(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                schema_version=artifact.schema_version,
                content_hash=artifact.content_hash,
            ),
            corrected_from_report_id=report.supersedes_report_id,
            published_at=report.created_at,
        )

    @staticmethod
    def _report_summary(report) -> ReportSummary:
        if (
            report.period_start is None
            or report.period_end is None
            or report.summary is None
        ):
            raise state_conflict("当前 Report 投影字段不完整")
        return ReportSummary(
            report_id=report.report_id,
            report_key=report.report_key,
            report_type=report.report_type,
            report_version=int(report.report_version),
            title=report.title,
            status=report.status,
            target_id=report.target_id,
            template_ref=getattr(
                report, "template_id", "system:diagnosis.standard"
            ),
            template_version=getattr(report, "template_version", "1"),
            source_conversation_id=getattr(
                report, "source_conversation_id", None
            ),
            period_start=report.period_start,
            period_end=report.period_end,
            summary=report.summary,
            published_at=report.created_at,
        )

    async def get_pending_input(
        self,
        *,
        ops_run_id: UUID,
        domain_id: int,
        actor_id: str,
    ) -> PendingInputView:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_run_scoped(
                ops_run_id=ops_run_id,
                domain_id=domain_id,
            )
            if run is None or run.actor_id != actor_id:
                raise resource_not_found("待补充数据")
            hitl = await uow.changes.get_pending_hitl_for_run(
                ops_run_id=ops_run_id,
                assignee_user_id=actor_id,
            )
            if hitl is None:
                raise resource_not_found("待补充数据")
            return await self._pending_input_view(uow, hitl)

    async def get_hitl_input(
        self,
        *,
        hitl_id: UUID,
        domain_id: int,
        actor_id: str,
    ) -> PendingInputView:
        async with self._uow_factory() as uow:
            hitl = await uow.changes.get_hitl_scoped(
                hitl_id=hitl_id,
                domain_id=domain_id,
            )
            if (
                hitl is None
                or hitl.assignee_user_id != actor_id
                or hitl.status != "PENDING"
            ):
                raise resource_not_found("待补充数据")
            return await self._pending_input_view(uow, hitl)

    async def respond_hitl(
        self,
        *,
        hitl_id: UUID,
        domain_id: int,
        actor_id: str,
        response: HitlResponse,
        idempotency_key: str,
        trace_id: str,
    ) -> HitlResult:
        """校验用户结果并恢复原 Run，不创建新的诊断 Run。"""
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            preliminary_hitl = await uow.changes.get_hitl_scoped(
                hitl_id=hitl_id,
                domain_id=domain_id,
            )
            if (
                preliminary_hitl is None
                or preliminary_hitl.assignee_user_id != actor_id
            ):
                raise resource_not_found("待补充数据")
            run = await uow.runs.get_run(
                ops_run_id=preliminary_hitl.ops_run_id, lock=True
            )
            task = await uow.runs.get_task(
                ops_task_id=preliminary_hitl.ops_task_id, lock=True
            )
            hitl = await uow.changes.get_hitl(
                hitl_id=hitl_id, lock=True
            )
            if (
                task is None
                or run is None
                or hitl is None
                or run.actor_id != actor_id
            ):
                raise resource_not_found("待补充数据")
            if hitl.request_type == "DIAGNOSTIC_QUERY_APPROVAL":
                raise validation_failed("诊断查询审批必须提交批准或拒绝决定")
            request_artifact = await self._hitl_request_artifact(
                uow, hitl
            )
            request = ManualSqlRequest.model_validate(
                request_artifact.payload_json
            )
            normalized = self._normalize_hitl_response(
                hitl_id=hitl_id,
                request=request,
                response=response,
            )
            submitted_payload = {
                "hitl_id": str(hitl_id),
                "responses": [
                    item.model_dump(mode="json") for item in normalized
                ],
                "note": response.note,
            }
            response_hash = sha256_json(submitted_payload)
            artifact_key = self._artifact_key(task)
            if hitl.status == "ANSWERED":
                if hitl.response_hash != response_hash:
                    raise _runtime_error(
                        "OPS_IDEMPOTENCY_CONFLICT",
                        "该人工补证请求已经用不同内容答复",
                    )
                existing = await uow.runs.get_artifact_by_key(
                    ops_run_id=run.ops_run_id,
                    artifact_key=artifact_key,
                )
                if existing is None:
                    raise state_conflict("人工补证结果 Artifact 不存在")
                return HitlResult(
                    hitl_id=hitl.hitl_id,
                    status="ANSWERED",
                    row_version=int(hitl.row_version),
                    accepted_artifact=self._artifact_ref(existing),
                )
            if hitl.status != "PENDING":
                raise state_conflict("人工补证请求当前不能答复")
            if int(hitl.row_version) != response.expected_row_version:
                raise _runtime_error(
                    "OPS_ROW_VERSION_CHANGED",
                    "人工补证请求版本已变化",
                    status_code=412,
                )
            if hitl.expires_at <= now:
                raise _runtime_error(
                    "OPS_HITL_EXPIRED",
                    "人工补证请求已经过期",
                    status_code=410,
                )
            if task.status != DomainOpsTaskStatus.WAITING_INPUT.value:
                raise state_conflict("人工补证 Task 未处于等待状态")
            self._ensure_hitl_budget(run, now=now)
            submission_body = {
                "hitl_id": str(hitl_id),
                "submitted_by": actor_id,
                "submitted_at": now,
                "target_display_name": request.target_display_name,
                "used_readonly_account": True,
                "note": response.note,
                "results": [
                    item.model_dump(mode="json") for item in normalized
                ],
            }
            submission = UserDiagnosticSubmission(
                **submission_body,
                submission_sha256=sha256_json(submission_body),
            )
            outcome = HitlOutcome(
                hitl_id=str(hitl_id),
                status="ANSWERED",
                submission=submission.model_dump(mode="json"),
            )
            outcome_payload = outcome.model_dump(mode="json")
            artifact = await uow.runs.add_artifact(
                OpsArtifactEntity(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    artifact_key=artifact_key,
                    artifact_type="HITL_OUTCOME",
                    schema_version="HITL_OUTCOME.v1",
                    payload_json=outcome_payload,
                    content_hash=sha256_json(outcome_payload),
                    byte_size=len(canonical_bytes(outcome_payload)),
                    provenance_json={
                        "producer": "user",
                        "producer_version": "manual-result.v1",
                        "actor_id": actor_id,
                        "idempotency_key": idempotency_key,
                    },
                    trust_level="USER_PROVIDED",
                    security_level=1,
                )
            )
            changed = await uow.changes.answer_hitl(
                hitl_id=hitl.hitl_id,
                expected_version=int(hitl.row_version),
                allowed_statuses=("PENDING",),
                new_status="ANSWERED",
                responded_by=actor_id,
                responded_at=now,
                response_json=submitted_payload,
                response_uri=None,
                response_hash=response_hash,
            )
            if not changed:
                raise _runtime_error(
                    "OPS_ROW_VERSION_CHANGED",
                    "人工补证请求版本已变化",
                    status_code=412,
                )
            ensure_task_transition(
                DomainOpsTaskStatus(task.status),
                DomainOpsTaskStatus.SUCCEEDED,
            )
            task.status = DomainOpsTaskStatus.SUCCEEDED.value
            task.output_artifact_id = artifact.artifact_id
            task.completed_at = now
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            released = self._release_successors(tasks, now=now)
            await self._sync_run_budget(uow=uow, run=run, tasks=tasks, now=now)
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="diagnostic.input_received",
                event_key=f"hitl:{hitl.hitl_id}:answered",
                visibility="USER",
                payload_json={
                    "hitl_id": str(hitl.hitl_id),
                    "status": "ANSWERED",
                    "accepted_artifact_id": str(artifact.artifact_id),
                    "trace_id": trace_id,
                },
            )
            for successor in released:
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=successor.ops_task_id,
                    event_type="task.status",
                    event_key=(
                        f"task:{successor.ops_task_id}:ready:"
                        f"{int(successor.attempt_count)}"
                    ),
                    visibility="USER",
                    payload_json={
                        "status": "READY",
                        "task_id": str(successor.ops_task_id),
                        "task_type": successor.task_type,
                        "task_key": successor.task_key,
                        "trace_id": trace_id,
                    },
                )
            await self._project_turn_interaction_status(uow=uow, run=run,
                status="WAITING_USER" if run.budget_paused_at is not None else "COLLECTING",
                event_type="diagnostic.input_received", payload={"hitl_id": str(hitl.hitl_id), "status": "ANSWERED"})
            await uow.commit()
            return HitlResult(
                hitl_id=hitl.hitl_id,
                status="ANSWERED",
                row_version=int(hitl.row_version) + 1,
                accepted_artifact=self._artifact_ref(artifact),
            )

    async def decide_diagnostic_query(
        self,
        *,
        hitl_id: UUID,
        domain_id: int,
        actor_id: str,
        decision: DiagnosticQueryApprovalDecision,
        idempotency_key: str,
        trace_id: str,
    ) -> HitlResult:
        """批准时恢复同一冻结查询，拒绝时记录 Gap 并继续诊断。"""
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            preliminary = await uow.changes.get_hitl_scoped(
                hitl_id=hitl_id,
                domain_id=domain_id,
            )
            if preliminary is None or preliminary.assignee_user_id != actor_id:
                raise resource_not_found("诊断查询审批")
            run = await uow.runs.get_run(
                ops_run_id=preliminary.ops_run_id, lock=True
            )
            task = await uow.runs.get_task(
                ops_task_id=preliminary.ops_task_id, lock=True
            )
            hitl = await uow.changes.get_hitl(hitl_id=hitl_id, lock=True)
            if (
                run is None
                or task is None
                or hitl is None
                or run.actor_id != actor_id
                or hitl.request_type != "DIAGNOSTIC_QUERY_APPROVAL"
            ):
                raise resource_not_found("诊断查询审批")
            target_status = (
                "APPROVED" if decision.decision == "APPROVE" else "REJECTED"
            )
            response_payload = {
                "decision": decision.decision,
                "note": decision.note,
                "idempotency_key": idempotency_key,
            }
            response_hash = sha256_json(response_payload)
            if hitl.status == target_status:
                if hitl.response_hash != response_hash:
                    raise _runtime_error(
                        "OPS_IDEMPOTENCY_CONFLICT",
                        "该诊断查询审批已经提交不同决定",
                    )
                return HitlResult(
                    hitl_id=hitl.hitl_id,
                    status=target_status,
                    row_version=int(hitl.row_version),
                )
            if hitl.status != "PENDING":
                raise state_conflict("诊断查询审批当前不能处理")
            if int(hitl.row_version) != decision.expected_row_version:
                raise _runtime_error(
                    "OPS_ROW_VERSION_CHANGED",
                    "诊断查询审批版本已变化",
                    status_code=412,
                )
            if hitl.expires_at <= now:
                raise _runtime_error(
                    "OPS_HITL_EXPIRED",
                    "诊断查询审批已经过期",
                    status_code=410,
                )
            if (
                task.status != DomainOpsTaskStatus.WAITING_APPROVAL.value
            ):
                raise state_conflict("诊断查询未处于等待审批状态")
            self._ensure_hitl_budget(run, now=now)
            request_artifact = await self._hitl_request_artifact(uow, hitl)
            request = DiagnosticQueryApprovalRequest.model_validate(
                request_artifact.payload_json
            )
            if (
                request.run_id != str(run.ops_run_id)
                or request.task_id != str(task.ops_task_id)
                or request.target_id != str(run.target_id)
                or request.expires_at <= now
            ):
                raise state_conflict("诊断查询审批与 Run 上下文不匹配")
            changed = await uow.changes.answer_hitl(
                hitl_id=hitl.hitl_id,
                expected_version=int(hitl.row_version),
                allowed_statuses=("PENDING",),
                new_status=target_status,
                responded_by=actor_id,
                responded_at=now,
                response_json=response_payload,
                response_uri=None,
                response_hash=response_hash,
            )
            if not changed:
                raise _runtime_error(
                    "OPS_ROW_VERSION_CHANGED",
                    "诊断查询审批版本已变化",
                    status_code=412,
                )
            accepted_artifact = None
            if decision.decision == "APPROVE":
                plan = dict(run.plan_snapshot_json or {})
                execution = dict(plan.get("investigation_execution") or {})
                invocations = dict(execution.get("dynamic_invocations") or {})
                invocation = dict(invocations.get(task.task_key) or {})
                validated = dict(invocation.get("validated_query") or {})
                if (
                    validated.get("query_sha256") != request.query_sha256
                    or validated.get("policy_sha256") != request.policy_sha256
                ):
                    raise state_conflict("诊断查询审批与冻结查询 Hash 不匹配")
                invocation["approval"] = {
                    "status": "APPROVED",
                    "hitl_id": str(hitl.hitl_id),
                    "approved_by": actor_id,
                    "approved_at": now.isoformat(),
                    "expires_at": request.expires_at.isoformat(),
                    "query_sha256": request.query_sha256,
                    "policy_sha256": request.policy_sha256,
                }
                invocations[task.task_key] = invocation
                execution["dynamic_invocations"] = invocations
                plan["investigation_execution"] = execution
                run.plan_snapshot_json = plan
                ensure_task_transition(
                    DomainOpsTaskStatus(task.status),
                    DomainOpsTaskStatus.READY,
                )
                task.status = DomainOpsTaskStatus.READY.value
                try:
                    resume_task_budget(task, now=now)
                except ExecutionBudgetError as exc:
                    raise state_conflict(str(exc)) from exc
                event_type = "diagnostic.query_approved"
            else:
                outcome = HitlOutcome(
                    hitl_id=str(hitl.hitl_id),
                    status="SKIPPED",
                    gap_code="DIAGNOSTIC_QUERY_REJECTED",
                )
                payload = outcome.model_dump(mode="json")
                artifact = await uow.runs.add_artifact(
                    OpsArtifactEntity(
                        ops_run_id=run.ops_run_id,
                        ops_task_id=task.ops_task_id,
                        artifact_key=self._artifact_key(task),
                        artifact_type="HITL_OUTCOME",
                        schema_version="HITL_OUTCOME.v1",
                        payload_json=payload,
                        content_hash=sha256_json(payload),
                        byte_size=len(canonical_bytes(payload)),
                        provenance_json={
                            "producer": "user",
                            "producer_version": "query-approval.v1",
                            "actor_id": actor_id,
                        },
                        trust_level="USER_PROVIDED",
                        security_level=1,
                    )
                )
                ensure_task_transition(
                    DomainOpsTaskStatus(task.status),
                    DomainOpsTaskStatus.SUCCEEDED,
                )
                task.status = DomainOpsTaskStatus.SUCCEEDED.value
                task.output_artifact_id = artifact.artifact_id
                task.completed_at = now
                tasks = await uow.runs.list_tasks(
                    ops_run_id=run.ops_run_id, lock=True
                )
                self._release_successors(tasks, now=now)
                accepted_artifact = self._artifact_ref(artifact)
                event_type = "diagnostic.query_rejected"
            tasks = await uow.runs.list_tasks(ops_run_id=run.ops_run_id, lock=True)
            await self._sync_run_budget(uow=uow, run=run, tasks=tasks, now=now)
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type=event_type,
                event_key=f"hitl:{hitl.hitl_id}:{target_status.lower()}",
                visibility="USER",
                payload_json={
                    "hitl_id": str(hitl.hitl_id),
                    "status": target_status,
                    "trace_id": trace_id,
                },
            )
            await self._project_turn_interaction_status(
                uow=uow,
                run=run,
                status="WAITING_USER" if run.budget_paused_at is not None else "COLLECTING",
                event_type=event_type,
                payload={
                    "hitl_id": str(hitl.hitl_id),
                    "status": target_status,
                },
            )
            await uow.commit()
            return HitlResult(
                hitl_id=hitl.hitl_id,
                status=target_status,
                row_version=int(hitl.row_version) + 1,
                accepted_artifact=accepted_artifact,
            )

    async def skip_hitl(
        self,
        *,
        hitl_id: UUID,
        domain_id: int,
        actor_id: str,
        expected_row_version: int,
        idempotency_key: str,
        trace_id: str,
    ) -> HitlResult:
        """跳过补证时写入受控 Gap，并继续完成同一个 Run。"""
        async with self._uow_factory() as uow:
            hitl = await uow.changes.get_hitl_scoped(
                hitl_id=hitl_id,
                domain_id=domain_id,
            )
            approval_request = (
                hitl is not None
                and hitl.assignee_user_id == actor_id
                and hitl.request_type == "DIAGNOSTIC_QUERY_APPROVAL"
            )
        if approval_request:
            return await self.decide_diagnostic_query(
                hitl_id=hitl_id,
                domain_id=domain_id,
                actor_id=actor_id,
                decision=DiagnosticQueryApprovalDecision(
                    expected_row_version=expected_row_version,
                    decision="REJECT",
                    note="用户选择不批准该诊断查询",
                ),
                idempotency_key=idempotency_key,
                trace_id=trace_id,
            )
        response = HitlResponse(
            expected_row_version=expected_row_version,
            responses=(
                {
                    "query_id": "__all__",
                    "status": "SKIPPED",
                    "error": "用户选择跳过人工补证",
                },
            ),
            note="用户选择跳过人工补证",
        )
        return await self._finish_skipped_hitl(
            hitl_id=hitl_id,
            domain_id=domain_id,
            actor_id=actor_id,
            response=response,
            idempotency_key=idempotency_key,
            trace_id=trace_id,
        )

    async def list_events(
        self,
        *,
        ops_run_id: UUID,
        domain_id: int,
        after_sequence: int,
        user_only: bool,
        limit: int = 200,
    ) -> OpsRunEventPage:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_run_scoped(
                ops_run_id=ops_run_id,
                domain_id=domain_id,
            )
            if run is None:
                raise resource_not_found("Ops Run")
            latest = await uow.runs.latest_event_sequence(
                ops_run_id=ops_run_id
            )
            if after_sequence > latest:
                raise _runtime_error(
                    "OPS_EVENT_CURSOR_INVALID",
                    "事件游标大于当前最新序号",
                )
            events = await uow.runs.list_events_after(
                ops_run_id=ops_run_id,
                after_sequence=after_sequence,
                visibility="USER" if user_only else None,
                limit=limit,
            )
            next_sequence = (
                int(events[-1].sequence_no)
                if events
                else after_sequence
            )
            return OpsRunEventPage(
                events=tuple(
                    OpsRunEventView(
                        ops_run_id=item.ops_run_id,
                        sequence_no=int(item.sequence_no),
                        event_type=item.event_type,
                        visibility=item.visibility,
                        payload=dict(item.payload_json),
                        occurred_at=item.created_at,
                    )
                    for item in events
                ),
                next_sequence=next_sequence,
                terminal=DomainOpsRunStatus(run.status)
                in TERMINAL_RUN_STATUSES,
            )

    async def _finish_skipped_hitl(
        self,
        *,
        hitl_id: UUID,
        domain_id: int,
        actor_id: str,
        response: HitlResponse,
        idempotency_key: str,
        trace_id: str,
    ) -> HitlResult:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            preliminary_hitl = await uow.changes.get_hitl_scoped(
                hitl_id=hitl_id,
                domain_id=domain_id,
            )
            if (
                preliminary_hitl is None
                or preliminary_hitl.assignee_user_id != actor_id
            ):
                raise resource_not_found("待补充数据")
            run = await uow.runs.get_run(
                ops_run_id=preliminary_hitl.ops_run_id, lock=True
            )
            task = await uow.runs.get_task(
                ops_task_id=preliminary_hitl.ops_task_id, lock=True
            )
            hitl = await uow.changes.get_hitl(
                hitl_id=hitl_id, lock=True
            )
            if (
                task is None
                or run is None
                or hitl is None
                or run.actor_id != actor_id
            ):
                raise resource_not_found("待补充数据")
            response_payload = response.model_dump(mode="json")
            response_hash = sha256_json(response_payload)
            artifact_key = self._artifact_key(task)
            if hitl.status == "SKIPPED":
                if hitl.response_hash != response_hash:
                    raise _runtime_error(
                        "OPS_IDEMPOTENCY_CONFLICT",
                        "该人工补证请求已经用不同内容跳过",
                    )
                existing = await uow.runs.get_artifact_by_key(
                    ops_run_id=run.ops_run_id,
                    artifact_key=artifact_key,
                )
                if existing is None:
                    raise state_conflict("人工补证结果 Artifact 不存在")
                return HitlResult(
                    hitl_id=hitl.hitl_id,
                    status="SKIPPED",
                    row_version=int(hitl.row_version),
                    accepted_artifact=self._artifact_ref(existing),
                )
            if (
                hitl.status != "PENDING"
                or task.status
                != DomainOpsTaskStatus.WAITING_INPUT.value
            ):
                raise state_conflict("人工补证请求当前不能跳过")
            if int(hitl.row_version) != response.expected_row_version:
                raise _runtime_error(
                    "OPS_ROW_VERSION_CHANGED",
                    "人工补证请求版本已变化",
                    status_code=412,
                )
            self._ensure_hitl_budget(run, now=now)
            if hitl.expires_at <= now:
                raise _runtime_error("OPS_HITL_EXPIRED", "人工补证请求已经过期", status_code=410)
            outcome = HitlOutcome(
                hitl_id=str(hitl_id),
                status="SKIPPED",
                gap_code="USER_SKIPPED_MANUAL_DIAGNOSTIC",
            )
            payload = outcome.model_dump(mode="json")
            artifact = await uow.runs.add_artifact(
                OpsArtifactEntity(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=task.ops_task_id,
                    artifact_key=artifact_key,
                    artifact_type="HITL_OUTCOME",
                    schema_version="HITL_OUTCOME.v1",
                    payload_json=payload,
                    content_hash=sha256_json(payload),
                    byte_size=len(canonical_bytes(payload)),
                    provenance_json={
                        "producer": "user",
                        "producer_version": "manual-result.v1",
                        "actor_id": actor_id,
                        "idempotency_key": idempotency_key,
                    },
                    trust_level="USER_PROVIDED",
                    security_level=1,
                )
            )
            changed = await uow.changes.answer_hitl(
                hitl_id=hitl.hitl_id,
                expected_version=int(hitl.row_version),
                allowed_statuses=("PENDING",),
                new_status="SKIPPED",
                responded_by=actor_id,
                responded_at=now,
                response_json=response_payload,
                response_uri=None,
                response_hash=response_hash,
            )
            if not changed:
                raise _runtime_error(
                    "OPS_ROW_VERSION_CHANGED",
                    "人工补证请求版本已变化",
                    status_code=412,
                )
            ensure_task_transition(
                DomainOpsTaskStatus(task.status),
                DomainOpsTaskStatus.SUCCEEDED,
            )
            task.status = DomainOpsTaskStatus.SUCCEEDED.value
            task.output_artifact_id = artifact.artifact_id
            task.completed_at = now
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            self._release_successors(tasks, now=now)
            await self._sync_run_budget(uow=uow, run=run, tasks=tasks, now=now)
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=task.ops_task_id,
                event_type="diagnostic.input_skipped",
                event_key=f"hitl:{hitl.hitl_id}:skipped",
                visibility="USER",
                payload_json={
                    "hitl_id": str(hitl.hitl_id),
                    "status": "SKIPPED",
                    "trace_id": trace_id,
                },
            )
            await self._project_turn_interaction_status(uow=uow, run=run,
                status="WAITING_USER" if run.budget_paused_at is not None else "COLLECTING",
                event_type="diagnostic.input_skipped", payload={"hitl_id": str(hitl.hitl_id), "status": "SKIPPED"})
            await uow.commit()
            return HitlResult(
                hitl_id=hitl.hitl_id,
                status="SKIPPED",
                row_version=int(hitl.row_version) + 1,
                accepted_artifact=self._artifact_ref(artifact),
            )

    async def _pending_input_view(self, uow, hitl) -> PendingInputView:
        artifact = await self._hitl_request_artifact(uow, hitl)
        return PendingInputView(
            hitl_id=hitl.hitl_id,
            ops_run_id=hitl.ops_run_id,
            hitl_type=hitl.request_type,
            status=hitl.status,
            request_artifact=self._artifact_ref(artifact),
            request=dict(artifact.payload_json or {}),
            expires_at=hitl.expires_at,
            row_version=int(hitl.row_version),
        )

    @staticmethod
    async def _hitl_request_artifact(uow, hitl):
        references = list(hitl.input_artifacts_json or [])
        if len(references) != 1:
            raise state_conflict("人工补证请求 Artifact 引用无效")
        artifact = await uow.runs.get_artifact(
            artifact_id=UUID(str(references[0]))
        )
        expected_schema = {
            "DATA_REQUIRED": "DATA_REQUEST.v1",
            "MANUAL_DIAGNOSTIC_SQL": "MANUAL_SQL_REQUEST.v1",
            "DIAGNOSTIC_QUERY_APPROVAL": (
                "DIAGNOSTIC_QUERY_APPROVAL_REQUEST.v1"
            ),
        }.get(hitl.request_type)
        if (
            artifact is None
            or artifact.ops_run_id != hitl.ops_run_id
            or expected_schema is None
            or artifact.schema_version != expected_schema
        ):
            raise state_conflict("人工补证请求 Artifact 不存在或类型无效")
        return artifact

    async def _project_turn_interaction_status(
        self,
        *,
        uow,
        run,
        status: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """把 Run 的交互暂停或恢复同步到 Agent Turn。"""
        if getattr(run, "workflow_kind", None) not in _AGENT_TURN_WORKFLOWS:
            return
        link = await uow.turns.get_run_link_by_ops_run_id(
            ops_run_id=run.ops_run_id
        )
        turn = (
            await uow.turns.get_turn(
                domain_id=int(run.domain_id),
                turn_id=link.turn_id,
                lock=True,
            )
            if link is not None
            else None
        )
        if turn is None:
            raise state_conflict("Agent Turn Run 缺少有效 Turn 关联")
        turn.status = status
        await self._append_turn_event(
            uow,
            turn,
            event_type=event_type,
            payload=payload,
        )

    @staticmethod
    def _normalize_hitl_response(
        *,
        hitl_id: UUID,
        request: ManualSqlRequest,
        response: HitlResponse,
    ):
        expected = {item.query_id: item for item in request.queries}
        received = {item.query_id: item for item in response.responses}
        if len(received) != len(response.responses) or set(received) != set(
            expected
        ):
            raise validation_failed("人工结果必须逐条对应请求中的 Query ID")
        normalized = []
        for query_id, query in expected.items():
            item = received[query_id]
            try:
                normalized.append(
                    normalize_raw_response(
                        hitl_id=str(hitl_id),
                        query_id=query_id,
                        status=str(item.status),
                        raw_output=item.raw_output,
                        error=item.error,
                        expected_columns=query.expected_columns,
                        max_rows=query.max_rows,
                    )
                )
            except (ValueError, json.JSONDecodeError) as exc:
                raise validation_failed(str(exc)) from exc
        identity = next(
            (
                item
                for item in normalized
                if item.query_id == "db.instance.identity"
            ),
            None,
        )
        if (
            identity is not None
            and identity.status == "SUCCEEDED"
            and identity.parse_status == "STRUCTURED"
        ):
            if len(identity.rows) != 1:
                raise validation_failed("实例身份查询必须只返回一行")
            version_index = identity.columns.index("version")
            configured = request.expected_instance_identity.get(
                "configured_version", ""
            )
            configured_major = next(
                iter(re.findall(r"\d+", configured)), ""
            )
            returned_major = next(
                iter(re.findall(r"\d+", str(identity.rows[0][version_index]))),
                "",
            )
            if (
                configured_major
                and returned_major
                and configured_major != returned_major
            ):
                raise validation_failed("人工结果来自不同数据库版本")
            row = dict(
                zip(identity.columns, identity.rows[0], strict=True)
            )
            for key in ("product", "instance_name"):
                expected_value = request.expected_instance_identity.get(key)
                if expected_value and str(row.get(key)) != expected_value:
                    raise validation_failed("人工结果来自不同数据库实例")
        return tuple(normalized)

    @staticmethod
    def _artifact_ref(artifact) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            schema_version=artifact.schema_version,
            content_hash=artifact.content_hash,
        )

    async def _lock_run_task(self, uow, task_id: UUID):
        preliminary = await uow.runs.get_task(ops_task_id=task_id)
        if preliminary is None:
            raise resource_not_found("Ops Task")
        run = await uow.runs.get_run(
            ops_run_id=preliminary.ops_run_id, lock=True
        )
        task = await uow.runs.get_task(
            ops_task_id=task_id, lock=True
        )
        if run is None or task is None:
            raise resource_not_found("Ops Task")
        return run, task

    async def _replay_concurrent_create(
        self, command: CreateOpsRunCommand
    ) -> OpsRunReceipt:
        async with self._uow_factory() as uow:
            existing = await uow.runs.get_by_idempotency(
                target_id=command.target_id,
                trigger_type=str(command.trigger_type),
                actor_id=command.actor_id,
                idempotency_key=command.idempotency_key,
            )
            if existing is None:
                raise _runtime_error(
                    "OPS_RUN_CREATE_CONFLICT",
                    "Run 创建发生并发约束冲突，请重试",
                    retryable=True,
                )
            if (
                existing.agent_id != command.agent_id
                or existing.original_request != command.input
            ):
                raise _runtime_error(
                    "OPS_IDEMPOTENCY_CONFLICT",
                    "相同 Idempotency-Key 对应的 Run 请求不同",
                )
            cursor = await uow.runs.latest_event_sequence(
                ops_run_id=existing.ops_run_id
            )
            return self._run_receipt(existing, cursor)

    async def _input_artifacts(self, uow, *, run_id: UUID, task):
        required = set(task.input_artifacts_json or [])
        if not required:
            return ()
        tasks = await uow.runs.list_tasks(ops_run_id=run_id)
        producer_ids = {
            item.ops_task_id
            for item in tasks
            if item.task_key in required
        }
        artifacts = await uow.runs.list_artifacts(ops_run_id=run_id)
        selected = tuple(
            LeasedArtifact(
                artifact_id=item.artifact_id,
                artifact_key=item.artifact_key,
                artifact_type=item.artifact_type,
                schema_version=item.schema_version,
                payload=item.payload_json,
                payload_uri=item.payload_uri,
                content_hash=item.content_hash,
                provenance=dict(item.provenance_json),
                trust_level=item.trust_level,
                security_level=int(item.security_level),
            )
            for item in artifacts
            if (
                item.ops_task_id in producer_ids
                or item.artifact_key in required
                or str(item.artifact_id) in required
            )
        )

        from aiops_agent.application.investigation.future_observation_evidence import revalidate_future_observation_inputs
        selected = await revalidate_future_observation_inputs(uow=uow, run_id=run_id, artifacts=artifacts, selected=selected)
        from aiops_agent.application.investigation.prior_sql_replay import revalidate_prior_sql_inputs
        selected = await revalidate_prior_sql_inputs(uow=uow, run_id=run_id, artifacts=artifacts, selected=selected)
        from aiops_agent.application.investigation.sql_plan_evidence import revalidate_plan_capture_inputs
        return await revalidate_plan_capture_inputs(uow=uow, run_id=run_id, artifacts=artifacts,
            selected=selected, reader=getattr(self, "_plan_capture_reader", None))

    def _task_lease(
        self, run, task, artifacts, *, lease_token: UUID
    ) -> TaskLease:
        assert task.lease_until is not None
        return TaskLease(
            task_id=task.ops_task_id,
            run_id=run.ops_run_id,
            task_key=task.task_key,
            task_type=task.task_type,
            handler_id=task.handler_id,
            handler_version=task.handler_version,
            input_schema_version=task.input_schema_version,
            output_schema_version=task.output_schema_version,
            lease_token=lease_token,
            lease_until=task.lease_until,
            attempt=int(task.attempt_count),
            max_attempts=int(task.max_attempts),
            timeout_seconds=int(task.timeout_seconds),
            row_version=int(task.row_version),
            target_id=run.target_id,
            agent_id=run.agent_id,
            actor_id=run.actor_id,
            trace_id=run.trace_id,
            original_request=run.original_request or "",
            deadline_at=run.deadline_at,
            task_deadline_at=task_deadline(task),
            plan_snapshot=dict(run.plan_snapshot_json or {}),
            policy_snapshot=dict(run.policy_snapshot_json or {}),
            input_artifacts=artifacts,
        )

    def _ensure_lease(
        self,
        *,
        run,
        task,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
        allow_expired: bool = False,
    ) -> None:
        if (
            task.status != DomainOpsTaskStatus.RUNNING.value
            or task.lease_owner != worker_id
            or task.lease_token != lease_token
            or task.lease_until is None
            or (not allow_expired and task.lease_until <= now)
        ):
            raise self._stale_lease()
        if run.cancel_requested_at is not None:
            raise _runtime_error(
                "OPS_RUN_CANCEL_REQUESTED",
                "Run 已请求取消，拒绝提交新的任务结果",
            )
        if not allow_expired and str(getattr(run, "status", "RUNNING")) in TERMINAL_RUN_STATUSES:
            raise self._stale_lease()
        if not allow_expired and run.deadline_at is not None and run.deadline_at <= now:
            raise _runtime_error(
                "OPS_RUN_DEADLINE_EXCEEDED",
                "Run 已超过截止时间",
            )
        execution_deadline = task_deadline(task)
        if not allow_expired and execution_deadline is not None and execution_deadline <= now:
            raise _runtime_error("OPS_TASK_DEADLINE_EXCEEDED", "Task 剩余执行预算已用尽")

    @staticmethod
    def _ensure_hitl_budget(run, *, now):
        try:
            ensure_run_resumable(run, now=now)
        except ExecutionBudgetError as exc:
            raise state_conflict(str(exc)) from exc

    async def _sync_run_budget(self, *, uow, run, tasks, now):
        if str(run.status) in TERMINAL_RUN_STATUSES or run.cancel_requested_at is not None:
            return
        previous = (run.status, run.budget_paused_at)
        sync_run_budget(run, tasks, now=now)
        if previous == (run.status, run.budget_paused_at):
            return
        paused = run.budget_paused_at is not None
        payload = {"status": run.status, "deadline_at": run.deadline_at.isoformat() if run.deadline_at else None,
            "remaining_milliseconds": int(run.budget_remaining_ms) if run.budget_remaining_ms is not None else None,
            "human_wait_until": run.human_wait_until.isoformat() if run.human_wait_until else None,
            "public_summary": "等待人工处理，已暂停机器执行预算" if paused else "继续执行，仅结转尚未使用的机器预算"}
        await uow.runs.append_event(ops_run_id=run.ops_run_id, event_type="run.budget_changed",
            event_key=None, visibility="USER", payload_json=payload)
        await self._project_turn_interaction_status(uow=uow, run=run,
            status="WAITING_USER" if paused else "COLLECTING", event_type="run.budget_changed", payload=payload)

    @staticmethod
    async def _close_pending_hitls(*, uow, run, now, status):
        """调用方持 Run/Task 锁；终止后不遗留可恢复的人工请求。"""
        for hitl in await uow.changes.list_pending_hitls(ops_run_id=run.ops_run_id, lock=True):
            hitl.status = status
            hitl.responded_at = now
            hitl.responded_by = None
            hitl.response_json = {"reason": f"RUN_{status}"}
            hitl.response_hash = sha256_json(hitl.response_json)

    @staticmethod
    def _stale_lease() -> AIOpsApplicationError:
        return _runtime_error(
            "OPS_STALE_LEASE", "Task 租约无效、过期或已被接管"
        )

    def _validate_artifact(self, task, artifact: ArtifactInput) -> None:
        manifest = self._handlers.resolve(
            task.handler_id, task.handler_version
        )
        if (
            artifact.producer != task.handler_id
            or artifact.producer_version != task.handler_version
            or artifact.schema_version != task.output_schema_version
            or manifest.output_schema_version
            != task.output_schema_version
        ):
            raise _runtime_error(
                "OPS_OUTPUT_SCHEMA_INVALID",
                "Artifact Producer 或 Schema 与冻结 Task 不匹配",
                status_code=422,
            )

    @staticmethod
    def _artifact_key(task) -> str:
        return (
            f"task:{task.ops_task_id}:output:default:"
            f"v{task.output_schema_version}"
        )

    @staticmethod
    def _ensure_same_artifact(
        existing, submitted: ArtifactInput, content_hash: str
    ) -> None:
        producer = dict(existing.provenance_json or {}).get("producer")
        producer_version = dict(existing.provenance_json or {}).get(
            "producer_version"
        )
        if (
            existing.content_hash != content_hash
            or existing.schema_version != submitted.schema_version
            or producer != submitted.producer
            or producer_version != submitted.producer_version
        ):
            raise _runtime_error(
                "OPS_ARTIFACT_IDEMPOTENCY_CONFLICT",
                "相同 Artifact Key 对应的内容或 Producer 不同",
            )

    @staticmethod
    def _release_successors(tasks, *, now: datetime) -> list:
        statuses = {item.task_key: item.status for item in tasks}
        released = []
        for item in tasks:
            if item.status != DomainOpsTaskStatus.PENDING.value:
                continue
            dependencies = item.depends_on_json or []
            if all(
                statuses.get(key)
                == DomainOpsTaskStatus.SUCCEEDED.value
                for key in dependencies
            ):
                ensure_task_transition(
                    DomainOpsTaskStatus(item.status),
                    DomainOpsTaskStatus.READY,
                )
                item.status = DomainOpsTaskStatus.READY.value
                item.available_at = max(item.available_at, now) if item.available_at is not None else now
                released.append(item)
        return released

    @staticmethod
    def _block_unreachable(tasks, *, failed_key: str) -> None:
        blocked = {failed_key}
        changed = True
        while changed:
            changed = False
            for item in tasks:
                if item.status not in {
                    DomainOpsTaskStatus.PENDING.value,
                    DomainOpsTaskStatus.READY.value,
                    DomainOpsTaskStatus.RETRY_WAIT.value,
                }:
                    continue
                if blocked.intersection(item.depends_on_json or []):
                    ensure_task_transition(
                        DomainOpsTaskStatus(item.status),
                        DomainOpsTaskStatus.BLOCKED,
                    )
                    item.status = DomainOpsTaskStatus.BLOCKED.value
                    blocked.add(item.task_key)
                    changed = True

    @staticmethod
    def _clear_lease(task) -> None:
        task.lease_owner = None
        task.lease_token = None
        task.lease_until = None
        task.heartbeat_at = None

    @staticmethod
    def _retry_delay(task) -> timedelta:
        attempt = int(task.attempt_count)
        base = min(2 ** max(attempt - 1, 0), 60)
        digest = hashlib.sha256(
            f"{task.ops_task_id}:{attempt}".encode("utf-8")
        ).digest()
        jitter_ms = int.from_bytes(digest[:2], "big") % 1000
        return timedelta(seconds=base, milliseconds=jitter_ms)

    @staticmethod
    async def _add_outbox(
        uow,
        *,
        aggregate_id: UUID,
        event_type: str,
        idempotency_key: str,
        payload: dict[str, Any],
        trace_id: str,
        now: datetime,
    ) -> None:
        await uow.outbox.add(
            OutboxEntity(
                aggregate_type="OPS_RUN",
                aggregate_id=aggregate_id,
                event_type=event_type,
                idempotency_key=idempotency_key,
                payload_json=payload,
                payload_hash=sha256_json(payload),
                status="PENDING",
                available_at=now,
                max_attempts=5,
                trace_id=trace_id,
            )
        )

    async def _add_multi_resource_terminal_outbox(
        self, uow, *, run, trace_id: str, now: datetime
    ) -> None:
        if run.workflow_kind != "RESOURCE_DIAGNOSIS_CHILD":
            return
        await self._add_outbox(
            uow,
            aggregate_id=run.ops_run_id,
            event_type="OPS_RUN_TERMINATED",
            idempotency_key=f"run:{run.ops_run_id}:terminated",
            payload={
                "ops_run_id": str(run.ops_run_id),
                "status": str(run.status),
            },
            trace_id=trace_id,
            now=now,
        )

    @staticmethod
    def _run_receipt(run, cursor: int) -> OpsRunReceipt:
        return OpsRunReceipt(
            ops_run_id=run.ops_run_id,
            status=run.status,
            row_version=int(run.row_version),
            event_cursor=cursor,
        )

    @staticmethod
    def _mutation_receipt(
        run, task, cursor: int, artifact_id: UUID | None
    ) -> TaskMutationReceipt:
        return TaskMutationReceipt(
            task_id=task.ops_task_id,
            run_id=run.ops_run_id,
            task_status=task.status,
            run_status=run.status,
            task_row_version=int(task.row_version),
            run_row_version=int(run.row_version),
            event_cursor=cursor,
            artifact_id=artifact_id,
        )
