"""AIOps Run/Task/Artifact/Event 的确定性事务内核。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from aiops_agent.application.errors import (
    AIOpsApplicationError,
    dependency_unavailable,
    resource_not_found,
    state_conflict,
    validation_failed,
)
from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    SignedCursorCodec,
)
from aiops_agent.application.monitoring_snapshot import MonitoringSnapshotBuilder
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
    OutboxEntity,
    ReportEntity,
    ReportSourceEntity,
)
from aiops_agent.contracts.evidence import ObservationSet
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
from aiops_agent.application.reporting import (
    ReportTemplate,
    closed_period_window,
    normalize_report_source,
    render_pdf,
    report_presentation,
    resolve_report_template_reference,
    resolve_system_template,
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
    HitlResponse,
    HitlResult,
    InspectionFirePage,
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
    ):
        self._uow_factory = uow_factory
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
                "target_id": str(target.target_id),
                "domain_id": int(target.domain_id),
                "display_name": target.display_name,
                "db_type": target.db_type,
                "version_code": target.version_code,
                "environment": target.environment,
                "db_role": target.db_role,
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
                "security_level": int(target.security_level),
                "capabilities": dict(target.capabilities_json or {}),
                "row_version": int(target.row_version),
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
        collection_ids = rules.get("aiops_collection_ids", [])
        if not isinstance(collection_ids, list):
            raise validation_failed("策略中的 AIOps Collection 范围无效")
        try:
            normalized_collection_ids = tuple(
                str(UUID(str(item))) for item in collection_ids
            )
        except (TypeError, ValueError):
            raise validation_failed(
                "策略中的 AIOps Collection ID 无效"
            ) from None
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
            "allowed_collection_ids": tuple(
                normalized_collection_ids
            ),
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
                now + timedelta(seconds=int(task.timeout_seconds)),
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
                    f"{int(task.attempt_count)}"
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
            await uow.commit()
            return self._task_lease(
                run, task, artifacts, lease_token=lease_token
            )

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
            timeout_until = (task.started_at or now) + timedelta(
                seconds=int(task.timeout_seconds)
            )
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
            content = (
                command.artifact.payload
                if command.artifact.payload is not None
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
                    payload_json=command.artifact.payload,
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
            released = self._release_successors(tasks, now=now)
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
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    DomainOpsRunStatus.COMPLETED,
                )
                run.status = DomainOpsRunStatus.COMPLETED.value
                final_artifact = artifact
                # 巡检、告警和智能诊断均只保留原始终态产物。正式报告只能
                # 经用户显式请求进入统一报告生成器，不能在 Run 完成时自动创建。
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
                run.final_artifact_id = final_artifact.artifact_id
                run.completed_at = now
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
                    event_type="run.completed",
                    event_key=f"run:{run.ops_run_id}:terminal",
                    visibility="USER",
                    payload_json={
                        "status": DomainOpsRunStatus.COMPLETED.value,
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
        elif artifact.schema_version == "DBA_SUFFICIENCY.v1":
            assessment = DbaSufficiencyAssessment.model_validate(payload)
            turn.sufficiency_status = str(assessment.status)
            turn.sufficiency_json = assessment.model_dump(mode="json")
            turn.sufficiency_artifact_id = artifact.artifact_id
            turn.assessment_artifact_id = artifact.artifact_id
            if (
                assessment.investigation is not None
                and not assessment.investigation.progress_made
            ) or (
                assessment.investigation is None
                and not assessment.evidence
            ):
                turn.no_progress_count = int(turn.no_progress_count or 0) + 1
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
            should_replan = self._should_replan_investigation(
                assessment=assessment,
                deterministic_replan=deterministic_replan,
                no_progress_count=int(turn.no_progress_count or 0),
                current_plan_revision=int(turn.current_plan_revision or 1),
            )
            if should_replan:
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
                    "public_summary": "证据已整理，正在形成回答",
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
        deterministic_replan: bool,
        no_progress_count: int,
        current_plan_revision: int,
    ) -> bool:
        """只要持续取得进展，就在 Run 截止时间内继续自动补证。"""
        requested = (
            assessment.investigation.next_action == "REPLAN"
            if assessment.investigation is not None
            else deterministic_replan
        )
        blocked_by_model = (
            assessment.investigation is not None
            and assessment.investigation.next_action
            in {"ASK_USER", "STOP_UNSAFE"}
        )
        return (
            (requested or deterministic_replan)
            and not blocked_by_model
            and no_progress_count < 2
            and current_plan_revision < 2
        )

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
                "public_summary": "首轮证据仍有关键缺口，正在调整调查计划",
            },
        )

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
        for tool_row in tool_rows:
            if tool_row.ops_task_id != task.ops_task_id:
                continue
            outcome = outcomes_by_tool.get(tool_row.tool_id)
            if outcome is None:
                continue
            tool_row.status = (
                "SUCCEEDED" if outcome.observation is not None
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
                    if outcome.observation is not None
                    else "tool.gap"
                ),
                payload={
                    "tool_invocation_id": str(tool_row.tool_invocation_id),
                    "tool_id": tool_row.tool_id,
                    "status": tool_row.status,
                    "public_summary": (
                        "数据库只读观测已经完成"
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
                    evidence_kind="ALERT_LOG",
                    confidence=1,
                    evidence_role="SUPPORTS",
                    measurement_semantics="HISTORICAL_SAMPLES",
                    observed_at=result.collected_at,
                    window_start_at=result.window_start,
                    window_end_at=result.window_end,
                    freshness_status="FRESH",
                    usage_reason="用于回答本轮问题的 Oracle Alert Log",
                    linked_by="aiops.turn-projector",
                )
            )
            await self._append_turn_event(
                uow,
                turn,
                event_type="evidence.added",
                payload={
                    "source_kind": "LOG",
                    "evidence_kind": "ALERT_LOG",
                    "public_summary": "新的 Oracle Alert Log 依据已加入本轮调查",
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
                "public_summary": "Oracle Alert Log 查询已经完成",
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
            if str(block.block_type) == "MARKDOWN"
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
            if str(block.block_type) == "MARKDOWN"
        ).strip()
        status = "READY" if source.status == "COMPLETED" else "PARTIAL"
        summary = markdown or "Agent 已完成巡检，但未生成文字摘要"
        security_level = await self._report_security_level(
            uow=uow,
            run=run,
            plan=plan,
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
            },
            facts=(
                {
                    "kind": "agent_health_inspection",
                    "markdown": markdown,
                    "sufficiency_status": str(source.sufficiency_status),
                },
            ),
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
            },
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
                template_id=inspection["template_id"],
                template_version=inspection["template_version"],
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
                template_id=inspection["template_id"],
                template_version=inspection["template_version"],
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
        """优先使用 Run 快照；旧 Run 缺失时读取权威 Target。"""
        target_snapshot = plan.get("target")
        if (
            isinstance(target_snapshot, dict)
            and "security_level" in target_snapshot
        ):
            return int(target_snapshot["security_level"])
        target = await uow.targets.get_scoped(
            target_id=run.target_id,
            domain_id=int(run.domain_id),
        )
        if target is None:
            raise resource_not_found("Target")
        return int(target.security_level)

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
        report_key = template.template_ref.removeprefix("system:")
        root = dict(source.get("root_cause") or {})
        grade = str(root.get("effective_level") or "INCONCLUSIVE")
        status = (
            "READY"
            if source.get("status") == "READY"
            and grade != "INCONCLUSIVE"
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
        summary = f"根因等级：{grade}。{rationale}"[:2000]
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
                "inspection_coverage": source.get(
                    "inspection_coverage",
                    rationale if source_kind == "INSPECTION" else None,
                ),
                "report_period_kind": period_kind,
            },
            facts=tuple(
                {
                    "fact_id": item.get("fact_id"),
                    "summary": item.get("fact_summary")
                    or item.get("summary"),
                    "trust_level": item.get("trust_level"),
                }
                for item in source.get("facts", ())
            ),
            gaps=tuple(
                {"code": str(code)} for code in source.get("gaps", ())
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
                    if isinstance(item, dict)
                ),
            ),
            recommendations=recommendations,
            provenance={
                "deterministic_report_decision": True,
                "source_artifact_hash": source_artifact.content_hash,
                "model_receipt_hashes": list(
                    source.get("model_receipt_hashes") or ()
                ),
                "template": {
                    "template_ref": template.template_ref,
                    "version": template.version,
                    "content_hash": template.content_hash,
                    "definition": template.definition,
                },
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
                target_id=run.target_id,
                report_key=report_key,
                report_version=1,
                is_current=0,
                report_type=report_type,
                title=content.title,
                status=status,
                period_start=period_start,
                period_end=period_end,
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
        for evidence in source.get("evidence_refs", ()):
            if not isinstance(evidence, dict) or not evidence.get("source_run_id"):
                continue
            try:
                evidence_run_id = UUID(str(evidence["source_run_id"]))
                if evidence_run_id == run.ops_run_id:
                    continue
                source_rows.append(ReportSourceEntity(
                    report_id=report.report_id,
                    ops_run_id=evidence_run_id,
                    source_artifact_id=UUID(str(evidence["artifact_id"])),
                    source_kind="INSPECTION",
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
                security_level=int(
                    (run.plan_snapshot_json or {})["target"][
                        "security_level"
                    ]
                ),
            )
        )
        source_payload = dict(source_result.payload_json or {})
        execution_id = source_payload.get("execution_id")
        suffix = execution_id or str(proposal.proposal_id)
        report_key = f"comparison.action.{suffix}"
        report_status = (
            "PARTIAL" if result == "INCONCLUSIVE" else "READY"
        )
        summary = {
            "RESOLVED": "处理后的验证证据表明目标问题已经解决",
            "IMPROVED": "处理后的直接效果指标已改善",
            "UNCHANGED": "处理后的直接效果指标未发生预期变化",
            "DEGRADED": "处理后发现直接效果或护栏指标退化",
            "INCONCLUSIVE": "处理后证据不足，无法形成可靠对比结论",
        }[result]
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
            },
            facts=(
                {
                    "kind": "comparison_result",
                    "result": result,
                    "primary_signals": comparison.primary_signals,
                    "guardrail_signals": comparison.guardrail_signals,
                    "rationale_codes": list(rationale),
                    "causal_limitations": list(
                        comparison.causal_limitations
                    ),
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
            ),
            provenance={
                "deterministic": True,
                "llm_used": False,
                "comparison_result_hash": comparison_hash,
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
                template_id=proposal.action_template_id,
                template_version=proposal.action_template_version,
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
        await self._release_next_action_proposal(
            uow=uow,
            source_run=source_run,
            source_proposal=proposal,
            verification=verification,
            now=now,
            trace_id=trace_id,
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
                security_level=int(target_snapshot.get("security_level", 1)),
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
        block_no = max(int(item.block_no) for item in blocks) + 1
        block_payload = proposal_summary_payload(snapshot)
        answer_block_id = uuid7()
        await uow.turns.add_answer_block(
            OpsAnswerBlockEntity(
                answer_block_id=answer_block_id,
                turn_id=turn.turn_id,
                message_id=message_id,
                block_no=block_no,
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
                "block_no": block_no,
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
            return "RESOLVED", ("ACTION_EFFECT_VERIFIED",)
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
            run_waiting_status = (
                DomainOpsRunStatus.WAITING_APPROVAL
                if waiting_for_approval
                else DomainOpsRunStatus.WAITING_INPUT
            )
            ensure_task_transition(
                DomainOpsTaskStatus(task.status), task_waiting_status
            )
            task.status = task_waiting_status.value
            self._clear_lease(task)
            ensure_run_transition(
                DomainOpsRunStatus(run.status), run_waiting_status
            )
            run.status = run_waiting_status.value
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
                status="WAITING_USER",
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
                security_level=int(
                    target_snapshot.get("security_level", 1)
                ),
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
            if observation.get("metric_code") == "db.availability"
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
                event = await uow.runs.append_event(
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
                assert uow.platform_notifications is not None
                await uow.platform_notifications.emit_run_event(
                    run=run,
                    event_type="aiops.run.failed",
                    summary=policy.safe_message,
                    actor_id=run.actor_id,
                )
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
                }:
                    ensure_task_transition(
                        current, DomainOpsTaskStatus.CANCELLED
                    )
                    task.status = DomainOpsTaskStatus.CANCELLED.value
                    task.completed_at = now
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
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    DomainOpsRunStatus.EXPIRED,
                )
                run.status = DomainOpsRunStatus.EXPIRED.value
                run.completed_at = now
                if run.workflow_kind in _AGENT_TURN_WORKFLOWS:
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
                    event_type="run.expired",
                    event_key=f"run:{run.ops_run_id}:terminal",
                    visibility="USER",
                    payload_json={
                        "status": DomainOpsRunStatus.EXPIRED.value,
                        "trace_id": trace_id,
                    },
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
                expected_run_status = (
                    DomainOpsRunStatus.WAITING_APPROVAL
                    if approval_request
                    else DomainOpsRunStatus.WAITING_INPUT
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
                    or run_status != expected_run_status
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
                ensure_run_transition(
                    run_status,
                    DomainOpsRunStatus.RUNNING,
                )
                run.status = DomainOpsRunStatus.RUNNING.value
                tasks = await uow.runs.list_tasks(
                    ops_run_id=run.ops_run_id, lock=True
                )
                self._release_successors(tasks, now=now)
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
                    status="COLLECTING",
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
                            security_level=int(
                                (run.plan_snapshot_json or {})[
                                    "target"
                                ]["security_level"]
                            ),
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
                            event_key=f"run:{run.ops_run_id}:terminal",
                            visibility="USER",
                            payload_json={
                                "status": "CANCELLED",
                                "trace_id": trace_id,
                            },
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
        status: str | None, severity: str | None,
        cursor: str | None, limit: int,
    ) -> SituationPage:
        filters = {"target_id": str(target_id) if target_id else None,
                   "status": status, "severity": severity}
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._decode_cursor(token=cursor, scope=scope, filters=filters)
        async with self._uow_factory() as uow:
            entities = await uow.situations.page_situations(
                domain_id=scope.domain_id, target_id=target_id, status=status,
                severity=severity, before_created_at=before_at,
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
            return self._run_summary(run, final_artifact=final)

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
                payload=(
                    artifact.payload_json
                    if artifact is not None
                    else None
                ),
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
            summary = self._fire_summary(fire)
            return InspectionFireView(
                **summary.model_dump(),
                run_ids=tuple(item.ops_run_id for item in runs),
                created_at=fire.created_at,
                completed_at=fire.completed_at,
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
            if (
                artifact is None
                or artifact.schema_version != "REPORT_CONTENT.v1"
                or artifact.content_hash != report.content_hash
            ):
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
            if (
                artifact is None
                or artifact.schema_version != "REPORT_CONTENT.v1"
                or artifact.content_hash != report.content_hash
            ):
                raise state_conflict("Report 内容引用不完整")
            content = ReportContent.model_validate(
                dict(artifact.payload_json or {})
            )
            snapshot = dict(content.provenance.get("template") or {})
            definition = snapshot.get("definition")
            if isinstance(definition, dict):
                template = validate_template_definition(definition)
            else:
                template = resolve_report_template_reference(str(report.template_id))
            if template is None:
                raise state_conflict("历史报告缺少可重现的模板快照")
            protected = {"EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX"}
            overrides: dict[str, tuple[str, ...]] = {}
            for section in sections:
                if section.kind in protected:
                    raise validation_failed("证据边界和证据索引不能人工编辑")
                if section.kind not in template.sections:
                    raise validation_failed("报告包含不属于模板的章节")
                if section.kind in overrides:
                    raise validation_failed("同一报告章节只能编辑一次")
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
                },
                trace_id=trace_id,
                now=now,
            )
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
    ) -> ReportEntity:
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
                "DB_DIAGNOSTIC_REPORT.v1",
            }:
                raise validation_failed("当前诊断结果不支持生成正式报告")
            task = next(
                (
                    item for item in await uow.runs.list_tasks(
                        ops_run_id=run.ops_run_id
                    )
                    if item.output_artifact_id == source_artifact.artifact_id
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
            current = await uow.inspections.get_current_report_for_run_template(
                ops_run_id=run.ops_run_id,
                template_id=template.template_ref,
            )
            if current is not None and (
                period_start is None or current.period_start == period_start
            ):
                return current
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
            await uow.commit()
            return report

    async def _aggregate_inspection_sources(
        self, *, uow, runs, period_kind: str, period_start: datetime,
        period_end: datetime,
    ) -> dict[str, Any]:
        """以同一 Target 的终态巡检产物构建可复现的周期 ReportContext。"""
        facts, gaps, evidence_refs = [], [], []
        partial_count = 0
        failed_count = 0
        for item in runs:
            failed_count += getattr(item, "status", "COMPLETED") in {
                "FAILED", "CANCELLED"
            }
            if item.final_artifact_id is None:
                partial_count += 1
                gaps.append({
                    "code": "MISSING_FINAL_ARTIFACT",
                    "source_run_id": str(item.ops_run_id),
                })
                continue
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
            source = normalize_report_source(
                schema_version=artifact.schema_version,
                payload=dict(artifact.payload_json or {}),
                source_kind="INSPECTION",
            )
            partial_count += source["status"] != "READY"
            facts.extend(source.get("facts") or ())
            gaps.extend(source.get("gaps") or ())
            evidence_refs.append({
                "artifact_id": str(artifact.artifact_id),
                "content_hash": artifact.content_hash,
                "schema_version": artifact.schema_version,
                "source_run_id": str(item.ops_run_id),
                "observed_at": item.completed_at.isoformat(),
            })
        return {
            "status": "PARTIAL" if partial_count or gaps else "READY",
            "root_cause": {"effective_level": "INCONCLUSIVE"},
            "diagnosis_rationale": (
                f"{period_kind} 周期覆盖 {len(runs)} 次巡检，"
                f"其中 {partial_count} 次结果不完整，{failed_count} 次执行失败。"
            ),
            "facts": tuple(facts), "gaps": tuple(gaps), "solution": {},
            "evidence_refs": tuple(evidence_refs),
            "inspection_coverage": (
                f"报告时间窗内完成 {len(runs)} 次巡检，"
                f"不完整结果 {partial_count} 次，执行失败 {failed_count} 次。"
            ),
            "period_start": period_start, "period_end": period_end,
        }

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
            if artifact is None:
                raise state_conflict("Report 内容引用不完整")
            payload = dict(artifact.payload_json or {})
            if artifact.content_hash != sha256_json(payload):
                raise state_conflict("Report 内容引用不完整")
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
                template = resolve_report_template_reference(str(report.template_id))
                if template is None:
                    raise state_conflict("历史报告缺少可重现的模板快照")
            return report_presentation(payload=payload, template=template)

    async def get_report_source_agent_id(
        self, *, report_id: UUID, domain_id: int,
    ) -> UUID:
        """在展示或下载前复核报告来源 Agent 的私有授权范围。"""
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
            updated_at=last.created_at,
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
            final_artifact=final_artifact, row_version=int(run.row_version),
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
        return InspectionFireSummary(
            fire_id=fire.inspection_fire_id,
            plan_id=fire.inspection_plan_id,
            scheduled_at=fire.scheduled_for,
            status=fire.status,
            target_count=int(fire.target_count),
            completed_count=int(fire.completed_count),
            failed_count=int(fire.failed_count),
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
            period_start=report.period_start,
            period_end=report.period_end,
            summary=report.summary,
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
            ensure_run_transition(
                DomainOpsRunStatus(run.status),
                DomainOpsRunStatus.RUNNING,
            )
            run.status = DomainOpsRunStatus.RUNNING.value
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            released = self._release_successors(tasks, now=now)
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
                or run.status != DomainOpsRunStatus.WAITING_APPROVAL.value
            ):
                raise state_conflict("诊断查询未处于等待审批状态")
            request_artifact = await self._hitl_request_artifact(uow, hitl)
            request = DiagnosticQueryApprovalRequest.model_validate(
                request_artifact.payload_json
            )
            if (
                request.run_id != str(run.ops_run_id)
                or request.task_id != str(task.ops_task_id)
                or request.target_id != str(run.target_id)
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
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    DomainOpsRunStatus.RUNNING,
                )
                run.status = DomainOpsRunStatus.RUNNING.value
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
                ensure_run_transition(
                    DomainOpsRunStatus(run.status),
                    DomainOpsRunStatus.RUNNING,
                )
                run.status = DomainOpsRunStatus.RUNNING.value
                tasks = await uow.runs.list_tasks(
                    ops_run_id=run.ops_run_id, lock=True
                )
                self._release_successors(tasks, now=now)
                accepted_artifact = self._artifact_ref(artifact)
                event_type = "diagnostic.query_rejected"
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
                status="COLLECTING",
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
            ensure_run_transition(
                DomainOpsRunStatus(run.status),
                DomainOpsRunStatus.RUNNING,
            )
            run.status = DomainOpsRunStatus.RUNNING.value
            tasks = await uow.runs.list_tasks(
                ops_run_id=run.ops_run_id, lock=True
            )
            self._release_successors(tasks, now=now)
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
        return tuple(
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
        if run.deadline_at is not None and run.deadline_at <= now:
            raise _runtime_error(
                "OPS_RUN_DEADLINE_EXCEEDED",
                "Run 已超过截止时间",
            )

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
                item.available_at = now
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
