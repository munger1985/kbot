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
from aiops_agent.application.managed_credentials import AIOpsManagedCredentialService
from aiops_agent.domain.operations import (
    ERROR_CATALOG,
    TASK_TYPE_TO_RUN_PHASE,
    TERMINAL_RUN_STATUSES,
    ensure_run_transition,
    ensure_task_transition,
)
from aiops_agent.adapters.diagnostic_sources.catalog import (
    MetricCatalog,
    load_metric_catalog,
)
from aiops_agent.domain.evidence import DEFAULT_BASELINE_METRICS
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_EVENT_QUERY,
    CAPABILITY_LOG_QUERY,
    CAPABILITY_METRIC_QUERY_RANGE,
)
from aiops_agent.domain.states import (
    DomainOpsRunStatus,
    DomainOpsTaskStatus,
)
from aiops_agent.entities import (
    ChangeProposalEntity,
    EvidenceRequestEntity,
    HitlEntity,
    OpsArtifactEntity,
    OpsConversationMessageEntity,
    OpsRunEntity,
    OpsTaskEntity,
    OutboxEntity,
    ReportEntity,
)
from aiops_agent.contracts.change import (
    ActionVerification,
    ExecutionResultArtifact,
    ProposalOutcome,
)
from aiops_agent.contracts.report import (
    ComparisonPlan,
    ComparisonResult,
    ReportContent,
)
from aiops_agent.orchestration import (
    BlueprintRegistry,
    build_advisory_verification_blueprint,
    build_database_diagnostic_blueprint,
    build_multi_round_diagnosis_blueprint,
    build_monitor_observe_blueprint,
)
from aiops_agent.orchestration.diagnosis import DiagnosisPromptRegistry
from aiops_agent.orchestration.hitl import normalize_raw_response
from aiops_agent.contracts.hitl import (
    HitlOutcome,
    ManualSqlRequest,
    UserDiagnosticSubmission,
)
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.workers.handlers import HandlerRegistry
from platform_core.contracts.aiops import (
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
    ReportSummary,
    ReportVersionPage,
    ReportVersionSummary,
    ReportView,
    SignalEventSummary,
    SituationPage,
    SituationSummary,
    SituationView,
)
from platform_core.contracts.aiops.types import ArtifactRef
from platform_core.identity import uuid7


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
        diagnosis_prompt_registry: DiagnosisPromptRegistry | None = None,
        agent_catalog=None,
        cursor_codec: SignedCursorCodec | None = None,
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
            if target is None or target.status != "ENABLED":
                raise resource_not_found("可用 Target")
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
                    "source_root_cause_grade": source_run.root_cause_level,
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
                "database_endpoint_configured": bool(target.endpoint_json),
                "diagnostic_secret_configured": bool(
                    target.diagnostic_credential_id
                ),
                "execution_secret_configured": bool(
                    target.execution_credential_id
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
                mandatory = {
                    "db.session.active",
                    "db.session.blocking_chain",
                }
                if not mandatory.issubset(requested_tools):
                    raise validation_failed(
                        "Advisory 验证必须检查活动会话和阻塞链"
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
                            policy is not None
                            and policy.rules_json.get(
                                "allow_agent_execution"
                            )
                            is not True,
                            "POLICY_MUTATION_DENIED",
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
                plan_snapshot["diagnosis"] = self._diagnosis_snapshot(
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
                        investigation_mode={
                            "ALERT": "INCIDENT",
                            "SCHEDULE": "INSPECTION",
                        }.get(str(command.trigger_type), "ON_DEMAND"),
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
                    task_type=spec.task_type,
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
                    root_cause_grade=(
                        run.root_cause_level or "INCONCLUSIVE"
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

    def _diagnosis_snapshot(
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
            "prompts": self._diagnosis_prompts.snapshot,
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
        policy_rules = dict(policy.rules_json) if policy is not None else {}
        policy_allowed = policy_rules.get(
            "readonly_database_enabled", True
        )
        if not policy_allowed:
            initial_gaps.append(
                {
                    "code": "DIAGNOSTIC_POLICY_DENIED",
                    "detail": "当前策略禁止数据库直连诊断",
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
            policy_allowed
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
        if (command.observation_start is None) != (
            command.observation_end is None
        ):
            raise validation_failed("观测窗口起止时间必须同时提供")
        window_end = command.observation_end or now
        window_start = command.observation_start or (
            window_end
            - timedelta(seconds=self._default_observation_window)
        )
        if window_start >= window_end or window_end > now + timedelta(
            seconds=5
        ):
            raise validation_failed("观测窗口无效或结束时间位于未来")
        monitors = await uow.targets.list_source_bindings(
            target_id=target.target_id,
            domain_id=command.domain_id,
            active_only=True,
        )
        if allowed_source_ids is not None:
            allowed = set(allowed_source_ids)
            monitors = [
                monitor
                for monitor in monitors
                if monitor.diagnostic_source_id in allowed
            ]
        snapshots = []
        initial_gaps = []
        observation_binding_ids = []
        log_binding_ids = []
        for monitor in monitors:
            source = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=monitor.diagnostic_source_id,
                domain_id=command.domain_id,
            )
            if source is None or source.status != "ENABLED":
                initial_gaps.append(
                    {
                        "binding_id": str(monitor.target_source_binding_id),
                        "source_id": str(monitor.diagnostic_source_id),
                        "code": "DIAGNOSTIC_SOURCE_INACTIVE",
                        "detail": "监控源不存在或未激活",
                    }
                )
                continue
            if source.connectivity_status not in {"CONNECTED", "DEGRADED"}:
                initial_gaps.append(
                    {
                        "binding_id": str(monitor.target_source_binding_id),
                        "source_id": str(monitor.diagnostic_source_id),
                        "code": "DIAGNOSTIC_SOURCE_UNAVAILABLE",
                        "detail": "监控源当前不可连接",
                    }
                )
                continue
            requested = (monitor.capability_scope_json or {}).get(
                "metric_codes", DEFAULT_BASELINE_METRICS
            )
            if (
                not isinstance(requested, (list, tuple))
                or not requested
                or len(requested) > 64
                or not all(
                    isinstance(item, str) and item for item in requested
                )
            ):
                raise validation_failed("监控绑定的 metric_codes 格式无效")
            requested_codes = tuple(dict.fromkeys(requested))
            try:
                selected = self._metric_catalog.select(
                    requested_codes, db_type=target.db_type
                )
            except KeyError as exc:
                raise validation_failed("监控绑定引用了未知标准指标") from exc
            supported = tuple(
                item
                for item in selected
                if source.source_type in item.providers
            )
            declared_capabilities = set(
                (source.declared_capabilities_json or {}).keys()
            )
            requested_capabilities = (
                monitor.capability_scope_json or {}
            ).get("capabilities")
            if requested_capabilities is not None:
                if (
                    not isinstance(requested_capabilities, (list, tuple))
                    or not requested_capabilities
                    or not all(
                        isinstance(item, str) and item
                        for item in requested_capabilities
                    )
                ):
                    raise validation_failed(
                        "Source Binding capabilities 格式无效"
                    )
                effective_capabilities = declared_capabilities.intersection(
                    requested_capabilities
                )
            else:
                effective_capabilities = declared_capabilities
            binding_id = str(monitor.target_source_binding_id)
            if effective_capabilities.intersection(
                {CAPABILITY_METRIC_QUERY_RANGE, CAPABILITY_EVENT_QUERY}
            ):
                observation_binding_ids.append(binding_id)
            if CAPABILITY_LOG_QUERY in effective_capabilities:
                log_binding_ids.append(binding_id)
            snapshots.append(
                {
                    "binding_id": binding_id,
                    "binding_version": int(monitor.row_version),
                    "role": monitor.role,
                    "priority": int(monitor.priority),
                    "source_locator_key": monitor.source_locator_key,
                    "source_locator": dict(monitor.source_locator_json),
                    "source_locator_fingerprint": hashlib.sha256(
                        monitor.source_locator_key.encode("utf-8")
                    ).hexdigest(),
                    "mapping_overrides": dict(
                        monitor.mapping_overrides_json or {}
                    ),
                    "query_budget": dict(monitor.query_budget_json or {}),
                    "effective_capabilities": sorted(
                        effective_capabilities
                    ),
                    "source": {
                        "source_id": str(source.diagnostic_source_id),
                        "source_type": source.source_type,
                        "adapter_id": source.adapter_id,
                        "adapter_version": source.adapter_version,
                        "config_version": int(source.row_version),
                        "endpoint": source.endpoint,
                        "secret_ref": (
                            AIOpsManagedCredentialService.reference(
                                domain_id=int(source.domain_id),
                                external_key=source.diagnostic_source_id,
                                credential_kind="diagnostic_source",
                                credential_id=source.auth_credential_id,
                            )
                            if source.auth_credential_id
                            else None
                        ),
                        "declared_capabilities": dict(
                            source.declared_capabilities_json or {}
                        ),
                        "config": dict(source.config_json or {}),
                    },
                    "metrics": [
                        item.model_dump(mode="json")
                        for item in supported
                        if CAPABILITY_METRIC_QUERY_RANGE
                        in effective_capabilities
                    ],
                    "unsupported_metrics": sorted(
                        (
                            set(requested_codes)
                            - {item.metric_code for item in supported}
                        )
                        if CAPABILITY_METRIC_QUERY_RANGE
                        in effective_capabilities
                        else ()
                    ),
                }
            )
        blueprint = build_monitor_observe_blueprint(
            tuple(observation_binding_ids)
        )
        return blueprint, {
            "window": {
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
            },
            "catalog_version": self._metric_catalog.version,
            "catalog_hash": self._metric_catalog.manifest_hash,
            "max_response_bytes": self._max_monitor_response_bytes,
            "bindings": snapshots,
            "observation_binding_ids": sorted(observation_binding_ids),
            "log_binding_ids": sorted(log_binding_ids),
            "initial_gaps": initial_gaps,
        }

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
            artifacts = await self._input_artifacts(
                uow, run_id=run.ops_run_id, task=task
            )
            await uow.commit()
            return self._task_lease(
                run, task, artifacts, lease_token=lease_token
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
                if (
                    run.trigger_type == "SCHEDULE"
                    and run.inspection_fire_id is not None
                    and artifact.schema_version
                    == "DB_DIAGNOSTIC_REPORT.v1"
                ):
                    final_artifact = (
                        await self._publish_inspection_report(
                            uow=uow,
                            run=run,
                            task=task,
                            source_artifact=artifact,
                            now=now,
                            trace_id=command.trace_id,
                        )
                    )
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
                if (
                    artifact.schema_version
                    == "DIAGNOSIS_REPORT_DRAFT.v1"
                    and (artifact.payload_json or {}).get("output_kind")
                    == "DIAGNOSIS_REPORT"
                ):
                    await self._publish_diagnosis_report(
                        uow=uow,
                        run=run,
                        task=task,
                        source_artifact=artifact,
                        now=now,
                        trace_id=command.trace_id,
                    )
                run.final_artifact_id = final_artifact.artifact_id
                if artifact.schema_version in {
                    "OBSERVE_REPORT.v1",
                    "DB_DIAGNOSTIC_REPORT.v1",
                }:
                    run.root_cause_level = "INCONCLUSIVE"
                elif (
                    artifact.schema_version
                    == "DIAGNOSIS_REPORT_DRAFT.v1"
                ):
                    root_cause = (
                        (command.artifact.payload or {}).get(
                            "root_cause", {}
                        )
                    )
                    run.root_cause_level = root_cause.get(
                        "effective_level", "INCONCLUSIVE"
                    )
                run.completed_at = now
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
                security_level=int(plan["target"]["security_level"]),
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
                security_level=int(plan["target"]["security_level"]),
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

    async def _publish_diagnosis_report(
        self,
        *,
        uow,
        run,
        task,
        source_artifact,
        now: datetime,
        trace_id: str,
    ) -> None:
        """仅在动态决策要求留档时发布正式诊断报告。"""
        assert uow.inspections is not None
        plan = dict(run.plan_snapshot_json or {})
        source = dict(source_artifact.payload_json or {})
        question = str(
            plan.get("diagnosis", {}).get("question_summary") or ""
        )
        performance_keywords = (
            "性能",
            "慢",
            "响应",
            "吞吐",
            "连接",
            "锁",
            "等待",
            "performance",
            "latency",
        )
        report_type = (
            "PERFORMANCE"
            if any(item in question.lower() for item in performance_keywords)
            else "INCIDENT"
        )
        report_key = (
            "diagnosis.performance"
            if report_type == "PERFORMANCE"
            else "diagnosis.incident"
        )
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
        content = ReportContent(
            report_key=report_key,
            report_type=report_type,
            ops_run_id=str(run.ops_run_id),
            target_id=str(run.target_id),
            title=(
                "数据库性能诊断报告"
                if report_type == "PERFORMANCE"
                else "数据库故障诊断报告"
            ),
            status=status,
            summary=summary,
            period_start=run.created_at,
            period_end=now,
            scope={
                "question_summary": question,
                "root_cause_grade": grade,
                "report_decision_reasons": list(
                    source.get("report_decision_reasons") or ()
                ),
                "effective_capabilities": dict(
                    plan.get("effective_capabilities") or {}
                ),
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
            ),
            recommendations=recommendations,
            provenance={
                "deterministic_report_decision": True,
                "source_artifact_hash": source_artifact.content_hash,
                "model_receipt_hashes": list(
                    source.get("model_receipt_hashes") or ()
                ),
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
                    "producer": "aiops.report-publisher",
                    "producer_version": "1",
                    "source_artifact_id": str(source_artifact.artifact_id),
                },
                trust_level="SOURCE_VERIFIED",
                security_level=int(plan["target"]["security_level"]),
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
                period_start=run.created_at,
                period_end=now,
                template_id="diagnosis.dynamic",
                template_version="1",
                generated_by_task_id=task.ops_task_id,
                content_artifact_id=report_artifact.artifact_id,
                content_hash=content_hash,
                summary=summary,
                security_level=int(plan["target"]["security_level"]),
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
        return report_artifact

    @staticmethod
    def _comparison_result(
        verification: ActionVerification,
    ) -> tuple[str, tuple[str, ...]]:
        """把只读 Verification 结果映射为稳定、可审计的对比结论。"""
        if verification.status == "ADVERSE":
            return "DEGRADED", ("VERIFICATION_ADVERSE",)
        if (
            verification.status == "INCONCLUSIVE"
            or verification.gap_codes
            or verification.target_still_present is None
            or verification.blocking_still_present is None
        ):
            return "INCONCLUSIVE", ("EVIDENCE_NOT_COMPARABLE",)
        if (
            verification.status == "VERIFIED"
            and not verification.target_still_present
            and not verification.blocking_still_present
        ):
            return "RESOLVED", ("TARGET_AND_BLOCKING_EFFECT_CLEARED",)
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
                not in {"DATA_REQUIRED", "MANUAL_DIAGNOSTIC_SQL"}
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
            conversation_id = (run.plan_snapshot_json or {}).get(
                "client_metadata", {}
            ).get("conversation_id")
            conversations = getattr(uow, "conversations", None)
            if conversation_id and conversations is not None:
                try:
                    parsed_conversation_id = UUID(str(conversation_id))
                except (TypeError, ValueError):
                    raise state_conflict("Run 的 Conversation 关联无效") from None
                conversation = await conversations.get_conversation(
                    domain_id=int(run.domain_id),
                    conversation_id=parsed_conversation_id,
                    lock=True,
                )
                if (
                    conversation is None
                    or conversation.agent_id != run.agent_id
                    or conversation.created_by != run.actor_id
                ):
                    raise state_conflict("Run 的 Conversation 关联不可信")
                queries = list(content.get("queries") or [])
                sql_blocks = [
                    f"-- [{item.get('query_id', 'query')}]\n{item.get('sql_text', '')}"
                    for item in queries
                    if item.get("sql_text")
                ]
                await conversations.add_evidence_request(
                    EvidenceRequestEntity(
                        request_id=command.hitl_id,
                        conversation_id=parsed_conversation_id,
                        purpose=command.prompt_text,
                        suggested_sql="\n\n".join(sql_blocks) or None,
                        sql_hash=(
                            hashlib.sha256(
                                "\n\n".join(sql_blocks).encode()
                            ).hexdigest()
                            if sql_blocks
                            else None
                        ),
                        requested_by=command.request_artifact.producer,
                    )
                )
                conversation.status = "WAITING_EVIDENCE"
                await conversations.add_message(
                    OpsConversationMessageEntity(
                        conversation_id=parsed_conversation_id,
                        sequence_no=await conversations.next_message_sequence(
                            conversation_id=parsed_conversation_id
                        ),
                        role="AGENT",
                        message_type="EVIDENCE_REQUEST",
                        payload_json={
                            "request_id": str(command.hitl_id),
                            "purpose": command.prompt_text,
                            "suggested_sql": "\n\n".join(sql_blocks) or None,
                            "query_ids": [
                                str(item.get("query_id")) for item in queries
                            ],
                        },
                    )
                )
            ensure_task_transition(
                DomainOpsTaskStatus(task.status),
                DomainOpsTaskStatus.WAITING_INPUT,
            )
            task.status = DomainOpsTaskStatus.WAITING_INPUT.value
            self._clear_lease(task)
            ensure_run_transition(
                DomainOpsRunStatus(run.status),
                DomainOpsRunStatus.WAITING_INPUT,
            )
            run.status = DomainOpsRunStatus.WAITING_INPUT.value
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
                event_type="diagnostic.input_required",
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
                summary="诊断需要补充输入",
                actor_id=run.actor_id,
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
        status = (
            "PENDING_APPROVAL"
            if snapshot.mode == "AGENT_EXECUTE"
            else "ADVISORY_READY"
        )
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
                    (run.plan_snapshot_json or {})["target"][
                        "security_level"
                    ]
                ),
            )
        )
        proposal = ChangeProposalEntity(
            proposal_id=UUID(snapshot.proposal_id),
            ops_run_id=run.ops_run_id,
            ops_task_id=task.ops_task_id,
            target_id=run.target_id,
            solution_group_key=snapshot.solution_group_key,
            command_ordinal=snapshot.command_ordinal,
            proposal_version=snapshot.proposal_version,
            action_type=snapshot.mode,
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
            domain_id=int(
                (run.plan_snapshot_json or {})["target"]["domain_id"]
            ),
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
                    or task_status
                    != DomainOpsTaskStatus.WAITING_INPUT
                    or run_status
                    != DomainOpsRunStatus.WAITING_INPUT
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
                    gap_code="MANUAL_DIAGNOSTIC_EXPIRED",
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
                    "reason": "MANUAL_DIAGNOSTIC_EXPIRED"
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
                    DomainOpsRunStatus.DIAGNOSING,
                )
                run.status = DomainOpsRunStatus.DIAGNOSING.value
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
            events = await uow.situations.list_events_for_situation(situation_id=situation_id)
            runs = await uow.runs.list_by_situation(situation_id=situation_id)
            return SituationView(
                **self._situation_summary(situation).model_dump(),
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
                root_cause_grade=run.root_cause_level,
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
            report = await uow.inspections.get_current_report_scoped(
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
            return ReportView(
                report_id=report.report_id,
                report_key=report.report_key,
                report_type=report.report_type,
                report_version=int(report.report_version),
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
        elif source_type == "diagnostic.input_required":
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
                or run.root_cause_level
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
            investigation_mode=run.investigation_mode, status=run.status,
            root_cause_grade=run.root_cause_level,
            source_proposal_id=getattr(run, "source_proposal_id", None),
            source_result_artifact_id=getattr(run, "source_result_artifact_id", None),
            final_artifact=final_artifact, row_version=int(run.row_version),
            created_at=run.created_at, completed_at=run.completed_at,
        )

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
                DomainOpsRunStatus.DIAGNOSING,
            )
            run.status = DomainOpsRunStatus.DIAGNOSING.value
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
                DomainOpsRunStatus.DIAGNOSING,
            )
            run.status = DomainOpsRunStatus.DIAGNOSING.value
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
        if (
            artifact is None
            or artifact.ops_run_id != hitl.ops_run_id
            or artifact.schema_version != "MANUAL_SQL_REQUEST.v1"
        ):
            raise state_conflict("人工补证请求 Artifact 不存在或类型无效")
        return artifact

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
            if item.ops_task_id in producer_ids
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
    ) -> None:
        if (
            task.status != DomainOpsTaskStatus.RUNNING.value
            or task.lease_owner != worker_id
            or task.lease_token != lease_token
            or task.lease_until is None
            or task.lease_until <= now
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
