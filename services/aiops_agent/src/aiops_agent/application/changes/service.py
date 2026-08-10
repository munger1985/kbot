"""Advisory Proposal 的读取、驳回和人工结果记录。"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from aiops_agent.actions import (
    ActionRegistry,
    ActionRenderer,
    MutationGrantCodec,
)
from aiops_agent.application.errors import (
    resource_not_found,
    state_conflict,
)
from aiops_agent.contracts.change import (
    ApprovalDecision,
    AdvisoryActionResult,
    ExecutionResultArtifact,
    ProposalOutcome,
)
from aiops_agent.entities import (
    ApprovalTokenEntity,
    ExecutionEntity,
    HitlEntity,
    InboxEntity,
    OpsArtifactEntity,
    OutboxEntity,
)
from platform_core.contracts.aiops.public import (
    ApprovalCommand,
    ApprovalReceipt,
    ManualResultCommand,
    ManualResultReceipt,
    ProposalView,
    RejectionCommand,
)
from platform_core.contracts.aiops.executor import (
    ExecutionStatusEvent,
    MutationClaimReceipt,
    MutationClaimRequest,
    MutationExecutionGrant,
)
from platform_core.contracts.aiops.internal import EventReceipt
from platform_core.contracts.aiops.types import ArtifactRef
from platform_core.identity import uuid7


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


def _hash(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class AIOpsChangeService:
    def __init__(
        self,
        *,
        uow_factory,
        action_registry: ActionRegistry | None = None,
        approval_enabled: bool = False,
        mutation_enabled: bool = False,
        approval_ttl_seconds: int = 300,
        mutation_grant_codec: MutationGrantCodec | None = None,
        mutation_grant_issuer: str = "kbot-aiops-api",
        mutation_grant_audience: str = "kbot-aiops-db-executor",
        mutation_grant_ttl_seconds: int = 30,
        mutation_statement_timeout_seconds: int = 60,
    ):
        self._uow_factory = uow_factory
        self._action_registry = action_registry
        self._renderer = ActionRenderer()
        self._approval_enabled = approval_enabled
        self._mutation_enabled = mutation_enabled
        self._approval_ttl_seconds = approval_ttl_seconds
        self._mutation_grant_codec = mutation_grant_codec
        self._mutation_grant_issuer = mutation_grant_issuer
        self._mutation_grant_audience = mutation_grant_audience
        self._mutation_grant_ttl_seconds = (
            mutation_grant_ttl_seconds
        )
        self._mutation_statement_timeout_seconds = (
            mutation_statement_timeout_seconds
        )

    async def get_proposal(
        self,
        *,
        proposal_id: UUID,
        domain_id: int,
    ) -> ProposalView:
        async with self._uow_factory() as uow:
            proposal = await uow.changes.get_proposal_scoped(
                proposal_id=proposal_id,
                domain_id=domain_id,
            )
            if proposal is None:
                raise resource_not_found("Proposal")
            snapshot = await self._snapshot(uow, proposal)
            return self._view(proposal, snapshot)

    async def reject_proposal(
        self,
        *,
        proposal_id: UUID,
        domain_id: int,
        actor_id: str,
        command: RejectionCommand,
        trace_id: str,
    ) -> ProposalView:
        async with self._uow_factory() as uow:
            preliminary = await uow.changes.get_proposal_scoped(
                proposal_id=proposal_id,
                domain_id=domain_id,
            )
            if preliminary is None:
                raise resource_not_found("Proposal")
            run = await uow.runs.get_run(
                ops_run_id=preliminary.ops_run_id, lock=True
            )
            proposal = await uow.changes.get_proposal(
                proposal_id=proposal_id, lock=True
            )
            if (
                run is None
                or proposal is None
                or run.actor_id != actor_id
            ):
                raise resource_not_found("Proposal")
            if int(proposal.row_version) != command.expected_row_version:
                raise state_conflict("Proposal 版本已变化")
            if proposal.status not in {
                "ADVISORY_READY",
                "PENDING_APPROVAL",
            }:
                raise state_conflict("Proposal 当前不能驳回")
            now = await uow.runs.database_now()
            proposal.status = "REJECTED"
            proposal.updated_at = now
            approval_hitl = await uow.changes.get_pending_hitl(
                ops_task_id=proposal.ops_task_id,
                request_type="CHANGE_APPROVAL",
                lock=True,
            )
            if (
                approval_hitl is not None
                and approval_hitl.proposal_id == proposal.proposal_id
            ):
                approval_hitl.status = "REJECTED"
                approval_hitl.responded_by = actor_id
                approval_hitl.responded_at = now
                approval_hitl.response_json = {
                    "reason_hash": _hash(command.reason)
                }
                approval_hitl.response_hash = _hash(
                    approval_hitl.response_json
                )
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=proposal.ops_task_id,
                event_type="proposal.rejected",
                event_key=f"proposal:{proposal.proposal_id}:rejected",
                visibility="USER",
                payload_json={
                    "proposal_id": str(proposal.proposal_id),
                    "reason_hash": _hash(command.reason),
                    "trace_id": trace_id,
                },
            )
            assert uow.platform_notifications is not None
            await uow.platform_notifications.emit_proposal_event(
                run=run,
                proposal=proposal,
                event_type="aiops.proposal.rejected",
                summary="变更方案已驳回",
                actor_id=actor_id,
            )
            snapshot = await self._snapshot(uow, proposal)
            await uow.commit()
            view = self._view(proposal, snapshot)
            return view.model_copy(
                update={"row_version": int(proposal.row_version) + 1}
            )

    async def approve_proposal(
        self,
        *,
        proposal_id: UUID,
        domain_id: int,
        actor_id: str,
        command: ApprovalCommand,
        idempotency_key: str,
        trace_id: str,
    ) -> ApprovalReceipt:
        """原子签发一次性授权并创建尚未投递的 Execution。"""
        if (
            not self._approval_enabled
            or not self._mutation_enabled
            or self._action_registry is None
        ):
            raise state_conflict("受控执行部署开关尚未启用")
        async with self._uow_factory() as uow:
            replay = await uow.changes.get_execution_by_idempotency(
                idempotency_key=idempotency_key
            )
            if replay is not None:
                if (
                    replay.proposal_id != proposal_id
                    or replay.proposal_hash
                    != command.expected_proposal_hash
                ):
                    raise state_conflict("审批幂等键对应的请求不同")
                token = await uow.changes.get_approval_token(
                    approval_token_id=replay.approval_token_id
                )
                if token is None:
                    raise state_conflict("审批幂等记录不完整")
                return self._approval_receipt(replay, token)

            preliminary = await uow.changes.get_proposal_scoped(
                proposal_id=proposal_id,
                domain_id=domain_id,
            )
            if preliminary is None:
                raise resource_not_found("Proposal")
            run = await uow.runs.get_run(
                ops_run_id=preliminary.ops_run_id, lock=True
            )
            proposal = await uow.changes.get_proposal(
                proposal_id=proposal_id, lock=True
            )
            if run is None or proposal is None:
                raise resource_not_found("Proposal")
            if int(proposal.row_version) != command.expected_row_version:
                raise state_conflict("Proposal 版本已变化")
            if proposal.proposal_hash != command.expected_proposal_hash:
                raise state_conflict("用户看到的 Proposal 已发生变化")
            if proposal.status != "PENDING_APPROVAL":
                raise state_conflict("Proposal 当前不能审批")
            now = await uow.runs.database_now()
            if (
                proposal.expires_at is None
                or proposal.expires_at <= now
            ):
                raise state_conflict("Proposal 已过期")
            hitl = await uow.changes.get_pending_hitl(
                ops_task_id=proposal.ops_task_id,
                request_type="CHANGE_APPROVAL",
                lock=True,
            )
            if hitl is None or hitl.proposal_id != proposal_id:
                raise state_conflict("Proposal 缺少有效审批节点")
            if hitl.assignee_user_id != actor_id:
                raise resource_not_found("Proposal")

            target = await uow.targets.get_scoped(
                target_id=proposal.target_id,
                domain_id=domain_id,
                lock=True,
            )
            if (
                target is None
                or target.status != "ACTIVE"
                or not target.execution_credential_id
            ):
                raise state_conflict("Target 当前不允许受控执行")
            snapshot = await self._snapshot(uow, proposal)
            if int(target.row_version) != snapshot.target_version:
                raise state_conflict("Target 配置版本已变化")
            binding = await uow.targets.get_agent_binding(
                target_id=target.target_id,
                agent_id=run.agent_id,
                domain_id=domain_id,
                lock=True,
            )
            if (
                binding is None
                or binding.status != "ACTIVE"
                or not bool(binding.allow_mutation)
                or proposal.action_template_id
                not in set(binding.allowed_actions_json or ())
            ):
                raise state_conflict("Agent Binding 当前不允许该动作")
            policy = (
                await uow.policies.get_scoped(
                    policy_id=binding.policy_id,
                    domain_id=domain_id,
                    lock=True,
                )
                if binding.policy_id is not None
                else None
            )
            rules = dict(policy.rules_json) if policy is not None else {}
            frozen_policy = dict(run.policy_snapshot_json or {})
            if (
                policy is None
                or policy.status != "ACTIVE"
                or rules.get("allow_agent_execution") is not True
                or frozen_policy.get("policy_hash") != policy.policy_hash
            ):
                raise state_conflict("当前 Policy 不再允许该动作")
            capabilities = {
                name
                for name, enabled in dict(
                    target.capabilities_json or {}
                ).items()
                if enabled is True
            }
            try:
                template = self._action_registry.resolve(
                    action_template_id=proposal.action_template_id,
                    version=proposal.action_template_version,
                    db_type=target.db_type,
                    db_version=target.version_code or "UNKNOWN",
                    capabilities=capabilities,
                    entitlements=set(rules.get("entitlements", ())),
                    environment=target.environment,
                )
                rendered = self._renderer.render(
                    template, dict(proposal.parameters_json)
                )
            except (LookupError, ValueError):
                raise state_conflict(
                    "Action Catalog 或参数当前不可执行"
                ) from None
            if (
                rendered.execution_capability
                != "EXECUTABLE_AFTER_APPROVAL"
                or rendered.template_hash
                != proposal.action_template_hash
                or rendered.parameters_hash != proposal.parameters_hash
                or rendered.command_hash != proposal.command_hash
            ):
                raise state_conflict("Action Catalog 或渲染结果已变化")

            token_id = uuid7()
            execution_id = uuid7()
            executor_request_id = uuid7()
            expires_at = min(
                proposal.expires_at,
                now + timedelta(seconds=self._approval_ttl_seconds),
            )
            nonce = str(uuid7())
            approval_claims = {
                "proposal_id": str(proposal_id),
                "proposal_hash": proposal.proposal_hash,
                "target_id": str(target.target_id),
                "target_version": int(target.row_version),
                "action_template_id": proposal.action_template_id,
                "action_template_version": (
                    proposal.action_template_version
                ),
                "action_template_hash": proposal.action_template_hash,
                "command_hash": proposal.command_hash,
                "parameters_hash": proposal.parameters_hash,
                "policy_decision_hash": (
                    proposal.policy_decision_hash
                ),
                "approver_id": actor_id,
                "issued_at": now,
                "expires_at": expires_at,
                "nonce": nonce,
            }
            token = await uow.changes.add_approval_token(
                ApprovalTokenEntity(
                    approval_token_id=token_id,
                    proposal_id=proposal_id,
                    hitl_id=hitl.hitl_id,
                    token_hash=_hash(approval_claims),
                    nonce_hash=_hash(nonce),
                    approver_id=actor_id,
                    policy_decision_hash=(
                        proposal.policy_decision_hash
                    ),
                    target_version=int(target.row_version),
                    parameters_hash=proposal.parameters_hash,
                    status="ISSUED",
                    issued_at=now,
                    expires_at=expires_at,
                )
            )
            execution = await uow.changes.add_execution(
                ExecutionEntity(
                    execution_id=execution_id,
                    proposal_id=proposal_id,
                    ops_run_id=run.ops_run_id,
                    ops_task_id=proposal.ops_task_id,
                    target_id=target.target_id,
                    idempotency_key=idempotency_key,
                    executor_request_id=str(executor_request_id),
                    proposal_hash=proposal.proposal_hash,
                    action_type=proposal.action_type,
                    action_template_id=proposal.action_template_id,
                    action_template_version=(
                        proposal.action_template_version
                    ),
                    action_template_hash=(
                        proposal.action_template_hash
                    ),
                    parameters_hash=proposal.parameters_hash,
                    command_hash=proposal.command_hash,
                    execution_kind="MUTATION",
                    approval_token_id=token_id,
                    status="CREATED",
                    status_version=1,
                    deadline_at=expires_at,
                )
            )
            decision = ApprovalDecision(
                proposal_id=str(proposal_id),
                proposal_hash=proposal.proposal_hash,
                approval_token_id=str(token_id),
                execution_id=str(execution_id),
                approver_id=actor_id,
                approved_at=now,
                expires_at=expires_at,
                policy_hash=policy.policy_hash,
                target_version=int(target.row_version),
                parameters_hash=proposal.parameters_hash,
                note_hash=(
                    _hash(command.note)
                    if command.note is not None
                    else None
                ),
            )
            decision_payload = decision.model_dump(mode="json")
            await uow.runs.add_artifact(
                OpsArtifactEntity(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=proposal.ops_task_id,
                    artifact_key=(
                        f"proposal:{proposal_id}:approval:v1"
                    ),
                    artifact_type="APPROVAL_DECISION",
                    schema_version="APPROVAL_DECISION.v1",
                    payload_json=decision_payload,
                    content_hash=_hash(decision_payload),
                    byte_size=len(_canonical(decision_payload)),
                    provenance_json={
                        "producer": "aiops.change-service",
                        "producer_version": "approval.v1",
                        "approver_id": actor_id,
                    },
                    trust_level="SOURCE_VERIFIED",
                    security_level=int(target.security_level),
                )
            )
            proposal.status = "APPROVED"
            proposal.updated_at = now
            hitl.status = "APPROVED"
            hitl.responded_by = actor_id
            hitl.responded_at = now
            hitl.response_json = {
                "approval_token_id": str(token_id),
                "execution_id": str(execution_id),
            }
            hitl.response_hash = _hash(hitl.response_json)
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=proposal.ops_task_id,
                event_type="proposal.approved",
                event_key=f"proposal:{proposal_id}:approved",
                visibility="USER",
                payload_json={
                    "proposal_id": str(proposal_id),
                    "execution_id": str(execution_id),
                    "authorization_expires_at": (
                        expires_at.isoformat()
                    ),
                    "trace_id": trace_id,
                },
            )
            assert uow.platform_notifications is not None
            await uow.platform_notifications.emit_proposal_event(
                run=run,
                proposal=proposal,
                event_type="aiops.proposal.approved",
                summary="变更方案已批准",
                actor_id=actor_id,
            )
            outbox_payload = {
                "execution_id": str(execution_id),
                "executor_request_id": str(executor_request_id),
                "trace_id": trace_id,
            }
            await uow.outbox.add(
                OutboxEntity(
                    aggregate_type="OPS_EXECUTION",
                    aggregate_id=execution_id,
                    event_type="OPS_EXECUTION_CREATED",
                    idempotency_key=(
                        f"execution:{execution_id}:created"
                    ),
                    payload_json=outbox_payload,
                    payload_hash=_hash(outbox_payload),
                    status="PENDING",
                    available_at=now,
                    max_attempts=5,
                    trace_id=trace_id,
                )
            )
            try:
                await uow.commit()
            except IntegrityError:
                await uow.rollback()
                return await self._replay_approval(
                    proposal_id=proposal_id,
                    idempotency_key=idempotency_key,
                    expected_proposal_hash=(
                        command.expected_proposal_hash
                    ),
                    actor_id=actor_id,
                )
            return self._approval_receipt(execution, token)

    async def record_manual_result(
        self,
        *,
        proposal_id: UUID,
        domain_id: int,
        actor_id: str,
        command: ManualResultCommand,
        idempotency_key: str,
        trace_id: str,
    ) -> ManualResultReceipt:
        async with self._uow_factory() as uow:
            preliminary = await uow.changes.get_proposal_scoped(
                proposal_id=proposal_id,
                domain_id=domain_id,
            )
            if preliminary is None:
                raise resource_not_found("Proposal")
            run = await uow.runs.get_run(
                ops_run_id=preliminary.ops_run_id, lock=True
            )
            proposal = await uow.changes.get_proposal(
                proposal_id=proposal_id, lock=True
            )
            if (
                run is None
                or proposal is None
                or run.actor_id != actor_id
            ):
                raise resource_not_found("Proposal")
            prior = await uow.changes.get_hitl_by_idempotency(
                ops_run_id=run.ops_run_id,
                idempotency_key=idempotency_key,
            )
            artifact_key = f"proposal:{proposal_id}:manual-result:v1"
            request_hash = _hash(command.model_dump(mode="json"))
            if prior is not None:
                if prior.proposal_id != proposal_id:
                    raise state_conflict("人工结果幂等键已用于其他 Proposal")
                artifact = await uow.runs.get_artifact_by_key(
                    ops_run_id=run.ops_run_id,
                    artifact_key=artifact_key,
                )
                if artifact is None:
                    raise state_conflict("人工结果幂等记录不完整")
                if prior.response_hash != request_hash:
                    raise state_conflict("相同幂等键对应的人工结果内容不同")
                return ManualResultReceipt(
                    proposal_id=proposal_id,
                    status=str(prior.response_json["status"]),
                    result_artifact=self._artifact_ref(artifact),
                )
            if int(proposal.row_version) != command.expected_row_version:
                raise state_conflict("Proposal 版本已变化")
            if proposal.status != "ADVISORY_READY":
                raise state_conflict("只有 Advisory Proposal 可回填人工结果")
            now = await uow.runs.database_now()
            if (
                proposal.expires_at is not None
                and proposal.expires_at <= now
            ):
                raise state_conflict("Advisory Proposal 已过期")
            if command.occurred_at > now:
                raise state_conflict("人工处理时间不能晚于当前时间")
            body = {
                "proposal_id": str(proposal_id),
                "status": str(command.status),
                "occurred_at": command.occurred_at,
                "submitted_at": now,
                "submitted_by": actor_id,
                "note": command.note,
                "bounded_output": command.bounded_output,
            }
            result = AdvisoryActionResult(
                **body,
                result_hash=_hash(body),
            )
            payload = result.model_dump(mode="json")
            artifact = await uow.runs.add_artifact(
                OpsArtifactEntity(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=proposal.ops_task_id,
                    artifact_key=artifact_key,
                    artifact_type="USER_PROVIDED_ACTION_RESULT",
                    schema_version="USER_PROVIDED_ACTION_RESULT.v1",
                    payload_json=payload,
                    content_hash=_hash(payload),
                    byte_size=len(_canonical(payload)),
                    provenance_json={
                        "producer": "user",
                        "producer_version": "manual-action-result.v1",
                        "actor_id": actor_id,
                    },
                    trust_level="USER_PROVIDED",
                    security_level=1,
                )
            )
            hitl_id = uuid7()
            await uow.changes.add_hitl(
                HitlEntity(
                    hitl_id=hitl_id,
                    ops_run_id=run.ops_run_id,
                    ops_task_id=proposal.ops_task_id,
                    proposal_id=proposal.proposal_id,
                    request_type="MANUAL_ACTION_RESULT",
                    assignee_user_id=actor_id,
                    prompt_text="用户回填 Advisory 处理结果",
                    response_schema_json={
                        "schema_version": (
                            "USER_PROVIDED_ACTION_RESULT.v1"
                        )
                    },
                    input_artifacts_json=[
                        str(proposal.snapshot_artifact_id)
                    ],
                    response_json={
                        "status": str(command.status),
                        "artifact_id": str(artifact.artifact_id),
                    },
                    response_hash=request_hash,
                    status="ANSWERED",
                    idempotency_key=idempotency_key,
                    requested_by=actor_id,
                    responded_by=actor_id,
                    requested_at=now,
                    responded_at=now,
                    expires_at=now,
                )
            )
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=proposal.ops_task_id,
                event_type="proposal.manual_result_recorded",
                event_key=f"proposal:{proposal_id}:manual-result",
                visibility="USER",
                payload_json={
                    "proposal_id": str(proposal_id),
                    "status": str(command.status),
                    "result_artifact_id": str(artifact.artifact_id),
                    "trace_id": trace_id,
                },
            )
            if str(command.status) == "EXECUTED":
                outbox_payload = {
                    "proposal_id": str(proposal_id),
                    "source_run_id": str(run.ops_run_id),
                    "result_artifact_id": str(artifact.artifact_id),
                                        "domain_id": domain_id,
                    "actor_id": actor_id,
                    "agent_id": str(run.agent_id),
                    "target_id": str(run.target_id),
                    "trace_id": trace_id,
                }
                await uow.outbox.add(
                    OutboxEntity(
                        aggregate_type="OPS_CHANGE_PROPOSAL",
                        aggregate_id=proposal_id,
                        event_type=(
                            "OPS_ADVISORY_RESULT_RECORDED"
                        ),
                        idempotency_key=(
                            f"proposal:{proposal_id}:verify:v1"
                        ),
                        payload_json=outbox_payload,
                        payload_hash=_hash(outbox_payload),
                        status="PENDING",
                        available_at=now,
                        max_attempts=5,
                        trace_id=trace_id,
                    )
                )
            await uow.commit()
            return ManualResultReceipt(
                proposal_id=proposal_id,
                status=str(command.status),
                result_artifact=self._artifact_ref(artifact),
            )

    async def claim_execution(
        self,
        *,
        execution_id: UUID,
        command: MutationClaimRequest,
        trace_id: str,
    ) -> MutationClaimReceipt:
        """DB Executor 以唯一实例 Claim 并消费一次性审批授权。"""
        if (
            not self._mutation_enabled
            or self._mutation_grant_codec is None
            or self._action_registry is None
        ):
            raise state_conflict("Mutation Claim 部署开关尚未启用")
        async with self._uow_factory() as uow:
            preliminary = await uow.changes.get_execution(
                execution_id=execution_id
            )
            if preliminary is None:
                raise resource_not_found("Execution")
            run = await uow.runs.get_run(
                ops_run_id=preliminary.ops_run_id, lock=True
            )
            proposal = await uow.changes.get_proposal(
                proposal_id=preliminary.proposal_id, lock=True
            )
            token = await uow.changes.get_approval_token(
                approval_token_id=preliminary.approval_token_id,
                lock=True,
            )
            hitl = (
                await uow.changes.get_hitl(
                    hitl_id=token.hitl_id, lock=True
                )
                if token is not None
                else None
            )
            execution = await uow.changes.get_execution(
                execution_id=execution_id, lock=True
            )
            if any(
                item is None
                for item in (run, proposal, token, hitl, execution)
            ):
                raise state_conflict("Execution 授权链不完整")
            assert run is not None
            assert proposal is not None
            assert token is not None
            assert hitl is not None
            assert execution is not None
            if (
                UUID(str(execution.executor_request_id))
                != command.executor_request_id
            ):
                raise state_conflict("Executor Request ID 不匹配")
            if command.action_catalog_hash != self._action_registry.catalog_hash:
                raise state_conflict("Executor Action Catalog 版本不一致")
            now = await uow.runs.database_now()
            replay = execution.status == "SUBMITTED"
            if replay:
                if (
                    execution.executor_instance_id
                    != command.executor_instance_id
                    or execution.claimed_at is None
                    or execution.grant_jti_hash
                    != _hash(str(execution.execution_id))
                ):
                    raise state_conflict("Execution 已被其他实例 Claim")
            elif (
                execution.status != "CREATED"
                or token.status != "ISSUED"
                or proposal.status != "APPROVED"
                or hitl.status != "APPROVED"
            ):
                raise state_conflict("Execution 当前不能 Claim")
            if token.expires_at <= now:
                raise state_conflict("审批授权已过期")

            target_snapshot = (
                run.plan_snapshot_json or {}
            ).get("target", {})
            domain_id = int(target_snapshot.get("domain_id", -1))
            target = await uow.targets.get_scoped(
                target_id=execution.target_id,
                domain_id=domain_id,
                lock=True,
            )
            if (
                target is None
                or target.status != "ACTIVE"
                or not target.execution_credential_id
                or int(target.row_version) != int(token.target_version)
            ):
                raise state_conflict("Target 当前不满足 Claim 条件")
            conflicting = (
                await uow.changes.get_active_execution_for_target(
                    target_id=target.target_id,
                    exclude_execution_id=execution.execution_id,
                )
            )
            if conflicting is not None:
                raise state_conflict("Target 已存在进行中的 Mutation")
            binding = await uow.targets.get_agent_binding(
                target_id=target.target_id,
                agent_id=run.agent_id,
                domain_id=domain_id,
                lock=True,
            )
            if (
                binding is None
                or binding.status != "ACTIVE"
                or not bool(binding.allow_mutation)
                or proposal.action_template_id
                not in set(binding.allowed_actions_json or ())
            ):
                raise state_conflict("Agent Binding 已不允许该动作")
            policy = (
                await uow.policies.get_scoped(
                    policy_id=binding.policy_id,
                    domain_id=domain_id,
                    lock=True,
                )
                if binding.policy_id is not None
                else None
            )
            rules = dict(policy.rules_json) if policy is not None else {}
            frozen_policy = dict(run.policy_snapshot_json or {})
            if (
                policy is None
                or policy.status != "ACTIVE"
                or rules.get("allow_agent_execution") is not True
                or frozen_policy.get("policy_hash") != policy.policy_hash
                or token.policy_decision_hash
                != proposal.policy_decision_hash
            ):
                raise state_conflict("Policy 已不允许该 Execution")
            capabilities = {
                name
                for name, enabled in dict(
                    target.capabilities_json or {}
                ).items()
                if enabled is True
            }
            try:
                template = self._action_registry.resolve(
                    action_template_id=proposal.action_template_id,
                    version=proposal.action_template_version,
                    db_type=target.db_type,
                    db_version=target.version_code or "UNKNOWN",
                    capabilities=capabilities,
                    entitlements=set(rules.get("entitlements", ())),
                    environment=target.environment,
                )
                rendered = self._renderer.render(
                    template, dict(proposal.parameters_json)
                )
            except (LookupError, ValueError):
                raise state_conflict(
                    "Action Catalog 或参数当前不可执行"
                ) from None
            if (
                rendered.template_hash
                != execution.action_template_hash
                or rendered.parameters_hash != execution.parameters_hash
                or rendered.command_hash != execution.command_hash
                or execution.proposal_hash != proposal.proposal_hash
                or token.parameters_hash != execution.parameters_hash
            ):
                raise state_conflict("Execution Hash 围栏校验失败")

            issued_at = execution.claimed_at if replay else now
            assert issued_at is not None
            expires_at = min(
                token.expires_at,
                issued_at
                + timedelta(
                    seconds=self._mutation_grant_ttl_seconds
                ),
            )
            if expires_at <= now:
                raise state_conflict("Mutation Grant 已过期")
            grant = MutationExecutionGrant(
                issuer=self._mutation_grant_issuer,
                audience=self._mutation_grant_audience,
                grant_id=execution.execution_id,
                issued_at=issued_at,
                expires_at=expires_at,
                execution_id=execution.execution_id,
                executor_request_id=command.executor_request_id,
                executor_instance_id=command.executor_instance_id,
                target_id=target.target_id,
                domain_id=int(target.domain_id),
                target_version=int(target.row_version),
                db_type=target.db_type,
                connection_profile=dict(target.endpoint_json or {}),
                execution_credential_id=target.execution_credential_id,
                action_template_id=proposal.action_template_id,
                action_template_version=(
                    proposal.action_template_version
                ),
                action_template_variant=rendered.variant,
                renderer_version=rendered.renderer_version,
                typed_parameters=rendered.typed_parameters,
                action_template_hash=rendered.template_hash,
                parameters_hash=rendered.parameters_hash,
                command_hash=rendered.command_hash,
                proposal_hash=proposal.proposal_hash,
                policy_decision_hash=proposal.policy_decision_hash,
                approval_token_hash=token.token_hash,
                approver_id=token.approver_id,
                action_catalog_hash=self._action_registry.catalog_hash,
                statement_timeout_seconds=(
                    self._mutation_statement_timeout_seconds
                ),
                trace_id=run.trace_id,
            )
            encoded = self._mutation_grant_codec.issue(grant)
            if not replay:
                token.status = "CONSUMED"
                token.consumed_at = now
                proposal.status = "CONSUMED"
                proposal.updated_at = now
                execution.status = "SUBMITTED"
                execution.executor_instance_id = (
                    command.executor_instance_id
                )
                execution.claimed_at = now
                execution.deadline_at = now + timedelta(
                    seconds=(
                        self._mutation_statement_timeout_seconds + 60
                    )
                )
                execution.grant_jti_hash = _hash(
                    str(execution.execution_id)
                )
                execution.status_version = 2
                execution.updated_at = now
                await uow.runs.append_event(
                    ops_run_id=run.ops_run_id,
                    ops_task_id=proposal.ops_task_id,
                    event_type="execution.claimed",
                    event_key=(
                        f"execution:{execution.execution_id}:claimed"
                    ),
                    visibility="USER",
                    payload_json={
                        "execution_id": str(execution.execution_id),
                        "status": "SUBMITTED",
                        "trace_id": trace_id,
                    },
                )
                await uow.commit()
            return MutationClaimReceipt(
                execution_id=execution.execution_id,
                executor_request_id=command.executor_request_id,
                status="SUBMITTED",
                grant=encoded,
                expires_at=expires_at,
            )

    async def apply_execution_event(
        self,
        *,
        event: ExecutionStatusEvent,
        trace_id: str,
    ) -> EventReceipt:
        """以 Inbox 去重并按严格状态版本应用 Executor 回调。"""
        payload = event.model_dump(mode="json")
        message_key = str(event.event_id)
        async with self._uow_factory() as uow:
            duplicate = await uow.inbox.get_by_message(
                source_system="aiops-db-executor",
                message_key=message_key,
            )
            if duplicate is not None:
                return EventReceipt(
                    event_id=event.event_id,
                    accepted=duplicate.status == "PROCESSED",
                    duplicate=True,
                )
            now = await uow.runs.database_now()
            inbox = await uow.inbox.add(
                InboxEntity(
                    source_system="aiops-db-executor",
                    message_key=message_key,
                    message_type="EXECUTION_STATUS",
                    payload_json=payload,
                    payload_hash=_hash(payload),
                    status="RECEIVED",
                    received_at=now,
                )
            )
            preliminary = await uow.changes.get_execution(
                execution_id=event.execution_id
            )
            if preliminary is None:
                raise resource_not_found("Execution")
            run = await uow.runs.get_run(
                ops_run_id=preliminary.ops_run_id, lock=True
            )
            proposal = await uow.changes.get_proposal(
                proposal_id=preliminary.proposal_id, lock=True
            )
            token = await uow.changes.get_approval_token(
                approval_token_id=preliminary.approval_token_id,
                lock=True,
            )
            execution = await uow.changes.get_execution(
                execution_id=event.execution_id, lock=True
            )
            if any(
                item is None
                for item in (run, proposal, token, execution)
            ):
                raise state_conflict("Execution 回调授权链不完整")
            assert run is not None
            assert proposal is not None
            assert token is not None
            assert execution is not None
            if (
                UUID(str(execution.executor_request_id))
                != event.executor_request_id
                or execution.executor_instance_id
                != event.executor_instance_id
                or execution.grant_jti_hash != event.grant_jti_hash
            ):
                raise state_conflict("Executor 回调围栏不匹配")
            current_version = int(execution.status_version)
            if int(event.status_version) != current_version + 1:
                raise state_conflict("Executor 状态版本不连续")
            if event.occurred_at > now + timedelta(minutes=1):
                raise state_conflict("Executor 事件时间超出允许偏差")
            if (
                execution.status == "SUBMITTED"
                and event.status == "RUNNING"
                and event.status_version == 3
            ):
                if (
                    execution.claimed_at is None
                    or event.occurred_at < execution.claimed_at
                ):
                    raise state_conflict("Executor RUNNING 时间无效")
                execution.status = "RUNNING"
                execution.status_version = 3
                execution.started_at = event.occurred_at
                execution.updated_at = now
            elif (
                execution.status == "RUNNING"
                and event.status
                in {"SUCCEEDED", "FAILED", "UNKNOWN"}
                and event.status_version == 4
            ):
                if (
                    execution.started_at is None
                    or event.occurred_at < execution.started_at
                ):
                    raise state_conflict("Executor 终态时间无效")
                result_body = event.bounded_result or {}
                if _hash(result_body) != event.result_hash:
                    raise state_conflict("Executor 结果 Hash 不匹配")
                result = ExecutionResultArtifact(
                    execution_id=str(execution.execution_id),
                    proposal_id=str(proposal.proposal_id),
                    executor_request_id=str(
                        event.executor_request_id
                    ),
                    executor_instance_id=event.executor_instance_id,
                    status=event.status,
                    status_version=event.status_version,
                    occurred_at=event.occurred_at,
                    bounded_result=event.bounded_result,
                    result_hash=event.result_hash,
                    error_code=event.error_code,
                    proposal_hash=execution.proposal_hash,
                    command_hash=execution.command_hash,
                    grant_jti_hash=event.grant_jti_hash,
                )
                result_payload = result.model_dump(mode="json")
                artifact = await uow.runs.add_artifact(
                    OpsArtifactEntity(
                        ops_run_id=run.ops_run_id,
                        ops_task_id=proposal.ops_task_id,
                        artifact_key=(
                            f"execution:{execution.execution_id}:result:v1"
                        ),
                        artifact_type="EXECUTION_RESULT",
                        schema_version="EXECUTION_RESULT.v1",
                        payload_json=result_payload,
                        content_hash=_hash(result_payload),
                        byte_size=len(_canonical(result_payload)),
                        provenance_json={
                            "producer": "aiops-db-executor",
                            "producer_version": "mutation.v1",
                            "executor_instance_id": (
                                event.executor_instance_id
                            ),
                        },
                        trust_level="SOURCE_VERIFIED",
                        security_level=int(
                            (run.plan_snapshot_json or {})["target"][
                                "security_level"
                            ]
                        ),
                    )
                )
                execution.status = event.status
                execution.status_version = 4
                execution.result_artifact_id = artifact.artifact_id
                execution.result_hash = event.result_hash
                execution.completed_at = event.occurred_at
                execution.error_code = event.error_code
                execution.error_message = (
                    "数据库动作执行未成功"
                    if event.status != "SUCCEEDED"
                    else None
                )
                execution.updated_at = now
                if event.status in {"SUCCEEDED", "UNKNOWN"}:
                    verification_payload = {
                        "execution_id": str(execution.execution_id),
                        "proposal_id": str(proposal.proposal_id),
                        "source_run_id": str(run.ops_run_id),
                        "result_artifact_id": str(artifact.artifact_id),
                        "domain_id": int(
                            (run.plan_snapshot_json or {})["target"][
                                "domain_id"
                            ]
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
                            payload_hash=_hash(
                                verification_payload
                            ),
                            status="PENDING",
                            available_at=now,
                            max_attempts=5,
                            trace_id=trace_id,
                        )
                    )
            else:
                raise state_conflict("Executor 状态迁移无效")
            inbox.status = "PROCESSED"
            inbox.processed_at = now
            await uow.runs.append_event(
                ops_run_id=run.ops_run_id,
                ops_task_id=proposal.ops_task_id,
                event_type="execution.status",
                event_key=f"executor-event:{event.event_id}",
                visibility="USER",
                payload_json={
                    "execution_id": str(execution.execution_id),
                    "status": execution.status,
                    "status_version": int(execution.status_version),
                    "trace_id": trace_id,
                },
            )
            await uow.commit()
            return EventReceipt(
                event_id=event.event_id,
                accepted=True,
                duplicate=False,
            )

    @staticmethod
    async def _snapshot(uow, proposal):
        artifact = await uow.runs.get_artifact(
            artifact_id=proposal.snapshot_artifact_id
        )
        if artifact is None or artifact.schema_version != "PROPOSAL_OUTCOME.v1":
            raise state_conflict("Proposal Snapshot Artifact 不存在")
        outcome = ProposalOutcome.model_validate(artifact.payload_json)
        if outcome.proposal is None:
            raise state_conflict("Proposal Snapshot 内容无效")
        return outcome.proposal

    @staticmethod
    def _view(proposal, snapshot) -> ProposalView:
        return ProposalView(
            proposal_id=proposal.proposal_id,
            ops_run_id=proposal.ops_run_id,
            target_id=proposal.target_id,
            target_version=snapshot.target_version,
            mode=snapshot.mode,
            action_template_id=proposal.action_template_id,
            action_template_version=proposal.action_template_version,
            action_template_hash=proposal.action_template_hash,
            parameters=dict(proposal.parameters_json),
            parameter_fact_refs=dict(snapshot.parameter_fact_refs),
            command_preview=snapshot.rendered_command,
            command_hash=proposal.command_hash,
            impact=snapshot.impact,
            risk=proposal.risk_level,
            prerequisites=tuple(snapshot.preconditions),
            rollback_plan=snapshot.rollback_plan,
            verification_plan="；".join(snapshot.verification_plan),
            evidence_refs=tuple(snapshot.evidence_refs),
            proposal_hash=proposal.proposal_hash,
            status=proposal.status,
            expires_at=proposal.expires_at,
            row_version=int(proposal.row_version),
        )

    @staticmethod
    def _artifact_ref(artifact) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            schema_version=artifact.schema_version,
            content_hash=artifact.content_hash,
        )

    @staticmethod
    def _approval_receipt(execution, token) -> ApprovalReceipt:
        return ApprovalReceipt(
            proposal_id=execution.proposal_id,
            proposal_status="APPROVED",
            approval_token_id=token.approval_token_id,
            execution_id=execution.execution_id,
            execution_status="CREATED",
            authorization_expires_at=token.expires_at,
        )

    async def _replay_approval(
        self,
        *,
        proposal_id: UUID,
        idempotency_key: str,
        expected_proposal_hash: str,
        actor_id: str,
    ) -> ApprovalReceipt:
        async with self._uow_factory() as uow:
            execution = await uow.changes.get_execution_by_idempotency(
                idempotency_key=idempotency_key
            )
            if execution is None:
                execution = await uow.changes.get_execution_by_proposal(
                    proposal_id=proposal_id
                )
            if (
                execution is None
                or execution.proposal_id != proposal_id
                or execution.proposal_hash != expected_proposal_hash
            ):
                raise state_conflict("Proposal 已被其他审批请求处理")
            token = await uow.changes.get_approval_token(
                approval_token_id=execution.approval_token_id
            )
            if token is None or token.approver_id != actor_id:
                raise state_conflict("Proposal 已被其他审批人处理")
            return self._approval_receipt(execution, token)
