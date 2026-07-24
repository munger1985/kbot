"""Advisory Proposal 的读取、驳回和人工结果记录。"""

from __future__ import annotations

import hashlib
import json
from uuid import UUID

from aiops_agent.application.errors import (
    resource_not_found,
    state_conflict,
)
from aiops_agent.contracts.change import (
    AdvisoryActionResult,
    ProposalOutcome,
)
from aiops_agent.entities import (
    HitlEntity,
    OpsArtifactEntity,
    OutboxEntity,
)
from platform_core.contracts.aiops.public import (
    ManualResultCommand,
    ManualResultReceipt,
    ProposalView,
    RejectionCommand,
)
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
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def get_proposal(
        self,
        *,
        proposal_id: UUID,
        app_id: int,
        domain_id: int,
    ) -> ProposalView:
        async with self._uow_factory() as uow:
            proposal = await uow.changes.get_proposal_scoped(
                proposal_id=proposal_id,
                app_id=app_id,
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
        app_id: int,
        domain_id: int,
        actor_id: str,
        command: RejectionCommand,
        trace_id: str,
    ) -> ProposalView:
        async with self._uow_factory() as uow:
            preliminary = await uow.changes.get_proposal_scoped(
                proposal_id=proposal_id,
                app_id=app_id,
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
            snapshot = await self._snapshot(uow, proposal)
            await uow.commit()
            view = self._view(proposal, snapshot)
            return view.model_copy(
                update={"row_version": int(proposal.row_version) + 1}
            )

    async def record_manual_result(
        self,
        *,
        proposal_id: UUID,
        app_id: int,
        domain_id: int,
        actor_id: str,
        command: ManualResultCommand,
        idempotency_key: str,
        trace_id: str,
    ) -> ManualResultReceipt:
        async with self._uow_factory() as uow:
            preliminary = await uow.changes.get_proposal_scoped(
                proposal_id=proposal_id,
                app_id=app_id,
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
                    "app_id": app_id,
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
