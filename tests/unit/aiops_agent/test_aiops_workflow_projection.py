"""工作流对账的来源身份、终态校验和服务边界回归。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiops_agent.api.notification_projection import router
from aiops_agent.api.dependencies import get_aiops_auth_context
from aiops_agent.application.notification_projection import WorkflowProjectionService
from main_api.application.aiops_workflow_reconciliation import AIOpsWorkflowReconciliation
from platform_core.contracts import PrincipalKind
from platform_core.contracts.aiops.notification_projection import WorkflowProjectionRequest, WorkflowProjectionSnapshot
from platform_core.identity import uuid7
from platform_core.notifications.aiops import presentation_severity, work_item_identity, work_item_source_key


@pytest.mark.parametrize("severity,urgency", [("INFO", "INFO"), ("WARNING", "WARNING"), ("HIGH", "CRITICAL"), ("CRITICAL", "CRITICAL")])
def test_original_severity_preserved_and_critical_is_not_downgraded(severity, urgency):
    facts = {"severity": severity}
    assert presentation_severity(facts, "WARNING") == urgency
    assert facts["severity"] == severity


def test_distinct_hitls_have_distinct_stable_keys():
    run, first, second = map(str, (uuid7(), uuid7(), uuid7()))
    first_key = work_item_source_key(resource_type="aiops_run", resource_id=run, facts={"hitl_id": first})
    assert first_key != work_item_source_key(resource_type="aiops_run", resource_id=run, facts={"hitl_id": second})
    assert work_item_identity(resource_type="aiops_run", resource_id=run, facts={"hitl_id": first}) == ("aiops_hitl", first)
@pytest.mark.parametrize("status,proposal_status,event_type", [
    ("CANCELLED", "SUPERSEDED", "aiops.proposal.invalidated"),
    ("EXPIRED", "EXPIRED", "aiops.proposal.expired"),
])
def test_run_termination_invalidates_pending_proposal_and_emits_atomic_closure(status, proposal_status, event_type):
    from datetime import datetime, timezone
    from aiops_agent.application.runtime.service import AIOpsRuntimeService
    async def run():
        source = SimpleNamespace(ops_run_id=uuid7(), actor_id="applicant")
        proposal = SimpleNamespace(proposal_id=uuid7(), ops_run_id=source.ops_run_id, status="PENDING_APPROVAL")
        hitl = SimpleNamespace(hitl_id=uuid7(), proposal_id=proposal.proposal_id, request_type="CHANGE_APPROVAL")
        notify = AsyncMock()
        uow = SimpleNamespace(changes=SimpleNamespace(list_pending_hitls=AsyncMock(return_value=[hitl]),
            get_proposal=AsyncMock(return_value=proposal)), platform_notifications=SimpleNamespace(emit_proposal_event=notify))
        await AIOpsRuntimeService._close_pending_hitls(uow=uow, run=source, now=datetime.now(timezone.utc), status=status)
        assert hitl.status == status and proposal.status == proposal_status
        assert notify.await_args.kwargs["event_type"] == event_type
    asyncio.run(run())

def test_report_or_terminal_run_does_not_replace_pending_interaction_state():
    async def run():
        request = WorkflowProjectionRequest(resource_type="aiops_run", resource_id=uuid7(), domain_id="73")
        entity = SimpleNamespace(ops_run_id=request.resource_id, status="COMPLETED", row_version=4)
        pending = []
        uow = SimpleNamespace(runs=SimpleNamespace(get_run_scoped=AsyncMock(return_value=entity)),
            changes=SimpleNamespace(list_pending_hitls=AsyncMock(side_effect=lambda **kwargs: pending)))
        context = AsyncMock(); context.__aenter__.return_value = uow
        service = WorkflowProjectionService(uow_factory=lambda: context, tenant_scoped=False)
        pending.append(SimpleNamespace(request_type="MANUAL_DIAGNOSTIC_SQL"))
        assert (await service.read(request)).work_item_status is None
        pending.clear()
        assert (await service.read(request)).work_item_status == "COMPLETED"
    asyncio.run(run())



@pytest.mark.parametrize("kind,status,resolution", [
    ("aiops_hitl", "PENDING", None), ("aiops_hitl", "ANSWERED", "COMPLETED"),
    ("aiops_hitl", "REJECTED", "COMPLETED"), ("aiops_hitl", "EXPIRED", "CANCELLED"),
    ("aiops_hitl", "CANCELLED", "CANCELLED"),
    ("aiops_proposal", "PENDING_APPROVAL", None), ("aiops_proposal", "APPROVED", "COMPLETED"),
    ("aiops_proposal", "REJECTED", "COMPLETED"), ("aiops_proposal", "EXPIRED", "CANCELLED"),
])
@pytest.mark.parametrize("tenant_scoped", [True, False])
def test_scoped_authority_maps_only_actual_resource_state(kind, status, resolution, tenant_scoped):
    async def run():
        request = WorkflowProjectionRequest(resource_type=kind, resource_id=uuid7(),
            domain_id=str(uuid7()) if tenant_scoped else "73",
            tenant_id=uuid7() if tenant_scoped else None)
        ops_run = SimpleNamespace(ops_run_id=uuid7(), tenant_id=request.tenant_id)
        entity = SimpleNamespace(ops_run_id=ops_run.ops_run_id, status=status, row_version=4)
        uow = SimpleNamespace(changes=SimpleNamespace(get_hitl_scoped=AsyncMock(return_value=entity),
            get_proposal_scoped=AsyncMock(return_value=entity)),
            runs=SimpleNamespace(get_run_scoped=AsyncMock(return_value=ops_run)))
        context = AsyncMock(); context.__aenter__.return_value = uow
        service = WorkflowProjectionService(uow_factory=lambda: context, tenant_scoped=tenant_scoped)
        result = await service.read(request)
        assert result.request == request and result.work_item_status == resolution and result.source_version == 4
        assert set(result.model_dump()) == {"request", "source_status", "source_version", "work_item_status"}
        if tenant_scoped:
            ops_run.tenant_id = uuid7()
            with pytest.raises(Exception):
                await service.read(request)
    asyncio.run(run())


def test_unknown_or_contradictory_authoritative_status_is_rejected():
    request = WorkflowProjectionRequest(resource_type="aiops_hitl", resource_id=uuid7(), domain_id="73")
    for status, resolution in [("UNKNOWN_FUTURE", "COMPLETED"), ("PENDING", "COMPLETED"), ("EXPIRED", "COMPLETED")]:
        with pytest.raises(ValueError):
            WorkflowProjectionSnapshot(request=request, source_status=status, source_version=1, work_item_status=resolution)


def test_reconciliation_rejects_scope_substitution_and_uses_service_identity():
    async def run():
        request = WorkflowProjectionRequest(resource_type="aiops_hitl", resource_id=uuid7(), domain_id="73")
        other = request.model_copy(update={"domain_id": "74"})
        client = SimpleNamespace(get_workflow_projection=AsyncMock(return_value=WorkflowProjectionSnapshot(
            request=other, source_status="ANSWERED", source_version=1, work_item_status="COMPLETED").model_dump(mode="json")))
        service = AIOpsWorkflowReconciliation(client=client, caller_service="notification-test")
        with pytest.raises(ValueError):
            await service.resolve(request)
        assert client.get_workflow_projection.await_args.kwargs["auth_context"].principal_kind == PrincipalKind.SERVICE
    asyncio.run(run())


def test_projection_endpoint_rejects_human_even_with_reconciliation_scope(monkeypatch):
    app = FastAPI(); app.include_router(router)
    app.state.workflow_projection_service = SimpleNamespace(read=AsyncMock())
    app.dependency_overrides[get_aiops_auth_context] = lambda: SimpleNamespace(principal_kind=PrincipalKind.PORTAL)
    monkeypatch.setattr("aiops_agent.api.notification_projection.require_service_scope", lambda *args: None)
    with TestClient(app) as client:
        response = client.post("/internal/v1/aiops/notifications:projection", json={
            "resource_type": "aiops_hitl", "resource_id": str(uuid7()), "domain_id": "73"})
    assert response.status_code == 403
    app.state.workflow_projection_service.read.assert_not_awaited()

@pytest.mark.parametrize("action", ["approve", "reject"])
@pytest.mark.parametrize("applicant_field", ["run", "hitl"])
def test_applicant_cannot_approve_or_reject_own_pending_proposal(action, applicant_field):
    from datetime import datetime, timedelta, timezone
    from aiops_agent.application.changes import AIOpsChangeService
    from aiops_agent.application.errors import AIOpsApplicationError
    from platform_core.contracts.aiops.public import ApprovalCommand, RejectionCommand
    async def run():
        actor, other, proposal_id = uuid7(), uuid7(), uuid7()
        now = datetime.now(timezone.utc)
        proposal = SimpleNamespace(proposal_id=proposal_id, ops_run_id=uuid7(), ops_task_id=uuid7(),
            row_version=1, proposal_hash="a" * 64, status="PENDING_APPROVAL",
            execution_mode="EXECUTABLE_AFTER_APPROVAL", effect_class="SESSION_TERMINATION",
            executor_kind="DATABASE", expires_at=now + timedelta(minutes=5))
        hitl = SimpleNamespace(proposal_id=proposal_id, assignee_user_id=None,
            requested_by=actor if applicant_field == "hitl" else other)
        changes = SimpleNamespace(get_execution_by_idempotency=AsyncMock(return_value=None),
            get_proposal_scoped=AsyncMock(return_value=proposal), get_proposal=AsyncMock(return_value=proposal),
            get_pending_hitl=AsyncMock(return_value=hitl))
        uow = SimpleNamespace(changes=changes, commit=AsyncMock(),
            runs=SimpleNamespace(get_run=AsyncMock(return_value=SimpleNamespace(
                actor_id=actor if applicant_field == "run" else other)), database_now=AsyncMock(return_value=now)))
        context = AsyncMock(); context.__aenter__.return_value = uow
        service = AIOpsChangeService(uow_factory=lambda: context, action_registry=object(),
            approval_enabled=True, mutation_enabled=True)
        with pytest.raises(AIOpsApplicationError, match="申请人与审批人必须为不同用户"):
            if action == "approve":
                await service.approve_proposal(proposal_id=proposal_id, domain_id=73, actor_id=actor,
                    command=ApprovalCommand(expected_row_version=1, expected_proposal_hash=proposal.proposal_hash),
                    idempotency_key="approval-test", trace_id="notification-test")
            else:
                await service.reject_proposal(proposal_id=proposal_id, domain_id=73, actor_id=actor,
                    command=RejectionCommand(expected_row_version=1, reason="拒绝原因"), trace_id="notification-test")
        uow.commit.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("case", ["wrong_actor", "wrong_domain", "valid"])
def test_approval_replay_requires_original_approver_and_current_resource_scope(case):
    from aiops_agent.application.changes import AIOpsChangeService
    from aiops_agent.application.errors import AIOpsApplicationError
    from platform_core.contracts.aiops.public import ApprovalCommand
    async def run():
        actor, proposal_id = uuid7(), uuid7()
        replay = SimpleNamespace(proposal_id=proposal_id, proposal_hash="a" * 64, approval_token_id=uuid7())
        changes = SimpleNamespace(get_execution_by_idempotency=AsyncMock(return_value=replay),
            get_approval_token=AsyncMock(return_value=SimpleNamespace(approver_id=uuid7() if case == "wrong_actor" else actor)),
            get_proposal_scoped=AsyncMock(return_value=None if case == "wrong_domain" else object()))
        context = AsyncMock(); context.__aenter__.return_value = SimpleNamespace(changes=changes)
        service = AIOpsChangeService(uow_factory=lambda: context, action_registry=object(),
            approval_enabled=True, mutation_enabled=True)
        receipt = object()
        service._approval_receipt = lambda *args: receipt
        call = service.approve_proposal(proposal_id=proposal_id, domain_id=73, actor_id=actor,
            command=ApprovalCommand(expected_row_version=1, expected_proposal_hash=replay.proposal_hash),
            idempotency_key="approval-test", trace_id="notification-test")
        if case == "valid":
            assert await call is receipt
        else:
            with pytest.raises(AIOpsApplicationError) as caught:
                await call
            assert caught.value.status_code == 404
    asyncio.run(run())
