"""DB Executor 反向 Claim 内部接口。"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from datetime import UTC, datetime
import hashlib

from aiops_agent.api.changes.routes import get_service
from aiops_agent.api.dependencies import require_service_scope
from aiops_agent.application.changes import AIOpsChangeService
from platform_core.contracts.aiops.executor import (
    CredentialIssueRequest,
    CredentialIssueResponse,
    ExecutionStatusEvent,
    MutationClaimReceipt,
    MutationClaimRequest,
)
from aiops_agent.diagnostics.grants import DiagnosticGrantError
from aiops_agent.actions import MutationGrantError
from aiops_agent.entities import InboxEntity
from platform_core.identity import uuid7
from platform_core.contracts.aiops.internal import EventReceipt


router = APIRouter(
    prefix="/internal/v1/aiops/executions",
    tags=["AIOps Executions"],
)
event_router = APIRouter(
    prefix="/internal/v1/aiops",
    tags=["AIOps Executions"],
)
Service = Annotated[AIOpsChangeService, Depends(get_service)]


@router.post(
    "/{execution_id}/claim",
    response_model=MutationClaimReceipt,
)
async def claim_execution(
    execution_id: UUID,
    body: MutationClaimRequest,
    request: Request,
    service: Service,
) -> MutationClaimReceipt:
    require_service_scope(request, "aiops.execution.claim")
    context = request.state.auth_context
    return await service.claim_execution(
        execution_id=execution_id,
        command=body,
        trace_id=context.trace_id,
    )


@event_router.post(
    "/executor-events",
    response_model=EventReceipt,
)
async def apply_execution_event(
    body: ExecutionStatusEvent,
    request: Request,
    service: Service,
) -> EventReceipt:
    require_service_scope(request, "aiops.execution.callback")
    context = request.state.auth_context
    return await service.apply_execution_event(
        event=body,
        trace_id=context.trace_id,
    )


@event_router.post("/credentials:issue", response_model=CredentialIssueResponse)
async def issue_credential(body: CredentialIssueRequest, request: Request) -> CredentialIssueResponse:
    """仅向已验证的 Executor 发放且消费一次连接材料。"""
    require_service_scope(request, "aiops.credentials.issue")
    try:
        try:
            grant = request.app.state.diagnostic_grant_codec.verify(body.grant)
            kind, credential_id, target_version = "DIAGNOSTIC", grant.diagnostic_credential_id, grant.target_row_version
        except DiagnosticGrantError:
            try:
                grant = request.app.state.diagnostic_grant_codec.verify_dynamic(body.grant)
                kind, credential_id, target_version = "DIAGNOSTIC", grant.diagnostic_credential_id, grant.target_row_version
            except DiagnosticGrantError:
                grant = request.app.state.mutation_grant_codec.verify(body.grant)
                kind, credential_id, target_version = "EXECUTION", grant.execution_credential_id, grant.target_version
    except (DiagnosticGrantError, MutationGrantError) as exc:
        raise HTTPException(status_code=403, detail={"code": "CREDENTIAL_GRANT_INVALID"}) from exc
    if grant.audience != "kbot-aiops-db-executor" or grant.expires_at <= datetime.now(UTC):
        raise HTTPException(status_code=403, detail={"code": "CREDENTIAL_GRANT_INVALID"})
    grant_hash = hashlib.sha256(str(grant.grant_id).encode()).hexdigest()
    service = request.app.state.configuration_service
    async with service._uow_factory() as uow:
        assert uow.inbox is not None and uow.targets is not None
        assert uow.managed_credentials is not None
        existing = await uow.inbox.get_by_message(source_system="AIOPS_CREDENTIAL_ISSUE", message_key=grant_hash, lock=True)
        if existing is not None:
            raise HTTPException(status_code=403, detail={"code": "CREDENTIAL_GRANT_INVALID"})
        target = await uow.targets.get_scoped(target_id=grant.target_id, domain_id=int(grant.domain_id), lock=True)
        if target is None or int(target.row_version) != int(target_version):
            raise HTTPException(status_code=403, detail={"code": "CREDENTIAL_GRANT_INVALID"})
        current_id = target.diagnostic_credential_id if kind == "DIAGNOSTIC" else target.execution_credential_id
        if current_id != credential_id:
            raise HTTPException(status_code=403, detail={"code": "CREDENTIAL_GRANT_INVALID"})
        try:
            values = await request.app.state.managed_credential_service.read(
                uow=uow,
                domain_id=int(grant.domain_id),
                credential_id=credential_id,
                credential_kind=f"target_{kind.lower()}",
                external_key=grant.target_id,
                lock=True,
            )
            username = values["username"]
            password = values["password"]
            if not isinstance(username, str) or not isinstance(password, str):
                raise ValueError("数据库凭据字段缺失")
        except (KeyError, ValueError) as exc:
            raise PermissionError("凭据发放授权无效") from exc
        await uow.inbox.add(InboxEntity(inbox_id=uuid7(), source_system="AIOPS_CREDENTIAL_ISSUE", message_key=grant_hash, message_type="CREDENTIAL_ISSUED", payload_json={}, payload_hash=grant_hash, status="PROCESSED", processed_at=datetime.now(UTC)))
        await uow.commit()
    return CredentialIssueResponse(username=username, password=password)
