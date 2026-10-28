"""签名凭据发放、当前授权及只读资源准入在同一个事务内完成。"""

from datetime import UTC, datetime
import hashlib
import secrets

from aiops_agent.diagnostics.grants import DiagnosticGrantError
from aiops_agent.actions import MutationGrantError
from aiops_agent.entities import InboxEntity
from aiops_agent.application.resource_governance import reserve_readonly_resource
from aiops_agent.domain.operations.budget import task_deadline
from platform_core.identity import uuid7
from platform_core.contracts.aiops.executor import CredentialIssueResponse


class CredentialIssueError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class CredentialIssuanceService:
    def __init__(self, *, uow_factory, diagnostic_codec, mutation_codec, credential_service, audience):
        self._uow_factory = uow_factory
        self._diagnostic_codec = diagnostic_codec
        self._mutation_codec = mutation_codec
        self._credentials = credential_service
        self._audience = audience

    async def issue(self, token):
        try:
            try:
                grant = self._diagnostic_codec.verify(token)
                kind, credential_id, target_version = "DIAGNOSTIC", grant.diagnostic_credential_id, grant.target_row_version
            except DiagnosticGrantError:
                try:
                    grant = self._diagnostic_codec.verify_dynamic(token)
                    kind, credential_id, target_version = "DIAGNOSTIC", grant.diagnostic_credential_id, grant.target_row_version
                except DiagnosticGrantError:
                    grant = self._mutation_codec.verify(token)
                    kind, credential_id, target_version = "EXECUTION", grant.execution_credential_id, grant.target_version
        except (DiagnosticGrantError, MutationGrantError) as exc:
            raise CredentialIssueError("CREDENTIAL_GRANT_INVALID") from exc
        if grant.audience != self._audience or grant.expires_at <= datetime.now(UTC):
            raise CredentialIssueError("CREDENTIAL_GRANT_INVALID")
        grant_hash = hashlib.sha256(str(grant.grant_id).encode()).hexdigest()
        resource_lease = None
        async with self._uow_factory() as uow:
            assert uow.inbox is not None and uow.targets is not None
            assert uow.managed_credentials is not None
            if kind == "DIAGNOSTIC":
                from aiops_agent.application.investigation.future_observation import FOLLOWUP_WORKFLOW, followup_access_error
                run = await uow.runs.get_run(ops_run_id=grant.run_id, lock=True)
                if (run is None or str(run.domain_id) != str(grant.domain_id)
                    or str(run.target_id) != str(grant.target_id)
                    or run.cancel_requested_at is not None
                    or (run.deadline_at is not None and run.deadline_at <= datetime.now(UTC))
                    or str(run.status) in {"CANCELLED", "CANCELLING", "FAILED", "COMPLETED", "PARTIAL", "EXPIRED"}):
                    raise CredentialIssueError("CREDENTIAL_GRANT_INVALID")
                if run.workflow_kind == FOLLOWUP_WORKFLOW:
                    if await followup_access_error(uow=uow, run=run, now=datetime.now(UTC)):
                        raise CredentialIssueError("CREDENTIAL_GRANT_INVALID")
                task = await uow.runs.get_task(ops_task_id=grant.task_id, lock=True)
                if (task is None or str(task.ops_run_id) != str(run.ops_run_id)
                    or str(task.status) != "RUNNING" or task.lease_token is None
                    or task.lease_until is None or task.lease_until <= datetime.now(UTC)
                    or task.started_at is None
                    or not secrets.compare_digest(hashlib.sha256(str(task.lease_token).encode()).hexdigest(), grant.lease_token_hash)):
                    raise CredentialIssueError("CREDENTIAL_GRANT_INVALID")
            target = await uow.targets.get_scoped(target_id=grant.target_id, domain_id=int(grant.domain_id), lock=True)
            if target is None or int(target.row_version) != int(target_version):
                raise CredentialIssueError("CREDENTIAL_GRANT_INVALID")
            current_id = target.diagnostic_credential_id if kind == "DIAGNOSTIC" else target.execution_credential_id
            if current_id != credential_id:
                raise CredentialIssueError("CREDENTIAL_GRANT_INVALID")
            existing = await uow.inbox.get_by_message(source_system="AIOPS_CREDENTIAL_ISSUE", message_key=grant_hash, lock=True)
            if existing is not None:
                raise CredentialIssueError("CREDENTIAL_GRANT_INVALID")
            if kind == "DIAGNOSTIC":
                resource_lease = await reserve_readonly_resource(
                    uow=uow, lease_id=grant.grant_id, run_id=grant.run_id, target_id=grant.target_id,
                    source_kind="DATABASE", source_id=grant.target_id,
                    deadline_at=min(grant.expires_at, run.deadline_at or grant.expires_at,
                        task.lease_until, task_deadline(task)),
                )
            if grant.expires_at <= datetime.now(UTC):
                raise CredentialIssueError("CREDENTIAL_GRANT_INVALID")
            try:
                values = await self._credentials.read(
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
        return CredentialIssueResponse(username=username, password=password, resource_lease=resource_lease)
