"""执行中的只读取证持续核对持久化围栏，不重新消费 Grant 或读取秘密。"""

from datetime import UTC, datetime
import hashlib
import secrets

from aiops_agent.application.resource_governance import TERMINAL_RUN_STATES, token_hash
from aiops_agent.diagnostics.grants import DiagnosticGrantError
from aiops_agent.domain.operations.budget import task_deadline
from platform_core.contracts.aiops.executor import ReadonlyExecutionFenceReceipt


class ExecutionFenceError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


async def validate_diagnostic_fence(*, uow, grant, now):
    """统一凭据签发与执行中复核的授权语义，固定 Run→Task→Target 锁序。"""
    from aiops_agent.application.investigation.future_observation import FOLLOWUP_WORKFLOW, followup_access_error

    run = await uow.runs.get_run(ops_run_id=grant.run_id, lock=True)
    if (run is None or str(run.domain_id) != str(grant.domain_id)
        or str(run.target_id) != str(grant.target_id)):
        raise ExecutionFenceError("EXECUTION_AUTHORIZATION_REVOKED")
    if run.cancel_requested_at is not None or str(run.status) in TERMINAL_RUN_STATES:
        raise ExecutionFenceError("EXECUTION_CANCELLED")
    if run.deadline_at is None or run.deadline_at <= now:
        raise ExecutionFenceError("EXECUTION_DEADLINE_EXCEEDED")
    if run.workflow_kind == FOLLOWUP_WORKFLOW:
        if await followup_access_error(uow=uow, run=run, now=now):
            raise ExecutionFenceError("EXECUTION_AUTHORIZATION_REVOKED")
    task = await uow.runs.get_task(ops_task_id=grant.task_id, lock=True)
    if (task is None or str(task.ops_run_id) != str(run.ops_run_id)
        or str(task.status) != "RUNNING" or task.lease_token is None
        or task.lease_until is None or task.lease_until <= now or task.started_at is None
        or not secrets.compare_digest(hashlib.sha256(str(task.lease_token).encode()).hexdigest(), grant.lease_token_hash)):
        raise ExecutionFenceError("EXECUTION_LEASE_INVALID")
    if task_deadline(task) <= now:
        raise ExecutionFenceError("EXECUTION_DEADLINE_EXCEEDED")
    target = await uow.targets.get_scoped(target_id=grant.target_id, domain_id=int(grant.domain_id), lock=True)
    if (target is None or target.status != "ENABLED" or not target.readonly_connection_enabled
        or int(target.row_version) != int(grant.target_row_version)
        or target.diagnostic_credential_id != grant.diagnostic_credential_id):
        raise ExecutionFenceError("EXECUTION_AUTHORIZATION_REVOKED")
    if run.workflow_kind == "RESOURCE_COLLECTION":
        # 资源策略是周期采集的授权来源，不读取 Agent，也不接受动态诊断查询。
        snapshot = dict(run.plan_snapshot_json or {})
        kind = dict(snapshot.get("workload_collection") or {}).get("kind")
        policy = (
            target.workload_snapshot_policy_json if kind == "WORKLOAD"
            else target.activity_sampler_policy_json if kind == "ACTIVITY" else {}
        )
        tool = next((item for item in dict(snapshot.get("database_diagnostics") or {}).get("tools", ())
                     if item.get("tool_id") == getattr(grant, "tool_id", None)), None)
        if (run.trigger_type != "SCHEDULE" or run.interaction_mode != "AUTONOMOUS"
            or run.agent_id is not None or run.agent_version_id is not None
            or not dict(policy or {}).get("enabled") or tool is None
            or task.handler_id != "database.diagnostic"
            or task.task_key != f"diagnostic:{tool['tool_id']}"
            or grant.tool_version != tool["tool_version"]
            or grant.template_sha256 != tool["template_sha256"]):
            raise ExecutionFenceError("EXECUTION_AUTHORIZATION_REVOKED")
    return run, task, target


class ReadonlyExecutionFenceService:
    def __init__(self, *, uow_factory, diagnostic_codec, audience):
        self._uow_factory = uow_factory
        self._codec = diagnostic_codec
        self._audience = audience

    async def check(self, request):
        try:
            try:
                grant = self._codec.verify(request.grant)
            except DiagnosticGrantError:
                grant = self._codec.verify_dynamic(request.grant)
            if grant.audience != self._audience or grant.expires_at <= datetime.now(UTC):
                raise ExecutionFenceError("EXECUTION_GRANT_INVALID")
            async with self._uow_factory() as uow:
                run, task, _target = await validate_diagnostic_fence(uow=uow, grant=grant, now=datetime.now(UTC))
                target_id = grant.target_id
                lease = await uow.resources.get_lease(request.resource_lease.lease_id, lock=True)
                now = await uow.resources.database_now()
                receipt = request.resource_lease
                if (lease is None or str(lease.lease_id) != str(grant.grant_id)
                    or str(lease.run_id) != str(run.ops_run_id) or str(lease.target_id) != str(target_id)
                    or lease.source_kind != "DATABASE" or str(lease.source_id) != str(target_id)
                    or lease.released_at is not None
                    or not secrets.compare_digest(lease.release_token_hash, token_hash(receipt.release_token))
                    or lease.execution_deadline_at != receipt.execution_deadline_at or lease.held_until != receipt.held_until):
                    raise ExecutionFenceError("EXECUTION_LEASE_INVALID")
                deadline = min(grant.expires_at, run.deadline_at, task.lease_until,
                    task_deadline(task), lease.execution_deadline_at)
                if deadline <= now:
                    raise ExecutionFenceError("EXECUTION_DEADLINE_EXCEEDED")
                # 不提交、不延长租约、预算、凭据或 Grant；退出 UoW 释放短事务行锁。
                return ReadonlyExecutionFenceReceipt(allowed=True, checked_at=now, execution_deadline_at=deadline)
        except DiagnosticGrantError:
            code = "EXECUTION_GRANT_INVALID"
        except ExecutionFenceError as exc:
            code = exc.code
        return ReadonlyExecutionFenceReceipt(allowed=False, reason=code, checked_at=datetime.now(UTC))
