"""所有只读来源共享控制面配额，不以进程数量扩大数据库负载。"""

from datetime import timedelta
import hashlib
import secrets

from aiops_agent.entities.resource_governance import ResourceLeaseEntity
from platform_core.contracts.aiops.executor import ResourceLeaseReceipt


POOL_KEY = "READONLY"
TERMINAL_RUN_STATES = frozenset({"CANCELLED", "CANCELLING", "FAILED", "COMPLETED", "PARTIAL", "EXPIRED"})


class ResourceAdmissionError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def token_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


async def reserve_readonly_resource(*, uow, lease_id, run_id, target_id, source_kind, source_id, deadline_at):
    """调用方先验授权；锁池后计数与插入必须在同一个 UoW 中提交。"""
    repository = uow.resources
    pool = await repository.lock_pool(POOL_KEY)
    if pool is None:
        raise ResourceAdmissionError("RESOURCE_QUOTA_UNAVAILABLE")
    stored_limits = (pool.global_limit, pool.target_limit, pool.source_limit)
    try:
        limits = tuple(int(value) for value in stored_limits)
        grace = int(pool.cleanup_grace_seconds)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ResourceAdmissionError("RESOURCE_QUOTA_POLICY_INVALID") from exc
    if any(isinstance(value, bool) or value != normalized or not 1 <= normalized <= 1024 for value, normalized in zip(stored_limits, limits, strict=True)) or grace != pool.cleanup_grace_seconds or not 5 <= grace <= 300:
        raise ResourceAdmissionError("RESOURCE_QUOTA_POLICY_INVALID")
    now = await repository.database_now()
    if deadline_at <= now:
        raise ResourceAdmissionError("EXECUTION_DEADLINE_EXCEEDED")
    if await repository.get_lease(lease_id) is not None:
        raise ResourceAdmissionError("RESOURCE_LEASE_ALREADY_CONSUMED")
    if source_kind not in {"DATABASE", "MONITOR"}:
        raise ResourceAdmissionError("RESOURCE_SOURCE_SCOPE_INVALID")
    counts = await repository.active_counts(pool_key=POOL_KEY, target_id=target_id, source_kind=source_kind, source_id=source_id, now=now)
    for count, limit, dimension in zip(counts, limits, ("GLOBAL", "TARGET", "SOURCE"), strict=True):
        if count >= limit:
            raise ResourceAdmissionError(f"RESOURCE_QUOTA_{dimension}_EXHAUSTED")
    token = secrets.token_urlsafe(32)
    held_until = deadline_at + timedelta(seconds=grace)
    await repository.add_lease(ResourceLeaseEntity(
        lease_id=lease_id, pool_key=POOL_KEY, run_id=run_id, target_id=target_id,
        source_kind=source_kind, source_id=source_id, release_token_hash=token_hash(token), acquired_at=now,
        execution_deadline_at=deadline_at, held_until=held_until,
    ))
    return ResourceLeaseReceipt(lease_id=lease_id, release_token=token,
        execution_deadline_at=deadline_at, held_until=held_until)


class ResourceGovernanceService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def release(self, receipt):
        async with self._uow_factory() as uow:
            lease = await uow.resources.get_lease(receipt.lease_id, lock=True)
            if lease is None or not secrets.compare_digest(lease.release_token_hash, token_hash(receipt.release_token)):
                raise ResourceAdmissionError("RESOURCE_LEASE_RELEASE_INVALID")
            if lease.released_at is None:
                lease.released_at = await uow.resources.database_now()
            await uow.commit()

    async def acquire_monitor(self, *, context, source_id, lease_id):
        """运行与任务围栏有效时才允许监控/日志工作占用额度。"""
        async with self._uow_factory() as uow:
            run = await uow.runs.get_run(ops_run_id=context.run_id, lock=True)
            now = await uow.resources.database_now()
            if (run is None or str(run.target_id) != str(context.target_id)
                or run.cancel_requested_at is not None or str(run.status) in TERMINAL_RUN_STATES
                or run.deadline_at is None or run.deadline_at <= now):
                raise ResourceAdmissionError("RESOURCE_EXECUTION_FENCE_INVALID")
            task = await uow.runs.get_task(ops_task_id=context.task_id, lock=True)
            if (task is None or str(task.ops_run_id) != str(run.ops_run_id) or task.status != "RUNNING"
                or str(task.lease_token) != str(context.lease_token)
                or task.lease_until is None or task.lease_until <= now or task.started_at is None):
                raise ResourceAdmissionError("RESOURCE_EXECUTION_FENCE_INVALID")
            bindings = (run.plan_snapshot_json or {}).get("monitoring", {}).get("bindings", ())
            if not any(str(item.get("source", {}).get("source_id")) == str(source_id) for item in bindings):
                raise ResourceAdmissionError("RESOURCE_SOURCE_SCOPE_INVALID")
            deadline = min(run.deadline_at, task.lease_until, task.started_at + timedelta(seconds=int(task.timeout_seconds)))
            receipt = await reserve_readonly_resource(uow=uow, lease_id=lease_id, run_id=run.ops_run_id,
                target_id=run.target_id, source_kind="MONITOR", source_id=source_id, deadline_at=deadline)
            await uow.commit()
        return receipt
