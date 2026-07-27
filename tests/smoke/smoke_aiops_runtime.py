"""在配置的 Oracle Schema 中验收完整 Run 闭环。"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, func, select, text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiops_agent.application.runtime import AIOpsRuntimeService
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.config import get_aiops_settings
from aiops_agent.entities import (
    OpsArtifactEntity,
    OpsRunEntity,
    OpsRunEventEntity,
    OpsTaskEntity,
    OutboxEntity,
    TargetBindingEntity,
    TargetEntity,
)
from aiops_agent.orchestration import create_kernel_blueprint_registry
from aiops_agent.persistence import create_aiops_uow_factory
from aiops_agent.workers import (
    AIOpsTaskWorker,
    create_kernel_handler_registry,
)
from main_api.entities import PlatformDomainEntity
from platform_core.contracts.aiops import (
    ArtifactInput,
    ClaimOpsTaskCommand,
    CompleteOpsTaskCommand,
    CreateOpsRunCommand,
)
from platform_core.database.oracle import create_database_runtime
from platform_core.identity import uuid7


async def main() -> None:
    settings = get_aiops_settings()
    database = create_database_runtime(settings)
    uow_factory = create_aiops_uow_factory(database.session_factory)
    handlers = create_kernel_handler_registry()
    service = AIOpsRuntimeService(
        uow_factory=uow_factory,
        blueprint_registry=create_kernel_blueprint_registry(),
        handler_registry=handlers,
        max_tasks_per_run=settings.limits.max_tasks_per_run,
        default_run_timeout_seconds=settings.limits.run_timeout_seconds,
    )
    target_id = uuid7()
    agent_id = uuid7()
    run_ids = []
    created_domain_id = None
    try:
        async with database.session_factory() as session:
            domain = (
                await session.execute(
                    select(PlatformDomainEntity)
                    .where(
                        PlatformDomainEntity.app_id
                        == settings.platform.app_id,
                        PlatformDomainEntity.status == "ACTIVE",
                    )
                    .order_by(PlatformDomainEntity.domain_id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if domain is None:
                domain = PlatformDomainEntity(
                    app_id=settings.platform.app_id,
                    name=f"runtime-smoke-{target_id}",
                    status="ACTIVE",
                    created_by="runtime-smoke",
                    updated_by="runtime-smoke",
                )
                session.add(domain)
                await session.flush()
                created_domain_id = int(domain.domain_id)
            domain_id = int(domain.domain_id)
            await session.commit()
        async with uow_factory() as uow:
            await uow.targets.add_target(
                TargetEntity(
                    target_id=target_id,
                    app_id=settings.platform.app_id,
                    domain_id=domain_id,
                    target_key=f"runtime-smoke-{target_id}",
                    display_name="运行内核 Smoke Target",
                    db_type="ORACLE",
                    environment="DEV",
                    db_role="PRIMARY",
                    execution_mode="ADVISORY",
                    security_level=1,
                    status="ACTIVE",
                    health_status="UNKNOWN",
                    created_by="runtime-smoke",
                    updated_by="runtime-smoke",
                )
            )
            await uow.targets.add_binding(
                TargetBindingEntity(
                    target_id=target_id,
                    agent_id=agent_id,
                    access_mode="DIAGNOSE",
                    status="ACTIVE",
                    created_by="runtime-smoke",
                    updated_by="runtime-smoke",
                )
            )
            await uow.commit()
        async def create_run(case: str):
            return await service.create_run(
                CreateOpsRunCommand(
                    command_id=uuid7(),
                    idempotency_key=f"runtime-smoke-{case}-{target_id}",
                    app_id=settings.platform.app_id,
                    domain_id=domain_id,
                    actor_id="runtime-smoke",
                    agent_id=agent_id,
                    target_id=target_id,
                    trigger_type="API",
                    input=f"验证确定性运行内核：{case}",
                )
            )

        receipt = await create_run("happy")
        run_id = receipt.ops_run_id
        run_ids.append(run_id)
        worker = AIOpsTaskWorker(
            runtime_service=service,
            handler_registry=handlers,
            worker_id="runtime-smoke-worker",
            lease_seconds=120,
            heartbeat_seconds=30,
            poll_interval_seconds=0.1,
        )
        for index in range(3):
            if not await worker.run_once():
                async with database.session_factory() as session:
                    tasks = (
                        await session.execute(
                            select(OpsTaskEntity)
                            .where(OpsTaskEntity.ops_run_id == run_id)
                            .order_by(OpsTaskEntity.task_key)
                        )
                    ).scalars()
                    state = [
                        (
                            item.task_key,
                            item.status,
                            item.error_code,
                            item.error_message,
                        )
                        for item in tasks
                    ]
                    diagnostic = (
                        await session.execute(
                            text(
                                """
                                SELECT t.STATUS, t.AVAILABLE_AT,
                                       CURRENT_TIMESTAMP AS DB_NOW,
                                       t.ATTEMPT_COUNT, t.MAX_ATTEMPTS,
                                       r.STATUS AS RUN_STATUS,
                                       r.CANCEL_REQUESTED_AT,
                                       r.DEADLINE_AT
                                FROM KBOT_OPS_TASK t
                                JOIN KBOT_OPS_RUN r
                                  ON r.OPS_RUN_ID = t.OPS_RUN_ID
                                WHERE t.OPS_RUN_ID = :run_id
                                """
                            ),
                            {"run_id": run_id.bytes},
                        )
                    ).all()
                raise RuntimeError(
                    f"第 {index + 1} 项 Blueprint Task 未被领取："
                    f"{state}，数据库状态={diagnostic}"
                )
        summary = await service.get_run(
            ops_run_id=run_id,
            app_id=settings.platform.app_id,
            domain_id=domain_id,
        )
        if summary.status != "COMPLETED":
            raise RuntimeError(f"Run 未完成：{summary.status}")
        page = await service.list_events(
            ops_run_id=run_id,
            app_id=settings.platform.app_id,
            domain_id=domain_id,
            after_sequence=0,
            user_only=True,
        )

        cancel_receipt = await create_run("cancel")
        cancel_run_id = cancel_receipt.ops_run_id
        run_ids.append(cancel_run_id)
        leases = await asyncio.gather(
            service.claim_task(
                ClaimOpsTaskCommand(
                    worker_id="runtime-smoke-cancel-1",
                    lease_seconds=120,
                    trace_id=str(uuid7()),
                )
            ),
            service.claim_task(
                ClaimOpsTaskCommand(
                    worker_id="runtime-smoke-cancel-2",
                    lease_seconds=120,
                    trace_id=str(uuid7()),
                )
            ),
        )
        claimed = [item for item in leases if item is not None]
        if len(claimed) != 1 or claimed[0].run_id != cancel_run_id:
            raise RuntimeError("并发 Claim 未保持单一租约")
        cancel_summary = await service.get_run(
            ops_run_id=cancel_run_id,
            app_id=settings.platform.app_id,
            domain_id=domain_id,
        )
        await service.request_cancel(
            ops_run_id=cancel_run_id,
            app_id=settings.platform.app_id,
            domain_id=domain_id,
            actor_id="runtime-smoke",
            expected_row_version=cancel_summary.row_version,
            idempotency_key=f"cancel-{cancel_run_id}",
            trace_id=str(uuid7()),
        )
        async with database.session_factory() as session:
            task = await session.get(
                OpsTaskEntity, claimed[0].task_id
            )
            task.lease_until = datetime.now(UTC) - timedelta(seconds=1)
            await session.commit()
        await service.reconcile_once(trace_id=str(uuid7()))
        cancel_summary = await service.get_run(
            ops_run_id=cancel_run_id,
            app_id=settings.platform.app_id,
            domain_id=domain_id,
        )
        if cancel_summary.status != "CANCELLED":
            raise RuntimeError("取消中的 Run 未被 Reconciler 收敛")

        stale_receipt = await create_run("stale")
        stale_run_id = stale_receipt.ops_run_id
        run_ids.append(stale_run_id)
        old_lease = await service.claim_task(
            ClaimOpsTaskCommand(
                worker_id="runtime-smoke-old",
                lease_seconds=120,
                trace_id=str(uuid7()),
            )
        )
        if old_lease is None or old_lease.run_id != stale_run_id:
            raise RuntimeError("未领取 Stale Lease 测试 Task")
        async with database.session_factory() as session:
            task = await session.get(OpsTaskEntity, old_lease.task_id)
            task.lease_until = datetime.now(UTC) - timedelta(seconds=1)
            await session.commit()
        await service.reconcile_once(trace_id=str(uuid7()))
        async with database.session_factory() as session:
            task = await session.get(OpsTaskEntity, old_lease.task_id)
            task.available_at = datetime.now(UTC) - timedelta(seconds=1)
            await session.commit()
        await service.reconcile_once(trace_id=str(uuid7()))
        new_lease = await service.claim_task(
            ClaimOpsTaskCommand(
                worker_id="runtime-smoke-new",
                lease_seconds=120,
                trace_id=str(uuid7()),
            )
        )
        if new_lease is None or new_lease.task_id != old_lease.task_id:
            raise RuntimeError("过期租约未被重新领取")
        try:
            await service.complete_task(
                CompleteOpsTaskCommand(
                    task_id=old_lease.task_id,
                    worker_id="runtime-smoke-old",
                    lease_token=old_lease.lease_token,
                    idempotency_key="stale-result",
                    trace_id=old_lease.trace_id,
                    artifact=ArtifactInput(
                        artifact_type="SCOPE_RESULT",
                        schema_version="SCOPE_RESULT.v1",
                        producer="kernel.scope",
                        producer_version="1",
                        payload={"stale": True},
                    ),
                )
            )
        except AIOpsApplicationError as exc:
            if exc.code != "OPS_STALE_LEASE":
                raise
        else:
            raise RuntimeError("旧 Worker 结果未被 Lease Token 拒绝")
        takeover_worker = AIOpsTaskWorker(
            runtime_service=service,
            handler_registry=handlers,
            worker_id="runtime-smoke-new",
            lease_seconds=120,
            heartbeat_seconds=30,
            poll_interval_seconds=0.1,
        )
        await takeover_worker._execute(new_lease)
        for _ in range(2):
            if not await worker.run_once():
                raise RuntimeError("接管后的 Run 未继续执行")
        stale_summary = await service.get_run(
            ops_run_id=stale_run_id,
            app_id=settings.platform.app_id,
            domain_id=domain_id,
        )
        if stale_summary.status != "COMPLETED":
            raise RuntimeError("Lease 接管后的 Run 未完成")

        concurrent_command = CreateOpsRunCommand(
            command_id=uuid7(),
            idempotency_key=f"runtime-smoke-idempotent-{target_id}",
            app_id=settings.platform.app_id,
            domain_id=domain_id,
            actor_id="runtime-smoke",
            agent_id=agent_id,
            target_id=target_id,
            trigger_type="API",
            input="验证并发幂等创建",
        )
        concurrent_receipts = await asyncio.gather(
            service.create_run(concurrent_command),
            service.create_run(concurrent_command),
        )
        if (
            concurrent_receipts[0].ops_run_id
            != concurrent_receipts[1].ops_run_id
        ):
            raise RuntimeError("并发幂等创建生成了多个 Run")
        idempotent_run_id = concurrent_receipts[0].ops_run_id
        run_ids.append(idempotent_run_id)
        await service.request_cancel(
            ops_run_id=idempotent_run_id,
            app_id=settings.platform.app_id,
            domain_id=domain_id,
            actor_id="runtime-smoke",
            expected_row_version=concurrent_receipts[0].row_version,
            idempotency_key=f"cancel-{idempotent_run_id}",
            trace_id=str(uuid7()),
        )

        print(
            "AIOps 运行内核 Smoke 成功："
            f"happy_run={run_id} events={len(page.events)} "
            f"cancel_run={cancel_run_id} stale_run={stale_run_id} "
            f"idempotent_run={idempotent_run_id}"
        )
    finally:
        async with database.session_factory() as session:
            for run_id in run_ids:
                await session.execute(
                    delete(OutboxEntity).where(
                        OutboxEntity.aggregate_id == run_id
                    )
                )
                await session.execute(
                    delete(OpsRunEventEntity).where(
                        OpsRunEventEntity.ops_run_id == run_id
                    )
                )
                await session.execute(
                    delete(OpsArtifactEntity).where(
                        OpsArtifactEntity.ops_run_id == run_id
                    )
                )
                await session.execute(
                    delete(OpsTaskEntity).where(
                        OpsTaskEntity.ops_run_id == run_id
                    )
                )
                await session.execute(
                    delete(OpsRunEntity).where(
                        OpsRunEntity.ops_run_id == run_id
                    )
                )
            bindings = (
                await session.execute(
                    select(TargetBindingEntity).where(
                        TargetBindingEntity.target_id == target_id
                    )
                )
            ).scalars()
            for binding in bindings:
                await session.delete(binding)
            target = await session.get(TargetEntity, target_id)
            if target is not None:
                await session.delete(target)
            if created_domain_id is not None:
                domain = await session.get(
                    PlatformDomainEntity, created_domain_id
                )
                if domain is not None:
                    await session.delete(domain)
            await session.commit()
            remaining_runs = (
                await session.execute(
                    select(func.count())
                    .select_from(OpsRunEntity)
                    .where(OpsRunEntity.actor_id == "runtime-smoke")
                )
            ).scalar_one()
            remaining_targets = (
                await session.execute(
                    select(func.count())
                    .select_from(TargetEntity)
                    .where(TargetEntity.target_key.like("runtime-smoke-%"))
                )
            ).scalar_one()
            if int(remaining_runs) or int(remaining_targets):
                raise RuntimeError("AIOps Runtime Smoke 测试数据未清理干净")
        await database.close()


if __name__ == "__main__":
    asyncio.run(main())
