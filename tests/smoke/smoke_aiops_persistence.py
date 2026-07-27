"""在开发 Oracle 上验收 AIOps UoW、租约和事件序列。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import secrets
import sys

from sqlalchemy import delete, select, text

# 支持从仓库根目录直接执行本脚本。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiops_agent.entities import (
    InboxEntity,
    OpsRunEntity,
    OpsRunEventEntity,
    OpsTaskEntity,
    OutboxEntity,
    TargetEntity,
)
from aiops_agent.persistence import create_aiops_uow_factory
from platform_core.config import get_settings
from platform_core.database.oracle import create_database_runtime
from platform_core.identity import uuid7


async def _cleanup(runtime, *, domain_id: int) -> None:
    """只清理本次 Smoke 使用的隔离 Domain。"""
    async with runtime.session_factory() as session:
        target_ids = select(TargetEntity.target_id).where(
            TargetEntity.domain_id == domain_id
        )
        run_ids = select(OpsRunEntity.ops_run_id).where(
            OpsRunEntity.target_id.in_(target_ids)
        )
        await session.execute(
            delete(OpsRunEventEntity).where(
                OpsRunEventEntity.ops_run_id.in_(run_ids)
            )
        )
        await session.execute(
            delete(OpsTaskEntity).where(OpsTaskEntity.ops_run_id.in_(run_ids))
        )
        await session.execute(
            delete(OutboxEntity).where(OutboxEntity.aggregate_id.in_(run_ids))
        )
        await session.execute(
            delete(OpsRunEntity).where(OpsRunEntity.ops_run_id.in_(run_ids))
        )
        await session.execute(
            delete(TargetEntity).where(TargetEntity.domain_id == domain_id)
        )
        await session.execute(
            text(
                "DELETE FROM KBOT_PLATFORM_DOMAIN "
                "WHERE DOMAIN_ID = :domain_id"
            ),
            {"domain_id": domain_id},
        )
        await session.commit()


async def smoke() -> None:
    runtime = create_database_runtime(get_settings())
    uow_factory = create_aiops_uow_factory(runtime.session_factory)
    domain_id = secrets.randbelow(10**18) + 10**18
    target_id = uuid7()
    run_id = uuid7()
    second_run_id = uuid7()
    task_id = uuid7()
    second_task_id = uuid7()
    outbox_id = uuid7()
    second_outbox_id = uuid7()
    rollback_inbox_id = uuid7()
    lease_owner = f"smoke-{uuid7()}"
    task_lease_token = uuid7()
    outbox_lease_token = uuid7()
    now = datetime.now(UTC)

    try:
        async with runtime.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO KBOT_PLATFORM_DOMAIN (
                        DOMAIN_ID, APP_ID, NAME, STATUS, CREATED_BY, UPDATED_BY
                    ) VALUES (
                        :domain_id, 1, :name, 'ACTIVE', 'smoke', 'smoke'
                    )
                    """
                ),
                {
                    "domain_id": domain_id,
                    "name": f"aiops-smoke-{domain_id}",
                },
            )
            await session.commit()

        async with uow_factory() as uow:
            await uow.targets.add_target(
                TargetEntity(
                    target_id=target_id,
                    app_id=1,
                    domain_id=domain_id,
                    target_key="smoke-target",
                    display_name="AIOps 持久化 Smoke",
                    db_type="ORACLE",
                    environment="DEV",
                    created_by="smoke",
                    updated_by="smoke",
                )
            )
            await uow.runs.add_run(
                OpsRunEntity(
                    ops_run_id=run_id,
                    target_id=target_id,
                    agent_id=uuid7(),
                    trigger_type="API",
                    actor_id="smoke",
                    idempotency_key=f"run-{run_id}",
                    status="CREATED",
                    trace_id=f"smoke-{run_id}",
                )
            )
            await uow.runs.add_run(
                OpsRunEntity(
                    ops_run_id=second_run_id,
                    target_id=target_id,
                    agent_id=uuid7(),
                    trigger_type="API",
                    actor_id="smoke",
                    idempotency_key=f"run-{second_run_id}",
                    status="CREATED",
                    trace_id=f"smoke-{second_run_id}",
                )
            )
            await uow.runs.add_task(
                OpsTaskEntity(
                    ops_task_id=task_id,
                    ops_run_id=run_id,
                    task_key="smoke-task",
                    task_type="SCOPE",
                    handler_id="smoke.handler",
                    handler_version="1",
                    input_schema_version="1",
                    output_schema_version="1",
                    status="READY",
                    priority=0,
                    available_at=now,
                    timeout_seconds=60,
                )
            )
            await uow.runs.add_task(
                OpsTaskEntity(
                    ops_task_id=second_task_id,
                    ops_run_id=second_run_id,
                    task_key="smoke-task-2",
                    task_type="SCOPE",
                    handler_id="smoke.handler",
                    handler_version="1",
                    input_schema_version="1",
                    output_schema_version="1",
                    status="READY",
                    priority=1,
                    available_at=now,
                    timeout_seconds=60,
                )
            )
            await uow.outbox.add(
                OutboxEntity(
                    outbox_id=outbox_id,
                    aggregate_type="OPS_RUN",
                    aggregate_id=run_id,
                    event_type="smoke.created",
                    idempotency_key=f"outbox-{run_id}",
                    payload_json={"ops_run_id": str(run_id)},
                    payload_hash="0" * 64,
                    available_at=now,
                    trace_id=f"smoke-{run_id}",
                )
            )
            await uow.outbox.add(
                OutboxEntity(
                    outbox_id=second_outbox_id,
                    aggregate_type="OPS_RUN",
                    aggregate_id=second_run_id,
                    event_type="smoke.created.2",
                    idempotency_key=f"outbox-{second_run_id}",
                    payload_json={"ops_run_id": str(second_run_id)},
                    payload_hash="0" * 64,
                    available_at=now,
                    trace_id=f"smoke-{second_run_id}",
                )
            )
            await uow.commit()

        async with runtime.session_factory() as session:
            ready_count = (
                await session.execute(
                    select(OpsTaskEntity).where(
                        OpsTaskEntity.ops_run_id.in_(
                            (run_id, second_run_id)
                        ),
                        OpsTaskEntity.status == "READY",
                        OpsTaskEntity.available_at <= now,
                    )
                )
            ).scalars().all()
            assert len(ready_count) == 2, len(ready_count)

        async with uow_factory() as uow:
            claimed_task = await uow.runs.claim_task(
                now=now,
                lease_owner=lease_owner,
                lease_token=task_lease_token,
                lease_until=now + timedelta(minutes=1),
            )
            assert claimed_task is not None
            claimed_task_version = int(claimed_task.row_version)
            claimed_outbox = await uow.outbox.claim(
                now=now,
                lease_owner=lease_owner,
                lease_token=outbox_lease_token,
                lease_until=now + timedelta(minutes=1),
            )
            assert claimed_outbox is not None

            async with uow_factory() as second_uow:
                second_claimed_task = await second_uow.runs.claim_task(
                    now=now,
                    lease_owner=f"{lease_owner}-2",
                    lease_token=uuid7(),
                    lease_until=now + timedelta(minutes=1),
                )
                assert second_claimed_task is not None
                assert second_claimed_task.ops_task_id != claimed_task.ops_task_id
                assert {
                    claimed_task.ops_run_id,
                    second_claimed_task.ops_run_id,
                } == {run_id, second_run_id}
                second_claimed_outbox = await second_uow.outbox.claim(
                    now=now,
                    lease_owner=f"{lease_owner}-2",
                    lease_token=uuid7(),
                    lease_until=now + timedelta(minutes=1),
                )
                assert second_claimed_outbox is not None
                assert (
                    second_claimed_outbox.outbox_id
                    != claimed_outbox.outbox_id
                )
                await second_uow.commit()

            first = await uow.runs.append_event(
                ops_run_id=run_id,
                event_type="task.claimed",
                visibility="INTERNAL",
                payload_json={"task_id": str(claimed_task.ops_task_id)},
                ops_task_id=claimed_task.ops_task_id,
            )
            second = await uow.runs.append_event(
                ops_run_id=run_id,
                event_type="progress",
                visibility="USER",
                payload_json={"message": "Smoke 处理中"},
            )
            assert (int(first.sequence_no), int(second.sequence_no)) == (1, 2)
            await uow.commit()

        async with uow_factory() as uow:
            stale_finish = await uow.runs.finish_task(
                ops_task_id=claimed_task.ops_task_id,
                lease_owner=lease_owner,
                lease_token=uuid7(),
                now=now,
                expected_version=claimed_task_version,
                new_status="SUCCEEDED",
            )
            assert stale_finish is False
            finished = await uow.runs.finish_task(
                ops_task_id=claimed_task.ops_task_id,
                lease_owner=lease_owner,
                lease_token=task_lease_token,
                now=now,
                expected_version=claimed_task_version,
                new_status="SUCCEEDED",
            )
            assert finished is True
            published = await uow.outbox.mark_published(
                outbox_id=claimed_outbox.outbox_id,
                lease_owner=lease_owner,
                lease_token=outbox_lease_token,
                now=now,
            )
            assert published is True
            await uow.commit()

        async with uow_factory() as uow:
            await uow.inbox.add(
                InboxEntity(
                    inbox_id=rollback_inbox_id,
                    source_system="smoke",
                    message_key=f"rollback-{rollback_inbox_id}",
                    message_type="smoke",
                    payload_json={"rollback": True},
                    payload_hash="0" * 64,
                )
            )

        async with runtime.session_factory() as session:
            rolled_back = (
                await session.execute(
                    select(InboxEntity).where(
                        InboxEntity.inbox_id == rollback_inbox_id
                    )
                )
            ).scalar_one_or_none()
            assert rolled_back is None

        print(
            "AIOps Persistence Smoke 通过："
            "显式提交、自动回滚、双 Worker SKIP LOCKED、"
            "Task/Outbox 租约、陈旧令牌拒绝、"
            "RunEvent 序列均正常"
        )
    finally:
        await _cleanup(runtime, domain_id=domain_id)
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(smoke())
