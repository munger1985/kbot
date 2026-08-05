"""真实 Oracle 的 S6 事务 Outbox、投影、隔离和续传 Smoke。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import socket
from types import SimpleNamespace
from uuid import UUID

import aiohttp
from fastapi import FastAPI, Request
from sqlalchemy import delete, select, text
import uvicorn

from agent_runtime.application.notifications import AgentRunOutboxPublisher
from agent_runtime.persistence import create_agent_runtime_uow
from data_query.persistence import create_data_query_uow_factory
from main_api.application import NotificationCenterService, NotificationDispatcher
from main_api.api.notifications import router as notification_router
from main_api.entities import PlatformDomainEntity
from main_api.persistence import create_main_api_uow
from platform_core.database.oracle import create_database_runtime
from platform_core.identity import uuid7
from platform_core.notifications import (
    NotificationEnvelope,
    NotificationOutboxEntity,
    publish_notification,
)


async def _verify_http_sse(
    *, center: NotificationCenterService, domain_id: int,
    actor_id: str, expected_sequence: int,
) -> None:
    """启动真实 HTTP Server，验证 SSE 首次投递与 Last-Event-ID 续传。"""
    app = FastAPI()
    app.state.notification_center_service = center
    app.state.main_api_settings = SimpleNamespace(
        notifications=SimpleNamespace(
            sse_poll_interval_seconds=0.01,
            sse_heartbeat_seconds=0.02,
        )
    )

    @app.middleware("http")
    async def inject_context(request: Request, call_next):
        request.state.auth_context = SimpleNamespace(
            domain_id=str(domain_id), asserted_user_id=actor_id,
        )
        return await call_next(request)

    app.include_router(notification_router)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = int(listener.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(
        app,
        log_level="warning",
        lifespan="off",
    ))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError("通知 SSE 测试服务器未能启动")
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{port}/api/v1/notifications/events"
            ) as response:
                assert response.status == 200
                assert response.headers["Content-Type"].startswith(
                    "text/event-stream"
                )
                event_id = ""
                while not event_id:
                    line = (await asyncio.wait_for(
                        response.content.readline(), timeout=2,
                    )).decode().strip()
                    if line.startswith("id: "):
                        event_id = line.removeprefix("id: ")
                assert int(event_id) == expected_sequence
            async with session.get(
                f"http://127.0.0.1:{port}/api/v1/notifications/events",
                headers={"Last-Event-ID": str(expected_sequence)},
            ) as response:
                assert response.status == 200
                heartbeat = await asyncio.wait_for(
                    response.content.readline(), timeout=2,
                )
                assert heartbeat == b": keep-alive\n"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


async def main() -> None:
    runtime = create_database_runtime()
    marker = f"s6-smoke-{uuid7()}"
    actor_id = marker
    try:
        async with runtime.session_factory() as session:
            domain_id = await session.scalar(
                select(PlatformDomainEntity.domain_id)
                .where(PlatformDomainEntity.status == "ACTIVE")
                .order_by(PlatformDomainEntity.domain_id)
                .limit(1)
            )
        if domain_id is None:
            raise RuntimeError("真实通知 Smoke 需要至少一个 ACTIVE Domain")
        domain_id = int(domain_id)
        agent_uow = create_agent_runtime_uow(runtime.session_factory)
        main_uow = create_main_api_uow(runtime.session_factory)
        dq_uow = create_data_query_uow_factory(runtime.session_factory)
        center = NotificationCenterService(uow_factory=main_uow)
        dispatcher = NotificationDispatcher(
            uow_factory=main_uow, batch_size=50,
            lease_seconds=30, max_attempts=1,
        )
        await center.set_preference(
            domain_id=domain_id,
            actor_id=actor_id,
            event_type="agent.run.input_required",
            enabled=True,
        )

        run_id = uuid7()
        run = SimpleNamespace(
            run_id=run_id, domain_id=domain_id, actor_id=actor_id,
            request_id=f"{marker}-request", trace_id=f"{marker}-trace",
            status="WAITING_INPUT", error_code=None,
        )
        for _ in range(2):
            async with agent_uow() as uow:
                await AgentRunOutboxPublisher().publish(
                    uow=uow, run=run,
                    event_type="agent.run.input_required",
                    actor_id="worker", payload={"status": "WAITING_INPUT"},
                )
                await uow.commit()
        await dispatcher.dispatch_once()
        await dispatcher.dispatch_once()

        first = await center.list_notifications(
            domain_id=domain_id, actor_id=actor_id,
            limit=20, before_sequence=None,
        )
        if len(first) != 1:
            raise AssertionError("重复业务事件产生了重复 Inbox")
        sequence = int(first[0]["event_sequence"])
        replay = await center.stream_events(
            domain_id=domain_id, actor_id=actor_id,
            after_sequence=0,
        )
        resumed = await center.stream_events(
            domain_id=domain_id, actor_id=actor_id,
            after_sequence=sequence,
        )
        if len(replay) != 1 or resumed:
            raise AssertionError("SSE 事件序号续传不符合预期")
        await _verify_http_sse(
            center=center,
            domain_id=domain_id,
            actor_id=actor_id,
            expected_sequence=sequence,
        )
        if await center.list_notifications(
            domain_id=domain_id + 999999,
            actor_id=actor_id, limit=20, before_sequence=None,
        ):
            raise AssertionError("通知跨 Domain 泄漏")
        operations = await center.list_operations(
            domain_id=domain_id, actor_id=actor_id, limit=100,
        )
        first_operation = next(
            row for row in operations
            if row["source_operation_id"] == str(run_id)
        )
        await center.watch_operation(
            operation_id=UUID(first_operation["operation_id"]),
            domain_id=domain_id,
            actor_id=actor_id,
            notify_terminal=True,
        )
        await center.unwatch_operation(
            operation_id=UUID(first_operation["operation_id"]),
            domain_id=domain_id,
            actor_id=actor_id,
        )

        now = datetime.now(timezone.utc)
        ordered_run = f"{marker}-ordered"
        async with agent_uow() as uow:
            for suffix, event_type, occurred_at in (
                ("complete", "agent.run.completed", now),
                ("old-input", "agent.run.input_required", now - timedelta(minutes=1)),
            ):
                await publish_notification(
                    uow=uow, producer_service="agent-runtime",
                    event_key=f"{marker}:{suffix}",
                    envelope=NotificationEnvelope(
                        domain_id=domain_id, event_type=event_type,
                        resource_type="agent_run", resource_id=ordered_run,
                        initiator_actor_id=actor_id,
                        recipient_actor_ids=[actor_id],
                        summary="乱序投影 Smoke。",
                        occurred_at=occurred_at,
                        correlation_id=f"{marker}-ordered-trace",
                        operation_id=ordered_run,
                    ),
                )
            await uow.commit()
        await dispatcher.dispatch_once()
        operations = await center.list_operations(
            domain_id=domain_id, actor_id=actor_id, limit=100,
        )
        ordered = next(
            row for row in operations if row["source_operation_id"] == ordered_run
        )
        if ordered["status"] != "SUCCEEDED":
            raise AssertionError("乱序事件回退了后台任务状态")

        system_operation = f"{marker}-system"
        async with dq_uow() as uow:
            await publish_notification(
                uow=uow, producer_service="data-query",
                event_key=f"{marker}:system",
                envelope=NotificationEnvelope(
                    domain_id=domain_id, event_type="data_query.run.failed",
                    resource_type="data_query_run", resource_id=system_operation,
                    recipient_actor_ids=[], summary="系统任务失败。",
                    correlation_id=f"{marker}-system-trace",
                    operation_id=system_operation,
                    safe_data={"error_code": "SMOKE_FAILURE"},
                ),
            )
            await uow.commit()
        before_system = len(await center.list_notifications(
            domain_id=domain_id, actor_id=actor_id,
            limit=100, before_sequence=None,
        ))
        await dispatcher.dispatch_once()
        after_system = len(await center.list_notifications(
            domain_id=domain_id, actor_id=actor_id,
            limit=100, before_sequence=None,
        ))
        if before_system != after_system:
            raise AssertionError("无 Actor 系统事件生成了无主 Inbox")
        if not any(
            row["source_operation_id"] == system_operation
            for row in await center.list_operations(
                domain_id=domain_id, actor_id=actor_id, limit=100,
            )
        ):
            raise AssertionError("无 Actor 系统事件未保留 Operation 审计")

        corrupt_id = uuid7()
        async with runtime.session_factory() as session:
            session.add(NotificationOutboxEntity(
                outbox_id=corrupt_id,
                producer_service="agent-runtime",
                event_key=f"{marker}:corrupt",
                event_type="agent.run.completed",
                event_version=1,
                domain_id=domain_id,
                payload_json={"corrupt": True},
            ))
            await session.commit()
        await dispatcher.dispatch_once()
        quarantined = await center.quarantine(domain_id=domain_id, limit=100)
        if str(corrupt_id) not in {row["outbox_id"] for row in quarantined}:
            raise AssertionError("坏事件未进入隔离状态")
        repaired = NotificationEnvelope(
            domain_id=domain_id, event_type="agent.run.completed",
            resource_type="agent_run", resource_id=f"{marker}-repaired",
            initiator_actor_id=actor_id, recipient_actor_ids=[actor_id],
            summary="隔离事件已修复。", correlation_id=f"{marker}-repair-trace",
            operation_id=f"{marker}-repaired",
        )
        async with runtime.session_factory() as session:
            row = await session.get(NotificationOutboxEntity, corrupt_id)
            row.payload_json = repaired.model_dump(mode="json")
            await session.commit()
        await center.retry_quarantined(domain_id=domain_id, outbox_id=corrupt_id)
        await dispatcher.dispatch_once()
        if any(row["outbox_id"] == str(corrupt_id) for row in await center.quarantine(
            domain_id=domain_id, limit=100,
        )):
            raise AssertionError("隔离事件修复后重试仍未完成")

        forgotten = await center.forget_actor(domain_id=domain_id, actor_id=actor_id)
        if forgotten["inbox"] < 1 or forgotten["preferences"] != 1:
            raise AssertionError("Actor 清理未删除 Inbox 和偏好")
        print(
            "S6 Oracle Smoke 通过：事务 Outbox、幂等投影、乱序保护、"
            "Domain 隔离、Actor 缺失、真实 HTTP SSE 续传、隔离重试和 Actor 清理"
        )
    finally:
        async with runtime.engine.begin() as connection:
            params = {"marker": f"{marker}%", "actor": actor_id}
            await connection.execute(text(
                "DELETE FROM KBOT_OPERATION_WATCH WHERE ACTOR_ID = :actor"
            ), params)
            await connection.execute(text(
                "DELETE FROM KBOT_NOTIFICATION_INBOX WHERE RECIPIENT_ACTOR_ID = :actor "
                "OR OUTBOX_ID IN (SELECT OUTBOX_ID FROM KBOT_NOTIFICATION_OUTBOX "
                "WHERE EVENT_KEY LIKE :marker)"
            ), params)
            await connection.execute(text(
                "DELETE FROM KBOT_WORK_ITEM WHERE ACTOR_ID = :actor "
                "OR OPENED_OUTBOX_ID IN (SELECT OUTBOX_ID FROM KBOT_NOTIFICATION_OUTBOX "
                "WHERE EVENT_KEY LIKE :marker)"
            ), params)
            await connection.execute(text(
                "DELETE FROM KBOT_BACKGROUND_OPERATION WHERE RESOURCE_ID LIKE :marker "
                "OR LAST_OUTBOX_ID IN (SELECT OUTBOX_ID FROM KBOT_NOTIFICATION_OUTBOX "
                "WHERE EVENT_KEY LIKE :marker)"
            ), params)
            await connection.execute(text(
                "DELETE FROM KBOT_NOTIFICATION_PREF WHERE ACTOR_ID = :actor"
            ), params)
            await connection.execute(text(
                "DELETE FROM KBOT_NOTIFICATION_OUTBOX WHERE EVENT_KEY LIKE :marker"
            ), params)
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
