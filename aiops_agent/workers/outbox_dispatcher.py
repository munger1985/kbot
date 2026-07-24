"""Outbox 至少一次投递循环。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Protocol

from loguru import logger

from platform_core.contracts.aiops import CreateOpsRunCommand
from platform_core.contracts.aiops.executor import (
    MutationExecutionRequest,
)
from platform_core.identity import uuid7


class OutboxSink(Protocol):
    async def publish(self, event_type: str, payload: dict) -> None: ...


class LoggingOutboxSink:
    """开发期 Sink；后续消息中间件只需实现同一 Port。"""

    async def publish(self, event_type: str, payload: dict) -> None:
        logger.info(
            "AIOps Outbox 投递：event_type={} aggregate_id={}",
            event_type,
            payload.get("ops_run_id", "unknown"),
        )


class AIOpsDomainOutboxSink:
    """消费 AIOps 内部领域命令；其他事件仍投递到外部 Sink。"""

    def __init__(
        self,
        *,
        runtime_service,
        fallback: OutboxSink,
        monitor_health_service=None,
        db_executor_client=None,
    ):
        self._runtime_service = runtime_service
        self._fallback = fallback
        self._monitor_health_service = monitor_health_service
        self._db_executor_client = db_executor_client

    async def publish(self, event_type: str, payload: dict) -> None:
        if (
            event_type == "MONITOR_HEALTH_CHECK_REQUESTED"
            and self._monitor_health_service is not None
        ):
            await self._monitor_health_service.execute(payload)
            return
        if event_type == "OPS_ADVISORY_RESULT_RECORDED":
            await self._create_verification_run(
                payload=payload,
                idempotency_key=(
                    f"proposal:{payload['proposal_id']}:manual-result:verify"
                ),
                trigger="advisory_manual_result",
            )
            logger.info(
                "Advisory 人工结果验证 Run 已创建：proposal_id={}",
                payload["proposal_id"],
            )
            return
        if event_type == "OPS_EXECUTION_VERIFY_REQUESTED":
            await self._create_verification_run(
                payload=payload,
                idempotency_key=(
                    f"execution:{payload['execution_id']}:verify"
                ),
                trigger="execution_result",
            )
            logger.info(
                "Execution 效果验证 Run 已创建：proposal_id={}",
                payload["proposal_id"],
            )
            return
        if (
            event_type == "OPS_EXECUTION_CREATED"
            and self._db_executor_client is not None
        ):
            await self._db_executor_client.request_execution(
                MutationExecutionRequest(
                    execution_id=payload["execution_id"],
                    executor_request_id=payload[
                        "executor_request_id"
                    ],
                    idempotency_key=(
                        f"execution:{payload['execution_id']}:dispatch"
                    ),
                ),
                trace_id=payload["trace_id"],
            )
            return
        if event_type != "OPS_ALERT_AUTO_RUN_REQUESTED":
            await self._fallback.publish(event_type, payload)
            return
        alert_id = payload["alert_id"]
        await self._runtime_service.create_run(
            CreateOpsRunCommand(
                command_id=uuid7(),
                idempotency_key=f"alert:{alert_id}:observe",
                app_id=payload["app_id"],
                domain_id=payload["domain_id"],
                actor_id="system:monitor-intake",
                agent_id=payload["agent_id"],
                target_id=payload["target_id"],
                trigger_type="ALERT",
                trigger_event_id=payload["event_id"],
                trigger_alert_id=alert_id,
                input="监控告警触发只观测报告",
                blueprint_id="monitor.observe-report",
                blueprint_version="1",
                client_metadata={
                    "trace_id": payload["trace_id"],
                    "trigger": "verified_monitor_alert",
                },
            )
        )
        logger.info(
            "严重告警只观测 Run 已创建：alert_id={}", alert_id
        )

    async def _create_verification_run(
        self,
        *,
        payload: dict,
        idempotency_key: str,
        trigger: str,
    ) -> None:
        await self._runtime_service.create_run(
            CreateOpsRunCommand(
                command_id=uuid7(),
                idempotency_key=idempotency_key,
                app_id=payload["app_id"],
                domain_id=payload["domain_id"],
                actor_id=payload["actor_id"],
                agent_id=payload["agent_id"],
                target_id=payload["target_id"],
                trigger_type="API",
                input="验证数据库动作的实际效果",
                blueprint_id="change.advisory-verify",
                blueprint_version="1",
                client_metadata={
                    "trace_id": payload["trace_id"],
                    "trigger": trigger,
                    "advisory_verification": {
                        key: payload[key]
                        for key in (
                            "proposal_id",
                            "source_run_id",
                            "result_artifact_id",
                        )
                    },
                },
            )
        )
class AIOpsOutboxDispatcher:
    def __init__(
        self,
        *,
        uow_factory,
        sink: OutboxSink,
        dispatcher_id: str,
        lease_seconds: int,
        interval_seconds: float,
    ):
        self._uow_factory = uow_factory
        self._sink = sink
        self._dispatcher_id = dispatcher_id
        self._lease_seconds = lease_seconds
        self._interval = interval_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> bool:
        token = uuid7()
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            recovered = await uow.outbox.recover_expired(
                now=now,
                available_at=now + timedelta(seconds=1),
            )
            if recovered:
                await uow.commit()
                return True
            message = await uow.outbox.claim(
                now=now,
                lease_owner=self._dispatcher_id,
                lease_token=token,
                lease_until=now + timedelta(seconds=self._lease_seconds),
            )
            if message is None:
                return False
            # 先提交 Claim，外部投递绝不占用数据库事务。
            snapshot = {
                "outbox_id": message.outbox_id,
                "event_type": message.event_type,
                "payload": dict(message.payload_json or {}),
                "attempt": int(message.attempt_count),
                "max_attempts": int(message.max_attempts),
            }
            await uow.commit()
        try:
            await self._sink.publish(
                snapshot["event_type"], snapshot["payload"]
            )
        except Exception as exc:
            async with self._uow_factory() as uow:
                now = await uow.runs.database_now()
                retry = snapshot["attempt"] < snapshot["max_attempts"]
                changed = await uow.outbox.release_failed(
                    outbox_id=snapshot["outbox_id"],
                    lease_owner=self._dispatcher_id,
                    lease_token=token,
                    now=now,
                    new_status="RETRY_WAIT" if retry else "FAILED",
                    available_at=now
                    + timedelta(seconds=min(2 ** snapshot["attempt"], 60)),
                    error_code="OUTBOX_PUBLISH_FAILED",
                    error_message=type(exc).__name__,
                )
                if changed:
                    await uow.commit()
            return True
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            changed = await uow.outbox.mark_published(
                outbox_id=snapshot["outbox_id"],
                lease_owner=self._dispatcher_id,
                lease_token=token,
                now=now,
            )
            if changed:
                await uow.commit()
        return True

    async def run_forever(self) -> None:
        logger.info("AIOps Outbox Dispatcher 开始运行")
        while not self._stop.is_set():
            worked = await self.run_once()
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._interval
                )
            except TimeoutError:
                pass
        logger.info("AIOps Outbox Dispatcher 已停止")
