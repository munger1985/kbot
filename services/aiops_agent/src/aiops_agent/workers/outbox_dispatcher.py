"""Outbox 至少一次投递循环。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
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
        diagnostic_source_connectivity_service=None,
        target_connectivity_service=None,
        db_executor_client=None,
        turn_queue_service=None,
    ):
        self._runtime_service = runtime_service
        self._fallback = fallback
        self._diagnostic_source_connectivity_service = (
            diagnostic_source_connectivity_service
        )
        self._target_connectivity_service = target_connectivity_service
        self._db_executor_client = db_executor_client
        self._turn_queue_service = turn_queue_service

    async def publish(self, event_type: str, payload: dict) -> None:
        if event_type == "aiops.turn.created" and self._turn_queue_service is not None:
            await self._turn_queue_service.accept_created(payload)
            return
        if (
            event_type == "SOURCE_CONNECTIVITY_CHECK_REQUESTED"
            and self._diagnostic_source_connectivity_service is not None
        ):
            await self._diagnostic_source_connectivity_service.execute(payload)
            return
        if (
            event_type == "TARGET_CONNECTIVITY_CHECK_REQUESTED"
            and self._target_connectivity_service is not None
        ):
            await self._target_connectivity_service.execute(payload)
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
        if event_type == "OPS_INSPECTION_RUN_REQUESTED":
            fire_id = payload["inspection_fire_id"]
            target_id = payload["target_id"]
            await self._runtime_service.create_run(
                CreateOpsRunCommand(
                    command_id=uuid7(),
                    idempotency_key=(
                        f"inspection:{fire_id}:target:{target_id}"
                    ),
                    domain_id=payload["domain_id"],
                    actor_id=payload["actor_id"],
                    agent_id=payload["agent_id"],
                    target_id=target_id,
                    trigger_type="SCHEDULE",
                    inspection_fire_id=fire_id,
                    deadline=datetime.now(UTC)
                    + timedelta(seconds=payload["timeout_seconds"]),
                    observation_start=datetime.fromisoformat(
                        payload["period_start"]
                    ),
                    observation_end=datetime.fromisoformat(
                        payload["period_end"]
                    ),
                    input="执行数据库定期巡检并生成报告",
                    blueprint_id="database.diagnostic-baseline",
                    blueprint_version="1",
                    client_metadata={
                        "trace_id": payload["trace_id"],
                        "trigger": "inspection_schedule",
                        "inspection": {
                            "fire_id": fire_id,
                            "template_id": payload["template_id"],
                            "template_version": payload[
                                "template_version"
                            ],
                            "schedule_type": payload["schedule_type"],
                            "timezone": payload["timezone"],
                            "period_start": payload["period_start"],
                            "period_end": payload["period_end"],
                            "template_overrides": payload[
                                "template_overrides"
                            ],
                        },
                    },
                )
            )
            logger.info(
                "巡检 Run 已创建：fire_id={} target_id={}",
                fire_id,
                target_id,
            )
            return
        if event_type != "OPS_SITUATION_AUTO_RUN_REQUESTED":
            await self._fallback.publish(event_type, payload)
            return
        situation_id = payload["situation_id"]
        await self._runtime_service.create_run(
            CreateOpsRunCommand(
                command_id=uuid7(),
                idempotency_key=f"situation:{situation_id}:observe",
                domain_id=payload["domain_id"],
                actor_id="system:signal-intake",
                agent_id=payload["agent_id"],
                target_id=payload["target_id"],
                trigger_type="ALERT",
                trigger_signal_event_id=payload["signal_event_id"],
                situation_id=situation_id,
                input="已验证故障信号触发主动根因诊断",
                blueprint_id="diagnosis.root-cause",
                blueprint_version="1",
                client_metadata={
                    "trace_id": payload["trace_id"],
                    "trigger": "verified_signal_event",
                },
            )
        )
        logger.info(
            "严重故障情境诊断 Run 已创建：situation_id={}", situation_id
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
