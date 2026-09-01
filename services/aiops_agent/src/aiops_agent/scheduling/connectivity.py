"""Target 与监控源的周期连通性探测调度。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from datetime import timedelta

from loguru import logger

from aiops_agent.entities import OutboxEntity
from platform_core.identity import uuid7


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


class AIOpsConnectivityScheduler:
    """每小时重试连通性，检查执行仍交给 Worker。"""

    def __init__(
        self,
        *,
        uow_factory,
        scheduler_id: str,
        interval_seconds: int,
        jitter_seconds: int,
        scan_interval_seconds: float,
    ):
        self._uow_factory = uow_factory
        self._scheduler_id = scheduler_id
        self._interval = interval_seconds
        self._jitter = jitter_seconds
        self._scan_interval = scan_interval_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> bool:
        return await self._request_one("TARGET") or await self._request_one(
            "DIAGNOSTIC_SOURCE"
        )

    async def _request_one(self, aggregate_type: str) -> bool:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            jitter = random.randint(0, self._jitter) if self._jitter else 0
            due_before = now - timedelta(seconds=self._interval + jitter)
            pending_before = now - timedelta(minutes=15)
            repository = (
                uow.targets
                if aggregate_type == "TARGET"
                else uow.diagnostic_sources
            )
            entity = await repository.claim_due_connectivity(
                due_before=due_before,
                pending_before=pending_before,
            )
            if entity is None:
                return False
            request_id = uuid7()
            entity.connectivity_status = "CHECKING"
            entity.connectivity_check_request_id = request_id
            entity.connectivity_check_requested_at = now
            entity.updated_by = self._scheduler_id
            entity.updated_at = now
            await uow.session.flush()
            aggregate_id = (
                entity.target_id
                if aggregate_type == "TARGET"
                else entity.diagnostic_source_id
            )
            event_type = (
                "TARGET_CONNECTIVITY_CHECK_REQUESTED"
                if aggregate_type == "TARGET"
                else "SOURCE_CONNECTIVITY_CHECK_REQUESTED"
            )
            payload = {
                "schema_version": "aiops.config.event.v1",
                "domain_id": int(entity.domain_id),
                "aggregate_type": aggregate_type,
                "aggregate_id": str(aggregate_id),
                "event_type": event_type,
                "row_version": int(entity.row_version),
                "actor_id": self._scheduler_id,
                "request_id": str(request_id),
                "trace_id": f"connectivity:{request_id}",
                "details": {
                    "connectivity_check_request_id": str(request_id),
                    "connectivity_version": int(entity.connectivity_version),
                },
            }
            digest = _payload_hash(payload)
            await uow.outbox.add(
                OutboxEntity(
                    outbox_id=uuid7(),
                    aggregate_type=aggregate_type,
                    aggregate_id=aggregate_id,
                    event_type=event_type,
                    idempotency_key=digest,
                    payload_json=payload,
                    payload_hash=digest,
                    status="PENDING",
                    trace_id=payload["trace_id"],
                )
            )
            await uow.commit()
            return True

    async def run_forever(self) -> None:
        logger.info("AIOps 连通性 Scheduler 开始运行")
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.opt(exception=exc).error(
                    "AIOps 连通性 Scheduler 本轮失败：{}",
                    type(exc).__name__,
                )
                worked = False
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._scan_interval
                )
            except TimeoutError:
                continue
        logger.info("AIOps 连通性 Scheduler 已停止")
