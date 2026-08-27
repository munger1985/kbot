"""多副本巡检 Scheduler：Plan Claim、Fire 创建与终态汇总。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import timedelta

from loguru import logger

from aiops_agent.entities import InspectionFireEntity, OutboxEntity
from platform_core.identity import uuid7

from .resolver import resolve_due_schedule


def _hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class AIOpsInspectionScheduler:
    def __init__(
        self,
        *,
        uow_factory,
        scheduler_id: str,
        system_agent_id,
        lease_seconds: int,
        interval_seconds: float,
        misfire_grace_seconds: int,
    ):
        self._uow_factory = uow_factory
        self._scheduler_id = scheduler_id
        self._system_agent_id = system_agent_id
        self._lease_seconds = lease_seconds
        self._interval = interval_seconds
        self._misfire_grace = misfire_grace_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> bool:
        if await self._schedule_one():
            return True
        return await self._reconcile_one()

    async def _schedule_one(self) -> bool:
        lease_token = uuid7()
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            assert uow.inspections is not None
            now = await uow.runs.database_now()
            plan = await uow.inspections.claim_due_plan(
                now=now,
                lease_owner=self._scheduler_id,
                lease_token=lease_token,
                lease_until=now
                + timedelta(seconds=self._lease_seconds),
            )
            if plan is None:
                return False
            if plan.next_run_at is None:
                return False
            resolution = resolve_due_schedule(
                cron_expression=plan.cron_expression,
                timezone_name=plan.timezone,
                schedule_type=plan.schedule_type,
                due_at=plan.next_run_at,
                now=now,
                misfire_policy=plan.misfire_policy,
                misfire_grace_seconds=self._misfire_grace,
                resolver_version=plan.schedule_resolver_version,
            )
            targets = await uow.inspections.list_active_targets(
                inspection_plan_id=plan.inspection_plan_id,
                domain_id=int(plan.domain_id),
            )
            target_snapshots = [
                {
                    "target_id": str(item.target_id),
                    "template_overrides": dict(
                        item.template_overrides_json or {}
                    ),
                }
                for item in targets
            ]
            open_fires = await uow.inspections.list_open_fires(
                inspection_plan_id=plan.inspection_plan_id,
                lock=True,
            )
            status = "RUNNING"
            skip_reason = resolution.skip_reason
            if resolution.skipped:
                status = "SKIPPED"
            elif not target_snapshots:
                status = "SKIPPED"
                skip_reason = "NO_ACTIVE_TARGETS"
            elif open_fires and plan.overlap_policy == "SKIP":
                status = "SKIPPED"
                skip_reason = "OVERLAP_SKIPPED"
            elif open_fires:
                status = "QUEUED"
                for queued in (
                    item for item in open_fires if item.status == "QUEUED"
                ):
                    queued.status = "SKIPPED"
                    queued.skip_reason = "SUPERSEDED_BY_LATEST"
                    queued.completed_at = now
                    queued.updated_at = now

            fire_id = uuid7()
            plan_snapshot = {
                "domain_id": int(plan.domain_id),
                "plan_id": str(plan.inspection_plan_id),
                "display_name": plan.display_name,
                "schedule_type": plan.schedule_type,
                "timezone": plan.timezone,
                "template_id": plan.template_id,
                "template_version": plan.template_version,
                "timeout_seconds": int(plan.timeout_seconds),
                "period_start": resolution.period_start.isoformat(),
                "period_end": resolution.period_end.isoformat(),
                "targets": target_snapshots,
            }
            await uow.inspections.add_fire(
                InspectionFireEntity(
                    inspection_fire_id=fire_id,
                    inspection_plan_id=plan.inspection_plan_id,
                    scheduled_for=resolution.scheduled_for,
                    status=status,
                    plan_row_version=int(plan.row_version),
                    template_id=plan.template_id,
                    template_version=plan.template_version,
                    schedule_resolver_version=(
                        plan.schedule_resolver_version
                    ),
                    plan_snapshot_json=plan_snapshot,
                    resolution_json=resolution.resolution,
                    target_count=len(target_snapshots),
                    run_count=0,
                    completed_count=0,
                    failed_count=0,
                    skip_reason=skip_reason,
                    started_at=now if status == "RUNNING" else None,
                    completed_at=now if status == "SKIPPED" else None,
                )
            )
            if status == "RUNNING":
                await self._enqueue_runs(
                    uow=uow,
                    fire_id=fire_id,
                    snapshot=plan_snapshot,
                    trace_id=str(uuid7()),
                    now=now,
                )
            advanced = await uow.inspections.advance_claimed_plan(
                inspection_plan_id=plan.inspection_plan_id,
                lease_owner=self._scheduler_id,
                lease_token=lease_token,
                now=now,
                expected_version=int(plan.row_version),
                scheduled_for=resolution.scheduled_for,
                next_run_at=resolution.next_run_at,
                updated_by=f"system:{self._scheduler_id}",
            )
            if not advanced:
                raise RuntimeError("巡检计划租约或版本围栏已失效")
            await uow.commit()
            logger.info(
                "巡检 Fire 已创建：fire_id={} status={} targets={}",
                fire_id,
                status,
                len(target_snapshots),
            )
            return True

    async def _reconcile_one(self) -> bool:
        terminal = {
            "COMPLETED",
            "PARTIAL",
            "FAILED",
            "CANCELLED",
            "EXPIRED",
        }
        success = {"COMPLETED", "PARTIAL"}
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            assert uow.inspections is not None
            now = await uow.runs.database_now()
            candidate = await uow.inspections.find_reconcilable_fire()
            if candidate is None:
                return False
            fire = await uow.inspections.get_fire(
                inspection_fire_id=candidate.inspection_fire_id,
                lock=True,
            )
            if fire is None:
                return False
            if fire.status == "QUEUED":
                open_fires = await uow.inspections.list_open_fires(
                    inspection_plan_id=fire.inspection_plan_id,
                    lock=True,
                )
                if any(
                    item.status == "RUNNING"
                    and item.inspection_fire_id
                    != fire.inspection_fire_id
                    for item in open_fires
                ):
                    return False
                fire.status = "RUNNING"
                fire.started_at = now
                fire.updated_at = now
                await self._enqueue_runs(
                    uow=uow,
                    fire_id=fire.inspection_fire_id,
                    snapshot=dict(fire.plan_snapshot_json),
                    trace_id=str(uuid7()),
                    now=now,
                )
                await uow.commit()
                return True
            runs = await uow.inspections.list_runs_for_fire(
                inspection_fire_id=fire.inspection_fire_id
            )
            request_events = (
                await uow.inspections.list_run_request_events_for_fire(
                    inspection_fire_id=fire.inspection_fire_id
                )
            )
            requests_finished = request_events and all(
                item.status in {"PUBLISHED", "FAILED"}
                for item in request_events
            )
            if len(runs) < int(fire.target_count) and not requests_finished:
                return False
            if any(item.status not in terminal for item in runs):
                return False
            completed = sum(item.status in success for item in runs)
            failed = int(fire.target_count) - completed
            if (
                len(runs) == int(fire.target_count)
                and completed == len(runs)
            ):
                status = "COMPLETED"
            elif completed:
                status = "PARTIAL"
            else:
                status = "FAILED"
            fire.status = status
            fire.run_count = len(runs)
            fire.completed_count = completed
            fire.failed_count = failed
            fire.completed_at = now
            fire.updated_at = now
            await uow.commit()
            logger.info(
                "巡检 Fire 已收敛：fire_id={} status={}",
                fire.inspection_fire_id,
                status,
            )
            return True

    async def _enqueue_runs(
        self,
        *,
        uow,
        fire_id,
        snapshot: dict,
        trace_id: str,
        now,
    ) -> None:
        for target in snapshot["targets"]:
            payload = {
                "inspection_fire_id": str(fire_id),
                                "domain_id": snapshot["domain_id"],
                "actor_id": "system:inspection-scheduler",
                "agent_id": str(self._system_agent_id),
                "target_id": target["target_id"],
                "template_id": snapshot["template_id"],
                "template_version": snapshot["template_version"],
                "schedule_type": snapshot["schedule_type"],
                "timezone": snapshot["timezone"],
                "period_start": snapshot["period_start"],
                "period_end": snapshot["period_end"],
                "timeout_seconds": snapshot["timeout_seconds"],
                "template_overrides": target["template_overrides"],
                "trace_id": trace_id,
            }
            await uow.outbox.add(
                OutboxEntity(
                    aggregate_type="OPS_INSPECTION_FIRE",
                    aggregate_id=fire_id,
                    event_type="OPS_INSPECTION_RUN_REQUESTED",
                    idempotency_key=(
                        f"inspection:{fire_id}:target:{target['target_id']}"
                    ),
                    payload_json=payload,
                    payload_hash=_hash(payload),
                    status="PENDING",
                    available_at=now,
                    max_attempts=8,
                    trace_id=trace_id,
                )
            )

    async def run_forever(self) -> None:
        logger.info("AIOps Inspection Scheduler 开始运行")
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
            except Exception as exc:
                logger.exception(
                    "AIOps Inspection Scheduler 本轮失败：{}",
                    type(exc).__name__,
                )
                worked = False
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._interval
                )
            except TimeoutError:
                pass
        logger.info("AIOps Inspection Scheduler 已停止")
