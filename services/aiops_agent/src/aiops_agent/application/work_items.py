"""DBA 工作项队列、自动归并和审计用例。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    SignedCursorCodec,
    canonical_json,
)
from aiops_agent.application.errors import (
    resource_not_found,
    row_version_changed,
    state_conflict,
)
from aiops_agent.entities import (
    WorkItemActivityEntity,
    WorkItemEntity,
    WorkItemLinkEntity,
    WorkItemOccurrenceEntity,
)
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.contracts.aiops import (
    FindingCard,
    FindingColumnGap,
    FindingCompilation,
    FindingSeverity,
    WorkItemActivityView,
    WorkItemAssignment,
    WorkItemCreate,
    WorkItemOccurrenceView,
    WorkItemPage,
    WorkItemPhase,
    WorkItemPriority,
    WorkItemQueueSummary,
    WorkItemResolutionCode,
    WorkItemResourceKind,
    WorkItemResourceLinkView,
    WorkItemRouteRun,
    WorkItemSourceKind,
    WorkItemStatus,
    WorkItemSummary,
    WorkItemTransition,
    WorkItemType,
    WorkItemView,
)
from platform_core.identity import uuid7


_TERMINAL_STATUSES = {
    WorkItemStatus.CLOSED.value,
    WorkItemStatus.CANCELLED.value,
}
_TRANSITIONS = {
    WorkItemStatus.PENDING_TRIAGE.value: {
        WorkItemStatus.OPEN.value,
        WorkItemStatus.CANCELLED.value,
    },
    WorkItemStatus.OPEN.value: {
        WorkItemStatus.IN_PROGRESS.value,
        WorkItemStatus.WAITING.value,
        WorkItemStatus.CANCELLED.value,
    },
    WorkItemStatus.IN_PROGRESS.value: {
        WorkItemStatus.WAITING.value,
        WorkItemStatus.PENDING_VERIFICATION.value,
        WorkItemStatus.RESOLVED.value,
        WorkItemStatus.CANCELLED.value,
    },
    WorkItemStatus.WAITING.value: {
        WorkItemStatus.IN_PROGRESS.value,
        WorkItemStatus.CANCELLED.value,
    },
    WorkItemStatus.PENDING_VERIFICATION.value: {
        WorkItemStatus.IN_PROGRESS.value,
        WorkItemStatus.RESOLVED.value,
    },
    WorkItemStatus.RESOLVED.value: {
        WorkItemStatus.IN_PROGRESS.value,
        WorkItemStatus.CLOSED.value,
    },
}
_SLA = {
    WorkItemPriority.P1.value: (15, 4, 8),
    WorkItemPriority.P2.value: (30, 8, 12),
    WorkItemPriority.P3.value: (240, 72, 96),
    WorkItemPriority.P4.value: (480, 168, 216),
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _priority(severity: str) -> str:
    return {
        FindingSeverity.CRITICAL.value: WorkItemPriority.P1.value,
        FindingSeverity.HIGH.value: WorkItemPriority.P2.value,
        FindingSeverity.MEDIUM.value: WorkItemPriority.P3.value,
    }.get(severity, WorkItemPriority.P4.value)


def _source_kind(trigger_type: str) -> str:
    return {
        "ALERT": WorkItemSourceKind.ALERT.value,
        "SCHEDULE": WorkItemSourceKind.INSPECTION.value,
        "CHAT": WorkItemSourceKind.CHAT.value,
    }.get(trigger_type, WorkItemSourceKind.MANUAL.value)


def _work_type(*, trigger_type: str, finding_type: str) -> str:
    if finding_type == "OBSERVABILITY_GAP":
        return WorkItemType.OBSERVABILITY_GAP.value
    if trigger_type == "ALERT":
        return WorkItemType.INCIDENT_RESPONSE.value
    if trigger_type == "SCHEDULE":
        return WorkItemType.RISK_REMEDIATION.value
    return WorkItemType.PROBLEM_INVESTIGATION.value


def _fingerprint(
    *, domain_id: int, target_id: UUID, work_type: str,
    finding_type: str, object_ref: dict[str, Any], condition_code: str,
) -> str:
    material = {
        "domain_id": domain_id,
        "target_id": str(target_id),
        "work_type": work_type,
        "finding_type": finding_type,
        "object_ref": object_ref,
        "condition_code": condition_code,
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def _sla(priority: str, observed_at: datetime) -> tuple[datetime, datetime, datetime]:
    acknowledgement_minutes, resolution_hours, verification_hours = _SLA[priority]
    return (
        observed_at + timedelta(minutes=acknowledgement_minutes),
        observed_at + timedelta(hours=resolution_hours),
        observed_at + timedelta(hours=verification_hours),
    )


class WorkItemService:
    """提供队列查询、人工操作和 Run 结果自动路由。"""

    def __init__(self, *, uow_factory, cursor_codec: SignedCursorCodec):
        self._uow_factory = uow_factory
        self._cursor_codec = cursor_codec

    async def list_items(
        self, *, scope: ConfigurationScope, status: str | None,
        priority: str | None, target_id: UUID | None,
        assignee_user_id: str | None, unassigned: bool,
        overdue: bool,
        cursor: str | None, limit: int,
    ) -> WorkItemPage:
        filters = {
            "status": status,
            "priority": priority,
            "target_id": str(target_id) if target_id else None,
            "assignee_user_id": assignee_user_id,
            "unassigned": unassigned,
            "overdue": overdue,
        }
        after_at = after_id = None
        if cursor:
            after_at, after_id = self._cursor_codec.decode(
                token=cursor, scope=scope, filters=filters
            )
        async with self._uow_factory() as uow:
            assert uow.work_items is not None
            now = _utc(await uow.runs.database_now())
            rows = await uow.work_items.page(
                domain_id=scope.domain_id,
                status=status,
                priority=priority,
                target_id=target_id,
                assignee_user_id=assignee_user_id,
                unassigned=unassigned,
                overdue_before=now if overdue else None,
                after_due_at=after_at,
                after_id=after_id,
                limit=limit + 1,
            )
            counts = await uow.work_items.queue_summary(
                domain_id=scope.domain_id, actor_id=scope.actor_id, now=now
            )
        has_more = len(rows) > limit
        selected = rows[:limit]
        next_cursor = None
        if has_more and selected:
            last = selected[-1][0]
            next_cursor = self._cursor_codec.encode(
                scope=scope,
                sort_at=_utc(last.resolution_due_at),
                resource_id=last.work_item_id,
                filters=filters,
            )
        return WorkItemPage(
            items=tuple(self._summary(item, target_name) for item, target_name in selected),
            queue_summary=WorkItemQueueSummary(**counts),
            has_more=has_more,
            next_cursor=next_cursor,
        )

    async def get_item(
        self, *, work_item_id: UUID, scope: ConfigurationScope
    ) -> WorkItemView:
        async with self._uow_factory() as uow:
            return await self._detail(uow, work_item_id=work_item_id, scope=scope)

    async def create_item(
        self, *, body: WorkItemCreate, scope: ConfigurationScope
    ) -> WorkItemView:
        async with self._uow_factory() as uow:
            assert uow.work_items is not None
            target_name = await uow.work_items.target_name(
                target_id=body.target_id, domain_id=scope.domain_id
            )
            if target_name is None:
                raise resource_not_found("数据库实例")
            now = _utc(await uow.runs.database_now())
            item_id = uuid7()
            ack_due, resolution_due, verification_due = _sla(body.priority, now)
            item = await uow.work_items.add(
                WorkItemEntity(
                    work_item_id=item_id,
                    domain_id=scope.domain_id,
                    item_key=f"WI-{item_id.hex[:12].upper()}",
                    target_id=body.target_id,
                    work_type=body.work_type,
                    source_kind=WorkItemSourceKind.MANUAL.value,
                    fingerprint=hashlib.sha256(str(item_id).encode()).hexdigest(),
                    title=body.title,
                    summary=body.summary,
                    severity=body.severity,
                    priority=body.priority,
                    status=WorkItemStatus.OPEN.value,
                    phase=WorkItemPhase.TRIAGE.value,
                    assignment_group=body.assignment_group,
                    assignee_user_id=body.assignee_user_id,
                    acknowledgement_due_at=ack_due,
                    resolution_due_at=resolution_due,
                    verification_due_at=verification_due,
                    first_observed_at=now,
                    last_observed_at=now,
                    occurrence_count=1,
                    reopen_count=0,
                    created_by=scope.actor_id,
                    updated_by=scope.actor_id,
                    created_at=now,
                    updated_at=now,
                    row_version=1,
                )
            )
            await self._activity(
                uow, item=item, activity_type="CREATED", actor_id=scope.actor_id,
                detail={"source": "MANUAL"}, now=now,
            )
            await uow.commit()
        return await self.get_item(work_item_id=item_id, scope=scope)

    async def assign(
        self, *, work_item_id: UUID, body: WorkItemAssignment,
        scope: ConfigurationScope,
    ) -> WorkItemView:
        async with self._uow_factory() as uow:
            assert uow.work_items is not None
            item = await uow.work_items.get_scoped(
                work_item_id=work_item_id, domain_id=scope.domain_id, lock=True
            )
            if item is None:
                raise resource_not_found("工作项")
            if int(item.row_version) != body.expected_row_version:
                raise row_version_changed()
            before = {
                "assignment_group": item.assignment_group,
                "assignee_user_id": item.assignee_user_id,
            }
            item.assignment_group = body.assignment_group
            item.assignee_user_id = body.assignee_user_id
            item.updated_by = scope.actor_id
            now = _utc(await uow.runs.database_now())
            item.updated_at = now
            await self._activity(
                uow, item=item, activity_type="ASSIGNED", actor_id=scope.actor_id,
                detail={"before": before, "after": body.model_dump(mode="json")},
                now=now,
            )
            await uow.commit()
        return await self.get_item(work_item_id=work_item_id, scope=scope)

    async def transition(
        self, *, work_item_id: UUID, body: WorkItemTransition,
        scope: ConfigurationScope,
    ) -> WorkItemView:
        async with self._uow_factory() as uow:
            assert uow.work_items is not None
            item = await uow.work_items.get_scoped(
                work_item_id=work_item_id, domain_id=scope.domain_id, lock=True
            )
            if item is None:
                raise resource_not_found("工作项")
            if int(item.row_version) != body.expected_row_version:
                raise row_version_changed()
            allowed = _TRANSITIONS.get(item.status, set())
            if body.status not in allowed:
                raise state_conflict(f"工作项不能从 {item.status} 迁移到 {body.status}")
            now = _utc(await uow.runs.database_now())
            from_status, from_phase = item.status, item.phase
            item.status = body.status
            item.phase = body.phase
            item.wait_reason = body.wait_reason
            item.resolution_code = body.resolution_code
            item.resolution_note = body.resolution_note
            item.resolved_at = now if body.status == WorkItemStatus.RESOLVED else None
            item.closed_at = now if body.status == WorkItemStatus.CLOSED else None
            item.updated_by = scope.actor_id
            item.updated_at = now
            await self._activity(
                uow, item=item, activity_type="STATUS_CHANGED",
                actor_id=scope.actor_id, from_status=from_status,
                to_status=item.status, from_phase=from_phase, to_phase=item.phase,
                detail={
                    "wait_reason": body.wait_reason,
                    "resolution_code": body.resolution_code,
                    "resolution_note": body.resolution_note,
                }, now=now,
            )
            await uow.commit()
        return await self.get_item(work_item_id=work_item_id, scope=scope)

    async def route_run(
        self, *, body: WorkItemRouteRun, scope: ConfigurationScope
    ) -> tuple[WorkItemSummary, ...]:
        async with self._uow_factory() as uow:
            run = await uow.runs.get_run_scoped(
                ops_run_id=body.ops_run_id, domain_id=scope.domain_id
            )
            if run is None:
                raise resource_not_found("诊断运行")
            now = _utc(await uow.runs.database_now())
            items = await route_completed_run_work_items(
                uow=uow,
                run=run,
                now=now,
                actor_id=scope.actor_id,
                selected_finding_ids=set(body.finding_ids),
                include_gaps=body.include_gaps,
                manual=True,
            )
            assert uow.work_items is not None
            names = {
                item.target_id: await uow.work_items.target_name(
                    target_id=item.target_id, domain_id=scope.domain_id
                ) or "未知实例"
                for item in items
            }
            await uow.commit()
        return tuple(self._summary(item, names[item.target_id]) for item in items)

    async def _detail(
        self, uow: AIOpsUnitOfWork, *, work_item_id: UUID,
        scope: ConfigurationScope,
    ) -> WorkItemView:
        assert uow.work_items is not None
        item = await uow.work_items.get_scoped(
            work_item_id=work_item_id, domain_id=scope.domain_id
        )
        if item is None:
            raise resource_not_found("工作项")
        target_name = await uow.work_items.target_name(
            target_id=item.target_id, domain_id=scope.domain_id
        ) or "未知实例"
        occurrences = await uow.work_items.list_occurrences(work_item_id=work_item_id)
        links = await uow.work_items.list_links(work_item_id=work_item_id)
        activities = await uow.work_items.list_activities(work_item_id=work_item_id)
        return WorkItemView(
            **self._summary(item, target_name).model_dump(),
            resolution_note=item.resolution_note,
            occurrences=tuple(
                WorkItemOccurrenceView(
                    occurrence_id=row.occurrence_id,
                    ops_run_id=row.ops_run_id,
                    situation_id=row.situation_id,
                    finding_id=row.finding_id,
                    finding_type=row.finding_type,
                    severity=row.severity,
                    confirmation=row.confirmation,
                    finding=(FindingCard.model_validate(row.finding_snapshot_json)
                             if row.finding_snapshot_json else None),
                    evidence_refs=tuple(row.evidence_refs_json or ()),
                    observed_at=_utc(row.observed_at),
                    created_at=_utc(row.created_at),
                ) for row in occurrences
            ),
            links=tuple(
                WorkItemResourceLinkView(
                    resource_kind=row.resource_kind,
                    resource_id=row.resource_id,
                    role=row.link_role,
                    label=(row.snapshot_json or {}).get("label"),
                    status=(row.snapshot_json or {}).get("status"),
                    created_at=_utc(row.created_at),
                ) for row in links
            ),
            activities=tuple(
                WorkItemActivityView(
                    activity_id=row.activity_id,
                    activity_type=row.activity_type,
                    actor_id=row.actor_id,
                    from_status=row.from_status,
                    to_status=row.to_status,
                    from_phase=row.from_phase,
                    to_phase=row.to_phase,
                    detail=row.detail_json or {},
                    created_at=_utc(row.created_at),
                ) for row in activities
            ),
        )

    @staticmethod
    def _summary(item: WorkItemEntity, target_name: str) -> WorkItemSummary:
        return WorkItemSummary(
            work_item_id=item.work_item_id,
            item_key=item.item_key,
            target_id=item.target_id,
            target_name=target_name,
            work_type=item.work_type,
            source_kind=item.source_kind,
            title=item.title,
            summary=item.summary,
            severity=item.severity,
            priority=item.priority,
            status=item.status,
            phase=item.phase,
            wait_reason=item.wait_reason,
            assignment_group=item.assignment_group,
            assignee_user_id=item.assignee_user_id,
            acknowledgement_due_at=(_utc(item.acknowledgement_due_at)
                                    if item.acknowledgement_due_at else None),
            resolution_due_at=(_utc(item.resolution_due_at)
                               if item.resolution_due_at else None),
            verification_due_at=(_utc(item.verification_due_at)
                                 if item.verification_due_at else None),
            first_observed_at=_utc(item.first_observed_at),
            last_observed_at=_utc(item.last_observed_at),
            occurrence_count=int(item.occurrence_count),
            reopen_count=int(item.reopen_count),
            resolution_code=item.resolution_code,
            resolved_at=_utc(item.resolved_at) if item.resolved_at else None,
            closed_at=_utc(item.closed_at) if item.closed_at else None,
            row_version=int(item.row_version),
            created_at=_utc(item.created_at),
            updated_at=_utc(item.updated_at),
        )

    @staticmethod
    async def _activity(
        uow: AIOpsUnitOfWork, *, item: WorkItemEntity, activity_type: str,
        actor_id: str, detail: dict[str, Any], now: datetime,
        from_status: str | None = None, to_status: str | None = None,
        from_phase: str | None = None, to_phase: str | None = None,
    ) -> None:
        assert uow.work_items is not None
        await uow.work_items.add_activity(
            WorkItemActivityEntity(
                activity_id=uuid7(), work_item_id=item.work_item_id,
                activity_type=activity_type, actor_id=actor_id,
                from_status=from_status, to_status=to_status,
                from_phase=from_phase, to_phase=to_phase,
                detail_json=detail, created_at=now,
            )
        )


async def route_completed_run_work_items(
    *, uow: AIOpsUnitOfWork, run, now: datetime, actor_id: str,
    selected_finding_ids: set[str] | None = None,
    include_gaps: bool = True, manual: bool = False,
) -> list[WorkItemEntity]:
    """在 Run 完成事务内创建或归并工作项；重复调用不会重复计数。"""
    assert uow.work_items is not None
    assert uow.turns is not None
    selected_finding_ids = selected_finding_ids or set()
    routed: list[WorkItemEntity] = []

    if run.workflow_kind == "VERIFICATION" and run.source_proposal_id:
        linked = await uow.work_items.list_by_resource(
            resource_kind=WorkItemResourceKind.PROPOSAL.value,
            resource_id=run.source_proposal_id,
        )
        for item in linked:
            await _ensure_link(
                uow, item=item, resource_kind=WorkItemResourceKind.VERIFICATION.value,
                resource_id=run.ops_run_id, role="VERIFIES",
                snapshot={"label": "验证运行", "status": run.status},
                actor_id=actor_id, now=now,
            )
            if item.status not in _TERMINAL_STATUSES:
                old_status, old_phase = item.status, item.phase
                item.status = WorkItemStatus.PENDING_VERIFICATION.value
                item.phase = WorkItemPhase.VERIFICATION.value
                item.updated_by = actor_id
                item.updated_at = now
                await WorkItemService._activity(
                    uow, item=item, activity_type="VERIFICATION_COMPLETED",
                    actor_id=actor_id, from_status=old_status,
                    to_status=item.status, from_phase=old_phase,
                    to_phase=item.phase,
                    detail={"ops_run_id": str(run.ops_run_id)}, now=now,
                )
            routed.append(item)

    if run.trigger_type == "CHAT" and not manual:
        return routed
    payload = (await uow.turns.list_finding_blocks_for_runs(
        ops_run_ids=(run.ops_run_id,)
    )).get(run.ops_run_id)
    if payload is None:
        return routed
    compilation = FindingCompilation.model_validate(payload)
    findings = list(compilation.findings)
    for finding in findings:
        if selected_finding_ids and finding.finding_id not in selected_finding_ids:
            continue
        if not manual and finding.severity not in {
            FindingSeverity.CRITICAL.value,
            FindingSeverity.HIGH.value,
            FindingSeverity.MEDIUM.value,
        }:
            continue
        routed.append(await _route_finding(
            uow=uow, run=run, finding=finding, now=now, actor_id=actor_id
        ))
    if include_gaps:
        for gap in compilation.gaps:
            gap_id = f"gap:{gap.finding_type}:{gap.source_tool_id}:{gap.column}:{gap.code}"
            if selected_finding_ids and gap_id not in selected_finding_ids:
                continue
            routed.append(await _route_gap(
                uow=uow, run=run, gap=gap, gap_id=gap_id,
                now=now, actor_id=actor_id,
            ))
    return list({item.work_item_id: item for item in routed}.values())


async def _route_finding(
    *, uow: AIOpsUnitOfWork, run, finding: FindingCard,
    now: datetime, actor_id: str,
) -> WorkItemEntity:
    object_ref = finding.object_ref.model_dump(mode="json", exclude_none=True)
    work_type = _work_type(
        trigger_type=run.trigger_type, finding_type=finding.finding_type
    )
    condition_code = (
        f"{finding.threshold.metric}:{finding.threshold.operator}"
        if finding.threshold else finding.finding_type
    )
    fingerprint = _fingerprint(
        domain_id=int(run.domain_id), target_id=run.target_id,
        work_type=work_type, finding_type=finding.finding_type,
        object_ref=object_ref, condition_code=condition_code,
    )
    title_object = (
        finding.object_ref.object_name
        or finding.object_ref.sql_id
        or finding.object_ref.object_kind
    )
    return await _upsert_occurrence(
        uow=uow, run=run, now=now, actor_id=actor_id,
        finding_id=finding.finding_id,
        finding_type=finding.finding_type,
        severity=finding.severity,
        confirmation=finding.confirmation,
        finding_snapshot=finding.model_dump(mode="json"),
        evidence_refs=list(finding.evidence_refs),
        work_type=work_type,
        fingerprint=fingerprint,
        title=f"{finding.finding_type} · {title_object}",
        summary=finding.impact,
    )


async def _route_gap(
    *, uow: AIOpsUnitOfWork, run, gap: FindingColumnGap, gap_id: str,
    now: datetime, actor_id: str,
) -> WorkItemEntity:
    work_type = WorkItemType.OBSERVABILITY_GAP.value
    object_ref = {"source_tool_id": gap.source_tool_id, "column": gap.column}
    return await _upsert_occurrence(
        uow=uow, run=run, now=now, actor_id=actor_id,
        finding_id=gap_id,
        finding_type="OBSERVABILITY_GAP",
        severity=FindingSeverity.MEDIUM.value,
        confirmation="CONFIRMED",
        finding_snapshot=None,
        evidence_refs=[gap.evidence_ref] if gap.evidence_ref else [],
        work_type=work_type,
        fingerprint=_fingerprint(
            domain_id=int(run.domain_id), target_id=run.target_id,
            work_type=work_type, finding_type="OBSERVABILITY_GAP",
            object_ref=object_ref, condition_code=gap.code,
        ),
        title=f"监控数据缺口 · {gap.column}",
        summary=gap.detail,
    )


async def _upsert_occurrence(
    *, uow: AIOpsUnitOfWork, run, now: datetime, actor_id: str,
    finding_id: str, finding_type: str, severity: str, confirmation: str,
    finding_snapshot: dict[str, Any] | None, evidence_refs: list[str],
    work_type: str, fingerprint: str, title: str, summary: str,
) -> WorkItemEntity:
    assert uow.work_items is not None
    item = await uow.work_items.get_open_by_fingerprint(
        domain_id=int(run.domain_id), target_id=run.target_id,
        fingerprint=fingerprint, lock=True,
    )
    priority = _priority(severity)
    if item is None:
        item_id = uuid7()
        ack_due, resolution_due, verification_due = _sla(priority, now)
        initial_status = (
            WorkItemStatus.PENDING_TRIAGE.value
            if priority in {WorkItemPriority.P3.value, WorkItemPriority.P4.value}
            else WorkItemStatus.OPEN.value
        )
        item = await uow.work_items.add(WorkItemEntity(
            work_item_id=item_id, domain_id=int(run.domain_id),
            item_key=f"WI-{item_id.hex[:12].upper()}", target_id=run.target_id,
            work_type=work_type, source_kind=_source_kind(run.trigger_type),
            fingerprint=fingerprint, title=title, summary=summary,
            severity=severity, priority=priority, status=initial_status,
            phase=WorkItemPhase.TRIAGE.value,
            acknowledgement_due_at=ack_due, resolution_due_at=resolution_due,
            verification_due_at=verification_due,
            first_observed_at=now, last_observed_at=now,
            occurrence_count=1, reopen_count=0,
            created_by=actor_id, updated_by=actor_id,
            created_at=now, updated_at=now, row_version=1,
        ))
        await WorkItemService._activity(
            uow, item=item, activity_type="CREATED", actor_id=actor_id,
            detail={"source": item.source_kind, "finding_id": finding_id}, now=now,
        )
    elif not await uow.work_items.occurrence_exists(
        work_item_id=item.work_item_id,
        ops_run_id=run.ops_run_id,
        finding_id=finding_id,
    ):
        old_status = item.status
        item.last_observed_at = now
        item.occurrence_count = int(item.occurrence_count) + 1
        item.severity = severity
        item.priority = min(item.priority, priority)
        item.summary = summary
        item.updated_by = actor_id
        item.updated_at = now
        if old_status in {
            WorkItemStatus.RESOLVED.value,
            WorkItemStatus.PENDING_VERIFICATION.value,
        }:
            item.status = WorkItemStatus.OPEN.value
            item.phase = WorkItemPhase.DIAGNOSIS.value
            item.reopen_count = int(item.reopen_count) + 1
            item.resolution_code = None
            item.resolution_note = None
            item.resolved_at = None
        await WorkItemService._activity(
            uow, item=item, activity_type="OCCURRENCE_MERGED",
            actor_id=actor_id, from_status=old_status, to_status=item.status,
            detail={"ops_run_id": str(run.ops_run_id), "finding_id": finding_id},
            now=now,
        )
    else:
        return item

    if not await uow.work_items.occurrence_exists(
        work_item_id=item.work_item_id,
        ops_run_id=run.ops_run_id,
        finding_id=finding_id,
    ):
        await uow.work_items.add_occurrence(WorkItemOccurrenceEntity(
            occurrence_id=uuid7(), work_item_id=item.work_item_id,
            ops_run_id=run.ops_run_id, situation_id=run.situation_id,
            finding_id=finding_id, finding_type=finding_type,
            severity=severity, confirmation=confirmation,
            finding_snapshot_json=finding_snapshot,
            evidence_refs_json=evidence_refs or None,
            observed_at=now, created_at=now,
        ))
    await _ensure_run_links(uow=uow, item=item, run=run, actor_id=actor_id, now=now)
    return item


async def _ensure_run_links(
    *, uow: AIOpsUnitOfWork, item: WorkItemEntity, run,
    actor_id: str, now: datetime,
) -> None:
    await _ensure_link(
        uow, item=item, resource_kind=WorkItemResourceKind.RUN.value,
        resource_id=run.ops_run_id, role="OCCURRENCE_SOURCE",
        snapshot={"label": "诊断运行", "status": run.status},
        actor_id=actor_id, now=now,
    )
    if run.situation_id:
        await _ensure_link(
            uow, item=item, resource_kind=WorkItemResourceKind.SITUATION.value,
            resource_id=run.situation_id, role="CORRELATED_SITUATION",
            snapshot={"label": "关联态势"}, actor_id=actor_id, now=now,
        )
    report = await uow.inspections.get_current_report_for_run(
        ops_run_id=run.ops_run_id
    )
    if report:
        await _ensure_link(
            uow, item=item, resource_kind=WorkItemResourceKind.REPORT.value,
            resource_id=report.report_id, role="SOURCE_REPORT",
            snapshot={"label": report.title, "status": report.status},
            actor_id=actor_id, now=now,
        )
    for proposal in await uow.changes.list_proposals_for_run(
        ops_run_id=run.ops_run_id
    ):
        await _ensure_link(
            uow, item=item, resource_kind=WorkItemResourceKind.PROPOSAL.value,
            resource_id=proposal.proposal_id, role="REMEDIATION_PROPOSAL",
            snapshot={"label": proposal.action_type, "status": proposal.status},
            actor_id=actor_id, now=now,
        )


async def _ensure_link(
    uow: AIOpsUnitOfWork, *, item: WorkItemEntity, resource_kind: str,
    resource_id: UUID, role: str, snapshot: dict[str, Any],
    actor_id: str, now: datetime,
) -> None:
    assert uow.work_items is not None
    if await uow.work_items.existing_link(
        work_item_id=item.work_item_id, resource_kind=resource_kind,
        resource_id=resource_id, link_role=role,
    ) is not None:
        return
    await uow.work_items.add_link(WorkItemLinkEntity(
        work_item_link_id=uuid7(), work_item_id=item.work_item_id,
        resource_kind=resource_kind, resource_id=resource_id,
        link_role=role, snapshot_json=json.loads(json.dumps(snapshot, default=str)),
        created_by=actor_id, created_at=now,
    ))
