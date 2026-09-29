"""面向 DBA 的异常优先 Dashboard 投影。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from platform_core.contracts.aiops.findings import (
    FindingCard,
    FindingCompilation,
    FindingType,
)
from platform_core.contracts.aiops.public import (
    DashboardActivityItem,
    DashboardAttentionItem,
    DashboardAutomationSummary,
    DashboardHealthCount,
    DashboardRiskItem,
    DashboardSummary,
    DashboardTargetRow,
    OpsDashboard,
)


_CLOSED_SITUATION_STATUSES = frozenset({"RESOLVED", "CLOSED"})
_UNREACHABLE_CONNECTIVITY = frozenset({"UNREACHABLE", "MISCONFIGURED"})
_CRITICAL_FINDING_TYPES = frozenset(
    {
        FindingType.LOCK_WAIT.value,
        FindingType.LONG_SESSION.value,
        FindingType.DG_LAG.value,
        FindingType.REPLICATION_LAG.value,
        FindingType.TABLESPACE.value,
        FindingType.CONNECTION_USAGE.value,
    }
)
_HIGH_SEVERITIES = frozenset({"HIGH", "CRITICAL"})
_SEVERITY_ORDER = {
    "CRITICAL": 0,
    "HIGH": 1,
    "WARNING": 2,
    "MEDIUM": 2,
    "LOW": 3,
    "INFO": 4,
}
_HEALTH_ORDER = {
    "CRITICAL": 0,
    "UNREACHABLE": 1,
    "WARNING": 2,
    "STALE": 3,
    "UNKNOWN": 4,
    "DISABLED": 5,
    "HEALTHY": 6,
}
_LAG_TYPES = frozenset(
    {FindingType.DG_LAG.value, FindingType.REPLICATION_LAG.value}
)


_RISK_TYPE_CATEGORY = {
    FindingType.TABLESPACE.value: "CAPACITY",
    FindingType.CONNECTION_USAGE.value: "CAPACITY",
    FindingType.ARCHIVE_HEADROOM.value: "CAPACITY",
    FindingType.DG_LAG.value: "REPLICATION",
    FindingType.REPLICATION_LAG.value: "REPLICATION",
    FindingType.BACKUP_FAILED.value: "BACKUP",
    FindingType.LOCK_WAIT.value: "SESSION",
    FindingType.LONG_SESSION.value: "SESSION",
    FindingType.LONG_TRANSACTION.value: "SESSION",
}
_RISK_LABELS = {
    FindingType.TABLESPACE.value: "表空间使用率",
    FindingType.CONNECTION_USAGE.value: "连接使用率",
    FindingType.ARCHIVE_HEADROOM.value: "归档空间余量",
    FindingType.DG_LAG.value: "Data Guard 延迟",
    FindingType.REPLICATION_LAG.value: "复制延迟",
    FindingType.BACKUP_FAILED.value: "备份失败",
    FindingType.LOCK_WAIT.value: "锁等待",
    FindingType.LONG_SESSION.value: "长会话",
    FindingType.LONG_TRANSACTION.value: "长事务",
}


@dataclass(frozen=True)
class DashboardTargetSnapshot:
    target_id: UUID
    display_name: str
    db_type: str
    environment: str
    db_role: str
    importance_level: int
    status: str
    connectivity_status: str
    observed_status: str
    readonly_connection_enabled: bool
    last_observed_at: datetime | None = None
    last_error_code: str | None = None


@dataclass(frozen=True)
class DashboardSituationSnapshot:
    situation_id: UUID
    target_id: UUID
    status: str
    severity: str
    title: str
    first_observed_at: datetime
    last_observed_at: datetime


@dataclass(frozen=True)
class DashboardRunSnapshot:
    target_id: UUID
    ops_run_id: UUID
    status: str
    created_at: datetime
    completed_at: datetime | None
    error_code: str | None = None
    findings: tuple[FindingCard, ...] = ()


def parse_finding_payload(payload: object) -> tuple[FindingCard, ...]:
    """无法解析的 Finding 块视为没有卡片，不中断 Dashboard。"""

    try:
        compilation = FindingCompilation.model_validate(payload)
    except Exception:
        return ()
    return compilation.findings


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _max_metric(
    findings: tuple[FindingCard, ...],
    *,
    types: frozenset[str],
    field_name: str | None = None,
    field_by_type: dict[str, str] | None = None,
) -> float | None:
    values: list[float] = []
    for finding in findings:
        finding_type = str(finding.finding_type)
        if finding_type not in types:
            continue
        key = field_name
        if field_by_type is not None:
            key = field_by_type.get(finding_type)
        if not key:
            continue
        number = _as_float(finding.fields.get(key))
        if number is not None:
            values.append(number)
    return max(values) if values else None


def _finding_health(findings: tuple[FindingCard, ...]) -> str:
    has_warning = False
    for finding in findings:
        finding_type = str(finding.finding_type)
        severity = str(finding.severity)
        if (
            severity in _HIGH_SEVERITIES
            and finding_type in _CRITICAL_FINDING_TYPES
        ):
            return "CRITICAL"
        if severity == "MEDIUM" or severity in _HIGH_SEVERITIES:
            has_warning = True
    return "WARNING" if has_warning else "HEALTHY"


def _latest(values: list[datetime | None]) -> datetime | None:
    available = [
        value.replace(tzinfo=UTC)
        if value.tzinfo is None or value.utcoffset() is None
        else value.astimezone(UTC)
        for value in values
        if value is not None
    ]
    return max(available) if available else None


def _freshness(
    observed_at: datetime | None,
    *,
    now: datetime,
    stale_after: timedelta,
) -> str:
    if observed_at is None:
        return "UNKNOWN"
    return "STALE" if observed_at < now - stale_after else "CURRENT"


def _top_situation(
    situations: list[DashboardSituationSnapshot],
) -> DashboardSituationSnapshot | None:
    if not situations:
        return None
    return min(
        situations,
        key=lambda item: (
            _SEVERITY_ORDER.get(item.severity, 5),
            item.first_observed_at,
            str(item.situation_id),
        ),
    )


def _target_health(
    *,
    target: DashboardTargetSnapshot,
    top_situation: DashboardSituationSnapshot | None,
    findings: tuple[FindingCard, ...],
    freshness: str,
    recent_failure: DashboardRunSnapshot | None,
) -> str:
    if target.status == "DISABLED":
        return "DISABLED"
    if target.connectivity_status in _UNREACHABLE_CONNECTIVITY or (
        target.readonly_connection_enabled and target.observed_status == "DOWN"
    ):
        return "UNREACHABLE"
    if top_situation is not None and top_situation.severity == "CRITICAL":
        return "CRITICAL"
    finding_health = _finding_health(findings)
    if freshness == "UNKNOWN":
        return "UNKNOWN"
    if freshness == "STALE":
        return "STALE"
    if finding_health == "CRITICAL":
        return "CRITICAL"
    if top_situation is not None or recent_failure is not None:
        return "WARNING"
    return finding_health


def _format_number(value: float, suffix: str) -> str:
    text = str(int(value)) if value.is_integer() else f"{value:.1f}"
    return f"{text}{suffix}"


def _risk_value(finding: FindingCard) -> str:
    finding_type = str(finding.finding_type)
    fields = finding.fields
    if finding_type == FindingType.TABLESPACE.value:
        value = _as_float(fields.get("used_percent"))
        return _format_number(value, "%") if value is not None else "已确认"
    if finding_type == FindingType.CONNECTION_USAGE.value:
        value = _as_float(fields.get("utilization_percent"))
        return _format_number(value, "%") if value is not None else "已确认"
    if finding_type in _LAG_TYPES:
        value = _as_float(fields.get("lag_seconds"))
        return _format_number(value, " 秒") if value is not None else "已确认"
    return "已确认"


def _risk_label(finding: FindingCard) -> str:
    finding_type = str(finding.finding_type)
    name = finding.object_ref.object_name
    base = _RISK_LABELS.get(finding_type, finding_type)
    return f"{base} · {name}" if name else base


def _risk_items(
    *,
    target: DashboardTargetSnapshot,
    run: DashboardRunSnapshot | None,
) -> list[DashboardRiskItem]:
    if run is None:
        return []
    result: list[DashboardRiskItem] = []
    for finding in run.findings:
        finding_type = str(finding.finding_type)
        category = _RISK_TYPE_CATEGORY.get(finding_type)
        if category is None:
            continue
        result.append(
            DashboardRiskItem(
                target_id=target.target_id,
                target_name=target.display_name,
                importance_level=target.importance_level,
                category=category,
                finding_type=finding_type,
                severity=str(finding.severity),
                label=_risk_label(finding),
                value_text=_risk_value(finding),
                observed_at=run.completed_at,
                run_id=run.ops_run_id,
            )
        )
    return result


def _attention_item(
    *,
    target: DashboardTargetSnapshot,
    health: str,
    top_situation: DashboardSituationSnapshot | None,
    recent_failure: DashboardRunSnapshot | None,
    top_risk: DashboardRiskItem | None,
    observed_at: datetime | None,
    latest_run_id: UUID | None,
) -> DashboardAttentionItem | None:
    if health in {"HEALTHY", "DISABLED"}:
        return None
    if health == "UNREACHABLE":
        return DashboardAttentionItem(
            target_id=target.target_id,
            target_name=target.display_name,
            environment=target.environment,
            importance_level=target.importance_level,
            health=health,
            category="CONNECTIVITY",
            severity="CRITICAL",
            title=(
                f"数据库不可达：{target.last_error_code}"
                if target.last_error_code
                else "数据库不可达或只读探活失败"
            ),
            status=target.connectivity_status,
            observed_at=observed_at,
            run_id=latest_run_id,
        )
    if top_situation is not None:
        return DashboardAttentionItem(
            target_id=target.target_id,
            target_name=target.display_name,
            environment=target.environment,
            importance_level=target.importance_level,
            health=health,
            category="ALERT",
            severity=top_situation.severity,
            title=top_situation.title,
            status=top_situation.status,
            started_at=top_situation.first_observed_at,
            observed_at=top_situation.last_observed_at,
            situation_id=top_situation.situation_id,
            run_id=latest_run_id,
        )
    if recent_failure is not None:
        return DashboardAttentionItem(
            target_id=target.target_id,
            target_name=target.display_name,
            environment=target.environment,
            importance_level=target.importance_level,
            health=health,
            category="AUTOMATION",
            severity="HIGH",
            title=(
                f"自动诊断失败：{recent_failure.error_code}"
                if recent_failure.error_code
                else "自动诊断执行失败"
            ),
            status=recent_failure.status,
            started_at=recent_failure.created_at,
            observed_at=recent_failure.completed_at or recent_failure.created_at,
            run_id=recent_failure.ops_run_id,
        )
    if top_risk is not None:
        return DashboardAttentionItem(
            target_id=target.target_id,
            target_name=target.display_name,
            environment=target.environment,
            importance_level=target.importance_level,
            health=health,
            category=top_risk.category,
            severity=top_risk.severity,
            title=f"{top_risk.label}：{top_risk.value_text}",
            status="CONFIRMED",
            observed_at=top_risk.observed_at,
            run_id=top_risk.run_id,
        )
    return DashboardAttentionItem(
        target_id=target.target_id,
        target_name=target.display_name,
        environment=target.environment,
        importance_level=target.importance_level,
        health=health,
        category="DATA_FRESHNESS",
        severity="WARNING",
        title="运维数据已过期" if health == "STALE" else "尚无有效运维数据",
        status=health,
        observed_at=observed_at,
        run_id=latest_run_id,
    )


def project_ops_dashboard(
    targets: (
        tuple[DashboardTargetSnapshot, ...] | list[DashboardTargetSnapshot]
    ),
    situations: (
        tuple[DashboardSituationSnapshot, ...]
        | list[DashboardSituationSnapshot]
    ),
    latest_runs: (
        tuple[DashboardRunSnapshot, ...] | list[DashboardRunSnapshot]
    ),
    recent_runs: (
        tuple[DashboardRunSnapshot, ...] | list[DashboardRunSnapshot]
    ),
    *,
    failed_runs: (
        tuple[DashboardRunSnapshot, ...] | list[DashboardRunSnapshot]
    ) = (),
    run_status_counts: dict[str, int] | None = None,
    inspection_status_counts: dict[str, int] | None = None,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(hours=24),
) -> OpsDashboard:
    generated_at = (now or datetime.now(UTC)).astimezone(UTC)
    situations_by_target: dict[UUID, list[DashboardSituationSnapshot]] = {}
    for situation in situations:
        if situation.status in _CLOSED_SITUATION_STATUSES:
            continue
        situations_by_target.setdefault(situation.target_id, []).append(situation)

    latest_by_target: dict[UUID, DashboardRunSnapshot] = {}
    for run in latest_runs:
        current = latest_by_target.get(run.target_id)
        if current is None:
            latest_by_target[run.target_id] = run
            continue
        current_at = current.completed_at
        run_at = run.completed_at
        if run_at is not None and (current_at is None or run_at > current_at):
            latest_by_target[run.target_id] = run
        elif (
            run_at == current_at
            and str(run.ops_run_id) > str(current.ops_run_id)
        ):
            latest_by_target[run.target_id] = run

    recent_failures: dict[UUID, DashboardRunSnapshot] = {}
    for run in failed_runs:
        if run.status != "FAILED":
            continue
        current = recent_failures.get(run.target_id)
        if current is None or (
            (run.completed_at or run.created_at)
            > (current.completed_at or current.created_at)
        ):
            recent_failures[run.target_id] = run

    target_rows: list[DashboardTargetRow] = []
    attention_items: list[DashboardAttentionItem] = []
    risk_items: list[DashboardRiskItem] = []
    target_names = {target.target_id: target.display_name for target in targets}
    for target in targets:
        run = latest_by_target.get(target.target_id)
        findings = run.findings if run is not None else ()
        target_situations = situations_by_target.get(target.target_id, [])
        top_situation = _top_situation(target_situations)
        target_risks = _risk_items(target=target, run=run)
        risk_items.extend(target_risks)
        top_risk = min(
            target_risks,
            key=lambda item: (
                _SEVERITY_ORDER.get(item.severity, 5),
                item.category,
            ),
            default=None,
        )
        recent_failure = recent_failures.get(target.target_id)
        observed_at = _latest([
            target.last_observed_at,
            run.completed_at if run is not None else None,
            (
                recent_failure.completed_at or recent_failure.created_at
                if recent_failure is not None
                else None
            ),
            *(item.last_observed_at for item in target_situations),
        ])
        freshness = _freshness(
            observed_at,
            now=generated_at,
            stale_after=stale_after,
        )
        health = _target_health(
            target=target,
            top_situation=top_situation,
            findings=findings,
            freshness=freshness,
            recent_failure=recent_failure,
        )
        capacity_candidates = [
            item
            for item in target_risks
            if item.category == "CAPACITY" and item.value_text.endswith("%")
        ]
        capacity_item = max(
            capacity_candidates,
            key=lambda item: float(item.value_text.removesuffix("%")),
            default=None,
        )
        lag_seconds = _max_metric(
            findings,
            types=_LAG_TYPES,
            field_name="lag_seconds",
        )
        attention = _attention_item(
            target=target,
            health=health,
            top_situation=top_situation,
            recent_failure=recent_failure,
            top_risk=top_risk,
            observed_at=observed_at,
            latest_run_id=run.ops_run_id if run is not None else None,
        )
        if attention is not None:
            attention_items.append(attention)
        target_rows.append(
            DashboardTargetRow(
                target_id=target.target_id,
                display_name=target.display_name,
                db_type=target.db_type,
                environment=target.environment,
                db_role=target.db_role,
                importance_level=target.importance_level,
                health=health,
                attention_reason=attention.title if attention is not None else None,
                open_alert_count=len(target_situations),
                critical_alert_count=sum(
                    item.severity == "CRITICAL" for item in target_situations
                ),
                high_alert_count=sum(
                    item.severity == "HIGH" for item in target_situations
                ),
                capacity_percent=(
                    float(capacity_item.value_text.removesuffix("%"))
                    if capacity_item is not None
                    else None
                ),
                capacity_label=(capacity_item.label if capacity_item else None),
                lag_seconds=lag_seconds,
                data_freshness=freshness,
                evidence_observed_at=observed_at,
                last_diagnosed_at=run.completed_at if run is not None else None,
                latest_run_id=run.ops_run_id if run is not None else None,
            )
        )

    target_rows.sort(
        key=lambda item: (
            _HEALTH_ORDER[item.health],
            -item.importance_level,
            item.display_name.casefold(),
            str(item.target_id),
        )
    )
    attention_items.sort(
        key=lambda item: (
            _HEALTH_ORDER[item.health],
            -item.importance_level,
            _SEVERITY_ORDER.get(item.severity, 5),
            item.started_at or item.observed_at or generated_at,
            item.target_name.casefold(),
        )
    )
    risk_items.sort(
        key=lambda item: (
            _SEVERITY_ORDER.get(item.severity, 5),
            -item.importance_level,
            item.target_name.casefold(),
            item.label,
        )
    )

    run_counts = run_status_counts or {}
    inspection_counts = inspection_status_counts or {}
    window_start = generated_at - timedelta(hours=24)
    automation = DashboardAutomationSummary(
        window_start=window_start,
        window_end=generated_at,
        run_total=sum(run_counts.values()),
        run_succeeded=run_counts.get("COMPLETED", 0),
        run_partial=run_counts.get("PARTIAL", 0),
        run_failed=run_counts.get("FAILED", 0),
        inspection_total=sum(inspection_counts.values()),
        inspection_succeeded=inspection_counts.get("COMPLETED", 0),
        inspection_partial=inspection_counts.get("PARTIAL", 0),
        inspection_failed=inspection_counts.get("FAILED", 0),
    )
    all_situations = [
        item for values in situations_by_target.values() for item in values
    ]
    activities: list[DashboardActivityItem] = [
        DashboardActivityItem(
            kind="ALERT",
            status=item.severity,
            title=item.title,
            occurred_at=item.last_observed_at,
            target_id=item.target_id,
            target_name=target_names.get(item.target_id),
            situation_id=item.situation_id,
        )
        for item in all_situations
    ]
    activities.extend(
        DashboardActivityItem(
            kind="DIAGNOSIS",
            status=run.status,
            title=(
                f"自动诊断失败：{run.error_code}"
                if run.status == "FAILED" and run.error_code
                else "自动诊断已结束"
            ),
            occurred_at=run.completed_at or run.created_at,
            target_id=run.target_id,
            target_name=target_names.get(run.target_id),
            run_id=run.ops_run_id,
        )
        for run in recent_runs
        if run.status in {"COMPLETED", "PARTIAL", "FAILED"}
    )
    activities.sort(key=lambda item: item.occurred_at, reverse=True)

    summary = DashboardSummary(
        target_count=len(target_rows),
        attention_target_count=len(attention_items),
        critical_alert_count=sum(
            item.severity == "CRITICAL" for item in all_situations
        ),
        high_alert_count=sum(item.severity == "HIGH" for item in all_situations),
        open_alert_count=len(all_situations),
        unreachable_count=sum(
            item.health == "UNREACHABLE" for item in target_rows
        ),
        stale_or_unknown_count=sum(
            item.health in {"STALE", "UNKNOWN"} for item in target_rows
        ),
        failed_automation_count=(
            automation.run_failed + automation.inspection_failed
        ),
        healthy_count=sum(item.health == "HEALTHY" for item in target_rows),
        disabled_count=sum(item.health == "DISABLED" for item in target_rows),
    )
    health_distribution = tuple(
        DashboardHealthCount(
            health=health,
            count=sum(item.health == health for item in target_rows),
        )
        for health in _HEALTH_ORDER
    )
    return OpsDashboard(
        generated_at=generated_at,
        stale_after_seconds=int(stale_after.total_seconds()),
        summary=summary,
        health_distribution=health_distribution,
        attention_items=tuple(attention_items[:8]),
        risk_items=tuple(risk_items[:20]),
        automation=automation,
        recent_activities=tuple(activities[:12]),
        targets=tuple(target_rows),
    )
