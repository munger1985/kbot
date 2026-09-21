"""库群总览投影；只汇总健康、告警、容量和延迟，不展开 SID。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from platform_core.contracts.aiops.findings import (
    FindingCard,
    FindingCompilation,
    FindingType,
)
from platform_core.contracts.aiops.public import (
    FleetDashboard,
    FleetSummary,
    FleetTargetCard,
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
_HEALTH_ORDER = {
    "CRITICAL": 0,
    "UNREACHABLE": 1,
    "WARNING": 2,
    "DISABLED": 3,
    "HEALTHY": 4,
}
_CAPACITY_FIELDS = {
    FindingType.TABLESPACE.value: "used_percent",
    FindingType.CONNECTION_USAGE.value: "utilization_percent",
}
_LAG_TYPES = frozenset(
    {FindingType.DG_LAG.value, FindingType.REPLICATION_LAG.value}
)


@dataclass(frozen=True)
class FleetTargetSnapshot:
    target_id: UUID
    display_name: str
    db_type: str
    environment: str
    status: str
    connectivity_status: str
    observed_status: str
    readonly_connection_enabled: bool


@dataclass(frozen=True)
class FleetSituationSnapshot:
    target_id: UUID
    status: str


@dataclass(frozen=True)
class FleetRunSnapshot:
    target_id: UUID
    ops_run_id: UUID
    completed_at: datetime | None
    findings: tuple[FindingCard, ...] = ()


def parse_finding_payload(payload: object) -> tuple[FindingCard, ...]:
    """无法解析的 Finding 块视为没有卡片，不中断库群总览。"""

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


def _target_health(
    *,
    target: FleetTargetSnapshot,
    open_alert_count: int,
    findings: tuple[FindingCard, ...],
) -> str:
    if target.status == "DISABLED":
        return "DISABLED"
    if target.connectivity_status in _UNREACHABLE_CONNECTIVITY or (
        target.readonly_connection_enabled and target.observed_status == "DOWN"
    ):
        return "UNREACHABLE"
    if open_alert_count > 0:
        return "CRITICAL"
    return _finding_health(findings)


def project_fleet_dashboard(
    targets: tuple[FleetTargetSnapshot, ...] | list[FleetTargetSnapshot],
    situations: tuple[FleetSituationSnapshot, ...] | list[FleetSituationSnapshot],
    runs: tuple[FleetRunSnapshot, ...] | list[FleetRunSnapshot],
) -> FleetDashboard:
    open_alerts: dict[UUID, int] = {}
    for situation in situations:
        if situation.status in _CLOSED_SITUATION_STATUSES:
            continue
        open_alerts[situation.target_id] = (
            open_alerts.get(situation.target_id, 0) + 1
        )

    latest_runs: dict[UUID, FleetRunSnapshot] = {}
    for run in runs:
        current = latest_runs.get(run.target_id)
        if current is None:
            latest_runs[run.target_id] = run
            continue
        current_at = current.completed_at
        run_at = run.completed_at
        if run_at is not None and (current_at is None or run_at > current_at):
            latest_runs[run.target_id] = run
        elif (
            run_at == current_at
            and str(run.ops_run_id) > str(current.ops_run_id)
        ):
            latest_runs[run.target_id] = run

    cards: list[FleetTargetCard] = []
    for target in targets:
        run = latest_runs.get(target.target_id)
        findings = run.findings if run is not None else ()
        open_alert_count = open_alerts.get(target.target_id, 0)
        cards.append(
            FleetTargetCard(
                target_id=target.target_id,
                display_name=target.display_name,
                db_type=target.db_type,
                environment=target.environment,
                health=_target_health(
                    target=target,
                    open_alert_count=open_alert_count,
                    findings=findings,
                ),
                open_alert_count=open_alert_count,
                max_capacity_percent=_max_metric(
                    findings,
                    types=frozenset(_CAPACITY_FIELDS),
                    field_by_type=_CAPACITY_FIELDS,
                ),
                max_lag_seconds=_max_metric(
                    findings,
                    types=_LAG_TYPES,
                    field_name="lag_seconds",
                ),
                last_diagnosed_at=(
                    run.completed_at if run is not None else None
                ),
                latest_run_id=run.ops_run_id if run is not None else None,
            )
        )
    cards.sort(
        key=lambda item: (
            _HEALTH_ORDER[item.health],
            item.display_name.casefold(),
            str(item.target_id),
        )
    )
    diagnosed = [
        item.last_diagnosed_at
        for item in cards
        if item.last_diagnosed_at is not None
    ]
    summary = FleetSummary(
        target_count=len(cards),
        healthy_count=sum(item.health == "HEALTHY" for item in cards),
        warning_count=sum(item.health == "WARNING" for item in cards),
        critical_count=sum(item.health == "CRITICAL" for item in cards),
        unreachable_count=sum(item.health == "UNREACHABLE" for item in cards),
        disabled_count=sum(item.health == "DISABLED" for item in cards),
        open_alert_count=sum(item.open_alert_count for item in cards),
        last_diagnosed_at=max(diagnosed) if diagnosed else None,
    )
    return FleetDashboard(summary=summary, items=tuple(cards))
