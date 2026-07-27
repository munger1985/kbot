"""确定性解析到期 Cron、Misfire 和报告窗口。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from zoneinfo import ZoneInfo

from aiops_agent.application.configuration.schedule import next_cron_run


def _timezone_database_version() -> str:
    try:
        return f"tzdata-{version('tzdata')}"
    except PackageNotFoundError:
        return "system-zoneinfo"


@dataclass(frozen=True)
class ScheduleResolution:
    scheduled_for: datetime
    next_run_at: datetime
    skipped: bool
    skip_reason: str | None
    resolution: dict
    period_start: datetime
    period_end: datetime


def resolve_due_schedule(
    *,
    cron_expression: str,
    timezone_name: str,
    schedule_type: str,
    due_at: datetime,
    now: datetime,
    misfire_policy: str,
    misfire_grace_seconds: int,
    resolver_version: str,
) -> ScheduleResolution:
    """把一个到期游标解析成至多一个 Fire 和下一游标。"""
    due = due_at.astimezone(UTC)
    current = now.astimezone(UTC)
    lateness = max(0, int((current - due).total_seconds()))
    misfired = lateness > misfire_grace_seconds
    selected = due
    skipped = False
    reason = None
    occurrences = 1
    if misfired and misfire_policy == "SKIP":
        skipped = True
        reason = "MISFIRE_SKIPPED"
        next_run = next_cron_run(
            expression=cron_expression,
            timezone_name=timezone_name,
            after=current,
        )
    elif misfired and misfire_policy == "LATEST_ONLY":
        candidate = due
        while occurrences < 10000:
            following = next_cron_run(
                expression=cron_expression,
                timezone_name=timezone_name,
                after=candidate,
            )
            if following > current:
                next_run = following
                break
            candidate = following
            occurrences += 1
        else:
            raise ValueError("Misfire 历史触发点超过安全上限")
        selected = candidate
    else:
        next_run = next_cron_run(
            expression=cron_expression,
            timezone_name=timezone_name,
            after=selected,
        )

    timezone = ZoneInfo(timezone_name)
    local_date = selected.astimezone(timezone).date()
    local_end = datetime.combine(
        local_date,
        datetime.min.time(),
        tzinfo=timezone,
    )
    days = 7 if schedule_type == "WEEKLY" else 1
    local_start = local_end - timedelta(days=days)
    return ScheduleResolution(
        scheduled_for=selected,
        next_run_at=next_run,
        skipped=skipped,
        skip_reason=reason,
        resolution={
            "resolver_version": resolver_version,
            "timezone": timezone_name,
            "timezone_database_version": _timezone_database_version(),
            "resolved_at": current.isoformat(),
            "original_due_at": due.isoformat(),
            "selected_scheduled_for": selected.isoformat(),
            "next_run_at": next_run.isoformat(),
            "lateness_seconds": lateness,
            "misfire_policy": misfire_policy,
            "occurrences_collapsed": occurrences,
        },
        period_start=local_start.astimezone(UTC),
        period_end=local_end.astimezone(UTC),
    )
