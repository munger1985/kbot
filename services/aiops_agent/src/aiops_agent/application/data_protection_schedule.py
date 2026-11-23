"""数据保护排程：用计划发生时刻判定备份类型和增量基线。"""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from platform_core.contracts.aiops.data_protection import ProtectionBackupKind, ProtectionBackupRotation, ProtectionPlanSettings


def _rotation_slot(
    rotation: ProtectionBackupRotation, day: date, zone: ZoneInfo,
) -> tuple[datetime | None, ProtectionBackupKind]:
    kind = "FULL" if (day - rotation.start_date).days % rotation.cycle_days == 0 else "INCREMENTAL"
    clock = rotation.full_time if kind == "FULL" else rotation.incremental_time
    local = datetime.combine(day, time.fromisoformat(clock), zone)
    try:
        utc = local.astimezone(UTC)
        wall_time = utc.astimezone(zone)
    except OverflowError:
        return None, kind
    # 跳过夏令时缺失的钟面时刻；重复的钟面时刻取第一次。
    return (utc if wall_time == local else None), kind


def next_backup_at(settings: ProtectionPlanSettings, after: datetime) -> datetime | None:
    if (rotation := settings.backup_rotation) is not None:
        zone = ZoneInfo(rotation.timezone)
        day = max(rotation.start_date, after.astimezone(zone).date())
        while True:
            at, _ = _rotation_slot(rotation, day, zone)
            if at is not None and at > after:
                return at
            if day == date.max:
                return None
            day += timedelta(days=1)

    # 配置服务也依赖任务用例，避免模块加载时循环导入。
    from aiops_agent.application.configuration.schedule import next_cron_run

    triggers = (
        next_cron_run(expression=schedule.cron, timezone_name=schedule.timezone, after=after,
                      first_occurrence_only=settings.incremental_enabled)
        for schedule in (settings.backup_schedule, settings.backup_incremental_schedule) if schedule
    )
    return min(triggers, default=None)


def scheduled_backup_kind(settings: ProtectionPlanSettings, at: datetime) -> ProtectionBackupKind | None:
    if (rotation := settings.backup_rotation) is not None:
        zone = ZoneInfo(rotation.timezone)
        day = at.astimezone(zone).date()
        if day < rotation.start_date:
            return None
        trigger, kind = _rotation_slot(rotation, day, zone)
        return kind if trigger == at else None

    from aiops_agent.application.configuration.schedule import backup_cron_matches

    if not settings.incremental_enabled:
        return "FULL" if settings.backup_schedule else None
    # 同时命中时只执行一次，优先全量，避免备份抢占和歧义。
    for kind, schedule in (("FULL", settings.backup_schedule), ("INCREMENTAL", settings.backup_incremental_schedule)):
        if backup_cron_matches(expression=schedule.cron, timezone_name=schedule.timezone, at=at):
            return kind
    return None


def required_full_at(settings: ProtectionPlanSettings, incremental_at: datetime) -> datetime | None:
    if (rotation := settings.backup_rotation) is not None:
        zone = ZoneInfo(rotation.timezone)
        day = incremental_at.astimezone(zone).date()
        phase = (day - rotation.start_date).days % rotation.cycle_days
        if day < rotation.start_date or phase == 0:
            return None
        at, _ = _rotation_slot(rotation, day - timedelta(days=phase), zone)
        return at

    from aiops_agent.application.configuration.schedule import previous_cron_run

    full = settings.backup_schedule
    return previous_cron_run(expression=full.cron, timezone_name=full.timezone, before=incremental_at)
