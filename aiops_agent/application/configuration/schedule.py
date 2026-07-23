"""巡检计划 Cron、时区与模板校验。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiops_agent.application.errors import validation_failed
from aiops_agent.config import InspectionTemplateRegistration


def _parse_field(expression: str, minimum: int, maximum: int) -> set[int]:
    values: set[int] = set()
    for part in expression.split(","):
        if not part:
            raise ValueError("空表达式")
        base, separator, step_text = part.partition("/")
        step = int(step_text) if separator else 1
        if step < 1:
            raise ValueError("步长必须大于零")
        if base == "*":
            start, end = minimum, maximum
        elif "-" in base:
            start_text, end_text = base.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(base)
        if start < minimum or end > maximum or start > end:
            raise ValueError("字段超出范围")
        values.update(range(start, end + 1, step))
    return values


def next_cron_run(
    *,
    expression: str,
    timezone_name: str,
    after: datetime | None = None,
) -> datetime:
    """计算五段 Cron 的下一次 UTC 触发时间。"""
    fields = expression.split()
    if len(fields) != 5:
        raise validation_failed("Cron 必须是规范的五段表达式")
    try:
        timezone = ZoneInfo(timezone_name)
        minutes = _parse_field(fields[0], 0, 59)
        hours = _parse_field(fields[1], 0, 23)
        month_days = _parse_field(fields[2], 1, 31)
        months = _parse_field(fields[3], 1, 12)
        week_days = {
            0 if value == 7 else value
            for value in _parse_field(fields[4], 0, 7)
        }
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise validation_failed(f"Cron 或 IANA 时区无效：{exc}") from exc

    current = (after or datetime.now(UTC)).astimezone(UTC)
    current = current.replace(second=0, microsecond=0) + timedelta(minutes=1)
    end = current + timedelta(days=370)
    day_of_month_wildcard = fields[2] == "*"
    day_of_week_wildcard = fields[4] == "*"
    while current <= end:
        local = current.astimezone(timezone)
        cron_weekday = (local.weekday() + 1) % 7
        dom_match = local.day in month_days
        dow_match = cron_weekday in week_days
        if day_of_month_wildcard:
            day_match = dow_match
        elif day_of_week_wildcard:
            day_match = dom_match
        else:
            day_match = dom_match or dow_match
        if (
            local.minute in minutes
            and local.hour in hours
            and local.month in months
            and day_match
        ):
            return current
        current += timedelta(minutes=1)
    raise validation_failed("Cron 在未来 370 天内没有可计算的触发时间")


class InspectionTemplateRegistry:
    """部署时冻结的巡检模板与覆盖字段白名单。"""

    def __init__(
        self,
        registrations: tuple[InspectionTemplateRegistration, ...],
    ):
        self._registrations = {
            (
                item.template_id,
                item.template_version,
                item.schedule_resolver_version,
            ): item
            for item in registrations
        }

    def validate(
        self,
        *,
        template_id: str,
        template_version: str,
        schedule_resolver_version: str,
    ) -> InspectionTemplateRegistration:
        registration = self._registrations.get(
            (template_id, template_version, schedule_resolver_version)
        )
        if registration is None:
            raise validation_failed("巡检模板或 Resolver 版本未登记")
        return registration

    def validate_overrides(
        self,
        *,
        registration: InspectionTemplateRegistration,
        overrides: dict | None,
    ) -> None:
        if overrides is None:
            return
        unsupported = set(overrides) - set(registration.allowed_override_keys)
        if unsupported:
            raise validation_failed(
                "模板覆盖包含未允许字段：" + ", ".join(sorted(unsupported))
            )
