"""Top SQL 时间窗口输入的冻结和校验。"""

from datetime import UTC, datetime, timedelta
from typing import Any


TOP_SQL_WINDOW_TOOL = "workload.sql.top_window"
TOP_SQL_REFERENCE_KIND = "TOP_SQL_CUMULATIVE_REFERENCE"
TOP_SQL_REFERENCE_CONDITION = "TOP_SQL_WINDOW_UNAVAILABLE"
_PRESET_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "24h": 86400}


def freeze_top_sql_window(parameters: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    """规划时固定窗口，后续重试不重新计算相对时间。"""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("当前时间必须包含时区")
    mode = str(parameters.get("window") or "15m")
    if mode == "custom":
        start = _parse_time(parameters.get("begin_time"))
        end = _parse_time(parameters.get("end_time"))
    elif mode in _PRESET_SECONDS:
        end = now.astimezone(UTC)
        start = end - timedelta(seconds=_PRESET_SECONDS[mode])
    else:
        raise ValueError("不支持的 Top SQL 时间窗口")
    return validate_top_sql_window({
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "limit": parameters.get("limit", 10),
    })


def _parse_time(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Top SQL 时间格式无效") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Top SQL 时间必须包含时区")
    return parsed.astimezone(UTC)


def validate_top_sql_window(parameters: dict[str, Any]) -> dict[str, Any]:
    if set(parameters) != {"period_start", "period_end", "limit"}:
        raise ValueError("Top SQL 查询参数无效")
    start = _parse_time(parameters["period_start"])
    end = _parse_time(parameters["period_end"])
    limit = parameters["limit"]
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("Top SQL 返回数量必须为 1 到 50")
    if not start < end or end - start > timedelta(days=365):
        raise ValueError("Top SQL 时间窗口必须大于零且不超过 365 天")
    return {"period_start": start.isoformat(), "period_end": end.isoformat(), "limit": limit}
