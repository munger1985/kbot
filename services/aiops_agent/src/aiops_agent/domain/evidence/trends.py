"""监控时间序列的有界精确趋势摘要；不将采样分位数当执行延迟分位数。"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, localcontext
from statistics import median

from aiops_agent.domain.evidence.observations import numeric_wire, parse_numeric_sample


MAX_TREND_SAMPLES = 4096
MAX_SLOPE_REPRESENTATIVES = 128


def normalized_samples(samples):
    """相同时间同值去重，冲突、无时区、非法数字或超预算时拒绝计算。"""
    if len(samples) > MAX_TREND_SAMPLES:
        return None
    by_time = {}
    for observed_at, raw in samples:
        number = parse_numeric_sample(raw)
        if not isinstance(observed_at, datetime) or observed_at.tzinfo is None or number is None:
            return None
        stamp = observed_at.astimezone(UTC)
        if stamp in by_time and by_time[stamp] != number:
            return None
        by_time[stamp] = number
    return sorted(by_time.items())


def decimal_seconds(delta):
    return Decimal(delta.days * 86400 + delta.seconds) + Decimal(delta.microseconds) / 1000000


def summarize_numeric_trend(samples, *, forecast_horizon_days=None):
    """日内末值代表样本与有界 Theil-Sen 斜率，所有公式保留精确十进制。"""
    ordered = normalized_samples(samples)
    if ordered is None or len(ordered) < 2:
        return None
    horizon = parse_numeric_sample(forecast_horizon_days)
    if forecast_horizon_days is not None and (horizon is None or horizon <= 0 or horizon > 365):
        return None
    precision = max(80, 2 * max(
        len(value.as_tuple().digits) + abs(value.as_tuple().exponent) for _, value in ordered
    ) + 40)
    with localcontext() as context:
        context.prec = precision
        elapsed_days = decimal_seconds(ordered[-1][0] - ordered[0][0]) / 86400
        if elapsed_days <= 0:
            return None
        representatives = ordered
        if elapsed_days >= 2:
            daily_last = {}
            for sample in ordered:
                daily_last[sample[0].date().isoformat()] = sample
            representatives = list(daily_last.values())
        # 等距索引缩小斜率对数，包含首尾；不对原始均值和极值做降采样。
        if len(representatives) > MAX_SLOPE_REPRESENTATIVES:
            representatives = [representatives[
                index * (len(representatives) - 1) // (MAX_SLOPE_REPRESENTATIVES - 1)
            ] for index in range(MAX_SLOPE_REPRESENTATIVES)]
        changes = [right[1] - left[1] for left, right in zip(representatives, representatives[1:])]
        slopes = [
            (right[1] - left[1]) / (decimal_seconds(right[0] - left[0]) / 86400)
            for index, left in enumerate(representatives[:-1])
            for right in representatives[index + 1:]
        ]
        first, latest = ordered[0][1], ordered[-1][1]
        result = {
            "first": first, "latest": latest,
            "minimum": min(value for _, value in ordered),
            "average": sum((value for _, value in ordered), Decimal(0)) / len(ordered),
            "maximum": max(value for _, value in ordered),
            "change": latest - first,
            "change_per_day": (latest - first) / elapsed_days,
            "trend_slope_per_day": median(slopes) if slopes else Decimal(0),
            "positive_change_ratio": Decimal(sum(value > 0 for value in changes)) / len(changes) if changes else Decimal(0),
            "sample_count": len(ordered), "representative_sample_count": len(representatives),
            "elapsed_days": elapsed_days,
        }
        if horizon is not None:
            change = result["trend_slope_per_day"] * horizon
            result.update(forecast_horizon_days=horizon, forecast_change=change, forecast_value=latest + change)
        return {key: numeric_wire(value) if isinstance(value, Decimal) else value for key, value in result.items()}

_TREND_FIELDS = (
    "metric_code",
    "dimensions",
    "unit",
    "first",
    "latest",
    "change",
    "change_per_day",
    "trend_slope_per_day",
    "history_elapsed_days",
)
_REQUIRED_TREND_FIELDS = ("metric_code", "first", "latest", "change_per_day")


def _column_name(column) -> str:
    if isinstance(column, dict):
        return str(column.get("name") or "")
    return str(getattr(column, "name", "") or "")


def extract_metric_trend_rows(evidence) -> tuple[dict[str, float | int | str], ...]:
    """从 metric.query_range 证据抽出服务端趋势字段，不重新计算。"""
    rows: list[dict[str, float | int | str]] = []
    for item in evidence or ():
        if getattr(item, "tool_id", None) != "metric.query_range":
            continue
        indexes = {
            _column_name(column): index
            for index, column in enumerate(getattr(item, "columns", ()) or ())
            if _column_name(column)
        }
        if any(field not in indexes for field in _REQUIRED_TREND_FIELDS):
            continue
        for row in getattr(item, "rows", ()) or ():
            first = row[indexes["first"]]
            latest = row[indexes["latest"]]
            change_per_day = row[indexes["change_per_day"]]
            if first is None or latest is None or change_per_day is None:
                continue
            payload: dict[str, float | int | str] = {
                "metric_code": row[indexes["metric_code"]],
                "dimensions": (
                    row[indexes["dimensions"]]
                    if "dimensions" in indexes and row[indexes["dimensions"]] not in {None, ""}
                    else "-"
                ),
                "first": first,
                "latest": latest,
                "change_per_day": change_per_day,
            }
            for field in _TREND_FIELDS:
                if field in payload or field not in indexes:
                    continue
                value = row[indexes[field]]
                if value is not None:
                    payload[field] = value
            rows.append(payload)
    return tuple(rows)
