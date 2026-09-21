"""监控时间序列的确定性趋势摘要。"""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import median


def summarize_numeric_trend(
    samples: tuple[tuple[datetime, float], ...],
    *,
    forecast_horizon_days: float | None = None,
) -> dict[str, float | int] | None:
    """先汇总历史变化，再按指定未来窗口生成可复核线性预测。"""
    ordered = sorted(
        (
            (observed_at.astimezone(UTC), float(value))
            for observed_at, value in samples
        ),
        key=lambda item: item[0],
    )
    if len(ordered) < 2:
        return None

    elapsed_seconds = (ordered[-1][0] - ordered[0][0]).total_seconds()
    if elapsed_seconds <= 0:
        return None

    elapsed_days = elapsed_seconds / 86_400
    representatives = ordered
    if elapsed_days >= 2:
        daily_last: dict[str, tuple[datetime, float]] = {}
        for sample in ordered:
            daily_last[sample[0].date().isoformat()] = sample
        representatives = list(daily_last.values())

    first_value = representatives[0][1]
    latest_value = representatives[-1][1]
    changes = [
        current[1] - previous[1]
        for previous, current in zip(
            representatives,
            representatives[1:],
            strict=False,
        )
    ]
    positive_ratio = (
        sum(1 for value in changes if value > 0) / len(changes)
        if changes
        else 0.0
    )
    slopes = []
    for left_index, left in enumerate(representatives[:-1]):
        for right in representatives[left_index + 1 :]:
            days = (right[0] - left[0]).total_seconds() / 86_400
            if days > 0:
                slopes.append((right[1] - left[1]) / days)

    result: dict[str, float | int] = {
        "first": first_value,
        "latest": latest_value,
        "minimum": min(value for _, value in ordered),
        "average": sum(value for _, value in ordered) / len(ordered),
        "maximum": max(value for _, value in ordered),
        "change": latest_value - first_value,
        "change_per_day": (
            (latest_value - first_value) / elapsed_days
        ),
        "trend_slope_per_day": median(slopes) if slopes else 0.0,
        "positive_change_ratio": positive_ratio,
        "sample_count": len(ordered),
        "representative_sample_count": len(representatives),
        "elapsed_days": elapsed_days,
    }
    if forecast_horizon_days is not None and forecast_horizon_days > 0:
        forecast_change = result["trend_slope_per_day"] * forecast_horizon_days
        result.update(
            {
                "forecast_horizon_days": forecast_horizon_days,
                "forecast_change": forecast_change,
                "forecast_value": latest_value + forecast_change,
            }
        )
    return result

_TREND_FIELDS = (
    "metric_code",
    "dimensions",
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
