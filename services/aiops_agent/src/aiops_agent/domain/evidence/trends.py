"""监控时间序列的确定性趋势摘要。"""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import median


def summarize_numeric_trend(
    samples: tuple[tuple[datetime, float], ...],
) -> dict[str, float | int] | None:
    """按时间排序并生成可复核的变化速度与持续性指标。"""
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

    return {
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
