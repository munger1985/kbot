"""观测值的纯领域统计函数。"""

import math
from statistics import fmean


def summarize_points(series) -> dict[str, float | int | None]:
    values = [
        float(point.value)
        for item in series
        for point in item.points
        if point.quality == "GOOD"
        and isinstance(point.value, (int, float))
        and not isinstance(point.value, bool)
        and math.isfinite(float(point.value))
    ]
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "avg": None,
            "p95": None,
            "last": None,
        }
    ordered = sorted(values)
    p95 = (
        ordered[math.ceil(len(ordered) * 0.95) - 1]
        if len(ordered) >= 20
        else None
    )
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": fmean(values),
        "p95": p95,
        "last": values[-1],
    }
