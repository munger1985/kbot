"""日报七天参考比较：只消费冻结事实，不重判数据库健康。"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from math import ceil, fsum, isfinite
from statistics import median
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiops_agent.application.display_dictionary import format_metric_value


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _time(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(UTC) if result.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def inspection_reference_window(inspection, *, maximum_seconds):
    """日巡检额外读取此前七天监控，不扩大数据库现场检查的观察区间。"""
    if inspection.get("schedule_type") != "DAILY":
        return None
    start, end = _time(inspection.get("period_start")), _time(inspection.get("period_end"))
    if start is None or end is None or start >= end:
        return None
    reference_start = start - timedelta(days=7)
    return (reference_start, end) if (end - reference_start).total_seconds() <= maximum_seconds else None


def observation_coverage(fact, *, observation_start, observation_end):
    """扩展历史后单独计算本期覆盖率，不能因旧历史缺失否定本期检查。"""
    left, right = _time(observation_start), _time(observation_end)
    start, end = _time(fact.get("window_start")), _time(fact.get("window_end"))
    if not left or not right or not start or not end or not start < left < right <= end:
        return None
    points = {}
    for item in fact.get("sample_points") or ():
        stamp, number = _time(item.get("observed_at")), _number(item.get("value"))
        if stamp is None or number is None or not start <= stamp <= end or (stamp in points and points[stamp] != number):
            return 0.0
        points[stamp] = number
    stamps = sorted(points)
    if len(stamps) < 2:
        return 0.0
    step = median((b - a).total_seconds() for a, b in zip(stamps, stamps[1:]))
    if not 0 < step <= 3600:
        return 0.0
    slots = {int((stamp - left).total_seconds() // step) for stamp in points if left <= stamp < right}
    return min(1.0, len(slots) / max(1, ceil((right - left).total_seconds() / step)))


def daily_baseline_comparison(fact, *, observation_start=None, observation_end=None, timezone="UTC"):
    """优先比较本期均值；只有七天汇总时明确使用最新样本，缺历史不造数。"""
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo("UTC")
    unit = str(fact.get("unit") or "")
    applicable = str(fact.get("value_kind") or "") not in {"STATE", "COUNTER"} and unit != "state"
    result = {
        "baseline_applicable": applicable, "baseline_status": "INSUFFICIENT",
        "baseline_average": None, "baseline_display": "暂无可比基线",
        "baseline_reason": "未取得完整七天的可比历史。" if applicable else "状态或累计计数不做普通均值偏离比较。",
        "comparison_basis": "LATEST_SAMPLE", "comparison_label": "最新样本",
        "comparison_value": _number(fact.get("latest")),
        "comparison_display": format_metric_value(_number(fact.get("latest")), unit),
        "baseline_window_label": "", "deviation": None, "deviation_display": "暂无可比基线",
        "deviation_percent": None, "change_display": "暂无可比基线", "trend_points": [],
    }
    if not applicable:
        return result
    start, end = _time(fact.get("window_start")), _time(fact.get("window_end"))
    coverage = _number(fact.get("coverage_ratio"))
    if not start or not end or start >= end or coverage is None or not 0.8 <= coverage <= 1:
        return result
    if set(fact.get("series_quality_flags") or ()) & {
        "TRUNCATED", "COVERAGE_UNKNOWN", "INVALID_OR_OUTSIDE_SAMPLE", "SERIES_COMPUTATION_BUDGET_EXCEEDED",
    }:
        result["baseline_reason"] = "来源截断或采样口径不明，暂不计算基线偏离。"
        return result
    baseline, value = None, result["comparison_value"]
    baseline_start, baseline_end = end - timedelta(days=7), end
    raw = fact.get("sample_points") or ()
    if raw:
        points = {}
        for item in raw:
            if not isinstance(item, dict):
                return result
            stamp, number = _time(item.get("observed_at")), _number(item.get("value"))
            if stamp is None or number is None or not start <= stamp <= end:
                result["baseline_reason"] = "冻结样本存在无效值或时间，暂不计算基线偏离。"
                return result
            if stamp in points and points[stamp] != number:
                result["baseline_reason"] = "同一时刻样本值冲突，暂不计算基线偏离。"
                return result
            points[stamp] = number
        stamps = sorted(points)
        if len(stamps) < 2:
            return result
        step = median((right - left).total_seconds() for left, right in zip(stamps, stamps[1:]))
        if not 0 < step <= 3600:
            result["baseline_reason"] = "原始样本过于稀疏，暂不计算七天基线。"
            return result

        def samples(left, right):
            # 半开区间去重，按时间格覆盖率防止密集补点掩盖长时间缺测。
            selected = [(stamp, number) for stamp, number in points.items() if left <= stamp < right]
            slots = {int((stamp - left).total_seconds() // step) for stamp, _ in selected}
            enough = len(slots) / max(1, ceil((right - left).total_seconds() / step)) >= 0.8
            return selected if enough else []

        period_start, period_end = _time(observation_start), _time(observation_end)
        if period_start and period_end and period_start < period_end and start <= period_start - timedelta(days=7) and period_end <= end:
            baseline_start, baseline_end = period_start - timedelta(days=7), period_start
            current = samples(period_start, period_end)
            if not current:
                result["baseline_reason"] = "本期采样不足，暂不计算本期均值偏离。"
                return result
            value = fsum(number / len(current) for _, number in current)
            result.update(comparison_basis="OBSERVATION_MEAN", comparison_label="本期均值", comparison_value=value, comparison_display=format_metric_value(value, unit))
        if start > baseline_start:
            return result
        previous = samples(baseline_start, baseline_end)
        if not previous or any(not samples(baseline_start + timedelta(days=day), baseline_start + timedelta(days=day + 1)) for day in range(7)):
            result["baseline_reason"] = "七天历史存在缺测，暂不计算基线偏离。"
            return result
        baseline = fsum(number / len(previous) for _, number in previous)
        daily = defaultdict(list)
        for stamp, number in sorted(previous):
            daily[stamp.astimezone(tz).date().isoformat()].append(number)
        result["trend_points"] = [{"date": date, "value": fsum(number / len(values) for number in values)} for date, values in sorted(daily.items())]
    elif abs((end - start).total_seconds() - 7 * 86400) <= 1:
        # 只有冻结汇总时不声称取得了本期均值，也不把任意长度窗口改称七天。
        baseline = _number(fact.get("average"))
        result["trend_points"] = [
            {"date": str(item.get("date") or ""), "value": number}
            for item in fact.get("daily_points") or ()
            if isinstance(item, dict) and (number := _number(item.get("value"))) is not None and item.get("date")
        ]
    if baseline is None or value is None or not isfinite(baseline) or not isfinite(value):
        return result
    delta = value - baseline
    relative = delta / abs(baseline) * 100 if baseline != 0 else None
    if not isfinite(delta):
        result["trend_points"] = []
        return result
    if relative is not None and not isfinite(relative):
        relative = None
    signed = lambda number: ("+" if number > 0 else "") + format_metric_value(number, unit)
    change = f"{delta:+.2f} 个百分点" if unit == "percent" and delta else "0 个百分点" if unit == "percent" else signed(delta)
    relative_display = f"{relative:+.2f}%" if relative is not None else "基线为零，不计算变化率" if baseline == 0 else "变化率超出可表示范围"
    result.update(
        baseline_status="AVAILABLE", baseline_average=baseline,
        baseline_display=format_metric_value(baseline, unit), baseline_reason="趋势参考，不等于异常判断。",
        comparison_value=value, comparison_display=format_metric_value(value, unit),
        baseline_window_label=f"{baseline_start.astimezone(tz):%m-%d %H:%M} 至 {baseline_end.astimezone(tz):%m-%d %H:%M}",
        deviation=delta, deviation_display=change,
        deviation_percent=round(relative, 2) if relative is not None else None,
        change_display=f"{change}（{relative_display}）",
    )
    return result
