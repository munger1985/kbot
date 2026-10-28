"""时间序列计算的可重放输出，不以累计最大值或累计计数器外推窗口。"""

from decimal import Decimal, localcontext
from math import ceil

from aiops_agent.domain.evidence.observations import numeric_wire, parse_numeric_sample
from aiops_agent.domain.evidence.trends import normalized_samples, summarize_numeric_trend


def summarize_series(samples, *, value_kind, forecast_horizon_days=None):
    ordered = normalized_samples(samples)
    if not ordered:
        return None
    numbers = [number for _, number in ordered]
    with localcontext() as context:
        context.prec = max(80, 2 * max(len(value.as_tuple().digits) + abs(value.as_tuple().exponent) for value in numbers) + 40)
        daily = {}
        for stamp, number in ordered:
            daily.setdefault(stamp.date().isoformat(), []).append(number)
        result = {
            "latest": numeric_wire(ordered[-1][1]),
            "average": numeric_wire(sum(numbers, Decimal(0)) / len(numbers)),
            "minimum": numeric_wire(min(numbers)), "maximum": numeric_wire(max(numbers)),
            "sample_count": len(numbers),
            "sample_p95": numeric_wire(sorted(numbers)[ceil(len(numbers) * Decimal("0.95")) - 1]) if len(numbers) >= 20 else None,
        }
        trend = summarize_numeric_trend(tuple(ordered), forecast_horizon_days=forecast_horizon_days) if value_kind == "GAUGE" else None
        if trend is not None:
            result.update(trend)
        daily_points = tuple(
            {"date": day, "value": numeric_wire(sum(values, Decimal(0)) / len(values))}
            for day, values in sorted(daily.items())
        )
    return result, daily_points


def replay_series_computation(trace):
    from aiops_agent.contracts.series_calculation import SeriesComputation
    trace = SeriesComputation.model_validate(trace.model_dump(mode="python") if isinstance(trace, SeriesComputation) else trace)
    result = summarize_series(trace.samples, value_kind=trace.value_kind, forecast_horizon_days=trace.forecast_horizon_days)
    if result is None or result[0] != trace.summary:
        raise ValueError("序列计算与冻结输入重放结果不一致")
    return trace.summary


def series_number(summary, key):
    return parse_numeric_sample(summary.get(key))
