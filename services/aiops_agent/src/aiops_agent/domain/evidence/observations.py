"""观测值的精确领域统计；采样分位数不等于执行延迟分位数。"""

import math
from decimal import Decimal, InvalidOperation, localcontext


def parse_numeric_sample(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or len(number.as_tuple().digits) > 1024 or abs(number.as_tuple().exponent) > 1024:
        return None
    return number


def numeric_wire(value):
    """浏览器不能精确表示的数值保留十进制字符串。"""
    if value is None:
        return None
    if value == value.to_integral_value() and abs(value) <= 9007199254740991:
        return int(value)
    approximate = float(value)
    if math.isfinite(approximate) and Decimal(str(approximate)) == value and abs(value) <= 9007199254740991:
        return approximate
    return str(value)


def summarize_points(series) -> dict[str, float | int | str | None]:
    samples = [
        (point.observed_at, number)
        for item in series for point in item.points
        if point.quality == "GOOD" and (number := parse_numeric_sample(point.value)) is not None
    ]
    if not samples:
        return {"count": 0, "min": None, "max": None, "avg": None, "p95": None, "last": None}
    values = [value for _, value in samples]
    ordered = sorted(values)
    # 有界精度随原始量级增加，避免 28 位默认精度舍掉计数器低位。
    precision = max(40, max(len(value.as_tuple().digits) + abs(value.as_tuple().exponent) for value in values) + len(str(len(values))) + 20)
    with localcontext() as context:
        context.prec = precision
        average = sum(values) / Decimal(len(values))
    latest_at = max(at for at, _ in samples)
    latest_values = {value for at, value in samples if at == latest_at}
    return {
        "count": len(values),
        "min": numeric_wire(ordered[0]), "max": numeric_wire(ordered[-1]),
        "avg": numeric_wire(average),
        "p95": numeric_wire(ordered[math.ceil(len(ordered) * 0.95) - 1]) if len(ordered) >= 20 else None,
        "last": numeric_wire(next(iter(latest_values))) if len(latest_values) == 1 else None,
    }
