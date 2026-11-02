"""时序指标窗口统一使用 UTC 毫秒精度，冻结前收窄边界，不扩大授权范围。"""

from datetime import UTC, datetime, timedelta


def canonical_metric_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """起点向上、终点向下对齐毫秒；不调整来源返回的原始样本时间。"""
    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("时序指标窗口必须带时区")
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    remainder = start.microsecond % 1000
    if remainder:
        start += timedelta(microseconds=1000 - remainder)
    end -= timedelta(microseconds=end.microsecond % 1000)
    if start >= end:
        raise ValueError("时序指标窗口对齐毫秒后必须保留有效区间")
    return start, end


def prometheus_timestamp(value: datetime) -> str:
    """使用整数生成 Unix 秒参数，避免浮点序列化再次改变窗口边界。"""
    if value.utcoffset() is None:
        raise ValueError("Prometheus 查询时间必须带时区")
    delta = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    seconds, fraction = divmod(abs(micros), 1_000_000)
    return ("-" if micros < 0 else "") + f"{seconds}.{fraction:06d}"
