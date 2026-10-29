"""身份与指标值同一原生查询返回；另外核对实际Prometheus抓取时点。"""

from datetime import UTC, datetime, timedelta

from aiops_agent.domain.evidence.observations import parse_numeric_sample
from aiops_agent.playbooks import canonical_hash
from .base import DiagnosticSourceAdapterError

ROW_METRIC = "db.native.active_sessions"


async def metric_row_provenance(adapter, *, request, query, step, raw_series, response_hash):
    clocks, clock_hash = await adapter._query_range(request=request, query="timestamp(" + query + ")", step=step)
    by_labels = {}
    for labels, points in clocks:
        identity = tuple(sorted((key, value) for key, value in labels.items() if key != "__name__"))
        if identity in by_labels:
            raise DiagnosticSourceAdapterError("SOURCE_ITEM_AMBIGUOUS", "同连接指标抓取时点存在重复序列")
        by_labels[identity] = dict(points)
        if len(by_labels[identity]) != len(points):
            raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "同连接指标存在重复求值时点")
    times = []
    seen = set()
    if len(raw_series) > 16:
        raise DiagnosticSourceAdapterError("SOURCE_RESULT_TRUNCATED", "同连接指标身份序列超过预算")
    for labels, points in raw_series:
        identity = tuple(sorted((key, value) for key, value in labels.items() if key != "__name__"))
        if identity in seen or len({when for when, _ in points}) != len(points):
            raise DiagnosticSourceAdapterError("SOURCE_ITEM_AMBIGUOUS", "同连接指标存在重复值序列或求值时点")
        seen.add(identity)
        clock = by_labels.get(identity, {})
        if len(points) > 120 or labels.get("metric_identity_version") != "1":
            raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "同连接指标协议或预算无法核验")
        stamps = []
        for when, _ in points:
            number = parse_numeric_sample(clock.get(when))
            if number is None or not 0 <= number <= 10000000000:
                raise DiagnosticSourceAdapterError("SOURCE_RESPONSE_INVALID", "同连接指标缺少抓取时点")
            sampled = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=int(number * 1000000))
            if not when - timedelta(seconds=120) <= sampled <= when:
                raise DiagnosticSourceAdapterError("SOURCE_SAMPLE_STALE", "同连接指标抓取时点陈旧或来自未来")
            stamps.append(sampled.isoformat())
        times.append(stamps)
    return {"metric_row_protocol": "DATABASE_METRIC_ROW.v1", "metric_row_scrape_times": times,
        "metric_row_value_response_hash": response_hash, "metric_row_clock_response_hash": clock_hash,
        "provider_response_hash": canonical_hash((response_hash, clock_hash))}
