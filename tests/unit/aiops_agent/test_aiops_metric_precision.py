"""监控采样不得在归一化或摘要时丢失计数器精度。"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from aiops_agent.contracts.evidence import MetricPoint, MetricSeries
from aiops_agent.domain.evidence.observations import numeric_wire, parse_numeric_sample, summarize_points


def test_large_integer_and_decimal_keep_exact_wire_values():
    assert numeric_wire(parse_numeric_sample("9007199254740993")) == "9007199254740993"
    assert numeric_wire(parse_numeric_sample("0.12345678901234567890123456789")) == "0.12345678901234567890123456789"
    assert numeric_wire(parse_numeric_sample("3.5")) == 3.5
    for raw in ("NaN", "Infinity", "not a metric", None, True):
        assert parse_numeric_sample(raw) is None


def test_summary_accepts_exact_numeric_strings_and_uses_latest_time_not_row_order():
    end = datetime.now(timezone.utc)
    series = (MetricSeries(points=(
        MetricPoint(observed_at=end, value="9007199254740995"),
        MetricPoint(observed_at=end - timedelta(seconds=1), value="9007199254740993"),
        MetricPoint(observed_at=end, value="NaN", quality="INVALID"),
    )),)
    summary = summarize_points(series)
    assert summary["count"] == 2
    assert summary["last"] == "9007199254740995"
    assert Decimal(summary["avg"]) == Decimal("9007199254740994")
    assert summary["p95"] is None


def test_last_sample_is_unknown_when_multiple_series_disagree_at_same_time():
    now = datetime.now(timezone.utc)
    series = tuple(MetricSeries(points=(MetricPoint(observed_at=now, value=value),)) for value in (1, 2))
    assert summarize_points(series)["last"] is None
