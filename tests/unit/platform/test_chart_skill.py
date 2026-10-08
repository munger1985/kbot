"""跨 App Chart Skill 的确定性合同测试。"""

from datetime import UTC, datetime, timedelta

from platform_core.visualization import ChartSkill


def test_time_series_normalizes_bytes_and_preserves_extrema() -> None:
    start = datetime(2026, 10, 1, tzinfo=UTC)
    points = [
        {
            "x": (start + timedelta(minutes=i)).isoformat(),
            "y": i * 1024 * 1024,
        }
        for i in range(480)
    ]
    points[212]["y"] = 900 * 1024 * 1024

    chart = ChartSkill.time_series(
        title="表空间使用量趋势",
        unit="bytes",
        series=[{"series_key": "SYSAUX", "name": "SYSAUX", "points": points}],
        source_ids=("evidence-1",),
    )

    assert chart.schema_version == "CHART_SPEC.v1"
    assert chart.chart_type == "LINE"
    assert chart.y_axis.unit == "MiB"
    assert len(chart.series[0].points) <= 240
    assert max(point.y for point in chart.series[0].points) == 900
    assert chart.source_ids == ("evidence-1",)


def test_capacity_distinguishes_current_free_and_expandable_headroom() -> None:
    chart = ChartSkill.capacity(
        title="表空间容量",
        source_ids=("evidence-2",),
        rows=[
            {
                "tablespace_name": "TEST01",
                "allocated_mb": 30,
                "used_mb": 29.06,
                "maximum_mb": 30,
            },
            {
                "tablespace_name": "SYSTEM",
                "allocated_mb": 440,
                "used_mb": 430.13,
                "maximum_mb": 32_000,
            },
        ],
    )

    assert chart.chart_type == "STACKED_BAR"
    assert chart.layout == "NORMALIZED"
    assert chart.series[2].points[0].y == 0
    assert chart.series[0].points[0].metadata["allocated_used_percent"] == 96.87
    assert chart.series[2].points[1].y > 98


def test_tabular_infers_iso_datetime_dimension_without_logical_type() -> None:
    chart = ChartSkill.tabular(
        title="请求量趋势",
        columns=(
            {"name": "observed_at", "type": "STRING"},
            {"name": "request_count", "type": "INTEGER"},
        ),
        rows=(
            {"observed_at": "2026-10-08T00:00:00Z", "request_count": 12},
            {"observed_at": "2026-10-08T01:00:00Z", "request_count": 18},
        ),
        source_ids=("query-result-1",),
    )

    assert chart.chart_type == "LINE"
    assert chart.intent == "TREND"
    assert chart.x_axis.axis_type == "TIME"
    assert chart.series[0].series_key == "request_count"
