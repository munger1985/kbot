"""原生来源、精确计算、窗口质量与访问探测有效期回归。"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.contracts.measurements import ScalarMeasurement
from aiops_agent.domain.measurements import derive_measurement, independent_origin_groups
from aiops_agent.tools.capabilities import _source_freshness
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.application.diagnosis.provenance import database_provenance, monitoring_provenance
from platform_core.identity import uuid7


NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)


def scalar(**overrides):
    return ScalarMeasurement.model_validate({
        "input_ref": "artifact:raw#row/0/value", "series_identity": (("database_oid", "16510"), ("query_id", "1285382359565720997")),
        "native_origin_id": "pg_stat_statements:instance-boot-1", "source_version": "1.10",
        "reset_marker": "reset-1", "observed_at": NOW, "value": "900719925474099312345",
        "unit": "execution", "kind": "ADDITIVE_COUNTER", "quality": "GOOD", **overrides,
    })


def test_counter_delta_and_rate_preserve_integer_precision():
    before = scalar()
    after = scalar(observed_at=NOW + timedelta(seconds=10), value=str(before.value + 3))
    delta = derive_measurement("AFTER_MINUS_BEFORE", before, after)
    rate = derive_measurement("DELTA_PER_SECOND", before, after)
    assert delta.value == 3 and rate.value == Decimal("0.3")
    assert delta.model_dump(mode="json")["inputs"][0]["value"] == "900719925474099312345"
    assert delta.native_origin_ids == (before.native_origin_id,)


def test_counter_precision_exceeds_process_default_decimal_context():
    prefix = "1234567890" * 6
    before = scalar(value=prefix + "001")
    after = scalar(value=prefix + "004", observed_at=NOW + timedelta(seconds=10))
    assert derive_measurement("AFTER_MINUS_BEFORE", before, after).value == 3
    assert derive_measurement("DELTA_PER_SECOND", before, after).value == Decimal("0.3")


@pytest.mark.parametrize("change", [
    {"kind": "MAXIMUM"}, {"kind": "PERCENTILE"}, {"reset_marker": "reset-2"},
    {"reset_marker": None}, {"source_version": "2"}, {"quality": "TRUNCATED"},
    {"series_identity": (("query_id", "other"),)}, {"value": "1"},
    {"trust_level": "USER_PROVIDED"}, {"observed_at": NOW - timedelta(seconds=1)},
])
def test_nonadditive_incomparable_reset_or_untrusted_inputs_cannot_generate_delta(change):
    before = scalar()
    after = scalar(observed_at=NOW + timedelta(seconds=10), **change) if "observed_at" not in change else scalar(**change)
    result = derive_measurement("AFTER_MINUS_BEFORE", before, after)
    assert result.value is None and result.quality != "GOOD"


def test_weighted_mean_uses_totals_and_counts_not_average_of_averages():
    total = scalar(kind="GAUGE", unit="millisecond", value="1000")
    count = scalar(kind="GAUGE", unit="execution", value="4")
    result = derive_measurement("TOTAL_DIVIDED_BY_COUNT", total, count)
    assert result.value == 250 and result.unit == "millisecond/execution"
    invalid = derive_measurement("TOTAL_DIVIDED_BY_COUNT", total, scalar(kind="GAUGE", value=0))
    assert invalid.value is None
    missing = derive_measurement("AFTER_MINUS_BEFORE", scalar(value=None, quality="MISSING"), scalar())
    assert missing.value is None and missing.quality == "MISSING"


def test_wire_evidence_does_not_round_large_ids_or_decimal_counters():
    fact = TurnEvidenceFact(
        evidence_ref="artifact:raw#row", artifact_id=str(uuid7()), source_id="db", step_id="a1", tool_id="db.statement",
        measurement_semantics="CUMULATIVE_SINCE_LOAD", presentation_kind="TABLE", captured_at=NOW.isoformat(),
        columns=({"name": "query_id"}, {"name": "value"}), rows=((1285382359565720997, Decimal("1.000000000000000001")),), row_count=1,
    )
    assert fact.model_dump(mode="json")["rows"] == [["1285382359565720997", "1.000000000000000001"]]


def test_expired_or_rotated_config_is_not_current_access_guarantee():
    source = SimpleNamespace(last_connectivity_check_at=NOW, updated_at=NOW, row_version=2, connectivity_status="CONNECTED")
    assert _source_freshness(source, NOW + timedelta(seconds=1))["health_state"] == "AVAILABLE"
    assert _source_freshness(source, NOW + timedelta(minutes=6))["health_state"] == "UNKNOWN"
    source.updated_at += timedelta(seconds=1)
    assert _source_freshness(source, NOW + timedelta(seconds=2))["health_state"] == "UNKNOWN"


def test_db_and_exporter_without_native_mapping_are_not_independent_confirmation():
    target = str(uuid7())
    observation = SimpleNamespace(provenance={}, target_id=target, db_version="16", variant="pg16",
        template_sha256="a" * 64, result_sha256="b" * 64, truncated=False, captured_at=NOW, duration_ms=2)
    db = database_provenance(observation)
    monitor = monitoring_provenance(SimpleNamespace(
        target_id=target, observations=(), model_dump=lambda **kw: {"target_id": target},
    ))
    assert not db.physical_identity
    groups = independent_origin_groups([SimpleNamespace(provenance=db), SimpleNamespace(provenance=monitor)])
    assert len(groups) == 1
