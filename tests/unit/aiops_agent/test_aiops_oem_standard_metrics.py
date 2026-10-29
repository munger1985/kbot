"""标准OEM指标必须使用官方接口、显式单位/维度映射及未知覆盖语义。"""

import asyncio
import copy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters.diagnostic_sources.registry import DiagnosticSourceAdapterCatalog
from aiops_agent.application.diagnosis.monitoring_calculations import build_monitoring_rows, validate_monitoring_trace_sources
from aiops_agent.application.diagnosis.provenance import monitoring_provenance
from aiops_agent.contracts.evidence import MetricDefinition, ObservationSet
from aiops_agent.ports.diagnostic_source import MetricsEvidenceRequest
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from aiops_agent.playbooks import canonical_hash
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import native_adapter, collection, series, GUID, NOW


def mapping():
    return {"target_type": "oracle_database", "metric_group": "Filesystems", "metric_name": "size",
        "native_unit": "MB", "unit": "bytes", "value_kind": "GAUGE", "scale": "1048576",
        "dimensions": {"mountPoint": "mount_point"}}


def request():
    definition = MetricDefinition(metric_code="db.storage.used", semantic_role="STORAGE_USED",
        semantic_version="1", name="已用容量", description="按精确挂载点返回", unit="bytes", value_kind="GAUGE",
        expected_dimensions=("mount_point",), default_window_seconds=900, min_step_seconds=60,
        max_points=100, max_series=10, providers={"OEM": {"template_id": "oem.storage.used",
        "template_version": "1", "target_type": "oracle_database", "metric_name": "Used"}})
    return MetricsEvidenceRequest(target_id=str(uuid7()), binding_id=str(uuid7()), source_locator_key=GUID,
        metric_definitions=(definition,), window_start=NOW-timedelta(minutes=15), window_end=NOW,
        requested_step_seconds=60, max_response_bytes=262144, trace_id="standard-test")


def configured(raw=None, payload=None):
    value, calls = native_adapter(payload)
    value.context = value.context.model_copy(update={"config": {
        "standard_metric_mappings": {"db.storage.used": mapping() if raw is None else raw}}})
    return value, calls


def test_unmapped_metric_is_an_actionable_gap_without_calling_an_invalid_legacy_endpoint():
    value, calls = native_adapter()
    result = asyncio.run(value.query_metrics(request()))
    assert not result.observations and result.gaps[0].code == "SOURCE_MAPPING_REQUIRED"
    assert not calls


def test_official_standard_path_preserves_large_values_and_never_manufactures_coverage():
    value, calls = configured(payload=collection([series("/"), series("/data")]))
    result = asyncio.run(value.query_metrics(request()))
    assert not result.gaps
    observed, = result.observations
    assert observed.series[0].points[0].value == str(90071992547409931 * 1048576)
    assert observed.series[0].dimensions == {"mount_point": "/"}
    assert observed.coverage_ratio == 0 and observed.expected_points == 0 and observed.actual_points == 2
    assert observed.provenance["native_unit"] == "MB"
    assert observed.provenance["history_resolution"] == "PROVIDER_MANAGED"
    assert len(calls) == 3 and all("/types/" not in url for url, _ in calls)
    assert calls[-1][0].endswith("/em/api/metricTimeSeries")


@pytest.mark.parametrize("field,bad,code", [
    ("native_unit", "MiB", "SOURCE_UNIT_MISMATCH"),
    ("unit", "seconds", "SOURCE_MAPPING_MISMATCH"),
    ("value_kind", "STATE", "SOURCE_CONFIG_INVALID"),
    ("target_type", "host", "SOURCE_MAPPING_MISMATCH"),
    ("dimensions", {}, "SOURCE_DIMENSIONS_MISMATCH"),
    ("dimensions", {"mountPoint": "other"}, "SOURCE_MAPPING_MISMATCH"),
])
def test_bad_native_mapping_is_not_a_successful_empty_or_mislabelled_observation(field, bad, code):
    raw = mapping()
    raw[field] = bad
    value, _ = configured(raw)
    result = asyncio.run(value.query_metrics(request()))
    assert not result.observations and result.gaps[0].code == code


@pytest.mark.parametrize("change", [
    {"scale": "NaN"}, {"scale": "0"}, {"scale": "1e9"}, {"scale": 10},
    {"scale": "0.0000000000000000001"}, {"native_unit": "bytes"}, {"unexpected": True},
    {"dimensions": {"a": "x", "b": "x"}}, {"filters": {"mountPoint": chr(10)}},
])
def test_invalid_or_ambiguous_configuration_is_rejected_before_persistence(change):
    raw = {**mapping(), **change}
    with pytest.raises((ValueError, TypeError)):
        DiagnosticSourceAdapterCatalog.normalize_config(source_type="OEM",
            config={"standard_metric_mappings": {"db.storage.used": raw}})


def test_exact_filter_does_not_merge_two_native_dimensions_or_sum_capacity():
    raw = mapping()
    raw.update(dimensions={}, filters={"mountPoint": "/data"})
    value, _ = configured(raw, collection([series("/"), series("/data")]))
    result = asyncio.run(value.query_metrics(request()))
    observed, = result.observations
    assert len(observed.series) == 1 and observed.actual_points == 1


def test_rollup_limitations_survive_computation_and_independent_source_replay():
    value, _ = configured()
    req = request()
    observed = asyncio.run(value.query_metrics(req))
    source = ObservationSet(target_id=req.target_id, binding_id=req.binding_id, source_id=value.context.source_id,
        observations=observed.observations, gaps=observed.gaps, collected_at=NOW)
    artifact_id, run_id = uuid7(), uuid7()
    fact = DbaEvidenceAssessmentHandler._monitoring_fact(artifact_id=str(artifact_id), result=source)
    assert "PROVIDER_RETENTION_ROLLUP_POSSIBLE" in monitoring_provenance(source).quality_flags
    assert "COVERAGE_UNKNOWN" in fact.series_computations[0].quality_flags
    assert fact.series_computations[0].forecast_horizon_days is None
    artifact = SimpleNamespace(ops_run_id=run_id, schema_version="OBSERVATION_SET.v1",
        payload_json=source.model_dump(mode="json"), content_hash=canonical_hash(source.model_dump(mode="json")))
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=artifact)))
    payload = {"evidence": [fact.model_dump(mode="json")]}
    assert asyncio.run(validate_monitoring_trace_sources(uow=uow, run=SimpleNamespace(ops_run_id=run_id,
        target_id=req.target_id), payload=payload))
    corrupted = copy.deepcopy(payload)
    corrupted["evidence"][0]["series_computations"][0]["quality_flags"].remove("COVERAGE_UNKNOWN")
    assert not asyncio.run(validate_monitoring_trace_sources(uow=uow, run=SimpleNamespace(ops_run_id=run_id,
        target_id=req.target_id), payload=corrupted))
