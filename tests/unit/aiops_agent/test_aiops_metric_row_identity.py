"""同一原生SQL返回身份和数值；普通身份探针不能替其他Exporter作证。"""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from pathlib import Path

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.contracts.evidence import ObservationSet
from aiops_agent.ports.diagnostic_source import MetricsEvidenceRequest
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from aiops_agent.domain.evidence.physical_identity import build_physical_identity_graph
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_physical_identity_graph import monitoring, direct, IDENTITIES, NOW


def row_metric(engine, *, stale=False, duplicate=False):
    _, _, base, adapter = monitoring(engine)
    labels = {"metric_identity_version": "1", "database_type": engine, "target_key": "mes",
        "job": "database-native", "instance": "collector:9399", **IDENTITIES[engine]}
    point_at = NOW-timedelta(seconds=30)
    scrape = NOW-timedelta(seconds=180 if stale else 35)
    clocks = [(labels, [(point_at, str(int(scrape.timestamp())))])]
    if duplicate: clocks *= 2
    adapter._query_range = AsyncMock(side_effect=[
        ([(labels, [(point_at, "7")])], "a"*64), (clocks, "b"*64)])
    request = MetricsEvidenceRequest(target_id=str(base.target_id), binding_id=str(base.binding_id),
        source_locator_key="mes", metric_definitions=(load_metric_catalog().get("db.native.active_sessions"),),
        window_start=base.observations[0].window_start, window_end=base.observations[0].window_end,
        requested_step_seconds=60, max_response_bytes=262144, trace_id="metric-row")
    observed = asyncio.run(adapter.query_metrics(request))
    result = ObservationSet(target_id=base.target_id, source_id=base.source_id, binding_id=base.binding_id,
        observations=observed.observations, gaps=observed.gaps, collected_at=NOW)
    return result, adapter


@pytest.mark.parametrize("engine", IDENTITIES)
def test_native_metric_row_keeps_exact_identity_value_and_actual_scrape_time(engine):
    result, adapter = row_metric(engine)
    assert not result.gaps and result.observations[0].series[0].points[0].value == 7
    fact = DbaEvidenceAssessmentHandler._monitoring_fact(artifact_id=str(uuid7()), result=result)
    witness, = fact.source_identity_witnesses
    assert witness.protocol_version == "DATABASE_METRIC_ROW.v1"
    assert witness.sample_times == (NOW-timedelta(seconds=35),)
    assert dict(witness.native_identity) == IDENTITIES[engine]
    assert adapter._query_range.call_args_list[1].kwargs["query"].startswith("timestamp(aiops_database_active_sessions")
    database, _ = direct(engine, result.target_id)
    edge, = build_physical_identity_graph((database, fact)).edges
    assert edge.relation != "UNKNOWN"
    assert "ONLY_THIS_NATIVE_QUERY_METRIC_REPORTED" in edge.reasons
    assert not edge.window_continuity_verified and not edge.independent_corroboration


@pytest.mark.parametrize("engine", IDENTITIES)
@pytest.mark.parametrize("failure", ["STALE", "DUPLICATE"])
def test_unverified_scrape_time_does_not_get_a_metric_assignment(engine, failure):
    result, _ = row_metric(engine, stale=failure=="STALE", duplicate=failure=="DUPLICATE")
    assert not result.observations and result.gaps


@pytest.mark.parametrize("engine", IDENTITIES)
def test_deployment_metric_and_identity_are_in_one_statement_without_secrets(engine):
    path = Path(__file__).resolve().parents[3]/"scripts/deployment/aiops_observability/configuration/identity"/(engine.lower()+"-metrics.collector.yml")
    config = yaml.safe_load(path.read_text())
    metric, = config["metrics"]
    assert metric["metric_name"] == "aiops_database_active_sessions"
    assert "active_sessions" in metric["query"] and "metric_identity_version" in metric["query"]
    assert "password" not in metric["query"].lower()
