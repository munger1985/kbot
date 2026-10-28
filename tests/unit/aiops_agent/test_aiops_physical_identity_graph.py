"""三库监控身份原始采样、有限匹配与诊断/报告/案例独立来源重放。"""

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact
from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
from aiops_agent.application.diagnosis.physical_identity import source_identity_witnesses
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.monitoring_snapshot import MonitoringSnapshotBuilder
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.contracts.evidence import ObservationSet
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft, InvestigationRecord
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.contracts.tool_execution import DbaToolResult, ToolOutcome
from aiops_agent.domain.evidence.physical_identity import build_physical_identity_graph
from aiops_agent.playbooks import canonical_hash
from aiops_agent.ports.diagnostic_source import DiagnosticSourceContext, MetricsEvidenceRequest
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops.executor import DatabaseObservation
from platform_core.identity import uuid7


NOW = datetime(2026, 10, 28, 7, tzinfo=timezone.utc)
IDENTITIES = {
    "POSTGRESQL": {"system_identifier": "7694909560417873958", "database_oid": "16510",
                   "instance_started_at": "2026-10-10T05:39:06+00:00"},
    "MYSQL": {"server_uuid": "aa6b4e59-410b-4b92-aa59-b4f07bff7411"},
    "ORACLE": {"dbid": "3017315513", "instance_number": "1", "container_id": "3",
               "instance_started_at": "2026-10-10T05:39:06"},
}


def monitoring(engine, *, identity=None, points=None, extra_series=(), target=None, source=None, binding=None):
    target, source, binding = (str(value) for value in (target or uuid7(), source or uuid7(), binding or uuid7()))
    labels = {"identity_version": "1", "database_type": engine, "job": "database-identity",
              "instance": "collector:9399", "target_key": "mes", **(identity or IDENTITIES[engine])}
    context = DiagnosticSourceContext(source_id=str(source), source_type="PROMETHEUS", adapter_id="prometheus",
        adapter_version="1.0.0", config_version=1, endpoint="http://prometheus.invalid")
    adapter = PrometheusAdapter(context=context, session=SimpleNamespace(),
        request_timeout_seconds=10, webhook_replay_seconds=300)
    default_points = [(NOW - timedelta(seconds=30), str(int((NOW - timedelta(seconds=35)).timestamp())))]
    adapter._query_range = AsyncMock(return_value=(
        [(labels, points or default_points)] + list(extra_series),
        "a" * 64,
    ))
    request = MetricsEvidenceRequest(target_id=str(target), binding_id=str(binding), source_locator_key="mes",
        metric_definitions=(load_metric_catalog().get("db.identity.info"),),
        window_start=NOW - timedelta(minutes=15), window_end=NOW, requested_step_seconds=60,
        max_response_bytes=262144, trace_id="identity-test")
    observed = asyncio.run(adapter.query_metrics(request))
    result = ObservationSet(target_id=target, binding_id=binding, source_id=source,
        observations=observed.observations, gaps=observed.gaps, collected_at=NOW)
    artifact_id = str(uuid7())
    fact = DbaEvidenceAssessmentHandler._monitoring_fact(artifact_id=artifact_id, result=result)
    artifact = SimpleNamespace(artifact_id=artifact_id, schema_version="OBSERVATION_SET.v1",
        trust_level="SOURCE_VERIFIED", payload_json=result.model_dump(mode="json"),
        content_hash=canonical_hash(result.model_dump(mode="json")), ops_run_id=None)
    return fact, artifact, result, adapter


def direct(engine, target):
    observation = DatabaseObservation(
        executor_request_id=str(uuid7()), target_id=str(target), tool_id="db.instance.identity", tool_version="1.0.0",
        variant="identity-test", template_sha256="b" * 64, db_type=engine,
        db_version={"POSTGRESQL": "16", "MYSQL": "8", "ORACLE": "19"}[engine],
        capability_snapshot_hash="c" * 64, captured_at=NOW, duration_ms=1,
        columns=({"name": "database_name", "logical_type": "STRING", "sensitivity": "PUBLIC"},),
        rows=(("mes",),), row_count=1, truncated=False, result_sha256=canonical_hash((("mes",),)),
        parameters_sha256="d" * 64,
        provenance={"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": IDENTITIES[engine]},
    )
    outcome = ToolOutcome(step_id="a1", tool_id="db.instance.identity", tool_version="1.0.0",
        status="SUCCEEDED", observation=observation)
    result = DbaToolResult(source_type="TOOL", source_id="database.identity", source_version="1.0.0",
        definition_hash="e" * 64, output_schema="DBA_TOOL_RESULT.v1", measurement_semantics="CURRENT_ACTIVITY",
        presentation_kind="TABLE", status="SUCCEEDED", tool_outcomes=(outcome,))
    artifact_id = str(uuid7())
    fact = database_evidence_fact(result=result, outcome=outcome, artifact_id=artifact_id)
    payload = result.model_dump(mode="json")
    artifact = SimpleNamespace(artifact_id=artifact_id, schema_version="DBA_TOOL_RESULT.v1",
        trust_level="SOURCE_VERIFIED", payload_json=payload, content_hash=canonical_hash(payload), ops_run_id=None)
    return fact, artifact


def investigation(engine):
    monitor, m_artifact, result, _ = monitoring(engine)
    database, d_artifact = direct(engine, result.target_id)
    run = SimpleNamespace(ops_run_id=uuid7(), target_id=result.target_id)
    m_artifact.ops_run_id = d_artifact.ops_run_id = run.ops_run_id
    draft = DiagnosisAnswerDraft(claims=({
        "claim_id": "c1", "kind": "FACT", "subject": "目标实例", "statement": "当前连接数据库为mes。",
        "level": "OBSERVED", "applicable_time": NOW.isoformat(), "reasoning": "数据库直连返回库名。",
        "evidence_fields": [{"evidence_ref": database.evidence_ref, "row_index": 0, "field": "database_name"}],
    },), goal_satisfied=True)
    context = SimpleNamespace(target_id=run.target_id, run_id=run.ops_run_id, task_id=uuid7(),
        plan_snapshot={"answer_context": {"question": "核实数据库身份", "task_frame": {}}})
    record = freeze_investigation(draft=draft, evidence=(database, monitor), context=context)
    payload = {"status": "COMPLETED", "blocks": [], "evidence": [
        database.model_dump(mode="json"), monitor.model_dump(mode="json")],
        "investigation_record": record.model_dump(mode="json")}
    artifacts = {str(item.artifact_id): item for item in (d_artifact, m_artifact)}
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(
        side_effect=lambda *, artifact_id: artifacts.get(str(artifact_id)))))
    return run, record, payload, uow, m_artifact


@pytest.mark.parametrize("engine", IDENTITIES)
def test_canonical_identity_query_uses_scrape_timestamp_and_preserves_large_native_labels(engine):
    fact, _, result, adapter = monitoring(engine)
    query = adapter._query_range.call_args.kwargs["query"]
    assert query == 'timestamp(aiops_database_identity_info{target_key="mes",identity_version="1"})'
    witness, = fact.source_identity_witnesses
    assert dict(witness.native_identity) == IDENTITIES[engine]
    assert witness.sample_times == (NOW - timedelta(seconds=35),)
    assert ("job", "database-identity") in witness.collector_scope
    assert witness.database_type == engine
    assert source_identity_witnesses(artifact_id=fact.artifact_id, result=result) == fact.source_identity_witnesses
    assert load_metric_catalog().get("db.identity.info").value_kind == "STATE"


@pytest.mark.parametrize("engine", IDENTITIES)
def test_same_database_family_never_becomes_exact_runtime_or_independent_corroboration(engine):
    monitor, _, result, _ = monitoring(engine)
    database, _ = direct(engine, result.target_id)
    graph = build_physical_identity_graph((database, monitor))
    edge, = graph.edges
    assert edge.relation == ("REPORTED_SERVER_MATCH" if engine == "MYSQL" else "REPORTED_DATABASE_FAMILY_MATCH")
    assert not edge.exact_runtime_instance_verified
    assert not edge.metric_series_assignment_verified
    assert not edge.window_continuity_verified and not edge.independent_corroboration
    assert "OTHER_METRIC_SERIES_NOT_ASSIGNED" in edge.reasons


@pytest.mark.parametrize("engine,key,value", [
    ("POSTGRESQL", "system_identifier", "7694909560417873959"),
    ("MYSQL", "server_uuid", "bb6b4e59-410b-4b92-aa59-b4f07bff7411"),
    ("ORACLE", "instance_number", "2"),
    ("ORACLE", "container_id", "4"),
])
def test_same_logical_target_does_not_hide_native_identity_conflicts(engine, key, value):
    monitor, _, result, _ = monitoring(engine, identity={**IDENTITIES[engine], key: value})
    database, _ = direct(engine, result.target_id)
    edge, = build_physical_identity_graph((database, monitor)).edges
    assert edge.relation == "CONFLICT" and key in edge.conflicting_fields


@pytest.mark.parametrize("raw", ["0", "1", "NaN", "Infinity", str(int((NOW + timedelta(seconds=60)).timestamp()))])
def test_stale_or_invalid_scrape_timestamp_cannot_use_query_evaluation_time(raw):
    monitor, _, result, _ = monitoring("MYSQL", points=[(NOW, raw)])
    database, _ = direct("MYSQL", result.target_id)
    edge, = build_physical_identity_graph((database, monitor)).edges
    assert edge.relation == "UNKNOWN"
    assert not monitor.source_identity_witnesses[0].sample_times


def test_one_stable_family_and_boot_time_does_not_prove_pg_replica_runtime_identity():
    monitor, _, result, _ = monitoring("POSTGRESQL")
    database, _ = direct("POSTGRESQL", result.target_id)
    edge, = build_physical_identity_graph((monitor, database)).edges
    assert edge.relation == "REPORTED_DATABASE_FAMILY_MATCH"
    assert "DATABASE_FAMILY_IS_NOT_RUNTIME_NODE" in edge.reasons


def test_oracle_timezone_unknown_never_becomes_utc_comparison():
    monitor, _, result, _ = monitoring("ORACLE", identity={**IDENTITIES["ORACLE"], "instance_started_at": "2026-10-10T13:39:06"})
    database, _ = direct("ORACLE", result.target_id)
    edge, = build_physical_identity_graph((monitor, database)).edges
    assert edge.relation == "REPORTED_DATABASE_FAMILY_MATCH"
    assert "instance_started_at" not in (*edge.matched_fields, *edge.conflicting_fields)


def test_multiple_identity_values_in_one_collector_scope_remain_ambiguous():
    second = {"identity_version": "1", "database_type": "MYSQL", "job": "database-identity",
              "instance": "collector:9399", "target_key": "mes",
              "server_uuid": "bb6b4e59-410b-4b92-aa59-b4f07bff7411"}
    monitor, _, result, _ = monitoring("MYSQL", extra_series=((second, [(NOW, str(int(NOW.timestamp())))]),))
    database, _ = direct("MYSQL", result.target_id)
    graph = build_physical_identity_graph((database, monitor))
    assert len(graph.edges) == 2 and all(edge.relation == "UNKNOWN" for edge in graph.edges)
    assert all("COLLECTOR_IDENTITY_CHANGED_OR_AMBIGUOUS" in edge.reasons for edge in graph.edges)


@pytest.mark.parametrize("engine", IDENTITIES)
@pytest.mark.parametrize("tamper", [None, "raw_native", "raw_time", "raw_hash", "foreign_run", "untrusted_source",
                                   "remove_witness", "remove_graph", "forge_graph", "forge_monitoring_physical"])
def test_answer_record_report_and_case_replay_original_identity_sources(engine, tamper):
    run, record, payload, uow, artifact = investigation(engine)
    normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", payload=payload, source_kind="DIAGNOSIS")
    assert record.physical_identity_graph.edges
    if tamper in {"raw_native", "raw_time"}:
        series = artifact.payload_json["observations"][0]["series"][0]
        if tamper == "raw_native":
            anchor = {"POSTGRESQL": "system_identifier", "MYSQL": "server_uuid", "ORACLE": "dbid"}[engine]
            series["dimensions"][anchor] = "0"
        else:
            series["points"][0]["value"] = "0"
        artifact.content_hash = canonical_hash(artifact.payload_json)
    elif tamper == "raw_hash":
        artifact.content_hash = "f" * 64
    elif tamper == "foreign_run":
        artifact.ops_run_id = uuid7()
    elif tamper == "untrusted_source":
        artifact.trust_level = "USER_PROVIDED"
    elif tamper == "remove_witness":
        payload["evidence"][1]["source_identity_witnesses"] = []
        payload["investigation_record"]["observations"][1]["source_identity_witnesses"] = []
    elif tamper == "remove_graph":
        payload["investigation_record"]["physical_identity_graph"] = None
    elif tamper == "forge_graph":
        payload["investigation_record"]["physical_identity_graph"]["edges"][0]["relation"] = "UNKNOWN"
    elif tamper == "forge_monitoring_physical":
        payload["evidence"][1]["provenance"]["physical_identity"] = list(IDENTITIES[engine].items())
        payload["investigation_record"]["observations"][1]["provenance"]["physical_identity"] = list(IDENTITIES[engine].items())
    assert asyncio.run(validate_investigation_sources(uow=uow, run=run, payload=payload)) == (tamper is None)


@pytest.mark.parametrize("engine", IDENTITIES)
def test_general_and_sql_model_inputs_use_same_deterministic_identity_graph(engine):
    run, record, payload, _, _ = investigation(engine)
    from aiops_agent.contracts.turn_answer import TurnEvidenceFact
    facts = tuple(TurnEvidenceFact.model_validate(raw) for raw in payload["evidence"])
    model = DbaAnswerComposeHandler._model_assessment_payload(DbaSufficiencyAssessment(
        status="ANSWERABLE", evidence=facts))
    assert model["physical_identity_graph"] == record.physical_identity_graph.model_dump(mode="json")
    assert model["identity_interpretation_policy"]["database_family_is_not_runtime_instance"]
    subject = {"POSTGRESQL": {"query_id": "1285382359565720997"},
               "MYSQL": {"digest": "a" * 64}, "ORACLE": {"sql_id": "36apy6dncjggq"}}[engine]
    sql = build_sql_evidence_bundle(task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject},
        database_type=engine, evidence=facts)
    assert sql["physical_identity_graph"] == model["physical_identity_graph"]
    assert sql["interpretation_policy"]["identity_metric_does_not_assign_other_series"]
    record.validate_observations(facts)
    with pytest.raises(ValueError):
        record.model_copy(update={"physical_identity_graph": None}).validate_observations(facts)


@pytest.mark.parametrize("engine", IDENTITIES)
def test_identity_collector_assets_have_one_same_connection_native_query_and_no_credentials(engine):
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts/deployment/aiops_observability/configuration/identity" / (engine.lower() + ".collector.yml")
    body = yaml.safe_load(path.read_text())
    metric, = body["metrics"]
    assert metric["metric_name"] == "aiops_database_identity_info"
    assert set(IDENTITIES[engine]) <= set(metric["key_labels"])
    assert metric["values"] == ["identity_info"]
    assert "password" not in path.read_text().lower()
    assert " dsn" not in path.read_text().lower()


@pytest.mark.parametrize("field,value", [
    ("job", ""), ("instance", ""), ("job", "bad\nlabel"), ("identity_version", "2"), ("system_identifier", "1e18"), ("target_key", "foreign-target"),
    ("system_identifier", "7694909560417873958\n"),
])
def test_unusable_native_or_collector_labels_never_become_a_match(field, value):
    monitor, _, source, _ = monitoring("POSTGRESQL",
        identity={**IDENTITIES["POSTGRESQL"], field: value})
    database, _ = direct("POSTGRESQL", source.target_id)
    edge, = build_physical_identity_graph((database, monitor)).edges
    assert edge.relation == "UNKNOWN"


@pytest.mark.parametrize("flag", ["TRUNCATED", "IDENTITY_PROTOCOL_UNVERIFIED", "IDENTITY_SAMPLE_BUDGET_EXCEEDED"])
def test_incomplete_witness_does_not_prove_identity_even_when_native_ids_equal(flag):
    monitor, _, source, _ = monitoring("MYSQL")
    witness = monitor.source_identity_witnesses[0]
    monitor = monitor.model_copy(update={"source_identity_witnesses": (
        witness.model_copy(update={"quality_flags": (flag,)}),)})
    database, _ = direct("MYSQL", source.target_id)
    edge, = build_physical_identity_graph((database, monitor)).edges
    assert edge.relation == "UNKNOWN"


def test_old_direct_observation_is_not_current_identity_conflict():
    monitor, _, source, _ = monitoring("MYSQL", identity={
        "server_uuid": "bb6b4e59-410b-4b92-aa59-b4f07bff7411"})
    database, _ = direct("MYSQL", source.target_id)
    database = database.model_copy(update={"provenance": database.provenance.model_copy(update={
        "collected_start": NOW - timedelta(hours=1), "collected_end": NOW - timedelta(hours=1)})})
    edge, = build_physical_identity_graph((database, monitor)).edges
    assert edge.relation == "UNKNOWN" and "NO_NEARBY_IDENTITY_SAMPLE" in edge.reasons


def test_no_identity_exporter_does_not_discard_other_valid_database_evidence():
    database, _ = direct("POSTGRESQL", uuid7())
    graph = build_physical_identity_graph((database,))
    assert len(graph.nodes) == 1 and not graph.edges


def test_graph_budget_remains_explicit_and_retains_both_sources():
    monitor, _, source, _ = monitoring("MYSQL")
    database, _ = direct("MYSQL", source.target_id)
    facts = tuple(database.model_copy(update={"evidence_ref": f"{database.evidence_ref}-{index}"}) for index in range(80))
    monitors = tuple(monitor.model_copy(update={"evidence_ref": f"{monitor.evidence_ref}-{index}"}) for index in range(80))
    graph = build_physical_identity_graph((*facts, *monitors))
    assert len(graph.nodes) == 64 and len(graph.edges) == 128
    assert {"DIRECT_DATABASE", "MONITORING_REPORT"} == {node.origin for node in graph.nodes}
    assert "IDENTITY_NODE_BUDGET_EXCEEDED" in graph.quality_flags
    assert "IDENTITY_EDGE_BUDGET_EXCEEDED" in graph.quality_flags


@pytest.mark.parametrize("authorized", [False, True])
def test_identity_query_is_frozen_only_when_explicit_binding_metric_scope_allows_it(authorized):
    source_id, binding_id = uuid7(), uuid7()
    monitor = SimpleNamespace(
        diagnostic_source_id=source_id, target_source_binding_id=binding_id,
        capability_scope_json={"metric_codes": ["db.identity.info"] if authorized else ["postgresql.availability"],
                               "capabilities": ["metric.query_range"]},
        role="PRIMARY", priority=10, source_locator_key="mes", source_locator_json={"instance": "mes"},
        mapping_overrides_json={}, query_budget_json={}, row_version=1,
    )
    source = SimpleNamespace(diagnostic_source_id=source_id, tenant_id=uuid7(), status="ENABLED",
        source_type="PROMETHEUS", adapter_id="prometheus", adapter_version="1.0.0", row_version=1,
        connectivity_status="CONNECTED", endpoint="http://prometheus.invalid",
        auth_credential_id=None, declared_capabilities_json={"metric.query_range": {}}, config_json={})
    uow = SimpleNamespace(targets=SimpleNamespace(list_source_bindings=AsyncMock(return_value=[monitor])),
        diagnostic_sources=SimpleNamespace(get_scoped=AsyncMock(return_value=source)))
    snapshot = asyncio.run(MonitoringSnapshotBuilder(
        metric_catalog=load_metric_catalog(), default_window_seconds=900, max_response_bytes=262144,
    ).build(uow=uow, domain_id=uuid7(), target=SimpleNamespace(target_id=uuid7(), db_type="POSTGRESQL"),
        now=NOW, requested_metric_codes=("db.identity.info",)))
    metrics = snapshot["bindings"][0]["metrics"]
    assert bool(metrics) == authorized
    if authorized:
        assert metrics[0] == load_metric_catalog().get("db.identity.info").model_dump(mode="json")


def test_plain_monitoring_without_witnesses_cannot_claim_direct_physical_identity():
    monitor, _, source, _ = monitoring("MYSQL")
    monitor = monitor.model_copy(update={"source_identity_witnesses": (), "provenance": monitor.provenance.model_copy(
        update={"physical_identity": tuple(IDENTITIES["MYSQL"].items())})})
    assert not build_physical_identity_graph((monitor,)).nodes


def test_witness_source_is_replayed_even_after_removing_record_and_witness_flags():
    run, _, payload, uow, _ = investigation("MYSQL")
    payload.pop("investigation_record")
    payload["evidence"][1]["source_identity_witnesses"] = []
    assert not asyncio.run(validate_investigation_sources(uow=uow, run=run, payload=payload))


@pytest.mark.parametrize("engine", IDENTITIES)
@pytest.mark.parametrize("stage", ["assessment", "replan"])
def test_identity_graph_reaches_actual_assessor_and_replanner_before_answer(engine, stage):
    from aiops_agent.application.investigation.reasoner import InvestigationReasoner
    run, record, payload, uow, _ = investigation(engine)
    model = SimpleNamespace(generate_structured=AsyncMock(side_effect=RuntimeError("model-input-captured")))
    prompt = SimpleNamespace(content="来源核验", ref=lambda: {})
    prompts = SimpleNamespace(resolve=AsyncMock(return_value=prompt))
    if stage == "replan":
        operation = InvestigationReasoner(model, prompts).replan(
            content=(), conversation_context=(), target_context={"database_type": engine},
            prompt_snapshot={}, source_run_evidence=None, task_frame={}, prior_plan={},
            assessment={"evidence": payload["evidence"]}, available_tools=(), available_playbooks=(),
            model_snapshot={"test": "model"}, deadline=None, idempotency_key="identity", revision_no=2)
    else:
        artifacts = [asyncio.run(uow.runs.get_artifact(artifact_id=item["artifact_id"])) for item in payload["evidence"]]
        context = SimpleNamespace(target_id=run.target_id, run_id=run.ops_run_id, task_id=uuid7(), attempt=1, deadline_at=None, trigger_type="MANUAL",
            input_artifacts=tuple({"artifact_id": str(artifact.artifact_id),
                "schema_version": artifact.schema_version, "payload": artifact.payload_json} for artifact in artifacts),
            plan_snapshot={"capability_snapshot": {"database_type": engine}, "answer_context": {
                "model": {"test": "model"}, "prompts": {}, "task_frame": {"objectives": ["DIAGNOSE"]}}})
        operation = DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts).execute(context)
    with pytest.raises(RuntimeError, match="model-input-captured"):
        asyncio.run(operation)
    supplied = model.generate_structured.await_args.kwargs["input_payload"]
    assert supplied["physical_identity_graph"] == record.physical_identity_graph.model_dump(mode="json")
    assert supplied["identity_interpretation_policy"]["identity_metric_does_not_assign_other_series"]


def test_timezone_overflow_in_optional_identity_does_not_discard_primary_evidence():
    monitor, _, source, _ = monitoring("POSTGRESQL",
        identity={**IDENTITIES["POSTGRESQL"], "instance_started_at": "9999-12-31T23:59:59-23:59"})
    database, _ = direct("POSTGRESQL", source.target_id)
    edge, = build_physical_identity_graph((monitor, database)).edges
    assert edge.relation == "UNKNOWN"


def test_identity_scrape_timestamps_are_not_rendered_as_a_performance_chart():
    points = [(NOW - timedelta(seconds=60), str(int((NOW - timedelta(seconds=65)).timestamp()))),
              (NOW, str(int((NOW - timedelta(seconds=5)).timestamp())))]
    fact, _, _, _ = monitoring("MYSQL", points=points)
    assert fact.source_identity_witnesses and not fact.visualizations
    assert fact.presentation_kind == "TABLE"


@pytest.mark.parametrize("engine", IDENTITIES)
@pytest.mark.parametrize("with_record", [False, True])
@pytest.mark.parametrize("tamper", [None, "raw_identity", "drop_witness"])
def test_actual_case_publication_and_retrieval_gate_replays_monitoring_even_without_record(engine, with_record, tamper):
    from aiops_agent.application.case_source_validation import validate_case_sources
    run, _, payload, uow, monitor = investigation(engine)
    if not with_record:
        payload.pop("investigation_record")
    if tamper == "raw_identity":
        monitor.payload_json["observations"][0]["series"][0]["dimensions"]["identity_version"] = "2"
        monitor.content_hash = canonical_hash(monitor.payload_json)
    elif tamper == "drop_witness":
        payload["evidence"][1]["source_identity_witnesses"] = []
        if with_record:
            payload["investigation_record"]["observations"][1]["source_identity_witnesses"] = []
    final_id, version_id = uuid7(), uuid7()
    final = SimpleNamespace(artifact_id=final_id, ops_run_id=run.ops_run_id, payload_json=payload,
        content_hash=canonical_hash(payload), provenance_json={})
    raw_lookup = uow.runs.get_artifact
    async def lookup(*, artifact_id):
        return final if str(artifact_id) == str(final_id) else await raw_lookup(artifact_id=artifact_id)
    uow.runs.get_artifact = AsyncMock(side_effect=lookup)
    uow.runs.get_run_scoped = AsyncMock(return_value=run)
    run.agent_id, run.plan_snapshot_json = uuid7(), {}
    source = SimpleNamespace(source_report_id=None, source_artifact_id=final_id, source_run_id=run.ops_run_id,
        content_hash=final.content_hash)
    uow.operations_knowledge = SimpleNamespace(get_version_relations=AsyncMock(return_value={"sources": (source,)}))
    uow.agents = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(current_version_id=uuid7())),
        version_has_target=AsyncMock(return_value=True), version_source_ids=AsyncMock(return_value=()))
    uow.turns = SimpleNamespace(get_run_link_by_ops_run_id=AsyncMock(return_value=None))
    assert asyncio.run(validate_case_sources(uow=uow, domain_id=uuid7(), version_id=version_id)) == (tamper is None)
