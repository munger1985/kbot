"""精确 Oracle 缓存计划、绑定元数据、正式执行链和原始来源重放。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.provenance import database_provenance
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.diagnosis.sql_plans import database_cursor_plan
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.investigation.discovery import available_tools, build_playbook_plan
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.contracts.investigation import InvestigationRecord
from aiops_agent.contracts.sql_plan import SqlPlanAssessment
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.diagnostics.grants import DiagnosticGrantCodec
from aiops_agent.domain.evidence.sql_plan_matching import assess_sql_plans
from aiops_agent.executor import DiagnosticExecutorService
from aiops_agent.executor.drivers import DiagnosticDriverError, DriverQueryResult
from aiops_agent.playbooks import PlaybookRegistry, canonical_hash
from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
from aiops_agent.workers.database_handlers import DatabaseDiagnosticHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.tool_handlers import DbaDiagnosticToolHandler
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler, DbaAnswerComposeHandler
from platform_core.contracts.aiops.executor import DatabaseColumn, DatabaseObservation, DiagnosticExecutionGrant, DiagnosticLimits
from platform_core.contracts.aiops.investigation import InvestigationAction
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from platform_core.identity import uuid7
from platform_core.contracts.aiops import MeasurementSemantics
from tests.support.aiops_resources import bind_resource_control
from tests.unit.aiops_agent.test_aiops_sql_plan_import import SUBJECTS, NOW, anchor, freeze, upload


PLAN = "db.oracle.statement.plan"
BINDS = "db.oracle.statement.bind_metadata"
SUBJECT = SUBJECTS["ORACLE"]
PARAMS = {key: value if key == "sql_id" else int(value) for key, value in SUBJECT.items()}
PHYSICAL = {"dbid": SUBJECT["dbid"], "container_id": SUBJECT["container_id"], "instance_number": SUBJECT["inst_id"], "instance_started_at": "2026-10-10T05:39:06"}


def tool(tool_id):
    return DiagnosticRegistry.load().resolve(tool_id=tool_id, tool_version="1.0.0", db_type="ORACLE", db_version="19.3",
        capabilities={"dynamic_performance_views"}, entitlements=set())


def observation(tool_id=PLAN, *, runtime=True, physical=None, **updates):
    resolved = tool(tool_id)
    base = {**PARAMS}
    if tool_id == PLAN:
        rows = [{**base, "plan_line_id": 0, "parent_id": None, "operation": "SELECT STATEMENT", "cost": 4,
                 "last_starts": 1 if runtime else None, "last_output_rows": 1 if runtime else None},
                {**base, "plan_line_id": 1, "parent_id": 0, "operation": "TABLE ACCESS", "options": "FULL", "object_owner": "MES",
                 "object_name": "ORDERS", "object_type": "TABLE", "cardinality": 10, "cost": 4,
                 "last_starts": 2 if runtime else None, "last_output_rows": 2000 if runtime else None,
                 "last_elapsed_time": 23250 if runtime else None, "last_cr_buffer_gets": 200 if runtime else None,
                 "last_disk_reads": 2 if runtime else None, "last_disk_writes": 0 if runtime else None}]
    else:
        rows = [{**base, "bind_position": 1, "datatype_name": "NUMBER", "was_captured": "YES", "last_captured": "2026-10-27T00:00:00"}]
    columns = tuple(DatabaseColumn(**column.model_dump(exclude={"nullable"})) for column in resolved.definition.output_columns)
    values = tuple(tuple(row.get(column.name) for column in columns) for row in rows)
    return DatabaseObservation(executor_request_id=uuid7(), target_id=uuid7(), tool_id=tool_id, tool_version="1.0.0",
        variant=resolved.definition.variant, template_sha256=resolved.definition.template_sha256,
        db_type="ORACLE", db_version="19.3", capability_snapshot_hash="a" * 64, captured_at=NOW, duration_ms=1,
        columns=columns, rows=values, row_count=len(rows), truncated=False, result_sha256=canonical_hash(values),
        parameters_sha256=canonical_hash(PARAMS), provenance={"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": PHYSICAL if physical is None else physical},
    ).model_copy(update=updates)


def fact(obs, artifact_id=None):
    identity = str(artifact_id or uuid7())
    document, warnings = database_cursor_plan(obs)
    return TurnEvidenceFact(evidence_ref=f"artifact:{identity}#a1", artifact_id=identity, source_id=obs.tool_id,
        step_id="a1", tool_id=obs.tool_id, trust_level="SOURCE_VERIFIED", measurement_semantics="CURRENT_ACTIVITY",
        presentation_kind="TABLE", captured_at=obs.captured_at.isoformat(),
        columns=tuple(column.model_dump(mode="json") for column in obs.columns), rows=obs.rows, row_count=obs.row_count,
        truncated=obs.truncated, provenance=database_provenance(obs), sql_plan=document, warnings=(*obs.warnings, *warnings))


def direct(physical=None):
    value = anchor("ORACLE")
    return value.model_copy(update={"provenance": value.provenance.model_copy(update={"physical_identity": tuple((PHYSICAL if physical is None else physical).items())})})


def assessment(plan=None, binds=None, statement=None):
    value = fact(observation()) if plan is None else plan
    facts = (statement or direct(), value, *(tuple(binds) if binds is not None else (fact(observation(BINDS)),)))
    return assess_sql_plans(database_type="ORACLE", subject=SUBJECT, evidence=facts)[0]


@pytest.mark.parametrize("tool_id", [PLAN, BINDS])
def test_exact_tools_are_bounded_parameterized_catalog_reads_without_sql_replay_or_values(tool_id):
    resolved = tool(tool_id)
    values = DiagnosticRegistry.load().validate_parameters(resolved, PARAMS)
    assert set(values) == {*SUBJECT, "limit"} and values["limit"] == (129 if tool_id == PLAN else 65)
    assert resolved.definition.limit_semantics == "BOUNDED_ENUMERATION"
    assert resolved.definition.timeout_seconds == 10 and resolved.definition.cost_level == "LOW"
    sql = resolved.sql.lower()
    for key in SUBJECT:
        assert ":" + key in sql
    for forbidden in ("value_string", "value_anydata", "access_predicates", "filter_predicates", "projection", "dbms_xplan", "explain", "awr", "dba_hist", "sql_text"):
        assert forbidden not in sql
    if tool_id == BINDS:
        assert "child_address" in sql and "datatype_string" in sql and "binds.name" not in sql
    for missing in SUBJECT:
        with pytest.raises(ValueError, match="缺少诊断参数"):
            DiagnosticRegistry.load().validate_parameters(resolved, {key: value for key, value in PARAMS.items() if key != missing})
    for invalid in ({**PARAMS, "inst_id": True}, {**PARAMS, "child_number": -1}, {**PARAMS, "limit": 130}, {**PARAMS, "sql": "SELECT 1"}):
        with pytest.raises(ValueError):
            DiagnosticRegistry.load().validate_parameters(resolved, invalid)


@pytest.mark.parametrize("runtime", [False, True])
def test_source_document_keeps_native_last_statistics_but_never_proves_representative_execution(runtime):
    obs = observation(runtime=runtime)
    document, warnings = database_cursor_plan(obs)
    assert document.origin == "DATABASE_CURSOR" and not warnings
    assert document.observed_at == NOW and document.declared_captured_at is None
    assert document.runtime_statistics == ("PRESENT" if runtime else "NONE")
    assert not document.parameter_values_available and not document.parameter_types
    row = document.nodes[1].metrics
    if runtime:
        assert row["actual_rows_per_start"] == "1000" and row["actual_to_estimated_rows_ratio"] == "100"
        assert row["actual_consistent_gets"] == "200" and row["actual_elapsed_time_us"] == "23250"
    result = assessment(fact(obs))
    assert result.identity_match == result.physical_match == "SOURCE_MATCH"
    assert result.time_relation == "SOURCE_NEAR_OBSERVATION" and len(result.bind_metadata_refs) == 1
    assert not result.production_execution_verified and result.representative_parameters == "UNVERIFIED"


@pytest.mark.parametrize("corrupt", ["truncated", "too_many", "mixed", "cycle", "parent", "duplicate", "bool", "missing_column", "variant", "version", "engine"])
def test_partial_or_unbound_cache_is_not_falsely_projected_as_a_complete_plan(corrupt):
    obs = observation()
    names = [column.name for column in obs.columns]
    rows = [list(row) for row in obs.rows]
    if corrupt == "truncated":
        obs = obs.model_copy(update={"truncated": True})
    elif corrupt == "too_many":
        obs = obs.model_copy(update={"rows": obs.rows * 65, "row_count": 130})
    elif corrupt in {"mixed", "cycle", "parent", "duplicate", "bool"}:
        key, value = {"mixed": ("child_number", 99), "cycle": ("parent_id", 1), "parent": ("parent_id", 99),
            "duplicate": ("plan_line_id", 0), "bool": ("last_starts", True)}[corrupt]
        rows[1][names.index(key)] = value
        obs = obs.model_copy(update={"rows": tuple(tuple(row) for row in rows)})
    elif corrupt == "missing_column":
        obs = obs.model_copy(update={"columns": obs.columns[:-1], "rows": obs.rows})
    else:
        obs = obs.model_copy(update={{"variant": "variant", "version": "tool_version", "engine": "db_type"}[corrupt]: {"variant": "unknown", "version": "2.0.0", "engine": "MYSQL"}[corrupt]})
    document, warnings = database_cursor_plan(obs)
    assert document is None and warnings


def test_zero_cached_rows_is_empty_evidence_not_parse_failure():
    assert database_cursor_plan(observation(rows=(), row_count=0)) == (None, ())


@pytest.mark.parametrize("case", ["unknown", "database", "container", "boot", "remote", "root", "identity", "time"])
def test_cache_subject_container_boot_and_time_are_not_borrowed_from_other_cursors(case):
    physical = dict(PHYSICAL)
    statement = direct()
    obs = observation()
    if case == "unknown":
        obs = observation(physical={})
    elif case == "database":
        obs = observation(physical={**physical, "dbid": "99"})
    elif case == "container":
        obs = observation(physical={**physical, "container_id": "9"})
    elif case == "boot":
        obs = observation(physical={**physical, "instance_started_at": "2026-10-27T00:00:00"})
    elif case == "remote":
        physical["instance_number"] = "2"
        obs, statement = observation(physical=physical), direct(physical)
    elif case == "root":
        physical["container_id"] = "1"
        obs, statement = observation(physical=physical), direct(physical)
    elif case == "identity":
        names = [column.name for column in obs.columns]
        rows = [list(row) for row in obs.rows]
        for row in rows:
            row[names.index("child_number")] = 99
        obs = obs.model_copy(update={"rows": tuple(tuple(row) for row in rows)})
    elif case == "time":
        obs = obs.model_copy(update={"captured_at": datetime(2026, 10, 27, tzinfo=UTC)})
    result = assessment(fact(obs), statement=statement)
    assert result.identity_match == ("MISMATCH" if case == "identity" else "SOURCE_MATCH")
    assert result.physical_match == ("INCOMPLETE" if case in {"unknown", "remote"} else "MISMATCH" if case in {"database", "container", "boot"} else "SOURCE_MATCH")
    assert result.time_relation == ("HISTORICAL_OR_DIFFERENT_TIME" if case == "time" else "SOURCE_NEAR_OBSERVATION")
    assert not result.production_execution_verified


@pytest.mark.parametrize("case", ["match", "different_child", "different_plan", "database", "remote", "old", "truncated", "untrusted", "historical", "empty"])
def test_bind_capture_metadata_must_match_exact_cursor_and_remains_unverified_parameters(case):
    obs = observation(BINDS)
    if case in {"different_child", "different_plan"}:
        names = [column.name for column in obs.columns]
        row = list(obs.rows[0])
        row[names.index("child_number" if case == "different_child" else "plan_hash_value")] = 99
        obs = obs.model_copy(update={"rows": (tuple(row),)})
    elif case in {"database", "remote"}:
        obs = observation(BINDS, physical={**PHYSICAL, "dbid" if case == "database" else "instance_number": "99"})
    elif case == "old":
        obs = obs.model_copy(update={"captured_at": datetime(2026, 10, 27, tzinfo=UTC)})
    elif case == "truncated":
        obs = obs.model_copy(update={"truncated": True})
    elif case == "empty":
        obs = obs.model_copy(update={"rows": (), "row_count": 0})
    captured = fact(obs)
    if case == "untrusted":
        captured = captured.model_copy(update={"trust_level": "USER_PROVIDED"})
    elif case == "historical":
        captured = captured.model_copy(update={"source_id": "prior_turn.sql_observation"})
    result = assessment(binds=(captured,))
    assert bool(result.bind_metadata_refs) == (case == "match")
    assert result.representative_parameters == "UNVERIFIED"


def test_uploaded_plan_cannot_change_origin_to_database_cursor():
    source, _ = upload("ORACLE")
    source["payload"]["structured_facts"].update(origin="DATABASE_CURSOR", observed_at=NOW, declared_captured_at=None, parameter_types=[])
    from aiops_agent.application.diagnosis.sql_plans import sql_plan_upload_fact
    with pytest.raises(ValueError, match="用户计划提取"):
        sql_plan_upload_fact(source)
    with pytest.raises(ValueError, match="人工计划"):
        SqlPlanAssessment(evidence_ref="fake", database_type="ORACLE", subject=SUBJECT, identity_match="SOURCE_MATCH",
            physical_match="SOURCE_MATCH", time_relation="SOURCE_NEAR_OBSERVATION", runtime_statistics="NONE", node_count=1, limitations=())


@pytest.mark.parametrize("full", [False, True])
def test_oracle_baseline_discovers_subject_first_or_reads_only_exact_cache(full):
    subject = {"subject_type": "ORACLE_SQL_ID", **(PARAMS if full else {"sql_id": SUBJECT["sql_id"]})}
    compact = SimpleNamespace(problem_statement="诊断 SQL", time_scope=None, requested_window_seconds=None, temporal_analysis_mode="CURRENT",
        forecast_scope=None, forecast_horizon_seconds=None, success_criteria=("定位原因",), selected_playbook_ids=(),
        implementation_profile="NONE", diagnostic_profile="SINGLE_SQL_PERFORMANCE", subject_ref=subject)
    result = TurnPlanningService._single_sql_investigation_output(question="诊断 SQL", compact=compact,
        target_context={"db_type": "ORACLE"}, subject=subject)
    assert [action.tool_id for action in result.plan.actions] == (["db.sql.cursor_details", PLAN, BINDS] if full else ["db.sql.cursor_details"])
    if full:
        assert all(action.input == PARAMS for action in result.plan.actions[1:])
    candidates = TurnPlanningService._profile_tool_ids(compact, {"db_type": "ORACLE"})
    assert PLAN in candidates and BINDS in candidates and "db.sql.display_cursor" not in candidates


@pytest.mark.parametrize("tool_id", [PLAN, BINDS])
@pytest.mark.parametrize("failure", [None, "PRIVILEGE_MISSING", "OBJECT_NOT_FOUND"])
@pytest.mark.parametrize("boundary", [False, True])
def test_exact_cache_tools_execute_through_signed_worker_executor_and_assessment(tool_id, failure, boundary):
    async def run():
        registry, playbooks = DiagnosticRegistry.load(), PlaybookRegistry.load()
        builder = ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=registry)
        caps = DbaCapabilitySnapshot(agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
            database_type="ORACLE", database_version="19.3", target_enabled=True, target_reachable=True,
            target_capabilities=("DB_READONLY", "dynamic_performance_views"))
        card = next(item for item in available_tools(builder, caps) if item["tool_id"] == tool_id)
        assert all(card["input"][key]["required"] for key in PARAMS)
        action = InvestigationAction(action_id="a1", tool_id=tool_id, question="读取精确游标已有证据", input=PARAMS,
            expected_evidence_kind="SQL_CURSOR_PLAN", measurement_semantics="CURRENT_ACTIVITY")
        plan = build_playbook_plan(playbooks)
        compiled = InvestigationTaskCompiler(playbooks).compile(plan, investigation_actions=(action,))
        scope = {"tenant_id": str(uuid7()), "domain_id": str(uuid7())} if "tenant_id" in DiagnosticExecutionGrant.model_fields else {"domain_id": 7}
        execution = builder.build(plan=plan, compiled=compiled, capabilities=caps, direct_actions=(action,),
            database_execution={**scope, "target_row_version": 1, "db_type": "ORACLE", "configured_version": "19.3",
                "diagnostic_credential_id": str(uuid7()), "automatic_access_enabled": True,
                "connection_profile": {"host": "db.internal", "port": 1521, "service": "PDB1"}})
        task = next(item for item in compiled.tasks if item.task_key == "diagnostic:a1")
        now = datetime.now(UTC)
        context = TaskExecutionContext(run_id=str(uuid7()), task_id=str(uuid7()), task_key=task.task_key,
            target_id=str(uuid7()), agent_id=str(uuid7()), actor_id=str(uuid7()), trigger_type="CHAT", trace_id="cursor-plan",
            attempt=1, max_attempts=1, deadline_at=(now + timedelta(seconds=90)).isoformat(),
            lease_until=(now + timedelta(seconds=90)).isoformat(), lease_token="lease-cache-plan",
            plan_snapshot={"investigation_execution": execution}, policy_snapshot={}, input_artifacts=())
        codec = DiagnosticGrantCodec(secret="s" * 32, issuer="worker", audience="executor")
        control = AsyncMock()
        control.issue_credential.return_value = SimpleNamespace(username="readonly", password="never-export")
        bind_resource_control(control, codec)
        obs = observation(tool_id)
        async def raw(**kwargs):
            assert kwargs["parameters"] == {**PARAMS, "limit": 129 if tool_id == PLAN else 65}
            if failure:
                raise DiagnosticDriverError(failure)
            raw_rows = tuple(tuple(datetime.fromisoformat(value) if column.logical_type == "DATETIME" and value is not None else value
                for column, value in zip(obs.columns, row, strict=True)) for row in obs.rows)
            if boundary:
                raw_rows = (raw_rows[0],) * kwargs["parameters"]["limit"]
            return DriverQueryResult(columns=tuple(column.name for column in obs.columns), rows=raw_rows,
                truncated=False, db_version="19.3", instance_identity=tuple(PHYSICAL.items()))
        driver = SimpleNamespace(db_type="ORACLE", execute=AsyncMock(side_effect=raw))
        executor = DiagnosticExecutorService(registry=registry, grant_codec=codec, control_plane=control, drivers=(driver,),
            hard_limits=DiagnosticLimits(statement_timeout_seconds=20, max_result_rows=1000, max_result_bytes=1048576), concurrency=2)
        async def invoke(request, **kwargs):
            grant = codec.verify(request.grant)
            assert str(grant.target_id) == str(context.target_id) and str(grant.task_id) == str(context.task_id)
            return await executor.execute(request)
        database = DatabaseDiagnosticHandler(executor_client=SimpleNamespace(execute_diagnostic=AsyncMock(side_effect=invoke)),
            grant_codec=codec, grant_issuer="worker", grant_audience="executor", grant_ttl_seconds=45)
        result = await DbaDiagnosticToolHandler(database_handler=database).execute(context)
        assert control.issue_credential.await_count == driver.execute.await_count == 1
        if failure:
            assert result.status == "FAILED" and result.tool_outcomes[0].gap.code == failure
            return
        assert result.status == "SUCCEEDED", result.model_dump(mode="json")
        evidence = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(input_artifacts=(
            {"artifact_id": uuid7(), "schema_version": result.schema_version, "payload": result.model_dump(mode="json")},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT"))
        value = evidence.evidence[0]
        assert value.truncated == boundary
        assert bool(value.sql_plan) == (tool_id == PLAN and not boundary)
        if boundary:
            assert "ENUMERATION_LIMIT_REACHED" in value.warnings
            assert not assessment(binds=(value,)).bind_metadata_refs
        if value.sql_plan:
            statement = direct().model_copy(update={"captured_at": value.captured_at})
            bundle = build_sql_evidence_bundle(task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": SUBJECT}, database_type="ORACLE", evidence=(statement, value))
            assert bundle["plan_assessments"][0]["identity_match"] == "SOURCE_MATCH"
        assert "never-export" not in result.model_dump_json()
    asyncio.run(run())


def source_artifact(obs, run):
    """构造正式工具原件与投影，供报告和案例共同的来源门禁重放。"""
    obs = obs.model_copy(update={"target_id": run.target_id})
    value = fact(obs)
    payload = {"schema_version": "DBA_TOOL_RESULT.v1", "source_type": "TOOL", "source_id": value.source_id,
        "source_version": "1.0.0", "definition_hash": "a" * 64, "output_schema": "DBA_TOOL_RESULT.v1",
        "measurement_semantics": value.measurement_semantics, "presentation_kind": "TABLE", "status": "SUCCEEDED",
        "tool_outcomes": [{"step_id": "a1", "tool_id": value.tool_id, "tool_version": "1.0.0", "status": "SUCCEEDED",
            "observation": obs.model_dump(mode="json")}], "sampling": None}
    original = SimpleNamespace(ops_run_id=run.ops_run_id, schema_version="DBA_TOOL_RESULT.v1", trust_level="SOURCE_VERIFIED",
        payload_json=payload, content_hash=canonical_hash(payload))
    return value, original


def original_anchor(run):
    value = direct()
    obs = observation().model_copy(update={"tool_id": value.tool_id,
        "columns": tuple(DatabaseColumn(**column, logical_type="STRING", sensitivity="PUBLIC") for column in value.columns),
        "rows": value.rows, "row_count": value.row_count})
    value, original = source_artifact(obs, run)
    value = value.model_copy(update={"measurement_semantics": MeasurementSemantics.CUMULATIVE_SINCE_LOAD})
    original.payload_json["measurement_semantics"] = value.measurement_semantics
    original.content_hash = canonical_hash(original.payload_json)
    return value, original


@pytest.mark.parametrize("tamper", ["none", "missing", "run", "hash", "trust", "schema", "rows", "document", "drop_document", "time", "origin", "binds", "anchor"])
def test_report_and_case_reopen_original_cursor_plan_binds_and_identity_anchor(tamper):
    run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
    statement, statement_source = original_anchor(run)
    plan, plan_source = source_artifact(observation(), run)
    binds, binds_source = source_artifact(observation(BINDS), run)
    values = (statement, plan, binds)
    record = freeze(values, "ORACLE").model_copy(update={"run_id": str(run.ops_run_id), "target_id": str(run.target_id)})
    assert record.sql_plan_assessments[0].bind_metadata_refs == (binds.evidence_ref,)
    record.validate_observations(values)
    payload = {"status": "COMPLETED", "blocks": [], "evidence": [value.model_dump(mode="json") for value in values],
        "investigation_record": record.model_dump(mode="json")}
    normalized = normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", payload=payload, source_kind="DIAGNOSIS")
    assert normalized["investigation_record"]["sql_plan_assessments"][0]["origin"] == "DATABASE_CURSOR"
    sources = {str(value.artifact_id): source for value, source in zip(values, (statement_source, plan_source, binds_source), strict=True)}
    if tamper == "missing":
        sources.pop(str(plan.artifact_id))
    elif tamper == "run":
        plan_source.ops_run_id = uuid7()
    elif tamper == "hash":
        plan_source.content_hash = "f" * 64
    elif tamper == "trust":
        plan_source.trust_level = "USER_PROVIDED"
    elif tamper == "schema":
        plan_source.schema_version = "USER_UPLOAD_EXTRACT.v1"
    elif tamper in {"rows", "time", "origin", "binds", "anchor"}:
        source = binds_source if tamper == "binds" else statement_source if tamper == "anchor" else plan_source
        raw = source.payload_json["tool_outcomes"][0]["observation"]
        if tamper == "time":
            raw["captured_at"] = "2026-10-27T00:00:00+00:00"
        elif tamper == "origin":
            raw["variant"] = "unknown"
        else:
            raw["rows"][0][-1] = "different"
        source.content_hash = canonical_hash(source.payload_json)
    elif tamper == "document":
        payload["evidence"][1]["sql_plan"]["nodes"][1]["operation"] = "INDEX RANGE SCAN"
    elif tamper == "drop_document":
        payload["evidence"][1]["sql_plan"] = None
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(side_effect=lambda *, artifact_id: sources.get(str(artifact_id)))))
    assert asyncio.run(validate_investigation_sources(uow=uow, run=run, payload=payload)) == (tamper == "none")


@pytest.mark.parametrize("document_present", [False, True])
def test_cursor_source_gate_does_not_require_a_record_or_plan_assessment(document_present):
    run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
    value, original = source_artifact(observation(), run)
    if not document_present:
        value = value.model_copy(update={"sql_plan": None})
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=original)))
    assert asyncio.run(validate_investigation_sources(uow=uow, run=run,
        payload={"evidence": [value.model_dump(mode="json")]})) == document_present


@pytest.mark.parametrize("tamper", ["drop", "identity", "bind_refs", "representative", "execution"])
def test_frozen_cursor_assessment_cannot_be_relabelled_as_a_verified_execution(tamper):
    plan_observation = observation()
    values = (direct(), fact(plan_observation), fact(observation(BINDS, target_id=plan_observation.target_id)))
    record = freeze(values, "ORACLE")
    body = record.model_dump(mode="json")
    if tamper == "drop":
        body["sql_plan_assessments"] = []
    elif tamper == "identity":
        body["sql_plan_assessments"][0]["identity_match"] = "DECLARED_MATCH"
    elif tamper == "bind_refs":
        body["sql_plan_assessments"][0]["bind_metadata_refs"] = []
    elif tamper == "representative":
        body["sql_plan_assessments"][0]["representative_parameters"] = "VERIFIED"
    else:
        body["sql_plan_assessments"][0]["production_execution_verified"] = True
    with pytest.raises(ValueError):
        InvestigationRecord.model_validate(body).validate_observations(values)


def test_per_subject_plan_budget_does_not_drop_later_exact_cursor():
    source = fact(observation())
    others = []
    for child in range(1, 5):
        document = source.sql_plan.model_copy(update={"subject": {**SUBJECT, "child_number": str(child)}})
        others.append(source.model_copy(update={"sql_plan": document, "evidence_ref": f"other:{child}"}))
    results = assess_sql_plans(database_type="ORACLE", subject=SUBJECT, evidence=(direct(), *others, source))
    assert len(results) == 4 and results[0].evidence_ref == source.evidence_ref
    assert results[0].identity_match == "SOURCE_MATCH"


def test_multiple_oracle_subjects_keep_separate_cursor_plan_and_bind_relationships():
    a = direct()
    names = [column["name"] for column in a.columns]
    row = list(a.rows[0])
    row[names.index("child_number")] = "1"
    b = a.model_copy(update={"evidence_ref": "other:a1", "rows": (tuple(row),)})
    frame = {"diagnostic_profile": "SQL_COMPARISON", "subject_ref": {"subjects": [
        {"subject_type": "ORACLE_SQL_ID", **SUBJECT}, {"subject_type": "ORACLE_SQL_ID", **SUBJECT, "child_number": "1"}]}}
    bundle = build_sql_evidence_bundle(task_frame=frame, database_type="ORACLE", evidence=(a, b, fact(observation()), fact(observation(BINDS))))
    assert [item["identity_match"] for item in bundle["plan_assessments"]] == ["SOURCE_MATCH", "MISMATCH"]
    assert [len(item["bind_metadata_refs"]) for item in bundle["plan_assessments"]] == [1, 0]


@pytest.mark.parametrize("root,expected", [(True, 1), (False, 0)])
def test_root_plan_row_uses_explicit_container_without_borrowing_pdb_scope(root, expected):
    from aiops_agent.domain.evidence.sql_object_graph import build_sql_object_graph
    physical = {**PHYSICAL, "container_id": "1" if root else "9"}
    plan = fact(observation(physical=physical))
    catalog = direct().model_copy(update={"tool_id": "db.sql.object_statistics", "columns": (
        {"name": "schema_name"}, {"name": "table_name"}), "rows": (("MES", "ORDERS"),)})
    graph = build_sql_object_graph(database_type="ORACLE", subject=SUBJECT, evidence=(plan, catalog))
    assert len(graph.relationships) == expected


@pytest.mark.parametrize("tamper", ["none", "document", "drop"])
def test_background_observation_reuse_also_replays_cursor_plan_projection(tamper):
    from unittest.mock import patch
    from aiops_agent.application.investigation.future_observation_evidence import validate_future_observation_facts
    run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
    value, original = source_artifact(observation(), run)
    background_id = uuid7()
    original.provenance_json = {"future_observation_source": {"source_turn_id": str(uuid7()),
        "background_run_id": str(background_id), "source_artifact_id": str(uuid7()), "source_content_hash": original.content_hash}}
    value = TurnEvidenceFact.model_validate({**value.model_dump(mode="python"), "followup_run_id": str(background_id)})
    if tamper == "document":
        value = value.model_copy(update={"sql_plan": value.sql_plan.model_copy(update={"nodes": (
            value.sql_plan.nodes[0].model_copy(update={"operation": "INDEX RANGE SCAN"}), *value.sql_plan.nodes[1:])})})
    elif tamper == "drop":
        value = value.model_copy(update={"sql_plan": None})
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=original)))
    with patch("aiops_agent.application.investigation.future_observation_evidence.validate_reused_future_observation", new=AsyncMock(return_value=True)):
        assert asyncio.run(validate_future_observation_facts(uow=uow, run=run,
            payload={"evidence": [value.model_dump(mode="json")]})) == (tamper == "none")


@pytest.mark.parametrize("tool_id,field,label", [(PLAN, "last_elapsed_time", "LAST 耗时（微秒）"),
    (BINDS, "last_captured", "最近绑定捕获时间（数据库原生时间）")])
def test_cursor_plan_tables_show_native_units_and_bind_time_without_rewriting_source(tool_id, field, label):
    value = fact(observation(tool_id))
    blocks = DbaAnswerComposeHandler._data_blocks((value,))
    columns = blocks[0].payload["columns"]
    assert next(column["label"] for column in columns if column["name"] == field) == label
    assert next(column["label"] for column in columns if column["name"] == "sql_id") == "SQL_ID"
    assert all("label" not in column for column in value.columns)


@pytest.mark.parametrize("tool_id", [PLAN, BINDS])
@pytest.mark.parametrize("semantics", ["CUMULATIVE_SINCE_LOAD", "SNAPSHOT_DELTA", "HISTORICAL_SAMPLES", "NOT_APPLICABLE"])
def test_catalog_time_semantics_cannot_be_relabelled_by_the_planner(tool_id, semantics):
    builder = ToolExecutionSnapshotBuilder(playbook_registry=PlaybookRegistry.load(), diagnostic_registry=DiagnosticRegistry.load())
    caps = DbaCapabilitySnapshot(agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
        database_type="ORACLE", database_version="19.3", target_enabled=True, target_reachable=True,
        target_capabilities=("DB_READONLY", "dynamic_performance_views"))
    card = next(item for item in available_tools(builder, caps) if item["tool_id"] == tool_id)
    assert card["measurement_semantics"] == "CURRENT_ACTIVITY"
    action = InvestigationAction(action_id="a1", tool_id=tool_id, question="读取精确游标缓存", input=PARAMS,
        expected_evidence_kind="SQL_CURSOR_PLAN", measurement_semantics=semantics)
    with pytest.raises(ValueError, match="观测口径必须使用目录声明"):
        builder.validate_direct_actions(actions=(action,), capabilities=caps)
