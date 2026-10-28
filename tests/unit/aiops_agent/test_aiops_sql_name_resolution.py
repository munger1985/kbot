"""未限定对象名必须支持实际目录补证，又不能把候选或诊断连接环境升级为历史解析结果。"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
from aiops_agent.application.diagnosis.provenance import database_provenance
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.investigation.discovery import available_tools, build_playbook_plan
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.diagnostics.grants import DiagnosticGrantCodec
from aiops_agent.domain.evidence.sql_object_graph import build_sql_object_graph, relation_names, replay_sql_object_graph
from aiops_agent.executor import DiagnosticExecutorService
from aiops_agent.executor.drivers import DiagnosticDriverError, DriverQueryResult
from aiops_agent.playbooks import PlaybookRegistry, canonical_hash
from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
from aiops_agent.workers.database_handlers import DatabaseDiagnosticHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.tool_handlers import DbaDiagnosticToolHandler
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops.executor import DatabaseColumn, DatabaseObservation, DiagnosticExecutionGrant, DiagnosticLimits
from platform_core.contracts.aiops.investigation import InvestigationAction
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from platform_core.contracts.aiops import MeasurementSemantics
from platform_core.identity import uuid7
from tests.support.aiops_resources import bind_resource_control
from tests.unit.aiops_agent.test_aiops_sql_object_graph import NOW, fact, scenario


TOOL = "db.sql.object_candidates"
ENGINES = ("POSTGRESQL", "MYSQL", "ORACLE")
VERSIONS = {"POSTGRESQL": "16", "MYSQL": "8.4", "ORACLE": "19"}


def unqualified(engine):
    subject, original, indexes = scenario(engine)
    row = dict(zip((item["name"] for item in original.columns), original.rows[0], strict=True))
    field = {"POSTGRESQL": "normalized_statement", "MYSQL": "digest_text", "ORACLE": "sql_text"}[engine]
    row[field] = "SELECT * FROM orders WHERE id = 1"
    return subject, fact(engine, original.tool_id, row), indexes


def candidate(engine, *, schema=None, table=None, **updates):
    return fact(engine, TOOL, {"schema_name": schema or ("MES" if engine == "ORACLE" else "mes"),
        "table_name": table or ("ORDERS" if engine == "ORACLE" else "orders"),
        "relation_kind": "TABLE", "native_object_id": None if engine == "MYSQL" else "12345",
        **({"container_id": 3} if engine == "ORACLE" else {})},
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY, **updates)


@pytest.mark.parametrize("sql,engine,expected", [
    ("SELECT * FROM orders", "POSTGRESQL", ((None, "orders"),)),
    ('SELECT * FROM "Orders"', "POSTGRESQL", ((None, "Orders"),)),
    ("SELECT * FROM orders", "ORACLE", ((None, "ORDERS"),)),
    ('SELECT * FROM "Orders@local"', "ORACLE", ((None, "Orders@local"),)),
    ("SELECT * FROM orders@remote", "ORACLE", ()),
    ("SELECT * FROM mes.orders@remote", "ORACLE", ()),
    ("SELECT * FROM db.schema.orders", "POSTGRESQL", ()),
    ("SELECT * FROM generate_series(1, 10)", "POSTGRESQL", ()),
    ("SELECT * FROM orders ...", "POSTGRESQL", ()),
    ("SELECT * FROM orders; SELECT * FROM other", "POSTGRESQL", ()),
    ("SELECT 'FROM private.secret' FROM orders -- JOIN private.other", "POSTGRESQL", ((None, "orders"),)),
    ("WITH orders AS (SELECT * FROM mes.base) SELECT * FROM orders JOIN lines ON true", "POSTGRESQL",
        ((None, "lines"), ("mes", "base"))),
    ("WITH RECURSIVE x AS (SELECT * FROM mes.base UNION ALL SELECT * FROM x) SELECT * FROM x", "POSTGRESQL", (("mes", "base"),)),
    ("WITH RECURSIVE a AS (SELECT * FROM b), b AS (SELECT * FROM mes.real) SELECT * FROM a", "POSTGRESQL", (("mes", "real"),)),
    ('WITH RECURSIVE "B" AS (SELECT * FROM b) SELECT * FROM "B"', "POSTGRESQL", ((None, "b"),)),
    ("SELECT * FROM mes.orders WHERE EXISTS (WITH orders AS (SELECT * FROM other.base) SELECT * FROM orders)", "POSTGRESQL",
        (("mes", "orders"), ("other", "base"))),
    ("UPDATE mes.orders SET x = 1 WHERE id IN (SELECT id FROM lines)", "POSTGRESQL", ((None, "lines"), ("mes", "orders"))),
    ("INSERT INTO orders (id) SELECT id FROM mes.lines", "MYSQL", ((None, "orders"), ("mes", "lines"))),
    ("DELETE o FROM orders o JOIN lines l ON o.id=l.id", "MYSQL", ((None, "lines"), (None, "orders"))),
    ("EXECUTE 'SELECT * FROM secret'", "POSTGRESQL", ()),
])
def test_names_preserve_scope_quotes_dml_and_non_executable_boundaries(sql, engine, expected):
    assert relation_names(sql, engine) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_unqualified_sql_produces_a_versioned_actionable_lookup_without_guessing_environment(engine):
    subject, text, _ = unqualified(engine)
    graph = build_sql_object_graph(database_type=engine, subject=subject, evidence=(text,))
    reference, = graph.references
    assert reference.relation_name == ("ORDERS" if engine == "ORACLE" else "orders")
    assert reference.lookup.tool_id == TOOL
    assert reference.lookup.input == {"table_name": reference.relation_name, "schema_name": "mes" if engine == "MYSQL" else "", "limit": 40}
    assert reference.resolution == ("NATIVE_DEFAULT_SCHEMA" if engine == "MYSQL" else "CONTEXT_REQUIRED")
    assert not graph.relationships and not reference.candidates
    replay_sql_object_graph(graph, (text,))


@pytest.mark.parametrize("engine", ENGINES)
def test_visible_candidate_is_not_a_production_plan_and_unknown_context_remains_unknown(engine):
    subject, text, indexes = unqualified(engine)
    discovered = candidate(engine)
    graph = build_sql_object_graph(database_type=engine, subject=subject, evidence=(text, discovered, indexes))
    reference, = graph.references
    assert len(reference.candidates) == 1 and reference.candidates[0].catalog_ref.evidence_ref == discovered.evidence_ref
    assert reference.lookup is None
    if engine == "MYSQL":
        assert reference.context_ref.evidence_ref == text.evidence_ref and reference.context_ref.field == "schema_name"
        assert "SQL_TEXT_DEFAULT_SCHEMA_CANDIDATE" in {edge.kind for edge in graph.relationships}
        assert all("不证明" in edge.limitation for edge in graph.relationships)
    else:
        assert reference.schema_name is None and reference.context_ref is None
        assert reference.resolution == "CONTEXT_REQUIRED" and not graph.relationships
    replay_sql_object_graph(graph, (text, discovered, indexes))


@pytest.mark.parametrize("engine", ENGINES)
def test_multiple_candidates_are_preserved_without_selecting_a_default_or_expanding_their_indexes(engine):
    subject, text, indexes = unqualified(engine)
    # 显式 NULL 默认数据库是 MySQL 的合法统计主体，不能借诊断连接的数据库名补齐。
    if engine == "MYSQL":
        subject = {**subject, "schema_name": None}
        row = dict(zip((item["name"] for item in text.columns), text.rows[0], strict=True))
        text = fact(engine, text.tool_id, {**row, "schema_name": None})
    evidence = (text, candidate(engine), candidate(engine, schema="other"), indexes)
    graph = build_sql_object_graph(database_type=engine, subject=subject, evidence=evidence)
    reference, = graph.references
    assert reference.resolution == "CONTEXT_REQUIRED" and len(reference.candidates) == 2
    assert not graph.relationships
    replay_sql_object_graph(graph, evidence)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("failure", ["untrusted", "scope", "unknown_identity", "old", "naive", "restart", "name", "prior"])
def test_candidates_cannot_bridge_bad_scope_time_name_or_trust(engine, failure):
    subject, text, _ = unqualified(engine)
    observed = candidate(engine)
    if failure == "untrusted":
        observed = observed.model_copy(update={"trust_level": "USER_PROVIDED"})
    elif failure == "unknown_identity":
        observed = observed.model_copy(update={"provenance": None})
    elif failure in {"scope", "restart"}:
        physical = dict(observed.provenance.physical_identity)
        key = "instance_started_at" if failure == "restart" else {"POSTGRESQL": "database_oid", "MYSQL": "server_uuid", "ORACLE": "container_id"}[engine]
        physical[key] = "99999"
        observed = observed.model_copy(update={"provenance": observed.provenance.model_copy(update={"physical_identity": tuple(physical.items())})})
    elif failure in {"old", "naive"}:
        observed = observed.model_copy(update={"captured_at": "2026-10-28T00:06:00+00:00" if failure == "old" else "2026-10-28T00:00:00"})
    elif failure == "name":
        observed = candidate(engine, table="unrelated")
    else:
        observed = observed.model_copy(update={"source_id": "prior_turn.sql_observation"})
    graph = build_sql_object_graph(database_type=engine, subject=subject, evidence=(text, observed))
    assert not graph.references[0].candidates and not graph.relationships


def test_mysql_defaults_are_row_scoped_and_never_leak_between_schema_subjects():
    subject, text, _ = unqualified("MYSQL")
    other = fact("MYSQL", text.tool_id, {**subject, "schema_name": "other", "digest_text": "SELECT * FROM orders"})
    bundle = build_sql_evidence_bundle(task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": {"digest": subject["digest"]}},
        database_type="MYSQL", evidence=(text, other, candidate("MYSQL"), candidate("MYSQL", schema="other")))
    assert bundle["subject_resolution"]["status"] == "COMPARISON"
    assert len(bundle["subjects"]) == 2
    for item in bundle["subjects"]:
        reference, = item["object_graph"]["references"]
        assert reference["schema_name"] == item["subject_ref"]["schema_name"]
        assert all(row["schema_name"] == reference["schema_name"] for row in reference["candidates"])


@pytest.mark.parametrize("engine", ["POSTGRESQL", "ORACLE"])
def test_mysql_default_context_cannot_be_relabeled_as_another_engine(engine):
    subject, text, _ = unqualified("MYSQL")
    graph = build_sql_object_graph(database_type="MYSQL", subject=subject, evidence=(text,))
    with pytest.raises(ValueError, match="只有 MySQL"):
        type(graph).model_validate({**graph.model_dump(mode="json"), "database_type": engine})


def test_oracle_parsing_schema_does_not_resolve_public_synonyms_and_root_does_not_borrow_pdb():
    subject, text, _ = unqualified("ORACLE")
    row = dict(zip((item["name"] for item in text.columns), text.rows[0], strict=True))
    text = fact("ORACLE", text.tool_id, {**row, "parsing_schema_name": "MES", "container_id": 3})
    root = candidate("ORACLE")
    physical = {**dict(root.provenance.physical_identity), "container_id": "1"}
    root = root.model_copy(update={"provenance": root.provenance.model_copy(update={"physical_identity": tuple(physical.items())})})
    root = root.model_copy(update={"rows": ((*root.rows[0][:-1], 1),)})
    graph = build_sql_object_graph(database_type="ORACLE", subject=subject, evidence=(text, root))
    assert graph.references[0].resolution == "CONTEXT_REQUIRED" and not graph.references[0].candidates


def test_oracle_root_candidates_with_explicit_native_container_only_match_root_cursor():
    subject, text, _ = unqualified("ORACLE")
    row = dict(zip((item["name"] for item in text.columns), text.rows[0], strict=True))
    text = fact("ORACLE", text.tool_id, {**row, "container_id": 1})
    observed = candidate("ORACLE")
    evidence = []
    for item in (text, observed):
        physical = {**dict(item.provenance.physical_identity), "container_id": "1"}
        item = item.model_copy(update={"provenance": item.provenance.model_copy(update={"physical_identity": tuple(physical.items())})})
        if item.tool_id == TOOL:
            item = item.model_copy(update={"rows": ((*item.rows[0][:-1], 1),)})
        evidence.append(item)
    graph = build_sql_object_graph(database_type="ORACLE", subject=subject, evidence=tuple(evidence))
    assert len(graph.references[0].candidates) == 1
    pdb_text = fact("ORACLE", text.tool_id, {**row, "container_id": 3})
    graph = build_sql_object_graph(database_type="ORACLE", subject=subject, evidence=(pdb_text, evidence[1]))
    assert not graph.references[0].candidates


def test_candidates_and_text_names_are_bounded_without_hiding_omitted_candidates():
    subject, text, _ = unqualified("POSTGRESQL")
    observed = candidate("POSTGRESQL")
    rows = tuple((f"schema_{index}", "orders", "TABLE", str(index)) for index in range(100))
    observed = observed.model_copy(update={"rows": rows, "row_count": len(rows)})
    graph = build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text, observed))
    assert graph.truncated and len(graph.references[0].candidates) == 16
    assert not graph.relationships
    assert relation_names("SELECT * FROM " + ",".join(f"mes.table_{index}" for index in range(33)), "POSTGRESQL") == ()
    replay_sql_object_graph(graph, (text, observed))


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders ...",
    "SELECT * FROM orders WHERE id IN (" + ",".join("1" for _ in range(6000)) + ")",
    "SELECT * FROM " + ",".join(f"mes.table_{index}" for index in range(33)),
])
def test_text_and_reference_budget_exhaustion_is_not_silently_reported_as_complete(sql):
    subject, text, _ = unqualified("POSTGRESQL")
    row = dict(zip((item["name"] for item in text.columns), text.rows[0], strict=True))
    text = fact("POSTGRESQL", text.tool_id, {**row, "normalized_statement": sql})
    graph = build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text,))
    assert graph.truncated and not graph.references and not graph.relationships


def test_name_exceeding_catalog_contract_is_retained_without_an_unexecutable_lookup():
    subject, text, _ = unqualified("POSTGRESQL")
    row = dict(zip((item["name"] for item in text.columns), text.rows[0], strict=True))
    name = "x" * 64
    text = fact("POSTGRESQL", text.tool_id, {**row, "normalized_statement": f'SELECT * FROM "{name}"'})
    graph = build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text,))
    assert graph.references[0].relation_name == name and graph.references[0].lookup is None
    assert graph.references[0].resolution == "CONTEXT_REQUIRED"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("failure", [None, "PRIVILEGE_MISSING", "OBJECT_NOT_FOUND", "EMPTY"])
def test_lookup_flows_through_formal_planning_signed_worker_executor_assessment_and_sql_bundle(engine, failure):
    async def run():
        registry, playbooks = DiagnosticRegistry.load(), PlaybookRegistry.load()
        builder = ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=registry)
        capabilities = DbaCapabilitySnapshot(agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
            database_type=engine, database_version=VERSIONS[engine], target_enabled=True, target_reachable=True, target_capabilities=("DB_READONLY",))
        card = next(item for item in available_tools(builder, capabilities) if item["tool_id"] == TOOL)
        assert card["input"]["table_name"]["required"] and "native_object_id" in card["returns"]
        subject, text, _ = unqualified(engine)
        graph = build_sql_object_graph(database_type=engine, subject=subject, evidence=(text,))
        action = InvestigationAction(action_id="a1", tool_id=TOOL, question="未限定名称有哪些可见目录候选？",
            input=graph.references[0].lookup.input, expected_evidence_kind="SQL_OBJECT_CANDIDATES", measurement_semantics="CURRENT_ACTIVITY")
        plan = build_playbook_plan(playbooks)
        compiled = InvestigationTaskCompiler(playbooks).compile(plan, investigation_actions=(action,))
        scope = {"tenant_id": str(uuid7()), "domain_id": str(uuid7())} if "tenant_id" in DiagnosticExecutionGrant.model_fields else {"domain_id": 7}
        execution = builder.build(plan=plan, compiled=compiled, capabilities=capabilities, direct_actions=(action,), database_execution={
            **scope, "target_row_version": 1, "db_type": engine, "configured_version": VERSIONS[engine],
            "diagnostic_credential_id": str(uuid7()), "automatic_access_enabled": True,
            "connection_profile": {"host": "db.internal", "port": 1521 if engine == "ORACLE" else 5432 if engine == "POSTGRESQL" else 3306,
                **({"service": "PDB1"} if engine == "ORACLE" else {"database": "demo"})}})
        task = next(item for item in compiled.tasks if item.task_key == "diagnostic:a1")
        now = datetime.now(timezone.utc)
        ctx = TaskExecutionContext(run_id=str(uuid7()), task_id=str(uuid7()), task_key=task.task_key, target_id=str(uuid7()),
            agent_id=str(uuid7()), actor_id=str(uuid7()), trigger_type="CHAT", trace_id="object-name-resolution", attempt=1,
            max_attempts=1, deadline_at=(now + timedelta(seconds=90)).isoformat(), lease_until=(now + timedelta(seconds=90)).isoformat(),
            lease_token="lease-object-lookup", plan_snapshot={"investigation_execution": execution}, policy_snapshot={}, input_artifacts=())
        codec = DiagnosticGrantCodec(secret="s" * 32, issuer="test-worker", audience="test-executor")
        control = AsyncMock()
        control.issue_credential.return_value = SimpleNamespace(username="readonly", password="never-export")
        bind_resource_control(control, codec)
        discovered = candidate(engine)

        async def raw(**kwargs):
            assert kwargs["tool"].definition.tool_id == TOOL and kwargs["parameters"] == action.input
            if failure not in {None, "EMPTY"}:
                raise DiagnosticDriverError(failure)
            return DriverQueryResult(columns=tuple(item["name"] for item in discovered.columns),
                rows=() if failure == "EMPTY" else discovered.rows, truncated=False, db_version=VERSIONS[engine],
                instance_identity=discovered.provenance.physical_identity)

        driver = SimpleNamespace(db_type=engine, execute=AsyncMock(side_effect=raw))
        executor = DiagnosticExecutorService(registry=registry, grant_codec=codec, control_plane=control, drivers=(driver,),
            hard_limits=DiagnosticLimits(statement_timeout_seconds=20, max_result_rows=1000, max_result_bytes=1048576), concurrency=2)

        async def invoke(request, **kwargs):
            grant = codec.verify(request.grant)
            assert str(grant.target_id) == str(ctx.target_id) and str(grant.task_id) == str(ctx.task_id)
            assert grant.parameters_sha256 and grant.template_sha256 == execution["direct_invocations"][task.task_key]["tool"]["template_sha256"]
            return await executor.execute(request)

        handler = DatabaseDiagnosticHandler(executor_client=SimpleNamespace(execute_diagnostic=AsyncMock(side_effect=invoke)),
            grant_codec=codec, grant_issuer="test-worker", grant_audience="test-executor", grant_ttl_seconds=45)
        result = await DbaDiagnosticToolHandler(database_handler=handler).execute(ctx)
        assert control.issue_credential.await_count == driver.execute.await_count == 1
        if failure not in {None, "EMPTY"}:
            assert result.status == "FAILED" and result.tool_outcomes[0].gap.code == failure
            assert result.tool_outcomes[0].observation is None
            return
        assert result.status == "SUCCEEDED"
        assessment = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": result.schema_version, "artifact_id": uuid7(), "payload": result.model_dump(mode="json")},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT"))
        observed = assessment.evidence[0]
        assert observed.row_count == (0 if failure == "EMPTY" else 1)
        text = text.model_copy(update={"captured_at": observed.captured_at})
        bundle = build_sql_evidence_bundle(task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject},
            database_type=engine, evidence=(text, observed))
        reference, = bundle["object_graph"]["references"]
        assert len(reference["candidates"]) == (0 if failure == "EMPTY" else 1)
        assert reference["resolution"] == ("NATIVE_DEFAULT_SCHEMA" if engine == "MYSQL" else "CONTEXT_REQUIRED")
        assert "never-export" not in result.model_dump_json()

    asyncio.run(run())


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("tamper", [None, "missing", "row", "scope", "trust", "run", "reference", "candidate", "context"])
def test_frozen_answer_report_and_case_reopen_unresolved_originals_without_sourced_claims_or_confirmed_object_edges(engine, tamper):
    subject, text, _ = unqualified(engine)
    discovered = candidate(engine)
    run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
    evidence, artifacts = [], {}
    for item in (text, discovered):
        observation = DatabaseObservation(executor_request_id=uuid7(), target_id=run.target_id, tool_id=item.tool_id,
            tool_version="1.0.0", variant="test", template_sha256="a" * 64, db_type=engine, db_version=VERSIONS[engine],
            capability_snapshot_hash="b" * 64, captured_at=NOW, duration_ms=1,
            columns=tuple(DatabaseColumn(**column, sensitivity="PUBLIC") for column in item.columns), rows=item.rows,
            row_count=item.row_count, truncated=False, result_sha256=canonical_hash(item.rows), parameters_sha256="c" * 64,
            provenance={"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": dict(item.provenance.physical_identity)})
        evidence.append(item.model_copy(update={"evidence_ref": f"artifact:{item.artifact_id}#a1",
            "columns": tuple(column.model_dump(mode="json") for column in observation.columns), "provenance": database_provenance(observation)}))
        body = {"schema_version": "DBA_TOOL_RESULT.v1", "source_type": "TOOL", "source_id": item.source_id,
            "source_version": "1.0.0", "definition_hash": "a" * 64, "output_schema": "DBA_TOOL_RESULT.v1",
            "measurement_semantics": item.measurement_semantics, "presentation_kind": "TABLE", "status": "SUCCEEDED",
            "tool_outcomes": [{"step_id": "a1", "tool_id": item.tool_id, "tool_version": "1.0.0", "status": "SUCCEEDED",
                "observation": observation.model_dump(mode="json")}], "sampling": None}
        artifacts[str(item.artifact_id)] = SimpleNamespace(ops_run_id=run.ops_run_id, schema_version="DBA_TOOL_RESULT.v1",
            trust_level="SOURCE_VERIFIED", payload_json=body, content_hash=canonical_hash(body))
    draft = DiagnosisAnswerDraft.model_validate({"claims": [{"claim_id": "c1", "kind": "INTERPRETATION", "subject": "解析环境",
        "statement": "原解析环境尚待核验，不能确认实际访问对象。", "level": "UNKNOWN", "applicable_time": NOW,
        "reasoning": "未限定名称只有目录候选。", "limitations": ["未取得原语句解析上下文或匹配执行计划。"]}], "goal_satisfied": False,
        "unresolved_questions": ["原语句的解析环境尚待核验，目录候选不能证明实际访问。"]})
    context = SimpleNamespace(run_id=run.ops_run_id, target_id=run.target_id, task_id=uuid7(), plan_snapshot={
        "capability_snapshot": {"database_type": engine}, "answer_context": {"task_frame": {
            "diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject}}})
    record = freeze_investigation(draft=draft, evidence=tuple(evidence), context=context)
    assert record.sql_object_graph.references[0].candidates
    payload = {"status": "PARTIAL", "blocks": [], "investigation_record": record.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in evidence]}
    normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", payload=payload, source_kind="DIAGNOSIS")
    original = artifacts[str(discovered.artifact_id)]
    reference = payload["investigation_record"]["sql_object_graph"]["references"][0]
    if tamper == "missing":
        artifacts.pop(str(discovered.artifact_id))
    elif tamper == "row":
        original.payload_json["tool_outcomes"][0]["observation"]["rows"][0][0] = "invented"
    elif tamper == "scope":
        original.payload_json["tool_outcomes"][0]["observation"]["provenance"]["instance_identity"] = {}
    elif tamper == "trust":
        original.trust_level = "USER_PROVIDED"
    elif tamper == "run":
        original.ops_run_id = uuid7()
    elif tamper == "reference":
        reference["relation_name"] = "invented"
    elif tamper == "candidate":
        reference["candidates"][0]["schema_name"] = "invented"
    elif tamper == "context":
        reference["resolution"] = "EXPLICIT_SCHEMA"
    original.content_hash = canonical_hash(original.payload_json)
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(side_effect=lambda *, artifact_id: artifacts.get(str(artifact_id)))))
    assert asyncio.run(validate_investigation_sources(uow=uow, run=run, payload=payload)) == (tamper is None)


@pytest.mark.parametrize("engine", ENGINES)
def test_tools_are_versioned_catalog_only_bounded_and_reachable_by_single_sql_profile(engine):
    registry = DiagnosticRegistry.load()
    tool = registry.resolve(tool_id=TOOL, tool_version="1.0.0", db_type=engine, db_version=VERSIONS[engine], capabilities=set(), entitlements=set())
    assert registry.validate_parameters(tool, {"table_name": "orders"}) == {"table_name": "orders", "schema_name": "", "limit": 40}
    assert tool.definition.measurement_semantics == "CURRENT_ACTIVITY"
    assert tool.definition.limit_semantics == "BOUNDED_ENUMERATION" and tool.definition.timeout_seconds == 10
    assert tool.definition.max_rows == 200 and tool.definition.max_bytes == 262144
    for invalid in ({}, {"table_name": "orders", "limit": 201}, {"table_name": "orders", "sql": "SELECT 1"}):
        with pytest.raises(ValueError):
            registry.validate_parameters(tool, invalid)
    compact = SimpleNamespace(implementation_profile="NONE", diagnostic_profile="SINGLE_SQL_PERFORMANCE",
        subject_ref={"subject_type": {"POSTGRESQL": "POSTGRESQL_QUERY_ID", "MYSQL": "MYSQL_DIGEST", "ORACLE": "ORACLE_SQL_ID"}[engine],
            **unqualified(engine)[0]})
    assert TOOL in TurnPlanningService._profile_tool_ids(compact, {"database_type": engine})


@pytest.mark.parametrize("engine", ENGINES)
def test_replanner_receives_rebuilt_sql_names_and_lookup_from_actual_assessment(engine):
    from aiops_agent.application.investigation.reasoner import InvestigationReasoner
    from tests.unit.aiops_agent.test_aiops_investigation_reasoner import _Model, _Prompts, _output

    async def run():
        subject, text, _ = unqualified(engine)
        frame = {"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject}
        graph = build_sql_object_graph(database_type=engine, subject=subject, evidence=(text,))
        payload = _output(tool_id=TOOL)
        payload["plan"]["revision_no"] = 2
        payload["plan"]["actions"][0]["input"] = graph.references[0].lookup.input
        model = _Model(payload)
        await InvestigationReasoner(model, _Prompts()).replan(content=(), conversation_context=(), target_context={"db_type": engine},
            prompt_snapshot={}, source_run_evidence=None, task_frame=frame, prior_plan={"actions": []},
            assessment={"status": "NEEDS_EVIDENCE", "evidence": [text.model_dump(mode="json")]},
            available_tools=({"tool_id": TOOL, "version": "1.0.0"},), available_playbooks=(), model_snapshot={},
            deadline=None, idempotency_key="object-name-replan", revision_no=2)
        bundle = model.calls[0]["input_payload"]["sql_evidence_bundle"]
        reference, = bundle["object_graph"]["references"]
        assert reference["lookup"] == graph.references[0].lookup.model_dump(mode="json")
        assert reference["anchor"]["evidence_ref"] == text.evidence_ref

    asyncio.run(run())


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("count,limit,truncated", [(0, 40, False), (1, 40, False), (40, 40, True), (1, 1, True)])
def test_candidate_enumeration_never_claims_completeness_when_its_limit_is_reached(engine, count, limit, truncated):
    tool = DiagnosticRegistry.load().resolve(tool_id=TOOL, tool_version="1.0.0", db_type=engine,
        db_version=VERSIONS[engine], capabilities=set(), entitlements=set())
    observed = candidate(engine)
    result = DiagnosticExecutorService._normalize(request=SimpleNamespace(executor_request_id=uuid7(), parameters={"limit": limit}),
        grant=SimpleNamespace(target_id=uuid7(), tool_id=TOOL, tool_version="1.0.0", variant=tool.definition.variant,
            template_sha256=tool.definition.template_sha256, db_type=engine, capability_snapshot_hash="a" * 64, parameters_sha256="b" * 64),
        tool=tool, raw=DriverQueryResult(columns=tuple(item["name"] for item in observed.columns), rows=observed.rows * count,
            truncated=False, db_version=VERSIONS[engine]), captured_at=datetime.now(timezone.utc), duration_ms=1,
        limits=DiagnosticLimits(statement_timeout_seconds=10, max_result_rows=200, max_result_bytes=262144))
    assert result.truncated == truncated
    assert ("ENUMERATION_LIMIT_REACHED" in result.warnings) == truncated
