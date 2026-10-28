"""视图及同义词目录取证必须贯穿正式计划、执行、关系重放及报告来源核验。"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
from aiops_agent.application.diagnosis.provenance import database_provenance
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.application.investigation.discovery import available_tools, build_playbook_plan
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
from aiops_agent.diagnostics.grants import DiagnosticGrantCodec
from aiops_agent.domain.evidence.sql_object_graph import build_sql_object_graph, replay_sql_object_graph
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
from platform_core.identity import uuid7
from tests.support.aiops_resources import bind_resource_control
from tests.unit.aiops_agent.test_aiops_sql_object_graph import NOW, fact, scenario


TOOL = "db.sql.object_dependencies"
ENGINES = ("POSTGRESQL", "MYSQL", "ORACLE")


def dependency(engine, source="orders", target="order_items", *, kind="VIEW", target_kind="TABLE", scope="LOCAL", **updates):
    schema = "MES" if engine == "ORACLE" else "mes"
    if engine == "ORACLE":
        source, target = source.upper(), target.upper()
    return fact(engine, TOOL, {"schema_name": schema, "table_name": source, "relation_kind": kind,
        "dependency_schema_name": schema, "dependency_name": target, "dependency_kind": target_kind,
        "reference_scope": scope}, **updates)


@pytest.mark.parametrize("engine", ENGINES)
def test_dependency_chain_reaches_only_related_base_table_indexes_independent_of_input_order(engine):
    subject, text, indexes = scenario(engine)
    first = dependency(engine, target="order_view", target_kind="VIEW")
    second = dependency(engine, source="order_view", target="base_orders")
    base_name = "BASE_ORDERS" if engine == "ORACLE" else "base_orders"
    indexes = indexes.model_copy(update={"rows": ((indexes.rows[0][0], base_name, base_name, "base_idx", "r"),)})
    unrelated = dependency(engine, source="secret", target="unrelated")
    for evidence in ((text, first, second, indexes, unrelated), (unrelated, indexes, second, first, text)):
        graph = build_sql_object_graph(database_type=engine, subject=subject, evidence=evidence)
        kinds = [edge.kind for edge in graph.relationships]
        assert kinds.count("CATALOG_VIEW_DEPENDS_ON") == 2
        assert kinds.count("CATALOG_INDEX_ON") == 1
        assert "CATALOG_NAME_MATCH" in kinds
        assert all(edge.relation_name.lower() not in {"secret", "unrelated"} for edge in graph.relationships)
        assert any(edge.kind == "CATALOG_VIEW_DEPENDS_ON" and edge.related.field == "dependency_name"
            and edge.anchor.field == "table_name" and edge.anchor.row_index == edge.related.row_index
            for edge in graph.relationships)
        replay_sql_object_graph(graph, evidence)


def test_oracle_synonym_then_view_expands_but_remote_synonym_never_borrows_a_local_table():
    subject, text, indexes = scenario("ORACLE")
    synonym = dependency("ORACLE", target="order_view", kind="SYNONYM", target_kind="UNKNOWN")
    view = dependency("ORACLE", source="order_view", target="base_orders")
    graph = build_sql_object_graph(database_type="ORACLE", subject=subject, evidence=(text, synonym, view))
    assert {"CATALOG_SYNONYM_REFERENCES", "CATALOG_VIEW_DEPENDS_ON"} <= {edge.kind for edge in graph.relationships}
    remote = dependency("ORACLE", target="orders", kind="SYNONYM", scope="REMOTE")
    graph = build_sql_object_graph(database_type="ORACLE", subject=subject, evidence=(text, remote, indexes))
    assert not any(edge.kind == "CATALOG_SYNONYM_REFERENCES" for edge in graph.relationships)
    assert any("远端引用" in item for item in graph.limitations)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("corrupt", ["untrusted", "scope", "unknown_scope", "old", "naive", "boot", "case", "missing_name", "remote", "unknown_reference_scope", "wrong_kind"])
def test_dependency_cannot_bridge_unknown_mismatched_or_remote_objects(engine, corrupt):
    subject, text, _ = scenario(engine)
    first = dependency(engine)
    if corrupt == "untrusted":
        first = first.model_copy(update={"trust_level": "USER_PROVIDED"})
    elif corrupt in {"scope", "boot", "unknown_scope"}:
        physical = dict(first.provenance.physical_identity)
        key = "instance_started_at" if corrupt == "boot" else {"POSTGRESQL": "system_identifier", "MYSQL": "server_uuid", "ORACLE": "dbid"}[engine]
        physical[key] = "other"
        first = first.model_copy(update={"provenance": None if corrupt == "unknown_scope" else first.provenance.model_copy(update={"physical_identity": tuple(physical.items())})})
    elif corrupt in {"old", "naive"}:
        first = first.model_copy(update={"captured_at": "2026-10-28T00:06:00+00:00" if corrupt == "old" else "2026-10-28T00:00:00"})
    else:
        row = dict(zip((column["name"] for column in first.columns), first.rows[0], strict=True))
        row.update({"case": {"table_name": "Orders"}, "missing_name": {"dependency_name": None},
            "remote": {"reference_scope": "REMOTE"}, "unknown_reference_scope": {"reference_scope": "UNKNOWN"},
            "wrong_kind": {"relation_kind": "PROCEDURE"}}[corrupt])
        first = first.model_copy(update={"rows": (tuple(row.values()),)})
    graph = build_sql_object_graph(database_type=engine, subject=subject, evidence=(text, first))
    assert not any(edge.kind in {"CATALOG_VIEW_DEPENDS_ON", "CATALOG_SYNONYM_REFERENCES"} for edge in graph.relationships)


def test_dependency_cycle_and_depth_bound_are_explicit_and_do_not_leak_disconnected_objects():
    subject, text, _ = scenario()
    loop = (dependency("POSTGRESQL", target="next", target_kind="VIEW"),
            dependency("POSTGRESQL", source="next", target="orders", target_kind="VIEW"))
    graph = build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text, *loop))
    assert len([edge for edge in graph.relationships if edge.kind == "CATALOG_VIEW_DEPENDS_ON"]) == 2
    assert not graph.truncated
    chain = tuple(dependency("POSTGRESQL", source="orders" if index == 0 else f"view_{index}",
        target=f"view_{index + 1}", target_kind="VIEW") for index in range(6))
    graph = build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text, *chain))
    assert graph.truncated
    assert len([edge for edge in graph.relationships if edge.kind == "CATALOG_VIEW_DEPENDS_ON"]) == 4
    assert not any(edge.relation_name == "view_5" for edge in graph.relationships)


def test_dependency_chain_cannot_accumulate_time_drift_across_five_minute_hops():
    subject, text, _ = scenario()
    first = dependency("POSTGRESQL", target="next", target_kind="VIEW", captured_at="2026-10-28T00:04:00+00:00")
    second = dependency("POSTGRESQL", source="next", target="base", captured_at="2026-10-28T00:08:00+00:00")
    graph = build_sql_object_graph(database_type="POSTGRESQL", subject=subject, evidence=(text, first, second))
    assert len([edge for edge in graph.relationships if edge.kind == "CATALOG_VIEW_DEPENDS_ON"]) == 1
    assert not any(edge.relation_name == "base" for edge in graph.relationships)


def test_remote_reference_outside_the_selected_sql_does_not_change_its_limitations():
    subject, text, _ = scenario("ORACLE")
    local = dependency("ORACLE")
    remote = dependency("ORACLE", source="unrelated", scope="REMOTE")
    graph = build_sql_object_graph(database_type="ORACLE", subject=subject, evidence=(text, local, remote))
    assert not any("远端引用" in item for item in graph.limitations)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("count,parameters,truncated", [
    (0, {}, False), (1, {}, False), (40, {}, True), (1, {"limit": 1}, True), (1, {"limit": 2}, False),
])
def test_catalog_reaching_requested_or_default_limit_cannot_claim_complete_enumeration(engine, count, parameters, truncated):
    registry = DiagnosticRegistry.load()
    tool = registry.resolve(tool_id=TOOL, tool_version="1.0.0", db_type=engine,
        db_version={"POSTGRESQL": "16", "MYSQL": "8.4", "ORACLE": "19"}[engine], capabilities=set(), entitlements=set())
    dep = dependency(engine)
    raw = DriverQueryResult(columns=tuple(column["name"] for column in dep.columns),
        rows=dep.rows * count, truncated=False, db_version="16")
    definition = tool.definition
    observation = DiagnosticExecutorService._normalize(
        request=SimpleNamespace(executor_request_id=uuid7(), parameters=parameters),
        grant=SimpleNamespace(target_id=uuid7(), tool_id=TOOL, tool_version=definition.version,
            variant=definition.variant, template_sha256=definition.template_sha256, db_type=engine,
            capability_snapshot_hash="a" * 64, parameters_sha256="b" * 64),
        tool=tool, raw=raw, captured_at=datetime.now(timezone.utc), duration_ms=1,
        limits=DiagnosticLimits(statement_timeout_seconds=10, max_result_rows=200, max_result_bytes=262144))
    assert observation.row_count == count and observation.truncated == truncated
    assert ("ENUMERATION_LIMIT_REACHED" in observation.warnings) == truncated
    with pytest.raises(ValueError, match="正整数"):
        type(definition).model_validate({**definition.model_dump(), "parameters": []})


@pytest.mark.parametrize("engine", ENGINES)
def test_dependency_tool_is_versioned_exact_bounded_and_contains_only_native_catalog_reads(engine):
    registry = DiagnosticRegistry.load()
    tool = registry.resolve(tool_id=TOOL, tool_version="1.0.0", db_type=engine,
        db_version={"POSTGRESQL": "16", "MYSQL": "8.4", "ORACLE": "19"}[engine], capabilities=set(), entitlements=set())
    params = {"schema_name": "Quoted Schema", "table_name": "View'Name"}
    assert registry.validate_parameters(tool, params) == {**params, "limit": 40}
    for invalid in ({}, {"schema_name": "mes"}, {**params, "limit": 201}, {**params, "sql": "SELECT 1"}):
        with pytest.raises(ValueError):
            registry.validate_parameters(tool, invalid)
    assert tool.definition.cost_level == "LOW" and tool.definition.timeout_seconds == 10
    assert tool.definition.max_rows == 200 and tool.definition.max_bytes == 262144
    assert "VIEW_TABLE_USAGE" in tool.sql if engine == "MYSQL" else "pg_catalog.pg_depend" in tool.sql if engine == "POSTGRESQL" else "all_synonyms" in tool.sql
    compact = SimpleNamespace(implementation_profile="NONE", diagnostic_profile="SINGLE_SQL_PERFORMANCE",
        subject_ref={"subject_type": {"POSTGRESQL": "POSTGRESQL_QUERY_ID", "MYSQL": "MYSQL_DIGEST", "ORACLE": "ORACLE_SQL_ID"}[engine],
            **scenario(engine)[0]})
    assert TOOL in TurnPlanningService._profile_tool_ids(compact, {"database_type": engine})


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("failure", [None, "PRIVILEGE_MISSING", "OBJECT_NOT_FOUND"])
def test_dependency_flows_through_formal_plan_signed_worker_executor_assessment_and_sql_bundle(engine, failure):
    async def run():
        registry, playbooks = DiagnosticRegistry.load(), PlaybookRegistry.load()
        builder = ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=registry)
        version = {"POSTGRESQL": "16", "MYSQL": "8.4", "ORACLE": "19"}[engine]
        capabilities = DbaCapabilitySnapshot(agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
            database_type=engine, database_version=version, target_enabled=True, target_reachable=True,
            target_capabilities=("DB_READONLY",))
        card = next(card for card in available_tools(builder, capabilities) if card["tool_id"] == TOOL)
        assert card["input"]["schema_name"]["required"] and "reference_scope" in card["returns"]
        dep = dependency(engine)
        selected = InvestigationAction(action_id="a1", tool_id=TOOL, question="视图引用了哪些对象？",
            input={"schema_name": dep.rows[0][0], "table_name": dep.rows[0][1]},
            expected_evidence_kind="OBJECT_DEPENDENCIES", measurement_semantics="CURRENT_ACTIVITY")
        plan = build_playbook_plan(playbooks)
        compiled = InvestigationTaskCompiler(playbooks).compile(plan, investigation_actions=(selected,))
        scope = ({"tenant_id": str(uuid7()), "domain_id": str(uuid7())}
            if "tenant_id" in DiagnosticExecutionGrant.model_fields else {"domain_id": 7})
        execution = builder.build(plan=plan, compiled=compiled, capabilities=capabilities, direct_actions=(selected,),
            database_execution={**scope, "target_row_version": 1,
                "db_type": engine, "configured_version": version, "diagnostic_credential_id": str(uuid7()),
                "automatic_access_enabled": True, "connection_profile": {"host": "db.internal",
                    "port": 1521 if engine == "ORACLE" else 5432 if engine == "POSTGRESQL" else 3306,
                    **({"service": "PDB1"} if engine == "ORACLE" else {"database": "demo"})}})
        task = next(item for item in compiled.tasks if item.task_key == "diagnostic:a1")
        assert task.handler_id == "dba.diagnostic.invoke"
        now = datetime.now(timezone.utc)
        ctx = TaskExecutionContext(run_id=str(uuid7()), task_id=str(uuid7()), task_key=task.task_key,
            target_id=str(uuid7()), agent_id=str(uuid7()), actor_id=str(uuid7()), trigger_type="CHAT", trace_id="view-dependencies",
            attempt=1, max_attempts=1, deadline_at=(now + timedelta(seconds=90)).isoformat(),
            lease_until=(now + timedelta(seconds=90)).isoformat(), lease_token="lease-dependencies",
            plan_snapshot={"investigation_execution": execution}, policy_snapshot={}, input_artifacts=())
        codec = DiagnosticGrantCodec(secret="s" * 32, issuer="test-worker", audience="test-executor")
        control = AsyncMock()
        control.issue_credential.return_value = SimpleNamespace(username="readonly", password="never-export")
        bind_resource_control(control, codec)

        async def raw(**kwargs):
            assert kwargs["tool"].definition.tool_id == TOOL
            assert kwargs["parameters"] == {"schema_name": dep.rows[0][0], "table_name": dep.rows[0][1], "limit": 40}
            if failure:
                raise DiagnosticDriverError(failure)
            return DriverQueryResult(columns=tuple(column["name"] for column in dep.columns), rows=dep.rows,
                truncated=False, db_version=version, instance_identity=dep.provenance.physical_identity)

        driver = SimpleNamespace(db_type=engine, execute=AsyncMock(side_effect=raw))
        executor = DiagnosticExecutorService(registry=registry, grant_codec=codec, control_plane=control, drivers=(driver,),
            hard_limits=DiagnosticLimits(statement_timeout_seconds=20, max_result_rows=1000, max_result_bytes=1048576), concurrency=2)
        async def invoke(request, **kwargs):
            grant = codec.verify(request.grant)
            assert str(grant.target_id) == str(ctx.target_id) and str(grant.task_id) == str(ctx.task_id)
            assert grant.parameters_sha256 and grant.template_sha256 == execution["direct_invocations"][task.task_key]["tool"]["template_sha256"]
            return await executor.execute(request)
        client = SimpleNamespace(execute_diagnostic=AsyncMock(side_effect=invoke))
        database = DatabaseDiagnosticHandler(executor_client=client, grant_codec=codec,
            grant_issuer="test-worker", grant_audience="test-executor", grant_ttl_seconds=45)
        result = await DbaDiagnosticToolHandler(database_handler=database).execute(ctx)
        assert control.issue_credential.await_count == driver.execute.await_count == 1
        if failure:
            assert result.status == "FAILED" and result.tool_outcomes[0].gap.code == failure
            assert result.tool_outcomes[0].observation is None
            return
        assert result.status == "SUCCEEDED" and result.tool_outcomes[0].observation.rows == dep.rows
        assessment = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": result.schema_version, "artifact_id": uuid7(), "payload": result.model_dump(mode="json")},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT"))
        subject, text, _ = scenario(engine)
        observed = assessment.evidence[0]
        # 使用同一组当前采集时间，保留执行器返回的同连接身份而非替身手填图。
        text = text.model_copy(update={"captured_at": observed.captured_at})
        bundle = build_sql_evidence_bundle(task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject},
            database_type=engine, evidence=(text, observed))
        assert any(edge["kind"] == "CATALOG_VIEW_DEPENDS_ON" for edge in bundle["object_graph"]["relationships"])
        assert "never-export" not in result.model_dump_json()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("tamper", [None, "row", "missing", "scope", "trust", "run", "graph"])
def test_answer_report_and_case_replay_dependency_from_original_tool_not_self_consistent_copy(engine, tamper):
    subject, text, _ = scenario(engine)
    dep = dependency(engine)
    run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
    evidence, artifacts = [], {}
    for item in (text, dep):
        observation = DatabaseObservation(executor_request_id=uuid7(), target_id=run.target_id,
            tool_id=item.tool_id, tool_version="1.0.0", variant="test", template_sha256="a" * 64,
            db_type=engine, db_version="16" if engine == "POSTGRESQL" else "8.4" if engine == "MYSQL" else "19",
            capability_snapshot_hash="b" * 64, captured_at=NOW, duration_ms=1,
            columns=tuple(DatabaseColumn(**column, sensitivity="PUBLIC") for column in item.columns),
            rows=item.rows, row_count=item.row_count, truncated=False, result_sha256=canonical_hash(item.rows),
            parameters_sha256="c" * 64, provenance={"identity_scope": "SAME_CONNECTION_CURRENT",
                "instance_identity": dict(item.provenance.physical_identity)})
        evidence.append(item.model_copy(update={"evidence_ref": f"artifact:{item.artifact_id}#a1",
            "columns": tuple(column.model_dump(mode="json") for column in observation.columns),
            "provenance": database_provenance(observation)}))
        body = {"schema_version": "DBA_TOOL_RESULT.v1", "source_type": "TOOL", "source_id": item.source_id,
            "source_version": "1.0.0", "definition_hash": "a" * 64, "output_schema": "DBA_TOOL_RESULT.v1",
            "measurement_semantics": item.measurement_semantics, "presentation_kind": "TABLE", "status": "SUCCEEDED",
            "tool_outcomes": [{"step_id": "a1", "tool_id": item.tool_id, "tool_version": "1.0.0",
                "status": "SUCCEEDED", "observation": observation.model_dump(mode="json")}], "sampling": None}
        artifacts[str(item.artifact_id)] = SimpleNamespace(ops_run_id=run.ops_run_id, schema_version="DBA_TOOL_RESULT.v1",
            trust_level="SOURCE_VERIFIED", payload_json=body, content_hash=canonical_hash(body))
    first = evidence[0]
    text_field = {"POSTGRESQL": "normalized_statement", "MYSQL": "digest_text", "ORACLE": "sql_text"}[engine]
    draft = DiagnosisAnswerDraft.model_validate({"claims": [{"claim_id": "c1", "kind": "FACT", "subject": "SQL",
        "statement": "语句直接引用的对象可连接原生视图依赖", "level": "OBSERVED", "applicable_time": NOW,
        "evidence_fields": [{"evidence_ref": first.evidence_ref, "row_index": 0, "field": text_field}],
        "reasoning": "语句文本及同物理范围原生依赖"}], "goal_satisfied": True})
    context = SimpleNamespace(run_id=run.ops_run_id, target_id=run.target_id, task_id=uuid7(), plan_snapshot={
        "capability_snapshot": {"database_type": engine}, "answer_context": {"task_frame": {
            "diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject}}})
    record = freeze_investigation(draft=draft, evidence=tuple(evidence), context=context)
    assert any(edge.kind == "CATALOG_VIEW_DEPENDS_ON" for edge in record.sql_object_graph.relationships)
    payload = {"status": "COMPLETED", "blocks": [], "investigation_record": record.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in evidence]}
    normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", payload=payload, source_kind="DIAGNOSIS")
    original = artifacts[str(dep.artifact_id)]
    if tamper == "graph":
        payload["investigation_record"]["sql_object_graph"]["relationships"][-1]["relation_name"] = "invented"
    elif tamper == "missing":
        artifacts.pop(str(dep.artifact_id))
    elif tamper == "row":
        original.payload_json["tool_outcomes"][0]["observation"]["rows"][0][4] = "invented"
    elif tamper == "scope":
        original.payload_json["tool_outcomes"][0]["observation"]["provenance"]["instance_identity"] = {}
    elif tamper == "trust":
        original.trust_level = "USER_PROVIDED"
    elif tamper == "run":
        original.ops_run_id = uuid7()
    original.content_hash = canonical_hash(original.payload_json)
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(side_effect=lambda *, artifact_id: artifacts.get(str(artifact_id)))))
    assert asyncio.run(validate_investigation_sources(uow=uow, run=run, payload=payload)) == (tamper is None)
