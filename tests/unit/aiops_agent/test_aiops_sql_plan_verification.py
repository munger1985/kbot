"""原生核验、授权隐私和来源重放的替身回归，不等于线上三库验收。"""

import asyncio
from copy import deepcopy
from decimal import Decimal
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.executor.drivers import sql_plan_verification as native
from aiops_agent.domain.evidence.sql_plan_replay import SqlPlanReplayRejected
from aiops_agent.application.errors import AIOpsApplicationError
from platform_core.contracts.aiops.sql_plan_verification import SqlPlanNativeVerification
from aiops_agent.contracts.sql_plan import SqlPlanDocument
from aiops_agent.contracts.sql_object_graph import SqlObjectGraph
from tests.unit.aiops_agent.test_aiops_sql_capture_approval import setup, request, decide, call
from tests.unit.aiops_agent.test_aiops_sql_capture_evidence import scenario, diagnosed
from tests.unit.aiops_agent.test_aiops_sql_plan_native_capture import protected, driver, postgresql, mysql, oracle
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.playbooks import canonical_hash

PRIVATE = "private-native-value-do-not-publish"


def replay(*, parameters=(PRIVATE,), types=("varchar2",), sql="SELECT * FROM mes.t AS a", relations=(("mes", "t"),)):
    return SimpleNamespace(parameters=parameters, native_types=types, sql=sql, relations=relations)


def verification():
    return SqlPlanNativeVerification(
        original_context=tuple(native.check(field) for field in ("DATABASE", "SCHEMA", "ROLE", "SEARCH_PATH", "OPTIMIZER_SETTINGS")),
        parameter_types=native.check("PARAMETER_TYPES"),
        parameter_values=native.check("PARAMETER_VALUES", "NOT_REQUESTED"),
        resolved_objects=({"node_id": 1, "object_kind": "TABLE", "schema_name": "mes", "object_name": "t",
            "native_object_id": "123", "resolution": "NATIVE_CATALOG"},),
        limitations=("PARTIAL_CONTEXT_ONLY", "POPULATION_NOT_OBSERVED"))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("actual,expected,status", [(PRIVATE, PRIVATE, "MATCH"), ("different", PRIVATE, "MISMATCH"),
    (None, PRIVATE, "UNKNOWN"), (PRIVATE, None, "UNKNOWN")])
def test_bind_value_comparison_is_opt_in_and_never_serializes_values(enabled, actual, expected, status):
    rows = [(1, "VARCHAR2(4000)", "YES", 4000, "2026-10-29", actual)]
    types, values = native.compare_captured_binds(rows, replay(parameters=(expected,)), enabled=enabled)
    assert types["status"] == "MATCH"
    assert values["status"] == (status if enabled else "NOT_REQUESTED")
    assert PRIVATE not in json.dumps((types, values))


@pytest.mark.parametrize("index,value", [(0, True), (0, "1"), (2, "NO"), (3, True), (3, 1), (4, None)])
def test_incomplete_boolean_or_truncated_bind_capture_stays_unknown(index, value):
    row = [1, "VARCHAR2(4000)", "YES", 4000, "2026-10-29", PRIVATE]
    row[index] = value
    _, values = native.compare_captured_binds([row], replay(), enabled=True)
    assert values["status"] == "UNKNOWN"


@pytest.mark.parametrize("raw,kind,expected,status", [("12345678901234567890.123456789", "number", Decimal("12345678901234567890.123456789"), "MATCH"),
    ("1,2", "number", Decimal("1.2"), "UNKNOWN"), ("2026-10-29", "date", "2026-10-29", "UNKNOWN"),
    ("NaN", "number", Decimal("1"), "UNKNOWN")])
def test_native_numbers_remain_exact_and_nls_dates_are_not_guessed(raw, kind, expected, status):
    _, values = native.compare_captured_binds([(1, kind.upper(), "YES", 4000, "now", raw)],
        replay(parameters=(expected,), types=(kind,)), enabled=True)
    assert values["status"] == status


@pytest.mark.parametrize("sql,expected", [
    ("SELECT * FROM mes.t AS a", {"a": ("mes", "t")}),
    ("SELECT * FROM mes.t", {"t": ("mes", "t")}),
    ("SELECT * FROM (SELECT * FROM mes.t AS x) AS a", {"x": ("mes", "t")}),
    ("WITH a AS (SELECT * FROM mes.t AS a) SELECT * FROM a", {}),
    ("SELECT * FROM mes.t AS a WHERE EXISTS (SELECT 1 FROM mes.t AS a)", {}),
    ("SELECT * FROM other.t AS a", {}),
])
def test_alias_mapping_rejects_cte_derived_scope_and_cross_scope_ambiguity(sql, expected):
    assert native._aliases(replay(sql=sql), "MYSQL") == expected


def inputs(engine="POSTGRESQL", **updates):
    context = {"database_name": "app", "schema_name": "mes", "role_name": "reader", **updates}
    exchanged = SimpleNamespace(source_subject={"database_oid": "123", "user_oid": "456"},
        native_bind_comparison=False)
    connection = SimpleNamespace(execute=AsyncMock())
    return context, exchanged, connection


def test_pg_native_subject_catalog_and_object_identity_are_separate_from_current_connection():
    context, exchanged, connection = inputs()
    async def fetch(sql, *args):
        if sql == native.PG_SOURCE:
            assert args == (123, 456)
            return [("app", "reader")]
        assert sql == native.PG_OBJECT and args == ("mes", "t")
        return [("789", "r")]
    connection.fetch = AsyncMock(side_effect=fetch)
    result = call(native.native_plan_verification(engine="POSTGRESQL", connection=connection, cursor=None,
        exchanged=exchanged, document={"execution_context": context}, replay=replay(types=("text",)),
        nodes=({"node_id": 0, "schema_name": "mes", "relation_name": "t"},)))
    checks = {check.field: check for check in result.original_context}
    assert checks["DATABASE"].status == checks["ROLE"].status == "MATCH"
    assert checks["SCHEMA"].status == checks["SEARCH_PATH"].status == "UNKNOWN"
    assert result.resolved_objects[0].native_object_id == "789"
    assert result.representativeness == "UNPROVEN"
    assert connection.execute.await_count == 6


def test_pg_optional_permission_failure_rolls_back_savepoint_and_continues_object_resolution():
    context, exchanged, connection = inputs()
    connection.fetch = AsyncMock(side_effect=[PermissionError(PRIVATE), [("789", "r")]])
    result = call(native.native_plan_verification(engine="POSTGRESQL", connection=connection, cursor=None,
        exchanged=exchanged, document={"execution_context": context}, replay=replay(),
        nodes=({"node_id": 0, "schema_name": "mes", "relation_name": "t"},)))
    assert result.resolved_objects and "NATIVE_SOURCE_UNAVAILABLE" in result.limitations
    assert result.original_context[0].status == "UNKNOWN" and PRIVATE not in result.model_dump_json()


def test_pg_failed_savepoint_cleanup_cannot_return_success():
    context, exchanged, connection = inputs()
    connection.fetch = AsyncMock(return_value=[])
    connection.execute.side_effect = [None, RuntimeError(PRIVATE)]
    with pytest.raises(SqlPlanReplayRejected, match="CLEANUP_UNCONFIRMED"):
        call(native.native_plan_verification(engine="POSTGRESQL", connection=connection, cursor=None,
            exchanged=exchanged, document={"execution_context": context}, replay=replay(), nodes=()))


def test_mysql_index_resolution_uses_table_alias_then_checks_index_belongs_to_that_table():
    context, exchanged, connection = inputs()
    exchanged.source_subject = {"schema_name": "app"}
    cursor = SimpleNamespace(execute=AsyncMock(), fetchmany=AsyncMock(side_effect=[[("BASE TABLE",)], [("idx_code",)]]))
    result = call(native.native_plan_verification(engine="MYSQL", connection=connection, cursor=cursor,
        exchanged=exchanged, document={"execution_context": context}, replay=replay(),
        nodes=({"node_id": 0, "relation_name": "a", "index_name": "idx_code"},)))
    assert result.resolved_objects[0].object_name == "idx_code"
    assert cursor.execute.await_args.args == (native.MYSQL_INDEX, ("mes", "t", "idx_code"))
    assert result.parameter_values.status == "UNSUPPORTED"


@pytest.mark.parametrize("allow", [False, True])
def test_oracle_complete_cursor_subject_and_sensitive_bind_projection_are_enforced(allow):
    context, exchanged, connection = inputs(settings={"optimizer_mode": "ALL_ROWS"})
    exchanged.source_subject = {"sql_id": "36apy6dncjggq", "dbid": "123", "inst_id": "1",
        "container_id": "3", "child_number": "0", "plan_hash_value": "456"}
    exchanged.native_bind_comparison = allow
    cursor = SimpleNamespace(execute=AsyncMock(), fetchmany=AsyncMock(side_effect=[
        [("mes", "reader")], [("optimizer_mode", "ALL_ROWS")],
        [(1, "VARCHAR2(4000)", "YES", 4000, "now", PRIVATE if allow else None)], [("789", "TABLE")]]))
    result = call(native.native_plan_verification(engine="ORACLE", connection=connection, cursor=cursor,
        exchanged=exchanged, document={"execution_context": context}, replay=replay(),
        nodes=({"node_id": 0, "schema_name": "mes", "relation_name": "t"},)))
    sql, binds = cursor.execute.await_args_list[2].args
    assert ("b.value_string" in sql) == allow
    assert ("CAST(NULL AS VARCHAR2(4000))" in sql) != allow
    assert binds == {"sql_id": "36apy6dncjggq", "dbid": 123, "inst_id": 1, "container_id": 3, "child_number": 0, "plan_hash_value": 456}
    checks = {item.field: item for item in result.original_context}
    assert checks["SCHEMA"].status == checks["ROLE"].status == checks["OPTIMIZER_SETTINGS"].status == "MATCH"
    assert result.parameter_values.status == ("MATCH" if allow else "NOT_REQUESTED")
    assert PRIVATE not in result.model_dump_json()
    assert result.representativeness == "UNPROVEN"


def test_missing_oracle_subject_never_reads_bind_capture():
    context, exchanged, connection = inputs()
    cursor = SimpleNamespace(execute=AsyncMock())
    result = call(native.native_plan_verification(engine="ORACLE", connection=connection, cursor=cursor,
        exchanged=exchanged, document={"execution_context": context}, replay=replay(), nodes=()))
    cursor.execute.assert_not_awaited()
    assert result.parameter_values.status == "NOT_REQUESTED"


@pytest.mark.parametrize("engine", ["MYSQL", "ORACLE"])
def test_session_settings_are_readonly_and_bound_and_unknown_settings_fail_closed(engine):
    cursor = SimpleNamespace(execute=AsyncMock(), fetchmany=AsyncMock(return_value=[("optimizer_mode", "ALL_ROWS")]))
    result = call(native.capture_session_settings(cursor, {"optimizer_mode": "ALL_ROWS"}, engine))
    assert result == {"optimizer_mode": "ALL_ROWS"}
    sql, binds = cursor.execute.await_args.args
    assert "ALL_ROWS" not in sql and not sql.startswith(("ALTER", "SET"))
    assert "optimizer_mode" in str(binds)
    cursor.fetchmany.return_value *= 2
    with pytest.raises(SqlPlanReplayRejected):
        call(native.capture_session_settings(cursor, {"optimizer_mode": "ALL_ROWS"}, engine))


@pytest.mark.parametrize("engine", ["MYSQL", "ORACLE"])
def test_capture_driver_can_verify_explicit_current_settings_before_explain(monkeypatch, engine):
    if engine == "MYSQL":
        _, cursor = mysql(monkeypatch)
    else:
        _, cursor, _ = oracle(monkeypatch)
    original_fetch = cursor.fetchmany
    async def fetch(limit):
        sql = cursor.executed[-1][0]
        if sql.startswith(("SELECT VARIABLE_NAME", "SELECT name, value")):
            return [("optimizer_mode", "ALL_ROWS")]
        return await original_fetch(limit)
    cursor.fetchmany = fetch
    item = protected(engine).protected_transport()
    doc = json.loads(item["protected_document"])
    doc["execution_context"]["settings"] = {"optimizer_mode": "ALL_ROWS"}
    item["protected_document"] = json.dumps(doc)
    from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureExchange
    assert call(driver().capture(SqlPlanCaptureExchange.model_validate(item)))[0]


@pytest.mark.parametrize("engine,enabled", [("ORACLE", True), ("ORACLE", False), ("POSTGRESQL", True)])
def test_native_bind_comparison_requires_policy_and_separate_explicit_ack(engine, enabled):
    store = setup(engine)
    store.policy.rules_json["sql_plan_capture"]["native_bind_comparison_enabled"] = enabled
    view = request(store)
    required = engine == "ORACLE" and enabled
    assert view.native_bind_comparison == required
    with pytest.raises(AIOpsApplicationError):
        decide(store, view, acknowledge_native_bind_comparison=not required)
    approved = decide(store, view, acknowledge_native_bind_comparison=required)
    assert approved.state == "APPROVED"
    assert PRIVATE not in approved.model_dump_json()


@pytest.mark.parametrize("index", [-1, True, 128])
def test_invalid_unresolved_node_numbers_are_rejected(index):
    data = verification().model_dump(mode="json")
    data["unresolved_object_nodes"] = [index]
    with pytest.raises(ValidationError):
        SqlPlanNativeVerification.model_validate(data)


def test_absent_verification_does_not_change_historical_document_or_graph_hash_shape():
    data = {"database_type": "POSTGRESQL", "native_format": "POSTGRESQL_EXPLAIN_JSON",
        "nodes": [{"node_id": 0, "operation": "Result"}]}
    document = SqlPlanDocument.model_validate(data)
    assert "native_verification" not in document.model_dump(mode="json")
    assert canonical_hash(document.model_dump(mode="json")) == canonical_hash(SqlPlanDocument.model_validate(
        document.model_dump(mode="json")).model_dump(mode="json"))
    graph = SqlObjectGraph(database_type="POSTGRESQL", subject={"query_id": "123"}, limitations=("边界",))
    assert "plan_objects" not in graph.model_dump(mode="json")


def test_verification_flows_from_independent_receipt_to_bundle_and_object_graph():
    store = scenario(native_verification=verification())
    async def run():
        _, assessment = await diagnosed(store)
        fact = next(item for item in assessment.evidence if item.sql_plan is not None)
        assert fact.sql_plan.native_verification == verification()
        bundle = build_sql_evidence_bundle(task_frame=store.frame, database_type="POSTGRESQL", evidence=assessment.evidence)
        assert bundle["plan_assessments"][0]["native_verification"]["resolved_objects"][0]["native_object_id"] == "123"
        graph = bundle["object_graph"]
        assert graph["plan_objects"][0]["source"] == "CONTROLLED_ESTIMATE_OBJECT"
        assert not graph["plan_objects"][0]["production_execution_verified"]
        assert await validate_investigation_sources(uow=store, run=store.current,
            payload={"evidence": [fact.model_dump(mode="json")]}, plan_capture_reader=store.reader)
        changed = fact.model_dump(mode="json")
        changed["sql_plan"]["native_verification"]["resolved_objects"][0]["object_name"] = "forged"
        assert not await validate_investigation_sources(uow=store, run=store.current,
            payload={"evidence": [changed]}, plan_capture_reader=store.reader)
    asyncio.run(run())


def test_native_verification_cancellation_propagates_and_pg_savepoint_is_cleaned():
    context, exchanged, connection = inputs()
    connection.fetch = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        call(native.native_plan_verification(engine="POSTGRESQL", connection=connection, cursor=None,
            exchanged=exchanged, document={"execution_context": context}, replay=replay(), nodes=()))
    assert [item.args[0] for item in connection.execute.await_args_list] == [
        "SAVEPOINT aiops_native_witness", "ROLLBACK TO SAVEPOINT aiops_native_witness",
        "RELEASE SAVEPOINT aiops_native_witness"]


def test_optional_native_timeout_is_bounded_and_keeps_explicit_unknown(monkeypatch):
    context, exchanged, connection = inputs()
    timeout = asyncio.timeout
    monkeypatch.setattr(native.asyncio, "timeout", lambda seconds: timeout(0.01 if seconds == 2 else seconds))
    async def wait(*_):
        await asyncio.Event().wait()
    connection.fetch = wait
    result = call(native.native_plan_verification(engine="POSTGRESQL", connection=connection, cursor=None,
        exchanged=exchanged, document={"execution_context": context}, replay=replay(),
        nodes=({"node_id": 0, "schema_name": "mes", "relation_name": "t"},)))
    assert result.unresolved_object_nodes == (0,) and "NATIVE_SOURCE_UNAVAILABLE" in result.limitations
    assert all(check.status == "UNKNOWN" for check in result.original_context)


@pytest.mark.parametrize("engine,enabled", [("ORACLE", True), ("ORACLE", False), ("POSTGRESQL", True), ("MYSQL", True)])
def test_production_exchange_uses_frozen_approved_native_bind_option(engine, enabled):
    from tests.unit.aiops_agent.test_aiops_sql_capture_execution import execution_store, claimed
    store = execution_store(engine, native_bind_comparison=enabled)
    claimed(store)
    exchanged = call(store.execution.exchange("signed-test"))
    assert exchanged.native_bind_comparison == (engine == "ORACLE" and enabled)


@pytest.mark.parametrize("enabled", [False, True])
def test_control_plane_result_cannot_claim_bind_comparison_without_approved_privacy_scope(enabled):
    from tests.unit.aiops_agent.test_aiops_sql_capture_execution import execution_store, claimed, report
    from aiops_agent.application.execution_fence import ExecutionFenceError
    store = execution_store("ORACLE", native_bind_comparison=enabled)
    claimed(store)
    exchanged = call(store.execution.exchange("signed-test"))
    data = verification().model_dump(mode="json")
    data.update(resolved_objects=[])
    data["parameter_values"] = native.check("PARAMETER_VALUES", "MATCH", "NATIVE_BIND_CAPTURE")
    command = report(store, exchanged.resource_lease, native_verification=SqlPlanNativeVerification.model_validate(data))
    if enabled:
        assert call(store.execution.report(command)).native_verification.parameter_values.status == "MATCH"
    else:
        with pytest.raises(ExecutionFenceError, match="RESULT_INVALID"):
            call(store.execution.report(command))


def test_oracle_parsing_user_lookup_never_borrows_current_container_user_for_other_pdb():
    assert "AND s.con_id=TO_NUMBER(SYS_CONTEXT('USERENV','CON_ID'))" in native.ORACLE_SOURCE
    assert "LEFT JOIN SYS.ALL_USERS" in native.ORACLE_SOURCE


def test_plan_object_field_reference_tracks_actual_projected_row_not_node_number():
    from aiops_agent.domain.evidence.sql_object_graph import matched_plan_objects
    store = scenario(native_verification=verification())
    async def run():
        _, assessment = await diagnosed(store)
        facts = list(assessment.evidence)
        index = next(index for index, fact in enumerate(facts) if fact.sql_plan is not None)
        fact = facts[index]
        facts[index] = fact.model_copy(update={"rows": tuple(reversed(fact.rows)),
            "sql_plan": fact.sql_plan.model_copy(update={"nodes": tuple(reversed(fact.sql_plan.nodes))})})
        objects = matched_plan_objects(database_type="POSTGRESQL", subject=store.frame["subject_ref"], evidence=tuple(facts))
        assert objects[0].node_id == 1 and objects[0].field_ref.row_index == 0
        assert not objects[0].production_execution_verified
    asyncio.run(run())
