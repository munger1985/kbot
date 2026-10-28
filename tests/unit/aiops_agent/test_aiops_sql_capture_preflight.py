"""三库预检、参数词法以及生产材料/Driver 链路；替身验证不冒充目标库验收。"""

import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.domain.evidence.sql_bindings import postgresql_named_bindings, SqlBindingInvalid
from aiops_agent.domain.evidence.sql_plan_capture import inspect_plan_capture, source_sql_text
from aiops_agent.executor.drivers.base import DiagnosticDriverError
from aiops_agent.executor.drivers.postgresql import PostgreSQLDiagnosticDriver
from tests.unit.aiops_agent.test_aiops_sql_plan_material import Store, command, document, SECRET
from aiops_agent.playbooks import canonical_hash
from aiops_agent.ports.secret_store import ResolvedSecret
from platform_core.contracts.aiops.executor import DiagnosticConnectionProfile, DiagnosticLimits


def driver_arguments():
    return {"profile": DiagnosticConnectionProfile(host="db.internal", port=5432, database="app", tls_enabled=False),
        "secret": ResolvedSecret(values={"username": "readonly", "password": "test-only"}, fingerprint="test"),
        "limits": DiagnosticLimits(statement_timeout_seconds=5, max_result_rows=2, max_result_bytes=65536),
        "trace_id": "lexical-bind-test"}


@pytest.mark.parametrize("sql,expected,names", [
    ("SELECT :limit, :limit, :other", "SELECT $1, $1, $2", ("limit", "other")),
    ("SELECT ':hidden', :count::int -- :comment", "SELECT ':hidden', $1::int -- :comment", ("count",)),
    ('SELECT "a:quoted", :real /* :not_a_bind */', 'SELECT "a:quoted", $1 /* :not_a_bind */', ("real",)),
    ("SELECT $$:hidden $1$$, $tag$:also_hidden$tag$, :value", "SELECT $$:hidden $1$$, $tag$:also_hidden$tag$, $1", ("value",)),
    ("SELECT 1 /* outer /* :nested */ :ignored */ + :n", "SELECT 1 /* outer /* :nested */ :ignored */ + $1", ("n",)),
    ("SELECT E'quoted\\\' :hidden', :val", "SELECT E'quoted\\\' :hidden', $1", ("val",)),
    ("SELECT :Name, :name, ':Name'", "SELECT $1, $2, ':Name'", ("Name", "name")),
    ("SELECT 'http://example/:name', '12:30'::time", "SELECT 'http://example/:name', '12:30'::time", ()),
    ("SELECT arr[1:upper], arr[:lower : :higher] FROM t", "SELECT arr[1:upper], arr[$1 : $2] FROM t", ("lower", "higher")),
])
def test_postgresql_lexical_replacement_preserves_every_non_bind_byte(sql, expected, names):
    assert postgresql_named_bindings(sql) == (expected, names)


@pytest.mark.parametrize("sql", ["SELECT $1", "SELECT :n, $2", "SELECT :name, 'unclosed"])
def test_invalid_or_mixed_bindings_do_not_escape_as_native_protocol(sql):
    with pytest.raises(SqlBindingInvalid) as caught:
        postgresql_named_bindings(sql)
    assert sql not in str(caught.value)


@pytest.mark.parametrize("engine,sql,count", [
    ("POSTGRESQL", "SELECT date_trunc($3, ts) FROM mes.t WHERE id=$1 AND n=$2", 3),
    ("POSTGRESQL", "WITH a AS (SELECT id FROM mes.t WHERE id=$1) SELECT * FROM a", 1),
    ("POSTGRESQL", "SELECT id FROM mes.a UNION ALL SELECT id FROM mes.b", 0),
    ("POSTGRESQL", "SELECT arr[1:upper] FROM mes.t", 0),
    ("MYSQL", "SELECT * FROM mes.t WHERE id=? AND name=?", 2),
    ("MYSQL", "SELECT '?' AS literal, `?` FROM mes.t WHERE id=? /* ? */", 1),
    ("ORACLE", "SELECT * FROM MES.T WHERE ID=:1 OR ID=:1", 1),
    ("ORACLE", "SELECT ':1' FROM DUAL WHERE :2=:1", 2),
])
def test_three_engines_static_acceptance_is_neither_authority_nor_native_validation(engine, sql, count):
    result = inspect_plan_capture(database_type=engine, sql=sql, parameter_count=count)
    assert result.status == "AWAITING_DATABASE_VALIDATION" and result.expected_parameter_count == count
    assert not result.execution_authorized and not result.reason_codes
    assert set(result.required_checks) == {"SOURCE_SQL_REPLAYABILITY", "ORIGINAL_PARSE_CONTEXT",
        "NATIVE_PARAMETER_TYPES", "OBJECT_AND_FUNCTION_ACCESS", "CAPTURE_POLICY", "INDEPENDENT_APPROVAL"}
    assert result.oracle_parameter_peeking == ("NOT_PROVEN" if engine == "ORACLE" else "NOT_APPLICABLE")


@pytest.mark.parametrize("engine,sql,count,code", [
    ("POSTGRESQL", None, 0, "SOURCE_SQL_TEXT_MISSING"),
    ("POSTGRESQL", "x" * 20001, 0, "SOURCE_SQL_LENGTH_EXCEEDED"),
    ("POSTGRESQL", "SELECT 1; SELECT 2", 0, "SINGLE_STATEMENT_REQUIRED"),
    ("POSTGRESQL", "INSERT INTO mes.t VALUES (1)", 0, "ESTIMATE_SELECT_ONLY"),
    ("POSTGRESQL", "WITH x AS (DELETE FROM mes.t RETURNING *) SELECT * FROM x", 0, "SIDE_EFFECT_OR_LOCK_FORBIDDEN"),
    ("POSTGRESQL", "SELECT * INTO tmp FROM mes.t", 0, "SIDE_EFFECT_OR_LOCK_FORBIDDEN"),
    ("POSTGRESQL", "SELECT * FROM mes.t FOR UPDATE", 0, "SIDE_EFFECT_OR_LOCK_FORBIDDEN"),
    ("POSTGRESQL", "SELECT * FROM mes.t WHERE id=$3", 3, "PARAMETER_LAYOUT_INCOMPLETE"),
    ("POSTGRESQL", "SELECT $0", 0, "PARAMETER_LAYOUT_INCOMPLETE"),
    ("POSTGRESQL", "SELECT $65", 64, "PARAMETER_LAYOUT_INCOMPLETE"),
    ("POSTGRESQL", "SELECT * FROM mes.t WHERE id=$1", 2, "PARAMETER_COUNT_MISMATCH"),
    ("POSTGRESQL", "SELECT * FROM mes.t WHERE id=:id", 1, "NAMED_BIND_MAPPING_REQUIRED"),
    ("POSTGRESQL", "SELECT * FROM mes.t WHERE id=?", 0, "NATIVE_BIND_STYLE_REQUIRED"),
    ("POSTGRESQL", "SELECT $1suffix", 1, "NATIVE_BIND_STYLE_REQUIRED"),
    ("MYSQL", "SELECT * FROM mes.t WHERE id=:id", 1, "NATIVE_BIND_STYLE_REQUIRED"),
    ("MYSQL", "SELECT * FROM mes.t WHERE id=?", 0, "PARAMETER_COUNT_MISMATCH"),
    ("MYSQL", "SELECT @sensitive", 0, "NATIVE_BIND_STYLE_REQUIRED"),
    ("ORACLE", "SELECT * FROM MES.T WHERE ID=:B1", 1, "NAMED_BIND_MAPPING_REQUIRED"),
    ("ORACLE", "SELECT * FROM MES.T WHERE ID=:1 AND ID=:3", 3, "PARAMETER_LAYOUT_INCOMPLETE"),
    ("POSTGRESQL", "SELECT x FROM", 0, "SOURCE_SQL_NOT_REPLAYABLE"),
    ("POSTGRESQL", "SELECT (" * 70 + "1" + ")" * 70, 0, "SOURCE_SQL_COMPLEXITY_EXCEEDED"),
])
def test_unsafe_unmatched_or_incomplete_sources_are_explicit_blockers(engine, sql, count, code):
    result = inspect_plan_capture(database_type=engine, sql=sql, parameter_count=count)
    assert result.status == "BLOCKED" and result.reason_codes == (code,)
    assert result.execution_authorized is False


def test_native_text_priority_does_not_guess_sql_from_another_source():
    assert source_sql_text({"digest_text": "SELECT ?", "sql_text": "SELECT 123"}, "MYSQL") == "SELECT ?"
    assert source_sql_text({"normalized_statement": "SELECT $1", "sql_text": "SELECT 123"}, "POSTGRESQL") == "SELECT $1"
    assert source_sql_text({"sql_text": "SELECT :1", "normalized_statement": "SELECT 123"}, "ORACLE") == "SELECT :1"
    assert source_sql_text({"question": "SELECT 123"}, "POSTGRESQL") is None


def test_parser_errors_and_fallback_logs_do_not_contain_original_sql_or_parameters(caplog):
    for sql in ("SELECT * FROM '" + SECRET, "EXECUTE " + SECRET, "SELECT " + SECRET + " FROM"):
        result = inspect_plan_capture(database_type="POSTGRESQL", sql=sql, parameter_count=0)
        assert result.status == "BLOCKED" and SECRET not in json.dumps(asdict(result))
    assert SECRET not in caplog.text


@pytest.mark.parametrize("engine,field,sql", [
    ("POSTGRESQL", "normalized_statement", "SELECT * FROM mes.t WHERE id=$1 OR n=$2"),
    ("MYSQL", "digest_text", "SELECT * FROM mes.t WHERE id=? OR n=?"),
    ("ORACLE", "sql_text", "SELECT * FROM MES.T WHERE ID=:1 OR N=:2"),
])
def test_production_material_submission_freezes_preflight_in_independent_encrypted_binding(engine, field, sql):
    store = Store(engine)
    def update(obs):
        index = next(i for i, column in enumerate(obs["columns"]) if column["name"] == field)
        obs["rows"][0][index] = sql
    store.update_observation(update)
    result = asyncio.run(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    assert result.capture_preflight.status == "AWAITING_DATABASE_VALIDATION"
    assert result.execution_authorized is False
    assert store.decrypt()["binding"]["capture_preflight"] == result.capture_preflight.model_dump(mode="json")
    assert store.decrypt()["document"] == document()
    for raw in (SECRET, sql, "private_schema", "private_role"):
        assert raw not in result.model_dump_json()
    assert asyncio.run(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items == (result,)
    revoked = asyncio.run(store.service.revoke(scope=store.scope, run_id=store.run.ops_run_id, material_id=result.material_id))
    assert revoked.capture_preflight is None and revoked.state == "REVOKED"


def test_blocked_preflight_does_not_drop_material_or_invent_parameter_types():
    store = Store()
    result = asyncio.run(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    assert result.capture_preflight.status == "BLOCKED"
    assert result.capture_preflight.reason_codes == ("PARAMETER_COUNT_MISMATCH",)
    assert store.decrypt()["document"] == document()


def test_postgresql_production_driver_passes_only_lexical_binds_to_original_cursor(monkeypatch):
    transaction = SimpleNamespace(start=AsyncMock(), rollback=AsyncMock())
    statement = SimpleNamespace(get_attributes=lambda: (SimpleNamespace(name="value", type=SimpleNamespace(name="int8")),),
        cursor=lambda *args, **kwargs: capture_args(args))
    connection = SimpleNamespace(fetchval=AsyncMock(return_value="16"),
        transaction=Mock(return_value=transaction), prepare=AsyncMock(return_value=statement), close=AsyncMock())
    captured = []
    async def empty_rows():
        for row in ():
            yield row
    def capture_args(args):
        captured.append(args)
        return empty_rows()
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr("aiops_agent.executor.drivers.postgresql.asyncpg.connect", connect)
    monkeypatch.setattr("aiops_agent.executor.identity.observe_postgresql_identity", AsyncMock(return_value=()))
    sql = "SELECT ':not_a_bind', $$:also_not$$, :count::int -- :ignored"
    asyncio.run(PostgreSQLDiagnosticDriver().execute_dynamic(
        **driver_arguments(), sql=sql, parameters={"count": 7}))
    connection.prepare.assert_awaited_once_with("SELECT ':not_a_bind', $$:also_not$$, $1::int -- :ignored")
    assert captured == [(7,)]
    connection.transaction.assert_called_once_with(readonly=True)
    transaction.rollback.assert_awaited_once()
    connection.close.assert_awaited_once()


@pytest.mark.parametrize("sql,parameters", [("SELECT :n", {}), ("SELECT 1", {"private": SECRET}), ("SELECT $1", {})])
def test_driver_bind_mismatch_fails_before_connecting_and_does_not_leak(monkeypatch, sql, parameters):
    connect = AsyncMock()
    monkeypatch.setattr("aiops_agent.executor.drivers.postgresql.asyncpg.connect", connect)
    with pytest.raises(DiagnosticDriverError) as caught:
        asyncio.run(PostgreSQLDiagnosticDriver().execute_dynamic(
            **driver_arguments(), sql=sql, parameters=parameters))
    connect.assert_not_awaited()
    assert str(caught.value) == "QUERY_BIND_INVALID" and SECRET not in str(caught.value)


def test_modified_preflight_cannot_self_attest_with_a_new_receipt_hash():
    store = Store()
    asyncio.run(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    receipt = store.artifacts[-1]
    receipt.payload_json["capture_preflight"]["status"] = "AWAITING_DATABASE_VALIDATION"
    receipt.payload_json["capture_preflight"]["reason_codes"] = []
    receipt.content_hash = canonical_hash(receipt.payload_json)
    result = asyncio.run(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0]
    assert result.state == "UNAVAILABLE" and result.capture_preflight is None
