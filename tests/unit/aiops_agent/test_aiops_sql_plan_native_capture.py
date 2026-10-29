"""生产Driver连接/回滚/取消路径的替身回归；不冒充三库真实计划执行验收。"""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.domain.evidence.sql_plan_replay import SqlPlanReplayRejected
from aiops_agent.executor.drivers.postgresql import PostgreSQLDiagnosticDriver
from aiops_agent.executor.readonly_lifecycle import collect_termination
from aiops_agent.executor.sql_plan_native import NativeSqlPlanCaptureDriver
from aiops_agent.executor.drivers.sql_plan_catalog import PG_RELATIONS, PG_SAFETY, MYSQL_RELATION, MYSQL_COLUMNS, MYSQL_INDEXES, ORACLE_RELATION, ORACLE_COLUMNS, ORACLE_INDEXES, ORACLE_POLICIES, ORACLE_EXTERNAL
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureExchange
from tests.unit.aiops_agent.test_aiops_sql_plan_replay import material, exchange, PRIVATE


def call(coro):
    return asyncio.run(coro)


def protected(engine="POSTGRESQL", **updates):
    original = exchange(engine).protected_transport()
    document = material(("varchar2" if engine == "ORACLE" else "text", PRIVATE))
    if engine == "MYSQL":
        document["execution_context"].update(schema_name="app", role_name="reader@%")
    original.update(source_sql={"POSTGRESQL": "SELECT id FROM mes.t WHERE code=$1",
        "MYSQL": "SELECT id FROM mes.t WHERE code=?", "ORACLE": "SELECT ID FROM MES.T WHERE CODE=:1"}[engine],
        protected_document=json.dumps(document))
    return SqlPlanCaptureExchange.model_validate({**original, **updates})


def postgresql(monkeypatch, *, oid=25, close_error=None, plan=None):
    transaction = SimpleNamespace(start=AsyncMock(), rollback=AsyncMock())
    statement = SimpleNamespace(get_parameters=lambda: (SimpleNamespace(oid=oid),),
        fetchval=AsyncMock(return_value=json.dumps(plan or [{"Plan": {"Node Type": "Seq Scan", "Schema": "mes",
            "Relation Name": "t", "Plan Rows": 2, "Filter": PRIVATE}}])))
    connection = SimpleNamespace(transaction=Mock(return_value=transaction),
        fetchrow=AsyncMock(return_value=("app", "mes", "reader", ["mes"])), fetch=AsyncMock(return_value=[]),
        prepare=AsyncMock(return_value=statement), fetchval=AsyncMock(return_value=25),
        close=AsyncMock(side_effect=close_error), terminate=Mock())
    connect = AsyncMock(return_value=connection)
    async def fetch(sql, *args):
        if sql == PG_RELATIONS:
            return [("123", "mes", "t", "r", False, False)]
        return connection.fetch.return_value
    async def fetchrow(sql, *args):
        if sql == PG_SAFETY:
            return dict.fromkeys(("unsafe_columns", "unsafe_dependencies", "unsafe_indexes", "shadow_functions", "shadow_operators", "unsafe_casts"), False)
        return connection.fetchrow.return_value
    connection.fetch.side_effect = fetch
    connection.fetchrow.side_effect = fetchrow
    connection.fetchval.side_effect = lambda sql, *args: ["pg_catalog", "mes"] if "current_schemas" in sql else 25
    monkeypatch.setattr("aiops_agent.executor.sql_plan_native.asyncpg.connect", connect)
    monkeypatch.setattr("aiops_agent.executor.sql_plan_native.observe_postgresql_identity",
        AsyncMock(return_value=(("database_oid", "123"),)))
    return connection, transaction, statement, connect


def driver():
    return NativeSqlPlanCaptureDriver(postgresql_driver=PostgreSQLDiagnosticDriver())


def test_postgresql_explain_is_bound_readonly_and_not_original_execution(monkeypatch):
    connection, transaction, statement, connect = postgresql(monkeypatch)
    item = protected()
    nodes, identity, verification = call(driver().capture(item))
    assert identity == {"database_oid": "123"} and PRIVATE not in json.dumps(nodes)
    assert connection.prepare.call_args.args == (
        "EXPLAIN (FORMAT JSON, VERBOSE TRUE, COSTS TRUE, SETTINGS FALSE) SELECT id FROM mes.t WHERE code=$1",)
    assert statement.fetchval.call_args.args == (PRIVATE,)
    assert connection.transaction.call_args.kwargs == {"readonly": True}
    transaction.start.assert_awaited_once()
    transaction.rollback.assert_awaited_once()
    connection.close.assert_awaited_once()
    assert connect.call_args.kwargs["password"] == PRIVATE


def test_postgresql_native_oid_mismatch_never_requests_explain_rows(monkeypatch):
    connection, _, statement, _ = postgresql(monkeypatch, oid=23)
    with pytest.raises(SqlPlanReplayRejected, match="NATIVE_PARAMETER_TYPE_MISMATCH"):
        call(driver().capture(protected()))
    statement.fetchval.assert_not_awaited()
    connection.close.assert_awaited_once()


def test_postgresql_settings_are_keyed_records_not_positional_dictionaries(monkeypatch):
    connection, _, _, _ = postgresql(monkeypatch)
    connection.fetch.return_value = [{"name": "TimeZone", "setting": "UTC"}]
    original = protected().protected_transport()
    document = json.loads(original["protected_document"])
    document["execution_context"]["settings"] = {"TimeZone": "UTC"}
    original["protected_document"] = json.dumps(document)
    assert call(driver().capture(SqlPlanCaptureExchange.model_validate(original)))[0]
    assert any(args.args[1] == ["TimeZone"] for args in connection.fetch.call_args_list)


@pytest.mark.parametrize("change", ["CONTEXT", "IDENTITY", "SQL_TYPE", "RAW_PLAN"])
def test_postgresql_failures_do_not_reveal_protected_values(monkeypatch, change):
    connection, _, statement, _ = postgresql(monkeypatch)
    item = protected()
    if change == "CONTEXT":
        connection.fetchrow.return_value = ("wrong", "mes", "reader", ["mes"])
    elif change == "IDENTITY":
        item = protected(source_identity={"database_oid": "999"})
    elif change == "SQL_TYPE":
        connection.prepare.side_effect = RuntimeError(PRIVATE)
    else:
        statement.fetchval.return_value = PRIVATE
    with pytest.raises(SqlPlanReplayRejected) as caught:
        call(driver().capture(item))
    assert PRIVATE not in str(caught.value) and caught.value.__cause__ is None
    connection.close.assert_awaited_once()


def test_postgresql_cleanup_failure_prevents_success_and_records_uncertainty(monkeypatch):
    connection, _, _, _ = postgresql(monkeypatch, close_error=RuntimeError(PRIVATE))
    with collect_termination("POSTGRESQL") as collected:
        with pytest.raises(SqlPlanReplayRejected, match="CLEANUP_UNCONFIRMED"):
            call(driver().capture(protected()))
    assert collected["observation"].status == "TERMINATION_UNKNOWN"
    connection.terminate.assert_called_once()


def test_postgresql_cancellation_closes_only_owned_connection_without_becoming_failure_result(monkeypatch):
    connection, _, statement, _ = postgresql(monkeypatch)
    statement.fetchval.side_effect = asyncio.CancelledError()
    cancel = AsyncMock()
    monkeypatch.setattr("aiops_agent.executor.sql_plan_native._cancel_owned_connection", cancel)
    with pytest.raises(asyncio.CancelledError):
        call(driver().capture(protected()))
    cancel.assert_awaited_once_with(connection)
    connection.close.assert_not_awaited()


class MySQLCursor:
    def __init__(self):
        self.executed = []
        self.result = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, sql, parameters=None):
        self.executed.append((sql, parameters))
        if sql == "SELECT DATABASE(), CURRENT_USER()":
            self.result = [("app", "reader@%")]
        elif sql == MYSQL_RELATION:
            self.result = [("BASE TABLE", "InnoDB", "")]
        elif sql in (MYSQL_COLUMNS, MYSQL_INDEXES):
            self.result = [(0,)]
        elif sql.startswith("EXPLAIN FORMAT=JSON "):
            self.result = [(json.dumps({"query_block": {"table": {"table_name": "t", "access_type": "ALL",
                "rows_examined_per_scan": 10, "attached_condition": PRIVATE}}}),)]

    async def fetchmany(self, _):
        return self.result


def mysql(monkeypatch):
    cursor = MySQLCursor()
    connection = SimpleNamespace(cursor=lambda: cursor, thread_id=lambda: 42, closed=False,
        rollback=AsyncMock(), close=Mock())
    monkeypatch.setattr("aiops_agent.executor.sql_plan_native.aiomysql.connect", AsyncMock(return_value=connection))
    monkeypatch.setattr("aiops_agent.executor.sql_plan_native.observe_mysql_connection_identity",
        AsyncMock(return_value=(("server_uuid", "12345678-1234-1234-1234-123456789abc"),)))
    return connection, cursor


def test_mysql_uses_parameter_protocol_rollback_and_owned_connection(monkeypatch):
    connection, cursor = mysql(monkeypatch)
    nodes, identity, verification = call(driver().capture(protected("MYSQL")))
    assert identity["server_uuid"] and PRIVATE not in json.dumps(nodes)
    explained = [(sql, values) for sql, values in cursor.executed if sql.startswith("EXPLAIN")]
    assert explained == [("EXPLAIN FORMAT=JSON SELECT id FROM mes.t WHERE code=%(plan_bind_1)s", {"plan_bind_1": PRIVATE})]
    assert not any(sql.startswith("SELECT id") for sql, _ in cursor.executed)
    connection.rollback.assert_awaited_once()
    connection.close.assert_called_once()


def test_mysql_context_failure_closes_connection_and_does_not_start_explain(monkeypatch):
    connection, cursor = mysql(monkeypatch)
    data = protected("MYSQL").protected_transport()
    document = json.loads(data["protected_document"])
    document["execution_context"]["role_name"] = "another"
    data["protected_document"] = json.dumps(document)
    with pytest.raises(SqlPlanReplayRejected, match="CONTEXT_MISMATCH"):
        call(driver().capture(SqlPlanCaptureExchange.model_validate(data)))
    assert not any(sql.startswith("EXPLAIN") for sql, _ in cursor.executed)
    connection.close.assert_called_once()


class OracleCursor:
    def __init__(self, *, plan=False, available=True):
        self.plan = plan
        self.available = available
        self.executed = []
        self.closed = False
        self.setinputsizes = Mock()

    async def execute(self, sql, parameters=None):
        self.executed.append((sql, parameters))

    async def fetchmany(self, _):
        if self.plan:
            return [(0, None, "SELECT STATEMENT", None, None, None, None, 1, 1),
                (1, 0, "TABLE ACCESS", "FULL", "MES", "T", "TABLE", 1, 1)]
        sql = self.executed[-1][0]
        if sql == ORACLE_RELATION:
            return [("N", "NO", "N", None, "TABLE", None)]
        if sql in (ORACLE_COLUMNS, ORACLE_INDEXES, ORACLE_POLICIES, ORACLE_EXTERNAL):
            return [(0,)]
        if "ALL_TABLES" in sql:
            return [("Y",)] if self.available else []
        return [("app", "mes", "reader")]

    def close(self):
        self.closed = True


def oracle(monkeypatch, *, available=True, close_error=None):
    first, plan = OracleCursor(available=available), OracleCursor(plan=True)
    connection = SimpleNamespace(cursor=Mock(side_effect=[first, plan]),
        rollback=AsyncMock(), close=AsyncMock(side_effect=close_error), cancel=Mock())
    monkeypatch.setattr("aiops_agent.executor.sql_plan_native.oracledb.connect_async", AsyncMock(return_value=connection))
    monkeypatch.setattr("aiops_agent.executor.sql_plan_native.observe_oracle_identity",
        AsyncMock(return_value=(("dbid", "123"), ("container_id", "3"))))
    return connection, first, plan


def test_oracle_explicit_temporary_plan_table_typed_binds_separate_read_cursor_and_rollback(monkeypatch):
    connection, first, plan = oracle(monkeypatch)
    nodes, identity, verification = call(driver().capture(protected("ORACLE")))
    assert identity == {"dbid": "123", "container_id": "3"} and len(nodes) == 2
    sql, binds = first.executed[-1]
    assert sql.startswith("EXPLAIN PLAN SET STATEMENT_ID = 'AIOPS_")
    assert "INTO SYS.PLAN_TABLE$ FOR SELECT ID FROM MES.T WHERE CODE=:plan_bind_1" in sql
    assert binds == {"plan_bind_1": PRIVATE} and PRIVATE not in sql
    assert first.setinputsizes.called and not plan.setinputsizes.called
    assert first.executed[0][0].endswith("FROM SYS.DUAL")
    assert any("FROM SYS.ALL_TABLES WHERE" in sql for sql, _ in first.executed)
    assert plan.executed[0][1]["statement_id"] in sql
    assert first.closed and plan.closed
    assert connection.rollback.await_count == 2
    connection.close.assert_awaited_once()


def test_oracle_will_not_create_plan_table_or_explain_without_verified_temporary_table(monkeypatch):
    connection, first, plan = oracle(monkeypatch, available=False)
    with pytest.raises(SqlPlanReplayRejected, match="TEMP_TABLE_UNAVAILABLE"):
        call(driver().capture(protected("ORACLE")))
    assert not plan.executed and first.closed
    assert not any("EXPLAIN" in sql or "CREATE" in sql or "GRANT" in sql for sql, _ in first.executed)
    connection.close.assert_awaited_once()


@pytest.mark.parametrize("kind,attribute", [("varchar2", "DB_TYPE_VARCHAR"), ("nvarchar2", "DB_TYPE_NVARCHAR"),
    ("char", "DB_TYPE_CHAR"), ("nchar", "DB_TYPE_NCHAR")])
def test_oracle_character_types_are_not_silently_downgraded(monkeypatch, kind, attribute):
    import oracledb
    _, first, _ = oracle(monkeypatch)
    data = protected("ORACLE").protected_transport()
    document = json.loads(data["protected_document"])
    document["parameters"][0]["database_type_name"] = kind
    data["protected_document"] = json.dumps(document)
    call(driver().capture(SqlPlanCaptureExchange.model_validate(data)))
    assert first.setinputsizes.call_args.kwargs == {"plan_bind_1": getattr(oracledb, attribute)}


def test_oracle_cleanup_failure_does_not_become_success(monkeypatch):
    oracle(monkeypatch, close_error=RuntimeError(PRIVATE))
    with collect_termination("ORACLE") as collected:
        with pytest.raises(SqlPlanReplayRejected, match="CLEANUP_UNCONFIRMED"):
            call(driver().capture(protected("ORACLE")))
    assert collected["observation"].status == "TERMINATION_UNKNOWN"


def test_invalid_material_never_opens_native_connection_and_has_fixed_error(monkeypatch):
    connect = AsyncMock()
    monkeypatch.setattr("aiops_agent.executor.sql_plan_native.asyncpg.connect", connect)
    with pytest.raises(SqlPlanReplayRejected, match="PROTECTED_MATERIAL_INVALID"):
        call(driver().capture(protected(protected_document=PRIVATE)))
    connect.assert_not_awaited()
