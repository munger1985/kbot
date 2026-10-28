"""原生目录求值风险核验的生产函数回归；替身不证明目标库已授权或已执行计划。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.domain.evidence.sql_plan_replay import compile_sql_plan_replay, SqlPlanReplayRejected
from aiops_agent.executor.drivers.sql_plan_catalog import (
    PG_RELATIONS, PG_SAFETY, MYSQL_RELATION, MYSQL_COLUMNS, MYSQL_INDEXES, MYSQL_SHADOWS,
    ORACLE_RELATION, ORACLE_COLUMNS, ORACLE_INDEXES, ORACLE_POLICIES, ORACLE_SHADOWS, ORACLE_EXTERNAL,
    validate_postgresql_catalog, validate_mysql_catalog, validate_oracle_catalog,
)
from tests.unit.aiops_agent.test_aiops_sql_plan_replay import material, PRIVATE
from tests.unit.aiops_agent.test_aiops_sql_plan_native_capture import driver, protected, postgresql, mysql, oracle


def replay(engine="POSTGRESQL"):
    return compile_sql_plan_replay(database_type=engine, sql="SELECT sum(id) FROM mes.t" if engine != "ORACLE" else "SELECT SUM(ID) FROM MES.T",
        document=material(), allowed_objects=("mes.t",))


def pg():
    return SimpleNamespace(fetch=AsyncMock(return_value=[("123", "mes", "t", "r", False, False)]),
        fetchrow=AsyncMock(return_value=dict.fromkeys(("unsafe_columns", "unsafe_dependencies", "unsafe_indexes",
            "shadow_functions", "shadow_operators", "unsafe_casts"), False)))


def test_pg_partition_hierarchy_and_dependencies_are_checked_on_the_original_connection():
    connection = pg()
    connection.fetch.return_value.append(("124", "mes", "t_202610", "r", False, False))
    asyncio.run(validate_postgresql_catalog(connection, replay()))
    assert connection.fetch.call_args.args == (PG_RELATIONS, "mes", "t")
    assert connection.fetchrow.call_args.args[0:2] == (PG_SAFETY, [123, 124])
    assert "sum" in connection.fetchrow.call_args.args[2]
    assert PRIVATE not in repr(connection.fetchrow.call_args)


@pytest.mark.parametrize("key,code", [
    ("unsafe_columns", "TYPE_UNSAFE"), ("unsafe_dependencies", "DEPENDENCY_UNSAFE"),
    ("unsafe_indexes", "INDEX_UNSAFE"), ("shadow_functions", "FUNCTION_SHADOWED"),
    ("shadow_operators", "OPERATOR_SHADOWED"), ("unsafe_casts", "CAST_UNSAFE"),
])
def test_pg_any_executable_dependency_shadow_or_custom_type_refuses_estimation(key, code):
    connection = pg()
    connection.fetchrow.return_value[key] = True
    with pytest.raises(SqlPlanReplayRejected, match=code):
        asyncio.run(validate_postgresql_catalog(connection, replay()))


@pytest.mark.parametrize("rows,code", [([], "OBJECT_UNAVAILABLE"),
    ([("123", "mes", "t", "v", False, False)], "OBJECT_UNSAFE"),
    ([("123", "mes", "t", "f", False, False)], "OBJECT_UNSAFE"),
    ([("123", "mes", "t", "r", True, False)], "POLICY_UNSAFE"),
    ([("123", "mes", "t", "r", False, True)], "POLICY_UNSAFE"),
    ([(str(index), "mes", "t", "r", False, False) for index in range(65)], "OBJECT_LIMIT"),
])
def test_pg_empty_views_remote_policies_and_inheritance_overflow_are_not_safe(rows, code):
    connection = pg()
    connection.fetch.return_value = rows
    with pytest.raises(SqlPlanReplayRejected, match=code):
        asyncio.run(validate_postgresql_catalog(connection, replay()))
    connection.fetchrow.assert_not_awaited()


@pytest.mark.parametrize("row", [None, {}, {"unsafe_columns": False},
    dict.fromkeys(("unsafe_columns", "unsafe_dependencies", "unsafe_indexes", "shadow_functions", "shadow_operators", "unsafe_casts"), 0)])
def test_pg_missing_or_untyped_catalog_result_is_not_an_empty_success(row):
    connection = pg()
    connection.fetchrow.return_value = row
    with pytest.raises(SqlPlanReplayRejected, match="INCOMPLETE"):
        asyncio.run(validate_postgresql_catalog(connection, replay()))


class Cursor:
    def __init__(self, engine):
        self.calls = []
        self.sql = None
        self.overrides = {}
        self.defaults = {MYSQL_RELATION: [("BASE TABLE", "InnoDB", "")], MYSQL_COLUMNS: [(0,)], MYSQL_INDEXES: [(0,)]} if engine == "MYSQL" else {
            ORACLE_RELATION: [("N", "NO", "N", None, "TABLE", None)], ORACLE_COLUMNS: [(0,)], ORACLE_INDEXES: [(0,)], ORACLE_POLICIES: [(0,)], ORACLE_EXTERNAL: [(0,)]}

    async def execute(self, sql, parameters=None):
        self.sql = sql
        self.calls.append((sql, parameters))

    async def fetchmany(self, count):
        value = self.overrides.get(self.sql, self.defaults.get(self.sql, []))
        if isinstance(value, BaseException):
            raise value
        return value


def validate(engine, cursor, item=None):
    return validate_mysql_catalog(cursor, item or replay(engine)) if engine == "MYSQL" else validate_oracle_catalog(cursor, item or replay(engine), schema="MES")


@pytest.mark.parametrize("engine", ["MYSQL", "ORACLE"])
def test_native_tables_are_checked_with_bound_exact_schema_and_name(engine):
    cursor = Cursor(engine)
    asyncio.run(validate(engine, cursor))
    assert cursor.calls[0][1] == (("mes", "t") if engine == "MYSQL" else {"owner": "MES", "name": "T"})
    sql, binds = cursor.calls[-1]
    assert sql.startswith("SELECT ROUTINE_NAME" if engine == "MYSQL" else "SELECT OBJECT_NAME")
    assert "sum" not in sql.lower() and ("sum" in binds if engine == "MYSQL" else binds["function_0"] == "SUM")


@pytest.mark.parametrize("engine,query,value,code", [
    ("MYSQL", MYSQL_RELATION, [], "INCOMPLETE"),
    ("MYSQL", MYSQL_RELATION, [("VIEW", None)], "OBJECT_UNSAFE"),
    ("MYSQL", MYSQL_RELATION, [("BASE TABLE", "FEDERATED", "")], "OBJECT_UNSAFE"),
    ("MYSQL", MYSQL_RELATION, [("BASE TABLE", "MyISAM", "")], "OBJECT_UNSAFE"),
    ("MYSQL", MYSQL_RELATION, [("BASE TABLE", "InnoDB", "secondary_engine=RAPID")], "DEPENDENCY_UNSAFE"),
    ("MYSQL", MYSQL_COLUMNS, [(1,)], "TYPE_UNSAFE"),
    ("MYSQL", MYSQL_INDEXES, [(1,)], "INDEX_UNSAFE"),
    ("ORACLE", ORACLE_RELATION, [], "INCOMPLETE"),
    ("ORACLE", ORACLE_RELATION, [("Y", "NO", "N", None, "TABLE", None)], "OBJECT_UNSAFE"),
    ("ORACLE", ORACLE_RELATION, [("N", "YES", "N", None, "TABLE", None)], "OBJECT_UNSAFE"),
    ("ORACLE", ORACLE_RELATION, [("N", "NO", "N", "IOT", "TABLE", None)], "OBJECT_UNSAFE"),
    ("ORACLE", ORACLE_RELATION, [("N", "NO", "N", None, "TABLE", "PRIVATE_CLUSTER")], "OBJECT_UNSAFE"),
    ("ORACLE", ORACLE_COLUMNS, [(1,)], "TYPE_UNSAFE"),
    ("ORACLE", ORACLE_INDEXES, [(1,)], "INDEX_UNSAFE"),
    ("ORACLE", ORACLE_POLICIES, [(1,)], "POLICY_UNSAFE"),
    ("ORACLE", ORACLE_EXTERNAL, [(1,)], "DEPENDENCY_UNSAFE"),
    ("ORACLE", ORACLE_POLICIES, RuntimeError(PRIVATE), "UNAVAILABLE"),
])
def test_hidden_objects_generated_columns_indexes_and_unreadable_policy_metadata_refuse_capture(engine, query, value, code):
    cursor = Cursor(engine)
    cursor.overrides[query] = value
    with pytest.raises(SqlPlanReplayRejected, match=code) as caught:
        asyncio.run(validate(engine, cursor))
    assert PRIVATE not in str(caught.value) and caught.value.__cause__ is None


@pytest.mark.parametrize("engine", ["MYSQL", "ORACLE"])
@pytest.mark.parametrize("value", [[], [(None,)], [(False,)], [("0",)], [(-1,)], [(0,), (0,)]])
def test_absent_or_malformed_risk_count_never_means_no_dependencies(engine, value):
    cursor = Cursor(engine)
    cursor.overrides[MYSQL_COLUMNS if engine == "MYSQL" else ORACLE_COLUMNS] = value
    with pytest.raises(SqlPlanReplayRejected, match="INCOMPLETE"):
        asyncio.run(validate(engine, cursor))


@pytest.mark.parametrize("engine", ["MYSQL", "ORACLE"])
def test_same_name_function_or_synonym_is_not_mistaken_for_a_native_function(engine):
    cursor = Cursor(engine)
    sql = MYSQL_SHADOWS.format(functions="%s") if engine == "MYSQL" else ORACLE_SHADOWS.format(functions=":function_0")
    cursor.overrides[sql] = [("SUM",)]
    with pytest.raises(SqlPlanReplayRejected, match="FUNCTION_SHADOWED"):
        asyncio.run(validate(engine, cursor))


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_catalog_cancellation_propagates_without_success_or_raw_error(engine):
    if engine == "POSTGRESQL":
        connection = pg()
        connection.fetch.side_effect = asyncio.CancelledError()
        operation = validate_postgresql_catalog(connection, replay())
    else:
        cursor = Cursor(engine)
        cursor.overrides[MYSQL_COLUMNS if engine == "MYSQL" else ORACLE_COLUMNS] = asyncio.CancelledError()
        operation = validate(engine, cursor)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(operation)


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_catalog_failure_in_real_driver_never_prepares_or_executes_explain(monkeypatch, engine):
    if engine == "POSTGRESQL":
        connection, _, statement, _ = postgresql(monkeypatch)
        connection.fetch.side_effect = RuntimeError(PRIVATE)
    elif engine == "MYSQL":
        connection, cursor = mysql(monkeypatch)
        original = cursor.execute
        async def execute(sql, parameters=None):
            if sql == MYSQL_RELATION:
                raise RuntimeError(PRIVATE)
            return await original(sql, parameters)
        cursor.execute = execute
    else:
        connection, first, plan = oracle(monkeypatch)
        original = first.execute
        async def execute(sql, parameters=None):
            if sql == ORACLE_POLICIES:
                raise RuntimeError(PRIVATE)
            return await original(sql, parameters)
        first.execute = execute
    with pytest.raises(SqlPlanReplayRejected) as caught:
        asyncio.run(driver().capture(protected(engine)))
    assert PRIVATE not in str(caught.value)
    if engine == "POSTGRESQL":
        connection.prepare.assert_not_awaited()
        statement.fetchval.assert_not_awaited()
    else:
        calls = cursor.executed if engine == "MYSQL" else first.executed
        assert not any(sql.startswith("EXPLAIN") for sql, _ in calls)


def test_pg_explicit_unsafe_search_path_is_rejected_before_any_catalog_joins(monkeypatch):
    connection, _, _, _ = postgresql(monkeypatch)
    connection.fetchval.side_effect = None
    connection.fetchval.return_value = ["mes", "pg_catalog"]
    with pytest.raises(SqlPlanReplayRejected, match="SEARCH_PATH_UNSAFE"):
        asyncio.run(driver().capture(protected()))
    connection.fetch.assert_not_awaited()
    connection.prepare.assert_not_awaited()
