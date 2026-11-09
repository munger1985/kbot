"""Oracle 能力来自真实只读查询，不从连接成功或异常文本猜测权限。"""

import asyncio
from types import SimpleNamespace

import oracledb
import pytest

from aiops_agent.adapters.oracle_capabilities import probe_oracle_read_capabilities


class Cursor:
    def __init__(self, denied=(), failure=None):
        self.denied = set(denied)
        self.failure = failure
        self.sql = []

    async def execute(self, sql):
        self.sql.append(sql)
        if self.failure:
            raise self.failure
        if any("FROM " + view + " " in sql for view in self.denied):
            raise oracledb.DatabaseError(SimpleNamespace(code=942))

    async def fetchall(self):
        return []


def test_successful_zero_rows_discovers_only_verified_read_capabilities():
    cursor = Cursor()
    capabilities, privileges, details = asyncio.run(probe_oracle_read_capabilities(cursor))
    assert capabilities == ("dba_catalog_views", "dynamic_performance_views")
    assert {"V_$DATABASE", "V_$INSTANCE", "V_$SQLSTATS", "GV_$SQL"} <= set(privileges)
    assert all(status == "AVAILABLE" for status in details["read_access"].values())
    assert all(sql.startswith("SELECT 1 FROM ") and sql.endswith(" WHERE 1=0") for sql in cursor.sql)
    assert len(cursor.sql) == 10
    assert not any(name in capabilities for name in ("session_management", "awr", "ash"))


@pytest.mark.parametrize("view,capability", [
    ("V$INSTANCE", "dynamic_performance_views"), ("V$SESSION", "dynamic_performance_views"),
    ("DBA_TABLES", "dba_catalog_views"), ("DBA_INDEXES", "dba_catalog_views"),
])
def test_one_denied_core_view_disables_only_its_capability(view, capability):
    capabilities, privileges, details = asyncio.run(probe_oracle_read_capabilities(Cursor(denied=[view])))
    assert capability not in capabilities
    assert len(capabilities) == 1
    assert view.replace("$", "_$") not in privileges
    assert details["read_access"][view] == "UNAVAILABLE"


def test_optional_sqlstats_denial_is_not_a_blanket_database_failure():
    capabilities, privileges, details = asyncio.run(probe_oracle_read_capabilities(Cursor(denied=["V$SQLSTATS"])))
    assert "dynamic_performance_views" in capabilities
    assert "V_$SQLSTATS" not in privileges
    assert details["read_access"]["V$SQLSTATS"] == "UNAVAILABLE"


def test_permission_withdrawal_does_not_reuse_success_from_previous_probe():
    asyncio.run(probe_oracle_read_capabilities(Cursor()))
    capabilities, _, _ = asyncio.run(probe_oracle_read_capabilities(Cursor(denied=["V$INSTANCE"])))
    assert "dynamic_performance_views" not in capabilities


@pytest.mark.parametrize("code", [1013, 3113, 12170])
def test_transient_database_error_is_not_published_as_empty_permissions(code):
    with pytest.raises(oracledb.DatabaseError):
        asyncio.run(probe_oracle_read_capabilities(Cursor(
            failure=oracledb.DatabaseError(SimpleNamespace(code=code)))))


def test_insufficient_privilege_is_a_verified_denial():
    capabilities, privileges, _ = asyncio.run(probe_oracle_read_capabilities(Cursor(
        failure=oracledb.DatabaseError(SimpleNamespace(code=1031)))))
    assert capabilities == privileges == ()


def test_cancellation_is_not_swallowed():
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(probe_oracle_read_capabilities(Cursor(failure=asyncio.CancelledError())))

@pytest.mark.parametrize("version", ["19c", "26ai"])
def test_probe_results_admit_oracle_collection_without_manual_capability_flags(version):
    from aiops_agent.diagnostics import DiagnosticRegistry
    from aiops_agent.scheduling.workload_collection import WorkloadCollectionScheduler

    capabilities, privileges, _ = asyncio.run(probe_oracle_read_capabilities(Cursor()))
    scheduler = WorkloadCollectionScheduler(uow_factory=None,
        diagnostic_registry=DiagnosticRegistry.load(), scheduler_id="test", actor_id="test", interval_seconds=1)
    tools, unavailable = scheduler._freeze_tools(target=SimpleNamespace(
        db_type="ORACLE", version_code=version,
        capabilities_json={"capabilities": list(capabilities), "privileges": list(privileges)}),
        kind="WORKLOAD", policy={})
    assert not unavailable
    assert {tool["tool_id"] for tool in tools} == {
        "db.instance.identity", "db.oracle.workload.identity", "db.oracle.workload.statements"}
