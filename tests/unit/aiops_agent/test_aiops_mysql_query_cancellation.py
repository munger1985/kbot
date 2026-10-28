"""MySQL 取消保留原连接所有权、同服务器围栏及协议回执的本地驱动回归。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import aiomysql
import pytest

from aiops_agent.executor.drivers.mysql import MySQLDiagnosticDriver
from aiops_agent.executor.drivers.mysql_cancellation import OwnedMySQLExecution
from aiops_agent.executor.drivers import DiagnosticDriverError
from aiops_agent.executor.resource_admission import readonly_credentials
from platform_core.contracts.aiops.executor import DiagnosticLimits
from tests.unit.aiops_agent.test_aiops_execution_fence import executor_fixture
import aiops_agent.executor.execution_fence as fence_module
import aiops_agent.executor.drivers.mysql_cancellation as cancel_module


class Cursor:
    def __init__(self, *, execute=None, rows=()):
        self.execute = execute or AsyncMock()
        self.fetchmany = AsyncMock(return_value=rows)
        self.description = (("value",),)
    async def __aenter__(self):
        return self
    async def __aexit__(self, *_args):
        return None


class Connection:
    def __init__(self, *, session_id=71, cursor=None):
        self.session_id = session_id
        self.closed = False
        self.close_count = 0
        self.current_cursor = cursor or Cursor()
        self.rollback = AsyncMock()
    def thread_id(self):
        return self.session_id
    def cursor(self, *_args):
        return self.current_cursor
    def close(self):
        self.closed = True
        self.close_count += 1
    def get_server_info(self):
        return "8.4"


@pytest.mark.parametrize("mode", [
    "valid", "wrong_server", "missing_identity", "closed", "changed_id",
    "bool_id", "missing_id", "permission", "network", "unacknowledged", "failed_drain",
])
def test_owned_query_only_cancels_same_server_and_pinned_connection_then_waits_for_query_receipt(monkeypatch, mode):
    monkeypatch.setattr(cancel_module, "MYSQL_CANCEL_TIMEOUT_SECONDS", .03)
    async def run():
        owner = OwnedMySQLExecution()
        server = str(uuid4())
        entered, complete_query = asyncio.Event(), asyncio.Event()
        connection = Connection()
        control = Connection()
        killed = []

        async def control_execute(sql):
            if sql.startswith("KILL QUERY"):
                assert not connection.closed
                assert sql == "KILL QUERY 71"
                killed.append(sql)
                if mode == "permission":
                    raise aiomysql.Error(1095, "权限不足")
                if mode == "unacknowledged":
                    await asyncio.Event().wait()
                if mode != "failed_drain":
                    complete_query.set()
        control.current_cursor = Cursor(execute=control_execute,
            rows=((str(uuid4()) if mode == "wrong_server" else server,),))
        open_control = AsyncMock(return_value=control)
        if mode == "network":
            open_control.side_effect = OSError("清理链路不可达")

        async def operation(ownership):
            ownership.attach(connection, () if mode == "missing_identity" else (("server_uuid", server),))
            if mode == "closed": connection.closed = True
            if mode == "changed_id": connection.session_id = 99
            if mode == "bool_id": owner.session_id = True
            if mode == "missing_id": owner.session_id = None
            entered.set()
            try:
                await complete_query.wait()
                return "late-result"
            finally:
                await ownership.finish(connection)

        task = asyncio.create_task(owner.run(operation, open_control_connection=open_control))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel("request-cancelled")
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        assert error.value.args == ("request-cancelled",)
        assert connection.closed
        assert owner.termination_confirmed is (mode == "valid")
        assert bool(killed) is (mode in {"valid", "permission", "unacknowledged", "failed_drain"})
        assert not any(t.get_name() in {"aiops-mysql-owned-query", "aiops-mysql-statement-deadline"} for t in asyncio.all_tasks())
        if open_control.await_count and mode != "network":
            assert control.closed
        if mode in {"closed", "changed_id", "bool_id", "missing_id", "missing_identity"}:
            open_control.assert_not_awaited()
    asyncio.run(run())


def test_statement_timeout_wakes_idle_sampling_and_does_not_issue_next_query():
    async def run():
        owner = OwnedMySQLExecution()
        server = str(uuid4())
        connection = Connection()
        commands = []
        async def control_execute(sql):
            commands.append(sql)
        control = Connection(cursor=Cursor(execute=control_execute, rows=((server,),)))
        async def operation(ownership):
            ownership.attach(connection, (("server_uuid", server),))
            try:
                async with ownership.statement_window(.01):
                    await ownership.pause(20)
                    pytest.fail("到期后不能进入下一条 SQL")
            finally:
                await ownership.finish(connection)
        with pytest.raises(TimeoutError):
            await owner.run(operation, open_control_connection=AsyncMock(return_value=control))
        assert commands == ["SELECT @@server_uuid", "KILL QUERY 71"]
        assert owner.termination_confirmed and connection.closed
    asyncio.run(run())


def test_natural_completion_racing_with_cancel_keeps_connection_pinned_until_kill_ack():
    async def run():
        owner = OwnedMySQLExecution()
        server = str(uuid4())
        connection = Connection()
        entered, finish = asyncio.Event(), asyncio.Event()
        async def control_execute(sql):
            if sql.startswith("KILL"):
                assert owner.query_finished.is_set()
                assert not connection.closed
        control = Connection(cursor=Cursor(execute=control_execute, rows=((server,),)))
        async def open_control():
            finish.set()
            await owner.query_finished.wait()
            return control
        async def operation(ownership):
            ownership.attach(connection, (("server_uuid", server),))
            entered.set()
            try:
                await finish.wait()
                return "completed-at-cancel"
            finally:
                await ownership.finish(connection)
        task = asyncio.create_task(owner.run(operation, open_control_connection=open_control))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owner.termination_confirmed and connection.closed
    asyncio.run(run())


@pytest.mark.parametrize("revoked", [False, True])
def test_formal_mysql_driver_and_executor_fence_cancel_server_query_before_socket_close(monkeypatch, revoked):
    async def run():
        server = str(uuid4())
        entered, stopped = asyncio.Event(), asyncio.Event()
        connection = Connection()
        kill_statements = []
        async def execute(sql, *_args):
            if sql == "SELECT 7 AS value":
                entered.set()
                await stopped.wait()
                raise aiomysql.Error(1317, "本查询已被取消")
        connection.current_cursor = Cursor(execute=execute)
        async def control_execute(sql):
            if sql.startswith("KILL QUERY"):
                assert not connection.closed
                kill_statements.append(sql)
                stopped.set()
        control = Connection(cursor=Cursor(execute=control_execute, rows=((server,),)))
        connect = AsyncMock(side_effect=[connection, control])
        monkeypatch.setattr("aiops_agent.executor.drivers.mysql.aiomysql.connect", connect)
        monkeypatch.setattr("aiops_agent.executor.identity.observe_mysql_connection_identity",
            AsyncMock(return_value=(("server_uuid", server),)))
        f = executor_fixture()
        f.grant.db_type = "MYSQL"
        if revoked:
            from datetime import datetime, timezone
            from platform_core.contracts.aiops.executor import ReadonlyExecutionFenceReceipt
            f.control.check_readonly_execution.side_effect = [f.allowed(),
                ReadonlyExecutionFenceReceipt(allowed=False, checked_at=datetime.now(timezone.utc), reason="EXECUTION_CANCELLED")]
            monkeypatch.setattr(fence_module, "FENCE_CHECK_INTERVAL_SECONDS", .01)
        async def operation():
            async with readonly_credentials(f.control, token="s" * 64, grant=f.grant):
                return await MySQLDiagnosticDriver().execute_dynamic(
                    profile=SimpleNamespace(host="db.invalid", port=3306, database="demo", tls_enabled=False, tls_profile_ref=None),
                    secret=SimpleNamespace(values={"username": "readonly", "password": "not-real"}),
                    sql="SELECT 7 AS value", parameters={},
                    limits=DiagnosticLimits(statement_timeout_seconds=2, max_result_rows=10,
                        max_result_bytes=65536, max_cell_chars=1024), trace_id="mysql-cancel",
                )
        task = asyncio.create_task(operation())
        await asyncio.wait_for(entered.wait(), 1)
        if revoked:
            with pytest.raises(DiagnosticDriverError, match="EXECUTION_CANCELLED"):
                await asyncio.wait_for(task, 2)
        else:
            task.cancel("http-caller-cancel")
            with pytest.raises(asyncio.CancelledError):
                await task
        assert kill_statements == ["KILL QUERY 71"]
        assert connection.closed and control.closed
        f.control.release_resource.assert_not_awaited()
        f.control.report_readonly_termination.assert_awaited_once()
        report = f.control.report_readonly_termination.await_args.args[0]
        assert report.resource_lease == f.receipt
        assert report.observation.db_type == "MYSQL"
        assert report.observation.status == "TERMINATION_CONFIRMED"
        assert report.observation.method == "MYSQL_KILL_OWN_QUERY"
        assert connect.await_count == 2
        assert connect.await_args_list[0].kwargs["user"] == connect.await_args_list[1].kwargs["user"]
    asyncio.run(run())
