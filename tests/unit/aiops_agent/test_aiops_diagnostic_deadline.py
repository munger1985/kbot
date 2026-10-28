"""排队不越权、凭据与执行共享期限、取消和迟到隔离的本地回归。"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.executor.diagnostic_deadline import diagnostic_execution_window
from aiops_agent.executor.drivers import DiagnosticDriverError
from platform_core.contracts.aiops.executor import DiagnosticLimits


def deadline(seconds):
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def test_queued_expiry_never_enters_credential_or_driver_and_leaks_no_permit():
    async def run():
        semaphore = asyncio.Semaphore(0)
        credential, driver = AsyncMock(), AsyncMock()
        with pytest.raises(DiagnosticDriverError, match="EXECUTION_DEADLINE_EXCEEDED"):
            async with diagnostic_execution_window(semaphore, expires_at=deadline(0.02)):
                await credential()
                await driver()
        credential.assert_not_awaited()
        driver.assert_not_awaited()
        semaphore.release()
        async with diagnostic_execution_window(semaphore, expires_at=deadline(1)):
            assert semaphore.locked()
        assert semaphore._value == 1
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["credential", "query"])
def test_absolute_deadline_cancels_each_phase_and_releases_permit(phase):
    async def run():
        semaphore = asyncio.Semaphore(1)
        finished = []
        async def blocking():
            try:
                await asyncio.Event().wait()
            finally:
                finished.append(phase)
        with pytest.raises(DiagnosticDriverError, match="EXECUTION_DEADLINE_EXCEEDED"):
            async with diagnostic_execution_window(semaphore, expires_at=deadline(0.02)):
                if phase == "query":
                    await AsyncMock()()
                await blocking()
        assert finished == [phase]
        assert semaphore._value == 1
    asyncio.run(run())


def test_caller_cancellation_propagates_instead_of_becoming_success_or_internal_error():
    async def run():
        semaphore = asyncio.Semaphore(1)
        entered = asyncio.Event()
        async def query():
            async with diagnostic_execution_window(semaphore, expires_at=deadline(1)):
                entered.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(query())
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert semaphore._value == 1
    asyncio.run(run())


def test_driver_swallowing_cancellation_cannot_return_late_success():
    async def run():
        with pytest.raises(DiagnosticDriverError, match="EXECUTION_DEADLINE_EXCEEDED"):
            async with diagnostic_execution_window(asyncio.Semaphore(1), expires_at=deadline(0.02)):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
    asyncio.run(run())


def test_expired_grant_never_acquires_capacity():
    async def run():
        semaphore = asyncio.Semaphore(1)
        with pytest.raises(DiagnosticDriverError, match="EXECUTION_DEADLINE_EXCEEDED"):
            async with diagnostic_execution_window(semaphore, expires_at=deadline(-1)):
                pytest.fail("过期 Grant 不得执行")
        assert semaphore._value == 1
    asyncio.run(run())


def test_postgresql_driver_cancellation_terminates_connection(monkeypatch):
    from aiops_agent.executor.drivers.postgresql import PostgreSQLDiagnosticDriver
    async def run():
        entered = asyncio.Event()
        async def prepare(_sql):
            entered.set()
            await asyncio.Event().wait()
        class Transaction:
            async def start(self): pass
            async def rollback(self):
                pytest.fail("取消查询不得等待事务回滚")
        connection = SimpleNamespace(
            fetchval=AsyncMock(return_value="16.15"), prepare=prepare,
            transaction=lambda **kwargs: Transaction(), terminate=Mock(), close=AsyncMock(),
        )
        monkeypatch.setattr("aiops_agent.executor.drivers.postgresql.asyncpg.connect", AsyncMock(return_value=connection))
        task = asyncio.create_task(PostgreSQLDiagnosticDriver().execute_dynamic(
            profile=SimpleNamespace(host="db.internal", port=5432, database="app",
                tls_profile_ref=None, tls_enabled=False),
            secret=SimpleNamespace(values={"username": "diag", "password": "hidden"}),
            sql="SELECT datname FROM pg_stat_database", parameters={},
            limits=DiagnosticLimits(statement_timeout_seconds=1, max_result_rows=200,
                max_result_bytes=131072), trace_id="test-cancel",
        ))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        connection.terminate.assert_not_called()
        connection.close.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("cancelled", [True, False])
def test_oracle_cancel_or_query_error_skips_rollback_and_closes(monkeypatch, cancelled):
    from aiops_agent.executor.drivers.oracle import OracleDiagnosticDriver
    async def run():
        entered = asyncio.Event()
        calls = 0
        async def execute(_sql, _parameters=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                if not cancelled:
                    raise TimeoutError
                await asyncio.Event().wait()
        cursor = SimpleNamespace(execute=execute, close=Mock())
        connection = SimpleNamespace(
            cursor=lambda: cursor, cancel=Mock(), rollback=AsyncMock(), close=AsyncMock(),
        )
        monkeypatch.setattr("aiops_agent.executor.drivers.oracle.oracledb.connect_async", AsyncMock(return_value=connection))
        task = asyncio.create_task(OracleDiagnosticDriver().execute_dynamic(
            profile=SimpleNamespace(host="db.internal", port=1521, service="app",
                tls_profile_ref=None, tls_enabled=False),
            secret=SimpleNamespace(values={"username": "diag", "password": "hidden"}),
            sql="SELECT sid FROM v$session", parameters={},
            limits=DiagnosticLimits(statement_timeout_seconds=1, max_result_rows=200,
                max_result_bytes=131072), trace_id="test-cancel",
        ))
        await asyncio.wait_for(entered.wait(), timeout=2)
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            connection.cancel.assert_called_once()
        else:
            with pytest.raises(DiagnosticDriverError, match="QUERY_TIMEOUT"):
                await task
            connection.cancel.assert_called_once()
        cursor.close.assert_called_once()
        connection.rollback.assert_not_awaited()
        connection.close.assert_awaited_once()
    asyncio.run(run())
