"""三库同连接物理身份、可选探测失败与取消边界的回归。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.application.diagnosis.provenance import database_provenance
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.executor.drivers import DriverQueryResult, PostgreSQLDiagnosticDriver
from aiops_agent.executor.identity import (
    MYSQL_IDENTITY_SQL, ORACLE_IDENTITY_SQL, POSTGRESQL_IDENTITY_SQL,
    observe_mysql_identity, observe_oracle_identity, observe_postgresql_identity,
)
from aiops_agent.executor.service import DiagnosticExecutorService
from platform_core.contracts.aiops.executor import DiagnosticLimits
from platform_core.identity import uuid7


NOW = datetime(2026, 10, 28, tzinfo=UTC)
MYSQL_UUID = "aa6b4e59-410b-4b92-aa59-b4f07bff7411"


def observer(engine, row=None, error=None):
    cursor = SimpleNamespace(execute=AsyncMock(side_effect=error),
        fetchmany=AsyncMock(return_value=[row] if row is not None else []), close=Mock())
    if engine == "POSTGRESQL":
        connection = SimpleNamespace(fetchrow=AsyncMock(return_value=row, side_effect=error))
        return observe_postgresql_identity(connection), connection
    if engine == "MYSQL":
        return observe_mysql_identity(cursor), cursor
    connection = SimpleNamespace(cursor=Mock(return_value=cursor))
    return observe_oracle_identity(connection), cursor


@pytest.mark.parametrize("engine,row,expected", [
    ("POSTGRESQL", ("7694909560417873958", NOW, "16510"), {"system_identifier": "7694909560417873958", "instance_started_at": NOW.isoformat(), "database_oid": "16510"}),
    ("MYSQL", (MYSQL_UUID,), {"server_uuid": MYSQL_UUID}),
    ("ORACLE", (3017315513, 1, NOW.replace(tzinfo=None), "3"), {"dbid": "3017315513", "instance_number": "1", "instance_started_at": NOW.replace(tzinfo=None).isoformat(), "container_id": "3"}),
])
def test_fixed_identity_queries_return_only_native_identity_fields(engine, row, expected):
    operation, resource = observer(engine, row=row)
    assert dict(asyncio.run(operation)) == expected
    sql = resource.fetchrow.call_args.args[0] if engine == "POSTGRESQL" else resource.execute.call_args.args[0]
    assert sql == {"POSTGRESQL": POSTGRESQL_IDENTITY_SQL, "MYSQL": MYSQL_IDENTITY_SQL, "ORACLE": ORACLE_IDENTITY_SQL}[engine]
    assert "PASSWORD" not in sql.upper()
    if engine == "ORACLE":
        resource.close.assert_called_once()


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_optional_permission_failure_preserves_primary_observation_contract(engine):
    operation, _ = observer(engine, error=PermissionError("denied"))
    assert asyncio.run(operation) == ()


@pytest.mark.parametrize("engine,row", [
    ("POSTGRESQL", ("0", NOW, "16510")),
    ("POSTGRESQL", ("7694909560417873958", "not-time", "16510")),
    ("POSTGRESQL", ("7694909560417873958", NOW, "-1")),
    ("MYSQL", ("00000000-0000-0000-0000-000000000000",)),
    ("MYSQL", ("bad",)), ("MYSQL", (None,)),
    ("ORACLE", (None, 1, NOW, 3)), ("ORACLE", (1, 1, "bad", 3)),
])
def test_unusable_metadata_does_not_become_verified_physical_identity(engine, row):
    operation, _ = observer(engine, row=row)
    assert asyncio.run(operation) == ()


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_identity_probe_never_swallows_outer_cancellation(engine):
    operation, _ = observer(engine, error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(operation)


def test_primary_pg_result_and_metadata_use_same_connection_after_readonly_rollback(monkeypatch):
    async def cursor_rows():
        yield ("demo", 1)
    statement = SimpleNamespace(
        get_attributes=lambda: (SimpleNamespace(name="database_name", type=SimpleNamespace(name="text")),
            SimpleNamespace(name="allocated_mb", type=SimpleNamespace(name="numeric"))),
        cursor=lambda *args, **kwargs: cursor_rows(),
    )
    connection = SimpleNamespace(fetchval=AsyncMock(return_value="16.15"),
        prepare=AsyncMock(return_value=statement), close=AsyncMock(), terminate=Mock())
    transaction = SimpleNamespace(start=AsyncMock(), rollback=AsyncMock())
    connection.transaction = Mock(return_value=transaction)

    async def identity(_sql):
        transaction.rollback.assert_awaited_once()
        return ("7694909560417873958", NOW, "16510")

    connection.fetchrow = AsyncMock(side_effect=identity)
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr("aiops_agent.executor.drivers.postgresql.asyncpg.connect", connect)
    tool = DiagnosticRegistry.load().resolve(tool_id="db.storage.capacity", tool_version="1.0.0",
        db_type="POSTGRESQL", db_version="16", capabilities=set(), entitlements=set())
    raw = asyncio.run(PostgreSQLDiagnosticDriver().execute(
        profile=SimpleNamespace(host="db.internal", port=5432, database="app", tls_enabled=False, tls_profile_ref=None),
        secret=SimpleNamespace(values={"username": "diag", "password": "hidden"}), tool=tool,
        parameters={}, limits=DiagnosticLimits(statement_timeout_seconds=5,
            max_result_rows=10, max_result_bytes=65536), trace_id="same-connection",
    ))
    assert raw.rows == (("demo", 1),)
    assert dict(raw.instance_identity)["system_identifier"] == "7694909560417873958"
    connect.assert_awaited_once()
    connection.close.assert_awaited_once()


def test_executor_freezes_current_identity_and_old_probe_markers_are_not_trusted():
    tool = DiagnosticRegistry.load().resolve(tool_id="db.storage.capacity", tool_version="1.0.0",
        db_type="POSTGRESQL", db_version="16", capabilities=set(), entitlements=set())
    grant = SimpleNamespace(target_id=uuid7(), tool_id=tool.definition.tool_id, tool_version="1.0.0",
        variant=tool.definition.variant, template_sha256=tool.definition.template_sha256,
        db_type="POSTGRESQL", capability_snapshot_hash="a" * 64, parameters_sha256="b" * 64)
    observation = DiagnosticExecutorService._normalize(
        request=SimpleNamespace(executor_request_id=uuid7()), grant=grant, tool=tool,
        raw=DriverQueryResult(columns=tuple(item.name for item in tool.definition.output_columns),
            rows=(), truncated=False, db_version="16.15",
            instance_identity=(("system_identifier", "7694909560417873958"), ("instance_started_at", NOW.isoformat()))),
        captured_at=NOW, duration_ms=2, limits=DiagnosticLimits(statement_timeout_seconds=5,
            max_result_rows=10, max_result_bytes=65536),
    )
    provenance = database_provenance(observation)
    assert ("system_identifier", "7694909560417873958") in provenance.physical_identity
    assert "PHYSICAL_IDENTITY_UNKNOWN" not in provenance.quality_flags
    old = observation.model_copy(update={"provenance": {"instance_identity": {"system_identifier": "old-probe"}}})
    assert not database_provenance(old).physical_identity
    assert "PHYSICAL_IDENTITY_UNKNOWN" in database_provenance(old).quality_flags


def test_mysql_uuid_does_not_claim_boot_generation_or_counter_reset_is_known():
    observation = SimpleNamespace(provenance={"identity_scope": "SAME_CONNECTION_CURRENT",
        "instance_identity": {"server_uuid": MYSQL_UUID}}, target_id=uuid7(), db_version="8.4",
        variant="mysql", template_sha256="a" * 64, result_sha256="b" * 64,
        captured_at=NOW, duration_ms=1, truncated=False)
    provenance = database_provenance(observation)
    assert provenance.physical_identity
    assert "BOOT_IDENTITY_UNKNOWN" in provenance.quality_flags


def test_verified_physical_aliases_do_not_become_independent_confirmation():
    from aiops_agent.domain.measurements import independent_origin_groups
    observations = [SimpleNamespace(provenance={"identity_scope": "SAME_CONNECTION_CURRENT",
        "instance_identity": {"system_identifier": "7694909560417873958", "instance_started_at": NOW.isoformat()}},
        target_id=uuid7(), db_type="POSTGRESQL", db_version="16.15", variant="pg",
        template_sha256="a" * 64, result_sha256="b" * 64, captured_at=NOW, duration_ms=1,
        truncated=False) for _ in range(2)]
    facts = [SimpleNamespace(provenance=database_provenance(item)) for item in observations]
    assert len(independent_origin_groups(facts)) == 1
    observations[1].provenance["instance_identity"]["system_identifier"] = "7694909560417873999"
    facts = [SimpleNamespace(provenance=database_provenance(item)) for item in observations]
    assert len(independent_origin_groups(facts)) == 2
    unknown = database_provenance(SimpleNamespace(provenance={}, target_id=uuid7(),
        db_type="POSTGRESQL", db_version="16", variant="pg", template_sha256="a" * 64,
        result_sha256="b" * 64, captured_at=NOW, duration_ms=1, truncated=False))
    assert len(independent_origin_groups([*facts, SimpleNamespace(provenance=unknown)])) == 1


def test_optional_probe_skips_when_cleanup_reserve_would_consume_primary_budget():
    from aiops_agent.executor.diagnostic_deadline import diagnostic_execution_window, identity_probe_budget
    async def run():
        connection = SimpleNamespace(fetchrow=AsyncMock())
        async with diagnostic_execution_window(asyncio.Semaphore(1), expires_at=datetime.now(UTC) + timedelta(seconds=1)):
            assert identity_probe_budget() == 0
            assert await observe_postgresql_identity(connection) == ()
        connection.fetchrow.assert_not_awaited()
        assert identity_probe_budget() == 1
    asyncio.run(run())
