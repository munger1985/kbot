"""访问观测缓存不得变成权限门禁、成功证明或跨对象权限推断。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.diagnostics.grants import DiagnosticGrantCodec, DiagnosticGrantError, canonical_sha256
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.diagnostics.dynamic_query import (
    DynamicQueryPolicySnapshot, OracleDynamicQueryPolicy,
    PostgreSQLDynamicQueryPolicy, PostgreSQLDynamicQueryPolicySnapshot,
    MySQLDynamicQueryPolicySnapshot,
)
from aiops_agent.diagnostics.mysql_dynamic_query import MySQLDynamicQueryPolicy
from aiops_agent.executor.access_observations import CACHED_ACCESS_DENIAL, QueryAccessObservations, catalog_object_group
from aiops_agent.executor.drivers import DiagnosticDriverError, DriverQueryResult
from aiops_agent.executor.dynamic_service import DynamicDiagnosticExecutorService
from aiops_agent.executor.service import DiagnosticExecutorService
from aiops_agent.workers.database_handlers import _database_gap_detail
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile, DiagnosticExecutionGrant, DiagnosticLimits,
    DynamicDiagnosticExecutionGrant, DynamicReadDiagnosticRequest,
    ReadDiagnosticRequest,
)
from platform_core.identity import uuid7


NOW = datetime(2026, 10, 28, tzinfo=UTC)


def clock_cache(capacity=1024):
    clock = SimpleNamespace(utc=NOW, monotonic=1.0)
    return QueryAccessObservations(capacity=capacity, clock=lambda: clock.utc,
                                  monotonic=lambda: clock.monotonic), clock


def harness(engine="POSTGRESQL", dynamic=False):
    cache, clock = clock_cache()
    codec = DiagnosticGrantCodec(secret="a" * 32, issuer="test-worker", audience="test-executor")
    registry = DiagnosticRegistry.load()
    now = datetime.now(UTC)
    limits = DiagnosticLimits(statement_timeout_seconds=5, max_result_rows=10, max_result_bytes=65536)
    common = dict(issuer="test-worker", audience="test-executor", grant_id=uuid7(), issued_at=now,
        expires_at=now + timedelta(seconds=30), run_id=uuid7(), task_id=uuid7(),
        lease_token_hash="a" * 64, target_id=uuid7(),
        domain_id=101 if DiagnosticExecutionGrant.model_fields["domain_id"].annotation is int else uuid7(),
        target_row_version=1, db_type=engine,
        connection_profile=DiagnosticConnectionProfile(host="db.internal", port=5432,
            database="app", service="app", tls_enabled=False),
        diagnostic_credential_id=uuid7(), capability_snapshot_hash="b" * 64,
        parameters_sha256=canonical_sha256({}), limits=limits, trace_id="access-test")
    if "tenant_id" in DiagnosticExecutionGrant.model_fields:
        common["tenant_id"] = uuid7()
    if dynamic:
        policy, snapshot, sql = {
            "POSTGRESQL": (PostgreSQLDynamicQueryPolicy, PostgreSQLDynamicQueryPolicySnapshot(),
                           "SELECT pid FROM pg_catalog.pg_stat_activity"),
            "MYSQL": (MySQLDynamicQueryPolicy, MySQLDynamicQueryPolicySnapshot(),
                      "SELECT thread_id FROM performance_schema.threads"),
            "ORACLE": (OracleDynamicQueryPolicy, DynamicQueryPolicySnapshot(),
                       "SELECT sid FROM v$session"),
        }[engine]
        validated = policy(snapshot).validate(sql, {})
        grant = DynamicDiagnosticExecutionGrant(**common,
            tool_id={"POSTGRESQL": "db.postgresql.readonly_query", "MYSQL": "db.mysql.readonly_query",
                     "ORACLE": "db.oracle.readonly_query"}[engine],
            variant={"POSTGRESQL": "postgresql-dynamic-readonly-v1", "MYSQL": "mysql-dynamic-readonly-v1",
                     "ORACLE": "oracle-dynamic-readonly-v1"}[engine],
            query_sha256=validated.query_sha256, policy_sha256=validated.policy_sha256,
            policy_snapshot=snapshot.model_dump(mode="json"),
            projected_columns=getattr(validated, "projected_columns", ("thread_id" if engine == "MYSQL" else "pid",)))
        result = DriverQueryResult(columns=grant.projected_columns, rows=((1,),), truncated=False, db_version="16")
    else:
        tool = registry.resolve(tool_id="db.instance.identity", tool_version="1.0.0", db_type=engine,
            db_version={"POSTGRESQL": "16", "MYSQL": "8", "ORACLE": "19"}[engine], capabilities=set(), entitlements=set())
        grant = DiagnosticExecutionGrant(**common, tool_id=tool.definition.tool_id,
            tool_version=tool.definition.version, variant=tool.definition.variant,
            template_sha256=tool.definition.template_sha256)
        result = DriverQueryResult(columns=tuple(item.name for item in tool.definition.output_columns),
                                  rows=(), truncated=False, db_version="16")
    driver = SimpleNamespace(db_type=engine, execute=AsyncMock(return_value=result), execute_dynamic=AsyncMock(return_value=result))
    control = SimpleNamespace(issue_credential=AsyncMock(return_value=SimpleNamespace(username="private-user", password="private-password")))
    kwargs = dict(grant_codec=codec, control_plane=control, hard_limits=limits,
                  concurrency=1, access_observations=cache)
    if dynamic:
        drivers = tuple(driver if key == engine else SimpleNamespace(db_type=key) for key in ("POSTGRESQL", "MYSQL", "ORACLE"))
        service = DynamicDiagnosticExecutorService(**kwargs, drivers=drivers)
    else:
        service = DiagnosticExecutorService(**kwargs, drivers=(driver,), registry=registry)

    def request(current=None):
        current = current or grant
        fields = dict(executor_request_id=uuid7(), parameters={}, idempotency_key=str(uuid7()))
        if dynamic:
            return DynamicReadDiagnosticRequest(**fields, grant=codec.issue_dynamic(current), sql=validated.normalized_sql)
        return ReadDiagnosticRequest(**fields, grant=codec.issue(current))

    return SimpleNamespace(cache=cache, clock=clock, grant=grant, driver=driver, control=control,
        service=service, request=request, method=driver.execute_dynamic if dynamic else driver.execute)


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_repeated_denial_uses_short_cache_but_always_redeems_current_authorization(engine, dynamic):
    h = harness(engine, dynamic)
    h.method.side_effect = DiagnosticDriverError("PRIVILEGE_MISSING")
    async def run():
        assert (await h.service.execute(h.request())).error_code == "PRIVILEGE_MISSING"
        assert (await h.service.execute(h.request())).error_code == CACHED_ACCESS_DENIAL
        assert h.method.await_count == 1
        assert h.control.issue_credential.await_count == 2
        h.clock.monotonic += 31
        h.clock.utc += timedelta(seconds=31)
        assert (await h.service.execute(h.request())).error_code == "PRIVILEGE_MISSING"
        assert h.method.await_count == 2
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_cached_success_is_never_reused_as_current_access_or_query_evidence(engine, dynamic):
    h = harness(engine, dynamic)
    async def run():
        first = await h.service.execute(h.request())
        assert first.status == "SUCCEEDED"
        metadata = first.observation.provenance["access_observation"]
        assert metadata["scope"] == "EXACT_QUERY_AND_PARAMETERS"
        assert metadata["grants_authority"] is False
        assert metadata["state"] == "SUCCEEDED"
        assert "private-password" not in first.model_dump_json()
        assert "private-user" not in first.model_dump_json()
        h.method.side_effect = DiagnosticDriverError("PRIVILEGE_MISSING")
        assert (await h.service.execute(h.request())).error_code == "PRIVILEGE_MISSING"
        assert h.method.await_count == 2
    asyncio.run(run())


@pytest.mark.parametrize("code", ["TIMEOUT", "QUERY_OBJECT_UNAVAILABLE", "AUTH_FAILED", "EXECUTOR_INTERNAL_ERROR"])
def test_non_permission_failures_are_not_negative_permission_cache(code):
    h = harness()
    h.method.side_effect = DiagnosticDriverError(code)
    async def run():
        for _ in range(2):
            assert (await h.service.execute(h.request())).error_code == code
        assert h.method.await_count == 2
    asyncio.run(run())


def test_revoked_current_grant_is_checked_before_cached_denial():
    h = harness()
    h.method.side_effect = DiagnosticDriverError("PRIVILEGE_MISSING")
    async def run():
        await h.service.execute(h.request())
        h.control.issue_credential.side_effect = DiagnosticGrantError("GRANT_REVOKED", "已撤权")
        with pytest.raises(DiagnosticGrantError):
            await h.service.execute(h.request())
    asyncio.run(run())


@pytest.mark.parametrize("field,value", [("target_row_version", 2), ("target_id", "another-target"),
    ("domain_id", "another-domain"), ("tenant_id", "another-tenant"),
    ("diagnostic_credential_id", "another-credential"), ("capability_snapshot_hash", "c" * 64),
    ("parameters_sha256", "d" * 64), ("db_type", "MYSQL")])
def test_target_credential_policy_parameter_and_tenant_changes_invalidate_observation(field, value):
    h = harness()
    if field not in type(h.grant).model_fields:
        return  # 平台合同没有该作用域字段时，不虚构一个租户边界。
    if field == "domain_id":
        value = 102 if type(h.grant).model_fields[field].annotation is int else uuid7()
    if field == "tenant_id":
        value = uuid7()
    issued = h.control.issue_credential.return_value
    key = h.cache.key(h.grant, issued=issued, query_sha256="e" * 64)
    changed = h.grant.model_copy(update={field: value})
    assert key != h.cache.key(changed, issued=issued, query_sha256="e" * 64)


def test_credential_content_endpoint_query_and_policy_changes_cannot_hit_old_entry():
    h = harness()
    issued = h.control.issue_credential.return_value
    original = h.cache.key(h.grant, issued=issued, query_sha256="e" * 64)
    assert original != h.cache.key(h.grant, issued=SimpleNamespace(username=issued.username, password="rotated"), query_sha256="e" * 64)
    assert original != h.cache.key(h.grant, issued=issued, query_sha256="f" * 64)
    grant = h.grant.model_copy(update={"connection_profile": h.grant.connection_profile.model_copy(update={"database": "other"})})
    assert original != h.cache.key(grant, issued=issued, query_sha256="e" * 64)
    other_process_cache, _ = clock_cache()
    assert original != other_process_cache.key(h.grant, issued=issued, query_sha256="e" * 64)


def test_dynamic_policy_change_invalidates_same_query_denial():
    h = harness(dynamic=True)
    issued = h.control.issue_credential.return_value
    original = h.cache.key(h.grant, issued=issued, query_sha256=h.grant.query_sha256)
    changed = h.grant.model_copy(update={"policy_sha256": "f" * 64})
    assert original != h.cache.key(changed, issued=issued, query_sha256=h.grant.query_sha256)


def test_cache_is_bounded_and_wall_clock_rollback_or_expiry_is_unknown():
    cache, clock = clock_cache(capacity=2)
    for key in ("one", "two", "three"):
        cache.remember(key, state="DENIED", objects=("pg_catalog.pg_stat_activity",))
    assert cache.get("one") is None
    clock.utc -= timedelta(seconds=1)
    assert cache.get("two") is None
    clock.utc += timedelta(seconds=301)
    assert cache.get("three") is None
    with pytest.raises(ValueError):
        cache.remember("invalid", state="TIMEOUT")


def test_union_query_denial_does_not_claim_each_object_is_denied():
    objects = catalog_object_group("WITH q AS (SELECT * FROM pg_catalog.pg_stat_activity) SELECT * FROM q JOIN pg_catalog.pg_database d ON TRUE", "POSTGRESQL")
    assert not any(item == "q" for item in objects)
    cache, _ = clock_cache()
    metadata = cache.remember("combined-query", state="DENIED", objects=objects).public_metadata()
    assert len(metadata["object_group"]) == 2
    assert metadata["scope"] == "EXACT_QUERY_AND_PARAMETERS"
    assert cache.get("individual-object") is None
    assert "本次未重复查库" in _database_gap_detail(CACHED_ACCESS_DENIAL)


def test_cancellation_is_neither_cached_nor_converted_to_a_gap():
    h = harness()
    h.method.side_effect = asyncio.CancelledError()
    async def run():
        with pytest.raises(asyncio.CancelledError):
            await h.service.execute(h.request())
        assert not h.cache._entries
    asyncio.run(run())
