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
from aiops_agent.executor.access_observations import (
    CACHED_ACCESS_DENIAL, OBJECT_CACHED_ACCESS_DENIAL, QueryAccessObservations,
    catalog_object_group, object_read_scope,
)
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


def harness(engine="POSTGRESQL", dynamic=False, *, readonly_semaphore=None):
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
                  concurrency=1, access_observations=cache, readonly_semaphore=readonly_semaphore)
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

    def query_request(sql, **updates):
        current = policy(snapshot).validate(sql, {})
        updated = grant.model_copy(update={
            "query_sha256": current.query_sha256, "policy_sha256": current.policy_sha256,
            "projected_columns": getattr(current, "projected_columns", ("*",)), **updates,
        })
        return DynamicReadDiagnosticRequest(executor_request_id=uuid7(), parameters={},
            idempotency_key=str(uuid7()), grant=codec.issue_dynamic(updated), sql=current.normalized_sql)

    return SimpleNamespace(cache=cache, clock=clock, grant=grant, driver=driver, control=control,
        service=service, request=request, query_request=query_request,
        codec=codec, snapshot=snapshot if dynamic else None, policy=policy if dynamic else None,
        method=driver.execute_dynamic if dynamic else driver.execute)


@pytest.mark.parametrize("first_dynamic", [False, True])
def test_fixed_and_dynamic_queries_share_capacity_before_credential_redemption(first_dynamic):
    async def run():
        shared = asyncio.Semaphore(1)
        first = harness(dynamic=first_dynamic, readonly_semaphore=shared)
        second = harness(dynamic=not first_dynamic, readonly_semaphore=shared)
        entered, release = asyncio.Event(), asyncio.Event()
        original = first.method.return_value

        async def blocked(**kwargs):
            entered.set()
            await release.wait()
            return original

        first.method.side_effect = blocked
        running = asyncio.create_task(first.service.execute(first.request()))
        await asyncio.wait_for(entered.wait(), 1)
        waiting = asyncio.create_task(second.service.execute(second.request()))
        await asyncio.sleep(0)
        second.control.issue_credential.assert_not_awaited()
        second.method.assert_not_awaited()
        release.set()
        one, two = await asyncio.wait_for(asyncio.gather(running, waiting), 1)
        assert one.status == two.status == "SUCCEEDED"
        assert shared._value == 1
        assert first.service._semaphore is second.service._semaphore
    asyncio.run(run())


@pytest.mark.parametrize("waiting_dynamic", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_shared_capacity_queue_cancellation_or_deadline_never_takes_credentials(waiting_dynamic, cancel):
    async def run():
        shared = asyncio.Semaphore(1)
        first = harness(dynamic=not waiting_dynamic, readonly_semaphore=shared)
        second = harness(dynamic=waiting_dynamic, readonly_semaphore=shared)
        entered, release = asyncio.Event(), asyncio.Event()
        original = first.method.return_value

        async def blocked(**kwargs):
            entered.set()
            await release.wait()
            return original

        first.method.side_effect = blocked
        running = asyncio.create_task(first.service.execute(first.request()))
        await asyncio.wait_for(entered.wait(), 1)
        grant = second.grant.model_copy(update={"expires_at": datetime.now(UTC) + timedelta(seconds=1 if cancel else 0.03)})
        waiting = asyncio.create_task(second.service.execute(second.request(grant)))
        if cancel:
            await asyncio.sleep(0)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        else:
            result = await asyncio.wait_for(waiting, 1)
            assert result.error_code == "EXECUTION_DEADLINE_EXCEEDED"
        second.control.issue_credential.assert_not_awaited()
        second.method.assert_not_awaited()
        release.set()
        assert (await asyncio.wait_for(running, 1)).status == "SUCCEEDED"
        assert shared._value == 1
    asyncio.run(run())


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


OBJECTS = [
    ("POSTGRESQL", "pg_catalog.pg_stat_activity", "pid", "datid"),
    ("MYSQL", "performance_schema.threads", "thread_id", "processlist_id"),
    ("ORACLE", "v$session", "sid", "username"),
]


@pytest.mark.parametrize("engine,table,column,other", OBJECTS)
def test_minimal_object_probe_denial_is_reused_for_different_sql_requiring_all_columns(engine, table, column, other):
    h = harness(engine, dynamic=True)
    h.method.side_effect = DiagnosticDriverError("PRIVILEGE_MISSING")
    async def run():
        first = await h.service.execute(h.query_request(f"SELECT {column} FROM {table} WHERE 1=0"))
        assert first.error_code == "PRIVILEGE_MISSING"
        observed, = first.object_access
        assert observed.state == "DENIED" and not observed.reused
        second = await h.service.execute(h.query_request(f"SELECT {other}, {column} FROM {table} WHERE {column} > 0"))
        assert second.error_code == OBJECT_CACHED_ACCESS_DENIAL
        reused, = second.object_access
        assert reused.reused and reused.checked_at == observed.checked_at
        assert reused.source_executor_request_id == observed.source_executor_request_id
        assert h.method.await_count == 1 and h.control.issue_credential.await_count == 2
        # 不相交的列不能受拒绝影响；SELECT * 与函数也不冒充完整的显式列集合。
        for sql in (f"SELECT {other} FROM {table}", f"SELECT * FROM {table}", f"SELECT COUNT(*) AS n FROM {table}"):
            result = await h.service.execute(h.query_request(sql))
            assert result.error_code == "PRIVILEGE_MISSING"
            assert not result.object_access
        assert h.method.await_count == 4
        h.clock.utc += timedelta(seconds=31)
        h.clock.monotonic += 31
        third = await h.service.execute(h.query_request(f"SELECT {column} FROM {table} WHERE {column} > 0"))
        assert third.error_code == "PRIVILEGE_MISSING" and not third.object_access
    asyncio.run(run())


@pytest.mark.parametrize("engine,table,column,other", OBJECTS)
def test_multicolumn_denial_does_not_guess_which_column_is_denied(engine, table, column, other):
    h = harness(engine, dynamic=True)
    h.method.side_effect = DiagnosticDriverError("PRIVILEGE_MISSING")
    async def run():
        await h.service.execute(h.query_request(f"SELECT {column}, {other} FROM {table} WHERE 0=1"))
        single = await h.service.execute(h.query_request(f"SELECT {column} FROM {table}"))
        assert single.error_code == "PRIVILEGE_MISSING" and not single.object_access
        full = await h.service.execute(h.query_request(f"SELECT {other}, {column} FROM {table}"))
        assert full.error_code == OBJECT_CACHED_ACCESS_DENIAL
        assert h.method.await_count == 2
    asyncio.run(run())


@pytest.mark.parametrize("engine,table,column,other", OBJECTS)
def test_probe_success_is_reusable_hint_not_fresh_data_or_row_visibility(engine, table, column, other):
    h = harness(engine, dynamic=True)
    h.method.return_value = DriverQueryResult(columns=(column,), rows=(), truncated=False, db_version="16")
    async def run():
        first = await h.service.execute(h.query_request(f"SELECT {column} FROM {table} WHERE 1=0"))
        assert first.status == "SUCCEEDED"
        probe, = first.observation.object_access
        assert probe.state == "READABLE" and not probe.grants_authority and not probe.reused
        h.method.side_effect = DiagnosticDriverError("PRIVILEGE_MISSING")
        second = await h.service.execute(h.query_request(f"SELECT {column} FROM {table} WHERE {column} > 0"))
        assert second.error_code == "PRIVILEGE_MISSING"
        prior, = second.object_access
        assert prior.state == "READABLE" and prior.reused
        assert prior.checked_at == probe.checked_at
        assert h.method.await_count == 2
        assert "private-user" not in first.model_dump_json()
        assert "private-password" not in second.model_dump_json()
    asyncio.run(run())


@pytest.mark.parametrize("code", ["QUERY_OBJECT_UNAVAILABLE", "QUERY_COLUMN_INVALID", "AUTH_FAILED", "TIMEOUT"])
@pytest.mark.parametrize("engine,table,column,other", OBJECTS)
def test_probe_non_permission_error_does_not_create_crossquery_denial(code, engine, table, column, other):
    h = harness(engine, dynamic=True)
    h.method.side_effect = DiagnosticDriverError(code)
    async def run():
        first = await h.service.execute(h.query_request(f"SELECT {column} FROM {table} WHERE 1=0"))
        if code == "QUERY_OBJECT_UNAVAILABLE":
            assert first.object_access[0].state == "OBJECT_UNAVAILABLE"
        else:
            assert not first.object_access
        second = await h.service.execute(h.query_request(f"SELECT {column} FROM {table}"))
        assert second.error_code == code and not second.object_access
        assert h.method.await_count == 2
    asyncio.run(run())


@pytest.mark.parametrize("sql", [
    "SELECT * FROM pg_catalog.pg_stat_activity WHERE 1=0",
    "SELECT count(pid) FROM pg_catalog.pg_stat_activity WHERE 1=0",
    "SELECT a.pid FROM pg_catalog.pg_stat_activity a JOIN pg_catalog.pg_database d ON TRUE WHERE 1=0",
    "WITH a AS (SELECT pid FROM pg_catalog.pg_stat_activity) SELECT pid FROM a WHERE 1=0",
    "SELECT a.pid FROM (SELECT pid FROM pg_catalog.pg_stat_activity) a WHERE 1=0",
    "SELECT pid FROM pg_catalog.pg_stat_activity UNION SELECT pid FROM pg_catalog.pg_stat_activity",
    "SELECT pid FROM pg_catalog.pg_stat_activity; SELECT datid FROM pg_catalog.pg_stat_activity",
    "SELECT other.pid FROM pg_catalog.pg_stat_activity WHERE 1=0",
])
def test_complex_or_ambiguous_queries_cannot_claim_single_object_probe(sql):
    assert object_read_scope(sql, "POSTGRESQL") is None


def test_probe_parser_retains_identifier_case_and_alias_semantics():
    a = object_read_scope("SELECT a.PID AS id FROM PG_CATALOG.PG_STAT_ACTIVITY a WHERE 1=0", "POSTGRESQL")
    b = object_read_scope("SELECT pid FROM pg_catalog.pg_stat_activity WHERE FALSE", "POSTGRESQL")
    assert a.is_probe and b.is_probe
    assert a.object_ref == b.object_ref and a.columns == b.columns
    quoted = object_read_scope('SELECT "PID" FROM pg_catalog.pg_stat_activity WHERE 1=0', "POSTGRESQL")
    assert quoted.columns != b.columns
    ordinary = object_read_scope("SELECT pid FROM pg_catalog.pg_stat_activity WHERE pid=0", "POSTGRESQL")
    assert not ordinary.is_probe
    assert object_read_scope("SELECT pid AS datid FROM pg_catalog.pg_stat_activity ORDER BY datid", "POSTGRESQL") is None
    annotated = object_read_scope("SELECT pid /* private-comment */ FROM pg_catalog.pg_stat_activity /* private-table-comment */ WHERE 1=0", "POSTGRESQL")
    assert annotated.is_probe and "private" not in repr(annotated)


def test_object_cache_shares_capacity_and_preserves_ttl_and_authorization_scopes():
    h = harness(dynamic=True)
    cache, clock = clock_cache(capacity=2)
    scope = object_read_scope("SELECT pid FROM pg_catalog.pg_stat_activity WHERE 1=0", "POSTGRESQL")
    key = cache.scope_key(h.grant, issued=h.control.issue_credential.return_value)
    cache.object_observation(key, scope=scope, state="DENIED", request_id=uuid7(),
        query_sha256="e" * 64, database_type="POSTGRESQL")
    assert cache.find_object(key, scope).state == "DENIED"
    assert cache.find_object("other-authorization", scope) is None
    # 查询和对象观察共用同一个 LRU，不增加第二份容量。
    cache.remember("one", state="SUCCEEDED")
    cache.remember("two", state="SUCCEEDED")
    assert cache.find_object(key, scope) is None
    cache.object_observation(key, scope=scope, state="DENIED", request_id=uuid7(),
        query_sha256="e" * 64, database_type="POSTGRESQL")
    clock.utc -= timedelta(seconds=1)
    assert cache.find_object(key, scope) is None


def test_object_cache_search_does_not_touch_unrelated_entries_and_readable_hint_expires():
    h = harness(dynamic=True)
    cache, clock = clock_cache(capacity=3)
    scope = object_read_scope("SELECT pid FROM pg_catalog.pg_stat_activity WHERE 1=0", "POSTGRESQL")
    key = cache.scope_key(h.grant, issued=h.control.issue_credential.return_value)
    cache.remember("first", state="SUCCEEDED")
    observed = cache.object_observation(key, scope=scope, state="READABLE", request_id=uuid7(),
        query_sha256="e" * 64, database_type="POSTGRESQL")
    cache.remember("last", state="SUCCEEDED")
    assert cache.find_object(key, scope).reused
    cache.remember("fourth", state="SUCCEEDED")
    assert cache.get("first") is None and cache.get("last") is not None
    assert cache.find_object(key, scope).checked_at == observed.checked_at
    clock.utc += timedelta(seconds=301)
    clock.monotonic += 301
    assert cache.find_object(key, scope) is None


def test_current_revocation_and_cancellation_still_precede_crossquery_denial():
    h = harness(dynamic=True)
    h.method.side_effect = DiagnosticDriverError("PRIVILEGE_MISSING")
    async def run():
        await h.service.execute(h.query_request("SELECT pid FROM pg_catalog.pg_stat_activity WHERE 1=0"))
        h.control.issue_credential.side_effect = DiagnosticGrantError("GRANT_REVOKED", "已撤权")
        with pytest.raises(DiagnosticGrantError):
            await h.service.execute(h.query_request("SELECT pid FROM pg_catalog.pg_stat_activity"))
        h.control.issue_credential.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await h.service.execute(h.query_request("SELECT pid FROM pg_catalog.pg_stat_activity"))
        assert h.method.await_count == 1
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["credential", "profile", "policy", "target", "capability", "domain"])
def test_crossquery_probe_isolated_by_current_credential_target_policy_and_endpoint(mutation):
    h = harness(dynamic=True)
    async def run():
        h.method.side_effect = DiagnosticDriverError("PRIVILEGE_MISSING")
        await h.service.execute(h.query_request("SELECT pid FROM pg_catalog.pg_stat_activity WHERE 1=0"))
        updates = {}
        if mutation == "credential":
            h.control.issue_credential.return_value = SimpleNamespace(username="private-user", password="rotated")
        elif mutation == "profile":
            updates["connection_profile"] = h.grant.connection_profile.model_copy(update={"database": "other"})
        elif mutation == "policy":
            # 冻结策略摘要改变时不能因旧访问观察跳过 SQL/Grant 重验。
            updates["policy_sha256"] = "d" * 64
        elif mutation == "target":
            updates["target_row_version"] = 2
        elif mutation == "capability":
            updates["capability_snapshot_hash"] = "e" * 64
        else:
            updates["domain_id"] = 102 if type(h.grant).model_fields["domain_id"].annotation is int else uuid7()
        request = h.query_request("SELECT pid FROM pg_catalog.pg_stat_activity", **updates)
        if mutation == "policy":
            with pytest.raises(DiagnosticGrantError):
                await h.service.execute(request)
        else:
            assert (await h.service.execute(request)).error_code == "PRIVILEGE_MISSING"
            assert h.method.await_count == 2
    asyncio.run(run())


def test_cancellation_is_neither_cached_nor_converted_to_a_gap():
    h = harness()
    h.method.side_effect = asyncio.CancelledError()
    async def run():
        with pytest.raises(asyncio.CancelledError):
            await h.service.execute(h.request())
        assert not h.cache._entries
    asyncio.run(run())


@pytest.mark.parametrize("engine,table,column,other", OBJECTS)
@pytest.mark.parametrize("denied", [False, True])
@pytest.mark.parametrize("reused", [False, True])
@pytest.mark.parametrize("tamper", ["none", "source", "frozen", "missing", "projection"])
def test_object_access_survives_worker_assessment_freeze_and_report_case_source_replay(
        engine, table, column, other, denied, reused, tamper):
    from aiops_agent.application.diagnosis.investigation_record import freeze_investigation, investigation_report_source
    from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
    from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
    from aiops_agent.contracts.turn_answer import AIOpsTurnResult, TurnAnswerBlock
    from aiops_agent.workers.database_handlers import DynamicQueryInvocationHandler
    from aiops_agent.workers.handlers import TaskExecutionContext
    from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler, DbaAnswerComposeHandler
    from aiops_agent.playbooks import canonical_hash

    h = harness(engine, dynamic=True)
    h.method.return_value = DriverQueryResult(columns=(column,), rows=(), truncated=False, db_version="16")
    if denied:
        h.method.side_effect = DiagnosticDriverError("PRIVILEGE_MISSING")
    validated = h.policy(h.snapshot).validate(f"SELECT {column} FROM {table}" + ("" if reused else " WHERE 1=0"), {})
    database = dict(db_type=engine, domain_id=h.grant.domain_id if isinstance(h.grant.domain_id, int) else str(h.grant.domain_id), target_row_version=1,
        connection_profile=h.grant.connection_profile.model_dump(mode="json"),
        diagnostic_credential_id=str(h.grant.diagnostic_credential_id), automatic_access_enabled=True)
    if "tenant_id" in type(h.grant).model_fields:
        database["tenant_id"] = str(h.grant.tenant_id)
    context = TaskExecutionContext(run_id=str(h.grant.run_id), task_id=str(h.grant.task_id), task_key="dynamic:a1",
        target_id=str(h.grant.target_id), agent_id=str(uuid7()), actor_id=uuid7(), trigger_type="CHAT",
        trace_id="access-pipeline", attempt=1, deadline_at=None, policy_snapshot={}, input_artifacts=(),
        lease_token="test-lease", lease_until=(datetime.now(UTC) + timedelta(seconds=60)).isoformat(),
        plan_snapshot={"capability_snapshot": {"database_type": engine},
            "answer_context": {"question": "核对所需诊断列的访问边界", "task_frame": {"objectives": ["ASSESS"]}},
            "investigation_execution": {"capability_snapshot_hash": h.grant.capability_snapshot_hash,
                "database": database, "dynamic_invocations": {"dynamic:a1": {
                    "action_id": "a1", "tool_id": h.grant.tool_id, "measurement_semantics": "CURRENT_ACTIVITY",
                    "policy_snapshot": h.snapshot.model_dump(mode="json"),
                    "validated_query": validated.model_dump(mode="json"),
                    "limits": h.grant.limits.model_dump(mode="json"),
                }}}})

    async def execute_request(request, trace_id):
        return await h.service.execute(request)

    handler = DynamicQueryInvocationHandler(
        executor_client=SimpleNamespace(execute_dynamic_diagnostic=AsyncMock(side_effect=execute_request)),
        grant_codec=h.codec, grant_issuer="test-worker", grant_audience="test-executor", grant_ttl_seconds=30)

    async def run():
        from dataclasses import replace
        if reused:
            await h.service.execute(h.query_request(f"SELECT {column} FROM {table} WHERE 1=0"))
        result = await handler.execute(context)
        outcome = result.tool_outcomes[0]
        observed = outcome.gap.object_access if denied else outcome.observation.object_access
        assert observed[0].state == ("DENIED" if denied else "READABLE")
        assert observed[0].reused is reused
        artifact_id = uuid7()
        original = result.model_dump(mode="json")
        artifact = {"artifact_id": str(artifact_id), "schema_version": "DBA_TOOL_RESULT.v1", "payload": original}
        assessment = await DbaEvidenceAssessmentHandler().execute(replace(context, input_artifacts=(artifact,)))
        projected = assessment.gaps[0].object_access if denied else assessment.evidence[0].object_access
        assert projected == observed
        if not denied:
            blocks = DbaAnswerComposeHandler._data_blocks(assessment.evidence)
            assert len(blocks) == 1 and str(blocks[0].block_type) == ("TABLE" if reused else "MARKDOWN")
            if not reused:
                assert "按设计不读取数据行" in blocks[0].payload["markdown"]
        draft = DiagnosisAnswerDraft(claims=({
            "claim_id": "c1", "kind": "INTERPRETATION", "subject": "业务数据可见范围",
            "statement": "仅凭零行访问探测无法判断业务数据的可见范围", "level": "UNKNOWN",
            "applicable_time": "当前调查", "reasoning": "访问探测不返回业务数据",
            "limitations": ("尚未取得业务数据观测",),
        },), goal_satisfied=False, unresolved_questions=("访问观察不代表业务数据可见",))
        record = freeze_investigation(draft=draft, evidence=assessment.evidence, context=context, gaps=assessment.gaps)
        report = investigation_report_source(record)
        frozen = report["investigation_record"]
        assert (frozen["access_gaps"][0] if denied else frozen["observations"][0])["object_access"]
        record.validate_observations(assessment.evidence)
        record.validate_access_gaps(assessment.gaps)
        payload = AIOpsTurnResult(status="PARTIAL", sufficiency_status=assessment.status,
            blocks=(TurnAnswerBlock(block_type="ANALYSIS_MARKDOWN", schema_version="AIOPS_ANALYSIS_BLOCK.v1",
                payload={"markdown": "访问观察不代表业务数据可见"}),),
            investigation_record=record, evidence=assessment.evidence,
            evidence_gaps=assessment.gaps).model_dump(mode="json")
        frozen = payload["investigation_record"]
        source = SimpleNamespace(ops_run_id=context.run_id, schema_version="DBA_TOOL_RESULT.v1",
            trust_level="SOURCE_VERIFIED", payload_json=original, content_hash=canonical_hash(original))
        if tamper == "source":
            parent = original["tool_outcomes"][0]["gap" if denied else "observation"]
            parent["object_access"][0]["read_columns"] = ["unrelated_column"]
            source.content_hash = canonical_hash(original)
        elif tamper == "frozen":
            parent = frozen["access_gaps"][0] if denied else frozen["observations"][0]
            parent["object_access"][0]["reused"] = not reused
        elif tamper == "projection":
            parent = payload["evidence_gaps"][0] if denied else payload["evidence"][0]
            parent["object_access"] = []
        uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=None if tamper == "missing" else source)))
        valid = await validate_investigation_sources(uow=uow,
            run=SimpleNamespace(ops_run_id=context.run_id, target_id=context.target_id), payload=payload)
        assert valid == (tamper == "none")
    asyncio.run(run())


@pytest.mark.parametrize("error,expected", [
    ("UndefinedTableError", "QUERY_OBJECT_UNAVAILABLE"),
    ("UndefinedColumnError", "QUERY_COLUMN_INVALID"),
    ("UndefinedFunctionError", "CAPABILITY_UNAVAILABLE"),
    ("InsufficientPrivilegeError", "PRIVILEGE_MISSING"),
])
def test_postgresql_native_errors_keep_object_column_and_privilege_distinct(error, expected):
    from unittest.mock import patch
    from aiops_agent.executor.drivers.postgresql import PostgreSQLDiagnosticDriver
    import asyncpg
    h = harness()
    async def run():
        with patch("aiops_agent.executor.drivers.postgresql.asyncpg.connect", AsyncMock(side_effect=getattr(asyncpg, error)("private detail"))):
            with pytest.raises(DiagnosticDriverError) as caught:
                await PostgreSQLDiagnosticDriver().execute_dynamic(profile=h.grant.connection_profile,
                    secret=SimpleNamespace(values={"username": "u", "password": "p"}),
                    sql="SELECT pid FROM pg_catalog.pg_stat_activity", parameters={},
                    limits=h.grant.limits, trace_id="native-errors")
            assert caught.value.code == expected
    asyncio.run(run())


@pytest.mark.parametrize("code,expected", [
    (1142, "PRIVILEGE_MISSING"), (1143, "PRIVILEGE_MISSING"),
    (1146, "QUERY_OBJECT_UNAVAILABLE"), (1054, "QUERY_COLUMN_INVALID"), (1045, "AUTH_FAILED"),
])
def test_mysql_native_column_privilege_and_object_errors_are_distinct(code, expected):
    from unittest.mock import patch
    from aiops_agent.executor.drivers.mysql import MySQLDiagnosticDriver
    import aiomysql
    h = harness("MYSQL")
    async def run():
        with patch("aiops_agent.executor.drivers.mysql.aiomysql.connect", AsyncMock(side_effect=aiomysql.Error(code, "private detail"))):
            with pytest.raises(DiagnosticDriverError) as caught:
                await MySQLDiagnosticDriver().execute_dynamic(profile=h.grant.connection_profile,
                    secret=SimpleNamespace(values={"username": "u", "password": "p"}),
                    sql="SELECT thread_id FROM performance_schema.threads", parameters={},
                    limits=h.grant.limits, trace_id="native-errors")
            assert caught.value.code == expected
    asyncio.run(run())
