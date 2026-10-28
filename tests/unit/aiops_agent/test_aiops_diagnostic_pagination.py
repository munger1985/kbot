"""三库同执行游标分页、签名预算、截断与正式证据来源重放回归。"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.investigation.discovery import available_tools, build_playbook_plan
from aiops_agent.application.investigation.progress import action_fingerprint, drop_repeated_plan_actions, execution_action_fingerprints
from aiops_agent.application.investigation.projection import safe_plan_projection
from aiops_agent.application.investigation.query_freezing import prepare_dynamic_queries
from aiops_agent.application.investigation.reasoner import InvestigationPlanValidationError
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.executor.drivers import DriverQueryResult, DiagnosticDriverError
from aiops_agent.executor.drivers.pagination import read_paged_rows
from aiops_agent.executor.dynamic_service import DynamicDiagnosticExecutorService, DynamicOutputValidationError
from aiops_agent.playbooks import PlaybookRegistry, canonical_hash
from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops.executor import DiagnosticLimits, DiagnosticPaginationObservation
from platform_core.contracts.aiops.investigation import DiagnosticPagination, InvestigationAction
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_dynamic_sampling import QUERIES, Clock, planning, wiring


def limits(**updates):
    return DiagnosticLimits(statement_timeout_seconds=20, max_result_rows=6,
        max_result_bytes=1048576, pagination={"page_size": 2, "max_pages": 3}, **updates)


def fetcher(records):
    remaining = list(records)
    async def fetch(count):
        batch = remaining[:count]
        del remaining[:count]
        return batch
    return AsyncMock(side_effect=fetch)


def paged_plan(engine, **updates):
    return planning(engine, sampling=False,
        pagination={"page_size": 2, "max_pages": 3}, **updates)


@pytest.mark.parametrize("engine", QUERIES)
def test_freeze_compiler_discovery_projection_and_repeat_budget(engine):
    planned, frozen = prepare_dynamic_queries(paged_plan(engine), database_type=engine)
    selected = planned.plan.actions[0]
    invocation = frozen[0]
    assert invocation["policy_snapshot"]["max_rows"] == 7
    assert invocation["limits"]["max_result_rows"] == 6
    assert invocation["limits"]["pagination"] == {"page_size": 2, "max_pages": 3}
    registry, playbooks = DiagnosticRegistry.load(), PlaybookRegistry.load()
    caps = DbaCapabilitySnapshot(agent_id=str(uuid7()), agent_version_id=str(uuid7()),
        target_id=str(uuid7()), database_type=engine, database_version={"ORACLE": "19", "POSTGRESQL": "16", "MYSQL": "8.4"}[engine],
        target_enabled=True, target_reachable=True, target_capabilities=("DB_READONLY",))
    builder = ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=registry)
    card = next(item for item in available_tools(builder, caps) if item["tool_id"] == selected.tool_id)
    assert card["pagination_policy"]["max_rows"] == 800
    compiled = InvestigationTaskCompiler(playbooks).compile(
        build_playbook_plan(playbooks), investigation_actions=planned.plan.actions)
    task = next(item for item in compiled.tasks if item.task_key == "dynamic:a1")
    assert task.max_attempts == 1
    projection = safe_plan_projection(planned.plan.model_dump(mode="json"),
        execution_snapshot={"dynamic_invocations": {"dynamic:a1": invocation}})
    assert projection["actions"][0]["pagination"] == {"page_size": 2, "max_pages": 3}
    original = action_fingerprint(selected.tool_id, selected.input, paged=True)
    changed_plan, _ = prepare_dynamic_queries(paged_plan(engine).model_copy(update={
        "plan": paged_plan(engine).plan.model_copy(update={"actions": (
            paged_plan(engine).plan.actions[0].model_copy(update={"pagination": DiagnosticPagination(page_size=3, max_pages=3)}),
        )}),
    }), database_type=engine)
    assert drop_repeated_plan_actions(changed_plan.plan, executed_fingerprints=(original,))[1] == ("a1",)
    assert execution_action_fingerprints({"dynamic_invocations": {"dynamic:a1": invocation}}) == (original,)


@pytest.mark.parametrize("pagination", [
    {"page_size": 201, "max_pages": 2}, {"page_size": 2, "max_pages": 5},
    {"page_size": True, "max_pages": 2}, {"page_size": 2, "max_pages": 1},
])
def test_contract_rejects_unbounded_or_coerced_page_budget(pagination):
    with pytest.raises(ValueError):
        DiagnosticPagination.model_validate(pagination)


@pytest.mark.parametrize("change", [
    {"sampling": {"count": 2, "interval_seconds": 2}},
    {"observation_schedule": {"count": 2, "interval_seconds": 30}},
    {"tool_id": "db.sql.top_current"}, {"evidence_need": ""},
])
def test_paging_is_not_a_sampling_or_fixed_tool_escape(change):
    raw = paged_plan("POSTGRESQL").plan.actions[0].model_dump(mode="json")
    with pytest.raises(ValueError):
        InvestigationAction.model_validate({**raw, **change})


def test_no_paging_budget_bleeds_into_the_next_action():
    source = paged_plan("POSTGRESQL")
    plain = planning("POSTGRESQL", sampling=False).plan.actions[0].model_copy(update={"action_id": "a2"})
    mixed = source.model_copy(update={"plan": source.plan.model_copy(update={"actions": (*source.plan.actions, plain)})})
    _, frozen = prepare_dynamic_queries(mixed, database_type="POSTGRESQL")
    assert frozen[0]["policy_snapshot"]["max_rows"] == 7
    assert frozen[1]["policy_snapshot"]["max_rows"] == 200
    assert "pagination" not in frozen[1]["limits"]
    twice = source.model_copy(update={"plan": source.plan.model_copy(update={"actions": (
        *source.plan.actions, source.plan.actions[0].model_copy(update={"action_id": "a2"}),
    )})})
    with pytest.raises(InvestigationPlanValidationError):
        prepare_dynamic_queries(twice, database_type="POSTGRESQL")


@pytest.mark.parametrize("engine,sql", [
    ("POSTGRESQL", "EXPLAIN SELECT pid FROM pg_catalog.pg_stat_activity"),
    ("POSTGRESQL", "SELECT * FROM business.orders"),
    ("ORACLE", "SELECT sql_id FROM dba_hist_sqlstat"),
    ("MYSQL", "SELECT * FROM business.orders"),
])
def test_paging_cannot_expand_object_license_or_plan_permissions(engine, sql):
    with pytest.raises(InvestigationPlanValidationError):
        prepare_dynamic_queries(paged_plan(engine, input={"sql": sql, "parameters": {}}), database_type=engine)


@pytest.mark.parametrize("size,counts,stop", [
    (0, (), "EXHAUSTED"), (3, (2, 1), "EXHAUSTED"),
    (6, (2, 2, 2), "EXHAUSTED"), (7, (2, 2, 2), "ROW_LIMIT"),
    (100, (2, 2, 2), "ROW_LIMIT"),
])
def test_cursor_reads_each_row_once_with_only_one_sentinel(size, counts, stop):
    fetch = fetcher([(index,) for index in range(size)])
    rows, summary = asyncio.run(read_paged_rows(fetch_page=fetch, limits=limits()))
    assert rows == tuple((index,) for index in range(min(size, 6)))
    assert summary.returned_page_rows == counts
    assert summary.stop_reason == stop
    assert summary.source_completeness == "UNKNOWN"
    assert sum(call.args[0] for call in fetch.await_args_list) <= 7
    assert DiagnosticPaginationObservation.model_validate(summary.model_dump(mode="json")) == summary


@pytest.mark.parametrize("records,budget,reason,expected", [
    ([("ok",), ("long",)], {"max_result_bytes": 8}, "BYTE_LIMIT", (("ok",),)),
    ([("ok",), ("too-long",)], {"max_cell_chars": 3}, "CELL_LIMIT", (("ok",),)),
    ([(1,), (1e-100,)], {"max_result_bytes": 20}, "BYTE_LIMIT", ((1,),)),
])
def test_total_byte_and_cell_limits_preserve_only_valid_prefix(records, budget, reason, expected):
    bounded = limits().model_copy(update=budget)
    rows, summary = asyncio.run(read_paged_rows(fetch_page=fetcher(records), limits=bounded))
    assert rows == expected and summary.stop_reason == reason
    assert summary.returned_page_rows == (1,)


def test_cancelled_fetch_is_not_caught_as_success():
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(read_paged_rows(fetch_page=AsyncMock(side_effect=asyncio.CancelledError), limits=limits()))


@pytest.mark.parametrize("engine,size", [(engine, size) for engine in QUERIES for size in (3, 8)])
def test_real_worker_signed_executor_and_assessor_keep_single_observation(engine, size):
    async def run():
        Clock.current = datetime.now(timezone.utc)
        wire = wiring(engine)
        planned, frozen = prepare_dynamic_queries(paged_plan(engine), database_type=engine)
        wire.ctx.plan_snapshot["investigation_execution"]["dynamic_invocations"]["dynamic:a1"] = frozen[0]
        async def raw(**kwargs):
            rows, summary = await read_paged_rows(fetch_page=fetcher([(i,) for i in range(size)]), limits=kwargs["limits"])
            return DriverQueryResult(columns=("session_id",), rows=rows,
                truncated=summary.stop_reason != "EXHAUSTED", db_version="16", pagination=summary)
        wire.driver.execute_dynamic.side_effect = raw
        result = await wire.handler.execute(wire.ctx)
        assert result.status == ("PARTIAL" if size > 6 else "SUCCEEDED")
        assert result.sampling is None and result.observation_series is None
        assert len(result.tool_outcomes) == len(wire.requests) == wire.control.issue_credential.await_count == 1
        assert wire.grants[0].limits.pagination == planned.plan.actions[0].pagination
        observation = result.tool_outcomes[0].observation
        assert observation.row_count == min(size, 6)
        assert observation.pagination.requested == planned.plan.actions[0].pagination
        assert DbaToolResult.model_validate(result.model_dump(mode="json")) == result
        if size > 6:
            invalid = result.model_dump(mode="json")
            invalid["status"] = "SUCCEEDED"
            with pytest.raises(ValueError):
                DbaToolResult.model_validate(invalid)
        invalid = result.model_dump(mode="json")
        invalid["tool_outcomes"].append(invalid["tool_outcomes"][0])
        with pytest.raises(ValueError):
            DbaToolResult.model_validate(invalid)
        artifact_id = uuid7()
        assessed = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": result.schema_version, "artifact_id": artifact_id,
                "payload": result.model_dump(mode="json")},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT"))
        assert len(assessed.evidence) == 1
        fact = assessed.evidence[0]
        assert fact.pagination == observation.pagination
        assert any("不是多个采样时点" in value for value in fact.warnings)
        source = SimpleNamespace(artifact_id=artifact_id, ops_run_id=wire.ctx.run_id, schema_version=result.schema_version,
            trust_level="SOURCE_VERIFIED", payload_json=result.model_dump(mode="json"),
            content_hash=canonical_hash(result.model_dump(mode="json")))
        uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=source)))
        current_run = SimpleNamespace(ops_run_id=wire.ctx.run_id, target_id=wire.ctx.target_id)
        payload = {"evidence": [fact.model_dump(mode="json")]}
        assert await validate_investigation_sources(uow=uow, run=current_run, payload=payload)
        # 只改副本页预算不改行数仍必须被原始来源复核拦截。
        payload["evidence"][0]["pagination"]["requested"]["max_pages"] = 4
        assert not await validate_investigation_sources(uow=uow, run=current_run, payload=payload)
    asyncio.run(run())


@pytest.mark.parametrize("engine", QUERIES)
def test_three_native_drivers_use_one_query_and_bounded_cursor(monkeypatch, engine):
    from aiops_agent.executor.drivers import PostgreSQLDiagnosticDriver, MySQLDiagnosticDriver, OracleDiagnosticDriver
    paged = fetcher([(index,) for index in range(10)])
    profile = SimpleNamespace(host="db.internal", port=5432, database="app", service="PDB1",
        tls_enabled=False, tls_profile_ref=None)
    secret = SimpleNamespace(values={"username": "readonly", "password": "not-a-real-secret"})
    transaction = SimpleNamespace(start=AsyncMock(), rollback=AsyncMock())
    if engine == "POSTGRESQL":
        cursor = SimpleNamespace(fetch=paged)
        statement = SimpleNamespace(get_attributes=lambda: (SimpleNamespace(name="session_id", type=SimpleNamespace(name="int8")),),
            cursor=AsyncMock(return_value=cursor))
        connection = SimpleNamespace(fetchval=AsyncMock(return_value="16"),
            transaction=Mock(return_value=transaction), prepare=AsyncMock(return_value=statement),
            close=AsyncMock(), terminate=Mock())
        monkeypatch.setattr("aiops_agent.executor.drivers.postgresql.asyncpg.connect", AsyncMock(return_value=connection))
        monkeypatch.setattr("aiops_agent.executor.identity.observe_postgresql_identity", AsyncMock(return_value=()))
        driver = PostgreSQLDiagnosticDriver()
    elif engine == "ORACLE":
        import oracledb
        cursor = SimpleNamespace(execute=AsyncMock(), fetchmany=paged,
            description=(("SESSION_ID", oracledb.DB_TYPE_NUMBER),), close=Mock())
        connection = SimpleNamespace(cursor=Mock(return_value=cursor), rollback=AsyncMock(),
            close=AsyncMock(), version="19", cancel=Mock())
        monkeypatch.setattr("aiops_agent.executor.drivers.oracle.oracledb.connect_async", AsyncMock(return_value=connection))
        monkeypatch.setattr("aiops_agent.executor.identity.observe_oracle_identity", AsyncMock(return_value=()))
        driver = OracleDiagnosticDriver()
    else:
        cursor = SimpleNamespace(execute=AsyncMock(), fetchmany=paged, description=(("SESSION_ID",),), close=AsyncMock())
        connection = SimpleNamespace(cursor=AsyncMock(return_value=cursor), close=Mock(),
            rollback=AsyncMock(), get_server_info=Mock(return_value="8.4"))
        monkeypatch.setattr("aiops_agent.executor.drivers.mysql.aiomysql.connect", AsyncMock(return_value=connection))
        monkeypatch.setattr("aiops_agent.executor.identity.observe_mysql_connection_identity", AsyncMock(return_value=()))
        driver = MySQLDiagnosticDriver()
    result = asyncio.run(driver.execute_dynamic(profile=profile, secret=secret, sql="SELECT session_id",
        parameters={}, limits=limits(), trace_id="bounded-cursor"))
    assert result.rows == tuple((i,) for i in range(6))
    assert result.pagination.stop_reason == "ROW_LIMIT"
    assert [call.args[0] for call in paged.await_args_list] == [2, 2, 2, 1]
    if engine == "POSTGRESQL":
        connection.prepare.assert_awaited_once()
        statement.cursor.assert_awaited_once()
        connection.close.assert_awaited_once()
    else:
        sql_calls = [call for call in cursor.execute.await_args_list if call.args[0] == "SELECT session_id"]
        assert len(sql_calls) == 1
        if engine == "MYSQL":
            import aiomysql
            connection.cursor.assert_awaited_once_with(aiomysql.SSCursor)
            cursor.close.assert_not_awaited()
            connection.rollback.assert_not_awaited()
            assert connection.close.called
        else:
            cursor.close.assert_called_once()


def test_native_mysql_paging_closes_connection_on_cancellation(monkeypatch):
    from aiops_agent.executor.drivers.mysql import _diagnostic_cursor
    cursor = SimpleNamespace(close=AsyncMock())
    connection = SimpleNamespace(cursor=AsyncMock(return_value=cursor), close=Mock())
    async def run():
        async with _diagnostic_cursor(connection, paged=True):
            raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    connection.close.assert_called_once()
    cursor.close.assert_not_awaited()

@pytest.mark.parametrize("corrupt", ["missing_summary", "wrong_budget", "false_exhaustion", "extra_rows"])
def test_executor_rejects_driver_pagination_that_does_not_match_signed_budget(corrupt):
    async def run():
        Clock.current = datetime.now(timezone.utc)
        wire = wiring("POSTGRESQL")
        _, frozen = prepare_dynamic_queries(paged_plan("POSTGRESQL"), database_type="POSTGRESQL")
        wire.ctx.plan_snapshot["investigation_execution"]["dynamic_invocations"]["dynamic:a1"] = frozen[0]
        async def raw(**kwargs):
            rows, summary = await read_paged_rows(fetch_page=fetcher([(i,) for i in range(8)]), limits=kwargs["limits"])
            if corrupt == "wrong_budget":
                summary = summary.model_copy(update={"requested": DiagnosticPagination(page_size=2, max_pages=4)})
            return DriverQueryResult(columns=("session_id",),
                rows=(*rows, (99,)) if corrupt == "extra_rows" else rows,
                truncated=corrupt != "false_exhaustion", db_version="16",
                pagination=None if corrupt == "missing_summary" else summary)
        wire.driver.execute_dynamic.side_effect = raw
        result = await wire.handler.execute(wire.ctx)
        assert result.status == "FAILED"
        assert result.tool_outcomes[0].observation is None
        assert result.tool_outcomes[0].gap.code == "OUTPUT_SCHEMA_INVALID"
        assert wire.client.execute_dynamic_diagnostic.await_count == 1
    asyncio.run(run())


@pytest.mark.parametrize("change", [
    {"returned_page_rows": [3]}, {"returned_page_rows": [1, 2]},
    {"returned_page_rows": [2, 2, 2, 2]}, {"returned_page_rows": [0]},
    {"source_completeness": "COMPLETE"},
])
def test_pagination_summary_rejects_false_coverage(change):
    with pytest.raises(ValueError):
        DiagnosticPaginationObservation.model_validate({
            "requested": {"page_size": 2, "max_pages": 3}, "returned_page_rows": [2, 1],
            "stop_reason": "EXHAUSTED", **change,
        })
