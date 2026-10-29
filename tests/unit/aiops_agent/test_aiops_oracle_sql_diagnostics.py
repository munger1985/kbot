"""Oracle 增强诊断的目录、精确关联、按需补证及交付回归，不冒充实库验收。"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aiops_agent.application.diagnosis.oracle_sql_evidence import (
    XPLAN, MONITOR, MONITOR_PLAN, ASH, ORACLE_SQL_TOOLS, ash_window,
    authorized_oracle_sql_tools, oracle_evidence_policy, pending_oracle_sql_actions,
)
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle, validate_sql_diagnosis
from aiops_agent.contracts.sql_diagnosis import OracleSqlEvidenceRequest, SqlDiagnosisDraft
from aiops_agent.contracts.sql_diagnostic_response import SqlDiagnosticResponse
from aiops_agent.diagnostics import DiagnosticRegistry
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_oracle_cursor_plan import PARAMS, direct, fact, observation
from tests.unit.aiops_agent.test_aiops_sql_plan_priority import frame, caps
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import diagnosis


PACKS = {"ORACLE_DIAGNOSTICS_PACK", "ORACLE_TUNING_PACK"}


def tool(tool_id, packs=PACKS):
    return DiagnosticRegistry.load().resolve(tool_id=tool_id, tool_version="1.0.0", db_type="ORACLE",
        db_version="19.3", capabilities={"dynamic_performance_views"}, entitlements=set(packs))


def runtime(tool_id, **values):
    fields = {**PARAMS, **values}
    original = direct()
    identity = str(uuid7())
    return type(original).model_validate({**original.model_dump(mode="python"),
        "artifact_id": identity, "evidence_ref": f"artifact:{identity}#a1",
        "tool_id": tool_id, "source_id": tool_id,
        "columns": tuple({"name": key} for key in fields), "rows": (tuple(fields.values()),),
        "row_count": 1, "truncated": False, "sql_plan": None,
        "measurement_semantics": "HISTORICAL_SAMPLES" if tool_id == ASH else "CURRENT_ACTIVITY",
    })


def actions(evidence=None, **kwargs):
    return pending_oracle_sql_actions(task_frame=kwargs.pop("task_frame", frame()),
        database_type=kwargs.pop("database_type", "ORACLE"),
        evidence=evidence if evidence is not None else (direct(),),
        available_tool_ids=kwargs.pop("available_tool_ids", ORACLE_SQL_TOOLS), **kwargs)


def subject_id():
    return build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE",
        evidence=(direct(),))["subjects"][0]["subject_id"]


@pytest.mark.parametrize("tool_id", ORACLE_SQL_TOOLS)
def test_exact_catalog_uses_full_cursor_without_replay(tool_id):
    resolved = tool(tool_id)
    for key in PARAMS:
        assert ":" + key in resolved.sql
    assert resolved.definition.timeout_seconds == 10
    assert set(PARAMS) <= {parameter.name for parameter in resolved.definition.parameters}
    assert resolved.definition.measurement_semantics == ("HISTORICAL_SAMPLES" if tool_id == ASH else "CURRENT_ACTIVITY")
    assert "value_string" not in resolved.sql.lower()
    assert "explain plan" not in resolved.sql.lower()
    if tool_id != XPLAN:
        assert "report_html" not in resolved.sql.lower()
        assert resolved.definition.required_entitlements


def test_xplan_can_only_use_local_instance_current_container_and_explicit_child():
    sql = tool(XPLAN).sql.lower()
    assert "c.inst_id = i.instance_number" in sql
    assert "sys_context('userenv', 'con_id')" in sql
    assert "s.sql_id, s.child_number" in sql
    assert "allstats last +predicate +alias +outline" in sql
    assert "peeked_binds" not in sql
    assert "dbms_xplan.display_cursor" in sql
    assert {"V_$SQL", "V_$SESSION", "V_$SQL_PLAN", "V_$SQL_PLAN_STATISTICS_ALL"} <= set(tool(XPLAN).definition.required_privileges)


def test_monitor_plan_uses_native_process_key_and_child_address_not_reused_child_number():
    sql = tool(MONITOR_PLAN).sql.lower()
    assert "m.key = p.key" in sql
    assert "m.sql_exec_id = p.sql_exec_id" in sql
    assert "m.sql_exec_start = p.sql_exec_start" in sql
    assert "s.child_address = p.sql_child_address" in sql
    # 19c 的序列号来自 Monitor，Plan Monitor 没有该列。
    assert "m.session_serial#" in sql
    assert "p.session_serial#" not in sql
    assert "s.plan_hash_value = m.sql_plan_hash_value" in tool(MONITOR).sql.lower()
    assert "GV_$SQL_MONITOR" in tool(MONITOR_PLAN).definition.required_privileges
    assert tool(MONITOR_PLAN).definition.limit_semantics == "BOUNDED_ENUMERATION"
    assert tool(ASH).definition.limit_semantics == "BOUNDED_ENUMERATION"


def test_pack_gate_is_shared_with_catalog_and_does_not_infer_license():
    assert authorized_oracle_sql_tools(caps().model_dump(mode="json")) == (XPLAN,)
    for tool_id in (MONITOR, MONITOR_PLAN, ASH):
        with pytest.raises(LookupError):
            tool(tool_id, packs=())
    with pytest.raises(LookupError):
        tool(MONITOR, packs={"ORACLE_DIAGNOSTICS_PACK"})
    assert tool(ASH, packs={"ORACLE_DIAGNOSTICS_PACK"})
    assert not authorized_oracle_sql_tools({"database_type": "ORACLE"})


def test_default_diagnostic_facts_are_optional_and_not_expensive_html_reports():
    result = actions()
    assert [action.tool_id for action in result] == [XPLAN, MONITOR]
    assert all(action.optional and action.input == PARAMS for action in result)
    assert ASH not in [action.tool_id for action in result]
    assert not actions(remaining_seconds=14)
    assert not actions(database_type="POSTGRESQL")
    assert not actions(available_tool_ids=())


@pytest.mark.parametrize("tool_id", (XPLAN, MONITOR, MONITOR_PLAN, ASH))
def test_failed_zero_or_scheduled_optional_tool_never_retries(tool_id):
    empty = runtime(tool_id).model_copy(update={"rows": (), "row_count": 0})
    request = OracleSqlEvidenceRequest(kind="ASH", subject_id=subject_id(), reason="判断等待是否来自锁")
    for kwargs in (
        {"evidence": (direct(), empty)},
        {"gaps": ({"source_id": tool_id, "code": "PRIVILEGE_MISSING"},)},
        {"prior_actions": ({"tool_id": tool_id, "input": PARAMS},)},
    ):
        result = actions(requests=(request,), **kwargs)
        assert tool_id not in [action.tool_id for action in result]


@pytest.mark.parametrize("dimension", ("sql_id", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value"))
def test_monitor_continuation_rejects_other_cursor_or_execution(dimension):
    wrong = dict(PARAMS)
    wrong[dimension] = "9999999999999" if dimension == "sql_id" else 999
    entry = runtime(MONITOR, **wrong, sql_exec_id=99, sql_exec_start_native="2026-10-29T10:00:00")
    assert MONITOR_PLAN not in [action.tool_id for action in actions((direct(), entry))]


def test_monitor_execution_binding_uses_raw_start_token_not_guessed_timezone():
    entry = runtime(MONITOR, sql_exec_id=99, sql_exec_start_native="2026-10-29T10:00:00", cpu_time_us=100)
    result = next(action for action in actions((direct(), entry)) if action.tool_id == MONITOR_PLAN)
    assert result.input == {**PARAMS, "sql_exec_id": 99, "sql_exec_start_native": "2026-10-29T10:00:00"}
    history = entry.model_copy(update={"source_id": "prior_turn.sql_observation"})
    assert MONITOR_PLAN not in [action.tool_id for action in actions((direct(), history))]
    values = DiagnosticRegistry.load().validate_parameters(tool(MONITOR_PLAN), result.input)
    assert values["limit"] == 128
    with pytest.raises(ValueError):
        DiagnosticRegistry.load().validate_parameters(tool(MONITOR_PLAN),
            {**result.input, "sql_exec_start_native": "2026-10-29T10:00:00Z"})


def test_ash_requires_typed_subject_request_and_frozen_bounded_utc_window():
    request = OracleSqlEvidenceRequest(kind="ASH", subject_id=subject_id(), reason="区分CPU和行锁等待")
    result = next(action for action in actions(requests=(request,)) if action.tool_id == ASH)
    assert result.measurement_semantics == "HISTORICAL_SAMPLES"
    assert result.input["end_time"] == direct().captured_at
    assert datetime.fromisoformat(result.input["end_time"]) - datetime.fromisoformat(result.input["begin_time"]) == timedelta(minutes=15)
    assert ash_window({**frame(), "requested_window_seconds": 90000}, (direct(),)) is None
    assert ASH not in [action.tool_id for action in actions(requests=(request,), available_tool_ids=(XPLAN,))]
    assert not ash_window(frame(), (direct().model_copy(update={"captured_at": "2026-10-29T10:00:00"}),))
    with pytest.raises(ValidationError):
        OracleSqlEvidenceRequest(kind="ASH", subject_id=subject_id(), reason="取证", sql="select 1")


@pytest.mark.parametrize("begin,end", [
    ("2026-10-29T10:00:00", "2026-10-29T11:00:00"),
    ("2026-10-29T11:00:00+00:00", "2026-10-29T10:00:00+00:00"),
    ("2026-10-27T10:00:00+00:00", "2026-10-29T10:00:00+00:00"),
])
def test_ash_registry_rejects_ambiguous_or_unbounded_window(begin, end):
    with pytest.raises(ValueError):
        DiagnosticRegistry.load().validate_parameters(tool(ASH), {**PARAMS, "begin_time": begin, "end_time": end})


def test_ash_registry_canonicalizes_offsets_without_changing_interval():
    result = DiagnosticRegistry.load().validate_parameters(tool(ASH),
        {**PARAMS, "begin_time": "2026-10-29T18:00:00+08:00", "end_time": "2026-10-29T18:15:00+08:00"})
    assert result["begin_time"] == "2026-10-29T10:00:00+00:00"
    assert result["end_time"] == "2026-10-29T10:15:00+00:00"
    assert "sample_time_utc" in tool(ASH).sql


def test_runtime_bundle_links_exact_execution_plan_line_and_keeps_native_values():
    monitor = runtime(MONITOR, sql_exec_id=99, sql_exec_start_native="2026-10-29T10:00:00", cpu_time_us=123)
    process = runtime(MONITOR_PLAN, sql_exec_id=99, sql_exec_start_native="2026-10-29T10:00:00",
        plan_line_id=1, output_rows=2000, workarea_tempseg=0, process_name="ora")
    ash = runtime(ASH, plan_line_id=1, session_state="WAITING", event="enq: TX - row lock contention",
        sample_count=7, total_sample_count=10, blocking_session=42)
    bundle = build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE",
        evidence=(direct(), fact(observation()), monitor, process, ash))
    context = bundle["oracle_runtime"][0]
    assert len(context["monitor_executions"][0]["process_rows"]) == 1
    assert context["plan_line_links"][0]["node_id"] == 1
    assert len(context["plan_line_links"][0]["related_rows"]) == 2
    assert context["ash_rows"][0]["evidence_ref"] == ash.evidence_ref
    assert next(item for item in bundle["observations"] if item["tool_id"] == MONITOR)["records"][0]["fields"]["cpu_time_us"] == "123"


def test_diagnostic_completion_cannot_ignore_existing_runtime_facts():
    anchor = direct()
    monitor = runtime(MONITOR, sql_exec_id=99, sql_exec_start_native="2026-10-29T10:00:00", cpu_time_us=123)
    bundle = build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE", evidence=(anchor, monitor))
    draft = diagnosis(anchor)
    with pytest.raises(ValueError, match="必须解释已取得"):
        validate_sql_diagnosis(draft, bundle)
    data = draft.model_dump(mode="json")
    data["conclusions"][0]["evidence_fields"].append({"evidence_ref": monitor.evidence_ref, "row_index": 0, "field": "cpu_time_us"})
    validate_sql_diagnosis(SqlDiagnosisDraft.model_validate(data), bundle)


def test_unknown_or_cross_engine_request_and_completed_request_are_rejected():
    anchor = direct()
    bundle = build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE", evidence=(anchor,))
    data = diagnosis(anchor, goal=False).model_dump(mode="json")
    data["oracle_evidence_requests"] = [{"kind": "ASH", "subject_id": "not-a-subject", "reason": "取证"}]
    with pytest.raises(ValueError, match="当前已解析"):
        validate_sql_diagnosis(SqlDiagnosisDraft.model_validate(data), bundle)
    data["oracle_evidence_requests"][0]["subject_id"] = subject_id()
    validate_sql_diagnosis(SqlDiagnosisDraft.model_validate(data), bundle)
    with pytest.raises(ValueError):
        validate_sql_diagnosis(SqlDiagnosisDraft.model_validate(data), {**bundle, "database_type": "MYSQL"})
    data["goal_satisfied"] = True
    with pytest.raises(ValidationError):
        SqlDiagnosisDraft.model_validate(data)


def test_xplan_projection_does_not_drop_predicates_after_line_forty():
    entry = runtime(XPLAN, plan_table_output="Predicate Information")
    row = entry.rows[0]
    entry = entry.model_copy(update={"rows": tuple(row for _ in range(80)), "row_count": 80})
    bundle = build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE", evidence=(direct(), entry))
    assert len(bundle["observations"][1]["records"]) == 80
    policy = oracle_evidence_policy(task_frame=frame(), evidence=(direct(), entry),
        available_tool_ids=(XPLAN,), gaps=())
    assert policy["attempted_tools"] == [XPLAN]
    assert policy["available_tools"] == [XPLAN]

def test_explicit_historical_ash_window_is_not_replaced_by_default_recent_window():
    request = OracleSqlEvidenceRequest(kind="ASH", subject_id=subject_id(), reason="验证用户明确的历史阻塞区间",
        begin_time="2026-10-28T08:00:00+08:00", end_time="2026-10-28T08:15:00+08:00")
    action = next(value for value in actions(requests=(request,)) if value.tool_id == ASH)
    assert action.input["begin_time"] == "2026-10-28T00:00:00+00:00"
    assert action.input["end_time"] == "2026-10-28T00:15:00+00:00"
    with pytest.raises(ValidationError):
        OracleSqlEvidenceRequest(kind="ASH", subject_id=subject_id(), reason="取证",
            begin_time="2026-10-28T08:00:00+08:00")
    with pytest.raises(ValidationError):
        OracleSqlEvidenceRequest(kind="ASH", subject_id=subject_id(), reason="取证",
            begin_time="2026-10-28T08:00:00", end_time="2026-10-28T08:15:00")


def test_native_xplan_is_attached_only_for_matching_plan_hash_and_physical_source():
    from aiops_agent.application.diagnosis.sql_plan_presentation import plan_table_block
    cached = fact(observation())
    text = runtime(XPLAN, plan_table_output="Plan hash value: " + str(PARAMS["plan_hash_value"]))
    block = plan_table_block(cached, native_evidence=(text,))
    assert block.payload["native_plan"]["lines"] == ["Plan hash value: " + str(PARAMS["plan_hash_value"])]
    assert block.evidence_refs == (cached.evidence_ref, text.evidence_ref)
    wrong = runtime(XPLAN, plan_table_output="Plan hash value: 99")
    assert plan_table_block(cached, native_evidence=(wrong,)).payload["native_plan"] is None
    unknown = text.model_copy(update={"provenance": None})
    assert plan_table_block(cached, native_evidence=(unknown,)).payload["native_plan"] is None


def test_ash_typed_request_enters_formal_replan_without_another_planner_call():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from aiops_agent.application.investigation.service import TurnPlanningService
    from aiops_agent.playbooks import PlaybookRegistry
    from aiops_agent.tools import ToolExecutionSnapshotBuilder

    async def exercise():
        registry = PlaybookRegistry.load()
        service = object.__new__(TurnPlanningService)
        service._playbook_registry = registry
        service._tool_snapshot_builder = ToolExecutionSnapshotBuilder(playbook_registry=registry,
            diagnostic_registry=DiagnosticRegistry.load())
        service._agent_catalog = SimpleNamespace(resolve_planner_model=AsyncMock(return_value={}),
            resolve_diagnosis_model=AsyncMock(return_value={}))
        service._investigation_reasoner = SimpleNamespace(replan=AsyncMock(side_effect=AssertionError("无需模型重规划")))
        empty = [runtime(tool_id).model_copy(update={"rows": (), "row_count": 0}) for tool_id in (XPLAN, MONITOR)]
        facts = (direct(), fact(observation()), *empty)
        inputs = SimpleNamespace(input_envelope=None, task_frame=frame(), prior_plan={"actions": []},
            assessment={"evidence": [item.model_dump(mode="json") for item in facts], "gaps": [],
                "sql_checkpoint": {"draft": {"oracle_evidence_requests": [
                    {"kind": "ASH", "subject_id": subject_id(), "reason": "区分等待与CPU"}]}}})
        service._load_replan_inputs = AsyncMock(return_value=inputs)
        service._prepare_query_inputs = Mock(side_effect=lambda **kw: (kw["investigation"], (), (), ()))
        service._compile_and_persist_replan = AsyncMock(return_value={"status": "COLLECTING"})
        context = SimpleNamespace(workflow_kind="CHAT", tenant_id=None, agent_id=uuid7(), domain_id=uuid7(),
            trace_id="oracle", capabilities=caps().model_copy(update={"entitlements": tuple(PACKS)}), question="诊断SQL")
        await service._execute_replan_once({"assessment_artifact_id": str(uuid7())}, context=context, revision_no=3)
        call = service._compile_and_persist_replan.await_args.kwargs
        assert call["planning_receipt"] is None
        assert [item.tool_id for item in call["investigation"].plan.actions] == [ASH]
        assert call["investigation"].plan.actions[0].input["sql_id"] == PARAMS["sql_id"]
        service._investigation_reasoner.replan.assert_not_awaited()
    asyncio.run(exercise())


@pytest.mark.parametrize("tool_id", ORACLE_SQL_TOOLS)
@pytest.mark.parametrize("outcome", ["rows", "empty", "denied", "boundary"])
def test_oracle_enhancements_execute_through_signed_worker_and_executor(tool_id, outcome):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact
    from aiops_agent.application.investigation.discovery import build_playbook_plan
    from aiops_agent.diagnostics.grants import DiagnosticGrantCodec
    from aiops_agent.executor import DiagnosticExecutorService
    from aiops_agent.executor.drivers import DiagnosticDriverError, DriverQueryResult
    from aiops_agent.playbooks import PlaybookRegistry
    from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
    from aiops_agent.workers.database_handlers import DatabaseDiagnosticHandler
    from aiops_agent.workers.handlers import TaskExecutionContext
    from aiops_agent.workers.tool_handlers import DbaDiagnosticToolHandler
    from platform_core.contracts.aiops.executor import DiagnosticExecutionGrant, DiagnosticLimits
    from platform_core.contracts.aiops.investigation import InvestigationAction
    from tests.support.aiops_resources import bind_resource_control
    from tests.unit.aiops_agent.test_aiops_oracle_cursor_plan import PHYSICAL

    async def exercise():
        registry, playbooks = DiagnosticRegistry.load(), PlaybookRegistry.load()
        builder = ToolExecutionSnapshotBuilder(playbook_registry=playbooks, diagnostic_registry=registry)
        capability = caps().model_copy(update={"entitlements": tuple(PACKS)})
        inputs = dict(PARAMS)
        if tool_id == MONITOR_PLAN:
            inputs.update(sql_exec_id=99, sql_exec_start_native="2026-10-29T10:00:00")
        elif tool_id == ASH:
            inputs.update(begin_time="2026-10-29T09:45:00+00:00", end_time="2026-10-29T10:00:00+00:00")
        definition = tool(tool_id).definition
        action = InvestigationAction(action_id="a1", tool_id=tool_id, input=inputs, question="核对原生运行证据",
            expected_evidence_kind="ORACLE_SQL_DIAGNOSTIC_FACTS", measurement_semantics=definition.measurement_semantics, optional=True)
        plan = build_playbook_plan(playbooks)
        compiled = InvestigationTaskCompiler(playbooks).compile(plan, investigation_actions=(action,))
        scope = {"tenant_id": str(uuid7()), "domain_id": str(uuid7())} if "tenant_id" in DiagnosticExecutionGrant.model_fields else {"domain_id": 7}
        execution = builder.build(plan=plan, compiled=compiled, capabilities=capability, direct_actions=(action,),
            database_execution={**scope, "target_row_version": 1, "db_type": "ORACLE", "configured_version": "19.3",
                "diagnostic_credential_id": str(uuid7()), "automatic_access_enabled": True,
                "connection_profile": {"host": "db.internal", "port": 1521, "service": "PDB1"}})
        now = datetime.now(UTC)
        context = TaskExecutionContext(run_id=str(uuid7()), task_id=str(uuid7()), task_key="diagnostic:a1",
            target_id=str(uuid7()), agent_id=str(uuid7()), actor_id=str(uuid7()), trigger_type="CHAT", trace_id="oracle-native",
            attempt=1, max_attempts=1, deadline_at=(now + timedelta(seconds=90)).isoformat(),
            lease_until=(now + timedelta(seconds=90)).isoformat(), lease_token="lease-native",
            plan_snapshot={"investigation_execution": execution}, policy_snapshot={}, input_artifacts=())
        codec = DiagnosticGrantCodec(secret="s" * 32, issuer="worker", audience="executor")
        control = AsyncMock()
        control.issue_credential.return_value = SimpleNamespace(username="readonly", password="never-export")
        bind_resource_control(control, codec)

        async def raw(**kwargs):
            assert all(kwargs["parameters"][key] == value for key, value in inputs.items())
            if outcome == "denied":
                raise DiagnosticDriverError("PRIVILEGE_MISSING")
            fields = {**inputs, "cpu_time_us": 123, "sample_count": 7, "plan_line_id": 1,
                "plan_table_output": "Plan hash value: " + str(PARAMS["plan_hash_value"])}
            row = tuple(fields.get(column.name) for column in definition.output_columns)
            count = 0 if outcome == "empty" else kwargs["parameters"].get("limit", 1) if outcome == "boundary" else 1
            return DriverQueryResult(columns=tuple(column.name for column in definition.output_columns),
                rows=(row,) * count, truncated=False, db_version="19.3", instance_identity=tuple(PHYSICAL.items()))

        driver = SimpleNamespace(db_type="ORACLE", execute=AsyncMock(side_effect=raw))
        executor = DiagnosticExecutorService(registry=registry, grant_codec=codec, control_plane=control, drivers=(driver,),
            hard_limits=DiagnosticLimits(statement_timeout_seconds=20, max_result_rows=1000, max_result_bytes=1048576), concurrency=2)
        async def invoke(request, **kwargs):
            return await executor.execute(request)
        database = DatabaseDiagnosticHandler(executor_client=SimpleNamespace(execute_diagnostic=invoke),
            grant_codec=codec, grant_issuer="worker", grant_audience="executor", grant_ttl_seconds=45)
        result = await DbaDiagnosticToolHandler(database_handler=database).execute(context)
        assert driver.execute.await_count == 1, result.model_dump(mode="json")
        if outcome == "denied":
            assert result.tool_outcomes[0].gap.code == "PRIVILEGE_MISSING"
            return
        assert result.status in {"SUCCEEDED", "PARTIAL"}, result.model_dump(mode="json")
        entry = database_evidence_fact(result=result, outcome=result.tool_outcomes[0], artifact_id=uuid7())
        assert entry.tool_id == tool_id and entry.row_count == (0 if outcome == "empty" else count_for(outcome, definition))
        assert entry.truncated == (outcome == "boundary" and tool_id in {MONITOR_PLAN, ASH})
        assert entry.provenance.physical_identity
        assert "never-export" not in result.model_dump_json()

    def count_for(outcome, definition):
        parameter = next((item for item in definition.parameters if item.name == "limit"), None)
        return parameter.default if outcome == "boundary" and parameter else 1

    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["restart", "missing_physical", "history", "missing_cursor"])
def test_runtime_enhancement_never_borrows_unmatched_physical_or_partial_cursor(change):
    entry = runtime(MONITOR, sql_exec_id=99, sql_exec_start_native="2026-10-29T10:00:00", cpu_time_us=100)
    if change == "restart":
        physical = dict(entry.provenance.physical_identity)
        physical["instance_started_at"] = "2026-10-29T09:00:00"
        entry = entry.model_copy(update={"provenance": entry.provenance.model_copy(update={
            "physical_identity": tuple(physical.items())})})
    elif change == "missing_physical":
        entry = entry.model_copy(update={"provenance": None})
    elif change == "history":
        entry = entry.model_copy(update={"source_id": "prior_turn.sql_observation"})
    else:
        index = next(i for i, column in enumerate(entry.columns) if column["name"] == "dbid")
        entry = entry.model_copy(update={"columns": entry.columns[:index] + entry.columns[index + 1:],
            "rows": (entry.rows[0][:index] + entry.rows[0][index + 1:],)})
    bundle = build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE", evidence=(direct(), entry))
    assert not bundle["oracle_runtime"][0]["monitor_executions"]
    assert MONITOR_PLAN not in [action.tool_id for action in actions((direct(), entry))]


def test_retained_monitor_execution_is_not_a_current_session_lock_witness():
    monitor = runtime(MONITOR, session_id=42, session_serial=9, status="DONE (ALL ROWS)")
    lock = runtime("db.sql.blocking", blocked_session_id=42, event="enq: TX - row lock contention")
    keep = ["blocked_session_id", "event"]
    positions = [index for index, column in enumerate(lock.columns) if column["name"] in keep]
    lock = lock.model_copy(update={"columns": tuple(lock.columns[index] for index in positions),
        "rows": (tuple(lock.rows[0][index] for index in positions),)})
    bundle = build_sql_evidence_bundle(task_frame=frame(), database_type="ORACLE",
        evidence=(direct(), monitor, lock))
    assert not bundle["relationships"]


@pytest.mark.parametrize("tool_id", ORACLE_SQL_TOOLS)
@pytest.mark.parametrize("tamper", [False, True])
def test_oracle_facts_remain_replayable_in_report_and_case_source_validation(tool_id, tamper):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
    from aiops_agent.application.reporting import normalize_report_source
    from aiops_agent.playbooks import canonical_hash
    from platform_core.contracts.aiops.executor import DatabaseColumn
    from tests.unit.aiops_agent.test_aiops_oracle_cursor_plan import original_anchor, source_artifact
    from tests.unit.aiops_agent.test_aiops_sql_plan_import import freeze

    run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
    statement, original = original_anchor(run)
    definition = tool(tool_id).definition
    fields = {**PARAMS, "sql_exec_id": 99, "sql_exec_start_native": "2026-10-29T10:00:00",
        "cpu_time_us": 123, "plan_line_id": 1, "sample_count": 7, "plan_table_output": "Predicate Information"}
    columns = tuple(DatabaseColumn(**column.model_dump(exclude={"nullable"})) for column in definition.output_columns)
    obs = observation().model_copy(update={"tool_id": tool_id, "variant": definition.variant,
        "template_sha256": definition.template_sha256, "columns": columns,
        "rows": (tuple(fields.get(column.name) for column in columns),), "row_count": 1})
    entry, raw = source_artifact(obs, run)
    values = (statement, entry)
    record = freeze(values, "ORACLE").model_copy(update={"run_id": str(run.ops_run_id), "target_id": str(run.target_id)})
    payload = {"status": "COMPLETED", "blocks": [], "evidence": [value.model_dump(mode="json") for value in values],
        "investigation_record": record.model_dump(mode="json")}
    normalized = normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", payload=payload, source_kind="DIAGNOSIS")
    assert tool_id in [item["tool_id"] for item in normalized["investigation_record"]["observations"]]
    if tamper:
        raw.payload_json["tool_outcomes"][0]["observation"]["rows"][0][-1] = "tampered"
        raw.content_hash = canonical_hash(raw.payload_json)
    sources = {str(statement.artifact_id): original, str(entry.artifact_id): raw}
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(
        side_effect=lambda *, artifact_id: sources.get(str(artifact_id)))))
    assert asyncio.run(validate_investigation_sources(uow=uow, run=run, payload=payload)) == (not tamper)


def test_diagnostic_handler_persists_typed_ash_request_and_stops_retrying_zero_samples():
    import asyncio
    from unittest.mock import patch
    from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
    from aiops_agent.application.diagnosis.sql_review import evidence_hash, scope_hash
    from aiops_agent.contracts.sql_diagnosis import SqlDiagnosticCheckpoint
    from tests.unit.aiops_agent.test_aiops_sql_review import setup, review

    async def exercise():
        _, task, context, model, prompts, assessment = setup("ORACLE")
        task.update(frame())
        context.plan_snapshot["capability_snapshot"] = caps().model_copy(update={
            "entitlements": tuple(PACKS)}).model_dump(mode="json")
        # 正式评估先尝试轻量增强；ASH 按需申请时它们已成功、零行或失败。
        facts = (direct(), *(runtime(tool_id).model_copy(update={"rows": (), "row_count": 0})
                            for tool_id in (XPLAN, MONITOR)))
        draft = review(facts[0], needs_more=False).model_copy(update={
            "goal_satisfied": False, "verdict": "INSUFFICIENT_EVIDENCE",
            "oracle_evidence_requests": (OracleSqlEvidenceRequest(
                kind="ASH", subject_id=subject_id(), reason="区分CPU与阻塞等待"),)})
        checkpoint = SqlDiagnosticCheckpoint(draft=draft, scope_hash=scope_hash(task, "ORACLE"),
            evidence_hash=evidence_hash(facts))
        assessment = assessment.model_copy(update={"evidence": facts})
        handler = DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)
        with patch("aiops_agent.workers.turn_answer_handlers.generate_sql_review", return_value=checkpoint):
            result = await handler._review_sql(context, assessment, task)
        assert result.investigation.next_action == "REPLAN"
        assert result.sql_checkpoint.draft.oracle_evidence_requests[0].subject_id == subject_id()
        assert any(gap.detail == "区分CPU与阻塞等待" for gap in result.gaps)
        empty = runtime(ASH).model_copy(update={"rows": (), "row_count": 0})
        evidence = (*facts, empty)
        checkpoint = checkpoint.model_copy(update={"evidence_hash": evidence_hash(evidence)})
        with patch("aiops_agent.workers.turn_answer_handlers.generate_sql_review", return_value=checkpoint):
            result = await handler._review_sql(context, assessment.model_copy(update={"evidence": evidence}), task)
        assert result.investigation.next_action == "ANSWER"
    asyncio.run(exercise())
