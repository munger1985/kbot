"""计划优先取证、三库展示及明确诊断交付；不以模型替身代替生产验收。"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.diagnosis.sql_plan_presentation import matching_plan_refs, plan_availability_block, plan_table_block
from aiops_agent.application.diagnosis.sql_evidence_answer import build_sql_evidence_answer
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.investigation.sql_plan_priority import authorized_plan_tools, pending_sql_plan_actions
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceGap
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import ToolExecutionSnapshotBuilder
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_oracle_cursor_plan import PLAN, SUBJECT, PARAMS, direct, fact, observation
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import diagnosis
from tests.unit.aiops_agent.test_aiops_sql_plan_import import SUBJECTS, anchor, upload
from tests.unit.aiops_agent.test_aiops_turn_answer import _context, _tool_artifact


def frame(engine="ORACLE", subject=None):
    return {"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "objectives": ["DIAGNOSE"],
            "problem_statement": "诊断指定 SQL", "success_criteria": ["解释访问路径和性能表现"],
            "subject_ref": subject or {"subject_type": "ORACLE_SQL_ID", "sql_id": SUBJECT["sql_id"]}}


def caps():
    return DbaCapabilitySnapshot(agent_id=str(uuid7()), agent_version_id=str(uuid7()), target_id=str(uuid7()),
        database_type="ORACLE", database_version="19.3", target_enabled=True, target_reachable=True,
        target_capabilities=("DB_READONLY", "dynamic_performance_views"))


def actions(evidence=None, **kwargs):
    return pending_sql_plan_actions(task_frame=kwargs.pop("task_frame", frame()), database_type="ORACLE",
        evidence=(direct(),) if evidence is None else evidence, available_tool_ids=kwargs.pop("available_tool_ids", (PLAN,)), **kwargs)


def test_sql_id_only_binds_plan_from_verified_current_cursor_without_model_or_replay():
    result = actions()
    assert len(result) == 1 and result[0].tool_id == PLAN
    assert result[0].input == PARAMS
    assert result[0].measurement_semantics == "CURRENT_ACTIVITY"
    assert "sql" not in result[0].input
    assert not actions(available_tool_ids=())
    assert authorized_plan_tools(caps().model_dump(mode="json")) == (PLAN,)
    assert not authorized_plan_tools({"database_type": "ORACLE"})
    # DB_READONLY 允许目录工具探测未知视图权限，但没有只读授权时不能规划。
    unknown = caps().model_copy(update={"target_capabilities": ("DB_READONLY",)})
    assert authorized_plan_tools(unknown.model_dump(mode="json")) == (PLAN,)
    denied = caps().model_copy(update={"target_capabilities": ()})
    assert not authorized_plan_tools(denied.model_dump(mode="json"))


@pytest.mark.parametrize("case", ["historical", "untrusted", "missing", "database", "container", "child", "plan", "background"])
def test_priority_never_borrows_or_guesses_other_cursor_identity(case):
    value = direct()
    subject = {"subject_type": "ORACLE_SQL_ID", **SUBJECT}
    if case == "historical":
        value = value.model_copy(update={"source_id": "prior_turn.sql_observation"})
    elif case == "untrusted":
        value = value.model_copy(update={"trust_level": "USER_PROVIDED"})
    elif case == "background":
        value = value.model_copy(update={"followup_run_id": str(uuid7())})
    elif case == "missing":
        value = value.model_copy(update={"columns": value.columns[:-1], "rows": (value.rows[0][:-1],)})
        # 显式删掉身份，不能用请求中的维度去填充数据源。
        index = next(i for i, column in enumerate(value.columns) if column["name"] == "dbid")
        value = value.model_copy(update={"columns": value.columns[:index] + value.columns[index + 1:],
            "rows": (value.rows[0][:index] + value.rows[0][index + 1:],)})
    else:
        key = {"database": "dbid", "container": "container_id", "child": "child_number", "plan": "plan_hash_value"}[case]
        subject[key] = "999"
    assert not actions((value,), task_frame=frame(subject=subject))


def test_zero_failed_or_already_attempted_plans_do_not_create_infinite_replans():
    empty = fact(observation(rows=(), row_count=0))
    assert not actions((direct(), empty))
    gap = TurnEvidenceGap(source_id=PLAN, step_id="a1", code="PRIVILEGE_MISSING", detail="无计划视图权限")
    assert not actions(gaps=(gap,))
    assert not actions(prior_actions=({"tool_id": PLAN, "input": PARAMS},))
    assert not actions((direct(), fact(observation())))


def test_automatic_cursor_coverage_is_bounded_without_aggregating_children():
    base = direct()
    names = [column["name"] for column in base.columns]
    rows = []
    for child in range(8):
        row = list(base.rows[0])
        row[names.index("child_number")] = str(child)
        rows.append(tuple(row))
    bounded = actions((base.model_copy(update={"rows": tuple(rows), "row_count": 8}),))
    assert len(bounded) == 4
    assert [action.input["child_number"] for action in bounded] == [0, 1, 2, 3]


def test_deterministic_plan_continuation_still_uses_formal_compile_and_persist():
    async def exercise():
        registry = PlaybookRegistry.load()
        builder = ToolExecutionSnapshotBuilder(playbook_registry=registry, diagnostic_registry=DiagnosticRegistry.load())
        service = object.__new__(TurnPlanningService)
        service._playbook_registry = registry
        service._tool_snapshot_builder = builder
        service._agent_catalog = SimpleNamespace(resolve_planner_model=AsyncMock(return_value={}), resolve_diagnosis_model=AsyncMock(return_value={}))
        service._investigation_reasoner = SimpleNamespace(replan=AsyncMock(side_effect=AssertionError("机械续取不应调用模型")))
        inputs = SimpleNamespace(input_envelope=None, task_frame=frame(), prior_plan={"actions": []},
            assessment={"evidence": [direct().model_dump(mode="json")], "gaps": []})
        service._load_replan_inputs = AsyncMock(return_value=inputs)
        service._prepare_query_inputs = Mock(side_effect=lambda **kw: (kw["investigation"], (), (), ()))
        service._compile_and_persist_replan = AsyncMock(return_value={"status": "COLLECTING"})
        context = SimpleNamespace(workflow_kind="CHAT", tenant_id=None, agent_id=uuid7(), domain_id=uuid7(),
            trace_id="priority", capabilities=caps(), question="诊断 SQL")
        result = await service._execute_replan_once({"assessment_artifact_id": str(uuid7())}, context=context, revision_no=2)
        assert result["status"] == "COLLECTING"
        call = service._compile_and_persist_replan.await_args.kwargs
        assert call["planning_receipt"] is None
        assert call["investigation"].plan.actions[0].input == PARAMS
        assert call["investigation"].plan.revision_no == 2
        service._prepare_query_inputs.assert_called_once()
        service._investigation_reasoner.replan.assert_not_awaited()
    asyncio.run(exercise())


def test_assessment_schedules_plan_before_llm_but_respects_answer_reserve():
    async def exercise():
        entry = direct()
        artifact = _tool_artifact(semantics="CUMULATIVE_SINCE_LOAD", tool_id="db.sql.cursor_details")
        raw = artifact["payload"]["tool_outcomes"][0]["observation"]
        raw.update(columns=[{**column, "logical_type": "STRING", "sensitivity": "PUBLIC"} for column in entry.columns],
                   rows=[list(entry.rows[0])], row_count=1)
        ctx = _context(artifacts=(artifact,), task_frame_overrides=frame())
        ctx.plan_snapshot["capability_snapshot"] = caps().model_dump(mode="json")
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=AssertionError("先补缓存计划，不需要模型评估")))
        handler = DbaEvidenceAssessmentHandler(model_client=model, prompts=SimpleNamespace())
        assessed = await handler.execute(ctx)
        assert assessed.investigation.next_action == "REPLAN"
        model.generate_structured.assert_not_awaited()
        ctx = replace(ctx, deadline_at=(datetime.now(UTC) + timedelta(seconds=1)).isoformat())
        exhausted = await handler.execute(ctx)
        assert exhausted.investigation is None
    asyncio.run(exercise())


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
def test_three_engines_display_plan_tree_native_units_and_safe_acquisition(engine):
    _, value = upload(engine)
    block = plan_table_block(value)
    assert block.payload["presentation_role"] == "PRIMARY"
    assert block.payload["result_kind"] == "SQL_EXECUTION_PLAN"
    assert not block.payload["has_row_actions"]
    assert any("└─" in row[1] for row in block.payload["rows"])
    assert block.payload["subject"] == SUBJECTS[engine]
    assert block.payload["runtime_statistics"] == value.sql_plan.runtime_statistics
    task = frame(engine, {"subject_type": {"ORACLE": "ORACLE_SQL_ID", "MYSQL": "MYSQL_DIGEST", "POSTGRESQL": "POSTGRESQL_QUERY_ID"}[engine], **SUBJECTS[engine]})
    bundle = build_sql_evidence_bundle(task_frame=task, database_type=engine, evidence=(anchor(engine), value))
    assert value.evidence_ref in matching_plan_refs(bundle)
    status = plan_availability_block(database_type=engine, bundle=bundle)
    assert status.payload["sql_plan_acquisition"]["status"] == "AVAILABLE"
    assert status.payload["sql_plan_acquisition"]["original_execution_authorized"] is False
    missing = plan_availability_block(database_type=engine, bundle={"plan_assessments": []})
    assert missing.payload["sql_plan_acquisition"]["status"] == "MISSING"
    assert "补齐执行计划" in missing.payload["markdown"]


@pytest.mark.parametrize("dimension", ["identity_match", "physical_match"])
@pytest.mark.parametrize("state", ["MISMATCH", "INCOMPLETE"])
def test_mismatched_plan_is_not_presented_as_the_target_plan(dimension, state):
    item = {"evidence_ref": "plan:1", "identity_match": "SOURCE_MATCH", "physical_match": "SOURCE_MATCH"}
    item[dimension] = state
    assert not matching_plan_refs({"plan_assessments": [item]})
    assert plan_availability_block(database_type="ORACLE", bundle={"plan_assessments": [item]}).payload["sql_plan_acquisition"]["status"] == "MISSING"


def test_no_anomaly_conclusion_and_plan_survive_both_answer_modes():
    async def exercise():
        entry, plan = direct(), fact(observation())
        original = diagnosis(entry)
        draft = original.model_copy(update={
            "summary": "在已观察范围未发现明显异常，暂不建议变更。", "verdict": "NO_OBVIOUS_ISSUE",
            "conclusions": (original.conclusions[0].model_copy(update={"statement": "已有匹配计划未显示明显异常。"}),),
            "recommendations": (original.recommendations[0].model_copy(update={
                "intent": "OBSERVE", "action": "暂不建议变更，保持同口径观测。", "requires_approval": False,
                "preconditions": (), "risks": (), "rollback_or_stop": "",
            }),),
        })
        model = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(output=draft, receipt=SimpleNamespace(model_dump=lambda **_: {}))))
        prompts = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="按证据诊断", ref=lambda: {})))
        ctx = _context(task_frame_overrides=frame(subject={"subject_type": "ORACLE_SQL_ID", **SUBJECT}))
        assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(entry, plan))
        ctx = replace(ctx, target_id=plan.provenance.target_id,
            input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},))
        handler = DbaAnswerComposeHandler(model_client=model, prompts=prompts)
        result = await handler.execute(ctx)
        assert result.status == "COMPLETED"
        assert result.blocks[0].payload["markdown"].startswith("结论：在已观察范围未发现明显异常")
        shown = next(block for block in result.blocks if block.payload.get("result_kind") == "SQL_EXECUTION_PLAN")
        assert shown.payload["presentation_role"] == "PRIMARY"
        assert shown.evidence_refs == (plan.evidence_ref,)
        streamed = [item async for item in handler.execute_stream(ctx)]
        assert streamed[-1].blocks == result.blocks
        assert "暂不建议变更" in "".join(item.payload["delta"] for item in streamed[:-1])
        assert "execution_plan" in model.generate_structured.await_args.kwargs["input_payload"]["delivery_requirements"]
    asyncio.run(exercise())


@pytest.mark.parametrize("engine,total,count,expected", [("ORACLE", "1.25", 20, "62.5"), ("POSTGRESQL", 2350, 100, "23.5"), ("MYSQL", 2500000000, 12, "0.208")])
def test_fallback_derives_native_average_without_inventing_anomaly_or_plan(engine, total, count, expected):
    value = anchor(engine)
    names = list(value.columns)
    fields = dict(zip((item["name"] for item in names), value.rows[0], strict=True))
    count_key, time_key = {"ORACLE": ("executions", "elapsed_seconds"), "POSTGRESQL": ("execution_count", "total_exec_time_ms"), "MYSQL": ("execution_count", "total_latency_picoseconds")}[engine]
    fields.update({count_key: count, time_key: total})
    value = value.model_copy(update={"columns": tuple({"name": key} for key in fields), "rows": (tuple(fields.values()),)})
    task = frame(engine, {"subject_type": {"ORACLE": "ORACLE_SQL_ID", "MYSQL": "MYSQL_DIGEST", "POSTGRESQL": "POSTGRESQL_QUERY_ID"}[engine], **SUBJECTS[engine]})
    result = build_sql_evidence_answer(task_frame=task, database_type=engine, evidence=(value,), fallback_reason="诊断生成未完成。")
    assert f"平均约 {expected} 毫秒／次" in result.analysis
    assert "不能仅凭均值认定故障或健康" in result.analysis
    assert "诊断生成未完成" in result.analysis

@pytest.mark.parametrize("sql_task", [False, True])
def test_replanner_projects_sql_evidence_once_without_restricting_open_tools(sql_task):
    from aiops_agent.application.investigation.reasoner import InvestigationReasoner
    from aiops_agent.application.investigation.selected_sql import selected_sql_investigation
    async def exercise():
        task = frame()
        initial = selected_sql_investigation(question="分析 SQL",
            target_context={"db_type": "ORACLE", "sql_selection": {"subject_ref": task["subject_ref"]}},
            tools=({"tool_id": "db.sql.cursor_details"},))
        output = initial.model_copy(update={"plan": initial.plan.model_copy(update={"revision_no": 2})})
        model = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(output=output, receipt=None)))
        prompts = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="规划", ref=lambda: {})))
        reasoner = InvestigationReasoner(model=model, prompts=prompts)
        assessment = {"status": "PARTIAL", "evidence": [direct().model_dump(mode="json")], "gaps": [], "reasons": ["继续取证"]}
        cards = ({"tool_id": "db.sql.cursor_details"}, {"tool_id": "db.oracle.readonly_query",
            "policy": {"allowed_objects": ["V$SESSION"], "requires_readonly": True}})
        await reasoner.replan(content=(), conversation_context=(), target_context={"database_type": "ORACLE"},
            prompt_snapshot={}, source_run_evidence=None, task_frame=task if sql_task else {},
            prior_plan={"actions": []}, assessment=assessment, available_tools=cards, available_playbooks=(),
            model_snapshot={}, deadline=None, idempotency_key="dedupe", revision_no=2)
        payload = model.generate_structured.await_args.kwargs["input_payload"]
        assert payload["available_tools"] == list(cards)
        if sql_task:
            assert "evidence" not in payload["assessment"]
            row = payload["sql_evidence_bundle"]["observations"][0]["records"][0]
            assert row["row_index"] == 0 and row["fields"]["sql_id"] == SUBJECT["sql_id"]
            assert payload["assessment"]["evidence_location"] == "sql_evidence_bundle.observations"
        else:
            assert payload["assessment"] == assessment and payload["sql_evidence_bundle"] is None
    asyncio.run(exercise())

def test_model_timeout_keeps_acquired_plan_and_does_not_request_the_same_plan_again():
    from aiops_agent.adapters.model_serving import AIOpsModelError
    async def exercise():
        entry, plan = direct(), fact(observation())
        ctx = _context(task_frame_overrides=frame(subject={"subject_type": "ORACLE_SQL_ID", **SUBJECT}))
        assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(entry, plan))
        ctx = replace(ctx, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},))
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=AIOpsModelError("MODEL_DEADLINE_EXCEEDED")))
        prompts = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="诊断", ref=lambda: {})))
        result = await DbaAnswerComposeHandler(model_client=model, prompts=prompts).execute(ctx)
        assert result.status == "PARTIAL"
        assert "TABLE ACCESS FULL" in result.blocks[0].payload["markdown"]
        assert "MODEL_DEADLINE_EXCEEDED" in result.blocks[0].payload["markdown"]
        solution = next(block for block in result.blocks if str(block.block_type) == "SOLUTION_MARKDOWN")
        assert "不需要重复采集同一材料" in solution.payload["markdown"]
        assert any(block.payload.get("result_kind") == "SQL_EXECUTION_PLAN" for block in result.blocks)
    asyncio.run(exercise())


def test_hard_deadline_keeps_only_revalidated_matching_plan_visible_without_new_work():
    from aiops_agent.application.diagnosis.deadline_result import deadline_result
    entry, plan = direct(), fact(observation())
    assessment_id = str(uuid7())
    source = SimpleNamespace(artifact_id=assessment_id, trust_level="SOURCE_VERIFIED",
        schema_version="DBA_SUFFICIENCY.v1", payload_json=DbaSufficiencyAssessment(status="PARTIAL", evidence=(entry, plan)).model_dump(mode="json"))
    args = dict(artifacts=(source,), task_frame=frame(subject={"subject_type": "ORACLE_SQL_ID", **SUBJECT}),
        database_type="ORACLE", projected_ids={assessment_id, str(entry.artifact_id), str(plan.artifact_id)})
    assert deadline_result(**args) is None
    result = deadline_result(**args, validated_assessment_ids=frozenset({assessment_id}))
    assert result.status == "PARTIAL"
    assert "TABLE ACCESS FULL" in result.blocks[0].payload["markdown"]
    assert any(block.payload.get("result_kind") == "SQL_EXECUTION_PLAN" for block in result.blocks)
    assert "RUN_DEADLINE_EXCEEDED" in {gap.code for gap in result.evidence_gaps}
