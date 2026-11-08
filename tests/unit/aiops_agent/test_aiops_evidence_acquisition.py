"""先采集再诊断、原生计划安全边界与真实命令展示回归。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aiops_agent.application.diagnosis.sql_plans import database_cursor_plan
from aiops_agent.application.implementation.diagnostic_runbook import compile_diagnostic_runbook
from aiops_agent.application.implementation.markdown import render_implementation_runbook_markdown
from aiops_agent.application.implementation.presentation import step_groups
from aiops_agent.application.investigation.sql_plan_priority import pending_sql_plan_actions, authorized_plan_tools
from aiops_agent.contracts.turn_answer import TurnEvidenceFact, TurnEvidenceGap
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import canonical_hash
from aiops_agent.domain.evidence.sql_plan_matching import assess_sql_plans
from aiops_agent.executor.drivers.base import DiagnosticDriverError
from aiops_agent.executor.drivers.statement_plan import (
    PLAN_COLUMNS, plan_source, readonly_plan_sql, safe_plan_row, postgresql_statement_plan, mysql_statement_plan,
)
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops.investigation import InvestigationAssessment
from tests.unit.aiops_agent.test_aiops_investigation_record import record
from tests.unit.aiops_agent.test_aiops_oracle_cursor_plan import observation
from tests.unit.aiops_agent.test_aiops_sql_plan_import import anchor, SUBJECTS
from tests.unit.aiops_agent.test_aiops_sql_plan_priority import caps, frame
from tests.unit.aiops_agent.test_aiops_turn_answer import _context, _tool_artifact
from platform_core.contracts.aiops.executor import DatabaseColumn
from platform_core.identity import uuid7


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL"])
def test_plan_priority_is_per_turn_exact_subject_and_never_guesses_sql_or_bind_values(engine):
    entry = anchor(engine)
    tool = "db.postgresql.statement.plan" if engine == "POSTGRESQL" else "db.mysql.statement.plan"
    task = frame(engine, {"subject_type": "POSTGRESQL_QUERY_ID" if engine == "POSTGRESQL" else "MYSQL_DIGEST", **SUBJECTS[engine]})
    def pending(facts=(entry,), **kwargs):
        return pending_sql_plan_actions(task_frame=task, database_type=engine, evidence=facts,
            available_tool_ids=(tool,), **kwargs)
    actions = pending()
    assert len(actions) == 1 and actions[0].tool_id == tool
    assert set(actions[0].input) == {*SUBJECTS[engine], "source_sql_sha256"}
    assert actions[0].measurement_semantics == "NOT_APPLICABLE"
    assert not pending((entry.model_copy(update={"source_id": "prior_turn.sql_observation"}),))
    assert not pending((entry.model_copy(update={"truncated": True}),))
    assert not pending(prior_actions=tuple(action.model_dump(mode="json") for action in actions))
    assert not pending(gaps=(TurnEvidenceGap(source_id=tool, step_id="a1", code="PRIVILEGE_MISSING", detail="无读取权限"),))
    assert not pending((entry, entry.model_copy(update={"tool_id": tool, "rows": (), "row_count": 0})))
    # 新一轮不携带本轮失败/尝试状态，重新采集，不永久缓存“缺证”。
    assert pending()
    capability = caps().model_copy(update={"database_type": engine, "database_version": "16" if engine == "POSTGRESQL" else "8.0"})
    assert tool in authorized_plan_tools(capability.model_dump(mode="json"))


@pytest.mark.parametrize("sql,code", [
    ("SELECT * FROM equipment WHERE id=$1", "SQL_PLAN_PARSE_CONTEXT_REQUIRED"),
    ("DELETE FROM mes.equipment", "SQL_PLAN_STATEMENT_UNSUPPORTED"),
    ("WITH x AS (DELETE FROM mes.equipment RETURNING *) SELECT * FROM x", "SQL_PLAN_STATEMENT_UNSUPPORTED"),
    ("SELECT * FROM mes.equipment FOR UPDATE", "SQL_PLAN_STATEMENT_UNSUPPORTED"),
    ("SELECT pg_sleep(10)", "SQL_PLAN_FUNCTION_UNSUPPORTED"),
    ("SELECT evil.custom_func()", "SQL_PLAN_FUNCTION_UNSUPPORTED"),
    ("SELECT 1; SELECT 2", "SQL_PLAN_STATEMENT_UNSUPPORTED"),
])
def test_plan_capture_does_not_execute_unsafe_source_or_invent_parse_context(sql, code):
    with pytest.raises(DiagnosticDriverError) as error:
        readonly_plan_sql(sql)
    assert error.value.code == code


def test_fixed_catalog_plan_source_rejects_ambiguous_or_changed_sql():
    assert plan_source([], expected_hash="a" * 64) is None
    with pytest.raises(DiagnosticDriverError, match="SQL_PLAN_SUBJECT_AMBIGUOUS"):
        plan_source([{}, {}], expected_hash="a" * 64)
    with pytest.raises(DiagnosticDriverError, match="SQL_PLAN_SOURCE_CHANGED"):
        plan_source([{"source_sql": "SELECT 1"}], expected_hash="a" * 64)


def test_missing_context_is_distinct_from_actual_plan_acquisition_failure_or_zero_rows():
    from aiops_agent.application.diagnosis.sql_plan_presentation import plan_availability_block
    def payload(bundle, gaps=()):
        return plan_availability_block(database_type="POSTGRESQL", bundle=bundle, gaps=gaps).payload["sql_plan_acquisition"]
    assert payload({})["collection_state"] == "NOT_ATTEMPTED"
    failed = payload({}, (TurnEvidenceGap(source_id="db.postgresql.statement.plan", step_id="a1",
        code="PRIVILEGE_MISSING", detail="目标拒绝计划读取权限"),))
    assert failed["collection_state"] == "FAILED" and failed["failure_codes"] == ["PRIVILEGE_MISSING"]
    empty = payload({"observations": [{"tool_id": "db.postgresql.statement.plan", "records": []}]})
    assert empty["collection_state"] == "NO_MATCHING_PLAN" and len(empty["attempts"]) == 1


def test_postgresql_capture_forces_generic_plan_and_does_not_use_analyze_or_original_execute():
    async def exercise():
        sql = "SELECT count(*) FROM mes.equipment WHERE id=$1"
        source = {"query_id": "1285382359565720997", "database_oid": 42, "user_oid": 11, "top_level": True, "source_sql": sql}
        native = [{"Plan": {"Node Type": "Index Scan", "Schema": "mes", "Relation Name": "equipment", "Plan Rows": 1,
            "Index Cond": "secret=$1", "Total Cost": 2}}]
        connection = SimpleNamespace(fetch=AsyncMock(return_value=[source]), execute=AsyncMock(),
            fetchval=AsyncMock(side_effect=[42, 1, json.dumps(native)]))
        parameters = {**{key: source[key] for key in ("query_id", "database_oid", "user_oid", "top_level")},
            "source_sql_sha256": canonical_hash(sql)}
        resolved = next(item for item in DiagnosticRegistry.load().tools if item.definition.tool_id == "db.postgresql.statement.plan")
        rows = await postgresql_statement_plan(connection, source_sql=resolved.sql, parameters=parameters)
        assert len(rows) == 1 and "secret" not in rows[0][2]
        assert connection.fetchval.await_args.args[0] == "EXPLAIN (FORMAT JSON, VERBOSE TRUE) EXECUTE aiops_statement_plan(NULL)"
        commands = [call.args[0] for call in connection.execute.await_args_list]
        assert "SET LOCAL plan_cache_mode = force_generic_plan" in commands
        assert "PREPARE aiops_statement_plan AS " + sql in commands
        assert not any(command.startswith("EXECUTE ") or "ANALYZE" in command for command in commands)
    asyncio.run(exercise())


def test_database_estimate_replays_original_receipt_and_retains_no_runtime_or_predicates():
    entry = anchor("POSTGRESQL")
    sql = next(value for column, value in zip(entry.columns, entry.rows[0], strict=True) if column["name"] == "normalized_statement")
    row = safe_plan_row([{"Plan": {"Node Type": "Seq Scan", "Schema": "mes", "Relation Name": "equipment", "Plan Rows": 10,
        "Filter": "password='secret'"}}], database_type="POSTGRESQL", subject=SUBJECTS["POSTGRESQL"],
        source_hash=canonical_hash(sql))
    tool = next(item for item in DiagnosticRegistry.load().tools if item.definition.tool_id == "db.postgresql.statement.plan")
    columns = tuple(DatabaseColumn(name=name, logical_type="STRING", sensitivity="PUBLIC") for name in PLAN_COLUMNS)
    obs = observation().model_copy(update={"db_type": "POSTGRESQL", "tool_id": tool.definition.tool_id,
        "variant": tool.definition.variant, "columns": columns, "rows": (row,), "row_count": 1})
    document, warnings = database_cursor_plan(obs)
    assert document and document.origin == "DATABASE_ESTIMATE" and not warnings
    assert document.runtime_statistics == "NONE" and "secret" not in document.model_dump_json()
    document = document.model_copy(update={"physical_identity": dict(entry.provenance.physical_identity)})
    identity = str(uuid7())
    fact = TurnEvidenceFact(evidence_ref=f"artifact:{identity}#a1", artifact_id=identity,
        source_id=tool.definition.tool_id, step_id="a1", tool_id=tool.definition.tool_id,
        measurement_semantics="NOT_APPLICABLE", presentation_kind="TABLE", captured_at=document.observed_at,
        columns=tuple(column.model_dump(mode="json") for column in columns), rows=(row,), row_count=1,
        provenance=entry.provenance, sql_plan=document)
    result = assess_sql_plans(database_type="POSTGRESQL", subject=SUBJECTS["POSTGRESQL"], evidence=(entry, fact))[0]
    assert result.identity_match == result.physical_match == "SOURCE_MATCH"
    assert not result.production_execution_verified and not result.bind_metadata_refs
    assert database_cursor_plan(obs.model_copy(update={"truncated": True}))[0] is None
    # 正式模型展开必须定位单行原始 JSON 中的真实节点，不能按节点号猜原始行号。
    from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
    from aiops_agent.application.diagnosis.sql_review import expand_diagnostic_response
    from tests.support.aiops_sql_review import diagnostic_response
    from tests.unit.aiops_agent.test_aiops_sql_diagnosis import diagnosis
    from aiops_agent.contracts.sql_diagnostic_response import SqlPlanReview
    bundle = build_sql_evidence_bundle(database_type="POSTGRESQL", evidence=(entry, fact),
        task_frame=frame("POSTGRESQL", {"subject_type": "POSTGRESQL_QUERY_ID", **SUBJECTS["POSTGRESQL"]}))
    response = diagnostic_response(diagnosis(entry)).model_copy(update={"plan_reviews": (SqlPlanReview(
        evidence_ref=fact.evidence_ref, node_ids=(0,), conclusion_id="c1",
        mechanism="顺序扫描返回估算输入行", impact="需结合统计读取量核对，不把估算当实际耗时"),)})
    expanded = expand_diagnostic_response(response, bundle, {})
    assert any(ref.evidence_ref == fact.evidence_ref and ref.field == "plan_nodes_json" and ref.row_index == 0
        for ref in expanded.conclusions[0].evidence_fields)


@pytest.mark.parametrize("changed", [False, True])
def test_mysql_only_explains_live_exact_event_and_detects_races(changed):
    async def exercise():
        sql = "SELECT * FROM equipment WHERE id = ?"
        fields = (21, 31, 41, "A" * 64, "mes", sql, "a" * 64)
        description = [(key,) for key in ("thread_id", "event_id", "connection_id", "digest", "schema_name", "source_sql", "expected_source_hash")]
        cursor = SimpleNamespace(execute=AsyncMock(), description=description,
            fetchmany=AsyncMock(side_effect=[[fields], [] if changed else [fields]]),
            fetchone=AsyncMock(return_value=(json.dumps({"query_block": {"table": {"table_name": "equipment", "access_type": "ALL", "rows_examined_per_scan": 10}}}),)))
        parameters = {"source_sql_sha256": canonical_hash(sql)}
        if changed:
            with pytest.raises(DiagnosticDriverError, match="SQL_PLAN_EXECUTION_CHANGED"):
                await mysql_statement_plan(cursor, bound_source_sql="SOURCE", parameters=parameters, ensure_active=lambda: None)
        else:
            assert await mysql_statement_plan(cursor, bound_source_sql="SOURCE", parameters=parameters, ensure_active=lambda: None)
        assert cursor.execute.await_args_list[1].args == ("EXPLAIN FORMAT=JSON FOR CONNECTION 41",)
        assert not any(sql == call.args[0] for call in cursor.execute.await_args_list)
    asyncio.run(exercise())


@pytest.mark.parametrize("profile,starter", [("SINGLE_SQL_PERFORMANCE", False), ("NONE", True), ("NONE", False)])
def test_every_diagnostic_entry_checks_acquisition_before_final_answer(profile, starter):
    async def exercise():
        ctx = _context(artifacts=(_tool_artifact(semantics="CURRENT_ACTIVITY"),), task_frame_overrides={"diagnostic_profile": profile, "time_scope": None})
        if starter:
            ctx.plan_snapshot["planning_route"] = {"mode": "CONVERSATION_STARTER"}
        model = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(output=InvestigationAssessment(
            round_no=1, sufficiency_status="PARTIAL", next_action="ANSWER", progress_made=True,
            reason="需要当前执行计划", evidence_gaps=("尚未采集当前计划",)))))
        prompts = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="补证评估", ref=lambda: {})))
        handler = DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)
        handler._review_sql = AsyncMock(side_effect=AssertionError("补证前不能输出最终诊断"))
        if profile == "SINGLE_SQL_PERFORMANCE":
            # 生产 SQL 入口的主体匹配已由独立测试覆盖，此处只隔离验证统一前置评估分支。
            bundle = {"subject_resolution": {"status": "RESOLVED"}, "observations": [{"trust_level": "SOURCE_VERIFIED", "records": [{"role": "SUBJECT"}]}]}
        else:
            bundle = None
        with patch("aiops_agent.application.investigation.sql_plan_priority.authorized_evidence_tools", return_value=({"tool_id": "db.postgresql.statement.plan"},)), \
             patch("aiops_agent.workers.turn_answer_handlers.build_sql_evidence_bundle", return_value=bundle):
            assessment = await handler.execute(ctx)
        assert assessment.investigation.next_action == "REPLAN"
        assert assessment.sql_checkpoint is None
        handler._review_sql.assert_not_awaited()
        call = model.generate_structured.await_args.kwargs
        assert call["purpose"] == "aiops.investigation-assessment" and "on_text_delta" not in call
        assert call["input_payload"]["evidence_acquisition"]["stage"] == "BEFORE_FINAL_ANSWER"
    asyncio.run(exercise())


def test_verification_is_kept_and_notes_never_impersonate_commands():
    value = record()
    runbook = compile_diagnostic_runbook(value, database_type="POSTGRESQL", database_version="16")
    assert runbook.status == "READY"
    step = runbook.phases[0].steps[0]
    assert step.step_id == "diagnosis.r1" and not step.commands and not step.verification_commands and not step.rollback
    assert step.verification_notes and step.rollback_conditions
    text = render_implementation_runbook_markdown(runbook.model_dump(mode="json"))
    assert "验证要点" in text and "停止或回退条件" in text and "```" not in text
    assert compile_diagnostic_runbook(value.model_copy(update={"stop_reason": "PARTIAL_ANSWER"}),
        database_type="POSTGRESQL", database_version="16").status == "PARTIAL_EVIDENCE"


def test_approved_draft_sql_is_copyable_but_manual_verification_is_plain_text():
    draft = {"command_type": "MANUAL", "executor": "PSQL", "content": "EXPLAIN SELECT 1"}
    note = {"command_type": "MANUAL", "executor": "MANUAL", "content": "比较同口径耗时"}
    groups = list(step_groups({"commands": [draft], "verification_commands": [note], "rollback_conditions": ["延迟增加时停止"]}))
    assert any(not manual and draft in entries for _, entries, manual in groups)
    assert any(manual and note in entries for _, entries, manual in groups)
    assert not any(not manual and note in entries for _, entries, manual in groups)


@pytest.mark.parametrize("tamper", [None, "nodes", "document", "hash", "missing"])
def test_native_plan_report_and_case_replay_fixed_tool_originals(tamper):
    from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact
    from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
    from aiops_agent.contracts.tool_execution import DbaToolResult
    from tests.unit.aiops_agent.test_aiops_sql_plan_import import freeze

    run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
    entry = anchor("POSTGRESQL")
    fields = dict(zip((column["name"] for column in entry.columns), entry.rows[0], strict=True))
    row = safe_plan_row([{"Plan": {"Node Type": "Seq Scan", "Plan Rows": 10}}], database_type="POSTGRESQL",
        subject=SUBJECTS["POSTGRESQL"], source_hash=canonical_hash(fields["normalized_statement"]))
    registry = DiagnosticRegistry.load()
    plan_tool = next(item for item in registry.tools if item.definition.tool_id == "db.postgresql.statement.plan")
    base = observation().model_copy(update={"target_id": run.target_id, "db_type": "POSTGRESQL", "db_version": "16",
        "provenance": {"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": dict(entry.provenance.physical_identity)}})
    plan_observation = base.model_copy(update={"tool_id": plan_tool.definition.tool_id, "variant": plan_tool.definition.variant,
        "template_sha256": plan_tool.definition.template_sha256, "columns": tuple(DatabaseColumn(name=n, logical_type="STRING", sensitivity="PUBLIC") for n in PLAN_COLUMNS),
        "rows": (row,), "row_count": 1})
    anchor_observation = base.model_copy(update={"tool_id": entry.tool_id,
        "columns": tuple(DatabaseColumn(**column, logical_type="STRING", sensitivity="PUBLIC") for column in entry.columns),
        "rows": entry.rows, "row_count": entry.row_count})
    def original(obs, semantics):
        result = DbaToolResult.model_validate({"source_type": "TOOL", "source_id": obs.tool_id, "source_version": "1.0.0",
            "definition_hash": "a" * 64, "output_schema": "DBA_TOOL_RESULT.v1", "measurement_semantics": semantics,
            "presentation_kind": "TABLE", "status": "SUCCEEDED", "tool_outcomes": [{"step_id": "a1", "tool_id": obs.tool_id,
                "tool_version": "1.0.0", "status": "SUCCEEDED", "observation": obs.model_dump(mode="json")} ]})
        identity = uuid7()
        fact = database_evidence_fact(result=result, outcome=result.tool_outcomes[0], artifact_id=identity)
        payload = result.model_dump(mode="json")
        artifact = SimpleNamespace(ops_run_id=run.ops_run_id, schema_version="DBA_TOOL_RESULT.v1", trust_level="SOURCE_VERIFIED",
            payload_json=payload, content_hash=canonical_hash(payload))
        return fact, artifact
    statement, statement_source = original(anchor_observation, "CUMULATIVE_SINCE_LOAD")
    plan, plan_source = original(plan_observation, "NOT_APPLICABLE")
    investigation = freeze((statement, plan)).model_copy(update={"run_id": str(run.ops_run_id)})
    assert investigation.sql_plan_assessments[0].origin == "DATABASE_ESTIMATE"
    payload = {"evidence": [fact.model_dump(mode="json") for fact in (statement, plan)],
        "investigation_record": investigation.model_dump(mode="json")}
    sources = {str(statement.artifact_id): statement_source, str(plan.artifact_id): plan_source}
    if tamper == "nodes":
        plan_source.payload_json["tool_outcomes"][0]["observation"]["rows"][0][2] = "[]"
        plan_source.content_hash = canonical_hash(plan_source.payload_json)
    elif tamper == "document":
        payload["evidence"][1]["sql_plan"]["nodes"][0]["operation"] = "Index Scan"
    elif tamper == "hash":
        plan_source.content_hash = "f" * 64
    elif tamper == "missing":
        sources.pop(str(plan.artifact_id))
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(side_effect=lambda *, artifact_id: sources.get(str(artifact_id)))))
    assert asyncio.run(validate_investigation_sources(uow=uow, run=run, payload=payload)) is (tamper is None)
