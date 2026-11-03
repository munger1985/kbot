"""三类数据库只列问题 SQL、完整空态及逐条 Agent 分析回归。"""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.inspections.assessment import assess_inspection
from aiops_agent.application.inspections.check_catalog import compile_selected_check_steps
from aiops_agent.application.inspections.sql_observations import format_duration_ms
from aiops_agent.application.inspections.sql_problems import (
    PROBLEM_SQL_COLUMNS, PROBLEM_SQL_TOOL, classify_problem_sql, problem_sql_policy,
)
from aiops_agent.application.inspections.sql_reviews import review_inspection_sql
from aiops_agent.application.reporting import report_presentation, render_pdf, resolve_system_template
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from platform_core.contracts.aiops.public import DailyInspectionPresentation
from platform_core.identity import uuid7


DATABASES = ("MYSQL", "ORACLE", "POSTGRESQL")
CHECKS = {"MYSQL": "mysql.sql.top", "ORACLE": "oracle.sql.problems", "POSTGRESQL": "postgresql.sql.problems"}


def observation(index=0, **overrides):
    row = {
        "sql_key": f"statement-{index}", "schema_name": "业务库", "sql_text": f"SELECT field_{index} FROM orders",
        "text_kind": "NORMALIZED", "execution_count": 490072, "total_elapsed_ms": 324033.9,
        "average_elapsed_ms": 0.7, "maximum_elapsed_ms": 92,
        "average_cpu_ms": None, "average_io_ms": None, "average_lock_ms": None,
        "average_reads": None, "average_rows_examined": None, "average_spills": None,
        "workarea_total_executions": None, "workarea_multipass_executions": None,
        "workarea_multipass_percent": None, "maximum_workarea_temp_mb": None,
        "collection_status": "AVAILABLE", **overrides,
    }
    if isinstance(row["average_elapsed_ms"], (int, float)) and row["average_elapsed_ms"] > 0:
        if "total_elapsed_ms" not in overrides and row["execution_count"] is not None:
            row["total_elapsed_ms"] = row["average_elapsed_ms"] * row["execution_count"]
        if "maximum_elapsed_ms" not in overrides:
            row["maximum_elapsed_ms"] = max(92, row["average_elapsed_ms"] * 2)
    return row


def pipeline(database_type="MYSQL", rows=(), *, truncated=False, reviews=()):
    steps = compile_selected_check_steps([CHECKS[database_type]], database_type=database_type)
    artifact_id = str(uuid7())
    raw_rows = rows or (dict.fromkeys(PROBLEM_SQL_COLUMNS) | {"collection_status": "AVAILABLE"},)
    evidence = TurnEvidenceFact(
        evidence_ref=f"artifact:{artifact_id}#sql", artifact_id=artifact_id, source_id="database.diagnostic",
        step_id="sql", tool_id=PROBLEM_SQL_TOOL, measurement_semantics="CUMULATIVE_SINCE_LOAD",
        presentation_kind="TABLE", captured_at="2026-11-03T10:00:00Z",
        columns=tuple({"name": name} for name in PROBLEM_SQL_COLUMNS),
        rows=tuple(tuple(row.get(name) for name in PROBLEM_SQL_COLUMNS) for row in raw_rows),
        row_count=len(raw_rows), truncated=truncated,
    )
    source = SimpleNamespace(evidence=(evidence,), evidence_gaps=())
    facts, gaps, _ = AIOpsRuntimeService._inspection_report_projection(
        inspection={"evidence_steps": steps, "period_end": "2026-11-03T10:00:00Z"},
        source=source, database_type=database_type, action_tool_ids={},
    )
    if reviews:
        facts = (*facts, {"kind": "inspection_sql_reviews", "reviews": list(reviews)})
    assessment = assess_inspection(facts, policy={})
    payload = {"report_type": "INSPECTION_DAILY", "title": "数据库日常巡检报告",
        "period_start": "2026-11-02T10:00:00Z", "period_end": "2026-11-03T10:00:00Z",
        "scope": {"inspection_assessment": assessment, "database_type": database_type},
        "facts": list(facts), "gaps": list(gaps)}
    original = deepcopy(payload)
    presentation = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
    assert payload == original
    DailyInspectionPresentation.model_validate(presentation["daily_inspection"])
    return steps, source, facts, assessment, presentation


@pytest.mark.parametrize("database_type", DATABASES)
def test_fast_frequent_sql_is_not_a_problem(database_type):
    *_, assessment, presentation = pipeline(database_type, [observation()])
    assert assessment["health_status"] == "NORMAL"
    group = presentation["daily_inspection"]["sql_details"][0]
    assert group["statements"] == []
    assert group["summary"] == "当前没有问题 SQL"


@pytest.mark.parametrize("database_type", DATABASES)
def test_no_problem_is_a_complete_explicit_empty_state(database_type):
    *_, assessment, presentation = pipeline(database_type)
    assert assessment["completion_status"] == "COMPLETE"
    assert presentation["daily_inspection"]["sql_details"][0]["summary"] == "当前没有问题 SQL"
    assert render_pdf(presentation).startswith(b"%PDF-")


@pytest.mark.parametrize("rule", problem_sql_policy()["rules"], ids=lambda rule: rule["code"])
def test_only_actual_per_execution_breach_enters_problem_list(rule):
    value = observation(average_elapsed_ms=10)
    value[rule["metric"]] = rule["limit"]
    database_type = rule.get("database_types", ["MYSQL"])[0]
    problem = classify_problem_sql(value, policy=problem_sql_policy(), database_type=database_type)
    assert problem is not None
    assert rule["code"] in {signal["code"] for signal in problem["signals"]}
    value[rule["metric"]] = rule["limit"] - 0.001
    assert classify_problem_sql(value, policy=problem_sql_policy(), database_type=database_type) is None


@pytest.mark.parametrize("invalid", [None, "", True, -1, "NaN", "Infinity"])
def test_invalid_resource_metrics_are_not_invented(invalid):
    assert classify_problem_sql(observation(average_cpu_ms=invalid),
        policy=problem_sql_policy(), database_type="MYSQL") is None


@pytest.mark.parametrize("database_type", DATABASES)
def test_all_problems_are_preserved_and_severity_controls_order(database_type):
    rows = [observation(i, average_elapsed_ms=1100 + i * 1000) for i in range(9)]
    _, _, facts, assessment, presentation = pipeline(database_type, rows)
    group = presentation["daily_inspection"]["sql_details"][0]
    assert len(group["statements"]) == 9
    assert len(assessment["findings"]) == 9
    assert group["statements"][0]["severity"] == "HIGH"
    assert group["statements"][-1]["severity"] == "MEDIUM"
    assert group["statements"][0]["statement_text"] == rows[-1]["sql_text"]
    assert group["shown_count"] == group["observed_count"] == 9
    assert len(facts[0]["observations"]) == 9
    assert render_pdf(presentation).startswith(b"%PDF-")


@pytest.mark.parametrize("database_type", DATABASES)
@pytest.mark.parametrize("truncated,collection", [(True, "AVAILABLE"), (False, "DISABLED")])
def test_partial_or_disabled_collection_never_claims_no_problem(database_type, truncated, collection):
    *_, presentation = pipeline(database_type, [observation(sql_key=None, collection_status=collection)], truncated=truncated)
    group = presentation["daily_inspection"]["sql_details"][0]
    assert group["status"] != "COMPLETE"
    assert group["summary"] != "当前没有问题 SQL"
    assert "不能确认" in group["summary"]


@pytest.mark.parametrize("value,expected", [(0.7, "0.7 ms"), (92, "92.0 ms"), (324033.9, "5.4 min"),
    (1000, "1.0 s"), (60000, "1.0 min"), (3600000, "1.0 h"), (None, "未记录")])
def test_latency_units_adapt_without_changing_source(value, expected):
    assert format_duration_ms(value) == expected


@pytest.mark.parametrize("database_type,version", [("MYSQL","8.4"),("ORACLE","19c"),
    ("POSTGRESQL","12"),("POSTGRESQL","13"),("POSTGRESQL","16"),("POSTGRESQL","17"),("POSTGRESQL","18")])
def test_source_query_filters_problems_before_any_limit(database_type, version):
    tool = DiagnosticRegistry.load().resolve(tool_id=PROBLEM_SQL_TOOL, tool_version="1.0.0",
        db_type=database_type, db_version=version,
        capabilities={"statement_digest","dynamic_performance_views","pg_stat_statements"}, entitlements=set())
    import sqlglot
    from sqlglot import exp
    tree = sqlglot.parse_one(tool.sql, read={"MYSQL":"mysql","ORACLE":"oracle","POSTGRESQL":"postgres"}[database_type])
    assert tree.find(exp.Limit) is None
    assert "qualified" in tool.sql and "average_elapsed_ms_limit" in tool.sql
    assert "total_elapsed_ms >=" not in tool.sql
    assert "execution_count >=" not in tool.sql
    assert set(PROBLEM_SQL_COLUMNS) == {column.name for column in tool.definition.output_columns}


@pytest.mark.parametrize("database_type", DATABASES)
def test_agent_analyzes_every_problem_and_preserves_per_sql_identity(database_type):
    async def run():
        rows = [observation(i, average_elapsed_ms=1100+i*1000) for i in range(8)]
        steps, source, facts, _, _ = pipeline(database_type, rows)
        async def generate(**arguments):
            problem = arguments["input_payload"]["problem"]
            return SimpleNamespace(output={
                "problem_id": problem["problem_id"], "analysis": "单次执行耗时偏高，应结合访问路径复核。",
                "possible_causes": [{"possible_cause": "可能存在代价较大的扫描或连接。",
                    "supporting_metrics": ["average_elapsed_ms"], "verification": "取得执行计划并复核。"}],
                "recommendations": [{"action": "核对过滤条件和现有访问路径。", "rationale": "减少不必要的处理。",
                    "validation": "对比相同业务参数的耗时和扫描量。", "requires_approval": False}],
                "limitations": ["未取得执行计划，不能确认根因。"], "evidence_refs": problem["evidence_refs"],
            }, receipt=SimpleNamespace(model_dump=lambda **_: {"model_id": "configured-agent-model"}))
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate))
        prompt = SimpleNamespace(content="逐条分析", ref=lambda: {"prompt_id":"inspection_sql_review"})
        prompts = SimpleNamespace(resolve=AsyncMock(return_value=prompt))
        context = SimpleNamespace(trigger_type="SCHEDULE", run_id=uuid7(), attempt=1,
            deadline_at=(datetime.now(UTC)+timedelta(seconds=60)).isoformat(),
            plan_snapshot={"target":{"db_type":database_type},"client_metadata":{"inspection":{"evidence_steps":steps}},
                "answer_context":{"model":{"model_id":"configured-agent-model"},"prompts":{"inspection_sql_review":{}}}})
        reviews = await review_inspection_sql(context=context, result=source, model=model, prompts=prompts, concurrency=2)
        assert len(reviews) == model.generate_structured.await_count == 8
        assert all(review["status"] == "COMPLETE" for review in reviews)
        assert [review["problem_id"] for review in reviews] == [problem["problem_id"] for problem in facts[0]["sql_problems"]]
        *_, presentation = pipeline(database_type, rows, reviews=reviews)
        assert all(item["agent_review"]["status"] == "COMPLETE"
            for item in presentation["daily_inspection"]["sql_details"][0]["statements"])
        assert render_pdf(presentation).startswith(b"%PDF-")
    asyncio.run(run())


def test_standard_steps_keep_query_inputs_and_rules_from_the_same_policy():
    from aiops_agent.application.inspections.assessment import compile_standard_steps, freeze_inspection_policy
    from aiops_agent.application.inspections.sql_problems import problem_sql_inputs
    step = compile_selected_check_steps(["mysql.sql.top"],database_type="MYSQL")[0]
    step["input"]["average_elapsed_ms_limit"] = 999999
    steps = compile_standard_steps((step,),database_type="MYSQL",schedule_type="DAILY",
        policy=freeze_inspection_policy("MYSQL"))
    frozen = next(item for item in steps if item["tool_id"] == PROBLEM_SQL_TOOL)
    assert frozen["input"] == problem_sql_inputs(frozen["sql_problem_policy"])
    assert frozen["measurement_semantics"] == "CUMULATIVE_SINCE_LOAD"
    obsolete = {**step, "tool_id": "db.mysql.statement.digest.top"}
    current = compile_standard_steps((obsolete,),database_type="MYSQL",schedule_type="DAILY",
        policy=freeze_inspection_policy("MYSQL"))
    assert PROBLEM_SQL_TOOL in {item["tool_id"] for item in current}
    assert "db.mysql.statement.digest.top" not in {item["tool_id"] for item in current}


def test_scheduled_empty_result_does_not_call_the_model():
    from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
    async def run():
        steps,source,_,_,_ = pipeline()
        model = SimpleNamespace(generate_structured=AsyncMock())
        result = await DbaAnswerComposeHandler(model_client=model,prompts=None).execute(review_context(steps,source))
        assert not result.inspection_sql_reviews
        model.generate_structured.assert_not_awaited()
        assert result.status == "COMPLETED"
    asyncio.run(run())


def test_prompt_integrity_failure_keeps_each_problem_without_calling_model():
    from platform_core.prompts import PromptIntegrityError
    async def run():
        steps,source,_,_,_ = pipeline(rows=[observation(average_elapsed_ms=2000)])
        model = SimpleNamespace(generate_structured=AsyncMock())
        reviews = await review_inspection_sql(context=review_context(steps,source),result=source,model=model,
            prompts=SimpleNamespace(resolve=AsyncMock(side_effect=PromptIntegrityError("冻结 Prompt 不匹配"))),concurrency=1)
        assert reviews[0]["status"] == "UNAVAILABLE"
        model.generate_structured.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("database_type", DATABASES)
@pytest.mark.parametrize("invalid", [{"execution_count": None}, {"average_elapsed_ms": None}])
def test_malformed_candidates_cannot_claim_current_no_problem(database_type, invalid):
    *_, presentation = pipeline(database_type, [observation(**invalid)])
    assert presentation["daily_inspection"]["sql_details"][0]["status"] == "PARTIAL"


def test_mismatched_query_thresholds_cannot_claim_a_complete_result():
    from aiops_agent.application.inspections.sql_problems import freeze_problem_sql_check
    steps, _, facts, _, _ = pipeline()
    step = deepcopy(steps[0])
    step["input"]["average_elapsed_ms_limit"] *= 10
    frozen = freeze_problem_sql_check(dict(facts[0]), step=step, database_type="MYSQL")
    assert frozen["sql_problem_status"] == "PARTIAL"
    assert frozen["summary"] != "当前没有问题 SQL"


def test_multiple_signals_are_one_problem_and_use_database_specific_units():
    row = observation(average_elapsed_ms=1500, average_cpu_ms=300, average_spills=2)
    for database_type, unit in (("MYSQL", "次/次"), ("POSTGRESQL", "块/次")):
        problem = classify_problem_sql(row, policy=problem_sql_policy(), database_type=database_type)
        assert len(problem["signals"]) == 3
        assert next(signal["unit"] for signal in problem["signals"] if signal["code"] == "DISK_SPILL") == unit


@pytest.mark.parametrize("database_type,version", [("MYSQL","8.4"),("ORACLE","19c"),
    ("POSTGRESQL","12"),("POSTGRESQL","13"),("POSTGRESQL","16"),("POSTGRESQL","17"),("POSTGRESQL","18")])
def test_actual_source_predicate_keeps_all_problem_rows_and_rejects_fast_frequent_rows(database_type, version):
    import sqlite3
    import sqlglot
    from sqlglot import exp
    from aiops_agent.application.inspections.sql_problems import problem_sql_inputs
    tool = DiagnosticRegistry.load().resolve(tool_id=PROBLEM_SQL_TOOL, tool_version="1.0.0",
        db_type=database_type, db_version=version,
        capabilities={"statement_digest","dynamic_performance_views","pg_stat_statements"}, entitlements=set())
    tree = sqlglot.parse_one(tool.sql, read={"MYSQL":"mysql","ORACLE":"oracle","POSTGRESQL":"postgres"}[database_type])
    qualified = next(cte.this for cte in tree.find_all(exp.CTE) if cte.alias_or_name == "qualified")
    columns = PROBLEM_SQL_COLUMNS[:-1]
    rows = [observation(0)] + [observation(i + 1, average_elapsed_ms=1500+i*1000) for i in range(12)]
    rows += [observation(20, average_cpu_ms=200), observation(21, average_io_ms=100),
             observation(22, average_lock_ms=100), observation(23, average_elapsed_ms=10, average_reads=1000),
             observation(24, average_elapsed_ms=10, average_rows_examined=10000),
             observation(25, average_elapsed_ms=10, average_spills=1)]
    if database_type == "ORACLE":
        rows += [observation(26, average_elapsed_ms=10, workarea_multipass_percent=10),
                 observation(27, average_elapsed_ms=10, maximum_workarea_temp_mb=64)]
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE metrics (" + ", ".join(columns) + ")")
        connection.executemany("INSERT INTO metrics VALUES (" + ", ".join("?" for _ in columns) + ")",
            [tuple(row[column] for column in columns) for row in rows])
        actual = {row[0] for row in connection.execute(qualified.sql(dialect="sqlite"), problem_sql_inputs(problem_sql_policy()))}
    expected = {row["sql_key"] for row in rows if classify_problem_sql(row, policy=problem_sql_policy(), database_type=database_type)}
    assert actual == expected
    assert "statement-0" not in actual and len(actual) == (20 if database_type == "ORACLE" else 18)


def review_context(steps, source, *, deadline=None):
    from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
    return SimpleNamespace(trigger_type="SCHEDULE", run_id=uuid7(), target_id=uuid7(), attempt=1,
        deadline_at=deadline, original_request="日常巡检",
        input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload":
            DbaSufficiencyAssessment(status="ANSWERABLE", evidence=source.evidence).model_dump(mode="json")},),
        plan_snapshot={"target":{"db_type":"MYSQL"}, "client_metadata":{"inspection":{
            "template_id":"database_daily", "template_version":"1.0.0",
            "evidence_steps":steps, "schedule_type":"DAILY", "timezone":"Asia/Shanghai",
            "period_start":"2026-11-02T10:00:00Z", "period_end":"2026-11-03T10:00:00Z"}},
            "answer_context":{"model":{"model_id":"agent-selected-model"},"prompts":{"inspection_sql_review":{}}}})


def review_output(problem):
    return {
        "problem_id":problem["problem_id"], "analysis":"单次耗时偏高，需复核访问路径。",
        "possible_causes":[{"possible_cause":"可能存在代价较高的扫描。",
            "supporting_metrics":["average_elapsed_ms"], "verification":"取得真实执行计划复核。"}],
        "recommendations":[{"action":"核对访问路径。", "rationale":"减少不必要处理。",
            "validation":"相同参数对比优化前后耗时。", "requires_approval":False}],
        "limitations":["未取得执行计划，不能确认根因。"], "evidence_refs":problem["evidence_refs"],
    }


@pytest.mark.parametrize("failure", ["wrong_id", "wrong_ref", "missing_metric", "non_metric", "empty_analysis"])
def test_each_model_response_is_identity_and_evidence_checked_without_losing_other_reviews(failure):
    async def run():
        steps, source, _, _, _ = pipeline(rows=[observation(i, average_elapsed_ms=2000) for i in range(2)])
        calls = 0
        async def generate(**arguments):
            nonlocal calls
            output = review_output(arguments["input_payload"]["problem"])
            calls += 1
            if calls == 1:
                if failure == "wrong_id":
                    output["problem_id"] = "other-sql"
                elif failure == "wrong_ref":
                    output["evidence_refs"] = ["artifact:unrelated#row"]
                elif failure == "missing_metric":
                    output["possible_causes"][0]["supporting_metrics"] = ["average_cpu_ms"]
                elif failure == "non_metric":
                    output["possible_causes"][0]["supporting_metrics"] = ["sql_text"]
                else:
                    output["analysis"] = ""
            return SimpleNamespace(output=output, receipt=SimpleNamespace(model_dump=lambda **_:{}))
        prompt = SimpleNamespace(content="逐条分析",ref=lambda:{"prompt_id":"inspection_sql_review"})
        reviews = await review_inspection_sql(context=review_context(steps,source),result=source,
            model=SimpleNamespace(generate_structured=AsyncMock(side_effect=generate)),
            prompts=SimpleNamespace(resolve=AsyncMock(return_value=prompt)),concurrency=1)
        assert [review["status"] for review in reviews] == ["UNAVAILABLE","COMPLETE"]
    asyncio.run(run())


@pytest.mark.parametrize("deadline,expired", [
    ("2000-01-01T00:00:00", True), ("2000-01-01T00:00:00Z", True),
    ("2099-01-01T00:00:00", False), ("2099-01-01T00:00:00Z", False),
])
def test_deadline_and_configured_concurrency_are_honored(deadline, expired):
    async def run():
        steps, source, _, _, _ = pipeline(rows=[observation(i, average_elapsed_ms=2000) for i in range(8)])
        active = maximum = 0
        async def generate(**arguments):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            active -= 1
            return SimpleNamespace(output=review_output(arguments["input_payload"]["problem"]),
                receipt=SimpleNamespace(model_dump=lambda **_:{}))
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate))
        prompt = SimpleNamespace(content="逐条分析",ref=lambda:{"prompt_id":"inspection_sql_review"})
        reviews = await review_inspection_sql(context=review_context(steps,source,deadline=deadline),
            result=source,model=model,prompts=SimpleNamespace(resolve=AsyncMock(return_value=prompt)),concurrency=2)
        assert len(reviews) == 8
        assert all(review["status"] == ("UNAVAILABLE" if expired else "COMPLETE") for review in reviews)
        assert maximum == (0 if expired else 2)
    asyncio.run(run())


@pytest.mark.parametrize("database_type", ["MYSQL", "ORACLE"])
def test_scheduled_composer_publishes_frozen_per_sql_reviews_into_the_actual_report(database_type):
    from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
    async def run():
        rows = [observation(i, average_elapsed_ms=1500+i*1000) for i in range(8)]
        if database_type == "ORACLE":
            rows = [observation(i, average_elapsed_ms=20, workarea_multipass_percent=20,
                workarea_total_executions=100, workarea_multipass_executions=20,
                maximum_workarea_temp_mb=64+i*64) for i in range(8)]
        steps, source, _, _, _ = pipeline(database_type, rows=rows)
        context = review_context(steps,source)
        context.plan_snapshot["target"]["db_type"] = database_type
        async def generate(**arguments):
            assert arguments["purpose"] == "aiops.inspection-sql-review"
            output = review_output(arguments["input_payload"]["problem"])
            if database_type == "ORACLE":
                output["possible_causes"][0]["supporting_metrics"] = ["workarea_multipass_percent"]
            return SimpleNamespace(output=output,
                receipt=SimpleNamespace(model_dump=lambda **_:{"model_id":"agent-selected-model"}))
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate))
        prompt = SimpleNamespace(content="逐条分析",ref=lambda:{"prompt_id":"inspection_sql_review"})
        result = await DbaAnswerComposeHandler(model_client=model,
            prompts=SimpleNamespace(resolve=AsyncMock(return_value=prompt))).execute(context)
        assert len(result.inspection_sql_reviews) == model.generate_structured.await_count == 8
        async def add_artifact(entity):
            entity.artifact_id = uuid7()
            return entity
        async def publish_report(entity):
            entity.is_current = 1
            return entity
        uow = SimpleNamespace(
            targets=SimpleNamespace(get_scoped=AsyncMock(return_value=SimpleNamespace(display_name="巡检样例库"))),
            inspections=SimpleNamespace(publish_report=AsyncMock(side_effect=publish_report),add_report_sources=AsyncMock()),
            runs=SimpleNamespace(add_artifact=AsyncMock(side_effect=add_artifact),append_event=AsyncMock()),
            outbox=SimpleNamespace(add=AsyncMock(side_effect=lambda entity:entity)),
            platform_notifications=SimpleNamespace(emit_report_ready=AsyncMock()),
        )
        service = AIOpsRuntimeService(uow_factory=AsyncMock(),blueprint_registry=AsyncMock(),handler_registry=AsyncMock())
        run = SimpleNamespace(ops_run_id=context.run_id, target_id=context.target_id,
            domain_id=uuid7(),inspection_fire_id=uuid7(),actor_id="system:inspection-scheduler",
            plan_snapshot_json=context.plan_snapshot)
        artifact = await service._publish_turn_inspection_report(
            uow=uow,run=run,task=SimpleNamespace(ops_task_id=uuid7()),
            source_artifact=SimpleNamespace(artifact_id=uuid7(),schema_version="AIOPS_TURN_RESULT.v1",
                content_hash="a"*64,payload_json=result.model_dump(mode="json")),
            now=datetime.now(UTC),trace_id="test-sql-review-report",
        )
        stored = deepcopy(artifact.payload_json)
        review_fact = next(fact for fact in stored["facts"] if fact["kind"] == "inspection_sql_reviews")
        assert len(review_fact["reviews"]) == 8
        template = resolve_system_template("system:inspection.daily")
        for _ in range(2):
            presentation = report_presentation(payload=artifact.payload_json,template=template)
            group = presentation["daily_inspection"]["sql_details"][0]
            assert len(group["statements"]) == 8
            assert all(item["agent_review"]["status"] == "COMPLETE" for item in group["statements"])
            assert render_pdf(presentation).startswith(b"%PDF-")
        assert artifact.payload_json == stored
        assert model.generate_structured.await_count == 8
    asyncio.run(run())


def test_analysis_failure_is_not_reported_as_success_or_no_problem():
    async def run():
        steps, source, _, _, _ = pipeline(rows=[observation(average_elapsed_ms=2000)])
        context = SimpleNamespace(trigger_type="SCHEDULE", run_id=uuid7(), attempt=1, deadline_at=None,
            plan_snapshot={"target":{"db_type":"MYSQL"},"client_metadata":{"inspection":{"evidence_steps":steps}},
                "answer_context":{"model":{},"prompts":{"inspection_sql_review":{}}}})
        reviews = await review_inspection_sql(context=context, result=source, model=None, prompts=None, concurrency=1)
        assert reviews[0]["status"] == "UNAVAILABLE"
        *_, presentation = pipeline(rows=[observation(average_elapsed_ms=2000)], reviews=reviews)
        group = presentation["daily_inspection"]["sql_details"][0]
        assert group["status"] == "COMPLETE" and len(group["statements"]) == 1
        assert group["statements"][0]["agent_review"]["status"] == "UNAVAILABLE"
    asyncio.run(run())
