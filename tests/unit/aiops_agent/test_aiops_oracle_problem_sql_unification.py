"""Oracle 旧排序入口收口、真实工作区统计及逐 SQL Agent 分析回归。"""

import asyncio
import sqlite3
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import sqlglot

from aiops_agent.application.inspections.assessment import compile_standard_steps, freeze_inspection_policy
from aiops_agent.application.inspections.check_catalog import compile_selected_check_steps, inspection_template_steps, load_check_catalog
from aiops_agent.application.inspections.sql_problems import (
    PROBLEM_SQL_TOOL, classify_problem_sql, problem_sql_inputs, problem_sql_policy,
)
from aiops_agent.application.reporting import render_pdf
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from tests.unit.aiops_agent.test_aiops_problem_sql_inspection import (
    observation, pipeline, review_context, review_output,
)


def oracle_tool():
    return DiagnosticRegistry.load().resolve(tool_id=PROBLEM_SQL_TOOL, tool_version="1.0.0",
        db_type="ORACLE", db_version="19c", capabilities={"dynamic_performance_views"}, entitlements=set())


def test_problem_sql_check_names_cover_all_resource_signals_not_only_sorting():
    checks = [item for group in load_check_catalog().groups for item in group.checks
              if item.report_section == "SQL"]
    assert {db_type for item in checks for db_type in item.supported_db_types} == {"ORACLE", "MYSQL", "POSTGRESQL"}
    assert all(item.display_name == "问题 SQL" and item.tool_id == PROBLEM_SQL_TOOL for item in checks)
    description = next(section["description"] for section in load_check_catalog().report_sections
                       if section["section_id"] == "SQL")
    for term in ("CPU", "I/O", "等待", "扫描", "临时空间", "Agent"):
        assert term in description


@pytest.mark.parametrize("old_tool", ["db.sql.top_current", "db.oracle.performance.heavy_sort_sql"])
@pytest.mark.parametrize("schedule_type", ["DAILY", "MONTHLY"])
def test_existing_sort_selection_uses_one_current_query(old_tool, schedule_type):
    legacy = {
        "title": "严重排序 SQL", "tool_id": old_tool, "input": {"limit": 20},
        "expected_evidence_kind": "TOP_SQL", "check_ids": ["oracle.performance.heavy_sort_sql"],
        "supported_db_types": ["ORACLE"],
    }
    current = compile_selected_check_steps(["oracle.sql.problems"], database_type="ORACLE")[0]
    original = deepcopy(legacy)
    steps = inspection_template_steps({"schema_version": "INSPECTION_TEMPLATE.v1",
        "evidence_steps": [legacy, current]}, database_type="ORACLE", schedule_type=schedule_type)
    assert legacy == original
    sql_steps = [step for step in steps if step["expected_evidence_kind"] == "TOP_SQL"]
    assert len(sql_steps) == 1
    assert sql_steps[0]["tool_id"] == PROBLEM_SQL_TOOL
    assert sql_steps[0]["title"] == "问题 SQL"
    assert set(sql_steps[0]["check_ids"]) == {"oracle.sql.problems", "oracle.performance.heavy_sort_sql"}
    assert sql_steps[0]["input"] == problem_sql_inputs(sql_steps[0]["sql_problem_policy"])
    assert "limit" not in sql_steps[0]["input"]
    assert not {step["tool_id"] for step in steps} & {"db.sql.top_current", "db.oracle.performance.heavy_sort_sql"}


def test_unlabelled_old_sql_snapshot_cannot_bypass_current_inspection_rules():
    steps = compile_standard_steps(({
        "title": "严重排序 SQL", "tool_id": "db.sql.top_current",
        "expected_evidence_kind": "TOP_SQL", "input": {"limit": 20},
    },), database_type="ORACLE", schedule_type="DAILY", policy=freeze_inspection_policy("ORACLE"))
    assert "db.sql.top_current" not in {step["tool_id"] for step in steps}
    assert PROBLEM_SQL_TOOL in {step["tool_id"] for step in steps}


def test_other_specialist_steps_and_unknown_custom_steps_are_not_silently_removed():
    specialist = compile_selected_check_steps(["oracle.object.invalid"], database_type="ORACLE")[0]
    specialist["input"] = {"limit": 73}
    custom = {"tool_id": "db.custom.readonly", "title": "专项核验", "input": {"sample": 2},
        "expected_evidence_kind": "CUSTOM"}
    steps = compile_standard_steps((specialist, custom), database_type="ORACLE", schedule_type="DAILY",
        policy=freeze_inspection_policy("ORACLE"))
    assert next(step for step in steps if step["tool_id"] == specialist["tool_id"])["input"] == {"limit": 73}
    assert next(step for step in steps if step["tool_id"] == custom["tool_id"])["input"] == {"sample": 2}


@pytest.mark.parametrize("metric,limit", [("workarea_multipass_percent", 10), ("maximum_workarea_temp_mb", 64)])
@pytest.mark.parametrize("elapsed,value,problem", [
    (0.7, 1000000, False), (9.99, 1000000, False), (10, 0.999, False), (10, 1, True), (10, 5, True),
])
def test_workarea_rules_use_per_execution_floor_and_actual_resource_breach(metric, limit, elapsed, value, problem):
    row = observation(average_elapsed_ms=elapsed, **{metric: limit * value})
    classified = classify_problem_sql(row, policy=problem_sql_policy(), database_type="ORACLE")
    assert bool(classified) == problem
    if classified:
        assert classified["severity"] == ("HIGH" if value >= 5 else "MEDIUM")
    for database_type in ("MYSQL", "POSTGRESQL"):
        assert classify_problem_sql(row, policy=problem_sql_policy(), database_type=database_type) is None


def test_oracle_real_query_aggregates_plans_and_workareas_without_multiplying_counts():
    tool = oracle_tool()
    query = sqlglot.parse_one(tool.sql, read="oracle").sql(dialect="sqlite")
    columns = ("con_id", "sql_id", "sql_text", "executions", "elapsed_time", "cpu_time",
        "user_io_wait_time", "application_wait_time", "disk_reads")
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE v$sqlstats (" + ", ".join(columns) + ")")
        connection.execute("CREATE TABLE v$sql_workarea (con_id, sql_id, total_executions, multipasses_executions, max_tempseg_size)")
        connection.execute("CREATE TABLE dual (dummy)")
        connection.execute("INSERT INTO dual VALUES (1)")
        # 同一 SQL 的多个计划先聚合；另一容器同名 SQL 不可串数据。
        connection.executemany("INSERT INTO v$sqlstats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            (3, "same", "SELECT * FROM orders", 100, 2000000, 0, 0, 0, 0),
            (3, "same", "SELECT * FROM orders", 300, 6000000, 0, 0, 0, 0),
            (4, "same", "SELECT * FROM orders", 100, 70000, 0, 0, 0, 0),
            (3, "fast", "SELECT 1", 490072, 343050400, 0, 0, 0, 0),
            (3, "onepass", "SELECT * FROM fast_sort", 100, 2000000, 0, 0, 0, 0),
        ])
        connection.executemany("INSERT INTO v$sql_workarea VALUES (?, ?, ?, ?, ?)", [
            (3, "same", 100, 20, 64 * 1048576),
            (3, "same", 300, 20, 32 * 1048576),
            (4, "same", 100, 100, 1024 * 1048576),
            (3, "fast", 490072, 490072, 1024 * 1048576),
            (3, "onepass", 100, 0, 0),
        ])
        rows = [dict(row) for row in connection.execute(query, problem_sql_inputs(problem_sql_policy()))]
    assert len(rows) == 1
    row = rows[0]
    assert row["sql_key"] == "3:same"
    assert row["execution_count"] == 400 and row["average_elapsed_ms"] == 20
    assert row["workarea_total_executions"] == 400 and row["workarea_multipass_executions"] == 40
    assert row["workarea_multipass_percent"] == 10
    assert row["maximum_workarea_temp_mb"] == 64
    problem = classify_problem_sql(row, policy=problem_sql_policy(), database_type="ORACLE")
    assert {signal["code"] for signal in problem["signals"]} == {"WORKAREA_MULTIPASS", "WORKAREA_TEMP"}
    assert "V_$SQL_WORKAREA" in tool.definition.required_privileges


def test_no_accumulated_seconds_or_onepass_alone_implies_a_sort_problem():
    row = observation(execution_count=10000000, average_elapsed_ms=0.7,
        workarea_total_executions=10000000, workarea_multipass_executions=0,
        workarea_multipass_percent=0, maximum_workarea_temp_mb=0)
    *_, assessment, presentation = pipeline("ORACLE", [row])
    assert not assessment["findings"]
    assert presentation["daily_inspection"]["sql_details"][0]["summary"] == "当前没有问题 SQL"


@pytest.mark.parametrize("version", ["12", "13", "14", "17"])
def test_postgresql_unavailable_workarea_metrics_are_numeric_nulls(version):
    from sqlglot import exp
    tool = DiagnosticRegistry.load().resolve(tool_id=PROBLEM_SQL_TOOL, tool_version="1.0.0",
        db_type="POSTGRESQL", db_version=version, capabilities={"pg_stat_statements"}, entitlements=set())
    tree = sqlglot.parse_one(tool.sql, read="postgres")
    metrics = next(cte.this for cte in tree.find_all(exp.CTE) if cte.alias_or_name == "metrics")
    for expression in metrics.expressions:
        if expression.alias_or_name.startswith("workarea_") or expression.alias_or_name == "maximum_workarea_temp_mb":
            assert isinstance(expression.this, exp.Cast)
            assert isinstance(expression.this.this, exp.Null)
            assert expression.this.args["to"].this in {exp.DataType.Type.BIGINT, exp.DataType.Type.DECIMAL}


@pytest.mark.parametrize("missing_metric", [False, True])
def test_sort_only_sql_gets_one_agent_review_and_one_risk_item(missing_metric):
    async def run():
        row = observation(average_elapsed_ms=20, workarea_total_executions=100,
            workarea_multipass_executions=20, workarea_multipass_percent=20, maximum_workarea_temp_mb=128)
        steps, source, _, assessment, _ = pipeline("ORACLE", [row])
        assert len(assessment["findings"]) == 1
        context = review_context(steps, source)
        context.plan_snapshot["target"]["db_type"] = "ORACLE"
        async def generate(**arguments):
            problem = arguments["input_payload"]["problem"]
            assert {signal["code"] for signal in problem["signals"]} == {"WORKAREA_MULTIPASS", "WORKAREA_TEMP"}
            output = review_output(problem)
            output["analysis"] = "工作区存在多遍处理，需结合实际执行计划复核排序、哈希或聚合开销。"
            output["possible_causes"][0]["supporting_metrics"] = (
                ["average_cpu_ms"] if missing_metric else ["workarea_multipass_percent", "maximum_workarea_temp_mb"])
            return SimpleNamespace(output=output, receipt=SimpleNamespace(model_dump=lambda **_: {}))
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate))
        prompt = SimpleNamespace(content="逐条分析", ref=lambda: {"prompt_id": "inspection_sql_review"})
        result = await DbaAnswerComposeHandler(model_client=model,
            prompts=SimpleNamespace(resolve=AsyncMock(return_value=prompt))).execute(context)
        assert model.generate_structured.await_count == len(result.inspection_sql_reviews) == 1
        reviews = result.inspection_sql_reviews
        assert reviews[0]["status"] == ("UNAVAILABLE" if missing_metric else "COMPLETE")
        *_, assessment, presentation = pipeline("ORACLE", [row], reviews=reviews)
        group = presentation["daily_inspection"]["sql_details"][0]
        assert len(assessment["findings"]) == len(group["statements"]) == 1
        assert group["title"] == "问题 SQL"
        assert group["statements"][0]["agent_review"]["status"] == reviews[0]["status"]
        assert assessment["findings"][0]["finding_id"] == reviews[0]["problem_id"]
        assert render_pdf(presentation).startswith(b"%PDF-")
    asyncio.run(run())
