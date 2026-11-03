"""MySQL 巡检实测情况展示来源 SQL，而不是把摘要哈希当正文。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from aiops_agent.application.inspections.assessment import assess_inspection
from aiops_agent.application.inspections.check_catalog import compile_selected_check_steps
from aiops_agent.application.inspections.sql_observations import format_picoseconds, inspection_sql_details
from aiops_agent.application.reporting import report_presentation, render_pdf, resolve_system_template
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from platform_core.identity import uuid7
from platform_core.contracts.aiops.public import DailyInspectionPresentation


DIGEST = "f2b9be5f0f32ca7507e4ddf133cd2fad0993a1c26fde9caa6f067b1af2b7603c"
SAMPLE = "SELECT order_id FROM orders WHERE customer_id = 42"
NORMALIZED = "SELECT order_id FROM orders WHERE customer_id = ?"


def pipeline(sample=SAMPLE, normalized=NORMALIZED, *, count=1):
    tool = DiagnosticRegistry.load().resolve(
        tool_id="db.mysql.statement.digest.top", tool_version="1.0.0",
        db_type="MYSQL", db_version="8.4", capabilities={"statement_digest"}, entitlements=set(),
    )
    names = [column.name for column in tool.definition.output_columns]
    raw = {"schema_name": "aiops_demo", "digest": DIGEST, "query_sample_text": sample,
           "digest_text": normalized, "execution_count": 592553, "total_latency_picoseconds": 1421368539162000,
           "average_latency_picoseconds": 2398719000, "maximum_latency_picoseconds": 239177605000,
           "rows_examined": 592553, "rows_sent": 10, "temp_disk_tables": 0}
    source = SimpleNamespace(evidence=(SimpleNamespace(
        tool_id=tool.definition.tool_id, columns=tuple({"name": name} for name in names),
        rows=tuple(tuple(raw.get(name) for name in names) for _ in range(count)), row_count=count,
        truncated=False, evidence_ref=f"artifact:{uuid7()}#digests",
    ),), evidence_gaps=())
    inspection = {"evidence_steps": tuple({**step, "tool_id": "db.mysql.statement.digest.top", "report_columns": list(raw)} for step in compile_selected_check_steps(
        ["mysql.sql.top"], database_type="MYSQL", schedule_type="DAILY")),
        "period_end": "2026-11-03T10:00:00Z"}
    facts, gaps, _ = AIOpsRuntimeService._inspection_report_projection(
        inspection=inspection, source=source, database_type="MYSQL", action_tool_ids={},
    )
    result = assess_inspection(facts, policy={})
    return raw, facts, gaps, result


def test_mysql_sql_sample_survives_real_catalog_and_report_projection():
    raw, facts, gaps, result = pipeline()
    check = result["checks"][0]
    assert SAMPLE in check["actual_display"]
    assert "SQL 文本（执行样本）" in check["actual_display"]
    assert NORMALIZED not in check["actual_display"]
    assert DIGEST not in check["actual_display"]
    assert "592553" in check["actual_display"]
    assert "239.2 ms" in check["actual_display"]
    assert "皮秒" not in check["actual_display"]
    assert not gaps
    fact = next(item for item in facts if item.get("kind") == "inspection_check")
    assert fact["observations"][0]["digest"] == raw["digest"]
    assert fact["observations"][0]["query_sample_text"] == SAMPLE
    assert fact["observations"][0]["maximum_latency_picoseconds"] == raw["maximum_latency_picoseconds"]
    assert result["completion_status"] == "COMPLETE"


@pytest.mark.parametrize("sample", [None, "", "   "])
def test_normalized_sql_is_used_and_labeled_when_no_execution_sample(sample):
    *_, result = pipeline(sample=sample)
    text = result["checks"][0]["actual_display"]
    assert NORMALIZED in text and "SQL 文本（归一化）" in text
    assert DIGEST not in text


@pytest.mark.parametrize("sample,normalized", [(None, None), ("", ""), (" ", " ")])
def test_missing_sql_is_explicit_and_hash_is_only_a_locator(sample, normalized):
    *_, result = pipeline(sample=sample, normalized=normalized)
    text = result["checks"][0]["actual_display"]
    assert "来源未返回 SQL 文本" in text
    assert "Digest" in text and DIGEST in text
    assert "592553" in text
    assert result["completion_status"] == "COMPLETE"


def test_sql_display_is_bounded_without_changing_frozen_source_text():
    sample = "SELECT " + ", ".join(f"field_{index}" for index in range(300)) + " FROM orders"
    _, facts, _, result = pipeline(sample=sample)
    assert len(result["checks"][0]["actual_display"]) < 1200
    assert "展示已截短" in result["checks"][0]["actual_display"]
    fact = next(item for item in facts if item.get("kind") == "inspection_check")
    assert fact["observations"][0]["query_sample_text"] == sample


@pytest.mark.parametrize("sample", [
    SAMPLE,
    "SELECT '<script>alert(1)</script>&' FROM orders WHERE amount < 42",
    "SELECT " + ", ".join(f"field_{index}" for index in range(300)) + " FROM orders",
], ids=("sample", "escaped-sql", "long-sql"))
def test_report_and_pdf_use_same_frozen_sql_and_do_not_mutate_old_report(sample):
    _, facts, gaps, result = pipeline(sample=sample, count=4)
    payload = {"report_type": "INSPECTION_DAILY", "title": "数据库日常巡检报告",
               "period_start": "2026-11-02T10:00:00Z", "period_end": "2026-11-03T10:00:00Z",
               "scope": {"inspection_assessment": result}, "facts": list(facts), "gaps": list(gaps)}
    original = deepcopy(payload)
    presentation = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
    daily = presentation["daily_inspection"]
    assert sample[:400] not in daily["checks"][0]["actual_display"]
    group = daily["sql_details"][0]
    assert group["status"] == "NOT_ASSESSED"
    assert group["statements"] == []
    typed = DailyInspectionPresentation.model_validate(daily)
    assert typed.sql_details[0].statements == ()
    assert render_pdf(presentation).startswith(b"%PDF-")
    assert payload == original
    old = deepcopy(payload)
    old["scope"]["inspection_assessment"]["checks"][0]["actual_display"] = "旧报告保留冻结的摘要内容"
    old_original = deepcopy(old)
    frozen = report_presentation(payload=old, template=resolve_system_template("system:inspection.daily"))
    assert frozen["daily_inspection"]["checks"] == daily["checks"]
    assert frozen["daily_inspection"]["conclusion"] == daily["conclusion"]
    assert old == old_original


def test_non_sql_observation_format_remains_unchanged():
    result = assess_inspection([{
        "kind": "inspection_check", "title": "连接", "check_status": "CLEAR",
        "observations": [{"current_utilization": 4, "limit_value": 100}],
        "evidence_refs": ["original"],
    }], policy={})
    assert result["checks"][0]["actual_display"] == "当前使用率：4，上限值：100"


@pytest.mark.parametrize("value,expected", [
    (1425880197358000, "23.8 min"), ("2398109000", "2.4 ms"),
    (239177605000, "239.2 ms"), (0, "0.0 ms"), (1, "0.0 ms"),
    (150000000, "0.2 ms"), (None, "未记录"), ("", "未记录"),
    ("invalid", "未记录"), (-1, "未记录"), ("NaN", "未记录"),
    ("Infinity", "未记录"), (True, "未记录"), (False, "未记录"),
    ("1e999999999", "未记录"),
])
def test_picoseconds_are_displayed_as_milliseconds(value, expected):
    assert format_picoseconds(value) == expected


def test_sql_details_preserve_source_order_and_missing_values_without_fabrication():
    _, facts, _, result = pipeline(sample=None, normalized=None, count=5)
    fact = next(item for item in facts if item.get("kind") == "inspection_check")
    fact["observations"][0]["execution_count"] = 12
    fact["observations"][1]["execution_count"] = None
    fact["observations"][2]["execution_count"] = -1
    fact["observations"][0]["average_latency_picoseconds"] = None
    original = deepcopy(facts)
    checks, groups = inspection_sql_details(list(facts), result["checks"])
    group = groups[0]
    assert group["status"] == "NOT_ASSESSED" and group["statements"] == []
    assert "重新运行" in group["summary"]
    assert "SQL 文本" not in checks[0]["actual_display"]
    assert facts == original


def test_no_frozen_observations_keeps_old_display_and_does_not_parse_text():
    _, facts, _, result = pipeline()
    fact = next(item for item in facts if item.get("kind") == "inspection_check")
    fact["observations"] = []
    result["checks"][0]["actual_display"] = "SQL 文本：" + SAMPLE + "，平均延迟皮秒：2398719000"
    checks, groups = inspection_sql_details(list(facts), result["checks"])
    assert groups[0]["status"] == "NOT_ASSESSED"
    assert SAMPLE not in checks[0]["actual_display"]
    assert result["checks"][0]["actual_display"].startswith("SQL 文本：" + SAMPLE)
