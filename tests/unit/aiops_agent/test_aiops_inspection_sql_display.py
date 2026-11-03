"""MySQL 巡检实测情况展示来源 SQL，而不是把摘要哈希当正文。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from aiops_agent.application.inspections.assessment import assess_inspection
from aiops_agent.application.inspections.check_catalog import compile_selected_check_steps
from aiops_agent.application.reporting import report_presentation, render_pdf, resolve_system_template
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from platform_core.identity import uuid7


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
    inspection = {"evidence_steps": compile_selected_check_steps(
        ["mysql.sql.top"], database_type="MYSQL", schedule_type="DAILY"),
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
    assert "239177605000" in check["actual_display"]
    assert not gaps
    fact = next(item for item in facts if item.get("kind") == "inspection_check")
    assert fact["observations"][0]["digest"] == raw["digest"]
    assert fact["observations"][0]["query_sample_text"] == SAMPLE
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
    assert presentation["daily_inspection"]["checks"][0]["actual_display"].count(sample[:400]) == 3
    assert render_pdf(presentation).startswith(b"%PDF-")
    assert payload == original
    old = deepcopy(payload)
    old["scope"]["inspection_assessment"]["checks"][0]["actual_display"] = "旧报告保留冻结的摘要内容"
    frozen = report_presentation(payload=old, template=resolve_system_template("system:inspection.daily"))
    assert frozen["daily_inspection"]["checks"][0]["actual_display"] == "旧报告保留冻结的摘要内容"


def test_non_sql_observation_format_remains_unchanged():
    result = assess_inspection([{
        "kind": "inspection_check", "title": "连接", "check_status": "CLEAR",
        "observations": [{"current_utilization": 4, "limit_value": 100}],
        "evidence_refs": ["original"],
    }], policy={})
    assert result["checks"][0]["actual_display"] == "当前使用率：4，上限值：100"
