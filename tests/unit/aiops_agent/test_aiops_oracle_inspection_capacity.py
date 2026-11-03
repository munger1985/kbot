"""Oracle 巡检正常表述与自动扩展容量规则的回归。"""

from copy import deepcopy
from datetime import UTC, datetime
import sqlite3
from types import SimpleNamespace

import pytest
from sqlglot import exp, parse_one

from aiops_agent.application.diagnosis.findings import compile_findings
from aiops_agent.application.inspections.assessment import assess_inspection
from aiops_agent.application.inspections.check_catalog import compile_selected_check_steps
from aiops_agent.application.reporting import report_presentation, render_pdf, resolve_system_template
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.turn_answer import TurnEvidenceFact, TurnEvidenceGap
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from platform_core.identity import uuid7


NOW = datetime(2026, 11, 3, 12, tzinfo=UTC)
CHECKS = ("oracle.session.long_running", "oracle.session.long_transaction",
          "oracle.session.lock_wait", "oracle.object.invalid")


def fact(tool, rows=()):
    return TurnEvidenceFact(
        artifact_id=str(uuid7()), evidence_ref=f"artifact:{uuid7()}#capacity", source_id=tool.definition.tool_id,
        step_id=tool.definition.tool_id, tool_id=tool.definition.tool_id, trust_level="SOURCE_VERIFIED",
        measurement_semantics="CURRENT_ACTIVITY", presentation_kind="TABLE", captured_at=NOW.isoformat(),
        columns=tuple({"name": column.name} for column in tool.definition.output_columns),
        rows=tuple(tuple(row.get(column.name) for column in tool.definition.output_columns) for row in rows),
        row_count=len(rows),
    )


def tool(tool_id):
    registry = DiagnosticRegistry.load()
    definition = next(item.definition for item in registry.tools
                      if item.definition.tool_id == tool_id and item.definition.db_type == "ORACLE")
    capabilities = {capability for item in registry.tools if item.definition.db_type == "ORACLE"
                    for capability in item.definition.required_capabilities}
    return registry.resolve(
        tool_id=tool_id, tool_version=definition.version, db_type="ORACLE", db_version="23.0",
        capabilities=capabilities, entitlements=set(),
    )


def project(check_ids, evidence=(), gaps=()):
    steps = compile_selected_check_steps(list(check_ids), database_type="ORACLE", schedule_type="DAILY")
    steps = tuple({**step, "required": True} for step in steps)
    facts, report_gaps, _ = AIOpsRuntimeService._inspection_report_projection(
        inspection={"evidence_steps": steps, "period_end": NOW.isoformat()}, database_type="ORACLE",
        source=SimpleNamespace(evidence=evidence, evidence_gaps=gaps), action_tool_ids={},
    )
    return facts, report_gaps, assess_inspection(facts, policy={}, gaps=report_gaps)


def capacity_row(name="SYSTEM", *, maximum=32768, used=991.1, allocated=1000, auto_files=1):
    return {
        "tablespace_name": name, "allocated_mb": allocated, "used_mb": used,
        "free_mb": allocated-used, "used_percent": used/allocated*100,
        "maximum_mb": maximum, "maximum_headroom_mb": maximum-used, "file_count": 2,
        "storage_type": "FILESYSTEM", "asm_diskgroups": None, "datafile_directories": "/oradata",
        "autoextensible_file_count": auto_files, "capacity_used_percent": used/maximum*100,
    }


@pytest.mark.parametrize("check_id", CHECKS)
def test_successful_empty_exception_list_is_unambiguously_normal(check_id):
    step = compile_selected_check_steps([check_id], database_type="ORACLE", schedule_type="DAILY")[0]
    _, gaps, result = project([check_id], (fact(tool(step["tool_id"])),))
    check = result["checks"][0]
    assert result["health_status"] == "NORMAL"
    assert result["completed_required_count"] == 1 and not gaps
    assert check["status_display"] == "正常"
    assert "未发现" in check["actual_display"]
    assert "不等于" not in check["actual_display"] + check["basis"]
    assert "无适用规则" not in check["basis"]
    assert "冻结" not in check["basis"]
    assert check["actual_display"] != check["basis"]


@pytest.mark.parametrize("name", ["SYSTEM", "SYSAUX", "APP_DATA"])
@pytest.mark.parametrize("auto_files,maximum,status", [(1, 32768, "NORMAL"), (0, 1000, "ABNORMAL"), (1, 1100, "ABNORMAL")])
def test_all_tablespaces_use_configured_autoextend_limit(name, auto_files, maximum, status):
    row = capacity_row(name, auto_files=auto_files, maximum=maximum)
    raw = fact(tool("db.storage.capacity"), (row,))
    compiled = compile_findings((raw,), database_type="ORACLE")
    _, gaps, result = project(["oracle.storage.tablespace_headroom"], (raw,))
    assert not gaps and result["completion_status"] == "COMPLETE"
    assert result["health_status"] == status
    if status == "NORMAL":
        assert not compiled.findings and not result["findings"]
    else:
        finding = compiled.findings[0]
        assert finding.threshold.metric == "capacity_used_percent"
        assert finding.threshold.current == pytest.approx(row["capacity_used_percent"])
        assert "配置容量上限" in result["findings"][0]["summary"]
    display = result["checks"][0]["actual_display"]
    assert "自动扩展" in display and "上限" in display
    assert "85" in result["checks"][0]["basis"]


def test_multiple_risky_tablespaces_do_not_repeat_same_threshold():
    raw = fact(tool("db.storage.capacity"), tuple(capacity_row(name, maximum=1000, auto_files=0)
                                                for name in ("SYSTEM", "SYSAUX", "APP_DATA")))
    *_, result = project(["oracle.storage.tablespace_headroom"], (raw,))
    assert len(result["findings"]) == 3
    for item in result["findings"]:
        assert item["basis"].count("85") == 1


@pytest.mark.parametrize("failure", ["TIMEOUT", "PRIVILEGE_MISSING"])
def test_failed_exception_query_remains_incomplete(failure):
    step = compile_selected_check_steps([CHECKS[2]], database_type="ORACLE", schedule_type="DAILY")[0]
    *_, result = project([CHECKS[2]], gaps=(TurnEvidenceGap(
        source_id="database", step_id=step["tool_id"], code=failure, detail="检查查询失败"),))
    assert result["health_status"] == "INCOMPLETE"
    assert result["completed_required_count"] == 0


def test_report_and_pdf_keep_capacity_assessment_and_frozen_history():
    raw = fact(tool("db.storage.capacity"), (capacity_row(),))
    facts, gaps, result = project(["oracle.storage.tablespace_headroom"], (raw,))
    payload = {"report_type": "INSPECTION_DAILY", "title": "Oracle 巡检",
               "period_start": "2026-11-02T12:00:00Z", "period_end": NOW.isoformat(),
               "scope": {"inspection_assessment": result}, "facts": list(facts), "gaps": list(gaps)}
    before = deepcopy(payload)
    presentation = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
    assert presentation["daily_inspection"]["conclusion"]["health_status"] == "NORMAL"
    assert render_pdf(presentation).startswith(b"%PDF-")
    assert payload == before


def test_capacity_sql_only_counts_enabled_files_as_growable():
    resolved = tool("db.storage.capacity")
    assert "autoextensible" in resolved.sql.lower()
    names = {column.name for column in resolved.definition.output_columns}
    assert {"autoextensible_file_count", "capacity_used_percent"} <= names
    assert resolved.definition.required_privileges == ("DBA_DATA_FILES", "DBA_FREE_SPACE")


@pytest.mark.parametrize("file_rows,maximum,auto_files", [
    ([(1000, 32768, "NO")], 1000, 0),
    ([(1000, 32768, "YES")], 32768, 1),
    ([(1000, 2000, "YES"), (1000, 32768, "NO")], 3000, 1),
    ([(1000, 1000, "YES")], 1000, 1),
    ([(1000, 0, "YES")], 1000, 1),
    ([(1000, None, "YES")], 1000, 1),
])
def test_real_sql_capacity_aggregate_respects_each_file_switch(file_rows, maximum, auto_files):
    # 仅抽取真实 Oracle SQL 的数值聚合，在内存中验证；不手写另一套容量公式。
    tree = parse_one(tool("db.storage.capacity").sql, read="oracle")
    aggregate = next(cte.this for cte in tree.find_all(exp.CTE) if cte.alias_or_name == "data_files")
    selected = [item.copy() for item in aggregate.expressions
                if item.alias_or_name in {"maximum_bytes", "autoextensible_file_count"}]
    assert len(selected) == 2
    query = exp.select(*selected).from_("data_file_detail").sql(dialect="sqlite")
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE data_file_detail (bytes, maxbytes, autoextensible)")
        connection.executemany("INSERT INTO data_file_detail VALUES (?, ?, ?)", file_rows)
        assert connection.execute(query).fetchone() == (maximum, auto_files)


@pytest.mark.parametrize("maximum", [0, -1, None, "无效值", float("nan"), float("inf")])
def test_invalid_capacity_limit_is_not_reported_normal(maximum):
    row = capacity_row()
    row["maximum_mb"] = maximum
    raw = fact(tool("db.storage.capacity"), (row,))
    *_, result = project(["oracle.storage.tablespace_headroom"], (raw,))
    assert result["health_status"] == "INCOMPLETE"
    assert result["completed_required_count"] == 0
