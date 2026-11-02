"""成功空结果按业务语义形成风险，不与查询失败或异常清单零行混淆。"""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.application.inspections.assessment import assess_inspection, monitoring_checks
from aiops_agent.application.inspections.backup_observations import backup_policy_result
from aiops_agent.application.inspections.check_catalog import load_check_catalog, compile_selected_check_steps
from aiops_agent.application.inspections.monthly import monthly_business_review
from aiops_agent.application.reporting import report_presentation, render_pdf, resolve_system_template
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.evidence import ObservationSet, ObservationGap
from aiops_agent.contracts.turn_answer import TurnEvidenceGap
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7


NOW = datetime(2026, 11, 2, 12, tzinfo=UTC)
CATALOG = [(item, db) for group in load_check_catalog().groups for item in group.checks
           if item.availability == "READY" and item.tool_id and item.tool_id.startswith("db.")
           for db in item.supported_db_types]


@pytest.mark.parametrize("item,db_type", CATALOG, ids=[f"{db}:{item.check_id}" for item, db in CATALOG])
def test_each_catalog_item_obeys_its_frozen_empty_result_rule(item, db_type):
    steps = compile_selected_check_steps([item.check_id], database_type=db_type, schedule_type="DAILY")
    source = SimpleNamespace(evidence=(SimpleNamespace(
        tool_id=item.tool_id, evidence_ref=f"artifact:{uuid7()}#result", row_count=0,
        columns=(), rows=(), truncated=False,
    ),), evidence_gaps=())
    facts, gaps, _ = AIOpsRuntimeService._inspection_report_projection(
        inspection={"evidence_steps": steps, "period_end": NOW.isoformat()}, source=source,
        database_type=db_type, action_tool_ids={},
    )
    fact = next(value for value in facts if value.get("kind") == "inspection_check")
    assert not gaps
    assert fact["check_status"] == item.zero_row_status
    assert fact["evidence_refs"]
    if item.zero_row_status == "FINDING":
        result = assess_inspection([{**fact, "required": True}], policy={})
        assert result["completion_status"] == "COMPLETE"
        assert result["health_status"] == "ATTENTION"
        assert result["completed_required_count"] == result["required_check_count"] == 1
        assert result["findings"] and not result["limitations"]
    elif item.zero_row_status in {"CLEAR", "NOT_APPLICABLE"}:
        assert not fact["findings"]


@pytest.mark.parametrize("error", ["TIMEOUT", "PRIVILEGE_MISSING", "QUERY_SYNTAX_INVALID", "SOURCE_UNREACHABLE"])
def test_failed_query_is_not_a_successful_empty_result(error):
    step = {"tool_id": "db.mysql.connection.utilization", "title": "连接使用率",
            "zero_row_status": "FINDING", "required": True, "check_ids": ["mysql.instance.connections"]}
    facts, gaps, _ = AIOpsRuntimeService._inspection_report_projection(
        inspection={"evidence_steps": [step]}, source=SimpleNamespace(
            evidence=(), evidence_gaps=(TurnEvidenceGap(
                source_id="database", step_id=step["tool_id"], code=error, detail="真实执行失败"),)),
        database_type="MYSQL", action_tool_ids={},
    )
    result = assess_inspection(facts, policy={}, gaps=gaps)
    assert result["completion_status"] == "PARTIAL"
    assert result["health_status"] == "INCOMPLETE"
    assert not result["findings"]


def metric_pipeline(code, gap_code="SOURCE_NO_DATA"):
    target, binding, source_id, artifact = (str(uuid7()) for _ in range(4))
    definition = load_metric_catalog().get(code)
    raw = ObservationSet(
        target_id=target, binding_id=binding, source_id=source_id, collected_at=NOW,
        gaps=(ObservationGap(metric_code=code, source_id=source_id, binding_id=binding,
                             code=gap_code, detail="成功但没有样本" if gap_code == "SOURCE_NO_DATA" else "查询失败"),),
    )
    assessed = asyncio.run(DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
        input_artifacts=({"schema_version": "OBSERVATION_SET.v1", "artifact_id": artifact,
                          "payload": raw.model_dump(mode="json")},),
        plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="USER",
        target_id=target, run_id=str(uuid7()), task_id=str(uuid7()),
    )))
    monitoring = {"current_window": {"start": "2026-11-02T11:45:00+00:00", "end": NOW.isoformat()},
                  "bindings": [{"binding_id": binding, "metrics": [definition.model_dump(mode="json")]}]}
    facts, gaps, _ = AIOpsRuntimeService._inspection_report_projection(
        inspection={"evidence_steps": [{"tool_id": "db.session.blocking_chain", "title": "阻塞",
                                       "zero_row_status": "CLEAR"}]},
        source=SimpleNamespace(evidence=(), evidence_gaps=assessed.gaps), monitoring=monitoring,
        database_type="MYSQL", action_tool_ids={},
    )
    return definition, assessed, facts, gaps


@pytest.mark.parametrize("code", [
    "mysql.availability", "runtime.cpu.utilization", "runtime.memory.utilization",
    "host.cpu.utilization", "host.memory.utilization", "host.filesystem.utilization", "host.disk.io.utilization",
])
def test_successful_empty_monitoring_query_is_a_completed_blind_spot_check(code):
    definition, source, facts, gaps = metric_pipeline(code)
    policy = {"checks": [], "required_metric_roles": {definition.semantic_role: definition.name}}
    checks = monitoring_checks(facts, profile={}, policy=policy)
    result = assess_inspection(checks, policy=policy, gaps=gaps)
    assert result["completion_status"] == "COMPLETE"
    assert result["health_status"] == "ATTENTION"
    assert result["completed_required_count"] == result["required_check_count"] == 1
    assert len(result["findings"]) == 1 and "监控盲区" in result["findings"][0]["summary"]
    assert not result["limitations"]
    assert not any(gap["code"] == "SOURCE_NO_DATA" for gap in gaps)
    absence = next(fact for fact in facts if fact.get("kind") == "inspection_monitoring_absence")
    original_gap = next(gap for gap in source.gaps if gap.code == "SOURCE_NO_DATA")
    assert absence["evidence_ref"] == original_gap.evidence_ref
    assert "#/gaps/0" in absence["evidence_ref"]
    assert all(check.get("latest") is None for check in checks)
    payload = {"report_type": "INSPECTION_DAILY", "title": "巡检报告", "period_start": "2026-11-01T12:00:00Z",
               "period_end": NOW.isoformat(), "facts": [*facts, *checks],
               "scope": {"inspection_assessment": result}, "gaps": gaps}
    frozen = deepcopy(payload)
    presentation = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
    assert presentation["daily_inspection"]["conclusion"]["health_status"] == "ATTENTION"
    assert render_pdf(presentation).startswith(b"%PDF-")
    assert payload == frozen


@pytest.mark.parametrize("error", ["SOURCE_UNREACHABLE", "SOURCE_AUTH_FAILED", "REFERENCE_SOURCE_NO_DATA"])
def test_monitoring_error_or_reference_absence_is_not_a_current_record_risk(error):
    definition, _, facts, gaps = metric_pipeline("runtime.cpu.utilization", error)
    checks = monitoring_checks(facts, profile={}, policy={"required_metric_roles": {definition.semantic_role: "CPU"}})
    result = assess_inspection(checks, policy={}, gaps=gaps)
    assert result["completion_status"] == "PARTIAL" and not result["findings"]


@pytest.mark.parametrize("error", ["SOURCE_UNREACHABLE", "TIMEOUT", "SOURCE_AUTH_FAILED"])
def test_backup_query_error_cannot_claim_no_backup_was_performed(error):
    fact, gaps = backup_policy_result(
        policy={"enabled": True, "source_types": ["MYSQL_LOGICAL_DUMP"], "success_max_age_hours": 26},
        database_type="MYSQL", facts=[], evidence_by_tool={}, reference_time=NOW.isoformat(),
        source_gaps=[{"step_id": "backup.last_success.timestamp", "code": error, "detail": "备份平台查询失败"}],
    )
    result = assess_inspection([fact], policy={}, gaps=gaps)
    assert result["completion_status"] == "PARTIAL" and not result["findings"]
    assert "查询失败" in result["limitations"][0]


def test_missing_backup_record_is_completed_risk_in_daily_pdf_and_monthly_ledger():
    fact, gaps = backup_policy_result(
        policy={"enabled": True, "source_types": ["MYSQL_LOGICAL_DUMP"], "success_max_age_hours": 26},
        database_type="MYSQL", facts=[], evidence_by_tool={}, reference_time=NOW.isoformat(),
    )
    result = assess_inspection([fact], policy={})
    assert not gaps and result["health_status"] == "ATTENTION"
    monthly = monthly_business_review(
        facts=[{**fact, "_source_target_id": "wms", "_source_target_name": "WMS",
                "_source_observed_at": NOW.isoformat(), "_source_run_id": "inspection-1"}],
        period_start=datetime(2026, 11, 1, tzinfo=UTC), period_end=datetime(2026, 12, 1, tzinfo=UTC),
        timezone="UTC",
    )
    assert len(monthly["issues"]) == 1 and "备份成功记录" in monthly["issues"][0]["summary"]
    assert monthly["issues"][0]["lifecycle"] == "PENDING_VERIFICATION"
    payload = {"report_type": "INSPECTION_DAILY", "title": "数据库巡检", "period_start": "2026-11-01T12:00:00Z",
               "period_end": NOW.isoformat(), "facts": [fact], "scope": {"inspection_assessment": result}}
    presentation = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
    assert presentation["daily_inspection"]["conclusion"]["health_status"] == "ATTENTION"
    assert render_pdf(presentation).startswith(b"%PDF-")


def test_arbitrary_unproven_breach_cannot_bypass_evidence_requirement():
    result = assess_inspection([{"kind": "backup_policy_check", "check_status": "BREACHED",
                                "summary": "没有来源核验的断言", "required": True}], policy={})
    assert result["completion_status"] == "PARTIAL"


@pytest.mark.parametrize("truncated", [False, True])
def test_empty_rows_with_query_error_or_truncation_do_not_confirm_record_absence(truncated):
    tool = "db.mysql.connection.utilization"
    source = SimpleNamespace(
        evidence=(SimpleNamespace(tool_id=tool, evidence_ref="original", row_count=0,
                                  columns=(), rows=(), truncated=truncated),),
        evidence_gaps=() if truncated else (TurnEvidenceGap(source_id="database", step_id=tool,
                                                          code="TIMEOUT", detail="查询中断"),),
    )
    facts, gaps, _ = AIOpsRuntimeService._inspection_report_projection(
        inspection={"evidence_steps": [{"tool_id": tool, "zero_row_status": "FINDING", "required": True}]},
        source=source, database_type="MYSQL", action_tool_ids={},
    )
    result = assess_inspection(facts, policy={}, gaps=gaps)
    assert result["completion_status"] == "PARTIAL"
    assert not result["findings"]
