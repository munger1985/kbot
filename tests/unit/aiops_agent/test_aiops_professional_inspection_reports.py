"""专业日报、月报业务判定与统一出版回归。"""

from __future__ import annotations

import asyncio
import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiops_agent.application.inspections.assessment import (
    assess_inspection, compile_standard_steps, freeze_inspection_policy, monitoring_checks,
)
from aiops_agent.application.inspections.coverage import inspection_evidence_coverage
from aiops_agent.application.inspections.monthly import monthly_business_review, merge_monthly_comparison
from aiops_agent.application.inspections.period_sources import inspection_period_sources
from aiops_agent.application.reporting import (
    monthly_inspection_summary, compare_monthly_inspection_summaries,
    report_presentation, resolve_system_template, render_pdf, normalize_report_source,
)
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.turn_answer import AIOpsTurnResult
from platform_core.identity import uuid7


START = datetime(2026, 10, 1, tzinfo=UTC)
END = datetime(2026, 11, 1, tzinfo=UTC)


def policy_checks(database_type="MYSQL"):
    policy = freeze_inspection_policy(database_type)
    checks = [{
        "kind": "inspection_check", "title": spec["title"], "tool_id": spec["tool_id"],
        "check_ids": [spec["check_id"]], "required": spec["required"],
        "check_status": "CLEAR", "summary": "未达到规则条件。",
        "evidence_refs": ["db-evidence"],
    } for spec in policy["checks"]]
    return policy, checks


def metric_facts(cpu=10):
    return [{
        "kind": "inspection_monitoring_metric", "title": title,
        "metric_code": code, "semantic_role": role, "latest": value,
        "average": value, "maximum": value, "unit": unit, "coverage_ratio": 1,
        "evidence_ref": "metric-evidence", "dimensions": "-",
        "window_end": "2026-11-02T10:00:00+00:00",
    } for code, role, title, value, unit in (
        ("mysql.availability", "AVAILABILITY", "数据库可用性", 1, "state"),
        ("runtime.cpu.utilization", "CPU_UTILIZATION", "CPU", cpu, "percent"),
        ("runtime.memory.utilization", "MEMORY_UTILIZATION", "内存", 20, "percent"),
        ("host.filesystem.utilization", "FILESYSTEM_UTILIZATION", "文件系统", 30, "percent"),
        ("mysql.transaction.throughput", "TRANSACTION_THROUGHPUT", "提交速率", 1000, "per_second"),
    )]


PROFILE = {"metric_thresholds": {
    code: {"warning_value": 70, "critical_value": 90, "direction": "ABOVE"}
    for code in ("runtime.cpu.utilization", "runtime.memory.utilization", "host.filesystem.utilization")
}}


def full_assessment(cpu=10):
    policy, checks = policy_checks()
    metrics = metric_facts(cpu)
    checks.extend(monitoring_checks(metrics, profile=PROFILE, policy=policy))
    return policy, checks, assess_inspection(checks, policy=policy)


def assessment_fact(day, health="NORMAL", target="wms", complete=True):
    return {
        "kind": "inspection_assessment", "_source_target_id": target,
        "_source_target_name": target.upper(), "_source_observed_at": day.isoformat(),
        "_source_run_id": f"{target}-{day.isoformat()}",
        "assessment": {"health_status": health, "completion_status": "COMPLETE" if complete else "PARTIAL",
                       "recommendations": []},
    }


def issue_fact(day, target="wms", run="run-1", verification=None):
    finding = {"finding_id": "lock-42", "finding_type": "LOCK_WAIT", "severity": "HIGH",
               "impact": "阻塞影响业务请求。", "object_ref": {"object_name": "42"}}
    if verification:
        finding["verification"] = verification
    return {
        "kind": "inspection_check", "tool_id": "db.session.blocking_chain", "title": "会话阻塞",
        "check_status": "FINDING", "findings": [finding], "_source_run_id": run,
        "_source_target_id": target, "_source_target_name": target.upper(),
        "_source_observed_at": day.isoformat(),
    }


class ProfessionalInspectionReportTest(unittest.TestCase):
    def test_frozen_report_normalization_preserves_business_assessment(self):
        _, _, assessment = full_assessment()
        source = normalize_report_source(
            schema_version="REPORT_CONTENT.v1", source_kind="INSPECTION",
            payload={"status": "READY", "scope": {"inspection_assessment": assessment,
                     "business_system_id": "system", "timezone": "Asia/Shanghai"}},
        )
        self.assertEqual(assessment, source["inspection_assessment"])
        self.assertEqual("Asia/Shanghai", source["timezone"])

    def test_oracle_session_resources_have_a_normal_and_breach_rule(self):
        from aiops_agent.application.diagnosis.findings import compile_findings
        from aiops_agent.contracts.turn_answer import TurnEvidenceFact
        for utilization, expected in ((10, 0), (90, 1)):
            evidence = TurnEvidenceFact.model_validate({
                "artifact_id": str(uuid7()), "presentation_kind": "TABLE",
                "evidence_ref": "session-utilization", "source_id": "db",
                "step_id": "sessions", "tool_id": "db.resource.session_utilization",
                "captured_at": START.isoformat(), "measurement_semantics": "CURRENT_ACTIVITY",
                "columns": [{"name": name} for name in ("resource_name", "current_utilization", "utilization_percent")],
                "rows": [["sessions", utilization, utilization]], "row_count": 1,
            })
            compiled = compile_findings((evidence,), database_type="ORACLE")
            self.assertEqual(expected, len(compiled.findings))
            self.assertEqual([], list(compiled.gaps))
            if not expected:
                self.assertTrue(compiled.empty_reasons)

    def test_finding_status_without_cards_cannot_be_normal(self):
        policy, checks, _ = full_assessment()
        checks[0]["check_status"] = "FINDING"
        result = assess_inspection(checks, policy=policy)
        self.assertEqual("ATTENTION", result["health_status"])

    def test_resolved_previous_issue_is_not_reopened_when_absent(self):
        before = {"schema_version": "MONTHLY_INSPECTION_SUMMARY.v1", "issues": [
            {"finding_id": "resolved", "lifecycle": "VERIFIED_RESOLVED"},
        ]}
        comparison = compare_monthly_inspection_summaries({"issues": [], "metrics": []}, before)
        self.assertEqual([], comparison["pending_verification_issues"])
        self.assertEqual("INCOMPARABLE", comparison["metric_comparison_status"])

    def test_monthly_naive_source_timestamp_is_not_accepted(self):
        fact = issue_fact(START)
        fact["_source_observed_at"] = "2026-10-01T10:00:00"
        result = monthly_business_review(facts=[fact], period_start=START, period_end=END, timezone="UTC")
        self.assertEqual([], result["issues"])
        self.assertEqual("INCOMPLETE", result["conclusion"]["health_status"])

    def test_repository_period_query_keeps_missing_runs_and_scopes_system(self):
        from aiops_agent.repositories.inspection import InspectionRepository
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(all=lambda: [])))
        asyncio.run(InspectionRepository(session, lambda: None).list_system_period_sources(
            domain_id=1, business_system_id=uuid7(), period_start=START, period_end=END,
        ))
        statement = session.execute.await_args.args[0]
        sql = str(statement).upper()
        self.assertIn("LEFT OUTER JOIN", sql)
        for field in ("DOMAIN_ID", "BUSINESS_SYSTEM_ID", "WORKFLOW_KIND", "SCHEDULED_FOR"):
            self.assertIn(field, sql)

    def test_complete_required_checks_are_normal(self):
        _, _, result = full_assessment()
        self.assertEqual("NORMAL", result["health_status"])
        self.assertEqual("COMPLETE", result["completion_status"])
        self.assertIn("数据库运行正常", result["text"])
        self.assertEqual([], result["recommendations"])

    def test_all_engines_have_executable_deduplicated_baselines(self):
        for database_type in ("ORACLE", "MYSQL", "POSTGRESQL"):
            with self.subTest(database_type=database_type):
                policy = freeze_inspection_policy(database_type)
                steps = compile_standard_steps((), database_type=database_type, schedule_type="DAILY", policy=policy)
                self.assertGreater(len(steps), 3)
                self.assertEqual(len(steps), len({item["tool_id"] for item in steps}))
                self.assertTrue(all(item["supported_db_types"] == [database_type] for item in steps))

    def test_single_tool_cannot_be_a_complete_normal_inspection(self):
        policy, checks = policy_checks()
        result = assess_inspection(checks[:1], policy=policy)
        self.assertEqual("INCOMPLETE", result["health_status"])
        self.assertTrue(result["limitations"])

    def test_explicit_critical_finding_survives_another_missing_check(self):
        policy, checks, _ = full_assessment(cpu=95)
        checks[0]["check_status"] = "MISSING"
        result = assess_inspection(checks, policy=policy)
        self.assertEqual("ABNORMAL", result["health_status"])
        self.assertEqual("PARTIAL", result["completion_status"])
        self.assertTrue(result["findings"])
        self.assertTrue(result["limitations"])

    def test_warning_uses_profile_not_relative_throughput(self):
        _, _, result = full_assessment(cpu=75)
        self.assertEqual("ATTENTION", result["health_status"])
        self.assertEqual(1, len(result["findings"]))
        self.assertNotIn("提交速率", result["findings"][0]["title"])

    def test_optional_gap_does_not_invalidate_completed_required_checks(self):
        policy, checks, _ = full_assessment()
        result = assess_inspection(checks, policy=policy, gaps=(
            {"step_id": "optional.source", "detail": "可选来源未接入"},
        ))
        self.assertEqual("NORMAL", result["health_status"])

    def test_required_observation_without_rule_does_not_mean_normal(self):
        policy, checks, _ = full_assessment()
        checks[0]["check_status"] = "OBSERVED"
        self.assertEqual("INCOMPLETE", assess_inspection(checks, policy=policy)["health_status"])

    def test_required_truncated_or_untraceable_check_cannot_be_normal(self):
        for key, value in (("truncated", True), ("evidence_refs", [])):
            policy, checks, _ = full_assessment()
            checks[0][key] = value
            self.assertEqual("INCOMPLETE", assess_inspection(checks, policy=policy)["health_status"])

    def test_nan_is_missing_not_normal(self):
        policy, _ = policy_checks()
        metrics = metric_facts(float("nan"))
        checked = monitoring_checks(metrics, profile=PROFILE, policy=policy)
        cpu = next(item for item in checked if item.get("semantic_role") == "CPU_UTILIZATION")
        self.assertEqual("MISSING", cpu["check_status"])

    def test_low_sampling_exposes_limitation_and_preserves_known_problem(self):
        policy, checks = policy_checks()
        metrics = metric_facts(95)
        metrics[1]["coverage_ratio"] = .2
        checks.extend(monitoring_checks(metrics, profile=PROFILE, policy=policy))
        result = assess_inspection(checks, policy=policy)
        self.assertEqual("ABNORMAL", result["health_status"])
        self.assertEqual("PARTIAL", result["completion_status"])

    def test_backup_gap_is_separate_from_log_chain_coverage(self):
        source = AIOpsTurnResult.model_validate({
            "status": "COMPLETED",
            "sufficiency_status": "ANSWERABLE", "blocks": [{
                "block_type": "MARKDOWN", "schema_version": "AIOPS_MARKDOWN_BLOCK.v1",
                "payload": {"markdown": "检查完成。"}, "evidence_refs": [],
            }], "evidence": [],
        })
        facts, gaps, _ = AIOpsRuntimeService._inspection_report_projection(
            inspection={"database_type": "MYSQL", "evidence_steps": [],
                        "backup_policy": {"enabled": True, "success_max_age_hours": 24}},
            source=source, database_type="MYSQL", action_tool_ids={},
        )
        backup_gap = next(item for item in gaps if item["code"] == "BACKUP_VERIFICATION_FAILED")
        self.assertEqual("inspection.backup-policy", backup_gap["step_id"])
        check = {"tool_id": "db.mysql.binlog.status", "check_ids": ["mysql.binlog.status"],
                 "check_status": "CLEAR", "evidence_refs": ["binlog"]}
        coverage = inspection_evidence_coverage((check,), (backup_gap,))
        self.assertEqual(100, coverage["evidence_coverage_percent"])
        backup = next(item for item in facts if item.get("kind") == "backup_policy_check")
        self.assertEqual("inspection.backup-policy", backup["tool_id"])

    def test_backup_missing_is_an_explicit_required_limitation(self):
        policy, checks, _ = full_assessment()
        checks.append({"kind": "backup_policy_check", "title": "备份成功核验",
                       "check_status": "NOT_OBSERVABLE", "tool_id": "inspection.backup-policy",
                       "summary": "未接入备份成功来源。"})
        result = assess_inspection(checks, policy=policy)
        self.assertEqual("INCOMPLETE", result["health_status"])
        self.assertIn("备份成功", result["limitations"][0])
        self.assertIn("Binlog", " ".join(result["recommendations"]))

    def test_daily_projection_uses_frozen_assessment_not_editor_or_live_rules(self):
        policy, checks, assessment = full_assessment(cpu=95)
        payload = {
            "report_type": "INSPECTION_DAILY", "period_start": "2026-11-01T00:00:00+00:00",
            "period_end": "2026-11-02T00:00:00+00:00", "title": "数据库日常巡检报告",
            "scope": {"inspection_assessment": assessment, "report_policy": policy,
                      "timezone": "Asia/Shanghai", "target": {"display_name": "WMS"}},
            "facts": [*checks, *metric_facts(95)], "gaps": [], "evidence_refs": [],
            "presentation_overrides": {"EXECUTIVE_SUMMARY": ["数据库全部正常"]},
        }
        with patch("aiops_agent.application.inspections.assessment.report_policy", side_effect=AssertionError("不能读取当前规则")):
            presentation = report_presentation(payload=payload, template=resolve_system_template("system:inspection.daily"))
        daily = presentation["daily_inspection"]
        self.assertEqual("ABNORMAL", daily["conclusion"]["health_status"])
        self.assertNotIn("全部正常", daily["conclusion"]["text"])
        self.assertNotIn("最大可计算偏离", daily["conclusion"]["text"])
        self.assertEqual(payload["period_start"], daily["header"]["observation_start"])
        self.assertTrue(daily["checks"])
        self.assertTrue(render_pdf(presentation).startswith(b"%PDF-"))

    def test_evidence_appendix_labels_identifiers_without_changing_frozen_values(self):
        from aiops_agent.application.inspection_pdf import inspection_pdf_story
        from reportlab.lib.styles import getSampleStyleSheet

        _, _, assessment = full_assessment()
        artifact_id, content_hash = str(uuid7()), "a" * 64
        styles = getSampleStyleSheet()
        for period in ("daily", "monthly"):
            with self.subTest(period=period):
                payload = {
                    "report_type": f"INSPECTION_{period.upper()}",
                    "period_start": START.isoformat(), "period_end": END.isoformat(),
                    "scope": {"inspection_assessment": assessment},
                    "facts": [], "gaps": [],
                    "evidence_refs": [
                        {"artifact_id": artifact_id, "content_hash": content_hash},
                        {"artifact_id": "missing-hash"},
                        {"content_hash": ""},
                    ],
                }
                frozen = deepcopy(payload)
                presentation = report_presentation(
                    payload=payload, template=resolve_system_template(f"system:inspection.{period}"),
                )
                data = presentation[f"{period}_inspection"]
                self.assertEqual([
                    f"证据编号：{artifact_id}\n内容指纹（Hash）：{content_hash}",
                    "证据编号：missing-hash\n内容指纹（Hash）：未提供哈希",
                    "证据编号：未记录\n内容指纹（Hash）：未提供哈希",
                ], data["evidence_boundary"]["sources"])
                self.assertEqual(frozen, payload)
                story = inspection_pdf_story(
                    data=data, monthly=period == "monthly", font_name="Helvetica",
                    body=styles["BodyText"], section=styles["Heading2"], cover_title=styles["Title"],
                )
                text = " ".join(item.getPlainText() for item in story if hasattr(item, "getPlainText"))
                for label in ("技术追溯信息", "证据编号用于定位原始记录", "不属于巡检问题", artifact_id, content_hash):
                    self.assertIn(label, text)
                self.assertTrue(render_pdf(presentation).startswith(b"%PDF-"))
                payload["evidence_refs"] = []
                empty = report_presentation(
                    payload=payload, template=resolve_system_template(f"system:inspection.{period}"),
                )
                self.assertEqual([], empty[f"{period}_inspection"]["evidence_boundary"]["sources"])

    def test_monthly_problem_identity_isolated_per_database(self):
        facts = [issue_fact(START+timedelta(days=1), target=target) for target in ("wms", "erp")]
        result = monthly_business_review(facts=facts, period_start=START, period_end=END, timezone="UTC")
        self.assertEqual(2, len(result["issues"]))
        self.assertEqual(2, len({item["finding_id"] for item in result["issues"]}))

    def test_monthly_observations_are_not_incidents(self):
        fact = issue_fact(START+timedelta(days=1))
        result = monthly_business_review(facts=[fact, deepcopy(fact)], period_start=START, period_end=END, timezone="UTC")
        self.assertEqual(1, result["issues"][0]["observation_count"])
        self.assertIsNone(result["issues"][0]["occurrence_count"])
        self.assertEqual("PENDING_VERIFICATION", result["issues"][0]["lifecycle"])

    def test_later_normal_check_does_not_resolve_prior_issue(self):
        facts = [issue_fact(START+timedelta(days=1)),
                 assessment_fact(START+timedelta(days=30), health="NORMAL")]
        result = monthly_business_review(facts=facts, period_start=START, period_end=END, timezone="UTC")
        self.assertEqual("ABNORMAL", result["conclusion"]["health_status"])
        self.assertEqual("NORMAL", result["target_results"][0]["latest_health_status"])
        self.assertEqual("PENDING_VERIFICATION", result["issues"][0]["lifecycle"])

    def test_only_dated_verified_evidence_can_resolve_an_issue(self):
        day = START+timedelta(days=1)
        verification = {"result": "RESOLVED", "evidence_refs": ["after"], "verified_at": day.isoformat()}
        result = monthly_business_review(facts=[issue_fact(day, verification=verification)], period_start=START, period_end=END, timezone="UTC")
        self.assertEqual("VERIFIED_RESOLVED", result["issues"][0]["lifecycle"])

    def test_recurrence_after_verified_recovery_reopens_issue(self):
        day = START+timedelta(days=1)
        verification = {"result": "RESOLVED", "evidence_refs": ["after"], "verified_at": day.isoformat()}
        facts = [issue_fact(day, verification=verification), issue_fact(day+timedelta(days=1), run="run-2")]
        result = monthly_business_review(facts=facts, period_start=START, period_end=END, timezone="UTC")
        self.assertEqual("PENDING_VERIFICATION", result["issues"][0]["lifecycle"])

    def test_month_without_records_is_not_normal(self):
        result = monthly_business_review(facts=[], period_start=START, period_end=END, timezone="UTC",
                                         members=[{"target_id": "wms", "display_name": "WMS"}])
        self.assertEqual("INCOMPLETE", result["conclusion"]["health_status"])
        self.assertEqual(1, len(result["target_results"]))

    def test_full_month_of_required_checks_can_be_normal(self):
        facts = [assessment_fact(START+timedelta(days=day)) for day in range(31)]
        result = monthly_business_review(facts=facts, period_start=START, period_end=END, timezone="UTC")
        self.assertEqual("NORMAL", result["conclusion"]["health_status"])

    def test_absent_previous_issue_requires_verification_not_resolved(self):
        previous = {"schema_version": "MONTHLY_INSPECTION_SUMMARY.v1", "metrics": [],
                    "issues": [{"finding_id": "old", "title": "旧问题", "target_id": "wms", "severity": "HIGH"}]}
        current = {"schema_version": "MONTHLY_INSPECTION_SUMMARY.v1", "metrics": [], "issues": [],
                   "conclusion": {"health_status": "NORMAL", "text": "本月检查正常。"}}
        comparison = compare_monthly_inspection_summaries(current, previous)
        self.assertEqual([], comparison["resolved_issues"])
        self.assertEqual(1, len(comparison["pending_verification_issues"]))
        combined = merge_monthly_comparison(current, comparison)
        self.assertEqual("ATTENTION", combined["conclusion"]["health_status"])
        self.assertEqual(1, combined["conclusion"]["issue_count"])

    def test_monthly_statistics_keep_target_and_sampling_granularity(self):
        facts = [{
            "kind": "inspection_monitoring_metric", "metric_code": "mysql.availability",
            "semantic_role": "AVAILABILITY", "value_kind": "STATE", "unit": "state",
            "_source_target_id": target, "_source_target_name": target,
            "_source_observed_at": "2026-10-02T00:00:00+00:00",
            "daily_points_timezone": "UTC",
            "daily_points": [{"date": "2026-10-01", "value": value}],
        } for target, value in (("wms", 1), ("erp", .5))]
        result = monthly_inspection_summary(facts=facts, period_start=START, period_end=END)
        self.assertEqual(2, len(result["metrics"]))
        self.assertTrue(all(item["statistic_basis"] == "DAILY_AGGREGATES" for item in result["metrics"]))
        self.assertTrue(all("状态日均" in item["monthly_average_display"] for item in result["metrics"]))
        self.assertTrue(all("日均最小" in item["minimum_display"] for item in result["metrics"]))

    def test_monthly_samples_use_report_timezone_and_half_open_month(self):
        start = datetime(2026, 9, 30, 16, tzinfo=UTC)
        end = datetime(2026, 10, 31, 16, tzinfo=UTC)
        fact = {
            "kind": "inspection_monitoring_metric", "metric_code": "runtime.cpu.utilization",
            "unit": "percent", "_source_target_id": "wms",
            "sample_points": [
                {"observed_at": "2026-09-30T15:59:59Z", "value": 900},
                {"observed_at": "2026-09-30T16:00:00Z", "value": 10},
                {"observed_at": "2026-10-01T15:00:00Z", "value": 30},
                {"observed_at": "2026-10-01T16:00:00Z", "value": 40},
                {"observed_at": "2026-10-31T16:00:00Z", "value": 900},
            ],
        }
        result = monthly_inspection_summary(
            facts=[fact, deepcopy(fact)], period_start=start, period_end=end, timezone="Asia/Shanghai",
        )
        points = result["metrics"][0]["daily_points"]
        self.assertEqual(["2026-10-01", "2026-10-02"], [point["date"] for point in points])
        self.assertEqual([20, 40], [point["value"] for point in points])
        self.assertEqual([2, 1], [point["sample_count"] for point in points])

    def test_monthly_rejects_aggregates_in_another_timezone(self):
        fact = {
            "kind": "inspection_monitoring_metric", "metric_code": "runtime.cpu.utilization",
            "daily_points_timezone": "UTC", "daily_points": [{"date": "2026-10-01", "value": 30}],
        }
        result = monthly_inspection_summary(
            facts=[fact], period_start=START, period_end=END, timezone="Asia/Shanghai",
        )
        self.assertEqual([], result["metrics"])
        self.assertTrue(any("日聚合" in item for item in result["limitations"]))

    def test_monthly_excludes_conflicting_overlapping_samples(self):
        fact = {
            "kind": "inspection_monitoring_metric", "metric_code": "runtime.cpu.utilization",
            "sample_points": [{"observed_at": "2026-10-01T01:00:00Z", "value": 30}],
        }
        conflict = {**fact, "sample_points": [
            {"observed_at": "2026-10-01T01:00:00Z", "value": 50},
            {"observed_at": "2026-10-01T02:00:00Z", "value": 10},
            {"observed_at": "2026-10-01T03:00:00", "value": 800},
            {"observed_at": "2026-10-01T04:00:00Z", "value": "NaN"},
        ]}
        result = monthly_inspection_summary(facts=[fact, conflict], period_start=START, period_end=END)
        self.assertEqual(10, result["metrics"][0]["monthly_average"])
        self.assertEqual(1, result["metrics"][0]["daily_points"][0]["sample_count"])
        self.assertTrue(any("冲突采样" in item for item in result["limitations"]))

    def test_daily_report_freezes_timestamped_monitoring_samples(self):
        evidence = SimpleNamespace(
            tool_id="metric.query_range", columns=({"name": "metric_code"}, {"name": "latest"}),
            rows=(("runtime.cpu.utilization", 10),), evidence_ref="metric-evidence",
            series_computations=(SimpleNamespace(
                row_index=0, samples=((START, 10), (START + timedelta(hours=1), 20)),
                effective_step_seconds=3600, dimensions=(),
            ),),
        )
        fact = AIOpsRuntimeService._inspection_monitoring_facts(SimpleNamespace(evidence=(evidence,)))[0]
        self.assertEqual("UTC", fact["daily_points_timezone"])
        self.assertEqual(START.isoformat(), fact["sample_points"][0]["observed_at"])
        self.assertEqual("10", fact["sample_points"][0]["value"])
        self.assertEqual(3600, fact["effective_step_seconds"])

    def test_incomplete_recommendations_follow_error_codes_not_error_text(self):
        for code, expected in (
            ("RESOURCE_QUOTA_TARGET_EXHAUSTED", "等待共享只读额度"),
            ("QUERY_SYNTAX_INVALID", "参数绑定"),
            ("PRIVILEGE_MISSING", "最小只读权限"),
            ("EXECUTOR_INTERNAL_ERROR", "执行器"),
            ("TARGET_UNREACHABLE", "目标连接"),
        ):
            policy, checks, _ = full_assessment()
            check = next(item for item in checks if item.get("required"))
            check["check_status"] = "MISSING"
            result = assess_inspection(checks, policy=policy, gaps=[
                {"step_id": check["tool_id"], "code": code, "detail": "补权限（该文字不参与分类）"},
            ])
            self.assertTrue(any(expected in action for action in result["recommendations"]), code)
            self.assertFalse(any("采集来源或判断规则" in action for action in result["recommendations"]))

    def test_monthly_scoped_backup_problem_is_not_filtered(self):
        fact = issue_fact(START+timedelta(days=1))
        fact["findings"][0]["finding_type"] = "BACKUP_FAILED"
        result = monthly_inspection_summary(facts=[fact], period_start=START, period_end=END)
        self.assertEqual(1, len(result["issues"]))

    def test_system_period_scope_uses_frozen_members_including_missing_run(self):
        system, target, absent = uuid7(), uuid7(), uuid7()
        fire = SimpleNamespace(inspection_fire_id=uuid7(), scheduled_for=START,
            plan_snapshot_json={"system_scope": {"business_system_name": "ERP",
                "target_ids": [str(target), str(absent)],
                "resources": [{"target_id": str(target), "display_name": "ERP", "db_type": "MYSQL"},
                              {"target_id": str(absent), "display_name": "WMS", "db_type": "MYSQL"}]}})
        run = SimpleNamespace(ops_run_id=uuid7(), target_id=target, plan_snapshot_json={})
        uow = SimpleNamespace(inspections=SimpleNamespace(list_system_period_sources=AsyncMock(return_value=[(fire, run)])))
        runs, scope, missing = asyncio.run(inspection_period_sources(
            uow=uow, run=run, inspection={"business_system_id": str(system)},
            domain_id=1, period_start=START, period_end=END))
        self.assertEqual([run], runs)
        self.assertEqual(2, scope["target_count"])
        self.assertEqual(str(absent), missing[0]["target_id"])


if __name__ == "__main__":
    unittest.main()
