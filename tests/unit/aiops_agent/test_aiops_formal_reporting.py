"""正式报告模板、展示投影与 PDF 渲染的无数据库测试。"""

from __future__ import annotations

import unittest
import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.report_lifecycle import report_input_hash
from aiops_agent.application.reporting import (
    REPORT_TYPE_SYSTEM_TEMPLATES,
    SYSTEM_REPORT_TEMPLATES,
    closed_period_window,
    compare_monthly_inspection_summaries,
    monthly_inspection_summary,
    render_pdf,
    report_presentation,
    normalize_report_source,
    resolve_historical_report_template,
    resolve_report_template_reference,
    validate_template_definition,
)
from aiops_agent.application.runtime.service import (
    AIOpsRuntimeService,
    sha256_json,
)
from platform_core.identity import uuid7


class FormalReportingTest(unittest.TestCase):
    def test_monthly_presentation_uses_frozen_summary_and_pdf(self) -> None:
        summary = {
            "schema_version": "MONTHLY_INSPECTION_SUMMARY.v1",
            "metrics": [{
                "metric_key": "db.availability|-|AVAILABILITY|percent",
                "metric_code": "db.availability", "semantic_role": "AVAILABILITY",
                "display_name": "数据库可用率", "dimension_label": "全部",
                "unit": "percent", "monthly_average": 99.9,
                "monthly_average_display": "99.90%", "p95": 100.0,
                "p95_display": "100.00%", "minimum": 99.0,
                "minimum_display": "99.00%", "maximum": 100.0,
                "maximum_display": "100.00%", "first": 100.0,
                "first_display": "100.00%", "last": 99.9,
                "last_display": "99.90%", "net_change": -0.1,
                "net_change_display": "-0.10%", "change_percent": -0.1,
                "effective_days": 28, "expected_days": 28,
                "coverage_percent": 100.0, "daily_points": [],
                "is_capacity": False,
            }],
            "issues": [], "database_checks": {"status_counts": {"CLEAR": 8}},
            "coverage": {"completed_run_count": 28, "metric_day_coverage_percent": 100.0},
        }
        payload = {
            "report_key": "inspection.monthly", "report_type": "INSPECTION_MONTHLY",
            "target_id": str(uuid7()), "title": "数据库月度巡检报告",
            "status": "READY", "summary": "月度巡检完成。",
            "period_start": "2026-02-01T00:00:00+00:00",
            "period_end": "2026-03-01T00:00:00+00:00",
            "scope": {
                "monthly_summary": summary,
                "previous_month_comparison": {"status": "NO_PREVIOUS_REPORT", "status_display": "无上月报告"},
                "scope_type": "TARGET", "scope_name": "Oracle 生产库", "target_count": 1,
                "database_types": ["ORACLE"],
            },
            "facts": [], "gaps": [], "evidence_refs": [], "recommendations": [],
        }
        presentation = report_presentation(
            payload=payload,
            template=SYSTEM_REPORT_TEMPLATES["system:inspection.monthly"],
            report_context={"report_id": str(uuid7()), "report_version": 1},
        )
        self.assertEqual("MONTHLY_INSPECTION_PRESENTATION.v1", presentation["monthly_inspection"]["schema_version"])
        self.assertNotIn("current", presentation["monthly_inspection"]["metrics"][0])
        self.assertGreater(len(render_pdf(presentation)), 1_000)

    def test_monthly_summary_deduplicates_days_and_excludes_backup(self) -> None:
        start = datetime(2026, 1, 31, 16, tzinfo=UTC)
        end = datetime(2026, 2, 28, 16, tzinfo=UTC)
        base = {
            "kind": "inspection_monitoring_metric",
            "metric_code": "db.storage.used_bytes",
            "semantic_role": "STORAGE_USED",
            "dimensions": "tablespace=USERS",
            "unit": "bytes",
        }
        facts = [
            {**base, "_source_observed_at": "2026-02-03T00:00:00+00:00", "daily_points": [
                {"date": "2026-02-01", "value": 10}, {"date": "2026-02-02", "value": 20},
            ]},
            {**base, "_source_observed_at": "2026-02-04T00:00:00+00:00", "daily_points": [
                {"date": "2026-02-02", "value": 30}, {"date": "2026-02-03", "value": 50},
            ]},
            {"kind": "inspection_check", "tool_id": "db.oracle.tablespace", "title": "表空间检查", "check_status": "FINDING", "_source_observed_at": "2026-02-03T00:00:00+00:00", "findings": [
                {"finding_id": "TABLESPACE_GROWTH", "severity": "MEDIUM", "summary": "增长较快"},
            ]},
            {"kind": "inspection_check", "tool_id": "db.oracle.rman_backup", "check_status": "FINDING", "findings": [
                {"finding_id": "BACKUP", "finding_type": "BACKUP_FAILED", "severity": "HIGH"},
            ]},
        ]
        summary = monthly_inspection_summary(
            facts=facts, period_start=start, period_end=end,
            timezone="Asia/Shanghai", run_summary={"completed_run_count": 2},
        )
        metric = summary["metrics"][0]
        self.assertEqual(30, metric["monthly_average"])
        self.assertEqual(3, metric["effective_days"])
        self.assertEqual(28, metric["expected_days"])
        self.assertEqual(40, metric["net_change"])
        self.assertEqual(["TABLESPACE_GROWTH"], [item["finding_id"] for item in summary["issues"]])
        self.assertEqual(
            "NO_PREVIOUS_REPORT",
            compare_monthly_inspection_summaries(summary, None)["status"],
        )

    def test_system_templates_cover_requested_periods(self) -> None:
        self.assertEqual(
            {
                "system:diagnosis.standard",
                "system:inspection.daily",
                "system:inspection.weekly",
                "system:inspection.custom",
                "system:inspection.monthly",
                "system:inspection.quarterly",
                "system:inspection.annual",
                "system:comparison.standard",
                "system:mysql.workload",
                "system:oracle.workload",
                "system:comparison.business",
                "system:oracle.workload_diff",
                "system:mysql.workload_diff",
                "system:mysql.activity",
                "system:mysql.sql_healthcheck",
                "system:postgresql.workload",
                "system:postgresql.workload_diff",
                "system:postgresql.activity",
                "system:postgresql.sql_healthcheck",
                "system:postgresql.pgbadger",
            },
            set(SYSTEM_REPORT_TEMPLATES),
        )
        self.assertEqual(
            {
                "INCIDENT",
                "PERFORMANCE",
                "INSPECTION_DAILY",
                "INSPECTION_WEEKLY",
                "INSPECTION_CUSTOM",
                "INSPECTION_MONTHLY",
                "INSPECTION_QUARTERLY",
                "INSPECTION_ANNUAL",
                "COMPARISON",
                "MYSQL_WORKLOAD",
                "ORACLE_WORKLOAD",
                "ORACLE_WORKLOAD_DIFF",
                "MYSQL_WORKLOAD_DIFF",
                "MYSQL_ACTIVITY",
                "MYSQL_SQL_HEALTHCHECK",
                "POSTGRESQL_WORKLOAD",
                "POSTGRESQL_WORKLOAD_DIFF",
                "POSTGRESQL_ACTIVITY",
                "POSTGRESQL_SQL_HEALTHCHECK",
                "POSTGRESQL_PGBADGER",
            },
            set(REPORT_TYPE_SYSTEM_TEMPLATES),
        )

    def test_historical_system_template_reference_is_resolvable(self) -> None:
        self.assertEqual(
            SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"],
            resolve_report_template_reference("diagnosis.standard"),
        )

    def test_historical_daily_inspection_template_is_resolvable(self) -> None:
        self.assertEqual(
            SYSTEM_REPORT_TEMPLATES["system:inspection.daily"],
            resolve_historical_report_template(
                template_ref="01a00000-0000-7000-8000-000000000000",
                report_type="INSPECTION_DAILY",
            ),
        )

    def test_historical_weekly_custom_and_comparison_templates_are_resolvable(self) -> None:
        cases = {
            "INSPECTION_WEEKLY": "system:inspection.weekly",
            "INSPECTION_CUSTOM": "system:inspection.custom",
            "COMPARISON": "system:comparison.standard",
        }
        for report_type, template_ref in cases.items():
            self.assertEqual(
                SYSTEM_REPORT_TEMPLATES[template_ref],
                resolve_historical_report_template(
                    template_ref="01a00000-0000-7000-8000-000000000000",
                    report_type=report_type,
                ),
            )

    def test_custom_template_cannot_hide_evidence_boundary(self) -> None:
        with self.assertRaises(AIOpsApplicationError):
            validate_template_definition({
                "sections": [{"kind": "EXECUTIVE_SUMMARY"}],
            })

    def test_presentation_and_pdf_share_the_frozen_template(self) -> None:
        template = SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"]
        presentation = report_presentation(
            template=template,
            payload={
                "title": "数据库故障诊断报告",
                "status": "PARTIAL",
                "summary": "已确认锁等待持续升高。",
                "period_start": "2026-09-01T00:00:00+00:00",
                "period_end": "2026-09-01T01:00:00+00:00",
                "scope": {"root_cause_grade": "PROBABLE"},
                "facts": [{"summary": "锁等待持续升高"}],
                "gaps": [
                    {"code": "MISSING_ASH"},
                    {"code": "MISSING_FINAL_RESULT", "turn_no": 3},
                ],
                "recommendations": ["确认阻塞会话"],
                "evidence_refs": [{"artifact_id": "evidence-1", "content_hash": "a" * 64}],
            },
        )
        sections = {
            item["kind"]: item for item in presentation["sections"]
        }
        kinds = list(sections)
        self.assertIn("EVIDENCE_BOUNDARY", kinds)
        self.assertEqual("部分完成", presentation["status_display"])
        self.assertEqual(
            "证据边界", sections["EVIDENCE_BOUNDARY"]["display_name"]
        )
        self.assertEqual(
            ["缺少 ASH 历史会话数据", "第 3 轮：缺少最终诊断结果"],
            sections["EVIDENCE_BOUNDARY"]["items"],
        )
        self.assertIn(
            "根因评估等级：很可能",
            sections["ROOT_CAUSE"]["items"],
        )
        visible_text = str([
            (item["display_name"], item["items"])
            for item in presentation["sections"]
        ])
        for internal_code in (
            "PROBABLE",
            "MISSING_ASH",
            "MISSING_FINAL_RESULT",
            "EVIDENCE_BOUNDARY",
            "EVIDENCE_APPENDIX",
        ):
            self.assertNotIn(internal_code, visible_text)
        pdf = render_pdf(presentation)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        # 标准生成器必须嵌入字体和 Unicode 映射，不能依赖阅读器安装中文字体。
        self.assertIn(b"/FontFile2", pdf)
        self.assertNotIn(b"FEFF", pdf)
        self.assertIn(b"/ToUnicode", pdf)

    def test_inspection_presentation_keeps_key_observations_and_gap_category(
        self,
    ) -> None:
        presentation = report_presentation(
            template=SYSTEM_REPORT_TEMPLATES["system:inspection.daily"],
            payload={
                "title": "数据库日常巡检报告",
                "status": "PARTIAL",
                "summary": "检查已完成。",
                "facts": [{
                    "kind": "inspection_check",
                    "summary": "归档配置存在恢复风险。",
                    "observations": [{"log_mode": "NOARCHIVELOG"}],
                }],
                "gaps": [{
                    "code": "PRIVILEGE_MISSING",
                    "detail": "缺少最小只读权限",
                    "category": "SOURCE_ACCESS",
                }],
            },
        )
        sections = {item["kind"]: item for item in presentation["sections"]}

        self.assertIn(
            "关键观测：日志模式：NOARCHIVELOG",
            sections["FINDINGS"]["items"][0],
        )
        self.assertTrue(
            sections["EVIDENCE_BOUNDARY"]["items"][0].startswith(
                "[数据库访问]"
            )
        )

    def test_daily_inspection_projects_seven_day_baseline_and_pdf(self) -> None:
        facts = [
            {
                "kind": "inspection_monitoring_metric",
                "metric_code": "db.response.latency",
                "semantic_role": "RESPONSE_LATENCY",
                "value_kind": "GAUGE",
                "dimensions": "instance=oracle-1, job=oracle",
                "latest": 38.0,
                "average": 28.0,
                "unit": "milliseconds",
                "coverage_ratio": 0.94,
                "window_start": "2026-10-13T08:30:00+00:00",
                "window_end": "2026-10-20T08:30:00+00:00",
                "sample_count": 168,
                "daily_points": [
                    {"date": f"2026-10-{day:02d}", "value": value}
                    for day, value in zip(
                        range(14, 21),
                        (25, 27, 26, 29, 28, 30, 38),
                        strict=True,
                    )
                ],
            },
            {
                "kind": "inspection_monitoring_metric",
                "metric_code": "db.storage.used_bytes",
                "semantic_role": "STORAGE_USED",
                "value_kind": "GAUGE",
                "dimensions": "tablespace=SYSTEM, type=PERMANENT",
                "latest": 82 * 1024**3,
                "average": 80 * 1024**3,
                "unit": "bytes",
                "coverage_ratio": 0.91,
                "window_start": "2026-10-13T08:30:00+00:00",
                "window_end": "2026-10-20T08:30:00+00:00",
                "sample_count": 168,
                "daily_points": [],
            },
            {
                "kind": "inspection_monitoring_metric",
                "metric_code": "host.cpu.utilization",
                "semantic_role": "CPU_UTILIZATION",
                "value_kind": "GAUGE",
                "dimensions": "instance=oracle-1",
                "latest": 45.0,
                "average": 40.0,
                "unit": "percent",
                "coverage_ratio": 0.72,
                "window_start": "2026-10-13T08:30:00+00:00",
                "window_end": "2026-10-20T08:30:00+00:00",
                "sample_count": 101,
                "daily_points": [],
            },
            {
                "kind": "inspection_monitoring_metric",
                "metric_code": "db.availability",
                "semantic_role": "AVAILABILITY",
                "value_kind": "STATE",
                "dimensions": "instance=oracle-1",
                "latest": 1.0,
                "average": 1.0,
                "unit": "state",
                "coverage_ratio": 1.0,
                "window_start": "2026-10-13T08:30:00+00:00",
                "window_end": "2026-10-20T08:30:00+00:00",
                "sample_count": 168,
                "daily_points": [],
            },
        ]
        presentation = report_presentation(
            template=SYSTEM_REPORT_TEMPLATES["system:inspection.daily"],
            report_context={
                "report_id": "01a1580c-42e5-72a2-b19b-9c73c77a5fff",
                "report_version": 2,
                "created_at": "2026-10-20T08:36:00+00:00",
            },
            payload={
                "report_type": "INSPECTION_DAILY",
                "report_key": "inspection.daily",
                "target_id": "target-1",
                "title": "数据库日常巡检报告",
                "status": "READY",
                "summary": "巡检完成。",
                "period_start": "2026-10-20T00:00:00+00:00",
                "period_end": "2026-10-20T08:30:00+00:00",
                "scope": {
                    "health_status": "MEDIUM",
                    "evidence_coverage_percent": 100.0,
                    "monitoring_sample_coverage_percent": 72.0,
                    "evidence_total_count": 4,
                    "target": {
                        "target_id": "target-1",
                        "display_name": "Oracle 生产库",
                        "database_type": "ORACLE",
                        "version_code": "19c",
                        "environment": "PROD",
                        "db_role": "PRIMARY",
                    },
                },
                "facts": facts,
                "gaps": (),
                "recommendations": ("复核响应延迟偏离。",),
                "evidence_refs": ({
                    "artifact_id": "artifact-1",
                    "content_hash": "a" * 64,
                },),
            },
        )

        daily = presentation["daily_inspection"]
        metrics = {
            item["metric_code"]: item
            for item in daily["baseline_metrics"]
        }
        self.assertAlmostEqual(
            35.71,
            metrics["db.response.latency"]["deviation_percent"],
            places=2,
        )
        self.assertEqual(
            "AVAILABLE",
            metrics["db.response.latency"]["baseline_status"],
        )
        self.assertEqual(
            "INSUFFICIENT",
            metrics["host.cpu.utilization"]["baseline_status"],
        )
        self.assertEqual(
            "INSUFFICIENT",
            metrics["db.availability"]["baseline_status"],
        )
        self.assertEqual(1, len(daily["capacity_rows"]))
        self.assertNotIn(
            "instance=oracle-1",
            daily["baseline_metrics"][0]["dimension_label"],
        )
        pdf = render_pdf(presentation)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertIn(b"/ToUnicode", pdf)

    def test_daily_inspection_replaces_legacy_uuid_target_name(self) -> None:
        target_id = str(uuid7())
        payload = {
            "report_type": "INSPECTION_DAILY",
            "target_id": target_id,
            "title": "数据库日常巡检报告",
            "status": "READY",
            "scope": {"target": {
                "target_id": target_id,
                "display_name": target_id,
            }},
            "facts": [],
            "gaps": [],
            "evidence_refs": [],
        }
        presentation = report_presentation(
            template=SYSTEM_REPORT_TEMPLATES["system:inspection.daily"],
            report_context={"target_display_name": "Oracle 核心生产库"},
            payload=payload,
        )

        self.assertEqual(
            "Oracle 核心生产库",
            presentation["daily_inspection"]["header"]["target_name"],
        )

        payload["scope"]["target"]["display_name"] = "Oracle 历史名称"
        preserved = report_presentation(
            template=SYSTEM_REPORT_TEMPLATES["system:inspection.daily"],
            report_context={"target_display_name": "Oracle 当前名称"},
            payload=payload,
        )
        self.assertEqual(
            "Oracle 历史名称",
            preserved["daily_inspection"]["header"]["target_name"],
        )

    def test_comparison_facts_are_projected_as_findings(self) -> None:
        template = SYSTEM_REPORT_TEMPLATES["system:comparison.standard"]
        presentation = report_presentation(
            template=template,
            payload={
                "summary": "处理后的验证证据表明目标问题已经解决",
                "facts": [{
                    "kind": "comparison_result",
                    "result": "RESOLVED",
                    "primary_signals": {
                        "target_absent": True,
                        "blocking_absent": True,
                    },
                    "rationale_codes": ["TARGET_ABSENT", "BLOCKING_ABSENT"],
                }],
                "gaps": (),
            },
        )
        findings = next(item for item in presentation["sections"] if item["kind"] == "FINDINGS")
        self.assertEqual(
            [
                "对比结论：已解决",
                "目标对象已消失：是",
                "阻塞关系已消失：是",
                "判定依据：目标对象已消失、阻塞关系已消失",
            ],
            findings["items"],
        )

    def test_presentation_translates_inconclusive_missing_turns(self) -> None:
        presentation = report_presentation(
            template=SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"],
            payload={
                "status": "PARTIAL",
                "summary": "部分诊断轮次尚未形成最终结果。",
                "scope": {"root_cause_grade": "INCONCLUSIVE"},
                "gaps": [
                    {"code": "MISSING_FINAL_RESULT", "turn_no": 2},
                    {"code": "MISSING_FINAL_RESULT", "turn_no": 4},
                ],
            },
        )

        sections = {
            item["kind"]: item for item in presentation["sections"]
        }
        self.assertEqual(
            [
                "第 2 轮：缺少最终诊断结果",
                "第 4 轮：缺少最终诊断结果",
            ],
            sections["EVIDENCE_BOUNDARY"]["items"],
        )
        self.assertIn(
            "根因评估等级：证据不足，无法定论",
            sections["ROOT_CAUSE"]["items"],
        )

    def test_inspection_markdown_facts_are_not_serialized_as_dicts(self) -> None:
        template = SYSTEM_REPORT_TEMPLATES["system:inspection.daily"]
        presentation = report_presentation(
            template=template,
            payload={
                "summary": "巡检已完成",
                "facts": [{
                    "kind": "agent_health_inspection",
                    "markdown": "# 巡检报告\n\n## 主要发现\n\n### 表空间容量临界\n\n使用率达到 **96.88%**。",
                }],
                "gaps": (),
            },
        )
        findings = next(item for item in presentation["sections"] if item["kind"] == "FINDINGS")
        self.assertEqual(
            ["【主要发现】", "【表空间容量临界】", "使用率达到 96.88%。"],
            findings["items"],
        )

    def test_evidence_boundary_translates_deduplicates_and_removes_generic_prefix(self) -> None:
        presentation = report_presentation(
            template=SYSTEM_REPORT_TEMPLATES["system:inspection.daily"],
            payload={
                "gaps": [
                    {
                        "code": "SOURCE_NO_DATA",
                        "detail": "Prometheus 主查询和回退查询均未返回采样",
                    },
                    {
                        "code": "SOURCE_NO_DATA",
                        "detail": "Prometheus 主查询和回退查询均未返回采样",
                    },
                    {
                        "code": "MONITORING_COVERAGE_INSUFFICIENT",
                        "detail": "监控时间序列有效采样覆盖率低于0.8",
                    },
                    {
                        "code": "INVESTIGATION_EVIDENCE_GAP",
                        "detail": "复制状态证据缺失",
                    },
                ],
            },
        )

        boundary = next(
            item for item in presentation["sections"]
            if item["kind"] == "EVIDENCE_BOUNDARY"
        )
        self.assertEqual(
            [
                "监控源未返回有效数据：Prometheus 主查询和回退查询均未返回采样",
                "监控采样覆盖不足：监控时间序列有效采样覆盖率低于0.8",
                "复制状态证据缺失",
            ],
            boundary["items"],
        )

    def test_inspection_result_uses_the_same_source_normalizer(self) -> None:
        source = normalize_report_source(
            schema_version="DB_DIAGNOSTIC_REPORT.v1",
            source_kind="INSPECTION",
            payload={
                "status": "PARTIAL",
                "observation_count": 8,
                "gap_count": 1,
                "tools": ["db.instance.identity"],
                "gaps": [{"code": "CAPABILITY_UNAVAILABLE"}],
            },
        )
        self.assertEqual("PARTIAL", source["status"])
        self.assertEqual("INCONCLUSIVE", source["root_cause"]["effective_level"])
        self.assertEqual("已完成 8 项观测", source["facts"][0]["summary"])

    def test_template_inspection_report_is_reusable_for_periodic_rollup(self) -> None:
        source = normalize_report_source(
            schema_version="REPORT_CONTENT.v1",
            source_kind="INSPECTION",
            payload={
                "status": "READY",
                "summary": "本期按冻结巡检模板完成 2/2 项检查。",
                "scope": {"inspection_coverage": "已完成 2/2 项检查"},
                "facts": [
                    {"summary": "实例性能指标：检查已完成"},
                    {"summary": "近期告警日志：检查已完成，采集 0 条观测"},
                ],
                "gaps": [],
                "recommendations": ["继续按既定周期执行巡检。"],
                "evidence_refs": [],
            },
        )
        self.assertEqual("READY", source["status"])
        self.assertEqual("已完成 2/2 项检查", source["inspection_coverage"])
        self.assertEqual(2, len(source["facts"]))
        self.assertEqual(
            ("继续按既定周期执行巡检。",),
            source["solution"]["long_term_remediations"],
        )

    def test_closed_monthly_window_uses_report_timezone(self) -> None:
        start, end = closed_period_window(
            period_kind="MONTHLY",
            timezone="Asia/Shanghai",
            now=datetime(2026, 3, 1, 1, tzinfo=UTC),
        )
        self.assertEqual(datetime(2026, 1, 31, 16, tzinfo=UTC), start)
        self.assertEqual(datetime(2026, 2, 28, 16, tzinfo=UTC), end)

    def test_periodic_inspection_aggregates_all_frozen_sources(self) -> None:
        first_id, second_id = uuid7(), uuid7()
        first_artifact, second_artifact = uuid7(), uuid7()
        runs = [
            SimpleNamespace(ops_run_id=first_id, final_artifact_id=first_artifact, completed_at=datetime(2026, 2, 2, tzinfo=UTC)),
            SimpleNamespace(ops_run_id=second_id, final_artifact_id=second_artifact, completed_at=datetime(2026, 2, 3, tzinfo=UTC)),
        ]
        artifacts = {
            first_artifact: SimpleNamespace(artifact_id=first_artifact, content_hash="a" * 64, schema_version="DB_DIAGNOSTIC_REPORT.v1", payload_json={"status": "READY", "observation_count": 4, "gap_count": 0, "tools": []}),
            second_artifact: SimpleNamespace(artifact_id=second_artifact, content_hash="b" * 64, schema_version="DB_DIAGNOSTIC_REPORT.v1", payload_json={"status": "PARTIAL", "observation_count": 3, "gap_count": 1, "tools": [], "gaps": [{"code": "MISSING_ASH"}]}),
        }
        uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(side_effect=lambda *, artifact_id: artifacts[artifact_id])))
        service = AIOpsRuntimeService(uow_factory=AsyncMock(), blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
        result = asyncio.run(service._aggregate_inspection_sources(uow=uow, runs=runs, period_kind="MONTHLY", period_start=datetime(2026, 1, 31, 16, tzinfo=UTC), period_end=datetime(2026, 2, 28, 16, tzinfo=UTC)))
        self.assertEqual("PARTIAL", result["status"])
        self.assertEqual(2, len(result["evidence_refs"]))
        self.assertIn("完成 2 次巡检", result["inspection_coverage"])

    def test_existing_report_receipt_survives_uow_exit(self) -> None:
        run_id = uuid7()
        artifact_id = uuid7()
        report_id = uuid7()
        report_artifact_id = uuid7()

        class ExpiringReport:
            expired = False

            def _value(self, value):
                if self.expired:
                    raise AssertionError("事务退出后不得读取 ORM 实体")
                return value

            @property
            def report_id(self):
                return self._value(report_id)

            @property
            def status(self):
                return self._value("READY")

            @property
            def template_id(self):
                return self._value("system:diagnosis.standard")

            @property
            def report_version(self):
                return self._value(1)

            @property
            def period_start(self):
                return self._value(None)

            @property
            def content_artifact_id(self):
                return self._value(report_artifact_id)

            @property
            def content_hash(self):
                return self._value("b" * 64)

        current = ExpiringReport()
        run = SimpleNamespace(
            ops_run_id=run_id,
            trigger_type="ALERT",
            status="COMPLETED",
            final_artifact_id=artifact_id,
            plan_snapshot_json={},
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id,
            schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
            provenance_json={},
            payload_json={"status": "READY", "diagnosis_rationale": "本次证据已核验"},
            content_hash="a" * 64,
        )
        report_artifact = SimpleNamespace(schema_version="REPORT_CONTENT.v1", content_hash="b" * 64,
            payload_json={"provenance": {"report_input_hash": report_input_hash(
                source_artifact=artifact,
                source=normalize_report_source(schema_version=artifact.schema_version,
                    payload=artifact.payload_json, source_kind="ALERT"),
                template=SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"],
            )}})
        uow = SimpleNamespace(
            inspections=SimpleNamespace(
                get_current_report_for_run_template=AsyncMock(
                    return_value=current
                )
            ),
            runs=SimpleNamespace(
                get_run_scoped=AsyncMock(return_value=run),
                get_artifact=AsyncMock(side_effect=lambda *, artifact_id: artifact
                    if artifact_id == run.final_artifact_id else report_artifact),
                list_tasks=AsyncMock(return_value=[SimpleNamespace(
                    output_artifact_id=artifact_id
                )]),
                database_now=AsyncMock(return_value=datetime.now(UTC)),
            ),
        )

        class UnitOfWorkContext:
            async def __aenter__(self):
                return uow

            async def __aexit__(self, exc_type, exc, traceback):
                current.expired = True

        service = AIOpsRuntimeService(
            uow_factory=lambda: UnitOfWorkContext(),
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        receipt = asyncio.run(service.generate_user_report(
            domain_id=8,
            actor_id="operator-1",
            ops_run_id=run_id,
            template=SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"],
            period_kind="AD_HOC",
            trace_id="trace-existing-report",
        ))

        self.assertEqual(report_id, receipt.report_id)
        self.assertEqual("READY", receipt.status)
        self.assertEqual("system:diagnosis.standard", receipt.template_id)
        self.assertEqual(1, receipt.report_version)

    def test_conversation_report_aggregates_all_completed_turns(self) -> None:
        first_turn, second_turn = uuid7(), uuid7()
        first_run, second_run = uuid7(), uuid7()
        service = AIOpsRuntimeService(
            uow_factory=AsyncMock(), blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        result = service._aggregate_conversation_sources(
            conversation=SimpleNamespace(
                conversation_id=uuid7(), title="生产库锁等待排查"
            ),
            source_rows=[
                (
                    SimpleNamespace(turn_id=first_turn, turn_no=1),
                    SimpleNamespace(
                        ops_run_id=first_run,
                        completed_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
                        created_at=datetime(2026, 9, 1, tzinfo=UTC),
                        plan_snapshot_json={"diagnosis": {"question_summary": "确认锁等待"}},
                    ),
                    SimpleNamespace(
                        artifact_id=uuid7(), content_hash="a" * 64,
                        schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                        payload_json={
                            "status": "READY",
                            "root_cause": {"effective_level": "POSSIBLE"},
                            "diagnosis_rationale": "锁等待来自未提交事务。",
                            "facts": [{"summary": "锁等待持续升高"}],
                            "solution": {"immediate_mitigations": ["确认阻塞会话"]},
                        },
                    ),
                ),
                (
                    SimpleNamespace(turn_id=second_turn, turn_no=2),
                    SimpleNamespace(
                        ops_run_id=second_run,
                        completed_at=datetime(2026, 9, 1, 2, tzinfo=UTC),
                        created_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
                        plan_snapshot_json={"diagnosis": {"question_summary": "验证根因"}},
                    ),
                    SimpleNamespace(
                        artifact_id=uuid7(), content_hash="b" * 64,
                        schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                        payload_json={
                            "status": "READY",
                            "root_cause": {"effective_level": "CONFIRMED"},
                            "diagnosis_rationale": "阻塞会话已定位。",
                            "facts": [{"summary": "阻塞会话已确认"}],
                            "solution": {"long_term_remediations": ["优化事务边界"]},
                        },
                    ),
                ),
            ],
            missing_turns=[{"code": "MISSING_FINAL_RESULT", "turn_no": 3}],
        )
        self.assertEqual("PARTIAL", result["status"])
        self.assertEqual("INCONCLUSIVE", result["root_cause"]["effective_level"])
        self.assertEqual(4, len(result["facts"]))
        self.assertEqual(2, len(result["evidence_refs"]))
        self.assertEqual(3, result["conversation"]["turn_count"])
        self.assertEqual(
            ["确认锁等待", "验证根因"],
            result["conversation"]["question_summaries"],
        )

    def test_generate_conversation_report_freezes_every_completed_turn(self) -> None:
        conversation_id, first_turn, second_turn = uuid7(), uuid7(), uuid7()
        first_run, second_run = uuid7(), uuid7()
        first_artifact, second_artifact = uuid7(), uuid7()
        now = datetime(2026, 9, 1, 3, tzinfo=UTC)
        conversation = SimpleNamespace(
            conversation_id=conversation_id, created_by="operator-1",
            title="生产库锁等待排查", agent_id=uuid7(),
        )
        turns = [
            SimpleNamespace(turn_id=first_turn, turn_no=1, status="COMPLETED"),
            SimpleNamespace(turn_id=second_turn, turn_no=2, status="PARTIAL"),
        ]
        runs = {
            first_turn: SimpleNamespace(
                ops_run_id=first_run, status="COMPLETED", final_artifact_id=first_artifact,
                created_at=datetime(2026, 9, 1, tzinfo=UTC), started_at=None,
                completed_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
                plan_snapshot_json={"diagnosis": {"question_summary": "确认锁等待"}},
            ),
            second_turn: SimpleNamespace(
                ops_run_id=second_run, status="PARTIAL", final_artifact_id=second_artifact,
                created_at=datetime(2026, 9, 1, 1, tzinfo=UTC), started_at=None,
                completed_at=datetime(2026, 9, 1, 2, tzinfo=UTC),
                plan_snapshot_json={"diagnosis": {"question_summary": "验证根因"}},
            ),
        }
        artifacts = {
            first_artifact: SimpleNamespace(
                artifact_id=first_artifact, content_hash="a" * 64,
                schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                payload_json={"status": "READY", "diagnosis_rationale": "锁等待来自未提交事务。", "facts": [{"summary": "锁等待持续升高"}]},
            ),
            second_artifact: SimpleNamespace(
                artifact_id=second_artifact, content_hash="b" * 64,
                schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                payload_json={"status": "PARTIAL", "diagnosis_rationale": "阻塞会话已定位。", "facts": [{"summary": "阻塞会话已确认"}]},
            ),
        }
        uow = SimpleNamespace(
            inspections=SimpleNamespace(),
            conversations=SimpleNamespace(get_conversation=AsyncMock(return_value=conversation)),
            turns=SimpleNamespace(
                list_all_turns=AsyncMock(return_value=turns),
                get_run_link=AsyncMock(side_effect=lambda *, turn_id, purpose: SimpleNamespace(ops_run_id=runs[turn_id].ops_run_id)),
            ),
            runs=SimpleNamespace(
                get_run_scoped=AsyncMock(side_effect=lambda *, ops_run_id, domain_id, lock: next(run for run in runs.values() if run.ops_run_id == ops_run_id)),
                get_artifact=AsyncMock(side_effect=lambda *, artifact_id: artifacts[artifact_id]),
                get_artifact_by_key=AsyncMock(return_value=None),
                list_tasks=AsyncMock(return_value=[SimpleNamespace(output_artifact_id=second_artifact)]),
                database_now=AsyncMock(return_value=now),
            ),
            commit=AsyncMock(),
        )

        class UnitOfWorkContext:
            async def __aenter__(self): return uow
            async def __aexit__(self, exc_type, exc, traceback): return None

        service = AIOpsRuntimeService(
            uow_factory=lambda: UnitOfWorkContext(), blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        service._publish_diagnosis_report = AsyncMock(return_value=SimpleNamespace(
            report_id=uuid7(),
            status="PARTIAL",
            template_id="system:diagnosis.standard",
            report_version=1,
        ))
        asyncio.run(service.generate_conversation_report(
            domain_id=8, actor_id="operator-1", conversation_id=conversation_id,
            template=SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"], trace_id="trace-session-report",
        ))
        source_override = service._publish_diagnosis_report.await_args.kwargs["source_override"]
        self.assertEqual(4, len(source_override["facts"]))
        self.assertEqual(2, len(source_override["evidence_refs"]))
        self.assertEqual("PARTIAL", source_override["status"])
        uow.commit.assert_awaited_once()

    def test_generate_conversation_report_reuses_same_snapshot_and_template(self) -> None:
        conversation_id, turn_id, run_id = uuid7(), uuid7(), uuid7()
        source_artifact_id, report_artifact_id = uuid7(), uuid7()
        report_id = uuid7()
        now = datetime(2026, 9, 1, 3, tzinfo=UTC)
        source_hash = "a" * 64
        conversation = SimpleNamespace(
            conversation_id=conversation_id,
            created_by="operator-1",
            title="生产库锁等待排查",
            agent_id=uuid7(),
        )
        turn = SimpleNamespace(
            turn_id=turn_id, turn_no=1, status="COMPLETED"
        )
        run = SimpleNamespace(
            ops_run_id=run_id,
            status="COMPLETED",
            final_artifact_id=source_artifact_id,
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
            started_at=None,
            completed_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
            plan_snapshot_json={
                "diagnosis": {"question_summary": "确认锁等待"}
            },
        )
        source_artifact = SimpleNamespace(
            artifact_id=source_artifact_id,
            content_hash=source_hash,
            schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
            payload_json={
                "status": "READY",
                "root_cause": {"effective_level": "PROBABLE"},
                "diagnosis_rationale": "锁等待已定位。",
                "facts": [],
            },
        )
        projector = AIOpsRuntimeService(uow_factory=AsyncMock(), blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
        report_payload = {
            "provenance": {"report_input_hash": report_input_hash(
                source_artifact=source_artifact,
                source=projector._aggregate_conversation_sources(
                    conversation=conversation, source_rows=[(turn, run, source_artifact)], missing_turns=[],
                ), template=SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"],
            )}
        }
        report_hash = sha256_json(report_payload)
        current_report = SimpleNamespace(
            report_id=report_id,
            status="READY",
            template_id="system:diagnosis.standard",
            template_version="1",
            report_version=2,
            content_artifact_id=report_artifact_id,
            content_hash=report_hash,
        )
        current_artifact = SimpleNamespace(
            artifact_id=report_artifact_id,
            schema_version="REPORT_CONTENT.v1",
            content_hash=report_hash,
            payload_json=report_payload,
        )
        uow = SimpleNamespace(
            inspections=SimpleNamespace(
                get_current_report_for_conversation_template=AsyncMock(
                    return_value=current_report
                )
            ),
            conversations=SimpleNamespace(
                get_conversation=AsyncMock(return_value=conversation)
            ),
            turns=SimpleNamespace(
                list_all_turns=AsyncMock(return_value=[turn]),
                get_run_link=AsyncMock(
                    return_value=SimpleNamespace(ops_run_id=run_id)
                ),
            ),
            runs=SimpleNamespace(
                get_run_scoped=AsyncMock(return_value=run),
                get_artifact_by_key=AsyncMock(return_value=None),
                get_artifact=AsyncMock(
                    side_effect=lambda *, artifact_id: (
                        source_artifact
                        if artifact_id == source_artifact_id
                        else current_artifact
                    )
                ),
                list_tasks=AsyncMock(return_value=[
                    SimpleNamespace(output_artifact_id=source_artifact_id)
                ]),
                database_now=AsyncMock(return_value=now),
            ),
            commit=AsyncMock(),
        )

        class UnitOfWorkContext:
            async def __aenter__(self): return uow
            async def __aexit__(self, exc_type, exc, traceback): return None

        service = AIOpsRuntimeService(
            uow_factory=lambda: UnitOfWorkContext(),
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        service._publish_diagnosis_report = AsyncMock()

        receipt = asyncio.run(service.generate_conversation_report(
            domain_id=8,
            actor_id="operator-1",
            conversation_id=conversation_id,
            template=SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"],
            trace_id="trace-session-report-retry",
        ))

        self.assertEqual(report_id, receipt.report_id)
        self.assertEqual(2, receipt.report_version)
        service._publish_diagnosis_report.assert_not_awaited()
        uow.commit.assert_not_awaited()

    def test_get_report_presentation_uses_report_type_when_snapshot_is_missing(self) -> None:
        cases = (
            ("INSPECTION_WEEKLY", "database_weekly", "system:inspection.weekly"),
            ("INSPECTION_CUSTOM", "database_custom", "system:inspection.custom"),
            ("COMPARISON", "oracle.kill-session", "system:comparison.standard"),
        )
        for report_type, stored_template_id, expected_ref in cases:
            with self.subTest(report_type=report_type):
                report_id, artifact_id = uuid7(), uuid7()
                payload = {
                    "title": "历史报告",
                    "status": "READY",
                    "summary": "已完成核验",
                    "period_start": "2026-09-01T00:00:00+00:00",
                    "period_end": "2026-09-08T00:00:00+00:00",
                    "facts": [{
                        "kind": "comparison_result" if report_type == "COMPARISON" else "agent_health_inspection",
                        "result": "RESOLVED",
                        "markdown": "## 主要发现\n\n健康。",
                        "primary_signals": {"target_absent": True},
                        "rationale_codes": ["TARGET_ABSENT"],
                    }],
                    "gaps": [],
                    "evidence_refs": [],
                    "provenance": {},
                }
                content_hash = sha256_json(payload)
                report = SimpleNamespace(
                    report_id=report_id,
                    content_artifact_id=artifact_id,
                    template_id=stored_template_id,
                    template_version="1.0.0",
                    report_type=report_type,
                    content_hash=content_hash,
                )
                artifact = SimpleNamespace(
                    artifact_id=artifact_id,
                    schema_version="REPORT_CONTENT.v1",
                    content_hash=content_hash,
                    payload_json=payload,
                )
                uow = SimpleNamespace(
                    inspections=SimpleNamespace(
                        get_report_scoped=AsyncMock(return_value=report)
                    ),
                    runs=SimpleNamespace(
                        get_artifact=AsyncMock(return_value=artifact)
                    ),
                )

                class UnitOfWorkContext:
                    async def __aenter__(self):
                        return uow

                    async def __aexit__(self, exc_type, exc, traceback):
                        return None

                service = AIOpsRuntimeService(
                    uow_factory=lambda: UnitOfWorkContext(),
                    blueprint_registry=AsyncMock(),
                    handler_registry=AsyncMock(),
                )
                presentation = asyncio.run(
                    service.get_report_presentation(
                        report_id=report_id, domain_id=8,
                    )
                )
                self.assertEqual("REPORT_PRESENTATION.v1", presentation["schema_version"])
                self.assertEqual(expected_ref, presentation["template"]["template_ref"])
                self.assertTrue(any(
                    item["kind"] == "EVIDENCE_BOUNDARY"
                    for item in presentation["sections"]
                ))


if __name__ == "__main__":
    unittest.main()
