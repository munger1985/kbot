"""官方 SQL Monitor 报告 Tool 的目录、发现与下载边界。"""

from __future__ import annotations

import hashlib
import inspect
import unittest
from pathlib import Path

import aiops_agent.api.conversations as conversations
from aiops_agent.application.investigation.discovery import (
    rewrite_incomplete_discovery_actions,
)
from aiops_agent.application.investigation.discovery_binding import (
    bind_discovery_parameters,
    catalog_tool_cards,
    decide_discovery_continuation,
)
from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from platform_core.contracts.aiops.investigation import (
    InvestigationAction,
    InvestigationPlan,
    InvestigationPlanningOutput,
)


SQL_ID = "8ab3c1d9e0f12"
CATALOG_SQL = (
    Path(__file__).resolve().parents[3]
    / "services/aiops_agent/src/aiops_agent/diagnostics/catalog/oracle/sql"
)


def _investigation(*, tool_id: str, input_values: dict, deferred: bool = False):
    return InvestigationPlanningOutput.model_validate(
        {
            "input_envelope": {
                "materials": [
                    {
                        "item_no": 1,
                        "material_kind": "QUESTION",
                        "summary": "生成 SQL Monitor 报告",
                        "confidence": 1,
                    }
                ],
                "explicit_question": "生成 SQL Monitor 报告",
            },
            "task_frame": {
                "objectives": ["DIAGNOSE"],
                "problem_statement": "查看单条 SQL 的官方监控报告",
                "success_criteria": ["取得原生 SQL Monitor HTML"],
            },
            "plan": {
                "revision_no": 1,
                "actions": [
                    {
                        "action_id": "a1",
                        "question": "生成官方 SQL Monitor 报告",
                        "tool_id": tool_id,
                        "input": input_values,
                        "expected_evidence_kind": "SQL_MONITOR_REPORT",
                        "measurement_semantics": "CURRENT_ACTIVITY",
                        "deferred": deferred,
                    }
                ],
            },
        }
    )


def _execution_result(*rows: tuple) -> dict:
    return {
        "schema_version": "DBA_TOOL_RESULT.v1",
        "tool_outcomes": [
            {
                "tool_id": "db.oracle.sql_monitor.executions",
                "status": "SUCCEEDED",
                "observation": {
                    "columns": [
                        {"name": "sql_id"},
                        {"name": "sql_exec_id"},
                        {"name": "sql_exec_start"},
                    ],
                    "rows": [list(row) for row in rows],
                },
            }
        ],
    }


class OracleSqlMonitorReportTest(unittest.TestCase):
    def test_catalog_pins_hash_package_and_discovery(self) -> None:
        registry = DiagnosticRegistry.load()
        report = registry.resolve(
            tool_id="db.oracle.sql_monitor.report",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="19c",
            capabilities={"dynamic_performance_views"},
            entitlements=set(),
        )
        executions = registry.resolve(
            tool_id="db.oracle.sql_monitor.executions",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="19c",
            capabilities={"dynamic_performance_views"},
            entitlements=set(),
        )
        report_sql = (CATALOG_SQL / "sql_monitor_report_html.sql").read_bytes()
        executions_sql = (CATALOG_SQL / "sql_monitor_executions.sql").read_bytes()
        self.assertEqual(
            "18b5eb8bc06304f8849dfc6adaf26ebfc6f0095114fb27f49858d247e84392bf",
            hashlib.sha256(report_sql).hexdigest(),
        )
        self.assertEqual(
            "ab0ca4e9276227b4aeb21e498e1edb3891ef7a7dd70ba20cd499a14c1db755b1",
            hashlib.sha256(executions_sql).hexdigest(),
        )
        self.assertEqual(("DBMS_SQLTUNE",), report.definition.allowed_packages)
        self.assertEqual(
            "db.oracle.sql_monitor.executions",
            report.definition.discovery_tool_id,
        )
        self.assertIsNone(executions.definition.discovery_tool_id)
        self.assertIn("dbms_sqltune.report_sql_monitor", report.sql.lower())
        self.assertNotIn("dbms_sqltune", executions.sql.lower())

    def test_missing_sql_id_rewrites_to_executions_and_defers_report(self) -> None:
        cards = catalog_tool_cards()
        rewritten = rewrite_incomplete_discovery_actions(
            investigation=_investigation(
                tool_id="db.oracle.sql_monitor.report",
                input_values={},
            ),
            available_tools=cards,
        )
        self.assertIsNotNone(rewritten)
        self.assertEqual(
            [
                "db.oracle.sql_monitor.executions",
                "db.oracle.sql_monitor.report",
            ],
            [action.tool_id for action in rewritten.plan.actions],
        )
        self.assertEqual({}, rewritten.plan.actions[0].input)
        self.assertFalse(rewritten.plan.actions[0].deferred)
        self.assertTrue(rewritten.plan.actions[1].deferred)
        self.assertIn(
            rewritten.plan.actions[0].action_id,
            rewritten.plan.actions[1].depends_on,
        )

    def test_confirmed_sql_id_does_not_rewrite_to_discovery(self) -> None:
        rewritten = rewrite_incomplete_discovery_actions(
            investigation=_investigation(
                tool_id="db.oracle.sql_monitor.report",
                input_values={"sql_id": SQL_ID},
            ),
            available_tools=catalog_tool_cards(),
        )
        self.assertIsNone(rewritten)

    def test_listing_does_not_guess_sql_id_even_when_unique(self) -> None:
        plan = InvestigationPlan(
            revision_no=1,
            actions=(
                InvestigationAction.model_validate(
                    {
                        "action_id": "a1",
                        "question": "列出可见 SQL Monitor 执行",
                        "tool_id": "db.oracle.sql_monitor.executions",
                        "input": {},
                        "expected_evidence_kind": "DISCOVERY_CATALOG",
                        "measurement_semantics": "CURRENT_ACTIVITY",
                    }
                ),
                InvestigationAction.model_validate(
                    {
                        "action_id": "a2",
                        "question": "生成官方 SQL Monitor 报告",
                        "tool_id": "db.oracle.sql_monitor.report",
                        "input": {},
                        "expected_evidence_kind": "SQL_MONITOR_REPORT",
                        "measurement_semantics": "CURRENT_ACTIVITY",
                        "depends_on": ("a1",),
                        "deferred": True,
                    }
                ),
            ),
        )
        bound = bind_discovery_parameters(
            plan=plan,
            tool_results=(
                _execution_result(
                    (SQL_ID, 16777216, "2026-09-20T10:00:00+08:00"),
                ),
            ),
        )
        self.assertEqual("UNBINDABLE", bound.status)
        self.assertEqual(("a2",), bound.unbound_action_ids)
        self.assertEqual({}, bound.plan.actions[1].input)
        self.assertTrue(bound.plan.actions[1].deferred)
        decision = decide_discovery_continuation(
            plan=plan,
            tool_results=(
                _execution_result(
                    (SQL_ID, 16777216, "2026-09-20T10:00:00+08:00"),
                ),
            ),
        )
        self.assertEqual("ASK_USER", decision.action)

    def test_confirmed_sql_id_binds_without_discovery_rows(self) -> None:
        plan = InvestigationPlan(
            revision_no=1,
            actions=(
                InvestigationAction.model_validate(
                    {
                        "action_id": "a1",
                        "question": "生成官方 SQL Monitor 报告",
                        "tool_id": "db.oracle.sql_monitor.report",
                        "input": {"sql_id": SQL_ID},
                        "expected_evidence_kind": "SQL_MONITOR_REPORT",
                        "measurement_semantics": "CURRENT_ACTIVITY",
                        "deferred": True,
                    }
                ),
            ),
        )
        bound = bind_discovery_parameters(plan=plan, tool_results=())
        self.assertEqual("BOUND", bound.status)
        self.assertEqual({"sql_id": SQL_ID}, bound.plan.actions[0].input)
        self.assertFalse(bound.plan.actions[0].deferred)

    def test_validate_rejects_unconfirmed_or_invalid_sql_id(self) -> None:
        registry = DiagnosticRegistry.load()
        tool = registry.resolve(
            tool_id="db.oracle.sql_monitor.report",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="19c",
            capabilities={"dynamic_performance_views"},
            entitlements=set(),
        )
        with self.assertRaisesRegex(ValueError, "缺少诊断参数：sql_id"):
            registry.validate_parameters(tool, {})
        with self.assertRaisesRegex(
            ValueError, "SQL Monitor 未确认 sql_id 不调用 REPORT_SQL_MONITOR"
        ):
            registry.validate_parameters(tool, {"sql_id": "not-a-sqlid"})
        with self.assertRaisesRegex(
            ValueError, "SQL Monitor 未确认 sql_id 不调用 REPORT_SQL_MONITOR"
        ):
            registry.validate_parameters(tool, {"sql_id": "short"})
        with self.assertRaisesRegex(ValueError, "UTC 偏移"):
            registry.validate_parameters(
                tool,
                {
                    "sql_id": SQL_ID,
                    "sql_exec_start": "2026-09-20T10:00:00",
                },
            )
        self.assertEqual(
            {
                "sql_id": SQL_ID,
                "sql_exec_id": 0,
                "sql_exec_start": "",
            },
            registry.validate_parameters(tool, {"sql_id": SQL_ID}),
        )
        self.assertEqual(
            {
                "sql_id": SQL_ID,
                "sql_exec_id": 16777216,
                "sql_exec_start": "2026-09-20T10:00:00+08:00",
            },
            registry.validate_parameters(
                tool,
                {
                    "sql_id": SQL_ID,
                    "sql_exec_id": 16777216,
                    "sql_exec_start": "2026-09-20T10:00:00+08:00",
                },
            ),
        )

    def test_download_whitelist_includes_sql_monitor_report(self) -> None:
        download_source = inspect.getsource(conversations.download_workload_report)
        content_source = inspect.getsource(
            ConversationTurnService.get_workload_report_content
        )
        self.assertIn("sql_monitor\\.report", download_source)
        self.assertIn("oracle-sql-monitor", download_source)
        self.assertIn("db.oracle.sql_monitor.report", content_source)
        self.assertNotIn("db.sql.plan_monitor", download_source)


if __name__ == "__main__":
    unittest.main()
