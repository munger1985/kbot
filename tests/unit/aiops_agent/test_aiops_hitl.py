"""AIOps 聊天人工补证的目录 SQL、结果解析与策略测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta

from aiops_agent.contracts.diagnosis import (
    DiagnosisRoundAssessment,
    EvidenceIndex,
)
from aiops_agent.contracts.hitl import InputSuspension
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.domain.diagnosis import normalize_evidence_artifacts
from aiops_agent.orchestration.hitl import (
    normalize_inline_response,
    validate_model_manual_sql,
)
from aiops_agent.workers.diagnosis_handlers import (
    InteractiveDiagnosisHandler,
)
from aiops_agent.workers.handlers import TaskExecutionContext


class ManualDiagnosticSqlTest(unittest.TestCase):
    def test_validator_rejects_business_table_and_mutation(self) -> None:
        with self.assertRaisesRegex(ValueError, "未授权对象"):
            validate_model_manual_sql(
                "SELECT * FROM customer_orders", db_type="ORACLE"
            )
        with self.assertRaisesRegex(ValueError, "只能"):
            validate_model_manual_sql(
                "DELETE FROM v$session", db_type="ORACLE"
            )

    def test_inline_csv_is_bounded_and_formula_is_neutralized(self) -> None:
        result = normalize_inline_response(
            hitl_id="hitl-1",
            query_id="db.test",
            status="SUCCEEDED",
            result_format="CSV",
            inline_data="name,value\nrow1,=danger\n",
            error=None,
            expected_columns=("name", "value"),
            max_rows=10,
        )
        self.assertEqual(result.rows, (("row1", "'=danger"),))
        with self.assertRaisesRegex(ValueError, "列"):
            normalize_inline_response(
                hitl_id="hitl-1",
                query_id="db.test",
                status="SUCCEEDED",
                result_format="CSV",
                inline_data="unexpected\n1\n",
                error=None,
                expected_columns=("name", "value"),
                max_rows=10,
            )


class InteractiveDiagnosisHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = DiagnosticRegistry.load()

    def test_non_chat_run_never_requests_manual_input(self) -> None:
        result = asyncio.run(
            InteractiveDiagnosisHandler(
                registry=self.registry
            ).execute(self._context("ALERT"))
        )
        self.assertEqual(result.status, "NOT_REQUIRED")

    def test_chat_connectivity_gap_creates_catalog_sql_request(self) -> None:
        result = asyncio.run(
            InteractiveDiagnosisHandler(
                registry=self.registry
            ).execute(self._context("CHAT"))
        )
        self.assertIsInstance(result, InputSuspension)
        self.assertEqual(result.request_type, "MANUAL_DIAGNOSTIC_SQL")
        queries = result.request_payload["queries"]
        self.assertGreaterEqual(len(queries), 1)
        self.assertEqual(queries[0]["query_id"], "db.instance.identity")
        self.assertTrue(queries[0]["sql_text"].lstrip().startswith("SELECT"))

    def test_user_result_is_normalized_as_unverified_evidence(self) -> None:
        index = normalize_evidence_artifacts(
            (
                {
                    "artifact_id": "artifact-1",
                    "schema_version": "HITL_OUTCOME.v1",
                    "payload": {
                        "status": "ANSWERED",
                        "submission": {
                            "results": [
                                {
                                    "query_id": "db.session.active",
                                    "status": "SUCCEEDED",
                                    "columns": ["session_id"],
                                    "rows": [[42]],
                                    "quality_flags": ["USER_PROVIDED"],
                                }
                            ]
                        },
                    },
                },
            ),
            target_id="target-1",
        )
        self.assertEqual(index.fact_count, 1)
        self.assertEqual(index.facts[0].source_type, "USER_RESULT")
        self.assertEqual(index.facts[0].trust_level, "USER_PROVIDED")

    def _context(self, trigger_type: str) -> TaskExecutionContext:
        tools = []
        for item in self.registry.tools:
            definition = item.definition
            if (
                definition.db_type == "ORACLE"
                and definition.tool_id
                in {"db.instance.identity", "db.session.active"}
            ):
                tools.append(
                    {
                        "tool_id": definition.tool_id,
                        "version": definition.version,
                        "variant": definition.variant,
                        "template_sha256": definition.template_sha256,
                        "parameters": {
                            parameter.name: parameter.default
                            for parameter in definition.parameters
                            if not parameter.required
                        },
                    }
                )
        assessment = DiagnosisRoundAssessment(
            round_no=3,
            suggested_root_cause_level="INCONCLUSIVE",
            recommended_next_step="STOP_INCONCLUSIVE",
            rationale_summary="数据库当前不可连接",
        )
        evidence = EvidenceIndex(
            target_id="target-1",
            gaps=(
                {
                    "code": "TARGET_UNREACHABLE",
                    "detail": "目标数据库不可连接",
                },
            ),
            fact_count=0,
            source_group_count=0,
            index_hash="0" * 64,
        )
        return TaskExecutionContext(
            run_id="run-1",
            task_id="task-1",
            task_key="diagnosis:interactive",
            target_id="target-1",
            agent_id="agent-1",
            trigger_type=trigger_type,
            actor_id="user-1",
            original_request="分析数据库连接故障",
            trace_id="trace-1",
            attempt=1,
            deadline_at=(
                datetime.now(UTC) + timedelta(hours=1)
            ).isoformat(),
            plan_snapshot={
                "target": {
                    "target_key": "oracle-prod",
                },
                "database_diagnostics": {
                    "db_type": "ORACLE",
                    "configured_version": "19.0.0",
                    "tools": tools,
                },
            },
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "EVIDENCE_INDEX.v1",
                    "payload": evidence.model_dump(mode="json"),
                },
                {
                    "schema_version": (
                        "DIAGNOSIS_ROUND_ASSESSMENT.v1"
                    ),
                    "payload": assessment.model_dump(mode="json"),
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
