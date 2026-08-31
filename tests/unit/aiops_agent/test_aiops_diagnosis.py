"""AIOps 步骤 7 证据、Planner Gate 与根因等级测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from pydantic import ValidationError

from aiops_agent.contracts.diagnosis import (
    DiagnosisRoundAssessment,
    DiagnosisRoundDraft,
    EvidenceRequestDraft,
    HypothesisAssessment,
    HypothesisDraft,
)
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.domain.diagnosis import (
    EvidenceRequestBudget,
    assess_root_cause,
    normalize_evidence_artifacts,
    validate_evidence_requests,
)
from aiops_agent.orchestration import (
    BlueprintRegistry,
    build_diagnosis_blueprint,
    build_multi_round_diagnosis_blueprint,
)
from aiops_agent.orchestration.diagnosis.prompts import PROMPT_KEYS
from aiops_agent.workers.diagnosis_handlers import (
    DiagnosisReportHandler,
    DiagnosisRoundAssessmentHandler,
    DiagnosisRoundDraftHandler,
)
from aiops_agent.workers.handlers import TaskExecutionContext
from platform_core.prompts import load_prompt_catalog


class _TestPrompts:
    async def resolve(self, *_args, **_kwargs):
        raise AssertionError("当前测试路径不应调用模型 Prompt")


def _monitor_artifact() -> dict:
    now = datetime(2026, 7, 24, 1, tzinfo=UTC).isoformat()
    return {
        "artifact_id": "monitor-artifact",
        "schema_version": "OBSERVATION_SET.v1",
        "payload": {
            "target_id": "target-1",
            "binding_id": "binding-1",
            "source_id": "source-1",
            "observations": [
                {
                    "metric_code": "db.cpu.utilization",
                    "unit": "percent",
                    "window_start": now,
                    "window_end": now,
                    "source_id": "source-1",
                    "binding_id": "binding-1",
                    "summary": {"max": 98.5, "avg": 91.2},
                    "coverage_ratio": 1.0,
                    "truncated": False,
                    "warnings": [],
                }
            ],
            "active_alerts": [],
            "gaps": [],
        },
    }


def _database_artifact() -> dict:
    now = datetime(2026, 7, 24, 1, tzinfo=UTC).isoformat()
    return {
        "artifact_id": "database-artifact",
        "schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1",
        "payload": {
            "status": "SUCCEEDED",
            "observation": {
                "tool_id": "db.session.active",
                "columns": [
                    {"name": "status"},
                    {"name": "wait_event"},
                ],
                "rows": [["ACTIVE", "CPU"]],
                "captured_at": now,
                "truncated": False,
            },
        },
    }


class EvidenceIndexTest(unittest.TestCase):
    def test_monitor_and_database_keep_independent_lineage(self) -> None:
        index = normalize_evidence_artifacts(
            (_monitor_artifact(), _database_artifact()),
            target_id="target-1",
        )
        self.assertEqual(3, index.fact_count)
        self.assertEqual(2, index.source_group_count)
        self.assertEqual(64, len(index.index_hash))
        self.assertEqual(
            {
                "METRIC_OBSERVATION",
                "DATABASE_OBSERVATION",
            },
            {item.source_type for item in index.facts},
        )

    def test_same_input_produces_same_fact_ids(self) -> None:
        first = normalize_evidence_artifacts(
            (_monitor_artifact(),), target_id="target-1"
        )
        second = normalize_evidence_artifacts(
            (_monitor_artifact(),), target_id="target-1"
        )
        self.assertEqual(first.index_hash, second.index_hash)
        self.assertEqual(
            [item.fact_id for item in first.facts],
            [item.fact_id for item in second.facts],
        )

    def test_storage_remaining_question_uses_monitor_fact_directly(
        self,
    ) -> None:
        artifact = _monitor_artifact()
        observation = artifact["payload"]["observations"][0]
        observation["metric_code"] = "db.storage.utilization"
        observation["unit"] = "percent"
        observation["summary"] = {
            "last": 1.0492,
            "max": 1.2,
            "avg": 1.0,
        }
        evidence = normalize_evidence_artifacts(
            (artifact,), target_id="target-1"
        )

        answer = DiagnosisReportHandler._direct_answer(
            question="表空间还有多少？",
            evidence=evidence,
        )

        self.assertIsNotNone(answer)
        self.assertEqual("PARTIAL", answer.status)
        self.assertIn("剩余约 98.95%", answer.answer_text)
        self.assertIn("没有 tablespace 标签", answer.answer_text)
        self.assertEqual(1, len(answer.fact_refs))

    def test_storage_remaining_bytes_marks_partial_answer(self) -> None:
        artifact = _monitor_artifact()
        observation = artifact["payload"]["observations"][0]
        observation["metric_code"] = "db.storage.utilization"
        observation["unit"] = "percent"
        observation["summary"] = {"last": 25.0}
        evidence = normalize_evidence_artifacts(
            (artifact,), target_id="target-1"
        )

        answer = DiagnosisReportHandler._direct_answer(
            question="表空间还剩多少 GB？",
            evidence=evidence,
        )

        self.assertIsNotNone(answer)
        self.assertEqual("PARTIAL", answer.status)
        self.assertIn("剩余约 75.00%", answer.answer_text)
        self.assertIn("不能据此计算剩余 GB/TB", answer.answer_text)

    def test_storage_series_answer_keeps_tablespace_dimensions(
        self,
    ) -> None:
        artifact = _monitor_artifact()
        observation = artifact["payload"]["observations"][0]
        observation["metric_code"] = "db.storage.utilization"
        observation["unit"] = "percent"
        observation["summary"] = {"max": 80.0}
        observation["series"] = [
            {
                "dimensions": {"tablespace": "USERS"},
                "points": [
                    {
                        "observed_at": datetime.now(UTC).isoformat(),
                        "value": 80.0,
                        "quality": "GOOD",
                    }
                ],
            },
            {
                "dimensions": {"tablespace": "SYSTEM"},
                "points": [
                    {
                        "observed_at": datetime.now(UTC).isoformat(),
                        "value": 40.0,
                        "quality": "GOOD",
                    }
                ],
            },
        ]
        free_observation = {
            **observation,
            "metric_code": "db.storage.free_bytes",
            "unit": "bytes",
            "summary": {"last": 80 * 1024**3},
            "series": [
                {
                    "dimensions": {"tablespace": "USERS"},
                    "points": [
                        {
                            "observed_at": datetime.now(UTC).isoformat(),
                            "value": 80 * 1024**3,
                            "quality": "GOOD",
                        }
                    ],
                },
                {
                    "dimensions": {"tablespace": "SYSTEM"},
                    "points": [
                        {
                            "observed_at": datetime.now(UTC).isoformat(),
                            "value": 60 * 1024**3,
                            "quality": "GOOD",
                        }
                    ],
                },
            ],
        }
        artifact["payload"]["observations"].append(free_observation)
        evidence = normalize_evidence_artifacts(
            (artifact,), target_id="target-1"
        )

        answer = DiagnosisReportHandler._direct_answer(
            question="各表空间还剩多少？",
            evidence=evidence,
        )

        self.assertIsNotNone(answer)
        self.assertIn("| 表空间 | 可用空间 | 剩余比例 |", answer.answer_text)
        self.assertIn("| USERS | 80.00 GiB | 20.00% |", answer.answer_text)
        self.assertIn("| SYSTEM | 60.00 GiB | 60.00% |", answer.answer_text)
        self.assertEqual(4, len(answer.fact_refs))

        utilization_answer = DiagnosisReportHandler._direct_answer(
            question="表空间使用率是多少？",
            evidence=evidence,
        )
        self.assertIsNotNone(utilization_answer)
        self.assertEqual("ANSWERED", utilization_answer.status)
        self.assertIn("| 表空间 | 使用率 |", utilization_answer.answer_text)
        self.assertIn("| USERS | 80.00% |", utilization_answer.answer_text)
        self.assertIn("| SYSTEM | 40.00% |", utilization_answer.answer_text)
        self.assertNotIn("可用", utilization_answer.answer_text)


class PlannerGateTest(unittest.TestCase):
    def _draft(self, tool_id="db.transaction.long_running"):
        return DiagnosisRoundDraft(
            round_no=1,
            hypotheses=(
                HypothesisDraft(
                    hypothesis_key="cpu_pressure",
                    statement="活跃事务造成 CPU 压力",
                    mechanism="长事务持续消耗 CPU",
                    causal_role="ROOT",
                ),
            ),
            evidence_requests=(
                EvidenceRequestDraft(
                    request_key="check_long_tx",
                    tool_id=tool_id,
                    parameters={"min_seconds": 300},
                    hypothesis_keys=("cpu_pressure",),
                    diagnostic_question="是否存在长事务",
                    supports_if="存在运行超过阈值的事务",
                    contradicts_if="没有长事务",
                    priority_reason="可区分事务与非事务负载",
                ),
            ),
            stop_recommendation="CONTINUE",
            stop_reason="需要区分性证据",
        )

    @staticmethod
    def _snapshot() -> dict:
        registry = DiagnosticRegistry.load()
        tool = registry.resolve(
            tool_id="db.transaction.long_running",
            tool_version="1.0.0",
            db_type="ORACLE",
            db_version="23ai",
            capabilities={"dynamic_performance_views"},
            entitlements=set(),
        )
        definition = tool.definition
        return {
            "db_type": "ORACLE",
            "target_row_version": 1,
            "tools": [
                {
                    "tool_id": definition.tool_id,
                    "version": definition.version,
                    "variant": definition.variant,
                    "template_sha256": definition.template_sha256,
                }
            ],
        }

    def test_unknown_tool_is_rejected(self) -> None:
        plan = validate_evidence_requests(
            self._draft("db.unknown"),
            database_snapshot=self._snapshot(),
            registry=DiagnosticRegistry.load(),
            budget=EvidenceRequestBudget(remaining_tool_calls=4),
        )
        self.assertFalse(plan.accepted)
        self.assertEqual(
            "TOOL_NOT_AVAILABLE", plan.rejected[0].reason_code
        )

    def test_typed_catalog_request_is_accepted(self) -> None:
        plan = validate_evidence_requests(
            self._draft(),
            database_snapshot=self._snapshot(),
            registry=DiagnosticRegistry.load(),
            budget=EvidenceRequestBudget(remaining_tool_calls=4),
        )
        self.assertEqual(1, len(plan.accepted))
        self.assertEqual("COLLECT", plan.decision)

    def test_model_contract_rejects_sql_and_limits(self) -> None:
        payload = {
            "request_key": "unsafe",
            "tool_id": "db.session.active",
            "parameters": {},
            "hypothesis_keys": ["h1"],
            "diagnostic_question": "检查会话",
            "supports_if": "存在活跃会话",
            "contradicts_if": "不存在活跃会话",
            "priority_reason": "区分负载",
            "sql": "SELECT * FROM users",
            "timeout": 999,
        }
        with self.assertRaises(ValidationError):
            EvidenceRequestDraft.model_validate(payload)


class RootCausePolicyTest(unittest.TestCase):
    def test_two_independent_groups_with_direct_test_caps_probable(self) -> None:
        evidence = normalize_evidence_artifacts(
            (_monitor_artifact(), _database_artifact()),
            target_id="target-1",
        )
        refs = tuple(item.fact_id for item in evidence.facts)
        assessment = DiagnosisRoundAssessment(
            round_no=1,
            hypothesis_assessments=(
                HypothesisAssessment(
                    hypothesis_key="cpu_pressure",
                    status="SUPPORTED",
                    causal_role="ROOT",
                    supporting_fact_refs=refs,
                    test_results=(
                        {
                            "request_key": "active_sessions",
                            "outcome": "SUPPORTS",
                            "strength": "DIRECT",
                            "fact_refs": (refs[-1],),
                        },
                    ),
                ),
            ),
            suggested_root_cause_level="PROBABLE",
            recommended_next_step="FINALIZE",
            rationale_summary="监控与数据库事实共同支持",
        )
        result = assess_root_cause(
            target_id="target-1",
            evidence=evidence,
            assessment=assessment,
        )
        self.assertEqual("PROBABLE", result.eligible_ceiling)
        self.assertEqual("PROBABLE", result.effective_level)

    def test_single_source_cannot_reach_probable(self) -> None:
        evidence = normalize_evidence_artifacts(
            (_monitor_artifact(),), target_id="target-1"
        )
        assessment = DiagnosisRoundAssessment(
            round_no=1,
            hypothesis_assessments=(
                HypothesisAssessment(
                    hypothesis_key="cpu_pressure",
                    status="SUPPORTED",
                    causal_role="ROOT",
                    supporting_fact_refs=tuple(
                        item.fact_id for item in evidence.facts
                    ),
                ),
            ),
            suggested_root_cause_level="PROBABLE",
            recommended_next_step="FINALIZE",
            rationale_summary="只有单一监控来源",
        )
        result = assess_root_cause(
            target_id="target-1",
            evidence=evidence,
            assessment=assessment,
        )
        self.assertEqual("POSSIBLE", result.effective_level)

    def test_knowledge_citation_cannot_prove_current_state(self) -> None:
        evidence = normalize_evidence_artifacts(
            (
                {
                    "artifact_id": "knowledge-artifact",
                    "schema_version": "KNOWLEDGE_CITATION_PACK.v1",
                    "payload": {
                        "query": "CPU 故障",
                        "citations": [
                            {
                                "citation_label": "K1",
                                "bundle_id": "bundle-1",
                                "items": [
                                    {
                                        "evidence": {
                                            "retrieval_text": (
                                                "CPU 高时应检查活跃会话"
                                            )
                                        }
                                    }
                                ],
                            }
                        ],
                    },
                },
            ),
            target_id="target-1",
        )
        ref = evidence.facts[0].fact_id
        assessment = DiagnosisRoundAssessment(
            round_no=1,
            hypothesis_assessments=(
                HypothesisAssessment(
                    hypothesis_key="cpu_pressure",
                    status="SUPPORTED",
                    causal_role="ROOT",
                    supporting_fact_refs=(ref,),
                ),
            ),
            suggested_root_cause_level="PROBABLE",
            recommended_next_step="FINALIZE",
            rationale_summary="只有知识库依据",
        )
        result = assess_root_cause(
            target_id="target-1",
            evidence=evidence,
            assessment=assessment,
        )
        self.assertEqual("INCONCLUSIVE", result.effective_level)

    def test_empty_knowledge_scope_is_not_an_evidence_gap(self) -> None:
        evidence = normalize_evidence_artifacts(
            (
                {
                    "artifact_id": "knowledge-artifact",
                    "schema_version": "KNOWLEDGE_CITATION_PACK.v1",
                    "payload": {
                        "query": "分析数据库响应变慢",
                        "citations": [],
                        "gap_code": None,
                    },
                },
            ),
            target_id="target-1",
        )
        self.assertEqual((), evidence.gaps)
        self.assertEqual((), evidence.facts)


class DirectAnswerShortCircuitTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _evidence():
        artifact = _monitor_artifact()
        observation = artifact["payload"]["observations"][0]
        observation["metric_code"] = "db.storage.utilization"
        observation["unit"] = "percent"
        observation["summary"] = {"last": 66.14}
        observation["series"] = [
            {
                "dimensions": {"tablespace": "USERS"},
                "points": [
                    {
                        "observed_at": datetime.now(UTC).isoformat(),
                        "value": 66.14,
                        "quality": "GOOD",
                    }
                ],
            }
        ]
        return normalize_evidence_artifacts(
            (artifact,), target_id="target-1"
        )

    @staticmethod
    def _context(*, task_key, artifacts):
        return TaskExecutionContext(
            run_id="run-1",
            task_id="task-1",
            task_key=task_key,
            target_id="target-1",
            agent_id="agent-1",
            trigger_type="CHAT",
            actor_id="user-1",
            original_request="当前表空间使用率是多少",
            trace_id="trace-1",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "diagnosis": {
                    "question_summary": "当前表空间使用率是多少",
                    "model": {"enabled": True},
                }
            },
            policy_snapshot={},
            input_artifacts=artifacts,
        )

    async def test_round_draft_skips_llm_for_complete_monitor_answer(self):
        model = AsyncMock()
        context = self._context(
            task_key="diagnosis:r1:draft",
            artifacts=(
                {
                    "schema_version": "EVIDENCE_INDEX.v1",
                    "payload": self._evidence().model_dump(mode="json"),
                    "provenance": {"task_key": "diagnosis:evidence:r0"},
                },
            ),
        )

        draft = await DiagnosisRoundDraftHandler(
            model_client=model,
            prompts=_TestPrompts(),
        ).execute(context)

        self.assertEqual("FINALIZE", draft.stop_recommendation)
        self.assertEqual((), draft.hypotheses)
        model.generate_structured.assert_not_awaited()

    async def test_round_assessment_skips_llm_for_complete_monitor_answer(self):
        model = AsyncMock()
        evidence = self._evidence()
        context = self._context(
            task_key="diagnosis:r1:assess",
            artifacts=(
                {
                    "schema_version": "EVIDENCE_INDEX.v1",
                    "payload": evidence.model_dump(mode="json"),
                    "provenance": {"task_key": "diagnosis:evidence:r1"},
                },
                {
                    "schema_version": "DIAGNOSIS_ROUND_DRAFT.v1",
                    "payload": DiagnosisRoundDraft(
                        round_no=1,
                        stop_recommendation="FINALIZE",
                        stop_reason="可信监控事实已直接回答用户问题",
                    ).model_dump(mode="json"),
                },
            ),
        )

        assessment = await DiagnosisRoundAssessmentHandler(
            model_client=model,
            prompts=_TestPrompts(),
        ).execute(context)

        self.assertEqual("FINALIZE", assessment.recommended_next_step)
        self.assertIsNone(assessment.model_gap_code)
        model.generate_structured.assert_not_awaited()


class DiagnosisAssetsTest(unittest.TestCase):
    def test_all_aiops_prompts_have_database_seed_baselines(self) -> None:
        catalog = load_prompt_catalog()
        entries = {
            item.prompt_key: item
            for item in catalog.entries
            if item.owner_service == "aiops_agent"
        }

        self.assertEqual(set(PROMPT_KEYS.values()), set(entries))
        self.assertTrue(
            all(item.version == "1.0.0" for item in entries.values())
        )
        self.assertTrue(all(len(item.sha256) == 64 for item in entries.values()))

    def test_blueprint_is_bounded_and_never_calls_mutation(self) -> None:
        blueprint = build_multi_round_diagnosis_blueprint(
            binding_ids=("binding-1",),
            tool_ids=(
                "db.instance.identity",
                "db.session.active",
            ),
            max_rounds=3,
        )
        BlueprintRegistry.validate(blueprint, max_tasks=32)
        self.assertEqual("diagnosis:report", blueprint.final_task_key)
        self.assertIn(
            "diagnosis:r3:assess",
            {item.task_key for item in blueprint.tasks},
        )
        self.assertFalse(
            any(
                "mutation" in item.handler_id.lower()
                for item in blueprint.tasks
            )
        )


class DiagnosisOutputDecisionTest(unittest.TestCase):
    def test_fact_question_returns_simple_conclusion(self) -> None:
        decision = DiagnosisReportHandler._output_decision(
            question="表空间还有多少",
            trigger_type="CHAT",
            root_level="INCONCLUSIVE",
            has_direct_answer=True,
            has_recommendations=False,
            status="READY",
        )
        self.assertEqual(
            (
                "SIMPLE_CONCLUSION",
                "NONE",
                ("INFORMATIONAL_QUERY",),
                False,
            ),
            decision,
        )

    def test_detected_issue_publishes_full_report(self) -> None:
        decision = DiagnosisReportHandler._output_decision(
            question="为什么数据库响应变慢",
            trigger_type="CHAT",
            root_level="PROBABLE",
            has_direct_answer=False,
            has_recommendations=True,
            status="READY",
        )
        self.assertEqual("DIAGNOSIS_REPORT", decision[0])
        self.assertEqual("FULL", decision[1])
        self.assertIn("ISSUE_DETECTED", decision[2])
        self.assertTrue(decision[3])

    def test_alert_always_publishes_report_without_user_interaction(
        self,
    ) -> None:
        decision = DiagnosisReportHandler._output_decision(
            question="Critical alert",
            trigger_type="ALERT",
            root_level="INCONCLUSIVE",
            has_direct_answer=False,
            has_recommendations=False,
            status="PARTIAL",
        )
        self.assertEqual("DIAGNOSIS_REPORT", decision[0])
        self.assertIn("AUTOMATIC_TRIGGER", decision[2])


if __name__ == "__main__":
    unittest.main()
