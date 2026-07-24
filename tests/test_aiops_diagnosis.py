"""AIOps 步骤 7 证据、Planner Gate 与根因等级测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

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
from aiops_agent.orchestration.diagnosis import DiagnosisPromptRegistry


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
                "MONITOR_METRIC",
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


class DiagnosisAssetsTest(unittest.TestCase):
    def test_prompt_assets_are_versioned_and_hashed(self) -> None:
        registry = DiagnosisPromptRegistry.load()
        prompt = registry.resolve("round_draft")
        self.assertEqual(64, len(prompt.sha256))
        self.assertNotIn("SELECT", prompt.content)

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


if __name__ == "__main__":
    unittest.main()
