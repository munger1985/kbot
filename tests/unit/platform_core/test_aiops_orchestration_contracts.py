"""AIOps 统一助手编排共享合同的聚焦测试。"""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

import platform_core.contracts.aiops as aiops_contracts
from platform_core.contracts.aiops.orchestration import (
    AIOpsCapability,
    AIOpsStructuredAnswerV2,
    BusinessSystemRefV1,
    ClarificationCandidateV1,
    ClarificationRequestV1,
    ClarificationResponseV1,
    DelegationSpecV1,
    EvidenceDescriptorV2,
    ExecutionBudgetV1,
    ResolvedExecutionTargetV1,
    ScopeResolutionV1,
    StructuredAnswerBlockV2,
    StructuredAnswerCitationV2,
    TimeRangeV1,
    TurnPlanStepV1,
    TurnPlanV1,
)
from platform_core.identity import uuid7


_HASH = "a" * 64


def _resolved_scope() -> ScopeResolutionV1:
    return ScopeResolutionV1(
        status="RESOLVED",
        scope_type="SYSTEM",
        business_systems=(
            BusinessSystemRefV1(
                business_system_id=uuid7(),
                display_name="MES",
                environment="PROD",
            ),
        ),
        resolution_basis=("EXPLICIT_NAME",),
        confidence=0.98,
        authorization_snapshot_hash=_HASH,
        resource_graph_snapshot_hash="b" * 64,
    )


class AIOpsOrchestrationContractTest(unittest.TestCase):
    def test_public_package_exports_versioned_contracts(self) -> None:
        for name in (
            "TurnPlanV1",
            "ScopeResolutionV1",
            "ResolvedExecutionTargetV1",
            "DelegationSpecV1",
            "DelegationResultV1",
            "EvidenceDescriptorV2",
            "ClarificationRequestV1",
            "ClarificationResponseV1",
            "AIOpsStructuredAnswerV2",
        ):
            self.assertTrue(hasattr(aiops_contracts, name), name)

    def test_capability_catalog_is_product_specific(self) -> None:
        self.assertEqual(
            {
                "RESOURCE_CATALOG_QUERY",
                "KNOWLEDGE_QA",
                "MONITORING_QUERY",
                "ALERT_QUERY",
                "INSPECTION_QUERY",
                "REPORT_CASE_QUERY",
                "ACTIVE_DIAGNOSIS",
                "CONTROLLED_CHANGE",
            },
            {item.value for item in AIOpsCapability},
        )

    def test_compound_plan_validates_time_budget_and_dependency_dag(self) -> None:
        plan = TurnPlanV1(
            turn_id=uuid7(),
            revision_no=1,
            requested_capabilities=(
                "RESOURCE_CATALOG_QUERY",
                "MONITORING_QUERY",
                "ACTIVE_DIAGNOSIS",
            ),
            requested_facts=("MES 主机", "CPU 异常原因"),
            time_range=TimeRangeV1(relative_window_seconds=3600),
            freshness="LIVE_REQUIRED",
            requires_active_diagnosis=True,
            output_mode="DIAGNOSIS",
            budget=ExecutionBudgetV1(
                max_delegations=3,
                max_parallel_delegations=2,
                max_live_queries=8,
                max_diagnosis_runs=3,
            ),
            steps=(
                TurnPlanStepV1(
                    step_id="s1",
                    capability="RESOURCE_CATALOG_QUERY",
                    objective="解析 MES 资源拓扑",
                    scope_type="SYSTEM",
                ),
                TurnPlanStepV1(
                    step_id="s2",
                    capability="MONITORING_QUERY",
                    objective="定位 CPU 异常资源",
                    scope_type="RESOURCE_SET",
                    depends_on=("s1",),
                ),
                TurnPlanStepV1(
                    step_id="s3",
                    capability="ACTIVE_DIAGNOSIS",
                    objective="诊断焦点资源",
                    scope_type="RESOURCE",
                    depends_on=("s2",),
                ),
            ),
            policy_snapshot_hash=_HASH,
        )
        self.assertEqual(3, len(plan.steps))

        with self.assertRaises(ValidationError):
            TurnPlanV1(
                **{
                    **plan.model_dump(),
                    "steps": (
                        plan.steps[0].model_copy(update={"depends_on": ("s3",)}),
                        plan.steps[1],
                        plan.steps[2],
                    ),
                }
            )
        with self.assertRaises(ValidationError):
            TimeRangeV1(
                start_at=datetime.now(UTC),
                end_at=datetime.now(UTC) - timedelta(minutes=1),
            )
        with self.assertRaises(ValidationError):
            ExecutionBudgetV1(
                max_delegations=2,
                max_parallel_delegations=3,
            )

    def test_public_scope_never_contains_target_identity(self) -> None:
        schema = json.dumps(
            ScopeResolutionV1.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
        )
        self.assertNotIn("target_id", schema)

        scope = _resolved_scope()
        execution_target = ResolvedExecutionTargetV1(
            business_system_id=scope.business_systems[0].business_system_id,
            system_resource_id=uuid7(),
            resource_id=uuid7(),
            target_id=uuid7(),
            readiness="READY",
        )
        delegation = DelegationSpecV1(
            delegation_id=uuid7(),
            turn_id=uuid7(),
            plan_revision_no=1,
            step_id="s1",
            capability="ACTIVE_DIAGNOSIS",
            objective="诊断 MES 数据库",
            scope=scope,
            execution_targets=(execution_target,),
            idempotency_key="delegation-1",
            timeout_seconds=300,
        )
        self.assertEqual(execution_target.target_id, delegation.execution_targets[0].target_id)

    def test_clarification_uses_opaque_candidate_and_exclusive_response(self) -> None:
        candidate = ClarificationCandidateV1(
            candidate_id="Y2FuZGlkYXRlX29uZQ",
            label="MES 生产系统",
            scope_type="SYSTEM",
        )
        request = ClarificationRequestV1(
            clarification_id=uuid7(),
            turn_id=uuid7(),
            question="请选择你指的 MES 系统",
            reason_code="SYSTEM_AMBIGUOUS",
            candidates=(
                candidate,
                candidate.model_copy(
                    update={
                        "candidate_id": "Y2FuZGlkYXRlX3R3bw",
                        "label": "MES 测试系统",
                    }
                ),
            ),
        )
        response = ClarificationResponseV1(
            clarification_id=request.clarification_id,
            idempotency_key="clarification-1",
            candidate_id=candidate.candidate_id,
        )
        self.assertEqual(candidate.candidate_id, response.candidate_id)
        with self.assertRaises(ValidationError):
            ClarificationResponseV1(
                clarification_id=request.clarification_id,
                idempotency_key="clarification-2",
                candidate_id=candidate.candidate_id,
                free_text="生产系统",
            )

    def test_evidence_and_structured_answer_keep_citations_closed(self) -> None:
        scope = _resolved_scope()
        evidence = EvidenceDescriptorV2(
            evidence_id=uuid7(),
            artifact_id=uuid7(),
            source_kind="MONITORING",
            evidence_kind="CPU_UTILIZATION",
            trust_level="SOURCE_VERIFIED",
            captured_at=datetime.now(UTC),
            freshness="FRESH",
            summary="MES 节点 CPU 使用率为 95%",
        )
        citation = StructuredAnswerCitationV2(
            citation_id="C1",
            evidence_id=evidence.evidence_id,
            label="CPU 监控",
        )
        answer = AIOpsStructuredAnswerV2(
            turn_id=uuid7(),
            status="COMPLETED",
            scope=scope,
            capabilities_used=("MONITORING_QUERY",),
            generated_at=datetime.now(UTC),
            coverage="SUPPORTED",
            evidence=(evidence,),
            citations=(citation,),
            blocks=(
                StructuredAnswerBlockV2(
                    block_id="B1",
                    block_type="MONITORING_SUMMARY",
                    schema_version="aiops.monitoring-summary.v1",
                    payload={"summary": "CPU 使用率异常"},
                    citation_ids=("C1",),
                ),
            ),
        )
        self.assertEqual("C1", answer.blocks[0].citation_ids[0])
        with self.assertRaises(ValidationError):
            answer.model_copy(
                update={
                    "blocks": (
                        answer.blocks[0].model_copy(
                            update={"citation_ids": ("C2",)}
                        ),
                    )
                }
            ).model_validate(
                answer.model_copy(
                    update={
                        "blocks": (
                            answer.blocks[0].model_copy(
                                update={"citation_ids": ("C2",)}
                            ),
                        )
                    }
                ).model_dump()
            )


if __name__ == "__main__":
    unittest.main()
