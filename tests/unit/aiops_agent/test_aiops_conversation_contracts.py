"""验证专业 DBA 对话、Turn 和 Skill 计划的共享契约。"""

from datetime import UTC, datetime
import unittest

from pydantic import ValidationError

from platform_core.contracts.aiops import (
    ConversationSourceContext,
    ConversationSourceType,
    DbaIntent,
    DbaIntentPlan,
    DbaSkillPlan,
    EvidenceResponseCreate,
    IntentCandidate,
    MeasurementSemantics,
    SkillPlanItem,
    TurnCreate,
    TurnEventView,
)
from platform_core.identity import uuid7


class AIOpsConversationContractTest(unittest.TestCase):
    def test_turn_create_requires_explicit_idempotency_key(self) -> None:
        turn = TurnCreate(
            content=({"content_type": "TEXT", "text": "分析当前数据库上的 Top SQL"},),
            idempotency_key="turn-1",
        )

        self.assertTrue(turn.content[0].text.endswith("Top SQL"))
        self.assertIsNone(turn.target_id)

        with self.assertRaises(ValidationError):
            TurnCreate(
                content=({"content_type": "TEXT", "text": "分析当前数据库上的 Top SQL"},),
                idempotency_key="",
            )


    def test_conversation_source_accepts_only_matching_resource(self) -> None:
        source_run_id = uuid7()
        source = ConversationSourceContext(
            source_type=ConversationSourceType.RUN,
            run_id=source_run_id,
        )

        self.assertEqual(source.run_id, source_run_id)

        with self.assertRaises(ValidationError):
            ConversationSourceContext(
                source_type=ConversationSourceType.RUN,
                report_id=uuid7(),
            )


    def test_intent_plan_requires_primary_intent_in_candidates(self) -> None:
        with self.assertRaises(ValidationError):
            DbaIntentPlan(
                primary_intent=DbaIntent.OBSERVE,
                candidates=(
                    IntentCandidate(
                        intent=DbaIntent.DIAGNOSE,
                        confidence=0.8,
                        reason="用户要求解释异常原因",
                    ),
                ),
                primary_domain="sql_performance",
            )


    def test_skill_plan_requires_contiguous_forward_only_dag(self) -> None:
        digest = "a" * 64
        valid = DbaSkillPlan(
            catalog_hash=digest,
            items=(
                SkillPlanItem(
                    ordinal=1,
                    skill_id="oracle.sql.top_current",
                    skill_version="1.0.0",
                    manifest_hash=digest,
                    reason="回答当前 Top SQL",
                    evidence_question="哪些 SQL 的累计资源消耗最高",
                    measurement_semantics=MeasurementSemantics.CUMULATIVE_SINCE_LOAD,
                ),
                SkillPlanItem(
                    ordinal=2,
                    skill_id="oracle.sql.detail",
                    skill_version="1.0.0",
                    manifest_hash=digest,
                    reason="补充最高负载 SQL 的详情",
                    evidence_question="最高负载 SQL 的执行统计是什么",
                    measurement_semantics=MeasurementSemantics.CUMULATIVE_SINCE_LOAD,
                    depends_on=(1,),
                ),
            ),
        )

        self.assertEqual(len(valid.items), 2)

        with self.assertRaises(ValidationError):
            DbaSkillPlan(
                catalog_hash=digest,
                items=(valid.items[1],),
            )


    def test_evidence_response_is_explicit_and_exclusive(self) -> None:
        response = EvidenceResponseCreate(
            idempotency_key="evidence-1",
            text="执行结果：no rows selected",
        )
        self.assertIsNotNone(response.text)

        with self.assertRaises(ValidationError):
            EvidenceResponseCreate(idempotency_key="evidence-2")
        with self.assertRaises(ValidationError):
            EvidenceResponseCreate(
                idempotency_key="evidence-3",
                text="结果",
                upload_id=uuid7(),
            )


    def test_turn_event_contract_has_monotonic_cursor_field(self) -> None:
        event = TurnEventView(
            turn_id=uuid7(),
            sequence_no=1,
            event_type="turn.created",
            payload={"status": "QUEUED"},
            occurred_at=datetime.now(UTC),
        )

        self.assertEqual(event.sequence_no, 1)


if __name__ == "__main__":
    unittest.main()
