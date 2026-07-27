import unittest

from knowledge_core.application.retrieval_plan import (
    DeterministicCandidateSelector, DeterministicEvidenceSupportJudge, RetrievalQueryPlanner,
)


class RetrievalPlanTest(unittest.IsolatedAsyncioTestCase):
    def test_plan_extracts_explicit_phrase_and_identifier(self):
        plan = RetrievalQueryPlanner().plan(query='列出 "AIOps Case" 资产编号 ASSET_2024', task_mode="DISCOVER", target_level="BUNDLE", coverage_mode="BREADTH")
        self.assertEqual("READY", plan.plan_status)
        self.assertEqual(("AIOps Case",), plan.exact_phrases)
        self.assertIn("ASSET_2024", plan.identifiers)

    async def test_degraded_selection_is_object_level(self):
        decisions = await DeterministicCandidateSelector().select(
            query="告警降噪案例", candidates=[{"candidate_label": "bundle-1", "display_title": "告警降噪案例"}],
        )
        self.assertEqual("bundle-1", decisions[0].candidate_label)
        self.assertIn(decisions[0].relevance, {"DIRECT", "STRONG"})

    async def test_support_judge_never_promotes_context_only(self):
        decisions = await DeterministicEvidenceSupportJudge().judge(
            query="告警降噪", groups=[{"group_label": "g1", "primary_text": "仅介绍背景"}],
        )
        self.assertEqual("NO_SUPPORT", decisions[0].support)


if __name__ == "__main__":
    unittest.main()
