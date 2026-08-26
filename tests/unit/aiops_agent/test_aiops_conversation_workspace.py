"""AIOps 三入口会话工作区补证输入测试。"""

from types import SimpleNamespace
import unittest

from fastapi import HTTPException

from aiops_agent.api.conversations import _hitl_response


class ConversationEvidenceInputTest(unittest.TestCase):
    def test_single_query_uses_the_whole_conversation_input(self) -> None:
        pending = SimpleNamespace(
            row_version=3,
            request={"queries": [{"query_id": "db.session.active"}]},
        )

        response = _hitl_response(pending, "SID STATUS\n42 ACTIVE")

        self.assertEqual(3, response.expected_row_version)
        self.assertEqual("db.session.active", response.responses[0].query_id)
        self.assertEqual("SID STATUS\n42 ACTIVE", response.responses[0].raw_output)

    def test_multiple_queries_are_split_by_visible_query_heading(self) -> None:
        pending = SimpleNamespace(
            row_version=5,
            request={
                "queries": [
                    {"query_id": "db.instance.identity"},
                    {"query_id": "db.session.active"},
                ]
            },
        )

        response = _hitl_response(
            pending,
            "[db.instance.identity]\nNAME VERSION\nDEV 19c\n"
            "[db.session.active]\nSID STATUS\n42 ACTIVE",
        )

        self.assertEqual(2, len(response.responses))
        self.assertIn("DEV 19c", response.responses[0].raw_output)
        self.assertIn("42 ACTIVE", response.responses[1].raw_output)

    def test_multiple_queries_reject_ambiguous_pasted_output(self) -> None:
        pending = SimpleNamespace(
            row_version=1,
            request={
                "queries": [
                    {"query_id": "first"},
                    {"query_id": "second"},
                ]
            },
        )

        with self.assertRaises(HTTPException) as captured:
            _hitl_response(pending, "无法区分归属的两段结果")

        self.assertEqual(422, captured.exception.status_code)


if __name__ == "__main__":
    unittest.main()
