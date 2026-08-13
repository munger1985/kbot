"""KM Asset App 公开 BFF 契约单元测试。"""

import unittest
from uuid import UUID

from pydantic import ValidationError

from main_api.api.km_asset_app import AgentCreatePayload, _km_turn_receipt


class KmAssetAppContractTest(unittest.TestCase):
    def test_public_agent_create_rejects_caller_supplied_capabilities(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            AgentCreatePayload(
                source_id=UUID("01900000-0000-7000-8000-000000000001"),
                display_name="KM Agent",
                enabled_capabilities=["conversation", "document", "data_query"],
            )

        self.assertEqual("extra_forbidden", raised.exception.errors()[0]["type"])
        self.assertEqual(("enabled_capabilities",), raised.exception.errors()[0]["loc"])

    def test_turn_receipt_uses_km_owned_event_stream(self) -> None:
        run_id = UUID("019ff999-fb22-7d92-8e87-49a20b1d18fa")
        upstream = {
            "run_id": str(run_id),
            "status": "RUNNING",
            "event_cursor": 1,
            "events_url": (
                "/api/v1/apps/knowledge-retrieval/runs/"
                f"{run_id}/events"
            ),
        }

        receipt = _km_turn_receipt(upstream)

        self.assertEqual(
            f"/api/v1/apps/km-asset/runs/{run_id}/events",
            receipt["events_url"],
        )
        self.assertIn("knowledge-retrieval", upstream["events_url"])

    def test_turn_receipt_without_run_keeps_events_url_empty(self) -> None:
        receipt = _km_turn_receipt({"run_id": None, "events_url": None})

        self.assertIsNone(receipt["events_url"])


if __name__ == "__main__":
    unittest.main()
