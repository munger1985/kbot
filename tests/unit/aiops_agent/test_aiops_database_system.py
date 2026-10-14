"""数据库系统 Wizard 共享契约测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from platform_core.contracts.aiops import (
    DatabaseSystemCreate,
    DatabaseSystemTopologyPut,
)
from platform_core.identity import uuid7


class DatabaseSystemContractTest(unittest.TestCase):
    def test_engine_is_selected_before_compatible_architecture(self) -> None:
        oracle = DatabaseSystemCreate(
            display_name="ERP RAC + DG",
            db_type="ORACLE",
            version_code="19c",
            deployment_shape="HIGH_AVAILABILITY",
            deployment_category="COMPOSITE",
            capabilities=("RAC", "DATA_GUARD"),
            environment="PROD",
        )
        self.assertEqual(("RAC", "DATA_GUARD"), oracle.capabilities)
        with self.assertRaises(ValidationError):
            DatabaseSystemCreate(
                display_name="错误组合",
                db_type="POSTGRESQL",
                version_code="16",
                deployment_shape="HIGH_AVAILABILITY",
                deployment_category="RAC",
                capabilities=("RAC",),
                environment="PROD",
            )

    def test_topology_references_are_structured_and_closed(self) -> None:
        primary = uuid7()
        standby = uuid7()
        topology = DatabaseSystemTopologyPut(
            sites=(
                {"site_code": "PRIMARY", "display_name": "主站点", "expected_role": "PRIMARY"},
                {"site_code": "STANDBY", "display_name": "备站点", "expected_role": "STANDBY"},
            ),
            members=(
                {"member_key": "PRIMARY-1", "target_id": primary, "site_code": "PRIMARY", "member_type": "INSTANCE", "expected_role": "PRIMARY"},
                {"member_key": "STANDBY-1", "target_id": standby, "site_code": "STANDBY", "member_type": "INSTANCE", "expected_role": "STANDBY"},
            ),
            relations=(
                {"relation_key": "STREAM-1", "source_member_key": "PRIMARY-1", "target_member_key": "STANDBY-1", "relation_type": "POSTGRES_STREAMING", "transport_mode": "ASYNC"},
            ),
        )
        self.assertEqual("PRIMARY-1", topology.relations[0].source_member_key)

    def test_target_cannot_be_reused_as_two_members(self) -> None:
        target_id = uuid7()
        with self.assertRaises(ValidationError):
            DatabaseSystemTopologyPut(
                sites=(
                    {"site_code": "PRIMARY", "display_name": "主站点", "expected_role": "PRIMARY"},
                    {"site_code": "STANDBY", "display_name": "备站点", "expected_role": "STANDBY"},
                ),
                members=(
                    {"member_key": "PRIMARY-1", "target_id": target_id, "site_code": "PRIMARY", "member_type": "INSTANCE", "expected_role": "PRIMARY"},
                    {"member_key": "STANDBY-1", "target_id": target_id, "site_code": "STANDBY", "member_type": "INSTANCE", "expected_role": "STANDBY"},
                ),
            )


if __name__ == "__main__":
    unittest.main()
