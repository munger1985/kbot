"""数据库系统 Wizard 共享契约测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from platform_core.contracts.aiops import (
    DatabaseSystemCommit,
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

    def test_atomic_commit_supports_new_and_reused_members(self) -> None:
        request = DatabaseSystemCommit.model_validate({
            "system": {"display_name": "ERP DG", "db_type": "ORACLE", "version_code": "19c", "deployment_shape": "HIGH_AVAILABILITY", "deployment_category": "REPLICATION", "capabilities": ["DATA_GUARD"], "environment": "PROD"},
            "sites": [{"site_code": "PRIMARY", "display_name": "主站点"}, {"site_code": "STANDBY", "display_name": "备站点"}],
            "members": [
                {"member_key": "PRIMARY-1", "site_code": "PRIMARY", "member_type": "INSTANCE", "expected_role": "PRIMARY", "mode": "NEW", "target": {"display_name": "ERP-PRI", "db_type": "ORACLE", "version_code": "19c", "environment": "PROD"}},
                {"member_key": "STANDBY-1", "site_code": "STANDBY", "member_type": "INSTANCE", "expected_role": "STANDBY", "mode": "REUSE", "target_id": uuid7()},
            ],
            "relations": [{"relation_key": "DG-1", "source_member_key": "PRIMARY-1", "target_member_key": "STANDBY-1", "relation_type": "DATA_GUARD", "transport_mode": "ASYNC"}],
        })
        self.assertEqual("NEW", request.members[0].mode)
        self.assertEqual("REUSE", request.members[1].mode)
        with self.assertRaises(ValidationError):
            DatabaseSystemCommit.model_validate({**request.model_dump(mode="json"), "members": [{**request.members[0].model_dump(mode="json"), "target_id": str(uuid7())}]})


if __name__ == "__main__":
    unittest.main()
