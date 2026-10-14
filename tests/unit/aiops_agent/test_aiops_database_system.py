"""业务系统 Wizard 共享契约测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from platform_core.contracts.aiops import BusinessSystemCommit, BusinessSystemTopologyPut
from platform_core.identity import uuid7


def _component(group_id):
    return {"component_key": "DATABASE-1", "display_name": "ERP 数据库", "component_type": "DATABASE", "responsibility_group_id": group_id}


class BusinessSystemContractTest(unittest.TestCase):
    def test_topology_is_inferred_from_relations_and_cluster_groups(self) -> None:
        topology = BusinessSystemTopologyPut.model_validate({
            "components": [_component(uuid7())],
            "sites": [{"site_code": "MAIN", "display_name": "默认站点"}],
            "members": [
                {"member_key": "RAC-1", "component_key": "DATABASE-1", "site_code": "MAIN", "cluster_group_key": "ERP-RAC", "expected_role": "PRIMARY", "mode": "REUSE", "target_id": uuid7()},
                {"member_key": "RAC-2", "component_key": "DATABASE-1", "site_code": "MAIN", "cluster_group_key": "ERP-RAC", "expected_role": "STANDBY", "mode": "REUSE", "target_id": uuid7()},
            ],
            "relations": [{"relation_key": "DG-1", "source_component_key": "DATABASE-1", "target_component_key": "DATABASE-1", "source_member_key": "RAC-1", "target_member_key": "RAC-2", "relation_type": "DATA_GUARD", "transport_mode": "ASYNC"}],
        })
        self.assertEqual("ERP-RAC", topology.members[0].cluster_group_key)
        self.assertFalse(hasattr(topology, "deployment_shape"))

    def test_target_cannot_be_reused_as_two_members(self) -> None:
        target_id = uuid7()
        with self.assertRaises(ValidationError):
            BusinessSystemTopologyPut.model_validate({
                "components": [_component(uuid7())], "sites": [{"site_code": "MAIN", "display_name": "默认站点"}],
                "members": [
                    {"member_key": "DB-1", "component_key": "DATABASE-1", "site_code": "MAIN", "expected_role": "INSTANCE", "mode": "REUSE", "target_id": target_id},
                    {"member_key": "DB-2", "component_key": "DATABASE-1", "site_code": "MAIN", "expected_role": "INSTANCE", "mode": "REUSE", "target_id": target_id},
                ],
            })

    def test_relation_member_must_belong_to_declared_component(self) -> None:
        group_id = uuid7()
        with self.assertRaisesRegex(ValidationError, "成员必须属于对应"):
            BusinessSystemTopologyPut.model_validate({
                "components": [
                    _component(group_id),
                    {**_component(group_id), "component_key": "DATABASE-2"},
                ],
                "sites": [{"site_code": "MAIN", "display_name": "默认站点"}],
                "members": [
                    {"member_key": "DB-1", "component_key": "DATABASE-1", "site_code": "MAIN", "expected_role": "SOURCE", "mode": "REUSE", "target_id": uuid7()},
                    {"member_key": "DB-2", "component_key": "DATABASE-2", "site_code": "MAIN", "expected_role": "REPLICA", "mode": "REUSE", "target_id": uuid7()},
                ],
                "relations": [{"relation_key": "OGG-1", "source_component_key": "DATABASE-2", "target_component_key": "DATABASE-1", "source_member_key": "DB-1", "target_member_key": "DB-2", "relation_type": "OGG"}],
            })

    def test_atomic_commit_can_update_reused_target_version(self) -> None:
        group_id = uuid7()
        request = BusinessSystemCommit.model_validate({
            "system": {"system_code": "ERP-PROD", "display_name": "ERP 生产系统", "environment": "PROD", "responsibility_group_id": group_id},
            "components": [_component(group_id)], "sites": [{"site_code": "MAIN", "display_name": "默认站点"}],
            "members": [{"member_key": "DB-1", "component_key": "DATABASE-1", "site_code": "MAIN", "expected_role": "INSTANCE", "mode": "REUSE", "target_id": uuid7(), "target_patch": {"display_name": "ERP-DB", "version_code": "23ai", "environment": "PROD"}, "target_row_version": 1}],
        })
        self.assertEqual("23ai", request.members[0].target_patch.version_code)


if __name__ == "__main__":
    unittest.main()
