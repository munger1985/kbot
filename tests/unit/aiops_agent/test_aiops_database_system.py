"""业务系统资源图谱共享契约测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from pydantic import ValidationError

from aiops_agent.application.configuration.business_system_service import _summary
from platform_core.contracts.aiops import (
    BusinessSystemResourceGraphCommit,
    BusinessSystemResourceGraphDetail,
    ManagedResourcePatch,
)
from platform_core.identity import uuid7


def _resource(draft_id, resource_id, group_id, name):
    return {
        "draft_resource_id": draft_id,
        "mode": "REUSE",
        "resource_id": resource_id,
        "resource_type_key": "DATABASE",
        "display_name": name,
        "owner_group_id": group_id,
        "architecture_layer_key": "DATA",
    }


class BusinessSystemResourceGraphContractTest(unittest.TestCase):
    def test_detail_is_projected_to_strict_list_summary(self) -> None:
        now = datetime.now(UTC)
        detail = BusinessSystemResourceGraphDetail(
            business_system_id=uuid7(), system_code="SYS-123456", display_name="ERP 生产系统",
            environment="PROD", importance_level=3, responsibility_group_id=uuid7(),
            description="详情字段不应进入摘要校验", status="DRAFT",
            protection_status="UNCONFIGURED", resource_count=2,
            resource_type_summary=({"resource_type_key": "DATABASE", "display_name": "数据库", "count": 2},),
            architecture_labels=("DATA_GUARD",), row_version=1, created_at=now, updated_at=now,
        )
        summary = _summary(detail)
        self.assertEqual(detail.business_system_id, summary.business_system_id)
        self.assertEqual(2, summary.resource_count)
        self.assertNotIn("description", summary.model_dump())
        self.assertNotIn("resources", summary.model_dump())

    def test_commit_uses_resources_and_multi_participant_relations(self) -> None:
        group_id = uuid7()
        first, second = uuid7(), uuid7()
        request = BusinessSystemResourceGraphCommit.model_validate({
            "system": {"display_name": "ERP 生产系统", "environment": "PROD", "responsibility_group_id": group_id},
            "resources": [_resource(first, uuid7(), group_id, "ERP 主库"), _resource(second, uuid7(), group_id, "ERP 备库")],
            "relations": [{"draft_relation_id": uuid7(), "relation_type_key": "DATA_GUARD", "participants": [{"draft_resource_id": first, "participant_role": "PRIMARY"}, {"draft_resource_id": second, "participant_role": "STANDBY"}]}],
        })
        self.assertEqual(2, len(request.resources))
        self.assertEqual("DATA_GUARD", request.relations[0].relation_type_key)
        self.assertEqual(("PRIMARY", "STANDBY"), tuple(item.participant_role for item in request.relations[0].participants))
        self.assertFalse(hasattr(request, "components"))

    def test_relation_cannot_repeat_a_resource(self) -> None:
        group_id = uuid7()
        draft_id = uuid7()
        with self.assertRaisesRegex(ValidationError, "同一资源不能在一个关系中重复出现"):
            BusinessSystemResourceGraphCommit.model_validate({
                "system": {"display_name": "MES", "environment": "PROD", "responsibility_group_id": group_id},
                "resources": [_resource(draft_id, uuid7(), group_id, "MES DB")],
                "relations": [{"draft_relation_id": uuid7(), "relation_type_key": "ORACLE_RAC", "participants": [{"draft_resource_id": draft_id, "participant_role": "MEMBER"}, {"draft_resource_id": draft_id, "participant_role": "MEMBER"}]}],
            })

    def test_relation_must_reference_a_resource_in_the_same_draft(self) -> None:
        group_id = uuid7()
        first = uuid7()
        with self.assertRaisesRegex(ValidationError, "不在当前业务系统草稿中的资源"):
            BusinessSystemResourceGraphCommit.model_validate({
                "system": {"display_name": "MES", "environment": "PROD", "responsibility_group_id": group_id},
                "resources": [_resource(first, uuid7(), group_id, "MES DB")],
                "relations": [{"draft_relation_id": uuid7(), "relation_type_key": "DATA_GUARD", "participants": [{"draft_resource_id": first, "participant_role": "PRIMARY"}, {"draft_resource_id": uuid7(), "participant_role": "STANDBY"}]}],
            })

    def test_database_version_environment_and_role_are_mutable(self) -> None:
        patch = ManagedResourcePatch(database={"version_code": "26ai", "environment": "STG", "db_role": "STANDBY"})
        self.assertEqual("26ai", patch.database.version_code)
        self.assertEqual("STANDBY", patch.database.db_role)


if __name__ == "__main__":
    unittest.main()
