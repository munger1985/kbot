"""业务系统资源图谱共享契约测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from aiops_agent.application.configuration import (
    AIOpsConfigurationService,
    ConfigurationScope,
)
from aiops_agent.application.configuration.business_system_service import _summary
from platform_core.contracts.aiops import (
    BusinessSystemLifecycleCommand,
    BusinessSystemResourceGraphCommit,
    BusinessSystemResourceGraphDetail,
    ManagedResourcePatch,
)
from platform_core.identity import uuid7


def _resource(draft_id, resource_id, group_id, name):
    return {
        "draft_resource_id": draft_id,
        "mode": "NEW",
        "resource_type_key": "DATABASE",
        "display_name": name,
        "owner_group_id": group_id,
        "architecture_layer_key": "DATA",
        "database": {
            "target": {"display_name": name, "db_type": "ORACLE", "version_code": "19c", "environment": "PROD"},
            "monitoring": {"source_id": uuid7(), "candidate_ref": "signed-candidate"},
        },
    }


class BusinessSystemResourceGraphContractTest(unittest.TestCase):
    def test_lifecycle_command_limits_reason_length(self) -> None:
        self.assertIsNone(BusinessSystemLifecycleCommand().reason)
        with self.assertRaises(ValidationError):
            BusinessSystemLifecycleCommand(reason="x" * 1001)

    def test_existing_resource_cannot_be_attached_to_another_system(self) -> None:
        payload = _resource(uuid7(), uuid7(), uuid7(), "共享数据库")
        payload["mode"] = "REUSE"
        payload["resource_id"] = uuid7()
        with self.assertRaises(ValidationError):
            BusinessSystemResourceGraphCommit.model_validate({"system": {"display_name": "MES", "environment": "PROD", "responsibility_group_id": payload["owner_group_id"]}, "resources": [payload]})

    def test_new_database_requires_monitoring_mapping(self) -> None:
        payload = _resource(uuid7(), uuid7(), uuid7(), "核心数据库")
        payload["database"].pop("monitoring")
        with self.assertRaises(ValidationError):
            BusinessSystemResourceGraphCommit.model_validate({"system": {"display_name": "MES", "environment": "PROD", "responsibility_group_id": payload["owner_group_id"]}, "resources": [payload]})

    def test_rac_requires_oracle_and_node_number(self) -> None:
        group_id = uuid7()
        payload = _resource(uuid7(), uuid7(), group_id, "MES RAC")
        payload["database"]["target"]["deployment_type"] = "RAC"
        with self.assertRaisesRegex(ValidationError, "必须填写节点编号"):
            BusinessSystemResourceGraphCommit.model_validate({"system": {"display_name": "MES", "environment": "PROD", "responsibility_group_id": group_id}, "resources": [payload]})

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
        patch = ManagedResourcePatch(database={"version_code": "26ai", "environment": "STG", "deployment_type": "RAC", "rac_node_number": 2, "db_role": "STANDBY"})
        self.assertEqual("26ai", patch.database.version_code)
        self.assertEqual("RAC", patch.database.deployment_type)
        self.assertEqual(2, patch.database.rac_node_number)
        self.assertEqual("STANDBY", patch.database.db_role)


class BusinessSystemResourceGraphServiceTest(unittest.IsolatedAsyncioTestCase):
    def scope(self) -> ConfigurationScope:
        return ConfigurationScope(
            domain_id=100,
            principal_id="PORTAL:aiops",
            actor_id="operator-1",
            request_id="request-1",
            trace_id="trace-1",
        )

    async def test_disable_requires_reason_before_writing(self) -> None:
        service = object.__new__(AIOpsConfigurationService)
        service._idempotent = AsyncMock()
        with self.assertRaisesRegex(Exception, "请填写下线原因"):
            await service.command_business_system(
                scope=self.scope(),
                business_system_id=uuid7(),
                command="disable",
                request=BusinessSystemLifecycleCommand(reason="   "),
                expected_version=1,
                idempotency_key=str(uuid7()),
            )
        service._idempotent.assert_not_awaited()

    async def test_delete_draft_is_soft_and_does_not_delete_resources(self) -> None:
        entity = SimpleNamespace(
            business_system_id=uuid7(),
            status="DRAFT",
            row_version=1,
            updated_by=None,
            updated_at=None,
        )
        repository = SimpleNamespace(get_scoped=AsyncMock(return_value=entity))
        uow = SimpleNamespace(business_systems=repository, flush=AsyncMock())

        class UowContext:
            async def __aenter__(self):
                return uow

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        service = object.__new__(AIOpsConfigurationService)
        service._uow_factory = UowContext
        service._now = lambda: datetime.now(UTC)
        scope = self.scope()
        with patch(
            "aiops_agent.application.configuration.business_system_service.add_configuration_event",
            new_callable=AsyncMock,
        ) as add_event:
            await service.delete_business_system_draft(
                scope=scope,
                business_system_id=entity.business_system_id,
                expected_version=1,
            )

        self.assertEqual("DELETED", entity.status)
        repository.get_scoped.assert_awaited_once_with(
            business_system_id=entity.business_system_id,
            lock=True,
            include_deleted=True,
            domain_id=scope.domain_id,
        )
        uow.flush.assert_awaited_once()
        self.assertEqual({"cascade": False}, add_event.await_args.kwargs["details"])

    async def test_list_business_systems_awaits_each_detail_projection(self) -> None:
        now = datetime.now(UTC)

        def detail(name: str) -> BusinessSystemResourceGraphDetail:
            return BusinessSystemResourceGraphDetail(
                business_system_id=uuid7(),
                system_code=f"SYS-{name}",
                display_name=name,
                environment="PROD",
                importance_level=3,
                responsibility_group_id=uuid7(),
                status="ENABLED",
                protection_status="UNCONFIGURED",
                resource_count=1,
                resource_type_summary=(
                    {
                        "resource_type_key": "DATABASE",
                        "display_name": "数据库",
                        "count": 1,
                    },
                ),
                row_version=1,
                created_at=now,
                updated_at=now,
            )

        rows = [SimpleNamespace(), SimpleNamespace(), SimpleNamespace()]
        repository = SimpleNamespace(
            list_scoped=AsyncMock(return_value=rows),
        )
        uow = SimpleNamespace(business_systems=repository)

        class UowContext:
            async def __aenter__(self):
                return uow

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        service = object.__new__(AIOpsConfigurationService)
        service._uow_factory = UowContext
        service._system_detail = AsyncMock(
            side_effect=(detail("MES"), detail("ERP"), detail("WMS")),
        )
        scope = ConfigurationScope(
            domain_id=100,
            principal_id="PORTAL:aiops",
            actor_id="operator-1",
            request_id="request-1",
            trace_id="trace-1",
        )

        result = await service.list_business_systems(
            scope=scope,
            status=None,
            limit=200,
        )

        self.assertEqual(
            ("MES", "ERP", "WMS"),
            tuple(item.display_name for item in result.items),
        )
        self.assertEqual(3, service._system_detail.await_count)
        repository.list_scoped.assert_awaited_once_with(
            status=None,
            limit=200,
            domain_id=scope.domain_id,
        )


if __name__ == "__main__":
    unittest.main()
