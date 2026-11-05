"""共享资源范围、资源目录及拓扑投影的回归测试。"""

from __future__ import annotations

import unittest
from copy import deepcopy

from pydantic import ValidationError

from aiops_agent.application.turn_planner import ResourceCatalogAdapter, TurnPlannerService
from platform_core.contracts.aiops import (
    AIOpsCapability,
    ScopeResolutionV1,
    TurnPlanStepV1,
    TurnPlanV1,
)
from platform_core.identity import uuid7


_HASH = "a" * 64


def _systems() -> list[dict]:
    shared_id = str(uuid7())
    systems = []
    for code in ("ERP", "MES"):
        system_id = str(uuid7())
        host = {
            "resource_id": shared_id,
            "system_resource_id": str(uuid7()),
            "resource_code": "SHARED-HOST",
            "display_name": "共享主机",
            "resource_type_key": "HOST",
            "topology_roles": ["HOST"],
            "attributes": {},
            "target_id": None,
            "target_ready": False,
        }
        database = {
            "resource_id": str(uuid7()),
            "system_resource_id": str(uuid7()),
            "resource_code": f"{code}-DB",
            "display_name": f"{code}数据库",
            "resource_type_key": "DATABASE",
            "topology_roles": ["WORKLOAD"],
            "attributes": {},
            "target_id": str(uuid7()),
            "target_ready": True,
        }
        systems.append({
            "business_system_id": system_id,
            "system_code": code,
            "display_name": code,
            "environment": "PROD",
            "resources": [host, database],
            "relations": [{
                "relation_id": str(uuid7()),
                "relation_type_key": "RUNS_ON",
                "display_name": f"{code}承载关系",
                "participants": [
                    {
                        "system_resource_id": resource["system_resource_id"],
                        "resource_id": resource["resource_id"],
                        "display_name": resource["display_name"],
                        "participant_role": resource["topology_roles"][0],
                        "display_order": index,
                    }
                    for index, resource in enumerate((database, host))
                ],
            }],
        })
    return systems


def _scope_payload(systems: list[dict]) -> dict:
    return {
        "status": "RESOLVED",
        "scope_type": "SYSTEM",
        "business_systems": [
            {
                key: system[key]
                for key in ("business_system_id", "display_name", "environment")
            }
            for system in systems
        ],
        "resources": [
            {
                "business_system_id": system["business_system_id"],
                **{
                    key: resource[key]
                    for key in (
                        "resource_id", "system_resource_id", "display_name",
                        "resource_type_key", "topology_roles",
                    )
                },
            }
            for system in systems
            for resource in system["resources"]
        ],
        "authorization_snapshot_hash": _HASH,
    }


class SharedResourceScopeContractTest(unittest.TestCase):
    def test_shared_resource_preserves_both_system_associations(self) -> None:
        payload = _scope_payload(_systems())
        scope = ScopeResolutionV1.model_validate(payload)

        self.assertEqual(4, len(scope.resources))
        self.assertEqual(scope.resources[0].resource_id, scope.resources[2].resource_id)
        self.assertNotEqual(
            scope.resources[0].system_resource_id, scope.resources[2].system_resource_id
        )
        self.assertEqual(
            scope, ScopeResolutionV1.model_validate_json(scope.model_dump_json())
        )

    def test_duplicate_association_is_rejected(self) -> None:
        payload = _scope_payload(_systems())
        payload["resources"].append(deepcopy(payload["resources"][0]))
        with self.assertRaisesRegex(ValidationError, "资源范围不能重复"):
            ScopeResolutionV1.model_validate(payload)

    def test_association_id_cannot_be_reused_across_systems(self) -> None:
        payload = _scope_payload(_systems())
        payload["resources"][2]["system_resource_id"] = (
            payload["resources"][0]["system_resource_id"]
        )
        with self.assertRaisesRegex(ValidationError, "资源范围不能重复"):
            ScopeResolutionV1.model_validate(payload)

    def test_same_system_resource_pair_cannot_use_two_association_ids(self) -> None:
        payload = _scope_payload(_systems())
        duplicate = {**payload["resources"][0], "system_resource_id": str(uuid7())}
        payload["resources"].append(duplicate)
        with self.assertRaisesRegex(ValidationError, "资源范围不能重复"):
            ScopeResolutionV1.model_validate(payload)

    def test_shared_resource_does_not_bypass_system_membership(self) -> None:
        payload = _scope_payload(_systems())
        payload["resources"][2]["business_system_id"] = str(uuid7())
        with self.assertRaisesRegex(ValidationError, "资源必须属于已解析的业务系统"):
            ScopeResolutionV1.model_validate(payload)

    def test_resource_scope_still_requires_one_association(self) -> None:
        payload = _scope_payload(_systems())
        payload["scope_type"] = "RESOURCE"
        payload["resources"] = [payload["resources"][0], payload["resources"][2]]
        with self.assertRaisesRegex(ValidationError, "RESOURCE 范围必须且只能包含一个资源"):
            ScopeResolutionV1.model_validate(payload)
        payload["resources"] = payload["resources"][:1]
        self.assertEqual(1, len(ScopeResolutionV1.model_validate(payload).resources))

    def test_denied_and_not_required_scopes_cannot_expose_shared_resources(self) -> None:
        for status, scope_type, message in (
            ("DENIED", "SYSTEM", "DENIED 范围不能泄露资源引用"),
            ("NOT_REQUIRED", "NONE", "NOT_REQUIRED 范围必须为空"),
        ):
            with self.subTest(status=status):
                payload = _scope_payload(_systems())
                payload.update(status=status, scope_type=scope_type)
                with self.assertRaisesRegex(ValidationError, message):
                    ScopeResolutionV1.model_validate(payload)


class SharedResourceCatalogTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.systems = _systems()
        self.planner = TurnPlannerService(uow_factory=lambda: None)
        self.plan = TurnPlanV1(
            turn_id=uuid7(),
            revision_no=1,
            requested_capabilities=(AIOpsCapability.RESOURCE_CATALOG_QUERY,),
            steps=(TurnPlanStepV1(
                step_id="s1",
                capability=AIOpsCapability.RESOURCE_CATALOG_QUERY,
                objective="列出授权业务系统及资源拓扑",
                scope_type="SYSTEM",
            ),),
            policy_snapshot_hash=_HASH,
        )

    def _resolve(self, question: str):
        return self.planner._resolve_scope({
            "question": question,
            "policy_snapshot_hash": _HASH,
            "systems": self.systems,
            "context_hint": {},
        }, self.plan)

    async def test_catalog_keeps_shared_resources_types_and_each_system_topology(self) -> None:
        scope, internal, clarification = self._resolve(
            "列出我有权访问的业务系统及其系统资源、资源类型和拓扑关系。"
        )

        self.assertIsNone(clarification)
        self.assertEqual("RESOLVED", scope.status)
        self.assertEqual(2, len(scope.business_systems))
        self.assertEqual(4, len(scope.resources))
        self.assertEqual(2, len(scope.relations))
        self.assertEqual(4, len(internal["resource_catalog"]))
        self.assertEqual(2, len(internal["execution_targets"]))
        self.assertEqual(2, len(internal["topology_relations"]))
        for system, relation in zip(self.systems, internal["topology_relations"]):
            self.assertEqual(system["business_system_id"], relation["business_system_id"])
            self.assertEqual(
                {item["system_resource_id"] for item in system["resources"]},
                {item["system_resource_id"] for item in relation["participants"]},
            )

        result = await ResourceCatalogAdapter().execute(
            capability=AIOpsCapability.RESOURCE_CATALOG_QUERY,
            plan=self.plan,
            scope=scope,
            internal_scope=internal,
        )
        self.assertTrue(result.supported)
        for code in ("ERP", "MES"):
            self.assertIn(f"- {code}（PROD）", result.markdown)
            self.assertIn(f"{code}数据库 [DATABASE]", result.markdown)
            self.assertIn(f"{code}承载关系", result.markdown)
            self.assertIn(
                f"{code}数据库（WORKLOAD） → 共享主机（HOST）", result.markdown
            )
        self.assertEqual(2, result.markdown.count("共享主机 [HOST]"))

    async def test_catalog_details_are_selected_by_system_resource_association(self) -> None:
        scope, internal, _ = self._resolve("列出所有业务系统资源")
        shared_id = self.systems[0]["resources"][0]["resource_id"]
        codes = {
            system["business_system_id"]: system["system_code"]
            for system in self.systems
        }
        for item in internal["resource_catalog"]:
            if item["resource_id"] == shared_id:
                item["location_name"] = f"{codes[item['business_system_id']]}范围位置"

        result = await ResourceCatalogAdapter().execute(
            capability=AIOpsCapability.RESOURCE_CATALOG_QUERY,
            plan=self.plan,
            scope=scope,
            internal_scope=internal,
        )

        self.assertEqual(1, result.markdown.count("位置 ERP范围位置"))
        self.assertEqual(1, result.markdown.count("位置 MES范围位置"))

    async def test_explicit_shared_resource_keeps_both_associations(self) -> None:
        scope, internal, clarification = self._resolve("列出共享主机")
        self.assertIsNone(clarification)
        self.assertEqual("RESOURCE_SET", scope.scope_type)
        self.assertEqual(2, len(scope.resources))
        self.assertEqual(2, len(internal["resource_catalog"]))
        self.assertEqual(1, len({item.resource_id for item in scope.resources}))

    async def test_single_system_keeps_only_its_own_associations(self) -> None:
        scope, internal, clarification = self._resolve("列出 ERP 的系统资源和拓扑关系")
        self.assertIsNone(clarification)
        self.assertEqual(["ERP"], [item.display_name for item in scope.business_systems])
        self.assertEqual(2, len(scope.resources))
        self.assertEqual(1, len(internal["topology_relations"]))
        self.assertEqual(
            {self.systems[0]["business_system_id"]},
            {item["business_system_id"] for item in internal["resource_catalog"]},
        )


if __name__ == "__main__":
    unittest.main()
