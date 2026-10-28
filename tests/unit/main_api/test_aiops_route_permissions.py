"""AIOps 公开路由核心权限登记完整性测试。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.routing import APIRoute

from main_api.api.ops import (
    AIOPS_ROUTE_PERMISSIONS,
    _resolve_work_item_visibility,
    claim_work_item,
    router,
)
from platform_core.contracts.aiops import WorkItemVersionCommand
from platform_core.identity import uuid7


class AIOpsRoutePermissionsTest(unittest.TestCase):
    def test_every_public_endpoint_has_core_permission(self) -> None:
        endpoint_names = {
            route.endpoint.__name__
            for route in router.routes
            if isinstance(route, APIRoute)
        }

        self.assertEqual(set(), endpoint_names - set(AIOPS_ROUTE_PERMISSIONS))
        self.assertEqual(
            "aiops:plan_manage",
            AIOPS_ROUTE_PERMISSIONS["delete_inspection_plan"],
        )
        self.assertEqual(
            "aiops:use", AIOPS_ROUTE_PERMISSIONS["get_recovery_profile"]
        )
        self.assertEqual(
            "aiops:work_item_handle",
            AIOPS_ROUTE_PERMISSIONS["create_recovery_drill"],
        )
        self.assertEqual(
            "aiops:work_item_manage",
            AIOPS_ROUTE_PERMISSIONS["review_recovery_drill"],
        )

    def test_material_and_capture_http_entrypoints_enforce_registered_permissions(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from main_api.api import ops

        app = FastAPI()
        app.include_router(router)
        app.state.allowed_permission = "aiops:use"
        calls = []

        async def authorize(request, *, app_id, permission):
            calls.append(permission)
            if permission != app.state.allowed_permission:
                raise HTTPException(status_code=403)
            request.state.auth_context = object()
            return None, None, SimpleNamespace(permissions=frozenset({permission}))

        downstream = SimpleNamespace(
            list_sql_plan_material_sources=AsyncMock(return_value=[]),
            list_sql_plan_materials=AsyncMock(return_value={"items": []}),
            list_sql_plan_captures=AsyncMock(return_value={"items": [], "can_request": False,
                "can_review": False, "availability": "DISABLED"}),
            list_sql_plan_capture_reviews=AsyncMock(return_value={"items": [], "can_request": False,
                "can_review": True, "availability": "DISABLED"}),
        )
        base = f"/api/v1/apps/aiops/runs/{uuid7()}"
        with patch.object(ops, "authorize_app_request", side_effect=authorize), patch.object(ops, "_client", return_value=downstream):
            client = TestClient(app)
            for suffix in ("/sql-plan-material-sources", "/sql-plan-materials", "/sql-plan-captures"):
                result = client.get(base + suffix)
                self.assertEqual(200, result.status_code, result.text)
                self.assertEqual("no-store", result.headers["cache-control"])
            self.assertEqual(403, client.get(base + "/sql-plan-capture-reviews").status_code)
            downstream.list_sql_plan_capture_reviews.assert_not_awaited()
            app.state.allowed_permission = "aiops:proposal:approve"
            self.assertEqual(200, client.get(base + "/sql-plan-capture-reviews").status_code)
            self.assertEqual(403, client.get(base + "/sql-plan-materials").status_code)
            self.assertEqual(["aiops:use"] * 3 + ["aiops:proposal:approve"] * 2 + ["aiops:use"], calls)

    def test_legacy_target_agent_binding_routes_are_not_public(self) -> None:
        paths = {
            route.path
            for route in router.routes
            if isinstance(route, APIRoute)
        }
        self.assertFalse(any("agent-bindings" in path for path in paths))
        self.assertTrue(
            {
                "list_agent_bindings",
                "create_agent_binding",
                "patch_agent_binding",
                "command_agent_binding",
            }.isdisjoint(AIOPS_ROUTE_PERMISSIONS)
        )

    def test_elevated_work_item_visibility_requires_manage_permission(self) -> None:
        ordinary = SimpleNamespace(state=SimpleNamespace(
            aiops_access_snapshot=SimpleNamespace(permissions=frozenset({"aiops:use"}))
        ))
        manager = SimpleNamespace(state=SimpleNamespace(
            aiops_access_snapshot=SimpleNamespace(
                permissions=frozenset({"aiops:use", "aiops:work_item_manage"})
            )
        ))

        self.assertEqual("SCOPE", _resolve_work_item_visibility(ordinary, "SCOPE"))
        with self.assertRaises(HTTPException) as raised:
            _resolve_work_item_visibility(ordinary, "ALL")
        self.assertEqual(403, raised.exception.status_code)
        self.assertEqual("ALL", _resolve_work_item_visibility(manager, "ALL"))
        self.assertEqual(
            "EXCEPTIONS",
            _resolve_work_item_visibility(manager, "EXCEPTIONS"),
        )

    def test_claim_uses_asserted_user_id_for_group_eligibility(self) -> None:
        request = SimpleNamespace(state=SimpleNamespace(
            auth_context=SimpleNamespace(asserted_user_id="dba-user-1")
        ))
        response = SimpleNamespace(headers={})
        expected = object()

        with patch(
            "main_api.api.ops._require_eligible_aiops_group_member",
            new_callable=AsyncMock,
        ) as require_member, patch(
            "main_api.api.ops._work_item_action",
            new_callable=AsyncMock,
            return_value=expected,
        ) as action:
            actual = asyncio.run(claim_work_item(
                uuid7(),
                WorkItemVersionCommand(expected_row_version=1),
                request,
                response,
            ))

        self.assertIs(expected, actual)
        require_member.assert_awaited_once_with(request, "dba-user-1")
        action.assert_awaited_once()

    def test_claim_rejects_missing_user_context(self) -> None:
        request = SimpleNamespace(state=SimpleNamespace(
            auth_context=SimpleNamespace(asserted_user_id=None)
        ))

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(claim_work_item(
                uuid7(),
                WorkItemVersionCommand(expected_row_version=1),
                request,
                SimpleNamespace(headers={}),
            ))

        self.assertEqual(401, raised.exception.status_code)


if __name__ == "__main__":
    unittest.main()
