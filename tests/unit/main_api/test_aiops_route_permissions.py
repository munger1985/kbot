"""AIOps 公开路由核心权限登记完整性测试。"""

import unittest
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.routing import APIRoute

from main_api.api.ops import (
    AIOPS_ROUTE_PERMISSIONS,
    _resolve_work_item_visibility,
    router,
)


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


if __name__ == "__main__":
    unittest.main()
