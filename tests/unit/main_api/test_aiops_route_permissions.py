"""AIOps 公开路由核心权限登记完整性测试。"""

import unittest

from fastapi.routing import APIRoute

from main_api.api.ops import AIOPS_ROUTE_PERMISSIONS, router


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


if __name__ == "__main__":
    unittest.main()
