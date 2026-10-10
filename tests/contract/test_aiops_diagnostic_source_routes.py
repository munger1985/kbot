"""验证监控源命令路由不会遮蔽固定功能端点。"""

from __future__ import annotations

import unittest

from fastapi.routing import APIRoute

from aiops_agent.api.management.routes import router as internal_router
from main_api.api.ops import router as public_router


class AIOpsDiagnosticSourceRouteContractTest(unittest.TestCase):
    def test_commands_use_explicit_paths_without_dynamic_catch_all(self) -> None:
        for router in (public_router, internal_router):
            paths = {
                route.path
                for route in router.routes
                if isinstance(route, APIRoute)
            }
            prefix = router.prefix
            self.assertIn(
                f"{prefix}/diagnostic-sources/{{source_id}}/enable", paths
            )
            self.assertIn(
                f"{prefix}/diagnostic-sources/{{source_id}}/disable", paths
            )
            self.assertIn(
                f"{prefix}/diagnostic-sources/{{source_id}}/instance-discoveries",
                paths,
            )
            self.assertNotIn(
                f"{prefix}/diagnostic-sources/{{source_id}}/{{command}}", paths
            )


if __name__ == "__main__":
    unittest.main()
