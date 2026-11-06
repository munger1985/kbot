"""验证 4.1 的公开与内部 API 版本边界。"""

from __future__ import annotations

import unittest

from fastapi.routing import APIRoute

from knowledge_core.api.parse_jobs_router import create_parse_jobs_router
from knowledge_retrieval_app.api.knowledge_retrieval import create_knowledge_retrieval_router
from knowledge_retrieval_app.api.knowledge_library import create_library_router
from main_api.api import knowledge_router
from model_serving.common.management_router import create_model_management_router
from model_serving.common.openai_router import create_openai_models_router
from platform_core.contracts import INTERNAL_API_V1, PUBLIC_API_V1


async def access_dependency():
    return None


INTERNAL_ROUTERS = (create_parse_jobs_router(service=None, access_dependency=access_dependency, source_resolver=None),
    create_knowledge_retrieval_router(access_dependency=access_dependency), create_library_router(),
    create_model_management_router(category=1),)


class ApiRouteVersionsTest(unittest.TestCase):
    def test_api_version_constants(self) -> None:
        self.assertEqual("/api/v1", PUBLIC_API_V1)
        self.assertEqual("/internal/v1", INTERNAL_API_V1)

    def test_internal_service_routes_use_internal_v1(self) -> None:
        paths = [
            route.path
            for router in INTERNAL_ROUTERS
            for route in router.routes
            if isinstance(route, APIRoute)
        ]

        self.assertTrue(paths)
        self.assertTrue(
            all(path.startswith(f"{INTERNAL_API_V1}/") for path in paths)
        )
        self.assertFalse(
            any(path.startswith(f"{PUBLIC_API_V1}/") for path in paths)
        )

    def test_main_api_routes_use_public_v1(self) -> None:
        paths = [
            route.path
            for router in (knowledge_router,)
            for route in router.routes
            if isinstance(route, APIRoute)
        ]
        self.assertTrue(paths)
        self.assertTrue(
            all(path.startswith(f"{PUBLIC_API_V1}/") for path in paths)
        )
        self.assertFalse(
            any(path.startswith(f"{INTERNAL_API_V1}/") for path in paths)
        )

    def test_model_catalog_has_separate_public_and_internal_routes(self) -> None:
        internal_paths = {
            route.path
            for route in create_model_management_router(category=1).routes
            if isinstance(route, APIRoute)
        }
        public_paths = {
            route.path
            for route in create_openai_models_router(category=1).routes
            if isinstance(route, APIRoute)
        }
        self.assertTrue(
            all(path.startswith(INTERNAL_API_V1) for path in internal_paths)
        )
        self.assertEqual({f"{PUBLIC_API_V1}/models"}, public_paths)


if __name__ == "__main__":
    unittest.main()
