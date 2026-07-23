import unittest
from types import SimpleNamespace

from model_serving.common.management_router import create_model_management_router
from model_serving.common.model_registry import ModelRegistryService


class AiModelsHandlerTest(unittest.TestCase):
    def test_model_projection_never_returns_api_key(self):
        result = ModelRegistryService._safe(SimpleNamespace(
            model_id=7, app_id=1, display_name="embed", model_name="bge",
            category=2, provider="local", api_endpoint=None, api_key="secret",
            status=1, embedding_dimension=1536, model_params={"device": "cpu"},
            descs="test", created_by="a", updated_by="b",
        ))
        self.assertNotIn("api_key", result)
        self.assertEqual(1536, result["embedding_dimension"])

    def test_each_process_gets_category_scoped_management_routes(self):
        router = create_model_management_router(category=2)
        paths = {route.path for route in router.routes}
        self.assertIn("/internal/v1/models", paths)
        self.assertIn("/internal/v1/models/{model_id}", paths)


if __name__ == "__main__":
    unittest.main()
