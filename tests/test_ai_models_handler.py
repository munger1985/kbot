import unittest
from types import SimpleNamespace

from model_serving.common.management_router import (
    ModelUpdateRequest,
    create_model_management_router,
)
from model_serving.common.entities.ai_model import AIModelEntity
from model_serving.common.model_registry import ModelRegistryService
from platform_core.identity import uuid7
from platform_core.persistence import UUIDv7Type


class AiModelsHandlerTest(unittest.TestCase):
    def test_model_projection_never_returns_api_key(self):
        result = ModelRegistryService._safe(SimpleNamespace(
            model_id=uuid7(), app_id=1, served_model_name="embed-prod",
            display_name="Embedding", provider_model_name="bge",
            category=2, provider="local", api_endpoint=None, api_key="secret",
            status=1, embedding_dimension=1536, model_params={"device": "cpu"},
            descs="test", created_by="a", updated_by="b",
        ))
        self.assertNotIn("api_key", result)
        self.assertEqual(1536, result["embedding_dimension"])
        self.assertEqual("embed-prod", result["served_model_name"])

    def test_each_process_gets_category_scoped_management_routes(self):
        router = create_model_management_router(category=2)
        paths = {route.path for route in router.routes}
        self.assertIn("/internal/v1/models", paths)
        self.assertIn("/internal/v1/models/{model_id}", paths)

    def test_model_entity_separates_internal_and_serving_identity(self):
        self.assertIsInstance(AIModelEntity.__table__.c.model_id.type, UUIDv7Type)
        self.assertIn("served_model_name", AIModelEntity.__table__.c)
        self.assertIn("provider_model_name", AIModelEntity.__table__.c)
        self.assertNotIn("model_name", AIModelEntity.__table__.c)

    def test_identity_fields_are_not_patchable(self):
        fields = set(ModelUpdateRequest.model_fields)
        self.assertNotIn("served_model_name", fields)
        self.assertNotIn("provider_model_name", fields)
        self.assertNotIn("embedding_dimension", fields)


if __name__ == "__main__":
    unittest.main()
