import unittest
from decimal import Decimal
from types import SimpleNamespace

from sqlalchemy.dialects import oracle
from sqlalchemy.schema import CreateTable
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from model_serving.common.management_router import (
    ModelUpdateRequest,
    create_model_management_router,
)
from model_serving.common.entities.ai_model import AIModelEntity
from model_serving.common.model_registry import ModelRegistryService
from platform_core.identity import uuid7
from platform_core.persistence import OracleNativeJSON, UUIDv7Type


class AiModelsHandlerTest(unittest.TestCase):
    def test_model_projection_never_returns_api_key(self):
        result = ModelRegistryService._safe(SimpleNamespace(
            model_id=uuid7(),
            display_name="Embedding", provider_model_name="bge",
            category=2, provider="local", api_endpoint=None, api_key="secret",
            status=1, model_params={"embedding_dimension": 1536, "device": "cpu"},
            descs="test", created_by="a", updated_by="b", row_version=1,
        ))
        self.assertNotIn("api_key", result)
        self.assertEqual(1536, result["model_params"]["embedding_dimension"])
        self.assertNotIn("served_model_name", result)
        self.assertEqual("ACTIVE", result["status"])

    def test_each_process_gets_category_scoped_management_routes(self):
        router = create_model_management_router(category=2)
        paths = {route.path for route in router.routes}
        self.assertIn("/internal/v1/models", paths)
        self.assertIn("/internal/v1/models/{model_id}", paths)

    def test_list_route_uses_safe_lifecycle_contract(self):
        class Service:
            async def list(self, *, category):
                return [{
                    "model_id": str(uuid7()),
                    "display_name": "Chat",
                    "provider_model_name": "qwen",
                    "category": category,
                    "provider": "api_qwen",
                    "api_endpoint": "https://example.invalid/v1",
                    "status": "ACTIVE",
                    "model_params": {"max_tokens": 128},
                    "description": None,
                    "row_version": 2,
                }]

        app = FastAPI()
        app.state.model_registry = Service()
        app.include_router(create_model_management_router(category=1))
        response = TestClient(app).get("/internal/v1/models")
        self.assertEqual(200, response.status_code)
        self.assertEqual("ACTIVE", response.json()[0]["status"])
        self.assertNotIn("api_key", response.text)

    def test_model_entity_uses_uuid_without_alias(self):
        self.assertIsInstance(AIModelEntity.__table__.c.model_id.type, UUIDv7Type)
        self.assertNotIn("served_model_name", AIModelEntity.__table__.c)
        self.assertIn("provider_model_name", AIModelEntity.__table__.c)
        self.assertNotIn("model_name", AIModelEntity.__table__.c)

    def test_model_params_uses_oracle_native_json(self):
        column_type = AIModelEntity.__table__.c.model_params.type
        self.assertIsInstance(column_type, OracleNativeJSON)
        ddl = str(
            CreateTable(AIModelEntity.__table__).compile(
                dialect=oracle.dialect()
            )
        )
        self.assertIn("model_params JSON", ddl)
        restored = column_type.result_processor(
            oracle.dialect(),
            None,
        )(
            {
                "ratio": Decimal("0.2"),
                "count": Decimal("2"),
                "nested": [Decimal("1.5")],
            }
        )
        self.assertEqual(
            {"ratio": 0.2, "count": 2, "nested": [1.5]},
            restored,
        )

    def test_identity_fields_are_not_patchable(self):
        fields = set(ModelUpdateRequest.model_fields)
        self.assertNotIn("served_model_name", fields)
        self.assertIn("provider_model_name", fields)
        self.assertNotIn("embedding_dimension", fields)
        self.assertIn("model_params", fields)
        self.assertNotIn("supports_x_search", fields)
        self.assertNotIn("supports_image_generation", fields)
        self.assertNotIn("supports_responses_streaming", fields)

    def test_patch_provider_model_name_rejects_empty_and_preserves_case(self):
        baseline = ModelUpdateRequest(expected_row_version=1)
        self.assertNotIn("provider_model_name", baseline.model_dump(exclude_unset=True))
        update = ModelUpdateRequest(expected_row_version=1, provider_model_name=" Vendor/Model-2 ")
        self.assertEqual("Vendor/Model-2", update.provider_model_name)
        for invalid in (None, "", "   ", "a" * 257):
            with self.subTest(value=invalid), self.assertRaises(ValidationError):
                ModelUpdateRequest(expected_row_version=1, provider_model_name=invalid)

    def test_model_entity_does_not_persist_provider_capability_flags(self):
        columns = AIModelEntity.__table__.c
        self.assertNotIn("supports_image_generation", columns)
        self.assertNotIn("supports_x_search", columns)
        self.assertNotIn("supports_responses_streaming", columns)
        self.assertNotIn("capability_verified_at", columns)


if __name__ == "__main__":
    unittest.main()
