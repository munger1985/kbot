"""模型测试从已保存配置发起推理，不返回供应商响应或凭据。"""

import asyncio
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from main_api.api.system_models import router as platform_models_router
from model_serving.common.management_router import create_model_management_router
from platform_clients import AIModelConfigClientError
from platform_core.contracts import GeneratedImageArtifact, ImageGenerationResult
from platform_core.dictionary import ModelCategory
from platform_core.identity import uuid7
from platform_core.security import create_service_auth_context


class ModelConnectionProbeTest(TestCase):
    def setUp(self):
        self.model_id = uuid7()
        self.model = {
            "status": "ACTIVE",
            "model_params": {"retrieval_directions": ["IMAGE_TO_IMAGE"]},
        }

    def model_client(self, category, service):
        app = FastAPI()
        registry = SimpleNamespace(get=AsyncMock(return_value=self.model))
        app.state.model_registry = registry
        app.include_router(create_model_management_router(
            category=category, probe_service=lambda _: service,
        ))

        @app.middleware("http")
        async def auth(request, call_next):
            request.state.auth_context = create_service_auth_context(caller_service="main-api-test")
            return await call_next(request)

        return TestClient(app), registry

    def test_six_categories_call_real_inference_adapters(self):
        text = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])
        artifact = GeneratedImageArtifact(mime_type="image/png", content=b"mock-image")
        cases = [
            (ModelCategory.LLM, "chat", text),
            (ModelCategory.TXT_EMBEDDING, "embed_texts", SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 0.0])],
            )),
            (ModelCategory.IMG_EMBEDDING, "represent", {"vectors": [[1.0]], "mask": [True]}),
            (ModelCategory.VLM, "inference", {"choices": [{"message": {"content": "KBOT"}}]}),
            (ModelCategory.OCR, "infer", {"text": "KBOT"}),
            (ModelCategory.IMAGE_GENERATION, "generate_image", ImageGenerationResult(
                status="COMPLETED", artifacts=(artifact,),
            )),
        ]
        for category, method, result in cases:
            with self.subTest(category=category):
                infer = AsyncMock(return_value=result)
                client, registry = self.model_client(category, SimpleNamespace(**{method: infer}))
                with client:
                    response = client.post(f"/internal/v1/models/{self.model_id}:test-connection")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(set(response.json()), {"latency_ms"})
                self.assertGreaterEqual(response.json()["latency_ms"], 0)
                registry.get.assert_awaited_once_with(self.model_id, category=category)
                infer.assert_awaited_once()
                if category == ModelCategory.IMAGE_GENERATION:
                    self.assertEqual(infer.call_args.args[0].count, 1)
                    self.assertEqual(infer.call_args.args[0].model_id, self.model_id)
                elif category == ModelCategory.IMG_EMBEDDING:
                    self.assertEqual(infer.call_args.kwargs["input_kind"], "IMAGE")
                    self.assertTrue(infer.call_args.kwargs["image_base64"].startswith("iVBOR"))
                elif category == ModelCategory.VLM:
                    self.assertIn("image_url", infer.call_args.kwargs["messages"][0]["content"][1])

    def test_provider_failure_never_echoes_credentials_and_draft_never_calls(self):
        infer = AsyncMock(side_effect=RuntimeError("provider response: secret-token"))
        client, _ = self.model_client(ModelCategory.LLM, SimpleNamespace(chat=infer))
        path = f"/internal/v1/models/{self.model_id}:test-connection"
        with client:
            response = client.post(path)
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json()["detail"]["code"], "MODEL_CONNECTION_TEST_FAILED")
            self.assertNotIn("secret-token", response.text)
            infer.side_effect = asyncio.TimeoutError
            response = client.post(path)
            self.assertEqual(response.status_code, 504)
            infer.reset_mock()
            self.model["status"] = "DRAFT"
            response = client.post(path)
            self.assertEqual(response.status_code, 409)
            infer.assert_not_awaited()

    def test_image_failure_payload_is_not_success(self):
        infer = AsyncMock(return_value=ImageGenerationResult(
            status="FAILED", error_message="private provider details",
        ))
        client, _ = self.model_client(
            ModelCategory.IMAGE_GENERATION, SimpleNamespace(generate_image=infer),
        )
        with client:
            response = client.post(f"/internal/v1/models/{self.model_id}:test-connection")
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private provider details", response.text)

    def test_platform_route_requires_model_permission_and_propagates_result(self):
        app = FastAPI()
        app.include_router(platform_models_router)
        access = AsyncMock()
        run = AsyncMock(return_value={"latency_ms": 34})
        app.state.access_control_service = SimpleNamespace(require_platform=access)
        app.state.model_config_clients = (SimpleNamespace(category=1, test_model_connection=run),)
        context = create_service_auth_context(caller_service="main-api-test")

        @app.middleware("http")
        async def auth(request, call_next):
            entry_kind = request.headers.get("test-entry-kind")
            request.state.user_token_claims = SimpleNamespace(entry_kind=entry_kind, user_id="admin")
            request.state.auth_context = context
            return await call_next(request)

        path = f"/api/v1/platform/models/1/{self.model_id}:test-connection"
        with TestClient(app) as client:
            denied = client.post(path, headers={"test-entry-kind": "BUSINESS"})
            self.assertEqual(denied.status_code, 403)
            run.assert_not_awaited()
            response = client.post(path, headers={"test-entry-kind": "PLATFORM"})
            self.assertEqual(response.json(), {"latency_ms": 34})
            run.assert_awaited_once_with(self.model_id, auth_context=context)
            access.assert_awaited_once_with(user_id="admin", permission_code="platform:model_manage")
            run.side_effect = AIModelConfigClientError(
                status_code=504, code="MODEL_CONNECTION_TEST_TIMEOUT", message="测试超时",
            )
            response = client.post(path, headers={"test-entry-kind": "PLATFORM"})
            self.assertEqual(response.status_code, 504)
