"""Responses 图片能力真实验收测试。"""

import unittest

from model_serving.llm.responses.errors import GenerativeAdapterError
from model_serving.llm.responses.fake_adapter import FakeGenerativeAdapter
from model_serving.llm.responses.service import ResponsesService
from platform_core.contracts import (
    AuthContext,
    GeneratedImageArtifact,
    ImageGenerationResult,
    PrincipalKind,
)
from platform_core.identity import uuid7


class _Registry:
    def __init__(self):
        self.material_request = None
        self.verification = None

    async def get_capability_verification_material(
        self, model_id, *, expected_row_version, category,
    ):
        self.material_request = (model_id, expected_row_version, category)
        return {
            "provider": "oci",
            "api_endpoint": "https://inference.example.test",
            "provider_model_name": "xai.grok-test",
            "served_model_name": "grok-test",
            "model_params": {},
            "row_version": expected_row_version,
        }

    async def record_capability_verification(self, model_id, **values):
        self.verification = (model_id, values)
        return {
            "model_id": str(model_id),
            "supports_image_generation": values["supports_image_generation"],
        }


def _auth_context() -> AuthContext:
    return AuthContext(
        principal_kind=PrincipalKind.PORTAL,
        client_id="main-api",
        request_id="capability-request",
        trace_id="capability-trace",
        api_key_id="portal-key",
        domain_id="41",
        asserted_user_id="operator",
    )


class ResponsesCapabilityVerificationTest(unittest.IsolatedAsyncioTestCase):
    async def test_canary_records_capability_only_after_real_image(self):
        model_id = uuid7()
        registry = _Registry()
        adapter = FakeGenerativeAdapter(image_result=ImageGenerationResult(
            status="COMPLETED",
            artifacts=(GeneratedImageArtifact(
                mime_type="image/png",
                content=b"\x89PNG\r\n\x1a\ncanary",
            ),),
        ))
        service = ResponsesService(registry=registry, adapter=adapter)

        result = await service.verify_image_generation(
            model_id,
            expected_row_version=7,
            actor_id="operator",
            trace_id="capability-trace",
            auth_context=_auth_context(),
        )

        self.assertTrue(result["supports_image_generation"])
        self.assertEqual(7, registry.material_request[1])
        self.assertEqual("image", adapter.calls[0][0])
        self.assertTrue(registry.verification[1]["supports_image_generation"])
        self.assertEqual(7, registry.verification[1]["expected_row_version"])

    async def test_canary_without_artifact_does_not_record_capability(self):
        registry = _Registry()
        adapter = FakeGenerativeAdapter(image_result=ImageGenerationResult(
            status="FAILED",
            error_code="PROVIDER_UNAVAILABLE",
            error_message="上游没有返回图片",
        ))
        service = ResponsesService(registry=registry, adapter=adapter)

        with self.assertRaises(GenerativeAdapterError) as raised:
            await service.verify_image_generation(
                uuid7(),
                expected_row_version=3,
                actor_id="operator",
                trace_id="capability-trace",
                auth_context=_auth_context(),
            )

        self.assertEqual("PROVIDER_UNAVAILABLE", raised.exception.code)
        self.assertIsNone(registry.verification)


if __name__ == "__main__":
    unittest.main()
