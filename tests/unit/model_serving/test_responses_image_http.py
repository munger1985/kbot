"""文生图内部 HTTP 响应必须能序列化 JPEG 字节。"""

from unittest.mock import AsyncMock
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_serving.llm.responses.router import create_responses_router
from platform_core.contracts import (
    INTERNAL_API_V1,
    GeneratedImageArtifact,
    ImageGenerationResult,
)
from platform_core.identity import uuid7


JPEG = b"\xff\xd8\xff\xdb" + b"\x00" * 32


class ResponsesImageHttpTest(unittest.TestCase):
    def test_completed_jpeg_result_returns_json_instead_of_500(self):
        app = FastAPI()
        app.include_router(create_responses_router())
        app.state.responses_service = AsyncMock()
        app.state.responses_service.generate_image.return_value = ImageGenerationResult(
            status="COMPLETED",
            artifacts=(GeneratedImageArtifact(mime_type="image/jpeg", content=JPEG),),
            provider_request_id="resp-img-1",
        )
        response = TestClient(app).post(
            f"{INTERNAL_API_V1}/responses/image-generations",
            json={
                "model_id": str(uuid7()),
                "prompt": "a red cube",
                "trace_id": "trace-image",
            },
        )
        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("COMPLETED", payload["status"])
        restored = ImageGenerationResult.model_validate(payload)
        self.assertEqual(JPEG, restored.artifacts[0].content)
        self.assertEqual("image/jpeg", restored.artifacts[0].mime_type)


if __name__ == "__main__":
    unittest.main()
