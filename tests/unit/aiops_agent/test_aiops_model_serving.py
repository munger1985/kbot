"""AIOps 结构化模型客户端测试。"""

from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from aiops_agent.adapters.model_serving import AIOpsStructuredModelClient


class _StructuredOutput(BaseModel):
    schema_version: str = "TEST_OUTPUT.v1"
    answer: str


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self):
        return {
            "id": "provider-request",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"schema_version":"TEST_OUTPUT.v1",'
                            '"answer":"正常"}'
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class _Session:
    def __init__(self):
        self.payload = None

    def post(self, url, *, headers, json, timeout):
        del url, headers, timeout
        self.payload = json
        return _Response()


class AIOpsStructuredModelClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_json_mode_and_validates_schema_locally(self) -> None:
        session = _Session()
        client = AIOpsStructuredModelClient(
            base_url="http://model-serving",
            audience="model-serving",
            caller_service="aiops-agent",
            timeout_seconds=30,
            session=session,
        )
        prompt = "只根据证据回答。"
        with patch(
            "aiops_agent.adapters.model_serving."
            "build_internal_auth_headers",
            return_value={"Authorization": "Bearer test"},
        ):
            result = await client.generate_structured(
                purpose="diagnosis.test",
                output_model=_StructuredOutput,
                model_snapshot={
                    "technical_name": "deepseek-chat",
                    "revision": "test-revision",
                },
                prompt_ref={
                    "content": prompt,
                    "prompt_id": "test",
                    "prompt_version": "1.0.0",
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode()
                    ).hexdigest(),
                },
                input_payload={"question": "数据库慢"},
                deadline=None,
                idempotency_key="test",
            )

        self.assertEqual(
            session.payload["response_format"],
            {"type": "json_object"},
        )
        self.assertNotIn("max_tokens", session.payload)
        system_prompt = session.payload["messages"][0]["content"]
        self.assertIn("JSON Schema", system_prompt)
        self.assertIn('"answer"', system_prompt)
        self.assertEqual(result.output.answer, "正常")
        self.assertEqual(
            result.receipt.schema_id,
            "TEST_OUTPUT.v1",
        )


if __name__ == "__main__":
    unittest.main()
