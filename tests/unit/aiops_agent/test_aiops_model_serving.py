"""AIOps 结构化模型客户端测试。"""

from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from aiops_agent.adapters.model_serving import (
    AIOpsModelError,
    AIOpsStructuredModelClient,
)


class _StructuredOutput(BaseModel):
    schema_version: str = "TEST_OUTPUT.v1"
    answer: str


class _CompositeStructuredOutput(BaseModel):
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


class _SequencedResponse(_Response):
    def __init__(self, content: str, request_id: str):
        self._content = content
        self._request_id = request_id

    async def json(self):
        return {
            "id": self._request_id,
            "choices": [{
                "message": {"content": self._content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class _SequencedSession:
    def __init__(self, contents: tuple[str, ...]):
        self._contents = list(contents)
        self.payloads = []

    def post(self, url, *, headers, json, timeout):
        del url, headers, timeout
        self.payloads.append(json)
        index = len(self.payloads)
        return _SequencedResponse(
            self._contents.pop(0),
            f"provider-request-{index}",
        )


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

    async def test_uses_model_name_when_top_level_schema_version_is_absent(
        self,
    ) -> None:
        """组合输出契约没有顶层版本字段时仍须生成可审计回执。"""
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
                purpose="diagnosis.composite-test",
                output_model=_CompositeStructuredOutput,
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

        self.assertEqual(result.output.answer, "正常")
        self.assertEqual(
            result.receipt.schema_id,
            "_CompositeStructuredOutput",
        )

    async def test_repairs_schema_invalid_json_with_validation_feedback(
        self,
    ) -> None:
        session = _SequencedSession((
            '{"schema_version":"TEST_OUTPUT.v1"}',
            '{"schema_version":"TEST_OUTPUT.v1","answer":"已修正"}',
        ))
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
                purpose="diagnosis.repair-test",
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

        self.assertEqual("已修正", result.output.answer)
        self.assertEqual(2, len(session.payloads))
        repair_messages = session.payloads[1]["messages"]
        self.assertEqual("assistant", repair_messages[-2]["role"])
        self.assertIn("未通过既定输出合同", repair_messages[-1]["content"])
        self.assertIn("answer", repair_messages[-1]["content"])
        self.assertEqual(20, result.receipt.prompt_tokens)
        self.assertEqual(10, result.receipt.completion_tokens)
        self.assertEqual(
            "provider-request-2",
            result.receipt.provider_request_id,
        )

    async def test_rejects_output_when_schema_repair_still_invalid(
        self,
    ) -> None:
        session = _SequencedSession(("{}", "{}"))
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
            with self.assertRaises(AIOpsModelError) as raised:
                await client.generate_structured(
                    purpose="diagnosis.invalid-test",
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

        self.assertEqual("MODEL_OUTPUT_INVALID", raised.exception.code)
        self.assertEqual(2, len(session.payloads))


if __name__ == "__main__":
    unittest.main()
