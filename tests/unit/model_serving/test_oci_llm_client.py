"""OCI LLM 模型家族请求与正文解析回归测试。"""

import unittest
from types import SimpleNamespace

from model_serving.llm.model.oci_client import (
    OCIClient,
    OCILLMConfig,
    extract_oci_text_content,
    resolve_oci_chat_profile,
)


def _client(model_name: str, *, max_tokens: int = 8192) -> OCIClient:
    return OCIClient(OCILLMConfig(
        model_name=model_name,
        provider="oci",
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        api_endpoint="https://example.invalid",
        compartment_id="ocid1.compartment.oc1..test",
        config_file={"region": "us-chicago-1"},
    ))


class OCILLMClientTest(unittest.TestCase):
    def test_gpt5_uses_max_completion_tokens_only(self):
        request = _client("openai.gpt-5.6-sol")._build_chat_request(
            [{"role": "user", "content": "你好"}],
        )

        self.assertEqual(8192, request.max_completion_tokens)
        self.assertIsNone(request.max_tokens)
        self.assertIsNone(request.temperature)
        self.assertIsNone(request.top_p)
        self.assertIsNone(request.top_k)

    def test_gpt5_preserves_per_call_output_limit(self):
        request = _client("openai.gpt-5.6-sol")._build_chat_request(
            [{"role": "user", "content": "你好"}],
            max_tokens=2048,
        )

        self.assertEqual(2048, request.max_completion_tokens)

    def test_grok_keeps_existing_max_tokens_contract(self):
        request = _client("xai.grok-4", max_tokens=24000)._build_chat_request(
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(20000, request.max_tokens)
        self.assertIsNone(request.max_completion_tokens)
        self.assertEqual(0.7, request.temperature)
        self.assertEqual(1.0, request.top_p)
        self.assertEqual(0, request.top_k)

    def test_profile_matches_whole_gpt5_family(self):
        profile = resolve_oci_chat_profile("openai.gpt-5.6-sol")

        self.assertEqual("generic", profile.request_format)
        self.assertEqual("max_completion_tokens", profile.output_token_field)

    def test_extracts_all_text_blocks_from_sdk_and_stream_content(self):
        content = [
            {"type": "TEXT", "text": "第一段"},
            SimpleNamespace(text="第二段"),
            {"type": "IMAGE"},
            {"type": "REASONING", "text": "隐藏推理"},
        ]

        self.assertEqual("第一段第二段", extract_oci_text_content(content))


if __name__ == "__main__":
    unittest.main()
