"""OCI LLM 模型家族请求与正文解析回归测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from model_serving.common.oci_auth import ModelConfigOciAuth, oci_sdk_service_endpoint

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
    def test_sdk_endpoint_accepts_only_service_origin(self):
        host = "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com"

        self.assertEqual(host, oci_sdk_service_endpoint(host))
        for endpoint in (
            f"{host}/20231130/actions/v1",
            f"{host}/20231130/actions/v1/responses",
            f"{host}?region=iad",
            f"{host}#responses",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(ValueError, "Origin"):
                    oci_sdk_service_endpoint(endpoint)

    def test_sdk_endpoint_rejects_non_http_address(self):
        with self.assertRaisesRegex(ValueError, "HTTP\\(S\\)"):
            oci_sdk_service_endpoint("inference.example.invalid/actions/v1")

    def test_official_iam_auth_signs_httpx_request_body(self):
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        key_content = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("ascii")
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(dict(request.headers))
            return httpx.Response(200, json={"ok": True}, request=request)

        auth = ModelConfigOciAuth({
            "tenancy": "ocid1.tenancy.oc1..test",
            "user": "ocid1.user.oc1..test",
            "fingerprint": "00:11:22:33:44:55:66:77",
            "key_content": key_content,
        })
        with httpx.Client(auth=auth, transport=httpx.MockTransport(handler)) as client:
            response = client.post(
                "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/"
                "openai/v1/responses",
                json={"model": "xai.grok-4.6", "input": "hello"},
            )

        self.assertEqual(200, response.status_code)
        self.assertIn("Signature", captured["authorization"])
        self.assertIn("x-content-sha256", captured)
        self.assertIn("content-length", captured)

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

    def test_gpt5_propagates_json_object_response_format(self):
        request = _client("openai.gpt-5.6-sol")._build_chat_request(
            [{"role": "user", "content": "只输出 JSON"}],
            response_format={"type": "json_object"},
        )

        self.assertEqual("JSON_OBJECT", request.response_format.type)

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


class OCIStartupEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_startup_forwards_configured_origin_to_sdk(self):
        host = "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com"
        client = _client("xai.grok-4.6")
        client.config.api_endpoint = host

        with (
            patch(
                "model_serving.llm.model.oci_client.validated_oci_config",
                return_value={"region": "us-ashburn-1"},
            ),
            patch(
                "model_serving.llm.model.oci_client.oci.generative_ai_inference."
                "GenerativeAiInferenceClient"
            ) as client_factory,
        ):
            await client.startup()

        self.assertEqual(
            host,
            client_factory.call_args.kwargs["service_endpoint"],
        )


if __name__ == "__main__":
    unittest.main()
