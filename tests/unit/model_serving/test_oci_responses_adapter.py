"""OCI Responses 标准 SDK、project、图片结果与错误日志回归测试。"""

import json
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import httpx
import openai

from platform_core.contracts import ImageGenerationRequest, ImageGenerationResult, ResearchRequest
from platform_core.identity import uuid7

from model_serving.llm.responses.errors import GenerativeAdapterError
from model_serving.llm.responses.oci_adapter import (
    OciGrokResponsesAdapter,
    classify_provider_http_error,
    oci_generative_ai_project,
    parse_image_response,
    provider_error_fields,
)


HOST = "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com"
DATED_API_BASE = f"{HOST}/20231130/actions/v1"
PROJECT = "ocid1.generativeaiproject.oc1..test"
COMPARTMENT = "ocid1.compartment.oc1..test"


def _request() -> ResearchRequest:
    return ResearchRequest(
        model_id=uuid7(),
        input="过去3天 Oracle 官方 twitter",
        trace_id="trace-x-search",
    )


def _material(**params):
    model_params = {"compartment_id": COMPARTMENT}
    model_params.update(params)
    return {
        "provider": "OCI",
        "api_endpoint": HOST,
        "provider_model_name": "xai.grok-4.6",
        "served_model_name": "grok-4.6",
        "model_params": model_params,
    }


def _openai_client(handler) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{HOST}/openai/v1",
        api_key="not-used",
        project=PROJECT,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class OciResponsesProjectTest(unittest.TestCase):
    def test_missing_project_is_a_configuration_error(self):
        with self.assertRaises(GenerativeAdapterError) as raised:
            oci_generative_ai_project({"compartment_id": COMPARTMENT})
        self.assertEqual(503, raised.exception.status_code)
        self.assertEqual("PROVIDER_UNAVAILABLE", raised.exception.code)
        self.assertIn("project", raised.exception.message.lower())

    def test_blank_project_is_rejected(self):
        with self.assertRaises(GenerativeAdapterError):
            oci_generative_ai_project({"project": "  "})


class OciResponsesAdapterTest(unittest.IsolatedAsyncioTestCase):
    def test_client_uses_openai_sdk_base_and_oci_iam_auth(self):
        adapter = OciGrokResponsesAdapter(timeout_seconds=45)
        config = {"tenancy": "ocid1.tenancy.oc1..test"}
        auth = object()
        http_client = object()
        sdk_client = object()

        with (
            patch(
                "model_serving.llm.responses.oci_adapter.validated_oci_config",
                return_value=config,
            ),
            patch(
                "model_serving.llm.responses.oci_adapter.ModelConfigOciAuth",
                return_value=auth,
            ) as auth_factory,
            patch(
                "model_serving.llm.responses.oci_adapter.httpx.Client",
                return_value=http_client,
            ) as http_client_factory,
            patch(
                "model_serving.llm.responses.oci_adapter.openai.OpenAI",
                return_value=sdk_client,
            ) as sdk_factory,
        ):
            result = adapter._client(_material(project=PROJECT, config_file=config))

        self.assertIs(sdk_client, result)
        auth_factory.assert_called_once_with(config)
        self.assertIs(auth, http_client_factory.call_args.kwargs["auth"])
        self.assertEqual(f"{HOST}/openai/v1", sdk_factory.call_args.kwargs["base_url"])
        self.assertEqual(PROJECT, sdk_factory.call_args.kwargs["project"])
        self.assertEqual(0, sdk_factory.call_args.kwargs["max_retries"])

    def test_client_uses_configured_api_base_and_bearer_key(self):
        adapter = OciGrokResponsesAdapter(timeout_seconds=45)
        http_client = object()
        sdk_client = object()
        material = _material() | {
            "api_endpoint": DATED_API_BASE,
            "api_key": "current-codex-key",
        }

        with (
            patch(
                "model_serving.llm.responses.oci_adapter.ModelConfigOciAuth",
            ) as auth_factory,
            patch(
                "model_serving.llm.responses.oci_adapter.httpx.Client",
                return_value=http_client,
            ) as http_client_factory,
            patch(
                "model_serving.llm.responses.oci_adapter.openai.OpenAI",
                return_value=sdk_client,
            ) as sdk_factory,
        ):
            result = adapter._client(material)

        self.assertIs(sdk_client, result)
        auth_factory.assert_not_called()
        self.assertIsNone(http_client_factory.call_args.kwargs["auth"])
        self.assertEqual(DATED_API_BASE, sdk_factory.call_args.kwargs["base_url"])
        self.assertEqual("current-codex-key", sdk_factory.call_args.kwargs["api_key"])
        self.assertIsNone(sdk_factory.call_args.kwargs["project"])

    async def test_api_key_mode_requires_complete_api_base_path(self):
        adapter = OciGrokResponsesAdapter()
        with self.assertRaises(GenerativeAdapterError) as raised:
            await adapter.research(
                _request(), _material() | {"api_key": "current-codex-key"},
            )
        self.assertEqual(503, raised.exception.status_code)
        self.assertIn("Endpoint", raised.exception.message)

    async def test_missing_project_does_not_call_upstream(self):
        adapter = OciGrokResponsesAdapter()
        with patch("model_serving.llm.responses.oci_adapter.openai.OpenAI") as client:
            with self.assertRaises(GenerativeAdapterError) as raised:
                await adapter.research(_request(), _material())
        self.assertEqual(503, raised.exception.status_code)
        client.assert_not_called()

    async def test_endpoint_with_api_path_is_rejected(self):
        adapter = OciGrokResponsesAdapter()
        with self.assertRaises(GenerativeAdapterError) as raised:
            await adapter.research(
                _request(),
                _material(project=PROJECT) | {
                    "api_endpoint": f"{HOST}/20231130/actions/v1",
                },
            )
        self.assertEqual(503, raised.exception.status_code)
        self.assertIn("Endpoint", raised.exception.message)

    async def test_posts_openai_responses_with_project_header(self):
        adapter = OciGrokResponsesAdapter()
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(
                url=str(request.url),
                json=json.loads(request.content),
                headers=dict(request.headers),
            )
            return httpx.Response(200, json={
                "id": "resp-1",
                "object": "response",
                "created_at": 1,
                "model": "xai.grok-4.6",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "msg-1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok"}],
                }],
            }, request=request)

        with patch.object(adapter, "_client", return_value=_openai_client(handler)):
            result = await adapter.research(_request(), _material(project=PROJECT))

        self.assertEqual(f"{HOST}/openai/v1/responses", captured["url"])
        self.assertEqual(PROJECT, captured["headers"]["openai-project"])
        self.assertNotIn("compartmentid", captured["headers"])
        self.assertEqual("x_search", captured["json"]["tools"][0]["type"])
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual("ok", result.answer)


class OciResponsesErrorLogTest(unittest.TestCase):
    def test_http_400_is_mapped_to_provider_unavailable(self):
        error = classify_provider_http_error(400, {"error": {"code": "invalid_request"}})
        self.assertEqual(502, error.status_code)
        self.assertEqual("PROVIDER_UNAVAILABLE", error.code)

    def test_error_fields_do_not_copy_secret_keys(self):
        code, message = provider_error_fields({
            "error": {
                "code": "invalid_request",
                "message": "project is required",
                "api_key": "sk-secret",
            }
        })
        self.assertEqual("invalid_request", code)
        self.assertEqual("project is required", message)
        self.assertNotIn("sk-secret", code or "")
        self.assertNotIn("sk-secret", message or "")

    def test_failed_http_log_omits_secret_payload_fields(self):
        response = SimpleNamespace(
            status_code=400,
            text="ignored",
            json=lambda: {
                "error": {
                    "code": "invalid_request",
                    "message": "project is required",
                    "api_key": "sk-secret",
                }
            },
        )
        with patch("model_serving.llm.responses.oci_adapter.logger") as logger:
            with self.assertRaises(GenerativeAdapterError):
                OciGrokResponsesAdapter._parse_http(
                    response,
                    url=f"{HOST}/openai/v1/responses",
                )
        rendered = " ".join(str(item) for item in logger.warning.call_args)
        self.assertIn("400", rendered)
        self.assertIn("invalid_request", rendered)
        self.assertIn("/openai/v1/responses", rendered)
        self.assertNotIn("sk-secret", rendered)


JPEG = b"\xff\xd8\xff\xdb" + b"\x00" * 32


class _HttpResponse:
    def __init__(self, status_code, content, headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def json(self):
        import json
        return json.loads(self.content.decode("utf-8"))

    @property
    def text(self):
        return self.content.decode("utf-8")


def _image_request() -> ImageGenerationRequest:
    return ImageGenerationRequest(
        model_id=uuid7(),
        prompt="a red cube",
        trace_id="trace-image",
    )


class OciResponsesImageBodyTest(unittest.TestCase):
    def test_jpeg_http_body_is_wrapped_as_image_payload(self):
        payload = OciGrokResponsesAdapter._parse_http(_HttpResponse(200, JPEG))
        result = parse_image_response(payload)
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(1, len(result.artifacts))
        self.assertEqual("image/jpeg", result.artifacts[0].mime_type)
        self.assertEqual(JPEG, result.artifacts[0].content)

    def test_json_image_generation_call_still_parses(self):
        import base64
        import json
        body = json.dumps({
            "id": "resp-img-1",
            "status": "completed",
            "output": [{
                "type": "image_generation_call",
                "id": "ig_1",
                "result": base64.b64encode(JPEG).decode("ascii"),
            }],
        }).encode("utf-8")
        payload = OciGrokResponsesAdapter._parse_http(_HttpResponse(200, body))
        result = parse_image_response(payload)
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(JPEG, result.artifacts[0].content)
        self.assertEqual("resp-img-1", result.provider_request_id)

    def test_non_utf8_error_body_does_not_raise_unicode_error(self):
        with self.assertRaises(GenerativeAdapterError) as raised:
            OciGrokResponsesAdapter._parse_http(_HttpResponse(400, b"\x80\x81not-json"))
        self.assertEqual(502, raised.exception.status_code)
        self.assertEqual("PROVIDER_UNAVAILABLE", raised.exception.code)

    def test_parsed_jpeg_result_can_dump_as_json(self):
        import json
        payload = OciGrokResponsesAdapter._parse_http(_HttpResponse(200, JPEG))
        result = parse_image_response(payload)
        encoded = json.dumps(result.model_dump(mode="json"))
        restored = ImageGenerationResult.model_validate(json.loads(encoded))
        self.assertEqual(JPEG, restored.artifacts[0].content)


class OciResponsesImageAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_generate_image_accepts_raw_jpeg_body(self):
        adapter = OciGrokResponsesAdapter()
        captured: dict = {}

        def images_handler(request: httpx.Request) -> httpx.Response:
            captured.setdefault("urls", []).append(str(request.url))
            return httpx.Response(
                404,
                json={"error": {"code": "not_found", "message": "no images api"}},
                request=request,
            )

        def responses_handler(request: httpx.Request) -> httpx.Response:
            captured.setdefault("urls", []).append(str(request.url))
            captured.update(url=str(request.url), json=json.loads(request.content))
            return httpx.Response(
                200, content=JPEG, headers={"Content-Type": "image/jpeg"}, request=request,
            )

        with patch.object(
            adapter,
            "_client",
            side_effect=[_openai_client(images_handler), _openai_client(responses_handler)],
        ):
            result = await adapter.generate_image(
                _image_request(),
                _material(project=PROJECT),
            )

        self.assertIn(f"{HOST}/openai/v1/images/generations", captured["urls"])
        self.assertEqual(f"{HOST}/openai/v1/responses", captured["url"])
        self.assertEqual("image_generation", captured["json"]["tools"][0]["type"])
        self.assertNotIn("aspect_ratio", captured["json"]["tools"][0])
        self.assertNotIn("instructions", captured["json"])
        self.assertIn("1:1", captured["json"]["input"])
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(JPEG, result.artifacts[0].content)
        self.assertEqual("image/jpeg", result.artifacts[0].mime_type)


class OciResponsesImageParseVariantsTest(unittest.TestCase):
    def test_http_200_error_body_is_not_treated_as_success(self):
        import json
        body = json.dumps({
            "error": {
                "code": "invalid_request",
                "message": "Please pass in correct format of request.",
            }
        }).encode("utf-8")
        with patch("model_serving.llm.responses.oci_adapter.logger") as logger:
            with self.assertRaises(GenerativeAdapterError) as raised:
                OciGrokResponsesAdapter._parse_http(_HttpResponse(200, body))
        self.assertEqual("PROVIDER_UNAVAILABLE", raised.exception.code)
        rendered = " ".join(str(item) for item in logger.warning.call_args)
        self.assertIn("invalid_request", rendered)
        self.assertIn("Please pass in correct format of request.", rendered)

    def test_whitespace_prefixed_jpeg_still_parses(self):
        payload = OciGrokResponsesAdapter._parse_http(_HttpResponse(200, b"\n" + JPEG))
        result = parse_image_response(payload)
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(JPEG, result.artifacts[0].content)

    def test_nested_b64_result_object_parses(self):
        import base64
        import json
        body = json.dumps({
            "status": "completed",
            "data": {
                "output": {
                    "type": "image_generation_call",
                    "id": "ig_nested",
                    "result": {"b64_json": base64.b64encode(JPEG).decode("ascii")},
                }
            },
        }).encode("utf-8")
        payload = OciGrokResponsesAdapter._parse_http(_HttpResponse(200, body))
        result = parse_image_response(payload)
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(JPEG, result.artifacts[0].content)
        self.assertEqual("ig_nested", result.artifacts[0].provider_artifact_id)

    def test_missing_image_logs_structure_without_payload_body(self):
        payload = {
            "id": "resp-no-image",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        }
        with patch("model_serving.llm.responses.oci_adapter.logger") as logger:
            result = parse_image_response(payload)
        self.assertEqual("FAILED", result.status)
        self.assertEqual("PROVIDER_UNAVAILABLE", result.error_code)
        self.assertEqual("ok", result.error_message)
        summary = logger.warning.call_args.args[1]
        self.assertEqual(["message"], summary["item_types"])
        self.assertEqual("list:1", summary["items"][0]["fields"]["content"])
        self.assertEqual("ok", summary["output_excerpt"])

    def test_message_data_uri_is_used_when_image_call_result_is_empty(self):
        import base64
        payload = {
            "id": "resp-oci-empty-result",
            "status": "completed",
            "truncation": "disabled",
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "image_generation_call",
                    "id": "ig_1",
                    "status": "completed",
                    "prompt": "a red cube",
                    "result": None,
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_image",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode("ascii"),
                        },
                    }],
                },
            ],
        }
        result = parse_image_response(payload)
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(JPEG, result.artifacts[0].content)

    def test_markdown_data_uri_in_message_text_still_parses(self):
        import base64
        encoded = base64.b64encode(JPEG).decode("ascii")
        payload = {
            "status": "completed",
            "output": [
                {"type": "image_generation_call", "id": "ig_md", "result": None},
                {
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": f"done ![image](data:image/jpeg;base64,{encoded})",
                    }],
                },
            ],
        }
        result = parse_image_response(payload)
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(JPEG, result.artifacts[0].content)


class OciResponsesImageUrlDownloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_generate_image_downloads_https_result_url(self):
        import json
        adapter = OciGrokResponsesAdapter()
        body = json.dumps({
            "id": "resp-url",
            "status": "completed",
            "output": [
                {"type": "reasoning"},
                {
                    "type": "image_generation_call",
                    "id": "ig_url",
                    "status": "completed",
                    "result": None,
                },
                {
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "![image](https://cdn.example.test/generated.jpg)",
                    }],
                },
            ],
        }).encode("utf-8")

        def images_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": {"code": "not_found"}}, request=request)

        def responses_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, request=request)

        def fake_get(url, timeout=None):
            self.assertEqual("https://cdn.example.test/generated.jpg", url)
            return _HttpResponse(200, JPEG, headers={"Content-Type": "image/jpeg"})

        with (
            patch.object(
                adapter,
                "_client",
                side_effect=[_openai_client(images_handler), _openai_client(responses_handler)],
            ),
            patch("model_serving.llm.responses.oci_adapter.requests.get", fake_get),
        ):
            result = await adapter.generate_image(
                _image_request(),
                _material(project=PROJECT),
            )

        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(JPEG, result.artifacts[0].content)


class OciImagesApiAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_generate_image_prefers_images_api_with_aspect_ratio(self):
        import base64
        import json

        adapter = OciGrokResponsesAdapter()
        captured: list[dict] = []
        body = json.dumps({
            "created": 1,
            "data": [{"b64_json": base64.b64encode(JPEG).decode("ascii")}],
        }).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append({"url": str(request.url), "json": json.loads(request.content)})
            return httpx.Response(200, content=body, request=request)

        with patch.object(adapter, "_client", return_value=_openai_client(handler)):
            result = await adapter.generate_image(
                _image_request(),
                _material(project=PROJECT),
            )

        self.assertEqual(1, len(captured))
        self.assertEqual(f"{HOST}/openai/v1/images/generations", captured[0]["url"])
        self.assertEqual("1:1", captured[0]["json"]["aspect_ratio"])
        self.assertEqual("a red cube", captured[0]["json"]["prompt"])
        self.assertEqual("b64_json", captured[0]["json"]["response_format"])
        self.assertNotIn("tools", captured[0]["json"])
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(JPEG, result.artifacts[0].content)


class OciFailedImageToolTest(unittest.TestCase):
    def test_failed_image_tool_logs_excerpt_without_prompt_or_url(self):
        payload = {
            "id": "resp-failed-tool",
            "status": "completed",
            "output": [
                {
                    "type": "image_generation_call",
                    "id": "ig_failed",
                    "status": "failed",
                    "prompt": "secret-prompt-xyz",
                },
                {
                    "type": "message",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "Imagine backend is currently unavailable. https://cdn.example.test/secret.jpg",
                    }],
                },
            ],
        }
        with patch("model_serving.llm.responses.oci_adapter.logger") as logger:
            result = parse_image_response(payload)
        self.assertEqual("FAILED", result.status)
        self.assertEqual("PROVIDER_UNAVAILABLE", result.error_code)
        self.assertIn("Imagine backend is currently unavailable", result.error_message or "")
        self.assertNotIn("https://", result.error_message or "")
        self.assertNotIn("secret-prompt-xyz", result.error_message or "")
        rendered = " ".join(str(item) for item in logger.warning.call_args)
        self.assertIn("工具执行失败", rendered)
        self.assertNotIn("secret-prompt-xyz", rendered)
        self.assertNotIn("cdn.example.test", rendered)
        summary = logger.warning.call_args.args[1]
        self.assertEqual(["failed"], summary["image_call_statuses"])
        self.assertIn("[url]", summary["output_excerpt"])

    def test_failed_image_tool_with_policy_text_is_rejected(self):
        payload = {
            "status": "completed",
            "output": [
                {"type": "image_generation_call", "status": "failed", "prompt": "secret prompt"},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "This request violates the content policy."}],
                },
            ],
        }
        result = parse_image_response(payload)
        self.assertEqual("REJECTED", result.status)
        self.assertEqual("CONTENT_REJECTED", result.error_code)
        self.assertIn("content policy", (result.error_message or "").lower())
        self.assertNotIn("secret prompt", result.error_message or "")


if __name__ == "__main__":
    unittest.main()
