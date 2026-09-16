"""OCI Responses 适配器的 URL、project、compartment 与错误日志回归测试。"""

from types import SimpleNamespace
from unittest.mock import patch
import unittest

from platform_core.contracts import ResearchRequest
from platform_core.identity import uuid7

from model_serving.llm.responses.errors import GenerativeAdapterError
from model_serving.llm.responses.oci_adapter import (
    OciGrokResponsesAdapter,
    build_oci_responses_url,
    classify_provider_http_error,
    oci_compartment_id,
    oci_generative_ai_project,
    provider_error_fields,
)


HOST = "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com"
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


class OciResponsesUrlTest(unittest.TestCase):
    def test_chat_service_endpoint_gains_openai_responses_path(self):
        self.assertEqual(
            f"{HOST}/openai/v1/responses",
            build_oci_responses_url(HOST),
        )

    def test_openai_compatible_base_is_not_duplicated(self):
        self.assertEqual(
            f"{HOST}/openai/v1/responses",
            build_oci_responses_url(f"{HOST}/openai/v1/"),
        )
        self.assertEqual(
            f"{HOST}/openai/v1/responses",
            build_oci_responses_url(f"{HOST}/openai/v1/responses"),
        )

    def test_legacy_v1_suffix_is_rewritten_to_openai_path(self):
        self.assertEqual(
            f"{HOST}/openai/v1/responses",
            build_oci_responses_url(f"{HOST}/v1"),
        )

    def test_dated_actions_v1_endpoint_appends_responses(self):
        endpoint = f"{HOST}/20231130/actions/v1"
        expected = f"{endpoint}/responses"
        self.assertEqual(expected, build_oci_responses_url(endpoint))
        self.assertEqual(expected, build_oci_responses_url(endpoint + "/"))
        self.assertEqual(expected, build_oci_responses_url(expected))


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


class OciResponsesCompartmentTest(unittest.TestCase):
    def test_missing_compartment_is_a_configuration_error(self):
        with self.assertRaises(GenerativeAdapterError) as raised:
            oci_compartment_id({"project": PROJECT})
        self.assertEqual(503, raised.exception.status_code)
        self.assertEqual("PROVIDER_UNAVAILABLE", raised.exception.code)
        self.assertIn("compartment", raised.exception.message.lower())

    def test_blank_compartment_is_rejected(self):
        with self.assertRaises(GenerativeAdapterError):
            oci_compartment_id({"project": PROJECT, "compartment_id": "  "})


class OciResponsesAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_project_does_not_call_upstream(self):
        adapter = OciGrokResponsesAdapter()
        with patch("model_serving.llm.responses.oci_adapter.requests.post") as post:
            with self.assertRaises(GenerativeAdapterError) as raised:
                await adapter.research(_request(), _material())
        self.assertEqual(503, raised.exception.status_code)
        post.assert_not_called()

    async def test_missing_compartment_does_not_call_upstream(self):
        adapter = OciGrokResponsesAdapter()
        with patch("model_serving.llm.responses.oci_adapter.requests.post") as post:
            with self.assertRaises(GenerativeAdapterError) as raised:
                await adapter.research(
                    _request(),
                    _material(project=PROJECT, compartment_id=None),
                )
        self.assertEqual(503, raised.exception.status_code)
        self.assertIn("compartment", raised.exception.message.lower())
        post.assert_not_called()

    async def test_posts_openai_responses_with_project_header(self):
        adapter = OciGrokResponsesAdapter()
        captured: dict = {}
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "id": "resp-1",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }],
            },
        )

        def fake_post(url, json=None, auth=None, headers=None, timeout=None):
            captured.update(url=url, json=json, headers=headers, auth=auth)
            return response

        with (
            patch.object(adapter, "_signer", return_value=object()),
            patch("model_serving.llm.responses.oci_adapter.requests.post", fake_post),
        ):
            result = await adapter.research(_request(), _material(project=PROJECT))

        self.assertEqual(f"{HOST}/openai/v1/responses", captured["url"])
        self.assertEqual(PROJECT, captured["headers"]["OpenAI-Project"])
        self.assertEqual(COMPARTMENT, captured["headers"]["CompartmentId"])
        self.assertEqual("x_search", captured["json"]["tools"][0]["type"])
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual("ok", result.answer)

    async def test_posts_dated_actions_v1_responses_without_openai_rewrite(self):
        adapter = OciGrokResponsesAdapter()
        captured: dict = {}
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "id": "resp-1",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }],
            },
        )

        def fake_post(url, json=None, auth=None, headers=None, timeout=None):
            captured.update(url=url, json=json, headers=headers, auth=auth)
            return response

        endpoint = f"{HOST}/20231130/actions/v1"
        with (
            patch.object(adapter, "_signer", return_value=object()),
            patch("model_serving.llm.responses.oci_adapter.requests.post", fake_post),
        ):
            await adapter.research(
                _request(),
                _material(project=PROJECT) | {"api_endpoint": endpoint},
            )

        self.assertEqual(f"{endpoint}/responses", captured["url"])
        self.assertEqual(PROJECT, captured["headers"]["OpenAI-Project"])
        self.assertEqual(COMPARTMENT, captured["headers"]["CompartmentId"])


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
                    url=f"{HOST}/20231130/actions/v1/responses",
                )
        rendered = " ".join(str(item) for item in logger.warning.call_args)
        self.assertIn("400", rendered)
        self.assertIn("invalid_request", rendered)
        self.assertIn("/20231130/actions/v1/responses", rendered)
        self.assertNotIn("sk-secret", rendered)


if __name__ == "__main__":
    unittest.main()
