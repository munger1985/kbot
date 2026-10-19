"""Model Serving S4 生命周期与一致性测试。"""

from decimal import Decimal
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from model_serving.common.model_registry import (
    ModelDefinitionNotFound,
    ModelRegistryConflict,
    ModelRegistryService,
)
from model_serving.common.provider_catalog import (
    list_provider_options,
    merge_model_params_preserving_secrets,
    validate_provider_config,
)
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.dictionary import ModelCategory
from platform_core.identity import uuid7


class _Repository:
    def __init__(self, rows):
        self.rows = rows

    async def get_by_id(self, model_id, *, lock=False):
        return self.rows[model_id]

    async def list_by_scope(self, *, category=None):
        return [
            row for row in self.rows.values()
            if category is None or int(row.category) == category
        ]

    async def add(self, row):
        row.row_version = 1
        self.rows[row.model_id] = row
        return row

    async def delete(self, row):
        self.rows.pop(row.model_id)


class _Uow:
    def __init__(self, repository):
        self.models = repository
        self._before = {}

    async def __aenter__(self):
        self._before = {
            model_id: int(row.row_version)
            for model_id, row in self.models.rows.items()
        }
        return self

    async def flush(self):
        for model_id, row in self.models.rows.items():
            if model_id in self._before:
                row.row_version = self._before[model_id] + 1

    async def commit(self):
        return None

    async def __aexit__(self, exc_type, exc, traceback):
        return None


def _auth_context():
    return AuthContext(
        principal_kind=PrincipalKind.SERVICE,
        client_id="model-test",
        calling_service="kbot-model-llm",
        request_id="request",
        trace_id="trace",
    )


def _portal_auth_context():
    return AuthContext(
        principal_kind=PrincipalKind.PORTAL,
        client_id="portal",
        request_id="request",
        trace_id="trace",
        api_key_id="portal-key",
        domain_id="10",
        asserted_user_id="operator",
    )


def _model(*, status=1, category=1, model_params=None):
    return SimpleNamespace(
        model_id=uuid7(),
        served_model_name="chat-prod",
        display_name="生产对话模型",
        provider_model_name="provider-chat",
        category=category,
        provider="api_qwen",
        api_endpoint="https://example.invalid/v1",
        api_key="top-secret",
        status=status,
        model_params=model_params or {"max_tokens": 4096},
        descs=None,
        created_by="tester",
        updated_by="tester",
        row_version=1,
    )


class ModelRegistryS4Test(unittest.IsolatedAsyncioTestCase):
    async def test_create_persists_requested_active_status(self):
        repository = _Repository({})
        service = ModelRegistryService(
            uow_factory=lambda: _Uow(repository),
        )

        result = await service.create(
            {
                "served_model_name": "chat-active",
                "display_name": "已启用对话模型",
                "provider_model_name": "provider-chat",
                "category": ModelCategory.LLM.value,
                "provider": "api_qwen",
                "api_endpoint": "https://models.example.test/v1",
                "api_key": "write-only-secret",
                "status": "ACTIVE",
                "model_params": {"max_tokens": 4096},
            },
            actor_id="operator",
        )

        self.assertEqual("ACTIVE", result["status"])
        persisted = next(iter(repository.rows.values()))
        self.assertEqual(1, persisted.status)

    def test_secret_model_params_survive_safe_patch(self):
        current = {
            "max_tokens": 4096,
            "config_file": {"key_content": "private-key"},
        }

        merged = merge_model_params_preserving_secrets(
            category=1,
            provider="oci",
            current=current,
            updated={"max_tokens": 8192},
        )

        self.assertEqual(8192, merged["max_tokens"])
        self.assertEqual(current["config_file"], merged["config_file"])

    def test_explicit_secret_model_param_rotates_value(self):
        merged = merge_model_params_preserving_secrets(
            category=1,
            provider="oci",
            current={"config_file": {"key_content": "old-key"}},
            updated={"config_file": {"key_content": "new-key"}},
        )

        self.assertEqual(
            {"key_content": "new-key"}, merged["config_file"],
        )

    def test_manual_oci_auth_replaces_legacy_config_file(self):
        manual = {"key_content": "manual-key"}
        merged = merge_model_params_preserving_secrets(
            category=ModelCategory.IMAGE_GENERATION.value,
            provider="api_oci_image",
            current={"config_file": {"key_content": "old-key"}},
            updated={"oci_auth": manual},
        )

        self.assertEqual(manual, merged["oci_auth"])
        self.assertNotIn("config_file", merged)

    def test_safe_projects_decimal_status_and_category(self):
        row = _model(status=Decimal("1"), category=Decimal("2"))
        payload = ModelRegistryService._safe(row)
        self.assertEqual(2, payload["category"])
        self.assertEqual("ACTIVE", payload["status"])
        self.assertNotIn("api_key", payload)

    async def test_runtime_reload_failure_is_published_for_request_actor(self):
        row = _model()
        published = []

        async def fail_reload(event):
            del event
            raise RuntimeError("reload failed")

        class _Publisher:
            async def publish_reload_failed(self, **values):
                published.append(values)

        service = ModelRegistryService(
            uow_factory=lambda: _Uow(_Repository({row.model_id: row})),
            on_model_changed=fail_reload,
            notification_publisher=_Publisher(),
        )
        with self.assertRaisesRegex(RuntimeError, "reload failed"):
            await service.update(
                row.model_id, {"display_name": "触发运行时刷新"},
                expected_row_version=1, actor_id="operator",
                auth_context=_portal_auth_context(),
            )
        self.assertEqual(1, len(published))
        self.assertEqual("RUNTIMEERROR", published[0]["error_code"])
        self.assertEqual("10", published[0]["auth_context"].domain_id)

    async def test_update_checks_version_invalidates_and_hides_secret(self):
        row = _model(model_params={
            "max_tokens": 4096,
            "config_file": {"key_content": "private-key"},
        })
        events = []
        async def capture(event):
            events.append(event)
        repository = _Repository({row.model_id: row})
        service = ModelRegistryService(
            uow_factory=lambda: _Uow(repository),
            on_model_changed=capture,
        )

        result = await service.update(
            row.model_id,
            {"display_name": "新名称", "api_key": "new-secret"},
            expected_row_version=1,
            actor_id="operator",
        )

        self.assertEqual(2, result["row_version"])
        self.assertNotIn("api_key", result)
        self.assertNotIn("config_file", result["model_params"])
        self.assertNotIn("new-secret", repr(events))
        self.assertNotIn("private-key", repr(events))
        self.assertEqual("chat-prod", events[0]["served_model_name"])
        with self.assertRaisesRegex(ModelRegistryConflict, "版本已变化"):
            await service.update(
                row.model_id, {"display_name": "过期更新"},
                expected_row_version=1, actor_id="operator",
            )

    async def test_delete_fails_closed_when_referenced_or_service_unavailable(self):
        row = _model(status=2)
        repository = _Repository({row.model_id: row})

        async def referenced(model_id, auth_context):
            return [{
                "service": "agent-runtime",
                "resource_type": "agent",
                "resource_id": "agent-1",
                "usage": "router_llm",
            }]

        service = ModelRegistryService(
            uow_factory=lambda: _Uow(repository),
            reference_resolvers={"agent-runtime": referenced},
        )
        with self.assertRaisesRegex(ModelRegistryConflict, "仍被") as raised:
            await service.delete(
                row.model_id, expected_row_version=1,
                auth_context=_auth_context(),
            )
        self.assertEqual(
            "agent-1", raised.exception.details["references"][0]["resource_id"]
        )

        async def unavailable(model_id, auth_context):
            raise TimeoutError

        service = ModelRegistryService(
            uow_factory=lambda: _Uow(repository),
            reference_resolvers={"knowledge-core": unavailable},
        )
        with self.assertRaisesRegex(ModelRegistryConflict, "不可用"):
            await service.delete(
                row.model_id, expected_row_version=1,
                auth_context=_auth_context(),
            )

    async def test_archive_unloads_then_unreferenced_delete_succeeds(self):
        row = _model()
        events = []
        async def capture(event):
            events.append(event)
        repository = _Repository({row.model_id: row})
        service = ModelRegistryService(
            uow_factory=lambda: _Uow(repository),
            on_model_changed=capture,
            reference_resolvers={"agent-runtime": lambda *_: _empty()},
            is_model_loaded=lambda _name: False,
        )
        archived, summary = await service.archive(
            row.model_id, expected_row_version=1, actor_id="operator",
            auth_context=_auth_context(),
        )
        self.assertEqual("ARCHIVED", archived["status"])
        self.assertEqual((), summary.references)
        await service.delete(
            row.model_id,
            expected_row_version=archived["row_version"],
            auth_context=_auth_context(),
        )
        self.assertNotIn(row.model_id, repository.rows)
        self.assertEqual(2, len(events))

    async def test_archive_continues_when_reference_service_is_unavailable(self):
        row = _model()
        repository = _Repository({row.model_id: row})

        async def unavailable(model_id, auth_context):
            del model_id, auth_context
            raise TimeoutError

        service = ModelRegistryService(
            uow_factory=lambda: _Uow(repository),
            reference_resolvers={"data-query": unavailable},
        )

        archived, summary = await service.archive(
            row.model_id,
            expected_row_version=1,
            actor_id="operator",
            auth_context=_auth_context(),
        )

        self.assertEqual("ARCHIVED", archived["status"])
        self.assertEqual(("data-query",), summary.unavailable_services)

    async def test_embedding_dimension_cannot_change_in_place(self):
        row = _model(
            category=2,
            model_params={"embedding_dimension": 1536},
        )
        row.provider = "api_qwen"
        repository = _Repository({row.model_id: row})
        service = ModelRegistryService(
            uow_factory=lambda: _Uow(repository),
        )
        with self.assertRaisesRegex(ModelRegistryConflict, "不可原地修改"):
            await service.update(
                row.model_id,
                {"model_params": {"embedding_dimension": 2048}},
                expected_row_version=1,
                actor_id="operator",
            )

    async def test_connection_material_requires_active_model(self):
        row = _model()
        repository = _Repository({row.model_id: row})
        service = ModelRegistryService(uow_factory=lambda: _Uow(repository))

        material = await service.get_connection_material(row.model_id)
        self.assertEqual("provider-chat", material["provider_model_name"])

        row.status = 0
        with self.assertRaisesRegex(ModelRegistryConflict, "未启用"):
            await service.get_connection_material(row.model_id)

    async def test_x_search_gate_uses_active_grok_identity(self):
        row = _model()
        row.provider_model_name = "xai.grok-4.6"
        repository = _Repository({row.model_id: row})
        service = ModelRegistryService(uow_factory=lambda: _Uow(repository))

        result = await service.require_grok_model(row.model_id)
        self.assertEqual(str(row.model_id), result["model_id"])

        row.provider_model_name = "xai.command-r-plus"
        with self.assertRaisesRegex(ModelRegistryConflict, "Grok"):
            await service.require_grok_model(row.model_id)

        row.provider_model_name = "grok-4"
        row.status = 0
        with self.assertRaisesRegex(ModelRegistryConflict, "未启用"):
            await service.require_grok_model(row.model_id)

        row.status = 1
        row.category = ModelCategory.IMAGE_GENERATION.value
        with self.assertRaises(ModelDefinitionNotFound):
            await service.require_grok_model(row.model_id)


async def _empty():
    return []


class ProviderCatalogS4Test(unittest.TestCase):
    def test_image_generation_has_dedicated_oci_provider_contract(self):
        options = list_provider_options(
            category=ModelCategory.IMAGE_GENERATION.value,
        )
        self.assertEqual(["api_oci_image"], [item.provider for item in options])
        option = options[0]
        self.assertIn("model_params.project", option.required_fields)
        self.assertNotIn("model_params.config_file", option.required_fields)
        self.assertIn("model_params.config_file", option.secret_fields)
        self.assertIn("model_params.oci_auth", option.secret_fields)

    def test_oci_accepts_manual_auth_parameters_without_config_file(self):
        auth = {
            "tenancy": "ocid1.tenancy.oc1..test",
            "user": "ocid1.user.oc1..test",
            "fingerprint": "00:11:22:33",
            "region": "us-ashburn-1",
            "key_content": "private-key",
        }
        with patch(
            "model_serving.common.provider_catalog.validated_oci_config",
            return_value=auth,
        ) as validate:
            validate_provider_config({
                "category": ModelCategory.IMAGE_GENERATION.value,
                "provider": "api_oci_image",
                "api_endpoint": "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
                "model_params": {
                    "compartment_id": "ocid1.compartment.oc1..test",
                    "project": "ocid1.generativeaiproject.oc1..test",
                    "oci_auth": auth,
                },
            })

        validate.assert_called_once_with(auth)

    def test_oci_requires_uploaded_or_manual_auth_parameters(self):
        with self.assertRaisesRegex(ValueError, "上传 config 文件或填写"):
            validate_provider_config({
                "category": ModelCategory.IMAGE_GENERATION.value,
                "provider": "api_oci_image",
                "api_endpoint": "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
                "model_params": {
                    "compartment_id": "ocid1.compartment.oc1..test",
                    "project": "ocid1.generativeaiproject.oc1..test",
                },
            })

    def test_local_deepseek_is_openai_compatible_llm_option(self):
        options = list_provider_options(category=1)
        providers = {item.provider for item in options}
        self.assertIn("local_deepseek", providers)
        local = next(item for item in options if item.provider == "local_deepseek")
        self.assertEqual(("api_endpoint", "api_key"), local.required_fields)
        self.assertTrue(local.supports_tool_calling)

    def test_provider_options_do_not_contain_secret_values(self):
        options = list_provider_options(category=1)
        self.assertTrue(options)
        oci_option = next(item for item in options if item.provider == "oci")
        self.assertIn("model_params.config_file", oci_option.secret_fields)
        self.assertIn("project", oci_option.allowed_model_params)
        self.assertNotIn("model_params.project", oci_option.required_fields)
        self.assertNotIn("top-secret", repr(options).lower())

    def test_unknown_model_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知参数"):
            validate_provider_config({
                "category": 1,
                "provider": "api_qwen",
                "api_endpoint": "https://example.invalid/v1",
                "api_key": "secret",
                "model_params": {"unexpected": True},
            })

    def test_malformed_oci_user_is_rejected_before_activation(self):
        with self.assertRaisesRegex(ValueError, "user"):
            validate_provider_config({
                "category": 1,
                "provider": "oci",
                "api_endpoint": "https://example.invalid",
                "model_params": {
                    "compartment_id": "ocid1.compartment.oc1..test",
                    "config_file": {
                        "tenancy": "ocid1.tenancy.oc1..test",
                        "user": "broken-user",
                        "fingerprint": (
                            "00:11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff"
                        ),
                        "region": "us-chicago-1",
                        "key_content": "not-a-private-key",
                    },
                },
            })

    def test_oci_endpoint_must_be_service_origin(self):
        with patch(
            "model_serving.common.provider_catalog.validated_oci_config",
            return_value={},
        ):
            with self.assertRaisesRegex(ValueError, "Origin"):
                validate_provider_config({
                    "category": 1,
                    "provider": "oci",
                    "api_endpoint": (
                        "https://inference.generativeai.us-ashburn-1."
                        "oci.oraclecloud.com/20231130/actions/v1"
                    ),
                    "model_params": {
                        "compartment_id": "ocid1.compartment.oc1..test",
                        "config_file": {"secret": "not-read-in-this-test"},
                    },
                })


if __name__ == "__main__":
    unittest.main()
