"""Model Serving S4 生命周期与一致性测试。"""

import unittest
from types import SimpleNamespace

from model_serving.common.model_registry import (
    ModelRegistryConflict,
    ModelRegistryService,
)
from model_serving.common.provider_catalog import (
    list_provider_options,
    validate_provider_config,
)
from platform_core.contracts import AuthContext, PrincipalKind
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


async def _empty():
    return []


class ProviderCatalogS4Test(unittest.TestCase):
    def test_provider_options_do_not_contain_secret_values(self):
        options = list_provider_options(category=1)
        self.assertTrue(options)
        oci_option = next(item for item in options if item.provider == "oci")
        self.assertIn("model_params.config_file", oci_option.secret_fields)
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


if __name__ == "__main__":
    unittest.main()
