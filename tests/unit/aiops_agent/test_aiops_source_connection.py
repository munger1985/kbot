"""诊断源连接预检单元测试。"""

import unittest

from aiops_agent.application.configuration.source_connection_test import (
    test_diagnostic_source_connection as run_connection_test,
)
from aiops_agent.ports.diagnostic_source import (
    DiagnosticSourceAdapterDescriptor,
    SourceHealthResult,
)
from platform_core.contracts.aiops import DiagnosticSourceCreate


class _Adapter:
    async def health_check(self, request):
        del request
        return SourceHealthResult(
            healthy=True,
            adapter_id="prometheus",
            adapter_version="1.0.0",
            discovered_capabilities=("health.check",),
        )


class _Registry:
    def describe_source_type(self, *, source_type):
        if source_type != "PROMETHEUS":
            raise LookupError("missing")
        return DiagnosticSourceAdapterDescriptor(
            adapter_id="prometheus",
            adapter_version="1.0.0",
            source_types=frozenset({"PROMETHEUS"}),
            capabilities=frozenset(
                {"health.check", "metric.query_range", "event.query"}
            ),
        )

    def create(self, context, *, capability):
        self.context = context
        self.capability = capability
        return _Adapter()

    def normalize_config(self, *, source_type, config):
        del source_type
        return config


class DiagnosticSourceConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_connection_uses_temporary_adapter_context(self):
        registry = _Registry()
        request = DiagnosticSourceCreate.model_validate(
            {
                "display_name": "测试 Prometheus",
                "source_type": "PROMETHEUS",
                "endpoint": "http://prometheus.internal:9090",
                "credentials": {"token": "temporary"},
            }
        )

        result = await run_connection_test(
            request, diagnostic_source_registry=registry
        )

        self.assertTrue(result.ok)
        self.assertEqual("PROMETHEUS", registry.context.source_type)
        self.assertEqual("prometheus", registry.context.adapter_id)
        self.assertIn(
            "metric.query_range", registry.context.declared_capabilities
        )
        self.assertEqual("temporary", registry.context.credentials["token"])
        self.assertEqual("health.check", registry.capability)

    async def test_invalid_adapter_is_returned_as_safe_error(self):
        class InvalidRegistry:
            def describe_source_type(self, *, source_type):
                del source_type
                raise LookupError("missing")

            def create(self, context, *, capability):
                del context, capability
                raise LookupError("missing")

        request = DiagnosticSourceCreate.model_validate(
            {
                "display_name": "未知源",
                "source_type": "UNKNOWN",
                "endpoint": "http://unknown.internal",
            }
        )
        result = await run_connection_test(
            request, diagnostic_source_registry=InvalidRegistry()
        )

        self.assertFalse(result.ok)
        self.assertEqual("SOURCE_ADAPTER_INVALID", result.error_code)

    async def test_invalid_config_is_returned_as_safe_error(self):
        class InvalidConfigRegistry(_Registry):
            def normalize_config(self, *, source_type, config):
                del source_type, config
                raise ValueError("unsupported")

        request = DiagnosticSourceCreate.model_validate(
            {
                "display_name": "Prometheus",
                "source_type": "PROMETHEUS",
                "endpoint": "http://prometheus.internal:9090",
                "config": {"custom_query": "up"},
            }
        )
        result = await run_connection_test(
            request, diagnostic_source_registry=InvalidConfigRegistry()
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            "SOURCE_CONFIGURATION_INVALID", result.error_code
        )


if __name__ == "__main__":
    unittest.main()
