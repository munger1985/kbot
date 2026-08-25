"""诊断源连接预检单元测试。"""

import unittest

from aiops_agent.application.configuration.source_connection_test import (
    test_diagnostic_source_connection as run_connection_test,
)
from aiops_agent.ports.diagnostic_source import SourceHealthResult
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
    def create(self, context, *, capability):
        self.context = context
        self.capability = capability
        return _Adapter()


class DiagnosticSourceConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_connection_uses_temporary_adapter_context(self):
        registry = _Registry()
        request = DiagnosticSourceCreate.model_validate(
            {
                "display_name": "测试 Prometheus",
                "source_type": "PROMETHEUS",
                "adapter_id": "prometheus",
                "adapter_version": "1.0.0",
                "endpoint": "http://prometheus.internal:9090",
                "credentials": {"token": "temporary"},
            }
        )

        result = await run_connection_test(
            request, diagnostic_source_registry=registry
        )

        self.assertTrue(result.ok)
        self.assertEqual("PROMETHEUS", registry.context.source_type)
        self.assertEqual("temporary", registry.context.credentials["token"])
        self.assertEqual("health.check", registry.capability)

    async def test_invalid_adapter_is_returned_as_safe_error(self):
        class InvalidRegistry:
            def create(self, context, *, capability):
                del context, capability
                raise LookupError("missing")

        request = DiagnosticSourceCreate.model_validate(
            {
                "display_name": "未知源",
                "source_type": "UNKNOWN",
                "adapter_id": "unknown",
                "adapter_version": "1.0.0",
                "endpoint": "http://unknown.internal",
            }
        )
        result = await run_connection_test(
            request, diagnostic_source_registry=InvalidRegistry()
        )

        self.assertFalse(result.ok)
        self.assertEqual("SOURCE_ADAPTER_INVALID", result.error_code)


if __name__ == "__main__":
    unittest.main()
