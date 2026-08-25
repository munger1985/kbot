"""诊断源配置在落库前执行只读连接预检。"""

from __future__ import annotations

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_HEALTH_CHECK,
    DiagnosticSourceContext,
    SourceHealthRequest,
)
from platform_core.contracts.aiops import (
    DiagnosticSourceConnectionTestResult,
    DiagnosticSourceCreate,
)
from platform_core.identity import uuid7


async def test_diagnostic_source_connection(
    request: DiagnosticSourceCreate, *, diagnostic_source_registry
) -> DiagnosticSourceConnectionTestResult:
    """使用临时上下文调用 Adapter 健康检查，不保存配置或凭据。"""

    try:
        adapter = diagnostic_source_registry.create(
            DiagnosticSourceContext(
                source_id=str(uuid7()),
                source_type=request.source_type,
                adapter_id=request.adapter_id,
                adapter_version=request.adapter_version,
                config_version=1,
                endpoint=str(request.endpoint) if request.endpoint else None,
                credentials={
                    **dict(request.credentials or {}),
                    **dict(request.webhook_credentials or {}),
                },
                declared_capabilities=dict(request.declared_capabilities),
                config=dict(request.config),
            ),
            capability=CAPABILITY_HEALTH_CHECK,
        )
        result = await adapter.health_check(
            SourceHealthRequest(trace_id=str(uuid7()))
        )
        return DiagnosticSourceConnectionTestResult(
            ok=result.healthy,
            error_code=result.error_code,
            discovered_capabilities=result.discovered_capabilities,
        )
    except DiagnosticSourceAdapterError as exc:
        return DiagnosticSourceConnectionTestResult(
            ok=False, error_code=exc.code
        )
    except LookupError:
        return DiagnosticSourceConnectionTestResult(
            ok=False, error_code="SOURCE_ADAPTER_INVALID"
        )
    except Exception:
        return DiagnosticSourceConnectionTestResult(
            ok=False, error_code="SOURCE_UNREACHABLE"
        )
