"""按 adapter_id、版本和 Capability 创建诊断源 Adapter。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import aiohttp

from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_EVENT_QUERY,
    CAPABILITY_EVENT_RECEIVE,
    CAPABILITY_HEALTH_CHECK,
    CAPABILITY_LOG_QUERY,
    CAPABILITY_METRIC_QUERY_RANGE,
    DiagnosticSourceAdapterDescriptor,
    DiagnosticSourceContext,
)

from .base import BaseDiagnosticSourceAdapter
from .alertmanager import AlertmanagerAdapter
from .oem import OEMAdapter
from .loki import LokiAdapter
from .prometheus import PrometheusAdapter
from .zabbix import ZabbixAdapter


AdapterType: TypeAlias = type[BaseDiagnosticSourceAdapter]


@dataclass(frozen=True)
class DiagnosticSourceAdapterRegistration:
    adapter_id: str
    adapter_version: str
    source_types: frozenset[str]
    capabilities: frozenset[str]
    adapter_type: AdapterType


class DiagnosticSourceAdapterCatalog:
    """内置 Adapter 元数据目录，可在创建网络资源前使用。"""

    _registrations = (
        DiagnosticSourceAdapterRegistration(
            adapter_id="loki",
            adapter_version="1.0.0",
            source_types=frozenset({"LOKI"}),
            capabilities=frozenset(
                {CAPABILITY_HEALTH_CHECK, CAPABILITY_LOG_QUERY}
            ),
            adapter_type=LokiAdapter,
        ),
        DiagnosticSourceAdapterRegistration(
            adapter_id="prometheus",
            adapter_version="1.0.0",
            source_types=frozenset({"PROMETHEUS"}),
            capabilities=frozenset(
                {
                    CAPABILITY_HEALTH_CHECK,
                    CAPABILITY_METRIC_QUERY_RANGE,
                    CAPABILITY_EVENT_QUERY,
                }
            ),
            adapter_type=PrometheusAdapter,
        ),
        DiagnosticSourceAdapterRegistration(
            adapter_id="alertmanager",
            adapter_version="1.0.0",
            source_types=frozenset({"ALERTMANAGER"}),
            capabilities=frozenset(
                {CAPABILITY_HEALTH_CHECK, CAPABILITY_EVENT_RECEIVE}
            ),
            adapter_type=AlertmanagerAdapter,
        ),
        DiagnosticSourceAdapterRegistration(
            adapter_id="zabbix",
            adapter_version="1.0.0",
            source_types=frozenset({"ZABBIX"}),
            capabilities=frozenset(
                {
                    CAPABILITY_HEALTH_CHECK,
                    CAPABILITY_EVENT_RECEIVE,
                    CAPABILITY_EVENT_QUERY,
                    CAPABILITY_METRIC_QUERY_RANGE,
                }
            ),
            adapter_type=ZabbixAdapter,
        ),
        DiagnosticSourceAdapterRegistration(
            adapter_id="oem",
            adapter_version="1.0.0",
            source_types=frozenset({"OEM"}),
            capabilities=frozenset(
                {
                    CAPABILITY_HEALTH_CHECK,
                    CAPABILITY_EVENT_QUERY,
                    CAPABILITY_METRIC_QUERY_RANGE,
                }
            ),
            adapter_type=OEMAdapter,
        ),
    )

    def __init__(self):
        self._by_identity = {
            (item.adapter_id, item.adapter_version): item
            for item in self._registrations
        }
        self._by_source_type = {
            source_type: item
            for item in self._registrations
            for source_type in item.source_types
        }

    def resolve(
        self, *, adapter_id: str, adapter_version: str
    ) -> DiagnosticSourceAdapterRegistration:
        try:
            return self._by_identity[(adapter_id, adapter_version)]
        except KeyError as exc:
            raise LookupError(
                f"未注册的诊断源 Adapter：{adapter_id}@{adapter_version}"
            ) from exc

    def describe(
        self, *, adapter_id: str, adapter_version: str
    ) -> DiagnosticSourceAdapterDescriptor:
        registration = self.resolve(
            adapter_id=adapter_id,
            adapter_version=adapter_version,
        )
        return DiagnosticSourceAdapterDescriptor(
            adapter_id=registration.adapter_id,
            adapter_version=registration.adapter_version,
            source_types=registration.source_types,
            capabilities=registration.capabilities,
        )

    def describe_source_type(
        self, *, source_type: str
    ) -> DiagnosticSourceAdapterDescriptor:
        try:
            registration = self._by_source_type[source_type]
        except KeyError as exc:
            raise LookupError(
                f"未注册的诊断源类型：{source_type}"
            ) from exc
        return DiagnosticSourceAdapterDescriptor(
            adapter_id=registration.adapter_id,
            adapter_version=registration.adapter_version,
            source_types=registration.source_types,
            capabilities=registration.capabilities,
        )

    @staticmethod
    def normalize_config(
        *, source_type: str, config: dict[str, object]
    ) -> dict[str, object]:
        allowed_fields = {
            "LOKI": {"tenant_id"},
        }.get(source_type, set())
        unsupported = sorted(set(config) - allowed_fields)
        if unsupported:
            raise ValueError(
                f"{source_type} 不支持配置项：" + ", ".join(unsupported)
            )
        normalized: dict[str, object] = {}
        for name in allowed_fields:
            if name not in config:
                continue
            value = config[name]
            if not isinstance(value, str):
                raise ValueError(f"{name} 必须是字符串")
            value = value.strip()
            if not value or len(value) > 256 or any(
                character in value for character in "\r\n"
            ):
                raise ValueError(f"{name} 格式无效")
            normalized[name] = value
        if source_type == "ALERTMANAGER":
            normalized["target_label"] = "target_key"
        return normalized


class DiagnosticSourceAdapterRegistry:
    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        catalog: DiagnosticSourceAdapterCatalog | None = None,
        request_timeout_seconds: float = 30,
        webhook_replay_seconds: int = 300,
    ):
        self._session = session
        self._request_timeout_seconds = request_timeout_seconds
        self._webhook_replay_seconds = webhook_replay_seconds
        self._catalog = catalog or DiagnosticSourceAdapterCatalog()

    def create(
        self,
        context: DiagnosticSourceContext,
        *,
        capability: str | None = None,
    ) -> BaseDiagnosticSourceAdapter:
        registration = self._catalog.resolve(
            adapter_id=context.adapter_id,
            adapter_version=context.adapter_version,
        )
        if context.source_type not in registration.source_types:
            raise LookupError(
                f"Adapter {context.adapter_id} 不支持 Source Type "
                f"{context.source_type}"
            )
        if capability is not None:
            if capability not in registration.capabilities:
                raise LookupError(
                    f"Adapter {context.adapter_id} 不支持能力 {capability}"
                )
            if (
                capability != CAPABILITY_HEALTH_CHECK
                and capability not in context.declared_capabilities
            ):
                raise LookupError(
                    f"Diagnostic Source 未声明能力 {capability}"
                )
        return registration.adapter_type(
            context=context,
            session=self._session,
            request_timeout_seconds=self._request_timeout_seconds,
            webhook_replay_seconds=self._webhook_replay_seconds,
            supported_capabilities=registration.capabilities,
        )

    def describe(
        self, *, adapter_id: str, adapter_version: str
    ) -> DiagnosticSourceAdapterDescriptor:
        return self._catalog.describe(
            adapter_id=adapter_id,
            adapter_version=adapter_version,
        )

    def describe_source_type(
        self, *, source_type: str
    ) -> DiagnosticSourceAdapterDescriptor:
        return self._catalog.describe_source_type(source_type=source_type)

    def normalize_config(
        self, *, source_type: str, config: dict[str, object]
    ) -> dict[str, object]:
        return self._catalog.normalize_config(
            source_type=source_type, config=config
        )
