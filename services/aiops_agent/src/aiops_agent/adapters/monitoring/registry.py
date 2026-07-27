"""按已验证 Source Type 创建 Monitor Adapter。"""

from __future__ import annotations

import aiohttp

from aiops_agent.ports.monitor import MonitorProviderContext

from .base import BaseMonitorAdapter
from .oem import OEMAdapter
from .prometheus import PrometheusAdapter
from .zabbix import ZabbixAdapter


class MonitorProviderRegistry:
    _adapters = {
        "PROMETHEUS": PrometheusAdapter,
        "ZABBIX": ZabbixAdapter,
        "OEM": OEMAdapter,
    }

    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        request_timeout_seconds: float = 30,
        webhook_replay_seconds: int = 300,
    ):
        self._session = session
        self._request_timeout_seconds = request_timeout_seconds
        self._webhook_replay_seconds = webhook_replay_seconds

    def create(
        self, context: MonitorProviderContext
    ) -> BaseMonitorAdapter:
        try:
            adapter = self._adapters[context.source_type]
        except KeyError as exc:
            raise LookupError(
                f"不支持的 Monitor Provider：{context.source_type}"
            ) from exc
        return adapter(
            context=context,
            session=self._session,
            request_timeout_seconds=self._request_timeout_seconds,
            webhook_replay_seconds=self._webhook_replay_seconds,
        )
