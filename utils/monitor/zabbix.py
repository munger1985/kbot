"""
Zabbix 监控数据源 Provider (预留桩实现)。

当前阶段暂不实现 Zabbix JSON-RPC API 对接。
当需要支持传统 Zabbix 客户时，按照 BaseMonitorProvider 接口填充实现即可。
也可以通过 zabbix-exporter 桥接器将 Zabbix 数据转为 Prometheus 格式，
从而复用 PrometheusClient。
"""

from typing import Any

from loguru import logger

from .base import BaseMonitorProvider, MetricResult


class ZabbixProvider(BaseMonitorProvider):
    """
    Zabbix 监控数据源 Provider —— 暂未实现。

    当需要原生 Zabbix 支持时，此 Provider 将:
      1. 通过 Zabbix JSON-RPC API (api_jsonrpc.php) 通信
      2. 先用 host.get 解析 hostid
      3. 再用 item.get 根据 key 查 itemid
      4. 最后用 history.get 获取历史数据
      5. 将结果转为标准 MetricResult 格式

    架构保证: Agent 核心链路完全不感知底层是 Prometheus 还是 Zabbix，
    只需通过工厂模式注入对应的 Provider 即可。
    """

    def __init__(self, api_url: str = "", token: str = ""):
        self.api_url = api_url
        self.token = token
        logger.warning(
            "[ZabbixProvider] Zabbix 驱动当前为桩实现，所有调用将抛出 NotImplementedError。"
            " 如需 Zabbix 支持，建议优先使用 zabbix-exporter 桥接为 Prometheus 格式。"
        )

    def format_query(self, template: str, params: dict[str, Any]) -> str:
        """Zabbix Item Key 模板渲染 (预留)"""
        raise NotImplementedError(
            "Zabbix 驱动正在灰度测试中，敬请期待。"
            " 当前建议通过 zabbix-exporter 将 Zabbix 数据桥接为 Prometheus 格式后，"
            " 使用 PrometheusClient 进行查询。"
        )

    async def query_instant(self, query_str: str) -> MetricResult:
        """Zabbix 瞬时查询 (预留)"""
        raise NotImplementedError(
            "Zabbix 驱动正在灰度测试中，敬请期待。"
        )

    async def query_range(
        self, query_str: str, start: int, end: int, step: str = "60s"
    ) -> MetricResult:
        """Zabbix 范围查询 (预留)"""
        raise NotImplementedError(
            "Zabbix 驱动正在灰度测试中，敬请期待。"
        )
