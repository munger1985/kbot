# utils/monitor/__init__.py — 监控子系统
#
# 包含:
#   - 抽象基类 (BaseMonitorProvider) 与统一数据模型 (MetricResult)
#   - Prometheus / Zabbix HTTP 客户端
#   - 统一指标注册中心 (UnifiedMetricRegistry)

from .base import BaseMonitorProvider, MetricResult
from .prometheus import PrometheusClient
from .zabbix import ZabbixProvider
from .registry import UnifiedMetricRegistry


def get_monitor_provider(monitor_type: str = "prometheus") -> BaseMonitorProvider:
    """根据监控类型返回对应的 Provider 实例。

    Args:
        monitor_type: "prometheus" | "zabbix"

    Returns:
        实现了 BaseMonitorProvider 的客户端实例
    """
    if monitor_type == "zabbix":
        return ZabbixProvider()
    return PrometheusClient()


__all__ = [
    "BaseMonitorProvider",
    "MetricResult",
    "PrometheusClient",
    "ZabbixProvider",
    "UnifiedMetricRegistry",
    "get_monitor_provider",
]
