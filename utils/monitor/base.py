# utils/clients/base_monitor_provider.py
"""
监控数据源抽象基类。
面向 Prometheus / Zabbix / 国产监控工具的统一接口定义，
所有监控 Provider 必须实现此基类。
"""

from abc import ABC, abstractmethod
from typing import Any


class MetricResult:
    """
    统一标准化的监控指标返回对象。

    无论底层是 Prometheus (HTTP API)、Zabbix (JSON-RPC) 还是数据库 SQL 直查，
    最终返回给 Agent 的数据格式必须统一为此结构。
    """

    def __init__(
        self,
        metric_code: str,
        data_type: str = "vector",
        series: list[dict[str, Any]] | None = None,
    ):
        """
        Args:
            metric_code: 统一指标编码 (如 "db_cpu_utilization")
            data_type: "vector" (瞬时单值) 或 "matrix" (时间序列)
            series: 标准化数据点列表
        """
        self.metric_code = metric_code
        self.data_type = data_type
        self.series: list[dict[str, Any]] = series or []

    def to_summary(self) -> dict[str, Any]:
        """返回给 LLM 消费的精简摘要"""
        return {
            "metric_code": self.metric_code,
            "data_type": self.data_type,
            "series": self.series,
        }

    @classmethod
    def from_prometheus(cls, metric_code: str, raw_response: dict) -> "MetricResult":
        """
        从 Prometheus 原始 HTTP 响应解析为标准 MetricResult。

        Prometheus API 返回的 resultType 可能是:
          - "vector":  瞬时查询结果
          - "matrix":  范围查询结果
        """
        data = raw_response.get("data", {})
        result_type = data.get("resultType", "vector")
        results = data.get("result", [])

        series: list[dict[str, Any]] = []
        for item in results:
            metric_labels = item.get("metric", {})

            if result_type == "vector":
                # 瞬时查询: {"value": [timestamp, "value_string"]}
                value_pair = item.get("value", [0, "0"])
                timestamp = int(value_pair[0]) if value_pair else 0
                value = float(value_pair[1]) if len(value_pair) > 1 else 0.0
                series.append({
                    "labels": metric_labels,
                    "timestamp": timestamp,
                    "value": value,
                })
            elif result_type == "matrix":
                # 范围查询: {"values": [[t1, "v1"], [t2, "v2"], ...]}
                values = item.get("values", [])
                datapoints = [[int(t), float(v)] for t, v in values]
                series.append({
                    "labels": metric_labels,
                    "datapoints": datapoints,
                })

        return cls(
            metric_code=metric_code,
            data_type=result_type,
            series=series,
        )

    def __repr__(self) -> str:
        return f"<MetricResult(code={self.metric_code}, type={self.data_type}, series_count={len(self.series)})>"

    @classmethod
    def from_zabbix(cls, metric_code: str, items: list[dict]) -> "MetricResult":
        """
        从 Zabbix item.get / history.get 响应解析为标准 MetricResult。

        Zabbix item.get 返回:
          [{"itemid": "1", "name": "...", "key_": "...", "lastvalue": "42", "lastclock": "1712345678", "value_type": "3"}, ...]

        Zabbix history.get 返回:
          [{"itemid": "1", "clock": "1712345678", "value": "42", "ns": "123456789"}, ...]
        """
        series: list[dict[str, Any]] = []
        for item in items:
            entry: dict[str, Any] = {"labels": {}}
            # item.get 返回 lastvalue / lastclock
            if "lastvalue" in item and "lastclock" in item:
                entry["value"] = _parse_zabbix_value(item.get("lastvalue", "0"))
                entry["timestamp"] = int(item.get("lastclock", 0))
                entry["labels"]["itemid"] = item.get("itemid", "")
                entry["labels"]["name"] = item.get("name", "")
                entry["labels"]["key_"] = item.get("key_", "")
            # history.get 返回 clock / value（批量）
            elif "clock" in item:
                entry["value"] = _parse_zabbix_value(item.get("value", "0"))
                entry["timestamp"] = int(item.get("clock", 0))
                if "ns" in item:
                    entry["labels"]["ns"] = item.get("ns", "")
            if "value" in entry:
                series.append(entry)

        return cls(
            metric_code=metric_code,
            data_type="vector",
            series=series,
        )


def _parse_zabbix_value(raw: Any) -> float:
    """安全解析 Zabbix 值（可能为字符串或数字）"""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


class BaseMonitorProvider(ABC):
    """
    监控数据源抽象基类。

    所有监控工具 (Prometheus, Zabbix, Nightingale 等) 的 Provider
    都必须实现此接口，确保 Agent 核心链路与具体监控工具完全解耦。
    """

    @abstractmethod
    async def query_instant(self, query_str: str) -> MetricResult:
        """
        执行即时查询，返回当前最新值。

        Args:
            query_str: 已渲染好的查询语句 (PromQL / Zabbix Key / ...)

        Returns:
            标准化的 MetricResult 对象
        """
        ...

    @abstractmethod
    async def query_range(
        self, query_str: str, start: int, end: int, step: str = "60s"
    ) -> MetricResult:
        """
        执行范围查询，返回历史时间序列数据。

        Args:
            query_str: 已渲染好的查询语句
            start: 起始 Unix 时间戳 (秒)
            end: 结束 Unix 时间戳 (秒)
            step: 采样步长 (如 "60s", "5m")

        Returns:
            标准化的 MetricResult 对象
        """
        ...

    @abstractmethod
    def format_query(self, template: str, params: dict[str, Any]) -> str:
        """
        将统一占位符模板渲染为该监控工具的具体查询语句。

        Args:
            template: 带 {param_name} 占位符的模板字符串
            params: 参数键值对

        Returns:
            渲染后的最终查询语句
        """
        ...
