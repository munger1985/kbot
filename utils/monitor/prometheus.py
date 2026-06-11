"""
Prometheus HTTP API 客户端适配器。

封装 Prometheus 标准的 REST API (/api/v1/query 和 /api/v1/query_range)，
处理网络超时、认证和错误重试。
"""

import aiohttp
from typing import Any
from loguru import logger

from core.config.settings import get_prometheus_config
from .base import BaseMonitorProvider, MetricResult


class PrometheusClient(BaseMonitorProvider):
    """
    Prometheus API 交互客户端。

    用法:
        client = PrometheusClient(base_url="http://localhost:9090")
        result = await client.query_instant("up{job='oracle'}")
        result = await client.query_range("rate(cpu[5m])", start=..., end=...)
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: int | None = None,
    ):
        """
        初始化 Prometheus 客户端。

        Args:
            base_url: Prometheus Server 地址，默认从配置读取
            token: Bearer Token (可选)
            timeout: HTTP 请求超时 (秒)
        """
        config = get_prometheus_config()
        self.base_url = (base_url or config.base_url).rstrip("/")
        self.token = token or config.token
        self.timeout = timeout or config.timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            self._headers["Authorization"] = f"Bearer {self.token}"

        logger.info(f"[PrometheusClient] 初始化完成 | Server: {self.base_url} | Timeout: {self.timeout}s")

    def format_query(self, template: str, params: dict[str, Any]) -> str:
        """
        将 {param_name} 占位符模板渲染为最终 PromQL。

        示例:
            template = 'oracledb_cpu_utilization_ratio{instance="{instance}"}'
            params   = {"instance": "192.168.1.50:9161"}
            → 'oracledb_cpu_utilization_ratio{instance="192.168.1.50:9161"}'
        """
        try:
            return template.format(**params)
        except KeyError as e:
            missing_key = str(e).strip("'")
            logger.error(f"[PrometheusClient] PromQL 渲染缺失参数: {missing_key} | 可用参数: {list(params.keys())}")
            raise ValueError(f"PromQL 模板渲染失败，缺失参数: {missing_key}") from e

    async def query_instant(self, query_str: str) -> MetricResult:
        """
        执行 Prometheus 瞬时查询 (/api/v1/query)。

        Args:
            query_str: 已渲染好的 PromQL 语句

        Returns:
            标准化的 MetricResult 对象
        """
        url = f"{self.base_url}/api/v1/query"
        params = {"query": query_str}

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self._headers,
        ) as session:
            try:
                logger.debug(f"[PrometheusClient] 瞬时查询 | URL: {url} | PromQL: {query_str}")
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    raw = await response.json()

                    # 检查 Prometheus 返回状态
                    if raw.get("status") != "success":
                        error_msg = raw.get("error", "未知 Prometheus 错误")
                        logger.error(f"[PrometheusClient] Prometheus 查询失败: {error_msg}")
                        raise RuntimeError(f"Prometheus 查询失败: {error_msg}")

                    return MetricResult.from_prometheus(
                        metric_code="",  # 由上层调用者填充
                        raw_response=raw,
                    )

            except aiohttp.ClientConnectorError as e:
                logger.error(f"[PrometheusClient] 无法连接 Prometheus Server: {self.base_url} | {e}")
                raise ConnectionError(f"无法连接 Prometheus Server ({self.base_url}): {e}") from e
            except aiohttp.ClientResponseError as e:
                logger.error(f"[PrometheusClient] HTTP 错误 {e.status}: {e.message}")
                raise
            except Exception as e:
                logger.exception(f"[PrometheusClient] 非预期异常: {e}")
                raise

    async def query_range(
        self,
        query_str: str,
        start: int,
        end: int,
        step: str | None = None,
    ) -> MetricResult:
        """
        执行 Prometheus 范围查询 (/api/v1/query_range)。

        Args:
            query_str: 已渲染好的 PromQL 语句
            start: 起始 Unix 时间戳 (秒)
            end: 结束 Unix 时间戳 (秒)
            step: 采样步长，默认从配置读取

        Returns:
            标准化的 MetricResult 对象
        """
        if step is None:
            step = get_prometheus_config().default_step

        url = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": query_str,
            "start": start,
            "end": end,
            "step": step,
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self._headers,
        ) as session:
            try:
                logger.debug(
                    f"[PrometheusClient] 范围查询 | URL: {url} | PromQL: {query_str} "
                    f"| Range: [{start}, {end}] | Step: {step}"
                )
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    raw = await response.json()

                    if raw.get("status") != "success":
                        error_msg = raw.get("error", "未知 Prometheus 错误")
                        logger.error(f"[PrometheusClient] Prometheus 范围查询失败: {error_msg}")
                        raise RuntimeError(f"Prometheus 范围查询失败: {error_msg}")

                    return MetricResult.from_prometheus(
                        metric_code="",
                        raw_response=raw,
                    )

            except aiohttp.ClientConnectorError as e:
                logger.error(f"[PrometheusClient] 无法连接 Prometheus Server: {self.base_url} | {e}")
                raise ConnectionError(f"无法连接 Prometheus Server ({self.base_url}): {e}") from e
            except aiohttp.ClientResponseError as e:
                logger.error(f"[PrometheusClient] HTTP 错误 {e.status}: {e.message}")
                raise
            except Exception as e:
                logger.exception(f"[PrometheusClient] 非预期异常: {e}")
                raise

    async def health_check(self) -> bool:
        """检查 Prometheus Server 是否可达"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5),
                headers=self._headers,
            ) as session:
                url = f"{self.base_url}/api/v1/status/config"
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False
