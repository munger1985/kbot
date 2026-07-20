"""
Oracle Enterprise Manager REST API Provider — 原生 Oracle 深度监控数据源。

通过 OEM 13c R4/R5 的 REST API 获取 Oracle 数据库原生深度监控指标。
支持 OEM 内部的自动发现、Incident 告警框架和 AWR/ASH 等 Oracle 专有诊断能力。

认证流程:
  13.4: POST /em/api/{version}/login → 返回 auth token cookie
  13.5: POST /em/api/{version}/login → 返回 bearer token (兼容 13.4 模式)
  也支持预配置的 Token 直接认证（避免周期登录）

指标查询流程:
  query_instant:  GET /em/rest/{version}/targets/{targetName}/metrics/{metricName}/{collection}
  query_range:    OEM REST API 默认返回最近采集值，历史趋势需走 /metrics/{metric}/{collection}?reporting=avg&from=...&to=...
"""

import re
from typing import Any
from loguru import logger

import aiohttp

from core.config.settings import get_oem_config
from .base import BaseMonitorProvider, MetricResult


class OEMProvider(BaseMonitorProvider):
    """
    Oracle Enterprise Manager REST API 数据源 Provider。

    用法:
        client = OEMProvider()
        result = await client.query_instant("orcl|sessions|response")
        result = await client.query_range("orcl|sessions|response", start=..., end=...)
    """

    OEM_API_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")

    def __init__(
        self,
        base_url: str | None = None,
        api_version: str | None = None,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        timeout: int | None = None,
        verify_ssl: bool | None = None,
    ):
        """
        初始化 OEM 客户端。

        Args:
            base_url:   OEM 基础 URL，如 https://oem-server:7803/em，默认从配置读取
            api_version: OEM REST API 版本（路径段，如 "v1"），默认从配置读取
            username:    OEM 登录用户名
            password:    OEM 登录密码
            token:       OEM 预生成 API Token（可选，优先级高于 user/password）
            timeout:     HTTP 请求超时 (秒)
            verify_ssl:  是否验证 SSL 证书
        """
        config = get_oem_config()
        self.base_url = (base_url or config.base_url).rstrip("/")
        self.api_version = api_version or config.api_version
        self.username = username or config.username
        self.password = password or config.password
        self.token = token or config.token
        self.timeout = timeout or config.timeout
        self.verify_ssl = verify_ssl if verify_ssl is not None else config.verify_ssl

        self._auth_token: str | None = None
        self._session_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.token:
            logger.info(
                f"[OEMProvider] 初始化完成 | URL: {self.base_url} | "
                f"API: v{self.api_version} | 使用 Token 认证 | Timeout: {self.timeout}s"
            )
        else:
            logger.info(
                f"[OEMProvider] 初始化完成 | URL: {self.base_url} | "
                f"API: v{self.api_version} | 用户: {self.username} | Timeout: {self.timeout}s"
            )

    # ======================================================================
    # 认证
    # ======================================================================

    async def _ensure_authenticated(self) -> dict[str, str]:
        """
        确保已认证，返回请求头。
        
        策略: 如果有预配置 token，直接使用；
              否则通过 user.login 获取 token。
        """
        if self._auth_token:
            return {**self._session_headers, "Authorization": f"Bearer {self._auth_token}"}

        if self.token:
            self._auth_token = self.token
            logger.debug("[OEMProvider] 使用预配置 Token")
            return {**self._session_headers, "Authorization": f"Bearer {self._auth_token}"}

        # POST /em/api/{version}/login 获取 token
        login_url = f"{self.base_url}/api/{self.api_version}/login"
        payload = {
            "username": self.username,
            "password": self.password,
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(ssl=self.verify_ssl),
        ) as session:
            try:
                logger.debug(f"[OEMProvider] 正在登录 OEM | URL: {login_url}")
                async with session.post(
                    login_url,
                    json=payload,
                    headers=self._session_headers,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # OEM 13.4/13.5 返回 {"token": "..."} 或 {"authToken": "..."}
                        self._auth_token = data.get("token") or data.get("authToken")
                        if self._auth_token:
                            logger.success("[OEMProvider] OEM 登录成功")
                            return {**self._session_headers, "Authorization": f"Bearer {self._auth_token}"}

                    logger.error(f"[OEMProvider] OEM 登录失败 | HTTP {resp.status}")
                    raise RuntimeError(f"OEM 登录失败 (HTTP {resp.status})")

            except aiohttp.ClientConnectorError as e:
                logger.error(f"[OEMProvider] 无法连接 OEM: {self.base_url} | {e}")
                raise ConnectionError(f"无法连接 OEM Server ({self.base_url}): {e}") from e

    async def _clear_auth(self):
        """清除认证 token（登录过期时调用）"""
        self._auth_token = None

    # ======================================================================
    # 目标查询
    # ======================================================================

    async def list_targets(
        self,
        target_type: str | None = "oracle_database",
        search: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        查询 OEM 中被管目标列表。

        Args:
            target_type: 目标类型过滤（如 "oracle_database", "host"），None 表示全部
            search:      模糊搜索目标名称
            limit:       最大返回数

        Returns:
            目标列表，每项含 targetName, targetType, status 等字段
        """
        headers = await self._ensure_authenticated()
        params: dict[str, Any] = {"limit": limit}

        if target_type:
            params["type"] = target_type
        if search:
            params["search"] = search

        url = f"{self.base_url}/rest/{self.api_version}/targets"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(ssl=self.verify_ssl),
            headers=headers,
        ) as session:
            try:
                logger.debug(f"[OEMProvider] 目标查询 | type={target_type} | search={search}")
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    items = data.get("items", []) if isinstance(data, dict) else data
                    logger.debug(f"[OEMProvider] 目标查询成功 | count={len(items)}")
                    return items
            except Exception as e:
                logger.error(f"[OEMProvider] 目标查询失败: {e}")
                raise

    # ======================================================================
    # BaseMonitorProvider 接口实现
    # ======================================================================

    OEM_QUERY_PATTERN = re.compile(r"^(.+?)\|(.+?)\|(.+)$")

    def format_query(self, template: str, params: dict[str, Any]) -> str:
        """
        将 {param_name} 占位符模板渲染为 OEM 查询字符串。

        OEM 查询字符串格式: "targetName|metricName|collectionName"

        示例:
            template = '{target}|sessions|response'
            params   = {"target": "orcl"}
            → 'orcl|sessions|response'
        """
        try:
            return template.format(**params)
        except KeyError as e:
            missing_key = str(e).strip("'")
            logger.error(f"[OEMProvider] OEM 查询模板渲染缺失参数: {missing_key}")
            raise ValueError(f"OEM 查询模板渲染失败，缺失参数: {missing_key}") from e

    def _parse_query(self, query_str: str) -> tuple[str, str, str]:
        """
        解析 OEM 查询字符串 "targetName|metricName|collectionName"。

        Returns:
            (target_name, metric_name, collection_name)
        """
        m = self.OEM_QUERY_PATTERN.match(query_str)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        raise ValueError(
            f"OEM 查询字符串格式错误: '{query_str}'。"
            f"期望格式: 'targetName|metricName|collectionName'"
        )

    async def query_instant(self, query_str: str, _retry: int = 1) -> MetricResult:
        """
        执行 OEM 即时查询 — 获取当前最新采集指标值。

        Args:
            query_str: "targetName|metricName|collectionName" 格式字符串
            _retry:    内部使用，401 重试次数（默认 1 次）

        Returns:
            标准化的 MetricResult 对象
        """
        target_name, metric_name, collection = self._parse_query(query_str)
        headers = await self._ensure_authenticated()

        url = (
            f"{self.base_url}/rest/{self.api_version}/targets/"
            f"{target_name}/metrics/{metric_name}/{collection}"
        )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(ssl=self.verify_ssl),
            headers=headers,
        ) as session:
            try:
                logger.debug(
                    f"[OEMProvider] 即时查询 | target={target_name} | "
                    f"metric={metric_name} | collection={collection}"
                )
                async with session.get(url) as resp:
                    if resp.status == 401 and _retry > 0:
                        logger.warning("[OEMProvider] Token 过期，重新认证并重试")
                        await self._clear_auth()
                        return await self.query_instant(query_str, _retry=_retry - 1)

                    resp.raise_for_status()
                    raw = await resp.json()

                    logger.debug(
                        f"[OEMProvider] 查询成功 | target={target_name} | "
                        f"metric={metric_name} | HTTP {resp.status}"
                    )
                    return MetricResult.from_oem("", raw)

            except aiohttp.ClientConnectorError as e:
                logger.error(f"[OEMProvider] 无法连接 OEM: {self.base_url} | {e}")
                raise ConnectionError(f"无法连接 OEM Server ({self.base_url}): {e}") from e
            except aiohttp.ClientResponseError as e:
                logger.error(f"[OEMProvider] HTTP 错误 {e.status}: {e.message}")
                raise
            except Exception as e:
                logger.exception(f"[OEMProvider] 非预期异常: {e}")
                raise

    async def query_range(
        self,
        query_str: str,
        start: int,
        end: int,
        step: str = "60s",
        _retry: int = 1,
    ) -> MetricResult:
        """
        执行 OEM 范围查询 — 获取历史时间序列数据。

        OEM REST API 通过 reporting 参数获取历史聚合数据。

        Args:
            query_str: "targetName|metricName|collectionName" 格式字符串
            start:     起始 Unix 时间戳 (秒)
            end:       结束 Unix 时间戳 (秒)
            step:      采样步长
            _retry:    内部使用，401 重试次数（默认 1 次），对应 OEM 的 reporting 参数
                     
                       OEM reporting 值映射:
                       "60s"  → "realtime"   (最近 1 小时)
                       "300s" → "hourly"     (最近 7 天)
                       "3600s"→ "daily"      (最近 31 天)

        Returns:
            标准化的 MetricResult 对象
        """
        target_name, metric_name, collection = self._parse_query(query_str)
        headers = await self._ensure_authenticated()

        # 将 step 映射为 OEM 的 reporting 模式
        reporting = self._step_to_reporting(step)

        url = (
            f"{self.base_url}/rest/{self.api_version}/targets/"
            f"{target_name}/metrics/{metric_name}/{collection}"
        )

        params: dict[str, Any] = {
            "reporting": reporting,
            "from": start,
            "to": end,
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(ssl=self.verify_ssl),
            headers=headers,
        ) as session:
            try:
                logger.debug(
                    f"[OEMProvider] 范围查询 | target={target_name} | "
                    f"metric={metric_name} | collection={collection} | "
                    f"reporting={reporting} | range=[{start}, {end}]"
                )
                async with session.get(url, params=params) as resp:
                    if resp.status == 401 and _retry > 0:
                        logger.warning("[OEMProvider] Token 过期，重新认证并重试")
                        await self._clear_auth()
                        return await self.query_range(query_str, start, end, step, _retry=_retry - 1)

                    resp.raise_for_status()
                    raw = await resp.json()

                    logger.debug(
                        f"[OEMProvider] 范围查询成功 | target={target_name} "
                        f"| HTTP {resp.status}"
                    )
                    return MetricResult.from_oem("", raw)

            except aiohttp.ClientConnectorError as e:
                logger.error(f"[OEMProvider] 无法连接 OEM: {self.base_url} | {e}")
                raise ConnectionError(f"无法连接 OEM Server ({self.base_url}): {e}") from e
            except aiohttp.ClientResponseError as e:
                logger.error(f"[OEMProvider] HTTP 错误 {e.status}: {e.message}")
                raise
            except Exception as e:
                logger.exception(f"[OEMProvider] 非预期异常: {e}")
                raise

    # ======================================================================
    # Incident / Alert 查询
    # ======================================================================

    async def list_incidents(
        self,
        target_name: str,
        severity: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        查询 OEM 目标的最新 Incident（告警事件）。

        Args:
            target_name: OEM 目标名称
            severity:    过滤严重级别: "FATAL", "CRITICAL", "WARNING", "MINOR"
            status:      过滤状态: "OPEN", "ACKNOWLEDGED", "CLOSED"
            limit:       最大返回条数

        Returns:
            Incident 列表
        """
        headers = await self._ensure_authenticated()
        params: dict[str, Any] = {"limit": limit}
        if severity:
            params["severity"] = severity
        if status:
            params["status"] = status

        url = (
            f"{self.base_url}/rest/{self.api_version}/targets/"
            f"{target_name}/incidents"
        )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(ssl=self.verify_ssl),
            headers=headers,
        ) as session:
            try:
                logger.debug(
                    f"[OEMProvider] 查询 Incident | target={target_name} | "
                    f"severity={severity} | status={status}"
                )
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    items = data.get("items", []) if isinstance(data, dict) else data
                    logger.debug(f"[OEMProvider] Incident 查询成功 | count={len(items)}")
                    return items
            except Exception as e:
                logger.error(f"[OEMProvider] Incident 查询失败: {e}")
                raise

    # ======================================================================
    # 辅助方法
    # ======================================================================

    @staticmethod
    def _step_to_reporting(step: str) -> str:
        """
        将 Prometheus 风格的采样步长映射为 OEM reporting 模式。

        OEM reporting 模式:
          - "realtime":  real-time, 细粒度, 通常保存最近 1 小时
          - "hourly":    小时聚合, 保存最近 7 天
          - "daily":     天聚合, 保存最近 31 天

        Mapping 规则:
          step ≤ 60s   → realtime
          step ≤ 3600s → hourly
          step > 3600s → daily
        """
        try:
            # 解析 step 字符串, 如 "60s", "300s", "5m", "1h"
            seconds = 0
            if step.endswith("s"):
                seconds = int(step[:-1])
            elif step.endswith("m"):
                seconds = int(step[:-1]) * 60
            elif step.endswith("h"):
                seconds = int(step[:-1]) * 3600
            else:
                seconds = int(step)

            if seconds <= 60:
                return "realtime"
            elif seconds <= 3600:
                return "hourly"
            else:
                return "daily"
        except (ValueError, TypeError):
            return "hourly"

    async def health_check(self) -> bool:
        """检查 OEM API 是否可达"""
        try:
            headers = await self._ensure_authenticated()
            url = f"{self.base_url}/rest/{self.api_version}/targets"
            params = {"limit": 1}
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=aiohttp.TCPConnector(ssl=self.verify_ssl),
                headers=headers,
            ) as session:
                async with session.get(url, params=params) as resp:
                    return resp.status == 200
        except Exception:
            return False
