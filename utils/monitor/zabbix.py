# utils/clients/zabbix_provider.py
"""
Zabbix 监控数据源 Provider — Zabbix JSON-RPC API 实现。

通过 Zabbix 原生的 JSON-RPC 接口 (api_jsonrpc.php) 获取监控数据，
流程:
  1. user.login → 获取 auth token
  2. host.get   → 按 host name 解析 hostid
  3. item.get   → 按 key 查 itemid 及最新值
  4. history.get → 获取历史时间序列数据
"""

import re
from typing import Any

import aiohttp
from loguru import logger

from platform_core.config.settings import get_zabbix_config
from .base import BaseMonitorProvider, MetricResult


class ZabbixProvider(BaseMonitorProvider):
    """
    Zabbix 监控数据源 Provider。

    用法:
        client = ZabbixProvider()
        result = await client.query_instant("oracledb_sessions_value[{host}]")
        result = await client.query_range("oracledb_sessions_value[{host}]", start=..., end=...)
    """

    def __init__(
        self,
        api_url: str | None = None,
        token: str | None = None,
        user: str | None = None,
        password: str | None = None,
        timeout: int | None = None,
    ):
        """
        初始化 Zabbix 客户端。

        Args:
            api_url: Zabbix JSON-RPC API 地址，默认从配置读取
            token: Zabbix API Token（可选，优先级高于 user/password）
            user: Zabbix API 登录用户
            password: Zabbix API 登录密码
            timeout: HTTP 请求超时 (秒)
        """
        config = get_zabbix_config()
        self.api_url = (api_url or config.api_url).rstrip("/")
        self.token = token or config.token
        self.user = user or config.user
        self.password = password or config.password
        self.timeout = timeout or config.timeout
        self._auth_token: str | None = None

        if self.token:
            logger.info(f"[ZabbixProvider] 初始化完成 | API: {self.api_url} | 使用 Token 认证 | Timeout: {self.timeout}s")
        else:
            logger.info(f"[ZabbixProvider] 初始化完成 | API: {self.api_url} | 使用用户: {self.user} | Timeout: {self.timeout}s")

    # ======================================================================
    # Zabbix JSON-RPC 核心调用
    # ======================================================================

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> dict:
        """
        执行 Zabbix JSON-RPC 调用。

        Args:
            method: JSON-RPC 方法名 (如 "host.get", "item.get")
            params: 方法参数

        Returns:
            JSON-RPC 响应中的 result 部分

        Raises:
            ConnectionError: 网络连接失败
            RuntimeError: API 返回错误
        """
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }

        # 除 login 方法外，其他调用需要 auth token
        if method != "user.login":
            auth = await self._get_auth_token()
            payload["auth"] = auth

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as session:
            try:
                logger.debug(f"[ZabbixProvider] JSON-RPC 调用 | method={method}")
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json-rpc"},
                ) as response:
                    response.raise_for_status()
                    raw = await response.json()

                    # 检查 Zabbix API 错误
                    if "error" in raw:
                        error_data = raw["error"]
                        err_msg = error_data.get("data", error_data.get("message", "未知 Zabbix 错误"))
                        err_code = error_data.get("code", -1)
                        logger.error(f"[ZabbixProvider] API 错误 [{err_code}]: {err_msg}")
                        raise RuntimeError(f"Zabbix API 错误 [{err_code}]: {err_msg}")

                    return raw.get("result", {})

            except aiohttp.ClientConnectorError as e:
                logger.error(f"[ZabbixProvider] 无法连接 Zabbix API: {self.api_url} | {e}")
                raise ConnectionError(f"无法连接 Zabbix API ({self.api_url}): {e}") from e
            except aiohttp.ClientResponseError as e:
                logger.error(f"[ZabbixProvider] HTTP 错误 {e.status}: {e.message}")
                raise
            except Exception as e:
                logger.exception(f"[ZabbixProvider] 非预期异常: {e}")
                raise

    async def _get_auth_token(self) -> str:
        """获取 Zabbix API 认证 token（缓存式）"""
        if self._auth_token:
            return self._auth_token

        if self.token:
            self._auth_token = self.token
            logger.debug("[ZabbixProvider] 使用预配置的 Token 认证")
            return self._auth_token

        # 通过 user.login 获取 token
        try:
            result = await self._call("user.login", {
                "user": self.user,
                "password": self.password,
            })
            if isinstance(result, str):
                self._auth_token = result
                logger.debug(f"[ZabbixProvider] user.login 成功 | user={self.user}")
            else:
                raise RuntimeError(f"user.login 返回非预期结果: {result}")
        except Exception as e:
            logger.error(f"[ZabbixProvider] 认证失败: {e}")
            raise

        return self._auth_token

    async def _get_host_id(self, host_name: str) -> str | None:
        """
        根据主机名称获取 Zabbix hostid。

        Args:
            host_name: Zabbix 中配置的主机名称

        Returns:
            hostid 字符串，未找到时返回 None
        """
        try:
            result = await self._call("host.get", {
                "filter": {"host": [host_name]},
                "output": ["hostid"],
            })
            if isinstance(result, list) and len(result) > 0:
                host_id = result[0].get("hostid")
                logger.debug(f"[ZabbixProvider] host.get 成功 | host={host_name} | hostid={host_id}")
                return host_id
            else:
                logger.warning(f"[ZabbixProvider] 未找到 Zabbix 主机: {host_name}")
                return None
        except Exception as e:
            logger.error(f"[ZabbixProvider] host.get 失败: host={host_name} | {e}")
            return None

    # ======================================================================
    # BaseMonitorProvider 接口实现
    # ======================================================================

    def format_query(self, template: str, params: dict[str, Any]) -> str:
        """
        将 {param_name} 占位符模板渲染为最终 Zabbix Item Key。

        示例:
            template = 'oracledb_sessions_value[{host}]'
            params   = {"host": "db-server-01"}
            → 'oracledb_sessions_value[db-server-01]'
        """
        try:
            return template.format(**params)
        except KeyError as e:
            missing_key = str(e).strip("'")
            logger.error(f"[ZabbixProvider] Item Key 渲染缺失参数: {missing_key} | 可用参数: {list(params.keys())}")
            raise ValueError(f"Zabbix Item Key 渲染失败，缺失参数: {missing_key}") from e

    async def query_instant(self, query_str: str) -> MetricResult:
        """
        执行 Zabbix 瞬时查询 — 获取当前最新值。

        通过 item.get 按 key_ 搜索，返回 lastvalue / lastclock。

        Args:
            query_str: 已渲染好的 Zabbix Item Key（如 "oracledb_sessions_value[db-01]"）

        Returns:
            标准化的 MetricResult 对象
        """
        # 从查询字符串中提取主机和 item key
        host_name, item_key = self._parse_item_key(query_str)

        if not host_name:
            # 如果解析不出主机名，直接按 key 搜索
            return await self._query_item_by_key(item_key or query_str)

        # 1. 获取 hostid
        host_id = await self._get_host_id(host_name)
        if not host_id:
            logger.warning(f"[ZabbixProvider] 主机未找到: {host_name}，按 key 全量搜索")
            return await self._query_item_by_key(item_key or query_str)

        # 2. 获取 item 最新值
        try:
            result = await self._call("item.get", {
                "hostids": host_id,
                "search": {"key_": item_key},
                "output": ["itemid", "name", "key_", "lastvalue", "lastclock", "value_type"],
                "sortfield": "name",
            })

            if isinstance(result, list) and len(result) > 0:
                logger.info(
                    f"[ZabbixProvider] item.get 成功 | host={host_name} | "
                    f"key={item_key} | items={len(result)}"
                )
                return MetricResult.from_zabbix("", result)
            else:
                logger.warning(f"[ZabbixProvider] 未找到匹配的 item | host={host_name} | key={item_key}")
                return MetricResult.from_zabbix("", [])

        except Exception as e:
            logger.error(f"[ZabbixProvider] item.get 失败: host={host_name} | key={item_key} | {e}")
            raise

    async def query_range(
        self, query_str: str, start: int, end: int, step: str = "60s"
    ) -> MetricResult:
        """
        执行 Zabbix 范围查询 — 获取历史时间序列数据。

        通过 history.get 获取指定时间范围内的历史数据。

        Args:
            query_str: 已渲染好的 Zabbix Item Key
            start: 起始 Unix 时间戳 (秒)
            end: 结束 Unix 时间戳 (秒)
            step: 采样步长（Zabbix 侧由 server 决定，此参数仅做日志标记）

        Returns:
            标准化的 MetricResult 对象
        """
        host_name, item_key = self._parse_item_key(query_str)
        if not host_name:
            host_name = "unknown"

        # 先获取 itemid
        host_id = await self._get_host_id(host_name)
        if not host_id:
            return MetricResult.from_zabbix("", [])

        try:
            item_result = await self._call("item.get", {
                "hostids": host_id,
                "search": {"key_": item_key},
                "output": ["itemid", "value_type"],
                "limit": 1,
            })

            if not isinstance(item_result, list) or len(item_result) == 0:
                logger.warning(f"[ZabbixProvider] 范围查询: 未找到 item | host={host_name} | key={item_key}")
                return MetricResult.from_zabbix("", [])

            item = item_result[0]
            item_id = item.get("itemid")
            value_type = item.get("value_type", "3")

            # 获取历史数据
            history_result = await self._call("history.get", {
                "itemids": item_id,
                "history": value_type,
                "sortfield": "clock",
                "sortorder": "DESC",
                "time_from": start,
                "time_till": end,
                "limit": 1000,
            })

            if isinstance(history_result, list):
                # history.get 返回按时间倒序，需要反转
                history_result.reverse()
                logger.info(
                    f"[ZabbixProvider] history.get 成功 | itemid={item_id} | "
                    f"points={len(history_result)} | range=[{start}, {end}]"
                )
                return MetricResult.from_zabbix("", history_result)

            return MetricResult.from_zabbix("", [])

        except Exception as e:
            logger.error(f"[ZabbixProvider] 范围查询失败: host={host_name} | key={item_key} | {e}")
            raise

    async def _query_item_by_key(self, item_key: str) -> MetricResult:
        """
        直接按 Item Key 全量搜索（不指定主机）。
        作为无法解析主机名时的降级策略。
        """
        try:
            result = await self._call("item.get", {
                "search": {"key_": item_key},
                "output": ["itemid", "name", "key_", "lastvalue", "lastclock", "value_type"],
                "sortfield": "name",
                "limit": 20,
            })
            if isinstance(result, list):
                logger.info(f"[ZabbixProvider] 按 key 全量搜索 | key={item_key} | items={len(result)}")
                return MetricResult.from_zabbix("", result)
            return MetricResult.from_zabbix("", [])
        except Exception as e:
            logger.error(f"[ZabbixProvider] 按 key 搜索失败: key={item_key} | {e}")
            return MetricResult.from_zabbix("", [])

    @staticmethod
    def _parse_item_key(query_str: str) -> tuple[str | None, str]:
        """
        解析 Zabbix Item Key 格式，提取主机名和 item key。

        支持的格式:
          - "key[hostname]"                 → ("hostname", "key")
          - "key[{host}]" (未渲染模板)     → (None, "key[{host}]")
          - "key" (无方括号)               → (None, "key")

        Returns:
            (host_name, item_key)
        """
        # 匹配末尾的 "[值]" 部分，值不含方括号、不是模板占位符
        m = re.match(r"^(.+?)\[([^{}\[\]]+)\]$", query_str)
        if m:
            return m.group(2), m.group(1)
        return None, query_str

    async def health_check(self) -> bool:
        """检查 Zabbix API 是否可达"""
        try:
            await self._call("apiinfo.version")
            return True
        except Exception:
            return False
