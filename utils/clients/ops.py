# utils/clients/ops.py

import os
import aiohttp
from loguru import logger
from typing import Any, Literal

from core.config.settings import get_executor_config
from core.exceptions import InternalServerError


class OpsDBExecutor:
    """
    智能运维专职内核执行客户端。
    对接微服务 /api/v1/ops/execute 端点, 专供运维线各自愈、诊断 Skill 使用。
    """

    def __init__(self):
        """初始化运维服务中心配置"""
        self.executor_config = get_executor_config()
        self.service_host = self.executor_config.service_host
        self.service_port = self.executor_config.service_port

        # 从环境变量提取用于微服务互信的通信令牌
        self.internal_token = os.getenv("INTERNAL_OPS_TOKEN", "SECRET_TOKEN_FOR_PROMETHEUS_2026")

        # 延迟导入运维专用的 CMDB 元数据服务
        from services.basic import OpsDBInstanceService
        self.ops_service = OpsDBInstanceService()

    async def execute_readonly_ops_sql(
        self,
        instance_id: str,
        sql: str,
        limit: int = 50,
        params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        【轨道 A】执行只读内核指标探测 SQL（供 db-metric-skill 听诊器调用）。
        支持可选的 params 字典以进行参数化查询防注入。
        """
        result = await self._dispatch_to_ops_service(
            instance_id=instance_id,
            sql=sql,
            run_mode="read_only",
            limit=limit,
            params=params
        )
        return result.get("data", [])

    async def execute_mutation_ops_sql(
        self,
        instance_id: str,
        sql: str
    ) -> dict[str, Any]:
        """
        【轨道 B】执行高危控制面运维变更指令（供 kill-session-skill 等强杀/热刷自愈工具调用）
        """
        return await self._dispatch_to_ops_service(
            instance_id=instance_id,
            sql=sql,
            run_mode="mutation",
            limit=None
        )

    async def _dispatch_to_ops_service(
        self,
        instance_id: str,
        sql: str,
        run_mode: Literal["read_only", "mutation"],
        limit: int | None = None,
        params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        底层的 HTTP 调度网关, 负责装配物理凭证、拼装加密 Headers 并投递请求。
        """
        # 1. 从物理运维资产中心动态检索实例拓扑与认证元数据
        instance_meta = await self.ops_service.get_instance_by_id(instance_id)
        db_type = instance_meta["db_type"]
        connection_config = instance_meta["connection_config"]
        environment = instance_meta.get("environment", "prod")

        # 2. 构造专职运维端点的请求报文与安全 Headers
        url = f"http://{self.service_host}:{self.service_port}/api/v1/ops/execute"

        headers = {
            "X-KBot-Internal-Ops-Token": self.internal_token,
            "Content-Type": "application/json"
        }

        payload = {
            "instance_id": instance_id,
            "db_type": db_type,
            "sql": sql,
            "connection_config": connection_config,
            "environment": environment,
            "run_mode": run_mode,
            "limit": limit,
            "params": params
        }

        # 运维操作涉及超时终止等较重任务, 超时设为 45 秒
        timeout = aiohttp.ClientTimeout(total=45)

        # 3. 异步非阻塞投递至微服务运维专用引擎
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                logger.warning(f"[OpsClient] 发起物理内核调用 | 实例: {instance_id} | 模式: {run_mode} | 环境: {environment}")

                async with session.post(url, json=payload) as response:
                    if response.status == 403:
                        logger.critical("[OpsClient] 密钥被微服务拒绝! 请检查 INTERNAL_OPS_TOKEN 配置。")
                        return {"status": "error", "error_message": "运维通道鉴权失败, 拒绝访问。"}

                    if response.status != 200:
                        err_text = await response.text()
                        logger.error(f"[OpsClient] 物理网关响应硬错误: {response.status} | {err_text}")
                        return {"status": "error", "error_message": f"物理通道网关异常: {err_text}"}

                    res_json = await response.json()

                    # 4. 透传内核报错与异常
                    if res_json.get("status") == "error":
                        error_message = res_json.get("error_message", "未知物理内核错误")
                        logger.error(f"[OpsClient] 内核拒绝执行该运维指令: {error_message}")
                        return {
                            "status": "error",
                            "error_message": error_message,
                            "instance_id": instance_id
                        }

                    # 5. 执行成功
                    return res_json

        except aiohttp.ClientConnectorError:
            logger.error(f"连接数据库执行微服务失败, 物理链路断开 | URL: {url}")
            raise InternalServerError("集群网络故障: 无法触达物理数据库执行微服务")
        except Exception as e:
            logger.exception("[OpsClient] 调用运维通道时发生非预期内的系统崩溃")
            raise InternalServerError(f"运维微服务链路异常: {e}")
