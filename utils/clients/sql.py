"""数据库执行微服务调用类。

集成数据库执行微服务的远程调用逻辑，支持动态数据库配置获取。
"""
import os
import aiohttp
from loguru import logger
from typing import Any

from core.config.settings import get_executor_config
from core.exceptions import *

class SQLClient:
    """数据库执行微服务调用类。"""

    def __init__(self):
        """初始化配置。"""
        # 统一从配置中心获取微服务访问地址
        self.executor_config = get_executor_config()
        self.service_host = self.executor_config.service_host
        self.service_port = self.executor_config.service_port

        # 延迟导入以避免循环依赖
        from services.kb import KBService
        # 初始化元数据仓库（用于取连接字符串）
        self.kb_service = KBService()

    @staticmethod
    def _auth_headers() -> dict[str, str]:
        """获取内部服务认证请求头。"""
        token = os.getenv("KBOT_INTERNAL_SERVICE_TOKEN", "kbot-internal-dev-token-2026")
        return {"X-KBot-Internal-Token": token}

    async def execute_sql(
        self, 
        kb_id: int, 
        sql: str, 
        limit: int = 100
    ) -> dict[str, Any]:
        """调用数据库执行微服务。

        Args:
            kb_id: 知识库ID，用于检索连接信息。
            sql: 待执行的 SQL 语句。
            limit: 结果集行数限制。

        Returns:
            dict[str, Any]: 执行结果，包含 status, data, row_count 等。
        """
        
        # 1. 根据 kb_id 获取数据库连接配置 (从项目元数据库获取)
        try:
            db_config = await self.kb_service.get_dbconf_of_kb(kb_id)
            db_type = db_config["db_type"]
            connection_config = db_config["connection_config"]

        except Exception as e:
            handle_exception(e, "获取数据库配置失败")

        # 2. 构造请求报文
        url = f"http://{self.service_host}:{self.service_port}/api/v1/execute"
        
        payload = {
            "db_type": db_type,
            "connection_config": connection_config,
            "sql": sql,
            "limit": limit,
            "kb_id": kb_id
        }

        # 超时设置 (通常 SQL 执行可能较慢，建议设长一点)
        timeout = aiohttp.ClientTimeout(total=60) 

        # 3. 发起请求
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info(f"发起 SQL 执行请求 | 目标库: {kb_id} | 类型: {db_config['db_type']}")

                async with session.post(url, json=payload, headers=self._auth_headers()) as response:
                    if response.status != 200:
                        err_text = await response.text()
                        logger.error(f"执行服务响应错误: {response.status} | {err_text}")
                        return {"status": "error", "error_message": f"微服务网络错误: {err_text}"}

                    res_json = await response.json()
                    
                    # 关键修改点：直接透传微服务的 status 和 error_message
                    if res_json.get("status") == "error":
                        error_message = res_json.get("error_message", "未知执行错误")
                        logger.warning(f"SQL 执行逻辑失败: {error_message}")
                        return {
                            "status": "error", 
                            "error_message": error_message,
                            "kb_id": kb_id
                        }

                    # 执行成功，返回数据
                    return {
                        "status": "success",
                        "data": res_json.get("data"),
                        "row_count": res_json.get("row_count"),
                        "kb_id": kb_id
                    }

        except aiohttp.ClientConnectorError:
            logger.error(f"连接执行微服务失败 | URL: {url}")
            raise InternalServerError("无法连接至数据库执行微服务")
        except Exception as e:
            logger.exception("调用执行微服务时发生未预期异常")
            raise InternalServerError(f"问数微服务调用异常: {e}")