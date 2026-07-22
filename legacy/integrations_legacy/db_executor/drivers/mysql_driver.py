# microservices/db_executor/drivers/mysql_driver.py

import aiomysql
import pandas as pd
from loguru import logger
from .base_driver import BaseDriver
from ..schemas.db_config import MySQLConfig

class MySqlDriver(BaseDriver):
    def __init__(self, config: MySQLConfig):
        super().__init__(config)
        # 提取连接参数，转为 aiomysql 识别的小写规范
        self.conn_params = {
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user,
            "password": self.config.password,
            "db": self.config.database,
            "charset": self.config.charset or "utf8mb4",
            "autocommit": True, # 运维线建议变更高危指令即时落盘，防止隐式事务锁表
            "connect_timeout": 10
        }

    async def connect(self):
        """建立真正的异步、非阻塞 MySQL 物理连接"""
        if not self.connection:
            logger.debug(f"[MySqlDriver] 正在建立异步物理连接至: {self.config.host}:{self.config.port}")
            # 使用 aiomysql.connect 开启原生异步模式
            self.connection = await aiomysql.connect(**self.conn_params)

    # =====================================================================
    # 轨道 A：专门用于指标探测、明细查询（READ_ONLY - 纯异步版）
    # =====================================================================
    async def execute_query(self, sql: str, params: tuple | dict | None = None) -> pd.DataFrame:
        """负责带结果集的只读查询（彻底告别同步阻塞的 pd.read_sql）"""
        if not self.connection:
            await self.connect()

        if not self.connection:
            raise Exception("数据库连接未初始化")

        logger.debug(f"[MySqlDriver] 执行异步查询: {sql[:100]}...")
        
        # 异步连接配合异步 DictCursor，直接返回带列名的字典结构，极大方便转为 DataFrame
        async with self.connection.cursor(aiomysql.DictCursor) as cursor:
            try:
                await cursor.execute(sql)
                rows = await cursor.fetchall()
                
                if not rows:
                    return pd.DataFrame()
                
                # 由于 DictCursor 返回的就是 list[dict]，直接喂给 DataFrame
                return pd.DataFrame(rows)
                
            except aiomysql.Error as mysql_err:
                logger.error(f"[MySqlDriver] 查询失败: {str(mysql_err)}")
                raise mysql_err

    # =====================================================================
    # 轨道 B：全新注入，专供运维线高危变更使用（MUTATION - 纯异步版）
    # =====================================================================
    async def execute_non_query(self, sql: str, params: tuple | dict | None = None) -> str:
        """
        专门处理没有结果集的 MySQL 运维变更动作。
        支持：KILL CONNECTION <pid>、FLUSH PRIVILEGES、SET GLOBAL ... 等管理指令。
        """
        if not self.connection:
            await self.connect()
            
        if not self.connection:
            raise Exception("数据库连接未初始化")

        logger.warning(f"[MySqlDriver] 接收到 MySQL 物理控制面变更指令: {sql}")
        
        async with self.connection.cursor() as cursor:
            try:
                # 执行变更管理指令
                await cursor.execute(sql)
                
                status_msg = f"MySQL 内核成功执行。受影响行数: {cursor.rowcount}"
                logger.success(f"[MySqlDriver] 运维变更成功投递。内核反馈: {status_msg}")
                return status_msg
                
            except aiomysql.Error as mysql_err: # 🎯 捕获标准的 aiomysql 内核级异常
                # mysql_err.args 通常包含 (error_code, error_message)
                err_code = mysql_err.args[0] if len(mysql_err.args) > 0 else "未知"
                err_msg = mysql_err.args[1] if len(mysql_err.args) > 1 else str(mysql_err)
                
                logger.error(f"[MySqlDriver] MySQL 内核拒绝执行该运维指令 | 错误码: {err_code} | 原因: {err_msg}")
                raise mysql_err

    async def close(self):
        if self.connection:
            # 异步连接关闭必须使用 await self.connection.ensure_closed() 或 close()
            self.connection.close()
            await self.connection.ensure_closed()
            self.connection = None
            logger.debug("[MySqlDriver] 异步连接已安全释放")