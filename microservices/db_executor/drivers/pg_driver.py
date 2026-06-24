import asyncpg
import pandas as pd
from loguru import logger
from .base_driver import BaseDriver
from ..schemas.db_config import PGConfig

class PgDriver(BaseDriver):
    def __init__(self, config: PGConfig):
        super().__init__(config)
        self.conn_params = {
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user,
            "password": self.config.password,
            "database": self.config.database,
            "timeout": 10
        }

    async def connect(self):
        if not self.connection:
            self.connection = await asyncpg.connect(**self.conn_params)

    # =====================================================================
    # 轨道 A：专门用于指标探测、明细查询（READ_ONLY）
    # =====================================================================
    async def execute_query(self, sql: str, params: tuple | dict | None = None) -> pd.DataFrame:
        """只负责带有结果集(SELECT)的性能指标、元数据查询。支持参数化查询防注入。"""
        if not self.connection:
            await self.connect()

        if not self.connection:
            raise Exception("数据库连接未初始化")

        logger.debug(f"[PgDriver] 执行指标探测查询: {sql[:100]}...")
        if params:
            rows = await self.connection.fetch(sql, *params) if isinstance(params, tuple) else await self.connection.fetch(sql, params)
        else:
            rows = await self.connection.fetch(sql)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([dict(r) for r in rows])

    # =====================================================================
    # 轨道 B：全新注入，专供运维线高危变更使用（MUTATION）
    # =====================================================================
    async def execute_non_query(self, sql: str, params: tuple | dict | None = None) -> str:
        """
        专门处理没有结果集的运维变更动作（如 KILL 连接、配置热刷、表空间管理等）。
        返回 PostgreSQL 协议底层的 Command Status 字符串。
        """
        if not self.connection:
            await self.connect()

        if not self.connection:
            raise Exception("数据库连接未初始化")

        logger.warning(f"[PgDriver] 接收到物理控制面变更指令: {sql}")

        try:
            # 使用 connection.execute() 执行管理命令
            if params:
                status_text = await self.connection.execute(sql, *params) if isinstance(params, tuple) else await self.connection.execute(sql, params)
            else:
                status_text = await self.connection.execute(sql)
            logger.success(f"[PgDriver] 运维变更成功落盘。底层内核反馈: {status_text}")
            return status_text

        except asyncpg.PostgresError as pg_err:
            logger.error(f"[PgDriver] PG内核拒绝执行该运维指令 | 错误码: {getattr(pg_err, 'sqlstate', '未知')} | 原因: {str(pg_err)}")
            raise pg_err
        
    async def close(self):
        if self.connection:
            await self.connection.close()
            self.connection = None