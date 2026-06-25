# microservices/db_executor/drivers/oracle_driver.py

import oracledb
import pandas as pd
from loguru import logger
from .base_driver import BaseDriver
from ..schemas.db_config import OracleConfig

class OracleDriver(BaseDriver):
    def __init__(self, config: OracleConfig):
        super().__init__(config)
        self.dsn = self.config.dsn or f"{self.config.host}:{self.config.port}/{self.config.service_name}"
        logger.info(
            f"[OracleDriver] DSN 构建完成: {self.dsn} "
            f"(raw_dsn={self.config.dsn!r}, host={self.config.host}, "
            f"port={self.config.port}, service_name={self.config.service_name})"
        )
        
    async def connect(self):
        """建立真正的异步、非阻塞 Oracle 物理内核连接"""
        if not self.connection:
            logger.debug(f"[OracleDriver] 正在建立异步 Thin 连接至: {self.dsn}")
            # 使用 oracledb.connect_async 开启原生异步模式
            self.connection = await oracledb.connect_async(
                user=self.config.user,
                password=self.config.password,
                dsn=self.dsn
            )

    # =====================================================================
    # 轨道 A：专门用于指标探测、明细查询（READ_ONLY - 纯异步版本）
    # =====================================================================
    async def execute_query(self, sql: str, params: tuple | dict | None = None) -> pd.DataFrame:
        """只负责带结果集的只读性能视图查询（全面抛弃阻塞的 pd.read_sql）"""
        if not self.connection:
            await self.connect()

        if not self.connection:
            raise Exception("数据库连接未初始化")

        logger.debug(f"[OracleDriver] 执行异步指标查询: {sql[:100]}...")

        # 异步连接必须配合使用异步上下文游标 async with ... cursor()
        async with self.connection.cursor() as cursor:
            try:
                if params:
                    await cursor.execute(sql, params)
                else:
                    await cursor.execute(sql)
                # 异步捞取所有行记录
                rows = await cursor.fetchall()
                
                if not rows:
                    return pd.DataFrame()
                
                # 显式提取游标中的列名，用于组装完美的 DataFrame
                columns = [col[0] for col in cursor.description] # type: ignore
                return pd.DataFrame(rows, columns=columns)
                
            except oracledb.Error as ora_err:
                logger.error(f"[OracleDriver] 指标查询失败: {str(ora_err)}")
                raise ora_err

    # =====================================================================
    # 轨道 B：全新注入，专供运维线高危变更使用（MUTATION - 纯异步版本）
    # =====================================================================
    async def execute_non_query(self, sql: str, params: tuple | dict | None = None) -> str:
        """
        专门处理没有结果集的 Oracle 运维变更动作。
        支持：ALTER SYSTEM KILL SESSION、ALTER SYSTEM FLUSH BUFFER_CACHE 等。
        """
        if not self.connection:
            await self.connect()
            
        if not self.connection:
            raise Exception("数据库连接未初始化")

        logger.warning(f"[OracleDriver] 接收到 Oracle 物理控制面变更指令: {sql}")
        
        async with self.connection.cursor() as cursor:
            try:
                # 执行变更管理指令
                await cursor.execute(sql)
                
                # 💡 极其重要：由于是运维变更命令（DDL / 管理控制语句），
                # 在 Oracle 中多数 DDL 会自动提交，但如果是通过 DML 进行的运维控制面调整（如改写某些元数据表），
                # 必须显式调用 commit 以防止锁库挂起。
                # 考虑到宁缺毋滥，如果非纯 DDL 语句，可由自愈引擎按需 commit。
                
                status_msg = f"Oracle 内核成功受影响。受影响行数/状态: {cursor.rowcount}"
                logger.success(f"⚙️ [OracleDriver] 运维变更成功投递。反馈: {status_msg}")
                return status_msg
                
            except oracledb.DatabaseError as ora_err:
                # oracledb.DatabaseError 包含了特殊的 code（如 ORA-00031: session marked for kill）
                error_obj = ora_err.args[0] if ora_err.args else None
                err_code = getattr(error_obj, "code", "未知")
                err_msg = getattr(error_obj, "message", str(ora_err))
                
                logger.error(f"[OracleDriver] Oracle 内核拒绝执行该运维指令 | 错误码: ORA-{err_code} | 原因: {err_msg}")
                raise ora_err

    async def close(self):
        if self.connection:
            # 异步连接关闭必须使用 await self.connection.close()
            await self.connection.close()
            self.connection = None
            logger.debug("[OracleDriver] 异步连接已安全释放")