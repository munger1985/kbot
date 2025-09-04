import oracledb
import asyncio
import threading
import hashlib
from loguru import logger
from contextlib import asynccontextmanager


class OracleConnParams:
    """
    Oracle数据库连接参数类。
    封装连接参数并提供验证功能。
    """
    def __init__(self, user: str, password: str, dsn: str):
        """
        初始化连接参数。
        :param user: 用户名
        :param password: 密码
        :param dsn: 数据源名称
        """
        self.user = user
        self.password = password
        self.dsn = dsn

    def to_dict(self) -> dict[str, str]:
        """将连接参数转换为字典格式"""
        return {
            'user': self.user,
            'password': self.password,
            'dsn': self.dsn
        }

    def validate(self) -> bool:
        """验证连接参数是否有效"""
        return all([self.user, self.password, self.dsn])

class AsyncOracleConnectionPoolManager:
    """
    异步Oracle数据库连接池管理器。
    根据不同的连接参数创建和管理多个连接池，支持异步操作。
    线程安全，单例模式。
    """
    _instance = None
    _lock = threading.Lock()
    _pools: dict[str, oracledb.ConnectionPool] = {}
    _pool_configs: dict[str, dict] = {}
    _loop: asyncio.AbstractEventLoop | None = None

    def __new__(cls):
        """实现管理器本身的单例，确保全局一个管理器实例。"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AsyncOracleConnectionPoolManager, cls).__new__(cls)
                cls._instance._loop = asyncio.get_event_loop()
            return cls._instance

    def _get_pool_key(self, conn_params: OracleConnParams) -> str:
        """根据连接参数生成唯一的键。"""
        param_str = f"{conn_params.user}@{conn_params.dsn}".encode('utf-8')
        return hashlib.md5(param_str).hexdigest()

    async def get_pool(self, conn_params: OracleConnParams) -> oracledb.ConnectionPool:
        """
        异步获取或创建连接池。
        """
        if not conn_params.validate():
            raise ValueError("Invalid connection parameters")

        pool_key = self._get_pool_key(conn_params)

        if pool_key not in self._pools:
            try:
                # 设置推荐的默认池参数
                default_pool_params = {
                    'min': 2,
                    'max': 10,
                    'increment': 2,
                    'getmode': oracledb.POOL_GETMODE_WAIT,
                    'max_lifetime_session': 3600,  # 连接最大存活时间1小时
                    'retry_delay': 3,
                    'retry_count': 100,
                    'ping_interval': 60,
                    'ping_timeout': 5000
                }
                # 合并用户参数和默认参数
                effective_params = {**default_pool_params, **conn_params.to_dict()}

                # 在线程池中执行同步的创建连接池操作
                def create_pool_sync():
                    return oracledb.create_pool(**effective_params)

                new_pool = await self._loop.run_in_executor(None, create_pool_sync) # type: ignore
                
                with self._lock:
                    self._pools[pool_key] = new_pool
                    self._pool_configs[pool_key] = effective_params
                    logger.info(f"为新参数创建了连接池，Key: {pool_key}")
                    
            except oracledb.Error as e:
                logger.error(f"使用参数 {conn_params.dsn} 创建连接池失败: {e}")
                raise
        else:
            logger.debug(f"找到现有连接池，Key: {pool_key}")

        return self._pools[pool_key]

    @asynccontextmanager
    async def get_connection_ctx(self, conn_params: OracleConnParams):
        """
        异步上下文管理器，用于自动获取和释放连接。
        """
        pool = await self.get_pool(conn_params)
        conn = None
        try:
            conn = await self._loop.run_in_executor(None, pool.acquire) # type: ignore
            yield conn
        finally:
            if conn:
                await self._loop.run_in_executor(None, pool.release, conn) # type: ignore

    async def execute_sql(
        self,
        conn_params: OracleConnParams,
        sql: str,
        params: dict | list | tuple | None = None,
        operation_type: str = "query"
    ) -> list[tuple] | int:
        """
        执行SQL语句的通用异步方法。
        """
        async with self.get_connection_ctx(conn_params) as conn:
            cursor = conn.cursor()
            try:
                if params is None:
                    await self._loop.run_in_executor(None, cursor.execute, sql) # type: ignore
                else:
                    await self._loop.run_in_executor(None, cursor.execute, sql, params) # type: ignore
                
                if operation_type == "query":
                    result = await self._loop.run_in_executor(None, cursor.fetchall) # type: ignore
                    return result
                else:
                    rowcount = cursor.rowcount
                    await self._loop.run_in_executor(None, conn.commit) # type: ignore
                    return rowcount
                    
            except oracledb.Error as e:
                await self._loop.run_in_executor(None, conn.rollback) # type: ignore
                logger.error(f"SQL执行错误: {e}, SQL: {sql}")
                raise
            finally:
                await self._loop.run_in_executor(None, cursor.close) # type: ignore

    async def query(
        self,
        conn_params: OracleConnParams,
        sql: str,
        params: dict | list | tuple | None = None
    ) -> list[tuple]:
        """执行查询SQL（SELECT）"""
        return await self.execute_sql(conn_params, sql, params, "query") # type: ignore

    async def execute_dml(
        self,
        conn_params: OracleConnParams,
        sql: str,
        params: dict | list | tuple | None = None
    ) -> int:
        """执行DML SQL（INSERT/UPDATE/DELETE）"""
        return await self.execute_sql(conn_params, sql, params, "dml") # type: ignore
    
    async def close_all_pools(self):
        """异步关闭所有连接池。"""
        for key, pool in list(self._pools.items()):
            try:
                await self._loop.run_in_executor(None, pool.close) # type: ignore
                with self._lock:
                    del self._pools[key]
                logger.info(f"连接池 {key} 已关闭")
            except oracledb.Error as e:
                logger.error(f"关闭连接池 {key} 时出错: {e}")

