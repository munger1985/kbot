import oracledb
import asyncio
import threading
import hashlib
from loguru import logger
from contextlib import asynccontextmanager
from typing import Optional, Any
import time

class OracleConnParams:
    """
    Oracle数据库连接参数类。
    封装连接参数并提供验证功能。
    """
    def __init__(self, user: str, password: str, dsn: str, **kwargs):
        """
        初始化连接参数。
        :param user: 用户名
        :param password: 密码
        :param dsn: 数据源名称
        :param kwargs: 其他连接参数
        """
        self.user = user
        self.password = password
        self.dsn = dsn
        self.extra_params = kwargs

    def to_dict(self) -> dict[str, Any]:
        """将连接参数转换为字典格式"""
        base_params = {
            'user': self.user,
            'password': self.password,
            'dsn': self.dsn
        }
        base_params.update(self.extra_params)
        return base_params

    def validate(self) -> bool:
        """验证连接参数是否有效"""
        return all([self.user, self.password, self.dsn])

class AsyncOracleConnectionPoolManager:
    """
    异步Oracle数据库连接池管理器。
    针对Oracle Cloud Infrastructure (OCI) DBCS 优化。
    """
    _instance = None
    _lock = threading.Lock()
    _pools: dict[str, oracledb.ConnectionPool] = {}
    _pool_configs: dict[str, dict] = {}
    _pool_last_used: dict[str, float] = {}
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

    def _validate_connection(self, conn: oracledb.Connection) -> bool:
        """验证连接是否仍然有效"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
            cursor.fetchone()
            cursor.close()
            return True
        except (oracledb.Error, oracledb.InterfaceError):
            return False

    async def get_pool(self, conn_params: OracleConnParams) -> oracledb.ConnectionPool:
        """
        异步获取或创建连接池，增加连接验证和重连机制。
        """
        if not conn_params.validate():
            raise ValueError("Invalid connection parameters")

        pool_key = self._get_pool_key(conn_params)

        if pool_key in self._pools:
            # 检查连接池是否仍然有效
            pool = self._pools[pool_key]
            try:
                # 尝试获取一个连接来验证池的健康状态
                test_conn = await self._loop.run_in_executor(None, pool.acquire) # type: ignore
                
                # 如果连接无效，关闭池并重新创建
                if not self._validate_connection(test_conn):
                    logger.warning(f"连接池 {pool_key} 中的连接无效，重新创建...")
                    await self._loop.run_in_executor(None, pool.close) # type: ignore
                    with self._lock:
                        del self._pools[pool_key]
                        if pool_key in self._pool_configs:
                            del self._pool_configs[pool_key]
                else:
                    await self._loop.run_in_executor(None, pool.release, test_conn) # type: ignore
                    self._pool_last_used[pool_key] = time.time()
                    return pool
            except (oracledb.Error, oracledb.InterfaceError) as e:
                logger.warning(f"连接池 {pool_key} 验证失败: {e}，重新创建...")
                with self._lock:
                    if pool_key in self._pools:
                        del self._pools[pool_key]
                    if pool_key in self._pool_configs:
                        del self._pool_configs[pool_key]

        # 创建新的连接池
        try:
            # 正确的连接池参数（只使用oracledb.create_pool支持的参数）
            default_pool_params = {
                'min': 2,
                'max': 10,
                'increment': 1,
                'getmode': oracledb.POOL_GETMODE_WAIT,
                'max_lifetime_session': 1800,  # 30分钟，适应云环境
                'wait_timeout': 30,           # 等待连接超时30秒
                'max_sessions_per_shard': 1,
                'ping_interval': 30,          # 更频繁的健康检查
                'timeout': 30,                # 连接超时30秒
            }
            
            # 获取基础连接参数
            conn_params_dict = conn_params.to_dict()
            
            # 分离连接池参数和连接参数
            pool_params = {k: v for k, v in default_pool_params.items()}
            connection_params = {
                'user': conn_params_dict['user'],
                'password': conn_params_dict['password'],
                'dsn': conn_params_dict['dsn']
            }
            
            # 添加其他连接参数（如encoding等）
            for key, value in conn_params_dict.items():
                if key not in ['user', 'password', 'dsn'] and key not in pool_params:
                    connection_params[key] = value

            # 合并参数
            effective_params = {**pool_params, **connection_params}

            def create_pool_sync():
                return oracledb.create_pool(**effective_params)

            new_pool = await self._loop.run_in_executor(None, create_pool_sync) # type: ignore
            
            with self._lock:
                self._pools[pool_key] = new_pool
                self._pool_configs[pool_key] = effective_params
                self._pool_last_used[pool_key] = time.time()
                
            logger.info(f"创建新的连接池，Key: {pool_key}")
            return new_pool
            
        except oracledb.Error as e:
            logger.error(f"使用参数 {conn_params.dsn} 创建连接池失败: {e}")
            # 打印具体参数以便调试
            logger.debug(f"连接池参数: {effective_params}")
            raise
        except TypeError as e:
            logger.error(f"连接池参数错误: {e}")
            raise ValueError(f"无效的连接池参数: {e}")

    @asynccontextmanager
    async def get_connection_ctx(self, conn_params: OracleConnParams, retries: int = 3):
        """
        异步上下文管理器，支持重试机制。
        """
        for attempt in range(retries):
            try:
                pool = await self.get_pool(conn_params)
                conn = await self._loop.run_in_executor(None, pool.acquire) # type: ignore
                
                # 验证连接是否有效
                if not self._validate_connection(conn):
                    logger.warning(f"获取的连接无效，尝试 {attempt + 1}/{retries}")
                    await self._loop.run_in_executor(None, pool.release, conn) # type: ignore
                    if attempt == retries - 1:
                        raise oracledb.Error("无法获取有效的数据库连接")
                    await asyncio.sleep(1)  # 等待后重试
                    continue
                    
                try:
                    yield conn
                    break  # 成功执行，退出重试循环
                finally:
                    await self._loop.run_in_executor(None, pool.release, conn) # type: ignore
                    
            except (oracledb.Error, oracledb.InterfaceError) as e:
                if attempt == retries - 1:
                    logger.error(f"数据库连接失败，已达最大重试次数: {e}")
                    raise
                logger.warning(f"数据库连接失败，尝试 {attempt + 1}/{retries}: {e}")
                await asyncio.sleep(2 ** attempt)  # 指数退避

    async def execute_sql(
        self,
        conn_params: OracleConnParams,
        sql: str,
        params: dict | list | tuple | None = None,
        operation_type: str = "query",
        retries: int = 2
    ) -> list[tuple] | int: # type: ignore
        """
        执行SQL语句的通用异步方法，支持重试。
        """
        for attempt in range(retries):
            try:
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
                        if attempt == retries - 1:
                            logger.error(f"SQL执行错误: {e}, SQL: {sql}")
                            raise
                        logger.warning(f"SQL执行失败，重试 {attempt + 1}/{retries}: {e}")
                        await asyncio.sleep(1)
                    finally:
                        await self._loop.run_in_executor(None, cursor.close) # type: ignore
            except (oracledb.Error, oracledb.InterfaceError) as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def query(
        self,
        conn_params: OracleConnParams,
        sql: str,
        params: dict | list | tuple | None = None,
        retries: int = 2
    ) -> list[tuple]:
        """执行查询SQL（SELECT）"""
        return await self.execute_sql(conn_params, sql, params, "query", retries) # type: ignore

    async def execute_dml(
        self,
        conn_params: OracleConnParams,
        sql: str,
        params: dict | list | tuple | None = None,
        retries: int = 2
    ) -> int:
        """执行DML SQL（INSERT/UPDATE/DELETE）"""
        return await self.execute_sql(conn_params, sql, params, "dml", retries) # type: ignore
    
    async def close_all_pools(self):
        """异步关闭所有连接池。"""
        for key, pool in list(self._pools.items()):
            try:
                await self._loop.run_in_executor(None, pool.close) # type: ignore
                with self._lock:
                    del self._pools[key]
                    if key in self._pool_configs:
                        del self._pool_configs[key]
                    if key in self._pool_last_used:
                        del self._pool_last_used[key]
                logger.info(f"连接池 {key} 已关闭")
            except oracledb.Error as e:
                logger.error(f"关闭连接池 {key} 时出错: {e}")

    async def cleanup_idle_pools(self, idle_timeout: int = 3600):
        """清理空闲时间过长的连接池"""
        current_time = time.time()
        for key, last_used in list(self._pool_last_used.items()):
            if current_time - last_used > idle_timeout:
                logger.info(f"清理空闲连接池: {key}")
                await self.close_pool_by_key(key)

    async def close_pool_by_key(self, pool_key: str):
        """关闭指定键的连接池"""
        if pool_key in self._pools:
            try:
                pool = self._pools[pool_key]
                await self._loop.run_in_executor(None, pool.close) # type: ignore
                with self._lock:
                    del self._pools[pool_key]
                    if pool_key in self._pool_configs:
                        del self._pool_configs[pool_key]
                    if pool_key in self._pool_last_used:
                        del self._pool_last_used[pool_key]
                logger.info(f"连接池 {pool_key} 已关闭")
            except oracledb.Error as e:
                logger.error(f"关闭连接池 {pool_key} 时出错: {e}")

    def get_supported_pool_params(self):
        """获取支持的连接池参数列表"""
        import inspect
        sig = inspect.signature(oracledb.create_pool)
        return list(sig.parameters.keys())