import asyncio
from typing import Any
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError
from core.config import settings

class AsyncRedisPool:
    """
    异步 Redis 连接池工具类
    
    示例:
        >>> redis_pool = AsyncRedisPool("redis://localhost")
        >>> await redis_pool.set("key", "value")
        >>> await redis_pool.get("key")
    """

    def __init__(self):
        """
        初始化连接池
        
        :param url: Redis连接URL (e.g. redis://:password@host:port/db)
        :param max_connections: 最大连接数
        """
        self._url = settings["redis"]["url"]
        self._pool: ConnectionPool | None = None
        self._redis: Redis | None = None
        self._max_connections = settings["redis"]["max_connections"] or 20
        self._is_initialized = False

    async def initialize(self) -> None:
        """初始化连接池"""
        if not self._is_initialized:
            self._pool = ConnectionPool.from_url(
                self._url,
                max_connections=self._max_connections,
                decode_responses=True  # 自动将bytes解码为字符串
            )
            self._redis = Redis(
                connection_pool=self._pool,
                decode_responses=True  # 双重保障确保解码
            )
            self._is_initialized = True
            # 测试连接
            try:
                await self._redis.ping()
            except RedisError as e:
                await self.close()
                raise ConnectionError(f"Redis连接失败: {str(e)}")

    async def close(self) -> None:
        """关闭所有连接"""
        if self._redis is not None:
            await self._redis.close()
        if self._pool is not None:
            await self._pool.disconnect()
        self._is_initialized = False

    async def __aenter__(self) -> "AsyncRedisPool":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def execute_command(self, *args, **kwargs) -> Any:
        """执行原始Redis命令"""
        if not self._is_initialized:
            await self.initialize()
        try:
            return await self._redis.execute_command(*args, **kwargs) # type: ignore
        except RedisError as e:
            # 自动重试一次
            try:
                await self.close()
                await self.initialize()
                return await self._redis.execute_command(*args, **kwargs) # type: ignore
            except Exception as retry_e:
                raise RedisError(f"操作失败: {str(retry_e)}") from retry_e

    # 以下是常用方法封装
    async def get(self, key: str) -> str | None:
        """获取字符串值"""
        return await self.execute_command("GET", key)

    async def set(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        设置键值
        :param ex: 过期时间(秒)
        :param px: 过期时间(毫秒)
        :param nx: 仅当key不存在时设置
        :param xx: 仅当key存在时设置
        :return: 是否设置成功
        """
        args = ["SET", key, value]
        if ex is not None:
            args.extend(["EX", ex])
        if px is not None:
            args.extend(["PX", px])
        if nx:
            args.append("NX")
        if xx:
            args.append("XX")
        return await self.execute_command(*args)

    async def hgetall(self, key: str) -> dict[str, str]:
        """获取哈希表所有字段"""
        return await self.execute_command("HGETALL", key)

    async def delete(self, *keys: str) -> int:
        """删除键(支持批量)"""
        return await self.execute_command("DEL", *keys)

    async def exists(self, *keys: str) -> int:
        """检查键是否存在"""
        return await self.execute_command("EXISTS", *keys)

    async def pipeline(self):
        """获取pipeline对象"""
        if not self._is_initialized:
            await self.initialize()
        return self._redis.pipeline() # type: ignore

    # 使用示例
    # async with AsyncRedisPool() as redis:
    #     await redis.set("test_key", "value", ex=60)
    #     value = await redis.get("test_key")
    #     print(f"获取值: {value}")