from typing import Any, AsyncIterator
from contextlib import asynccontextmanager
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from .nacos_manager import load_config, DBConfig

class AsyncRedisPool:
    """
    异步 Redis 连接池工具类 (兼容 redis==5.0.1)
    
    示例:
        >>> async with AsyncRedisPool(db=1) as redis:
        >>>     await redis.hset("key", mapping={"field": "value"})
        >>>     data = await redis.hgetall("key")
    """

    def __init__(self, db: int = 0):
        self._db = db
        self._max_connections = 20
        self._pool: ConnectionPool | None = None
        self._redis: Redis | None = None
        self._is_initialized = False
        self._config_loaded = False

    async def _load_config(self) -> None:
        """从 Nacos 加载 Redis 配置"""
        if self._config_loaded:
            return
            
        try:
            db_config = load_config("db_config")
            
            if isinstance(db_config, DBConfig):
                self._host = db_config.redis.host
                self._port = db_config.redis.port
                self._password = db_config.redis.password
                self._max_connections = (
                    db_config.redis.max_connections or 
                    self._max_connections
                )
                self.socket_connect_timeout = db_config.redis.socket_connect_timeout or 3
                self.socket_timeout = db_config.redis.socket_timeout or 5
                self.retry_on_timeout = db_config.redis.retry_on_timeout or True
                self.health_check_interval = db_config.redis.health_check_interval or 30

            else:
                raise ValueError("Invalid database configuration")
                
            self._config_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Redis configuration loading failed: {e}") from e

    async def initialize(self) -> None:
        """初始化连接池"""
        if self._is_initialized:
            return
            
        await self._load_config()
        
        try:
            self._pool = ConnectionPool.from_url(
                f"redis://:{self._password}@{self._host}:{self._port}/{self._db}",
                max_connections=self._max_connections,
                decode_responses=True,
                socket_connect_timeout=self.socket_connect_timeout, # 连接建立超时（秒）
                socket_timeout=self.socket_timeout,  # 读写操作超时（秒）
                retry_on_timeout=self.retry_on_timeout,
                health_check_interval=self.health_check_interval
            )
            
            self._redis = Redis(connection_pool=self._pool)
            await self._redis.ping()
            self._is_initialized = True
            
        except (ConnectionError, TimeoutError) as e:
            await self.close()
            raise ConnectionError(f"无法连接到 Redis: {e}") from e
        except Exception as e:
            await self.close()
            raise

    async def close(self) -> None:
        """关闭所有连接"""
        if self._redis:
            await self._redis.close()

        if self._pool:
            await self._pool.disconnect()

        self._is_initialized = False
        self._redis = None
        self._pool = None

    async def __aenter__(self) -> "AsyncRedisPool":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[Redis]:
        """获取连接的上下文管理器"""
        await self.initialize()
        try:
            if self._redis is None:
                raise RuntimeError("Redis connection not available")
            yield self._redis
        except RedisError as e:
            await self.close()
            await self.initialize()
            raise
        except Exception as e:
            raise

    def __getattr__(self, name: str):
        """动态代理未定义的方法到 Redis 客户端"""
        async def method(*args, **kwargs):
            async with self.get_connection() as redis:
                if not hasattr(redis, name):
                    raise AttributeError(f"Redis method {name} not exists")
                return await getattr(redis, name)(*args, **kwargs)
        return method

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self.get_connection() as redis:
                return await redis.ping()
        except Exception:
            return False

    async def get_info(self) -> dict[str, Any]:
        """获取 Redis 服务器信息"""
        async with self.get_connection() as redis:
            return await redis.info()
