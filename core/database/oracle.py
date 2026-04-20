from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from sqlalchemy.dialects import oracle
from typing import AsyncIterator
from loguru import logger
from contextlib import asynccontextmanager
from core.config.settings import get_settings
from core.exceptions import DataNotFoundException
import json
from sqlalchemy import event
from decimal import Decimal

# Load database configuration from settings
db_config = get_settings()
username = db_config.oracle.username
password = db_config.oracle.password
host = db_config.oracle.host
port = db_config.oracle.port
service_name = db_config.oracle.service_name
url = f"oracle+oracledb://{username}:{password}@{host}:{port}/?service_name={service_name}"
echo = db_config.sqlalchemy.echo
pool_size = db_config.sqlalchemy.pool_size
pool_timeout = db_config.sqlalchemy.pool_timeout
max_overflow = db_config.sqlalchemy.max_overflow
pool_pre_ping = db_config.sqlalchemy.pool_pre_ping
pool_recycle = db_config.sqlalchemy.pool_recycle
pool_use_lifo = db_config.sqlalchemy.pool_use_lifo

# Create async database engine
try:
    async_engine = create_async_engine(
        url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        pool_recycle=pool_recycle,
        pool_timeout=pool_timeout,
        pool_use_lifo=pool_use_lifo,
        future=True
    )
    logger.info("Async database engine created successfully")
    # 【核心补丁】手动为异步 Dialect 注入缺失的序列化器
    def extended_json_dumps(obj, **kwargs):
        def default_encoder(item):
            if isinstance(item, Decimal):
                return float(item)  # 或者 str(item)，看你对精度的要求
            raise TypeError(f"Object of type {item.__class__.__name__} is not JSON serializable")
        
        # 确保 ensure_ascii=False 以支持中文
        kwargs.setdefault('ensure_ascii', False)
        return json.dumps(obj, default=default_encoder, **kwargs)

    def flexible_json_loads(value):
        if value is None:
            return None
        # 如果已经是字典或列表（驱动已经帮我们转好了），直接返回
        if isinstance(value, (dict, list)):
            return value
        # 如果是字符串或字节流，执行标准的 json.loads
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
    
    # 应用补丁
    @event.listens_for(async_engine.sync_engine, "connect")
    def register_json_processors(dbapi_connection, connection_record):
        dialect = async_engine.dialect
        # 强制覆盖序列化器，使其支持 Decimal
        dialect._json_serializer = extended_json_dumps  # type: ignore
        dialect._json_deserializer = flexible_json_loads  # type: ignore
        
        # 针对 oracledb 驱动的特殊设置：确保它能处理 python dict
        # 这样可以避免驱动尝试将 dict 识别为其他类型

except Exception as e:
    logger.error(f"Failed to create async database engine: {str(e)}")
    raise RuntimeError(f"Failed to create async database engine: {str(e)}") from e

async def close_engine() -> None:
    """Close database engine and release connection pool resources.
    
    Raises:
        RuntimeError: If engine disposal fails
    """
    try:
        await async_engine.dispose()
        logger.info("Async database engine closed successfully")
    except Exception as e:
        logger.error(f"Failed to close async database engine: {str(e)}")
        raise RuntimeError(f"Failed to close async database engine: {str(e)}") from e

# Create async session factory
async_session = async_sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession, autoflush=False
)

@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Async database session context manager with automatic transaction handling.
    
    Yields:
        AsyncSession: Async database session instance
        
    Raises:
        RuntimeError: If database operation fails (after rollback)
        
    Example:
        async with get_session() as session:
            result = await session.execute(text("SELECT * FROM table"))
            data = result.scalars().all()
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except DataNotFoundException as e:
            logger.warning("Data not found: {}", str(e))
        except Exception as e:
            # These are actual database/infrastructure errors
            await session.rollback()
            logger.error(
                "Database operation failed, rollback executed - "
                "error type: {}, "
                "error message: {}, "
                "error module: {}",
                type(e).__name__,
                repr(str(e)),
                type(e).__module__,
                exc_info=True
            )
            raise RuntimeError(f"Database operation failed: {str(e)}")
        finally:
            await session.close()

async def test_connection() -> bool:
    """Test database connection availability.
    
    Returns:
        bool: True if connection is successful, False otherwise
        
    Logs:
        Info: Connection success
        Error: Connection failure or unexpected result
    """
    try:
        async with get_session() as session:
            result = await session.execute(text("SELECT 1 FROM DUAL"))
            test_result = result.scalar()
            if test_result == 1:
                logger.info("Database connection test succeeded")
                return True
            else:
                logger.error("Database connection test failed: Unexpected result returned")
                return False
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False