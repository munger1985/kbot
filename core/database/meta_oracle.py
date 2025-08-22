import os
import oracledb
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from typing import AsyncIterator
from loguru import logger
from contextlib import asynccontextmanager
from core.nacos_manager import load_config, DBConfig



# 通过 nacos_manager 获取 database 配置
try:
    db_config = load_config("db_config")
    if isinstance(db_config, DBConfig):
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
    else:
        # 如果获取 database 配置失败，则抛出异常
        raise ValueError
    
except Exception as e:
    # 如果获取 database 配置失败，则抛出异常
    logger.error(f"Failed to get database config from nacos: {str(e)}")
    raise RuntimeError(f"Failed to get database config from nacos: {str(e)}") from e

try:
    # 检查常见的 Oracle Instant Client 路径
    possible_lib_dirs = [
        "/usr/lib/oracle/21/client64/lib",
        "/usr/lib/oracle/19/client64/lib", 
        "/usr/lib/oracle/18/client64/lib",
        "/opt/oracle/instantclient",
        "/opt/oracle/instantclient_21_10",
        "/opt/oracle/instantclient_19_20"
    ]
    
    lib_dir = None
    for possible_dir in possible_lib_dirs:
        if os.path.exists(possible_dir) and os.path.isdir(possible_dir):
            lib_dir = possible_dir
            break
    
    if lib_dir:
        oracledb.init_oracle_client(lib_dir=lib_dir)
        logger.info(f"Oracle Thick mode initialized with lib_dir: {lib_dir}")
    else:
        # 尝试自动查找（如果 Instant Client 在标准路径）
        oracledb.init_oracle_client()
        logger.info("Oracle Thick mode initialized with auto-detection")
        
except Exception as e:
    logger.warning(f"Failed to initialize Oracle Thick mode: {str(e)}")
    logger.warning("Falling back to Thin mode. Some features may be limited.")
    # 继续使用 Thin 模式，但某些功能可能不可用

try:
    async_engine = create_async_engine(
        url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        pool_recycle=pool_recycle,
        pool_timeout=pool_timeout,
        future=True,  # Enable SQLAlchemy 2.0 features
    )
    logger.info("Async database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {str(e)}")
    raise RuntimeError(f"Failed to create database engine: {str(e)}") from e

async def close_engine() -> None:
    """Dispose the database engine and clean up resources."""
    try:
        await async_engine.dispose()
        logger.info("Database engine disposed successfully")
    except Exception as e:
        logger.error(f"Error disposing database engine: {str(e)}")
        raise

async_session = async_sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession, autoflush=False
)

@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Asynchronous context manager for database sessions with automatic transaction handling.
    
    Yields:
        AsyncSession: An async database session
        
    Raises:
        Exception: Any database operation errors will be raised after rollback
        
    Example:
        async with get_session() as session:
            result = await session.execute(query)
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database operation failed, rolled back: {str(e)}")
            raise RuntimeError(f"Database operation failed: {str(e)}") from e
        finally:
            await session.close()


async def test_connection() -> bool:
    """Test database connection."""
    try:
        async with get_session() as session:
            result = await session.execute(text("SELECT 1 FROM DUAL"))
            test_result = result.scalar()
            if test_result == 1:
                logger.info("Database connection test successful")
                return True
            else:
                logger.error("Database connection test failed: unexpected result")
                return False
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False