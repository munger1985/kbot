from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from typing import AsyncIterator
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
        max_overflow = db_config.sqlalchemy.max_overflow
        pool_pre_ping = db_config.sqlalchemy.pool_pre_ping
        pool_recycle = db_config.sqlalchemy.pool_recycle
    else:
        # 如果获取 database 配置失败，则抛出异常
        raise ValueError
    
except Exception as e:
    # 如果获取 database 配置失败，则抛出异常
    raise RuntimeError(f"Failed to get database config from nacos: {str(e)}") from e

try:
    async_engine = create_async_engine(
        url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        pool_recycle=pool_recycle,
        future=True,  # Enable SQLAlchemy 2.0 features
    )
except Exception as e:
    raise RuntimeError(f"Failed to create database engine: {str(e)}") from e

async def close_engine() -> None:
    """Dispose the database engine and clean up resources."""
    await async_engine.dispose()

async_session = async_sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession
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
            raise RuntimeError(f"Database operation failed: {str(e)}") from e
        finally:
            await session.close()

