
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text
from typing import AsyncIterator
from contextlib import asynccontextmanager


from backend.core.config import settings

# Declare the asynchronous base class
Base = declarative_base()

# Default to environment variable connection strings when available
url = settings["database"]["url"]
echo = settings["database"]["echo"]
pool_size = settings["database"]["pool_size"]
max_overflow = settings["database"]["max_overflow"]
pool_pre_ping = settings["database"]["pool_pre_ping"]
pool_recycle = settings["database"]["pool_recycle"]

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



async def test_connection() -> bool:
    """Test the database connection by executing a simple query.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        async with get_session() as session:
            # 使用与数据库类型无关的通用测试语句
            await session.execute(text("SELECT 1"))
            await session.commit()  # 确保测试查询被提交
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False