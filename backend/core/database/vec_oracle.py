from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import TypeDecorator
from typing import AsyncIterator
import oracledb
from contextlib import asynccontextmanager

oracledb.version = "8.3.0"
Base = declarative_base()

@asynccontextmanager
async def create_oracle_session(connection_string: str) -> AsyncIterator[AsyncSession]:
    """
    创建Oracle数据库异步连接session
    :param connection_string: Oracle连接字符串，格式为oracle+oracledb://user:password@host:port/service_name
    :return: SQLAlchemy AsyncSession对象
    """
    async_engine = create_async_engine(connection_string)
    async_session = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise RuntimeError(f"Database connection failed: {str(e)}") from e
        finally:
            await session.close()
