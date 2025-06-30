from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import TypeDecorator
from typing import AsyncIterator
import asyncpg

async def create_pg_session(connection_string: str) -> AsyncIterator[AsyncSession]:
    """
    创建 Postgre SQL 数据库异步连接session
    :param connection_string: Postgre SQL连接字符串，格式为postgresql+asyncpg://user:password@host:port/database
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

class PgVectorType(TypeDecorator):
    """PostgreSQL VECTOR类型适配器"""
    
    impl = "vector"  # PostgreSQL的vector类型

    def __init__(self, dimensions=None):
        super().__init__()
        self.dimensions = dimensions

    def process_bind_param(self, value, dialect):
        """将Python list转为PostgreSQL vector格式"""
        if value is None:
            return None
        return value  # PostgreSQL可以直接处理Python list

    def process_result_value(self, value, dialect):
        """将PostgreSQL vector解析为Python list"""
        if value is None:
            return None
        return list(value)  # 转换为Python list