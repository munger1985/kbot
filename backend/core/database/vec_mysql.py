from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import TypeDecorator
from typing import AsyncIterator
import json

class MySQLVectorType(TypeDecorator):
    """MySQL VECTOR类型适配器(使用JSON存储)"""
    
    impl = "JSON"  # MySQL使用JSON存储向量数据

    def __init__(self, dimensions=None):
        super().__init__()
        self.dimensions = dimensions

    def process_bind_param(self, value, dialect):
        """将Python list转为JSON格式"""
        if value is None:
            return None
        return json.dumps(value)  # 序列化为JSON字符串

    def process_result_value(self, value, dialect):
        """将JSON解析为Python list"""
        if value is None:
            return None
        return json.loads(value)  # 反序列化为Python list

async def create_mysql_session(connection_string: str) -> AsyncIterator[AsyncSession]:
    """
    创建 MySQL 数据库异步连接session
    :param connection_string: MySQL连接字符串，格式为mysql+mysqlconnector://user:password@host:port/database
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