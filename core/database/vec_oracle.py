from typing import Any, AsyncIterator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from contextlib import asynccontextmanager
from core.dictionary import DbType

@asynccontextmanager
async def create_session(db_type: int, connection_info: dict[str, Any]) -> AsyncIterator[AsyncSession]:
    """
    根据数据库类型和连接信息创建异步数据库session
    :param db_type: 数据库类型，支持oracle/mysql/pg
    :param connection_info: 连接信息字典，包含连接所需参数
    :return: SQLAlchemy AsyncSession对象
    """

    connection_string = _build_connection_string(db_type, connection_info)
    
    if db_type == DbType.ORACLE.value:
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

    else:
        raise ValueError(f"不支持的数据库类型: {db_type}")

@staticmethod
def _build_connection_string(db_type: int, connection_info: dict[str, Any]) -> str:
    """
    构建数据库连接字符串
    :param db_type: 数据库类型
    :param connection_info: 连接信息字典
    :return: 连接字符串
    """
    user = connection_info.get("user")
    password = connection_info.get("password")
    host = connection_info.get("host")
    port = connection_info.get("port")
    database = connection_info.get("database") or connection_info.get("service_name")

    if not all([user, password, host, port, database]):
        raise ValueError("缺少必要的连接参数")

    if db_type == DbType.ORACLE.value:
        return f"oracle+oracledb://{user}:{password}@{host}:{port}/?service_name={database}"
    else:
        raise ValueError(f"不支持的数据库类型: {db_type}")
    
    