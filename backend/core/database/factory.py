from typing import Dict, Any
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from .vec_mysql import create_mysql_session
from .vec_oracle import create_oracle_session
from .vec_pg import create_pg_session

@asynccontextmanager
async def create_session(db_type: str, connection_info: Dict[str, Any]) -> AsyncIterator[AsyncSession]:
    """
    根据数据库类型和连接信息创建异步数据库session
    :param db_type: 数据库类型，支持oracle/mysql/pg
    :param connection_info: 连接信息字典，包含连接所需参数
    :return: SQLAlchemy AsyncSession对象
    """

    connection_string = _build_connection_string(db_type, connection_info)
    
    if db_type == "oracle":
        yield await create_oracle_session(connection_string)
    elif db_type == "mysql":
        yield await create_mysql_session(connection_string)
    elif db_type == "pg":
        yield await create_pg_session(connection_string)
    else:
        raise ValueError(f"不支持的数据库类型: {db_type}")

@staticmethod
def _build_connection_string(db_type: str, connection_info: Dict[str, Any]) -> str:
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

    if db_type == "oracle":
        return f"oracle+oracledb://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "mysql":
        return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "pg":
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    else:
        raise ValueError(f"不支持的数据库类型: {db_type}")