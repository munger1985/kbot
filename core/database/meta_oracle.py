from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from typing import AsyncIterator
from loguru import logger
from contextlib import asynccontextmanager
from configuration import ConfigManager


# 通过 nacos_manager 获取 database 配置
try:
    db_config = ConfigManager.get_db_config()
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
    max_identifier_length = db_config.sqlalchemy.max_identifier_length
    hide_parameters = db_config.sqlalchemy.hide_parameters
    echo_pool = db_config.sqlalchemy.echo_pool
    
except Exception as e:
    # 如果获取 database 配置失败，则抛出异常
    logger.error(f"无法从 nacos 获取 database 配置: {str(e)}")
    raise RuntimeError(f"无法从 nacos 获取 database 配置: {str(e)}") from e


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
        pool_use_lifo=pool_use_lifo,
        max_identifier_length=max_identifier_length,
        hide_parameters=hide_parameters,
        echo_pool=echo_pool
    )
    logger.info("成功创建数据库引擎")
except Exception as e:
    logger.error(f"创建数据库引擎失败: {str(e)}")
    raise RuntimeError(f"创建数据库引擎失败: {str(e)}") from e

async def close_engine() -> None:
    """关闭数据库引擎并释放连接池资源。"""
    try:
        await async_engine.dispose()
        logger.info("数据库引擎已关闭")
    except Exception as e:
        logger.error(f"数据库引擎关闭失败: {str(e)}")
        raise

async_session = async_sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession, autoflush=False
)

@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """异步数据库会话上下文管理器，支持自动事务处理。
    
    Yields:
        AsyncSession: 异步数据库会话
        
    Raises:
        Exception: 数据库操作错误将在回滚后抛出
        
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
            logger.error(f"数据库操作失败，已执行回滚: {str(e)}")
            raise RuntimeError(f"数据库操作失败: {str(e)}") from e
        finally:
            await session.close()


async def test_connection() -> bool:
    """测试数据库连接"""
    try:
        async with get_session() as session:
            result = await session.execute(text("SELECT 1 FROM DUAL"))
            test_result = result.scalar()
            if test_result == 1:
                logger.info("数据库连接测试成功")
                return True
            else:
                logger.error("数据库连接测试失败：返回意外结果")
                return False
    except Exception as e:
        logger.error(f"数据库连接测试失败: {str(e)}")
        return False