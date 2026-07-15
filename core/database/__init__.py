# core/database/__init__.py — 数据库适配层

from contextlib import asynccontextmanager
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession
from .oracle import async_session


class _OracleSessionFactory:
    """兼容 NexusCube db_instance() 接口的 Oracle 适配器"""

    @staticmethod
    def get_session_maker():
        """返回 async_sessionmaker 实例"""
        return async_session

    @asynccontextmanager
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """获取一个受管理的数据库 session（自动 commit/rollback/close）"""
        session = async_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


_db_factory = _OracleSessionFactory()


def db_instance() -> _OracleSessionFactory:
    """获取数据库 session factory（兼容 NexusCube 接口签名）"""
    return _db_factory
