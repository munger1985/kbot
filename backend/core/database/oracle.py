# -*- coding: utf-8 -*-

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from backend.core.config import load_config

# 定义异步Base类
Base = declarative_base()


config = load_config()

# 优先使用环境变量中的连接字符串
DB_CON_STRING = os.getenv("DB_CON_STRING")
DATABASE_URL = DB_CON_STRING if DB_CON_STRING else config["database"]["url"]

async_engine = create_async_engine(
    DATABASE_URL,
    echo=config["database"]["echo"],
    pool_size=config["database"]["pool_size"],
    max_overflow=config["database"]["max_overflow"],
    pool_pre_ping=config["database"]["pool_pre_ping"],
    pool_recycle=config["database"]["pool_recycle"],
    future=True  # 启用SQLAlchemy 2.0特性
)

async_session = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db() -> AsyncSession:
    """Get an async database session."""
    async with async_session() as session:
        yield session