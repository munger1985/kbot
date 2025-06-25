# -*- coding: utf-8 -*-

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from backend.core.config import load_config

# Declare the asynchronous base class
Base = declarative_base()


config = load_config()

# Default to environment variable connection strings when available
DB_CON_STRING = os.getenv("DB_CON_STRING")
DATABASE_URL = DB_CON_STRING if DB_CON_STRING else config["database"]["url"]

async_engine = create_async_engine(
    DATABASE_URL,
    echo=config["database"]["echo"],
    pool_size=config["database"]["pool_size"],
    max_overflow=config["database"]["max_overflow"],
    pool_pre_ping=config["database"]["pool_pre_ping"],
    pool_recycle=config["database"]["pool_recycle"],
    future=True  # Enable SQLAlchemy 2.0 features
)

async_session = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db() -> AsyncSession:
    """Get an async database session."""
    async with async_session() as session:
        yield session