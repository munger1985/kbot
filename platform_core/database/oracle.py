"""App-owned Oracle runtime primitives.

Each deployable App creates one :class:`DatabaseRuntime` during startup and
injects its ``session_factory`` into repositories/UoWs.  The module deliberately
does not create a process-global engine or expose a global session helper.
"""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from platform_core.config.settings import Settings, get_settings


@dataclass(frozen=True)
class DatabaseRuntime:
    """Database resources owned by one deployable App process."""

    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]

    async def close(self) -> None:
        """Dispose the App's connection pool during shutdown."""
        await self.engine.dispose()


def _database_url(settings: Settings) -> str:
    oracle = settings.database.oracle
    return (
        f"oracle+oracledb://{oracle.username}:{oracle.require_password()}"
        f"@{oracle.host}:{oracle.port}/?service_name={oracle.service_name}"
    )


def create_database_runtime(settings: Settings | None = None) -> DatabaseRuntime:
    """Create an isolated engine and Session Factory for one App.

    The database section is the only deployment-specific input.  Repositories
    and UoWs remain unchanged if a future deployment points the App at a
    different database or schema.
    """
    config = settings or get_settings()
    sqlalchemy_config = config.database.sqlalchemy
    engine = create_async_engine(
        _database_url(config),
        echo=sqlalchemy_config.echo,
        pool_size=sqlalchemy_config.pool_size,
        max_overflow=sqlalchemy_config.max_overflow,
        pool_pre_ping=sqlalchemy_config.pool_pre_ping,
        pool_recycle=sqlalchemy_config.pool_recycle,
        pool_timeout=sqlalchemy_config.pool_timeout,
        pool_use_lifo=sqlalchemy_config.pool_use_lifo,
        future=True,
    )
    return DatabaseRuntime(
        engine=engine,
        session_factory=async_sessionmaker(
            bind=engine,
            expire_on_commit=False,
            class_=AsyncSession,
            autoflush=False,
        ),
    )
