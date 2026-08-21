"""将已编译查询安全绑定到受治理的数据源。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import ssl
from uuid import UUID

import asyncpg
import aiomysql
import oracledb

from data_query.adapters.credential_cipher import DatabaseCredentialService
from data_query.connectors.postgresql import (
    CompiledPostgreSQLQuery, NormalizedQueryResult, PostgreSQLExecutionLimits,
    PostgreSQLReadOnlyExecutor,
    normalize_rows,
)
from data_query.contracts import DataSourceEndpoint
from data_query.persistence import DataQueryUnitOfWork


class DataSourceExecutorResolver:
    """在执行时重读数据源状态，杜绝 Run 冻结后绕过停用或凭据轮换。"""

    def __init__(self, *, uow_factory: Callable[[], DataQueryUnitOfWork], credential_service: DatabaseCredentialService) -> None:
        self._uow_factory = uow_factory
        self._credential_service = credential_service

    async def execute(
        self, *, connector_type: str, data_source_id: UUID,
        policy_budget: dict[str, object], compiled: CompiledPostgreSQLQuery,
    ) -> NormalizedQueryResult:
        async with self._uow_factory() as uow:
            assert uow.data_sources
            source = await uow.data_sources.get_by_id(data_source_id=data_source_id)
            await uow.commit()
        if source is None or source.status != "ACTIVE":
            raise ValueError("DATA_SOURCE_NOT_ACTIVE")
        endpoint = DataSourceEndpoint.model_validate(source.configuration_json)
        username, password = await self._credential_service.read_database_credentials(
            credential_id=source.credential_id,
            domain_id=int(source.domain_id),
            data_source_id=source.data_source_id,
        )
        if connector_type == "MYSQL":
            return await self._execute_mysql(endpoint, username, password, policy_budget, compiled)
        if connector_type == "ORACLE":
            return await self._execute_oracle(endpoint, username, password, policy_budget, compiled)
        if connector_type != "POSTGRESQL":
            raise ValueError("CONNECTOR_NOT_SUPPORTED")

        @asynccontextmanager
        async def connection_factory() -> AsyncIterator[asyncpg.Connection]:
            connection = await asyncpg.connect(
                host=endpoint.host, port=endpoint.port, database=endpoint.database,
                user=username, password=password,
                ssl="require" if endpoint.tls_enabled else False, timeout=15,
            )
            try:
                yield connection
            finally:
                await connection.close()

        def integer(name: str, fallback: int) -> int:
            value = policy_budget.get(name, fallback)
            return value if isinstance(value, int) else fallback

        return await PostgreSQLReadOnlyExecutor(
            connection_factory=connection_factory,
            limits=PostgreSQLExecutionLimits(
                statement_timeout_seconds=integer("statement_timeout_seconds", 30),
                lock_timeout_seconds=min(integer("statement_timeout_seconds", 30), 10),
                max_rows=integer("max_rows", 1000),
                max_result_bytes=integer("max_result_bytes", 1_048_576),
                search_path=endpoint.allowed_schemas,
            ),
        ).execute(compiled)

    @staticmethod
    async def _execute_mysql(endpoint, username, password, budget, compiled) -> NormalizedQueryResult:
        connection = None
        try:
            connection = await aiomysql.connect(host=endpoint.host, port=endpoint.port, db=endpoint.database, user=username, password=password, autocommit=False, connect_timeout=15, ssl=ssl.create_default_context() if endpoint.tls_enabled else None)
            async with connection.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("START TRANSACTION READ ONLY")
                await cursor.execute("SET SESSION MAX_EXECUTION_TIME=%s", (int(budget.get("statement_timeout_seconds", 30)) * 1000,))
                await cursor.execute(compiled.sql, compiled.parameters)
                rows = await cursor.fetchall()
            await connection.rollback()
        except (aiomysql.Error, OSError, TimeoutError) as exc:
            code = int(exc.args[0]) if getattr(exc, "args", ()) and isinstance(exc.args[0], int) else 0
            raise ValueError("DATA_SOURCE_AUTHENTICATION_FAILED" if code in {1044, 1045} else "DATA_SOURCE_CONNECTION_TIMEOUT" if isinstance(exc, TimeoutError) else "DATA_SOURCE_CONNECTION_FAILED") from exc
        finally:
            if connection is not None: connection.close()
        return normalize_rows(rows=rows, max_rows=int(budget.get("max_rows", 1000)), max_result_bytes=int(budget.get("max_result_bytes", 1048576)))

    @staticmethod
    async def _execute_oracle(endpoint, username, password, budget, compiled) -> NormalizedQueryResult:
        dsn = (
            f"tcps://{endpoint.host}:{endpoint.port}/{endpoint.database}"
            if endpoint.tls_enabled else
            oracledb.makedsn(endpoint.host, endpoint.port, service_name=endpoint.database)
        )
        connection = None
        try:
            connection = await asyncio.wait_for(
                oracledb.connect_async(
                    user=username, password=password, dsn=dsn,
                    tcp_connect_timeout=15,
                ),
                timeout=17,
            )
        except (oracledb.Error, OSError, TimeoutError) as exc:
            code = getattr(getattr(exc, "args", [None])[0], "code", None)
            raise ValueError(
                "DATA_SOURCE_AUTHENTICATION_FAILED"
                if code in {1017, 28000, 28001}
                else "DATA_SOURCE_CONNECTION_TIMEOUT"
                if isinstance(exc, TimeoutError)
                else "DATA_SOURCE_CONNECTION_FAILED"
            ) from exc
        try:
            connection.call_timeout = int(budget.get("statement_timeout_seconds", 30)) * 1000
            cursor = connection.cursor()
            try:
                await cursor.execute("SET TRANSACTION READ ONLY")
                binds = {f"p{index + 1}": value for index, value in enumerate(compiled.parameters)}
                await cursor.execute(compiled.sql, binds)
                columns = [item[0] for item in cursor.description]
                raw = await cursor.fetchall()
                rows = [dict(zip(columns, row, strict=True)) for row in raw]
            finally:
                cursor.close()
            await connection.rollback()
        except oracledb.Error as exc:
            raise ValueError("DATA_QUERY_EXECUTION_FAILED") from exc
        except (OSError, TimeoutError) as exc:
            raise ValueError("DATA_QUERY_EXECUTION_FAILED") from exc
        finally:
            if connection is not None: await connection.close()
        return normalize_rows(rows=rows, max_rows=int(budget.get("max_rows", 1000)), max_result_bytes=int(budget.get("max_result_bytes", 1048576)))
