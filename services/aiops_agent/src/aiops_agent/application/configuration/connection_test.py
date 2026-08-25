"""创建 Target 前执行不落库的最小只读数据库连接测试。"""

from __future__ import annotations

import asyncio
import socket
import ssl

import aiomysql
import asyncpg
import oracledb

from platform_core.contracts.aiops import (
    TargetConnectionTest,
    TargetConnectionTestResult,
)


async def test_target_connection(
    request: TargetConnectionTest,
) -> TargetConnectionTestResult:
    """仅验证连接、认证和最小查询，不保存连接信息或凭据。"""
    try:
        if request.db_type == "ORACLE":
            version = await _test_oracle(request)
        elif request.db_type == "MYSQL":
            version = await _test_mysql(request)
        else:
            version = await _test_postgresql(request)
        return TargetConnectionTestResult(ok=True, database_version=version)
    except TimeoutError:
        return TargetConnectionTestResult(ok=False, error_code="TIMEOUT")
    except (OSError, ssl.SSLError):
        return TargetConnectionTestResult(
            ok=False, error_code="TARGET_UNREACHABLE"
        )
    except Exception as exc:
        return TargetConnectionTestResult(
            ok=False,
            error_code=_stable_error_code(request.db_type, exc),
        )


async def _test_oracle(request: TargetConnectionTest) -> str:
    endpoint = request.endpoint
    credential = request.diagnostic_credential
    dsn = (
        f"tcps://{endpoint.host}:{endpoint.port}/{endpoint.service}"
        if endpoint.tls_enabled
        else oracledb.makedsn(
            endpoint.host,
            endpoint.port,
            service_name=endpoint.service,
        )
    )
    connection = None
    try:
        async with asyncio.timeout(12):
            connection = await oracledb.connect_async(
                user=credential.username,
                password=credential.password,
                dsn=dsn,
                tcp_connect_timeout=10,
                ssl_server_dn_match=endpoint.tls_enabled,
            )
            connection.call_timeout = 10_000
            cursor = connection.cursor()
            try:
                await cursor.execute("SELECT 1 FROM DUAL")
                await cursor.fetchone()
            finally:
                cursor.close()
            return str(connection.version)
    finally:
        if connection is not None:
            await connection.close()


async def _test_mysql(request: TargetConnectionTest) -> str:
    endpoint = request.endpoint
    credential = request.diagnostic_credential
    connection = None
    try:
        async with asyncio.timeout(12):
            connection = await aiomysql.connect(
                host=endpoint.host,
                port=endpoint.port,
                user=credential.username,
                password=credential.password,
                db=endpoint.database,
                connect_timeout=10,
                ssl=(
                    ssl.create_default_context()
                    if endpoint.tls_enabled
                    else None
                ),
            )
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT 1")
                await cursor.fetchone()
            return str(connection.get_server_info())
    finally:
        if connection is not None:
            connection.close()


async def _test_postgresql(request: TargetConnectionTest) -> str:
    endpoint = request.endpoint
    credential = request.diagnostic_credential
    connection = None
    try:
        async with asyncio.timeout(12):
            connection = await asyncpg.connect(
                host=endpoint.host,
                port=endpoint.port,
                database=endpoint.database,
                user=credential.username,
                password=credential.password,
                ssl="require" if endpoint.tls_enabled else False,
                timeout=10,
                command_timeout=10,
            )
            await connection.fetchval("SELECT 1")
            return str(await connection.fetchval("SHOW server_version"))
    finally:
        if connection is not None:
            await connection.close()


def _stable_error_code(db_type: str, exc: Exception) -> str:
    if db_type == "ORACLE" and isinstance(exc, oracledb.Error):
        code = getattr(getattr(exc, "args", [None])[0], "code", None)
        if code in {1017, 28000, 28001}:
            return "AUTH_FAILED"
        if code in {12154, 12514, 12541, 12543, 12545}:
            return "TARGET_UNREACHABLE"
        if code in {12170, 12535}:
            return "TIMEOUT"
    if db_type == "MYSQL" and isinstance(exc, aiomysql.Error):
        code = (
            int(exc.args[0])
            if exc.args and isinstance(exc.args[0], int)
            else 0
        )
        if code in {1044, 1045}:
            return "AUTH_FAILED"
        if code in {1049, 2002, 2003, 2005, 2013}:
            return "TARGET_UNREACHABLE"
    if db_type == "POSTGRESQL" and isinstance(exc, asyncpg.PostgresError):
        if isinstance(
            exc,
            (
                asyncpg.InvalidPasswordError,
                asyncpg.InvalidAuthorizationSpecificationError,
            ),
        ):
            return "AUTH_FAILED"
        if isinstance(
            exc,
            (asyncpg.InvalidCatalogNameError, asyncpg.CannotConnectNowError),
        ):
            return "TARGET_UNREACHABLE"
    if isinstance(exc, socket.gaierror):
        return "TARGET_UNREACHABLE"
    return "CONNECTION_FAILED"
