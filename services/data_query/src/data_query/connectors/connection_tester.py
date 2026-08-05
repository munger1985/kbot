"""外部数据库连接测试 Connector。"""

import asyncio
import socket
import ssl

import aiomysql
import asyncpg
import oracledb

from data_query.contracts import DataSourceConnectionTest, DataSourceConnectionTestResult
from data_query.domain.errors import DataSourceConnectionError

_CONNECTION_ERROR_MESSAGES = {
    "DATA_SOURCE_AUTHENTICATION_FAILED": "数据库身份验证失败，请检查只读账号和密码。",
    "DATA_SOURCE_DATABASE_NOT_FOUND": "数据库或 Oracle Service Name 不存在，请检查名称和监听器配置。",
    "DATA_SOURCE_HOST_NOT_FOUND": "无法解析数据库主机地址，请检查主机名或 DNS。",
    "DATA_SOURCE_CONNECTION_REFUSED": "数据库端口拒绝连接，请确认数据库已启动、端口正确且允许当前主机访问。",
    "DATA_SOURCE_CONNECTION_TIMEOUT": "数据库连接超时，请检查网络、防火墙和访问控制。",
    "DATA_SOURCE_TLS_FAILED": "数据库监听未接受 TLS/TCPS 连接。本地普通 TCP Listener 请关闭“使用加密连接”；生产环境请正确配置数据库证书后重试。",
    "DATA_SOURCE_METADATA_PERMISSION_DENIED": "连接已建立，但只读账号缺少读取数据库版本或 Schema 元数据的权限。",
    "DATA_SOURCE_SCHEMA_NOT_FOUND": "连接已建立，但配置的允许 Schema 不存在或当前账号不可见。",
    "DATA_SOURCE_CONNECTION_FAILED": "数据库连接失败，请检查主机、端口、数据库名称及连接加密方式。",
}


async def test_data_source_connection(
    *, command: DataSourceConnectionTest
) -> DataSourceConnectionTestResult:
    """最小权限数据库连通性检查；不落库、不返回服务器地址或驱动异常。"""
    if command.source_type == "MYSQL":
        return await _test_mysql_connection(command)
    if command.source_type == "ORACLE":
        return await _test_oracle_connection(command)
    connection = None
    try:
        connection = await asyncpg.connect(
            host=command.endpoint.host, port=command.endpoint.port,
            database=command.endpoint.database,
            user=command.credentials.username, password=command.credentials.password,
            ssl="require" if command.endpoint.tls_enabled else False,
            timeout=10, command_timeout=10,
        )
        version = await connection.fetchval("SHOW server_version")
        schemas = await connection.fetch(
            """SELECT schema_name FROM information_schema.schemata
               WHERE schema_name NOT LIKE 'pg_%' AND schema_name <> 'information_schema'
               ORDER BY schema_name"""
        )
        _validate_allowed_schemas(
            configured=command.endpoint.allowed_schemas,
            visible=(str(row[0]) for row in schemas),
        )
        return DataSourceConnectionTestResult(
            ok=True, database_version=str(version),
            capabilities={
                "connector": "POSTGRESQL", "tls": command.endpoint.tls_enabled,
                "schemas": [str(row[0]) for row in schemas],
            },
        )
    except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
        raise DataSourceConnectionError(
            _connection_error_code(exc, tls_enabled=command.endpoint.tls_enabled)
        ) from exc
    finally:
        if connection is not None:
            await connection.close()


def _connection_error_code(exc: BaseException, *, tls_enabled: bool = False) -> str:
    if isinstance(exc, (asyncpg.InvalidPasswordError, asyncpg.InvalidAuthorizationSpecificationError)):
        return "DATA_SOURCE_AUTHENTICATION_FAILED"
    if isinstance(exc, (asyncpg.InvalidCatalogNameError, asyncpg.InvalidSchemaNameError)):
        return "DATA_SOURCE_DATABASE_NOT_FOUND"
    if isinstance(exc, asyncpg.InsufficientPrivilegeError):
        return "DATA_SOURCE_METADATA_PERMISSION_DENIED"
    if isinstance(exc, socket.gaierror):
        return "DATA_SOURCE_HOST_NOT_FOUND"
    if isinstance(exc, ConnectionRefusedError):
        return "DATA_SOURCE_CONNECTION_REFUSED"
    if isinstance(exc, TimeoutError):
        return "DATA_SOURCE_CONNECTION_TIMEOUT"
    if tls_enabled and isinstance(exc, (ssl.SSLError, ConnectionResetError, ConnectionAbortedError)):
        return "DATA_SOURCE_TLS_FAILED"
    return "DATA_SOURCE_CONNECTION_FAILED"


async def _test_mysql_connection(command: DataSourceConnectionTest) -> DataSourceConnectionTestResult:
    connection = None
    try:
        connection = await aiomysql.connect(
            host=command.endpoint.host, port=command.endpoint.port,
            db=command.endpoint.database, user=command.credentials.username,
            password=command.credentials.password, connect_timeout=10,
            ssl=ssl.create_default_context() if command.endpoint.tls_enabled else None,
        )
        async with connection.cursor() as cursor:
            await cursor.execute("SELECT VERSION()")
            row = await cursor.fetchone()
            await cursor.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('information_schema','mysql','performance_schema','sys') "
                "ORDER BY schema_name"
            )
            schemas = [str(item[0]) for item in await cursor.fetchall()]
        _validate_allowed_schemas(
            configured=command.endpoint.allowed_schemas,
            visible=schemas,
        )
        return DataSourceConnectionTestResult(
            ok=True, database_version=str(row[0]) if row else None,
            capabilities={"connector": "MYSQL", "tls": command.endpoint.tls_enabled, "schemas": schemas},
        )
    except (aiomysql.Error, OSError, TimeoutError) as exc:
        code = int(exc.args[0]) if getattr(exc, "args", ()) and isinstance(exc.args[0], int) else 0
        stable = (
            "DATA_SOURCE_AUTHENTICATION_FAILED" if code in {1044, 1045}
            else "DATA_SOURCE_DATABASE_NOT_FOUND" if code == 1049
            else "DATA_SOURCE_METADATA_PERMISSION_DENIED" if code in {1142, 1227}
            else "DATA_SOURCE_HOST_NOT_FOUND" if isinstance(exc, socket.gaierror)
            else "DATA_SOURCE_CONNECTION_REFUSED" if isinstance(exc, ConnectionRefusedError) or code in {2002, 2003}
            else "DATA_SOURCE_CONNECTION_TIMEOUT" if isinstance(exc, TimeoutError)
            else "DATA_SOURCE_TLS_FAILED" if command.endpoint.tls_enabled and isinstance(exc, (ssl.SSLError, ConnectionResetError, ConnectionAbortedError))
            else "DATA_SOURCE_CONNECTION_FAILED"
        )
        raise DataSourceConnectionError(stable) from exc
    finally:
        if connection is not None:
            connection.close()


async def _test_oracle_connection(command: DataSourceConnectionTest) -> DataSourceConnectionTestResult:
    connection = None
    try:
        dsn = (
            f"tcps://{command.endpoint.host}:{command.endpoint.port}/{command.endpoint.database}"
            if command.endpoint.tls_enabled else
            oracledb.makedsn(command.endpoint.host, command.endpoint.port, service_name=command.endpoint.database)
        )
        connection = await asyncio.wait_for(
            oracledb.connect_async(
                user=command.credentials.username, password=command.credentials.password,
                dsn=dsn, tcp_connect_timeout=10,
            ),
            timeout=12,
        )
        cursor = connection.cursor()
        try:
            await cursor.execute(
                "SELECT DISTINCT owner FROM all_objects WHERE object_type IN ('TABLE','VIEW') "
                "AND owner NOT IN ('SYS','SYSTEM') ORDER BY owner"
            )
            schemas = [str(item[0]) for item in await cursor.fetchall()]
        finally:
            cursor.close()
        _validate_allowed_schemas(
            configured=command.endpoint.allowed_schemas,
            visible=schemas,
        )
        return DataSourceConnectionTestResult(
            ok=True, database_version=str(connection.version),
            capabilities={"connector": "ORACLE", "tls": command.endpoint.tls_enabled, "schemas": schemas},
        )
    except (oracledb.Error, OSError, TimeoutError) as exc:
        code = getattr(getattr(exc, "args", [None])[0], "code", None)
        stable = (
            "DATA_SOURCE_AUTHENTICATION_FAILED" if code in {1017, 28000, 28001}
            else "DATA_SOURCE_DATABASE_NOT_FOUND" if code in {12154, 12514}
            else "DATA_SOURCE_METADATA_PERMISSION_DENIED" if code in {942, 1031}
            else "DATA_SOURCE_HOST_NOT_FOUND" if isinstance(exc, socket.gaierror) or code == 12545
            else "DATA_SOURCE_CONNECTION_REFUSED" if isinstance(exc, ConnectionRefusedError) or code in {12541, 12543}
            else "DATA_SOURCE_CONNECTION_TIMEOUT" if isinstance(exc, TimeoutError) or code in {12170, 12535}
            else "DATA_SOURCE_TLS_FAILED" if command.endpoint.tls_enabled
            else "DATA_SOURCE_CONNECTION_FAILED"
        )
        raise DataSourceConnectionError(stable) from exc
    finally:
        if connection is not None:
            await connection.close()


def _validate_allowed_schemas(*, configured, visible) -> None:
    available = {str(item).casefold() for item in visible}
    missing = [item for item in configured if item.casefold() not in available]
    if missing:
        raise DataSourceConnectionError("DATA_SOURCE_SCHEMA_NOT_FOUND")
