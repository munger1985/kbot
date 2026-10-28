"""三库受控估算Driver；只编译计划，不执行原查询或ANALYZE，不修改业务对象。"""

import asyncio
from contextlib import suppress
import json
import ssl
from uuid import uuid4

import aiomysql
import asyncpg
import oracledb

from aiops_agent.domain.evidence.sql_plan_material import parse_sql_plan_material, SqlPlanMaterialInvalid
from aiops_agent.domain.evidence.sql_plan_replay import compile_sql_plan_replay, SqlPlanReplayRejected
from aiops_agent.domain.evidence.sql_plan import MAX_PLAN_BYTES, parse_sql_plan_json
from aiops_agent.executor.drivers.base import DiagnosticDriverError
from aiops_agent.executor.drivers.postgresql import _cancel_owned_connection
from aiops_agent.executor.drivers.mysql import _mysql_bound_sql
from aiops_agent.executor.drivers.mysql_cancellation import OwnedMySQLExecution
from aiops_agent.executor.identity import observe_postgresql_identity, observe_mysql_connection_identity, observe_oracle_identity
from aiops_agent.executor.readonly_lifecycle import observe_termination


def check_context(document, *, database, schema, role, search_path=None, settings=None):
    context = document["execution_context"]
    if (context["database_name"] != database or context["schema_name"] != schema or context["role_name"] != role
        or any(context[key] is None for key in ("database_name", "schema_name", "role_name"))):
        raise SqlPlanReplayRejected("PLAN_CURRENT_CONTEXT_MISMATCH")
    if "search_path" in context and context["search_path"] != (search_path or []):
        raise SqlPlanReplayRejected("PLAN_SEARCH_PATH_MISMATCH")
    requested = context.get("settings", {})
    if any((settings or {}).get(key) != value for key, value in requested.items()):
        raise SqlPlanReplayRejected("PLAN_CURRENT_SETTINGS_MISMATCH")


def check_identity(expected, observed):
    current = dict(observed)
    if any(current.get(key) != value for key, value in expected.items()):
        raise SqlPlanReplayRejected("PLAN_SOURCE_PHYSICAL_IDENTITY_MISMATCH")
    return current


def project_plan(raw, database_type):
    if not isinstance(raw, str):
        raw = json.dumps(raw, default=str)
    if len(raw.encode("utf-8")) > MAX_PLAN_BYTES:
        raise SqlPlanReplayRejected("PLAN_RESULT_TOO_LARGE")
    if database_type == "ORACLE":
        raw = json.dumps({"schema_version": "SQL_PLAN_IMPORT.v1", "database_type": "ORACLE", "plan": json.loads(raw)})
    plan = parse_sql_plan_json(raw)
    if plan is None or plan.database_type != database_type or plan.runtime_statistics != "NONE":
        raise SqlPlanReplayRejected("PLAN_RESULT_INVALID")
    return tuple({key: value for key, value in node.model_dump().items() if key != "object_name_semantics"} for node in plan.nodes)


class NativeSqlPlanCaptureDriver:
    """独占新连接；诊断身份与声明不一致时不切换角色或猜测原环境。"""
    def __init__(self, *, postgresql_driver):
        self._pg = postgresql_driver

    async def capture(self, exchanged):
        try:
            document = parse_sql_plan_material(exchanged.protected_document.get_secret_value())
            replay = compile_sql_plan_replay(database_type=exchanged.database_type,
                sql=exchanged.source_sql.get_secret_value(), document=document, allowed_objects=exchanged.allowed_objects)
            handler = {"POSTGRESQL": self._postgresql, "MYSQL": self._mysql, "ORACLE": self._oracle}[exchanged.database_type]
            return await handler(exchanged, document, replay)
        except SqlPlanMaterialInvalid:
            raise SqlPlanReplayRejected("PLAN_PROTECTED_MATERIAL_INVALID") from None
        except (SqlPlanReplayRejected, DiagnosticDriverError, asyncio.CancelledError):
            raise
        except TimeoutError:
            raise SqlPlanReplayRejected("PLAN_CAPTURE_TIMEOUT") from None
        except Exception:
            # 原生错误可含绑定值、SQL或连接串，不使用普通诊断错误详情/日志通道。
            raise SqlPlanReplayRejected("PLAN_NATIVE_CAPTURE_FAILED") from None

    async def _postgresql(self, exchanged, document, replay):
        profile = exchanged.connection_profile
        connection = None
        completed = False
        try:
            tls = self._pg._resolve_tls_profile(profile.tls_profile_ref) if profile.tls_profile_ref else "require" if profile.tls_enabled else False
            async with asyncio.timeout(10):
                connection = await asyncpg.connect(host=profile.host, port=profile.port, database=profile.database,
                    user=exchanged.username.get_secret_value(), password=exchanged.password.get_secret_value(),
                    ssl=tls, timeout=10, command_timeout=exchanged.statement_timeout_seconds,
                    server_settings={"application_name": "aiops-controlled-plan", "statement_timeout": str(exchanged.statement_timeout_seconds * 1000)})
            async with asyncio.timeout(exchanged.statement_timeout_seconds):
                transaction = connection.transaction(readonly=True)
                await transaction.start()
                row = await connection.fetchrow("SELECT current_database(), current_schema(), current_user, current_schemas(false)")
                requested = document["execution_context"].get("settings", {})
                settings = await connection.fetch("SELECT name, setting FROM pg_catalog.pg_settings WHERE name=ANY($1::text[])", list(requested))
                check_context(document, database=row[0], schema=row[1], role=row[2], search_path=list(row[3]),
                    settings={item["name"]: item["setting"] for item in settings})
                identity = check_identity(exchanged.source_identity, await observe_postgresql_identity(connection))
                statement = await connection.prepare("EXPLAIN (FORMAT JSON, VERBOSE TRUE, COSTS TRUE, SETTINGS FALSE) " + replay.sql)
                inferred = statement.get_parameters()
                if len(inferred) != len(replay.native_types):
                    raise SqlPlanReplayRejected("PLAN_NATIVE_BIND_LAYOUT_MISMATCH")
                for parameter, name in zip(inferred, replay.native_types, strict=True):
                    oid = await connection.fetchval(
                        "SELECT t.oid FROM pg_catalog.pg_type t JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace WHERE n.nspname='pg_catalog' AND t.typname=$1", name)
                    if oid is None or parameter.oid != oid:
                        raise SqlPlanReplayRejected("PLAN_NATIVE_PARAMETER_TYPE_MISMATCH")
                raw = await statement.fetchval(*replay.parameters)
                nodes = project_plan(raw, "POSTGRESQL")
                await transaction.rollback()
                completed = True
                return nodes, identity
        except (asyncio.CancelledError, TimeoutError):
            if connection is not None:
                await _cancel_owned_connection(connection)
                connection = None
            raise
        finally:
            if connection is not None:
                try:
                    async with asyncio.timeout(2):
                        await connection.close()
                except asyncio.CancelledError:
                    connection.terminate()
                    observe_termination("POSTGRESQL", "TERMINATION_UNKNOWN", "POSTGRESQL_PROTOCOL_CLOSE")
                    raise
                except Exception:
                    connection.terminate()
                    observe_termination("POSTGRESQL", "TERMINATION_UNKNOWN", "POSTGRESQL_PROTOCOL_CLOSE")
                    if completed:
                        raise SqlPlanReplayRejected("PLAN_CAPTURE_CLEANUP_UNCONFIRMED") from None

    async def _mysql(self, exchanged, document, replay):
        profile = exchanged.connection_profile
        if profile.tls_profile_ref:
            raise SqlPlanReplayRejected("PLAN_TLS_PROFILE_UNSUPPORTED")
        owner = OwnedMySQLExecution()
        async def connect(*, control=False):
            return await aiomysql.connect(host=profile.host, port=profile.port, db=profile.database,
                user=exchanged.username.get_secret_value(), password=exchanged.password.get_secret_value(),
                autocommit=control, connect_timeout=2 if control else 10,
                ssl=ssl.create_default_context() if profile.tls_enabled else None)
        async def operation(ownership):
            # 包括连接及身份检查在内的整次操作共用时限；不只给最后EXPLAIN计时。
            async with ownership.statement_window(exchanged.statement_timeout_seconds):
                connection = await connect()
                try:
                    identity = check_identity(exchanged.source_identity, await observe_mysql_connection_identity(connection))
                    ownership.attach(connection, tuple(identity.items()))
                    ownership.ensure_active()
                    return await self._mysql_plan(connection, ownership, exchanged, document, replay, identity)
                finally:
                    await ownership.finish(connection)
        return await owner.run(operation, open_control_connection=lambda: connect(control=True))

    async def _mysql_plan(self, connection, ownership, exchanged, document, replay, identity):
        async with connection.cursor() as cursor:
            await cursor.execute("START TRANSACTION READ ONLY")
            await cursor.execute("SET SESSION MAX_EXECUTION_TIME=%s", (exchanged.statement_timeout_seconds * 1000,))
            await cursor.execute("SELECT DATABASE(), CURRENT_USER()")
            row = (await cursor.fetchmany(1))[0]
            if document["execution_context"].get("settings"):
                raise SqlPlanReplayRejected("PLAN_CONTEXT_SETTINGS_UNSUPPORTED")
            check_context(document, database=row[0], schema=row[0], role=row[1])
            ownership.ensure_active()
            await cursor.execute("EXPLAIN FORMAT=JSON " + _mysql_bound_sql(replay.sql),
                {"plan_bind_" + str(index): value for index, value in enumerate(replay.parameters, start=1)})
            rows = await cursor.fetchmany(2)
            if len(rows) != 1 or len(rows[0]) != 1:
                raise SqlPlanReplayRejected("PLAN_RESULT_INVALID")
            nodes = project_plan(rows[0][0], "MYSQL")
            await connection.rollback()
            return nodes, identity

    async def _oracle(self, exchanged, document, replay):
        profile = exchanged.connection_profile
        if profile.tls_profile_ref:
            raise SqlPlanReplayRejected("PLAN_TLS_PROFILE_UNSUPPORTED")
        dsn = f"tcps://{profile.host}:{profile.port}/{profile.service}" if profile.tls_enabled else oracledb.makedsn(profile.host, profile.port, service_name=profile.service)
        connection = None
        completed = False
        try:
            async with asyncio.timeout(10):
                connection = await oracledb.connect_async(user=exchanged.username.get_secret_value(),
                    password=exchanged.password.get_secret_value(), dsn=dsn)
            connection.autocommit = False
            connection.call_timeout = exchanged.statement_timeout_seconds * 1000
            connection.module = "aiops-controlled-plan"
            async with asyncio.timeout(exchanged.statement_timeout_seconds):
                cursor = connection.cursor()
                try:
                    await cursor.execute("SELECT SYS_CONTEXT('USERENV','CON_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA'), SYS_CONTEXT('USERENV','SESSION_USER') FROM DUAL")
                    row = (await cursor.fetchmany(1))[0]
                    if document["execution_context"].get("settings"):
                        raise SqlPlanReplayRejected("PLAN_CONTEXT_SETTINGS_UNSUPPORTED")
                    check_context(document, database=row[0], schema=row[1], role=row[2])
                    identity = check_identity(exchanged.source_identity, await observe_oracle_identity(connection))
                    # 显式使用Oracle自带临时计划表，不能解析到用户创建的同名表/触发器。
                    await cursor.execute("SELECT temporary FROM ALL_TABLES WHERE owner='SYS' AND table_name='PLAN_TABLE$'")
                    available = await cursor.fetchmany(2)
                    if len(available) != 1 or tuple(available[0]) != ("Y",):
                        raise SqlPlanReplayRejected("PLAN_ORACLE_TEMP_TABLE_UNAVAILABLE")
                    statement_id = "AIOPS_" + uuid4().hex[:24]
                    binds = {"plan_bind_" + str(index): value for index, value in enumerate(replay.parameters, start=1)}
                    types = {}
                    for index, name in enumerate(replay.native_types, start=1):
                        types["plan_bind_" + str(index)] = {"number": oracledb.DB_TYPE_NUMBER,
                            "date": oracledb.DB_TYPE_DATE, "timestamp": oracledb.DB_TYPE_TIMESTAMP,
                            "varchar2": oracledb.DB_TYPE_VARCHAR, "nvarchar2": oracledb.DB_TYPE_NVARCHAR,
                            "char": oracledb.DB_TYPE_CHAR, "nchar": oracledb.DB_TYPE_NCHAR}[name]
                    if types:
                        cursor.setinputsizes(**types)
                    await cursor.execute("EXPLAIN PLAN SET STATEMENT_ID = '" + statement_id + "' INTO SYS.PLAN_TABLE$ FOR " + replay.sql, binds)
                    # 新游标隔离代表参数的类型/绑定状态，不能遗留到计划表读取语句。
                    cursor.close()
                    cursor = connection.cursor()
                    await cursor.execute("SELECT id,parent_id,operation,options,object_owner,object_name,object_type,cardinality,cost FROM SYS.PLAN_TABLE$ WHERE statement_id=:statement_id ORDER BY id", {"statement_id": statement_id})
                    rows = await cursor.fetchmany(129)
                    if not rows or len(rows) > 128:
                        raise SqlPlanReplayRejected("PLAN_RESULT_TOO_LARGE")
                    plan_rows = []
                    for row in rows:
                        plan_rows.append({"id": int(row[0]), "parent_id": row[1],
                            "operation": row[2] + (" " + row[3] if row[3] else ""),
                            "object_owner": row[4], "object_name": row[5], "object_type": row[6],
                            "cardinality": row[7], "cost": row[8]})
                    nodes = project_plan(plan_rows, "ORACLE")
                    await connection.rollback()
                    completed = True
                    return nodes, identity
                finally:
                    cursor.close()
        except (asyncio.CancelledError, TimeoutError):
            if connection is not None:
                observe_termination("ORACLE", "TERMINATION_UNKNOWN", "ORACLE_CANCEL")
                with suppress(oracledb.Error):
                    connection.cancel()
                    observe_termination("ORACLE", "CANCEL_REQUESTED", "ORACLE_CANCEL")
            raise
        finally:
            if connection is not None:
                try:
                    async with asyncio.timeout(2):
                        await connection.rollback()
                        await connection.close()
                except asyncio.CancelledError:
                    observe_termination("ORACLE", "TERMINATION_UNKNOWN", "ORACLE_CANCEL")
                    raise
                except Exception:
                    observe_termination("ORACLE", "TERMINATION_UNKNOWN", "ORACLE_CANCEL")
                    if completed:
                        raise SqlPlanReplayRejected("PLAN_CAPTURE_CLEANUP_UNCONFIRMED") from None
