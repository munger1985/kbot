"""PG、MySQL、Oracle 数据库结构采集 Connector。"""

from __future__ import annotations

import asyncio
import ssl
from typing import Any

import aiomysql
import asyncpg
import oracledb


class DatabaseSchemaIntrospector:
    async def discover(
        self, context: _SourceContext, username: str, password: str,
    ) -> tuple[list[tuple[str, str, str]], str]:
        schemas = tuple(context.endpoint.allowed_schemas)
        if context.source_type == "POSTGRESQL":
            connection = await asyncpg.connect(
                host=context.endpoint.host, port=context.endpoint.port,
                database=context.endpoint.database, user=username, password=password,
                ssl="require" if context.endpoint.tls_enabled else False, timeout=15,
            )
            try:
                version = str(await connection.fetchval("SHOW server_version"))
                rows = await connection.fetch(
                    """SELECT table_schema, table_name, table_type
                       FROM information_schema.tables
                       WHERE table_schema = ANY($1::text[])
                       ORDER BY table_schema, table_name""", list(schemas),
                )
            finally:
                await connection.close()
            return [(str(r[0]), str(r[1]), "VIEW" if "VIEW" in str(r[2]) else "TABLE") for r in rows], version
        if context.source_type == "MYSQL":
            connection = await aiomysql.connect(
                host=context.endpoint.host, port=context.endpoint.port,
                db=context.endpoint.database, user=username, password=password,
                ssl=ssl.create_default_context() if context.endpoint.tls_enabled else None, connect_timeout=15,
            )
            try:
                async with connection.cursor() as cursor:
                    await cursor.execute("SELECT VERSION()")
                    version = str((await cursor.fetchone())[0])
                    placeholders = ",".join(["%s"] * len(schemas))
                    await cursor.execute(
                        f"SELECT table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema IN ({placeholders}) ORDER BY table_schema, table_name",  # noqa: S608 - placeholders are generated, values remain bound
                        schemas,
                    )
                    rows = await cursor.fetchall()
            finally:
                connection.close()
            return [(str(r[0]), str(r[1]), "VIEW" if "VIEW" in str(r[2]) else "TABLE") for r in rows], version
        if context.source_type == "ORACLE":
            connection = await self._oracle_connect(context, username, password)
            try:
                cursor = connection.cursor()
                try:
                    version = str(connection.version)
                    upper = tuple(item.upper() for item in schemas)
                    binds = ",".join(f":s{i}" for i in range(len(upper)))
                    await cursor.execute(
                        f"SELECT owner, object_name, object_type FROM all_objects WHERE owner IN ({binds}) AND object_type IN ('TABLE','VIEW') ORDER BY owner, object_name",  # noqa: S608
                        {f"s{i}": value for i, value in enumerate(upper)},
                    )
                    rows = await cursor.fetchall()
                finally:
                    cursor.close()
            finally:
                await connection.close()
            return [(str(r[0]), str(r[1]), str(r[2])) for r in rows], version
        raise ValueError("CONNECTOR_NOT_SUPPORTED")

    async def capture_object(
        self, context: _ObjectContext, username: str, password: str,
    ) -> dict[str, object]:
        if context.source_type == "POSTGRESQL":
            connection = await asyncpg.connect(
                host=context.endpoint.host, port=context.endpoint.port,
                database=context.endpoint.database, user=username, password=password,
                ssl="require" if context.endpoint.tls_enabled else False, timeout=15,
            )
            try:
                rows = await connection.fetch(
                    """SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod),
                              CASE WHEN a.attnotnull THEN 'NO' ELSE 'YES' END, a.attnum,
                              pg_get_expr(ad.adbin, ad.adrelid), col_description(c.oid, a.attnum)
                       FROM pg_catalog.pg_class c
                       JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
                       JOIN pg_catalog.pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
                       LEFT JOIN pg_catalog.pg_attrdef ad ON ad.adrelid=c.oid AND ad.adnum=a.attnum
                       WHERE n.nspname=$1 AND c.relname=$2 ORDER BY a.attnum""",
                    context.schema_name, context.object_name,
                )
                constraint_rows = await connection.fetch(
                    """SELECT conname, contype, pg_get_constraintdef(con.oid, true)
                       FROM pg_catalog.pg_constraint con
                       JOIN pg_catalog.pg_class c ON c.oid=con.conrelid
                       JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
                       WHERE n.nspname=$1 AND c.relname=$2 ORDER BY conname""",
                    context.schema_name, context.object_name,
                )
                index_rows = await connection.fetch(
                    "SELECT indexname, indexdef FROM pg_catalog.pg_indexes WHERE schemaname=$1 AND tablename=$2 ORDER BY indexname",
                    context.schema_name, context.object_name,
                )
                object_comment = await connection.fetchval(
                    """SELECT obj_description(c.oid, 'pg_class') FROM pg_catalog.pg_class c
                       JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
                       WHERE n.nspname=$1 AND c.relname=$2""",
                    context.schema_name, context.object_name,
                )
            finally:
                await connection.close()
        elif context.source_type == "MYSQL":
            connection = await aiomysql.connect(
                host=context.endpoint.host, port=context.endpoint.port,
                db=context.endpoint.database, user=username, password=password,
                ssl=ssl.create_default_context() if context.endpoint.tls_enabled else None, connect_timeout=15,
            )
            try:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT column_name, column_type, is_nullable, ordinal_position, column_default, column_comment FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
                        (context.schema_name, context.object_name),
                    )
                    rows = await cursor.fetchall()
                    await cursor.execute(
                        """SELECT tc.constraint_name, tc.constraint_type,
                                  GROUP_CONCAT(kcu.column_name ORDER BY kcu.ordinal_position)
                           FROM information_schema.table_constraints tc
                           LEFT JOIN information_schema.key_column_usage kcu
                             ON kcu.constraint_schema=tc.constraint_schema AND kcu.table_name=tc.table_name
                            AND kcu.constraint_name=tc.constraint_name
                           WHERE tc.table_schema=%s AND tc.table_name=%s
                           GROUP BY tc.constraint_name, tc.constraint_type ORDER BY tc.constraint_name""",
                        (context.schema_name, context.object_name),
                    )
                    constraint_rows = await cursor.fetchall()
                    await cursor.execute(
                        """SELECT index_name, GROUP_CONCAT(column_name ORDER BY seq_in_index)
                           FROM information_schema.statistics WHERE table_schema=%s AND table_name=%s
                           GROUP BY index_name ORDER BY index_name""",
                        (context.schema_name, context.object_name),
                    )
                    index_rows = await cursor.fetchall()
                    await cursor.execute(
                        "SELECT table_comment FROM information_schema.tables WHERE table_schema=%s AND table_name=%s",
                        (context.schema_name, context.object_name),
                    )
                    comment_row = await cursor.fetchone()
                    object_comment = comment_row[0] if comment_row else None
            finally:
                connection.close()
        elif context.source_type == "ORACLE":
            connection = await self._oracle_connect(context, username, password)
            try:
                cursor = connection.cursor()
                try:
                    await cursor.execute(
                        """SELECT c.column_name, c.data_type, c.nullable, c.column_id, c.data_default, cc.comments
                           FROM all_tab_columns c LEFT JOIN all_col_comments cc
                             ON cc.owner=c.owner AND cc.table_name=c.table_name AND cc.column_name=c.column_name
                           WHERE c.owner=:owner AND c.table_name=:name ORDER BY c.column_id""",
                        {"owner": context.schema_name.upper(), "name": context.object_name.upper()},
                    )
                    rows = await cursor.fetchall()
                    await cursor.execute(
                        """SELECT ac.constraint_name, ac.constraint_type,
                                  LISTAGG(acc.column_name, ',') WITHIN GROUP (ORDER BY acc.position)
                           FROM all_constraints ac LEFT JOIN all_cons_columns acc
                             ON acc.owner=ac.owner AND acc.constraint_name=ac.constraint_name
                           WHERE ac.owner=:owner AND ac.table_name=:name
                           GROUP BY ac.constraint_name, ac.constraint_type ORDER BY ac.constraint_name""",
                        {"owner": context.schema_name.upper(), "name": context.object_name.upper()},
                    )
                    constraint_rows = await cursor.fetchall()
                    await cursor.execute(
                        """SELECT ai.index_name,
                                  LISTAGG(aic.column_name, ',') WITHIN GROUP (ORDER BY aic.column_position)
                           FROM all_indexes ai JOIN all_ind_columns aic
                             ON aic.index_owner=ai.owner AND aic.index_name=ai.index_name
                           WHERE ai.table_owner=:owner AND ai.table_name=:name
                           GROUP BY ai.index_name ORDER BY ai.index_name""",
                        {"owner": context.schema_name.upper(), "name": context.object_name.upper()},
                    )
                    index_rows = await cursor.fetchall()
                    await cursor.execute(
                        "SELECT comments FROM all_tab_comments WHERE owner=:owner AND table_name=:name",
                        {"owner": context.schema_name.upper(), "name": context.object_name.upper()},
                    )
                    comment_row = await cursor.fetchone()
                    object_comment = comment_row[0] if comment_row else None
                finally:
                    cursor.close()
            finally:
                await connection.close()
        else:
            raise ValueError("CONNECTOR_NOT_SUPPORTED")
        columns = [
            {
                "name": str(row[0]), "type": str(row[1]),
                "nullable": str(row[2]).upper() in {"YES", "Y"},
                "ordinal": int(row[3]), "default": None if row[4] is None else str(row[4]),
                "comment": None if len(row) < 6 or row[5] is None else str(row[5]),
            }
            for row in rows
        ]
        if not columns:
            raise ValueError("SCHEMA_OBJECT_HAS_NO_VISIBLE_COLUMNS")
        return {
            "schema": context.schema_name, "name": context.object_name,
            "object_type": context.object_type,
            "columns": [str(column["name"]) for column in columns],
            "column_details": columns,
            "comment": None if object_comment is None else str(object_comment),
            "constraints": [
                {"name": str(row[0]), "type": str(row[1]), "definition": None if row[2] is None else str(row[2])}
                for row in constraint_rows
            ],
            "indexes": [
                {"name": str(row[0]), "definition": None if row[1] is None else str(row[1])}
                for row in index_rows
            ],
        }

    @staticmethod
    async def _oracle_connect(context: _SourceContext, username: str, password: str):
        dsn = (
            f"tcps://{context.endpoint.host}:{context.endpoint.port}/{context.endpoint.database}"
            if context.endpoint.tls_enabled else
            oracledb.makedsn(
                context.endpoint.host, context.endpoint.port,
                service_name=context.endpoint.database,
            )
        )
        connection = await asyncio.wait_for(
            oracledb.connect_async(
                user=username, password=password, dsn=dsn, tcp_connect_timeout=15,
            ),
            timeout=17,
        )
        connection.call_timeout = 15_000
        return connection

    @staticmethod
    def error_code(error: Exception | None) -> str:
        if isinstance(error, asyncpg.InvalidPasswordError):
            return "DATA_SOURCE_AUTHENTICATION_FAILED"
        if isinstance(error, aiomysql.Error):
            code = int(error.args[0]) if error.args and isinstance(error.args[0], int) else 0
            return "DATA_SOURCE_AUTHENTICATION_FAILED" if code in {1044, 1045} else "SCHEMA_SNAPSHOT_FAILED"
        if isinstance(error, oracledb.Error):
            code = getattr(error.args[0], "code", None) if error.args else None
            return "DATA_SOURCE_AUTHENTICATION_FAILED" if code in {1017, 28000, 28001} else "SCHEMA_SNAPSHOT_FAILED"
        if isinstance(error, TimeoutError):
            return "DATA_SOURCE_CONNECTION_TIMEOUT"
        if error is not None and str(error) in {"CONNECTOR_NOT_SUPPORTED", "SCHEMA_OBJECT_HAS_NO_VISIBLE_COLUMNS", "SCHEMA_DISCOVERY_OBJECT_LIMIT_EXCEEDED"}:
            return str(error)
        return "SCHEMA_SNAPSHOT_FAILED"

    @staticmethod
    def error_message(error: Exception | None) -> str:
        code = DatabaseSchemaIntrospector.error_code(error)
        return {
            "DATA_SOURCE_AUTHENTICATION_FAILED": "数据库身份验证失败，请检查只读账号和密码。",
            "DATA_SOURCE_CONNECTION_TIMEOUT": "数据库连接超时，请检查网络和访问控制。",
            "SCHEMA_OBJECT_HAS_NO_VISIBLE_COLUMNS": "当前账号无法读取该对象的字段结构。",
            "CONNECTOR_NOT_SUPPORTED": "当前数据库类型不支持结构采集。",
            "SCHEMA_DISCOVERY_OBJECT_LIMIT_EXCEEDED": "发现对象超过 10000 个，请缩小允许的 Schema 范围。",
        }.get(code, "结构采集失败，请检查数据库权限后重试。")
