"""K2 知识检索 App 定向建表与验收支撑；禁止访问其他 App 的写路径。"""

import argparse
import asyncio
import importlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in ROOT.glob("packages/*/src")] + [
    str(p) for p in ROOT.glob("services/*/src")
]

from sqlalchemy import event, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import AddConstraint, CreateIndex, CreateTable
from sqlalchemy.sql.dml import Delete, Insert, Update


def knowledge_tables():
    """只从固定知识检索 App 模块取表，不遍历公共 metadata。"""
    module = importlib.import_module("knowledge_retrieval_app.entities.knowledge_ingestion")
    tables = tuple(getattr(module, name).__table__ for name in module.__all__)
    prefix = "KBOT_KR_KNOWLEDGE_" if (ROOT / "database/oracle").is_dir() else "knowledge_retrieval_knowledge_"
    if len(tables) != 18 or len({table.name for table in tables}) != 18:
        raise RuntimeError("知识检索表清单必须恰好为 18 张")
    if any(not table.name.startswith(prefix) for table in tables):
        raise RuntimeError("表清单越出知识检索 App")
    for table in tables:
        if any(fk.referred_table not in tables for fk in table.foreign_key_constraints):
            raise RuntimeError("外键越出知识检索表清单")
    return tables


def assert_write_allowed(statement, tables):
    """仅放行指向固定表的结构化语句；拒绝裸 SQL 写入及 DROP/TRUNCATE。"""
    if isinstance(statement, (Insert, Update, Delete)):
        target = statement.table
        if isinstance(statement, (Update, Delete)) and statement.whereclause is None:
            raise RuntimeError("验收写入必须包含范围条件")
    elif isinstance(statement, CreateTable):
        target = statement.element
    elif isinstance(statement, (CreateIndex, AddConstraint)):
        target = statement.element.table
    else:
        raise RuntimeError("拒绝未获授权的数据库写语句")
    if (target.schema, target.name) not in {(table.schema, table.name) for table in tables}:
        raise RuntimeError("拒绝写入知识检索清单之外的表")


def guard_engine(engine, tables):
    """拦截每条游标语句，包含 exec_driver_sql；只读元数据查询不改状态。"""
    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def before_cursor_execute(connection, cursor, statement, parameters, context, executemany):
        del connection, cursor, parameters, executemany
        command = statement.lstrip().split(None, 1)[0].upper()
        if command == "SELECT" and ";" not in statement.rstrip().rstrip(";"):
            return
        compiled = getattr(context, "compiled", None)
        if compiled is None:
            raise RuntimeError("拒绝非结构化数据库写入")
        assert_write_allowed(compiled.statement, tables)


def configured_engine():
    """复用本仓 typed 配置，不接受任意 DSN 或运行时表名。"""
    if (ROOT / "database/oracle").is_dir():
        import oracledb
        from platform_core.config.settings import get_oracle_config
        config = get_oracle_config()
        url = URL.create("oracle+oracledb", username=config.username,
                         password=config.require_password(),
                         query={"dsn": oracledb.makedsn(config.host, config.port,
                                                       service_name=config.service_name)})
        engine = create_async_engine(url, hide_parameters=True,
                                     connect_args={"tcp_connect_timeout": 10})
        @event.listens_for(engine.sync_engine, "connect")
        def configure_oracle_session(dbapi_connection, connection_record):
            del connection_record
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("ALTER SESSION SET TIME_ZONE = '+00:00'")
            finally:
                cursor.close()
        schema = None
    else:
        from platform_core.config.settings import get_postgresql_config
        config = get_postgresql_config()
        schema = config.schema_name
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
            raise RuntimeError("配置的 PostgreSQL Schema 名无效")
        url = URL.create("postgresql+asyncpg", username=config.username,
                         password=config.require_password(), host=config.host,
                         port=config.port, database=config.database)
        engine = create_async_engine(url, hide_parameters=True, connect_args={
            "timeout": 10, "server_settings": {"search_path": schema, "timezone": "UTC"}
        })
    tables = knowledge_tables()
    guard_engine(engine, tables)
    return engine, tables, schema


def oracle_column_type(row):
    """直接核对原生类型，不依赖旧版 SQLAlchemy 对 JSON 的反射支持。"""
    name = row["data_type"]
    if name == "NUMBER":
        if row["data_precision"] is None:
            return name
        return f"NUMBER({row['data_precision']},{row['data_scale'] or 0})"
    if name == "VARCHAR2":
        unit = "CHAR" if row["char_used"] == "C" else "BYTE"
        length = row["char_length"] if unit == "CHAR" else row["data_length"]
        return f"VARCHAR2({length}{unit})"
    if name == "RAW":
        return f"RAW({row['data_length']})"
    return name.replace("(6)", "").replace(" ", "")


def index_columns_match(actual, index, *, oracle=False):
    """Oracle 对带时区列自动生成 UTC 函数索引；仅接受该精确等价表达式。"""
    names = actual["column_names"]
    columns = tuple(index.columns)
    if len(names) != len(columns):
        return False
    for position, (name, column) in enumerate(zip(names, columns)):
        if name is not None:
            if name.casefold() != column.name.casefold():
                return False
        else:
            expressions = actual.get("expressions", ())
            if not oracle or not getattr(column.type, "timezone", False) or position >= len(expressions):
                return False
            expression = expressions[position].replace('"', "").replace(" ", "").casefold()
            if expression != "sys_extract_utc(" + column.name.casefold() + ")":
                return False
    return True


def inspect_tables(connection, tables):
    """部署/验收时检查，仅查看固定表；不在服务启动或业务运行时调用。"""
    inspector = inspect(connection)
    missing = []
    for table in tables:
        if not inspector.has_table(table.name):
            missing.append(table.name)
            continue
        if connection.dialect.name == "oracle":
            rows = connection.execute(text(
                "SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE, "
                "CHAR_LENGTH, CHAR_USED, NULLABLE FROM USER_TAB_COLUMNS WHERE TABLE_NAME=:table_name"
            ), {"table_name": table.name}).mappings().all()
            columns = {row["column_name"].casefold(): {
                "signature": oracle_column_type(row), "nullable": row["nullable"] == "Y"
            } for row in rows}
        else:
            columns = {column["name"].casefold(): column for column in inspector.get_columns(table.name)}
        expected = {column.name.casefold() for column in table.columns}
        if set(columns) != expected:
            raise RuntimeError("既有知识检索表列不匹配，拒绝自动改表")
        for column in table.columns:
            actual = columns[column.name.casefold()]
            expected_type = column.type.compile(dialect=connection.dialect).upper().replace(" ", "")
            if connection.dialect.name == "oracle":
                expected_type = expected_type.replace("NUMERIC", "NUMBER")
                actual_type = actual["signature"]
            else:
                actual_type = actual["type"].compile(dialect=connection.dialect).upper().replace(" ", "")
            if actual_type != expected_type or actual["nullable"] != column.nullable:
                raise RuntimeError("既有知识检索列类型或非空约束不匹配")
        primary = inspector.get_pk_constraint(table.name)["constrained_columns"]
        if {name.casefold() for name in primary} != {column.name.casefold() for column in table.primary_key}:
            raise RuntimeError("既有知识检索主键不匹配")
        foreign = inspector.get_foreign_keys(table.name)
        expected_foreign = {(tuple(c.parent.name.casefold() for c in fk.elements),
                             fk.referred_table.name.casefold(),
                             tuple(c.column.name.casefold() for c in fk.elements))
                            for fk in table.foreign_key_constraints}
        actual_foreign = {(tuple(name.casefold() for name in fk["constrained_columns"]),
                           fk["referred_table"].casefold(),
                           tuple(name.casefold() for name in fk["referred_columns"]))
                          for fk in foreign}
        if actual_foreign != expected_foreign:
            raise RuntimeError("既有知识检索外键不匹配")
        indexes = {index["name"].casefold(): index for index in inspector.get_indexes(table.name)}
        for index in table.indexes:
            actual = indexes.get(index.name.casefold())
            if (actual is None or actual["unique"] or not index_columns_match(
                    actual, index, oracle=connection.dialect.name == "oracle")):
                raise RuntimeError("既有知识检索访问索引不匹配")
    return missing


async def prepare(*, apply=False):
    engine, tables, schema = configured_engine()
    try:
        async with engine.begin() as connection:
            missing = await connection.run_sync(lambda sync: inspect_tables(sync, tables))
            print(json.dumps({"scope": "knowledge_retrieval_only", "tables": 18,
                              "missing_tables": missing, "apply": apply}, ensure_ascii=False), flush=True)
            if apply and missing:
                if len(missing) != len(tables):
                    raise RuntimeError("目标存在部分表，拒绝隐式修补；需核对上次执行结果")
                # 显式限定 tables，绝不调用全库 initializer 或公共 metadata.create_all()。
                await connection.run_sync(lambda sync: tables[0].metadata.create_all(
                    sync, tables=tables, checkfirst=False))
                if await connection.run_sync(lambda sync: inspect_tables(sync, tables)):
                    raise RuntimeError("知识检索定向建表验收失败")
                print("知识检索 App 18 张规范表创建并核验通过；没有删除既有表", flush=True)
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="显式批准仅创建缺少的 18 张知识检索表")
    args = parser.parse_args()
    try:
        asyncio.run(prepare(apply=args.apply))
    except Exception as exc:
        # 不输出凭据、连接 URL 或异常绑定参数。
        print("知识检索定向检查失败：" + type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
