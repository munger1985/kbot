"""从当前规范 DDL 显式增补缺表；不修改已有表、数据或基础种子。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

import oracledb

from platform_core.config.settings import get_oracle_config
from scripts.db.apply_oracle_schema import SchemaStatement, load_schema_statements


def declaration_sql(statement: str) -> str:
    """移除语句前导注释，不触碰字符串或 SQL 正文。"""
    return re.sub(
        r"^\s*(?:(?:--[^\n]*(?:\n|$))|(?:/\*.*?\*/\s*))*", "", statement,
        flags=re.DOTALL,
    ).strip()


def compile_additions(
    statements: list[SchemaStatement], existing_tables: set[str],
) -> dict:
    """按规范顺序选择缺表及其索引、延迟外键，绝不重放整份建库脚本。"""
    definitions = {}
    for statement in statements:
        sql = declaration_sql(statement.sql)
        match = re.match(r"CREATE\s+TABLE\s+([A-Z0-9_]+)\s*\(", sql, re.I)
        if match:
            name = match[1].upper()
            if name in definitions:
                raise RuntimeError(f"规范包含重复表声明：{name}")
            definitions[name] = statement
    missing = set(definitions) - existing_tables
    available = set(existing_tables)
    selected = []
    for statement in statements:
        sql = declaration_sql(statement.sql)
        create = re.match(r"CREATE\s+TABLE\s+([A-Z0-9_]+)\s*\(", sql, re.I)
        index = re.match(r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+[A-Z0-9_]+\s+ON\s+([A-Z0-9_]+)\b", sql, re.I)
        alter = re.match(r"ALTER\s+TABLE\s+([A-Z0-9_]+)\b", sql, re.I)
        name = (create or index or alter)[1].upper() if (create or index or alter) else None
        if name not in missing:
            continue
        if alter and not re.match(
            r"ALTER\s+TABLE\s+[A-Z0-9_]+\s+ADD\s+CONSTRAINT\s+[A-Z0-9_]+\s+FOREIGN\s+KEY\b", sql, re.I,
        ):
            raise RuntimeError(f"拒绝增补计划中的非外键 ALTER：{statement.label}")
        references = {value.upper() for value in re.findall(r"\bREFERENCES\s+([A-Z0-9_]+)\b", sql, re.I)}
        if references - (available | {name}):
            raise RuntimeError(f"规范依赖未满足：{statement.label}：{sorted(references - available - {name})}")
        if create:
            available.add(name)
        selected.append({"source": str(statement.script), "ordinal": statement.ordinal, "sql": sql})
    return {"missing_tables": sorted(missing), "statements": selected}


def read_plan(connection) -> dict:
    """读取精确目标及对象快照；不查询业务记录内容。"""
    with connection.cursor() as cursor:
        cursor.execute("""SELECT SYS_CONTEXT('USERENV','CON_NAME'),
            SYS_CONTEXT('USERENV','CURRENT_SCHEMA'), SYS_CONTEXT('USERENV','DB_UNIQUE_NAME'),
            SYS_CONTEXT('USERENV','SERVER_HOST') FROM DUAL""")
        target = dict(zip(("container", "schema", "database", "server"), cursor.fetchone()))
        cursor.execute("SELECT object_name, object_type FROM user_objects WHERE object_type IN ('TABLE','VIEW','INDEX') ORDER BY object_type,object_name")
        objects = [list(row) for row in cursor]
    tables = {name for name, kind in objects if kind == "TABLE"}
    plan = {"target": target, **compile_additions(load_schema_statements(), tables)}
    occupied = {name for name, kind in objects if kind in {"TABLE", "VIEW"}}
    conflicts = occupied.intersection(plan["missing_tables"])
    if conflicts:
        raise RuntimeError(f"缺表名称被其他对象占用，拒绝执行：{sorted(conflicts)}")
    payload = json.dumps({"plan": plan, "objects": objects}, sort_keys=True)
    plan["plan_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    return plan


def apply_plan(connection, plan: dict, expected_hash: str) -> None:
    """每条 DDL 单独审计；Oracle 自动提交，失败后停止且绝不删除回滚。"""
    fresh = read_plan(connection)
    if fresh != plan or fresh["plan_hash"] != expected_hash:
        raise RuntimeError("目标或规范结构已变化；请重新预览，不执行任何 DDL")
    with connection.cursor() as cursor:
        cursor.execute("ALTER SESSION SET DDL_LOCK_TIMEOUT = 5")
        for statement in fresh["statements"]:
            print(json.dumps({"applying": statement["source"], "ordinal": statement["ordinal"]}), flush=True)
            cursor.execute(statement["sql"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-plan-hash", help="省略时只读预览；与当前计划一致时才执行")
    args = parser.parse_args()
    config = get_oracle_config()
    with oracledb.connect(
        user=config.username, password=config.require_password(),
        dsn=oracledb.makedsn(config.host, config.port, service_name=config.service_name),
    ) as connection:
        connection.call_timeout = 120000
        plan = read_plan(connection)
        if args.expected_plan_hash:
            apply_plan(connection, plan, args.expected_plan_hash)
        print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
