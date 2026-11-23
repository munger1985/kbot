"""只清理 2026-11-17 报告获准的本地 Oracle 四个旧模型字段；不备份、不删除行。"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json

import oracledb

from model_serving.common.entities.ai_model import AIModelEntity
from platform_core.config.settings import get_oracle_config


RETIRED_COLUMNS = {
    "KBOT_AI_MODEL": (
        "SUPPORTS_IMAGE_GENERATION", "SUPPORTS_RESPONSES_STREAMING", "SUPPORTS_X_SEARCH", "CAPABILITY_VERIFIED_AT",
    ),
}


def canonical_columns() -> dict[str, set[str]]:
    """当前规范实体是保留字段的唯一来源，旧列重新被采用时拒绝清理。"""
    return {AIModelEntity.__tablename__: {column.name.upper() for column in AIModelEntity.__table__.columns}}


def compile_retirement(canonical: dict[str, set[str]], actual: dict[str, set[str]]) -> dict:
    statements = []
    removed = []
    for table, allowed in RETIRED_COLUMNS.items():
        if set(allowed) & canonical[table]:
            raise RuntimeError(f"当前规范仍使用旧字段，拒绝删除：{table}")
        if canonical[table] - actual.get(table, set()):
            raise RuntimeError(f"规范字段缺失，拒绝继续清理：{table}")
        selected = [column for column in allowed if column in actual[table]]
        if selected:
            names = ", ".join(f'"{column}"' for column in selected)
            statements.append(f'ALTER TABLE "{table}" DROP ({names})')
            removed.extend([table, column] for column in selected)
    return {"removed_columns": removed, "statements": statements, "backup": False}


def encode(value):
    """仅在内存中规范化后计算指纹，不输出模型密钥或业务内容。"""
    if isinstance(value, oracledb.LOB):
        value = value.read()
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, (date, datetime)):
        return {"time": value.isoformat()}
    if value is None or isinstance(value, (str, int, float, bool, dict, list)):
        return value
    raise TypeError(type(value).__name__)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=encode).encode()).hexdigest()


def read_plan(connection) -> dict:
    canonical = canonical_columns()
    with connection.cursor() as cursor:
        cursor.execute("""SELECT SYS_CONTEXT('USERENV','CON_NAME'),
            SYS_CONTEXT('USERENV','CURRENT_SCHEMA'), SYS_CONTEXT('USERENV','DB_UNIQUE_NAME'),
            SYS_CONTEXT('USERENV','SERVER_HOST') FROM DUAL""")
        target = dict(zip(("container", "schema", "database", "server"), cursor.fetchone()))
        if target["container"] != "KBOT4" or target["schema"] != "KBOTDEV":
            raise RuntimeError("清理目标不是获准的 KBOT4 / KBOTDEV")
        actual = {}
        for table in RETIRED_COLUMNS:
            cursor.execute("SELECT column_name FROM user_tab_columns WHERE table_name=:name", name=table)
            actual[table] = {row[0] for row in cursor}
        compilation = compile_retirement(canonical, actual)
        protected = {}
        for table, names in canonical.items():
            projection = ", ".join(f'"{name}"' for name in sorted(names))
            cursor.execute(f'SELECT {projection} FROM "{table}"')
            rows = [digest(list(row)) for row in cursor]
            protected[table] = {"rows": len(rows), "sha256": digest(sorted(rows))}
    plan = {"target": target, **compilation, "protected": protected}
    plan["plan_hash"] = digest({"plan": plan, "actual": {table: sorted(names) for table, names in actual.items()}})
    return plan


def apply_retirement(connection, plan: dict, *, expected_hash: str, discard_reported_fields: bool) -> None:
    if not discard_reported_fields:
        raise RuntimeError("执行需要显式许可删除报告中的四个旧模型字段")
    fresh = read_plan(connection)
    if fresh != plan or fresh["plan_hash"] != expected_hash:
        raise RuntimeError("目标、规范或受保护数据已变化；未执行 DDL")
    with connection.cursor() as cursor:
        cursor.execute("ALTER SESSION SET DDL_LOCK_TIMEOUT = 5")
        for statement in plan["statements"]:
            cursor.execute(statement)
    after = read_plan(connection)
    if after["statements"] or after["protected"] != plan["protected"]:
        raise RuntimeError("Oracle 自动提交后的复验未通过，请核查；不执行额外删除或恢复操作")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-plan-hash")
    parser.add_argument("--discard-reported-fields", action="store_true")
    args = parser.parse_args()
    config = get_oracle_config()
    if (config.host not in {"localhost", "127.0.0.1"} or config.port != 1521
            or config.service_name.lower() != "kbot4"):
        raise RuntimeError("此清理仅获准用于本地 kbot4 数据库")
    with oracledb.connect(user=config.username, password=config.require_password(),
            dsn=oracledb.makedsn(config.host, config.port, service_name=config.service_name)) as connection:
        connection.call_timeout = 30000
        plan = read_plan(connection)
        if args.expected_plan_hash:
            apply_retirement(connection, plan, expected_hash=args.expected_plan_hash,
                             discard_reported_fields=args.discard_reported_fields)
        print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
