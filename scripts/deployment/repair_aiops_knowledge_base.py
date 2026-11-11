"""从仓库根目录执行的 Oracle 知识库定向修复；默认只读，不迁移旧 KC 内容。"""

import argparse
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


BASE = "KBOT_OPS_KNOWLEDGE_BASE"
PROTECTED = ("KBOT_OPS_KNOWLEDGE_ASSET", "KBOT_OPS_AGENT_VERSION_KNOW_BASE")
CONFIGURATION = "CONFIGURATION_JSON"
LEGACY = "COLLECTION_ID"
EXPECTED = {
    "KNOWLEDGE_BASE_ID", "DOMAIN_ID", "DISPLAY_NAME", "DESCRIPTION", CONFIGURATION,
    "STATUS", "CREATED_BY", "UPDATED_BY", "CREATED_AT", "UPDATED_AT", "ROW_VERSION",
}
ADD = f"ALTER TABLE {BASE} ADD ({CONFIGURATION} JSON)"
FILL = f"UPDATE {BASE} SET {CONFIGURATION}=JSON('{{}}') WHERE {CONFIGURATION} IS NULL"
REQUIRE = f"ALTER TABLE {BASE} MODIFY ({CONFIGURATION} NOT NULL)"
RELAX = f"ALTER TABLE {BASE} MODIFY ({LEGACY} NULL)"


def encode(value):
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    raise TypeError(type(value).__name__)


def serialized(value):
    return json.dumps(value, default=encode, ensure_ascii=False, sort_keys=True)


def digest(value):
    return sha256(serialized(value).encode()).hexdigest()


def snapshot(connection):
    """仅快照知识库元数据和两张关联表；不读取账号、凭据或文档正文。"""
    cursor = connection.cursor()
    cursor.execute("SELECT SYS_CONTEXT('USERENV','DB_UNIQUE_NAME'), "
        "SYS_CONTEXT('USERENV','CON_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM DUAL")
    result = {"target": digest(list(cursor.fetchone())), "tables": {}}
    for table in (BASE, *PROTECTED):
        cursor.execute("SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE "
            "FROM USER_TAB_COLUMNS WHERE TABLE_NAME=:name ORDER BY COLUMN_NAME", name=table)
        columns = [list(row) for row in cursor]
        if not columns:
            raise ValueError("缺少本次定向修复所需的基础表")
        # DDL 前后列集合会变化；不用可能复用旧结果元数据的 SELECT *。
        projection = ", ".join('"' + column[0].replace('"', '""') + '"' for column in columns)
        cursor.execute("SELECT " + projection + " FROM " + table)
        names = [item[0].upper() for item in cursor.description]
        rows = sorted((dict(zip(names, row)) for row in cursor), key=serialized)
        result["tables"][table] = {"columns": columns, "rows": rows}
    return result


def build_plan(before):
    """固定补齐原生 JSON 配置，保留旧绑定值并解除弃用列的必填限制。"""
    columns = {row[0]: row[1:] for row in before["tables"][BASE]["columns"]}
    if not EXPECTED - {CONFIGURATION} <= columns.keys():
        raise ValueError("知识库缺少其他必要字段，拒绝扩大修复范围")
    if columns.keys() - EXPECTED - {LEGACY}:
        raise ValueError("知识库出现未知字段，需先人工核对")
    if CONFIGURATION in columns and columns[CONFIGURATION][0] != "JSON":
        raise ValueError("既有配置列不是规范原生 JSON 类型")
    if LEGACY in columns and columns[LEGACY][:2] != ["RAW", 16]:
        raise ValueError("弃用绑定列的类型不符合已核对的结构")
    plan = []
    if CONFIGURATION not in columns:
        plan.extend((ADD, FILL, REQUIRE))
    elif columns[CONFIGURATION][-1] == "Y":
        plan.extend((FILL, REQUIRE))
    if LEGACY in columns and columns[LEGACY][-1] == "N":
        plan.append(RELAX)
    return plan


def row_fingerprint(rows, names):
    return digest(sorted(serialized({key: row[key] for key in names}) for row in rows))


def verify_unchanged(before, after):
    """现有字段逐值保持，包括旧 Collection ID、版本号、模型配置和所有关联。"""
    if before["target"] != after["target"]:
        raise ValueError("数据库目标发生变化")
    for table in (BASE, *PROTECTED):
        old, new = before["tables"][table], after["tables"][table]
        names = [column[0] for column in old["columns"]
            if table != BASE or column[0] != CONFIGURATION]
        if row_fingerprint(old["rows"], names) != row_fingerprint(new["rows"], names):
            raise ValueError("现有业务记录发生变化：" + table)
        if table in PROTECTED and old["columns"] != new["columns"]:
            raise ValueError("关联表结构发生变化")
    # 中断后重跑只允许把 NULL 初始化为 {}，绝不覆盖用户已有的配置。
    new_rows = {serialized(row["KNOWLEDGE_BASE_ID"]): row for row in after["tables"][BASE]["rows"]}
    for row in before["tables"][BASE]["rows"]:
        if row.get(CONFIGURATION) is not None:
            current = new_rows[serialized(row["KNOWLEDGE_BASE_ID"])][CONFIGURATION]
            if serialized(row[CONFIGURATION]) != serialized(current):
                raise ValueError("已有模型配置发生变化")
    if build_plan(after):
        raise ValueError("知识库修复仍未完成")
    if any(row[CONFIGURATION] is None for row in after["tables"][BASE]["rows"]):
        raise ValueError("知识库仍存在空配置")


def save_private(directory, name, value):
    descriptor = os.open(directory / name,
        os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(serialized(value))
        output.flush()
        os.fsync(output.fileno())


def run(args):
    from platform_core.config.settings import get_oracle_config
    import oracledb

    canonical = Path("database/oracle/aiops_agent/010_ops_knowledge.sql").read_text()
    if "CONFIGURATION_JSON JSON NOT NULL" not in canonical:
        raise ValueError("当前工作目录不是已核对的规范仓库")
    config = get_oracle_config()
    with oracledb.connect(user=config.username, password=config.require_password(),
            dsn=oracledb.makedsn(config.host, config.port, service_name=config.service_name),
            tcp_connect_timeout=10) as connection:
        connection.call_timeout = 15000
        # 定向 DDL 工具不复用修改前的语句结果缓存。
        connection.stmtcachesize = 0
        before = snapshot(connection)
        statements = build_plan(before)
        token = digest({"before": before, "statements": statements})
        summary = {"scope": BASE, "target_fingerprint": before["target"],
            "plan_hash": token, "statements": statements, "apply": args.apply,
            "preserved_rows": {name: len(data["rows"]) for name, data in before["tables"].items()}}
        print(serialized(summary), flush=True)
        if not args.apply:
            return
        if token != args.expected_plan_hash or not args.report_dir:
            raise ValueError("执行必须提供匹配的只读计划指纹和私有审计目录")
        directory = args.report_dir.resolve()
        if args.report_dir.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o077:
            raise ValueError("审计目录必须已存在、不可为符号链接且权限为 0700")
        save_private(directory, "before.json", before)
        save_private(directory, "plan.json", summary)
        completed = []
        try:
            cursor = connection.cursor()
            cursor.execute("ALTER SESSION SET DDL_LOCK_TIMEOUT=5")
            for statement in statements:
                if statement not in (ADD, FILL, REQUIRE, RELAX):
                    raise ValueError("SQL 超出固定修复白名单")
                cursor.execute(statement)
                completed.append(statement)
            after = snapshot(connection)
            verify_unchanged(before, after)
            save_private(directory, "after.json", after)
            save_private(directory, "completed.json", {"statements": completed, "verified": True})
            print(serialized({"verified": True, "statement_count": len(completed),
                "existing_data_unchanged": True, "models_auto_bound": False}), flush=True)
        except Exception:
            # Oracle DDL 自动提交，失败只记录已执行步骤，不能声称整体回滚。
            save_private(directory, "partial.json", {"statements": completed, "verified": False})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-plan-hash")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        # 不输出可能包含数据库连接信息的异常正文。
        print("知识库定向修复未完成：" + type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
