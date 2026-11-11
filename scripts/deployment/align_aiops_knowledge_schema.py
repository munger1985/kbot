"""AIOps 知识结构定向对齐；显式批准后删除旧解析，不备份、不迁移、不重新解析。"""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


PREFIX = "KBOT_OPS_KNOWLEDGE_"
BASE, ASSET, VERSION, INDEX = (PREFIX + name for name in ("BASE", "ASSET", "VERSION", "INDEX"))
BINDING = "KBOT_OPS_AGENT_VERSION_KNOW_BASE"
CHILDREN = tuple(PREFIX + name for name in ("SCOPE", "SOURCE", "INDEX", "REVIEW"))
LEGACY = {"COLLECTION_ID", "BUNDLE_ID", "BUNDLE_REVISION_ID"}
DOCUMENT = {"DOCUMENT_ID", "DOCUMENT_VERSION_ID", "INGESTION_JOB_ID", "PARSE_RUN_ID", "INDEX_GENERATION_ID"}
DROP_BASE = f"ALTER TABLE {BASE} DROP COLUMN COLLECTION_ID"
DROP_INDEX = f"ALTER TABLE {INDEX} DROP (COLLECTION_ID, BUNDLE_ID, BUNDLE_REVISION_ID)"
ADD_INDEX = f"ALTER TABLE {INDEX} ADD (DOCUMENT_ID RAW(16) NOT NULL, " \
    "DOCUMENT_VERSION_ID RAW(16) NOT NULL, INGESTION_JOB_ID RAW(16) NOT NULL, " \
    "PARSE_RUN_ID RAW(16), INDEX_GENERATION_ID RAW(16))"
RESET_ASSET = f"UPDATE {ASSET} SET CURRENT_VERSION_ID=NULL, STATUS='DRAFT', " \
    "UPDATED_BY='system:knowledge-schema-alignment', UPDATED_AT=SYSTIMESTAMP, " \
    "ROW_VERSION=ROW_VERSION+1 WHERE ASSET_ID=:id AND ROW_VERSION=:expected"
DELETE = {table: f"DELETE FROM {table} WHERE ASSET_VERSION_ID=:id" for table in (*CHILDREN, VERSION)}


def encode(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def digest(value):
    return sha256(json.dumps(value, default=encode, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_type(value):
    value = value.upper().replace("NUMERIC", "NUMBER").replace(" ", "").replace("TIMESTAMP(6)", "TIMESTAMP")
    value = re.sub(r"NUMBER\((\d+)\)", r"NUMBER(\1,0)", value)
    return re.sub(r"VARCHAR2\((\d+)\)", r"VARCHAR2(\1BYTE)", value)


def canonical(root):
    """只接受当前清单中两份规范 SQL 的固定工件，不接受外部 SQL。"""
    directory = root / "database/oracle/aiops_agent"
    manifest = json.loads((directory / "schema_manifest.json").read_text())
    scripts = {item["name"]: item["sha256"] for item in manifest["scripts"]}
    statements, tables, indexes, constraints = [], {}, {}, {}
    for name in ("010_ops_knowledge.sql", "014_ops_knowledge_ingestion.sql"):
        raw = (directory / name).read_bytes()
        if sha256(raw).hexdigest() != scripts[name]:
            raise ValueError("规范 SQL 指纹不一致")
        statements.extend(part.strip() for part in re.sub(r"--[^\n]*", "", raw.decode()).split(";") if part.strip())
    for statement in statements:
        constraint = re.match(r"ALTER TABLE (\w+) ADD CONSTRAINT (\w+) FOREIGN KEY", " ".join(statement.split()))
        if constraint:
            constraints[constraint[2]] = {"table": constraint[1], "sql": statement}
            continue
        match = re.match(r"CREATE TABLE (\w+)\s*\(", statement)
        if match:
            name = match[1]
            if not (name.startswith(PREFIX) or name == BINDING):
                raise ValueError("规范表越出 AIOps 知识范围")
            columns = {}
            for line in statement.splitlines()[1:]:
                column = re.match(r"\s*(\w+)\s+((?:RAW|NUMBER|NUMERIC|VARCHAR2)\([^)]*\)|JSON|CLOB|TIMESTAMP(?:\(6\))?(?: WITH TIME ZONE)?)(.*)", line)
                if column:
                    columns[column[1]] = [normalize_type(column[2]), "N" if "NOT NULL" in column[3] or "PRIMARY KEY" in column[3] else "Y"]
            # 表级主键列在规范中均已显式声明 NOT NULL。
            tables[name] = {"sql": statement, "columns": columns}
        else:
            match = re.match(r"CREATE INDEX (\w+)\s+ON (\w+)\s*\(", statement)
            if not match:
                raise ValueError("规范含未知语句")
            indexes[match[1]] = {"table": match[2], "sql": statement}
    if len(tables) != 28 or len(constraints) != 3 or any(
            item["table"] not in tables for item in (*indexes.values(), *constraints.values())):
        raise ValueError("知识表清单与已批准范围不一致")
    return tables, indexes, constraints


def columns(cursor, table):
    cursor.execute("SELECT COLUMN_NAME, DATA_TYPE, DATA_PRECISION, DATA_SCALE, CHAR_LENGTH, "
        "CHAR_USED, DATA_LENGTH, NULLABLE FROM USER_TAB_COLUMNS WHERE TABLE_NAME=:name ORDER BY COLUMN_NAME", name=table)
    result = {}
    for name, kind, precision, scale, chars, unit, length, nullable in cursor:
        if kind == "NUMBER" and precision is not None:
            kind += f"({precision},{scale or 0})"
        elif kind == "VARCHAR2":
            kind += f"({chars}{'CHAR' if unit == 'C' else 'BYTE'})"
        elif kind == "RAW":
            kind += f"({length})"
        result[name] = [normalize_type(kind), nullable]
    return result


def row_hash(cursor, table, names):
    projection = ", ".join('"' + name + '"' for name in sorted(names))
    cursor.execute("SELECT " + projection + " FROM " + table)
    return digest(sorted(digest(list(row)) for row in cursor))


def inventory(connection, tables, indexes, constraints):
    """仅保留结构、数量及内存指纹；不生成备份或持久化业务内容。"""
    cursor = connection.cursor()
    cursor.execute("SELECT SYS_CONTEXT('USERENV','DB_UNIQUE_NAME'), SYS_CONTEXT('USERENV','CON_NAME'), "
        "SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM DUAL")
    result = {"target": digest(list(cursor.fetchone())), "columns": {}, "indexes": [], "constraints": [],
        "counts": {}, "versions": [], "assets": [], "asset_versions": {}, "protected": {}}
    for table in tables:
        result["columns"][table] = columns(cursor, table)
        if result["columns"][table]:
            cursor.execute("SELECT COUNT(*) FROM " + table)
            result["counts"][table] = cursor.fetchone()[0]
    for name in indexes:
        cursor.execute("SELECT COUNT(*) FROM USER_INDEXES WHERE INDEX_NAME=:name", name=name)
        if cursor.fetchone()[0]:
            result["indexes"].append(name)
    for name in constraints:
        cursor.execute("SELECT COUNT(*) FROM USER_CONSTRAINTS WHERE CONSTRAINT_NAME=:name AND STATUS='ENABLED'", name=name)
        if cursor.fetchone()[0]:
            result["constraints"].append(name)
    actual = result["columns"][INDEX]
    legacy = LEGACY <= actual.keys() and not DOCUMENT & actual.keys()
    if legacy:
        cursor.execute(f"SELECT DISTINCT ASSET_VERSION_ID FROM {INDEX} ORDER BY ASSET_VERSION_ID")
        result["versions"] = [row[0] for row in cursor]
        cursor.execute(f"SELECT DISTINCT v.ASSET_ID FROM {VERSION} v JOIN {INDEX} i "
            "ON i.ASSET_VERSION_ID=v.ASSET_VERSION_ID ORDER BY v.ASSET_ID")
        result["assets"] = [row[0] for row in cursor]
        # 只允许此次已确认的手册解析；遇到混合新旧历史或其他业务类型时拒绝扩大删除。
        cursor.execute(f"SELECT COUNT(*) FROM {VERSION} v WHERE NOT EXISTS "
            f"(SELECT 1 FROM {INDEX} i WHERE i.ASSET_VERSION_ID=v.ASSET_VERSION_ID)")
        if cursor.fetchone()[0]:
            raise ValueError("存在无旧索引版本，拒绝扩大清理范围")
        cursor.execute(f"SELECT COUNT(*) FROM {ASSET} WHERE ASSET_KIND<>'MANUAL'")
        if cursor.fetchone()[0]:
            raise ValueError("存在非手册资产，需单独批准清理范围")
    for table in (BASE, BINDING):
        result["protected"][table] = row_hash(cursor, table, tables[table]["columns"])
    stable_asset = tables[ASSET]["columns"].keys() - {"CURRENT_VERSION_ID", "STATUS", "UPDATED_BY", "UPDATED_AT", "ROW_VERSION"}
    result["protected"][ASSET] = row_hash(cursor, ASSET, stable_asset)
    cursor.execute(f"SELECT ASSET_ID, ROW_VERSION FROM {ASSET} ORDER BY ASSET_ID")
    result["asset_versions"] = {identity.hex(): int(version) for identity, version in cursor}
    return result


def build_plan(before, tables, indexes, constraints):
    """新增缺少表/索引，只有已核对的旧列允许删除。"""
    creation, alteration = [], []
    for table, expected in tables.items():
        actual = before["columns"][table]
        if not actual:
            if table in (BASE, ASSET, VERSION, INDEX, BINDING, *CHILDREN):
                raise ValueError("缺少旧业务基础表，拒绝当作空库初始化")
            creation.append(expected["sql"])
            continue
        desired = expected["columns"]
        if table == BASE and set(actual) == set(desired) | {"COLLECTION_ID"}:
            alteration.append(DROP_BASE)
            actual = {key: value for key, value in actual.items() if key != "COLLECTION_ID"}
        elif table == INDEX and set(actual) == (set(desired) - DOCUMENT) | LEGACY:
            alteration.extend((DROP_INDEX, ADD_INDEX))
            actual = {key: value for key, value in actual.items() if key not in LEGACY}
            desired = {key: value for key, value in desired.items() if key not in DOCUMENT}
        elif table == INDEX and set(actual) == set(desired) - DOCUMENT and not before["counts"][INDEX]:
            # 旧列已删除但新增列尚未完成的中断态：空表才允许接着补齐。
            alteration.append(ADD_INDEX)
            desired = {key: value for key, value in desired.items() if key not in DOCUMENT}
        if actual != desired:
            difference = {name: {"actual": actual.get(name), "expected": desired.get(name)}
                for name in sorted(actual.keys() | desired.keys()) if actual.get(name) != desired.get(name)}
            print(json.dumps({"table": table, "unsupported_column_differences": difference}), flush=True)
            raise ValueError("存在未批准的结构差异：" + table)
    creation.extend(item["sql"] for name, item in constraints.items() if name not in before["constraints"])
    creation.extend(item["sql"] for name, item in indexes.items() if name not in before["indexes"])
    return {"create": creation, "alter": alteration, "discard_versions": len(before["versions"]),
        "reset_assets": len(before["assets"])}


def clean_legacy(connection, before):
    """删除已定位版本的派生记录；保留资料容器、库和全部 Agent 绑定。"""
    cursor = connection.cursor()
    for table in (ASSET, VERSION, *CHILDREN):
        cursor.execute("LOCK TABLE " + table + " IN EXCLUSIVE MODE NOWAIT")
    cursor.execute(f"SELECT DISTINCT ASSET_VERSION_ID FROM {INDEX} ORDER BY ASSET_VERSION_ID")
    if [row[0] for row in cursor] != before["versions"]:
        raise ValueError("待清理版本在预演后发生变化")
    deleted = {}
    for identity in before["assets"]:
        cursor.execute(RESET_ASSET, id=identity, expected=before["asset_versions"][identity.hex()])
        if cursor.rowcount != 1:
            raise ValueError("资料条目已变化")
    for table in (*CHILDREN, VERSION):
        deleted[table] = 0
        for identity in before["versions"]:
            cursor.execute(DELETE[table], id=identity)
            deleted[table] += cursor.rowcount
        if deleted[table] != before["counts"][table]:
            raise ValueError("删除数量与已批准计划不一致")
    connection.commit()
    return deleted


def run(args):
    from platform_core.config.settings import get_oracle_config
    import oracledb

    tables, indexes, constraints = canonical(Path.cwd())
    config = get_oracle_config()
    with oracledb.connect(user=config.username, password=config.require_password(),
            dsn=oracledb.makedsn(config.host, config.port, service_name=config.service_name),
            tcp_connect_timeout=10) as connection:
        connection.call_timeout, connection.stmtcachesize = 15000, 0
        before = inventory(connection, tables, indexes, constraints)
        plan = build_plan(before, tables, indexes, constraints)
        token = digest({"before": before, "plan": plan})
        print(json.dumps({"target_fingerprint": before["target"], "plan_hash": token,
            "create_statements": len(plan["create"]), "alter_statements": plan["alter"],
            "discard_versions": plan["discard_versions"], "reset_assets": plan["reset_assets"],
            "legacy_rows": {table: before["counts"][table] for table in (*CHILDREN, VERSION)},
            "backup": False, "apply": args.apply}), flush=True)
        if not args.apply:
            return
        if not args.discard_legacy_parses or token != args.expected_plan_hash:
            raise ValueError("执行必须显式批准删除旧解析并提供匹配的预演指纹")
        completed, deleted = [], {}
        try:
            cursor = connection.cursor()
            cursor.execute("ALTER SESSION SET DDL_LOCK_TIMEOUT=5")
            for statement in plan["create"]:
                cursor.execute(statement)
                completed.append(statement.split("(", 1)[0].strip())
            if before["versions"]:
                deleted = clean_legacy(connection, before)
            for statement in plan["alter"]:
                cursor.execute(statement)
                completed.append(statement)
            after = inventory(connection, tables, indexes, constraints)
            remaining = build_plan(after, tables, indexes, constraints)
            if (after["target"] != before["target"] or after["protected"] != before["protected"]
                or remaining["create"] or remaining["alter"] or remaining["discard_versions"]):
                raise ValueError("结构或受保护记录验收失败")
            print(json.dumps({"verified": True, "tables_aligned": len(tables),
                "deleted_rows": deleted, "preserved_counts": {name: after["counts"][name]
                    for name in (BASE, ASSET, BINDING)}, "backup": False}), flush=True)
        except Exception:
            connection.rollback()
            # DDL 和已提交清理不可回滚，不伪称原子恢复；只输出语句名称和数量。
            print(json.dumps({"verified": False, "completed": completed,
                "deleted_rows": deleted, "backup": False}), flush=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--discard-legacy-parses", action="store_true")
    parser.add_argument("--expected-plan-hash")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print("AIOps 知识结构对齐未完成：" + type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
