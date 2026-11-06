"""知识检索 Oracle 非破坏式升级；默认只读，不处理旧 KC 数据或 AIOps 表。"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = "KBOT_KR_KNOWLEDGE_LIBRARY"
VERSION = "KBOT_KR_AGENT_VERSION"
ADD_LIBRARY = "ALTER TABLE KBOT_KR_AGENT_VERSION ADD (LIBRARY_ID RAW(16))"
DEFAULT_RETIRED = "ALTER TABLE KBOT_KR_AGENT_VERSION MODIFY (DO_RERANK DEFAULT 0)"


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()


def canonical():
    """只加载固定规范，不接受调用者提供 DDL 或目标表。"""
    directory = ROOT / "database/oracle/knowledge_retrieval_app"
    manifest = json.loads((directory / "schema_manifest.json").read_text())
    for script in manifest["scripts"]:
        raw = (directory / script["name"]).read_bytes()
        if sha256(raw).hexdigest() != script["sha256"]:
            raise ValueError("规范 SQL 指纹不一致")
    sql = (directory / "004_knowledge_library.sql").read_text()
    sql = re.sub(r"--[^\n]*", "", sql)
    statements = [part.strip() for part in sql.split(";") if part.strip()]
    if (len(statements) != 2 or
            not statements[0].startswith("CREATE TABLE " + LIBRARY + " (") or
            statements[1] != "CREATE INDEX IX_KR_LIBRARY_DOMAIN ON " + LIBRARY + " (domain_id, status)"):
        raise ValueError("资料库规范 DDL 越出固定范围")
    return manifest, statements


def execute_rows(cursor, sql, **parameters):
    cursor.execute(sql, parameters)
    return [list(row) for row in cursor]


def snapshot(connection):
    """只读元数据及最小绑定；不读取文档正文、凭据或 AIOps 业务数据。"""
    cursor = connection.cursor()
    tables = [row[0] for row in execute_rows(cursor,
        "SELECT TABLE_NAME FROM USER_TABLES WHERE TABLE_NAME LIKE 'KBOT_KR_%' ORDER BY TABLE_NAME")]
    columns = execute_rows(cursor,
        "SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE FROM USER_TAB_COLUMNS "
        "WHERE TABLE_NAME=:name ORDER BY COLUMN_NAME", name=VERSION)
    names = {row[0] for row in columns}
    original = "KNOWLEDGE_CORE_ID" if "KNOWLEDGE_CORE_ID" in names else None
    selected = ["AGENT_VERSION_ID"] + ([original] if original else []) + (
        ["DO_RERANK"] if "DO_RERANK" in names else [])
    bindings = execute_rows(cursor, "SELECT " + ", ".join(selected) + " FROM " + VERSION +
        " ORDER BY AGENT_VERSION_ID")
    bindings = [[value.hex() if isinstance(value, bytes) else value for value in row] for row in bindings]
    retired_default = None
    if "DO_RERANK" in names:
        cursor.execute("SELECT DATA_DEFAULT FROM USER_TAB_COLUMNS WHERE TABLE_NAME=:name "
            "AND COLUMN_NAME='DO_RERANK'", name=VERSION)
        value = cursor.fetchone()[0]
        retired_default = value.strip() if value is not None else None
    library_columns = execute_rows(cursor, "SELECT COLUMN_NAME FROM USER_TAB_COLUMNS "
        "WHERE TABLE_NAME=:name ORDER BY COLUMN_NAME", name=LIBRARY)
    library_index = execute_rows(cursor, "SELECT COLUMN_NAME FROM USER_IND_COLUMNS "
        "WHERE INDEX_NAME=\'IX_KR_LIBRARY_DOMAIN\' ORDER BY COLUMN_POSITION")
    protected = {}
    for prefix in ("KBOT_OPS_", "KBOT_KC_"):
        # 元数据指纹只是部署前后比较，不加入服务启动或 readiness。
        values = []
        for query in (
            "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE "
            "FROM USER_TAB_COLUMNS WHERE TABLE_NAME LIKE :prefix ORDER BY TABLE_NAME,COLUMN_NAME",
            "SELECT TABLE_NAME, CONSTRAINT_NAME, CONSTRAINT_TYPE, STATUS "
            "FROM USER_CONSTRAINTS WHERE TABLE_NAME LIKE :prefix ORDER BY TABLE_NAME,CONSTRAINT_NAME",
            "SELECT TABLE_NAME, INDEX_NAME, UNIQUENESS, STATUS "
            "FROM USER_INDEXES WHERE TABLE_NAME LIKE :prefix ORDER BY TABLE_NAME,INDEX_NAME"):
            values.append(execute_rows(cursor, query, prefix=prefix + "%"))
        protected[prefix] = digest(values)
    cursor.execute("SELECT SYS_CONTEXT('USERENV','DB_UNIQUE_NAME'), "
        "SYS_CONTEXT('USERENV','CON_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM DUAL")
    identity = digest(list(cursor.fetchone()))
    return {"tables": tables, "version_columns": columns, "binding_fields": selected,
        "bindings": bindings, "retired_default": retired_default,
        "library_columns": library_columns, "library_index": library_index,
        "protected_metadata": protected, "target_fingerprint": identity}


def build_plan(before, manifest, library_statements):
    """仅补充新结构；旧集合 ID 不自动成为新库 ID，旧绑定不被清空。"""
    required = set(manifest["tables"]) - {LIBRARY}
    if not required <= set(before["tables"]):
        raise ValueError("缺少基础表，请先执行固定 K2 定向建表工具")
    columns = {row[0]: row for row in before["version_columns"]}
    if "LIBRARY_ID" in columns and columns["LIBRARY_ID"][1:] != ["RAW", 16, "Y"]:
        raise ValueError("既有 LIBRARY_ID 类型或空值约束不符合规范")
    if LIBRARY in before["tables"]:
        expected = {"LIBRARY_ID", "DOMAIN_ID", "DISPLAY_NAME", "DESCRIPTION", "STATUS", "GRANTS_JSON",
            "CONFIGURATION_JSON", "CREATED_BY", "CREATED_AT", "UPDATED_AT", "ROW_VERSION"}
        if {row[0] for row in before["library_columns"]} != expected:
            raise ValueError("既有资料库结构与规范不一致")
        if before["library_index"] not in ([], [["DOMAIN_ID"], ["STATUS"]]):
            raise ValueError("既有资料库索引与规范不一致")
        statements = [] if before["library_index"] else [library_statements[1]]
    else:
        statements = list(library_statements)
    if "LIBRARY_ID" not in columns:
        statements.append(ADD_LIBRARY)
    if "DO_RERANK" in columns and before["retired_default"] is None:
        # 保留已有值，仅避免弃用的非空列阻断新版本 INSERT；新运行路径不读取它。
        statements.append(DEFAULT_RETIRED)
    return statements


def verify_unchanged(before, after):
    if before["target_fingerprint"] != after["target_fingerprint"]:
        raise ValueError("数据库目标发生变化")
    if before["protected_metadata"] != after["protected_metadata"]:
        raise ValueError("受保护 KC/AIOps 元数据发生变化")
    if before["binding_fields"] != after["binding_fields"] or before["bindings"] != after["bindings"]:
        raise ValueError("旧绑定或弃用列的既有值发生变化")
    if set(after["tables"]) - set(before["tables"]) not in (set(), {LIBRARY}):
        raise ValueError("出现计划外新表")
    if not set(before["tables"]) <= set(after["tables"]):
        raise ValueError("既有表消失")


def save_private(directory, name, value):
    """写入私有审计工件，绝不在终端输出历史绑定。"""
    path = directory / name
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())


def run(args):
    from platform_core.config.settings import get_oracle_config
    import oracledb

    manifest, library_statements = canonical()
    config = get_oracle_config()
    with oracledb.connect(user=config.username, password=config.require_password(),
            dsn=oracledb.makedsn(config.host, config.port, service_name=config.service_name),
            tcp_connect_timeout=10) as connection:
        connection.call_timeout = 15000
        before = snapshot(connection)
        statements = build_plan(before, manifest, library_statements)
        token = digest({"before": before, "manifest": manifest, "statements": statements})
        summary = {"scope": "knowledge_retrieval_only", "plan_hash": token,
            "target_fingerprint": before["target_fingerprint"],
            "statements": statements, "agent_versions_preserved": len(before["bindings"]),
            "legacy_nonnull_bindings": sum(row[before["binding_fields"].index("KNOWLEDGE_CORE_ID")] is not None
                for row in before["bindings"]) if "KNOWLEDGE_CORE_ID" in before["binding_fields"] else 0,
            "apply": args.apply}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if not args.apply:
            return
        if not args.report_dir or args.expected_plan_hash != token:
            raise ValueError("写入需要私有审计目录及完全匹配的只读计划指纹")
        directory = args.report_dir.resolve()
        if (not directory.is_dir() or args.report_dir.is_symlink() or
                directory.stat().st_mode & 0o077):
            raise ValueError("审计目录必须是现有私有目录且不能是符号链接")
        save_private(directory, "before.json", before)
        save_private(directory, "plan.json", summary)
        cursor = connection.cursor()
        cursor.execute("ALTER SESSION SET DDL_LOCK_TIMEOUT=10")
        completed = []
        try:
            for statement in statements:
                if statement not in library_statements + [ADD_LIBRARY, DEFAULT_RETIRED]:
                    raise ValueError("DDL 越出白名单")
                cursor.execute(statement)
                completed.append(statement)
            after = snapshot(connection)
            verify_unchanged(before, after)
            if build_plan(after, manifest, library_statements):
                raise ValueError("升级后仍存在缺少的结构")
            save_private(directory, "after.json", after)
            save_private(directory, "completed.json", {"statements": completed, "verified": True})
            print(json.dumps({"verified": True, "ddl_count": len(completed),
                "protected_kc_aiops_unchanged": True, "original_bindings_unchanged": True,
                "new_library_bindings_migrated": False}), flush=True)
        except Exception:
            # Oracle DDL 自动提交，逐条记录已完成动作，不伪称事务回滚。
            save_private(directory, "partial.json", {"completed": completed, "verified": False})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--expected-plan-hash")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print("知识检索 Schema 升级未完成：" + type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
