"""显式对齐 Oracle 资源采集 Run 的两列可空约束；默认只读，不进入运行时。"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys


TABLE = "KBOT_OPS_RUN"
AGENT_COLUMNS = ("AGENT_ID", "AGENT_VERSION_ID")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rows(cursor, sql):
    cursor.execute(sql)
    return [list(row) for row in cursor]


def snapshot(connection):
    """仅取固定表元数据及历史主键指纹，不读取诊断正文、凭据或连接配置。"""
    with connection.cursor() as cursor:
        identity = rows(cursor, "SELECT SYS_CONTEXT('USERENV','DB_UNIQUE_NAME'), "
            "SYS_CONTEXT('USERENV','CON_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA'), "
            "SYS_CONTEXT('USERENV','SESSION_USER') FROM DUAL")[0]
        if not all(identity) or identity[2] != identity[3]:
            raise ValueError("必须使用 Schema Owner 连接固定目标")
        columns = rows(cursor, "SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE "
            "FROM USER_TAB_COLUMNS WHERE TABLE_NAME='KBOT_OPS_RUN' ORDER BY COLUMN_NAME")
        constraints = rows(cursor, "SELECT CONSTRAINT_NAME, CONSTRAINT_TYPE, STATUS, VALIDATED, "
            "DEFERRABLE, DEFERRED, R_OWNER, R_CONSTRAINT_NAME, DELETE_RULE, SEARCH_CONDITION_VC "
            "FROM USER_CONSTRAINTS WHERE TABLE_NAME='KBOT_OPS_RUN' ORDER BY CONSTRAINT_NAME")
        indexes = rows(cursor, "SELECT INDEX_NAME, UNIQUENESS, STATUS FROM USER_INDEXES "
            "WHERE TABLE_NAME='KBOT_OPS_RUN' ORDER BY INDEX_NAME")
        index_columns = rows(cursor, "SELECT INDEX_NAME, COLUMN_NAME, COLUMN_POSITION, DESCEND "
            "FROM USER_IND_COLUMNS WHERE TABLE_NAME='KBOT_OPS_RUN' ORDER BY INDEX_NAME,COLUMN_POSITION")
        run_ids = rows(cursor, "SELECT RAWTOHEX(OPS_RUN_ID) FROM KBOT_OPS_RUN ORDER BY OPS_RUN_ID")
    return {"target_fingerprint": digest(identity), "columns": columns,
        "constraints": constraints, "indexes": indexes, "index_columns": index_columns,
        "run_count": len(run_ids), "run_ids_hash": digest(run_ids)}


def build_plan(before):
    """只允许已存在的 RAW(16) Agent 列放宽空值；已可空时不重复执行 DDL。"""
    columns = {row[0]: row for row in before["columns"]}
    pending = []
    for name in AGENT_COLUMNS:
        column = columns.get(name)
        if column is None or column[1:3] != ["RAW", 16] or column[3] not in {"Y", "N"}:
            raise ValueError("目标 Agent 列缺失或类型不符合规范")
        if column[3] == "N":
            pending.append(name + " NULL")
    return ["ALTER TABLE " + TABLE + " MODIFY (" + ", ".join(pending) + ")"] if pending else []


def nullable_constraint(row):
    """Oracle 列级 NOT NULL 的目录表达式；其他 CHECK、主外键全部保留。"""
    expression = re.sub(r"\s+", " ", (row[-1] or "").strip()).upper()
    return row[1] == "C" and expression in {
        '"AGENT_ID" IS NOT NULL', '"AGENT_VERSION_ID" IS NOT NULL',
    }


def verify(before, after):
    expected = dict(before)
    expected["columns"] = [row[:3] + ["Y"] if row[0] in AGENT_COLUMNS else row
        for row in before["columns"]]
    expected["constraints"] = [row for row in before["constraints"] if not nullable_constraint(row)]
    if expected != after or build_plan(after):
        raise ValueError("定向对齐后校验失败：目标、历史 Run 或受保护元数据不一致")


def save_private(directory, name, value):
    descriptor = os.open(directory / name,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())


def align(connection, *, apply=False, expected_plan_hash=None, report_dir=None):
    before = snapshot(connection)
    statements = build_plan(before)
    token = digest({"before": before, "statements": statements})
    summary = {"scope": "aiops_resource_collection_agent_nullability", "plan_hash": token,
        "target_fingerprint": before["target_fingerprint"], "run_count": before["run_count"],
        "statements": statements, "apply": apply}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not apply:
        return summary
    if report_dir is None or expected_plan_hash != token:
        raise ValueError("写入需要私有审计目录及完全匹配的只读计划指纹")
    directory = Path(report_dir)
    if (directory.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o077
            or directory.stat().st_uid != os.getuid()):
        raise ValueError("审计目录必须归当前用户所有、权限为 0700 且不是符号链接")
    save_private(directory, "before.json", before)
    save_private(directory, "plan.json", summary)
    completed = []
    try:
        with connection.cursor() as cursor:
            cursor.execute("ALTER SESSION SET DDL_LOCK_TIMEOUT=10")
            for statement in statements:
                cursor.execute(statement)
                completed.append(statement)
        after = snapshot(connection)
        save_private(directory, "after.json", after)
        verify(before, after)
        save_private(directory, "completed.json", {"statements": completed, "verified": True})
        print(json.dumps({"verified": True, "ddl_count": len(completed),
            "historical_run_ids_preserved": True, "other_constraints_indexes_preserved": True}), flush=True)
    except Exception:
        # Oracle DDL 隐式提交；失败时记录实际进度，不声称已经回滚。
        save_private(directory, "partial.json", {"statements": completed, "verified": False})
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-plan-hash")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    try:
        from aiops_agent.config import get_aiops_settings
        import oracledb

        config = get_aiops_settings().database.oracle
        with oracledb.connect(user=config.username, password=config.require_password(),
                dsn=oracledb.makedsn(config.host, config.port, service_name=config.service_name),
                tcp_connect_timeout=10) as connection:
            connection.call_timeout = 15000
            align(connection, apply=args.apply, expected_plan_hash=args.expected_plan_hash,
                report_dir=args.report_dir)
    except Exception as exc:
        # 不输出可能包含连接身份或密码的异常正文。
        print("资源采集列约束对齐未完成：" + type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
