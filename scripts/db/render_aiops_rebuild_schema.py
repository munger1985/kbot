"""从规范 DDL 生成 SQL Developer 可直接执行的 AIOps 重建脚本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"
MANIFEST_PATH = SCHEMA_DIR / "schema_manifest.json"
OUTPUT_PATH = SCHEMA_DIR / "rebuild_aiops_schema.sql"

HEADER = """-- KBot 4.0 AIOps Schema 全量重建脚本。
-- 本文件由 scripts/db/render_aiops_rebuild_schema.py 生成，请勿手工复制规范 DDL。
-- 使用 KBot Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- 本脚本永久删除当前 Schema 内全部 KBOT_OPS_% 表、KBOT_V_OPS_% 视图及其数据。
-- 执行前必须停止 AIOps API、Worker、Scheduler，并备份需要保留的数据。
-- 平台用户、Domain、权限、角色以及 KC Collection 不在删除范围内。
-- Oracle DDL 会自动提交；失败后应修复原因并重新执行本脚本。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

SET SERVEROUTPUT ON
SET VERIFY OFF

PROMPT === 正在删除旧 AIOps 视图和表 ===

DECLARE
BEGIN
    FOR view_row IN (
        SELECT view_name
        FROM user_views
        WHERE view_name LIKE 'KBOT\\_V\\_OPS\\_%' ESCAPE '\\'
        ORDER BY view_name
    ) LOOP
        EXECUTE IMMEDIATE
            'DROP VIEW ' || dbms_assert.enquote_name(view_row.view_name, FALSE);
        dbms_output.put_line('已删除视图 ' || view_row.view_name);
    END LOOP;

    FOR table_row IN (
        SELECT table_name
        FROM user_tables
        WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
        ORDER BY table_name
    ) LOOP
        EXECUTE IMMEDIATE
            'DROP TABLE '
            || dbms_assert.enquote_name(table_row.table_name, FALSE)
            || ' CASCADE CONSTRAINTS PURGE';
        dbms_output.put_line('已删除表 ' || table_row.table_name);
    END LOOP;
END;
/

PROMPT === 正在执行当前规范 AIOps DDL ===
"""

FOOTER = """
PROMPT === 正在验证 AIOps Schema ===

DECLARE
    l_table_count PLS_INTEGER;
    l_view_count PLS_INTEGER;
    l_invalid_count PLS_INTEGER;
    l_bad_constraint_count PLS_INTEGER;
    l_bad_index_count PLS_INTEGER;
    l_workflow_kind_count PLS_INTEGER;
    l_component VARCHAR2(32);
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64);
BEGIN
    SELECT COUNT(*)
      INTO l_table_count
      FROM user_tables
     WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\';

    SELECT COUNT(*)
      INTO l_view_count
      FROM user_views
     WHERE view_name LIKE 'KBOT\\_V\\_OPS\\_%' ESCAPE '\\';

    SELECT COUNT(*)
      INTO l_invalid_count
      FROM user_objects
     WHERE (
            object_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
         OR object_name LIKE 'KBOT\\_V\\_OPS\\_%' ESCAPE '\\'
     )
       AND object_type IN ('TABLE', 'VIEW', 'INDEX')
       AND status <> 'VALID';

    SELECT COUNT(*)
      INTO l_bad_constraint_count
      FROM user_constraints
     WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
       AND (status <> 'ENABLED' OR validated <> 'VALIDATED');

    SELECT COUNT(*)
      INTO l_bad_index_count
      FROM user_indexes
     WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
       AND status <> 'VALID';

    SELECT COUNT(*)
      INTO l_workflow_kind_count
      FROM user_tab_columns
     WHERE table_name = 'KBOT_OPS_RUN'
       AND column_name = 'WORKFLOW_KIND'
       AND nullable = 'N';

    SELECT component, schema_version, contract_version
      INTO l_component, l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF l_table_count <> {table_count} OR l_view_count <> {view_count} THEN
        raise_application_error(
            -20001,
            'AIOps 对象数量错误：表=' || l_table_count || '，视图=' || l_view_count
        );
    END IF;
    IF l_invalid_count <> 0 THEN
        raise_application_error(-20002, '存在无效的 AIOps 对象。');
    END IF;
    IF l_bad_constraint_count <> 0 THEN
        raise_application_error(-20003, '存在未启用或未验证的 AIOps 约束。');
    END IF;
    IF l_bad_index_count <> 0 THEN
        raise_application_error(-20004, '存在无效的 AIOps 索引。');
    END IF;
    IF l_workflow_kind_count <> 1 THEN
        raise_application_error(-20005, 'KBOT_OPS_RUN.WORKFLOW_KIND 缺失或允许为空。');
    END IF;
    IF l_component <> 'AIOPS'
       OR l_schema_version <> {schema_version}
       OR l_contract_version <> '{contract_version}' THEN
        raise_application_error(
            -20006,
            'AIOps Schema 合同错误：'
            || l_component || '/' || l_schema_version || '/' || l_contract_version
        );
    END IF;

    dbms_output.put_line(
        '验证通过：{table_count} 张表、{view_count} 个视图，Schema Version '
        || '{schema_version}，合同 {contract_version}。'
    );
END;
/

SELECT component, schema_version, contract_version
FROM KBOT_V_OPS_SCHEMA_VERSION;

PROMPT === AIOps Schema 重建完成；启动服务后检查 AIOps /ready ===
"""


def _load_manifest() -> dict:
    """读取并返回 AIOps Schema Manifest。"""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def render_rebuild_sql() -> str:
    """生成包含全部规范 DDL 的单文件重建脚本。"""
    manifest = _load_manifest()
    sections = [HEADER.rstrip()]
    for definition in manifest["scripts"]:
        name = str(definition["name"])
        path = SCHEMA_DIR / name
        content = path.read_text(encoding="utf-8")
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual_hash != definition["sha256"]:
            raise RuntimeError(f"规范 DDL 哈希与 Manifest 不一致：{name}")
        sections.extend(
            (
                f"-- ===== 开始规范 DDL：{name} =====",
                content.rstrip(),
                f"-- ===== 结束规范 DDL：{name} =====",
            )
        )
    sections.append(
        FOOTER.format(
            table_count=len(manifest["tables"]),
            view_count=len(manifest["views"]),
            schema_version=int(manifest["schema_version"]),
            contract_version=str(manifest["contract_version"]),
        ).strip()
    )
    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查已生成脚本是否与当前规范 DDL 一致",
    )
    args = parser.parse_args()
    rendered = render_rebuild_sql()
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print("AIOps 重建脚本已过期，请重新运行生成器。")
            return 1
        print("AIOps 重建脚本与当前规范 DDL 一致。")
        return 0
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"已生成 SQL Developer 单文件脚本：{OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
