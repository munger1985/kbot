"""从规范 DDL 生成 SQL Developer 可直接执行的 AIOps维护脚本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"
MANIFEST_PATH = SCHEMA_DIR / "schema_manifest.json"
OUTPUT_PATH = SCHEMA_DIR / "rebuild_aiops_schema.sql"
UPGRADE_OUTPUT_PATH = SCHEMA_DIR / "upgrade_aiops_v12_to_v13.sql"

PRESERVED_CONFIGURATION_TABLES = (
    "KBOT_OPS_TARGET",
    "KBOT_OPS_POLICY",
    "KBOT_OPS_TARGET_BINDING",
    "KBOT_OPS_NOTIFICATION_SUBSCRIPTION",
    "KBOT_OPS_DIAGNOSTIC_SOURCE",
    "KBOT_OPS_TARGET_SOURCE_BINDING",
    "KBOT_OPS_AGENT",
    "KBOT_OPS_AGENT_VERSION",
    "KBOT_OPS_AGENT_VERSION_SOURCE",
    "KBOT_OPS_AGENT_GRANT",
)

UPGRADE_SCRIPT_NAMES = (
    "002_ops_runtime.sql",
    "003_ops_change.sql",
    "004_ops_inspection.sql",
    "005_ops_messaging.sql",
    "006_ops_fks_views.sql",
    "008_ops_conversations_reports.sql",
)

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

UPGRADE_HEADER = """-- KBot 4.0 AIOps Schema v12 到 v13 开发库升级脚本。
-- 本文件由 scripts/db/render_aiops_rebuild_schema.py 生成，请勿手工复制规范 DDL。
-- 使用 KBot Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- 本脚本保留 Target、监控源、绑定、策略、Agent、Agent版本和授权配置。
-- 本脚本永久删除并重建其他 KBOT_OPS_% 表，运行、告警、巡检、变更和对话历史会丢失。
-- 执行前必须停止 AIOps API、Worker、Scheduler，并备份需要保留的历史数据。
-- Oracle DDL 会自动提交；脚本只接受 12 / aiops-oracle-v2，不能用于其他版本。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

SET SERVEROUTPUT ON
SET VERIFY OFF

PROMPT === 正在验证 AIOps v12 升级前置条件 ===

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64);
    l_preserved_count PLS_INTEGER;
BEGIN
    SELECT schema_version, contract_version
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE component = 'AIOPS';

    IF l_schema_version <> 12 OR l_contract_version <> 'aiops-oracle-v2' THEN
        raise_application_error(
            -20120,
            '只允许从 AIOPS/12/aiops-oracle-v2 升级，当前为 AIOPS/'
            || l_schema_version || '/' || l_contract_version
        );
    END IF;

    SELECT COUNT(*)
      INTO l_preserved_count
      FROM user_tables
     WHERE table_name IN (
        'KBOT_OPS_TARGET',
        'KBOT_OPS_POLICY',
        'KBOT_OPS_TARGET_BINDING',
        'KBOT_OPS_NOTIFICATION_SUBSCRIPTION',
        'KBOT_OPS_DIAGNOSTIC_SOURCE',
        'KBOT_OPS_TARGET_SOURCE_BINDING',
        'KBOT_OPS_AGENT',
        'KBOT_OPS_AGENT_VERSION',
        'KBOT_OPS_AGENT_VERSION_SOURCE',
        'KBOT_OPS_AGENT_GRANT'
     );

    IF l_preserved_count <> 10 THEN
        raise_application_error(-20121, 'AIOps v12 配置表不完整，拒绝升级。');
    END IF;

    dbms_output.put_line('前置检查通过：将保留 10 张配置表。');
END;
/

PROMPT === 正在删除旧运行历史、对话对象和投影视图 ===

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
          AND table_name NOT IN (
            'KBOT_OPS_TARGET',
            'KBOT_OPS_POLICY',
            'KBOT_OPS_TARGET_BINDING',
            'KBOT_OPS_NOTIFICATION_SUBSCRIPTION',
            'KBOT_OPS_DIAGNOSTIC_SOURCE',
            'KBOT_OPS_TARGET_SOURCE_BINDING',
            'KBOT_OPS_AGENT',
            'KBOT_OPS_AGENT_VERSION',
            'KBOT_OPS_AGENT_VERSION_SOURCE',
            'KBOT_OPS_AGENT_GRANT'
          )
        ORDER BY table_name
    ) LOOP
        EXECUTE IMMEDIATE
            'DROP TABLE '
            || dbms_assert.enquote_name(table_row.table_name, FALSE)
            || ' CASCADE CONSTRAINTS PURGE';
        dbms_output.put_line('已删除历史表 ' || table_row.table_name);
    END LOOP;

    FOR constraint_row IN (
        SELECT table_name, constraint_name
        FROM user_constraints
        WHERE constraint_name IN (
            'FK_OPS_TARGET_DIAG_CRED',
            'FK_OPS_TARGET_EXEC_CRED',
            'FK_OPS_SOURCE_AUTH_CRED',
            'FK_OPS_SOURCE_WEBHOOK_CRED'
        )
    ) LOOP
        EXECUTE IMMEDIATE
            'ALTER TABLE '
            || dbms_assert.enquote_name(constraint_row.table_name, FALSE)
            || ' DROP CONSTRAINT '
            || dbms_assert.enquote_name(constraint_row.constraint_name, FALSE);
    END LOOP;

    FOR index_row IN (
        SELECT index_name
        FROM user_indexes
        WHERE index_name IN (
            'IX_OPS_TARGET_DIAG_CRED',
            'IX_OPS_TARGET_EXEC_CRED',
            'IX_OPS_SOURCE_AUTH_CRED',
            'IX_OPS_SOURCE_WEBHOOK_CRED',
            'UX_OPS_POLICY_ACTIVE'
        )
    ) LOOP
        EXECUTE IMMEDIATE
            'DROP INDEX ' || dbms_assert.enquote_name(index_row.index_name, FALSE);
    END LOOP;
END;
/

PROMPT === 正在补齐 v13 配置表所有权约束 ===

ALTER TABLE KBOT_OPS_TARGET ADD CONSTRAINT UK_OPS_TARGET_OWNER
    UNIQUE (TARGET_ID, DOMAIN_ID);

ALTER TABLE KBOT_OPS_AGENT_VERSION ADD CONSTRAINT UK_OPS_AGENT_VER_OWNER
    UNIQUE (AGENT_VERSION_ID, AGENT_ID);

PROMPT === 正在执行当前规范 AIOps v13 运行时 DDL ===
"""


def _load_manifest() -> dict:
    """读取并返回 AIOps Schema Manifest。"""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _read_verified_script(manifest: dict, name: str) -> str:
    """读取经过Manifest哈希校验的单份规范DDL。"""
    definitions = {
        str(item["name"]): item for item in manifest["scripts"]
    }
    definition = definitions.get(name)
    if definition is None:
        raise RuntimeError(f"Manifest缺少规范DDL：{name}")
    path = SCHEMA_DIR / name
    content = path.read_text(encoding="utf-8")
    actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if actual_hash != definition["sha256"]:
        raise RuntimeError(f"规范 DDL 哈希与 Manifest 不一致：{name}")
    return content


def _ddl_section(name: str, content: str) -> tuple[str, str, str]:
    """返回生成脚本中的规范DDL边界。"""
    return (
        f"-- ===== 开始规范 DDL：{name} =====",
        content.rstrip(),
        f"-- ===== 结束规范 DDL：{name} =====",
    )


def _footer(manifest: dict, *, action: str) -> str:
    """生成与Manifest一致的终态校验。"""
    return FOOTER.format(
        table_count=len(manifest["tables"]),
        view_count=len(manifest["views"]),
        schema_version=int(manifest["schema_version"]),
        contract_version=str(manifest["contract_version"]),
    ).replace("Schema 重建完成", f"Schema {action}完成")


def render_rebuild_sql() -> str:
    """生成包含全部规范 DDL 的单文件重建脚本。"""
    manifest = _load_manifest()
    sections = [HEADER.rstrip()]
    for definition in manifest["scripts"]:
        name = str(definition["name"])
        sections.extend(_ddl_section(name, _read_verified_script(manifest, name)))
    sections.append(_footer(manifest, action="重建").strip())
    return "\n\n".join(sections) + "\n"


def render_upgrade_sql() -> str:
    """生成保留配置表的v12到v13开发库升级脚本。"""
    manifest = _load_manifest()
    if (
        int(manifest["schema_version"]) != 13
        or str(manifest["contract_version"]) != "aiops-oracle-v3"
    ):
        raise RuntimeError("v12到v13升级脚本只能由AIOps v13 Manifest生成")
    manifest_tables = set(map(str, manifest["tables"]))
    if not set(PRESERVED_CONFIGURATION_TABLES).issubset(manifest_tables):
        raise RuntimeError("Manifest缺少需要保留的AIOps配置表")
    sections = [UPGRADE_HEADER.rstrip()]
    for name in UPGRADE_SCRIPT_NAMES:
        sections.extend(_ddl_section(name, _read_verified_script(manifest, name)))
    sections.append(_footer(manifest, action="升级").strip())
    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upgrade-v12-v13",
        action="store_true",
        help="生成保留配置表的v12到v13开发库升级脚本",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查已生成脚本是否与当前规范 DDL 一致",
    )
    args = parser.parse_args()
    if args.upgrade_v12_v13:
        rendered = render_upgrade_sql()
        output_path = UPGRADE_OUTPUT_PATH
        script_label = "AIOps v12到v13升级脚本"
    else:
        rendered = render_rebuild_sql()
        output_path = OUTPUT_PATH
        script_label = "AIOps重建脚本"
    if args.check:
        if (
            not output_path.is_file()
            or output_path.read_text(encoding="utf-8") != rendered
        ):
            print(f"{script_label}已过期，请重新运行生成器。")
            return 1
        print(f"{script_label}与当前规范 DDL 一致。")
        return 0
    output_path.write_text(rendered, encoding="utf-8")
    print(f"已生成 SQL Developer 单文件脚本：{output_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
