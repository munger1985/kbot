"""从规范 DDL 生成 SQL Developer 可直接执行的 AIOps 重建脚本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"
MANIFEST_PATH = SCHEMA_DIR / "schema_manifest.json"
OUTPUT_PATH = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "aiops_agent"
    / "rebuild_aiops_schema.sql"
)
PRESERVING_OUTPUT_PATH = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "aiops_agent"
    / "rebuild_aiops_preserve_sources_targets.sql"
)

PRESERVED_TABLES = (
    ("KBOT_OPS_TARGET", "KBOT_KEEP_AIOPS_TARGET"),
    ("KBOT_OPS_DIAGNOSTIC_SOURCE", "KBOT_KEEP_AIOPS_SOURCE"),
    ("KBOT_OPS_TARGET_FACT", "KBOT_KEEP_AIOPS_TARGET_FACT"),
    ("KBOT_OPS_RECOVERY_PROFILE", "KBOT_KEEP_AIOPS_REC_PROFILE"),
    ("KBOT_OPS_TARGET_SOURCE_BINDING", "KBOT_KEEP_AIOPS_TSRC_BIND"),
)

HEADER = """-- KBot 4.0 AIOps Schema 全量重建脚本。
-- 本文件由 tools/db/render_aiops_rebuild_schema.py 生成，请勿手工复制规范 DDL。
-- 使用 KBot Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- 本脚本永久删除当前 Schema 内全部 KBOT_OPS_% 表、KBOT_V_OPS_% 视图及其数据。
-- 执行前必须停止 AIOps API、Worker、Scheduler 和 DB Executor，并备份需要保留的数据。
-- 平台用户、Domain、权限、角色以及 KC Collection 不在删除范围内。
-- Oracle DDL 会自动提交；失败后应修复原因并重新执行本脚本。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

SET SERVEROUTPUT ON
SET VERIFY OFF
SET SQLBLANKLINES ON

PROMPT === 正在检查 AIOps 重建前置条件 ===

DECLARE
    l_domain_key_count PLS_INTEGER;
    l_credential_key_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_domain_key_count
      FROM user_constraints constraint_row
     WHERE constraint_row.table_name = 'KBOT_PLATFORM_DOMAIN'
       AND constraint_row.constraint_type IN ('P', 'U')
       AND constraint_row.status = 'ENABLED'
       AND constraint_row.validated = 'VALIDATED'
       AND (
            SELECT COUNT(*)
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
       ) = 1
       AND EXISTS (
            SELECT 1
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
               AND column_row.position = 1
               AND column_row.column_name = 'DOMAIN_ID'
       );

    SELECT COUNT(*)
      INTO l_credential_key_count
      FROM user_constraints constraint_row
     WHERE constraint_row.table_name = 'KBOT_MANAGED_CREDENTIAL'
       AND constraint_row.constraint_type IN ('P', 'U')
       AND constraint_row.status = 'ENABLED'
       AND constraint_row.validated = 'VALIDATED'
       AND (
            SELECT COUNT(*)
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
       ) = 2
       AND EXISTS (
            SELECT 1
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
               AND column_row.position = 1
               AND column_row.column_name = 'CREDENTIAL_ID'
       )
       AND EXISTS (
            SELECT 1
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
               AND column_row.position = 2
               AND column_row.column_name = 'DOMAIN_ID'
       );

    IF l_domain_key_count = 0 THEN
        raise_application_error(
            -20010,
            '重建前置条件错误：KBOT_PLATFORM_DOMAIN(DOMAIN_ID) 主键或唯一键不可用。'
        );
    END IF;
    IF l_credential_key_count = 0 THEN
        raise_application_error(
            -20011,
            '重建前置条件错误：KBOT_MANAGED_CREDENTIAL(CREDENTIAL_ID, DOMAIN_ID) 唯一键不可用。'
        );
    END IF;
END;
/

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
    l_missing_table_count PLS_INTEGER;
    l_missing_view_count PLS_INTEGER;
    l_invalid_count PLS_INTEGER;
    l_bad_constraint_count PLS_INTEGER;
    l_bad_index_count PLS_INTEGER;
    l_workflow_kind_count PLS_INTEGER;
    l_required_column_count PLS_INTEGER;
    l_report_summary_count PLS_INTEGER;
    l_business_check_constraint_count PLS_INTEGER;
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
      INTO l_missing_table_count
      FROM TABLE(sys.odcivarchar2list(
{expected_tables}
      )) expected
     WHERE NOT EXISTS (
            SELECT 1
              FROM user_tables actual
             WHERE actual.table_name = expected.column_value
     );

    SELECT COUNT(*)
      INTO l_missing_view_count
      FROM TABLE(sys.odcivarchar2list(
{expected_views}
      )) expected
     WHERE NOT EXISTS (
            SELECT 1
              FROM user_views actual
             WHERE actual.view_name = expected.column_value
     );

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

    SELECT COUNT(*)
      INTO l_required_column_count
      FROM user_tab_columns
     WHERE nullable = 'N'
       AND (
            (table_name = 'KBOT_OPS_TASK' AND column_name = 'TASK_TYPE')
         OR (table_name = 'KBOT_OPS_CHANGE_PROPOSAL' AND column_name = 'TURN_ID')
         OR (table_name = 'KBOT_OPS_CHANGE_PROPOSAL' AND column_name = 'ACTION_FAMILY')
         OR (table_name = 'KBOT_OPS_CHANGE_PROPOSAL' AND column_name = 'EFFECT_CLASS')
         OR (table_name = 'KBOT_OPS_CHANGE_PROPOSAL' AND column_name = 'EXECUTION_MODE')
         OR (table_name = 'KBOT_OPS_CHANGE_PROPOSAL' AND column_name = 'EXECUTOR_KIND')
         OR (table_name = 'KBOT_OPS_CHANGE_PROPOSAL' AND column_name = 'LOCK_IMPACT')
         OR (table_name = 'KBOT_OPS_CHANGE_PROPOSAL' AND column_name = 'ESTIMATED_DURATION_SECONDS')
         OR (table_name = 'KBOT_OPS_AGENT_VERSION_TARGET'
             AND column_name = 'CONTROLLED_ACTION_POLICY_JSON')
         OR (table_name = 'KBOT_OPS_CONVERSATION_TURN'
             AND column_name = 'CURRENT_PLAN_REVISION')
         OR (table_name = 'KBOT_OPS_INVESTIGATION_REVISION'
             AND column_name = 'REVISION_ID')
         OR (table_name = 'KBOT_OPS_PLAYBOOK_INVOCATION'
             AND column_name = 'PLAYBOOK_INVOCATION_ID')
         OR (table_name = 'KBOT_OPS_TOOL_INVOCATION'
             AND column_name = 'TOOL_INVOCATION_ID')
         OR (table_name = 'KBOT_OPS_TURN_EVIDENCE'
             AND column_name = 'EVIDENCE_ROLE')
         OR (table_name = 'KBOT_OPS_INSPECTION_PLAN'
             AND column_name = 'AGENT_ID')
         OR (table_name = 'KBOT_OPS_INSPECTION_PLAN'
             AND column_name = 'INSPECTION_TEMPLATE_ID')
         OR (table_name = 'KBOT_OPS_INSPECTION_PLAN'
             AND column_name = 'INSPECTION_TEMPLATE_VERSION_ID')
         OR (table_name = 'KBOT_OPS_INSPECTION_FIRE'
             AND column_name = 'INSPECTION_TEMPLATE_ID')
         OR (table_name = 'KBOT_OPS_INSPECTION_FIRE'
             AND column_name = 'INSPECTION_TEMPLATE_VERSION_ID')
         OR (table_name = 'KBOT_OPS_INSPECTION_TEMPLATE'
             AND column_name = 'CURRENT_VERSION_ID')
         OR (table_name = 'KBOT_OPS_INSPECTION_TEMPLATE_VER'
             AND column_name = 'DEFINITION_JSON')
         OR (table_name = 'KBOT_OPS_TARGET'
             AND column_name = 'IMPORTANCE_LEVEL')
         OR (table_name = 'KBOT_OPS_KNOWLEDGE_ASSET'
             AND column_name = 'ASSET_KIND')
         OR (table_name = 'KBOT_OPS_KNOWLEDGE_VERSION'
             AND column_name = 'SOURCE_HASH')
         OR (table_name = 'KBOT_OPS_RECOVERY_PROFILE'
             AND column_name = 'RPO_SECONDS')
         OR (table_name = 'KBOT_OPS_RECOVERY_PROFILE'
             AND column_name = 'RTO_SECONDS')
         OR (table_name = 'KBOT_OPS_RECOVERY_PROFILE'
             AND column_name = 'REQUIRED_ASSURANCE_LEVEL')
         OR (table_name = 'KBOT_OPS_RECOVERY_DRILL'
             AND column_name = 'RECOVERY_MARKER_JSON')
         OR (table_name = 'KBOT_OPS_RECOVERY_DRILL'
             AND column_name = 'EVIDENCE_JSON')
         OR (table_name = 'KBOT_OPS_RECOVERY_DRILL'
             AND column_name = 'SOURCE_TRUST_LEVEL')
       );

    SELECT COUNT(*)
      INTO l_report_summary_count
      FROM user_tab_columns
     WHERE table_name = 'KBOT_OPS_REPORT'
       AND column_name = 'SUMMARY'
       AND data_type = 'CLOB';

    SELECT COUNT(*)
      INTO l_business_check_constraint_count
      FROM user_constraints
     WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
       AND constraint_type = 'C'
       AND generated = 'USER NAME';

    SELECT component, schema_version, contract_version
      INTO l_component, l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF l_table_count <> {table_count} OR l_view_count <> {view_count} THEN
        raise_application_error(
            -20001,
            'AIOps 对象数量错误：表=' || l_table_count || '，视图=' || l_view_count
        );
    END IF;
    IF l_missing_table_count <> 0 OR l_missing_view_count <> 0 THEN
        raise_application_error(
            -20007,
            'AIOps 规范对象缺失：表=' || l_missing_table_count
            || '，视图=' || l_missing_view_count
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
    IF l_required_column_count <> 30 THEN
        raise_application_error(-20008, 'Schema {schema_version} 必需列缺失或允许为空。');
    END IF;
    IF l_report_summary_count <> 1 THEN
        raise_application_error(-20013, 'KBOT_OPS_REPORT.SUMMARY 必须为 CLOB。');
    END IF;
    IF l_business_check_constraint_count <> 0 THEN
        raise_application_error(-20009, 'AIOps 业务表不得包含命名 CHECK 约束。');
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
"""


def _load_manifest() -> dict:
    """读取并返回 AIOps Schema Manifest。"""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _analyze_canonical_sql(name: str, content: str) -> int:
    """检查规范 DDL 的字符串、注释、语句边界和括号。"""
    statement_count = 0
    statement_started = False
    parenthesis_depth = 0
    state = "code"
    index = 0
    line_number = 1
    while index < len(content):
        character = content[index]
        next_character = content[index + 1] if index + 1 < len(content) else ""
        if state == "line_comment":
            if character == "\n":
                state = "code"
                line_number += 1
            index += 1
            continue
        if state == "block_comment":
            if character == "*" and next_character == "/":
                state = "code"
                index += 2
                continue
            if character == "\n":
                line_number += 1
            index += 1
            continue
        if state in {"string", "identifier"}:
            delimiter = "'" if state == "string" else '"'
            if character == delimiter:
                if next_character == delimiter:
                    index += 2
                    continue
                state = "code"
            if character == "\n":
                line_number += 1
            index += 1
            continue
        if character == "-" and next_character == "-":
            state = "line_comment"
            index += 2
            continue
        if character == "/" and next_character == "*":
            state = "block_comment"
            index += 2
            continue
        if character == "'":
            state = "string"
            statement_started = True
            index += 1
            continue
        if character == '"':
            state = "identifier"
            statement_started = True
            index += 1
            continue
        if character == "(":
            parenthesis_depth += 1
            statement_started = True
        elif character == ")":
            parenthesis_depth -= 1
            statement_started = True
            if parenthesis_depth < 0:
                raise RuntimeError(f"规范 DDL 括号提前结束：{name}:{line_number}")
        elif character == ";":
            if not statement_started:
                raise RuntimeError(f"规范 DDL 出现空语句：{name}:{line_number}")
            if parenthesis_depth != 0:
                raise RuntimeError(f"规范 DDL 括号未闭合：{name}:{line_number}")
            statement_count += 1
            statement_started = False
        elif not character.isspace():
            statement_started = True
        if character == "\n":
            line_number += 1
        index += 1
    if state in {"block_comment", "string", "identifier"}:
        raise RuntimeError(f"规范 DDL 词法结构未闭合：{name}:{line_number}")
    if statement_started or parenthesis_depth != 0:
        raise RuntimeError(f"规范 DDL 语句未以分号结束：{name}:{line_number}")
    return statement_count


def _format_expected_names(names: list[str]) -> str:
    """生成 Oracle ODCIVARCHAR2LIST 的缩进参数。"""
    return ",\n".join(f"          '{name}'" for name in names)


def _load_canonical_sections() -> tuple[dict, list[tuple[str, str]]]:
    """校验 Manifest，并返回按规范顺序排列的 DDL。"""
    manifest = _load_manifest()
    canonical_sections: list[tuple[str, str]] = []
    for definition in manifest["scripts"]:
        name = str(definition["name"])
        path = SCHEMA_DIR / name
        content = path.read_text(encoding="utf-8")
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual_hash != definition["sha256"]:
            raise RuntimeError(f"规范 DDL 哈希与 Manifest 不一致：{name}")
        actual_statements = _analyze_canonical_sql(name, content)
        expected_statements = int(definition["statements"])
        if actual_statements != expected_statements:
            raise RuntimeError(
                f"规范 DDL 语句数与 Manifest 不一致：{name}，"
                f"实际 {actual_statements}，期望 {expected_statements}"
            )
        canonical_sections.append((name, content))
    return manifest, canonical_sections


def _render_canonical_sections(
    canonical_sections: list[tuple[str, str]],
) -> list[str]:
    """生成自包含脚本中的规范 DDL 区段。"""
    sections: list[str] = []
    for name, content in canonical_sections:
        sections.extend(
            (
                f"-- ===== 开始规范 DDL：{name} =====",
                content.rstrip(),
                f"-- ===== 结束规范 DDL：{name} =====",
            )
        )
    return sections


def _render_validation(manifest: dict) -> str:
    """生成重建后的 Schema 合同验证。"""
    return FOOTER.format(
        table_count=len(manifest["tables"]),
        view_count=len(manifest["views"]),
        expected_tables=_format_expected_names(manifest["tables"]),
        expected_views=_format_expected_names(manifest["views"]),
        schema_version=int(manifest["schema_version"]),
        contract_version=str(manifest["contract_version"]),
    ).strip()


def _extract_table_columns(
    canonical_sections: list[tuple[str, str]], table_name: str
) -> tuple[str, ...]:
    """从规范 CREATE TABLE 中提取列名，避免保存脚本复制表结构。"""
    marker = f"CREATE TABLE {table_name} ("
    for name, content in canonical_sections:
        if marker not in content:
            continue
        body = content.split(marker, 1)[1]
        columns: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if line.startswith("        "):
                continue
            if stripped.startswith("CONSTRAINT ") or stripped == ");":
                break
            column_name = stripped.split(None, 1)[0].rstrip(",")
            if not column_name.replace("_", "").isalnum():
                raise RuntimeError(
                    f"无法解析规范表列：{name}:{table_name}:{stripped}"
                )
            columns.append(column_name)
        if not columns:
            raise RuntimeError(f"规范表没有可保存列：{name}:{table_name}")
        return tuple(columns)
    raise RuntimeError(f"规范 DDL 缺少待保存表：{table_name}")


def _format_columns(columns: tuple[str, ...], *, indent: str = "    ") -> str:
    """按每行一个列名输出稳定 SQL。"""
    return ",\n".join(f"{indent}{column}" for column in columns)


def _render_preserving_header(
    manifest: dict,
    canonical_sections: list[tuple[str, str]],
) -> str:
    """生成保存监控源、运维目标及直接子数据的重建前半段。"""
    source_tables = ", ".join(f"'{table}'" for table, _ in PRESERVED_TABLES)
    backup_tables = ", ".join(f"'{backup}'" for _, backup in PRESERVED_TABLES)
    backup_statements: list[str] = []
    count_queries: list[str] = []
    for table_name, backup_name in PRESERVED_TABLES:
        columns = _extract_table_columns(canonical_sections, table_name)
        backup_statements.append(
            "\n".join(
                (
                    f"CREATE TABLE {backup_name} AS",
                    "SELECT",
                    _format_columns(columns),
                    f"FROM {table_name};",
                )
            )
        )
        count_queries.append(
            f"SELECT '{table_name}' AS OBJECT_NAME, COUNT(*) AS ROW_COUNT "
            f"FROM {backup_name}"
        )
    backups_sql = "\n\n".join(backup_statements)
    counts_sql = "\nUNION ALL\n".join(count_queries) + ";"
    return f"""-- KBot 4.0 AIOps 保留配置数据的全量重建脚本。
-- 本文件由 tools/db/render_aiops_rebuild_schema.py 生成，请勿手工修改内嵌 DDL。
-- 使用 KBot Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- 仅保留运维目标、目标事实、恢复目标、监控源以及目标与监控源绑定；其他 AIOps 数据全部清空。
-- Managed Credential、平台用户、Domain、权限、角色和 KC Collection 位于共享表，不会删除。
-- 执行前必须停止 AIOps API、Worker、Scheduler 和 DB Executor，并完成数据库备份。
-- Oracle DDL 会自动提交；中途失败时 KBOT_KEEP_AIOPS_% 备份表会保留，请勿直接删除。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

SET SERVEROUTPUT ON
SET VERIFY OFF
SET SQLBLANKLINES ON

PROMPT === 正在检查保留式重建前置条件 ===

DECLARE
    l_component VARCHAR2(32 CHAR);
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_source_table_count PLS_INTEGER;
    l_backup_table_count PLS_INTEGER;
    l_external_fk_count PLS_INTEGER;
    l_domain_key_count PLS_INTEGER;
    l_credential_key_count PLS_INTEGER;
BEGIN
    SELECT component, schema_version, contract_version
      INTO l_component, l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF l_component <> 'AIOPS'
       OR l_schema_version <> {int(manifest['schema_version'])}
       OR l_contract_version <> '{manifest['contract_version']}' THEN
        raise_application_error(
            -20100,
            '仅支持 AIOPS/{int(manifest['schema_version'])}/'
            || '{manifest['contract_version']}，当前为 '
            || l_component || '/' || l_schema_version || '/' || l_contract_version
        );
    END IF;

    SELECT COUNT(*)
      INTO l_source_table_count
      FROM user_tables
     WHERE table_name IN ({source_tables});
    IF l_source_table_count <> {len(PRESERVED_TABLES)} THEN
        raise_application_error(-20101, '待保留的 AIOps 配置表不完整。');
    END IF;

    SELECT COUNT(*)
      INTO l_backup_table_count
      FROM user_tables
     WHERE table_name IN ({backup_tables});
    IF l_backup_table_count <> 0 THEN
        raise_application_error(
            -20102,
            '发现上次执行留下的 KBOT_KEEP_AIOPS_% 表，请先核实并人工处理。'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_external_fk_count
      FROM user_constraints child_constraint
      JOIN user_constraints parent_constraint
        ON parent_constraint.constraint_name = child_constraint.r_constraint_name
     WHERE child_constraint.constraint_type = 'R'
       AND parent_constraint.table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
       AND child_constraint.table_name NOT LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\';
    IF l_external_fk_count <> 0 THEN
        raise_application_error(
            -20103,
            '存在非 AIOps 表指向 KBOT_OPS_% 的外键，禁止自动重建。'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_domain_key_count
      FROM user_constraints constraint_row
     WHERE constraint_row.table_name = 'KBOT_PLATFORM_DOMAIN'
       AND constraint_row.constraint_type IN ('P', 'U')
       AND constraint_row.status = 'ENABLED'
       AND constraint_row.validated = 'VALIDATED'
       AND (
            SELECT COUNT(*)
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
       ) = 1
       AND EXISTS (
            SELECT 1
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
               AND column_row.position = 1
               AND column_row.column_name = 'DOMAIN_ID'
       );

    SELECT COUNT(*)
      INTO l_credential_key_count
      FROM user_constraints constraint_row
     WHERE constraint_row.table_name = 'KBOT_MANAGED_CREDENTIAL'
       AND constraint_row.constraint_type IN ('P', 'U')
       AND constraint_row.status = 'ENABLED'
       AND constraint_row.validated = 'VALIDATED'
       AND (
            SELECT COUNT(*)
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
       ) = 2
       AND EXISTS (
            SELECT 1
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
               AND column_row.position = 1
               AND column_row.column_name = 'CREDENTIAL_ID'
       )
       AND EXISTS (
            SELECT 1
              FROM user_cons_columns column_row
             WHERE column_row.constraint_name = constraint_row.constraint_name
               AND column_row.table_name = constraint_row.table_name
               AND column_row.position = 2
               AND column_row.column_name = 'DOMAIN_ID'
       );

    IF l_domain_key_count = 0 OR l_credential_key_count = 0 THEN
        raise_application_error(-20104, '共享 Domain 或 Managed Credential 父键不可用。');
    END IF;
END;
/

PROMPT === 正在备份监控源和运维目标配置 ===

{backups_sql}

PROMPT === 已保存的数据行数 ===

{counts_sql}

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

PROMPT === 正在执行当前规范 AIOps DDL ==="""


def _render_restore_and_verify(
    canonical_sections: list[tuple[str, str]],
) -> str:
    """生成配置恢复、行数验证与非配置数据清空验证。"""
    restore_statements: list[str] = []
    row_pairs: list[str] = []
    for table_name, backup_name in PRESERVED_TABLES:
        columns = _extract_table_columns(canonical_sections, table_name)
        restore_statements.append(
            "\n".join(
                (
                    f"INSERT INTO {table_name} (",
                    _format_columns(columns),
                    ")",
                    "SELECT",
                    _format_columns(columns),
                    f"FROM {backup_name};",
                )
            )
        )
        row_pairs.append(
            f"SELECT '{table_name}' AS TABLE_NAME, '{backup_name}' AS BACKUP_NAME "
            "FROM DUAL"
        )
    restore_sql = "\n\n".join(restore_statements)
    pairs_sql = "\n        UNION ALL\n        ".join(row_pairs)
    preserved_names = ", ".join(f"'{table}'" for table, _ in PRESERVED_TABLES)
    return f"""PROMPT === 正在恢复监控源和运维目标配置 ===

{restore_sql}

COMMIT;

PROMPT === 正在验证保留行数和清空边界 ===

DECLARE
    l_current_count PLS_INTEGER;
    l_backup_count PLS_INTEGER;
    l_other_row_count PLS_INTEGER := 0;
BEGIN
    FOR keep_row IN (
        {pairs_sql}
    ) LOOP
        EXECUTE IMMEDIATE
            'SELECT COUNT(*) FROM '
            || dbms_assert.enquote_name(keep_row.table_name, FALSE)
            INTO l_current_count;
        EXECUTE IMMEDIATE
            'SELECT COUNT(*) FROM '
            || dbms_assert.enquote_name(keep_row.backup_name, FALSE)
            INTO l_backup_count;
        IF l_current_count <> l_backup_count THEN
            raise_application_error(
                -20110,
                keep_row.table_name || ' 恢复行数不一致：当前='
                || l_current_count || '，备份=' || l_backup_count
            );
        END IF;
        dbms_output.put_line(
            '已恢复 ' || keep_row.table_name || '：' || l_current_count || ' 行'
        );
    END LOOP;

    FOR table_row IN (
        SELECT table_name
          FROM user_tables
         WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
           AND table_name NOT IN ({preserved_names})
         ORDER BY table_name
    ) LOOP
        EXECUTE IMMEDIATE
            'SELECT COUNT(*) FROM '
            || dbms_assert.enquote_name(table_row.table_name, FALSE)
            INTO l_current_count;
        l_other_row_count := l_other_row_count + l_current_count;
    END LOOP;

    IF l_other_row_count <> 0 THEN
        raise_application_error(-20111, '非保留 AIOps 表仍存在业务数据。');
    END IF;
END;
/"""


def _render_backup_cleanup() -> str:
    """生成成功后删除临时备份表的 SQL。"""
    backup_names = "\n        UNION ALL\n        ".join(
        f"SELECT '{backup}' AS TABLE_NAME FROM DUAL"
        for _, backup in PRESERVED_TABLES
    )
    return f"""PROMPT === Schema 与数据验证通过，正在删除临时备份表 ===

DECLARE
BEGIN
    FOR backup_row IN (
        {backup_names}
    ) LOOP
        EXECUTE IMMEDIATE
            'DROP TABLE '
            || dbms_assert.enquote_name(backup_row.table_name, FALSE)
            || ' PURGE';
        dbms_output.put_line('已删除临时备份表 ' || backup_row.table_name);
    END LOOP;
END;
/

PROMPT === AIOps Schema 重建完成；配置数据已恢复，其他 AIOps 数据已清空 ===
PROMPT === 启动服务后检查 AIOps /ready ==="""


def render_rebuild_sql() -> str:
    """生成清空全部 AIOps 数据的单文件重建脚本。"""
    manifest, canonical_sections = _load_canonical_sections()
    sections = [HEADER.rstrip()]
    sections.extend(_render_canonical_sections(canonical_sections))
    sections.append(_render_validation(manifest))
    sections.append("PROMPT === AIOps Schema 重建完成；启动服务后检查 AIOps /ready ===")
    return "\n\n".join(sections) + "\n"


def render_preserving_rebuild_sql() -> str:
    """生成仅保留监控源和运维目标配置的单文件重建脚本。"""
    manifest, canonical_sections = _load_canonical_sections()
    sections = [_render_preserving_header(manifest, canonical_sections).rstrip()]
    sections.extend(_render_canonical_sections(canonical_sections))
    sections.append(_render_restore_and_verify(canonical_sections))
    sections.append(_render_validation(manifest))
    sections.append(_render_backup_cleanup())
    return "\n\n".join(sections) + "\n"


def _outputs() -> tuple[tuple[Path, str], ...]:
    """返回全部生成物及其当前内容。"""
    return (
        (OUTPUT_PATH, render_rebuild_sql()),
        (PRESERVING_OUTPUT_PATH, render_preserving_rebuild_sql()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查已生成脚本是否与当前规范 DDL 一致",
    )
    args = parser.parse_args()
    outputs = _outputs()
    if args.check:
        stale_paths = [
            path.relative_to(ROOT)
            for path, rendered in outputs
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered
        ]
        if stale_paths:
            for path in stale_paths:
                print(f"AIOps 重建脚本已过期：{path}")
            return 1
        print("AIOps 重建脚本与当前规范 DDL 一致。")
        return 0
    for path, rendered in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"已生成 SQL Developer 单文件脚本：{path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
