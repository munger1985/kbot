"""从 AIOps 规范 DDL 生成只读 Schema 完整性验证脚本。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tools.db.render_aiops_rebuild_schema import (
    ROOT,
    _extract_table_columns,
    _load_canonical_sections,
)


OUTPUT_PATH = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "aiops_agent"
    / "validate_aiops_schema.sql"
)


def _extract_named_objects(
    canonical_sections: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """提取规范 DDL 中显式索引及命名约束。"""
    canonical_sql = "\n".join(content for _, content in canonical_sections)
    explicit_indexes = set(
        re.findall(
            r"(?im)^\s*CREATE\s+(?:UNIQUE\s+)?INDEX\s+([A-Z][A-Z0-9_$#]*)\b",
            canonical_sql,
        )
    )
    constraints = set(
        re.findall(
            r"(?i)\bCONSTRAINT\s+([A-Z][A-Z0-9_$#]*)\b",
            canonical_sql,
        )
    )
    constraint_indexes = set(
        re.findall(
            r"(?is)\bCONSTRAINT\s+([A-Z][A-Z0-9_$#]*)\s+"
            r"(?:PRIMARY\s+KEY|UNIQUE)\b",
            canonical_sql,
        )
    )
    return sorted(explicit_indexes | constraint_indexes), sorted(constraints)


def _format_names(names: list[str], *, indent: str = "            ") -> str:
    """生成 Oracle ODCIVARCHAR2LIST 参数。"""
    return ",\n".join(f"{indent}'{name}'" for name in names)


def _render_set_check(
    *,
    label: str,
    expected_names: list[str],
    actual_query: str,
    error_code: str,
) -> str:
    """生成期望集合与数据库实际集合的双向差异检查。"""
    expected_sql = _format_names(expected_names)
    return f"""    l_issue_count := 0;
    FOR missing_row IN (
        SELECT column_value AS object_name
          FROM TABLE(sys.odcivarchar2list(
{expected_sql}
          ))
        MINUS
{actual_query}
    ) LOOP
        l_issue_count := l_issue_count + 1;
        dbms_output.put_line('[缺失] {label}: ' || missing_row.object_name);
    END LOOP;

    FOR extra_row IN (
{actual_query}
        MINUS
        SELECT column_value AS object_name
          FROM TABLE(sys.odcivarchar2list(
{expected_sql}
          ))
    ) LOOP
        l_issue_count := l_issue_count + 1;
        dbms_output.put_line('[多余] {label}: ' || extra_row.object_name);
    END LOOP;

    IF l_issue_count > 0 THEN
        l_error_count := l_error_count + l_issue_count;
        dbms_output.put_line('[失败] {error_code}: {label}集合不一致，共 '
                             || l_issue_count || ' 项。');
    ELSE
        dbms_output.put_line('[通过] {label}: {len(expected_names)} 项。');
    END IF;
"""


def render_validation_sql() -> str:
    """生成 SQL Developer 可直接执行的只读完整性检查。"""
    manifest, canonical_sections = _load_canonical_sections()
    indexes, constraints = _extract_named_objects(canonical_sections)
    columns = sorted(
        f"{table_name}|{column_name}"
        for table_name in manifest["tables"]
        for column_name in _extract_table_columns(
            canonical_sections, table_name
        )
    )

    checks = [
        _render_set_check(
            label="AIOps 表",
            expected_names=list(manifest["tables"]),
            actual_query=(
                "        SELECT table_name AS object_name\n"
                "          FROM user_tables\n"
                "         WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'"
            ),
            error_code="AIOPS_SCHEMA_TABLES",
        ),
        _render_set_check(
            label="AIOps 视图",
            expected_names=list(manifest["views"]),
            actual_query=(
                "        SELECT view_name AS object_name\n"
                "          FROM user_views\n"
                "         WHERE view_name LIKE 'KBOT\\_V\\_OPS\\_%' ESCAPE '\\'"
            ),
            error_code="AIOPS_SCHEMA_VIEWS",
        ),
        _render_set_check(
            label="AIOps 命名索引",
            expected_names=indexes,
            actual_query=(
                "        SELECT index_name AS object_name\n"
                "          FROM user_indexes\n"
                "         WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'\n"
                "           AND index_name NOT LIKE 'SYS\\_%' ESCAPE '\\'"
            ),
            error_code="AIOPS_SCHEMA_INDEXES",
        ),
        _render_set_check(
            label="AIOps 命名约束",
            expected_names=constraints,
            actual_query=(
                "        SELECT constraint_name AS object_name\n"
                "          FROM user_constraints\n"
                "         WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'\n"
                "           AND constraint_name NOT LIKE 'SYS\\_%' ESCAPE '\\'"
            ),
            error_code="AIOPS_SCHEMA_CONSTRAINTS",
        ),
        _render_set_check(
            label="AIOps 表列",
            expected_names=columns,
            actual_query=(
                "        SELECT table_name || '|' || column_name AS object_name\n"
                "          FROM user_tab_columns\n"
                "         WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'"
            ),
            error_code="AIOPS_SCHEMA_COLUMNS",
        ),
    ]

    return f"""-- KBot 4.0 AIOps Schema 只读完整性验证脚本。
-- 本文件由 tools/db/render_aiops_schema_validation.py 根据规范 DDL 生成，请勿手工维护对象清单。
-- 使用 KBot Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- 脚本只读取 USER_* 数据字典和版本视图，不创建、修改或删除任何数据库对象及数据。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

SET SERVEROUTPUT ON
SET VERIFY OFF
SET SQLBLANKLINES ON

PROMPT === 正在验证 AIOps Schema 完整性 ===

DECLARE
    l_error_count PLS_INTEGER := 0;
    l_issue_count PLS_INTEGER;
    l_component VARCHAR2(32 CHAR);
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
{''.join(checks)}
    l_issue_count := 0;
    FOR invalid_row IN (
        SELECT object_type, object_name, status
          FROM user_objects
         WHERE (
                object_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
             OR object_name LIKE 'KBOT\\_V\\_OPS\\_%' ESCAPE '\\'
         )
           AND status <> 'VALID'
         ORDER BY object_type, object_name
    ) LOOP
        l_issue_count := l_issue_count + 1;
        dbms_output.put_line(
            '[无效] ' || invalid_row.object_type || ': '
            || invalid_row.object_name || '，状态=' || invalid_row.status
        );
    END LOOP;
    l_error_count := l_error_count + l_issue_count;
    IF l_issue_count = 0 THEN
        dbms_output.put_line('[通过] 所有 AIOps 数据库对象状态均为 VALID。');
    END IF;

    l_issue_count := 0;
    FOR bad_constraint IN (
        SELECT table_name, constraint_name, status, validated
          FROM user_constraints
         WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
           AND (status <> 'ENABLED' OR validated <> 'VALIDATED')
         ORDER BY table_name, constraint_name
    ) LOOP
        l_issue_count := l_issue_count + 1;
        dbms_output.put_line(
            '[异常约束] ' || bad_constraint.table_name || '.'
            || bad_constraint.constraint_name || '，状态='
            || bad_constraint.status || '/' || bad_constraint.validated
        );
    END LOOP;
    l_error_count := l_error_count + l_issue_count;
    IF l_issue_count = 0 THEN
        dbms_output.put_line('[通过] 所有 AIOps 约束均已启用并验证。');
    END IF;

    l_issue_count := 0;
    FOR bad_index IN (
        SELECT table_name, index_name, status
          FROM user_indexes
         WHERE table_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
           AND status <> 'VALID'
         ORDER BY table_name, index_name
    ) LOOP
        l_issue_count := l_issue_count + 1;
        dbms_output.put_line(
            '[异常索引] ' || bad_index.table_name || '.'
            || bad_index.index_name || '，状态=' || bad_index.status
        );
    END LOOP;
    l_error_count := l_error_count + l_issue_count;
    IF l_issue_count = 0 THEN
        dbms_output.put_line('[通过] 所有 AIOps 索引状态均为 VALID。');
    END IF;

    l_issue_count := 0;
    FOR unexpected_row IN (
        SELECT object_type, object_name
          FROM user_objects
         WHERE (
                object_name LIKE 'KBOT\\_OPS\\_%' ESCAPE '\\'
             OR object_name LIKE 'KBOT\\_V\\_OPS\\_%' ESCAPE '\\'
         )
           AND object_type NOT IN ('TABLE', 'VIEW', 'INDEX')
         ORDER BY object_type, object_name
    ) LOOP
        l_issue_count := l_issue_count + 1;
        dbms_output.put_line(
            '[多余对象类型] ' || unexpected_row.object_type
            || ': ' || unexpected_row.object_name
        );
    END LOOP;
    l_error_count := l_error_count + l_issue_count;
    IF l_issue_count = 0 THEN
        dbms_output.put_line('[通过] 不存在规范之外的 AIOps 命名对象类型。');
    END IF;

    BEGIN
        SELECT component, schema_version, contract_version
          INTO l_component, l_schema_version, l_contract_version
          FROM KBOT_V_OPS_SCHEMA_VERSION;

        IF l_component <> 'AIOPS'
           OR l_schema_version <> {int(manifest['schema_version'])}
           OR l_contract_version <> '{manifest['contract_version']}' THEN
            l_error_count := l_error_count + 1;
            dbms_output.put_line(
                '[失败] AIOps Schema 版本错误：当前='
                || l_component || '/' || l_schema_version || '/'
                || l_contract_version || '，期望=AIOPS/'
                || '{int(manifest['schema_version'])}/{manifest['contract_version']}'
            );
        ELSE
            dbms_output.put_line(
                '[通过] Schema 合同：AIOPS/{int(manifest['schema_version'])}/'
                || '{manifest['contract_version']}'
            );
        END IF;
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            l_error_count := l_error_count + 1;
            dbms_output.put_line('[失败] Schema 版本视图没有返回记录。');
        WHEN TOO_MANY_ROWS THEN
            l_error_count := l_error_count + 1;
            dbms_output.put_line('[失败] Schema 版本视图返回多条记录。');
        WHEN OTHERS THEN
            l_error_count := l_error_count + 1;
            dbms_output.put_line(
                '[失败] 无法读取 Schema 版本视图：' || SQLERRM
            );
    END;

    IF l_error_count > 0 THEN
        raise_application_error(
            -20200,
            'AIOps Schema 验证失败，共发现 ' || l_error_count
            || ' 个对象或状态问题；请查看上方明细。'
        );
    END IF;

    dbms_output.put_line(
        '验证通过：{len(manifest['tables'])} 张表、{len(manifest['views'])} 个视图、'
        || '{len(indexes)} 个命名索引、{len(constraints)} 个命名约束、'
        || '{len(columns)} 个表列；Schema 与当前规范一致。'
    );
END;
/

PROMPT === AIOps Schema 完整性验证完成 ===
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查已生成验证脚本是否与当前规范 DDL 一致",
    )
    args = parser.parse_args()
    rendered = render_validation_sql()
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(
            encoding="utf-8"
        ) != rendered:
            print(f"AIOps Schema 验证脚本已过期：{OUTPUT_PATH.relative_to(ROOT)}")
            return 1
        print("AIOps Schema 验证脚本与当前规范 DDL 一致。")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"已生成 AIOps Schema 验证脚本：{OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
