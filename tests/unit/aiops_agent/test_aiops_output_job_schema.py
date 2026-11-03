"""输出任务增量发布工件只新增规范对象，不删除或回填既有业务数据。"""

import re

from tools.db.render_aiops_output_jobs_upgrade import OUTPUT_PATH, render_upgrade_sql
from tools.db.render_aiops_rebuild_schema import _extract_table_columns, _load_canonical_sections


def test_upgrade_matches_canonical_and_is_non_destructive():
    rendered = render_upgrade_sql()
    assert OUTPUT_PATH.read_text(encoding="utf-8") == rendered
    tables = re.findall(r"CREATE TABLE\s+(\w+)", rendered)
    assert tables == ["KBOT_OPS_OUTPUT_JOB", "KBOT_OPS_OUTPUT_JOB_ATTEMPT"]
    assert not re.search(r"(?im)^\s*(DROP|TRUNCATE|DELETE|UPDATE|INSERT)\b", rendered)
    assert "仅允许从已验收的 AIOps Schema 55 升级" in rendered
    assert "禁止覆盖" in rendered
    assert "56 AS SCHEMA_VERSION" in rendered


def test_output_jobs_do_not_add_business_unique_or_check_constraints():
    _, sections = _load_canonical_sections()
    sql = dict(sections)["005_ops_messaging.sql"]
    added = sql[sql.index("-- 独立产物任务不复用诊断 Run 身份"):]
    assert not re.search(r"\bUNIQUE\b|\bCHECK\s*\(", added)
    assert "FK_OPS_OUTPUT_ATTEMPT_JOB" in added


def test_canonical_column_parser_does_not_treat_unnamed_primary_key_as_column():
    _, sections = _load_canonical_sections()
    columns = _extract_table_columns(sections, "KBOT_OPS_RESPONSIBILITY_GROUP_MEMBER")
    assert "PRIMARY" not in columns
    assert {"RESPONSIBILITY_GROUP_ID", "USER_ID"} <= set(columns)
