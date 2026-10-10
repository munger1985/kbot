"""AIOps Oracle 全量重建脚本合同测试。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from tools.db.render_aiops_rebuild_schema import (
    _analyze_canonical_sql,
    render_rebuild_sql,
    render_preserving_rebuild_sql,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"
REBUILD_SCRIPT = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "aiops_agent"
    / "rebuild_aiops_schema.sql"
)
PRESERVING_REBUILD_SCRIPT = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "aiops_agent"
    / "rebuild_aiops_preserve_sources_targets.sql"
)
MANIFEST = SCHEMA_DIR / "schema_manifest.json"
UPGRADE_SCHEMA_19 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "upgrade_aiops_schema_19.sql"
)
APPLY_SCHEMA_20 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_20.sql"
)
APPLY_SCHEMA_21 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_21.sql"
)
APPLY_SCHEMA_22 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_22.sql"
)
APPLY_SCHEMA_23 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_23.sql"
)
APPLY_SCHEMA_26 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_26.sql"
)
APPLY_SCHEMA_28 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_28.sql"
)
APPLY_SCHEMA_29 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_29.sql"
)
APPLY_SCHEMA_30 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_30.sql"
)
APPLY_SCHEMA_36 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_36.sql"
)
APPLY_SCHEMA_37 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "apply_aiops_schema_37.sql"
)
CHECK_CATALOG = (
    ROOT
    / "services"
    / "aiops_agent"
    / "src"
    / "aiops_agent"
    / "application"
    / "inspections"
    / "check_catalog.json"
)


class AIOpsRebuildSchemaScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sql = REBUILD_SCRIPT.read_text(encoding="utf-8")
        self.preserving_sql = PRESERVING_REBUILD_SCRIPT.read_text(
            encoding="utf-8"
        )
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_rebuild_uses_every_canonical_script_in_manifest_order(self) -> None:
        sections = re.findall(
            r"^-- ===== 开始规范 DDL：([0-9]{3}_[a-z0-9_]+\.sql) =====$",
            self.sql,
            re.MULTILINE,
        )
        expected = [item["name"] for item in self.manifest["scripts"]]

        self.assertEqual(expected, sections)
        self.assertEqual(render_rebuild_sql(), self.sql)
        self.assertNotRegex(self.sql, r"(?m)^@@")
        self.assertEqual(
            len(self.manifest["tables"]),
            len(re.findall(r"(?im)^\s*CREATE\s+TABLE\s+", self.sql)),
        )
        self.assertEqual(
            len(self.manifest["views"]),
            len(
                re.findall(
                    r"(?im)^\s*CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+",
                    self.sql,
                )
            ),
        )

    def test_preserving_rebuild_uses_current_canonical_ddl(self) -> None:
        self.assertEqual(render_preserving_rebuild_sql(), self.preserving_sql)
        self.assertNotRegex(self.preserving_sql, r"(?m)^@@")
        sections = re.findall(
            r"^-- ===== 开始规范 DDL：([0-9]{3}_[a-z0-9_]+\.sql) =====$",
            self.preserving_sql,
            re.MULTILINE,
        )
        self.assertEqual(
            [item["name"] for item in self.manifest["scripts"]],
            sections,
        )

    def test_preserving_rebuild_keeps_only_sources_targets_and_direct_data(
        self,
    ) -> None:
        preserved = {
            "KBOT_OPS_TARGET": "KBOT_KEEP_AIOPS_TARGET",
            "KBOT_OPS_TARGET_FACT": "KBOT_KEEP_AIOPS_TARGET_FACT",
            "KBOT_OPS_DIAGNOSTIC_SOURCE": "KBOT_KEEP_AIOPS_SOURCE",
            "KBOT_OPS_TARGET_SOURCE_BINDING": "KBOT_KEEP_AIOPS_TSRC_BIND",
        }
        for table_name, backup_name in preserved.items():
            self.assertIn(f"CREATE TABLE {backup_name} AS", self.preserving_sql)
            self.assertIn(f"FROM {table_name};", self.preserving_sql)
            self.assertIn(f"INSERT INTO {table_name} (", self.preserving_sql)
            self.assertIn(f"FROM {backup_name};", self.preserving_sql)

        self.assertNotIn("SELECT *", self.preserving_sql.upper())
        self.assertNotIn("KBOT_KEEP_AIOPS_AGENT", self.preserving_sql)
        self.assertNotIn("KBOT_KEEP_AIOPS_POLICY", self.preserving_sql)
        self.assertIn("非保留 AIOps 表仍存在业务数据", self.preserving_sql)
        self.assertIn("正在删除临时备份表", self.preserving_sql)
        self.assertLess(
            self.preserving_sql.index("正在验证保留行数和清空边界"),
            self.preserving_sql.index("正在删除临时备份表"),
        )

    def test_canonical_schema_does_not_enumerate_business_values(
        self,
    ) -> None:
        canonical = (SCHEMA_DIR / "008_ops_conversations_reports.sql").read_text(
            encoding="utf-8"
        )
        self.assertNotRegex(canonical, r"\bCHECK\s*\(")
        self.assertNotRegex(self.sql, r"\bCONSTRAINT\s+CK_OPS_")
        self.assertNotRegex(self.preserving_sql, r"\bCONSTRAINT\s+CK_OPS_")

    def test_rebuild_validation_matches_manifest_contract(self) -> None:
        self.assertIn(f"l_table_count <> {len(self.manifest['tables'])}", self.sql)
        self.assertIn(f"l_view_count <> {len(self.manifest['views'])}", self.sql)
        self.assertIn(
            f"l_schema_version <> {self.manifest['schema_version']}", self.sql
        )
        self.assertIn(
            f"l_contract_version <> '{self.manifest['contract_version']}'",
            self.sql,
        )
        self.assertIn(
            f"{self.manifest['schema_version']}，合同 "
            f"{self.manifest['contract_version']}。",
            self.sql,
        )
        self.assertIn("l_missing_table_count <> 0", self.sql)
        self.assertIn("l_missing_view_count <> 0", self.sql)
        self.assertIn("l_required_column_count <> 30", self.sql)
        self.assertIn("l_report_summary_count <> 1", self.sql)
        self.assertIn("l_business_check_constraint_count <> 0", self.sql)
        self.assertNotIn("CK_OPS_TASK_TYPE", self.sql)
        for table_name in self.manifest["tables"]:
            self.assertIn(f"'{table_name}'", self.sql)
        for view_name in self.manifest["views"]:
            self.assertIn(f"'{view_name}'", self.sql)

    def test_schema_version_view_matches_manifest_contract(self) -> None:
        canonical = (SCHEMA_DIR / "006_ops_fks_views.sql").read_text(
            encoding="utf-8"
        )
        version_view = re.search(
            r"CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS"
            r"(?P<body>.*?)FROM DUAL;",
            canonical,
            re.DOTALL,
        )

        self.assertIsNotNone(version_view)
        body = version_view.group("body")
        self.assertIn(
            f"{self.manifest['schema_version']} AS SCHEMA_VERSION",
            body,
        )
        self.assertIn(
            f"'{self.manifest['contract_version']}' AS CONTRACT_VERSION",
            body,
        )

    def test_report_summary_uses_clob_without_clob_aggregation(self) -> None:
        inspection_sql = (SCHEMA_DIR / "004_ops_inspection.sql").read_text(
            encoding="utf-8"
        )
        views_sql = (SCHEMA_DIR / "006_ops_fks_views.sql").read_text(
            encoding="utf-8"
        )

        self.assertRegex(inspection_sql, r"(?m)^\s*SUMMARY CLOB,")
        self.assertNotIn("MAX(SUMMARY)", views_sql)
        self.assertIn("ROW_NUMBER() OVER", views_sql)

    def test_schema_19_upgrade_is_incremental_and_repairs_orphan_turns(
        self,
    ) -> None:
        sql = UPGRADE_SCHEMA_19.read_text(encoding="utf-8")

        self.assertIn("SUMMARY_CLOB CLOB", sql)
        self.assertIn("SUMMARY_CLOB = TO_CLOB(SUMMARY)", sql)
        self.assertIn("DROP COLUMN SUMMARY", sql)
        self.assertIn("RENAME COLUMN SUMMARY_CLOB TO SUMMARY", sql)
        self.assertNotIn("MAX(SUMMARY)", sql)
        self.assertIn("schema19-recovery:", sql)
        self.assertIn("KBOT_OPS_TURN_EVENT", sql)
        self.assertIn("19 AS SCHEMA_VERSION", sql)
        self.assertIn("'aiops-oracle-v9' AS CONTRACT_VERSION", sql)
        self.assertNotIn("DROP TABLE", sql.upper())

    def test_schema_20_apply_preserves_data_and_closes_execution_by_default(
        self,
    ) -> None:
        sql = APPLY_SCHEMA_20.read_text(encoding="utf-8")
        normalized = sql.upper()

        self.assertNotIn("DROP TABLE", normalized)
        self.assertNotIn("TRUNCATE TABLE", normalized)
        self.assertNotRegex(normalized, r"\bDELETE\s+FROM\b")
        for column_name in (
            "ACTION_FAMILY",
            "EFFECT_CLASS",
            "EXECUTION_MODE",
            "EXECUTOR_KIND",
            "CANONICAL_OBJECT_REF_JSON",
            "LOCK_IMPACT",
            "ESTIMATED_DURATION_SECONDS",
            "CONTROLLED_ACTION_POLICY_JSON",
        ):
            self.assertIn(column_name, normalized)
        self.assertIn("COALESCE(EXECUTION_MODE, 'MANUAL_ONLY')", normalized)
        self.assertIn("COALESCE(EXECUTOR_KIND, 'NONE')", normalized)
        self.assertIn("'ENABLED' VALUE 'FALSE' FORMAT JSON", normalized)
        self.assertIn("20 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V10' AS CONTRACT_VERSION", normalized)

    def test_schema_21_apply_is_incremental_and_matches_reporting_contract(
        self,
    ) -> None:
        sql = APPLY_SCHEMA_21.read_text(encoding="utf-8")
        normalized = sql.upper()

        self.assertNotIn("DROP TABLE", normalized)
        self.assertNotIn("TRUNCATE TABLE", normalized)
        self.assertNotRegex(normalized, r"\bDELETE\s+FROM\b")
        self.assertIn("KBOT_OPS_REPORT_SOURCE", normalized)
        self.assertIn("IX_OPS_RPT_SOURCE_RUN", normalized)
        self.assertIn("IX_OPS_RPT_SOURCE_ART", normalized)
        self.assertIn("INSPECTION_MONTHLY", normalized)
        self.assertIn("INSPECTION_QUARTERLY", normalized)
        self.assertIn("INSPECTION_ANNUAL", normalized)
        self.assertIn("21 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V11' AS CONTRACT_VERSION", normalized)

    def test_schema_22_apply_preserves_data_and_allows_user_evidence(
        self,
    ) -> None:
        sql = APPLY_SCHEMA_22.read_text(encoding="utf-8")
        normalized = sql.upper()

        self.assertNotIn("DROP TABLE", normalized)
        self.assertNotIn("TRUNCATE TABLE", normalized)
        self.assertNotRegex(normalized, r"\bDELETE\s+FROM\b")
        self.assertIn("DROP CONSTRAINT CK_OPS_TOOL_INV_CLASS", normalized)
        self.assertIn("'USER_EVIDENCE'", normalized)
        self.assertIn("22 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V12' AS CONTRACT_VERSION", normalized)

    def test_schema_23_apply_preserves_data_and_requires_container_recheck(
        self,
    ) -> None:
        sql = APPLY_SCHEMA_23.read_text(encoding="utf-8")
        normalized = sql.upper()

        self.assertNotIn("DROP TABLE", normalized)
        self.assertNotIn("TRUNCATE TABLE", normalized)
        self.assertNotRegex(normalized, r"\bDELETE\s+FROM\b")
        for column_name in (
            "ORACLE_CONTAINER_SCOPE",
            "ORACLE_PDB_NAME",
            "OBSERVED_ORACLE_CONTAINER_SCOPE",
            "OBSERVED_ORACLE_CONTAINER_NAME",
            "OBSERVED_ORACLE_CONTAINER_NUMBER",
            "OBSERVED_ORACLE_DATABASE_NAME",
        ):
            self.assertIn(column_name, normalized)
        self.assertIn("DROP CONSTRAINT CK_OPS_TARGET_CONNECTIVITY", normalized)
        self.assertIn("'MISCONFIGURED'", normalized)
        self.assertIn("ORACLE_CONTAINER_SCOPE IS NOT NULL", normalized)
        self.assertIn("OBSERVED_ORACLE_CONTAINER_SCOPE IS NOT NULL", normalized)
        self.assertIn("READONLY_CONNECTION_ENABLED = 0", normalized)
        self.assertIn(
            "LAST_ERROR_CODE = 'ORACLE_CONTAINER_CONFIGURATION_REQUIRED'",
            normalized,
        )
        self.assertIn("CREATE OR REPLACE VIEW KBOT_V_OPS_TARGET", normalized)
        self.assertIn("23 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V13' AS CONTRACT_VERSION", normalized)
        self.assertIn("ALTER SESSION SET TIME_ZONE = '+00:00'", normalized)
        self.assertIn("UPDATED_AT = SYSTIMESTAMP", normalized)
        self.assertNotIn("UPDATED_AT = CURRENT_TIMESTAMP", normalized)

        roots_sql = (SCHEMA_DIR / "001_ops_roots.sql").read_text(
            encoding="utf-8"
        )
        target_table = roots_sql.split(
            "CREATE TABLE KBOT_OPS_POLICY", maxsplit=1
        )[0]
        self.assertIn(
            "CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP",
            target_table,
        )
        self.assertIn(
            "UPDATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP",
            target_table,
        )

    def test_schema_26_apply_adds_selected_checks_and_target_facts(
        self,
    ) -> None:
        sql = APPLY_SCHEMA_26.read_text(encoding="utf-8")
        normalized = sql.upper()
        catalog = json.loads(CHECK_CATALOG.read_text(encoding="utf-8"))
        ready_ids = [
            item["check_id"]
            for group in catalog["groups"]
            for item in group["checks"]
            if item.get("availability") == "READY"
            and not str(item.get("tool_id") or "").startswith("user.")
            and item["check_id"] in sql
        ]

        self.assertNotIn("DROP TABLE", normalized)
        self.assertNotIn("TRUNCATE TABLE", normalized)
        self.assertNotRegex(normalized, r"\bDELETE\s+FROM\b")
        self.assertIn("SELECTED_CHECKS_JSON", normalized)
        self.assertIn("CREATE TABLE KBOT_OPS_TARGET_FACT", normalized)
        self.assertIn("UX_OPS_TARGET_FACT_ACTIVE", normalized)
        self.assertIn("CREATE OR REPLACE VIEW KBOT_V_OPS_INSPECTION_PLAN", normalized)
        self.assertIn("26 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V16' AS CONTRACT_VERSION", normalized)
        self.assertIn("AIOPS-ORACLE-V13", normalized)
        self.assertIn("AIOPS-ORACLE-V14", normalized)
        self.assertEqual(18, len(ready_ids))
        for check_id in ready_ids:
            self.assertIn(check_id, sql)
        self.assertNotIn("user.report", sql)

        roots_sql = (SCHEMA_DIR / "001_ops_roots.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("CREATE TABLE KBOT_OPS_TARGET_FACT", roots_sql)

    def test_canonical_statement_counts_and_parentheses_match_manifest(self) -> None:
        for definition in self.manifest["scripts"]:
            content = (SCHEMA_DIR / definition["name"]).read_text(encoding="utf-8")
            self.assertEqual(
                definition["statements"],
                _analyze_canonical_sql(definition["name"], content),
            )

    def test_schema_28_apply_removes_business_checks_without_data_changes(
        self,
    ) -> None:
        sql = APPLY_SCHEMA_28.read_text(encoding="utf-8")
        normalized = sql.upper()

        self.assertNotIn("DROP TABLE", normalized)
        self.assertNotIn("TRUNCATE TABLE", normalized)
        self.assertNotRegex(normalized, r"\bDELETE\s+FROM\b")
        self.assertIn("CONSTRAINT_TYPE = 'C'", normalized)
        self.assertIn("GENERATED = 'USER NAME'", normalized)
        self.assertIn("DROP CONSTRAINT", normalized)
        self.assertIn("DDL_LOCK_TIMEOUT = 60", normalized)
        self.assertIn("SQLCODE = -54", normalized)
        self.assertIn("-20064", normalized)
        self.assertIn("SCHEMA_VERSION = 28", normalized)
        self.assertIn("CONTRACT_VERSION = 'AIOPS-ORACLE-V18'", normalized)
        self.assertIn("28 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V18' AS CONTRACT_VERSION", normalized)

    def test_schema_29_apply_adds_target_importance_without_business_constraints(
        self,
    ) -> None:
        sql = APPLY_SCHEMA_29.read_text(encoding="utf-8")
        normalized = sql.upper()

        self.assertNotIn("DROP TABLE", normalized)
        self.assertNotIn("TRUNCATE TABLE", normalized)
        self.assertNotRegex(normalized, r"\bDELETE\s+FROM\b")
        self.assertIn("IMPORTANCE_LEVEL NUMBER(1) DEFAULT 3 NOT NULL", normalized)
        self.assertNotIn("CHECK (IMPORTANCE_LEVEL", normalized)
        self.assertIn("DDL_LOCK_TIMEOUT = 60", normalized)
        self.assertIn("29 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V19' AS CONTRACT_VERSION", normalized)

    def test_schema_30_apply_freezes_inspection_templates_and_removes_old_fields(
        self,
    ) -> None:
        sql = APPLY_SCHEMA_30.read_text(encoding="utf-8")
        normalized = sql.upper()

        self.assertNotIn("DROP TABLE", normalized)
        self.assertNotIn("TRUNCATE TABLE", normalized)
        self.assertNotRegex(normalized, r"\bDELETE\s+FROM\b")
        self.assertIn("CREATE TABLE KBOT_OPS_INSPECTION_TEMPLATE", normalized)
        self.assertIn("CREATE TABLE KBOT_OPS_INSPECTION_TEMPLATE_VER", normalized)
        self.assertIn("INSPECTION_TEMPLATE.V1", normalized)
        self.assertIn("SELECTED_CHECK_IDS", normalized)
        self.assertIn("EVIDENCE_STEPS", normalized)
        self.assertIn("INSPECTION_TEMPLATE_VERSION_ID NOT NULL", normalized)
        self.assertIn("DROP_COLUMN_IF_PRESENT", normalized)
        self.assertIn("UK_OPS_REPORT_TEMPLATE_NAME", normalized)
        self.assertIn("DDL_LOCK_TIMEOUT = 60", normalized)
        self.assertLess(
            normalized.index("L_TEMPLATE_ID RAW(16);"),
            normalized.index("FUNCTION UUID7_RAW"),
        )
        self.assertIn("FUNCTION CLOB_FINGERPRINT", normalized)
        self.assertIn("DBMS_LOB.SUBSTR(P_VALUE, 900, L_OFFSET)", normalized)
        self.assertIn("JSON(L_DEFINITION)", normalized)
        self.assertNotIn("DBMS_LOB.SUBSTR(L_DEFINITION, 32767", normalized)
        self.assertIn("30 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V20' AS CONTRACT_VERSION", normalized)

    def test_schema_36_apply_removes_target_security_level_safely(self) -> None:
        sql = APPLY_SCHEMA_36.read_text(encoding="utf-8")
        normalized = sql.upper()

        self.assertIn("CREATE OR REPLACE VIEW KBOT_V_OPS_TARGET", normalized)
        self.assertIn("DROP COLUMN SECURITY_LEVEL", normalized)
        self.assertIn("USER_TAB_COLUMNS", normalized)
        self.assertIn("SCHEMA_VERSION = 35", normalized)
        self.assertIn("SCHEMA_VERSION = 36", normalized)
        self.assertIn("36 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V26' AS CONTRACT_VERSION", normalized)
        target_view = normalized.split(
            "CREATE OR REPLACE VIEW KBOT_V_OPS_TARGET AS", maxsplit=1
        )[1].split(
            "PROMPT === 正在移除 TARGET 安全级别列 ===", maxsplit=1
        )[0]
        self.assertNotIn("T.SECURITY_LEVEL", target_view)
        self.assertNotIn("DROP TABLE", normalized)
        self.assertNotRegex(normalized, r"\bDELETE\s+FROM\b")

    def test_schema_37_apply_removes_legacy_source_links_without_guessing(
        self,
    ) -> None:
        sql = APPLY_SCHEMA_37.read_text(encoding="utf-8")
        normalized = sql.upper()

        self.assertIn("USER_TABLES", normalized)
        self.assertIn(
            "DROP TABLE KBOT_OPS_AGENT_VERSION_SOURCE "
            "CASCADE CONSTRAINTS PURGE",
            normalized,
        )
        self.assertIn("KBOT_OPS_TARGET_SOURCE_BINDING", normalized)
        self.assertIn("KBOT_OPS_DIAGNOSTIC_SOURCE", normalized)
        self.assertIn("KBOT_OPS_AGENT_VERSION_TARGET", normalized)
        self.assertGreaterEqual(normalized.count("SET STATUS = 'DISABLED'"), 2)
        self.assertIn("ROW_VERSION = TARGET.ROW_VERSION + 1", normalized)
        self.assertIn("ROW_VERSION = AGENT.ROW_VERSION + 1", normalized)
        self.assertIn("'SCHEMA-UPGRADE-37'", normalized)
        self.assertIn("SCHEMA_VERSION = 36", normalized)
        self.assertIn("SCHEMA_VERSION = 37", normalized)
        self.assertIn("37 AS SCHEMA_VERSION", normalized)
        self.assertIn("'AIOPS-ORACLE-V27' AS CONTRACT_VERSION", normalized)
        self.assertNotIn(
            "INSERT INTO KBOT_OPS_AGENT_VERSION_TARGET",
            normalized,
        )

    def test_canonical_analyzer_rejects_unclosed_check_constraint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "括号未闭合"):
            _analyze_canonical_sql(
                "broken.sql",
                "CREATE TABLE T (A NUMBER, CONSTRAINT CK_T CHECK (A IN (1, 2));",
            )

    def test_rebuild_checks_shared_parent_keys_before_drop(self) -> None:
        preflight_position = self.sql.index("PROMPT === 正在检查 AIOps 重建前置条件 ===")
        drop_position = self.sql.index("PROMPT === 正在删除旧 AIOps 视图和表 ===")
        preflight = self.sql[preflight_position:drop_position]
        self.assertLess(preflight_position, drop_position)
        self.assertIn("KBOT_PLATFORM_DOMAIN", preflight)
        self.assertIn("KBOT_MANAGED_CREDENTIAL", preflight)
        self.assertIn("CREDENTIAL_ID", preflight)
        self.assertIn("DOMAIN_ID", preflight)

    def test_rebuild_is_non_interactive_and_scoped_to_aiops_objects(self) -> None:
        self.assertNotIn("ACCEPT ", self.sql.upper())
        self.assertNotRegex(self.sql, r"&[a-zA-Z][a-zA-Z0-9_]*")
        self.assertIn("DROP TABLE ", self.sql)
        self.assertIn(" CASCADE CONSTRAINTS PURGE", self.sql)
        self.assertIn("KBOT\\_OPS\\_%", self.sql)
        self.assertIn("KBOT\\_V\\_OPS\\_%", self.sql)
        self.assertIn("Worker、Scheduler 和 DB Executor", self.sql)
        self.assertIn("SET SQLBLANKLINES ON", self.sql)
        self.assertNotIn("DROP USER", self.sql.upper())

if __name__ == "__main__":
    unittest.main()
