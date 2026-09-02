"""AIOps Oracle 全量重建脚本合同测试。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from tools.db.render_aiops_rebuild_schema import (
    _analyze_canonical_sql,
    render_rebuild_sql,
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
MANIFEST = SCHEMA_DIR / "schema_manifest.json"
UPGRADE_SCHEMA_19 = (
    ROOT
    / "database"
    / "oracle"
    / "operations"
    / "upgrade_aiops_schema_19.sql"
)


class AIOpsRebuildSchemaScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sql = REBUILD_SCRIPT.read_text(encoding="utf-8")
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
        self.assertIn("l_required_column_count <> 8", self.sql)
        self.assertIn("l_report_summary_count <> 1", self.sql)
        self.assertIn("l_task_type_constraint_count <> 1", self.sql)
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

    def test_canonical_statement_counts_and_parentheses_match_manifest(self) -> None:
        for definition in self.manifest["scripts"]:
            content = (SCHEMA_DIR / definition["name"]).read_text(encoding="utf-8")
            self.assertEqual(
                definition["statements"],
                _analyze_canonical_sql(definition["name"], content),
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

    def test_turn_status_check_is_closed_before_next_constraint(self) -> None:
        self.assertIn(
            "'PROPOSAL_PENDING', 'COMPLETED', 'PARTIAL', 'FAILED', "
            "'CANCELLED'\n    )),\n"
            "    CONSTRAINT CK_OPS_TURN_SUFFICIENCY CHECK (",
            self.sql,
        )


if __name__ == "__main__":
    unittest.main()
