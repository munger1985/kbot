"""AIOps Oracle 全量重建脚本合同测试。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from scripts.db.render_aiops_rebuild_schema import (
    PRESERVED_CONFIGURATION_TABLES,
    UPGRADE_SCRIPT_NAMES,
    render_rebuild_sql,
    render_upgrade_sql,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"
REBUILD_SCRIPT = SCHEMA_DIR / "rebuild_aiops_schema.sql"
UPGRADE_SCRIPT = SCHEMA_DIR / "upgrade_aiops_v12_to_v13.sql"
MANIFEST = SCHEMA_DIR / "schema_manifest.json"


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

    def test_rebuild_is_non_interactive_and_scoped_to_aiops_objects(self) -> None:
        self.assertNotIn("ACCEPT ", self.sql.upper())
        self.assertNotRegex(self.sql, r"&[a-zA-Z][a-zA-Z0-9_]*")
        self.assertIn("DROP TABLE ", self.sql)
        self.assertIn(" CASCADE CONSTRAINTS PURGE", self.sql)
        self.assertIn("KBOT\\_OPS\\_%", self.sql)
        self.assertIn("KBOT\\_V\\_OPS\\_%", self.sql)
        self.assertNotIn("DROP USER", self.sql.upper())


class AIOpsUpgradeSchemaScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sql = UPGRADE_SCRIPT.read_text(encoding="utf-8")
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_upgrade_is_generated_from_selected_canonical_scripts(self) -> None:
        sections = re.findall(
            r"^-- ===== 开始规范 DDL：([0-9]{3}_[a-z0-9_]+\.sql) =====$",
            self.sql,
            re.MULTILINE,
        )

        self.assertEqual(list(UPGRADE_SCRIPT_NAMES), sections)
        self.assertEqual(render_upgrade_sql(), self.sql)
        self.assertNotRegex(self.sql, r"(?m)^@@")
        self.assertEqual(
            len(self.manifest["tables"])
            - len(PRESERVED_CONFIGURATION_TABLES),
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

    def test_upgrade_requires_v12_and_preserves_configuration_tables(self) -> None:
        self.assertIn("l_schema_version <> 12", self.sql)
        self.assertIn("l_contract_version <> 'aiops-oracle-v2'", self.sql)
        self.assertIn("table_name NOT IN", self.sql)
        for table_name in PRESERVED_CONFIGURATION_TABLES:
            self.assertIn(f"'{table_name}'", self.sql)
        self.assertIn("UK_OPS_TARGET_OWNER", self.sql)
        self.assertIn("UK_OPS_AGENT_VER_OWNER", self.sql)

    def test_upgrade_validates_v13_manifest_contract(self) -> None:
        self.assertIn(f"l_table_count <> {len(self.manifest['tables'])}", self.sql)
        self.assertIn(f"l_view_count <> {len(self.manifest['views'])}", self.sql)
        self.assertIn(
            f"l_schema_version <> {self.manifest['schema_version']}",
            self.sql,
        )
        self.assertIn(
            f"l_contract_version <> '{self.manifest['contract_version']}'",
            self.sql,
        )
        self.assertNotIn("DROP USER", self.sql.upper())


if __name__ == "__main__":
    unittest.main()
