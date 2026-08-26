"""AIOps Oracle 全量重建脚本合同测试。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"
REBUILD_SCRIPT = SCHEMA_DIR / "rebuild_aiops_schema.sql"
MANIFEST = SCHEMA_DIR / "schema_manifest.json"


class AIOpsRebuildSchemaScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sql = REBUILD_SCRIPT.read_text(encoding="utf-8")
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_rebuild_uses_every_canonical_script_in_manifest_order(self) -> None:
        includes = re.findall(r"^@@([0-9]{3}_[a-z0-9_]+\.sql)$", self.sql, re.MULTILINE)
        expected = [item["name"] for item in self.manifest["scripts"]]

        self.assertEqual(expected, includes)
        self.assertNotRegex(self.sql, r"(?im)^\s*CREATE\s+TABLE\s+")
        self.assertNotRegex(self.sql, r"(?im)^\s*CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+")

    def test_rebuild_validation_matches_manifest_contract(self) -> None:
        expected_summary = (
            f"验证通过：{len(self.manifest['tables'])} 张表、"
            f"{len(self.manifest['views'])} 个视图，Schema Version "
            f"{self.manifest['schema_version']}，合同 "
            f"{self.manifest['contract_version']}。"
        )

        self.assertIn(f"l_table_count <> {len(self.manifest['tables'])}", self.sql)
        self.assertIn(f"l_view_count <> {len(self.manifest['views'])}", self.sql)
        self.assertIn(
            f"l_schema_version <> {self.manifest['schema_version']}", self.sql
        )
        self.assertIn(
            f"l_contract_version <> '{self.manifest['contract_version']}'",
            self.sql,
        )
        self.assertIn(expected_summary, self.sql)

    def test_rebuild_requires_target_and_destructive_confirmations(self) -> None:
        for required_fragment in (
            "ACCEPT expected_pdb",
            "ACCEPT expected_schema",
            "ACCEPT services_stopped",
            "ACCEPT rebuild_confirmation",
            "<> 'STOPPED'",
            "<> 'REBUILD_AIOPS'",
            "child_constraint.table_name NOT LIKE 'KBOT\\_OPS\\_%'",
        ):
            self.assertIn(required_fragment, self.sql)

        self.assertIn("DROP TABLE ", self.sql)
        self.assertIn(" CASCADE CONSTRAINTS PURGE", self.sql)
        self.assertNotIn("DROP USER", self.sql.upper())


if __name__ == "__main__":
    unittest.main()
