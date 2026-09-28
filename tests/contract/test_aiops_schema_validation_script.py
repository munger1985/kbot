"""AIOps Oracle Schema 完整性验证脚本合同测试。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from tools.db.render_aiops_schema_validation import (
    _extract_named_objects,
    render_validation_sql,
)
from tools.db.render_aiops_rebuild_schema import _load_canonical_sections
from tools.db.render_aiops_rebuild_schema import _extract_table_columns


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"
MANIFEST_PATH = SCHEMA_DIR / "schema_manifest.json"
VALIDATION_SCRIPT = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "aiops_agent"
    / "validate_aiops_schema.sql"
)


class AIOpsSchemaValidationScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.sql = VALIDATION_SCRIPT.read_text(encoding="utf-8")

    def test_generated_script_matches_current_canonical_ddl(self) -> None:
        self.assertEqual(render_validation_sql(), self.sql)
        self.assertNotRegex(self.sql, r"(?m)^@@")
        self.assertNotRegex(
            self.sql.upper(),
            r"\b(?:CREATE|ALTER|DROP|TRUNCATE|INSERT|UPDATE|DELETE|MERGE)\b"
            r"(?![^\n]*--)",
        )

    def test_script_contains_every_table_view_index_and_constraint(self) -> None:
        _, canonical_sections = _load_canonical_sections()
        indexes, constraints = _extract_named_objects(canonical_sections)

        for name in self.manifest["tables"]:
            self.assertIn(f"'{name}'", self.sql)
        for name in self.manifest["views"]:
            self.assertIn(f"'{name}'", self.sql)
        for name in indexes:
            self.assertIn(f"'{name}'", self.sql)
        for name in constraints:
            self.assertIn(f"'{name}'", self.sql)

        self.assertIn(f"{len(indexes)} 个命名索引", self.sql)
        self.assertIn(f"{len(constraints)} 个命名约束", self.sql)

    def test_script_checks_bidirectional_differences_and_health(self) -> None:
        for marker in (
            "[缺失] AIOps 表",
            "[多余] AIOps 表",
            "[缺失] AIOps 视图",
            "[缺失] AIOps 命名索引",
            "[缺失] AIOps 命名约束",
            "[缺失] AIOps 表列",
            "status <> 'VALID'",
            "status <> 'ENABLED' OR validated <> 'VALIDATED'",
            "raise_application_error(",
        ):
            self.assertIn(marker, self.sql)

    def test_script_checks_current_schema_contract(self) -> None:
        self.assertIn(
            f"l_schema_version <> {self.manifest['schema_version']}",
            self.sql,
        )
        self.assertIn(
            f"l_contract_version <> '{self.manifest['contract_version']}'",
            self.sql,
        )
        self.assertIn("FROM KBOT_V_OPS_SCHEMA_VERSION", self.sql)

    def test_every_canonical_table_column_is_embedded(self) -> None:
        table_columns = re.findall(
            r"'((?:KBOT_OPS_[A-Z0-9_]+)\|(?:[A-Z][A-Z0-9_]*))'",
            self.sql,
        )
        self.assertGreater(len(set(table_columns)), 300)
        self.assertEqual(len(table_columns), len(set(table_columns)) * 2)

    def test_virtual_column_continuation_is_not_treated_as_column(self) -> None:
        _, canonical_sections = _load_canonical_sections()
        columns = _extract_table_columns(
            canonical_sections,
            "KBOT_OPS_INSPECTION_FIRE",
        )

        self.assertIn("SCHEDULED_FOR_UTC", columns)
        self.assertIn("STATUS", columns)
        self.assertNotIn("GENERATED", columns)
        self.assertNotIn("KBOT_OPS_INSPECTION_FIRE|GENERATED", self.sql)


if __name__ == "__main__":
    unittest.main()
