"""AIOps Oracle Schema 的离线契约测试。"""

import hashlib
import json
from pathlib import Path
import re
import unittest

from scripts.db.apply_oracle_schema import split_oracle_statements
from tests.acceptance.check_oracle_schema import SERVICE_TABLES, SERVICE_VIEWS


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"


class AIOpsOracleSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scripts = sorted(SCHEMA_DIR.glob("[0-9][0-9][0-9]_*.sql"))
        self.sql = "\n".join(
            path.read_text(encoding="utf-8") for path in self.scripts
        )
        self.upper_sql = self.sql.upper()
        self.manifest = json.loads(
            (SCHEMA_DIR / "schema_manifest.json").read_text(encoding="utf-8")
        )

    def test_manifest_freezes_script_order_hash_and_statement_count(self) -> None:
        entries = self.manifest["scripts"]
        self.assertEqual(
            [path.name for path in self.scripts],
            [entry["name"] for entry in entries],
        )
        for path, entry in zip(self.scripts, entries):
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                entry["sha256"],
            )
            self.assertEqual(
                len(split_oracle_statements(path.read_text(encoding="utf-8"))),
                entry["statements"],
            )

    def test_manifest_owns_exact_tables_and_views(self) -> None:
        self.assertEqual(
            SERVICE_TABLES["aiops_agent"],
            set(self.manifest["tables"]),
        )
        self.assertEqual(
            SERVICE_VIEWS["aiops_agent"],
            set(self.manifest["views"]),
        )
        self.assertEqual(
            SERVICE_TABLES["aiops_agent"],
            set(
                re.findall(
                    r"\bCREATE\s+TABLE\s+([A-Z][A-Z0-9_]*)",
                    self.upper_sql,
                )
            ),
        )

    def test_concurrency_and_history_constraints_are_present(self) -> None:
        self.assertEqual(
            len(self.manifest["deferred_foreign_keys"]),
            self.upper_sql.count("DEFERRABLE INITIALLY DEFERRED"),
        )
        for name in self.manifest["function_unique_indexes"]:
            self.assertRegex(
                self.upper_sql,
                rf"\bCREATE\s+UNIQUE\s+INDEX\s+{name}\b",
            )
        for definition in self.manifest["foreign_key_indexes"]:
            columns = r"\s*,\s*".join(definition["columns"])
            self.assertRegex(
                self.upper_sql,
                rf"\bCREATE\s+INDEX\s+{definition['index']}\s+"
                rf"ON\s+{definition['table']}\s*\(\s*{columns}\s*\)",
            )
        self.assertIn("STATUS = 'RUNNING' AND LEASE_OWNER IS NOT NULL", self.upper_sql)
        self.assertIn(
            "STATUS = 'PUBLISHING' AND LEASE_OWNER IS NOT NULL",
            self.upper_sql,
        )
        self.assertIn("THEN CORRELATION_HASH", self.upper_sql)
        self.assertNotIn("THEN FINGERPRINT", self.upper_sql)

    def test_oracle_26ai_physical_adaptations_are_explicit(self) -> None:
        self.assertNotRegex(self.upper_sql, r"\bMODE\s+VARCHAR2\b")
        self.assertIn("EXECUTION_KIND VARCHAR2(16 CHAR)", self.upper_sql)
        self.assertIn(
            "SYS_EXTRACT_UTC(SCHEDULED_FOR)",
            self.upper_sql,
        )
        self.assertIn(
            "UNIQUE (INSPECTION_PLAN_ID, SCHEDULED_FOR_UTC)",
            self.upper_sql,
        )

    def test_apex_views_do_not_project_secrets_or_command_payloads(self) -> None:
        protected_tokens = (
            "DIAGNOSTIC_SECRET_REF",
            "EXECUTION_SECRET_REF",
            "SECRET_REF",
            "WEBHOOK_SECRET_REF",
            "WEBHOOK_KEY_HASH",
            "PREVIOUS_WEBHOOK_KEY_HASH",
            "ENDPOINT_JSON",
            "PARAMETERS_JSON",
            "EVIDENCE_ARTIFACTS_JSON",
            "RULES_JSON",
        )
        view_script = (
            SCHEMA_DIR / "006_ops_fks_views.sql"
        ).read_text(encoding="utf-8").upper()
        view_area = view_script[
            view_script.index("CREATE OR REPLACE VIEW KBOT_V_OPS_TARGET") :
        ]
        for token in protected_tokens:
            self.assertNotIn(token, view_area)

    def test_schema_contains_no_seed_or_destructive_ddl(self) -> None:
        for token in (
            "INSERT INTO",
            "MERGE INTO",
            "DROP TABLE",
            "DROP VIEW",
            "KBOT_MD_",
            "TXTCHUNK",
        ):
            self.assertNotIn(token, self.upper_sql)


if __name__ == "__main__":
    unittest.main()
