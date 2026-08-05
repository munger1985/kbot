"""Data Query Oracle Schema 的离线契约测试。"""

import hashlib
import json
from pathlib import Path
import re
import unittest

from scripts.db.apply_oracle_schema import split_oracle_statements
from tests.acceptance.check_oracle_schema import SERVICE_TABLES, SERVICE_VIEWS


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = ROOT / "database" / "oracle" / "data_query"


class DataQueryOracleSchemaTest(unittest.TestCase):
    def setUp(self):
        self.scripts = sorted(SCHEMA_DIR.glob("[0-9][0-9][0-9]_*.sql"))
        self.sql = "\n".join(path.read_text(encoding="utf-8") for path in self.scripts)
        self.upper_sql = self.sql.upper()
        self.manifest = json.loads((SCHEMA_DIR / "schema_manifest.json").read_text(encoding="utf-8"))

    def test_manifest_freezes_order_hash_and_statement_count(self):
        entries = self.manifest["scripts"]
        self.assertEqual([path.name for path in self.scripts], [item["name"] for item in entries])
        for path, item in zip(self.scripts, entries):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            self.assertEqual(len(split_oracle_statements(path.read_text(encoding="utf-8"))), item["statements"])

    def test_manifest_owns_exact_tables_and_views(self):
        self.assertEqual(SERVICE_TABLES["data_query"], set(self.manifest["tables"]))
        self.assertEqual(SERVICE_VIEWS["data_query"], set(self.manifest["views"]))
        self.assertEqual(
            SERVICE_TABLES["data_query"],
            set(re.findall(r"\bCREATE\s+TABLE\s+([A-Z][A-Z0-9_]*)", self.upper_sql)),
        )

    def test_credentials_and_leases_are_constrained(self):
        self.assertIn("USERNAME_NONCE RAW(12) NOT NULL", self.upper_sql)
        self.assertIn("PASSWORD_NONCE RAW(12) NOT NULL", self.upper_sql)
        self.assertIn("STATUS = 'EXECUTING' AND LEASE_OWNER IS NOT NULL", self.upper_sql)
        self.assertIn("STATUS = 'RUNNING' AND LEASE_OWNER IS NOT NULL", self.upper_sql)
        for name in self.manifest["function_unique_indexes"]:
            self.assertRegex(self.upper_sql, rf"\bCREATE\s+UNIQUE\s+INDEX\s+{name}\b")

    def test_schema_has_no_secret_refs_or_user_role_policy(self):
        for token in ("SECRET_REF", "SUBJECT_SELECTOR", "ROLE_ID", "INSERT INTO", "MERGE INTO", "DROP TABLE"):
            self.assertNotIn(token, self.upper_sql)


if __name__ == "__main__":
    unittest.main()
