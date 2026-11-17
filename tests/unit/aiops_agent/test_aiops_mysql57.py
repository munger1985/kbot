"""MySQL 5.7 的版本、监控与只读诊断能力边界回归。"""

import asyncio
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from aiops_agent.application.configuration.connection_test import (
    _database_version_error, _probe_mysql_capabilities,
    test_target_connection as run_connection_test,
)
from aiops_agent.actions import ActionRegistry
from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.application.diagnosis.native_measurements import native_mapping_catalog
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.monitoring.catalog import load_monitoring_profile_catalog
from platform_core.contracts.aiops import TargetConnectionTest
from platform_core.contracts.aiops.configuration import (
    DATABASE_SQL_PROFILES, validate_supported_database_version,
)

CONNECTION_MODULE = "aiops_agent.application.configuration.connection_test"
ROOT = Path(__file__).resolve().parents[3]


def request(version="5.7"):
    return TargetConnectionTest.model_validate({
        "db_type": "MYSQL", "version_code": version,
        "endpoint": {"host": "db.internal", "port": 3306, "database": "app", "tls_enabled": False},
        "diagnostic_credential": {"username": "test", "password": "test-only"},
    })


class MySQL57SupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = DiagnosticRegistry.load()

    def test_contract_accepts_only_published_mysql_versions(self):
        for version in ("5.7", "8.4"):
            validate_supported_database_version("MYSQL", version)
            self.assertEqual(request(version).version_code, version)
        self.assertEqual(DATABASE_SQL_PROFILES[("MYSQL", "5.7")], "mysql_5_7")
        for version in ("5.6", "5", "5.7.44", "8.0", "9.0"):
            with self.subTest(version=version), self.assertRaises(ValidationError):
                request(version)
        with self.assertRaises(ValueError):
            validate_supported_database_version("POSTGRESQL", "5.7")

    def test_detection_distinguishes_mismatch_and_unsupported(self):
        for selected, detected, expected in (
            ("5.7", "5.7.44", None),
            ("8.4", "8.4.0", None),
            ("8.4", "5.7.44", "DATABASE_VERSION_MISMATCH"),
            ("5.7", "8.4.0", "DATABASE_VERSION_MISMATCH"),
            ("5.7", "5.6.51", "UNSUPPORTED_DATABASE_VERSION"),
            ("5.7", "8.0.40", "UNSUPPORTED_DATABASE_VERSION"),
        ):
            with self.subTest(selected=selected, detected=detected):
                self.assertEqual(_database_version_error(db_type="MYSQL",
                    selected_version=selected, detected_version=detected), expected)

    def test_connection_result_preserves_detected_57(self):
        async def check():
            with patch(CONNECTION_MODULE + "._test_mysql", new=AsyncMock(
                return_value=("5.7.44", None, None, ("information_schema",), ("PROCESS",), {}))):
                result = await run_connection_test(request())
                self.assertTrue(result.ok)
                self.assertEqual(result.database_version, "5.7.44")
                self.assertEqual(result.supported_version_code, "5.7")
                mismatch = await run_connection_test(request("8.4"))
                self.assertFalse(mismatch.ok)
                self.assertEqual(mismatch.error_code, "DATABASE_VERSION_MISMATCH")
        asyncio.run(check())

    def test_capability_probe_does_not_claim_8_only_features(self):
        async def available(_cursor, sql):
            return not any(token in sql for token in ("data_lock_waits", "column_statistics"))
        async def check():
            with patch(CONNECTION_MODULE + "._mysql_query_succeeds", side_effect=available), \
                 patch(CONNECTION_MODULE + "._mysql_fetchone", new=AsyncMock(return_value=None)), \
                 patch(CONNECTION_MODULE + "._mysql_current_privileges", new=AsyncMock(return_value=("PROCESS",))):
                result = await _probe_mysql_capabilities(cursor=object(), database_version="5.7.44")
                self.assertNotIn("session_management", result[3])
                self.assertNotIn("histograms", result[3])
                self.assertNotIn("data_lock_waits", result[3])
                self.assertEqual(result[5]["data_lock_waits"], "UNAVAILABLE")
        asyncio.run(check())

    def test_overview_metrics_are_available_for_both_versions(self):
        profiles = load_monitoring_profile_catalog(load_metric_catalog())
        old = profiles.resolve("mysql-overview", (("MYSQL", "8.4"),))
        new = profiles.resolve("mysql-overview", (("MYSQL", "5.7"),))
        self.assertEqual(old.metric_codes, new.metric_codes)
        self.assertEqual(len(new.metric_codes), 7)
        self.assertEqual(profiles.overview_for_database("MYSQL", "5.7").profile_id, "mysql-overview")
        with self.assertRaises(LookupError):
            profiles.resolve("mysql-overview", (("MYSQL", "5.6"),))

    def test_57_diagnostics_resolve_without_changing_8_variants(self):
        tools = [item for item in self.registry.tools if item.definition.db_type == "MYSQL"
                 and item.definition.variant.startswith("mysql_5_7_")]
        self.assertEqual(len(tools), 15)
        bindings = native_mapping_catalog()[1]
        for item in tools:
            definition = item.definition
            with self.subTest(tool_id=definition.tool_id):
                common = dict(tool_id=definition.tool_id, tool_version=definition.version,
                              db_type="MYSQL", capabilities=set(definition.required_capabilities))
                resolved = self.registry.resolve(**common, db_version="5.7")
                self.assertEqual(resolved.definition.variant, definition.variant)
                common["capabilities"].add("sys_schema")
                self.assertFalse(self.registry.resolve(**common, db_version="8.4")
                                 .definition.variant.startswith("mysql_5_7_"))
                self.assertNotIn("data_lock_waits", item.sql.lower())
                if definition.tool_id in {"db.instance.performance", "db.mysql.instance.throughput"}:
                    self.assertIn(("MYSQL", definition.tool_id, definition.version,
                                   definition.variant, definition.template_sha256), bindings)

    def test_8_only_diagnostics_and_mutations_are_unavailable_on_57(self):
        for tool_id in ("db.mysql.lock.current", "db.mysql.replication.lag",
                        "db.sql.object_dependencies", "db.mysql.statistics.health",
                        "db.sql.problem_candidates"):
            with self.subTest(tool_id=tool_id), self.assertRaises(LookupError):
                self.registry.resolve(tool_id=tool_id, tool_version="1.0.0", db_type="MYSQL",
                    db_version="5.7", capabilities={"data_lock_waits", "information_schema",
                                                  "replication_views", "statement_digest"})
        actions = ActionRegistry.load()
        self.assertEqual(actions.compatible(db_type="MYSQL", db_version="5.7",
            capabilities={"session_management"}, environment="DEVELOPMENT"), ())

    def test_openapi_version_enums_include_57(self):
        files = ("aiops_internal_v1.json", "aiops_public_v1.json", "main_api_public_v1.json")
        def enums(node):
            if isinstance(node, dict):
                if "enum" in node and "26ai" in node["enum"]:
                    yield node["enum"]
                for value in node.values():
                    yield from enums(value)
            elif isinstance(node, list):
                for value in node:
                    yield from enums(value)
        for filename in files:
            with self.subTest(filename=filename):
                values = list(enums(json.loads((ROOT / "docs/openapi" / filename).read_text())))
                self.assertEqual(len(values), 4)
                self.assertTrue(all("5.7" in value for value in values))

    def test_workspace_offers_mysql_57(self):
        source = (ROOT / "ui/aiops/js/aiops-business-system-workspace.js").read_text()
        self.assertIn('MYSQL: ["5.7", "8.4"]', source)


if __name__ == "__main__":
    unittest.main()
