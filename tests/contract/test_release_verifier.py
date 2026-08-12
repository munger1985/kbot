"""发布证据生成器的离线契约测试。"""

import unittest

from scripts.release.verify_release import (
    ACTIVE_PACKAGES,
    _checks,
    build_input_manifest,
    resolve_profile_options,
)


class ReleaseVerifierTest(unittest.TestCase):
    def test_rc_profile_enforces_external_gates_and_clean_worktree(self):
        self.assertEqual(
            (True, True, True),
            resolve_profile_options(
                profile="rc",
                include_oracle=False,
                include_external_databases=False,
                require_clean=False,
            ),
        )

    def test_manifest_covers_schema_configuration_and_openapi(self):
        manifest = build_input_manifest()

        self.assertIn(
            "database/oracle/aiops_agent/schema_manifest.json",
            manifest,
        )
        self.assertIn(
            "configuration/kbot.toml.example",
            manifest,
        )
        self.assertIn("docs/openapi/aiops_internal_v1.json", manifest)
        self.assertIn("docs/openapi/data_query_internal_v1.json", manifest)
        self.assertIn("docs/openapi/main_api_public_v1.json", manifest)
        self.assertIn(
            "docs/openapi/knowledge_core_internal_v1.json",
            manifest,
        )
        self.assertIn("docs/openapi/km_asset_app_internal_v1.json", manifest)
        self.assertIn("database/oracle/km_asset_app/schema_manifest.json", manifest)
        self.assertIn("resources/topology.toml", manifest)
        self.assertNotIn("var/release/sbom/python-direct.cdx.json", manifest)
        self.assertTrue(
            all(len(digest) == 64 for digest in manifest.values())
        )

    def test_all_active_service_packages_are_compiled(self):
        for package in (
            "services/knowledge_core/src/knowledge_core",
            "services/km_asset_app/src/km_asset_app",
            "services/agent_runtime/src/agent_runtime",
            "services/aiops_agent/src/aiops_agent",
            "services/data_query/src/data_query",
            "services/main_api/src/main_api",
            "services/model_serving/src/model_serving",
        ):
            self.assertIn(package, ACTIVE_PACKAGES)

    def test_component_contract_uses_pytest_for_async_test_discovery(self):
        command = next(
            command
            for name, command, _ in _checks(include_oracle=False)
            if name == "unit_component_contract"
        )
        self.assertEqual("pytest", command[2])
        self.assertIn("tests/unit", command)
        self.assertIn("tests/contract", command)

    def test_oracle_profile_has_preflight_catalog_checks(self):
        names = {
            name
            for name, _, _ in _checks(include_oracle=True)
        }

        self.assertIn("oracle_object_catalog", names)
        self.assertIn("oracle_all_entity_catalog", names)
        self.assertIn("oracle_aiops_entity_catalog", names)
        self.assertIn("oracle_cross_service_uow", names)
        self.assertIn("oracle_aiops_persistence", names)
        self.assertIn("oracle_aiops_runtime", names)
        self.assertIn("oracle_data_query_runtime", names)
        self.assertIn("oracle_agent_memory", names)
        self.assertIn("oracle_knowledge_core_s3", names)
        self.assertIn("oracle_model_serving_s4", names)
        self.assertIn("oracle_notifications_s6", names)

    def test_external_database_smoke_is_explicitly_enabled(self):
        names = {
            name
            for name, _, _ in _checks(
                include_oracle=False,
                include_external_databases=True,
            )
        }

        self.assertIn("data_query_external_databases", names)

    def test_prometheus_check_is_explicitly_enabled(self):
        names = {
            name
            for name, _, _ in _checks(
                include_oracle=False,
                prometheus_url="http://localhost:9161/metrics",
            )
        }

        self.assertIn("prometheus_metrics", names)

    def test_release_test_programs_are_loaded_from_tests(self):
        commands = [
            command
            for _, command, _ in _checks(include_oracle=True)
        ]
        misplaced = [
            argument
            for command in commands
            for argument in command
            if argument.startswith(("scripts/check_", "scripts/smoke_"))
        ]
        self.assertEqual([], misplaced)


if __name__ == "__main__":
    unittest.main()
