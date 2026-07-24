"""发布证据生成器的离线契约测试。"""

import unittest

from scripts.verify_release import (
    ACTIVE_PACKAGES,
    _checks,
    build_input_manifest,
)


class ReleaseVerifierTest(unittest.TestCase):
    def test_manifest_covers_schema_configuration_and_openapi(self):
        manifest = build_input_manifest()

        self.assertIn(
            "database/oracle/aiops_agent/schema_manifest.json",
            manifest,
        )
        self.assertIn(
            "configuration/example/services/agent_runtime/base.toml.example",
            manifest,
        )
        self.assertIn("docs/openapi/aiops_internal_v1.json", manifest)
        self.assertIn("configuration/process_topology.toml", manifest)
        self.assertTrue(
            all(len(digest) == 64 for digest in manifest.values())
        )

    def test_all_active_service_packages_are_compiled(self):
        for package in (
            "knowledge_core",
            "agent_runtime",
            "aiops_agent",
            "main_api",
            "model_serving",
        ):
            self.assertIn(package, ACTIVE_PACKAGES)

    def test_oracle_profile_has_preflight_catalog_checks(self):
        names = {
            name
            for name, _, _ in _checks(include_oracle=True)
        }

        self.assertIn("oracle_object_catalog", names)
        self.assertIn("oracle_aiops_entity_catalog", names)


if __name__ == "__main__":
    unittest.main()
