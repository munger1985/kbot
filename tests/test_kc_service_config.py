"""Focused checks for the isolated Knowledge Core service configuration."""

import unittest

from knowledge_core.config import KnowledgeCoreSettings


class KnowledgeCoreConfigTest(unittest.TestCase):
    def test_defaults_use_dedicated_service_port(self):
        config = KnowledgeCoreSettings()

        self.assertEqual(config.api.service_port, 18090)
        self.assertEqual(config.api.service_version, "4.0.0")


if __name__ == "__main__":
    unittest.main()
