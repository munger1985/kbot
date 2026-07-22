"""Focused checks for the isolated Knowledge Core service configuration."""

import unittest

from platform_core.config.settings import KnowledgeCoreConfig


class KnowledgeCoreConfigTest(unittest.TestCase):
    def test_defaults_use_dedicated_service_port(self):
        config = KnowledgeCoreConfig()

        self.assertEqual(config.service_port, 18090)
        self.assertEqual(config.service_version, "3.5.0")


if __name__ == "__main__":
    unittest.main()
