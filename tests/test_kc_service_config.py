"""Focused checks for the isolated Knowledge Core service configuration."""

import unittest
import subprocess
import sys
from pathlib import Path

from knowledge_core.config import KnowledgeCoreSettings


ROOT = Path(__file__).resolve().parents[1]


class KnowledgeCoreConfigTest(unittest.TestCase):
    def test_defaults_use_dedicated_service_port(self):
        config = KnowledgeCoreSettings()

        self.assertEqual(config.api.service_port, 18090)
        self.assertEqual(config.api.service_version, "4.0.0")

    def test_parser_process_module_can_be_imported(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import apps.knowledge_core_parser.main",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
