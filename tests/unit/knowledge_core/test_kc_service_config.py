"""Knowledge Core 独立服务配置检查。"""

import os
import unittest
import subprocess
import sys
from pathlib import Path

from knowledge_core.config import KnowledgeCoreSettings


ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = os.pathsep.join(
    (
        str(ROOT / "packages" / "platform_core" / "src"),
        str(ROOT / "packages" / "platform_clients" / "src"),
        str(ROOT / "services" / "knowledge_core" / "src"),
    )
)


class KnowledgeCoreConfigTest(unittest.TestCase):
    def test_defaults_use_dedicated_service_port(self):
        config = KnowledgeCoreSettings()

        self.assertEqual(config.api.service_port, 18090)
        self.assertEqual(config.api.service_version, "4.0.0")
        self.assertEqual(config.job_wakeup.mode, "DBMS_ALERT")
        self.assertEqual(
            config.job_wakeup.notification_timeout_seconds,
            30,
        )

    def test_parser_process_module_can_be_imported(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import knowledge_core.entrypoints.parser",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": SOURCE_PATH},
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
