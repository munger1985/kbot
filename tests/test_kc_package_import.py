"""Knowledge Core 包边界导入回归测试。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class KnowledgeCorePackageImportTest(unittest.TestCase):
    def test_persistence_can_be_imported_in_fresh_process(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from knowledge_core.persistence import create_kc_uow",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
