"""验证业务系统数据库的历史 SQL 采集入口。"""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


class WorkloadSnapshotUiTest(unittest.TestCase):
    def test_node_behaviour(self):
        subprocess.run(["node", str(ROOT / "tests/contract/aiops_workload_snapshot.mjs")], check=True, cwd=ROOT)

    def test_workspace_requires_target_manage_permission(self):
        shell = (ROOT / "ui/aiops/js/aiops-shell.js").read_text(encoding="utf-8")
        self.assertIn('"business-system-workspace": "aiops:target_manage"', shell)


if __name__ == "__main__":
    unittest.main()
