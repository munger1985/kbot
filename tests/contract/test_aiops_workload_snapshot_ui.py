"""验证业务系统数据库的历史 SQL 采集入口。"""

from pathlib import Path
import json
import subprocess
import unittest

from platform_core.contracts.aiops.workload import WorkloadPolicy

ROOT = Path(__file__).resolve().parents[2]


class WorkloadSnapshotUiTest(unittest.TestCase):
    def test_new_resource_defaults_match_backend_contract(self):
        script = (ROOT / "ui/aiops/js/aiops-business-system-workspace.js").read_text(encoding="utf-8")
        helper = script[script.index("  function createWorkloadPolicy("):script.index("  function validWorkloadPolicy(")]
        result = subprocess.run(["node", "-e", helper + "process.stdout.write(JSON.stringify(createWorkloadPolicy()));"], check=True, capture_output=True, text=True)
        self.assertEqual(WorkloadPolicy().model_dump(mode="json"), json.loads(result.stdout))

    def test_node_behaviour(self):
        subprocess.run(["node", str(ROOT / "tests/contract/aiops_workload_snapshot.mjs")], check=True, cwd=ROOT)

    def test_publication_behaviour(self):
        subprocess.run(["node", str(ROOT / "tests/contract/aiops_workspace_publication.mjs")], check=True, cwd=ROOT)

    def test_workspace_requires_target_manage_permission(self):
        shell = (ROOT / "ui/aiops/js/aiops-shell.js").read_text(encoding="utf-8")
        self.assertIn('"business-system-workspace": "aiops:target_manage"', shell)

    def test_resource_drawer_rows_do_not_clip_contents(self):
        css = (ROOT / "ui/aiops/css/aiops-forms.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"\.business-resource-drawer-body\s*\{[^}]*grid-auto-rows\s*:\s*max-content")


if __name__ == "__main__":
    unittest.main()
