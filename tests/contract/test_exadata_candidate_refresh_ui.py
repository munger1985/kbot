"""验证 Exadata 硬件发现与发布前候选引用刷新，不连接真实监控源。"""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ExadataCandidateRefreshUITest(unittest.TestCase):
    def test_candidate_refresh_behaviour(self):
        subprocess.run(
            ["node", str(ROOT / "tests/contract/aiops_exadata_candidate_refresh.mjs")],
            check=True, cwd=ROOT,
        )

    def test_publication_behaviour(self):
        script = ROOT / "tests/contract/aiops_workspace_publication.mjs"
        if not script.exists():
            script = ROOT / "tests/contract/aiops_oem_onboarding.mjs"
        subprocess.run(["node", str(script)], check=True, cwd=ROOT)

    def test_existing_hardware_behaviour(self):
        subprocess.run(
            ["node", str(ROOT / "tests/contract/aiops_resource_monitoring.mjs")],
            check=True, cwd=ROOT,
        )


if __name__ == "__main__":
    unittest.main()
