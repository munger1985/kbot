"""业务系统发布前刷新短期监控候选引用的静态契约测试。"""

from __future__ import annotations

import unittest
from pathlib import Path


AIOPS_ROOT = Path(__file__).resolve().parents[2] / "ui" / "aiops"


class BusinessSystemCandidateRefreshTest(unittest.TestCase):
    def test_commit_refreshes_discovery_refs_before_persisting_draft(self) -> None:
        script = (AIOPS_ROOT / "js" / "aiops-business-system-workspace.js").read_text(
            encoding="utf-8"
        )
        commit = script.split("async function commit()", 1)[1].split(
            "async function saveDraft()", 1
        )[0]

        self.assertIn("function selectedDiscoveryLocators", script)
        self.assertIn("candidate_locator_key", script)
        self.assertIn("host_candidate_locator_key", script)
        self.assertIn("async function refreshDiscoveryReferences", script)
        self.assertLess(
            commit.index("await refreshDiscoveryReferences()"),
            commit.index("await persistDraft(false)"),
        )

    def test_workspace_cache_key_changes_with_candidate_refresh_fix(self) -> None:
        page = (AIOPS_ROOT / "business-system-workspace.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("aiops-business-system-workspace.js?v=20261021_1", page)


if __name__ == "__main__":
    unittest.main()
