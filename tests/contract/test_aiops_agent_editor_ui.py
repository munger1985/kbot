"""AIOps Agent 编辑器静态交互边界测试。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ui" / "aiops" / "js" / "aiops-agents.js"
PAGE = ROOT / "ui" / "aiops" / "agents.html"


class AIOpsAgentEditorUiTest(unittest.TestCase):
    def test_source_mapping_only_traverses_monitoring_source_cards(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'document.querySelectorAll("#agent-sources '
            '.agent-source-card[data-source-id]")',
            script,
        )
        self.assertIn("sourceCards().forEach", script)
        self.assertIn("const choice = card.querySelector", script)
        self.assertIn("if (!choice || !mapping) return", script)
        self.assertNotIn(
            'document.querySelectorAll(".agent-source-card").forEach',
            script,
        )

    def test_page_uses_repaired_agent_editor_bundle(self) -> None:
        page = PAGE.read_text(encoding="utf-8")

        self.assertIn("aiops-agents.js?v=20260902-2", page)


if __name__ == "__main__":
    unittest.main()
