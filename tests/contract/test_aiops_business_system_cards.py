"""KBot 业务系统目录与已确认的卡片方案保持一致。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PAGE = (ROOT / "ui/aiops/business-systems.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "ui/aiops/js/aiops-business-systems.js").read_text(encoding="utf-8")
STYLES = (ROOT / "ui/aiops/css/aiops-forms.css").read_text(encoding="utf-8")
GLOBAL_STYLES = (ROOT / "ui/aiops/css/aiops.css").read_text(encoding="utf-8")


class BusinessSystemCardContractTests(unittest.TestCase):
    def test_add_entry_is_a_grid_card_not_a_header_button(self):
        header = PAGE.split('<header class="ops-head">', 1)[1].split("</header>", 1)[0]
        self.assertNotIn("新增业务系统", header)
        self.assertIn('class="business-system-add-card"', SCRIPT)
        self.assertIn("＋", SCRIPT)

    def test_labels_are_localized_and_do_not_overclaim_protection(self):
        self.assertIn('SINGLE_INSTANCE: "单实例"', SCRIPT)
        self.assertIn('MYSQL_REPLICATION: "MySQL 主从复制"', SCRIPT)
        self.assertIn('POSTGRES_STREAMING: "PostgreSQL 流复制"', SCRIPT)
        self.assertNotIn("个组件 ·", SCRIPT)
        self.assertNotIn("已覆盖", SCRIPT)
        self.assertIn("<dt>数据保护</dt>", SCRIPT)
        self.assertNotIn("<dt>恢复目标</dt>", SCRIPT)

    def test_cards_resolve_responsibility_groups_and_reuse_global_tokens(self):
        self.assertIn('request("/responsibility-groups")', SCRIPT)
        self.assertIn("responsibilityGroupName", SCRIPT)
        self.assertIn(".business-system-directory-card", STYLES)
        self.assertIn("background:var(--paper)", STYLES)
        self.assertIn("border:1px solid var(--line)", STYLES)
        self.assertIn("--card-border:", GLOBAL_STYLES)
        self.assertIn("--card-shadow:", GLOBAL_STYLES)
        self.assertIn('[class$="-card"]', GLOBAL_STYLES)

    def test_cards_use_subtle_importance_tones_and_badges(self):
        self.assertIn('class="business-importance"', SCRIPT)
        self.assertIn("importance-level-${importanceLevel}", SCRIPT)
        self.assertIn("box-shadow:var(--shadow)", STYLES)
        for level in range(1, 6):
            self.assertIn(f".business-system-directory-card.importance-level-{level}", STYLES)


if __name__ == "__main__":
    unittest.main()
