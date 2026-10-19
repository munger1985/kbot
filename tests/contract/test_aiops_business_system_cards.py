"""KBot 业务系统目录与已确认的卡片方案保持一致。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PAGE = (ROOT / "ui/aiops/business-systems.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "ui/aiops/js/aiops-business-systems.js").read_text(encoding="utf-8")
WORKSPACE_PAGE = (ROOT / "ui/aiops/business-system-workspace.html").read_text(encoding="utf-8")
WORKSPACE_SCRIPT = (ROOT / "ui/aiops/js/aiops-business-system-workspace.js").read_text(encoding="utf-8")
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
        self.assertIn('"数据保护"', SCRIPT)
        self.assertNotIn("<dt>恢复目标</dt>", SCRIPT)

    def test_cards_resolve_responsibility_groups_and_reuse_global_tokens(self):
        self.assertIn('request("/responsibility-groups")', SCRIPT)
        self.assertIn("responsibilityGroupName", SCRIPT)
        self.assertIn(".business-system-directory-card", STYLES)
        self.assertIn("background:var(--paper)", STYLES)
        self.assertIn("border:1px solid var(--line)", STYLES)
        self.assertIn("--card-border:", GLOBAL_STYLES)
        self.assertIn("--card-shadow:", GLOBAL_STYLES)
        self.assertIn("--card-shadow-hover:", GLOBAL_STYLES)
        self.assertIn("box-shadow:var(--card-shadow-hover)", GLOBAL_STYLES)
        self.assertIn("transform:translateY(-2px)", GLOBAL_STYLES)
        self.assertIn('[class$="-card"]', GLOBAL_STYLES)
        hover_rule = next(
            rule for rule in GLOBAL_STYLES.split("\n")
            if "box-shadow:var(--card-shadow-hover)" in rule
        )
        self.assertNotIn(".ops-panel", hover_rule)

    def test_cards_use_importance_badges_without_tinting_the_card(self):
        self.assertIn('class="business-importance"', SCRIPT)
        self.assertIn("importance-level-${importanceLevel}", SCRIPT)
        self.assertNotIn(".business-system-directory-card::before", STYLES)
        self.assertNotIn("background:linear-gradient", STYLES)
        self.assertNotIn(".business-system-directory-card:hover", STYLES)
        for level in range(1, 6):
            self.assertIn(f".business-system-directory-card.importance-level-{level}", STYLES)

    def test_card_actions_use_two_rows_to_avoid_horizontal_overflow(self):
        self.assertIn(".business-system-directory-card>.business-system-card-actions{display:grid", STYLES)
        self.assertIn(".business-system-card-actions>span{grid-column:1/-1", STYLES)
        self.assertIn(".business-system-card-actions>div{display:flex;grid-column:1/-1", STYLES)

    def test_workspace_recovers_from_a_stale_draft_reference(self):
        self.assertIn("function clearStaleDraftReference()", WORKSPACE_SCRIPT)
        self.assertIn('currentUrl.searchParams.delete("draft")', WORKSPACE_SCRIPT)
        self.assertIn("if (error.status !== 404) throw error;", WORKSPACE_SCRIPT)
        self.assertIn("draftRevision >= 1", WORKSPACE_SCRIPT)
        self.assertIn('method: "POST"', WORKSPACE_SCRIPT)
        self.assertIn(
            "aiops-business-system-workspace.js?v=20261019_4",
            WORKSPACE_PAGE,
        )

    def test_workspace_uses_the_public_oracle_container_scope_contract(self):
        self.assertIn(
            'const oracleScopeOptions = [["NON_CDB", "非 CDB"], ["CDB_ROOT", "CDB"], ["PDB", "PDB"]]',
            WORKSPACE_SCRIPT,
        )
        self.assertEqual(
            2,
            WORKSPACE_SCRIPT.count(
                "selectOptions(oracleScopeOptions, database.oracle_scope)"
            ),
        )
        self.assertIn(
            'database.oracle_scope === "CDB" ? "CDB_ROOT" : database.oracle_scope',
            WORKSPACE_SCRIPT,
        )

    def test_workspace_shows_connection_test_feedback_inside_the_drawer(self):
        self.assertIn("const connectionTestStates = new Map()", WORKSPACE_SCRIPT)
        self.assertIn(
            'id="resource-connection-result" class="ops-connection-result"',
            WORKSPACE_SCRIPT,
        )
        self.assertIn('status: "CHECKING", message: "正在测试数据库连接…"', WORKSPACE_SCRIPT)
        self.assertIn('connectionTestRunning ? "正在测试…" : "测试连接"', WORKSPACE_SCRIPT)
        validation = WORKSPACE_SCRIPT.split(
            "async function validateResource(item)", 1
        )[1].split("function topologyBoardHtml()", 1)[0]
        self.assertIn('status: "SUCCEEDED"', validation)
        self.assertIn('status: "FAILED"', validation)
        self.assertNotIn("shell.toast", validation)


if __name__ == "__main__":
    unittest.main()
