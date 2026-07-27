"""KBot 4.0 Vanilla JavaScript 测试页面的静态契约检查。"""

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = ROOT / "tools" / "dev_console"


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.scripts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "script" and values.get("src"):
            self.scripts.append(values["src"])
        if tag == "link" and values.get("href"):
            self.links.append(values["href"])


class UiStaticPagesTest(unittest.TestCase):
    def test_pages_reference_existing_local_assets(self):
        for page_name in (
            "index.html",
            "knowledge-core.html",
            "agent-chat.html",
            "agent-debug.html",
            "operations-logs.html",
        ):
            page = UI_ROOT / page_name
            parser = _PageParser()
            parser.feed(page.read_text(encoding="utf-8"))
            for reference in [*parser.scripts, *parser.links]:
                if reference.startswith("./"):
                    self.assertTrue(
                        (UI_ROOT / reference[2:]).is_file(),
                        f"{page_name} 缺少资源 {reference}",
                    )

    def test_feature_pages_expose_required_controls(self):
        requirements = {
            "knowledge-core.html": {
                "auth-form",
                "domain-form",
                "collection-form",
                "generate-collection-key",
                "upload-form",
                "tracking-rows",
                "status-form",
                "agent-form",
                "agent-rows",
                "binding-form",
                "resource-setup",
                "ingestion-flow",
                "agent-setup",
                "diagnostics",
            },
            "agent-chat.html": {
                "auth-form",
                "agent-select",
                "conversation-select",
                "turn-form",
                "live-stream",
                "timeline",
                "result-output",
            },
            "operations-logs.html": {
                "auth-form",
                "log-filter-form",
                "service-filter",
                "log-type-filter",
                "refresh-interval",
                "log-rows",
                "event-detail",
            },
            "agent-debug.html": {
                "auth-form",
                "run-filter-form",
                "run-list",
                "run-overview",
                "run-timeline",
                "tab-retrieval",
                "tab-models",
                "tab-tasks",
                "tab-events",
                "tab-artifacts",
                "tab-logs",
                "tab-errors",
                "debug-detail",
            },
        }
        for page_name, expected in requirements.items():
            parser = _PageParser()
            parser.feed(
                (UI_ROOT / page_name).read_text(encoding="utf-8")
            )
            self.assertTrue(expected.issubset(parser.ids))

    def test_javascript_syntax(self):
        for script in (
            "shared.js",
            "knowledge-core.js",
            "agent-chat.js",
            "agent-debug.js",
            "operations-logs.js",
        ):
            result = subprocess.run(
                ["node", "--check", str(UI_ROOT / script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_ui_uses_explicit_development_auth_bypass(self):
        source = (UI_ROOT / "shared.js").read_text(encoding="utf-8")
        self.assertIn('"X-KBot-Test-Auth": "true"', source)
        self.assertNotIn("Authorization:", source)
        self.assertNotIn("apiKey", source)

    def test_agent_chat_defaults_to_full_test_security_visibility(self):
        html = (UI_ROOT / "agent-chat.html").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'name="securityLevel" type="number" min="0" max="3" value="3"',
            html,
        )

    def test_knowledge_page_supports_direct_file_grouping(self):
        html = (UI_ROOT / "knowledge-core.html").read_text(
            encoding="utf-8"
        )
        script = (UI_ROOT / "knowledge-core.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('value="EACH_FILE"', html)
        self.assertIn('value="SINGLE_BUNDLE"', html)
        self.assertIn(
            "/ingestions/user-files",
            script,
        )
        self.assertIn('KBotUI.api("/api/v1/domains"', script)
        self.assertIn("generateCollectionKey()", script)
        self.assertIn('KBotUI.api("/api/v1/agents"', script)
        self.assertIn("enabled_capabilities", script)
        self.assertIn("memory_embedding", script)
        self.assertIn("error_id=", script)
        self.assertIn("item.error_code", script)
        self.assertNotIn("km-assets", script)

    def test_development_log_page_includes_two_logs_and_all_levels(self):
        html = (UI_ROOT / "operations-logs.html").read_text(
            encoding="utf-8"
        )
        script = (UI_ROOT / "operations-logs.js").read_text(
            encoding="utf-8"
        )
        for value in (
            "RUNTIME",
            "ACCESS",
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ):
            self.assertIn(f'value="{value}"', html)
        self.assertIn("/api/v1/development/logs/events", script)
        self.assertIn("/api/v1/development/logs/services", script)

    def test_agent_debug_page_uses_run_aggregation_api(self):
        script = (UI_ROOT / "agent-debug.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("/api/v1/development/agent-runs", script)
        self.assertIn("retrieval_report", script)
        self.assertIn("diagnostics.discovery", script)
        self.assertIn("diagnostics.evidence", script)


if __name__ == "__main__":
    unittest.main()
