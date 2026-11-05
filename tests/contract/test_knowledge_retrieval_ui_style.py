"""知识检索复用 AIOps 视觉基础的静态合同，不连接真实服务。"""

import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "ui" / "knowledge_retrieval"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []
        self.styles = []
        self.body = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "body":
            self.body = values
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"])
        if tag == "link" and values.get("rel") == "stylesheet":
            self.styles.append(values["href"])
            self.assets.append(values["href"])


class KnowledgeRetrievalUiStyleTest(unittest.TestCase):
    def test_all_pages_load_the_same_aiops_design_source_before_business_styles(self):
        pages = list(UI.glob("*.html"))
        self.assertEqual(len(pages), 8)
        for page in pages:
            with self.subTest(page=page.name):
                parser = PageParser()
                parser.feed(page.read_text(encoding="utf-8"))
                self.assertEqual(parser.styles[0], "../aiops/css/ammolite-shared.css?v=20261103_2")
                self.assertEqual(parser.styles[1], "./css/knowledge.css?v=20261105_2")
                self.assertIn("business-app-aiops", parser.body["class"].split())
                self.assertEqual(parser.body["data-page"], page.stem)
                for asset in parser.assets:
                    self.assertTrue((page.parent / asset.split("?", 1)[0]).is_file(), asset)

    def test_shell_assets_are_versioned_and_javascript_is_valid(self):
        for page in UI.glob("*.html"):
            if page.stem != "login":
                self.assertIn('src="./js/knowledge-shell.js?v=20261105_2"', page.read_text(encoding="utf-8"))
        for script in (UI / "js").glob("*.js"):
            result = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_knowledge_styles_reference_shared_tokens_instead_of_old_green_theme(self):
        css = (UI / "css" / "knowledge.css").read_text(encoding="utf-8")
        for token in ("--knowledge-accent: var(--accent)", "--knowledge-bg: var(--canvas)", "--knowledge-radius: var(--card-radius)"):
            self.assertIn(token, css)
        for obsolete in ("#20312d", "#176b52", ".knowledge-sidebar", ".knowledge-topbar"):
            self.assertNotIn(obsolete, css)
        self.assertIn("[hidden] { display: none !important; }", css)
        self.assertIn("@container knowledge-content", css)

    def test_shell_keeps_knowledge_authentication_and_permission_boundaries(self):
        script = (UI / "js" / "knowledge-shell.js").read_text(encoding="utf-8")
        self.assertIn('KBotKnowledgeApi.json("/access", "GET")', script)
        self.assertIn("session.must_change_password", script)
        self.assertIn("KBotKnowledgeAuth.clear()", script)
        self.assertIn("pages.filter(([id]) => granted.has(PAGE_PERMISSIONS[id]))", script)
        self.assertIn("if (!visible.length) return", script)
        self.assertNotIn("KBotAIOpsAuth", script)
        self.assertNotIn("/apps/aiops", script)

    def test_domain_switcher_uses_authorized_session_domains_and_reload(self):
        script = (UI / "js" / "knowledge-shell.js").read_text(encoding="utf-8")
        self.assertIn('KBotKnowledgeApi.json("/api/v1/auth/domains", "GET")', script)
        self.assertIn('KBotKnowledgeApi.json("/api/v1/auth/switch-domain", "POST"', script)
        self.assertIn("KBotKnowledgeAuth.save({ ...current, ...next })", script)
        self.assertIn("location.reload()", script)
        self.assertIn('aria-controls="knowledge-domain-menu"', script)
        self.assertNotIn("knowledge-current-domain", script)
        domains = (UI / "js" / "knowledge-domains.js").read_text(encoding="utf-8")
        self.assertIn("KBotKnowledgeShell.switchDomain(row.domain_id)", domains)


if __name__ == "__main__":
    unittest.main()
