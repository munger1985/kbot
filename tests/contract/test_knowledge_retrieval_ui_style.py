"""知识检索与 Ammolite 生产页面的静态映射和授权合同。"""

import json
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
    def test_all_pages_declare_reference_and_load_generated_knowledge_styles(self):
        pages = list(UI.glob("*.html"))
        manifest = json.loads((UI / "reference-pages.json").read_text(encoding="utf-8"))
        self.assertEqual({page.stem for page in pages}, set(manifest["pages"]))
        for page in pages:
            with self.subTest(page=page.name):
                parser = PageParser()
                parser.feed(page.read_text(encoding="utf-8"))
                self.assertEqual(parser.styles[0], "./css/knowledge.css?v=20261105_3")
                self.assertEqual(parser.styles[1], "./css/ammolite-shared.css?v=20261105_3")
                self.assertIn("business-app-knowledge_retrieval", parser.body["class"].split())
                self.assertTrue(manifest["pages"][page.stem]["sources"] or manifest["pages"][page.stem].get("platform_difference"))
                self.assertEqual(parser.body["data-page"], page.stem)
                for asset in parser.assets:
                    self.assertTrue((page.parent / asset.split("?", 1)[0]).is_file(), asset)

    def test_shell_assets_are_versioned_and_javascript_is_valid(self):
        for page in UI.glob("*.html"):
            if page.stem != "login":
                self.assertIn('src="./js/knowledge-shell.js?v=20261105_3"', page.read_text(encoding="utf-8"))
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

    def test_daily_operations_are_separate_from_configuration(self):
        collections = (UI / "knowledge-cores.html").read_text(encoding="utf-8")
        self.assertNotIn('id="kc-approval-rows"', collections)
        self.assertNotIn('id="kc-processing-rows"', collections)
        for name in ("upload", "reviews", "processing", "files"):
            self.assertIn('id="library-collection"', (UI / f"{name}.html").read_text(encoding="utf-8"))
        library = (UI / "js/knowledge-library.js").read_text(encoding="utf-8")
        self.assertIn('comment', library)
        self.assertIn('SINGLE_BUNDLE', library)
        self.assertIn('page_size:state.size', library)
        self.assertIn('request !== state.request', library)

    def test_generated_styles_are_current(self):
        result = subprocess.run(["node", "tests/contract/sync_knowledge_reference_styles.mjs"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
