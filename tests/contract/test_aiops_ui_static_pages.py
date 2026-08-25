"""AIOps 正式页面静态契约检查。"""

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
AIOPS_ROOT = ROOT / "ui" / "aiops"


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"])
        if tag == "link" and values.get("href"):
            self.assets.append(values["href"])


class AIOpsUiStaticPagesTest(unittest.TestCase):
    pages = {
        "dashboard", "chat", "situations", "runs", "run-detail",
        "reports", "report-detail", "inspections", "changes",
        "notifications", "targets", "target-detail",
        "diagnostic-sources", "diagnostic-source-detail", "knowledge-core",
        "agents",
        "policies", "inspection-plans", "inspection-plan-detail",
        "report-templates", "notification-subscriptions", "api-clients",
        "login",
    }

    def test_exact_page_inventory_and_assets(self):
        actual = {path.stem for path in AIOPS_ROOT.glob("*.html")}
        self.assertEqual(self.pages, actual)
        for page in AIOPS_ROOT.glob("*.html"):
            parser = _Parser()
            parser.feed(page.read_text(encoding="utf-8"))
            for reference in parser.assets:
                asset = reference.partition("?")[0]
                self.assertTrue(
                    (page.parent / asset).resolve().is_file(),
                    f"{page.name} 缺少资源 {reference}",
                )

    def test_javascript_syntax_and_public_boundary(self):
        scripts = list((AIOPS_ROOT / "js").glob("*.js"))
        self.assertEqual(4, len(scripts))
        source = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
        self.assertIn("/api/v1/apps/aiops", source)
        self.assertNotIn("/internal/v1", source)
        for script in scripts:
            result = subprocess.run(
                ["node", "--check", str(script)],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_pages_do_not_embed_demo_records_or_api_keys(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in AIOPS_ROOT.rglob("*") if path.is_file()
        )
        self.assertNotIn("140.238.44.208", source)
        self.assertNotIn("kbot_ak_", source)
        self.assertNotIn("/metrics", source)

    def test_knowledge_core_page_uses_fixed_public_bff(self):
        page = (AIOPS_ROOT / "knowledge-core.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-knowledge-core.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="model-form"', page)
        self.assertIn('id="manual-form"', page)
        self.assertIn("/knowledge-core/models", script)
        self.assertIn("/knowledge-core/manuals", script)
        self.assertIn("embedding_change_allowed", script)
        self.assertIn("visual_embedding_change_allowed", script)
        self.assertNotIn("/api/v1/knowledge", script)

    def test_login_uses_fixed_aiops_domain_contract(self):
        login = (AIOPS_ROOT / "login.html").read_text(encoding="utf-8")
        auth = (AIOPS_ROOT / "js" / "aiops-auth.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("aiops_portal", login)
        self.assertNotIn('name="domain_id"', login)
        self.assertIn("/api/v1/apps/aiops/auth/login", auth)


if __name__ == "__main__":
    unittest.main()
