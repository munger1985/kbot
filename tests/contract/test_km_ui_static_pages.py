"""KM Asset 正式页面静态契约检查。"""

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = ROOT / "ui"
KM_ROOT = UI_ROOT / "km"


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.assets: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"])
        if tag == "link" and values.get("href"):
            self.assets.append(values["href"])


class KmUiStaticPagesTest(unittest.TestCase):
    pages = {
        "dashboard.html": {"metric-sources", "metric-ready", "metric-failed", "dashboard-job-rows"},
        "metadb.html": {"metadb-form", "metadb-source", "metadb-rows", "metadb-detail-dialog"},
        "sources.html": {"source-form", "source-edit-form", "source-rows", "data-model-dialog", "collection-page-link"},
        "assets.html": {"asset-form", "asset-rows", "asset-detail-dialog"},
        "jobs.html": {"job-form", "job-rows", "job-detail-dialog"},
        "agents.html": {"agent-form", "agent-rows", "agent-source"},
        "chat.html": {"chat-agent", "conversation-list", "chat-stream", "chat-form", "reference-dialog"},
    }

    def test_pages_reference_existing_assets_and_controls(self):
        for page_name, expected_ids in self.pages.items():
            page = KM_ROOT / page_name
            parser = _PageParser()
            parser.feed(page.read_text(encoding="utf-8"))
            self.assertTrue(expected_ids.issubset(parser.ids), page_name)
            for reference in parser.assets:
                self.assertTrue(
                    (page.parent / reference).resolve().is_file(),
                    f"{page_name} 缺少资源 {reference}",
                )

    def test_javascript_syntax(self):
        scripts = [
            *(UI_ROOT / "shared").glob("*.js"),
            *(KM_ROOT / "js").glob("*.js"),
        ]
        self.assertGreaterEqual(len(scripts), 9)
        for script in scripts:
            result = subprocess.run(
                ["node", "--check", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_public_client_does_not_store_trusted_identity(self):
        source = (UI_ROOT / "shared" / "kbot-api-client.js").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "X-KBot-Test-Auth",
            "X-KBot-Domain-ID",
            "X-KBot-User-ID",
            "localStorage",
            "sessionStorage",
            "apiKey",
        ):
            self.assertNotIn(forbidden, source)

    def test_pages_only_use_km_public_bff_contract(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (KM_ROOT / "js").glob("*.js")
        )
        self.assertIn('/api/v1/apps/km-asset', source)
        self.assertIn('/api/v1/model-catalog', source)
        self.assertIn('"PATCH"', source)
        self.assertNotIn('/internal/v1', source)
        for path in (
            "/metadb/assets/",
            "/data-model/reconcile",
            "/assets/",
            "/agents/",
            "/conversations/",
            "/references/",
        ):
            self.assertIn(path, source)

    def test_default_user_script_is_sql_developer_standalone(self):
        source = (
            ROOT / "scripts" / "db" / "bootstrap_km_default_user.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("DEFINE KM_DEFAULT_USER_ID", source)
        self.assertIn("KBOT_PLATFORM_USER", source)
        self.assertIn("KBOT_APP_MEMBER_ROLE", source)
        self.assertIn("'km_asset' AS APP_ID", source)
        self.assertNotIn("@@", source)


if __name__ == "__main__":
    unittest.main()
