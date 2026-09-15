"""智能工作台静态页契约：脚本链、鉴权、blob 预览，且不暴露内部路由。"""

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = ROOT / "ui" / "assistant"
SCRIPT_CHAIN = (
    "../runtime-config.js",
    "./js/assistant-auth.js",
    "./js/assistant-api.js",
    "./js/assistant-shell.js",
)


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


def _parse(page: Path) -> _PageParser:
    parser = _PageParser()
    parser.feed(page.read_text(encoding="utf-8"))
    return parser


class AssistantUiStaticPagesTest(unittest.TestCase):
    pages = {
        "login.html": {"login-form", "password-form"},
        "dashboard.html": {"dashboard-setup", "dashboard-runs"},
        "knowledge.html": {"knowledge-form"},
        "x-search.html": {
            "x-search-history",
            "x-search-result",
            "x-search-sources",
            "x-search-capability",
            "x-search-form",
            "x-search-image",
            "x-search-video",
            "x-search-handle-mode",
        },
        "image-generation.html": {
            "image-history",
            "image-result",
            "image-preview",
            "image-meta",
            "image-capability",
            "image-form",
        },
        "domains.html": {"domain-dialog"},
        "knowledge-cores.html": {"kc-dialog"},
        "data-models.html": {"data-model-dialog"},
        "agents.html": {"agent-dialog"},
        "model-bindings.html": {"binding-rows", "model-binding-dialog"},
        "media-assets.html": {"asset-rows"},
        "usage-runs.html": {"run-rows"},
    }
    page_scripts = {
        "dashboard.html": "./js/assistant-dashboard.js",
        "knowledge.html": "./js/assistant-knowledge.js",
        "x-search.html": "./js/assistant-x-search.js",
        "image-generation.html": "./js/assistant-image-generation.js",
        "domains.html": "./js/assistant-resources.js",
        "knowledge-cores.html": "./js/assistant-resources.js",
        "data-models.html": "./js/assistant-resources.js",
        "agents.html": "./js/assistant-resources.js",
        "model-bindings.html": "./js/assistant-management.js",
        "media-assets.html": "./js/assistant-management.js",
        "usage-runs.html": "./js/assistant-management.js",
    }

    def test_pages_exist_with_required_controls_and_assets(self):
        self.assertTrue({"login.html", *self.page_scripts}.issubset(self.pages))
        for page_name, expected_ids in self.pages.items():
            page = UI_ROOT / page_name
            parser = _parse(page)
            self.assertTrue(expected_ids.issubset(parser.ids), page_name)
            for reference in [*parser.scripts, *parser.links]:
                if reference.startswith("data:"):
                    continue
                asset_path = reference.partition("?")[0]
                self.assertTrue(
                    (page.parent / asset_path).resolve().is_file(),
                    f"{page_name} 缺少资源 {reference}",
                )

    def test_business_pages_use_auth_api_shell_script_chain(self):
        for page_name, page_script in self.page_scripts.items():
            parser = _parse(UI_ROOT / page_name)
            expected = [*SCRIPT_CHAIN, page_script]
            self.assertEqual(expected, parser.scripts[:5], page_name)
            self.assertNotIn("./js/assistant-resources.js", parser.scripts) if page_name == "model-bindings.html" else None
            if page_name == "model-bindings.html":
                self.assertNotIn("./js/assistant-resources.js", parser.scripts)

    def test_login_page_only_loads_auth_client(self):
        parser = _parse(UI_ROOT / "login.html")
        self.assertEqual(["../runtime-config.js", "./js/assistant-auth.js"], parser.scripts)
        html = (UI_ROOT / "login.html").read_text(encoding="utf-8")
        self.assertIn("assistant_portal", html)
        self.assertIn("assistantadmin", html)
        self.assertIn("assistant-login", html)

    def test_javascript_syntax(self):
        scripts = sorted((UI_ROOT / "js").glob("*.js"))
        self.assertGreaterEqual(len(scripts), 8)
        for script in scripts:
            result = subprocess.run(
                ["node", "--check", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, f"{script.name}: {result.stderr}")

    def test_ui_does_not_call_internal_routes_or_hardcode_model(self):
        sources = [
            *(UI_ROOT.glob("*.html")),
            *( (UI_ROOT / "js").glob("*.js") ),
            UI_ROOT / "css" / "assistant.css",
        ]
        blob = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        self.assertNotIn("/internal/v1", blob)
        self.assertNotIn("grok-4.6", blob)
        self.assertNotIn("EventSource", blob)

    def test_api_client_sends_authorization_and_supports_blob_preview(self):
        api = (UI_ROOT / "js" / "assistant-api.js").read_text(encoding="utf-8")
        image = (UI_ROOT / "js" / "assistant-image-generation.js").read_text(encoding="utf-8")
        management = (UI_ROOT / "js" / "assistant-management.js").read_text(encoding="utf-8")
        x_search = (UI_ROOT / "js" / "assistant-x-search.js").read_text(encoding="utf-8")
        self.assertIn("Authorization", api)
        self.assertIn("Bearer", api)
        self.assertIn("requestBlob", api)
        self.assertIn("Idempotency-Key", x_search)
        self.assertIn("Idempotency-Key", image)
        self.assertIn("requestBlob", image)
        self.assertIn("createObjectURL", image)
        self.assertIn("requestBlob", management)
        self.assertIn("createObjectURL", management)
        self.assertNotIn("img.src = ", image)
        self.assertNotIn("content_path", image)

    def test_pages_wait_for_shell_ready(self):
        for name in (
            "assistant-dashboard.js",
            "assistant-knowledge.js",
            "assistant-resources.js",
            "assistant-x-search.js",
            "assistant-image-generation.js",
            "assistant-management.js",
        ):
            source = (UI_ROOT / "js" / name).read_text(encoding="utf-8")
            self.assertIn("KBotAssistantShell.ready", source, name)
            self.assertNotIn("DOMContentLoaded", source, name)


if __name__ == "__main__":
    unittest.main()
