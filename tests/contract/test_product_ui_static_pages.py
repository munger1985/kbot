"""知识检索与多媒体创作静态页契约。"""

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
APP_CONFIGS = {
    "knowledge_retrieval": {
        "root": ROOT / "ui" / "knowledge_retrieval",
        "prefix": "knowledge",
        "global": "KBotKnowledge",
        "pages": {
            "login.html": {"login-form", "password-form"},
            "dashboard.html": set(),
            "knowledge.html": {"knowledge-form", "knowledge-agent", "knowledge-chat-stream"},
            "x-search.html": {"x-search-form", "x-search-agent", "x-search-result", "x-search-sources"},
            "domains.html": {"domain-dialog", "domain-rows", "domain-form"},
            "knowledge-cores.html": {"kc-dialog", "kc-rows", "kc-form"},
            "data-models.html": {"data-source-dialog", "data-source-list", "data-model-dialog"},
            "agents.html": {"agent-dialog", "agent-form", "agent-x-search-llm"},
        },
    },
    "media_studio": {
        "root": ROOT / "ui" / "media_studio",
        "prefix": "media",
        "global": "KBotMedia",
        "pages": {
            "login.html": {"login-form", "password-form"},
            "dashboard.html": set(),
            "image-generation.html": {"image-form", "image-history", "image-result"},
            "model-bindings.html": {"binding-rows", "model-binding-dialog"},
            "media-assets.html": {"asset-rows", "asset-preview-dialog"},
            "usage-runs.html": {"run-rows"},
        },
    },
}


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


class ProductUiStaticPagesTest(unittest.TestCase):
    def test_pages_and_referenced_assets_exist(self):
        for app_name, config in APP_CONFIGS.items():
            root = config["root"]
            for page_name, expected_ids in config["pages"].items():
                page = root / page_name
                parser = _parse(page)
                self.assertTrue(expected_ids.issubset(parser.ids), f"{app_name}/{page_name}")
                for reference in [*parser.scripts, *parser.links]:
                    if reference.startswith("data:"):
                        continue
                    asset_path = reference.partition("?")[0]
                    self.assertTrue(
                        (page.parent / asset_path).resolve().is_file(),
                        f"{app_name}/{page_name} 缺少资源 {reference}",
                    )

    def test_all_javascript_has_valid_syntax(self):
        for config in APP_CONFIGS.values():
            scripts = sorted((config["root"] / "js").glob("*.js"))
            self.assertGreaterEqual(len(scripts), 4)
            for script in scripts:
                result = subprocess.run(
                    ["node", "--check", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(0, result.returncode, f"{script.name}: {result.stderr}")

    def test_public_ui_does_not_call_internal_routes_or_hardcode_model(self):
        for config in APP_CONFIGS.values():
            root = config["root"]
            sources = [*root.glob("*.html"), *(root / "js").glob("*.js"), *(root / "css").glob("*.css")]
            blob = "\n".join(path.read_text(encoding="utf-8") for path in sources)
            self.assertNotIn("/internal/v1", blob)
            self.assertNotIn("grok-4.6", blob)

    def test_knowledge_retrieval_owns_agent_and_x_search(self):
        root = APP_CONFIGS["knowledge_retrieval"]["root"]
        agents = (root / "js" / "knowledge-agents.js").read_text(encoding="utf-8")
        api = (root / "js" / "knowledge-api.js").read_text(encoding="utf-8")
        search = (root / "js" / "knowledge-x-search.js").read_text(encoding="utf-8")
        self.assertIn("x_search_llm", agents)
        self.assertIn("isGrokModel", api)
        self.assertIn("provider_model_name", api)
        self.assertIn('json("/agents", "GET")', search)
        self.assertIn("Idempotency-Key", search)
        self.assertIn("knowledge-history-delete", search)

    def test_media_studio_only_exposes_media_workflows(self):
        root = APP_CONFIGS["media_studio"]["root"]
        blob = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.*"))
        self.assertIn("/image-generations/runs", blob)
        self.assertIn("/media-assets", blob)
        self.assertIn("/bindings", blob)
        self.assertNotIn("/x-search", blob)
        self.assertNotIn("/agents", blob)
        self.assertNotIn("/knowledge-cores", blob)


if __name__ == "__main__":
    unittest.main()
