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
MARKDOWN_CHAIN = (
    "../vendor/marked.umd.js?v=18.0.9",
    "../vendor/purify.min.js?v=3.4.13",
    "../shared/kbot-markdown.js?v=20260827_1",
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
        "knowledge.html": {
            "knowledge-form", "knowledge-agent", "knowledge-agent-summary",
            "knowledge-context-agent", "knowledge-context-core",
            "knowledge-context-models",
        },
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
        "domains.html": {"domain-dialog", "domain-rows", "domain-form"},
        "knowledge-cores.html": {
            "kc-dialog", "kc-rows", "kc-form", "kc-upload-dialog",
            "kc-upload-form", "kc-file-input", "kc-processing-rows",
        },
        "data-models.html": {
            "data-source-dialog",
            "data-source-form",
            "data-source-list",
            "schema-object-rows",
            "confirm-schema-selection",
            "data-model-dialog",
            "semantic-model-dialog",
            "semantic-model-definition",
            "manual-ddl-dialog",
        },
        "agents.html": {
            "agent-refresh",
            "agent-create",
            "agent-count",
            "agent-filter",
            "agent-status-filter",
            "agent-rows",
            "agent-dialog",
            "agent-form",
            "agent-name",
            "agent-status",
            "agent-description",
            "agent-instruction",
            "agent-knowledge-core",
            "agent-data-models",
            "agent-context-llm",
            "agent-composer-llm",
            "agent-memory-llm",
            "agent-memory-embedding",
            "agent-config",
        },
        "model-bindings.html": {"binding-rows", "model-binding-dialog"},
        "media-assets.html": {"asset-rows", "asset-preview-dialog", "asset-preview-body"},
        "usage-runs.html": {"run-rows"},
    }
    page_scripts = {
        "dashboard.html": "./js/assistant-dashboard.js",
        "knowledge.html": "./js/assistant-knowledge.js",
        "x-search.html": "./js/assistant-x-search.js",
        "image-generation.html": "./js/assistant-image-generation.js",
        "domains.html": "./js/assistant-domains.js",
        "knowledge-cores.html": "./js/assistant-knowledge-cores.js",
        "data-models.html": "./js/assistant-data-models.js",
        "agents.html": "./js/assistant-agents.js",
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
            extra = MARKDOWN_CHAIN if page_name == "x-search.html" else ()
            expected = [*SCRIPT_CHAIN, *extra, page_script]
            self.assertEqual(expected, parser.scripts[: len(expected)], page_name)
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
        self.assertIn('"DELETE"', x_search)
        self.assertIn('"DELETE"', image)
        self.assertIn("assistant-history-delete", x_search)
        self.assertIn("assistant-history-delete", image)
        self.assertIn("globalThis.confirm", x_search)
        self.assertIn("globalThis.confirm", image)
        self.assertIn("requestBlob", image)
        self.assertIn("createObjectURL", image)
        self.assertIn("requestBlob", management)
        self.assertIn("createObjectURL", management)
        self.assertIn("asset-preview-dialog", management)
        self.assertIn("showModal", management)
        self.assertNotIn('closest("td")', management)
        self.assertIn("KBotMarkdown.render", x_search)
        self.assertIn("KBotMarkdown.copyCode", x_search)
        self.assertNotIn("parseAnswerBlocks", x_search)
        self.assertNotIn("img.src = ", image)
        self.assertNotIn("content_path", image)

    def test_pages_wait_for_shell_ready(self):
        for name in (
            "assistant-dashboard.js",
            "assistant-knowledge.js",
            "assistant-agents.js",
            "assistant-domains.js",
            "assistant-knowledge-cores.js",
            "assistant-data-models.js",
            "assistant-x-search.js",
            "assistant-image-generation.js",
            "assistant-management.js",
        ):
            source = (UI_ROOT / "js" / name).read_text(encoding="utf-8")
            self.assertIn("KBotAssistantShell.ready", source, name)
            self.assertNotIn("DOMContentLoaded", source, name)

    def test_data_models_page_uses_governed_workflow_routes(self):
        source = (UI_ROOT / "js" / "assistant-data-models.js").read_text(
            encoding="utf-8"
        )
        for fragment in (
            "/data-sources/test-connection",
            "/snapshots",
            "/selection",
            "/semantic-model-draft",
            "/validations",
            "/submit-review",
            "/publish",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("/policy-bindings", source)
        self.assertNotIn("/agent-bindings", source)
        self.assertNotIn("domain_id", source)
        self.assertNotIn("/internal/v1", source)

    def test_agents_page_uses_lifecycle_and_safe_option_routes(self):
        source = (UI_ROOT / "js" / "assistant-agents.js").read_text(
            encoding="utf-8"
        )
        for fragment in (
            'json("/agents", "GET")',
            'json("/agents/options", "GET")',
            '"POST"',
            '"PATCH"',
            '"DELETE"',
            "expected_row_version",
            "context_llm",
            "composer_llm",
            "memory_llm",
            "memory_embedding",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("agent_bindings", source)
        self.assertNotIn("configure-query", source)
        self.assertIn("globalThis.confirm", source)
        self.assertNotIn("domain_id", source)
        self.assertNotIn("/internal/v1", source)

    def test_knowledge_core_catalog_uses_contract_helpers(self):
        api = (UI_ROOT / "js" / "assistant-api.js").read_text(encoding="utf-8")
        kc = (UI_ROOT / "js" / "assistant-knowledge-cores.js").read_text(encoding="utf-8")
        management = (UI_ROOT / "js" / "assistant-management.js").read_text(encoding="utf-8")
        self.assertIn("function items(", api)
        self.assertIn("function modelCategory(", api)
        self.assertIn("function isActiveModel(", api)
        self.assertIn("TXT_EMBEDDING", api)
        self.assertIn("IMG_EMBEDDING", api)
        self.assertIn("KBotAssistantApi.items(", kc)
        self.assertIn("KBotAssistantApi.isActiveModel(", kc)
        self.assertIn("KBotAssistantApi.modelCategory(", kc)
        self.assertIn("ModelCategory.TXT_EMBEDDING", kc)
        self.assertIn("ModelCategory.IMG_EMBEDDING", kc)
        self.assertIn("/processing", kc)
        self.assertIn("/ingestions/user-files", kc)
        self.assertIn("/reprocess", kc)
        self.assertIn("data-reprocess-file", kc)
        self.assertIn("Idempotency-Key", kc)
        self.assertNotIn("Number(row.category) === 2", kc)
        self.assertNotIn("Number(item.category) === 2", kc)
        self.assertIn("KBotAssistantApi.items(", management)
        self.assertIn("KBotAssistantApi.isActiveModel(", management)

    def test_catalog_helpers_unwrap_and_fill_embedding_categories(self):
        script = UI_ROOT / "js" / "assistant-api.js"
        result = subprocess.run(
            [
                "node",
                "-e",
                """
const fs = require("fs");
const vm = require("vm");
const sandbox = { console };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const api = sandbox.KBotAssistantApi;
const payload = { items: [
  { model_id: "text-1", category: "TXT_EMBEDDING", status: 1 },
  { model_id: "text-2", category: 2, status: "ENABLED" },
  { model_id: "visual-1", category: "VISUAL_EMBEDDING", status: "ACTIVE" },
  { model_id: "draft", category: 2, status: 0 },
]};
const rows = api.items(payload).filter(api.isActiveModel);
if (rows.length !== 3) process.exit(2);
if (rows.filter((row) => api.modelCategory(row) === api.ModelCategory.TXT_EMBEDDING).length !== 2) process.exit(3);
if (rows.filter((row) => api.modelCategory(row) === api.ModelCategory.IMG_EMBEDDING).length !== 1) process.exit(4);
""",
                str(script),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr or result.stdout)

    def test_shell_renders_chrome_before_access_then_removes_by_permission(self):
        source = (UI_ROOT / "js" / "assistant-shell.js").read_text(encoding="utf-8")
        self.assertIn('insertAdjacentHTML("afterbegin"', source)
        self.assertIn("data-permission", source)
        self.assertIn("function pruneNavigation", source)
        self.assertIn("pruneNavigation(permissions);", source)
        self.assertIn('json("/access"', source)
        self.assertLess(source.index('insertAdjacentHTML("afterbegin"'), source.index('json("/access"'))
        self.assertGreater(source.index("pruneNavigation(permissions);"), source.index('json("/access"'))
        self.assertNotIn("permissions.has(PAGE_PERMISSIONS[id])", source)
        self.assertNotIn("cursor: not-allowed", source)


if __name__ == "__main__":
    unittest.main()
