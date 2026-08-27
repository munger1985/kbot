"""KM Asset 正式页面静态契约检查。"""

from html.parser import HTMLParser
import json
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
        "knowledge-core.html": {
            "knowledge-core-form",
            "knowledge-core-summary",
            "knowledge-core-model-rows",
            "create-knowledge-core",
        },
        "metadb.html": {"metadb-form", "metadb-source", "metadb-rows", "metadb-detail-dialog"},
        "sources.html": {"source-form", "source-edit-form", "source-rows", "data-model-dialog"},
        "assets.html": {
            "asset-form",
            "asset-rows",
            "asset-select-all",
            "asset-bulk-reindex",
            "asset-detail-dialog",
        },
        "jobs.html": {
            "job-form",
            "job-tree",
            "source-job-tree",
            "job-status-text",
            "job-detail-dialog",
        },
        "agents.html": {"agent-form", "agent-rows", "agent-source"},
        "chat.html": {"chat-agent", "conversation-list", "chat-stream", "chat-form", "reference-dialog"},
        "api-clients.html": {
            "api-client-form",
            "api-client-rows",
            "api-client-subject",
            "api-client-agents",
            "api-key-dialog",
        },
    }

    def test_pages_reference_existing_assets_and_controls(self):
        for page_name, expected_ids in self.pages.items():
            page = KM_ROOT / page_name
            parser = _PageParser()
            parser.feed(page.read_text(encoding="utf-8"))
            self.assertTrue(expected_ids.issubset(parser.ids), page_name)
            for reference in parser.assets:
                if reference.startswith("data:"):
                    continue
                asset_path = reference.partition("?")[0]
                self.assertTrue(
                    (page.parent / asset_path).resolve().is_file(),
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

    def test_asset_page_allows_discovery_publication_recovery(self):
        source = (KM_ROOT / "js" / "km-assets.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('row.ingestion_status === "KC_ACCEPTED"', source)
        self.assertIn('row.failure_stage === "KC_STATUS_SYNC"', source)
        self.assertIn('function requiresIndexRecovery(row)', source)
        self.assertIn('function reindexLabel(row)', source)
        self.assertIn('"恢复索引"', source)

    def test_agent_page_has_no_rerank_product_control(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                KM_ROOT / "agents.html",
                KM_ROOT / "js" / "km-agents-v3.js",
            )
        ).lower()
        self.assertNotIn("do_rerank", source)
        self.assertNotIn("reranker", source)
        self.assertNotIn("重排", source)

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
        self.assertIn('/model-catalog', source)
        self.assertNotIn('/api/v1/apps/knowledge-retrieval', source)
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

    def test_km_chat_retries_turn_confirmation_with_same_idempotency_key(self):
        source = (
            ROOT / "ui" / "km" / "js" / "km-chat-v6.js"
        ).read_text(encoding="utf-8")
        self.assertIn("createTurn(input, idempotencyKey)", source)
        self.assertIn('"Idempotency-Key": idempotencyKey', source)
        self.assertIn("isRetryableTurnTransportError", source)
        self.assertIn("正在确认已提交的 Turn", source)
        self.assertIn('run.status !== "COMPLETED"', source)
        self.assertIn("run.error_message", source)

    def test_km_chat_exposes_existing_conversation_delete_contract(self):
        source = (
            ROOT / "ui" / "km" / "js" / "km-chat-v6.js"
        ).read_text(encoding="utf-8")
        self.assertIn("deleteConversation", source)
        self.assertIn("expected_row_version=${row.row_version}", source)
        self.assertIn('method: "DELETE"', source)
        self.assertIn("会话内容删除后无法恢复", source)

    def test_km_chat_streams_answer_deltas_as_accumulated_markdown(self):
        source = (
            ROOT / "ui" / "km" / "js" / "km-chat-v6.js"
        ).read_text(encoding="utf-8")
        html = (ROOT / "ui" / "km" / "chat.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('eventType === "answer.delta"', source)
        self.assertIn("enqueueAnswerDelta(pending, payload.delta)", source)
        self.assertIn("typingUnits", source)
        self.assertIn("typingBatchSize", source)
        self.assertIn("pending.displayedMarkdown", source)
        self.assertIn("waitForTypewriter(pending)", source)
        self.assertIn('eventType === "answer.completed"', source)
        self.assertIn(
            "renderReferences(pending.message.dataset.runId, payload.references)",
            source,
        )
        self.assertIn(
            "turn.assistant_item?.content?.references",
            source,
        )
        self.assertNotIn("Promise.allSettled(runIds", source)
        self.assertIn("updateActiveConversation()", source)
        self.assertIn('message.classList.add("is-typing")', source)
        self.assertIn("onEvent: (item) => applyRunEvent(pending, item)", source)
        self.assertIn('content.setAttribute("aria-live", "polite")', source)
        self.assertIn("km-chat-v6.js?v=20260827_1", html)

    def test_km_chat_uses_interactive_paper_style_citations(self):
        source = (
            ROOT / "ui" / "km" / "js" / "km-chat-v6.js"
        ).read_text(encoding="utf-8")
        styles = (
            ROOT / "ui" / "km" / "css" / "km-chat-markdown.css"
        ).read_text(encoding="utf-8")
        html = (ROOT / "ui" / "km" / "chat.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("renderAssistantMarkdown", source)
        self.assertIn("document.createTreeWalker", source)
        self.assertIn('marker.className = "km-citation-marker"', source)
        self.assertIn("marker.dataset.citationLabel", source)
        self.assertIn("prepareCitationMarker", source)
        self.assertNotIn("showQueryReference", source)
        self.assertIn('const citationPattern = /\\[(C\\d+)\\]/g;', source)
        self.assertIn("prepareAssetReference", source)
        self.assertIn("appendAssetAttachments", source)
        self.assertIn("data-attachment-url", source)
        self.assertIn('closest("code, pre, a, sup")', source)
        self.assertIn(".km-citation-marker", styles)
        self.assertIn(".km-reference-asset-preview", styles)
        self.assertIn("vertical-align: super", styles)
        self.assertIn('id="reference-asset-preview"', html)
        self.assertNotIn('id="reference-query-preview"', html)
        self.assertNotIn('id="open-reference"', html)

    def test_jobs_page_uses_asset_revision_step_tree(self):
        source = (KM_ROOT / "js" / "km-jobs.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("loadAllAssets", source)
        self.assertIn("asset_revision_id", source)
        self.assertIn("bundle_revision_id", source)
        self.assertIn("/jobs/processing", source)
        self.assertIn("data-asset", source)
        self.assertIn("data-chain", source)
        self.assertIn("data-source-jobs", source)
        self.assertIn("data-source-more", source)
        self.assertIn("SOURCE_JOB_PAGE_SIZE = 10", source)
        self.assertNotIn("job-rows", source)

    def test_km_chat_does_not_supply_user_security_level(self):
        source = (
            ROOT / "ui" / "km" / "js" / "km-chat-v6.js"
        ).read_text(encoding="utf-8")
        self.assertNotIn("security_level:", source)

    def test_km_chat_uses_local_safe_markdown_renderer(self):
        renderer = ROOT / "ui" / "shared" / "kbot-markdown.js"
        source = renderer.read_text(encoding="utf-8")
        chat = (
            ROOT / "ui" / "km" / "js" / "km-chat-v6.js"
        ).read_text(encoding="utf-8")
        self.assertIn("KBotMarkdown.render(value)", chat)
        self.assertIn("escapeHtml(source)", source)
        self.assertIn('target="_blank" rel="noopener noreferrer"', source)
        self.assertNotIn("innerHTML = value", source)

    def test_km_markdown_uses_ammolite_gfm_runtime(self):
        source = (
            ROOT / "ui" / "shared" / "kbot-markdown.js"
        ).read_text(encoding="utf-8")
        chat = (ROOT / "ui" / "km" / "chat.html").read_text(
            encoding="utf-8"
        )
        marked = (ROOT / "ui" / "vendor" / "marked.umd.js").read_text(
            encoding="utf-8"
        )
        purifier = (
            ROOT / "ui" / "vendor" / "purify.min.js"
        ).read_text(encoding="utf-8")

        self.assertIn("new markedApi.Marked", source)
        self.assertIn("gfm: true", source)
        self.assertIn("breaks: true", source)
        self.assertIn("purifier.sanitize", source)
        self.assertIn("FORBID_TAGS", source)
        self.assertIn("marked v18.0.9", marked)
        self.assertIn("DOMPurify 3.4.13", purifier)
        self.assertIn("marked.umd.js?v=18.0.9", chat)
        self.assertIn("purify.min.js?v=3.4.13", chat)
        self.assertIn("kbot-markdown.js?v=20260819_3", chat)

    def test_km_markdown_renders_safe_responsive_tables(self):
        renderer = (
            ROOT / "ui" / "shared" / "kbot-markdown.js"
        ).read_text(encoding="utf-8")
        styles = (
            ROOT / "ui" / "km" / "css" / "km-chat-markdown.css"
        ).read_text(encoding="utf-8")
        chat = (ROOT / "ui" / "km" / "chat.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("gfm: true", renderer)
        self.assertIn("purifier.sanitize", renderer)
        self.assertIn("overflow-x: auto", styles)
        self.assertIn("border-collapse: separate", styles)
        self.assertIn("km-chat-markdown.css?v=20260824_1", chat)

    def test_sources_page_has_no_apex_collection_shortcut(self):
        html = (KM_ROOT / "sources.html").read_text(encoding="utf-8")
        script = (KM_ROOT / "js" / "km-sources.js").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("APEX Collection", html)
        self.assertNotIn("collection-page-link", html)
        self.assertNotIn("collectionPageUrl", script)

    def test_existing_schema_permission_script_is_idempotent_and_standalone(self):
        source = (
            ROOT / "scripts" / "db" / "bootstrap_km_asset_permissions.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("MERGE INTO KBOT_PERMISSION", source)
        self.assertIn("MERGE INTO KBOT_APP_ROLE", source)
        self.assertIn("MERGE INTO KBOT_APP_ROLE_PERMISSION", source)
        self.assertIn("MERGE INTO KBOT_APP_MEMBER_ROLE", source)
        self.assertIn("km_asset:operations_manage", source)
        self.assertIn("km_asset:knowledge_manage", source)
        self.assertIn("km_asset:api_key_manage", source)
        self.assertIn("member.IS_INITIAL_ADMIN = 'Y'", source)
        self.assertIn("legacy_role.ROLE_CODE = 'manager'", source)
        self.assertIn("target.SCOPE_MODE = 'ALL_APP_DOMAINS'", source)
        self.assertIn("KM Asset App 管理员最终授权", source)
        self.assertIn("member_role.ROLE_CODE = 'app_admin'", source)
        self.assertNotIn("@@", source)

    def test_existing_schema_auto_sync_upgrade_defaults_to_off(self):
        source = (
            ROOT / "scripts" / "db" / "upgrade_km_source_auto_sync.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("AUTO_SYNC_ENABLED NUMBER(1) DEFAULT 0 NOT NULL", source)
        self.assertIn("CK_KM_SOURCE_AUTO_SYNC", source)
        self.assertNotIn("&&", source)
        self.assertNotIn("@@", source)

    def test_api_client_key_copy_supports_insecure_http_context(self):
        html = (KM_ROOT / "api-clients.html").read_text(encoding="utf-8")
        script = (KM_ROOT / "js" / "km-api-clients.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("km-api-clients.js?v=20260825_1", html)
        self.assertIn("navigator.clipboard?.writeText", script)
        self.assertIn("window.isSecureContext", script)
        self.assertIn('document.execCommand("copy")', script)
        self.assertIn("自动复制失败，请手动选择并复制密钥", script)

    def test_api_client_can_choose_a_non_expiring_key(self):
        html = (KM_ROOT / "api-clients.html").read_text(encoding="utf-8")
        script = (KM_ROOT / "js" / "km-api-clients.js").read_text(
            encoding="utf-8"
        )
        self.assertEqual(2, html.count('name="never_expires"'))
        self.assertIn("永不过期", html)
        self.assertIn("value.setFullYear(value.getFullYear() + 100)", script)
        self.assertIn(
            "if (form.elements.never_expires.checked) return longTermExpiry()",
            script,
        )

    def test_initial_admin_script_bootstraps_full_km_access(self):
        source = (
            ROOT / "scripts" / "db" / "bootstrap_km_initial_admin.sql"
        ).read_text(encoding="utf-8")
        for statement in (
            "MERGE INTO KBOT_PLATFORM_DOMAIN",
            "INSERT INTO KBOT_KC_COLLECTION",
            "MERGE INTO KBOT_PERMISSION",
            "MERGE INTO KBOT_APP_ROLE",
            "MERGE INTO KBOT_APP_ROLE_PERMISSION",
            "MERGE INTO KBOT_PLATFORM_USER",
            "MERGE INTO KBOT_APP_MEMBER",
            "MERGE INTO KBOT_APP_MEMBER_ROLE",
        ):
            self.assertIn(statement, source)
        self.assertIn("'app_admin' AS ROLE_CODE", source)
        self.assertIn("km_asset:api_key_manage", source)
        self.assertIn("km_asset:knowledge_manage", source)
        self.assertIn("'ALL_APP_DOMAINS'", source)
        self.assertIn("'APP_INITIAL_ADMIN'", source)
        self.assertIn("WHERE domain.NAME = 'km_portal'", source)
        self.assertIn("DISPLAY_NAME = 'assets'", source)
        self.assertNotIn("CATEGORY = 1", source)
        self.assertIn("CATEGORY = 2", source)
        self.assertIn("REMOVE '$.parser_llm'", source)
        self.assertIn("'kmadmin' AS USER_ID", source)
        self.assertIn("'KM Asset 管理员' AS DISPLAY_NAME", source)
        self.assertIn("target.MUST_CHANGE_PASSWORD = 'N'", source)
        self.assertIn("source.USER_ID, source.PASSWORD_HASH, 'N'", source)
        self.assertNotIn("VARIABLE KM_ADMIN", source)
        self.assertNotIn(":KM_ADMIN", source)
        self.assertNotIn("&&", source)
        self.assertNotIn("@@", source)

    def test_km_pages_use_login_token_flow(self):
        login_html = (ROOT / "ui" / "km" / "login.html").read_text(
            encoding="utf-8"
        )
        login_js = (ROOT / "ui" / "km" / "js" / "km-login.js").read_text(
            encoding="utf-8"
        )
        adapter = (
            ROOT / "ui" / "shared" / "kbot-km-auth-v2.js"
        ).read_text(encoding="utf-8")
        self.assertIn("km-login-form", login_html)
        self.assertNotIn("Main API 地址", login_html)
        self.assertNotIn('name="baseUrl"', login_html)
        self.assertIn("KBotKmAuth.login", login_js)
        self.assertNotIn("km-password-form", login_html)
        self.assertNotIn("changePassword", login_js)
        self.assertNotIn('name="domainId"', login_html)
        self.assertNotIn("domain_id:", login_js)
        self.assertNotIn("saveConnection", login_js)
        self.assertIn("/api/v1/apps/km-asset/auth/login", adapter)
        self.assertIn("KBOT_UI_CONFIG?.mainApiBaseUrl", adapter)
        self.assertIn("sessionStorage", adapter)
        self.assertIn("Authorization: `Bearer ${session.access_token}`", adapter)
        self.assertIn('/api/v1/auth/refresh', adapter)
        self.assertIn("refreshFlight", adapter)
        self.assertIn('cache: "no-store"', adapter)
        self.assertNotIn("X-KBot-Test-Auth", adapter)
        self.assertNotIn("X-KBot-User-ID", adapter)
        for page_name in (*self.pages, "login.html"):
            source = (KM_ROOT / page_name).read_text(encoding="utf-8")
            self.assertIn(
                "kbot-km-auth-v2.js?v=20260818_2", source, page_name
            )

    def test_km_source_creation_uses_server_fixed_collection(self):
        source_html = (ROOT / "ui" / "km" / "sources.html").read_text(
            encoding="utf-8"
        )
        source_js = (ROOT / "ui" / "km" / "js" / "km-sources.js").read_text(
            encoding="utf-8"
        )
        api_source = (
            ROOT / "services" / "main_api" / "src" / "main_api" / "api"
            / "km_asset_app.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('name="collection_id"', source_html)
        self.assertNotIn("values.collection_id", source_js)
        self.assertIn("const form = event.target", source_js)
        self.assertNotIn("const form = event.currentTarget", source_js)
        self.assertIn('KM_ASSET_COLLECTION_NAME = "assets"', api_source)
        self.assertIn("_fixed_collection_id", api_source)
        self.assertIn("auto_sync_enabled", source_js)
        self.assertIn("开启后台同步", source_js)
        self.assertIn("关闭后台同步", source_js)
        self.assertIn('error?.code === "ROW_VERSION_CONFLICT"', source_js)
        self.assertIn("refreshConflict(values.source_id, form)", source_js)
        self.assertIn("已保留当前输入", source_js)
        self.assertIn("km-sources.js?v=20260818_1", source_html)
        openapi = json.loads(
            (ROOT / "docs" / "openapi" / "main_api_public_v1.json").read_text(
                encoding="utf-8"
            )
        )
        schemas = openapi["components"]["schemas"]
        self.assertNotIn("domain_id", schemas["KmLoginPayload"]["properties"])
        self.assertNotIn(
            "collection_id", schemas["SourceCreatePayload"]["properties"]
        )
        self.assertIn(
            "auto_sync_enabled", schemas["SourceUpdatePayload"]["properties"]
        )

    def test_every_km_page_loads_server_runtime_configuration_first(self):
        for page_name in (*self.pages, "login.html"):
            parser = _PageParser()
            parser.feed((KM_ROOT / page_name).read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(parser.assets), 1, page_name)
            scripts = [
                item for item in parser.assets
                if item.partition("?")[0].endswith(".js")
            ]
            self.assertEqual("../runtime-config.js", scripts[0], page_name)

    def test_ui_server_injects_main_api_address_from_deployment_config(self):
        server = (ROOT / "tools" / "dev_console" / "server.py").read_text(
            encoding="utf-8"
        )
        example = (
            ROOT / "configuration" / "kbot.toml.example"
        ).read_text(encoding="utf-8")
        self.assertIn('os.getenv("KBOT_CONFIG_FILE")', server)
        self.assertIn('get("main_api_base_url")', server)
        self.assertIn('"/ui/runtime-config.js"', server)
        self.assertIn('urlsplit(self.path).path != "/favicon.ico"', server)
        self.assertIn("self.send_response(204)", server)
        self.assertIn("main_api_base_url =", example)

    def test_km_knowledge_core_page_manages_only_fixed_collection(self):
        html = (KM_ROOT / "knowledge-core.html").read_text(encoding="utf-8")
        script = (
            KM_ROOT / "js" / "km-knowledge-core.js"
        ).read_text(encoding="utf-8")
        api_source = (
            ROOT / "services" / "main_api" / "src" / "main_api" / "api"
            / "km_asset_app.py"
        ).read_text(encoding="utf-8")
        shell = (UI_ROOT / "shared" / "kbot-shell.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('value="assets" disabled', html)
        self.assertIn("/knowledge-core/models", script)
        self.assertIn("/knowledge-core/status", script)
        self.assertIn('name="parser_vlm"', html)
        self.assertNotIn('name="parser_llm"', html)
        self.assertIn("model_policy", script)
        self.assertIn("embedding_change_allowed", script)
        self.assertNotIn('name="display_name"', html)
        self.assertNotIn("/api/v1/apps/knowledge-retrieval", script)
        self.assertIn('"owner_app_id": "km_asset"', api_source)
        self.assertIn('"fixed_resource": True', api_source)
        self.assertIn('"km_asset:knowledge_manage"', api_source)
        self.assertIn('"km_asset:knowledge_manage"', shell)
        self.assertNotIn("APEX 页面", shell)


if __name__ == "__main__":
    unittest.main()
