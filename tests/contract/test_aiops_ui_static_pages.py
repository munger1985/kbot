"""AIOps 正式页面静态契约检查。"""

import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path

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
        "chat", "situations", "dashboard", "monitoring", "run-detail", "report-detail", "reports", "inspections",
        "targets", "target-detail",
        "recovery-drills",
        "diagnostic-sources", "diagnostic-source-detail", "operations-knowledge",
        "agents",
        "inspection-plans", "inspection-plan-detail",
        "report-templates",
        "api-clients",
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
        self.assertEqual(12, len(scripts))
        source = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
        self.assertIn("/api/v1/apps/aiops", source)
        self.assertNotIn("/internal/v1", source)
        for script in scripts:
            result = subprocess.run(
                ["node", "--check", str(script)],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_dashboard_is_exception_first_and_truthful_about_missing_data(self):
        page = (AIOPS_ROOT / "dashboard.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-dashboard.js").read_text(
            encoding="utf-8"
        )
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('["dashboard", "Dashboard"]', shell)
        self.assertNotIn('["fleet", "库群总览"]', shell)
        self.assertIn('href="./dashboard.html"', shell)
        self.assertIn("location.replace('./dashboard.html')", (
            AIOPS_ROOT / "login.html"
        ).read_text(encoding="utf-8"))
        self.assertIn("/dashboard", script)
        self.assertIn("优先处理队列", page)
        self.assertIn("风险热点", page)
        self.assertIn("最近 24 小时运维结果", page)
        self.assertIn("无数据不会计入健康", page)
        self.assertIn('UNKNOWN: "无有效数据"', script)
        self.assertIn('STALE: "数据过期"', script)
        self.assertIn('health === "BLIND_SPOT"', script)
        self.assertIn("data-risk-category=\"CAPACITY\"", page)
        self.assertIn("data-risk-category=\"REPLICATION\"", page)
        self.assertIn("data-risk-category=\"BACKUP\"", page)
        self.assertIn("data-risk-category=\"SESSION\"", page)
        self.assertNotIn("scrollIntoView", script)
        self.assertFalse((AIOPS_ROOT / "fleet.html").exists())
        self.assertFalse((AIOPS_ROOT / "js" / "aiops-fleet.js").exists())

    def test_monitoring_is_independent_local_and_contract_driven(self):
        page = (AIOPS_ROOT / "monitoring.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-monitoring.js").read_text(encoding="utf-8")
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(encoding="utf-8")
        vendor = ROOT / "ui" / "vendor"
        self.assertIn('["monitoring", "实时监控"]', shell)
        self.assertIn('data-page="monitoring"', page)
        self.assertIn("../vendor/echarts.min.js?v=6.1.0", page)
        self.assertTrue((vendor / "echarts.min.js").is_file())
        self.assertTrue((vendor / "echarts-LICENSE.txt").is_file())
        self.assertTrue((vendor / "echarts-NOTICE.txt").is_file())
        self.assertIn('const api = "/api/v1/apps/aiops/monitoring"', script)
        self.assertNotIn("/internal/v1", script)
        self.assertNotIn("PromQL", script)
        self.assertNotIn("prometheus_queries", script)
        self.assertIn("instance_ids: state.selected", script)
        self.assertIn("compare_source_id: state.compareSourceId", script)
        self.assertIn("/targets/${encodeURIComponent(state.selected[0])}/monitoring/view?${params}", script)
        self.assertIn("source_display_name", script)
        self.assertIn('id="monitoring-compare-source"', page)
        self.assertIn("aiops-monitoring.js?v=20261010_2", page)
        self.assertIn("state.selected.length >= 12", script)
        self.assertIn("setTimeout(loadView, 250)", script)
        self.assertIn("state.controller?.abort()", script)
        self.assertIn("state.contextController?.abort()", script)
        self.assertIn("connectNulls: false", script)
        self.assertIn("查看数据", script)
        self.assertIn(".ops-monitoring-main [hidden]{display:none!important}", (
            AIOPS_ROOT / "css" / "aiops.css"
        ).read_text(encoding="utf-8"))
        for text in (
            "尚未配置监控源", "监控源未启用", "监控源连接失败",
            "尚未关联数据库实例", "无有效采样", "部分成功",
        ):
            self.assertIn(text, script)
        self.assertNotIn("<iframe", page.lower())
        self.assertNotIn("iframe", script.lower())
        self.assertNotIn("locator", page.lower())
        self.assertNotIn("locator", script.lower())

    def test_chat_code_copy_supports_insecure_http_context(self):
        renderer = (ROOT / "ui" / "shared" / "kbot-markdown.js").read_text(
            encoding="utf-8"
        )
        chat = (AIOPS_ROOT / "chat.html").read_text(encoding="utf-8")
        self.assertIn("navigator.clipboard?.writeText", renderer)
        self.assertIn("window.isSecureContext", renderer)
        self.assertIn('document.execCommand("copy")', renderer)
        self.assertIn("kbot-markdown.js?v=20260827_1", chat)

    def test_chat_selects_attachment_before_submitting_it(self):
        chat = (AIOPS_ROOT / "chat.html").read_text(encoding="utf-8")
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("选择诊断材料", chat)
        self.assertIn("点击“发送”时上传", chat)
        self.assertIn('id="evidence-file" type="file" multiple', chat)
        self.assertIn(".trc,.trace", chat)
        self.assertIn("AWR/ASH（HTML/TXT）", chat)
        change_handler = workspace.split(
            'document.getElementById("evidence-file").onchange =', 1
        )[1].split("await loadConversationList();", 1)[0]
        self.assertNotIn("conversation-uploads", change_handler)
        self.assertIn("点击发送时上传", change_handler)
        self.assertIn("selectedFiles", workspace)
        self.assertIn("HTML_TEXT_EXTRACT", (ROOT / "services" / "aiops_agent" / "src" / "aiops_agent" / "application" / "conversation_inputs.py").read_text(encoding="utf-8"))

    def test_chat_exposes_target_aware_conversation_starters(self):
        chat = (AIOPS_ROOT / "chat.html").read_text(encoding="utf-8")
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="open-starter-menu"', chat)
        self.assertIn('id="starter-dialog"', chat)
        self.assertIn("/conversation-starters?agent_id=", workspace)
        self.assertIn("starterCatalogVersion", workspace)
        self.assertIn("executeStarter", workspace)
        self.assertIn("datetime-local", workspace)
        self.assertIn('field.type === "oracle_datetime"', workspace)
        self.assertIn('value.replace("T", " ")', workspace)
        self.assertNotIn("includes(\"ADG\")", workspace)

    def test_trend_charts_warn_when_monitoring_coverage_is_low(self):
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        stylesheet = (AIOPS_ROOT / "css" / "workspaces.css").read_text(
            encoding="utf-8"
        )
        for page_name in ("chat", "situations", "inspections"):
            page = (AIOPS_ROOT / f"{page_name}.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("workspaces.css?v=20261008_2", page)
            self.assertIn("aiops-workspaces.js?v=20261008_2", page)
        self.assertIn("coverage_warning", workspace)
        self.assertIn("ops-chart-warning", workspace)
        self.assertIn(".ops-chart-warning", stylesheet)

    def test_chat_reloads_images_through_authenticated_api(self):
        auth = (AIOPS_ROOT / "js" / "aiops-auth.js").read_text(
            encoding="utf-8"
        )
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("async function requestBlob", auth)
        self.assertIn("Authorization: `Bearer ${session.access_token}`", auth)
        self.assertIn("imageAttachmentsHtml", workspace)
        self.assertIn("hydrateConversationImages", workspace)
        self.assertIn("/inputs/${item.item_no}/content", workspace)
        self.assertNotIn("file://", workspace)

    def test_chat_downloads_generated_native_oracle_reports(self):
        auth = (AIOPS_ROOT / "js" / "aiops-auth.js").read_text(
            encoding="utf-8"
        )
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('action.status === "SUCCEEDED"', workspace)
        self.assertIn("下载原生 AWR 报告", workspace)
        self.assertIn("下载原生 AWR 对比报告", workspace)
        self.assertIn("下载原生 ASH 报告", workspace)
        self.assertIn("下载原生 SQL Monitor 报告", workspace)
        self.assertIn("oracle-awr-report.html", workspace)
        self.assertIn("oracle-awr-diff-report.html", workspace)
        self.assertIn("oracle-ash-report.html", workspace)
        self.assertIn("oracle-sql-monitor.html", workspace)
        self.assertIn("data-workload-report-action", workspace)
        self.assertIn("/workload-reports/${toolId}?action_id=${actionId}", workspace)
        self.assertNotIn("reports.set(action.tool_id", workspace)
        self.assertIn('"text/html"', workspace)
        self.assertIn('accept = "application/pdf"', auth)
        self.assertIn("Accept: accept", auth)
        self.assertIn("data-download-implementation-runbook", workspace)
        self.assertIn("implementation-runbook.${format.extension}", workspace)
        self.assertIn('commandGroup("人工确认项", manualItems, true)', workspace)
        self.assertIn('"application/pdf"', workspace)
        self.assertIn('"text/markdown"', workspace)
        self.assertNotIn('data-download-implementation-runbook="json"', workspace)
        self.assertIn('"application/zip"', workspace)
        self.assertIn("bindImplementationRunbookActions(panel)", workspace)
        pages = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(encoding="utf-8")
        self.assertIn("data-extract-case", pages)
        self.assertIn("/operations-knowledge/reports/", pages)

    def test_pages_do_not_embed_demo_records_or_api_keys(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in AIOPS_ROOT.rglob("*") if path.is_file()
        )
        self.assertNotIn("140.238.44.208", source)
        self.assertNotIn("kbot_ak_", source)
        self.assertNotIn("/metrics", source)
        self.assertNotIn("Idempotency-Key':crypto.randomUUID()", source)
        self.assertNotIn("client_file_id: crypto.randomUUID()", source)
        self.assertIn("KBotAIOpsAuth.uuid()", source)

    def test_operations_knowledge_page_uses_business_registry(self):
        page = (AIOPS_ROOT / "operations-knowledge.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-operations-knowledge.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="manual-form"', page)
        self.assertIn('id="search-form"', page)
        self.assertIn("/operations-knowledge", script)
        self.assertIn("/search-preview", script)
        self.assertIn("X-Expected-Asset-Row-Version", script)
        self.assertIn('data-review="publish"', script)
        self.assertNotIn("Knowledge Core", page)
        self.assertNotIn("Collection", page)
        self.assertNotIn("Embedding", page)
        self.assertNotIn("/api/v1/knowledge", script)

    def test_login_uses_fixed_aiops_domain_contract(self):
        login = (AIOPS_ROOT / "login.html").read_text(encoding="utf-8")
        auth = (AIOPS_ROOT / "js" / "aiops-auth.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("aiops_portal", login)
        self.assertNotIn('name="domain_id"', login)
        self.assertIn("/api/v1/apps/aiops/auth/login", auth)

    def test_request_id_falls_back_without_crypto_random_uuid(self):
        script = """
const fs = require("node:fs");
const vm = require("node:vm");
const sandbox = {
  crypto: {},
  sessionStorage: {getItem: () => null, setItem: () => {}, removeItem: () => {}},
  location: {replace: () => {}},
  FormData: class {},
};
vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const value = sandbox.KBotAIOpsAuth.uuid();
if (!/^ui-[0-9]+-[0-9a-f]+$/.test(value)) process.exit(1);
"""
        result = subprocess.run(
            ["node", "-e", script, str(AIOPS_ROOT / "js" / "aiops-auth.js")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_api_validation_error_is_human_readable(self):
        auth = (AIOPS_ROOT / "js" / "aiops-auth.js").read_text(
            encoding="utf-8"
        )
        agent = (AIOPS_ROOT / "js" / "aiops-agents.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("Array.isArray(detail)", auth)
        self.assertIn('button.textContent = editing ? "保存中…" : "创建中…"', agent)
        self.assertIn("shell.toast(error.message)", agent)
        self.assertIn('status: editing ? form.elements.status.value : "DRAFT"', agent)
        self.assertIn("controlled_change_enabled", agent)
        self.assertIn("逐条人工审批", agent)
        self.assertNotIn("selected && !executionConfigured", agent)
        pages = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('page !== "agents"', pages)
        self.assertIn("!paths[page] || !panel", pages)

    def test_source_connectivity_toast_displays_final_result(self):
        pages = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("连通性检查结果：", pages)
        self.assertIn("result.connectivity_status", pages)
        self.assertIn("result.last_error_code", pages)
        self.assertNotIn("连通性检查已完成", pages)
        self.assertNotIn("Target 连通性检查已提交", pages)

    def test_target_create_form_uses_public_contract_fields(self):
        page = (AIOPS_ROOT / "targets.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-targets.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="target-dialog"', page)
        self.assertIn('id="target-form"', page)
        self.assertIn('id="test-target-connection"', page)
        self.assertIn("diagnostic_credential", script)
        self.assertIn("Idempotency-Key", script)
        self.assertIn("/targets/test-connection", script)
        self.assertIn('oracle ? "service" : "database"', script)
        self.assertIn('name="oracle_container_scope"', page)
        self.assertIn('value="CDB_ROOT"', page)
        self.assertIn('value="PDB"', page)
        self.assertIn('value="NON_CDB"', page)
        self.assertIn('name="oracle_pdb_name"', page)
        self.assertIn("oracle_container_scope: oracleScope.value", script)
        self.assertIn('oracleScope.value === "PDB"', script)
        self.assertIn("? oraclePdbName.value.trim()", script)
        self.assertIn("实际 CON_NAME 比对", page)
        self.assertIn('method: "PATCH"', script)
        self.assertIn("diagnostic-credential:rotate", script)
        self.assertIn("execution-credential:rotate", script)
        self.assertIn('name="execution_username"', page)
        self.assertIn('name="execution_password"', page)
        self.assertIn('name="controlled_change_enabled"', page)
        self.assertIn("数据库访问与变更边界", page)
        self.assertIn("数据库写操作必须在这里显式开启", page)
        self.assertIn('id="target-access-summary"', page)
        self.assertIn('name="importance_level"', page)
        self.assertIn("importance_level: Number", script)
        self.assertIn("openEdit", script)
        self.assertIn('get("edit")', script)
        self.assertIn('credentialConfigured("execution")', script)
        self.assertIn("db_type", script)
        pages_script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        detail_page = (AIOPS_ROOT / "target-detail.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('["_access", "访问模式", "target-access"]', pages_script)
        self.assertIn('data-target-action="edit"', pages_script)
        self.assertIn('id="edit-target-access"', detail_page)
        self.assertIn("targets.html?edit=", pages_script)
        self.assertNotIn("engine_type", script)

    def test_target_detail_owns_current_user_notification_subscription(self):
        page = (AIOPS_ROOT / "target-detail.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        forms = (AIOPS_ROOT / "css" / "aiops-forms.css").read_text(
            encoding="utf-8"
        )
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="target-subscription-form"', page)
        self.assertIn('id="target-subscription-dialog"', page)
        self.assertIn('id="open-target-subscription"', page)
        self.assertIn("data-close-target-subscription", page)
        self.assertIn('name="follow_target"', page)
        self.assertIn('name="minimum_severity"', page)
        self.assertIn('value="SITUATION_DETECTED"', page)
        self.assertIn('value="DIAGNOSIS_STARTED"', page)
        self.assertIn('value="REPORT_READY"', page)
        self.assertIn('value="SITUATION_RECOVERED"', page)
        self.assertIn("initializeTargetSubscription", script)
        self.assertIn("/notification-subscriptions/targets/", script)
        self.assertIn("dialog.showModal()", script)
        self.assertIn("dialog.close()", script)
        self.assertNotIn("target-subscription-panel", page)
        self.assertNotIn(".target-subscription-panel{position:sticky", forms)
        self.assertNotIn('["notification-subscriptions", "主动分享"]', shell)
        self.assertFalse(
            (AIOPS_ROOT / "notification-subscriptions.html").exists()
        )

    def test_target_detail_uses_business_overview_instead_of_raw_payload(self):
        page = (AIOPS_ROOT / "target-detail.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        overview = script.split("function targetOverviewHtml", 1)[1].split(
            "function reportPresentationHtml", 1
        )[0]
        self.assertIn('id="ops-detail" class="ops-panel span-12"', page)
        self.assertIn("targetOverviewHtml(data)", script)
        self.assertIn("数据库范围", overview)
        self.assertIn("连接状态", overview)
        self.assertIn("凭据与变更", overview)
        self.assertIn("观测与采集", overview)
        for internal_field in (
            "schema_version", "target_id", "created_at", "created_by", "updated_by"
        ):
            self.assertNotIn(internal_field, overview)
        self.assertNotIn("JSON.stringify", overview)

    def test_report_center_translates_backend_dictionary_values(self):
        script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("function reportDisplayText", script)
        self.assertIn('EVIDENCE_BOUNDARY: "证据边界"', script)
        self.assertIn('INCONCLUSIVE: "证据不足，无法定论"', script)
        self.assertIn(
            'MISSING_FINAL_RESULT: "缺少最终诊断结果"', script
        )
        self.assertIn(
            "section.display_name || reportDisplayText(section.kind", script
        )
        self.assertIn(
            "data.status_display || reportDisplayText(data.status", script
        )

    def test_recovery_drills_use_independent_operations_workspace(self):
        target_page = (AIOPS_ROOT / "target-detail.html").read_text(
            encoding="utf-8"
        )
        drill_page = (AIOPS_ROOT / "recovery-drills.html").read_text(
            encoding="utf-8"
        )
        drill_script = (AIOPS_ROOT / "js" / "aiops-recovery-drills.js").read_text(
            encoding="utf-8"
        )
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('id="target-recovery-drill-form"', target_page)
        self.assertIn('id="target-recovery-drills-link"', target_page)
        self.assertIn('id="recovery-drill-form"', drill_page)
        self.assertIn('id="recovery-drill-history"', drill_page)
        self.assertIn("/recovery-drills", drill_script)
        self.assertNotIn("Oracle SCN", drill_page)
        self.assertNotIn("Resetlogs ID", drill_page)
        self.assertNotIn("Incarnation", drill_page)
        self.assertNotIn('name="evidence_kind"', drill_page)
        self.assertNotIn('name="evidence_reference"', drill_page)
        self.assertNotIn('name="evidence_hash"', drill_page)
        self.assertIn("recovery_marker: { kind: currentTarget.db_type }", drill_script)
        self.assertIn("evidence: []", drill_script)
        self.assertIn("数据时间不能晚于模拟故障时间", drill_script)
        self.assertIn("验证完成时间不能早于模拟故障时间", drill_script)
        self.assertIn('["recovery-drills", "恢复演练"]', shell)

    def test_target_detail_and_chat_own_target_fact_confirmation(self):
        page = (AIOPS_ROOT / "target-detail.html").read_text(encoding="utf-8")
        pages = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        workspaces_css = (AIOPS_ROOT / "css" / "workspaces.css").read_text(
            encoding="utf-8"
        )
        forms_css = (AIOPS_ROOT / "css" / "aiops-forms.css").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="target-facts"', page)
        self.assertIn("initializeTargetFacts", pages)
        self.assertIn("/facts", pages)
        self.assertIn("FACT_CONFIRMATION", workspace)
        self.assertIn("data-confirm-target-fact", workspace)
        self.assertIn("target-facts:confirm", workspace)
        self.assertIn(".ops-fact-confirmation", workspaces_css)
        self.assertIn(".target-facts-list", forms_css)
        self.assertIn(".target-facts-form", forms_css)
        self.assertNotIn("command_preview", workspace.split("function factConfirmationHtml")[1].split("function htmlReportLinksHtml")[0])

    def test_configuration_pages_open_real_create_and_edit_dialogs(self):
        pages = {
            "diagnostic-sources.html": "diagnostic-source-dialog",
            "inspection-plans.html": "inspection-plan-dialog",
        }
        for filename, dialog_id in pages.items():
            source = (AIOPS_ROOT / filename).read_text(encoding="utf-8")
            self.assertIn(f'id="{dialog_id}"', source)
            self.assertIn("aiops-configurations.js", source)
        script = (AIOPS_ROOT / "js" / "aiops-configurations.js").read_text(
            encoding="utf-8"
        )
        diagnostic_page = (
            AIOPS_ROOT / "diagnostic-sources.html"
        ).read_text(encoding="utf-8")
        self.assertIn("/diagnostic-sources/test-connection", script)
        self.assertNotIn("声明能力（JSON 对象）", diagnostic_page)
        self.assertNotIn("Adapter 配置（JSON 对象）", diagnostic_page)
        self.assertNotIn('name="adapter_version"', diagnostic_page)
        self.assertNotIn('name="target_label"', diagnostic_page)
        self.assertNotIn("form.elements.target_label", script)
        self.assertIn('name="tenant_id"', diagnostic_page)
        self.assertIn('id="generate-source-webhook-secret"', diagnostic_page)
        self.assertIn('id="rotate-source-webhook-key"', diagnostic_page)
        self.assertIn('id="source-webhook-onboarding"', diagnostic_page)
        self.assertIn('id="copy-created-webhook-ini"', diagnostic_page)
        self.assertIn("创建并生成接入凭据", script)
        self.assertIn("showWebhookOnboarding", script)
        self.assertIn("requestWebhookKey(saved)", script)
        self.assertIn("crypto.getRandomValues", script)
        self.assertIn("webhook-key:rotate", script)
        self.assertIn('"If-Match": `"rv-${editing.row_version}"`', script)
        self.assertIn("copyWebhookSecret", script)
        self.assertIn("copyWebhookKey", script)
        self.assertIn("renderSourceType", script)
        pages_script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('data-source-action="connectivity"', pages_script)
        self.assertIn('data-source-action="enable"', pages_script)
        self.assertIn('data-source-action="disable"', pages_script)
        self.assertIn("connectivity_check_pending", pages_script)
        self.assertIn(
            'hasOwnProperty.call(item, "readonly_connection_enabled")',
            pages_script,
        )
        self.assertIn('data-target-action="connectivity"', pages_script)
        self.assertIn('data-target-action="detail"', pages_script)
        self.assertIn('data-target-action="enable"', pages_script)
        self.assertNotIn('data-target-action="maintenance"', pages_script)
        self.assertIn('data-target-action="disable"', pages_script)
        self.assertIn('method: editing ? "PATCH" : "POST"', script)
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('["policies", "执行策略"]', shell)
        self.assertFalse((AIOPS_ROOT / "policies.html").exists())
        self.assertIn('"If-Match"', script)
        self.assertIn("openEdit", script)

    def test_auth_rejects_missing_runtime_configuration(self):
        script = (AIOPS_ROOT / "js" / "aiops-auth.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("KBOT_UI_CONFIG?.mainApiBaseUrl", script)
        self.assertIn("AIOps UI 未加载 Main API 部署配置", script)

    def test_inspection_plan_uses_visual_schedule_builder(self):
        page = (AIOPS_ROOT / "inspection-plans.html").read_text(
            encoding="utf-8"
        )
        script = (
            AIOPS_ROOT / "js" / "aiops-configurations.js"
        ).read_text(encoding="utf-8")

        self.assertIn('name="cron_expression" type="hidden"', page)
        self.assertIn('name="schedule_type" type="hidden"', page)
        self.assertNotIn("Cron 表达式", page)
        self.assertIn('name="schedule_mode" value="DAILY"', page)
        self.assertIn('name="schedule_mode" value="WEEKLY"', page)
        self.assertIn('name="schedule_mode" value="MONTHLY"', page)
        self.assertIn('name="schedule_mode" value="INTERVAL"', page)
        self.assertIn('id="inspection-schedule-summary"', page)
        self.assertIn('name="agent_id" required', page)
        self.assertIn("创建并启用", page)
        self.assertIn('id="plan-inspection-template-id"', page)
        self.assertIn('id="inspection-template-preview"', page)
        self.assertIn("模板检查内容", page)
        self.assertIn(">03<", page)
        self.assertIn(">04<", page)
        self.assertIn("运行策略", page)
        self.assertIn("inspection_template_id", script)
        self.assertIn("function renderInspectionTemplatePreview", script)
        self.assertIn("/inspection-templates", script)
        self.assertIn("function buildSchedule(form)", script)
        self.assertIn("function hydrateScheduleBuilder(form, plan)", script)
        self.assertIn("function renderSchedule(form)", script)
        self.assertIn('cron: "*/15 * * * *"', script)
        pages_script = (
            AIOPS_ROOT / "js" / "aiops-pages.js"
        ).read_text(encoding="utf-8")
        self.assertIn('["schedule_type", "调度周期", "schedule"]', pages_script)
        self.assertIn('DAILY: "每天", WEEKLY: "每周", CRON: "灵活周期"', pages_script)
        self.assertIn('data-inspection-action="${action}"', pages_script)
        self.assertIn('["PAUSED", "DISABLED"].includes(item.status)', pages_script)
        self.assertIn('/inspection-plans/${encodeURIComponent(item.plan_id)}/${button.dataset.inspectionAction}', pages_script)

    def test_agent_form_owns_resources_and_approval_is_not_a_policy_input(self):
        page = (AIOPS_ROOT / "agents.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-agents.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('name="diagnostic_source_ids"', script)
        self.assertIn('name="target_ids"', script)
        self.assertIn('id="agent-targets"', page)
        self.assertIn("监控源必选；数据库 Target 可选", page)
        self.assertIn("数据库 Target 为可选项", script)
        self.assertNotIn("至少选择一个逻辑 Target", script)
        self.assertNotIn('name="allow_change_execution"', page)
        self.assertIn('id="agent-controlled-actions"', page)
        self.assertIn("controlled_action_execution:", script)
        self.assertIn('/action-catalog/', script)
        self.assertIn("与全部所选监控源的有效映射", script)
        self.assertIn("默认仅允许只读诊断", page)
        self.assertIn('data-scope-kind="dynamic_parameters"', script)
        self.assertIn("selectedDynamicParameters", script)
        self.assertIn("data-action-scope", script)
        self.assertIn("scope_requirements", script)
        self.assertIn("当前 Target 没有可选项", script)
        self.assertNotIn("parseDynamicParameters", script)
        self.assertNotIn("逗号分隔", script)
        self.assertNotIn("分号分隔", script)
        self.assertIn('name="auto_alert_enabled"', page)
        self.assertIn('name="auto_observe_min_target_level"', page)
        self.assertIn("auto_observe_min_target_level:", script)
        self.assertIn('name="diagnosis_model_id" required', page)
        self.assertIn('name="planner_model_id" required', page)
        self.assertIn('planner_llm: plannerModelId', script)
        self.assertIn('diagnosis_llm: diagnosisModelId', script)
        self.assertIn('request("/api/v1/model-catalog")', script)
        self.assertNotIn('`${api}/model-catalog`', script)
        self.assertNotIn('models: modelId ? { diagnosis:', script)
        self.assertIn("只适用于自动触发", page)
        self.assertIn('id="agent-binding-summary"', page)
        self.assertIn('class="agent-form-section"', page)
        self.assertIn("/source-bindings", script)
        self.assertIn("assertExistingBindings", script)
        self.assertIn("Agent 不会创建或修改映射", script)
        self.assertIn("locator_hint", script)
        self.assertNotIn("ensureSourceBindings", script)
        self.assertNotIn("source_locator_key", script)
        self.assertNotIn("data-loki-target-label", script)
        self.assertNotIn("data-prometheus-host-target", script)
        self.assertNotIn("scrollIntoView", script)
        self.assertNotIn('name="policy_id"', page)
        self.assertNotIn("max_risk_level", page)
        self.assertNotIn("allowed_action_types", page)

    def test_workspace_separates_approval_and_manual_actions(self):
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('payload.execution_mode === "MANUAL_ONLY"', workspace)
        self.assertIn("仅供人工执行", workspace)
        self.assertIn("data-manual-proposal", workspace)
        self.assertIn("/manual-result", workspace)
        self.assertIn("data-copy-code", workspace)
        self.assertIn('state.permissions.has("aiops:proposal:approve")', workspace)

    def test_shell_renders_navigation_from_access_permissions(self):
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("pagePermissions", shell)
        self.assertIn('agents: "aiops:agent_manage"', shell)
        self.assertIn('"api-clients": "aiops:api_key_manage"', shell)
        self.assertIn("shellMarkup(access)", shell)

    def test_business_workspace_includes_report_center(self):
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        situations = (AIOPS_ROOT / "situations.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('["chat", "智能运维"]', shell)
        self.assertIn('["situations", "告警诊断"]', shell)
        self.assertIn('["inspections", "日常巡检"]', shell)
        self.assertNotIn('["runs", "诊断运行"]', shell)
        self.assertIn('["reports", "报告中心"]', shell)
        self.assertIn('["report-templates", "模板管理"]', shell)
        business_workspace = shell.split('["资源配置"', 1)[0]
        self.assertNotIn('["report-templates", "模板管理"]', business_workspace)
        self.assertIn("source_run_id", workspace)
        self.assertIn("source_situation_id", workspace)
        self.assertIn("正在等待 Agent 自动诊断任务启动", workspace)
        self.assertIn("监控来源", workspace)
        self.assertIn("告警内容", workspace)
        self.assertIn("累计 ${esc(detail.event_count)} 次观测", workspace)
        self.assertNotIn("个监控信号", workspace)
        self.assertIn("Agent 正在诊断", workspace)
        self.assertIn("自动诊断状态", workspace)
        self.assertIn("run.error_code", workspace)
        self.assertIn("run.error_message", workspace)
        self.assertIn('class="ops-error"', workspace)
        self.assertNotIn("本次自动诊断已结束但未形成可展示结果", workspace)
        self.assertIn("scheduleSituationRefresh", workspace)
        self.assertIn("terminalRunStatuses", workspace)
        self.assertIn('schemaVersion === "AIOPS_TURN_RESULT.v1"', workspace)
        self.assertIn("conversationAnswerHtml(result)", workspace)
        self.assertIn("诊断过程", workspace)
        self.assertIn("ops-progress-timeline", workspace)
        self.assertIn('"assessment.started"', workspace)
        self.assertIn("updateProgressElapsed", workspace)
        self.assertIn('id="target-select"', (AIOPS_ROOT / "chat.html").read_text(encoding="utf-8"))
        self.assertIn("target_id: targetId", workspace)
        self.assertIn("当前 Agent 未绑定所选 Target", workspace)
        self.assertIn(
            'content: [{ content_type: "TEXT", text: fields.message }]',
            workspace,
        )
        self.assertNotIn(
            "JSON.stringify({ ...body, source_run_id: run.ops_run_id })",
            workspace,
        )
        self.assertIn("KBotAIOpsAuth.stream", workspace)
        self.assertIn('event === "answer.delta"', workspace)
        self.assertIn('id="case-filters"', situations)
        self.assertIn('name="agent_id"', situations)
        self.assertIn('name="severity"', situations)
        self.assertIn('name="target_id"', situations)
        self.assertIn('["agent_id", "severity", "target_id"]', workspace)
        self.assertIn(
            'progress.insertAdjacentHTML("afterend", messageHtml("AGENT", ""))',
            workspace,
        )
        self.assertIn("const message = progress.nextElementSibling", workspace)
        self.assertNotIn(
            'progress.insertAdjacentHTML("beforebegin", messageHtml("AGENT", ""))',
            workspace,
        )
        self.assertIn("${plan}${progress}${answer}", workspace)
        self.assertIn("enqueueAnswerDelta", workspace)
        self.assertIn("waitForTyping", workspace)
        self.assertIn("evidenceDetails", workspace)
        self.assertIn('class="ops-evidence"', workspace)
        self.assertIn('block.block_type === "TABLE"', workspace)
        self.assertIn('block.block_type === "CHART"', workspace)
        self.assertIn('block.block_type === "EVIDENCE_REFERENCES"', workspace)
        self.assertNotIn("tablespaceChartHtml", workspace)
        self.assertIn("inspectionMarkdown", workspace)
        self.assertIn("inspectionAnswerHtml", workspace)
        self.assertIn("markdown.render(inspectionMarkdown(result))", workspace)
        self.assertIn("inspectionReportHtml", workspace)
        self.assertIn(
            "const findings = checks.flatMap((item) => values(item?.findings))",
            workspace,
        )
        self.assertIn(
            "empty_reasons: findings.length ? [] : [emptyFindings]",
            workspace,
        )
        self.assertIn('schemaVersion === "REPORT_CONTENT.v1"', workspace)
        self.assertIn('block.block_type === "FINDING_CARDS"', workspace)
        self.assertIn("conversationAnswerHtml(result)", workspace)
        self.assertIn("bindWorkloadReportActions(panel)", workspace)
        css = (AIOPS_ROOT / "css" / "workspaces.css").read_text(encoding="utf-8")
        self.assertIn(".ops-findings", css)
        self.assertIn(".ops-finding-card", css)
        self.assertIn(".ops-finding-fields", css)
        self.assertIn(".ops-analysis", css)
        self.assertIn(".ops-solution", css)
        self.assertIn("inspectionBullets", workspace)
        self.assertIn("本期没有需要报告的记录，结果正常。", workspace)
        self.assertIn("未发现数据缺口，全部检查已形成可验证观测。", workspace)
        self.assertIn("继续按既定周期执行该巡检模板并关注趋势变化。", workspace)
        self.assertNotIn('`**根因等级：**', workspace)
        self.assertIn("answerBlockHtml", workspace)
        self.assertIn("turnEvidenceHtml", workspace)
        self.assertIn(
            'const dataBlocks = blocks.filter((block) => '
            'block.block_type === "TABLE")',
            workspace,
        )
        self.assertIn('payload.chart_type === "LINE"', workspace)
        self.assertIn('payload.chart_type === "STACKED_BAR"', workspace)
        self.assertIn('const narrativeBlocks = answerBlocks.filter', workspace)
        self.assertIn("原始取证结果", workspace)
        self.assertNotIn("answerBlocks.map(answerBlockHtml).join", workspace)
        self.assertIn("investigationPlanHtml", workspace)
        inspection_workspace = workspace.split(
            "async function showInspection", 1
        )[1].split("async function initCases", 1)[0]
        self.assertIn("inspectionAnswerHtml(result)", inspection_workspace)
        self.assertNotIn(
            "conversationAnswerHtml(result)", inspection_workspace
        )
        self.assertIn(
            "本次巡检已完成，但未生成可展示的巡检结论。",
            workspace,
        )
        self.assertIn("巡检尚未形成最终报告", workspace)
        self.assertIn("showInvestigationPlan", workspace)
        self.assertIn("调查计划与判断依据", workspace)
        progress_rule = css.split(".ops-progress {", 1)[1].split("}", 1)[0]
        self.assertIn("max-height: 320px", progress_rule)
        self.assertIn("overflow-y: auto", progress_rule)
        self.assertIn("scrollbar-gutter: stable", progress_rule)
        self.assertIn(".ops-investigation-plan", css)
        self.assertIn("max-height: 360px", css)
        self.assertIn("overflow-y: auto", css)
        self.assertIn("待验证假设", workspace)
        self.assertIn("预期证据", workspace)
        self.assertIn("payload.public_sections", workspace)
        self.assertIn('"planning.route.selected"', workspace)
        self.assertIn("turn.investigation_plan", workspace)
        self.assertIn("payload.plan", workspace)
        self.assertIn("diagnosticQueryApprovalHtml", workspace)
        self.assertIn("diagnosticQueryDecision", workspace)
        self.assertIn("data-query-decision", workspace)
        self.assertIn("request.sql_text", workspace)
        self.assertIn("request.parameters", workspace)
        self.assertIn("turn.ops_run_id", workspace)
        self.assertIn("data-generate-report-conversation", workspace)
        self.assertIn("conversation_id: conversationId", workspace)
        self.assertIn("全部已完成 Turn", workspace)
        self.assertNotIn("${plan}${progress}${answer}${report}", workspace)
        self.assertIn('["WAITING_INPUT", "WAITING_APPROVAL"]', workspace)
        self.assertIn(
            "includes(run?.status)",
            workspace,
        )
        self.assertIn("审批已提交，正在继续诊断", workspace)
        self.assertIn("const evidence = new Map()", workspace)
        self.assertIn("${evidence}</div></article>", workspace)
        self.assertIn("followTurn", workspace)
        self.assertIn('"Last-Event-ID": lastEventId', workspace)
        self.assertIn("terminalTurnStatuses.has(turn.status)", workspace)
        self.assertIn("let streamFailed = false", workspace)
        self.assertIn("诊断仍在后台运行，正在继续获取进度", workspace)
        self.assertIn('"WAITING_USER", "COMPLETED"', workspace)
        self.assertIn("await followTurn(receipt.conversation_id", workspace)
        self.assertIn("await loadConversation(receipt.conversation_id)", workspace)
        self.assertIn("resumeActiveTurns(conversation.conversation_id, turns)", workspace)
        self.assertIn("activeTurnFollowers.has(followerKey)", workspace)
        self.assertNotIn("followTurn(receipt.conversation_id, receipt.turn_id, progress)\n        .then", workspace)
        self.assertIn("请先选择 Target 和 Agent 查看会话历史", workspace)
        self.assertIn("?agent_id=${encodeURIComponent(selectedAgent)}", workspace)
        self.assertIn("archiveConversation", workspace)
        self.assertIn('method: "DELETE"', workspace)
        self.assertIn('class="ops-workspace-delete"', workspace)
        self.assertIn('data-delete-id="${esc(item.conversation_id)}"', workspace)
        self.assertIn("会话将从聊天历史中移除", workspace)
        self.assertIn("关联的诊断、证据和变更审计记录仍会保留", workspace)
        self.assertIn("upload", workspace.lower())
        for obsolete in (
            "fleet.html", "runs.html",
            "changes.html", "notifications.html",
        ):
            self.assertFalse((AIOPS_ROOT / obsolete).exists())

    def test_report_center_supports_history_download_and_versioned_editing(self):
        page = (AIOPS_ROOT / "reports.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("查看已生成的正式报告", page)
        self.assertIn("ops-report-list", page)
        self.assertIn("实际发布时间", page)
        self.assertIn('reports: { path: "/reports"', script)
        self.assertIn("data-report-version", script)
        self.assertIn("data-download-report", script)
        self.assertIn("data-edit-report", script)
        self.assertIn('method: "PATCH"', script)
        self.assertIn('"If-Match": `"rv-${report.report_version}"`', script)
        self.assertIn("不会覆盖旧版", script)

    def test_report_template_configuration_is_not_a_business_entry(self):
        page = (AIOPS_ROOT / "report-templates.html").read_text(
            encoding="utf-8"
        )
        script = (AIOPS_ROOT / "js" / "aiops-report-templates.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("系统预设模板只读", page)
        self.assertIn("证据边界（必选）", page)
        self.assertIn("/inspection-templates", script)
        self.assertIn("/session-report-templates", script)
        self.assertIn("SESSION_REPORT_TEMPLATE.v1", script)
        self.assertIn("data-edit-inspection-template", script)
        self.assertIn("data-edit-session-template", script)
        self.assertIn("expected_row_version", script)


if __name__ == "__main__":
    unittest.main()
