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
        "chat", "situations", "run-detail", "report-detail", "inspections",
        "targets", "target-detail",
        "diagnostic-sources", "diagnostic-source-detail", "knowledge-core",
        "agents",
        "inspection-plans", "inspection-plan-detail",
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
        self.assertEqual(8, len(scripts))
        source = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
        self.assertIn("/api/v1/apps/aiops", source)
        self.assertNotIn("/internal/v1", source)
        for script in scripts:
            result = subprocess.run(
                ["node", "--check", str(script)],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_chat_code_copy_supports_insecure_http_context(self):
        renderer = (ROOT / "ui" / "shared" / "kbot-markdown.js").read_text(
            encoding="utf-8"
        )
        chat = (AIOPS_ROOT / "chat.html").read_text(encoding="utf-8")
        self.assertIn("navigator.clipboard?.writeText", renderer)
        self.assertIn("window.isSecureContext", renderer)
        self.assertIn('document.execCommand("copy")', renderer)
        self.assertIn("kbot-markdown.js?v=20260827_1", chat)

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
        self.assertIn("实际执行仍逐条审批", agent)
        self.assertNotIn("selected && !executionConfigured", agent)
        pages = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('page !== "agents"', pages)
        self.assertIn("!paths[page] || !panel", pages)

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
        self.assertIn('method: "PATCH"', script)
        self.assertIn("diagnostic-credential:rotate", script)
        self.assertIn("execution-credential:rotate", script)
        self.assertIn('name="execution_username"', page)
        self.assertIn('name="execution_password"', page)
        self.assertIn("openEdit", script)
        self.assertIn("db_type", script)
        self.assertNotIn("engine_type", script)

    def test_target_detail_owns_current_user_notification_subscription(self):
        page = (AIOPS_ROOT / "target-detail.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="target-subscription-form"', page)
        self.assertIn('name="follow_target"', page)
        self.assertIn('name="minimum_severity"', page)
        self.assertIn('value="SITUATION_DETECTED"', page)
        self.assertIn('value="DIAGNOSIS_STARTED"', page)
        self.assertIn('value="REPORT_READY"', page)
        self.assertIn('value="SITUATION_RECOVERED"', page)
        self.assertIn("initializeTargetSubscription", script)
        self.assertIn("/notification-subscriptions/targets/", script)
        self.assertNotIn('["notification-subscriptions", "主动分享"]', shell)
        self.assertFalse(
            (AIOPS_ROOT / "notification-subscriptions.html").exists()
        )

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
        self.assertIn('/inspection-plans/${encodeURIComponent(item.plan_id)}/${button.dataset.inspectionAction}', pages_script)

    def test_agent_form_owns_resources_and_policy_inputs(self):
        page = (AIOPS_ROOT / "agents.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-agents.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('name="diagnostic_source_ids"', script)
        self.assertIn('name="target_ids"', script)
        self.assertIn('id="agent-targets"', page)
        self.assertNotIn('name="allow_change_execution"', page)
        self.assertIn('id="agent-controlled-actions"', page)
        self.assertIn('controlled_action_execution:', script)
        self.assertIn('/action-catalog/', script)
        self.assertIn("data-action-dynamic-parameters", script)
        self.assertIn("data-action-resource-plans", script)
        self.assertIn("data-action-privilege-grantees", script)
        self.assertIn("data-action-system-privileges", script)
        self.assertIn("data-action-object-privileges", script)
        self.assertIn("parseDynamicParameters", script)
        self.assertIn('name="auto_alert_enabled"', page)
        self.assertIn('name="diagnosis_model_id" required', page)
        self.assertIn('name="planner_model_id" required', page)
        self.assertIn('planner_llm: plannerModelId', script)
        self.assertIn('diagnosis_llm: diagnosisModelId', script)
        self.assertIn('`${api}/model-catalog`', script)
        self.assertNotIn('models: modelId ? { diagnosis:', script)
        self.assertIn("只适用于告警自动触发", page)
        self.assertIn('id="agent-binding-summary"', page)
        self.assertIn('class="agent-form-section"', page)
        self.assertIn("/source-bindings", script)
        self.assertIn("ensureSourceBindings", script)
        self.assertIn("data-loki-target-label", script)
        self.assertIn("data-prometheus-host-target", script)
        self.assertIn("Node Exporter 两个 target_key", script)
        self.assertIn('"Idempotency-Key"', script)
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

    def test_business_workspace_has_exact_three_entry_points(self):
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('["chat", "智能诊断"]', shell)
        self.assertIn('["situations", "告警诊断"]', shell)
        self.assertIn('["inspections", "日常巡检"]', shell)
        self.assertNotIn('["runs", "诊断运行"]', shell)
        self.assertNotIn('["reports", "报告中心"]', shell)
        self.assertNotIn('["report-templates", "报告模板"]', shell)
        self.assertIn("source_run_id", workspace)
        self.assertIn("source_situation_id", workspace)
        self.assertIn("正在等待 Agent 自动诊断任务启动", workspace)
        self.assertIn("监控来源", workspace)
        self.assertIn("告警内容", workspace)
        self.assertIn("累计 ${esc(detail.event_count)} 次观测", workspace)
        self.assertNotIn("个监控信号", workspace)
        self.assertIn("Agent 正在诊断", workspace)
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
        self.assertNotIn('`**根因等级：**', workspace)
        self.assertIn("answerBlockHtml", workspace)
        self.assertIn("turnEvidenceHtml", workspace)
        self.assertIn(
            'const dataBlocks = blocks.filter((block) => '
            '["TABLE", "CHART"].includes(block.block_type))',
            workspace,
        )
        self.assertIn('const narrativeBlocks = answerBlocks.filter', workspace)
        self.assertIn("原始取证结果", workspace)
        self.assertNotIn("answerBlocks.map(answerBlockHtml).join", workspace)
        self.assertIn("investigationPlanHtml", workspace)
        self.assertIn("showInvestigationPlan", workspace)
        self.assertIn("调查计划与判断依据", workspace)
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
        self.assertIn("关联的诊断、证据和变更审计记录仍会保留", workspace)
        self.assertIn("upload", workspace.lower())
        for obsolete in (
            "dashboard.html", "runs.html", "reports.html",
            "changes.html", "notifications.html", "report-templates.html",
        ):
            self.assertFalse((AIOPS_ROOT / obsolete).exists())


if __name__ == "__main__":
    unittest.main()
