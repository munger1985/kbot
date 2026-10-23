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
    def test_page_heads_share_one_left_aligned_anatomy(self):
        pages = [path for path in AIOPS_ROOT.glob("*.html") if path.name != "login.html"]
        for path in pages:
            page = path.read_text(encoding="utf-8")
            self.assertIn('class="ops-head', page, path.name)

        stylesheet = (AIOPS_ROOT / "css" / "aiops.css").read_text(encoding="utf-8")
        self.assertIn(".ops-main>.ops-head{box-sizing:border-box;align-items:flex-start", stylesheet)
        self.assertIn("border-bottom:1px solid var(--line);text-align:left", stylesheet)
        self.assertIn(".ops-main>.ops-head.agent-page-head>div:first-child{padding-left:0;border-left:0}", stylesheet)

    pages = {
        "chat", "situations", "dashboard", "work-item-detail", "responsibility-groups", "monitoring", "runs", "run-detail", "report-detail", "reports", "inspections",
        "business-systems", "business-system-workspace", "targets", "target-detail",
        "recovery-drills",
        "diagnostic-sources", "diagnostic-source-detail", "operations-knowledge", "knowledge-bases", "knowledge-tags",
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

    def test_pages_share_current_interactive_shell(self):
        expected = "./js/aiops-shell.js?v=20261023_1"
        pages = [path for path in AIOPS_ROOT.glob("*.html") if path.name != "login.html"]
        for path in pages:
            parser = _Parser()
            parser.feed(path.read_text(encoding="utf-8"))
            shell_assets = [asset for asset in parser.assets if "aiops-shell.js" in asset]
            self.assertEqual([expected], shell_assets, path.name)

    def test_pages_share_current_navigation_styles(self):
        expected = "./css/aiops.css?v=20261020_3"
        for path in AIOPS_ROOT.glob("*.html"):
            parser = _Parser()
            parser.feed(path.read_text(encoding="utf-8"))
            style_assets = [asset for asset in parser.assets if "css/aiops.css" in asset]
            self.assertEqual([expected], style_assets, path.name)

    def test_javascript_syntax_and_public_boundary(self):
        scripts = list((AIOPS_ROOT / "js").glob("*.js"))
        self.assertEqual(18, len(scripts))
        source = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
        self.assertIn("/api/v1/apps/aiops", source)
        self.assertNotIn("/internal/v1", source)
        for script in scripts:
            result = subprocess.run(
                ["node", "--check", str(script)],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_dashboard_is_the_action_first_aiops_workbench(self):
        page = (AIOPS_ROOT / "dashboard.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-dashboard.js").read_text(
            encoding="utf-8"
        )
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('["chat", "Agent 工作台", "agents"]', shell)
        self.assertIn('["dashboard", "运维总览"]', shell)
        self.assertNotIn('["work-items", "工作项中心"]', shell)
        self.assertLess(
            shell.index('["business-systems", "业务系统"]'),
            shell.index('["diagnostic-sources", "监控源"]'),
        )
        self.assertNotIn('["targets", "成员 Target"]', shell)
        self.assertNotIn('["fleet", "库群总览"]', shell)
        self.assertIn('href="./${id}.html"', shell)
        self.assertIn("location.replace('./dashboard.html')", (
            AIOPS_ROOT / "login.html"
        ).read_text(encoding="utf-8"))
        self.assertIn("/dashboard", script)
        self.assertIn("/reports?limit=1", script)
        self.assertIn("行动优先工作台", page)
        self.assertIn("工作项是唯一承载责任人、SLA、处理和验证的对象", page)
        self.assertNotIn("ops-action-scope", page)
        self.assertNotIn("管理员可在筛选中切换全部工作项或异常归属", page)
        self.assertIn("以下是系统观测，不代表已形成个人责任", page)
        self.assertIn("需关注目标", page)
        self.assertIn("最新正式报告", page)
        self.assertNotIn("优先处理队列", page)
        self.assertIn('UNKNOWN: "无有效数据"', script)
        self.assertIn('STALE: "数据过期"', script)
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
        self.assertIn('baseline: state.baseline', script)
        self.assertNotIn("compare_source_id", script)
        self.assertIn("/targets/${encodeURIComponent(state.selected[0])}/monitoring/view?${params}", script)
        self.assertIn("source_display_name", script)
        self.assertIn('id="monitoring-baseline"', page)
        self.assertIn("过去 1 天平均值", page)
        self.assertIn("过去 1 周平均值", page)
        self.assertIn("过去 1 月平均值", page)
        self.assertNotIn('id="monitoring-compare-source"', page)
        self.assertNotIn("monitoring-query-drawer", page)
        self.assertIn("aiops.css?v=20261020_3", page)
        self.assertIn("aiops-monitoring.js?v=20261019_1", page)
        self.assertIn('class="ops-main ops-special-page ops-monitoring-main"', page)
        self.assertIn('id="monitoring-workspace" class="ops-special-workspace ops-monitoring-workspace"', page)
        self.assertIn('id="monitoring-rail-toggle"', page)
        self.assertIn('railCollapsed: false', script)
        self.assertIn('setRailCollapsed(!state.railCollapsed)', script)
        self.assertIn('percent: "%"', script)
        self.assertIn('return `${text} %`', script)
        self.assertIn('transactions_per_second: "次/秒"', script)
        self.assertIn('bytes: "GB"', script)
        self.assertIn('return `${gigabytes.toLocaleString', script)
        self.assertIn('return "<0.01 GB"', script)
        self.assertIn("function formatByteRate(value)", script)
        self.assertIn('unit: "MB/秒"', script)
        self.assertIn('if (unit === "bytes_per_second") return formatByteRate(number)', script)
        self.assertIn('bytes_per_second: "B/秒（自适应）"', script)
        self.assertNotIn('<span>指标 ${esc(panel.metric_code)}</span>', script)
        self.assertIn('<span>单位 ${esc(unitLabel(panel.unit) || "无量纲")}</span>', script)
        self.assertIn('<span>覆盖率 ${(averageCoverage * 100).toFixed(1)} %</span>', script)
        self.assertIn("const tooltipSeriesLabel = (series)", script)
        self.assertIn("item ? tooltipSeriesLabel(item) : entry.seriesName", script)
        self.assertIn('sourceNames ? `来源：${esc(sourceNames)}` : ""', script)
        self.assertIn("axisValueLabel(value, panel.unit)", script)
        self.assertIn('unit === "state"', script)
        self.assertIn("deltaText(series.points, panel.unit)", script)
        for metric_code in (
            "db.availability",
            "mysql.availability",
            "postgresql.availability",
        ):
            self.assertNotIn(f'"{metric_code}"', script)
        self.assertIn('panel.semantic_role !== "AVAILABILITY"', script)
        self.assertIn("item.is_default", script)
        self.assertIn("function gapIdentity(gap)", script)
        self.assertIn("if (seen.has(identity)) return false", script)
        self.assertIn("state.view.panels || []).slice(0, 6)", script)
        self.assertNotIn("const importantMetrics", script)
        self.assertIn("capacityMetricRoles", script)
        self.assertIn("function mergeCapacityPanels", script)
        self.assertIn('visualization: "CAPACITY_BAR"', script)
        self.assertIn("function capacityRows(panel)", script)
        self.assertIn("function baselinePresentation(panel)", script)
        self.assertIn("function conclusionTitle(panel", script)
        self.assertIn("function cardSubtitle(panel", script)
        self.assertNotIn("<p>${esc(panel.description)}</p>", script)
        self.assertIn("<p>${esc(cardSubtitle(panel, toneValue))}</p>", script)
        self.assertIn("当前值处于正常区间", script)
        self.assertIn("function thresholdPresentation(panel)", script)
        self.assertIn("ops-monitoring-current-value", script)
        self.assertIn("ops-monitoring-baseline", script)
        self.assertIn("ops-monitoring-threshold-rail", script)
        self.assertIn("ops-capacity-threshold-rail", script)
        self.assertIn("未提供容量上限", script)
        self.assertIn("柱长按当前最大逻辑占用归一", script)
        self.assertIn("不表示使用率", script)
        self.assertIn("__capacity_scope", script)
        self.assertIn("host.filesystem.", script)
        self.assertIn("ops-capacity-bar-shell-absolute", script)
        self.assertIn("database.storage.capacity-trend", script)
        self.assertIn("数据库逻辑占用趋势", script)
        self.assertIn("数据库容量使用率趋势", script)
        stylesheet = (AIOPS_ROOT / "css" / "aiops.css").read_text(encoding="utf-8")
        self.assertNotIn(".ops-monitoring-panel.is-capacity-comparison{grid-column:1/-1}", stylesheet)
        self.assertIn(".ops-monitoring-panel.is-capacity-comparison{display:flex", stylesheet)
        self.assertIn("grid-column:auto", stylesheet)
        self.assertIn(".ops-capacity-overview{height:310px;overflow:auto}", stylesheet)
        self.assertIn("使用率趋势", script)
        self.assertIn("capacity_warning_percent", script)
        self.assertIn("capacity_critical_percent", script)
        self.assertIn('STORAGE_USED: "已用容量"', script)
        self.assertIn('DATABASE_SIZE: "数据库容量"', script)
        self.assertIn("const rows = capacityRows(panel)", script)
        self.assertNotIn('id="monitoring-profile"', page)
        self.assertNotIn("function renderCapacityTable", script)
        self.assertIn('class="ops-monitoring-instance-list" role="radiogroup" aria-label="数据库资源单选"', page)
        self.assertIn('id="monitoring-instance-drawer" class="ops-monitoring-instance-drawer"', page)
        self.assertIn('id="open-monitoring-instance-drawer"', page)
        self.assertIn('class="ops-monitoring-target-option ${selected ? "is-selected" : ""}"', script)
        self.assertIn('document.getElementById("monitoring-current-instance").textContent', script)
        self.assertIn('closeInstanceDrawer();', script)
        self.assertIn("function targetPickerStatus(instance, result)", script)
        self.assertIn('instance.monitoring_readiness === "READY" ? "监控就绪"', script)
        self.assertNotIn('selected ? "待查询" : "未选择"', script)
        self.assertIn(".ops-monitoring-target-option.is-selected", (
            AIOPS_ROOT / "css" / "aiops.css"
        ).read_text(encoding="utf-8"))
        self.assertNotIn("实例状态矩阵", page)
        self.assertNotIn("monitoring-status-body", script)
        self.assertIn('type="radio" name="monitoring_instance"', script)
        self.assertIn("state.selected.length === 1", script)
        self.assertIn("void loadView()", script)
        self.assertIn('/targets/${encodeURIComponent(targetId)}/monitoring/profiles', script)
        self.assertNotIn('/sources/${encodeURIComponent(state.sourceId)}/profiles', script)
        self.assertIn("void selectInstance(input.dataset.instanceId)", script)
        self.assertIn("void selectInstance(instanceId)", script)
        self.assertNotIn("setTimeout(loadView, 250)", script)
        self.assertNotIn("monitoring-select-filtered", page)
        self.assertNotIn("monitoring-clear-selection", page)
        self.assertIn("state.controller?.abort()", script)
        self.assertIn("state.contextController?.abort()", script)
        self.assertIn("connectNulls: false", script)
        self.assertIn("查看数据", script)
        self.assertIn('id="monitoring-data-drawer"', page)
        self.assertIn('class="ops-monitoring-data-drawer"', page)
        self.assertIn('aria-labelledby="monitoring-data-drawer-title"', page)
        self.assertIn("当前值与区间概览", page)
        self.assertIn("区间统计", page)
        self.assertIn("原始数据仅用于核对证据", page)
        self.assertIn("seriesInsightsHtml(panel)", script)
        self.assertIn("intervalTableHtml(panel)", script)
        self.assertIn("numericValue(previousPoint.value) === numericValue(point.value)", script)
        self.assertIn("if (runs.length <= 12) return runs", script)
        self.assertIn("Math.ceil(points.length / 12)", script)
        self.assertIn("openDataDrawer(panel, button)", script)
        self.assertIn("drawer.showModal()", script)
        self.assertNotIn("data-panel-data", script)
        self.assertNotIn("收起数据", script)
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

    def test_work_items_are_a_compact_operational_queue(self):
        page = (AIOPS_ROOT / "dashboard.html").read_text(encoding="utf-8")
        detail = (AIOPS_ROOT / "work-item-detail.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-work-items.js").read_text(
            encoding="utf-8"
        )
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('["chat", "Agent 工作台", "agents"]', shell)
        self.assertIn('["dashboard", "运维总览"]', shell)
        self.assertNotIn('["work-items", "工作项中心"]', shell)
        self.assertFalse((AIOPS_ROOT / "work-items.html").exists())
        self.assertIn("处理队列", page)
        self.assertIn("解决 SLA", page)
        self.assertIn('class="ops-panel ops-work-item-queue-panel"', page)
        self.assertIn('class="ops-panel ops-work-item-filter-panel"', page)
        self.assertIn("选择队列可快速聚焦需要立即处理的事项", page)
        self.assertIn('aria-pressed="${queueIsActive(filter)}"', script)
        self.assertIn("等待责任组内 DBA 领取", script)
        self.assertIn(".ops-work-item-filters>div{grid-column:-2/-1}", (
            AIOPS_ROOT / "css" / "aiops.css"
        ).read_text(encoding="utf-8"))
        self.assertIn('name="target_id"', page)
        self.assertIn("责任范围内全部实例", page)
        self.assertIn("事实与发生记录", detail)
        self.assertIn("关联资源", detail)
        self.assertIn("活动时间线", detail)
        self.assertIn('class="ops-button ops-back-link"', detail)
        self.assertIn("返回运维总览", detail)
        self.assertIn('document.body.dataset.page === "dashboard"', script)
        self.assertIn("./dashboard.html?visibility=", script)
        self.assertIn("/api/v1/apps/aiops/work-items", script)
        self.assertIn("loadScopedTargets", script)
        self.assertIn("target.responsibility_group_id", script)
        self.assertNotIn("agent.default_responsibility_group_id", script)
        self.assertIn("userGroupIds", script)
        self.assertIn("target_id: values.target_id", script)
        self.assertNotIn("work-item-agent-filter", page)
        self.assertNotIn("来源 Agent / 数据库实例", page)
        self.assertIn('PENDING_VERIFICATION: "处理完成"', script)
        self.assertIn('data-handle-id', script)
        self.assertIn('data-assignment-id', script)
        self.assertIn('id="work-item-processing-dialog"', page)
        self.assertIn('id="work-item-assignment-dialog"', page)
        self.assertIn("标记工作完成", page)
        self.assertIn("确认分派", page)
        self.assertIn('<option value="FIXED">已修复</option>', page)
        self.assertIn('<option value="FALSE_POSITIVE">误报</option>', page)
        self.assertIn('FIXED: "已修复"', script)
        self.assertIn('text(item.resolution_code)', script)
        self.assertIn(':complete', script)
        self.assertIn('id="work-item-alert-context"', page)
        self.assertIn("错误 / 告警信息", page)
        self.assertIn('id="work-item-agent-diagnosis"', page)
        self.assertIn("初步诊断结果", page)
        self.assertIn('id="work-item-continue-diagnosis"', page)
        self.assertIn("继续深入诊断", page)
        self.assertIn("loadProcessingContext", script)
        self.assertIn("/runs/${encodeURIComponent(runId)}/result", script)
        self.assertIn("/situations/${encodeURIComponent(situationId)}", script)
        self.assertIn("source_run_id", script)
        self.assertIn("source_situation_id", script)
        self.assertIn("`${rootApi}/conversations`", script)
        self.assertIn("Idempotency-Key", script)
        self.assertIn("./chat.html?conversation=", script)
        self.assertIn("../shared/kbot-markdown.js", page)
        self.assertIn('member.status === "ACTIVE"', script)
        self.assertIn('member.user_id) !== String(state.access.user_id)', script)
        self.assertIn('`${groupApi}/${groupId}/member-candidates`', script)
        self.assertIn('member.display_name || member.username || member.user_id', script)
        self.assertNotIn('id="work-item-assignment-panel"', detail)
        self.assertNotIn('id="work-item-handle-panel"', detail)
        self.assertNotIn("loadAssignmentMembers", script)
        self.assertNotIn(":return-to-group", script)
        self.assertNotIn('id="work-item-transition"', detail)
        self.assertNotIn('/transitions', script)
        self.assertNotIn("fake", script.lower())

    def test_responsibility_groups_and_default_routing_are_manageable(self):
        page = (AIOPS_ROOT / "responsibility-groups.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-responsibility-groups.js").read_text(encoding="utf-8")
        agents = (AIOPS_ROOT / "agents.html").read_text(encoding="utf-8")
        agent_script = (AIOPS_ROOT / "js" / "aiops-agents.js").read_text(encoding="utf-8")
        targets = (AIOPS_ROOT / "targets.html").read_text(encoding="utf-8")
        target_script = (AIOPS_ROOT / "js" / "aiops-targets.js").read_text(encoding="utf-8")
        self.assertIn("选择人员", page)
        self.assertIn("不修改该用户的 AIOps 角色或权限", page)
        self.assertNotIn("组内角色", page)
        self.assertIn("active_work_item_count", script)
        table = page.split('<table class="ops-table ops-responsibility-group-table">', 1)[1].split("</table>", 1)[0]
        self.assertNotIn("绑定 Agent", table)
        self.assertNotIn("关联 Target", table)
        self.assertNotIn("未领取", table)
        self.assertNotIn("unassigned_work_item_count", script)
        self.assertIn("Target 的责任归属在运维目标中维护", page)
        self.assertIn('<dialog id="group-editor-dialog"', page)
        self.assertIn('<dialog id="group-members-dialog"', page)
        self.assertIn('<dialog id="group-confirm-dialog"', page)
        self.assertIn('id="group-create-open"', page)
        self.assertNotIn('id="group-create"', page)
        self.assertNotIn("ops-work-item-detail-grid", page)
        self.assertIn("data-edit-group", script)
        self.assertIn("data-delete-group", script)
        self.assertIn("askConfirmation", script)
        self.assertIn('status: "INACTIVE"', script)
        self.assertIn('filter((item) => item.status === "ACTIVE")', script)
        self.assertNotIn("group-editor-status-field", page)
        self.assertNotIn("payload.status", script)
        self.assertIn("/member-candidates", script)
        self.assertIn("/members/${encodeURIComponent", script)
        self.assertIn('body: JSON.stringify({ member_role: "MEMBER" })', script)
        self.assertIn("setLead", script)
        self.assertIn("void addMember(event.currentTarget)", script)
        self.assertNotIn("event.currentTarget.reset()", script)
        main_api = (ROOT / "services" / "main_api" / "src" / "main_api" / "api" / "ops.py").read_text(encoding="utf-8")
        candidates = main_api.split("async def _eligible_aiops_group_members", 1)[1].split("async def _require_eligible_aiops_group_member", 1)[0]
        self.assertIn('"user_id": str(row["user_id"])', candidates)
        self.assertNotIn('"roles"', candidates)
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(encoding="utf-8")
        business_groups, configuration_governance = shell.split('["configuration-governance", "配置与治理"', 1)
        self.assertNotIn('["responsibility-groups", "责任组"]', business_groups)
        self.assertIn('["responsibility-groups", "责任组"]', configuration_governance)
        self.assertNotIn("responsibility_group_id", agents)
        self.assertNotIn("responsibility_group_id", agent_script)
        self.assertIn('name="responsibility_group_id"', targets)
        self.assertIn("responsibility_group_id", target_script)
        self.assertNotIn("default_responsibility_group_id", targets)
        self.assertNotIn("default_responsibility_group_id", target_script)

    def test_chat_code_copy_supports_insecure_http_context(self):
        renderer = (ROOT / "ui" / "shared" / "kbot-markdown.js").read_text(
            encoding="utf-8"
        )
        chat = (AIOPS_ROOT / "chat.html").read_text(encoding="utf-8")
        self.assertIn("navigator.clipboard?.writeText", renderer)
        self.assertIn("window.isSecureContext", renderer)
        self.assertIn('document.execCommand("copy")', renderer)
        self.assertIn("kbot-markdown.js?v=20260827_1", chat)

    def test_chat_uses_the_monitoring_workspace_bone_and_starts_collapsed(self):
        chat = (AIOPS_ROOT / "chat.html").read_text(encoding="utf-8")
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(encoding="utf-8")
        stylesheet = (AIOPS_ROOT / "css" / "aiops.css").read_text(encoding="utf-8")
        self.assertIn('id="chat-workspace" class="ops-special-workspace ops-workspace is-rail-collapsed"', chat)
        self.assertIn('id="chat-rail-toggle"', chat)
        self.assertIn('workspace.classList.toggle("is-rail-collapsed", collapsed)', workspace)
        self.assertIn(".ops-special-workspace.is-rail-collapsed", stylesheet)
        self.assertIn("grid-template-columns:238px minmax(0,1fr)", stylesheet)

    def test_chat_lists_all_user_sessions_and_pins_deep_link(self):
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("function mergePinnedConversation", workspace)
        self.assertIn("`${api}/conversations?limit=50`", workspace)
        self.assertNotIn(
            "`${api}/conversations?agent_id=${encodeURIComponent(selectedAgent)}&target_id=${encodeURIComponent(selectedTarget)}`",
            workspace,
        )
        self.assertIn(
            "renderConversationList(mergePinnedConversation(rows, state.conversation))",
            workspace,
        )
        self.assertIn("还没有诊断会话", workspace)
        self.assertIn("也可以从左侧打开历史会话", workspace)
        self.assertNotIn("才会显示对应的会话历史", workspace)

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

    def test_chat_exposes_capability_aware_conversation_starters(self):
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
        versions = {
            "chat": "20261020_4",
            "situations": "20261020_4",
            "inspections": "20261020_4",
        }
        for page_name, version in versions.items():
            page = (AIOPS_ROOT / f"{page_name}.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("workspaces.css?v=20261020_4", page)
            self.assertIn("aiops-shell.js?v=20261023_1", page)
            self.assertIn(f"aiops-workspaces.js?v={version}", page)
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
        main_api = (
            ROOT / "services/main_api/src/main_api/api/aiops_app.py"
        ).read_text(encoding="utf-8")
        client = (
            ROOT / "packages/platform_clients/src/platform_clients/aiops.py"
        ).read_text(encoding="utf-8")
        policy = (
            ROOT / "packages/platform_core/src/platform_core/authorization/policy.py"
        ).read_text(encoding="utf-8")
        self.assertIn("REPORT_ARTIFACT_LINKS", workspace)
        self.assertIn("data-download-report-artifact", workspace)
        self.assertIn("report.artifact_id", workspace)
        self.assertIn(
            "/artifacts/${artifactId}/content",
            workspace,
        )
        self.assertNotIn("/workload-reports/", workspace)
        self.assertNotIn("data-download-workload-report", workspace)
        self.assertIn('"*/*"', workspace)
        artifact_route = (
            "/conversations/{conversation_id}/turns/{turn_id}/"
            "artifacts/{artifact_id}/content"
        )
        self.assertIn(artifact_route, main_api)
        self.assertIn("download_conversation_artifact", client)
        self.assertIn("/artifacts/{artifact_id}/content", client)
        self.assertIn("/artifacts/[0-9a-fA-F-]{36}/content", policy)
        self.assertNotIn("/workload-reports/", main_api)
        self.assertNotIn("download_workload_report", client)
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
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(encoding="utf-8")
        self.assertIn('id="manual-form"', page)
        self.assertIn("<h1>运维知识</h1>", page)
        self.assertIn('class="ops-table knowledge-table"', page)
        self.assertIn('id="knowledge-search"', page)
        self.assertIn('id="knowledge-status"', page)
        self.assertIn('class="knowledge-drawer knowledge-upload-drawer"', page)
        self.assertIn('class="knowledge-drawer knowledge-detail-drawer"', page)
        self.assertIn("/operations-knowledge", script)
        self.assertIn('groupingMode: "EACH_FILE"', script)
        self.assertIn('"SINGLE_BUNDLE"', script)
        self.assertIn('body.set("grouping_mode", state.groupingMode)', script)
        self.assertIn('body.set("knowledge_base_id", selectedKnowledgeBaseId())', script)
        self.assertIn('body.set("metadata", JSON.stringify(metadata))', script)
        self.assertIn('`${api}/uploads`', script)
        self.assertIn("X-Expected-Asset-Row-Version", script)
        self.assertIn('data-review="publish"', script)
        self.assertNotIn('id="knowledge-overview"', page)
        self.assertNotIn('class="knowledge-cards"', page)
        self.assertNotIn('id="search-form"', page)
        self.assertNotIn('id="knowledge-chat-form"', page)
        self.assertNotIn("loadAnswerConversations", script)
        self.assertIn('"operations-knowledge": "aiops:knowledge_manage"', shell)
        self.assertIn('["operations-knowledge", "运维知识"]', shell)
        self.assertIn('["knowledge-bases", "运维知识库"]', shell)
        self.assertNotIn("operations-knowledge-chat", shell)
        self.assertFalse((AIOPS_ROOT / "operations-knowledge-chat.html").exists())
        self.assertFalse((AIOPS_ROOT / "js" / "aiops-operations-knowledge-chat.js").exists())
        self.assertNotIn("Knowledge Core", page)
        self.assertNotIn("Collection", page)
        self.assertNotIn("Embedding", page)
        self.assertNotIn("/api/v1/knowledge", script)

    def test_knowledge_base_editor_manages_model_bindings(self):
        page = (AIOPS_ROOT / "knowledge-bases.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-knowledge-bases.js").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("知识问答 Agent 绑定", page)
        self.assertNotIn("knowledge-answering/binding", script)
        for role in (
            "parser_ocr", "parser_vlm", "parser_llm", "retrieval_llm",
            "embedding", "visual_embedding",
        ):
            self.assertIn(f'name="{role}"', page)
            self.assertIn(role, script)
        self.assertIn('/api/v1/model-catalog', script)
        self.assertIn('/models`)', script)
        self.assertIn('const payload = { display_name:', script)

    def test_knowledge_answering_is_part_of_the_unified_agent(self):
        page = (AIOPS_ROOT / "agents.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-agents.js").read_text(
            encoding="utf-8"
        )
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('value="KNOWLEDGE_QA"', script)
        self.assertIn("绑定运维知识库", page)
        self.assertIn("只检索已审批版本", page)
        self.assertNotIn("knowledge-answering-agent", shell)
        self.assertFalse((AIOPS_ROOT / "knowledge-answering-agent.html").exists())
        self.assertFalse((AIOPS_ROOT / "js" / "aiops-knowledge-answering-agent.js").exists())

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
        self.assertIn('button.textContent = wasEditing ? "保存中…" : "创建中…"', agent)
        self.assertIn("shell.toast(error.message)", agent)
        self.assertIn('showResult(error.message, "bad")', agent)
        self.assertIn("controlled_action_execution: values(editing?.controlled_action_execution)", agent)
        self.assertIn("新增授权需在资源治理中完成", agent)
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
        forms = (AIOPS_ROOT / "css" / "aiops-forms.css").read_text(
            encoding="utf-8"
        )
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
        version_patch = script.index(
            "body: JSON.stringify({ version_code: version.value })"
        )
        credential_rotation = script.index("diagnostic-credential:rotate")
        self.assertLess(version_patch, credential_rotation)
        self.assertIn("version.value !== editingTarget.version_code", script)
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
        self.assertIn(".ops-dialog .ops-field{align-content:start}", forms)
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
        editor_page = (AIOPS_ROOT / "targets.html").read_text(encoding="utf-8")
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
        self.assertIn("管理信息", overview)
        self.assertIn("target.target_id", overview)
        self.assertIn("target.created_at", overview)
        self.assertIn("target.updated_by", overview)
        self.assertNotIn("schema_version", overview)
        self.assertNotIn("JSON.stringify", overview)
        self.assertNotIn('id="target-monitor-binding-form"', page)
        self.assertNotIn('id="target-recovery-profile-form"', page)
        self.assertNotIn('id="target-facts-form"', page)
        self.assertNotIn('id="target-recovery-profile-form"', editor_page)
        self.assertIn('id="target-facts-form"', editor_page)

    def test_target_editor_uses_three_steps_without_undesignated_protection_page(self):
        page = (AIOPS_ROOT / "targets.html").read_text(encoding="utf-8")
        targets = (AIOPS_ROOT / "js" / "aiops-targets.js").read_text(encoding="utf-8")
        systems = (AIOPS_ROOT / "js" / "aiops-business-system-workspace.js").read_text(encoding="utf-8")
        workspace = (AIOPS_ROOT / "business-system-workspace.html").read_text(encoding="utf-8")
        identity = page.index('data-target-wizard-step="identity"')
        monitoring = page.index('data-target-wizard-step="monitoring"')
        access = page.index('data-target-wizard-step="access"')
        self.assertLess(identity, monitoring)
        self.assertLess(monitoring, access)
        self.assertIn("目标身份与责任归属", page)
        self.assertIn('name="responsibility_group_id" required', page)
        self.assertNotIn('name="backup_policy_enabled"', page)
        self.assertNotIn("saveTargetRecovery", targets)
        self.assertFalse((AIOPS_ROOT / "data-protection.html").exists())
        self.assertFalse((AIOPS_ROOT / "js" / "aiops-data-protection.js").exists())
        self.assertIn('["DATABASE", "数据库"', systems)
        self.assertIn('["NETWORK_DEVICE", "网络设备"', systems)
        self.assertNotIn("REUSE", systems)
        self.assertIn("配置${esc(typeName(catalogType))}", systems)
        self.assertNotIn("添加 Target", systems)
        self.assertNotIn('/targets?limit=200', systems)
        self.assertIn('request("/business-system-drafts"', systems)
        self.assertIn(":commit", systems)
        self.assertNotIn('system-protection-form', systems)
        self.assertIn("← 返回业务系统", workspace)
        self.assertIn("系统构成", workspace)
        self.assertIn("business-topology-property", systems)
        self.assertIn("点击画布中的资源或关系", systems)
        self.assertIn("data-topology-relation", systems)
        self.assertIn("function renderTopology()", systems)
        self.assertIn('business-system-workspace.html?id=', targets)
        self.assertIn('const searchParams = new URLSearchParams(location.search)', systems)
        self.assertIn('searchParams.get("id")', systems)

    def test_business_system_workspace_uses_compact_header_and_flow_actions(self):
        systems = (AIOPS_ROOT / "js" / "aiops-business-system-workspace.js").read_text(
            encoding="utf-8"
        )
        workspace = (AIOPS_ROOT / "business-system-workspace.html").read_text(
            encoding="utf-8"
        )
        forms = (AIOPS_ROOT / "css" / "aiops-forms.css").read_text(
            encoding="utf-8"
        )
        shell = (AIOPS_ROOT / "js" / "aiops-shell.js").read_text(
            encoding="utf-8"
        )

        actions = workspace.split(
            'class="business-workspace-actions"', 1
        )[1].split("</footer>", 1)[0]
        self.assertLess(actions.index("返回业务系统"), actions.index("保存草稿"))
        self.assertLess(actions.index("保存草稿"), actions.index("上一步"))
        self.assertLess(actions.index("上一步"), actions.index("下一步"))
        self.assertNotIn(".business-workspace-actions{position:fixed", forms)
        self.assertIn('id="workspace-code" class="business-workspace-code"', workspace)
        self.assertIn(
            'getElementById("workspace-code").textContent = value.system_code',
            systems,
        )
        self.assertNotIn("`编辑 · ${value.display_name}`", systems)
        self.assertIn('.business-form-grid label{align-content:start}', forms)
        self.assertIn('errors.push("业务信息：请填写业务系统名称。")', systems)
        self.assertIn('errors.push("业务信息：请选择业务责任组。")', systems)
        self.assertNotIn('message("请填写业务系统名称并选择业务责任组。"', systems)
        self.assertNotIn("描述业务影响和整体责任，不在这里填写数据库部署形态。", systems)
        self.assertNotIn("保存后按 SYS-xxxxxx 规则自动生成", systems)
        self.assertNotIn("编码由租户规则生成，创建后不可修改。", systems)
        self.assertIn('role="tablist" aria-label="系统构成"', systems)
        self.assertIn('class="business-resource-card"', systems)
        self.assertIn('id="business-resource-drawer"', workspace)
        self.assertIn(".business-resource-card-grid", forms)
        self.assertIn("overflow-y:auto", forms)
        self.assertIn('activeLayer === "DATA"', systems)
        self.assertIn("catalogTypesByLayer[activeLayer].includes(item[0])", systems)
        self.assertIn('DATA: ["DATABASE", "CACHE"]', systems)
        self.assertNotIn('["DATA_WAREHOUSE", "数据仓库"', systems)
        self.assertIn("功能未完成，敬请期待", systems)
        self.assertNotIn("关联已有数据库", systems)
        self.assertNotIn("复用已有数据库", systems)
        self.assertIn('shell.toast("至少有两个资源后才能建立关系。")', systems)
        self.assertNotIn('message("至少有两个资源后才能建立关系。"', systems)
        self.assertIn('return `${label}：未配置数据库监控 Target。`', systems)
        self.assertIn('return `${label}：未配置主机监控 Target。`', systems)
        self.assertIn('id="resource-connection-result"', systems)
        self.assertIn('"数据库连接测试通过；正式提交前不会创建资源。"', systems)
        self.assertIn("连接测试为可选项，不影响保存", systems)
        self.assertNotIn("function validateStep()", systems)
        self.assertIn('errors.push("系统构成：至少接入一个资源。")', systems)
        self.assertIn("function publicationErrors()", systems)
        self.assertIn('request("/business-system-drafts"', systems)
        self.assertNotIn("sessionStorage", systems)
        publication_validation = systems.split("function publicationErrors()", 1)[1].split(
            "function systemPayload", 1
        )[0]
        self.assertIn("resourceConfigurationError(item)", publication_validation)
        self.assertNotIn("!item.saved", publication_validation)
        self.assertNotIn("完成所有数据库的连接验证", systems)
        self.assertNotIn("function message(", systems)
        self.assertNotIn('id="workspace-result"', workspace)
        self.assertNotIn("business-validation-list", systems)
        self.assertIn('id="workspace-leave-dialog"', workspace)
        self.assertIn('id="workspace-leave-form"', workspace)
        self.assertIn('document.addEventListener("click"', systems)
        self.assertIn('addEventListener("beforeunload"', systems)
        self.assertNotIn("架构层是当前系统中的展示位置，不会改变资源类型。", systems)
        self.assertNotIn('name="architecture_layer_key"', systems)
        self.assertIn("item.architecture_layer_key = activeLayer;", systems)
        self.assertNotIn("本系统责任覆盖", systems)
        self.assertNotIn("关联只建立业务系统归属，不会修改数据库名称、技术责任组或监控配置。", systems)
        self.assertNotIn("REUSE", systems)
        self.assertIn("deployment_type", systems)
        self.assertIn("rac_node_number", systems)
        self.assertIn("locator_key", systems)
        single_connection = systems.split(
            'if (item.mode === "NEW" && database && !rac)', 1
        )[1].split('if (item.mode === "NEW" && database && rac)', 1)[0]
        first_row = [
            single_connection.index("数据库地址"),
            single_connection.index("端口"),
            single_connection.index("Service Name"),
            single_connection.index("容器范围"),
        ]
        self.assertEqual(first_row, sorted(first_row))
        for group in ("连接端点", "数据库访问", "监控映射"):
            self.assertIn(f"<p>{group}</p>", single_connection)
        self.assertIn("business-connection-endpoint-row", single_connection)
        self.assertIn("business-connection-access-row", single_connection)
        self.assertIn("business-connection-monitoring-row", single_connection)
        self.assertIn('<span>TLS</span><select name="tls_enabled">', single_connection)
        self.assertIn('[["false", "不启用"], ["true", "启用"]]', single_connection)
        self.assertIn('database.oracle_scope === "PDB"', single_connection)
        self.assertIn("<span>PDB 名称</span>", single_connection)
        second_and_third_rows = [
            single_connection.index("诊断用户名"),
            single_connection.index("诊断密码"),
            single_connection.index("监控源（必选）"),
        ]
        self.assertEqual(second_and_third_rows, sorted(second_and_third_rows))
        self.assertIn(".business-connection-endpoint-row.oracle{grid-template-columns:repeat(4,minmax(0,1fr))}", forms)
        self.assertIn(".business-connection-endpoint-row{grid-template-columns:2fr 1fr 3fr 1.3fr}", forms)
        self.assertIn(".business-connection-access-row.pdb,.business-connection-monitoring-row.discovery{grid-template-columns:repeat(3,minmax(0,1fr))}", forms)
        self.assertNotIn("discover-resources", systems)
        self.assertNotIn("event.target === drawer", systems)
        self.assertIn("function enforceDrawerClosePolicy()", shell)
        self.assertIn('["pointerdown", "mousedown", "click"]', shell)
        self.assertIn("event.stopImmediatePropagation()", shell)
        self.assertIn('document.addEventListener("cancel"', shell)
        self.assertIn("responsibility_group_id: null", systems)
        self.assertIn(".business-resource-drawer{", forms)
        self.assertIn("width:min(760px,100vw)", forms)

    def test_business_system_workspace_adds_oracle_rac_as_one_multi_node_group(self):
        systems = (AIOPS_ROOT / "js" / "aiops-business-system-workspace.js").read_text(
            encoding="utf-8"
        )
        forms = (AIOPS_ROOT / "css" / "aiops-forms.css").read_text(
            encoding="utf-8"
        )

        for contract in (
            "rac_group_id",
            "rac_cluster_name",
            "ensureRacGroup",
            "newRacNode",
            "addRacNode",
            "removeRacNode",
            "managedRacRelation",
            "isManagedRacRelation",
        ):
            self.assertIn(contract, systems)
        self.assertIn('relation_type_key: "ORACLE_RAC"', systems)
        self.assertIn('participant_role: "MEMBER"', systems)
        self.assertIn("if (racMembers(item).length < 2)", systems)
        self.assertIn("Oracle RAC 至少需要两个节点。", systems)
        self.assertIn("＋ 增加 RAC 节点", systems)
        self.assertIn("诊断用户名（所有节点共用）", systems)
        self.assertIn("诊断密码（所有节点共用）", systems)
        self.assertIn("监控源（所有节点共用）", systems)
        for label in ("Host", "Service Name", "数据库监控 Target", "主机监控 Target"):
            self.assertIn(f"<span>{label}</span>", systems)

        configuration_validation = systems.split("function resourceConfigurationError", 1)[1].split(
            "async function validateResource", 1
        )[0]
        validation = systems.split("async function validateResource", 1)[1].split(
            "function topologyBoardHtml", 1
        )[0]
        self.assertIn("for (const member of members)", validation)
        self.assertIn('await request("/targets/test-connection"', validation)
        self.assertIn("members.length < 2", configuration_validation)
        self.assertIn("new Set(entry[1]).size !== entry[1].length", configuration_validation)
        self.assertNotIn('["数据库监控 Target", members.map', configuration_validation)
        self.assertNotIn('uniqueFields.push(["主机监控 Target"', configuration_validation)
        self.assertNotIn('mapping_status !== "MAPPED"', systems)
        self.assertIn("（已配置，可复用）", systems)
        self.assertIn("members.forEach((member) => { member.saved = true; })", validation)
        self.assertIn(".business-rac-node-card", forms)

    def test_business_system_workspace_uses_explicit_database_tls_setting(self):
        systems = (AIOPS_ROOT / "js" / "aiops-business-system-workspace.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("tls_enabled: database.tls_enabled === true", systems)
        self.assertIn("tls_enabled: false", systems)
        self.assertIn('name === "tls_enabled" ? input.value === "true"', systems)
        self.assertIn("tls_enabled: !oracle && database.tls_enabled === true", systems)
        self.assertNotIn("tls_enabled: !oracle }", systems)
        self.assertIn("无法连接数据库，请检查地址、端口和 TLS 设置。", systems)

    def test_business_system_workspace_infers_database_relations_and_draws_topology(self):
        systems = (AIOPS_ROOT / "js" / "aiops-business-system-workspace.js").read_text(
            encoding="utf-8"
        )
        forms = (AIOPS_ROOT / "css" / "aiops-forms.css").read_text(
            encoding="utf-8"
        )

        for contract in (
            "syncInferredDatabaseRelations",
            "DATABASE_ROLE:${dbType}",
            "DATA_GUARD",
            "MYSQL_REPLICATION",
            "POSTGRES_STREAMING",
            "primaries.length !== 1 || !standbys.length",
            "auto_inferred",
            "isManagedRelation",
            "TOPOLOGY PREVIEW",
            "business-relation-flow",
            "business-relation-connector",
            "自动推导",
            "不会自动猜测",
        ):
            self.assertIn(contract, systems)
        self.assertIn('origin: "DISCOVERY"', systems)
        self.assertIn('inference_rule: "RAC_DEPLOYMENT"', systems)
        self.assertIn(".business-relation-flow", forms)
        self.assertIn(".business-standalone-map", forms)

    def test_aiops_importance_labels_increase_from_low_to_core(self):
        workspace = (AIOPS_ROOT / "js" / "aiops-business-system-workspace.js").read_text(
            encoding="utf-8"
        )
        systems = (AIOPS_ROOT / "js" / "aiops-business-systems.js").read_text(
            encoding="utf-8"
        )
        pages = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        targets = (AIOPS_ROOT / "targets.html").read_text(encoding="utf-8")

        for label in ["L1 低", "L2 一般", "L3 重要", "L4 关键", "L5 核心"]:
            self.assertIn(label, workspace)
        self.assertIn('{ 1: "低", 2: "一般", 3: "重要", 4: "关键", 5: "核心" }', systems)
        self.assertIn('{ 1: "低", 2: "一般", 3: "重要", 4: "关键", 5: "核心" }', pages)
        self.assertIn("L1 · 低", targets)
        self.assertIn("L5 · 核心", targets)
        self.assertNotIn("可按需忽略", targets)
        self.assertNotIn("L2 较低", workspace)
        self.assertNotIn("级别越高", workspace)
        self.assertNotIn("级别越高", targets)

    def test_business_system_directory_exposes_business_lifecycle_actions(self):
        page = (AIOPS_ROOT / "business-systems.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-business-systems.js").read_text(encoding="utf-8")

        for label in ("全部", "运行中", "草稿", "已下线"):
            self.assertIn(label, page + script)
        self.assertIn("下线原因", page)
        self.assertIn("删除草稿", script)
        self.assertIn("重新上线", script)
        self.assertIn("已接入的资源会保留，不会删除资源、拓扑或历史数据", script)
        self.assertIn('request("/business-system-drafts?limit=200")', script)
        self.assertIn('"未发布草稿"', script)
        self.assertIn("尚未发布的资源配置将被删除", script)
        self.assertNotIn("已停用", script)

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
        self.assertNotIn('id="target-recovery-drills-link"', target_page)
        self.assertIn('id="recovery-drill-form"', drill_page)
        self.assertIn('id="recovery-drill-history"', drill_page)
        self.assertIn("/recovery-drills", drill_script)
        self.assertNotIn("Oracle SCN", drill_page)
        self.assertNotIn("Resetlogs ID", drill_page)
        self.assertNotIn("Incarnation", drill_page)
        self.assertNotIn('name="evidence_kind"', drill_page)
        self.assertNotIn('name="evidence_reference"', drill_page)
        self.assertNotIn('name="evidence_hash"', drill_page)
        self.assertIn("recovery_marker: { kind: currentResource.db_type }", drill_script)
        self.assertIn("evidence: []", drill_script)
        self.assertIn("数据时间不能晚于模拟故障时间", drill_script)
        self.assertIn("验证完成时间不能早于模拟故障时间", drill_script)
        self.assertIn('query.get("work_item_id")', drill_script)
        self.assertIn("source_work_item_id: sourceWorkItemId || null", drill_script)
        self.assertIn("登记恢复演练", drill_page)
        self.assertIn("计划外演练记录已提交", drill_script)
        self.assertIn('["recovery-drills", "恢复演练"]', shell)
        systems = (AIOPS_ROOT / "js" / "aiops-business-systems.js").read_text(encoding="utf-8")
        self.assertNotIn("protection-profile", systems)
        self.assertIn("/business-systems?limit=200", systems)
        self.assertFalse((AIOPS_ROOT / "data-protection.html").exists())

    def test_target_edit_and_chat_own_target_fact_confirmation(self):
        page = (AIOPS_ROOT / "target-detail.html").read_text(encoding="utf-8")
        editor_page = (AIOPS_ROOT / "targets.html").read_text(encoding="utf-8")
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
        self.assertNotIn('id="target-facts-form"', page)
        self.assertIn('id="target-facts-form"', editor_page)
        self.assertIn("initializeTargetFacts", pages)
        self.assertIn("/facts", pages)
        self.assertIn("FACT_CONFIRMATION", workspace)
        self.assertIn("data-confirm-target-fact", workspace)
        self.assertIn("target-facts:confirm", workspace)
        self.assertIn(".ops-fact-confirmation", workspaces_css)
        self.assertIn(".target-facts-list", forms_css)
        self.assertIn(".target-facts-form", forms_css)
        self.assertNotIn(
            "command_preview",
            workspace.split("function factConfirmationHtml")[1].split(
                "function reportArtifactLinksHtml"
            )[0],
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
        self.assertIn('data-source-action="detail"', pages_script)
        self.assertIn('data-source-action="edit"', pages_script)
        self.assertIn('data-source-action="enable"', pages_script)
        self.assertIn('data-source-action="disable"', pages_script)
        self.assertIn('data-source-action="delete"', pages_script)
        self.assertIn('shell.toast("请先停用监控源，再执行删除")', pages_script)
        self.assertIn('method: "DELETE"', pages_script)
        self.assertIn('"If-Match": `"rv-${item.row_version}"`', pages_script)
        self.assertIn('"Idempotency-Key": KBotAIOpsAuth.uuid()', pages_script)
        self.assertIn("删除会撤销该监控源的访问凭据和 Webhook 凭据", pages_script)
        self.assertIn("如果仍有运维目标或运行历史引用它", pages_script)
        self.assertIn("connectivity_check_pending", pages_script)
        self.assertIn(
            'hasOwnProperty.call(item, "readonly_connection_enabled")',
            pages_script,
        )
        self.assertIn('data-target-action="connectivity"', pages_script)
        self.assertIn('data-target-action="detail"', pages_script)
        target_config = pages_script.split("targets:", 1)[1].split(
            '"diagnostic-sources":', 1
        )[0]
        self.assertNotIn("detail:", target_config)
        self.assertIn('data-target-action="enable"', pages_script)
        self.assertNotIn('data-target-action="delete"', pages_script)
        self.assertNotIn('shell.toast("请先停用运维目标，再执行删除")', pages_script)
        self.assertNotIn("删除会一并清理该 Target 的监控映射、数据库凭据", pages_script)
        self.assertNotIn(
            "KBotAIOpsAuth.request(`${appApi}/targets/${targetId}`",
            pages_script,
        )
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

    def test_target_create_and_edit_include_required_monitor_binding(self):
        page = (AIOPS_ROOT / "targets.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-targets.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="target-monitor-editor-source"', page)
        self.assertIn('id="discover-target-monitor-editor-labels"', page)
        self.assertIn('id="target-monitor-editor-host-candidates"', page)
        self.assertIn("主机 Label（Prometheus 必选）", page)
        self.assertIn("Target 必须至少绑定一条有效监控映射", page)
        self.assertIn("loadMonitorEditor()", script)
        self.assertIn("loadMonitorEditor(target.target_id)", script)
        self.assertIn("monitorSelectionIsValid()", script)
        self.assertIn("saveSelectedMonitorBinding(created)", script)
        self.assertIn("saveSelectedMonitorBinding(updated)", script)
        self.assertIn("monitor_host_candidate_ref", script)
        self.assertIn("host_candidate_ref", script)
        self.assertIn("binding.host_locator_hint", script)
        self.assertNotIn("binding.source_locator?.host_target_key", script)
        self.assertIn('data-monitor-binding-action="${action}"', script)
        self.assertIn("const refreshedTarget = await KBotAIOpsAuth.request", script)
        self.assertIn("editingTarget = refreshedTarget", script)

    def test_diagnostic_source_detail_is_operator_readable(self):
        page = (AIOPS_ROOT / "diagnostic-source-detail.html").read_text(
            encoding="utf-8"
        )
        script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        detail_renderer = script.split(
            "function sourceDetailHtml(source)", maxsplit=1
        )[1].split("async function initializeTargetMonitorMappings", maxsplit=1)[0]

        self.assertIn("aiops-forms.css", page)
        self.assertIn('class="ops-panel-head source-overview-head"', detail_renderer)
        for label in (
            "接入信息",
            "连接与健康",
            "凭据与 Webhook",
            "管理信息",
            "系统声明能力",
            "最近验证能力",
        ):
            self.assertIn(label, detail_renderer)
        self.assertIn('"metric.query_range": "查询时序指标"', detail_renderer)
        self.assertIn('"instance.discover": "发现监控实例"', detail_renderer)
        self.assertIn("具体监控 Label 与数据库对象的映射", detail_renderer)
        self.assertNotIn("JSON.stringify(source", detail_renderer)
        self.assertNotIn("Object.entries(source)", detail_renderer)

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
        self.assertIn('data-inspection-action="edit"', pages_script)
        self.assertIn('data-inspection-action="detail"', pages_script)
        self.assertIn('data-inspection-action="delete"', pages_script)
        self.assertIn('method: deleting ? "DELETE" : "POST"', pages_script)
        self.assertNotIn('"inspection-plans": globalThis.KBotAIOpsConfigurations', pages_script)

    def test_agent_form_owns_business_system_capability_and_knowledge_scope(self):
        page = (AIOPS_ROOT / "agents.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-agents.js").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('name="diagnostic_source_ids"', script)
        self.assertNotIn('name="target_ids"', script)
        self.assertNotIn('id="agent-targets"', page)
        self.assertIn("selectionSets.systems", script)
        self.assertIn('id="agent-business-system-summary"', page)
        self.assertIn('id="open-agent-system-picker"', page)
        self.assertIn("请至少选择一个业务系统", script)
        self.assertIn('id="agent-capabilities"', page)
        self.assertIn('name="capabilities"', script)
        self.assertIn('id="agent-knowledge-policy-fields"', page)
        self.assertIn('name="knowledge_enabled" type="checkbox" checked', page)
        self.assertIn("绑定运维知识库", page)
        self.assertIn('id="open-agent-knowledge-base-picker"', page)
        self.assertNotIn(
            'class="agent-compact-switch"><input name="knowledge_enabled"', page
        )
        self.assertIn('name="knowledge_source_kinds"', page)
        self.assertIn("selectionSets.tags", script)
        self.assertNotIn('name="allow_change_execution"', page)
        self.assertIn("controlled_action_execution:", script)
        self.assertNotIn('/action-catalog/', script)
        self.assertIn('name="auto_alert_enabled"', page)
        self.assertIn('name="auto_observe_min_target_level"', page)
        self.assertIn("auto_observe_min_target_level:", script)
        self.assertNotIn('name="diagnosis_model_id" required', page)
        self.assertNotIn('name="planner_model_id" required', page)
        self.assertNotIn('name="composer_model_id" required', page)
        self.assertNotIn('class="agent-advanced"', page)
        self.assertIn('data-agent-step="4"', page)
        self.assertIn('data-agent-step="5"', page)
        self.assertIn('<h3>模型与指令</h3>', page)
        self.assertIn("setStep(4)", script)
        self.assertIn("运行中的 Agent 必须配置", script)
        self.assertIn('planner_llm: form.elements.planner_model_id.value', script)
        self.assertIn('diagnosis_llm: form.elements.diagnosis_model_id.value', script)
        self.assertIn('`${api}/agent-model-catalog`', script)
        self.assertNotIn('models: modelId ? { diagnosis:', script)
        self.assertIn("自动触发不影响聊天、知识问答和报告查询", page)
        self.assertNotIn('id="agent-binding-summary"', page)
        self.assertIn('class="agent-step-panel"', page)
        self.assertIn('class="agent-picker-dialog"', page)
        self.assertNotIn("/source-bindings", script)
        self.assertNotIn("assertExistingBindings", script)
        self.assertNotIn("ensureSourceBindings", script)
        self.assertNotIn("source_locator_key", script)
        self.assertNotIn("data-loki-target-label", script)
        self.assertNotIn("data-prometheus-host-target", script)
        self.assertNotIn("scrollIntoView", script)
        self.assertNotIn('name="policy_id"', page)
        self.assertNotIn("max_risk_level", page)
        self.assertNotIn("allowed_action_types", page)

    def test_target_owns_monitor_source_label_mapping(self):
        page = (AIOPS_ROOT / "target-detail.html").read_text(encoding="utf-8")
        script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        editor_page = (AIOPS_ROOT / "targets.html").read_text(encoding="utf-8")
        editor_script = (AIOPS_ROOT / "js" / "aiops-targets.js").read_text(
            encoding="utf-8"
        )

        self.assertNotIn('id="target-monitor-binding-form"', page)
        self.assertIn("监控源与 Label 映射", page)
        self.assertIn("只读信息", page)
        self.assertIn("initializeTargetMonitorMappings", script)
        self.assertIn("binding.host_locator_hint", script)
        self.assertNotIn("binding.source_locator?.host_target_key", script)
        self.assertIn("/source-bindings", script)
        self.assertIn('id="target-monitor-editor-source"', editor_page)
        self.assertIn("target_key label 值", editor_script)
        self.assertIn("OEM Target Name", editor_script)
        self.assertIn("monitor_candidate_ref", editor_script)
        self.assertIn("monitor_host_candidate_ref", editor_script)
        self.assertIn("host_candidate_ref", editor_script)
        self.assertIn("/instance-discoveries", editor_script)
        self.assertIn("/instance-mappings", editor_script)
        self.assertIn("data-monitor-binding-delete", editor_script)

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
        navigation = shell.split("const pinnedPages", 1)[1].split("const iconPaths", 1)[0]
        expected = [
            '["chat", "Agent 工作台", "agents"]',
            '["situations-events", "态势与事件", "data"',
            '["inspections-drills", "巡检与演练", "audit"',
            '["reports-knowledge", "报告与知识", "knowledge"',
            '["configuration-governance", "配置与治理", "settings"',
            '["dashboard", "运维总览"]',
            '["monitoring", "实时监控"]', '["situations", "告警诊断"]',
            '["inspections", "日常巡检"]',
            '["recovery-drills", "恢复演练"]', '["reports", "运维报告"]',
            '["operations-knowledge", "运维知识"]',
            '["business-systems", "业务系统"]', '["diagnostic-sources", "监控源"]',
            '["responsibility-groups", "责任组"]', '["knowledge-bases", "运维知识库"]',
            '["agents", "Agent 管理"]', '["report-templates", "模板中心"]',
            '["runs", "运行记录"]',
        ]
        for item in expected:
            self.assertIn(item, navigation)
        self.assertNotIn('["resource-management", "资源管理"', navigation)
        self.assertNotIn('["operations-knowledge-chat", "知识问答"]', navigation)
        self.assertNotIn('["data-protection", "数据保护"]', navigation)
        self.assertNotIn('["knowledge-answering-agent", "知识问答 Agent"]', navigation)
        self.assertNotIn('["api-clients", "API 客户端"]', navigation)
        self.assertIn("bindNavigationGroups", shell)
        self.assertIn('?.key || ""', shell)
        self.assertNotIn('?.key || groups[0]?.key', shell)
        self.assertIn("const shouldOpen = candidate === group && !isOpen", shell)
        self.assertNotIn("items.querySelector", shell)
        stylesheet = (AIOPS_ROOT / "css" / "aiops.css").read_text(encoding="utf-8")
        self.assertIn(
            ".ops-nav-group>.ops-nav-items[hidden]{display:none!important}",
            stylesheet,
        )
        self.assertIn(".ops-nav-group.is-open", stylesheet)
        self.assertIn(".ops-nav-footer-mark", stylesheet)

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
        inspections = (AIOPS_ROOT / "inspections.html").read_text(
            encoding="utf-8"
        )
        plans = (AIOPS_ROOT / "inspection-plans.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('["chat", "Agent 工作台", "agents"]', shell)
        self.assertIn('["situations", "告警诊断"]', shell)
        self.assertIn('["inspections", "日常巡检"]', shell)
        self.assertNotIn('["inspection-plans", "巡检计划"]', shell)
        self.assertIn('["runs", "运行记录"]', shell)
        self.assertIn('["reports", "运维报告"]', shell)
        self.assertIn('["report-templates", "模板中心"]', shell)
        self.assertIn('aria-label="日常巡检视图"', inspections)
        self.assertIn('data-plan-manage hidden', inspections)
        self.assertIn('aria-current="page" href="./inspection-plans.html"', plans)
        self.assertIn("plan_display_name", workspace)
        self.assertIn("detail.runs || []", workspace)
        self.assertIn("inspection-run-selector", workspace)
        business_groups = shell.split('["configuration-governance", "配置与治理"', 1)[0]
        self.assertNotIn('["report-templates", "模板中心"]', business_groups)
        self.assertIn("source_run_id", workspace)
        self.assertIn("source_situation_id", workspace)
        self.assertIn("正在等待 Agent 自动诊断任务启动", workspace)
        self.assertIn("监控来源", workspace)
        self.assertIn("告警内容", workspace)
        self.assertIn('RESOLVED: "当前无活动告警"', workspace)
        self.assertIn("shell.badge(item.latest_status, situationStatusText(item.latest_status))", workspace)
        self.assertIn("shell.badge(detail.status, situationStatusText(detail.status))", workspace)
        self.assertNotIn("shell.badge(detail.status)} · ${esc(situationStatusText", workspace)
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
        self.assertNotIn('id="target-select"', (AIOPS_ROOT / "chat.html").read_text(encoding="utf-8"))
        self.assertNotIn("当前 Agent 未绑定所选 Target", workspace)
        self.assertIn("无需预先选择 Target", workspace)
        self.assertIn(': { agent_id: agentId, content: [], starter }', workspace)
        self.assertIn(': { agent_id: agentId, content }', workspace)
        self.assertNotIn("state.targets", workspace)
        self.assertIn('request(`${api}/targets?limit=200`)', workspace)
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
        self.assertIn("inspectionReportQualityText", workspace)
        self.assertIn("function diagnosisQuality(result, runStatus", workspace)
        self.assertIn("function diagnosisStatusHtml(run, result)", workspace)
        self.assertIn("evidence_coverage_percent", workspace)
        self.assertIn("sufficiency_status === \"ANSWERABLE\"", workspace)
        self.assertIn("证据有缺口", workspace)
        self.assertIn("证据覆盖", workspace)
        self.assertIn("历史报告未记录覆盖率", workspace)
        self.assertIn("数据库检查覆盖率：${coverage}%", workspace)
        self.assertIn('typeof coverage === "number" && Number.isFinite(coverage)', workspace)
        self.assertIn('typeof rawPercent === "number" && Number.isFinite(rawPercent)', workspace)
        self.assertIn("个目标完成", workspace)
        self.assertIn('block.block_type === "FINDING_CARDS"', workspace)
        self.assertIn("conversationAnswerHtml(result)", workspace)
        self.assertIn("bindReportArtifactActions(panel)", workspace)
        css = (AIOPS_ROOT / "css" / "workspaces.css").read_text(encoding="utf-8")
        self.assertIn(".ops-findings", css)
        self.assertIn(".ops-finding-card", css)
        self.assertIn(".ops-finding-fields", css)
        self.assertIn(".ops-analysis", css)
        self.assertIn(".ops-solution", css)
        self.assertIn("inspectionBullets", workspace)
        self.assertIn("需按检查项规则判断", workspace)
        self.assertIn("inspectionChecksHtml", workspace)
        self.assertIn("关键观测（最多 5 条）", workspace)
        self.assertIn("未发现数据缺口，全部检查已形成可验证观测。", workspace)
        self.assertIn("继续按既定周期执行该巡检模板并关注趋势变化。", workspace)
        self.assertNotIn('`**根因等级：**', workspace)
        self.assertIn("answerBlockHtml", workspace)
        self.assertIn("turnEvidenceHtml", workspace)
        self.assertIn("turn.display_dictionary", workspace)
        self.assertIn("result?.payload?.display_dictionary", workspace)
        self.assertIn("displayText", workspace)
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
        self.assertIn("data-clarification-candidate", workspace)
        self.assertIn("data-clarification-response", workspace)
        self.assertIn("/clarification-responses", workspace)
        self.assertIn("clarification_id: clarificationId", workspace)
        self.assertIn("idempotency_key: clarificationIdempotencyKey(clarificationId)", workspace)
        self.assertIn("return `clarification:${clarificationId}`", workspace)
        self.assertIn("candidate_id: candidateId", workspace)
        self.assertIn("free_text: normalizedText", workspace)
        self.assertIn("await followTurn(receipt.conversation_id, receipt.turn_id, progress)", workspace)
        self.assertIn("请先在当前轮回答范围确认", workspace)
        self.assertIn("也可以从左侧打开历史会话", workspace)
        self.assertIn("`${api}/conversations?limit=50`", workspace)
        self.assertIn("archiveConversation", workspace)
        self.assertIn('method: "DELETE"', workspace)
        self.assertIn('class="ops-workspace-delete"', workspace)
        self.assertIn('data-delete-id="${esc(item.conversation_id)}"', workspace)
        self.assertIn("会话将从聊天历史中移除", workspace)
        self.assertIn("关联的诊断、证据和变更审计记录仍会保留", workspace)
        self.assertIn("upload", workspace.lower())
        for obsolete in (
            "fleet.html",
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
        self.assertIn("按报告生成时间排序", page)
        self.assertIn("报告生成时间", script)
        self.assertNotIn("ops-report-period", script)
        self.assertIn('reports: { path: "/reports"', script)
        self.assertIn("data-report-version", script)
        self.assertIn("data-download-report", script)
        self.assertIn("data-edit-report", script)
        self.assertIn('method: "PATCH"', script)
        self.assertIn('"If-Match": `"rv-${report.report_version}"`', script)
        self.assertIn("不会覆盖旧版", script)
        self.assertIn("function reportMetadataHtml", script)
        self.assertIn("历史报告未记录覆盖率", script)
        self.assertIn("evidence_coverage_percent", script)
        self.assertIn('typeof rawCoverage === "number" && Number.isFinite(rawCoverage)', script)
        self.assertIn("monitoring_profile", script)
        self.assertIn("health_status", script)
        self.assertIn("database_type", script)
        self.assertIn("monitoring_sample_coverage_percent", script)
        self.assertIn("数据库检查覆盖率", script)
        self.assertIn("监控采样覆盖率", script)
        self.assertIn("证据完整性", script)
        self.assertIn("target?.display_name", script)
        self.assertIn("/targets/${encodeURIComponent(report.target_id)}", script)
        css = (AIOPS_ROOT / "css" / "aiops.css").read_text(encoding="utf-8")
        self.assertIn(".ops-report-metadata", css)

    def test_daily_inspection_report_uses_structured_formal_layout(self):
        script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        css = (AIOPS_ROOT / "css" / "aiops.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("DAILY_INSPECTION_PRESENTATION.v1", script)
        self.assertIn("关键指标与过去 7 日基线偏离", script)
        self.assertIn("七日趋势 · 小倍数", script)
        self.assertIn("七日趋势与容量明细", script)
        self.assertIn("无数据不等于异常", script)
        self.assertIn("frozenTargetIsIdentifier", script)
        self.assertIn('["巡检目标", targetName]', script)
        self.assertIn(".daily-formal-report", css)
        self.assertIn(".daily-report-table", css)
        self.assertIn(".daily-trend-svg", css)
        self.assertIn("grid-template-columns:repeat(4,minmax(0,1fr))", css)

    def test_monthly_inspection_report_uses_natural_month_layout(self):
        script = (AIOPS_ROOT / "js" / "aiops-pages.js").read_text(
            encoding="utf-8"
        )
        css = (AIOPS_ROOT / "css" / "aiops.css").read_text(
            encoding="utf-8"
        )
        self.assertIn("MONTHLY_INSPECTION_PRESENTATION.v1", script)
        self.assertIn("上月对比", script)
        self.assertIn("月度关键指标", script)
        self.assertIn("容量与增长", script)
        self.assertIn("问题台账", script)
        self.assertIn("扩展接口状态", script)
        self.assertIn(".monthly-formal-report", css)

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
        self.assertIn("data-history-session-template", script)
        self.assertIn("data-default-session-template", script)
        self.assertIn("data-toggle-session-template", script)
        self.assertIn("/session-report-templates/default", script)
        self.assertIn("/versions", script)
        self.assertIn("expected_row_version", script)
        workspace = (AIOPS_ROOT / "js" / "aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("template_version_id", workspace)


if __name__ == "__main__":
    unittest.main()
