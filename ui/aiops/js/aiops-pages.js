(function () {
  "use strict";
  const appApi = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const configs = {
    situations: { path: "/situations", cols: [["title", "情境"], ["severity", "严重度", "badge"], ["status", "状态", "badge"], ["event_count", "事件数"], ["last_observed_at", "最近观测", "date"]] },
    runs: { path: "/runs", cols: [["ops_run_id", "运行 ID", "id"], ["trigger_type", "触发方式"], ["investigation_mode", "模式"], ["status", "状态", "badge"], ["created_at", "创建时间", "date"]], detail: "run-detail.html?id=" },
    reports: { path: "/reports", cols: [["report_key", "报告"], ["report_type", "类型"], ["status", "状态", "badge"], ["summary", "摘要"], ["period_end", "周期结束", "date"]], detail: "report-detail.html?id=" },
    inspections: { path: "/inspection-fires", cols: [["fire_id", "执行 ID", "id"], ["scheduled_at", "计划时间", "date"], ["status", "状态", "badge"], ["target_count", "目标数"], ["failed_count", "失败数"]] },
    changes: { path: "/proposals", cols: [["proposal_id", "建议 ID", "id"], ["action_template_id", "动作模板"], ["risk", "风险", "badge"], ["status", "状态", "badge"], ["expires_at", "失效时间", "date"]] },
    targets: { path: "/targets", cols: [["display_name", "目标"], ["engine_type", "数据库"], ["environment", "环境"], ["status", "状态", "badge"], ["updated_at", "更新时间", "date"]], detail: "target-detail.html?id=" },
    "diagnostic-sources": { path: "/diagnostic-sources", cols: [["display_name", "诊断源"], ["provider_type", "类型"], ["health_status", "健康", "badge"], ["status", "状态", "badge"], ["updated_at", "更新时间", "date"]], detail: "diagnostic-source-detail.html?id=" },
    policies: { path: "/policies", cols: [["display_name", "策略"], ["policy_type", "类型"], ["status", "状态", "badge"], ["row_version", "版本"]] },
    "inspection-plans": { path: "/inspection-plans", cols: [["display_name", "计划"], ["schedule_expression", "调度"], ["status", "状态", "badge"], ["updated_at", "更新时间", "date"]], detail: "inspection-plan-detail.html?id=" },
    "notification-subscriptions": { path: "/notification-subscriptions", cols: [["target_id", "目标", "id"], ["channel_type", "渠道"], ["minimum_severity", "最低级别", "badge"], ["enabled", "启用"]] },
  };
  const resourceId = (item) => item.ops_run_id || item.report_id || item.target_id || item.diagnostic_source_id || item.inspection_plan_id;
  function cell(item, [key, , type]) {
    const value = item?.[key];
    if (type === "badge") return shell.badge(value);
    if (type === "date") return shell.escape(shell.fmt(value));
    if (type === "id") return `<code>${shell.escape(shell.short(value))}</code>`;
    return shell.escape(value ?? "—");
  }
  async function renderList(page) {
    const cfg = configs[page];
    const head = document.getElementById("ops-table-head");
    const body = document.getElementById("ops-table-body");
    head.innerHTML = `<tr>${cfg.cols.map((col) => `<th>${col[1]}</th>`).join("")}</tr>`;
    try {
      const payload = await KBotAIOpsAuth.request(appApi + cfg.path);
      const items = Array.isArray(payload) ? payload : payload?.items || [];
      if (!items.length) {
        body.innerHTML = `<tr><td class="ops-empty" colspan="${cfg.cols.length}">当前范围内暂无数据</td></tr>`;
        return;
      }
      body.innerHTML = items.map((item) => `<tr ${cfg.detail ? `data-href="${cfg.detail}${encodeURIComponent(resourceId(item))}"` : ""}>${cfg.cols.map((col) => `<td>${cell(item, col)}</td>`).join("")}</tr>`).join("");
      body.querySelectorAll("[data-href]").forEach((row) => {
        row.style.cursor = "pointer";
        row.addEventListener("click", () => { location.href = row.dataset.href; });
      });
    } catch (error) {
      body.innerHTML = `<tr><td class="ops-empty" colspan="${cfg.cols.length}">${shell.escape(error.message)}</td></tr>`;
    }
  }
  async function renderDetail(page) {
    const id = new URLSearchParams(location.search).get("id");
    const paths = { "run-detail": "/runs/", "report-detail": "/reports/", "target-detail": "/targets/", "diagnostic-source-detail": "/diagnostic-sources/", "inspection-plan-detail": "/inspection-plans/" };
    const panel = document.getElementById("ops-detail");
    if (!id) { panel.innerHTML = '<div class="ops-error">URL 缺少资源 id</div>'; return; }
    try {
      const data = await KBotAIOpsAuth.request(appApi + paths[page] + encodeURIComponent(id));
      panel.innerHTML = `<dl class="ops-detail">${Object.entries(data).filter(([, value]) => typeof value !== "object").map(([key, value]) => `<dt>${shell.escape(key)}</dt><dd>${shell.escape(value ?? "—")}</dd>`).join("")}</dl><pre class="ops-code">${shell.escape(JSON.stringify(data, null, 2))}</pre>`;
    } catch (error) { panel.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`; }
  }
  async function renderDashboard() {
    const paths = ["/situations?status=OPEN&limit=5", "/runs?limit=5", "/proposals?status=PENDING_APPROVAL&limit=5", "/reports?limit=5"];
    const results = await Promise.allSettled(paths.map((path) => KBotAIOpsAuth.request(appApi + path)));
    results.forEach((result, index) => { document.querySelector(`[data-metric="${index}"]`).textContent = result.status === "fulfilled" ? result.value?.items?.length ?? 0 : "—"; });
    const rows = results.flatMap((result, index) => result.status === "fulfilled" ? (result.value?.items || []).map((value) => ({ type: ["故障情境", "诊断运行", "变更待办", "报告"][index], value })) : []);
    document.getElementById("dashboard-stream").innerHTML = rows.length ? rows.slice(0, 12).map(({ type, value }) => `<tr><td>${type}</td><td>${shell.escape(value.title || value.summary || value.status || "—")}</td><td>${shell.fmt(value.last_observed_at || value.created_at || value.period_end)}</td></tr>`).join("") : '<tr><td class="ops-empty" colspan="3">当前范围内暂无活动</td></tr>';
  }
  async function renderSimple(page) {
    const paths = { agents: `${appApi}/agents`, "report-templates": `${appApi}/report-templates`, "api-clients": `${appApi}/api-clients`, notifications: "/api/v1/notifications" };
    const panel = document.getElementById("ops-simple");
    if (!paths[page]) return;
    try { panel.innerHTML = `<pre class="ops-code">${shell.escape(JSON.stringify(await KBotAIOpsAuth.request(paths[page]), null, 2))}</pre>`; }
    catch (error) { panel.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`; }
  }
  shell.ready.then(() => {
    document.querySelectorAll("header.ops-head button:not([onclick])").forEach((button) => {
      button.disabled = true;
      button.title = "该写操作将在对应配置表单接入后开放";
    });
    const page = document.body.dataset.page;
    if (configs[page]) renderList(page);
    else if (page.endsWith("-detail")) renderDetail(page);
    else if (page === "dashboard") renderDashboard();
    else renderSimple(page);
  });
})();
