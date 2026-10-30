(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const healthLabels = {
    CRITICAL: "严重", UNREACHABLE: "不可达", WARNING: "警告",
    STALE: "数据过期", UNKNOWN: "待观测", HEALTHY: "健康", DISABLED: "停用",
  };
  const reportStatusLabels = {
    READY: "结论就绪", PARTIAL: "证据不完整", GENERATING: "生成中", FAILED: "生成异常",
  };
  const healthRank = {
    CRITICAL: 0, UNREACHABLE: 1, WARNING: 2, STALE: 3,
    UNKNOWN: 4, HEALTHY: 5, DISABLED: 6,
  };
  let dashboard = null;
  let latestReport = null;

  const esc = shell.escape;

  function healthBadge(health) {
    const value = String(health || "UNKNOWN");
    const tone = ["CRITICAL","UNREACHABLE"].includes(value) ? "status-danger" : ["WARNING","STALE","UNKNOWN"].includes(value) ? "status-warning" : value === "DISABLED" ? "status-muted" : "status-success";
    return `<span class="status-pill ${tone}">${esc(healthLabels[value] || value)}</span>`;
  }

  function reportBadge(status) {
    const value = String(status || "UNKNOWN");
    const tone = value === "FAILED" ? "status-danger" : ["PARTIAL","GENERATING"].includes(value) ? "status-warning" : "status-success";
    return `<span class="status-pill ${tone}">${esc(reportStatusLabels[value] || value)}</span>`;
  }

  function renderSummary() {
    const summary = dashboard.summary || {};
    const metrics = [
      ["attention_target_count", "需关注目标"],
      ["critical_alert_count", "严重告警"],
      ["unreachable_count", "不可达"],
      ["stale_or_unknown_count", "数据陈旧"],
    ];
    document.getElementById("dashboard-summary").innerHTML = metrics.map(([key, label]) => `
      <div><dt>${esc(label)}</dt><dd>${Number(summary[key] || 0)}</dd></div>
    `).join("");
    document.getElementById("dashboard-attention-count").textContent = `${Number(summary.open_alert_count || 0)} 个未关闭告警`;
  }

  function renderAttention() {
    const items = [...(dashboard.targets || [])]
      .filter((item) => !["HEALTHY", "DISABLED"].includes(item.health))
      .sort((left, right) => (healthRank[left.health] ?? 99) - (healthRank[right.health] ?? 99)
        || Number(left.importance_level || 99) - Number(right.importance_level || 99))
      .slice(0, 5);
    document.getElementById("dashboard-attention").innerHTML = items.length
      ? `<ul>${items.map((item) => `<li>
          <div><strong>${esc(item.display_name || "未命名数据库")}</strong><small>${esc(item.db_type || "—")} · ${esc(item.environment || "—")} · ${esc(item.attention_reason || `${Number(item.open_alert_count || 0)} 个未关闭告警`)}</small></div>
          ${healthBadge(item.health)}
        </li>`).join("")}</ul>`
      : '<p class="context-empty">当前没有需要特别关注的运维目标。</p>';
  }

  function renderAutomation() {
    const item = dashboard.automation || {};
    const groups = [
      ["诊断运行", item.run_total, item.run_failed, item.run_partial],
      ["巡检执行", item.inspection_total, item.inspection_failed, item.inspection_partial],
    ];
    document.getElementById("dashboard-automation").innerHTML = groups.map(([label, total, failed, partial]) => `
      <div><dt>${esc(label)}</dt><dd>${Number(total || 0)}</dd><small>失败 ${Number(failed || 0)} · 部分 ${Number(partial || 0)}</small></div>
    `).join("");
  }

  function renderLatestReport() {
    const node = document.getElementById("dashboard-latest-report");
    if (!latestReport) {
      node.innerHTML = '<p class="context-empty">当前业务空间还没有正式报告。</p>';
      return;
    }
    node.innerHTML = `<article class="latest-report-context"><div><strong>${esc(latestReport.title)}</strong>${reportBadge(latestReport.status)}</div><p>${esc(latestReport.summary)}</p><footer><time>${esc(shell.fmt(latestReport.published_at))}</time><a href="./report-detail.html?id=${encodeURIComponent(latestReport.report_id)}">打开报告 →</a></footer></article>`;
  }

  function render() {
    renderSummary();
    renderAttention();
    renderAutomation();
    renderLatestReport();
    document.getElementById("dashboard-freshness").textContent = shell.fmt(dashboard.generated_at);
  }

  async function load() {
    const button = document.getElementById("refresh-dashboard");
    button.disabled = true;
    button.textContent = "正在刷新…";
    document.getElementById("dashboard-error").hidden = true;
    try {
      const [dashboardResult, reportsResult] = await Promise.all([
        KBotAIOpsAuth.request(`${api}/dashboard`),
        KBotAIOpsAuth.request(`${api}/reports?limit=6`),
      ]);
      dashboard = dashboardResult;
      latestReport = (Array.isArray(reportsResult) ? reportsResult : reportsResult?.items || [])[0] || null;
      render();
    } catch (error) {
      const notice = document.getElementById("dashboard-error");
      notice.hidden = false;
      notice.querySelector("span").textContent = error.message || "无法读取运行态势";
    } finally { if (!button.dataset.refreshing) {button.disabled=false;button.textContent="刷新工作台";} }
  }

  shell.ready.then(() => {
    document.getElementById("refresh-dashboard").onclick = async event => {
      const button = event.currentTarget;
      if (button.dataset.refreshing) return;
      button.dataset.refreshing = "true";
      try {await Promise.all([load(),globalThis.KBotAIOpsWorkItems?.refresh()]);}
      finally {delete button.dataset.refreshing;button.disabled=false;button.textContent="刷新工作台";}
    };
    return load();
  }).catch((error) => shell.toast(error.message));
})();
