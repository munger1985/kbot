(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const healthLabels = {
    CRITICAL: "严重", UNREACHABLE: "不可达", WARNING: "警告",
    STALE: "数据过期", UNKNOWN: "无有效数据", HEALTHY: "健康", DISABLED: "停用",
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
  let refreshTimer = null;

  const esc = shell.escape;

  function healthBadge(health) {
    const value = String(health || "UNKNOWN");
    return `<span class="ops-dashboard-state state-${esc(value.toLowerCase())}">${esc(healthLabels[value] || value)}</span>`;
  }

  function reportBadge(status) {
    const value = String(status || "UNKNOWN");
    const tone = value === "READY" ? "good" : value === "FAILED" ? "bad" : "warn";
    return `<span class="ops-badge ${tone}">${esc(reportStatusLabels[value] || value)}</span>`;
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
      <article><span>${esc(label)}</span><strong>${Number(summary[key] || 0)}</strong></article>
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
      ? items.map((item) => `<article class="ops-action-attention-row">
          <div><strong>${esc(item.display_name || "未命名数据库")}</strong><small>${esc(item.db_type || "—")} · ${esc(item.environment || "—")} · ${esc(item.attention_reason || `${Number(item.open_alert_count || 0)} 个未关闭告警`)}</small></div>
          ${healthBadge(item.health)}
        </article>`).join("")
      : '<div class="ops-empty">当前没有需要特别关注的运维目标。</div>';
  }

  function renderAutomation() {
    const item = dashboard.automation || {};
    const groups = [
      ["诊断运行", item.run_total, item.run_failed, item.run_partial],
      ["巡检执行", item.inspection_total, item.inspection_failed, item.inspection_partial],
    ];
    document.getElementById("dashboard-automation").innerHTML = groups.map(([label, total, failed, partial]) => `
      <section><header><strong>${esc(label)}</strong><span>${Number(total || 0)} 次</span></header><p>失败 ${Number(failed || 0)} · 部分 ${Number(partial || 0)}</p></section>
    `).join("");
  }

  function renderLatestReport() {
    const node = document.getElementById("dashboard-latest-report");
    if (!latestReport) {
      node.innerHTML = '<div class="ops-empty">当前业务空间还没有正式报告。</div>';
      return;
    }
    node.innerHTML = `<article><header><strong>${esc(latestReport.title || "正式报告")}</strong>${reportBadge(latestReport.status)}</header><p>${esc(latestReport.summary || "报告未提供摘要")}</p><footer><time>${esc(shell.fmt(latestReport.published_at || latestReport.created_at))}</time><a href="./report-detail.html?id=${encodeURIComponent(latestReport.report_id)}">打开报告 →</a></footer></article>`;
  }

  function render() {
    renderSummary();
    renderAttention();
    renderAutomation();
    renderLatestReport();
    document.getElementById("dashboard-freshness").textContent = `生成于 ${shell.fmt(dashboard.generated_at)}`;
  }

  async function load() {
    const button = document.getElementById("refresh-dashboard");
    button.disabled = true;
    try {
      const [dashboardResult, reportsResult] = await Promise.allSettled([
        KBotAIOpsAuth.request(`${api}/dashboard`),
        KBotAIOpsAuth.request(`${api}/reports?limit=1`),
      ]);
      if (dashboardResult.status === "rejected") throw dashboardResult.reason;
      dashboard = dashboardResult.value;
      if (reportsResult.status === "fulfilled") {
        const payload = reportsResult.value;
        latestReport = (Array.isArray(payload) ? payload : payload?.items || [])[0] || null;
      }
      render();
    } catch (error) {
      document.getElementById("dashboard-summary").innerHTML = `<div class="ops-error">${esc(error.message || "无法读取运行态势")}</div>`;
      shell.toast(error.message || "无法读取运行态势");
    } finally {
      button.disabled = false;
    }
  }

  function scheduleRefresh() {
    window.clearInterval(refreshTimer);
    const seconds = Number(document.getElementById("dashboard-refresh-interval").value || 0);
    if (seconds > 0) refreshTimer = window.setInterval(load, seconds * 1000);
  }

  shell.ready.then(() => {
    document.getElementById("refresh-dashboard").onclick = load;
    document.getElementById("dashboard-refresh-interval").onchange = scheduleRefresh;
    scheduleRefresh();
    return load();
  }).catch((error) => shell.toast(error.message));
})();
