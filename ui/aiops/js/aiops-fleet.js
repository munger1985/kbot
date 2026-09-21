(function () {
  "use strict";
  const appApi = "/api/v1/apps/aiops";
  const healthLabel = {
    HEALTHY: "健康",
    WARNING: "警告",
    CRITICAL: "严重",
    UNREACHABLE: "不可达",
    DISABLED: "停用",
  };
  const summaryItems = [
    ["target_count", "目标库"],
    ["healthy_count", "健康"],
    ["warning_count", "警告"],
    ["critical_count", "严重"],
    ["unreachable_count", "不可达"],
    ["disabled_count", "停用"],
    ["open_alert_count", "未关闭告警"],
  ];

  function cardHref(item) {
    if ((item.open_alert_count || 0) > 0) return "./situations.html";
    if (item.latest_run_id) {
      return `./run-detail.html?id=${encodeURIComponent(item.latest_run_id)}`;
    }
    return "./chat.html";
  }

  function healthBadge(health) {
    const tone = {
      HEALTHY: "good",
      WARNING: "warn",
      CRITICAL: "bad",
      UNREACHABLE: "bad",
      DISABLED: "",
    }[health] || "";
    return `<span class="ops-badge ${tone}">${KBotAIOpsShell.escape(healthLabel[health] || health)}</span>`;
  }

  function formatNumber(value, suffix) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    const text = Number.isInteger(number) ? String(number) : number.toFixed(1);
    return suffix ? `${text}${suffix}` : text;
  }

  function render(dashboard) {
    const summary = dashboard.summary || {};
    const items = dashboard.items || [];
    const summaryNode = document.getElementById("fleet-summary");
    const gridNode = document.getElementById("fleet-grid");
    summaryNode.innerHTML = summaryItems.map(([key, label]) => (
      `<article class="ops-fleet-kpi"><span>${KBotAIOpsShell.escape(label)}</span><strong>${KBotAIOpsShell.escape(summary[key] ?? 0)}</strong></article>`
    )).join("");
    if (!items.length) {
      gridNode.innerHTML = '<div class="ops-empty">当前工作域还没有运维目标。</div>';
      return;
    }
    gridNode.innerHTML = items.map((item) => {
      const health = String(item.health || "UNKNOWN");
      return `<a class="ops-fleet-card health-${KBotAIOpsShell.escape(health.toLowerCase())}" href="${cardHref(item)}">
        <header><strong>${KBotAIOpsShell.escape(item.display_name || "未命名库")}</strong>${healthBadge(health)}</header>
        <p>${KBotAIOpsShell.escape(item.db_type || "—")} · ${KBotAIOpsShell.escape(item.environment || "—")}</p>
        <dl>
          <div><dt>未关闭告警</dt><dd>${KBotAIOpsShell.escape(item.open_alert_count ?? 0)}</dd></div>
          <div><dt>最高容量</dt><dd>${KBotAIOpsShell.escape(formatNumber(item.max_capacity_percent, "%"))}</dd></div>
          <div><dt>最高延迟</dt><dd>${KBotAIOpsShell.escape(formatNumber(item.max_lag_seconds, " 秒"))}</dd></div>
          <div><dt>最近诊断</dt><dd>${KBotAIOpsShell.escape(KBotAIOpsShell.fmt(item.last_diagnosed_at))}</dd></div>
        </dl>
      </a>`;
    }).join("");
  }

  async function load() {
    try {
      const dashboard = await KBotAIOpsAuth.request(`${appApi}/fleet`);
      render(dashboard);
    } catch (error) {
      document.getElementById("fleet-grid").innerHTML =
        `<div class="ops-error">${KBotAIOpsShell.escape(error.message || "无法读取库群总览")}</div>`;
    }
  }

  KBotAIOpsShell.initialize().then(() => {
    document.getElementById("refresh-fleet").onclick = load;
    return load();
  }).catch(() => {});
})();
