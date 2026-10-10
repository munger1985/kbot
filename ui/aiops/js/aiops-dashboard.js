(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const healthLabels = {
    CRITICAL: "严重",
    UNREACHABLE: "不可达",
    WARNING: "警告",
    STALE: "数据过期",
    UNKNOWN: "无有效数据",
    HEALTHY: "健康",
    DISABLED: "停用",
  };
  const categoryLabels = {
    ALERT: "告警",
    CONNECTIVITY: "连通性",
    AUTOMATION: "自动化",
    DATA_FRESHNESS: "数据新鲜度",
    CAPACITY: "容量",
    REPLICATION: "复制",
    BACKUP: "备份",
    SESSION: "会话与锁",
  };
  const statusLabels = {
    CURRENT: "当前有效",
    STALE: "数据过期",
    UNKNOWN: "未知",
    OPEN: "未恢复",
    RESOLVED: "当前无活动告警",
    COMPLETED: "已完成",
    SUCCEEDED: "成功",
    PARTIAL: "部分成功",
    FAILED: "失败",
    RUNNING: "执行中",
    CRITICAL: "严重",
    HIGH: "高",
    MEDIUM: "中",
    WARNING: "警告",
    LOW: "低",
    INFO: "提示",
    CONFIRMED: "已确认",
    CONNECTED: "已连接",
    UNREACHABLE: "不可达",
    MISCONFIGURED: "配置错误",
  };
  let dashboard = null;
  let riskCategory = "";
  let refreshTimer = null;

  const esc = shell.escape;

  function relativeTime(value) {
    if (!value) return "—";
    const time = new Date(value).getTime();
    if (!Number.isFinite(time)) return shell.fmt(value);
    const seconds = Math.max(0, Math.round((Date.now() - time) / 1000));
    if (seconds < 60) return `${seconds} 秒前`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
    return `${Math.floor(seconds / 86400)} 天前`;
  }

  function formatNumber(value, suffix = "") {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    const text = Number.isInteger(number) ? String(number) : number.toFixed(1);
    return `${text}${suffix}`;
  }

  function healthBadge(health) {
    const value = String(health || "UNKNOWN");
    return `<span class="ops-dashboard-state state-${esc(value.toLowerCase())}">${esc(healthLabels[value] || value)}</span>`;
  }

  function statusBadge(status) {
    const value = String(status || "UNKNOWN");
    return `<span class="ops-dashboard-status">${esc(statusLabels[value] || value)}</span>`;
  }

  function scrollToNode(node) {
    if (!node) return;
    const top = node.getBoundingClientRect().top + window.scrollY - 82;
    window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  }

  function targetLabel(item) {
    return `<strong>${esc(item.target_name || item.display_name || "未命名数据库")}</strong><small>${esc(item.environment || "—")} · L${esc(item.importance_level || "—")}</small>`;
  }

  function situationHref(item) {
    const query = new URLSearchParams();
    if (item.target_id) query.set("target_id", item.target_id);
    if (item.situation_id) query.set("situation", item.situation_id);
    return `./situations.html?${query}`;
  }

  function runHref(runId) {
    return `./run-detail.html?id=${encodeURIComponent(runId)}`;
  }

  function chatHref(targetId) {
    return `./chat.html?target_id=${encodeURIComponent(targetId)}`;
  }

  function itemHref(item) {
    if (item.situation_id) return situationHref(item);
    if (item.run_id) return runHref(item.run_id);
    return chatHref(item.target_id);
  }

  function setDashboardFilter(name, value) {
    const form = document.getElementById("dashboard-filters");
    if (name === "attention_only") form.elements.attention_only.checked = Boolean(value);
    else if (form.elements[name]) form.elements[name].value = value;
    renderTargets();
    scrollToNode(document.querySelector(".ops-dashboard-target-panel"));
  }

  function renderSummary() {
    const summary = dashboard.summary;
    const urgent = summary.critical_alert_count || summary.unreachable_count;
    const stable = summary.attention_target_count === 0;
    const headline = stable ? "当前态势稳定" : urgent ? "需要立即关注" : "存在待处理事项";
    const detail = stable
      ? `${summary.healthy_count} 个数据库健康，当前没有待处理异常。`
      : `${summary.attention_target_count} 个数据库需要关注，其中 ${summary.critical_alert_count} 个严重告警、${summary.unreachable_count} 个不可达。`;
    const metrics = [
      ["attention_target_count", "待处理数据库", "attention_only", true],
      ["critical_alert_count", "严重告警", "health", "CRITICAL"],
      ["unreachable_count", "不可达", "health", "UNREACHABLE"],
      ["stale_or_unknown_count", "数据盲区", "health", "BLIND_SPOT"],
      ["failed_automation_count", "自动化失败", "automation", "failed"],
    ];
    document.getElementById("dashboard-summary").innerHTML = `
      <article class="ops-dashboard-posture ${urgent ? "is-urgent" : stable ? "is-stable" : "is-warning"}">
        <span>当前态势</span><strong>${esc(headline)}</strong><p>${esc(detail)}</p>
      </article>
      ${metrics.map(([key, label, filter, value]) => `<button type="button" class="ops-dashboard-kpi" data-summary-filter="${esc(filter)}" data-summary-value="${esc(value)}"><span>${esc(label)}</span><strong>${esc(summary[key] ?? 0)}</strong><small>点击查看</small></button>`).join("")}`;
    document.querySelectorAll("[data-summary-filter]").forEach((button) => {
      button.onclick = () => {
        if (button.dataset.summaryFilter === "automation") {
          scrollToNode(document.getElementById("dashboard-automation"));
          return;
        }
        setDashboardFilter(button.dataset.summaryFilter, button.dataset.summaryValue === "true" ? true : button.dataset.summaryValue);
      };
    });
  }

  function renderAttention() {
    const items = dashboard.attention_items || [];
    document.getElementById("dashboard-attention-count").textContent = `${items.length} 项优先事项`;
    document.getElementById("dashboard-attention").innerHTML = items.length
      ? items.map((item) => `<article class="ops-dashboard-attention-row">
          <div class="ops-dashboard-target">${targetLabel(item)}</div>
          <div class="ops-dashboard-attention-copy"><div>${healthBadge(item.health)}<span>${esc(categoryLabels[item.category] || item.category)}</span>${statusBadge(item.status)}</div><strong>${esc(item.title)}</strong><small>${item.started_at ? `持续 ${esc(relativeTime(item.started_at).replace("前", ""))}` : "开始时间未知"} · 最近证据 ${esc(relativeTime(item.observed_at))}</small></div>
          <a class="ops-button" href="${esc(itemHref(item))}">查看处理</a>
        </article>`).join("")
      : '<div class="ops-empty">当前没有需要优先处理的数据库。</div>';
  }

  function renderHealth() {
    const values = dashboard.health_distribution || [];
    const maximum = Math.max(1, ...values.map((item) => Number(item.count || 0)));
    document.getElementById("dashboard-health").innerHTML = values.map((item) => `<button type="button" data-health-filter="${esc(item.health)}"><span>${healthBadge(item.health)}<strong>${esc(item.count)}</strong></span><i><b style="width:${Math.max(2, Number(item.count || 0) / maximum * 100)}%"></b></i></button>`).join("");
    document.querySelectorAll("[data-health-filter]").forEach((button) => {
      button.onclick = () => setDashboardFilter("health", button.dataset.healthFilter);
    });
  }

  function renderRisks() {
    const items = (dashboard.risk_items || []).filter((item) => !riskCategory || item.category === riskCategory);
    document.getElementById("dashboard-risks").innerHTML = items.length
      ? items.map((item) => `<tr><td><div class="ops-dashboard-target">${targetLabel(item)}</div></td><td>${esc(item.label)}</td><td>${esc(categoryLabels[item.category] || item.category)}</td><td><strong>${esc(item.value_text)}</strong></td><td>${statusBadge(item.severity)}</td><td title="${esc(shell.fmt(item.observed_at))}">${esc(relativeTime(item.observed_at))}</td><td><a href="${esc(runHref(item.run_id))}">查看诊断</a></td></tr>`).join("")
      : '<tr><td class="ops-empty" colspan="7">当前分类没有已确认风险。</td></tr>';
  }

  function renderAutomation() {
    const item = dashboard.automation;
    const groups = [
      ["自动诊断", item.run_total, item.run_succeeded, item.run_partial, item.run_failed],
      ["日常巡检", item.inspection_total, item.inspection_succeeded, item.inspection_partial, item.inspection_failed],
    ];
    document.getElementById("dashboard-automation").innerHTML = groups.map(([label, total, success, partial, failed]) => `<section><header><strong>${esc(label)}</strong><span>${esc(total)} 次</span></header><dl><div><dt>成功</dt><dd class="is-good">${esc(success)}</dd></div><div><dt>部分成功</dt><dd class="is-warn">${esc(partial)}</dd></div><div><dt>失败</dt><dd class="is-bad">${esc(failed)}</dd></div></dl></section>`).join("");
  }

  function renderActivities() {
    const items = dashboard.recent_activities || [];
    document.getElementById("dashboard-activities").innerHTML = items.length
      ? items.map((item) => `<a href="${esc(itemHref(item))}"><span>${esc(item.kind === "ALERT" ? "告警" : "诊断")}</span><div><strong>${esc(item.target_name || "未知数据库")}</strong><p>${esc(item.title)}</p></div>${statusBadge(item.status)}<time title="${esc(shell.fmt(item.occurred_at))}">${esc(relativeTime(item.occurred_at))}</time></a>`).join("")
      : '<div class="ops-empty">最近 24 小时没有可展示活动。</div>';
  }

  function targetMatches(item) {
    const form = document.getElementById("dashboard-filters");
    const search = form.elements.search.value.trim().toLowerCase();
    if (search && !`${item.display_name} ${item.attention_reason || ""}`.toLowerCase().includes(search)) return false;
    if (form.elements.environment.value && item.environment !== form.elements.environment.value) return false;
    if (form.elements.db_type.value && item.db_type !== form.elements.db_type.value) return false;
    if (Number(item.importance_level) < Number(form.elements.importance.value || 1)) return false;
    const health = form.elements.health.value;
    if (health === "BLIND_SPOT" && !["STALE", "UNKNOWN"].includes(item.health)) return false;
    if (health && health !== "BLIND_SPOT" && item.health !== health) return false;
    if (form.elements.attention_only.checked && ["HEALTHY", "DISABLED"].includes(item.health)) return false;
    return true;
  }

  function renderTargets() {
    if (!dashboard) return;
    const items = (dashboard.targets || []).filter(targetMatches);
    document.getElementById("dashboard-target-count").textContent = `显示 ${items.length} / ${dashboard.targets.length} 个数据库`;
    document.getElementById("dashboard-targets").innerHTML = items.length
      ? items.map((item) => `<tr class="health-${esc(item.health.toLowerCase())}"><td><strong>${esc(item.display_name)}</strong><small>${esc(item.db_type)}</small>${item.attention_reason ? `<p>${esc(item.attention_reason)}</p>` : ""}</td><td><strong>L${esc(item.importance_level)}</strong></td><td>${esc(item.environment)} / ${esc(item.db_role)}</td><td>${healthBadge(item.health)}</td><td>${esc(item.open_alert_count)}${item.critical_alert_count ? ` · <b>${esc(item.critical_alert_count)} 严重</b>` : ""}</td><td>${item.capacity_percent === null || item.capacity_percent === undefined ? "—" : `<strong>${esc(formatNumber(item.capacity_percent, "%"))}</strong><small>${esc(item.capacity_label || "容量")}</small>`}</td><td>${esc(formatNumber(item.lag_seconds, " 秒"))}</td><td>${statusBadge(item.data_freshness)}<small title="${esc(shell.fmt(item.evidence_observed_at))}">${esc(relativeTime(item.evidence_observed_at))}</small></td><td title="${esc(shell.fmt(item.last_diagnosed_at))}">${esc(relativeTime(item.last_diagnosed_at))}</td><td><div class="ops-dashboard-row-actions">${item.open_alert_count ? `<a href="${esc(situationHref(item))}">告警</a>` : ""}${item.latest_run_id ? `<a href="${esc(runHref(item.latest_run_id))}">诊断</a>` : ""}<a href="${esc(chatHref(item.target_id))}">智能运维</a></div></td></tr>`).join("")
      : '<tr><td class="ops-empty" colspan="10">当前筛选条件下没有数据库。</td></tr>';
  }

  function render() {
    renderSummary();
    renderAttention();
    renderHealth();
    renderRisks();
    renderAutomation();
    renderActivities();
    renderTargets();
    const blindSpots = dashboard.summary.stale_or_unknown_count;
    document.getElementById("dashboard-freshness").textContent = `生成于 ${shell.fmt(dashboard.generated_at)}${blindSpots ? ` · ${blindSpots} 个数据库数据已过期或未知` : " · 数据新鲜度正常"}`;
  }

  async function load() {
    const button = document.getElementById("refresh-dashboard");
    button.disabled = true;
    try {
      dashboard = await KBotAIOpsAuth.request(`${api}/dashboard`);
      render();
    } catch (error) {
      document.getElementById("dashboard-attention").innerHTML = `<div class="ops-error">${esc(error.message || "无法读取 Dashboard")}</div>`;
      shell.toast(error.message || "无法读取 Dashboard");
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
    const filters = document.getElementById("dashboard-filters");
    filters.oninput = renderTargets;
    filters.onchange = renderTargets;
    filters.onreset = () => window.setTimeout(renderTargets, 0);
    document.getElementById("refresh-dashboard").onclick = load;
    document.getElementById("dashboard-refresh-interval").onchange = scheduleRefresh;
    document.querySelectorAll("[data-risk-category]").forEach((button) => {
      button.onclick = () => {
        riskCategory = button.dataset.riskCategory || "";
        document.querySelectorAll("[data-risk-category]").forEach((item) => item.classList.toggle("is-active", item === button));
        renderRisks();
      };
    });
    scheduleRefresh();
    return load();
  }).catch((error) => shell.toast(error.message));
})();
