(function () {
  "use strict";
  const sections = [
    ["业务工作区", [
      ["chat", "智能诊断"], ["situations", "告警诊断"],
      ["inspections", "日常巡检"],
    ]],
    ["资源配置", [
      ["targets", "运维目标"], ["diagnostic-sources", "诊断源"],
      ["knowledge-core", "Knowledge Core"], ["agents", "AIOps Agent"],
      ["inspection-plans", "巡检计划"],
      ["report-templates", "报告模板"],
      ["api-clients", "API 客户端"],
    ]],
  ];
  const escape = (value) => String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
  const fmt = (value) => {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime())
      ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
  };
  const short = (value) => {
    const text = String(value || "");
    return text.length > 18 ? `${text.slice(0, 8)}…${text.slice(-6)}` : text || "—";
  };
  function badge(value) {
    const text = String(value || "UNKNOWN").toUpperCase();
    const tone = ["ACTIVE", "ENABLED", "CONNECTED", "UP", "COMPLETED", "SUCCEEDED", "HEALTHY", "RESOLVED", "PUBLISHED", "APPROVED"].includes(text)
      ? "good" : ["FAILED", "ERROR", "CRITICAL", "REJECTED", "UNHEALTHY", "UNREACHABLE", "DOWN"].includes(text)
        ? "bad" : ["RUNNING", "CHECKING", "OPEN", "PENDING", "WARNING", "DEGRADED"].includes(text)
          ? "warn" : "";
    return `<span class="ops-badge ${tone}">${escape(text)}</span>`;
  }
  function toast(message) {
    const node = document.createElement("div");
    node.className = "ops-toast";
    node.textContent = message;
    document.body.append(node);
    setTimeout(() => node.remove(), 5000);
  }
  function shellMarkup() {
    const current = document.body.dataset.page;
    return `<aside class="ops-sidebar"><a class="ops-brand" href="./chat.html"><span class="ops-brand-mark">AI</span><span><strong>Operations Desk</strong><small>KBot AIOps 4.0</small></span></a>${sections.map(([name, pages]) => `<div class="ops-nav-label">${name}</div><nav class="ops-nav">${pages.map(([id, label]) => `<a href="./${id}.html" ${id === current ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</nav>`).join("")}</aside><header class="ops-topbar"><div><small>当前工作域</small> <strong>KBot AIOps</strong></div><div class="ops-session"><span id="ops-domain">Domain —</span><span id="ops-user">验证用户中…</span><button id="ops-logout">退出登录</button></div></header>`;
  }
  async function initialize() {
    if (document.body.classList.contains("ops-login")) return null;
    if (!KBotAIOpsAuth.load()?.access_token) {
      location.replace("./login.html");
      return null;
    }
    document.body.insertAdjacentHTML("afterbegin", shellMarkup());
    document.getElementById("ops-logout").onclick = () => {
      KBotAIOpsAuth.clear();
      location.replace("./login.html");
    };
    try {
      const access = await KBotAIOpsAuth.request("/api/v1/apps/aiops/access");
      document.getElementById("ops-domain").textContent = `Domain ${access.domain_id}`;
      document.getElementById("ops-user").textContent = access.user_id;
      return access;
    } catch (error) {
      toast(error.message);
      throw error;
    }
  }
  globalThis.KBotAIOpsShell = {
    ready: new Promise((resolve, reject) => addEventListener(
      "DOMContentLoaded", () => initialize().then(resolve).catch(reject), { once: true },
    )),
    escape, fmt, short, badge, toast,
  };
})();
