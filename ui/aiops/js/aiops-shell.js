(function () {
  "use strict";
  const pagePermissions = {
    dashboard: "aiops:use", "work-item-detail": "aiops:use", "responsibility-groups": "aiops:work_item_manage", monitoring: "aiops:use", chat: "aiops:use", situations: "aiops:use", inspections: "aiops:use",
    reports: "aiops:use", "report-detail": "aiops:use",
    "run-detail": "aiops:use", targets: "aiops:target_manage",
    "business-systems": "aiops:target_manage",
    "business-system-workspace": "aiops:target_manage",
    "target-detail": "aiops:target_manage",
    "recovery-drills": "aiops:work_item_handle",
    "data-protection": "aiops:target_manage",
    "diagnostic-sources": "aiops:diagnostic_source_manage",
    "diagnostic-source-detail": "aiops:diagnostic_source_manage",
    "operations-knowledge-chat": "aiops:use",
    "operations-knowledge": "aiops:knowledge_manage", agents: "aiops:agent_manage",
    "inspection-plans": "aiops:plan_manage",
    "inspection-plan-detail": "aiops:plan_manage",
    "report-templates": "aiops:plan_manage",
    runs: "aiops:use",
    "api-clients": "aiops:api_key_manage",
  };
  const pinnedPages = [["dashboard", "AIOps 工作台", "overview"]];
  const navigationGroups = [
    ["monitoring-diagnosis", "监控与诊断", "data", [
      ["monitoring", "实时监控"], ["situations", "告警诊断"], ["chat", "智能诊断"],
    ]],
    ["operations-assurance", "运维保障", "audit", [
      ["inspections", "日常巡检"], ["recovery-drills", "恢复演练"], ["reports", "运维报告"],
      ["operations-knowledge-chat", "知识问答"], ["operations-knowledge", "运维知识"],
    ]],
    ["resource-management", "资源管理", "models", [
      ["business-systems", "业务系统"], ["diagnostic-sources", "监控源"], ["data-protection", "数据保护"],
    ]],
    ["configuration-governance", "配置与治理", "settings", [
      ["responsibility-groups", "责任组"], ["agents", "Agent 管理"], ["report-templates", "模板中心"], ["runs", "运行记录"],
    ]],
  ];
  const iconPaths = {
    overview: ["M4 4h6v6H4z", "M14 4h6v6h-6z", "M4 14h6v6H4z", "M14 14h6v6h-6z"],
    data: ["M4 6c0 1.7 3.6 3 8 3s8-1.3 8-3-3.6-3-8-3-8 1.3-8 3Z", "M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6", "M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"],
    audit: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z", "M14 2v6h6", "M8 13h8", "M8 17h6"],
    models: ["M12 3 4.5 7.2v9.6L12 21l7.5-4.2V7.2z", "M4.8 7.4 12 12l7.2-4.6", "M12 12v9"],
    settings: ["M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z", "M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.09a2 2 0 0 1 1 1.73v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.09a2 2 0 0 1-1-1.74v-.51a2 2 0 0 1 1-1.73l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2Z"],
    chevron: ["m9 18 6-6-6-6"],
  };
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
  function badge(value, label = value) {
    const text = String(value || "UNKNOWN").toUpperCase();
    const tone = ["ACTIVE", "ENABLED", "CONNECTED", "UP", "COMPLETED", "SUCCEEDED", "HEALTHY", "RESOLVED", "PUBLISHED", "APPROVED"].includes(text)
      ? "good" : ["FAILED", "ERROR", "CRITICAL", "REJECTED", "UNHEALTHY", "UNREACHABLE", "MISCONFIGURED", "DOWN"].includes(text)
        ? "bad" : ["RUNNING", "CHECKING", "OPEN", "PENDING", "WARNING", "DEGRADED"].includes(text)
          ? "warn" : "";
    return `<span class="ops-badge ${tone}">${escape(label || text)}</span>`;
  }
  function toast(message) {
    const node = document.createElement("div");
    node.className = "ops-toast";
    node.textContent = message;
    document.body.append(node);
    setTimeout(() => node.remove(), 5000);
  }
  function icon(name) {
    const paths = iconPaths[name] || iconPaths.overview;
    return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths.map((path) => `<path d="${path}"></path>`).join("")}</svg>`;
  }
  function enforceDrawerClosePolicy() {
    const blockBackdropInteraction = (event) => {
      if (event.target instanceof HTMLDialogElement && event.target.matches('[class*="drawer"]')) {
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    };
    ["pointerdown", "mousedown", "click"].forEach((eventName) => {
      document.addEventListener(eventName, blockBackdropInteraction, true);
    });
    document.addEventListener("cancel", (event) => {
      if (event.target instanceof HTMLDialogElement && event.target.matches('[class*="drawer"]')) {
        event.preventDefault();
      }
    }, true);
  }
  function shellMarkup(access) {
    const current = document.body.dataset.page;
    const aliases = {
      "inspection-plans": "inspections", "inspection-plan-detail": "inspections", "work-item-detail": "dashboard",
      "business-system-workspace": "business-systems", targets: "business-systems", "target-detail": "business-systems",
      "diagnostic-source-detail": "diagnostic-sources", "report-detail": "reports", "run-detail": "runs",
    };
    const currentNavigation = aliases[current] || current;
    const permissions = new Set(access.permissions || []);
    const visiblePinned = pinnedPages.filter(([id]) => permissions.has(pagePermissions[id]));
    const groups = navigationGroups.map(([key, name, groupIcon, pages]) => {
      const visible = pages.filter(([id]) => permissions.has(pagePermissions[id]));
      return visible.length ? { key, name, groupIcon, pages: visible } : null;
    }).filter(Boolean);
    const activeGroupKey = groups.find((group) => group.pages.some(([id]) => id === currentNavigation))?.key || groups[0]?.key || "";
    const pinned = visiblePinned.length
      ? `<section class="ops-nav-pinned" aria-label="常用功能"><nav class="ops-nav-items">${visiblePinned.map(([id, label, itemIcon]) => `<a class="ops-nav-item" href="./${id}.html" ${id === currentNavigation ? 'aria-current="page"' : ""}><span class="ops-nav-item-icon">${icon(itemIcon)}</span><span>${label}</span></a>`).join("")}</nav></section>`
      : "";
    const navigation = groups.map((group) => {
      const open = group.key === activeGroupKey;
      return `<section class="ops-nav-group${open ? " is-open" : ""}" data-group-key="${group.key}"><button class="ops-nav-group-toggle" type="button" aria-expanded="${open}" aria-controls="ops-nav-group-${group.key}"><span class="ops-nav-group-icon">${icon(group.groupIcon)}</span><strong>${group.name}</strong><span class="ops-nav-group-count" aria-label="${group.pages.length} 个功能">${group.pages.length}</span><span class="ops-nav-group-chevron">${icon("chevron")}</span></button><nav id="ops-nav-group-${group.key}" class="ops-nav-items"${open ? "" : " hidden"}>${group.pages.map(([id, label]) => `<a class="ops-nav-item" href="./${id}.html" ${id === currentNavigation ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</nav></section>`;
    }).join("");
    return `<aside class="ops-sidebar" aria-label="App 功能导航"><div class="ops-sidebar-navigation">${pinned}${navigation}</div><div class="ops-nav-footer"><span class="ops-nav-footer-mark">A</span><span><strong>AIOps</strong><small>Agent-native App</small></span></div></aside><header class="ops-topbar"><div><small>当前工作域</small> <strong>KBot AIOps</strong></div><div class="ops-session"><span id="ops-domain">Domain —</span><span id="ops-user">验证用户中…</span><button id="ops-logout">退出登录</button></div></header>`;
  }
  function bindNavigationGroups() {
    const groups = [...document.querySelectorAll(".ops-nav-group")];
    groups.forEach((group) => {
      const button = group.querySelector(".ops-nav-group-toggle");
      const items = group.querySelector(".ops-nav-items");
      button.addEventListener("click", () => {
        const isOpen = button.getAttribute("aria-expanded") === "true";
        if (isOpen && items.querySelector('[aria-current="page"]')) return;
        groups.forEach((candidate) => {
          const candidateButton = candidate.querySelector(".ops-nav-group-toggle");
          const candidateItems = candidate.querySelector(".ops-nav-items");
          const shouldOpen = candidate === group && !isOpen;
          candidate.classList.toggle("is-open", shouldOpen);
          candidateButton.setAttribute("aria-expanded", String(shouldOpen));
          candidateItems.hidden = !shouldOpen;
        });
      });
    });
  }
  async function initialize() {
    if (document.body.classList.contains("ops-login")) return null;
    if (!KBotAIOpsAuth.load()?.access_token) {
      location.replace("./login.html");
      return null;
    }
    try {
      enforceDrawerClosePolicy();
      const access = await KBotAIOpsAuth.request("/api/v1/apps/aiops/access");
      const current = document.body.dataset.page;
      if (!new Set(access.permissions || []).has(pagePermissions[current])) {
        location.replace("./dashboard.html");
        return null;
      }
      document.body.insertAdjacentHTML("afterbegin", shellMarkup(access));
      bindNavigationGroups();
      document.getElementById("ops-logout").onclick = () => {
        KBotAIOpsAuth.clear();
        location.replace("./login.html");
      };
      document.getElementById("ops-domain").textContent = `Domain ${access.domain_id}`;
      document.getElementById("ops-user").textContent = access.user_id;
      const permissions = new Set(access.permissions || []);
      document.querySelectorAll("[data-plan-manage]").forEach((element) => {
        element.hidden = !permissions.has("aiops:plan_manage");
      });
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
