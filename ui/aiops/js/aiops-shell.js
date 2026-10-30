(function () {
  "use strict";
  const pagePermissions = {
    dashboard: "aiops:use", "work-item-detail": "aiops:use", "responsibility-groups": "aiops:work_item_manage", monitoring: "aiops:use", chat: "aiops:use", situations: "aiops:use", inspections: "aiops:use",
    reports: "aiops:use", "report-detail": "aiops:use", cases: "aiops:use",
    "run-detail": "aiops:use",
    "business-systems": "aiops:target_manage",
    "business-system-workspace": "aiops:target_manage",
    "recovery-drills": "aiops:work_item_handle",
    "diagnostic-sources": "aiops:diagnostic_source_manage",
    "diagnostic-source-detail": "aiops:diagnostic_source_manage",
    "operations-knowledge": "aiops:knowledge_manage", "knowledge-bases": "aiops:knowledge_manage", "knowledge-tags": "aiops:knowledge_manage", agents: "aiops:agent_manage",
    "inspection-plans": "aiops:plan_manage",
    "inspection-plan-detail": "aiops:plan_manage",
    "report-templates": "aiops:plan_manage",
    runs: "aiops:use",
    "api-clients": "aiops:api_key_manage",
  };
  const pinnedPages = [["chat", "Agent 工作台", "agents"]];
  const navigationGroups = [
    ["situations-events", "态势与事件", "data", [
      ["dashboard", "运维总览"], ["monitoring", "实时监控"], ["situations", "告警诊断"],
    ]],
    ["inspections-drills", "巡检与演练", "audit", [
      ["inspections", "日常巡检"], ["recovery-drills", "恢复演练"],
    ]],
    ["reports-knowledge", "报告与知识", "knowledge", [
      ["reports", "运维报告"], ["cases", "历史诊断案例"], ["operations-knowledge", "运维知识"],
    ]],
    ["configuration-governance", "配置与治理", "settings", [
      ["business-systems", "业务系统"], ["diagnostic-sources", "监控源"], ["responsibility-groups", "责任组"], ["knowledge-bases", "运维知识库"], ["agents", "Agent 管理"], ["report-templates", "模板中心"], ["runs", "运行记录"],
    ]],
  ];
  const pageIcons = {
    dashboard: "overview", monitoring: "data", situations: "audit", inspections: "data",
    "recovery-drills": "audit", reports: "knowledge", cases: "knowledge", "operations-knowledge": "knowledge",
    "business-systems": "models", "diagnostic-sources": "data", "responsibility-groups": "users",
    "knowledge-bases": "knowledge", agents: "agents", "report-templates": "knowledge", runs: "data",
  };
  const iconPaths = {
    overview: ["M4 4h6v6H4z", "M14 4h6v6h-6z", "M4 14h6v6H4z", "M14 14h6v6h-6z"],
    agents: ["M8 9h8", "M8 13h5", "M9 3h6l1 3h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h2z"],
    knowledge: ["M5 4.5A2.5 2.5 0 0 1 7.5 2H11v17H7.5A2.5 2.5 0 0 0 5 21z", "M19 4.5A2.5 2.5 0 0 0 16.5 2H13v17h3.5A2.5 2.5 0 0 1 19 21z"],
    data: ["M4 6c0 1.7 3.6 3 8 3s8-1.3 8-3-3.6-3-8-3-8 1.3-8 3Z", "M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6", "M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"],
    audit: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z", "M14 2v6h6", "M8 13h8", "M8 17h6"],
    models: ["M12 3 4.5 7.2v9.6L12 21l7.5-4.2V7.2z", "M4.8 7.4 12 12l7.2-4.6", "M12 12v9"],
    users: ["M16 20v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2", "M9 10a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z", "M16 11a4 4 0 0 0 0-8", "M22 20v-2a4 4 0 0 0-3-3.87"],
    domains: ["M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Z", "M2 12h20", "M12 2a15 15 0 0 1 0 20", "M12 2a15 15 0 0 0 0 20"],
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
      ? String(value) : `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}:${String(date.getSeconds()).padStart(2, "0")}`;
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
    const directoryAliases = {
      "inspection-plans": "inspections", "inspection-plan-detail": "inspections",
      "diagnostic-source-detail": "diagnostic-sources", "run-detail": "runs",
    };
    const detailParents = {
      "work-item-detail": "dashboard", "business-system-workspace": "business-systems",
      "report-detail": "reports",
    };
    const selectedPage = directoryAliases[current] || current;
    const currentNavigation = detailParents[current] || selectedPage;
    const permissions = new Set(access.permissions || []);
    const visiblePinned = pinnedPages.filter(([id]) => permissions.has(pagePermissions[id]));
    const groups = navigationGroups.map(([key, name, groupIcon, pages]) => {
      const visible = pages.filter(([id]) => permissions.has(pagePermissions[id]));
      return visible.length ? { key, name, groupIcon, pages: visible } : null;
    }).filter(Boolean);
    const activeGroupKey = groups.find((group) => group.pages.some(([id]) => id === currentNavigation))?.key || "";
    const pinned = visiblePinned.length
      ? `<section class="business-nav-pinned" aria-label="常用功能"><div class="nav-group-items">${visiblePinned.map(([id, label, itemIcon]) => `<a class="nav-item business-nav-item${id === selectedPage ? " business-route-active" : ""}" href="./${id}.html" ${id === selectedPage ? 'aria-current="page"' : ""}><span class="nav-item-icon">${icon(itemIcon)}</span><span class="nav-item-label">${label}</span></a>`).join("")}</div></section>`
      : "";
    const navigation = groups.map((group) => {
      const open = group.key === activeGroupKey;
      return `<section class="business-nav-group is-collapsible${open ? " is-open" : ""}" data-group-key="${group.key}" data-current="${open}"><button class="business-nav-group-toggle" type="button" aria-expanded="${open}" aria-controls="ops-nav-group-${group.key}"><span class="business-nav-group-icon">${icon(group.groupIcon)}</span><span class="business-nav-group-copy"><strong>${group.name}</strong></span><span class="business-nav-group-count" aria-label="${group.pages.length} 个功能">${group.pages.length}</span><span class="business-nav-group-chevron">${icon("chevron")}</span></button><div id="ops-nav-group-${group.key}" class="nav-group-items"${open ? "" : " hidden"}>${group.pages.map(([id, label]) => `<a class="nav-item business-nav-item${id === selectedPage ? " business-route-active" : ""}" href="./${id}.html" ${id === selectedPage ? 'aria-current="page"' : ""}><span class="nav-item-icon">${icon(pageIcons[id])}</span><span class="nav-item-label">${label}</span></a>`).join("")}</div></section>`;
    }).join("");
    return `<header class="business-app-topbar"><a class="brand" href="../index.html"><span class="brand-mark">K</span><span class="brand-name">KBot</span><span class="brand-divider"></span><span class="brand-context">AIOps</span></a><button id="ops-nav-trigger" class="icon-button nav-trigger business-nav-trigger" type="button" aria-label="打开 App 导航" aria-controls="business-app-navigation" aria-expanded="false">☰</button><div class="business-context-cluster"><div class="business-context-control ops-current-domain"><span class="business-context-icon">${icon("domains")}</span><span class="business-context-copy"><small>业务空间</small><strong id="ops-domain">Domain —</strong></span><span class="business-context-action is-static">当前</span></div></div><div class="topbar-actions business-topbar-actions"><div class="account-menu"><button id="ops-account-toggle" class="account-button" type="button" aria-expanded="false" aria-haspopup="menu"><span class="avatar">我</span><span id="ops-user">我的账户</span><span class="muted">⌄</span></button><div id="ops-account-menu" class="account-menu-popover" role="menu" aria-label="账户菜单" hidden><button id="ops-logout" class="account-menu-item account-menu-item-danger" type="button" role="menuitem">退出登录</button></div></div></div></header><div id="ops-shell-layout" class="business-app-layout"><button id="ops-nav-scrim" class="shell-nav-scrim business-nav-scrim" type="button" aria-label="关闭 App 导航" hidden></button><aside id="business-app-navigation" class="business-app-sidebar" aria-label="App 功能导航"><nav>${pinned}${navigation}</nav><div class="nav-footer business-app-nav-footer"><span class="business-app-nav-footer-mark">A</span><span><strong>AIOps</strong><small>Agent-native App</small></span></div></aside><div id="ops-shell-content" class="business-app-content"></div></div>`;
  }
  function bindNavigationGroups() {
    const groups = [...document.querySelectorAll(".business-nav-group")];
    groups.forEach((group) => {
      const button = group.querySelector(".business-nav-group-toggle");
      button.addEventListener("click", () => {
        const isOpen = button.getAttribute("aria-expanded") === "true";
        if (isOpen && group.dataset.current === "true") return;
        groups.forEach((candidate) => {
          const candidateButton = candidate.querySelector(".business-nav-group-toggle");
          const candidateItems = candidate.querySelector(".nav-group-items");
          const shouldOpen = candidate === group && !isOpen;
          candidate.classList.toggle("is-open", shouldOpen);
          candidateButton.setAttribute("aria-expanded", String(shouldOpen));
          candidateItems.hidden = !shouldOpen;
        });
      });
    });
  }
  function mountPage() {
    document.body.classList.add("aiops-static-page");
    const main = document.querySelector("main.ops-main");
    const content = document.getElementById("ops-shell-content");
    if (!main || !content) return;
    content.append(main);
    // Portal 的业务抽屉留在页面内，Teleport 确认框才放在 body；保持相同的继承边界。
    document.querySelectorAll("body > dialog:not(.confirm-dialog):not(#monitoring-data-drawer):not(#business-resource-drawer)").forEach(dialog => main.append(dialog));
    main.classList.add("platform-home-page", "aiops-page", "page-shell");
    if (["chat", "monitoring"].includes(document.body.dataset.page)) {
      content.classList.add("is-viewport-contained");
      main.classList.add("app-page-rail-workspace");
    }
    if (document.body.dataset.page === "business-system-workspace") content.classList.add("is-viewport-contained");
    const aliases = [
      [".ops-head:not(.detail-hero)", "app-page-head"], [".ops-panel", "table-card"],
      [".ops-panel-head", "resource-list-header"], [".ops-panel-head > div:first-child", "resource-list-heading"],
      [".ops-panel-body", "resource-list-body"], [".ops-table-wrap", "table-scroll"],
      [".ops-actions", "button-row"], [".ops-related-link", "related-page-link"],
      [".ops-related-link > span", "related-page-link-copy"], [".ops-related-link > b", "related-page-link-arrow"],
    ];
    for (const [selector, name] of aliases) main.querySelectorAll(selector).forEach((node) => node.classList.add(name));
    const upgradeControls = (container) => {
      container.querySelectorAll('button.primary, a.ops-button.primary').forEach((node) => node.classList.add("primary-button"));
      container.querySelectorAll('button.danger').forEach((node) => node.classList.add("danger-button"));
      container.querySelectorAll('button:not([class]), a.ops-button:not(.primary)').forEach((node) => {
        if (node.closest('.business-system-filters, .segmented-control, [role="tablist"]')) return;
        node.classList.add(node.closest("table") ? "row-action" : "secondary-button");
      });
    };
    upgradeControls(main);
    document.querySelectorAll("dialog").forEach(upgradeControls);
    new MutationObserver((records) => {
      for (const record of records) for (const node of record.addedNodes) if (node.nodeType === 1) upgradeControls(node.parentElement || node);
    }).observe(main, { childList: true, subtree: true });
  }
  function bindShellActions() {
    const trigger = document.getElementById("ops-nav-trigger");
    const scrim = document.getElementById("ops-nav-scrim");
    const sidebar = document.getElementById("business-app-navigation");
    const account = document.getElementById("ops-account-toggle");
    const menu = document.getElementById("ops-account-menu");
    const closeNavigation = () => { sidebar.classList.remove("is-open"); trigger.setAttribute("aria-expanded", "false"); scrim.hidden = true; };
    trigger.onclick = () => {
      const open = trigger.getAttribute("aria-expanded") !== "true";
      sidebar.classList.toggle("is-open", open); trigger.setAttribute("aria-expanded", String(open)); scrim.hidden = !open;
    };
    scrim.onclick = closeNavigation;
    account.onclick = () => { menu.hidden = !menu.hidden; account.setAttribute("aria-expanded", String(!menu.hidden)); };
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") { closeNavigation(); menu.hidden = true; account.setAttribute("aria-expanded", "false"); } });
    document.addEventListener("pointerdown", (event) => { if (!event.target.closest(".account-menu")) { menu.hidden = true; account.setAttribute("aria-expanded", "false"); } });
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
      // 与 Portal 一样把壳层限制在 App 容器内；body 上的模态框不继承侧栏控件规格。
      document.body.insertAdjacentHTML("afterbegin", `<div id="ops-app-shell" class="business-app-shell business-app-aiops">${shellMarkup(access)}</div>`);
      mountPage();
      bindNavigationGroups();
      bindShellActions();
      document.getElementById("ops-logout").onclick = () => {
        KBotAIOpsAuth.clear();
        location.replace("./login.html");
      };
      document.getElementById("ops-domain").textContent = access.domain_display_name || access.domain_name || `Domain ${access.domain_id}`;
      document.getElementById("ops-user").textContent = access.display_name || access.user_id;
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
