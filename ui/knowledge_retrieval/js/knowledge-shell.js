/* 知识检索公共 Shell：对齐 Ammolite 知识检索的应用壳层、业务导航与设计规格。 */
(function () {
  "use strict";
  const PAGE_PERMISSIONS = {
    dashboard: "knowledge_retrieval:use", knowledge: "knowledge_retrieval:knowledge_chat",
    "x-search": "knowledge_retrieval:x_search", domains: "knowledge_retrieval:domain_manage",
    "knowledge-cores": "knowledge_retrieval:knowledge_core_manage",
    upload: "knowledge_retrieval:knowledge_core_manage", reviews: "knowledge_retrieval:knowledge_core_manage",
    processing: "knowledge_retrieval:knowledge_core_manage", files: "knowledge_retrieval:knowledge_core_manage",
    "data-sources": "knowledge_retrieval:data_model_manage",
    "data-models": "knowledge_retrieval:data_model_manage", agents: "knowledge_retrieval:agent_manage",
  };
  const sections = [
    ["workspace", "业务功能", "knowledge", [
      ["knowledge", "知识检索", "knowledge"], ["upload", "上传资料", "upload"],
      ["reviews", "资料审核", "review"], ["processing", "解析任务", "processing"], ["files", "知识库文件", "files"],
    ]],
    ["resources", "管理配置", "models", [
      ["knowledge-cores", "知识库管理", "knowledge"], ["data-sources", "数据源", "data"],
      ["data-models", "语义模型", "models"], ["agents", "Agent 管理", "agents"],
    ]],
    ["extensions", "扩展功能", "overview", [
      ["dashboard", "工作台", "overview"], ["x-search", "X 实时搜索", "search"], ["domains", "Domain 管理", "domains"],
    ]],
  ];
  // 沿用 AIOps 的线性图标，不引入另一套图标资源。
  const iconPaths = {
    upload: ["M12 16V3", "m7 8 5-5 5 5", "M4 16v5h16v-5"],
    review: ["M5 3h14v18H5z", "m8 12 3 3 5-6"],
    processing: ["M12 3a9 9 0 1 0 9 9", "M12 7v5l3 2"],
    files: ["M4 3h10l6 6v12H4z", "M14 3v6h6"],
    overview: ["M4 4h6v6H4z", "M14 4h6v6h-6z", "M4 14h6v6H4z", "M14 14h6v6h-6z"],
    knowledge: ["M5 4.5A2.5 2.5 0 0 1 7.5 2H11v17H7.5A2.5 2.5 0 0 0 5 21z", "M19 4.5A2.5 2.5 0 0 0 16.5 2H13v17h3.5A2.5 2.5 0 0 1 19 21z"],
    search: ["M21 21l-5-5", "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14Z"],
    models: ["M12 3 4.5 7.2v9.6L12 21l7.5-4.2V7.2z", "M4.8 7.4 12 12l7.2-4.6", "M12 12v9"],
    domains: ["M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Z", "M2 12h20", "M12 2a15 15 0 0 1 0 20", "M12 2a15 15 0 0 0 0 20"],
    data: ["M4 6c0 1.7 3.6 3 8 3s8-1.3 8-3-3.6-3-8-3-8 1.3-8 3Z", "M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6", "M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"],
    agents: ["M8 9h8", "M8 13h5", "M9 3h6l1 3h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h2z"],
    chevron: ["m9 18 6-6-6-6"], menu: ["M4 6h16", "M4 12h16", "M4 18h16"],
  };
  const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
  const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${iconPaths[name].map((path) => `<path d="${path}"></path>`).join("")}</svg>`;

  function toast(message, kind = "info") {
    let region = document.getElementById("knowledge-toast-region");
    if (!region) {
      region = document.createElement("div");
      region.id = "knowledge-toast-region";
      region.className = "knowledge-toast-region";
      region.setAttribute("aria-live", "polite");
      document.body.append(region);
    }
    const item = document.createElement("div");
    item.className = `knowledge-toast ${kind === "error" ? "error" : ""}`;
    item.textContent = message;
    region.append(item);
    setTimeout(() => item.remove(), 4500);
  }
  function badge(value, tone = "") { return `<span class="knowledge-badge ${escapeHtml(tone)}">${escapeHtml(value)}</span>`; }
  function capability(access, key) { return access?.capabilities?.[key] || { bound: false, verified: false, ready: false }; }
  function capabilityLabel(item) { return item.ready ? ["已就绪", "good"] : [item.bound ? "未验收" : "未绑定", "warn"]; }

  let switchingDomain = false;
  async function switchDomain(domainId) {
    const current = KBotKnowledgeAuth.load();
    const target = Number(domainId);
    if (!Number.isSafeInteger(target) || target <= 0) throw new Error("请选择有效的工作域");
    if (switchingDomain || String(current?.domain_id) === String(target)) return;
    switchingDomain = true;
    KBotKnowledgeApi.cancelPending();
    dispatchEvent(new Event("kbot:knowledge-context-changing"));
    const trigger = document.getElementById("knowledge-domain-toggle");
    if (trigger) trigger.disabled = true;
    document.querySelectorAll("[data-switch-domain]").forEach(button => { button.disabled = true; });
    try {
      const next = await KBotKnowledgeApi.json("/api/v1/auth/switch-domain", "POST", { domain_id: target });
      if (!next?.access_token || String(next.domain_id) !== String(target)) throw new Error("工作域切换响应无效，请重试");
      KBotKnowledgeApi.cancelPending();
      KBotKnowledgeAuth.save({ ...current, ...next });
      // 重建页面和权限快照，不能让旧工作域的知识库、会话或配置留在新上下文中。
      location.reload();
    } catch (error) {
      switchingDomain = false;
      if (trigger) trigger.disabled = false;
      document.querySelectorAll("[data-switch-domain]").forEach(button => {
        button.disabled = String(button.dataset.switchDomain) === String(current?.domain_id);
      });
      dispatchEvent(new Event("kbot:knowledge-context-restored"));
      throw error;
    }
  }

  function bindDomainSwitcher() {
    const wrapper = document.getElementById("knowledge-domain-switcher");
    const trigger = document.getElementById("knowledge-domain-toggle");
    const menu = document.getElementById("knowledge-domain-menu");
    const options = document.getElementById("knowledge-domain-options");
    let domains = [];
    let loading = false;
    const close = (restoreFocus = false) => {
      menu.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
      if (restoreFocus) trigger.focus();
    };
    function renderOptions() {
      const currentId = String(KBotKnowledgeAuth.load()?.domain_id);
      options.innerHTML = domains.length ? domains.map(row => {
        const current = String(row.domain_id) === currentId;
        return `<button class="business-space-option${current ? " is-current" : ""}" type="button" data-switch-domain="${escapeHtml(row.domain_id)}"${current ? ' aria-current="true"' : ""}${current || switchingDomain ? " disabled" : ""}>
          <span class="business-space-option-copy"><strong>${escapeHtml(row.name)}</strong><small>Domain ${escapeHtml(row.domain_id)}</small></span>
          <span class="business-space-option-state">${current ? "当前" : "切换"}</span>
        </button>`;
      }).join("") : '<p class="business-space-menu-empty">当前账号暂无可切换工作域</p>';
    }
    async function loadDomains() {
      if (loading) return;
      loading = true;
      options.innerHTML = '<p class="business-space-menu-empty" role="status">正在读取可切换工作域…</p>';
      try {
        const payload = await KBotKnowledgeApi.json("/api/v1/auth/domains", "GET");
        domains = (payload.domains || []).filter(row => row.status === "ACTIVE");
        renderOptions();
      } catch (error) {
        options.innerHTML = '<p class="business-space-menu-empty" role="alert"></p><button class="business-space-option" type="button" data-domain-retry>重新加载</button>';
        options.querySelector("p").textContent = error.message || "无法读取可切换工作域";
      } finally { loading = false; }
    }
    trigger.addEventListener("click", () => {
      if (!menu.hidden) { close(); return; }
      menu.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      void loadDomains();
    });
    options.addEventListener("click", event => {
      if (event.target.closest("[data-domain-retry]")) { void loadDomains(); return; }
      const button = event.target.closest("[data-switch-domain]");
      if (!button || button.disabled) return;
      void switchDomain(button.dataset.switchDomain).catch(error => {
        renderOptions();
        toast(error.message || "无法切换工作域", "error");
      });
    });
    document.addEventListener("pointerdown", event => { if (!wrapper.contains(event.target)) close(); });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && !menu.hidden) { close(wrapper.contains(document.activeElement)); return; }
      if (event.key === "Tab" && !menu.hidden) {
        setTimeout(() => { if (!wrapper.contains(document.activeElement)) close(); }, 0);
      }
    });
  }

  function shellMarkup(session, access, current) {
    const granted = new Set(access.permissions || []);
    const navigation = sections.map(([key, title, groupIcon, pages]) => {
      const visible = pages.filter(([id]) => granted.has(PAGE_PERMISSIONS[id]));
      if (!visible.length) return "";
      const open = visible.some(([id]) => id === current);
      const links = visible.map(([id, label, itemIcon]) => `
        <a class="nav-item business-nav-item${id === current ? " business-route-active" : ""}"
           href="./${id}.html" data-permission="${PAGE_PERMISSIONS[id]}"${id === current ? ' aria-current="page"' : ""}>
          <span class="nav-item-icon">${icon(itemIcon)}</span><span class="nav-item-label">${escapeHtml(label)}</span>
        </a>`).join("");
      return `<section class="business-nav-group is-collapsible${open ? " is-open" : ""}" data-current="${open}">
        <button class="business-nav-group-toggle" type="button" aria-expanded="${open}" aria-controls="knowledge-nav-${key}">
          <span class="business-nav-group-icon">${icon(groupIcon)}</span>
          <span class="business-nav-group-copy"><strong>${escapeHtml(title)}</strong></span>
          <span class="business-nav-group-count" aria-label="${visible.length} 个功能">${visible.length}</span>
          <span class="business-nav-group-chevron">${icon("chevron")}</span>
        </button>
        <div id="knowledge-nav-${key}" class="nav-group-items"${open ? "" : " hidden"}>${links}</div>
      </section>`;
    }).join("");
    const user = access.display_name || session.display_name || access.user_id || session.user_id || "已登录";
    const domain = access.domain_name || session.domain_name || "名称暂不可用";
    return `<header class="business-app-topbar">
      <a class="brand" href="../index.html"><span class="brand-mark">K</span><span class="brand-name">KBot</span><span class="brand-divider"></span><span class="brand-context">知识检索</span></a>
      <button id="knowledge-nav-trigger" class="icon-button business-nav-trigger" type="button" aria-label="打开 App 导航" aria-controls="knowledge-app-navigation" aria-expanded="false">${icon("menu")}</button>
      <div class="business-context-cluster"><div id="knowledge-domain-switcher" class="business-space-switcher-wrap">
      <button id="knowledge-domain-toggle" class="business-context-control" type="button" aria-label="切换工作域" aria-expanded="false" aria-controls="knowledge-domain-menu">
        <span class="business-context-icon">${icon("domains")}</span>
        <span class="business-context-copy"><small>当前工作域</small><strong>${escapeHtml(domain)}</strong></span>
        <span class="business-context-action">切换 ${icon("chevron")}</span>
      </button>
      <div id="knowledge-domain-menu" class="business-space-menu" aria-label="选择工作域" hidden>
        <header class="business-space-menu-header"><strong>切换工作域</strong><span>仅显示已授权工作域</span></header>
        <div id="knowledge-domain-options" class="business-space-options"></div>
      </div></div></div>
      <div class="topbar-actions business-topbar-actions"><div class="account-menu">
        <button id="knowledge-account-toggle" class="account-button" type="button" aria-expanded="false" aria-controls="knowledge-account-menu">
          <span class="avatar">${escapeHtml(Array.from(user)[0])}</span><span class="knowledge-account-name">${escapeHtml(user)}</span><span class="muted">⌄</span>
        </button>
        <div id="knowledge-account-menu" class="account-menu-popover" aria-label="账户操作" hidden>
          <button id="knowledge-logout" class="account-menu-item account-menu-item-danger" type="button">退出登录</button>
        </div>
      </div></div>
    </header>
    <div class="business-app-layout">
      <button id="knowledge-nav-scrim" class="shell-nav-scrim business-nav-scrim" type="button" aria-label="关闭 App 导航" hidden></button>
      <aside id="knowledge-app-navigation" class="business-app-sidebar" aria-label="知识检索功能导航">
        <nav>${navigation}</nav>
        <div class="nav-footer business-app-nav-footer"><span class="business-app-nav-footer-mark">K</span><span><strong>知识检索</strong><small>Knowledge Retrieval</small></span></div>
      </aside>
      <div id="knowledge-shell-content" class="business-app-content"></div>
    </div>`;
  }

  function bindShellActions() {
    const trigger = document.getElementById("knowledge-nav-trigger");
    const scrim = document.getElementById("knowledge-nav-scrim");
    const sidebar = document.getElementById("knowledge-app-navigation");
    const account = document.getElementById("knowledge-account-toggle");
    const menu = document.getElementById("knowledge-account-menu");
    const closeNavigation = () => {
      const restoreFocus = sidebar.contains(document.activeElement) || document.activeElement === scrim;
      sidebar.classList.remove("is-open");
      trigger.setAttribute("aria-expanded", "false");
      scrim.hidden = true;
      if (restoreFocus) trigger.focus();
    };
    trigger.addEventListener("click", () => {
      if (trigger.getAttribute("aria-expanded") === "true") { closeNavigation(); return; }
      sidebar.classList.add("is-open");
      trigger.setAttribute("aria-expanded", "true");
      scrim.hidden = false;
      sidebar.querySelector('[aria-current="page"]')?.focus();
    });
    scrim.addEventListener("click", closeNavigation);
    const closeAccount = () => { menu.hidden = true; account.setAttribute("aria-expanded", "false"); };
    account.addEventListener("click", () => {
      menu.hidden = !menu.hidden;
      account.setAttribute("aria-expanded", String(!menu.hidden));
    });
    document.addEventListener("pointerdown", (event) => {
      if (!event.target.closest(".account-menu")) closeAccount();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      closeNavigation();
      if (menu.contains(document.activeElement)) account.focus();
      closeAccount();
    });
    matchMedia("(max-width: 900px)").addEventListener("change", closeNavigation);
    document.getElementById("knowledge-logout").addEventListener("click", () => {
      KBotKnowledgeAuth.clear();
      location.replace("./login.html");
    });
    document.querySelectorAll(".business-nav-group-toggle").forEach((button) => {
      button.addEventListener("click", () => {
        const group = button.closest(".business-nav-group");
        const open = button.getAttribute("aria-expanded") === "true";
        if (open && group.dataset.current === "true") return;
        document.querySelectorAll(".business-nav-group").forEach((candidate) => {
          const expanded = candidate === group && !open;
          candidate.classList.toggle("is-open", expanded);
          candidate.querySelector(".business-nav-group-toggle").setAttribute("aria-expanded", String(expanded));
          candidate.querySelector(".nav-group-items").hidden = !expanded;
        });
      });
    });
  }

  function mount(session, access, page) {
    if (document.getElementById("knowledge-app-shell")) return;
    document.body.insertAdjacentHTML("afterbegin", `<div id="knowledge-app-shell" class="business-app-shell knowledge-app-shell">${shellMarkup(session, access, page)}</div>`);
    const main = document.querySelector("main.knowledge-main");
    if (main) {
      main.classList.add("platform-home-page", "knowledge-retrieval-page");
      if (page === "knowledge") document.getElementById("knowledge-shell-content").classList.add("is-viewport-contained", "is-agent-chat");
      main.querySelector(".knowledge-page-head")?.classList.add("app-page-head");
      document.getElementById("knowledge-shell-content").append(main);
    }
    bindShellActions();
    bindDomainSwitcher();
  }

  async function initialize() {
    const page = document.body.dataset.page || "";
    if (page === "login" || document.body.classList.contains("knowledge-login")) return null;
    const session = globalThis.KBotKnowledgeAuth?.load?.();
    if (!session?.access_token || session.must_change_password) { location.replace("./login.html"); return null; }
    let access;
    try { access = await KBotKnowledgeApi.json("/access", "GET"); }
    catch (error) { toast(error.message || "无法读取访问权限", "error"); return null; }
    const granted = new Set(access.permissions || []);
    if (!granted.has("knowledge_retrieval:use")) { KBotKnowledgeAuth.clear(); location.replace("./login.html"); return null; }
    const required = PAGE_PERMISSIONS[page];
    if (required && !granted.has(required)) { location.replace("./dashboard.html"); return null; }
    mount(session, access, page);
    document.querySelectorAll("[data-permission]").forEach((node) => {
      if (!granted.has(node.dataset.permission)) node.remove();
    });
    document.body.dataset.access = "ready";
    return access;
  }

  globalThis.KBotKnowledgeShell = {
    PAGE_PERMISSIONS, badge, capability, capabilityLabel, escapeHtml, icon, toast, switchDomain,
    confirmAction: options => import("../../shared/kbot-confirm.js?v=20261105_1").then(module => module.confirmAction(options)),
    ready: new Promise((resolve, reject) => {
      const start = () => initialize().then(resolve, reject);
      document.readyState === "loading" ? addEventListener("DOMContentLoaded", start, { once: true }) : start();
    }),
  };
})();
