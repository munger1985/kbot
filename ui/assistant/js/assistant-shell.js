/* 智能工作台公共 Shell：先注入完整导航，再按权限删除，并在就绪后暴露 access。 */
(function () {
  "use strict";

  const PAGE_PERMISSIONS = {
    dashboard: "assistant:use",
    knowledge: "assistant:knowledge_chat",
    "x-search": "assistant:x_search",
    "image-generation": "assistant:image_generate",
    domains: "assistant:domain_manage",
    "knowledge-cores": "assistant:knowledge_core_manage",
    "data-models": "assistant:data_model_manage",
    agents: "assistant:agent_manage",
    "model-bindings": "assistant:model_binding_manage",
    "media-assets": "assistant:media_read",
    "usage-runs": "assistant:run_read",
  };

  const sections = [
    ["业务工作区", [
      ["dashboard", "工作台", "./dashboard.html"],
      ["knowledge", "知识问答", "./knowledge.html"],
      ["x-search", "X 实时搜索", "./x-search.html"],
      ["image-generation", "文生图", "./image-generation.html"],
    ]],
    ["资源配置", [
      ["domains", "Domains", "./domains.html"],
      ["knowledge-cores", "Knowledge Cores", "./knowledge-cores.html"],
      ["data-models", "问数模型", "./data-models.html"],
      ["agents", "Agents", "./agents.html"],
    ]],
    ["管理", [
      ["model-bindings", "模型绑定", "./model-bindings.html"],
      ["media-assets", "图片资产", "./media-assets.html"],
      ["usage-runs", "用量与运行记录", "./usage-runs.html"],
    ]],
  ];

  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

  function toast(message, kind = "info") {
    const region = document.getElementById("assistant-toast-region");
    if (!region) return;
    const item = document.createElement("div");
    item.className = `assistant-toast ${kind === "error" ? "error" : ""}`;
    item.textContent = message;
    region.append(item);
    setTimeout(() => item.remove(), 4500);
  }

  function badge(value, tone = "") {
    return `<span class="assistant-badge ${escapeHtml(tone)}">${escapeHtml(value)}</span>`;
  }

  function capability(access, key) {
    return access?.capabilities?.[key] || { bound: false, verified: false, ready: false };
  }

  function capabilityLabel(item) {
    if (item.ready) return ["已就绪", "good"];
    if (!item.bound) return ["未绑定", "warn"];
    return ["未验收", "warn"];
  }

  function shellMarkup(session, current) {
    const navigation = sections.map(([title, pages]) => {
      const links = pages.map(([id, label, href]) => {
        const permission = PAGE_PERMISSIONS[id];
        const currentAttr = id === current ? ' aria-current="page"' : "";
        const permissionAttr = permission ? ` data-permission="${escapeHtml(permission)}"` : "";
        return `<a href="${href}"${permissionAttr}${currentAttr}>${escapeHtml(label)}</a>`;
      }).join("");
      return `<div class="assistant-nav-label">${escapeHtml(title)}</div><nav class="assistant-nav" aria-label="${escapeHtml(title)}">${links}</nav>`;
    }).join("");
    const domain = session.domain_name || "assistant_portal";
    const user = session.display_name || session.user_id || "已登录";
    return `
      <aside class="assistant-sidebar">
        <a class="assistant-brand" href="./dashboard.html" aria-label="智能工作台首页">
          <span class="assistant-brand-mark">AS</span><span><strong>Intelligent Desk</strong><small>KBot 4.0</small></span>
        </a>
        ${navigation}
        <div class="assistant-sidebar-note">三个入口独立运行：企业知识、外部实时线索与创作资产不自动混合。</div>
      </aside>
      <header class="assistant-topbar">
        <div class="assistant-context"><small>当前工作域</small><strong>智能工作台</strong></div>
        <div class="assistant-session">
          <span id="assistant-domain">${escapeHtml(domain)}</span>
          <span class="assistant-session-user" id="assistant-user">${escapeHtml(user)}</span>
          <button class="small" type="button" id="assistant-logout">退出</button>
        </div>
      </header>
      <div id="assistant-toast-region" class="assistant-toast-region" aria-live="polite"></div>`;
  }

  function pruneNavigation(permissions) {
    document.querySelectorAll("[data-permission]").forEach((element) => {
      if (!permissions.has(element.dataset.permission)) element.remove();
    });
    document.querySelectorAll(".assistant-nav").forEach((nav) => {
      if (nav.querySelector("a")) return;
      const label = nav.previousElementSibling;
      if (label?.classList.contains("assistant-nav-label")) label.remove();
      nav.remove();
    });
  }

  function mountShell(session, current) {
    if (document.querySelector(".assistant-sidebar")) return;
    document.body.insertAdjacentHTML("afterbegin", shellMarkup(session, current));
    document.getElementById("assistant-logout")?.addEventListener("click", () => {
      KBotAssistantAuth.clear();
      location.replace("./login.html");
    });
  }

  async function initialize() {
    const page = document.body.dataset.page || "";
    if (page === "login" || document.body.classList.contains("assistant-login")) return null;
    const session = globalThis.KBotAssistantAuth?.load?.();
    if (!session?.access_token) {
      location.replace("./login.html");
      return null;
    }
    if (session.must_change_password) {
      location.replace("./login.html");
      return null;
    }
    mountShell(session, page);
    let access;
    try {
      access = await KBotAssistantApi.json("/access", "GET");
    } catch (error) {
      document.body.dataset.access = "denied";
      if (error.status === 401) return null;
      toast(error.message || "无法读取访问权限", "error");
      return null;
    }
    const permissions = new Set(access.permissions || []);
    pruneNavigation(permissions);
    if (!permissions.has("assistant:use")) {
      document.body.dataset.access = "denied";
      toast("没有智能工作台访问权限", "error");
      KBotAssistantAuth.clear();
      location.replace("./login.html");
      return null;
    }
    const required = PAGE_PERMISSIONS[page];
    if (page && page !== "dashboard" && required && !permissions.has(required)) {
      location.replace("./dashboard.html");
      return access;
    }
    document.body.dataset.access = "ready";
    return access;
  }

  function startReady(resolve, reject) {
    initialize().then(resolve, reject);
  }

  globalThis.KBotAssistantShell = {
    PAGE_PERMISSIONS,
    badge,
    capability,
    capabilityLabel,
    escapeHtml,
    ready: new Promise((resolve, reject) => {
      if (document.readyState === "loading") {
        addEventListener("DOMContentLoaded", () => startReady(resolve, reject), { once: true });
      } else {
        startReady(resolve, reject);
      }
    }),
    toast,
  };
})();
