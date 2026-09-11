/* 智能工作台测试页公共 Shell；真实鉴权与访问契约接入后替换测试上下文。 */
(function () {
  "use strict";

  const sections = [
    ["业务工作区", [
      ["dashboard", "工作台", "./dashboard.html"],
      ["knowledge", "知识问答", "./knowledge.html"],
      ["x-search", "X 实时搜索", "./x-search.html"],
      ["image-generation", "文生图", "./image-generation.html"],
    ]],
    ["资源配置", [
      ["domains", "Domains", "./domains.html"], ["knowledge-cores", "Knowledge Cores", "./knowledge-cores.html"],
      ["data-models", "问数模型", "./data-models.html"], ["agents", "Agents", "./agents.html"],
    ]],
    ["管理", [
      ["model-bindings", "模型绑定", "./model-bindings.html"], ["media-assets", "图片资产", "./media-assets.html"], ["usage-runs", "用量与运行记录", "./usage-runs.html"],
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

  function shellMarkup() {
    const current = document.body.dataset.page || "";
    const navigation = sections.map(([title, pages]) => {
      const links = pages.map(([id, label, href]) => href
        ? `<a href="${href}" ${id === current ? 'aria-current="page"' : ""}>${escapeHtml(label)}</a>`
        : `<button type="button" data-coming-soon="${escapeHtml(label)}" title="资源配置页将在下一阶段接入">${escapeHtml(label)}</button>`).join("");
      return `<div class="assistant-nav-label">${escapeHtml(title)}</div><nav class="assistant-nav" aria-label="${escapeHtml(title)}">${links}</nav>`;
    }).join("");
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
        <div class="assistant-session"><span id="assistant-domain">Domain 待接入</span><span class="assistant-session-user" id="assistant-user">测试页面</span></div>
      </header>
      <div id="assistant-toast-region" class="assistant-toast-region" aria-live="polite"></div>`;
  }

  function initialize() {
    document.body.insertAdjacentHTML("afterbegin", shellMarkup());
    document.querySelectorAll("[data-coming-soon]").forEach((button) => {
      button.addEventListener("click", () => toast(`${button.dataset.comingSoon} 页面将在资源配置阶段接入。`));
    });
  }

  globalThis.KBotAssistantShell = { badge, escapeHtml, ready: new Promise((resolve) => {
    addEventListener("DOMContentLoaded", () => { initialize(); resolve(); }, { once: true });
  }), toast };
})();
