/* 多媒体创作工作台公共 Shell。 */
(function () {
  "use strict";
  const PAGE_PERMISSIONS = {"dashboard": "media_studio:use", "image-generation": "media_studio:image_generate", "model-bindings": "media_studio:model_binding_manage", "media-assets": "media_studio:media_read", "usage-runs": "media_studio:run_read"};
  const sections = [["创作工作区", [["dashboard", "工作台", "./dashboard.html"], ["image-generation", "图片生成", "./image-generation.html"]]], ["管理", [["model-bindings", "模型绑定", "./model-bindings.html"], ["media-assets", "媒体资产", "./media-assets.html"], ["usage-runs", "运行记录", "./usage-runs.html"]]]];
  const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('\"', '&quot;');
  function toast(message, kind = "info") {
    const region = document.getElementById("media-toast-region");
    if (!region) return;
    const item = document.createElement("div"); item.className = `media-toast ${kind === "error" ? "error" : ""}`; item.textContent = message;
    region.append(item); setTimeout(() => item.remove(), 4500);
  }
  function badge(value, tone = "") { return `<span class="media-badge ${escapeHtml(tone)}">${escapeHtml(value)}</span>`; }
  function capability(access, key) { return access?.capabilities?.[key] || { bound: false, verified: false, ready: false }; }
  function capabilityLabel(item) { return item.ready ? ["已就绪", "good"] : [item.bound ? "未验收" : "未绑定", "warn"]; }
  function shellMarkup(session, current) {
    const navigation = sections.map(([title, pages]) => {
      const links = pages.map(([id, label, href]) => `<a href="${href}" data-permission="${PAGE_PERMISSIONS[id]}"${id === current ? ' aria-current=\"page\"' : ""}>${escapeHtml(label)}</a>`).join("");
      return `<div class="media-nav-label">${escapeHtml(title)}</div><nav class="media-nav">${links}</nav>`;
    }).join("");
    return `<aside class="media-sidebar"><a class="media-brand" href="./dashboard.html"><span class="media-brand-mark">MS</span><span><strong>多媒体创作工作台</strong><small>KBot 4.0</small></span></a>${navigation}</aside><header class="media-topbar"><div class="media-context"><small>当前工作域</small><strong>多媒体创作工作台</strong></div><div class="media-session"><span>${escapeHtml(session.domain_name || "default")}</span><span class="media-session-user">${escapeHtml(session.display_name || session.user_id || "已登录")}</span><button class="small" id="media-logout">退出</button></div></header><div id="media-toast-region" class="media-toast-region" aria-live="polite"></div>`;
  }
  function mount(session, page) {
    if (document.querySelector(".media-sidebar")) return;
    document.body.insertAdjacentHTML("afterbegin", shellMarkup(session, page));
    document.getElementById("media-logout")?.addEventListener("click", () => { KBotMediaAuth.clear(); location.replace("./login.html"); });
  }
  async function initialize() {
    const page = document.body.dataset.page || "";
    if (page === "login" || document.body.classList.contains("media-login")) return null;
    const session = globalThis.KBotMediaAuth?.load?.();
    if (!session?.access_token || session.must_change_password) { location.replace("./login.html"); return null; }
    mount(session, page);
    let access;
    try { access = await KBotMediaApi.json("/access", "GET"); } catch (error) { toast(error.message || "无法读取访问权限", "error"); return null; }
    const granted = new Set(access.permissions || []);
    document.querySelectorAll("[data-permission]").forEach((node) => { if (!granted.has(node.dataset.permission)) node.remove(); });
    if (!granted.has("media_studio:use")) { KBotMediaAuth.clear(); location.replace("./login.html"); return null; }
    const required = PAGE_PERMISSIONS[page]; if (required && !granted.has(required)) location.replace("./dashboard.html");
    document.body.dataset.access = "ready"; return access;
  }
  globalThis.KBotMediaShell = { PAGE_PERMISSIONS, badge, capability, capabilityLabel, escapeHtml, toast, ready: new Promise((resolve, reject) => { const start=()=>initialize().then(resolve,reject); document.readyState === "loading" ? addEventListener("DOMContentLoaded", start, {once:true}) : start(); }) };
})();
