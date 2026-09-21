/* 知识检索公共 Shell。 */
(function () {
  "use strict";
  const PAGE_PERMISSIONS = {"dashboard": "knowledge_retrieval:use", "knowledge": "knowledge_retrieval:knowledge_chat", "x-search": "knowledge_retrieval:x_search", "domains": "knowledge_retrieval:domain_manage", "knowledge-cores": "knowledge_retrieval:knowledge_core_manage", "data-models": "knowledge_retrieval:data_model_manage", "agents": "knowledge_retrieval:agent_manage"};
  const sections = [["知识工作区", [["dashboard", "工作台", "./dashboard.html"], ["knowledge", "知识问答", "./knowledge.html"], ["x-search", "X 实时搜索", "./x-search.html"]]], ["资源配置", [["domains", "Domains", "./domains.html"], ["knowledge-cores", "Knowledge Cores", "./knowledge-cores.html"], ["data-models", "问数模型", "./data-models.html"], ["agents", "Agents", "./agents.html"]]]];
  const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('\"', '&quot;');
  function toast(message, kind = "info") {
    const region = document.getElementById("knowledge-toast-region");
    if (!region) return;
    const item = document.createElement("div"); item.className = `knowledge-toast ${kind === "error" ? "error" : ""}`; item.textContent = message;
    region.append(item); setTimeout(() => item.remove(), 4500);
  }
  function badge(value, tone = "") { return `<span class="knowledge-badge ${escapeHtml(tone)}">${escapeHtml(value)}</span>`; }
  function capability(access, key) { return access?.capabilities?.[key] || { bound: false, verified: false, ready: false }; }
  function capabilityLabel(item) { return item.ready ? ["已就绪", "good"] : [item.bound ? "未验收" : "未绑定", "warn"]; }
  function shellMarkup(session, current) {
    const navigation = sections.map(([title, pages]) => {
      const links = pages.map(([id, label, href]) => `<a href="${href}" data-permission="${PAGE_PERMISSIONS[id]}"${id === current ? ' aria-current=\"page\"' : ""}>${escapeHtml(label)}</a>`).join("");
      return `<div class="knowledge-nav-label">${escapeHtml(title)}</div><nav class="knowledge-nav">${links}</nav>`;
    }).join("");
    return `<aside class="knowledge-sidebar"><a class="knowledge-brand" href="./dashboard.html"><span class="knowledge-brand-mark">KR</span><span><strong>知识检索</strong><small>KBot 4.0</small></span></a>${navigation}</aside><header class="knowledge-topbar"><div class="knowledge-context"><small>当前工作域</small><strong>知识检索</strong></div><div class="knowledge-session"><span>${escapeHtml(session.domain_name || "default")}</span><span class="knowledge-session-user">${escapeHtml(session.display_name || session.user_id || "已登录")}</span><button class="small" id="knowledge-logout">退出</button></div></header><div id="knowledge-toast-region" class="knowledge-toast-region" aria-live="polite"></div>`;
  }
  function mount(session, page) {
    if (document.querySelector(".knowledge-sidebar")) return;
    document.body.insertAdjacentHTML("afterbegin", shellMarkup(session, page));
    document.getElementById("knowledge-logout")?.addEventListener("click", () => { KBotKnowledgeAuth.clear(); location.replace("./login.html"); });
  }
  async function initialize() {
    const page = document.body.dataset.page || "";
    if (page === "login" || document.body.classList.contains("knowledge-login")) return null;
    const session = globalThis.KBotKnowledgeAuth?.load?.();
    if (!session?.access_token || session.must_change_password) { location.replace("./login.html"); return null; }
    mount(session, page);
    let access;
    try { access = await KBotKnowledgeApi.json("/access", "GET"); } catch (error) { toast(error.message || "无法读取访问权限", "error"); return null; }
    const granted = new Set(access.permissions || []);
    document.querySelectorAll("[data-permission]").forEach((node) => { if (!granted.has(node.dataset.permission)) node.remove(); });
    if (!granted.has("knowledge_retrieval:use")) { KBotKnowledgeAuth.clear(); location.replace("./login.html"); return null; }
    const required = PAGE_PERMISSIONS[page]; if (required && !granted.has(required)) location.replace("./dashboard.html");
    document.body.dataset.access = "ready"; return access;
  }
  globalThis.KBotKnowledgeShell = { PAGE_PERMISSIONS, badge, capability, capabilityLabel, escapeHtml, toast, ready: new Promise((resolve, reject) => { const start=()=>initialize().then(resolve,reject); document.readyState === "loading" ? addEventListener("DOMContentLoaded", start, {once:true}) : start(); }) };
})();
