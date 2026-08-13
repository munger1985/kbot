/* KM 页面公共 Shell、可信用户展示和无框架 UI 工具。 */
(function () {
  "use strict";

  const pages = [
    ["dashboard", "工作台", "../km/dashboard.html"],
    ["metadb", "MetaDB 数据", "../km/metadb.html"],
    ["sources", "数据来源", "../km/sources.html"],
    ["assets", "Asset 处理", "../km/assets.html"],
    ["jobs", "同步任务", "../km/jobs.html"],
    ["agents", "KM Agent", "../km/agents.html"],
    ["chat", "智能问答", "../km/chat.html"],
  ];
  let access = null;

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatDate(value) {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
  }

  function shortId(value) {
    const text = String(value || "");
    return text.length > 18 ? `${text.slice(0, 8)}…${text.slice(-6)}` : text || "—";
  }

  function badge(value) {
    const text = String(value || "UNKNOWN").toUpperCase();
    const tone = ["ACTIVE", "READY", "SUCCEEDED", "Y", "COMPLETED", "AUTO_ON"].includes(text)
      ? "good" : ["FAILED", "ERROR", "F", "REJECTED"].includes(text)
        ? "bad" : ["RUNNING", "PROCESSING", "PENDING", "RETRY_WAIT", "N"].includes(text)
          ? "warn" : "neutral";
    return `<span class="km-badge ${tone}">${escapeHtml(text)}</span>`;
  }

  function renderEmpty(tbody, columns, message) {
    tbody.innerHTML = `<tr><td class="km-empty" colspan="${columns}">${escapeHtml(message)}</td></tr>`;
  }

  function setBusy(button, busy, text) {
    if (!button) return;
    if (busy) {
      button.dataset.label = button.textContent;
      button.textContent = text || "处理中…";
      button.disabled = true;
    } else {
      button.textContent = button.dataset.label || button.textContent;
      button.disabled = false;
    }
  }

  function toast(message, kind = "info") {
    const region = document.getElementById("km-toast-region");
    if (!region) return;
    const item = document.createElement("div");
    item.className = `km-toast ${kind}`;
    item.textContent = message;
    region.appendChild(item);
    setTimeout(() => item.remove(), 5000);
  }

  function showError(error, fallback = "操作失败") {
    const request = error?.requestId ? ` · request_id: ${error.requestId}` : "";
    toast(`${error?.message || fallback}${request}`, "error");
  }

  function openDialog(id) { document.getElementById(id)?.showModal(); }
  function closeDialog(id) { document.getElementById(id)?.close(); }

  function shellMarkup() {
    const current = document.body.dataset.page || "";
    return `
      <aside class="km-sidebar">
        <a class="km-brand" href="./dashboard.html" aria-label="KM Asset 首页">
          <span class="km-brand-mark">KM</span><span><strong>Asset Desk</strong><small>KBot 4.0</small></span>
        </a>
        <nav class="km-nav" aria-label="KM 工作域">
          ${pages.map(([id, label, href]) => `<a href="${href}" ${id === current ? 'aria-current="page"' : ""}>${label}</a>`).join("")}
        </nav>
        <div class="km-sidebar-note">Collection 创建与文件上传继续使用现有 APEX 页面。</div>
      </aside>
      <header class="km-topbar">
        <div><span class="km-context-label">当前工作域</span><strong>KM Asset</strong></div>
        <div class="km-session"><span id="km-domain">Domain —</span><span class="km-session-user" id="km-user">验证用户中…</span><button id="km-logout" class="small">退出登录</button></div>
      </header>
      <div id="km-toast-region" class="km-toast-region" aria-live="polite"></div>
      `;
  }

  async function initialize() {
    KBotKmAuth.requireSession();
    document.body.insertAdjacentHTML("afterbegin", shellMarkup());
    document.querySelectorAll("dialog [data-close]").forEach((button) => {
      button.addEventListener("click", () => button.closest("dialog")?.close());
    });
    document.getElementById("km-logout").addEventListener("click", () => {
      KBotKmAuth.clearSession();
      location.replace("./login.html");
    });
    try {
      access = await KBotKmApi.request("/api/v1/apps/km-asset/access");
      document.getElementById("km-domain").textContent = `Domain ${access.domain_id}`;
      document.getElementById("km-user").textContent = access.user_id;
      document.body.dataset.access = "ready";
      return access;
    } catch (error) {
      document.body.dataset.access = "denied";
      document.getElementById("km-user").textContent = "未通过 KM 用户验证";
      showError(error, "当前登录用户没有 KM Asset 权限");
      throw error;
    }
  }

  const ready = new Promise((resolve, reject) => {
    window.addEventListener("DOMContentLoaded", () => initialize().then(resolve).catch(reject), { once: true });
  });

  globalThis.KBotKmShell = {
    badge, closeDialog, escapeHtml, formatDate, openDialog, ready,
    renderEmpty, setBusy, shortId, showError, toast,
    get access() { return access; },
  };
})();
