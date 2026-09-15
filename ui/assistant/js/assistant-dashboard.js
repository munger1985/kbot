/* 工作台首页：按真实能力摘要更新入口，并按权限加载最近运行。 */
(function () {
  "use strict";

  const { badge, capability, capabilityLabel, escapeHtml, toast } = KBotAssistantShell;

  function setupNotice(access) {
    const items = [
      ["knowledge", "知识问答", "需要当前 Domain 绑定可用的知识回答模型，并由已启用 Agent 承接会话。"],
      ["x_search", "X 实时搜索", "需要管理员绑定已通过 X Search 能力验收的模型。"],
      ["image_generation", "文生图", "需要管理员绑定已通过图像生成能力验收的模型。"],
    ];
    const notices = items.map(([key, title, copy]) => {
      const item = capability(access, key);
      const [label, tone] = capabilityLabel(item);
      const kind = item.ready ? "" : " warn";
      return `<div class="assistant-notice${kind}">${escapeHtml(title)}：${escapeHtml(item.ready ? "当前能力已就绪。" : copy)} ${badge(label, tone)}</div>`;
    });
    const node = document.getElementById("dashboard-setup");
    if (node) node.innerHTML = notices.join("");
  }

  function paintCapabilities(access) {
    document.querySelectorAll("[data-capability]").forEach((node) => {
      const item = capability(access, node.dataset.capability);
      const [label, tone] = capabilityLabel(item);
      node.className = `assistant-badge ${tone}`;
      node.textContent = label;
    });
  }

  function kindLabel(kind) {
    if (kind === "X_SEARCH") return "X Search";
    if (kind === "IMAGE_GENERATION") return "文生图";
    return "知识问答";
  }

  function renderRuns(rows) {
    const node = document.getElementById("dashboard-runs");
    if (!node) return;
    if (!rows.length) {
      node.innerHTML = `<div class="assistant-empty"><div><strong>尚无可显示的运行</strong><p>登录后按权限加载真实运行记录；不会伪造会话、搜索或图片生成历史。</p></div></div>`;
      return;
    }
    node.innerHTML = `<div class="assistant-list">${rows.slice(0, 8).map((row) => `
      <div class="assistant-history-item">
        <strong>${escapeHtml(kindLabel(row.kind))}</strong>
        <small>${escapeHtml(row.status)} · ${escapeHtml(row.model_display_name || "已绑定模型")} · ${escapeHtml(row.created_at || "")}</small>
      </div>`).join("")}</div>`;
  }

  KBotAssistantShell.ready.then(async (access) => {
    if (!access) return;
    paintCapabilities(access);
    setupNotice(access);
    const permissions = new Set(access.permissions || []);
    if (!permissions.has("assistant:run_read")) return;
    try {
      const rows = await KBotAssistantApi.json("/runs", "GET");
      renderRuns(Array.isArray(rows) ? rows : []);
    } catch (error) {
      toast(error.message || "无法加载最近运行", "error");
    }
  });
})();
