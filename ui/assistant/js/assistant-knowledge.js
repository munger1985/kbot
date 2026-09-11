/* 知识问答测试页：仅演示模式切换，不在后端未接入时伪造 Agent 或回答。 */
(function () {
  "use strict";
  const modes = {
    auto: ["智能选择", "在 Agent 已启用的知识、问数与普通对话能力中选择实际执行路径。"],
    document: ["问文", "只允许检索 Agent 绑定的 Knowledge Core；最终回答使用 [C] 文档引用。"],
    data: ["问数", "只允许使用 Agent 绑定且已发布的问数模型；最终结果使用 [Q] 查询结果引用。"],
    conversation: ["闲聊", "仅执行普通对话，不读取 Knowledge Core 或问数模型。"],
  };
  function selectMode(mode) {
    document.querySelectorAll("[data-mode]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
    document.getElementById("knowledge-mode-help").textContent = modes[mode][1];
    document.getElementById("knowledge-intro-title").textContent = `当前模式：${modes[mode][0]}`;
    document.getElementById("knowledge-intro-copy").textContent = modes[mode][1];
  }
  addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => selectMode(button.dataset.mode)));
    document.getElementById("knowledge-form").addEventListener("submit", (event) => { event.preventDefault(); KBotAssistantShell.toast("尚未连接 Agent Runtime，消息没有发送。", "error"); });
    ["knowledge-new-conversation", "knowledge-refresh"].forEach((id) => document.getElementById(id).addEventListener("click", () => KBotAssistantShell.toast("等待知识问答 API 与 Agent 列表接入。")));
  }, { once: true });
})();
