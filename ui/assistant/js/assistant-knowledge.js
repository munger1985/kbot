/* 知识问答测试页：读取已启用 Agent；Runtime 未接入前不伪造会话或回答。 */
(function () {
  "use strict";
  const { escapeHtml, toast } = KBotAssistantShell;
  const state = { agents: [], selectedAgentId: "" };
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

  function selectedAgent() {
    return state.agents.find((row) => String(row.agent_id) === state.selectedAgentId) || null;
  }

  function renderAgentContext() {
    const agent = selectedAgent();
    document.getElementById("knowledge-context-agent").textContent = agent?.display_name || "尚未选择";
    document.getElementById("knowledge-context-core").textContent = agent?.knowledge_core_id || "未绑定";
    const modelCount = Array.isArray(agent?.data_model_ids) ? agent.data_model_ids.length : 0;
    document.getElementById("knowledge-context-models").textContent = modelCount ? `${modelCount} 个已发布模型` : "未绑定";
    document.getElementById("knowledge-intro-title").textContent = agent ? `当前 Agent：${agent.display_name}` : "等待选择 Agent";
    document.getElementById("knowledge-intro-copy").textContent = agent
      ? "Agent 已可选择；知识问答 Runtime 接入完成后将在其授权范围内创建会话。"
      : "当前 Domain 没有可用于知识问答的已启用 Agent。";
    document.getElementById("knowledge-agent-summary").innerHTML = agent
      ? `<div><strong>${escapeHtml(agent.display_name || "未命名 Agent")}</strong><p>v${escapeHtml(agent.version_no || 1)} · ${modelCount ? `问文与问数（${modelCount} 个模型）` : "问文与闲聊"}</p></div>`
      : "<div><strong>尚无可用 Agent</strong><p>请先在 Agents 页面创建并启用 Agent。</p></div>";
  }

  async function loadAgents() {
    const select = document.getElementById("knowledge-agent");
    select.disabled = true;
    const payload = await KBotAssistantApi.json("/agents", "GET");
    const rows = Array.isArray(payload) ? payload : KBotAssistantApi.items(payload);
    state.agents = rows.filter((row) => row.status === "ACTIVE");
    const previous = state.selectedAgentId;
    select.innerHTML = state.agents.length
      ? state.agents.map((row) => `<option value="${escapeHtml(row.agent_id)}">${escapeHtml(row.display_name || row.agent_id)}</option>`).join("")
      : '<option value="">当前 Domain 没有已启用 Agent</option>';
    state.selectedAgentId = state.agents.some((row) => String(row.agent_id) === previous)
      ? previous
      : String(state.agents[0]?.agent_id || "");
    select.value = state.selectedAgentId;
    select.disabled = !state.agents.length;
    renderAgentContext();
  }

  KBotAssistantShell.ready.then(async (access) => {
    if (!access) return;
    document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => selectMode(button.dataset.mode)));
    document.getElementById("knowledge-agent").addEventListener("change", (event) => {
      state.selectedAgentId = event.target.value;
      renderAgentContext();
    });
    document.getElementById("knowledge-form").addEventListener("submit", (event) => {
      event.preventDefault();
      toast("尚未连接 Agent Runtime，消息没有发送。", "error");
    });
    document.getElementById("knowledge-new-conversation").addEventListener("click", () => {
      toast(selectedAgent() ? "已选择 Agent，等待知识问答 Runtime 接入。" : "当前没有已启用 Agent。", selectedAgent() ? "info" : "error");
    });
    document.getElementById("knowledge-refresh").addEventListener("click", () => loadAgents().catch((error) => toast(error.message || "Agent 列表刷新失败", "error")));
    try {
      await loadAgents();
    } catch (error) {
      toast(error.message || "Agent 列表加载失败", "error");
      document.getElementById("knowledge-agent").innerHTML = '<option value="">Agent 列表加载失败</option>';
      renderAgentContext();
    }
  });
})();
