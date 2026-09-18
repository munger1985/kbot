/* 智能工作台知识问答：复用 Agent Runtime 会话、Turn、事件和引用能力。 */
(function () {
  "use strict";

  const { escapeHtml, toast } = KBotAssistantShell;
  const $ = (id) => document.getElementById(id);
  const state = {
    agents: [],
    conversations: [],
    active: null,
    sending: false,
    runResults: new Map(),
  };
  const citationPattern = /\[((?:C|Q)\d+)\]/g;

  function selectedAgentId() { return $("knowledge-agent").value; }
  function selectedAgent() {
    return state.agents.find((row) => String(row.agent_id) === String(selectedAgentId())) || null;
  }
  function agentName(agentId) {
    return state.agents.find((row) => String(row.agent_id) === String(agentId))?.display_name || String(agentId || "未知 Agent").slice(0, 12);
  }
  function formatDate(value) {
    if (!value) return "";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
  }
  function showError(error, fallback) {
    const request = error?.requestId ? ` · 请求 ${error.requestId}` : "";
    toast(`${error?.message || fallback}${request}`, "error");
  }
  function setBusy(button, busy, label) {
    if (!button) return;
    if (busy) {
      button.dataset.originalText = button.textContent;
      button.textContent = label || "处理中…";
    } else if (button.dataset.originalText) {
      button.textContent = button.dataset.originalText;
      delete button.dataset.originalText;
    }
    button.disabled = busy;
  }
  function contentText(item) {
    const value = item?.content;
    if (typeof value === "string") return value;
    if (!value || typeof value !== "object") return "";
    return String(value.answer ?? value.text ?? value.input ?? value.content ?? "");
  }
  function renderMarkdown(value) {
    const template = document.createElement("template");
    template.innerHTML = KBotMarkdown.render(String(value || ""));
    const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (node.parentElement?.closest("code, pre, a, sup")) return;
      const text = node.textContent || "";
      citationPattern.lastIndex = 0;
      if (!citationPattern.test(text)) return;
      citationPattern.lastIndex = 0;
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      for (const match of text.matchAll(citationPattern)) {
        fragment.append(document.createTextNode(text.slice(cursor, match.index)));
        const marker = document.createElement("button");
        marker.type = "button";
        marker.className = "assistant-citation-marker";
        marker.textContent = match[1];
        marker.dataset.citationLabel = match[1];
        marker.setAttribute("aria-label", `引用 ${match[1]}`);
        fragment.append(marker);
        cursor = match.index + match[0].length;
      }
      fragment.append(document.createTextNode(text.slice(cursor)));
      node.replaceWith(fragment);
    });
    return template.innerHTML;
  }

  function renderAgentContext() {
    const agent = selectedAgent();
    const modelCount = Array.isArray(agent?.data_model_ids) ? agent.data_model_ids.length : 0;
    $("knowledge-context-agent").textContent = agent?.display_name || "尚未选择";
    $("knowledge-context-core").textContent = agent?.knowledge_core_id || "未绑定";
    $("knowledge-context-models").textContent = modelCount ? `${modelCount} 个已发布模型` : "未绑定";
    $("knowledge-agent-summary").innerHTML = agent
      ? `<strong>${escapeHtml(agent.display_name || "未命名 Agent")}</strong><p>v${escapeHtml(agent.version_no || 1)} · ${modelCount ? `问文、问数与对话（${modelCount} 个模型）` : "问文与对话"}</p>`
      : "<strong>尚无可用 Agent</strong><p>请先在 Agents 页面创建并启用 Agent。</p>";
    $("knowledge-new-conversation").disabled = !agent || state.sending;
    $("knowledge-input").disabled = !agent || state.sending;
    $("knowledge-send").disabled = !agent || state.sending;
    $("knowledge-input-hint").textContent = agent
      ? (state.active ? "Enter 换行，点击发送提交本轮问题。" : "可直接输入并发送，系统会自动创建会话。")
      : "请选择已启用 Agent。";
  }

  async function loadAgents() {
    const select = $("knowledge-agent");
    select.disabled = true;
    const payload = await KBotAssistantApi.request("/agents");
    state.agents = KBotAssistantApi.items(payload).filter((row) => row.status === "ACTIVE");
    const previous = selectedAgentId();
    select.innerHTML = state.agents.length
      ? state.agents.map((row) => `<option value="${escapeHtml(row.agent_id)}">${escapeHtml(row.display_name || row.agent_id)}</option>`).join("")
      : '<option value="">当前 Domain 没有已启用 Agent</option>';
    select.value = state.agents.some((row) => String(row.agent_id) === String(previous))
      ? previous
      : String(state.agents[0]?.agent_id || "");
    select.disabled = !state.agents.length;
    renderAgentContext();
    await loadConversations();
  }

  function renderConversationList() {
    const list = $("knowledge-conversation-list");
    if (!state.conversations.length) {
      list.innerHTML = '<li class="assistant-chat-empty">当前 Agent 暂无会话</li>';
      return;
    }
    list.innerHTML = state.conversations.map((row) => `
      <li class="assistant-conversation-row">
        <button class="assistant-conversation-select" type="button" data-conversation-id="${escapeHtml(row.conversation_id)}" aria-current="${String(state.active?.conversation_id) === String(row.conversation_id)}">
          <strong>${escapeHtml(row.title || "未命名会话")}</strong>
          <small>${escapeHtml(formatDate(row.last_active_at))}</small>
        </button>
        <button class="assistant-conversation-delete small" type="button" data-delete-conversation="${escapeHtml(row.conversation_id)}" aria-label="删除会话 ${escapeHtml(row.title || "未命名会话")}">删除</button>
      </li>`).join("");
  }

  function renderEmptyChat() {
    state.active = null;
    $("knowledge-title").textContent = "请选择或创建会话";
    $("knowledge-conversation-meta").textContent = selectedAgent()
      ? `当前 Agent：${agentName(selectedAgentId())}`
      : "会话由 Agent Runtime 持久化";
    $("knowledge-chat-stream").innerHTML = `<div class="assistant-chat-empty">${selectedAgent() ? "新建会话后即可开始问文、问数或普通对话。" : "当前没有可用 Agent。"}</div>`;
    renderConversationList();
    renderAgentContext();
  }

  async function loadConversations(selectId) {
    if (!selectedAgent()) {
      state.conversations = [];
      renderEmptyChat();
      return;
    }
    const rows = KBotAssistantApi.items(await KBotAssistantApi.request("/conversations?limit=50"));
    state.conversations = rows.filter((row) => String(row.agent_id) === String(selectedAgentId()));
    const targetId = selectId || (String(state.active?.agent_id) === String(selectedAgentId()) ? state.active?.conversation_id : "");
    const selected = state.conversations.find((row) => String(row.conversation_id) === String(targetId)) || state.conversations[0];
    if (selected) await selectConversation(selected);
    else renderEmptyChat();
  }

  async function createConversation() {
    if (!selectedAgent()) return null;
    const button = $("knowledge-new-conversation");
    setBusy(button, true, "创建中…");
    try {
      const created = await KBotAssistantApi.json("/conversations", "POST", {
        agent_id: selectedAgentId(),
        title: null,
        retention_policy: "DEFAULT",
      });
      await loadConversations(created.conversation_id);
      $("knowledge-input").focus();
      return state.active;
    } catch (error) {
      showError(error, "会话创建失败");
      return null;
    } finally {
      setBusy(button, false);
      renderAgentContext();
    }
  }

  async function selectConversation(row) {
    state.active = await KBotAssistantApi.request(`/conversations/${row.conversation_id}`);
    renderConversationList();
    $("knowledge-title").textContent = state.active.title || "未命名会话";
    $("knowledge-conversation-meta").textContent = `${agentName(state.active.agent_id)} · ${state.active.status} · v${state.active.row_version}`;
    renderAgentContext();
    await loadTurns();
  }

  async function deleteConversation(row) {
    if (!globalThis.confirm(`确认永久删除会话“${row.title || "未命名会话"}”吗？\n\n会话内容删除后无法恢复；正在执行的会话不能删除。`)) return;
    await KBotAssistantApi.request(`/conversations/${row.conversation_id}?expected_row_version=${row.row_version}`, { method: "DELETE" });
    if (String(state.active?.conversation_id) === String(row.conversation_id)) state.active = null;
    toast("会话已删除", "info");
    await loadConversations();
  }

  function messageMarkup(role, label, text, runId) {
    const content = role === "assistant" ? renderMarkdown(text) : escapeHtml(text);
    const run = runId ? ` data-run-id="${escapeHtml(runId)}"` : "";
    const references = runId ? `<div class="assistant-message-references" data-references-for="${escapeHtml(runId)}"></div>` : "";
    return `<div class="assistant-message ${role}"${run}><div class="assistant-message-meta">${escapeHtml(label)}</div><div class="assistant-message-content">${content}</div>${references}</div>`;
  }

  function referenceRows(source) {
    if (Array.isArray(source)) return source;
    return Array.isArray(source?.payload?.references) ? source.payload.references : [];
  }
  function renderReferences(runId, source) {
    if (!runId) return;
    const rows = referenceRows(source);
    state.runResults.set(String(runId), { payload: { references: rows } });
    const host = Array.from(document.querySelectorAll("[data-references-for]"))
      .find((node) => node.dataset.referencesFor === String(runId));
    if (!host || !rows.length) return;
    host.innerHTML = rows.map((row, index) => {
      const label = row.citation_label || `${row.reference_type === "QUERY" ? "Q" : "C"}${index + 1}`;
      const title = row.title || row.semantic_model_name || (row.reference_type === "DOCUMENT" ? "引用文档" : "查询结果");
      return `<button type="button" class="assistant-reference" data-run-id="${escapeHtml(runId)}" data-reference-label="${escapeHtml(label)}">${escapeHtml(label)} · ${escapeHtml(title)}</button>`;
    }).join("");
  }

  async function loadTurns() {
    if (!state.active) return renderEmptyChat();
    const payload = await KBotAssistantApi.request(`/conversations/${state.active.conversation_id}/turns?after=0&limit=200`);
    const turns = Array.isArray(payload?.turns) ? payload.turns : KBotAssistantApi.items(payload);
    const stream = $("knowledge-chat-stream");
    if (!turns.length) {
      stream.innerHTML = '<div class="assistant-chat-empty">这是一个新会话，请输入问题。</div>';
      return;
    }
    stream.innerHTML = turns.map((turn) => {
      const user = contentText(turn.user_item);
      const assistant = contentText(turn.assistant_item);
      const waiting = turn.status && turn.status !== "COMPLETED" ? `处理状态：${turn.status}` : "";
      return `${user ? messageMarkup("user", "你", user) : ""}${assistant ? messageMarkup("assistant", agentName(state.active.agent_id), assistant, turn.run_id) : (waiting ? messageMarkup("assistant", agentName(state.active.agent_id), waiting, turn.run_id) : "")}`;
    }).join("");
    turns.forEach((turn) => {
      const references = turn.assistant_item?.content?.references;
      if (turn.run_id && Array.isArray(references)) renderReferences(turn.run_id, references);
    });
    stream.scrollTop = stream.scrollHeight;
  }

  function appendPending(input) {
    const stream = $("knowledge-chat-stream");
    if (stream.querySelector(".assistant-chat-empty")) stream.innerHTML = "";
    stream.insertAdjacentHTML("beforeend", messageMarkup("user", "你", input));
    const wrapper = document.createElement("div");
    wrapper.className = "assistant-message assistant pending";
    wrapper.setAttribute("aria-busy", "true");
    wrapper.innerHTML = `<div class="assistant-message-meta">${escapeHtml(agentName(selectedAgentId()))}</div><div class="assistant-message-content" aria-live="polite">正在分析问题并选择问文、问数或普通对话路径…</div><div class="assistant-message-references"></div>`;
    stream.append(wrapper);
    stream.scrollTop = stream.scrollHeight;
    return {
      wrapper,
      content: wrapper.querySelector(".assistant-message-content"),
      references: wrapper.querySelector(".assistant-message-references"),
      answer: "",
      referencesReceived: false,
    };
  }

  function applyRunEvent(pending, item) {
    const data = item.json && typeof item.json === "object" ? item.json : {};
    const payload = data.payload && typeof data.payload === "object" ? data.payload : {};
    const eventType = item.type || data.event_type || "message";
    if (eventType === "answer.delta") {
      pending.answer += String(payload.delta || "");
      pending.content.innerHTML = renderMarkdown(pending.answer);
      $("knowledge-chat-stream").scrollTop = $("knowledge-chat-stream").scrollHeight;
      return;
    }
    if (eventType === "answer.completed" && Array.isArray(payload.references)) {
      pending.referencesReceived = true;
      renderReferences(pending.wrapper.dataset.runId, payload.references);
    }
    $("knowledge-progress").textContent = payload.public_summary || payload.summary || data.title || data.summary || eventType || "Agent 正在执行";
  }

  function isRetryable(error) {
    return error instanceof TypeError || [502, 503, 504].includes(Number(error?.status));
  }
  async function createTurn(input, idempotencyKey) {
    const path = `/conversations/${state.active.conversation_id}/turns`;
    const options = { headers: { "Idempotency-Key": idempotencyKey } };
    const payload = {
      input,
      expected_conversation_version: state.active.row_version,
      client_metadata: { source: "assistant-ui" },
    };
    try {
      return await KBotAssistantApi.json(path, "POST", payload, options);
    } catch (error) {
      if (!isRetryable(error)) throw error;
      $("knowledge-progress").textContent = "连接中断，正在确认已提交的 Turn";
      await new Promise((resolve) => window.setTimeout(resolve, 300));
      return KBotAssistantApi.json(path, "POST", payload, options);
    }
  }

  async function refreshActive() {
    if (state.active) state.active = await KBotAssistantApi.request(`/conversations/${state.active.conversation_id}`);
  }
  function updateActiveConversation() {
    if (!state.active) return;
    const index = state.conversations.findIndex((row) => String(row.conversation_id) === String(state.active.conversation_id));
    if (index >= 0) state.conversations[index] = state.active;
    else state.conversations.unshift(state.active);
    renderConversationList();
    $("knowledge-title").textContent = state.active.title || "未命名会话";
    $("knowledge-conversation-meta").textContent = `${agentName(state.active.agent_id)} · ${state.active.status} · v${state.active.row_version}`;
  }

  async function send(event) {
    event.preventDefault();
    const input = $("knowledge-input").value.trim();
    if (!input || !selectedAgent() || state.sending) return;
    if (!state.active && !(await createConversation())) return;
    const pending = appendPending(input);
    $("knowledge-input").value = "";
    state.sending = true;
    renderAgentContext();
    setBusy($("knowledge-send"), true, "处理中…");
    $("knowledge-progress").hidden = false;
    $("knowledge-progress").textContent = "正在创建 Turn";
    try {
      const receipt = await createTurn(input, KBotAssistantApi.requestId());
      if (receipt.run_id) {
        pending.wrapper.dataset.runId = String(receipt.run_id);
        pending.references.dataset.referencesFor = String(receipt.run_id);
        $("knowledge-progress").textContent = "Agent 正在执行";
        await KBotAssistantApi.stream(`/runs/${encodeURIComponent(receipt.run_id)}/events`, {
          lastEventId: receipt.event_cursor,
          onEvent: (item) => applyRunEvent(pending, item),
        });
        const run = await KBotAssistantApi.request(`/runs/${receipt.run_id}`);
        if (run.status !== "COMPLETED") {
          const error = new Error(run.error_message || `Run 执行结束但状态为 ${run.status}`);
          error.code = run.error_code || "ASSISTANT_RUN_NOT_COMPLETED";
          throw error;
        }
        if (!pending.answer || !pending.referencesReceived) {
          const result = await KBotAssistantApi.request(`/runs/${receipt.run_id}/result`);
          if (!pending.answer) {
            pending.answer = String(result?.payload?.answer || "");
            pending.content.innerHTML = renderMarkdown(pending.answer);
          }
          if (!pending.referencesReceived) renderReferences(receipt.run_id, result);
        }
        pending.wrapper.classList.remove("pending");
        pending.wrapper.setAttribute("aria-busy", "false");
      }
      await refreshActive();
      updateActiveConversation();
      if (!receipt.run_id) await loadTurns();
    } catch (error) {
      pending.wrapper.classList.remove("pending");
      pending.wrapper.classList.add("failed");
      pending.wrapper.setAttribute("aria-busy", "false");
      pending.content.textContent = error?.message || "本轮处理失败";
      showError(error, "对话请求失败");
      await refreshActive().catch(() => {});
      await loadTurns().catch(() => {});
    } finally {
      state.sending = false;
      $("knowledge-progress").hidden = true;
      setBusy($("knowledge-send"), false);
      renderAgentContext();
      $("knowledge-input").focus();
    }
  }

  async function resultForRun(runId) {
    let result = state.runResults.get(String(runId));
    if (!result) {
      result = await KBotAssistantApi.request(`/runs/${runId}/result`);
      state.runResults.set(String(runId), result);
    }
    return result;
  }
  async function openReference(runId, label) {
    const result = await resultForRun(runId);
    const references = referenceRows(result);
    const reference = references.find((row) => String(row.citation_label) === String(label));
    if (!reference) return toast("引用依据尚未加载", "error");
    if (reference.reference_type !== "DOCUMENT") {
      return toast(`${label} 为本轮问数查询结果，已体现在回答中。`, "info");
    }
    const preview = await KBotAssistantApi.request(`/runs/${runId}/references/${encodeURIComponent(label)}/preview`);
    $("knowledge-reference-title").textContent = preview.title || reference.title || "文档引用";
    const pages = preview.page_no ? ` · 第 ${preview.page_no}${preview.page_end && preview.page_end !== preview.page_no ? `–${preview.page_end}` : ""} 页` : "";
    $("knowledge-reference-meta").textContent = `${preview.citation_label} · ${preview.mime_type}${pages}`;
    $("knowledge-reference-body").innerHTML = `<p>该引用来自当前 Agent 绑定的 Knowledge Core。打开原文时仍会执行当前用户与 Domain 权限校验。</p><button class="primary" type="button" data-open-reference-content="${escapeHtml(preview.content_url)}" data-preview-type="${escapeHtml(preview.preview_type)}" data-page-no="${escapeHtml(preview.page_no || "")}">打开引用原文</button>`;
    $("knowledge-reference-dialog").showModal();
  }
  async function openReferenceContent(button) {
    const popup = window.open("about:blank", "_blank");
    try {
      const blob = await KBotAssistantApi.requestBlob(button.dataset.openReferenceContent);
      const url = URL.createObjectURL(blob);
      const target = button.dataset.previewType === "PDF" && button.dataset.pageNo ? `${url}#page=${button.dataset.pageNo}` : url;
      if (popup) popup.location.href = target;
      else window.open(target, "_blank");
      window.setTimeout(() => URL.revokeObjectURL(url), 10 * 60 * 1000);
    } catch (error) {
      popup?.close();
      showError(error, "引用原文打开失败");
    }
  }

  function bindEvents() {
    $("knowledge-agent").addEventListener("change", async () => {
      state.active = null;
      renderAgentContext();
      try { await loadConversations(); }
      catch (error) { showError(error, "会话列表加载失败"); renderEmptyChat(); }
    });
    $("knowledge-new-conversation").addEventListener("click", createConversation);
    $("knowledge-refresh").addEventListener("click", () => loadConversations(state.active?.conversation_id).catch((error) => showError(error, "会话列表刷新失败")));
    $("knowledge-form").addEventListener("submit", send);
    $("knowledge-conversation-list").addEventListener("click", (event) => {
      const deleteButton = event.target.closest("[data-delete-conversation]");
      if (deleteButton) {
        const row = state.conversations.find((item) => String(item.conversation_id) === deleteButton.dataset.deleteConversation);
        if (row) deleteConversation(row).catch((error) => showError(error, "会话删除失败"));
        return;
      }
      const button = event.target.closest("[data-conversation-id]");
      const row = state.conversations.find((item) => String(item.conversation_id) === button?.dataset.conversationId);
      if (row) selectConversation(row).catch((error) => showError(error, "会话读取失败"));
    });
    $("knowledge-chat-stream").addEventListener("click", (event) => {
      const copy = event.target.closest("[data-copy-code]");
      if (copy) return KBotMarkdown.copyCode(copy);
      const marker = event.target.closest("[data-citation-label]");
      if (marker) {
        const message = marker.closest("[data-run-id]");
        if (message?.dataset.runId) openReference(message.dataset.runId, marker.dataset.citationLabel).catch((error) => showError(error, "引用读取失败"));
        return;
      }
      const reference = event.target.closest("[data-reference-label]");
      if (reference) openReference(reference.dataset.runId, reference.dataset.referenceLabel).catch((error) => showError(error, "引用读取失败"));
    });
    $("knowledge-reference-dialog").addEventListener("click", (event) => {
      if (event.target.closest("[data-close-reference]")) $("knowledge-reference-dialog").close();
      const open = event.target.closest("[data-open-reference-content]");
      if (open) openReferenceContent(open);
    });
  }

  KBotAssistantShell.ready.then(async (access) => {
    if (!access) return;
    bindEvents();
    try { await loadAgents(); }
    catch (error) {
      showError(error, "知识问答工作区加载失败");
      $("knowledge-agent").innerHTML = '<option value="">Agent 列表加载失败</option>';
      renderEmptyChat();
    }
  }).catch(() => {});
})();
