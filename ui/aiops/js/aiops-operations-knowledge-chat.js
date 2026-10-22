(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/operations-knowledge";
  const shell = globalThis.KBotAIOpsShell;
  const markdown = globalThis.KBotMarkdown;
  const state = { conversations: [], active: null, sending: false };

  function answerContext() {
    return {
      database_type: document.getElementById("knowledge-context-db").value || null,
      database_major_version: document.getElementById("knowledge-context-version").value.trim() || null,
      topology: document.getElementById("knowledge-context-topology").value.trim() || null,
      problem_class: null,
      component: null,
    };
  }

  function answerText(item) {
    const content = item?.content;
    if (typeof content === "string") return content;
    return String(content?.answer || content?.text || content?.input || "");
  }

  function referenceButtons(runId, references) {
    return (references || [])
      .filter((item) => item.reference_type === "DOCUMENT")
      .map((item) => `<button type="button" class="knowledge-reference" data-citation-run="${shell.escape(runId)}" data-citation-label="${shell.escape(item.citation_label)}">${shell.escape(item.citation_label)} · ${shell.escape(item.title || "原文引用")}</button>`)
      .join("");
  }

  function answerMessage(role, text, runId = "", references = []) {
    const content = role === "agent" ? markdown.render(String(text || "")) : `<p>${shell.escape(text)}</p>`;
    return `<article class="ops-message ${role}"${runId ? ` data-run-id="${shell.escape(runId)}"` : ""}>
      <div class="ops-avatar">${role === "user" ? "我" : "AI"}</div>
      <div class="ops-message-body">
        <div class="ops-message-content ops-result-markdown">${content}</div>
        <div class="knowledge-reference-list">${referenceButtons(runId, references)}</div>
      </div>
    </article>`;
  }

  function renderConversations() {
    const host = document.getElementById("knowledge-conversation-list");
    host.innerHTML = state.conversations.length
      ? state.conversations.map((item) => `<button type="button" class="ops-workspace-item" data-knowledge-conversation="${shell.escape(item.conversation_id)}" aria-current="${String(state.active?.conversation_id) === String(item.conversation_id)}"><strong>${shell.escape(item.title || "未命名问答会话")}</strong><small>${shell.escape(shell.fmt(item.last_active_at))}</small></button>`).join("")
      : '<p class="ops-empty">还没有知识问答会话。</p>';
  }

  function resetConversation() {
    if (state.sending) return;
    state.active = null;
    renderConversations();
    document.getElementById("knowledge-chat-title").textContent = "运维知识助手";
    document.getElementById("knowledge-chat-meta").textContent = "新会话将在发送第一条问题时创建。";
    document.getElementById("knowledge-chat-stream").innerHTML = '<p class="ops-empty">可以询问 Exadata 存储节点维护方法、状态查看命令或故障处理步骤。</p>';
    document.getElementById("knowledge-chat-input").focus();
  }

  async function loadConversations(selectId = "", selectFirst = false) {
    state.conversations = await KBotAIOpsAuth.request(`${api}/conversations?limit=200`);
    state.conversations.sort((left, right) => new Date(right.last_active_at).getTime() - new Date(left.last_active_at).getTime());
    renderConversations();
    const selected = state.conversations.find((item) => String(item.conversation_id) === String(selectId))
      || (selectFirst ? state.conversations[0] : null);
    if (selected) await selectConversation(selected.conversation_id);
  }

  async function createConversation() {
    const created = await KBotAIOpsAuth.request(`${api}/conversations`, {
      method: "POST",
      body: JSON.stringify({ title: null, retention_policy: "DEFAULT", business_context: answerContext() }),
    });
    state.active = created;
    state.conversations.unshift(created);
    renderConversations();
    return created;
  }

  async function selectConversation(conversationId) {
    if (state.sending) return;
    state.active = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(conversationId)}`);
    document.getElementById("knowledge-chat-title").textContent = state.active.title || "运维知识助手";
    document.getElementById("knowledge-chat-meta").textContent = `会话版本 ${state.active.row_version} · 只使用有权访问的已发布知识`;
    renderConversations();
    await loadTurns();
  }

  async function loadTurns() {
    if (!state.active) return;
    const page = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(state.active.conversation_id)}/turns?after=0&limit=200`);
    const turns = page.turns || [];
    const host = document.getElementById("knowledge-chat-stream");
    if (!turns.length) {
      host.innerHTML = '<p class="ops-empty">这是一个新会话，请输入运维知识问题。</p>';
      return;
    }
    host.innerHTML = turns.map((turn) => {
      const user = answerText(turn.user_item);
      const assistant = answerText(turn.assistant_item);
      const waiting = turn.status !== "COMPLETED" ? `处理状态：${turn.status}` : "";
      const references = turn.assistant_item?.content?.references || [];
      return `${user ? answerMessage("user", user) : ""}${assistant || waiting ? answerMessage("agent", assistant || waiting, turn.run_id, references) : ""}`;
    }).join("");
    host.scrollTop = host.scrollHeight;
  }

  function applyAnswerEvent(pending, item) {
    const data = item.data && typeof item.data === "object" ? item.data : {};
    const payload = data.payload && typeof data.payload === "object" ? data.payload : {};
    const eventType = item.event || data.event_type || "message";
    if (eventType === "answer.delta") {
      pending.answer += String(payload.delta || "");
      pending.content.innerHTML = markdown.render(pending.answer);
    }
    if (eventType === "answer.completed" && Array.isArray(payload.references)) {
      pending.references.innerHTML = referenceButtons(pending.runId, payload.references);
    }
    document.getElementById("knowledge-chat-status").textContent = payload.public_summary || payload.summary || "正在形成可追溯回答…";
    const host = document.getElementById("knowledge-chat-stream");
    host.scrollTop = host.scrollHeight;
  }

  async function sendAnswer(event) {
    event.preventDefault();
    if (state.sending) return;
    const input = document.getElementById("knowledge-chat-input").value.trim();
    if (!input) return;
    if (!document.getElementById("knowledge-context-bases").options.length) { shell.toast("请先在运维知识库管理页为知识问答 Agent 绑定知识库"); return; }
    if (!state.active) await createConversation();
    state.sending = true;
    const button = document.getElementById("knowledge-chat-send");
    button.disabled = true;
    const host = document.getElementById("knowledge-chat-stream");
    if (host.querySelector(".ops-empty")) host.innerHTML = "";
    host.insertAdjacentHTML("beforeend", answerMessage("user", input));
    const pendingNode = document.createElement("article");
    pendingNode.className = "ops-message agent is-typing";
    pendingNode.innerHTML = '<div class="ops-avatar">AI</div><div class="ops-message-body"><div class="ops-message-content ops-result-markdown">正在检索已发布原文…</div><div class="knowledge-reference-list"></div></div>';
    host.append(pendingNode);
    host.scrollTop = host.scrollHeight;
    document.getElementById("knowledge-chat-input").value = "";
    const pending = { answer: "", runId: "", content: pendingNode.querySelector(".ops-message-content"), references: pendingNode.querySelector(".knowledge-reference-list") };
    try {
      const receipt = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(state.active.conversation_id)}/turns`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify({ input, expected_conversation_version: state.active.row_version, business_context: answerContext(), client_metadata: { source: "aiops-operations-knowledge-chat" } }),
      });
      pending.runId = receipt.run_id || "";
      if (!receipt.run_id) throw new Error("服务未返回运行标识");
      await KBotAIOpsAuth.stream(`${api}/runs/${encodeURIComponent(receipt.run_id)}/events`, (item) => applyAnswerEvent(pending, item), {
        headers: receipt.event_cursor ? { "Last-Event-ID": String(receipt.event_cursor) } : {},
      });
      const run = await KBotAIOpsAuth.request(`${api}/runs/${encodeURIComponent(receipt.run_id)}`);
      if (run.status !== "COMPLETED") throw new Error(run.error_message || `知识问答执行状态为 ${run.status}`);
      const result = await KBotAIOpsAuth.request(`${api}/runs/${encodeURIComponent(receipt.run_id)}/result`);
      const payload = result.payload || {};
      if (!pending.answer) pending.content.innerHTML = markdown.render(payload.answer || "没有形成可展示答案。");
      pending.references.innerHTML = referenceButtons(receipt.run_id, payload.references || []);
      if (["NO_EVIDENCE", "INSUFFICIENT_EVIDENCE"].includes(payload.status)) pending.content.insertAdjacentHTML("beforeend", '<p class="knowledge-warning">当前发布资料不足以支持答案，系统已停止生成。</p>');
      state.active = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(state.active.conversation_id)}`);
      await loadConversations(state.active.conversation_id);
    } catch (error) {
      pendingNode.classList.add("failed");
      pending.content.textContent = error.message || "知识问答失败";
      shell.toast(error.message || "知识问答失败");
    } finally {
      pendingNode.classList.remove("is-typing");
      state.sending = false;
      button.disabled = false;
      document.getElementById("knowledge-chat-status").textContent = "可以补充数据库环境以缩小适用范围；结论请通过引用原文核验。";
    }
  }

  async function openCitation(runId, label) {
    const preview = await KBotAIOpsAuth.request(`${api}/runs/${encodeURIComponent(runId)}/references/${encodeURIComponent(label)}/preview`);
    document.getElementById("citation-title").textContent = `${preview.citation_label} · ${preview.title}`;
    document.getElementById("citation-subtitle").textContent = [preview.preview_type, preview.page_no ? `第 ${preview.page_no} 页` : ""].filter(Boolean).join(" · ");
    document.getElementById("citation-body").innerHTML = markdown.render(`> ${preview.quote || ""}\n\n${preview.content_text || "没有可展示的原文文本，可下载原始文件核验。"}`);
    const button = document.getElementById("citation-download");
    button.disabled = !preview.download_available;
    button.onclick = () => KBotAIOpsAuth.download(preview.content_url, `${preview.title || label}`, "*/*").catch((error) => shell.toast(error.message));
    document.getElementById("citation-dialog").showModal();
  }

  shell.ready.then(async () => {
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    const workspace = document.getElementById("knowledge-chat-workspace");
    const railToggle = document.getElementById("knowledge-chat-rail-toggle");
    railToggle.onclick = () => {
      const collapsed = !workspace.classList.contains("is-rail-collapsed");
      workspace.classList.toggle("is-rail-collapsed", collapsed);
      railToggle.setAttribute("aria-expanded", String(!collapsed));
      railToggle.setAttribute("aria-label", collapsed ? "展开知识问答会话" : "收起知识问答会话");
    };
    document.getElementById("knowledge-new-conversation").onclick = resetConversation;
    document.getElementById("knowledge-chat-form").addEventListener("submit", (event) => sendAnswer(event).catch((error) => shell.toast(error.message)));
    document.getElementById("knowledge-chat-input").addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
      event.preventDefault();
      document.getElementById("knowledge-chat-form").requestSubmit();
    });
    document.getElementById("knowledge-conversation-list").onclick = (event) => {
      const button = event.target.closest("[data-knowledge-conversation]");
      if (button) selectConversation(button.dataset.knowledgeConversation).catch((error) => shell.toast(error.message));
    };
    document.getElementById("knowledge-chat-stream").onclick = (event) => {
      const button = event.target.closest("[data-citation-run]");
      if (button) openCitation(button.dataset.citationRun, button.dataset.citationLabel).catch((error) => shell.toast(error.message));
    };
    try {
      const binding = await KBotAIOpsAuth.request(`${api}/knowledge-answering/binding`);
      const bases = binding.knowledge_bases || [];
      document.getElementById("knowledge-context-bases").innerHTML = bases.map((item) => `<option value="${shell.escape(item.knowledge_base_id)}" selected>${shell.escape(item.display_name)}</option>`).join("");
      await loadConversations("", true);
    } catch (error) { shell.toast(error.message); }
  });
})();
