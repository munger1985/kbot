/* 知识检索知识问答：复用 Agent Runtime 会话、Turn、事件和引用能力。 */
(function () {
  "use strict";

  const { escapeHtml, icon, toast } = KBotKnowledgeShell;
  const $ = (id) => document.getElementById(id);
  const state = {
    agents: [],
    conversations: [],
    active: null,
    sending: false,
    runResults: new Map(),
  };
  const citationPattern = /\[((?:D|Q)\d+)\]/g;
  const userIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM5 20v-1a7 7 0 0 1 14 0v1"/></svg>';
  const narrowChat = matchMedia("(max-width: 900px)");
  const messageClock = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" });
  let menu = null;
  let menuTrigger = null;

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
  function formatMessageTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    const time = messageClock.format(date);
    if (date.toDateString() === new Date().toDateString()) return time;
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${date.getFullYear()}-${month}-${day} ${time}`;
  }
  function showError(error, fallback) {
    const request = error?.requestId ? ` · 请求 ${error.requestId}` : "";
    toast(`${error?.message || fallback}${request}`, "error");
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
        marker.className = "knowledge-citation-marker";
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
    $("knowledge-agent").disabled = !state.agents.length || state.sending;
    $("knowledge-refresh").disabled = state.sending;
    $("knowledge-new-conversation").disabled = !agent || state.sending;
    $("knowledge-input").disabled = !agent || state.sending;
    $("knowledge-query-images").disabled = !agent?.library_id || state.sending;
    $("knowledge-add-images").disabled = $("knowledge-query-images").disabled;
    $("knowledge-clear-images").disabled = state.sending;
    $("knowledge-send").disabled = !agent || state.sending || !$("knowledge-input").value.trim();
    $("knowledge-send").ariaLabel = $("knowledge-send").title = state.sending ? "正在生成回答" : "发送问题";
    $("knowledge-input-hint").textContent = agent
      ? state.sending ? "正在生成回答…" : state.active ? "在当前会话中继续提问。" : "发送第一条问题后自动创建会话。"
      : "请选择已启用 Agent。";
  }

  function renderImages() {
    const files = Array.from($("knowledge-query-images").files || []);
    $("knowledge-chat-attachments").hidden = !files.length;
    $("knowledge-chat-attachment-names").textContent = files.map(file => file.name).join("、");
  }
  function clearImages() {
    $("knowledge-query-images").value = "";
    renderImages();
  }

  async function loadAgents() {
    const select = $("knowledge-agent");
    select.disabled = true;
    const payload = await KBotKnowledgeApi.request("/agents");
    state.agents = KBotKnowledgeApi.items(payload).filter((row) => row.status === "ACTIVE");
    $("knowledge-workspace").hidden = !state.agents.length;
    if (!state.agents.length && !$("knowledge-no-access")) {
      $("knowledge-workspace").insertAdjacentHTML("afterend", '<section class="empty-state knowledge-chat-no-access" id="knowledge-no-access"><h2>没有可用于问答的 Agent</h2><p>请先在 Agent 管理中启用一个已绑定 ACTIVE 知识库的 Agent。</p></section>');
    }
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
    closeConversationMenu();
    $("knowledge-session-count").textContent = state.conversations.length + " 条记录";
    if (!state.conversations.length) {
      list.innerHTML = `<div class="knowledge-session-empty">${icon("agents")}<strong>还没有会话</strong><span>发送第一条消息后，会话会保存在这里。</span></div>`;
      return;
    }
    const now = new Date();
    const day = offset => new Date(now.getFullYear(), now.getMonth(), now.getDate() + offset).getTime();
    const [tomorrow, today, yesterday, week] = [1, 0, -1, -6].map(day);
    const groups = new Map(["未来", "今天", "昨天", "最近 7 天", "更早", "时间未知"].map(label => [label, []]));
    for (const row of state.conversations) {
      const time = row.last_active_at ? Date.parse(row.last_active_at) : NaN;
      const label = Number.isNaN(time) ? "时间未知" : time >= tomorrow ? "未来" : time >= today ? "今天"
        : time >= yesterday ? "昨天" : time >= week ? "最近 7 天" : "更早";
      groups.get(label).push(row);
    }
    list.innerHTML = Array.from(groups, ([label, rows]) => rows.length ? `
      <section class="knowledge-session-group" aria-label="${label}"><h3>${label}</h3>${rows.map((row, index) => {
        const title = row.title || "未命名会话";
        const badge = Array.from(String(row.title || "").trim().replace(/\s+/g, "")).slice(0, 2).join("") || String(index + 1);
        const id = escapeHtml(row.conversation_id), name = escapeHtml(title), date = escapeHtml(formatDate(row.last_active_at));
        const current = String(state.active?.conversation_id) === String(row.conversation_id);
        return `<div class="knowledge-conversation-row">
        <button class="knowledge-conversation-select agent-chat-session-item${current ? " is-selected" : ""}" type="button" data-conversation-id="${id}" aria-label="${name}" title="${name}${date ? " · " + date : ""}"${current ? ' aria-current="true"' : ""}>
          <span class="agent-chat-session-badge" aria-hidden="true">${escapeHtml(badge)}</span>
          <span class="knowledge-session-copy"><strong>${name}</strong><small>${date}</small></span>
        </button>
        <button class="knowledge-conversation-more" type="button" data-session-more="${id}" aria-label="会话操作：${name}" aria-haspopup="menu" aria-expanded="false"><span aria-hidden="true">⋮</span></button>
      </div>`;
      }).join("")}</section>` : "").join("");
  }

  function closeConversationMenu(restoreFocus = false) {
    menu?.remove();
    menu = null;
    menuTrigger?.setAttribute("aria-expanded", "false");
    if (restoreFocus) menuTrigger?.focus({ preventScroll: true });
    menuTrigger = null;
  }
  function openConversationMenu(button) {
    if (menuTrigger === button) return closeConversationMenu(true);
    const row = state.conversations.find(item => String(item.conversation_id) === button.dataset.sessionMore);
    if (!row || !$("knowledge-retrieval-sessions").classList.contains("open")) return;
    closeConversationMenu();
    menuTrigger = button;
    button.setAttribute("aria-expanded", "true");
    const rect = button.getBoundingClientRect();
    menu = document.createElement("div");
    menu.className = "knowledge-conversation-menu";
    menu.setAttribute("role", "menu");
    menu.setAttribute("aria-label", button.ariaLabel);
    menu.tabIndex = -1;
    menu.style.left = `${Math.max(8, Math.min(rect.right - 144, innerWidth - 152))}px`;
    menu.style.top = `${rect.bottom + 58 > innerHeight ? Math.max(8, rect.top - 58) : rect.bottom + 6}px`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.setAttribute("role", "menuitem");
    remove.textContent = "删除会话";
    remove.disabled = state.sending;
    remove.onclick = () => {
      closeConversationMenu(true);
      deleteConversation(row).catch(error => showError(error, "会话删除失败"));
    };
    menu.append(remove);
    document.body.append(menu);
    (remove.disabled ? menu : remove).focus({ preventScroll: true });
  }

  function renderEmptyChat() {
    state.active = null;
    $("knowledge-chat-stream").innerHTML = selectedAgent() ? `<div class="knowledge-chat-welcome"><span class="knowledge-welcome-mark">${icon("knowledge")}</span><p class="eyebrow">${escapeHtml(selectedAgent().display_name)}</p><h2>从问题开始问文或问数</h2><p>${escapeHtml(selectedAgent().description || "输入业务问题。系统会按 Agent 绑定的知识库和问数模型检索，并保留可追溯的资料引用。")}</p></div>` : '<div class="empty-state knowledge-chat-no-access"><h2>没有可用于问答的 Agent</h2><p>请先在 Agent 管理中启用一个已绑定 ACTIVE 知识库的 Agent。</p></div>';
    renderConversationList();
    renderAgentContext();
  }

  async function loadConversations(selectId) {
    if (!selectedAgent()) {
      state.conversations = [];
      renderEmptyChat();
      return;
    }
    const rows = KBotKnowledgeApi.items(await KBotKnowledgeApi.request("/conversations?limit=50"));
    state.conversations = rows.filter((row) => String(row.agent_id) === String(selectedAgentId()));
    const targetId = selectId || (String(state.active?.agent_id) === String(selectedAgentId()) ? state.active?.conversation_id : "");
    const selected = state.conversations.find((row) => String(row.conversation_id) === String(targetId)) || state.conversations[0];
    if (selected) await selectConversation(selected, Boolean(selectId));
    else renderEmptyChat();
  }

  async function createConversation() {
    if (!selectedAgent()) return null;
    try {
      const created = await KBotKnowledgeApi.json("/conversations", "POST", {
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
      renderAgentContext();
    }
  }

  async function selectConversation(row, duringCreate = false) {
    if (state.sending && !duringCreate) return;
    state.active = await KBotKnowledgeApi.request(`/conversations/${row.conversation_id}`);
    renderConversationList();
    renderAgentContext();
    await loadTurns();
  }

  async function deleteConversation(row) {
    if (state.sending) return;
    if (!await KBotKnowledgeShell.confirmAction({ title: "永久删除会话？", message: `确认永久删除会话“${row.title || "未命名会话"}”吗？\n\n会话内容删除后无法恢复；正在执行的会话不能删除。`, confirmLabel: "永久删除", tone: "danger" })) return;
    await KBotKnowledgeApi.request(`/conversations/${row.conversation_id}?expected_row_version=${row.row_version}`, { method: "DELETE" });
    if (String(state.active?.conversation_id) === String(row.conversation_id)) state.active = null;
    toast("会话已删除", "info");
    await loadConversations();
  }

  function messageMarkup(role, label, text, runId, createdAt, images) {
    const content = role === "knowledge" ? renderMarkdown(text) : escapeHtml(text);
    const run = runId ? ` data-run-id="${escapeHtml(runId)}"` : "";
    const clock = formatMessageTime(createdAt);
    const time = clock ? `<time datetime="${escapeHtml(createdAt)}" title="${escapeHtml(formatDate(createdAt))}">${clock}</time>` : "";
    const references = runId ? `<div class="knowledge-message-references" data-references-for="${escapeHtml(runId)}"></div>` : "";
    const attachments = role === "user" && Array.isArray(images) && images.length
      ? `<div aria-label="提问图片">${images.map(image => `<span class="knowledge-badge">${escapeHtml(image?.file_name || "图片附件")}</span>`).join(" ")}</div>` : "";
    return `<article class="knowledge-message ${role === "knowledge" ? "assistant" : role}"${run}><div class="chat-message-avatar" aria-hidden="true">${role === "user" ? userIcon : icon("agents")}</div><div class="knowledge-message-body"><header><strong>${escapeHtml(label)}</strong>${time}</header><div class="knowledge-message-content chat-message-bubble">${content}</div>${attachments}${references}</div></article>`;
  }

  function referenceRows(source) {
    if (Array.isArray(source)) return source;
    return Array.isArray(source?.payload?.references) ? source.payload.references : [];
  }
  function renderReferences(runId, source) {
    if (!runId) return;
    const rows = referenceRows(source);
    const composition = source?.payload?.knowledge_composition;
    state.runResults.set(String(runId), { payload: { references: rows, knowledge_composition: composition } });
    const host = Array.from(document.querySelectorAll("[data-references-for]"))
      .find((node) => node.dataset.referencesFor === String(runId));
    if (!host) return;
    host.innerHTML = composition
      ? `${composition.question_image_count > 0 ? `<span class="knowledge-badge">${questionImageStatus(composition)}</span>` : ""}<button type="button" class="secondary-button" data-open-retrieval aria-haspopup="dialog">查看检索结果 <span aria-hidden="true">›</span></button>`
      : "";
  }

  function questionImageStatus(value) {
    const count = Number(value.question_image_count);
    return value.question_image_kind === "OCR" ? `OCR 已识别 ${count} 张提问图片的文字` : `视觉模型收到 ${count} 张提问图片`;
  }

  let retrievalSession;
  const imageTypes = ["image/png", "image/jpeg", "image/webp"];
  function openRetrieval(runId) {
    const value = state.runResults.get(String(runId))?.payload?.knowledge_composition;
    if (!value) return;
    clearRetrieval();
    retrievalSession = { runId, abort: new AbortController(), urls: [] };
    const evidence = key => {
      const item = (value.items || []).find(row => row.key === key);
      if (!item) return "";
      const pages = [...new Set((item.locators || []).map(loc => loc.page_no).filter(page => Number.isSafeInteger(page) && page > 0))];
      return `<details class="knowledge-evidence-item" data-evidence-key="${escapeHtml(key)}">
        <summary>${imageTypes.includes(item.mime_type) ? "预览图片证据" : "预览文字证据"}${pages.length ? ` · 第 ${pages.join("、")} 页` : ""}</summary>
        <div class="knowledge-evidence-content"></div></details>`;
    };
    const hits = value.primary_hits || [];
    const delivery = { SENT: value.question_image_kind === "OCR" ? questionImageStatus(value) : value.question_image_count > 0
      ? `${Number(value.delivered_count) || 0} 项知识库资料 · ${Number(value.question_image_count)} 张提问图片`
      : `已交给 Agent ${Number(value.delivered_count) || 0} 项`, FAILED: "证据交给 Agent 失败",
      UNKNOWN_DELIVERY: "未确认哪些证据已交给 Agent", PREPARED: "证据已准备，尚未确认送达", NOT_APPLICABLE: "本轮未使用模型交付" };
    const coverage = { COMPLETE: "证据可支撑回答", PARTIAL: "证据只能支撑部分回答", INSUFFICIENT: "证据不足",
      UNKNOWN: value.question_image_kind === "OCR" ? "仅识别图中文字，未引用知识库资料"
        : value.question_image_count > 0 ? "根据提问图片回答，未引用知识库资料" : "无法确认覆盖程度" };
    $("knowledge-retrieval-body").innerHTML = `<dl class="knowledge-evidence-summary">
      <div><dt>主要结果</dt><dd>${hits.length} 项</dd></div><div><dt>交给 Agent</dt><dd>${escapeHtml(Object.hasOwn(delivery, value.model_delivery) ? delivery[value.model_delivery] : "未确认")}</dd></div>
      <div><dt>回答覆盖</dt><dd>${escapeHtml(Object.hasOwn(coverage, value.coverage_status) ? coverage[value.coverage_status] : "未确认")}</dd></div>
      </dl>${hits.length ? `<ol class="knowledge-evidence-hits">${hits.map(hit => {
        const group = (value.linked || []).find(row => row.anchor_key === hit.key);
        const keys = group?.linked_keys || [];
        return `<li><strong>结果 ${escapeHtml(hit.rank)}</strong>${evidence(hit.key)}
          ${keys.length || group?.incomplete_reason ? `<details class="knowledge-evidence-related"><summary>关联证据 · ${keys.length} 项</summary>${group.incomplete_reason ? "<p>部分关联证据未能加载。</p>" : ""}${keys.map(evidence).join("")}</details>` : ""}</li>`;
      }).join("")}</ol>` : "<p>本轮没有可展示的检索结果。</p>"}`;
    $("knowledge-retrieval-dialog").showModal();
  }

  function clearRetrieval() {
    if (!retrievalSession) return;
    retrievalSession.abort.abort();
    retrievalSession.urls.forEach(url => URL.revokeObjectURL(url));
    retrievalSession = null;
    $("knowledge-retrieval-body").replaceChildren();
  }

  async function readEvidence(details) {
    const session = retrievalSession, key = details.dataset.evidenceKey;
    if (!session || details.dataset.state === "ready" || details.dataset.state === "loading") return;
    const host = details.querySelector(".knowledge-evidence-content");
    const status = document.createElement("p");
    status.setAttribute("role", "status");
    status.textContent = "正在读取证据…";
    host.replaceChildren(status);
    details.dataset.state = "loading";
    try {
      const blob = await KBotKnowledgeApi.requestBlob(`/runs/${encodeURIComponent(session.runId)}/knowledge/content/${encodeURIComponent(key)}`,
        { signal: session.abort.signal });
      if (retrievalSession !== session) return;
      if (imageTypes.includes(blob.type)) {
        const image = document.createElement("img"), url = URL.createObjectURL(blob);
        session.urls.push(url);
        image.src = url; image.alt = "检索图片证据";
        host.replaceChildren(image);
      } else if (blob.type.startsWith("text/plain")) {
        const pre = document.createElement("pre"); pre.textContent = await blob.text();
        if (retrievalSession !== session) return;
        host.replaceChildren(pre);
      } else {
        status.textContent = "此证据不支持在列表内预览。";
      }
      details.dataset.state = "ready";
    } catch {
      if (retrievalSession !== session) return;
      details.dataset.state = "error";
      status.textContent = "证据暂不可读取，请检查当前访问权限或重试。";
      const retry = document.createElement("button");
      retry.type = "button"; retry.className = "secondary-button"; retry.textContent = "重试";
      retry.addEventListener("click", () => void readEvidence(details));
      host.append(retry);
    }
  }

  async function loadTurns() {
    if (!state.active) return renderEmptyChat();
    const payload = await KBotKnowledgeApi.request(`/conversations/${state.active.conversation_id}/turns?after=0&limit=200`);
    const turns = Array.isArray(payload?.turns) ? payload.turns : KBotKnowledgeApi.items(payload);
    const stream = $("knowledge-chat-stream");
    if (!turns.length) {
      stream.innerHTML = '<div class="knowledge-chat-empty">这是一个新会话，请输入问题。</div>';
      return;
    }
    stream.innerHTML = turns.map((turn) => {
      const user = contentText(turn.user_item);
      const knowledge = contentText(turn.assistant_item);
      const waiting = turn.status && turn.status !== "COMPLETED" ? `处理状态：${turn.status}` : "";
      const userAt = turn.user_item?.created_at ?? turn.created_at;
      const assistantAt = turn.assistant_item?.created_at ?? turn.completed_at;
      return `${user ? messageMarkup("user", "你", user, null, userAt, turn.user_item?.content?.images) : ""}${knowledge || waiting ? messageMarkup("knowledge", agentName(state.active.agent_id), knowledge || waiting, turn.run_id, assistantAt) : ""}`;
    }).join("");
    turns.forEach((turn) => {
      const content = turn.assistant_item?.content;
      if (turn.run_id && Array.isArray(content?.references)) renderReferences(turn.run_id, { payload: content });
    });
    stream.scrollTop = stream.scrollHeight;
  }

  function appendPending(input, createdAt, images) {
    const stream = $("knowledge-chat-stream");
    if (stream.querySelector(".knowledge-chat-empty, .knowledge-chat-welcome")) stream.innerHTML = "";
    stream.insertAdjacentHTML("beforeend", messageMarkup("user", "你", input, null, createdAt, images));
    const wrapper = document.createElement("div");
    wrapper.className = "knowledge-message assistant pending";
    wrapper.setAttribute("aria-busy", "true");
    wrapper.innerHTML = `<div class="chat-message-avatar" aria-hidden="true">${icon("agents")}</div><div class="knowledge-message-body"><header><strong>${escapeHtml(agentName(selectedAgentId()))}</strong><time hidden></time></header><div class="knowledge-message-content chat-message-bubble" aria-live="polite">正在分析问题并选择问文、问数或普通对话路径…</div><div class="knowledge-message-references"></div></div>`;
    stream.append(wrapper);
    stream.scrollTop = stream.scrollHeight;
    return {
      wrapper,
      time: wrapper.querySelector("time"),
      content: wrapper.querySelector(".knowledge-message-content"),
      references: wrapper.querySelector(".knowledge-message-references"),
      answer: "",
      preview: "",
      referencesReceived: false,
    };
  }

  function setReplyTime(pending, value) {
    const clock = formatMessageTime(value);
    if (!clock) return;
    pending.time.dateTime = value;
    pending.time.title = formatDate(value);
    pending.time.textContent = clock;
    pending.time.hidden = false;
  }

  function applyRunEvent(pending, item) {
    const data = item.json && typeof item.json === "object" ? item.json : {};
    const payload = data.payload && typeof data.payload === "object" ? data.payload : {};
    const eventType = item.type || data.event_type || "message";
    if (eventType === "answer.preview.delta") {
      if (payload.chunk_index === 1) pending.preview = "";
      pending.preview += String(payload.delta || "");
      if (pending.time.hidden && pending.preview) setReplyTime(pending, data.created_at ?? new Date().toISOString());
      pending.content.classList.add("knowledge-answer-preview");
      pending.content.textContent = pending.preview;
      $("knowledge-progress").textContent = "正在生成回答，引用核验后显示";
      $("knowledge-chat-stream").scrollTop = $("knowledge-chat-stream").scrollHeight;
      return;
    }
    if (eventType === "answer.delta") {
      if (pending.preview) $("knowledge-progress").textContent = "正在呈现最终回答";
      pending.preview = "";
      pending.content.classList.remove("knowledge-answer-preview");
      pending.answer += String(payload.delta || "");
      if (pending.time.hidden && pending.answer) setReplyTime(pending, data.created_at ?? new Date().toISOString());
      pending.content.innerHTML = renderMarkdown(pending.answer);
      $("knowledge-chat-stream").scrollTop = $("knowledge-chat-stream").scrollHeight;
      return;
    }
    if (eventType === "answer.completed") {
      setReplyTime(pending, data.created_at);
      if (Array.isArray(payload.references)) {
        pending.referencesReceived = true;
        renderReferences(pending.wrapper.dataset.runId, { payload });
      }
    }
    $("knowledge-progress").textContent = payload.public_summary || payload.summary || data.title || data.summary || eventType || "Agent 正在执行";
  }

  function isRetryable(error) {
    return error instanceof TypeError || [502, 503, 504].includes(Number(error?.status));
  }
  async function createTurn(input, idempotencyKey, images) {
    const path = `/conversations/${state.active.conversation_id}/turns`;
    const options = { headers: { "Idempotency-Key": idempotencyKey } };
    const payload = {
      input,
      expected_conversation_version: state.active.row_version,
      client_metadata: { source: "knowledge-ui" },
      images,
    };
    try {
      return await KBotKnowledgeApi.json(path, "POST", payload, options);
    } catch (error) {
      if (!isRetryable(error)) throw error;
      $("knowledge-progress").textContent = "连接中断，正在确认已提交的 Turn";
      await new Promise((resolve) => window.setTimeout(resolve, 300));
      return KBotKnowledgeApi.json(path, "POST", payload, options);
    }
  }

  async function refreshActive() {
    if (state.active) state.active = await KBotKnowledgeApi.request(`/conversations/${state.active.conversation_id}`);
  }
  function updateActiveConversation() {
    if (!state.active) return;
    const index = state.conversations.findIndex((row) => String(row.conversation_id) === String(state.active.conversation_id));
    if (index >= 0) state.conversations[index] = state.active;
    else state.conversations.unshift(state.active);
    renderConversationList();
  }

  async function send(event) {
    event.preventDefault();
    const input = $("knowledge-input").value.trim();
    if (!input || !selectedAgent() || state.sending) return;
    const submittedAt = new Date().toISOString();
    let images;
    try {
      const files = Array.from($("knowledge-query-images").files || []);
      if (files.length > 4 || files.reduce((sum, file) => sum + file.size, 0) > 8 * 1024 * 1024) throw new Error("查询图片最多 4 张，合计不超过 8 MiB");
      images = await Promise.all(files.map(async file => {
        if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) throw new Error("查询图片只支持 PNG、JPEG 或 WebP");
        const data = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = reject; reader.readAsDataURL(file); });
        return { file_name: file.name, mime_type: file.type, content_base64: data.slice(data.indexOf(",") + 1) };
      }));
    } catch (error) { showError(error, "查询图片读取失败"); return; }
    if (state.sending) return;
    state.sending = true;
    renderAgentContext();
    if (!state.active && !(await createConversation())) {
      state.sending = false; renderAgentContext(); return;
    }
    const pending = appendPending(input, submittedAt, images);
    $("knowledge-input").value = "";
    state.sending = true;
    renderAgentContext();
    closeConversationMenu();
    $("knowledge-progress").hidden = false;
    $("knowledge-progress").textContent = "正在创建 Turn";
    try {
      const receipt = await createTurn(input, KBotKnowledgeApi.requestId(), images);
      if (receipt.run_id) {
        pending.wrapper.dataset.runId = String(receipt.run_id);
        pending.references.dataset.referencesFor = String(receipt.run_id);
        $("knowledge-progress").textContent = "Agent 正在执行";
        await KBotKnowledgeApi.stream(`/runs/${encodeURIComponent(receipt.run_id)}/events`, {
          lastEventId: receipt.event_cursor,
          onEvent: (item) => applyRunEvent(pending, item),
        });
        const run = await KBotKnowledgeApi.request(`/runs/${receipt.run_id}`);
        setReplyTime(pending, run.completed_at);
        if (run.status !== "COMPLETED") {
          const error = new Error(run.error_message || `Run 执行结束但状态为 ${run.status}`);
          error.code = run.error_code || "KNOWLEDGE_RETRIEVAL_RUN_NOT_COMPLETED";
          throw error;
        }
        if (!pending.answer || !pending.referencesReceived) {
          const result = await KBotKnowledgeApi.request(`/runs/${receipt.run_id}/result`);
          if (!pending.answer) {
            pending.answer = String(result?.payload?.answer || "");
            pending.content.classList.remove("knowledge-answer-preview");
            pending.content.innerHTML = renderMarkdown(pending.answer);
          }
          if (!pending.referencesReceived) renderReferences(receipt.run_id, result);
        }
        if (pending.time.hidden) setReplyTime(pending, new Date().toISOString());
        pending.wrapper.classList.remove("pending");
        pending.wrapper.setAttribute("aria-busy", "false");
      }
      clearImages();
      await refreshActive();
      updateActiveConversation();
      if (!receipt.run_id) await loadTurns();
    } catch (error) {
      if (pending.time.hidden) setReplyTime(pending, new Date().toISOString());
      pending.wrapper.classList.remove("pending");
      pending.wrapper.classList.add("failed");
      pending.wrapper.setAttribute("aria-busy", "false");
      pending.content.classList.remove("knowledge-answer-preview");
      pending.content.textContent = error?.message || "本轮处理失败";
      showError(error, "对话请求失败");
      await refreshActive().catch(() => {});
      await loadTurns().catch(() => {});
    } finally {
      state.sending = false;
      $("knowledge-progress").hidden = true;
      renderAgentContext();
      $("knowledge-input").focus();
    }
  }

  async function resultForRun(runId) {
    let result = state.runResults.get(String(runId));
    if (!result) {
      result = await KBotKnowledgeApi.request(`/runs/${runId}/result`);
      state.runResults.set(String(runId), result);
    }
    return result;
  }
  let referenceSession;
  function clearReference() {
    if (!referenceSession) return;
    referenceSession.abort.abort();
    referenceSession.reader?.destroy();
    if (referenceSession.url) URL.revokeObjectURL(referenceSession.url);
    referenceSession = null;
    $("knowledge-reference-body").replaceChildren();
  }
  async function openReference(runId, label) {
    if (label.startsWith("Q")) {
      const rows = referenceRows(await resultForRun(runId));
      if (rows.some(row => row.citation_label === label && row.reference_type !== "APP_KNOWLEDGE"))
        toast(`${label} 为本轮问数查询结果，已体现在回答中。`, "info");
      else toast("引用依据尚未加载", "error");
      return;
    }
    clearReference();
    const session = referenceSession = { abort: new AbortController() };
    const dialog = $("knowledge-reference-dialog"), body = $("knowledge-reference-body");
    const current = () => referenceSession === session && dialog.open;
    const options = { signal: session.abort.signal };
    $("knowledge-reference-title").textContent = "引用原文";
    $("knowledge-reference-meta").textContent = `正文引用 ${label}`;
    body.innerHTML = '<p role="status">正在读取引用原文…</p>';
    if (!dialog.open) dialog.showModal();
    let excerpt = "", excerptImage;
    try {
      const preview = await KBotKnowledgeApi.request(`/runs/${encodeURIComponent(runId)}/references/${encodeURIComponent(label)}/preview`, options);
      if (!current()) return;
      const locators = preview.locators || [];
      const pages = [...new Set(locators.map(loc => loc.page_no).filter(page => Number.isSafeInteger(page) && page > 0))];
      $("knowledge-reference-title").textContent = preview.title || "引用原文";
      $("knowledge-reference-meta").textContent = `正文引用 ${label}${pages.length ? ` · 第 ${pages.join("、")} 页` : ""}`;
      const [evidence, pdf] = await Promise.allSettled([
        KBotKnowledgeApi.requestBlob(preview.content_url, options),
        pages.length && preview.pdf_url ? KBotKnowledgeApi.requestBlob(preview.pdf_url, options) : null,
      ]);
      if (evidence.value?.type.startsWith("text/plain")) excerpt = await evidence.value.text();
      else if (imageTypes.includes(evidence.value?.type)) excerptImage = evidence.value;
      if (!current()) return;
      if (pdf.value?.type === "application/pdf") {
        const module = await import(new URL("../vendor/pdf-search-reader.js", location.href).href).catch(() => null);
        if (!current()) return;
        if (module) {
          renderReferencePdf(session, module, pdf.value, pages[0], locators, excerpt);
          return;
        }
      }
      body.innerHTML = '<p role="status">原文暂不可预览，以下为本轮的引用证据。</p>';
      if (excerpt) appendReferenceExcerpt(body, excerpt);
      else if (excerptImage) {
        const image = document.createElement("img");
        session.url = URL.createObjectURL(excerptImage);
        image.src = session.url; image.alt = "引用图片证据";
        body.append(image);
      } else body.firstChild.textContent = "引用原文暂不可读取，可能已撤回、换版或不再有访问权限。";
    } catch (error) {
      if (!current()) return;
      const notice = document.createElement("p");
      notice.setAttribute("role", "alert");
      notice.textContent = error?.code === "AGENT_EVIDENCE_SERVICE_AUTH_FAILED"
        ? "引用服务暂不可用，请稍后重试。" : "引用原文暂不可读取，请检查当前访问权限或重试。";
      body.replaceChildren(notice);
    }
  }

  function appendReferenceExcerpt(host, excerpt) {
    const pre = document.createElement("pre");
    pre.className = "knowledge-evidence-text";
    pre.textContent = excerpt;
    host.append(pre);
  }

  function renderReferencePdf(session, module, blob, hitPage, locators, excerpt) {
    const body = $("knowledge-reference-body");
    body.innerHTML = `<div class="knowledge-reference-tools" role="group" aria-label="PDF 页码和缩放">
      <button type="button" class="icon-button" data-step="-1" aria-label="上一页">‹</button>
      <span data-page aria-live="polite"></span>
      <button type="button" class="icon-button" data-step="1" aria-label="下一页">›</button>
      <button type="button" class="secondary-button button-compact" data-hit-page>回到引用页</button>
      <button type="button" class="icon-button" data-scale="-25" aria-label="缩小">−</button>
      <span data-zoom></span>
      <button type="button" class="icon-button" data-scale="25" aria-label="放大">＋</button>
      <button type="button" class="secondary-button button-compact" data-scale="0">适合宽度</button>
      </div><div class="knowledge-pdf-viewport" role="region" aria-label="PDF 原页" tabindex="0"></div>`;
    if (excerpt) {
      const details = document.createElement("details");
      details.className = "knowledge-reference-excerpt";
      details.innerHTML = "<summary>查看引用摘录</summary>";
      appendReferenceExcerpt(details, excerpt);
      body.append(details);
    }
    let page = hitPage, zoom = 0, pageCount = 0;
    session.url = URL.createObjectURL(blob);
    const blocks = [{ anchor: true, locators, segments: excerpt ? [{ text: excerpt, matched: true }] : [] }];
    const updateControls = () => {
      body.querySelector("[data-page]").textContent = `第 ${page}${pageCount ? ` / ${pageCount}` : ""} 页`;
      body.querySelector('[data-step="-1"]').disabled = page <= 1;
      body.querySelector('[data-step="1"]').disabled = !pageCount || page >= pageCount;
      body.querySelector("[data-hit-page]").disabled = page === hitPage;
      body.querySelector("[data-zoom]").textContent = zoom ? zoom + "%" : "适宽";
    };
    session.reader = module.mountPdfSearchReader(body.querySelector(".knowledge-pdf-viewport"), count => {
      pageCount = count;
      updateControls();
    });
    const update = () => {
      updateControls();
      session.reader.update({ url: session.url, page, zoom, hitPage, blocks });
    };
    body.querySelector(".knowledge-reference-tools").addEventListener("click", event => {
      const button = event.target.closest("button");
      if (!button) return;
      if (button.dataset.step) page += Number(button.dataset.step);
      else if ("hitPage" in button.dataset) page = hitPage;
      else if ("scale" in button.dataset) {
        const step = Number(button.dataset.scale);
        zoom = step ? Math.max(50, Math.min(300, (zoom || 100) + step)) : 0;
      }
      update();
    });
    update();
  }

  function bindEvents() {
    $("knowledge-agent").addEventListener("change", async () => {
      state.active = null;
      clearImages();
      renderAgentContext();
      try { await loadConversations(); }
      catch (error) { showError(error, "会话列表加载失败"); renderEmptyChat(); }
    });
    let sidebarOpen = false;
    const setSidebar = open => {
      if (!open) closeConversationMenu();
      sidebarOpen = open;
      $("knowledge-workspace").classList.toggle("is-session-collapsed", !open);
      $("knowledge-retrieval-sessions").classList.toggle("open", open);
      const toggle = $("knowledge-session-toggle");
      toggle.setAttribute("aria-expanded", String(open));
      toggle.ariaLabel = toggle.title = open ? "收起会话列表" : "展开会话列表";
    };
    $("knowledge-session-toggle").addEventListener("click", () => setSidebar(!sidebarOpen));
    document.addEventListener("pointerdown", event => {
      if (menu && !menu.contains(event.target) && !menuTrigger?.contains(event.target)) closeConversationMenu();
    });
    document.addEventListener("keydown", event => {
      if (event.key !== "Escape") return;
      if (menu) { event.preventDefault(); closeConversationMenu(true); }
      else if (sidebarOpen && narrowChat.matches && !$("knowledge-reference-dialog").open && !document.querySelector("dialog[open]")) {
        setSidebar(false);
        $("knowledge-session-toggle").focus();
      }
    });
    document.addEventListener("scroll", () => closeConversationMenu(), true);
    addEventListener("resize", () => closeConversationMenu());
    addEventListener("kbot:knowledge-context-changing", () => closeConversationMenu());
    setSidebar(sidebarOpen);
    $("knowledge-new-conversation").addEventListener("click", () => {
      if (state.sending) return;
      if (narrowChat.matches) setSidebar(false);
      clearImages();
      $("knowledge-input").value = "";
      renderEmptyChat();
      $("knowledge-input").focus();
    });
    $("knowledge-add-images").addEventListener("click", () => $("knowledge-query-images").click());
    $("knowledge-query-images").addEventListener("change", renderImages);
    $("knowledge-clear-images").addEventListener("click", () => { clearImages(); $("knowledge-add-images").focus(); });
    $("knowledge-input").addEventListener("input", renderAgentContext);
    $("knowledge-input").addEventListener("keydown", event => {
      if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) return;
      event.preventDefault();
      if (!event.repeat && !$("knowledge-send").disabled) event.currentTarget.form?.requestSubmit();
    });
    $("knowledge-refresh").addEventListener("click", () => loadConversations(state.active?.conversation_id).catch((error) => showError(error, "会话列表刷新失败")));
    $("knowledge-form").addEventListener("submit", send);
    $("knowledge-conversation-list").addEventListener("click", (event) => {
      const more = event.target.closest("[data-session-more]");
      if (more) return openConversationMenu(more);
      const button = event.target.closest("[data-conversation-id]");
      const row = state.conversations.find((item) => String(item.conversation_id) === button?.dataset.conversationId);
      if (row && !state.sending) {
        if (narrowChat.matches) setSidebar(false);
        selectConversation(row).catch((error) => showError(error, "会话读取失败"));
      }
    });
    $("knowledge-chat-stream").addEventListener("click", (event) => {
      const copy = event.target.closest("[data-copy-code]");
      if (copy) return KBotMarkdown.copyCode(copy);
      const retrieval = event.target.closest("[data-open-retrieval]");
      if (retrieval) return openRetrieval(retrieval.closest("[data-run-id]")?.dataset.runId);
      const marker = event.target.closest("[data-citation-label]");
      if (marker) {
        const message = marker.closest("[data-run-id]");
        if (message?.dataset.runId) openReference(message.dataset.runId, marker.dataset.citationLabel).catch((error) => showError(error, "引用读取失败"));
        return;
      }
    });
    $("knowledge-reference-dialog").addEventListener("click", (event) => {
      if (event.target.closest("[data-close-reference]")) $("knowledge-reference-dialog").close();
    });
    $("knowledge-reference-dialog").addEventListener("close", () => {
      if (!$("knowledge-reference-dialog").open) clearReference();
    });
    const retrieval = $("knowledge-retrieval-dialog");
    retrieval.addEventListener("click", event => {
      if (event.target.closest("[data-close-retrieval]")) retrieval.close();
    });
    retrieval.addEventListener("toggle", event => {
      if (event.target.matches?.("details[data-evidence-key][open]")) void readEvidence(event.target);
    }, true);
    retrieval.addEventListener("close", () => {
      if (!retrieval.open) clearRetrieval();
    });
    addEventListener("kbot:knowledge-context-changing", () => {
      if (retrieval.open) retrieval.close();
      if ($("knowledge-reference-dialog").open) $("knowledge-reference-dialog").close();
    });
  }

  KBotKnowledgeShell.ready.then(async (access) => {
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
