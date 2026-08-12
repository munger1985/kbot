(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  let agents = [], conversations = [], active = null, currentPreview = null;

  function activeAgentId() { return $("chat-agent").value; }
  function agentName(id) { return agents.find((row) => String(row.agent_id) === String(id))?.display_name || KBotKmShell.shortId(id); }
  function contentText(item) {
    const value = item?.content;
    if (typeof value === "string") return value;
    if (!value || typeof value !== "object") return "";
    return String(value.answer ?? value.text ?? value.input ?? value.content ?? "");
  }

  async function initialize() {
    try {
      agents = KBotKmApi.items(await KBotKmApi.request(`${base}/agents`)).filter((row) => row.status === "ACTIVE");
      $("chat-agent").innerHTML = agents.length ? agents.map((row) => `<option value="${KBotKmShell.escapeHtml(row.agent_id)}">${KBotKmShell.escapeHtml(row.display_name)}</option>`).join("") : '<option value="">没有可用的 ACTIVE Agent</option>';
      if (!agents.length) { $("new-conversation").disabled = true; $("chat-input").disabled = true; return; }
      await loadConversations();
    } catch (error) { KBotKmShell.showError(error, "问答工作区加载失败"); }
  }
  async function loadConversations(selectId) {
    try {
      const all = KBotKmApi.items(await KBotKmApi.request(`${base}/conversations?limit=50`));
      conversations = all.filter((row) => String(row.agent_id) === String(activeAgentId()));
      renderConversationList();
      const selected = conversations.find((row) => String(row.conversation_id) === String(selectId || active?.conversation_id));
      if (selected) await selectConversation(selected);
      else { active = null; renderEmptyChat(); }
    } catch (error) { KBotKmShell.showError(error, "会话列表加载失败"); }
  }
  function renderConversationList() {
    const list = $("conversation-list");
    if (!conversations.length) { list.innerHTML = '<li class="km-empty">当前 Agent 暂无会话</li>'; return; }
    list.innerHTML = conversations.map((row) => `<li><button data-id="${KBotKmShell.escapeHtml(row.conversation_id)}" aria-current="${active?.conversation_id === row.conversation_id}"><span class="km-cell-main">${KBotKmShell.escapeHtml(row.title || "未命名会话")}</span><span class="km-cell-sub">${KBotKmShell.escapeHtml(KBotKmShell.formatDate(row.last_active_at))}</span></button></li>`).join("");
  }
  function renderEmptyChat() {
    $("conversation-title").textContent = "请选择或创建会话";
    $("conversation-meta").textContent = `当前 Agent：${agentName(activeAgentId())}`;
    $("chat-stream").innerHTML = '<div class="km-empty">创建会话后即可开始问文与问数。</div>';
  }
  async function createConversation() {
    if (!activeAgentId()) return;
    KBotKmShell.setBusy($("new-conversation"), true, "创建中…");
    try {
      const created = await KBotKmApi.json(`${base}/conversations`, "POST", { agent_id: activeAgentId(), title: null, retention_policy: "DEFAULT" });
      await loadConversations(created.conversation_id);
      $("chat-input").focus();
    } catch (error) { KBotKmShell.showError(error, "会话创建失败"); }
    finally { KBotKmShell.setBusy($("new-conversation"), false); }
  }
  async function selectConversation(row) {
    active = await KBotKmApi.request(`${base}/conversations/${row.conversation_id}`);
    renderConversationList();
    $("conversation-title").textContent = active.title || "未命名会话";
    $("conversation-meta").textContent = `${agentName(active.agent_id)} · ${active.status} · v${active.row_version}`;
    await loadTurns();
  }
  async function loadTurns() {
    if (!active) return renderEmptyChat();
    const payload = await KBotKmApi.request(`${base}/conversations/${active.conversation_id}/turns?after=0&limit=200`);
    const turns = Array.isArray(payload?.turns) ? payload.turns : KBotKmApi.items(payload);
    const stream = $("chat-stream");
    if (!turns.length) stream.innerHTML = '<div class="km-empty">这是一个新会话，请输入问题。</div>';
    else stream.innerHTML = turns.map((turn) => {
      const user = contentText(turn.user_item);
      const assistant = contentText(turn.assistant_item);
      return `${user ? messageMarkup("user", "你", user) : ""}${assistant ? messageMarkup("assistant", "KM Agent", assistant, turn.run_id) : (turn.status && turn.status !== "COMPLETED" ? messageMarkup("assistant", "KM Agent", `处理状态：${turn.status}`) : "")}`;
    }).join("");
    const runIds = turns.map((turn) => turn.run_id).filter(Boolean).slice(-20);
    await Promise.allSettled(runIds.map(async (runId) => {
      const result = await KBotKmApi.request(`${base}/runs/${runId}/result`);
      renderReferences(runId, result);
    }));
    stream.scrollTop = stream.scrollHeight;
  }
  function messageMarkup(role, label, text, runId) { return `<div class="km-message ${role}"><div class="meta">${KBotKmShell.escapeHtml(label)}</div><div class="content">${KBotKmShell.escapeHtml(text)}</div>${runId ? `<div data-references-for="${KBotKmShell.escapeHtml(runId)}"></div>` : ""}</div>`; }
  function appendPending(input) {
    const stream = $("chat-stream");
    if (stream.querySelector(".km-empty")) stream.innerHTML = "";
    stream.insertAdjacentHTML("beforeend", `${messageMarkup("user", "你", input)}${messageMarkup("assistant", "KM Agent", "正在分析问题并选择文档检索或元数据问数路径…")}`);
    stream.scrollTop = stream.scrollHeight;
  }
  async function send(event) {
    event.preventDefault();
    if (!active) return KBotKmShell.toast("请先创建或选择会话", "error");
    const input = $("chat-input").value.trim(); if (!input) return;
    appendPending(input); $("chat-input").value = ""; KBotKmShell.setBusy($("send-message"), true, "处理中…");
    $("chat-progress").hidden = false; $("chat-progress").textContent = "正在创建 Turn";
    try {
      const receipt = await KBotKmApi.json(`${base}/conversations/${active.conversation_id}/turns`, "POST", { input, expected_conversation_version: active.row_version, collection_ids: [], security_level: 0, client_metadata: { source: "km-asset-ui" }, images: [] }, { headers: { "Idempotency-Key": KBotKmApi.uuid() } });
      if (receipt.run_id && receipt.events_url) {
        $("chat-progress").textContent = "Agent 正在执行";
        await KBotKmApi.stream(receipt.events_url, { lastEventId: receipt.event_cursor, onEvent: (item) => { const data = item.json || {}; $("chat-progress").textContent = data.title || data.summary || item.type || "Agent 正在执行"; } });
        const result = await KBotKmApi.request(`${base}/runs/${receipt.run_id}/result`);
        await refreshActive();
        await loadConversations(active.conversation_id);
        renderReferences(receipt.run_id, result);
      } else { await refreshActive(); await loadTurns(); }
      if (!receipt.run_id || !receipt.events_url) await loadConversations(active.conversation_id);
    } catch (error) { KBotKmShell.showError(error, "对话请求失败"); await refreshActive().catch(() => {}); await loadTurns().catch(() => {}); }
    finally { $("chat-progress").hidden = true; KBotKmShell.setBusy($("send-message"), false); }
  }
  async function refreshActive() { if (active) active = await KBotKmApi.request(`${base}/conversations/${active.conversation_id}`); }
  function renderReferences(runId, result) {
    const refs = Array.isArray(result?.payload?.references) ? result.payload.references.filter((row) => row.reference_type === "DOCUMENT") : [];
    const host = Array.from(document.querySelectorAll("[data-references-for]"))
      .find((node) => node.dataset.referencesFor === String(runId));
    if (!host || !refs.length) return;
    host.innerHTML = refs.map((row, index) => `<button class="km-reference" data-run="${KBotKmShell.escapeHtml(runId)}" data-reference="${index}">${KBotKmShell.escapeHtml(row.citation_label || `[${index + 1}]`)} · ${KBotKmShell.escapeHtml(row.title || "引用文档")}</button>`).join("");
    host._references = refs;
  }
  async function prepareReference(button) {
    const host = button.parentElement; const reference = host._references?.[Number(button.dataset.reference)]; if (!reference) return;
    try {
      const preview = await KBotKmApi.request(`${base}/runs/${encodeURIComponent(button.dataset.run)}/references/${encodeURIComponent(reference.citation_label)}/preview`);
      currentPreview = preview; $("reference-title").textContent = preview.title || reference.title || "引用文档"; $("reference-meta").textContent = `${preview.mime_type} · ${preview.page_no ? `第 ${preview.page_no}${preview.page_end && preview.page_end !== preview.page_no ? `–${preview.page_end}` : ""} 页` : "未指定页码"}`; KBotKmShell.openDialog("reference-dialog");
    } catch (error) { KBotKmShell.showError(error, "引用描述读取失败"); }
  }
  async function openReference() {
    if (!currentPreview?.content_url) return;
    const popup = window.open("about:blank", "_blank");
    try { const file = await KBotKmApi.blob(currentPreview.content_url); const url = URL.createObjectURL(file.data); const target = currentPreview.preview_type === "PDF" && currentPreview.page_no ? `${url}#page=${currentPreview.page_no}` : url; if (popup) popup.location.href = target; else window.open(target, "_blank"); setTimeout(() => URL.revokeObjectURL(url), 10 * 60 * 1000); }
    catch (error) { popup?.close(); KBotKmShell.showError(error, "原文打开失败"); }
  }
  window.addEventListener("DOMContentLoaded", () => {
    $("chat-agent").addEventListener("change", () => loadConversations()); $("new-conversation").addEventListener("click", createConversation); $("refresh-conversations").addEventListener("click", () => loadConversations(active?.conversation_id)); $("chat-form").addEventListener("submit", send);
    $("conversation-list").addEventListener("click", (event) => { const button = event.target.closest("[data-id]"); const row = conversations.find((item) => String(item.conversation_id) === button?.dataset.id); if (row) selectConversation(row).catch((error) => KBotKmShell.showError(error)); });
    $("chat-stream").addEventListener("click", (event) => { const button = event.target.closest("[data-reference]"); if (button) prepareReference(button); }); $("open-reference").addEventListener("click", openReference);
  });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
