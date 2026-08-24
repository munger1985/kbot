/* KM 聊天独立入口；SSE 正文增量累计后使用安全 Markdown 渲染。 */
(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  let agents = [], conversations = [], active = null;
  const runResults = new Map();
  const citationPattern = /\[(C\d+)\]/g;
  const typingFrameMs = 22;
  const graphemeSegmenter = typeof Intl?.Segmenter === "function"
    ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
    : null;

  function activeAgentId() { return $("chat-agent").value; }
  function agentName(id) { return agents.find((row) => String(row.agent_id) === String(id))?.display_name || KBotKmShell.shortId(id); }
  function contentText(item) {
    const value = item?.content;
    if (typeof value === "string") return value;
    if (!value || typeof value !== "object") return "";
    return String(value.answer ?? value.text ?? value.input ?? value.content ?? "");
  }
  function renderAssistantMarkdown(value) {
    const template = document.createElement("template");
    template.innerHTML = KBotMarkdown.render(value);
    const walker = document.createTreeWalker(
      template.content,
      NodeFilter.SHOW_TEXT,
    );
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    textNodes.forEach((node) => {
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
        marker.className = "km-citation-marker";
        marker.textContent = match[1];
        marker.title = `引用 ${match[1]}`;
        marker.setAttribute("aria-label", `引用 ${match[1]}`);
        marker.dataset.citationLabel = match[1];
        fragment.append(marker);
        cursor = match.index + match[0].length;
      }
      fragment.append(document.createTextNode(text.slice(cursor)));
      node.replaceWith(fragment);
    });
    return template.innerHTML;
  }

  function typingUnits(value) {
    const text = String(value || "");
    if (!text) return [];
    if (!graphemeSegmenter) return Array.from(text);
    return Array.from(graphemeSegmenter.segment(text), (item) => item.segment);
  }

  function typingBatchSize(queueLength, finalizing) {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      return queueLength;
    }
    if (finalizing) return Math.max(1, Math.ceil(queueLength / 16));
    if (queueLength > 800) return 8;
    if (queueLength > 240) return 4;
    if (queueLength > 80) return 2;
    return 1;
  }

  function settleTypewriter(pending) {
    pending.typingTimer = null;
    pending.message.classList.remove("is-typing");
    pending.message.setAttribute("aria-busy", "false");
    pending.typingWaiters.splice(0).forEach((resolve) => resolve());
  }

  function typewriterTick(pending) {
    if (!pending.typingQueue.length) return settleTypewriter(pending);
    const count = typingBatchSize(
      pending.typingQueue.length,
      pending.finalizing,
    );
    pending.displayedMarkdown += pending.typingQueue.splice(0, count).join("");
    pending.content.innerHTML = renderAssistantMarkdown(pending.displayedMarkdown);
    $("chat-stream").scrollTop = $("chat-stream").scrollHeight;
    pending.typingTimer = window.setTimeout(
      () => typewriterTick(pending), typingFrameMs,
    );
  }

  function enqueueAnswerDelta(pending, delta) {
    const text = String(delta || "");
    if (!text) return;
    pending.markdown += text;
    pending.typingQueue.push(...typingUnits(text));
    pending.message.classList.add("is-typing");
    if (pending.typingTimer === null) typewriterTick(pending);
  }

  function waitForTypewriter(pending) {
    if (!pending.typingQueue.length && pending.typingTimer === null) {
      return Promise.resolve();
    }
    return new Promise((resolve) => pending.typingWaiters.push(resolve));
  }

  function finishTypewriterSoon(pending) {
    pending.finalizing = true;
  }

  function cancelTypewriter(pending) {
    if (pending.typingTimer !== null) window.clearTimeout(pending.typingTimer);
    pending.typingQueue.length = 0;
    settleTypewriter(pending);
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
    turns.forEach((turn) => {
      const references = turn.assistant_item?.content?.references;
      if (turn.run_id && Array.isArray(references)) {
        renderReferences(turn.run_id, references);
      }
    });
    stream.scrollTop = stream.scrollHeight;
  }
  function messageMarkup(role, label, text, runId) {
    const content = role === "assistant"
      ? renderAssistantMarkdown(text)
      : KBotKmShell.escapeHtml(text);
    return `<div class="km-message ${role}"${runId ? ` data-run-id="${KBotKmShell.escapeHtml(runId)}"` : ""}><div class="meta">${KBotKmShell.escapeHtml(label)}</div><div class="content">${content}</div>${runId ? `<div data-references-for="${KBotKmShell.escapeHtml(runId)}"></div>` : ""}</div>`;
  }
  function appendPending(input) {
    const stream = $("chat-stream");
    if (stream.querySelector(".km-empty")) stream.innerHTML = "";
    stream.insertAdjacentHTML("beforeend", messageMarkup("user", "你", input));
    const message = document.createElement("div");
    message.className = "km-message assistant";
    message.setAttribute("aria-busy", "true");
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = "KM Agent";
    const content = document.createElement("div");
    content.className = "content";
    content.setAttribute("aria-live", "polite");
    content.textContent = "正在分析问题并选择文档检索或元数据问数路径…";
    const references = document.createElement("div");
    message.append(meta, content, references);
    stream.append(message);
    stream.scrollTop = stream.scrollHeight;
    return {
      message, content, references, markdown: "", displayedMarkdown: "",
      typingQueue: [], typingTimer: null, typingWaiters: [], finalizing: false,
      referencesReceived: false,
    };
  }
  function applyRunEvent(pending, item) {
    const data = item.json && typeof item.json === "object" ? item.json : {};
    const payload = data.payload && typeof data.payload === "object" ? data.payload : {};
    const eventType = item.type || data.event_type || "message";
    if (eventType === "answer.delta") {
      enqueueAnswerDelta(pending, payload.delta);
      return;
    }
    if (eventType === "answer.completed") {
      finishTypewriterSoon(pending);
      if (Array.isArray(payload.references)) {
        pending.referencesReceived = true;
        renderReferences(pending.message.dataset.runId, payload.references);
      }
    }
    $("chat-progress").textContent = payload.public_summary
      || payload.summary
      || data.title
      || data.summary
      || eventType
      || "Agent 正在执行";
  }
  function isRetryableTurnTransportError(error) {
    return error instanceof TypeError || [502, 503, 504].includes(Number(error?.status));
  }
  async function createTurn(input, idempotencyKey) {
    const path = `${base}/conversations/${active.conversation_id}/turns`;
    const options = { headers: { "Idempotency-Key": idempotencyKey } };
    const payload = { input, expected_conversation_version: active.row_version, collection_ids: [], client_metadata: { source: "km-asset-ui" }, images: [] };
    try {
      return await KBotKmApi.json(path, "POST", payload, options);
    } catch (error) {
      if (!isRetryableTurnTransportError(error)) throw error;
      $("chat-progress").textContent = "连接中断，正在确认已提交的 Turn";
      await new Promise((resolve) => window.setTimeout(resolve, 300));
      return KBotKmApi.json(path, "POST", payload, options);
    }
  }
  async function send(event) {
    event.preventDefault();
    if (!active) return KBotKmShell.toast("请先创建或选择会话", "error");
    const input = $("chat-input").value.trim(); if (!input) return;
    const pending = appendPending(input); $("chat-input").value = ""; KBotKmShell.setBusy($("send-message"), true, "处理中…");
    $("chat-progress").hidden = false; $("chat-progress").textContent = "正在创建 Turn";
    try {
      const receipt = await createTurn(input, KBotKmApi.uuid());
      if (receipt.run_id) {
        pending.message.dataset.runId = String(receipt.run_id);
        pending.references.dataset.referencesFor = String(receipt.run_id);
        $("chat-progress").textContent = "Agent 正在执行";
        const eventsUrl = `${base}/runs/${encodeURIComponent(receipt.run_id)}/events`;
        await KBotKmApi.stream(eventsUrl, {
          lastEventId: receipt.event_cursor,
          onEvent: (item) => applyRunEvent(pending, item),
        });
        const run = await KBotKmApi.request(`${base}/runs/${receipt.run_id}`);
        if (run.status !== "COMPLETED") {
          const error = new Error(run.error_message || `Run 执行结束但状态为 ${run.status}`);
          error.code = run.error_code || "KM_RUN_NOT_COMPLETED";
          error.requestId = run.request_id || "";
          throw error;
        }
        let result = null;
        if (!pending.referencesReceived || !pending.markdown) {
          result = await KBotKmApi.request(`${base}/runs/${receipt.run_id}/result`);
          if (!pending.markdown) {
            enqueueAnswerDelta(pending, result?.payload?.answer);
          }
          if (!pending.referencesReceived) {
            renderReferences(receipt.run_id, result);
          }
        }
        finishTypewriterSoon(pending);
        await Promise.all([waitForTypewriter(pending), refreshActive()]);
        updateActiveConversation();
      } else { await refreshActive(); await loadTurns(); }
      if (!receipt.run_id) await loadConversations(active.conversation_id);
    } catch (error) { cancelTypewriter(pending); KBotKmShell.showError(error, "对话请求失败"); await refreshActive().catch(() => {}); await loadTurns().catch(() => {}); }
    finally { $("chat-progress").hidden = true; KBotKmShell.setBusy($("send-message"), false); }
  }
  async function refreshActive() { if (active) active = await KBotKmApi.request(`${base}/conversations/${active.conversation_id}`); }
  function updateActiveConversation() {
    if (!active) return;
    const index = conversations.findIndex(
      (row) => String(row.conversation_id) === String(active.conversation_id),
    );
    if (index >= 0) conversations[index] = active;
    else conversations.unshift(active);
    renderConversationList();
    $("conversation-title").textContent = active.title || "未命名会话";
    $("conversation-meta").textContent = `${agentName(active.agent_id)} · ${active.status} · v${active.row_version}`;
  }
  function referenceRows(source) {
    if (Array.isArray(source)) return source;
    return Array.isArray(source?.payload?.references)
      ? source.payload.references
      : [];
  }
  function renderReferences(runId, source) {
    if (!runId) return;
    const rows = referenceRows(source);
    const result = { payload: { references: rows } };
    runResults.set(String(runId), result);
    const refs = rows.filter((row) => row.reference_type === "DOCUMENT");
    const host = Array.from(document.querySelectorAll("[data-references-for]"))
      .find((node) => node.dataset.referencesFor === String(runId));
    if (!host || !refs.length) return;
    host.innerHTML = refs.map((row, index) => `<button class="km-reference" data-run="${KBotKmShell.escapeHtml(runId)}" data-reference="${index}">${KBotKmShell.escapeHtml(row.citation_label || `[${index + 1}]`)} · ${KBotKmShell.escapeHtml(row.title || "引用文档")}</button>`).join("");
    host._references = refs;
  }
  function appendAssetFields(host, fields) {
    const labels = { title: "标题", author: "作者", product: "产品", solution: "解决方案", industry: "行业", category: "类别", content_category: "内容类别", asset_date: "发布日期" };
    const entries = Object.entries(fields || {}).filter(([, value]) => value != null && String(value).trim());
    if (!entries.length) return;
    const details = document.createElement("dl");
    details.className = "km-reference-asset-fields";
    entries.forEach(([field, value]) => {
      const term = document.createElement("dt");
      term.textContent = labels[field] || field;
      const description = document.createElement("dd");
      description.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
      details.append(term, description);
    });
    host.append(details);
  }
  function appendAssetAttachments(host, attachments) {
    const heading = document.createElement("h3");
    heading.textContent = "Asset 附件";
    host.append(heading);
    if (!attachments.length) {
      const empty = document.createElement("p");
      empty.className = "km-help";
      empty.textContent = "该 Asset 没有附件；当前引用依据来自 Asset 本身内容。";
      host.append(empty);
      return;
    }
    const list = document.createElement("ul");
    list.className = "km-reference-attachments";
    attachments.forEach((attachment) => {
      const item = document.createElement("li");
      const copy = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = attachment.name || "附件";
      const meta = document.createElement("span");
      const pages = attachment.page_no ? ` · 第 ${attachment.page_no}${attachment.page_end && attachment.page_end !== attachment.page_no ? `–${attachment.page_end}` : ""} 页` : "";
      meta.textContent = `${attachment.mime_type || "文件"}${attachment.evidence_source ? " · 当前引用命中" : ""}${pages}`;
      copy.append(name, meta);
      const open = document.createElement("button");
      open.type = "button";
      open.className = "small";
      open.textContent = "打开附件";
      open.dataset.attachmentUrl = attachment.content_url;
      open.dataset.previewType = attachment.preview_type || "DOWNLOAD";
      open.dataset.pageNo = attachment.page_no || "";
      item.append(copy, open);
      list.append(item);
    });
    host.append(list);
  }
  async function prepareAssetReference(runId, reference) {
    if (!reference) return;
    try {
      const preview = await KBotKmApi.request(`${base}/runs/${encodeURIComponent(runId)}/references/${encodeURIComponent(reference.citation_label)}/preview`);
      $("reference-title").textContent = preview.title || reference.title || "Asset 引用";
      $("reference-meta").textContent = `${preview.citation_label || reference.citation_label} · 版本 ${preview.revision_no} · ${preview.status || "UNKNOWN"}${preview.is_current_revision ? " · 当前版本" : ""}`;
      $("reference-description").textContent = preview.asset_content_available ? "该引用首先指向 Asset 本身内容；下方附件属于同一 Asset。" : "该引用首先指向 Asset；当前版本未提供可预览的 Asset 内容文件。";
      const host = $("reference-asset-preview");
      host.replaceChildren();
      appendAssetFields(host, preview.asset_fields);
      appendAssetAttachments(host, Array.isArray(preview.attachments) ? preview.attachments : []);
      host.hidden = false;
      KBotKmShell.openDialog("reference-dialog");
    } catch (error) { KBotKmShell.showError(error, "引用描述读取失败"); }
  }
  async function prepareCitationMarker(marker) {
    const message = marker.closest(".km-message[data-run-id]");
    const runId = String(message?.dataset.runId || "");
    const label = String(marker.dataset.citationLabel || "");
    let result = runResults.get(runId);
    if (runId && !result) {
      try {
        result = await KBotKmApi.request(`${base}/runs/${runId}/result`);
        renderReferences(runId, result);
      } catch (error) {
        KBotKmShell.showError(error, "引用依据读取失败");
        return;
      }
    }
    const references = Array.isArray(result?.payload?.references) ? result.payload.references : [];
    const reference = references.find((row) => String(row?.citation_label || "") === label);
    if (!runId || !reference) return KBotKmShell.toast("引用依据尚未加载", "error");
    return prepareAssetReference(runId, reference);
  }
  async function prepareReference(button) {
    const host = button.parentElement;
    const reference = host._references?.[Number(button.dataset.reference)];
    return prepareAssetReference(button.dataset.run, reference);
  }
  async function openAttachment(button) {
    const contentUrl = String(button?.dataset?.attachmentUrl || "");
    if (!contentUrl) return;
    const popup = window.open("about:blank", "_blank");
    try { const file = await KBotKmApi.blob(contentUrl); const url = URL.createObjectURL(file.data); const target = button.dataset.previewType === "PDF" && button.dataset.pageNo ? `${url}#page=${button.dataset.pageNo}` : url; if (popup) popup.location.href = target; else window.open(target, "_blank"); setTimeout(() => URL.revokeObjectURL(url), 10 * 60 * 1000); }
    catch (error) { popup?.close(); KBotKmShell.showError(error, "附件打开失败"); }
  }
  window.addEventListener("DOMContentLoaded", () => {
    $("chat-agent").addEventListener("change", () => loadConversations()); $("new-conversation").addEventListener("click", createConversation); $("refresh-conversations").addEventListener("click", () => loadConversations(active?.conversation_id)); $("chat-form").addEventListener("submit", send);
    $("conversation-list").addEventListener("click", (event) => { const button = event.target.closest("[data-id]"); const row = conversations.find((item) => String(item.conversation_id) === button?.dataset.id); if (row) selectConversation(row).catch((error) => KBotKmShell.showError(error)); });
    $("chat-stream").addEventListener("click", (event) => {
      const copyButton = event.target.closest("[data-copy-code]");
      if (copyButton) { KBotMarkdown.copyCode(copyButton); return; }
      const citationMarker = event.target.closest("[data-citation-label]");
      if (citationMarker) { prepareCitationMarker(citationMarker); return; }
      const attachment = event.target.closest("[data-attachment-url]");
      if (attachment) { openAttachment(attachment); return; }
      const button = event.target.closest("[data-reference]");
      if (button) prepareReference(button);
    });
  });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
