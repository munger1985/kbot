(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const markdown = globalThis.KBotMarkdown;
  const state = { agents: [], conversation: null, selectedFile: null };
  const typingFrameMs = 22;
  const streamRecoveryAttempts = 120;
  const terminalTurnStatuses = new Set([
    "WAITING_USER", "COMPLETED", "PARTIAL", "FAILED", "CANCELLED",
  ]);
  const graphemeSegmenter = typeof Intl?.Segmenter === "function"
    ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
    : null;

  const esc = shell.escape;
  const values = (items) => Array.isArray(items) ? items : [];
  const bullets = (items) => values(items).length
    ? values(items).map((item) => `- ${typeof item === "string" ? item : item.fact_summary || item.summary || item.title || "已记录"}`).join("\n")
    : "- 无";

  function inspectionMarkdown(result) {
    const payload = result?.payload || {};
    if (!result?.final_artifact) {
      return `### 诊断尚未形成最终结论\n\n当前状态：${result?.status || "处理中"}`;
    }
    return `## ${payload.title || "巡检报告"}\n\n${payload.summary || ""}\n\n### 发现\n${bullets(payload.facts)}\n\n### 建议\n${bullets(payload.recommendations)}\n\n### 数据缺口\n${bullets(payload.gaps)}`;
  }

  function conversationAnswerMarkdown(result) {
    const payload = result?.payload || {};
    if (!result?.final_artifact) {
      return `诊断尚未完成，当前状态为 ${result?.status || "处理中"}。`;
    }
    const direct = payload.direct_answer?.answer_text;
    const solution = payload.solution || {};
    let answer = direct || payload.diagnosis_rationale || "现有证据还不足以回答这个问题。";
    const limitations = values(payload.direct_answer?.limitations).filter((item) => !answer.includes(String(item)));
    if (limitations.length) answer += `\n\n${limitations.map((item) => `> ${item}`).join("\n")}`;
    if (!direct) {
      const recommendations = [
        ...values(solution.immediate_mitigations),
        ...values(solution.long_term_remediations),
      ].filter(Boolean);
      if (recommendations.length) answer += `\n\n接下来可以这样处理：\n\n${bullets(recommendations)}`;
    }
    return answer;
  }

  function evidenceDetails(result) {
    const payload = result?.payload || {};
    const facts = values(payload.facts);
    const gaps = values(payload.gaps);
    const root = payload.root_cause || {};
    if (!facts.length && !gaps.length && !root.effective_level) return "";
    const factRows = facts.length
      ? `<ol class="ops-evidence-list">${facts.map((fact) => `<li><span>${esc(fact.fact_summary || "已验证事实")}</span><small>${esc(fact.source_type || "EVIDENCE")}${fact.captured_at ? ` · ${esc(shell.fmt(fact.captured_at))}` : ""}</small></li>`).join("")}</ol>`
      : '<p class="ops-evidence-empty">本次没有形成可展示的事实条目。</p>';
    const rootRow = root.effective_level
      ? `<div class="ops-evidence-assessment"><span>根因判断</span><strong>${esc(root.effective_level)}</strong></div>`
      : "";
    const gapRows = gaps.length
      ? `<div class="ops-evidence-gaps"><strong>仍缺少的证据</strong><ul>${gaps.map((item) => `<li>${esc(typeof item === "string" ? item : item.code || item.summary || "EVIDENCE_GAP")}</li>`).join("")}</ul></div>`
      : "";
    return `<details class="ops-evidence"><summary>诊断依据 <span>${facts.length} 项已验证事实</span></summary><div class="ops-evidence-body">${rootRow}${factRows}${gapRows}</div></details>`;
  }

  function messageHtml(role, text, meta = "", supplemental = "") {
    const user = role === "USER";
    return `<article class="ops-message ${user ? "user" : "agent"}"><div class="ops-avatar">${user ? "我" : "AI"}</div><div class="ops-message-body ops-result-markdown"><div class="ops-message-content">${markdown.render(text)}</div>${supplemental}${meta ? `<div class="ops-message-meta">${esc(meta)}</div>` : ""}</div></article>`;
  }

  async function agents() {
    const rows = await KBotAIOpsAuth.request(`${api}/agents`);
    state.agents = values(rows).filter((item) => item.status === "ACTIVE");
    return state.agents;
  }

  async function proposalAction(button) {
    const approving = Boolean(button.dataset.approveProposal);
    const proposalId = button.dataset.approveProposal || button.dataset.rejectProposal;
    if (approving && !confirm("确认批准并执行这一条受控变更吗？系统会继续执行前置校验，并在完成后验证效果。")) return;
    const reason = approving ? "用户在诊断对话中逐条确认" : prompt("请输入拒绝原因");
    if (!reason) return;
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${api}/proposals/${encodeURIComponent(proposalId)}/${approving ? "approve" : "reject"}`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify(approving ? { expected_row_version: Number(button.dataset.version), expected_proposal_hash: button.dataset.hash, note: reason } : { expected_row_version: Number(button.dataset.version), reason }),
      });
      shell.toast(approving ? "审批已提交，等待执行与验证" : "已拒绝该变更");
      if (state.conversation) await loadConversation(state.conversation.conversation_id);
      else button.closest(".ops-message")?.remove();
    } catch (error) { shell.toast(error.message); button.disabled = false; }
  }

  function answerBlockHtml(block) {
    const payload = block.payload || {};
    const citations = values(block.citations);
    const citationHtml = citations.length
      ? `<details class="ops-evidence"><summary>诊断依据 <span>${citations.length} 项证据</span></summary><div class="ops-evidence-body"><ol class="ops-evidence-list">${citations.map((item) => `<li><span>${esc(item.label || `证据 ${item.citation_no}`)}</span><small>${esc(shell.short(item.turn_evidence_id))}</small></li>`).join("")}</ol></div></details>`
      : "";
    if (block.block_type === "MARKDOWN") return `${markdown.render(payload.markdown || payload.text || "")}${citationHtml}`;
    if (block.block_type === "TABLE") {
      const columns = values(payload.columns);
      const cell = (row, column, index) => Array.isArray(row)
        ? row[index]
        : row?.[column.key || column.name || column];
      return `<div class="ops-table-wrap"><table><thead><tr>${columns.map((column) => `<th>${esc(column.label || column.name || column.key || column)}</th>`).join("")}</tr></thead><tbody>${values(payload.rows).map((row) => `<tr>${columns.map((column, index) => `<td>${esc(cell(row, column, index) ?? "-")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>${citationHtml}`;
    }
    if (block.block_type === "CHART") {
      const categories = values(payload.categories);
      const sourceSeries = values(payload.series);
      const series = sourceSeries.map((item, index) => typeof item === "object"
        ? item
        : { label: categories[index] ?? "-", value: item });
      const maximum = Math.max(0, ...series.map((item) => Number(item.value)).filter(Number.isFinite));
      return `<figure class="ops-tablespace-chart"><figcaption>${esc(payload.title || "指标对比")}</figcaption><div class="ops-chart-rows">${series.map((item) => { const raw = Number(item.value); const width = Number.isFinite(raw) && maximum > 0 ? Math.max(0, Math.min(100, raw / maximum * 100)) : 0; return `<div class="ops-chart-row"><span>${esc(item.label || item.name || "-")}</span><div class="ops-chart-track"><i style="width:${width}%"></i></div><strong>${esc(item.display_value ?? item.value ?? "-")}</strong></div>`; }).join("")}</div></figure>${citationHtml}`;
    }
    if (block.block_type === "EVIDENCE_REFERENCES") {
      const items = values(payload.items);
      return `<details class="ops-evidence"><summary>诊断依据 <span>${items.length} 项证据</span></summary><div class="ops-evidence-body"><ol class="ops-evidence-list">${items.map((item) => `<li><span>${esc(item.label || item.summary || "诊断证据")}</span><small>${esc(item.source || "EVIDENCE")}${item.observed_at ? ` · ${esc(shell.fmt(item.observed_at))}` : ""}</small></li>`).join("")}</ol></div></details>`;
    }
    return markdown.render(payload.markdown || payload.text || payload.instruction || "");
  }

  function turnHtml(turn) {
    const messages = values(turn.messages);
    const user = messages.find((item) => item.message_type === "USER_MESSAGE");
    const assistant = messages.find((item) => item.message_type === "ASSISTANT_MESSAGE");
    const blocks = values(turn.answer_blocks).map(answerBlockHtml).join("");
    const terminal = terminalTurnStatuses.has(turn.status);
    const answer = assistant || blocks ? `<article class="ops-message agent"><div class="ops-avatar">AI</div><div class="ops-message-body ops-result-markdown"><div class="ops-message-content">${blocks || markdown.render(assistant?.payload?.text || "")}</div></div></article>` : "";
    const progress = terminal && !turn.error_message ? "" : `<div class="ops-context-banner ops-progress" data-turn-progress="${esc(turn.turn_id)}">${esc(turn.error_message || `当前状态：${turn.status}`)}</div>`;
    return `${user ? messageHtml("USER", user.payload?.text || "", shell.fmt(user.created_at)) : ""}${answer}${progress}`;
  }

  async function renderConversation(conversation, turns) {
    state.conversation = conversation;
    state.turns = turns;
    document.getElementById("conversation-title").textContent = conversation.title || "诊断对话";
    document.getElementById("conversation-context").textContent = conversation.source_type === "RUN" ? "这次对话继承自告警或巡检结果。" : "人工发起的智能诊断。";
    const panel = document.getElementById("message-list");
    panel.innerHTML = conversation.source_run_id ? '<div class="ops-context-banner">已关联来源诊断；后续回答只会引用当前 Turn 明确关联的证据。</div>' : "";
    turns.forEach((turn) => panel.insertAdjacentHTML("beforeend", turnHtml(turn)));
    panel.scrollTop = panel.scrollHeight;
    document.querySelectorAll("[data-copy-code]").forEach((button) => { button.onclick = () => markdown.copyCode(button); });
  }

  async function loadConversation(id) {
    const conversation = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}`);
    const turnRows = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}/turns?limit=200`);
    const turns = await Promise.all(turnRows.map((turn) => KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}/turns/${encodeURIComponent(turn.turn_id)}`)));
    document.getElementById("agent-select").value = conversation.agent_id;
    await renderConversation(conversation, turns);
    history.replaceState(null, "", `./chat.html?conversation=${encodeURIComponent(id)}`);
    document.querySelectorAll(".ops-workspace-item").forEach((button) => { button.setAttribute("aria-current", String(button.dataset.id === id)); });
  }

  function resetConversationView({ agentSelected = false } = {}) {
    state.conversation = null;
    document.getElementById("conversation-title").textContent = agentSelected
      ? "开始一次数据库诊断"
      : "请先选择 Agent";
    document.getElementById("conversation-context").textContent = agentSelected
      ? "可以选择历史会话，或在下方发起一次新诊断。"
      : "选择 Agent 后，才会显示该 Agent 的会话历史。";
    document.getElementById("message-list").innerHTML = agentSelected
      ? '<div class="ops-empty">请在下方描述需要诊断的问题。</div>'
      : '<div class="ops-empty">请选择 Agent 以查看历史并开始诊断。</div>';
  }

  function setComposerAvailability(enabled) {
    const form = document.getElementById("conversation-form");
    form.elements.message.disabled = !enabled;
    form.querySelector('button[type="submit"]').disabled = !enabled;
  }

  function clearConversationUrl() {
    history.replaceState(null, "", "./chat.html");
  }

  async function archiveConversation(id, title) {
    if (!confirm(`确认删除会话“${title || "未命名会话"}”吗？\n\n会话将从聊天历史中移除；关联的诊断、证据和变更审计记录仍会保留。`)) return;
    await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (String(state.conversation?.conversation_id) === String(id)) {
      clearConversationUrl();
      resetConversationView({ agentSelected: true });
    }
    shell.toast("会话已从历史中移除");
    await loadConversationList();
  }

  function renderConversationList(rows) {
    const list = document.getElementById("conversation-list");
    list.innerHTML = rows.length ? rows.map((item) => `<div class="ops-workspace-row"><button class="ops-workspace-item" type="button" data-id="${esc(item.conversation_id)}"><strong>${esc(item.title || "未命名会话")}</strong><small>${esc(item.source_type)} · ${esc(shell.fmt(item.updated_at))}</small></button><button class="ops-workspace-delete" type="button" data-delete-id="${esc(item.conversation_id)}" data-delete-title="${esc(item.title || "未命名会话")}" aria-label="删除会话 ${esc(item.title || "未命名会话")}" title="删除会话">删除</button></div>`).join("") : '<div class="ops-empty">当前 Agent 还没有诊断会话</div>';
    list.querySelectorAll(".ops-workspace-item").forEach((button) => {
      button.onclick = () => loadConversation(button.dataset.id).catch((error) => shell.toast(error.message));
    });
    list.querySelectorAll(".ops-workspace-delete").forEach((button) => {
      button.onclick = () => archiveConversation(button.dataset.deleteId, button.dataset.deleteTitle).catch((error) => shell.toast(error.message));
    });
  }

  async function loadConversationList(preferredId) {
    const select = document.getElementById("agent-select");
    const requestedId = preferredId || new URLSearchParams(location.search).get("conversation");
    if (!select.value && requestedId) await loadConversation(requestedId);
    const selectedAgent = select.value;
    const list = document.getElementById("conversation-list");
    if (!selectedAgent) {
      list.innerHTML = '<div class="ops-empty">请先选择 Agent 查看其会话历史</div>';
      setComposerAvailability(false);
      resetConversationView();
      return;
    }
    setComposerAvailability(true);
    const rows = await KBotAIOpsAuth.request(`${api}/conversations?agent_id=${encodeURIComponent(selectedAgent)}`);
    renderConversationList(rows);
    if (requestedId && String(state.conversation?.conversation_id) !== String(requestedId)) {
      await loadConversation(requestedId);
    }
  }

  function typingUnits(value) {
    const text = String(value || "");
    if (!text) return [];
    if (!graphemeSegmenter) return Array.from(text);
    return Array.from(graphemeSegmenter.segment(text), (item) => item.segment);
  }

  function typingBatchSize(pending) {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return pending.queue.length;
    if (pending.finalizing) return Math.max(1, Math.ceil(pending.queue.length / 16));
    if (pending.queue.length > 800) return 8;
    if (pending.queue.length > 240) return 4;
    if (pending.queue.length > 80) return 2;
    return 1;
  }

  function settleTyping(pending) {
    pending.timer = null;
    pending.message.classList.remove("is-typing");
    pending.message.setAttribute("aria-busy", "false");
    pending.waiters.splice(0).forEach((resolve) => resolve());
  }

  function typingTick(pending) {
    if (!pending.queue.length) return settleTyping(pending);
    pending.displayedMarkdown += pending.queue.splice(0, typingBatchSize(pending)).join("");
    pending.content.innerHTML = markdown.render(pending.displayedMarkdown);
    const panel = document.getElementById("message-list");
    panel.scrollTop = panel.scrollHeight;
    pending.timer = window.setTimeout(() => typingTick(pending), typingFrameMs);
  }

  function enqueueAnswerDelta(pending, delta) {
    pending.queue.push(...typingUnits(delta));
    pending.message.classList.add("is-typing");
    pending.message.setAttribute("aria-busy", "true");
    if (pending.timer === null) typingTick(pending);
  }

  function waitForTyping(pending) {
    if (!pending.queue.length && pending.timer === null) return Promise.resolve();
    return new Promise((resolve) => pending.waiters.push(resolve));
  }

  async function followTurn(conversationId, turnId, progress) {
    let pending = null;
    let lastEventId = "";
    let completed = false;
    const path = `${api}/conversations/${encodeURIComponent(conversationId)}/turns/${encodeURIComponent(turnId)}/events`;
    const onEvent = ({ event, data, id }) => {
      if (id) lastEventId = id;
      const payload = data?.payload || {};
      if (["turn.created", "turn.status", "skill.status"].includes(event)) {
        progress.textContent = payload.public_summary || payload.summary || `当前状态：${payload.status || "处理中"}`;
      }
      if (event === "thinking.delta") {
        progress.textContent = payload.public_summary || payload.delta || "正在组织回答";
      }
      if (event === "answer.delta") {
        const delta = String(data?.payload?.delta || "");
        if (!pending) {
          progress.insertAdjacentHTML("beforebegin", messageHtml("AGENT", ""));
          const message = progress.previousElementSibling;
          pending = {
            message,
            content: message.querySelector(".ops-message-content"),
            displayedMarkdown: "",
            queue: [],
            timer: null,
            waiters: [],
            finalizing: false,
          };
        }
        enqueueAnswerDelta(pending, delta);
      }
      if (event === "answer.completed" && pending) pending.finalizing = true;
      if (event === "done") {
        completed = true;
        if (pending) pending.finalizing = true;
        progress.textContent = "诊断已完成，正在整理结论…";
      }
    };
    for (let attempt = 0; attempt < streamRecoveryAttempts && !completed; attempt += 1) {
      try {
        await KBotAIOpsAuth.stream(path, onEvent, {
          headers: lastEventId ? { "Last-Event-ID": lastEventId } : {},
        });
      } catch (_) {
        // 临时断流统一由权威 Turn 状态与续传游标恢复。
      }
      if (completed) break;
      try {
        const turn = await KBotAIOpsAuth.request(
          `${api}/conversations/${encodeURIComponent(conversationId)}/turns/${encodeURIComponent(turnId)}`,
        );
        if (terminalTurnStatuses.has(turn.status)) {
          completed = true;
          break;
        }
      } catch (_) {
        // 状态回读也可能遇到同一次短暂网络抖动，下一轮继续恢复。
      }
      progress.textContent = "事件流暂时中断，正在恢复诊断进度…";
      await new Promise((resolve) => window.setTimeout(resolve, Math.min(5000, 1000 * (attempt + 1))));
    }
    if (!completed) throw new Error("诊断仍在后台运行，请稍后刷新会话查看结果");
    if (pending) {
      pending.finalizing = true;
      await waitForTyping(pending);
    }
  }

  async function submitConversation(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button[type="submit"]');
    const agentId = document.getElementById("agent-select").value;
    const text = form.elements.message.value.trim();
    if (!agentId) return shell.toast("请先选择 Agent");
    if (!text) return shell.toast("请输入需要诊断的问题");
    if (state.selectedFile) return shell.toast("截图和文件补证将在补证阶段启用，请先发送文字问题");
    button.disabled = true;
    try {
      const path = state.conversation
        ? `${api}/conversations/${state.conversation.conversation_id}/turns`
        : `${api}/conversations`;
      const body = state.conversation ? { message: text } : { agent_id: agentId, message: text };
      const receipt = await KBotAIOpsAuth.request(path, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify(body),
      });
      form.reset(); state.selectedFile = null; document.getElementById("upload-preview").textContent = "";
      const panel = document.getElementById("message-list");
      panel.insertAdjacentHTML("beforeend", messageHtml("USER", text));
      panel.insertAdjacentHTML("beforeend", '<div id="live-progress" class="ops-context-banner ops-progress">正在建立诊断计划…</div>');
      const progress = document.getElementById("live-progress");
      await followTurn(receipt.conversation_id, receipt.turn_id, progress);
      await loadConversationList(receipt.conversation_id);
    } catch (error) { shell.toast(error.message); } finally { button.disabled = false; }
  }

  async function initChat() {
    const select = document.getElementById("agent-select");
    const rows = await agents();
    select.innerHTML = '<option value="">选择 Agent</option>' + rows.map((item) => `<option value="${esc(item.agent_id)}">${esc(item.display_name || item.name || item.agent_key || shell.short(item.agent_id))}</option>`).join("");
    select.onchange = () => {
      clearConversationUrl();
      resetConversationView({ agentSelected: Boolean(select.value) });
      loadConversationList().catch((error) => shell.toast(error.message));
    };
    document.getElementById("new-conversation").onclick = () => {
      if (!select.value) return shell.toast("请先选择 Agent");
      clearConversationUrl();
      resetConversationView({ agentSelected: true });
    };
    document.getElementById("conversation-form").onsubmit = submitConversation;
    document.getElementById("evidence-file").onchange = (event) => { state.selectedFile = event.target.files[0] || null; document.getElementById("upload-preview").textContent = state.selectedFile ? `待上传：${state.selectedFile.name}` : ""; };
    await loadConversationList();
  }

  function continueForm(run, title) {
    const allowed = state.agents.filter((item) => !item.target_id || String(item.target_id) === String(run.target_id));
    return `<div class="ops-inline-dialog"><h3>继续深入诊断</h3><p>系统会在服务端继承本次自动诊断证据；后续人工对话才可能按 Agent 权限产生审批待办。</p><form id="continue-form" class="ops-form"><div class="ops-field span-12"><label>Agent</label><select name="agent_id" required><option value="">请选择</option>${allowed.map((item) => `<option value="${esc(item.agent_id)}">${esc(item.display_name || item.name || item.agent_key || shell.short(item.agent_id))}</option>`).join("")}</select></div><div class="ops-field span-12"><label>继续追问</label><textarea name="message" required>${esc(`请基于“${title}”的自动诊断结果继续分析：`)}</textarea></div><div class="ops-filter-actions"><button class="primary" type="submit">进入对话</button></div></form></div>`;
  }

  async function bindContinue(run, title) {
    const form = document.getElementById("continue-form");
    if (!form) return;
    form.onsubmit = async (event) => {
      event.preventDefault();
      const body = Object.fromEntries(new FormData(form));
      try {
        const result = await KBotAIOpsAuth.request(`${api}/conversations`, { method: "POST", headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() }, body: JSON.stringify({ ...body, source_run_id: run.ops_run_id }) });
        location.href = `./chat.html?conversation=${encodeURIComponent(result.conversation_id)}`;
      } catch (error) { shell.toast(error.message); }
    };
  }

  async function showSituation(item) {
    const detail = await KBotAIOpsAuth.request(`${api}/situations/${encodeURIComponent(item.situation_id)}`);
    document.getElementById("case-title").textContent = detail.title;
    const panel = document.getElementById("case-detail");
    const runId = detail.run_ids[0];
    let run = null; let result = null;
    if (runId) { run = await KBotAIOpsAuth.request(`${api}/runs/${runId}`); result = await KBotAIOpsAuth.request(`${api}/runs/${runId}/result`); }
    panel.innerHTML = `<div class="ops-context-banner">${shell.badge(detail.severity)} ${shell.badge(detail.status)} · ${detail.event_count} 个监控信号 · 最近观测 ${esc(shell.fmt(detail.last_observed_at))}</div>${result ? `<div class="ops-result-markdown">${markdown.render(conversationAnswerMarkdown(result))}${evidenceDetails(result)}</div>${continueForm(run, detail.title)}` : '<div class="ops-empty">自动诊断尚未生成结果。</div>'}`;
    await bindContinue(run, detail.title);
  }

  async function showInspection(item) {
    const detail = await KBotAIOpsAuth.request(`${api}/inspection-fires/${encodeURIComponent(item.fire_id)}`);
    document.getElementById("case-title").textContent = `巡检 ${shell.fmt(detail.scheduled_at)}`;
    const panel = document.getElementById("case-detail");
    const runId = detail.run_ids[0];
    let run = null; let result = null;
    if (runId) { run = await KBotAIOpsAuth.request(`${api}/runs/${runId}`); result = await KBotAIOpsAuth.request(`${api}/runs/${runId}/result`); }
    panel.innerHTML = `<div class="ops-context-banner">${shell.badge(detail.status)} · ${detail.completed_count}/${detail.target_count} 个目标完成 · ${detail.failed_count} 个失败</div>${result ? `<div class="ops-result-markdown">${markdown.render(inspectionMarkdown(result))}</div>${continueForm(run, "本次日常巡检")}` : '<div class="ops-empty">本次巡检尚未形成可展示结果。</div>'}`;
    await bindContinue(run, "本次日常巡检");
  }

  async function initCases(page) {
    await agents();
    const endpoint = page === "situations" ? "/situations" : "/inspection-fires";
    const payload = await KBotAIOpsAuth.request(`${api}${endpoint}?limit=100`);
    const rows = payload.items || [];
    const list = document.getElementById("case-list");
    list.innerHTML = rows.length ? rows.map((item) => `<button class="ops-case-row" data-id="${esc(item.situation_id || item.fire_id)}"><strong>${esc(item.title || `巡检 ${shell.fmt(item.scheduled_at)}`)}</strong>${shell.badge(item.severity || item.status)}<p>${esc(item.summary || `${item.completed_count || 0}/${item.target_count || 0} 个目标已完成`)}</p></button>`).join("") : '<div class="ops-empty">当前范围内暂无记录</div>';
    list.querySelectorAll("button").forEach((button, index) => { button.onclick = () => (page === "situations" ? showSituation(rows[index]) : showInspection(rows[index])).catch((error) => shell.toast(error.message)); });
    document.getElementById("refresh-workspace").onclick = () => location.reload();
    if (rows[0]) await (page === "situations" ? showSituation(rows[0]) : showInspection(rows[0]));
  }

  addEventListener("click", (event) => {
    const copyButton = event.target.closest("[data-copy-code]");
    if (copyButton) markdown.copyCode(copyButton);
    const proposalButton = event.target.closest("[data-approve-proposal],[data-reject-proposal]");
    if (proposalButton) proposalAction(proposalButton);
  });
  shell.ready.then(() => {
    const page = document.body.dataset.page;
    if (page === "chat") return initChat();
    if (["situations", "inspections"].includes(page)) return initCases(page);
    return null;
  }).catch((error) => shell.toast(error.message));
})();
