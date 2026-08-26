(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const markdown = globalThis.KBotMarkdown;
  const state = { agents: [], conversation: null, selectedFile: null };

  const esc = shell.escape;
  const values = (items) => Array.isArray(items) ? items : [];
  const bullets = (items) => values(items).length
    ? values(items).map((item) => `- ${typeof item === "string" ? item : item.fact_summary || item.summary || item.title || "已记录"}`).join("\n")
    : "- 无";

  function resultMarkdown(result) {
    const payload = result?.payload || {};
    if (!result?.final_artifact) {
      return `### 诊断尚未形成最终结论\n\n当前状态：${result?.status || "处理中"}`;
    }
    if (payload.schema_version === "REPORT_CONTENT.v1" || payload.title) {
      return `## ${payload.title || "巡检报告"}\n\n${payload.summary || ""}\n\n### 发现\n${bullets(payload.facts)}\n\n### 建议\n${bullets(payload.recommendations)}\n\n### 数据缺口\n${bullets(payload.gaps)}`;
    }
    const root = payload.root_cause || {};
    const direct = payload.direct_answer?.answer_text;
    const solution = payload.solution || {};
    return [
      `## ${direct ? "诊断回答" : "诊断结论"}`,
      direct || payload.diagnosis_rationale || "当前证据未形成可展示的文字结论。",
      `**根因等级：** ${root.effective_level || result.root_cause_grade || "INCONCLUSIVE"}`,
      "### 已验证事实", bullets(payload.facts),
      "### 立即建议", bullets(solution.immediate_mitigations),
      "### 长期建议", bullets(solution.long_term_remediations),
      "### 尚缺证据", bullets(payload.gaps),
    ].join("\n\n");
  }

  function messageHtml(role, text, meta = "") {
    const user = role === "USER";
    return `<article class="ops-message ${user ? "user" : "agent"}"><div class="ops-avatar">${user ? "我" : "AI"}</div><div class="ops-message-body ops-result-markdown">${markdown.render(text)}${meta ? `<div class="ops-message-meta">${esc(meta)}</div>` : ""}</div></article>`;
  }

  function messageText(item) {
    const payload = item.payload || {};
    if (item.message_type === "EVIDENCE_REQUEST") {
      const markers = values(payload.query_ids).length > 1
        ? `\n\n这次包含多条查询。粘贴文本时，请依次使用 ${payload.query_ids.map((id) => `\`[${id}]\``).join("、")} 作为各段结果的标题；也可以直接上传完整截图。`
        : "";
      return `${payload.purpose || "为了继续判断，还需要补充一项证据。"}${payload.suggested_sql ? `\n\n请在目标数据库手工执行以下只读 SQL，并将结果或截图发回这里：\n\n\`\`\`sql\n${payload.suggested_sql}\n\`\`\`${markers}` : "\n\n请把相关输出或截图直接发回这里。"}`;
    }
    if (item.message_type === "AGENT_PROGRESS") return payload.summary || "正在诊断…";
    if (item.message_type === "EVIDENCE_FILE") return `已上传补充证据：${payload.filename || "结果截图"}`;
    if (item.message_type === "IMAGE_EVIDENCE_PROCESSED") return `已读取截图内容：\n\n${payload.text || "图片中未提取到可用文字。"}`;
    return payload.text || payload.summary || "";
  }

  function openEvidenceRequest(conversation) {
    const requests = conversation.messages.filter((item) => item.message_type === "EVIDENCE_REQUEST");
    const completed = new Set(conversation.messages.filter((item) => ["EVIDENCE_TEXT", "EVIDENCE_FILE", "EVIDENCE_SKIPPED"].includes(item.message_type)).map((item) => item.payload?.request_id));
    return [...requests].reverse().find((item) => !completed.has(item.payload?.request_id)) || null;
  }

  async function agents() {
    const rows = await KBotAIOpsAuth.request(`${api}/agents`);
    state.agents = values(rows).filter((item) => item.status === "ACTIVE");
    return state.agents;
  }

  async function renderRunResult(runId, container) {
    try {
      const result = await KBotAIOpsAuth.request(`${api}/runs/${encodeURIComponent(runId)}/result`);
      if (!result.final_artifact && !["COMPLETED", "DEGRADED", "FAILED", "CANCELLED", "REJECTED", "EXPIRED"].includes(result.status)) {
        container.insertAdjacentHTML("beforeend", messageHtml("AGENT", "诊断仍在进行中，页面会通过事件流持续更新。", result.status));
        return;
      }
      container.insertAdjacentHTML("beforeend", messageHtml("AGENT", resultMarkdown(result), `Run ${shell.short(runId)}`));
      try { await renderProposals(runId, container); } catch (_) { /* 无审批权限时仍展示诊断结果。 */ }
    } catch (error) {
      container.insertAdjacentHTML("beforeend", messageHtml("AGENT", `无法读取本次诊断结果：${error.message}`));
    }
  }

  async function renderProposals(runId, container) {
    const run = await KBotAIOpsAuth.request(`${api}/runs/${encodeURIComponent(runId)}`);
    const page = await KBotAIOpsAuth.request(`${api}/proposals?target_id=${encodeURIComponent(run.target_id)}&limit=100`);
    const proposals = values(page.items).filter((item) => String(item.ops_run_id) === String(runId));
    proposals.forEach((item) => {
      const executable = item.mode === "AGENT_EXECUTE" && item.status === "PENDING_APPROVAL";
      const body = `### ${executable ? "待人工审批的变更" : "处理建议"}\n\n**影响：** ${item.impact}\n\n**风险：** ${item.risk}\n\n**回滚：** ${item.rollback_plan}\n\n\`\`\`sql\n${item.command_preview}\n\`\`\``;
      container.insertAdjacentHTML("beforeend", `<article class="ops-message agent"><div class="ops-avatar">AI</div><div class="ops-message-body ops-result-markdown">${markdown.render(body)}${executable ? `<div class="ops-actions"><button type="button" class="primary" data-approve-proposal="${esc(item.proposal_id)}" data-version="${item.row_version}" data-hash="${esc(item.proposal_hash)}">审批并执行</button><button type="button" data-reject-proposal="${esc(item.proposal_id)}" data-version="${item.row_version}">拒绝</button></div>` : `<div class="ops-message-meta">${esc(item.status)} · 当前只提供人工建议</div>`}</div></article>`);
    });
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

  async function renderConversation(conversation) {
    state.conversation = conversation;
    document.getElementById("conversation-title").textContent = conversation.messages.find((item) => item.role === "USER")?.payload?.text || "诊断对话";
    document.getElementById("conversation-context").textContent = conversation.source_type === "ALERT" ? "这次对话继承自告警自动诊断。" : conversation.source_type === "INSPECTION" ? "这次对话继承自日常巡检。" : "人工发起的智能诊断。";
    const panel = document.getElementById("message-list");
    panel.innerHTML = conversation.source_run_id ? '<div class="ops-context-banner">已由服务端关联原始自动诊断证据；后续结论会继续核验，不会把历史推断当作新事实。</div>' : "";
    conversation.messages.forEach((item) => {
      panel.insertAdjacentHTML("beforeend", messageHtml(item.role, messageText(item), shell.fmt(item.created_at)));
    });
    for (const link of conversation.runs) {
      if (!["AUTO_DIAGNOSIS_SEED", "INSPECTION_SEED"].includes(link.purpose)) await renderRunResult(link.ops_run_id, panel);
    }
    panel.scrollTop = panel.scrollHeight;
    document.querySelectorAll("[data-copy-code]").forEach((button) => { button.onclick = () => markdown.copyCode(button); });
  }

  async function loadConversation(id) {
    const conversation = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}`);
    document.getElementById("agent-select").value = conversation.agent_id;
    await renderConversation(conversation);
    document.querySelectorAll(".ops-workspace-item").forEach((button) => { button.setAttribute("aria-current", String(button.dataset.id === id)); });
  }

  async function loadConversationList(preferredId) {
    const selectedAgent = document.getElementById("agent-select").value;
    const query = selectedAgent ? `?agent_id=${encodeURIComponent(selectedAgent)}` : "";
    const rows = await KBotAIOpsAuth.request(`${api}/conversations${query}`);
    const list = document.getElementById("conversation-list");
    list.innerHTML = rows.length ? rows.map((item) => `<button class="ops-workspace-item" data-id="${esc(item.conversation_id)}"><strong>${esc(item.title)}</strong><small>${esc(item.source_type)} · ${esc(shell.fmt(item.updated_at))}</small></button>`).join("") : '<div class="ops-empty">还没有诊断会话</div>';
    list.querySelectorAll("button").forEach((button) => { button.onclick = () => loadConversation(button.dataset.id).catch((error) => shell.toast(error.message)); });
    const id = preferredId || new URLSearchParams(location.search).get("conversation");
    if (id) await loadConversation(id);
  }

  async function followRun(runId, progress) {
    let summary = "诊断任务已经开始";
    await KBotAIOpsAuth.stream(`${api}/runs/${encodeURIComponent(runId)}/events`, ({ event, data }) => {
      if (event === "task.status") {
        const payload = data?.payload || {};
        summary = payload.task_key ? `正在执行：${payload.task_key}` : summary;
        progress.textContent = summary;
      }
      if (event === "done") progress.textContent = "诊断已完成，正在整理结论…";
    });
  }

  async function fileBase64(file) {
    if (file.size > 10 * 1024 * 1024) throw new Error("结果截图不能超过 10 MiB");
    const url = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(file); });
    return String(url).split(",", 2)[1];
  }

  async function submitConversation(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button[type="submit"]');
    const agentId = document.getElementById("agent-select").value;
    const text = form.elements.message.value.trim();
    if (!agentId) return shell.toast("请先选择 Agent");
    button.disabled = true;
    try {
      const request = state.conversation ? openEvidenceRequest(state.conversation) : null;
      if (state.selectedFile) {
        if (!request) throw new Error("只有 Agent 请求补充证据后才能上传结果截图");
        await KBotAIOpsAuth.request(`${api}/conversations/${state.conversation.conversation_id}/evidence-requests/${request.payload.request_id}/uploads`, { method: "POST", body: JSON.stringify({ filename: state.selectedFile.name, mime_type: state.selectedFile.type, content_base64: await fileBase64(state.selectedFile), text: text || null }) });
      } else if (request) {
        await KBotAIOpsAuth.request(`${api}/conversations/${state.conversation.conversation_id}/evidence-requests/${request.payload.request_id}/text`, { method: "POST", body: JSON.stringify({ text }) });
      }
      if (request) {
        const conversationId = state.conversation.conversation_id;
        const resumedRun = [...state.conversation.runs].reverse().find((item) => item.purpose === "QUESTION");
        const submittedFileName = state.selectedFile?.name;
        form.reset(); state.selectedFile = null; document.getElementById("upload-preview").textContent = "";
        const panel = document.getElementById("message-list");
        panel.insertAdjacentHTML("beforeend", messageHtml("USER", submittedFileName ? `已上传结果截图：${submittedFileName}` : text));
        panel.insertAdjacentHTML("beforeend", '<div id="live-progress" class="ops-context-banner ops-progress">已接收补充证据，正在继续原诊断…</div>');
        const progress = document.getElementById("live-progress");
        if (resumedRun) {
          try { await followRun(resumedRun.ops_run_id, progress); } catch (error) { shell.toast(`事件流已中断：${error.message}`); }
        }
        await loadConversationList(conversationId);
        return;
      }
      const receipt = await KBotAIOpsAuth.request(`${api}/conversations`, { method: "POST", body: JSON.stringify({ agent_id: agentId, message: text, conversation_id: state.conversation?.conversation_id || null }) });
      form.reset(); state.selectedFile = null; document.getElementById("upload-preview").textContent = "";
      const panel = document.getElementById("message-list");
      panel.insertAdjacentHTML("beforeend", messageHtml("USER", text));
      panel.insertAdjacentHTML("beforeend", '<div id="live-progress" class="ops-context-banner ops-progress">正在建立诊断计划…</div>');
      const progress = document.getElementById("live-progress");
      try { await followRun(receipt.run_id, progress); } catch (error) { shell.toast(`事件流已中断：${error.message}`); }
      await loadConversationList(receipt.conversation_id);
    } catch (error) { shell.toast(error.message); } finally { button.disabled = false; }
  }

  async function initChat() {
    const select = document.getElementById("agent-select");
    const rows = await agents();
    select.innerHTML = '<option value="">选择 Agent</option>' + rows.map((item) => `<option value="${esc(item.agent_id)}">${esc(item.display_name || item.name || item.agent_key || shell.short(item.agent_id))}</option>`).join("");
    select.onchange = () => { state.conversation = null; loadConversationList().catch((error) => shell.toast(error.message)); };
    document.getElementById("new-conversation").onclick = () => { state.conversation = null; history.replaceState(null, "", "./chat.html"); document.getElementById("message-list").innerHTML = '<div class="ops-empty">请在下方描述需要诊断的问题。</div>'; document.getElementById("conversation-title").textContent = "开始一次数据库诊断"; };
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
        const result = await KBotAIOpsAuth.request(`${api}/conversations`, { method: "POST", body: JSON.stringify({ ...body, source_run_id: run.ops_run_id }) });
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
    panel.innerHTML = `<div class="ops-context-banner">${shell.badge(detail.severity)} ${shell.badge(detail.status)} · ${detail.event_count} 个监控信号 · 最近观测 ${esc(shell.fmt(detail.last_observed_at))}</div>${result ? `<div class="ops-result-markdown">${markdown.render(resultMarkdown(result))}</div>${continueForm(run, detail.title)}` : '<div class="ops-empty">自动诊断尚未生成结果。</div>'}`;
    await bindContinue(run, detail.title);
  }

  async function showInspection(item) {
    const detail = await KBotAIOpsAuth.request(`${api}/inspection-fires/${encodeURIComponent(item.fire_id)}`);
    document.getElementById("case-title").textContent = `巡检 ${shell.fmt(detail.scheduled_at)}`;
    const panel = document.getElementById("case-detail");
    const runId = detail.run_ids[0];
    let run = null; let result = null;
    if (runId) { run = await KBotAIOpsAuth.request(`${api}/runs/${runId}`); result = await KBotAIOpsAuth.request(`${api}/runs/${runId}/result`); }
    panel.innerHTML = `<div class="ops-context-banner">${shell.badge(detail.status)} · ${detail.completed_count}/${detail.target_count} 个目标完成 · ${detail.failed_count} 个失败</div>${result ? `<div class="ops-result-markdown">${markdown.render(resultMarkdown(result))}</div>${continueForm(run, "本次日常巡检")}` : '<div class="ops-empty">本次巡检尚未形成可展示结果。</div>'}`;
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
