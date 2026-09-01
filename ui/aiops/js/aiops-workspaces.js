(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const markdown = globalThis.KBotMarkdown;
  const state = { agents: [], targets: [], conversation: null, selectedFile: null };
  const typingFrameMs = 22;
  const streamRecoveryAttempts = 120;
  const activeTurnFollowers = new Set();
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

  function uploadMediaType(file) {
    const suffix = String(file.name || "").toLowerCase().split(".").pop();
    if (["log", "txt"].includes(suffix)) return "text/plain";
    if (suffix === "csv") return "text/csv";
    if (suffix === "json") return "application/json";
    if (suffix === "sql") return "application/sql";
    if (file.type) return file.type;
    return "application/octet-stream";
  }

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

  function situationEvidence(detail) {
    const events = values(detail?.signal_events);
    if (!events.length) return '<p class="ops-evidence-empty">当前情境没有可展示的监控信号。</p>';
    const rows = events.map((item) => `<li><span>${esc(item.summary || item.event_class || "监控信号")}</span><small>${esc(item.event_class || item.signal_kind || "SIGNAL")} · ${esc(item.severity || "UNKNOWN")} · ${esc(item.normalized_status || "UNKNOWN")}${item.occurred_at ? ` · ${esc(shell.fmt(item.occurred_at))}` : ""}</small></li>`).join("");
    return `<details class="ops-evidence" open><summary>关联监控信号 <span>${events.length} 条</span></summary><div class="ops-evidence-body"><ol class="ops-evidence-list">${rows}</ol></div></details>`;
  }

  function messageHtml(role, text, meta = "", supplemental = "") {
    const user = role === "USER";
    return `<article class="ops-message ${user ? "user" : "agent"}"><div class="ops-avatar">${user ? "我" : "AI"}</div><div class="ops-message-body ops-result-markdown"><div class="ops-message-content">${markdown.render(text)}</div>${supplemental}${meta ? `<div class="ops-message-meta">${esc(meta)}</div>` : ""}</div></article>`;
  }

  function investigationPlanHtml(plan) {
    const actions = values(plan?.actions);
    const frame = plan?.task_frame || {};
    const hypotheses = values(plan?.hypotheses);
    if (!actions.length && !frame.problem_statement && !hypotheses.length) return "";
    const list = (title, items) => values(items).length
      ? `<div class="ops-plan-section"><strong>${esc(title)}</strong><ul>${values(items).map((item) => `<li>${esc(item)}</li>`).join("")}</ul></div>`
      : "";
    const frameHtml = frame.problem_statement
      ? `<div class="ops-plan-frame"><p><strong>问题定义：</strong>${esc(frame.problem_statement)}</p>${list("当前已知", frame.known_facts)}${list("待验证", frame.unknowns)}${list("完成标准", frame.success_criteria)}</div>`
      : "";
    const hypothesisHtml = hypotheses.length
      ? `<div class="ops-plan-hypotheses"><strong>待验证假设</strong><ol>${hypotheses.map((item) => {
        const confidence = Number.isFinite(Number(item.confidence)) ? ` · 初始置信度 ${Math.round(Number(item.confidence) * 100)}%` : "";
        return `<li><span>${esc(item.statement || "待验证假设")}</span>${item.rationale || confidence ? `<small>${item.rationale ? `判断依据摘要：${esc(item.rationale)}` : ""}${esc(confidence)}</small>` : ""}</li>`;
      }).join("")}</ol></div>`
      : "";
    const rows = actions.map((action) => {
      const approval = action.execution_mode === "APPROVAL_REQUIRED";
      const mode = approval ? "需人工审批" : "自动只读执行";
      const status = action.status || "PLANNED";
      const evidence = action.expected_evidence_kind ? ` · 预期证据 ${action.expected_evidence_kind}` : "";
      const dependency = values(action.depends_on).length ? ` · 依赖 ${values(action.depends_on).join("、")}` : "";
      const query = action.sql_text
        ? `<details class="ops-plan-query"><summary>查看待执行 SQL 与参数</summary><pre><code>${esc(action.sql_text)}</code></pre><strong>绑定参数</strong><pre><code>${esc(JSON.stringify(action.parameters || {}, null, 2))}</code></pre></details>`
        : "";
      return `<li data-plan-action="${esc(action.action_id)}"><span>${esc(action.question || "执行诊断步骤")}</span><small>${esc(action.tool_class || action.tool_id || "DIAGNOSTIC")} · ${esc(mode)} · ${esc(status)}${esc(evidence)}${esc(dependency)}</small>${query}</li>`;
    }).join("");
    const actionsHtml = actions.length ? `<div class="ops-plan-actions"><strong>取证步骤</strong><ol>${rows}</ol></div>` : `<p class="ops-plan-empty">现有材料已足够，本轮不需要调用额外诊断工具。</p>`;
    return `<section class="ops-investigation-plan" data-plan-revision="${esc(plan.revision_no || 1)}"><header><strong>调查计划与判断依据</strong><span>第 ${esc(plan.revision_no || 1)} 版 · ${actions.length} 个步骤</span></header>${frameHtml}${hypothesisHtml}${actionsHtml}</section>`;
  }

  function showInvestigationPlan(progress, plan) {
    const html = investigationPlanHtml(plan);
    if (!html) return;
    const existing = progress.parentElement?.querySelector(".ops-investigation-plan.is-live");
    if (existing) existing.remove();
    progress.insertAdjacentHTML("beforebegin", html.replace("ops-investigation-plan", "ops-investigation-plan is-live"));
  }

  function ensureProgressTimeline(progress) {
    if (progress.dataset.timelineReady === "true") return;
    const initial = progress.textContent.trim() || "正在建立诊断计划：先固定执行上下文，再理解问题并选择证据…";
    progress.dataset.timelineReady = "true";
    progress.dataset.startedAt = String(Date.now());
    progress.innerHTML = `<header><strong>诊断过程</strong><span class="ops-progress-elapsed">已运行 0 秒</span></header><ol class="ops-progress-timeline"></ol>`;
    appendProgress(progress, "client.started", {
      public_summary: initial,
      public_sections: [{
        title: "计划将包含",
        items: ["问题定义与完成标准", "当前已知和待验证项", "候选假设、取证步骤及每步预期证据"],
      }],
    }, "client.started");
  }

  function appendProgress(progress, event, payload = {}, eventId = "") {
    ensureProgressTimeline(progress);
    const summary = String(payload.public_summary || payload.summary || `当前状态：${payload.status || "处理中"}`);
    const key = String(eventId || `${event}:${payload.action_id || payload.status || payload.revision_no || "current"}`);
    const timeline = progress.querySelector(".ops-progress-timeline");
    timeline.querySelectorAll("li.is-active").forEach((item) => item.classList.remove("is-active"));
    let row = [...timeline.children].find((item) => item.dataset.progressKey === key);
    if (!row) {
      row = document.createElement("li");
      row.dataset.progressKey = key;
      row.innerHTML = `<i aria-hidden="true"></i><div class="ops-progress-content"><span></span><div class="ops-progress-details"></div></div><small></small>`;
      timeline.append(row);
    }
    row.classList.add("is-active");
    row.querySelector(".ops-progress-content > span").textContent = summary;
    const details = row.querySelector(".ops-progress-details");
    const sections = values(payload.public_sections).filter((section) => values(section?.items).length);
    details.innerHTML = sections.map((section) => `<section><strong>${esc(section.title || "阶段详情")}</strong><ul>${values(section.items).map((item) => `<li>${esc(item)}</li>`).join("")}</ul></section>`).join("");
    details.hidden = !sections.length;
    row.querySelector("small").textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function updateProgressElapsed(progress) {
    const startedAt = Number(progress.dataset.startedAt || Date.now());
    const seconds = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
    const label = progress.querySelector(".ops-progress-elapsed");
    if (label) label.textContent = `已运行 ${seconds} 秒`;
  }

  function diagnosticQueryApprovalHtml(pending) {
    const request = pending?.request || {};
    if (pending?.hitl_type !== "DIAGNOSTIC_QUERY_APPROVAL") return "";
    const reasons = values(request.reason_codes).length
      ? `<p><strong>审批原因：</strong>${values(request.reason_codes).map(esc).join("、")}</p>`
      : "";
    return `<section class="ops-query-approval" data-query-approval="${esc(pending.hitl_id)}"><header><strong>动态只读查询待审批</strong><span>${esc(request.target_display_name || "当前 Target")}</span></header><p>${esc(request.purpose || "执行动态只读诊断")}</p>${reasons}<strong>规范化 SQL</strong><pre><code>${esc(request.sql_text || "")}</code></pre><strong>绑定参数</strong><pre><code>${esc(JSON.stringify(request.parameters || {}, null, 2))}</code></pre><p class="ops-query-limits">最多 ${esc(request.max_rows || "-")} 行 · 超时 ${esc(request.timeout_seconds || "-")} 秒 · ${esc(shell.fmt(request.expires_at))} 前有效</p><div class="ops-query-actions"><button type="button" class="primary" data-query-decision="APPROVE" data-hitl-id="${esc(pending.hitl_id)}" data-row-version="${esc(pending.row_version)}">批准并继续</button><button type="button" data-query-decision="REJECT" data-hitl-id="${esc(pending.hitl_id)}" data-row-version="${esc(pending.row_version)}">拒绝并继续分析</button></div></section>`;
  }

  async function diagnosticQueryDecision(button) {
    const approving = button.dataset.queryDecision === "APPROVE";
    const note = approving
      ? "用户已核对并批准该动态只读查询"
      : prompt("请输入拒绝原因");
    if (!note) return;
    const card = button.closest(".ops-query-approval");
    card.querySelectorAll("button").forEach((item) => { item.disabled = true; });
    try {
      await KBotAIOpsAuth.request(`${api}/hitl/${encodeURIComponent(button.dataset.hitlId)}/decision`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify({
          expected_row_version: Number(button.dataset.rowVersion),
          decision: approving ? "APPROVE" : "REJECT",
          note,
        }),
      });
      card.querySelector(".ops-query-actions").innerHTML = `<p>${approving ? "已批准，诊断继续执行。" : "已拒绝，Agent 将依据现有证据继续分析。"}</p>`;
      const progress = card.nextElementSibling?.matches("[data-turn-progress]")
        ? card.nextElementSibling
        : null;
      const conversationId = state.conversation?.conversation_id;
      if (progress && progress.id !== "live-progress" && conversationId) {
        appendProgress(progress, "approval.submitted", { public_summary: "审批已提交，正在继续诊断…" });
        followTurn(conversationId, progress.dataset.turnProgress, progress)
          .then(() => loadConversation(conversationId))
          .catch((error) => shell.toast(error.message));
      }
    } catch (error) {
      shell.toast(error.message);
      card.querySelectorAll("button").forEach((item) => { item.disabled = false; });
    }
  }

  function bindDiagnosticQueryActions(root) {
    root.querySelectorAll("[data-query-decision]").forEach((button) => {
      button.onclick = () => diagnosticQueryDecision(button);
    });
  }

  async function showDiagnosticQueryApproval(progress, hitlId) {
    const pending = await KBotAIOpsAuth.request(`${api}/hitl/${encodeURIComponent(hitlId)}`);
    const html = diagnosticQueryApprovalHtml(pending);
    if (!html) return;
    progress.parentElement?.querySelector(`[data-query-approval="${String(hitlId)}"]`)?.remove();
    progress.insertAdjacentHTML("beforebegin", html);
    bindDiagnosticQueryActions(progress.previousElementSibling);
  }

  async function agents() {
    const [rows, targetPage] = await Promise.all([
      KBotAIOpsAuth.request(`${api}/agents`),
      KBotAIOpsAuth.request(`${api}/targets?status=ENABLED&limit=200`),
    ]);
    state.agents = values(rows).filter((item) => item.status === "ACTIVE");
    state.targets = values(targetPage?.items);
    return state.agents;
  }

  function syncAgentTargetContext() {
    const agentId = document.getElementById("agent-select").value;
    const targetId = document.getElementById("target-select").value;
    const context = document.getElementById("agent-target-context");
    const agent = state.agents.find((item) => String(item.agent_id) === String(agentId));
    const target = state.targets.find((item) => String(item.target_id) === String(targetId));
    context.textContent = !targetId
      ? "先选择要运维的数据库对象。"
      : !agentId
        ? `已选择 ${target?.display_name || shell.short(targetId)}，请选择 Agent。`
      : target
        ? `诊断 Target：${target.display_name || shell.short(target.target_id)} · ${target.readonly_connection_enabled ? (target.connectivity_status || "UNKNOWN") : "仅监控模式"}`
        : "当前 Target 不可用";
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
    if (block.block_type === "MARKDOWN") return markdown.render(payload.markdown || payload.text || "");
    if (block.block_type === "TABLE") {
      const columns = values(payload.columns);
      const cell = (row, column, index) => Array.isArray(row)
        ? row[index]
        : row?.[column.key || column.name || column];
      return `<div class="ops-table-wrap"><table><thead><tr>${columns.map((column) => `<th>${esc(column.label || column.name || column.key || column)}</th>`).join("")}</tr></thead><tbody>${values(payload.rows).map((row) => `<tr>${columns.map((column, index) => `<td>${esc(cell(row, column, index) ?? "-")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    }
    if (block.block_type === "CHART") {
      const categories = values(payload.categories);
      const sourceSeries = values(payload.series);
      const series = sourceSeries.map((item, index) => typeof item === "object"
        ? item
        : { label: categories[index] ?? "-", value: item });
      const maximum = Math.max(0, ...series.map((item) => Number(item.value)).filter(Number.isFinite));
      return `<figure class="ops-tablespace-chart"><figcaption>${esc(payload.title || "指标对比")}</figcaption><div class="ops-chart-rows">${series.map((item) => { const raw = Number(item.value); const width = Number.isFinite(raw) && maximum > 0 ? Math.max(0, Math.min(100, raw / maximum * 100)) : 0; return `<div class="ops-chart-row"><span>${esc(item.label || item.name || "-")}</span><div class="ops-chart-track"><i style="width:${width}%"></i></div><strong>${esc(item.display_value ?? item.value ?? "-")}</strong></div>`; }).join("")}</div></figure>`;
    }
    if (block.block_type === "PROPOSAL_SUMMARY") {
      const parameters = Object.entries(payload.parameters || {}).map(([key, value]) => `<li><code>${esc(key)}</code><span>${esc(value)}</span></li>`).join("");
      const pending = payload.status === "PENDING_APPROVAL";
      const actions = pending
        ? `<div class="ops-proposal-actions"><button type="button" class="primary" data-approve-proposal="${esc(payload.proposal_id)}" data-version="${esc(payload.row_version || 1)}" data-hash="${esc(payload.proposal_hash)}">批准并执行</button><button type="button" data-reject-proposal="${esc(payload.proposal_id)}" data-version="${esc(payload.row_version || 1)}">拒绝</button></div>`
        : `<p class="ops-proposal-status">当前状态：${esc(payload.status || "UNKNOWN")}</p>`;
      return `<section class="ops-proposal"><header><div><strong>受控变更待审批</strong><small>${esc(payload.action_template_id || "Action Template")} · ${esc(payload.risk_level || "UNKNOWN")}</small></div></header><p>${esc(payload.rationale || "")}</p><p><strong>影响范围：</strong>${esc(payload.impact || "-")}</p>${parameters ? `<ul class="ops-proposal-parameters">${parameters}</ul>` : ""}${actions}</section>`;
    }
    if (block.block_type === "EVIDENCE_REFERENCES") return "";
    return markdown.render(payload.markdown || payload.text || payload.instruction || "");
  }

  function turnEvidenceHtml(blocks, gaps = []) {
    const evidence = new Map();
    const dataBlocks = blocks.filter((block) => ["TABLE", "CHART"].includes(block.block_type));
    const add = (key, label, meta) => {
      const normalizedLabel = String(label || "诊断证据").trim();
      const normalizedKey = String(key || normalizedLabel.toLowerCase());
      if (!evidence.has(normalizedKey)) evidence.set(normalizedKey, { label: normalizedLabel, meta });
    };
    blocks.forEach((block) => {
      values(block.citations).forEach((item) => add(
        item.turn_evidence_id || `citation:${item.label || item.citation_no}`,
        item.label || `证据 ${item.citation_no}`,
        shell.short(item.turn_evidence_id),
      ));
      if (block.block_type !== "EVIDENCE_REFERENCES") return;
      values(block.payload?.items).forEach((item, index) => add(
        item.turn_evidence_id || item.artifact_id || `reference:${item.label || item.summary || index}`,
        item.label || item.summary || "诊断证据",
        `${item.source || "EVIDENCE"}${item.observed_at ? ` · ${shell.fmt(item.observed_at)}` : ""}`,
      ));
    });
    const rows = Array.from(evidence.values());
    const gapRows = values(gaps).map((item) => ({
      label: item.detail || item.code || "本次未取得证据",
      meta: `${item.code || "EVIDENCE_GAP"}${item.step_id ? ` · ${item.step_id}` : ""}`,
    }));
    if (!rows.length && !gapRows.length && !dataBlocks.length) return "";
    const evidenceRows = rows.length
      ? `<ol class="ops-evidence-list">${rows.map((item) => `<li><span>${esc(item.label)}</span><small>${esc(item.meta || "EVIDENCE")}</small></li>`).join("")}</ol>`
      : dataBlocks.length
        ? ""
        : '<p class="ops-evidence-empty">本次没有形成可展示的有效证据。</p>';
    const evidenceData = dataBlocks.length
      ? `<div class="ops-evidence-data"><strong>原始取证结果</strong>${dataBlocks.map((block) => { const payload = block.payload || {}; const meta = [payload.measurement_semantics, payload.captured_at ? shell.fmt(payload.captured_at) : ""].filter(Boolean).join(" · "); return `<section><header><span>${esc(payload.title || (block.block_type === "CHART" ? "指标图表" : "查询结果"))}</span>${meta ? `<small>${esc(meta)}</small>` : ""}</header>${answerBlockHtml(block)}</section>`; }).join("")}</div>`
      : "";
    const missingRows = gapRows.length
      ? `<div class="ops-evidence-gaps"><strong>未取得的证据</strong><ol class="ops-evidence-list">${gapRows.map((item) => `<li><span>${esc(item.label)}</span><small>${esc(item.meta)}</small></li>`).join("")}</ol></div>`
      : "";
    return `<details class="ops-evidence"><summary>诊断依据 <span>${rows.length} 项证据${dataBlocks.length ? ` · ${dataBlocks.length} 份原始结果` : ""}${gapRows.length ? ` · ${gapRows.length} 项缺口` : ""}</span></summary><div class="ops-evidence-body">${evidenceRows}${evidenceData}${missingRows}</div></details>`;
  }

  function turnHtml(turn) {
    const messages = values(turn.messages);
    const user = messages.find((item) => item.message_type === "USER_MESSAGE");
    const assistant = messages.find((item) => item.message_type === "ASSISTANT_MESSAGE");
    const answerBlocks = values(turn.answer_blocks);
    const narrativeBlocks = answerBlocks.filter((block) => !["TABLE", "CHART", "EVIDENCE_REFERENCES"].includes(block.block_type));
    const blocks = narrativeBlocks.map(answerBlockHtml).join("");
    const evidence = turnEvidenceHtml(answerBlocks, turn.evidence_gaps);
    const plan = investigationPlanHtml(turn.investigation_plan);
    const answer = assistant || blocks || evidence ? `<article class="ops-message agent"><div class="ops-avatar">AI</div><div class="ops-message-body ops-result-markdown"><div class="ops-message-content">${blocks || markdown.render(assistant?.payload?.text || "")}</div>${evidence}</div></article>` : "";
    const settled = ["COMPLETED", "PARTIAL", "CANCELLED"].includes(turn.status);
    const progress = settled && !turn.error_message ? "" : `<div class="ops-context-banner ops-progress" data-turn-progress="${esc(turn.turn_id)}">${esc(turn.error_message || `当前状态：${turn.status}`)}</div>`;
    return `${user ? messageHtml("USER", user.payload?.text || "", shell.fmt(user.created_at)) : ""}${plan}${progress}${answer}`;
  }

  async function renderConversation(conversation, turns) {
    state.conversation = conversation;
    state.turns = turns;
    document.getElementById("conversation-title").textContent = conversation.title || "诊断对话";
    document.getElementById("conversation-context").textContent = conversation.source_type === "RUN" ? "这次对话继承自告警或巡检结果。" : "人工发起的智能诊断。";
    const panel = document.getElementById("message-list");
    panel.innerHTML = conversation.source_run_id ? '<div class="ops-context-banner">已关联来源诊断；后续回答只会引用当前 Turn 明确关联的证据。</div>' : "";
    turns.forEach((turn) => panel.insertAdjacentHTML("beforeend", turnHtml(turn)));
    await Promise.all(turns.filter((turn) => turn.status === "WAITING_USER" && turn.ops_run_id).map(async (turn) => {
      const progress = panel.querySelector(`[data-turn-progress="${String(turn.turn_id)}"]`);
      if (!progress) return;
      try {
        const run = await KBotAIOpsAuth.request(
          `${api}/runs/${encodeURIComponent(turn.ops_run_id)}`,
        );
        if (!["WAITING_INPUT", "WAITING_APPROVAL"].includes(run?.status)) {
          return;
        }
        const pending = await KBotAIOpsAuth.request(`${api}/runs/${encodeURIComponent(turn.ops_run_id)}/pending-input`);
        const html = diagnosticQueryApprovalHtml(pending);
        if (!html) return;
        progress.insertAdjacentHTML("beforebegin", html);
        bindDiagnosticQueryActions(progress.previousElementSibling);
      } catch (_) {
        // 已处理或并发恢复的审批不再展示，Turn事件流会给出最终状态。
      }
    }));
    panel.scrollTop = panel.scrollHeight;
    document.querySelectorAll("[data-copy-code]").forEach((button) => { button.onclick = () => markdown.copyCode(button); });
    resumeActiveTurns(conversation.conversation_id, turns);
  }

  async function loadConversation(id) {
    const conversation = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}`);
    const turnRows = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}/turns?limit=200`);
    const turns = await Promise.all(turnRows.map((turn) => KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}/turns/${encodeURIComponent(turn.turn_id)}`)));
    document.getElementById("agent-select").value = conversation.agent_id;
    document.getElementById("target-select").value = conversation.target_id;
    renderAgentOptions(conversation.agent_id);
    syncAgentTargetContext();
    await renderConversation(conversation, turns);
    history.replaceState(null, "", `./chat.html?conversation=${encodeURIComponent(id)}`);
    document.querySelectorAll(".ops-workspace-item").forEach((button) => { button.setAttribute("aria-current", String(button.dataset.id === id)); });
  }

  function resetConversationView({ agentSelected = false } = {}) {
    state.conversation = null;
    syncAgentTargetContext();
    document.getElementById("conversation-title").textContent = agentSelected
      ? "开始一次数据库诊断"
      : "请先选择 Target 和 Agent";
    document.getElementById("conversation-context").textContent = agentSelected
      ? "可以选择历史会话，或在下方发起一次新诊断。"
      : "选择 Target 和 Agent 后，才会显示对应的会话历史。";
    document.getElementById("message-list").innerHTML = agentSelected
      ? '<div class="ops-empty">请在下方描述需要诊断的问题。</div>'
      : '<div class="ops-empty">请选择 Target 和 Agent 以查看历史并开始诊断。</div>';
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
    const selectedTarget = document.getElementById("target-select").value;
    const list = document.getElementById("conversation-list");
    if (!selectedAgent || !selectedTarget) {
      list.innerHTML = '<div class="ops-empty">请先选择 Target 和 Agent 查看会话历史</div>';
      setComposerAvailability(false);
      resetConversationView();
      return;
    }
    setComposerAvailability(true);
    const rows = await KBotAIOpsAuth.request(`${api}/conversations?agent_id=${encodeURIComponent(selectedAgent)}&target_id=${encodeURIComponent(selectedTarget)}`);
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
    ensureProgressTimeline(progress);
    const elapsedTimer = window.setInterval(() => updateProgressElapsed(progress), 1000);
    let pending = null;
    let lastEventId = "";
    let completed = false;
    const path = `${api}/conversations/${encodeURIComponent(conversationId)}/turns/${encodeURIComponent(turnId)}/events`;
    const onEvent = ({ event, data, id }) => {
      if (id) lastEventId = id;
      const payload = data?.payload || {};
      if ([
        "turn.created", "planning.started", "planning.route.selected", "turn.status",
        "input.analysis.started", "input.analysis.completed",
        "task.frame.completed", "investigation.planned",
        "playbook.completed", "tool.started", "tool.completed",
        "tool.gap", "evidence.added", "assessment.started", "assessment.completed",
        "investigation.replanned", "diagnostic.query_approval_required",
        "diagnostic.query_approved", "diagnostic.query_rejected",
      ].includes(event)) {
        appendProgress(progress, event, payload, id);
      }
      if (["investigation.planned", "investigation.replanned"].includes(event)) {
        showInvestigationPlan(progress, payload.plan);
      }
      if (event === "diagnostic.query_approval_required" && payload.hitl_id) {
        showDiagnosticQueryApproval(progress, payload.hitl_id).catch((error) => shell.toast(error.message));
      }
      if (event === "thinking.delta") {
        appendProgress(progress, event, {
          public_summary: payload.public_summary || payload.delta || "正在组织回答",
        }, id);
      }
      if (event === "answer.delta") {
        const delta = String(data?.payload?.delta || "");
        if (!pending) {
          progress.insertAdjacentHTML("afterend", messageHtml("AGENT", ""));
          const message = progress.nextElementSibling;
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
        appendProgress(progress, event, { public_summary: "诊断已完成，正在整理结论…" }, id);
      }
    };
    for (let attempt = 0; attempt < streamRecoveryAttempts && !completed; attempt += 1) {
      let streamFailed = false;
      try {
        await KBotAIOpsAuth.stream(path, onEvent, {
          headers: lastEventId ? { "Last-Event-ID": lastEventId } : {},
        });
      } catch (_) {
        streamFailed = true;
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
      appendProgress(progress, "stream.recovery", {
        public_summary: streamFailed
          ? "事件流暂时中断，正在恢复诊断进度…"
          : "诊断仍在后台运行，正在继续获取进度…",
      }, "stream.recovery");
      await new Promise((resolve) => window.setTimeout(resolve, Math.min(5000, 1000 * (attempt + 1))));
    }
    window.clearInterval(elapsedTimer);
    updateProgressElapsed(progress);
    if (!completed) throw new Error("诊断仍在后台运行，请稍后刷新会话查看结果");
    if (pending) {
      pending.finalizing = true;
      await waitForTyping(pending);
    }
  }

  function resumeActiveTurns(conversationId, turns) {
    turns.filter((turn) => !terminalTurnStatuses.has(turn.status)).forEach((turn) => {
      const turnId = String(turn.turn_id);
      const followerKey = `${conversationId}:${turnId}`;
      const progress = document.querySelector(`[data-turn-progress="${turnId}"]`);
      if (!progress || activeTurnFollowers.has(followerKey)) return;
      activeTurnFollowers.add(followerKey);
      followTurn(conversationId, turnId, progress)
        .then(() => {
          if (String(state.conversation?.conversation_id) === String(conversationId)) {
            return loadConversation(conversationId);
          }
          return null;
        })
        .catch((error) => shell.toast(error.message))
        .finally(() => activeTurnFollowers.delete(followerKey));
    });
  }

  async function submitConversation(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button[type="submit"]');
    const agentId = document.getElementById("agent-select").value;
    const targetId = document.getElementById("target-select").value;
    const agent = state.agents.find((item) => String(item.agent_id) === String(agentId));
    const text = form.elements.message.value.trim();
    const selectedFile = state.selectedFile;
    if (!agentId) return shell.toast("请先选择 Agent");
    if (!targetId) return shell.toast("请先选择逻辑 Target");
    if (!values(agent?.target_ids).includes(targetId)) return shell.toast("当前 Agent 未绑定所选 Target");
    if (!text && !selectedFile) return shell.toast("请输入问题或上传诊断材料");
    button.disabled = true;
    try {
      const path = state.conversation
        ? `${api}/conversations/${state.conversation.conversation_id}/turns`
        : `${api}/conversations`;
      const content = text ? [{ content_type: "TEXT", text }] : [];
      if (selectedFile) {
        const uploaded = await KBotAIOpsAuth.request(`${api}/conversation-uploads`, {
          method: "POST",
          headers: {
            "Content-Type": uploadMediaType(selectedFile),
            "X-File-Name": encodeURIComponent(selectedFile.name),
          },
          body: selectedFile,
        });
        content.push({
          content_type: uploaded.media_type.startsWith("image/") ? "IMAGE" : "FILE",
          upload_id: uploaded.upload_id,
          media_type: uploaded.media_type,
        });
      }
      const body = state.conversation
        ? { content }
        : { agent_id: agentId, target_id: targetId, content };
      const receipt = await KBotAIOpsAuth.request(path, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify(body),
      });
      form.reset(); state.selectedFile = null; document.getElementById("upload-preview").textContent = "";
      const panel = document.getElementById("message-list");
      const submittedText = [
        text,
        selectedFile ? `已上传诊断材料：${selectedFile.name}` : "",
      ].filter(Boolean).join("\n\n");
      panel.insertAdjacentHTML("beforeend", messageHtml("USER", submittedText));
      panel.insertAdjacentHTML("beforeend", '<section id="live-progress" class="ops-context-banner ops-progress" aria-live="polite">正在建立诊断计划：先固定执行上下文，再理解问题并选择证据…</section>');
      const progress = document.getElementById("live-progress");
      await followTurn(receipt.conversation_id, receipt.turn_id, progress);
      await loadConversation(receipt.conversation_id);
      await loadConversationList();
    } catch (error) { shell.toast(error.message); } finally { button.disabled = false; }
  }

  async function initChat() {
    const select = document.getElementById("agent-select");
    await agents();
    const targetSelect = document.getElementById("target-select");
    targetSelect.innerHTML = '<option value="">选择逻辑 Target</option>' + state.targets.map((item) => `<option value="${esc(item.target_id)}">${esc(item.display_name)} · ${esc(item.db_type)}${item.readonly_connection_enabled ? "" : " · 仅监控"}</option>`).join("");
    targetSelect.onchange = () => {
      clearConversationUrl();
      state.conversation = null;
      renderAgentOptions();
      syncAgentTargetContext();
      resetConversationView();
      loadConversationList().catch((error) => shell.toast(error.message));
    };
    select.onchange = () => {
      clearConversationUrl();
      syncAgentTargetContext();
      resetConversationView({ agentSelected: Boolean(select.value) });
      loadConversationList().catch((error) => shell.toast(error.message));
    };
    document.getElementById("new-conversation").onclick = () => {
      if (!select.value) return shell.toast("请先选择 Agent");
      clearConversationUrl();
      resetConversationView({ agentSelected: true });
    };
    document.getElementById("conversation-form").onsubmit = submitConversation;
    document.getElementById("evidence-file").onchange = (event) => {
      const file = event.target.files[0] || null;
      if (file && file.size > 20 * 1024 * 1024) {
        event.target.value = "";
        state.selectedFile = null;
        return shell.toast("诊断材料不能超过 20 MiB");
      }
      state.selectedFile = file;
      document.getElementById("upload-preview").textContent = file ? `待上传：${file.name}` : "";
    };
    await loadConversationList();
  }

  function renderAgentOptions(preferredId = "") {
    const targetId = document.getElementById("target-select").value;
    const select = document.getElementById("agent-select");
    const rows = state.agents.filter((item) => values(item.target_ids).includes(targetId));
    select.disabled = !targetId;
    select.innerHTML = '<option value="">选择 Agent</option>' + rows.map((item) => `<option value="${esc(item.agent_id)}">${esc(item.display_name || item.name || item.agent_key || shell.short(item.agent_id))}</option>`).join("");
    if (rows.some((item) => String(item.agent_id) === String(preferredId))) select.value = preferredId;
  }

  function continueForm(source, title) {
    const allowed = state.agents.filter((item) => values(item.target_ids).includes(String(source.target_id)));
    const inherited = source.source_run_id ? "自动诊断结果与证据" : "告警情境及其监控信号";
    const prompt = source.source_run_id
      ? `请基于“${title}”的自动诊断结果继续分析：`
      : `请基于告警情境“${title}”开始诊断，并核验关联监控证据：`;
    return `<div class="ops-inline-dialog"><h3>继续深入诊断</h3><p>系统会在服务端继承本次${inherited}；后续人工对话才可能按 Agent 权限产生审批待办。</p><form id="continue-form" class="ops-form"><div class="ops-field span-12"><label>Agent</label><select name="agent_id" required><option value="">请选择</option>${allowed.map((item) => `<option value="${esc(item.agent_id)}">${esc(item.display_name || item.name || item.agent_key || shell.short(item.agent_id))}</option>`).join("")}</select></div><div class="ops-field span-12"><label>继续追问</label><textarea name="message" required>${esc(prompt)}</textarea></div><div class="ops-filter-actions"><button class="primary" type="submit">进入对话</button></div></form></div>`;
  }

  async function bindContinue(source) {
    const form = document.getElementById("continue-form");
    if (!form) return;
    form.onsubmit = async (event) => {
      event.preventDefault();
      const fields = Object.fromEntries(new FormData(form));
      const body = {
        agent_id: fields.agent_id,
        target_id: source.target_id,
        content: [{ content_type: "TEXT", text: fields.message }],
      };
      if (source.source_run_id) body.source_run_id = source.source_run_id;
      if (source.source_situation_id) body.source_situation_id = source.source_situation_id;
      try {
        const result = await KBotAIOpsAuth.request(`${api}/conversations`, { method: "POST", headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() }, body: JSON.stringify(body) });
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
    if (runId) {
      run = await KBotAIOpsAuth.request(`${api}/runs/${runId}`);
      try {
        result = await KBotAIOpsAuth.request(`${api}/runs/${runId}/result`);
      } catch (error) {
        if (error.status !== 404 && error.status !== 409) throw error;
      }
    }
    const source = result
      ? { target_id: run.target_id, source_run_id: run.ops_run_id }
      : { target_id: detail.target_id, source_situation_id: detail.situation_id };
    const diagnosis = result
      ? `<div class="ops-result-markdown">${markdown.render(conversationAnswerMarkdown(result))}${evidenceDetails(result)}</div>`
      : '<div class="ops-empty">自动诊断尚未生成结果，可选择 Agent 立即开始诊断。</div>';
    panel.innerHTML = `<div class="ops-context-banner">${shell.badge(detail.severity)} ${shell.badge(detail.status)} · ${detail.event_count} 个监控信号 · 最近观测 ${esc(shell.fmt(detail.last_observed_at))}</div>${situationEvidence(detail)}${diagnosis}${continueForm(source, detail.title)}`;
    await bindContinue(source);
  }

  async function showInspection(item) {
    const detail = await KBotAIOpsAuth.request(`${api}/inspection-fires/${encodeURIComponent(item.fire_id)}`);
    document.getElementById("case-title").textContent = `巡检 ${shell.fmt(detail.scheduled_at)}`;
    const panel = document.getElementById("case-detail");
    const runId = detail.run_ids[0];
    let run = null; let result = null;
    if (runId) { run = await KBotAIOpsAuth.request(`${api}/runs/${runId}`); result = await KBotAIOpsAuth.request(`${api}/runs/${runId}/result`); }
    const source = run ? { target_id: run.target_id, source_run_id: run.ops_run_id } : null;
    panel.innerHTML = `<div class="ops-context-banner">${shell.badge(detail.status)} · ${detail.completed_count}/${detail.target_count} 个目标完成 · ${detail.failed_count} 个失败</div>${result ? `<div class="ops-result-markdown">${markdown.render(inspectionMarkdown(result))}</div>${continueForm(source, "本次日常巡检")}` : '<div class="ops-empty">本次巡检尚未形成可展示结果。</div>'}`;
    if (source) await bindContinue(source);
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
