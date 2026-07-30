(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const STORAGE_KEY = "kbot.ui.aiops.chat.v1";
  const state = {
    agents: [],
    targets: [],
    activeRunId: "",
    activeRun: null,
    activeHitl: null,
    pendingHitlId: "",
    activeProposal: null,
    cursor: 0,
    streamController: null,
    seenEvents: new Set(),
  };

  function optionalValue(value) {
    const normalized = String(value || "").trim();
    return normalized || null;
  }

  function option(value, label, selected) {
    return `<option value="${KBotUI.escapeHtml(value)}"${
      selected ? " selected" : ""
    }>${KBotUI.escapeHtml(label)}</option>`;
  }

  function loadLocalState() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    } catch (_) {
      return {};
    }
  }

  function persistLocalState() {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        agentId: $("#agent-select").value,
        targetId: $("#target-select").value,
        sessionId: $("#session-id").value,
        activeRunId: state.activeRunId,
        cursor: state.cursor,
      })
    );
  }

  function createSessionId() {
    return `ops-session-${KBotUI.uuid()}`;
  }

  function ensureSessionId() {
    if (!$("#session-id").value.trim()) {
      $("#session-id").value = createSessionId();
    }
    return $("#session-id").value.trim();
  }

  function setConversationState(label, kind) {
    const badge = $("#conversation-state");
    badge.textContent = label;
    badge.className = `badge${kind ? ` ${kind}` : ""}`;
  }

  function appendMessage(role, body, meta, allowHtml) {
    const stream = $("#conversation-stream");
    const article = document.createElement("article");
    article.className = `ops-message ${role}`;
    const author =
      role === "user"
        ? "用户"
        : role === "system"
          ? "系统"
          : "AIOps Agent";
    article.innerHTML = `
      <div class="ops-message-author">${author}${
        meta
          ? `<span>${KBotUI.escapeHtml(meta)}</span>`
          : ""
      }</div>
      <div class="ops-message-body">${
        allowHtml ? body : KBotUI.escapeHtml(body)
      }</div>`;
    stream.appendChild(article);
    stream.scrollTop = stream.scrollHeight;
    return article;
  }

  function appendTrace(title, body, kind, meta) {
    const stream = $("#trace-stream");
    if (stream.querySelector(":scope > .muted")) stream.innerHTML = "";
    const article = document.createElement("article");
    article.className = `ops-trace-item ${kind || ""}`;
    article.innerHTML = `
      <div class="ops-trace-dot"></div>
      <div>
        <div class="ops-trace-title">
          <strong>${KBotUI.escapeHtml(title)}</strong>
          <span>${KBotUI.escapeHtml(meta || "")}</span>
        </div>
        <p>${KBotUI.escapeHtml(body || "")}</p>
      </div>`;
    stream.appendChild(article);
    stream.scrollTop = stream.scrollHeight;
  }

  function appendRawEvent(event) {
    const row = document.createElement("pre");
    row.textContent = `${event.id || "-"} · ${event.type}\n${KBotUI.json(
      event.json
    )}`;
    $("#raw-events").appendChild(row);
  }

  function listHtml(values) {
    const items = (Array.isArray(values) ? values : [])
      .filter(Boolean)
      .map((item) => `<li>${KBotUI.escapeHtml(String(item))}</li>`)
      .join("");
    return items ? `<ul>${items}</ul>` : "<p>无</p>";
  }

  async function refreshContext() {
    KBotUI.setStatus($("#context-status"), "正在读取 Agent 和数据库目标…");
    try {
      const local = loadLocalState();
      const [agents, targetPage] = await Promise.all([
        KBotUI.api("/api/v1/agents"),
        KBotUI.api("/api/v1/ops/targets?limit=200"),
      ]);
      state.agents = agents.filter((item) =>
        (item.enabled_capabilities || []).includes("aiops")
      );
      state.targets = targetPage.items || [];
      const agentId = $("#agent-select").value || local.agentId;
      const targetId = $("#target-select").value || local.targetId;
      $("#agent-select").innerHTML = state.agents
        .map((item) =>
          option(
            item.agent_id,
            `${item.display_name} · ${item.status}`,
            item.agent_id === agentId
          )
        )
        .join("");
      $("#target-select").innerHTML = state.targets
        .map((item) =>
          option(
            item.target_id,
            `${item.display_name} · ${item.db_type} · ${item.status}`,
            item.target_id === targetId
          )
        )
        .join("");
      renderContextSummary();
      KBotUI.setStatus(
        $("#context-status"),
        `可用 Agent ${state.agents.length} 个，目标 ${state.targets.length} 个`,
        state.agents.length && state.targets.length ? "ok" : ""
      );
      persistLocalState();
    } catch (error) {
      KBotUI.setStatus($("#context-status"), error.message, "error");
    }
  }

  function renderContextSummary() {
    const agent = state.agents.find(
      (item) => item.agent_id === $("#agent-select").value
    );
    const target = state.targets.find(
      (item) => item.target_id === $("#target-select").value
    );
    $("#context-summary").innerHTML = `
      <dl class="ops-run-facts">
        <div><dt>诊断模型</dt><dd>${
          agent?.models?.diagnosis_llm ? "已配置" : "缺失"
        }</dd></div>
        <div><dt>数据库</dt><dd>${KBotUI.escapeHtml(
          target ? `${target.db_type} ${target.version_code || ""}` : "—"
        )}</dd></div>
        <div><dt>环境</dt><dd>${KBotUI.escapeHtml(
          target?.environment || "—"
        )}</dd></div>
        <div><dt>直连诊断</dt><dd>${
          target?.diagnostic_secret_configured ? "已配置" : "未配置 / 可能补数"
        }</dd></div>
      </dl>`;
  }

  function taskDescription(payload) {
    const key = String(payload.task_key || "");
    if (key === "scope") return "冻结目标、时间窗、权限和诊断预算";
    if (key.startsWith("observe:")) return "从监控平台采集并规范化指标";
    if (key.startsWith("diagnostic:db."))
      return `执行只读数据库工具 ${key.replace("diagnostic:", "")}`;
    if (key === "diagnosis:knowledge") return "检索已绑定的运维知识";
    if (key.includes(":draft")) return "生成可证伪的根因假设与补证计划";
    if (key.includes(":validate")) return "校验取证工具、参数和安全边界";
    if (key.includes(":collect")) return "执行补证计划并汇总新证据";
    if (key.includes(":assess")) return "评估证据、反证和下一步";
    if (key.includes("root-cause")) return "确定根因等级";
    if (key.includes("grounding")) return "验证结论与证据引用";
    if (key.includes("solution")) return "生成缓解、改进和验证方案";
    if (key.includes("proposal")) return "生成受控操作提案";
    if (key.includes("report")) return "生成结构化诊断报告";
    return payload.task_type || "执行诊断任务";
  }

  function normalizeQueries(request) {
    const queries = request?.queries || request?.sql_requests || [];
    if (queries.length) return queries;
    return (request?.fields || []).map((field, index) => ({
      query_id: field.field_id || field.name || `field-${index + 1}`,
      purpose: field.label || field.name || "补充诊断数据",
      diagnostic_question: field.description || "",
      sql_text: "",
      expected_columns: [],
      cost_warning: "",
    }));
  }

  async function showHitl(hitlId) {
    state.pendingHitlId = hitlId;
    try {
      const payload = await KBotUI.api(`/api/v1/ops/hitl/${hitlId}`);
      state.activeHitl = payload;
      state.pendingHitlId = "";
      const request = payload.request || {};
      const queries = normalizeQueries(request);
      $("#interaction-dock").hidden = false;
      $("#hitl-card").hidden = false;
      $("#hitl-expiry").textContent = `截止 ${new Date(
        payload.expires_at
      ).toLocaleString()}`;
      $("#hitl-prompt").textContent =
        request.instructions?.join("；") ||
        "请只在目标数据库使用只读账号执行，并完整保留列名。";
      $("#hitl-queries").innerHTML = queries
        .map(
          (query, index) => `
            <section class="ops-query" data-query-id="${KBotUI.escapeHtml(
              query.query_id
            )}">
              <div class="ops-query-head">
                <div>
                  <span>查询 ${index + 1}</span>
                  <strong>${KBotUI.escapeHtml(
                    query.purpose || query.diagnostic_question || query.query_id
                  )}</strong>
                </div>
                <button class="copy-query" type="button">复制 SQL</button>
              </div>
              ${
                query.sql_text
                  ? `<pre class="ops-sql">${KBotUI.escapeHtml(
                      query.sql_text
                    )}</pre>`
                  : `<p class="hint">${KBotUI.escapeHtml(
                      query.diagnostic_question || "请提供该字段的数据"
                    )}</p>`
              }
              ${
                query.expected_columns?.length
                  ? `<p class="hint">期望列：${KBotUI.escapeHtml(
                      query.expected_columns.join(", ")
                    )}</p>`
                  : ""
              }
              ${
                query.cost_warning
                  ? `<p class="ops-warning">${KBotUI.escapeHtml(
                      query.cost_warning
                    )}</p>`
                  : ""
              }
              <div class="field-row">
                <div class="field">
                  <label>执行状态</label>
                  <select class="query-status">
                    <option>SUCCEEDED</option>
                    <option>FAILED</option>
                    <option>SKIPPED</option>
                  </select>
                </div>
                <div class="field">
                  <label>结果或错误信息</label>
                  <textarea class="query-result" placeholder="直接粘贴数据库客户端的完整原始输出，系统会自动识别 SQL*Plus、JSON 或分隔文本"></textarea>
                </div>
              </div>
            </section>`
        )
        .join("");
      $("#hitl-queries")
        .querySelectorAll(".copy-query")
        .forEach((button) =>
          button.addEventListener("click", async () => {
            const sql =
              button.closest(".ops-query").querySelector(".ops-sql")
                ?.textContent || "";
            await navigator.clipboard.writeText(sql);
            button.textContent = "已复制";
          })
        );
      setConversationState("等待人工补证", "warning");
      appendMessage(
        "assistant",
        "现有证据不足，需要你执行页面中的只读查询并回贴结果。收到数据后会继续同一个诊断 Run。",
        "需要操作"
      );
      persistLocalState();
    } catch (error) {
      $("#interaction-dock").hidden = false;
      $("#hitl-card").hidden = false;
      $("#hitl-expiry").textContent = "详情加载失败";
      $("#hitl-prompt").textContent =
        "补证请求已经创建，但页面暂时无法读取 SQL 详情。";
      $("#hitl-queries").innerHTML = `
        <section class="ops-query">
          <p class="ops-warning">${KBotUI.escapeHtml(error.message)}</p>
          <button id="retry-hitl" type="button">重新加载补证内容</button>
        </section>`;
      $("#retry-hitl").addEventListener("click", () => showHitl(hitlId));
      KBotUI.setStatus($("#hitl-status"), error.message, "error");
      appendMessage(
        "system",
        `人工补证详情加载失败：${error.message}`,
        "可点击重新加载"
      );
    }
  }

  async function showProposal(proposalId, advisory) {
    try {
      const proposal = await KBotUI.api(
        `/api/v1/ops/proposals/${proposalId}`
      );
      state.activeProposal = proposal;
      $("#interaction-dock").hidden = false;
      $("#proposal-card").hidden = false;
      $("#proposal-mode").textContent = proposal.mode;
      $("#proposal-title").textContent = advisory
        ? "建议命令已生成"
        : "命令等待单次审批";
      $("#proposal-command").textContent = proposal.command_preview;
      $("#proposal-impact").textContent = proposal.impact;
      $("#proposal-risk").textContent = proposal.risk;
      const prerequisites = proposal.prerequisites || [];
      $("#proposal-prerequisites").innerHTML = prerequisites.length
        ? prerequisites
            .map((item) => `<li>${KBotUI.escapeHtml(String(item))}</li>`)
            .join("")
        : "<li>无额外前置条件</li>";
      $("#proposal-verification").textContent = proposal.verification_plan;
      $("#proposal-rollback").textContent = proposal.rollback_plan;
      $("#approve-proposal").disabled = advisory;
      $("#reject-proposal").disabled = advisory;
      $("#manual-result-toggle").hidden = !advisory;
      setConversationState(
        advisory ? "等待人工处理" : "等待命令审批",
        "warning"
      );
      appendMessage(
        "assistant",
        advisory
          ? "已生成建议命令。当前资源不具备自动变更能力；你可以人工处理后回填结果。"
          : "已生成一条受控变更命令。批准前不会执行；本次审批只对当前命令和参数有效。",
        "需要操作"
      );
    } catch (error) {
      KBotUI.setStatus($("#proposal-status"), error.message, "error");
    }
  }

  function eventPayload(event) {
    const record = event.json || {};
    return { record, payload: record.payload || record };
  }

  function handleEvent(event) {
    const eventKey = `${event.id}:${event.type}`;
    if (state.seenEvents.has(eventKey)) return;
    state.seenEvents.add(eventKey);
    appendRawEvent(event);
    state.cursor = Math.max(state.cursor, Number(event.id || 0));
    $("#active-cursor").textContent = String(state.cursor);
    persistLocalState();
    const { record, payload } = eventPayload(event);
    if (event.type === "task.status") {
      const status = payload.status || "UNKNOWN";
      appendTrace(
        taskDescription(payload),
        status === "RUNNING"
          ? "Worker 已领取，正在执行"
          : `任务状态：${status}`,
        status === "FAILED" ? "failed" : status === "SUCCEEDED" ? "done" : "",
        payload.task_key
      );
    } else if (event.type === "run.status") {
      $("#active-run-status").textContent = payload.status;
      setConversationState(payload.status);
      appendTrace("Run 阶段变化", payload.status, "phase", record.occurred_at);
    } else if (event.type === "diagnostic.input_required") {
      appendTrace(
        "需要人工补证",
        "数据库不可直连或当前证据不足",
        "attention",
        payload.hitl_id
      );
      showHitl(payload.hitl_id);
    } else if (
      event.type === "diagnostic.input_received" ||
      event.type === "diagnostic.input_skipped"
    ) {
      appendTrace(
        "人工补证状态",
        event.type.endsWith("received") ? "已接收，继续诊断" : "已跳过",
        "done",
        payload.hitl_id
      );
    } else if (event.type === "proposal.pending_approval") {
      appendTrace(
        "命令等待审批",
        "执行边界已冻结，等待用户显式批准",
        "attention",
        payload.proposal_id
      );
      showProposal(payload.proposal_id, false);
    } else if (event.type === "proposal.advisory_ready") {
      appendTrace(
        "建议命令已生成",
        "当前资源不具备自动变更能力",
        "done",
        payload.proposal_id
      );
      showProposal(payload.proposal_id, true);
    } else if (event.type.startsWith("proposal.")) {
      appendTrace("提案状态", event.type, "phase", payload.proposal_id);
    } else if (event.type === "execution.status") {
      appendTrace(
        "命令执行状态",
        payload.status,
        payload.status === "FAILED" ? "failed" : "phase",
        payload.execution_id
      );
      appendMessage(
        "system",
        `受控命令执行状态更新为 ${payload.status}`,
        payload.execution_id
      );
    } else if (event.type === "comparison.plan.created") {
      appendTrace(
        "已安排效果验证",
        "将使用同口径指标生成处理前后对比",
        "phase",
        payload.comparison_plan_id
      );
    } else if (event.type === "report.ready") {
      appendTrace(
        "报告已生成",
        `${payload.report_type} · ${payload.summary || ""}`,
        "done",
        payload.report_id
      );
    } else if (event.type.startsWith("run.")) {
      const terminal = [
        "run.completed",
        "run.failed",
        "run.cancelled",
        "run.expired",
      ].includes(event.type);
      $("#active-run-status").textContent = payload.status || event.type;
      if (terminal) {
        if (event.type === "run.completed") {
          loadRunResult();
        } else {
          setConversationState(payload.status || event.type, "warning");
          appendMessage(
            "system",
            `Run 已结束：${payload.status || event.type}`,
            state.activeRunId
          );
        }
        refreshRunSummary();
      }
    }
  }

  function resultSection(title, values) {
    const items = [
      ...new Set(
        (Array.isArray(values) ? values : [])
          .filter(Boolean)
          .map((item) => String(item))
      ),
    ];
    if (!items.length) return "";
    return `<section><strong>${KBotUI.escapeHtml(title)}</strong>${listHtml(
      items
    )}</section>`;
  }

  function renderFinalAnswer(result) {
    const payload = result.payload || {};
    const direct = payload.direct_answer;
    const root = payload.root_cause || {};
    const solution = payload.solution || {};
    const formalReport = payload.output_kind === "DIAGNOSIS_REPORT";
    const recommendationLevel = payload.recommendation_level || "BRIEF";
    const grade = result.root_cause_grade || root.effective_level || "未定级";
    const details = payload.hypothesis_details || [];
    const primary = details.find(
      (item) => item.hypothesis_key === root.primary_hypothesis_key
    );
    const title = direct
      ? direct.answer_text
      : grade === "INCONCLUSIVE"
        ? "诊断流程已完成，但证据不足，暂未确认根因。"
        : primary?.statement ||
          root.conclusion ||
          `已形成 ${grade} 等级的诊断结论。`;
    const facts = (payload.facts || [])
      .slice(0, 12)
      .map((item) => item.fact_summary || item.summary);
    if (direct) {
      const directRefs = new Set(direct.fact_refs || []);
      const directFacts = (payload.facts || [])
        .filter((item) => directRefs.has(item.fact_id))
        .map((item) => item.fact_summary || item.summary)
        .filter(Boolean);
      const body = `
        <div class="ops-answer-state answered">
          <span>${KBotUI.escapeHtml(direct.status)}</span>
          <strong>${KBotUI.escapeHtml(direct.answer_text)}</strong>
        </div>
        ${resultSection("监控依据", directFacts)}
        ${resultSection("口径限制", direct.limitations || [])}`;
      appendMessage("assistant", body, "直接回答", true);
      $("#active-root-grade").textContent = "不适用";
      setConversationState(
        direct.status === "ANSWERED" ? "已回答" : "已部分回答",
        direct.status === "PARTIAL" ? "warning" : ""
      );
      return;
    }
    const body = `
      <div class="ops-answer-state ${grade.toLowerCase()}">
        <span>${KBotUI.escapeHtml(grade)}</span>
        <strong>${KBotUI.escapeHtml(title)}</strong>
      </div>
      ${
        payload.diagnosis_rationale
          ? `<p>${KBotUI.escapeHtml(payload.diagnosis_rationale)}</p>`
          : ""
      }
      ${resultSection("关键证据", facts)}
      ${
        recommendationLevel === "NONE"
          ? ""
          : resultSection("立即缓解措施", solution.immediate_mitigations)
      }
      ${
        recommendationLevel === "FULL"
          ? resultSection("长期改进", solution.long_term_remediations)
          : ""
      }
      ${
        recommendationLevel === "FULL"
          ? resultSection("验证方法", solution.verification_plan)
          : ""
      }
      ${resultSection("尚缺证据", [
        ...(root.unresolved_gaps || []),
        ...(solution.limitations || []),
        ...(payload.gaps || []),
      ])}`;
    appendMessage(
      "assistant",
      body,
      formalReport ? "正式诊断报告" : "诊断结论",
      true
    );
    $("#active-root-grade").textContent = grade;
    setConversationState(
      grade === "INCONCLUSIVE"
        ? "已结束 · 未确诊"
        : `已结束 · ${grade}`,
      grade === "INCONCLUSIVE" ? "warning" : ""
    );
  }

  async function loadRunResult() {
    if (!state.activeRunId) return;
    try {
      const result = await KBotUI.api(
        `/api/v1/ops/runs/${state.activeRunId}/result`
      );
      $("#result-output").textContent = KBotUI.json(result);
      renderFinalAnswer(result);
    } catch (error) {
      appendMessage(
        "system",
        `Run 已结束，但最终结果读取失败：${error.message}`,
        state.activeRunId
      );
    }
  }

  async function refreshRunSummary() {
    if (!state.activeRunId) return null;
    const summary = await KBotUI.api(
      `/api/v1/ops/runs/${state.activeRunId}`
    );
    state.activeRun = summary;
    $("#run-output").textContent = KBotUI.json(summary);
    $("#active-run-label").textContent = summary.ops_run_id;
    $("#active-run-status").textContent = summary.status;
    $("#active-root-grade").textContent =
      summary.root_cause_grade || "—";
    persistLocalState();
    return summary;
  }

  async function listenRun(cursor) {
    if (!state.activeRunId) return;
    state.streamController?.abort();
    state.streamController = new AbortController();
    setConversationState("正在监听");
    try {
      await KBotUI.streamSse(
        `/api/v1/ops/runs/${state.activeRunId}/events`,
        {
          lastEventId: cursor || 0,
          onEvent: handleEvent,
        },
        state.streamController.signal
      );
    } catch (error) {
      if (error.name !== "AbortError") {
        appendMessage("system", `SSE 连接中断：${error.message}`, "可重新监听");
        setConversationState("监听中断", "warning");
      }
    }
  }

  async function resumeRun(runId) {
    state.activeRunId = runId;
    state.cursor = 0;
    state.seenEvents.clear();
    $("#active-run-label").textContent = runId;
    $("#active-cursor").textContent = "0";
    persistLocalState();
    try {
      const summary = await refreshRunSummary();
      if (summary.status === "WAITING_INPUT") {
        const pending = await KBotUI.api(
          `/api/v1/ops/runs/${runId}/pending-input`
        );
        await showHitl(pending.hitl_id);
      }
      listenRun(0);
      if (["COMPLETED", "DEGRADED"].includes(summary.status)) {
        await loadRunResult();
      }
    } catch (error) {
      appendMessage("system", `恢复 Run 失败：${error.message}`, runId);
    }
  }

  $("#question-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = event.currentTarget.elements.input.value.trim();
    const agentId = $("#agent-select").value;
    const targetId = $("#target-select").value;
    if (!agentId || !targetId) {
      appendMessage("system", "请先选择 AIOps Agent 和数据库目标。");
      return;
    }
    appendMessage("user", input, new Date().toLocaleTimeString());
    event.currentTarget.elements.input.value = "";
    hideInteractionCards();
    setConversationState("正在创建 Run");
    try {
      const receipt = await KBotUI.api("/api/v1/ops/runs", {
        method: "POST",
        headers: {
          "Idempotency-Key": KBotUI.idempotency("aiops-chat-run"),
        },
        body: JSON.stringify({
          agent_id: agentId,
          target_id: targetId,
          input,
          session_id: ensureSessionId(),
          observation_start: null,
          observation_end: null,
          client_metadata: {
            source: "dev_console_aiops_chat",
            interaction_mode: "FULL",
          },
        }),
      });
      state.activeRunId = receipt.ops_run_id;
      state.activeRun = receipt;
      state.cursor = Number(receipt.event_cursor || 0);
      state.seenEvents.clear();
      $("#active-run-label").textContent = receipt.ops_run_id;
      $("#active-run-status").textContent = receipt.status;
      $("#active-cursor").textContent = String(state.cursor);
      $("#run-output").textContent = KBotUI.json(receipt);
      appendMessage(
        "assistant",
        "诊断 Run 已创建。我会先采集监控和可用数据库证据，再决定是否需要你补充数据。",
        receipt.ops_run_id
      );
      persistLocalState();
      listenRun(receipt.event_cursor);
    } catch (error) {
      appendMessage("system", `创建诊断 Run 失败：${error.message}`);
      setConversationState("创建失败", "warning");
    }
  });

  function hideInteractionCards() {
    $("#interaction-dock").hidden = true;
    $("#hitl-card").hidden = true;
    $("#proposal-card").hidden = true;
    $("#manual-result-form").hidden = true;
    state.activeHitl = null;
    state.activeProposal = null;
  }

  $("#submit-hitl").addEventListener("click", async () => {
    if (!state.activeHitl) return;
    const rows = [...$("#hitl-queries").querySelectorAll(".ops-query")];
    const responses = rows.map((row) => {
      const status = row.querySelector(".query-status").value;
      const content = row.querySelector(".query-result").value;
      return {
        query_id: row.dataset.queryId,
        status,
        raw_output: status === "SUCCEEDED" ? content : null,
        error: status === "FAILED" ? content : null,
      };
    });
    if (
      responses.some(
        (item) =>
          (item.status === "SUCCEEDED" && !item.raw_output?.trim()) ||
          (item.status === "FAILED" && !item.error?.trim())
      )
    ) {
      KBotUI.setStatus(
        $("#hitl-status"),
        "成功查询必须粘贴结果，失败查询必须填写错误信息",
        "error"
      );
      return;
    }
    KBotUI.setStatus($("#hitl-status"), "正在提交人工证据…");
    try {
      const result = await KBotUI.api(
        `/api/v1/ops/hitl/${state.activeHitl.hitl_id}/response`,
        {
          method: "POST",
          headers: {
            "Idempotency-Key": KBotUI.idempotency("aiops-hitl-response"),
          },
          body: JSON.stringify({
            expected_row_version: state.activeHitl.row_version,
            responses,
            note: optionalValue($("#hitl-note").value),
          }),
        }
      );
      $("#hitl-card").hidden = true;
      $("#interaction-dock").hidden = $("#proposal-card").hidden;
      appendMessage(
        "user",
        `已提交 ${responses.length} 项数据库查询结果，请继续分析。`,
        result.hitl_id
      );
      appendTrace("人工证据已提交", "原 Run 将从挂起点继续", "done");
      setConversationState("继续诊断");
      state.activeHitl = null;
    } catch (error) {
      KBotUI.setStatus($("#hitl-status"), error.message, "error");
    }
  });

  $("#skip-hitl").addEventListener("click", async () => {
    if (!state.activeHitl) return;
    try {
      await KBotUI.api(
        `/api/v1/ops/hitl/${state.activeHitl.hitl_id}/skip`,
        {
          method: "POST",
          headers: {
            "If-Match": `"rv-${state.activeHitl.row_version}"`,
            "Idempotency-Key": KBotUI.idempotency("aiops-hitl-skip"),
          },
        }
      );
      $("#hitl-card").hidden = true;
      $("#interaction-dock").hidden = $("#proposal-card").hidden;
      appendMessage(
        "user",
        "跳过本次人工补证，请根据已有证据继续。",
        state.activeHitl.hitl_id
      );
      state.activeHitl = null;
      setConversationState("继续诊断");
    } catch (error) {
      KBotUI.setStatus($("#hitl-status"), error.message, "error");
    }
  });

  $("#approve-proposal").addEventListener("click", async () => {
    const proposal = state.activeProposal;
    if (!proposal) return;
    KBotUI.setStatus($("#proposal-status"), "正在批准当前命令…");
    try {
      const receipt = await KBotUI.api(
        `/api/v1/ops/proposals/${proposal.proposal_id}/approve`,
        {
          method: "POST",
          headers: {
            "Idempotency-Key": KBotUI.idempotency("aiops-proposal-approve"),
          },
          body: JSON.stringify({
            expected_row_version: proposal.row_version,
            expected_proposal_hash: proposal.proposal_hash,
            note: optionalValue($("#proposal-note").value),
          }),
        }
      );
      $("#proposal-card").hidden = true;
      $("#interaction-dock").hidden = $("#hitl-card").hidden;
      appendMessage(
        "user",
        "已批准当前这一条命令，等待执行和验证结果。",
        receipt.execution_id
      );
      appendTrace(
        "命令已获单次批准",
        `授权截止 ${new Date(
          receipt.authorization_expires_at
        ).toLocaleString()}`,
        "attention",
        receipt.execution_id
      );
      setConversationState("执行与验证中");
    } catch (error) {
      KBotUI.setStatus($("#proposal-status"), error.message, "error");
    }
  });

  $("#reject-proposal").addEventListener("click", async () => {
    const proposal = state.activeProposal;
    if (!proposal) return;
    const reason = $("#proposal-note").value.trim();
    if (!reason) {
      KBotUI.setStatus($("#proposal-status"), "驳回必须填写原因", "error");
      return;
    }
    try {
      await KBotUI.api(
        `/api/v1/ops/proposals/${proposal.proposal_id}/reject`,
        {
          method: "POST",
          headers: {
            "Idempotency-Key": KBotUI.idempotency("aiops-proposal-reject"),
          },
          body: JSON.stringify({
            expected_row_version: proposal.row_version,
            reason,
          }),
        }
      );
      $("#proposal-card").hidden = true;
      $("#interaction-dock").hidden = $("#hitl-card").hidden;
      appendMessage("user", `驳回命令：${reason}`, proposal.proposal_id);
      setConversationState("提案已驳回", "warning");
    } catch (error) {
      KBotUI.setStatus($("#proposal-status"), error.message, "error");
    }
  });

  $("#manual-result-toggle").addEventListener("click", () => {
    $("#manual-result-form").hidden = !$("#manual-result-form").hidden;
  });

  $("#submit-manual-result").addEventListener("click", async () => {
    const proposal = state.activeProposal;
    if (!proposal) return;
    try {
      const result = await KBotUI.api(
        `/api/v1/ops/proposals/${proposal.proposal_id}/manual-result`,
        {
          method: "POST",
          headers: {
            "Idempotency-Key": KBotUI.idempotency("aiops-manual-result"),
          },
          body: JSON.stringify({
            expected_row_version: proposal.row_version,
            status: $("#manual-result-status").value,
            occurred_at: new Date().toISOString(),
            bounded_output: optionalValue($("#manual-result-output").value),
            note: optionalValue($("#proposal-note").value),
          }),
        }
      );
      $("#proposal-card").hidden = true;
      $("#interaction-dock").hidden = $("#hitl-card").hidden;
      appendMessage(
        "user",
        `已回填人工执行结果：${result.status}`,
        proposal.proposal_id
      );
      appendTrace(
        "人工处理结果已登记",
        "系统将创建验证 Run 并生成处理前后对比",
        "done",
        result.result_artifact?.artifact_id
      );
    } catch (error) {
      KBotUI.setStatus($("#proposal-status"), error.message, "error");
    }
  });

  $("#refresh-context").addEventListener("click", refreshContext);
  $("#agent-select").addEventListener("change", () => {
    renderContextSummary();
    persistLocalState();
  });
  $("#target-select").addEventListener("change", () => {
    renderContextSummary();
    persistLocalState();
  });
  $("#new-session").addEventListener("click", () => {
    $("#session-id").value = createSessionId();
    state.activeRunId = "";
    state.activeRun = null;
    state.cursor = 0;
    state.seenEvents.clear();
    state.streamController?.abort();
    $("#active-run-label").textContent = "未开始";
    $("#active-run-status").textContent = "IDLE";
    $("#active-root-grade").textContent = "—";
    $("#active-cursor").textContent = "0";
    hideInteractionCards();
    appendMessage("system", "已开始新的诊断会话。");
    setConversationState("等待提问");
    persistLocalState();
  });
  $("#resume-run").addEventListener("click", () => {
    const runId = $("#resume-run-id").value.trim();
    if (runId) resumeRun(runId);
  });
  $("#reconnect-stream").addEventListener("click", () => {
    if (state.activeRunId) listenRun(state.cursor);
  });
  $("#cancel-run").addEventListener("click", async () => {
    try {
      const summary = await refreshRunSummary();
      if (!summary) return;
      await KBotUI.api(`/api/v1/ops/runs/${state.activeRunId}/cancel`, {
        method: "POST",
        headers: {
          "If-Match": `"rv-${summary.row_version}"`,
          "Idempotency-Key": KBotUI.idempotency("aiops-run-cancel"),
        },
      });
      appendMessage("user", "取消当前诊断 Run。", state.activeRunId);
    } catch (error) {
      appendMessage("system", `取消失败：${error.message}`);
    }
  });
  $("#clear-trace").addEventListener("click", () => {
    $("#trace-stream").innerHTML =
      '<p class="muted">轨迹已清空；不会影响正在运行的诊断。</p>';
    $("#raw-events").innerHTML = "";
  });

  KBotUI.bindAuthForm($("#auth-form"), refreshContext);
  const local = loadLocalState();
  $("#session-id").value = local.sessionId || createSessionId();
  $("#resume-run-id").value = local.activeRunId || "";
  refreshContext().then(() => {
    if (local.activeRunId) {
      resumeRun(local.activeRunId);
    }
  });
})();
