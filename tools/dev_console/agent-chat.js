(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    agents: [],
    conversations: [],
    activeConversation: null,
    activeTurnId: null,
    activeRunId: null,
    abortController: null,
    streamedAnswer: "",
    seenEvents: new Set(),
  };

  KBotUI.bindAuthForm($("#auth-form"), () => {
    KBotUI.setStatus($("#session-status"), "连接信息已保存", "ok");
  });

  function selectedAgent() {
    const id = $("#agent-select").value;
    return state.agents.find((item) => item.agent_id === id) || null;
  }

  function selectedConversation() {
    const id = $("#conversation-select").value;
    return state.conversations.find(
      (item) => item.conversation_id === id
    ) || null;
  }

  function renderSession(payload) {
    $("#session-output").textContent = KBotUI.json(payload);
  }

  function renderAgents() {
    $("#agent-select").innerHTML = state.agents
      .map(
        (item) =>
          `<option value="${item.agent_id}">${KBotUI.escapeHtml(
            item.display_name
          )} · ${KBotUI.escapeHtml(item.agent_key)}</option>`
      )
      .join("");
    const agent = selectedAgent();
    $("#rerank-select").value = String(Boolean(agent?.do_rerank));
  }

  function renderConversations() {
    $("#conversation-select").innerHTML = state.conversations
      .map(
        (item) =>
          `<option value="${item.conversation_id}">${KBotUI.escapeHtml(
            item.title || "未命名会话"
          )} · v${item.row_version}</option>`
      )
      .join("");
    state.activeConversation = selectedConversation();
  }

  async function refreshAgents() {
    try {
      state.agents = (await KBotUI.api("/api/v1/agents")) || [];
      renderAgents();
      renderSession(state.agents);
      KBotUI.setStatus(
        $("#session-status"),
        `已读取 ${state.agents.length} 个 Agent`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus($("#session-status"), error.message, "error");
    }
  }

  async function refreshConversations(preferredId) {
    try {
      state.conversations =
        (await KBotUI.api("/api/v1/conversations?limit=100")) || [];
      renderConversations();
      if (preferredId) $("#conversation-select").value = preferredId;
      state.activeConversation = selectedConversation();
      renderSession(state.activeConversation || state.conversations);
      KBotUI.setStatus(
        $("#session-status"),
        `已读取 ${state.conversations.length} 个会话`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus($("#session-status"), error.message, "error");
    }
  }

  $("#refresh-agents").addEventListener("click", refreshAgents);
  $("#refresh-conversations").addEventListener("click", () =>
    refreshConversations()
  );
  $("#agent-select").addEventListener("change", () => {
    const agent = selectedAgent();
    $("#rerank-select").value = String(Boolean(agent?.do_rerank));
    renderSession(agent);
  });
  $("#conversation-select").addEventListener("change", () => {
    state.activeConversation = selectedConversation();
    renderSession(state.activeConversation);
  });

  $("#save-rerank").addEventListener("click", async () => {
    const agent = selectedAgent();
    if (!agent) return;
    try {
      const updated = await KBotUI.api(`/api/v1/agents/${agent.agent_id}`, {
        method: "PATCH",
        body: JSON.stringify({
          expected_row_version: agent.row_version,
          do_rerank: $("#rerank-select").value === "true",
        }),
      });
      state.agents = state.agents.map((item) =>
        item.agent_id === updated.agent_id ? updated : item
      );
      renderAgents();
      renderSession(updated);
      KBotUI.setStatus($("#session-status"), "重排开关已保存", "ok");
    } catch (error) {
      KBotUI.setStatus($("#session-status"), error.message, "error");
    }
  });

  $("#conversation-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const agent = selectedAgent();
    if (!agent) {
      KBotUI.setStatus($("#session-status"), "请先选择 Agent", "error");
      return;
    }
    const title = event.currentTarget.elements.title.value.trim();
    try {
      const payload = await KBotUI.api("/api/v1/conversations", {
        method: "POST",
        body: JSON.stringify({
          agent_id: agent.agent_id,
          title: title || null,
          retention_policy: "DEFAULT",
        }),
      });
      await refreshConversations(payload.conversation_id);
      KBotUI.setStatus($("#session-status"), "会话创建成功", "ok");
    } catch (error) {
      KBotUI.setStatus($("#session-status"), error.message, "error");
    }
  });

  function eventPayload(event) {
    const envelope =
      event.json && typeof event.json === "object" ? event.json : {};
    return {
      envelope,
      payload:
        envelope.payload && typeof envelope.payload === "object"
          ? envelope.payload
          : {},
    };
  }

  function appendLiveEntry(kind, title, body, meta, detail) {
    const stream = $("#live-stream");
    stream.querySelector(".muted")?.remove();
    const node = document.createElement("article");
    node.className = `stream-entry ${kind}`;
    node.innerHTML = `
      <div class="stream-entry-head">
        <strong>${KBotUI.escapeHtml(title)}</strong>
        <span>${KBotUI.escapeHtml(meta || "")}</span>
      </div>
      <div class="stream-entry-body">${KBotUI.escapeHtml(body || "")}</div>
      ${
        detail
          ? `<pre>${KBotUI.escapeHtml(
              typeof detail === "string" ? detail : KBotUI.json(detail)
            )}</pre>`
          : ""
      }`;
    stream.appendChild(node);
    stream.scrollTop = stream.scrollHeight;
    return node;
  }

  function resetLiveStream(question) {
    state.seenEvents = new Set();
    $("#live-stream").innerHTML = "";
    appendLiveEntry("user", "用户", question, "问题");
  }

  function appendThinking(payload, event) {
    const text =
      payload.delta ||
      payload.public_summary ||
      payload.title ||
      "Agent 正在处理";
    appendLiveEntry(
      "thinking",
      "思考",
      text,
      `#${event.id} · 可公开过程`
    );
  }

  function updateLiveAnswer(delta, event) {
    let node = $("#live-answer");
    if (!node) {
      node = appendLiveEntry(
        "answer",
        "Agent",
        "",
        `#${event.id} · 流式回答`
      );
      node.id = "live-answer";
    }
    node.querySelector(".stream-entry-body").textContent =
      state.streamedAnswer || delta || "正在生成回答…";
    $("#live-stream").scrollTop = $("#live-stream").scrollHeight;
  }

  function readableEvent(event, payload) {
    const task = payload.task_key || payload.skill_id || "";
    const mappings = {
      RUN_CREATED: ["status", "运行", "已接收问题并创建 Agent Run"],
      RUN_STARTED: ["thinking", "规划", "已生成本轮执行计划"],
      TASK_READY: ["skill", "任务", `${task || "下一执行步骤"}已就绪`],
      TASK_STARTED: ["skill", "任务", `开始执行 ${task || "任务"}`],
      TASK_COMPLETED: ["skill", "任务", `${task || "任务"}执行完成`],
      TASK_RETRYING: ["warning", "重试", `${task || "任务"}准备重试`],
      TASK_FAILED: ["error", "任务失败", payload.error_code || task],
      "memory.context_loaded": [
        "memory",
        "记忆",
        payload.public_summary || "已加载会话上下文",
      ],
      "skill.started": [
        "skill",
        "Skill",
        payload.public_summary || `正在执行 ${task || "Skill"}`,
      ],
      "query.rewritten": [
        "thinking",
        "上下文理解",
        payload.public_summary ||
          `已改写为：${payload.standalone_query || ""}`,
      ],
      "retrieval.completed": [
        "retrieval",
        "知识检索",
        payload.public_summary ||
          `发现 ${payload.candidate_count || 0} 个候选，形成 ${
            payload.citation_count || 0
          } 组证据`,
      ],
      "data.query.completed": [
        "retrieval",
        "问数结果",
        payload.public_summary || `查询返回 ${payload.row_count || 0} 行`,
      ],
      "chart.completed": [
        "retrieval",
        "图表",
        payload.public_summary || "已生成 ECharts 配置",
      ],
      "answer.completed": [
        "status",
        "回答完成",
        `最终回答已生成，引用 ${payload.reference_count || 0} 个文档`,
      ],
      RUN_COMPLETED: ["status", "运行完成", "Agent Run 已成功完成"],
      RUN_FAILED: [
        "error",
        "运行失败",
        payload.error_message || payload.error_code || "Agent Run 执行失败",
      ],
      RUN_CANCELLED: ["warning", "运行取消", "Agent Run 已取消"],
    };
    return mappings[event.type] || null;
  }

  function renderReadableEvent(event, payload) {
    if (event.type === "thinking.delta") {
      appendThinking(payload, event);
      return;
    }
    if (event.type === "answer.delta") {
      updateLiveAnswer(String(payload.delta || ""), event);
      return;
    }
    if (event.type.startsWith("trace.")) {
      const envelope = event.json || {};
      appendLiveEntry(
        envelope.stage === "thinking" ? "thinking" : "status",
        envelope.title || envelope.stage || "Trace",
        envelope.summary || envelope.public_summary || "",
        `#${event.id} · 历史 Trace`
      );
      return;
    }
    if (event.type === "done") return;
    const readable = readableEvent(event, payload);
    if (!readable) return;
    const [kind, title, body] = readable;
    const detail =
      event.type === "retrieval.completed"
        ? {
            diagnostics: payload.diagnostics || {},
            rerank: payload.rerank || {},
            warnings: payload.warnings || [],
          }
        : null;
    appendLiveEntry(kind, title, body, `#${event.id}`, detail);
  }

  function addEvent(event) {
    const eventKey = `${event.type}:${event.id}`;
    if (state.seenEvents.has(eventKey)) return;
    state.seenEvents.add(eventKey);
    const { envelope, payload } = eventPayload(event);
    if (event.type === "answer.delta") {
      const normalizedDelta = String(payload.delta || "");
      if (normalizedDelta) {
        state.streamedAnswer += normalizedDelta;
        renderStreamingAnswer();
      }
    }
    renderReadableEvent(event, payload);
    const node = document.createElement("article");
    node.className = "event";
    const title =
      payload.public_summary ||
      payload.title ||
      envelope.event_type ||
      event.type;
    node.innerHTML = `
      <div class="event-head">
        <strong>${KBotUI.escapeHtml(event.type)}</strong>
        <span class="muted">#${KBotUI.escapeHtml(event.id)}</span>
      </div>
      <div>${KBotUI.escapeHtml(title || "")}</div>
      <pre>${KBotUI.escapeHtml(KBotUI.json(envelope))}</pre>`;
    $("#timeline").appendChild(node);
    $("#timeline").scrollTop = $("#timeline").scrollHeight;
  }

  function contentText(content) {
    if (content == null) return "";
    if (typeof content === "string") return content;
    return (
      content.answer ||
      content.text ||
      content.input ||
      content.content ||
      KBotUI.json(content)
    );
  }

  function renderTurns(payload) {
    const turns = payload?.turns || [];
    const messages = [];
    for (const turn of turns) {
      for (const item of [turn.user_item, turn.assistant_item]) {
        if (!item) continue;
        const normalizedRole = String(item.role || "").toLowerCase();
        messages.push(`
          <article class="message ${KBotUI.escapeHtml(normalizedRole)}">
            <div class="role">${KBotUI.escapeHtml(item.role)} · #${
              item.item_sequence
            }</div>
            <div>${KBotUI.escapeHtml(contentText(item.content))}</div>
          </article>`);
      }
    }
    $("#messages").innerHTML =
      messages.join("") || '<p class="muted">当前没有历史消息。</p>';
    $("#messages").scrollTop = $("#messages").scrollHeight;
  }

  function renderStreamingAnswer() {
    let node = $("#streaming-answer");
    if (!node) {
      node = document.createElement("article");
      node.id = "streaming-answer";
      node.className = "message assistant streaming";
      node.innerHTML = `
        <div class="role">ASSISTANT · 正在生成</div>
        <div class="content"></div>`;
      $("#messages").appendChild(node);
    }
    node.querySelector(".content").textContent =
      state.streamedAnswer || "正在等待回答内容…";
    $("#messages").scrollTop = $("#messages").scrollHeight;
  }

  async function loadHistory(options) {
    const conversation = selectedConversation();
    if (!conversation) return;
    const payload = await KBotUI.api(
      `/api/v1/conversations/${conversation.conversation_id}/turns?after=0&limit=200`
    );
    renderTurns(payload);
    if (state.activeTurnId && options?.includeTrace !== false) {
      const trace = await KBotUI.api(
        `/api/v1/conversations/${conversation.conversation_id}/turns/${state.activeTurnId}/trace?after=0&limit=500`
      );
      trace.forEach((item) =>
        addEvent({
          id: item.sequence_no,
          type: `trace.${item.stage}`,
          json: item,
        })
      );
    }
  }

  $("#load-history").addEventListener("click", async () => {
    try {
      await loadHistory({ includeTrace: true });
      KBotUI.setStatus($("#turn-status"), "历史记录已加载", "ok");
    } catch (error) {
      KBotUI.setStatus($("#turn-status"), error.message, "error");
    }
  });

  async function createTurn(form, conversation) {
    const files = Array.from(form.elements.images.files || []);
    const collectionIds = form.elements.collectionIds.value
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    const common = {
      input: form.elements.input.value.trim(),
      expected_conversation_version: conversation.row_version,
      collection_ids: collectionIds,
      security_level: Number(form.elements.securityLevel.value || 3),
      client_metadata: { source: "agent-chat-test-ui" },
    };
    const headers = {
      "Idempotency-Key": KBotUI.idempotency("ui-turn"),
    };
    if (!files.length) {
      return KBotUI.api(
        `/api/v1/conversations/${conversation.conversation_id}/turns`,
        { method: "POST", headers, body: JSON.stringify(common) }
      );
    }
    if (files.length > 8) throw new Error("查询图片最多 8 张");
    const data = new FormData();
    data.append("input", common.input);
    data.append(
      "expected_conversation_version",
      String(common.expected_conversation_version)
    );
    data.append("collection_ids_json", JSON.stringify(collectionIds));
    data.append("security_level", String(common.security_level));
    data.append(
      "client_metadata_json",
      JSON.stringify(common.client_metadata)
    );
    files.forEach((file) => data.append("images", file, file.name));
    return KBotUI.api(
      `/api/v1/conversations/${conversation.conversation_id}/turns/multipart`,
      { method: "POST", headers, body: data }
    );
  }

  async function loadResult(runId) {
    try {
      const payload = await KBotUI.api(`/api/v1/runs/${runId}/result`);
      $("#result-output").textContent = KBotUI.json(payload);
      renderResultInLiveStream(payload);
      return payload;
    } catch (error) {
      $("#result-output").textContent = KBotUI.json(
        error.payload || { error: error.message }
      );
      return null;
    }
  }

  function renderResultInLiveStream(result) {
    const payload =
      result?.payload && typeof result.payload === "object"
        ? result.payload
        : {};
    const answer = String(payload.answer || "");
    if (answer && !state.streamedAnswer) {
      state.streamedAnswer = answer;
      updateLiveAnswer(answer, { id: "result" });
    }
    const references = Array.isArray(payload.references)
      ? payload.references
      : [];
    if (!references.length || $("#live-references")) return;
    const node = appendLiveEntry(
      "references",
      "引用文档",
      references
        .map(
          (item) =>
            `${item.citation_label || "-"} · ${item.title || "未命名文档"}`
        )
        .join("\n"),
      `${references.length} 个实际引用`
    );
    node.id = "live-references";
  }

  $("#turn-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const conversation = selectedConversation();
    if (!conversation) {
      KBotUI.setStatus($("#turn-status"), "请先创建或选择会话", "error");
      return;
    }
    const button = $("#send-turn");
    button.disabled = true;
    state.streamedAnswer = "";
    resetLiveStream(event.currentTarget.elements.input.value.trim());
    $("#timeline").innerHTML = "";
    $("#result-output").textContent = "{}";
    KBotUI.setStatus($("#turn-status"), "正在创建 Turn…");
    try {
      const receipt = await createTurn(event.currentTarget, conversation);
      state.activeTurnId = receipt.turn_id;
      state.activeRunId = receipt.run_id;
      renderSession(receipt);
      await refreshConversations(conversation.conversation_id);
      await loadHistory({ includeTrace: false });
      if (!receipt.run_id || !receipt.events_url) {
        KBotUI.setStatus(
          $("#turn-status"),
          `Turn 状态：${receipt.turn_status}`,
          "ok"
        );
        await loadHistory({ includeTrace: false });
        return;
      }
      state.abortController?.abort();
      state.abortController = new AbortController();
      KBotUI.setStatus($("#turn-status"), "SSE 已连接，等待执行完成…");
      await KBotUI.streamSse(
        receipt.events_url,
        {
          lastEventId: receipt.event_cursor,
          onEvent: addEvent,
        },
        state.abortController.signal
      );
      const result = await loadResult(receipt.run_id);
      if (!state.streamedAnswer && result?.payload?.answer) {
        state.streamedAnswer = String(result.payload.answer);
        renderStreamingAnswer();
      }
      await refreshConversations(conversation.conversation_id);
      await loadHistory({ includeTrace: false });
      KBotUI.setStatus($("#turn-status"), "Run 已结束", "ok");
    } catch (error) {
      if (error.name === "AbortError") {
        KBotUI.setStatus($("#turn-status"), "SSE 监听已停止");
      } else {
        KBotUI.setStatus($("#turn-status"), error.message, "error");
      }
    } finally {
      button.disabled = false;
    }
  });

  $("#stop-stream").addEventListener("click", () => {
    state.abortController?.abort();
  });
})();
