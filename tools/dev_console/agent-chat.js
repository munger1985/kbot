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

  function addEvent(event) {
    const payload = event.json || {};
    const node = document.createElement("article");
    node.className = "event";
    const title =
      payload.payload?.public_summary ||
      payload.payload?.title ||
      payload.event_type ||
      event.type;
    node.innerHTML = `
      <div class="event-head">
        <strong>${KBotUI.escapeHtml(event.type)}</strong>
        <span class="muted">#${KBotUI.escapeHtml(event.id)}</span>
      </div>
      <div>${KBotUI.escapeHtml(title || "")}</div>
      <pre>${KBotUI.escapeHtml(KBotUI.json(payload))}</pre>`;
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
        messages.push(`
          <article class="message ${KBotUI.escapeHtml(item.role)}">
            <div class="role">${KBotUI.escapeHtml(item.role)} · #${
              item.item_sequence
            }</div>
            <div>${KBotUI.escapeHtml(contentText(item.content))}</div>
          </article>`);
      }
    }
    $("#messages").innerHTML =
      messages.join("") || '<p class="muted">当前没有历史消息。</p>';
  }

  async function loadHistory() {
    const conversation = selectedConversation();
    if (!conversation) return;
    const payload = await KBotUI.api(
      `/api/v1/conversations/${conversation.conversation_id}/turns?after=0&limit=200`
    );
    renderTurns(payload);
    if (state.activeTurnId) {
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
      await loadHistory();
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
      security_level: Number(form.elements.securityLevel.value || 0),
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
      return payload;
    } catch (error) {
      $("#result-output").textContent = KBotUI.json(
        error.payload || { error: error.message }
      );
      return null;
    }
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
    $("#timeline").innerHTML = "";
    $("#result-output").textContent = "{}";
    KBotUI.setStatus($("#turn-status"), "正在创建 Turn…");
    try {
      const receipt = await createTurn(event.currentTarget, conversation);
      state.activeTurnId = receipt.turn_id;
      state.activeRunId = receipt.run_id;
      renderSession(receipt);
      await refreshConversations(conversation.conversation_id);
      if (!receipt.run_id || !receipt.events_url) {
        KBotUI.setStatus(
          $("#turn-status"),
          `Turn 状态：${receipt.turn_status}`,
          "ok"
        );
        await loadHistory();
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
      await loadResult(receipt.run_id);
      await refreshConversations(conversation.conversation_id);
      await loadHistory();
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
