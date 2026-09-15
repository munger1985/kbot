/* X 实时搜索：未就绪时不创建 Run，就绪后轮询公开阶段与来源。 */
(function () {
  "use strict";

  const { badge, capability, capabilityLabel, escapeHtml, toast } = KBotAssistantShell;
  const TERMINAL = new Set(["COMPLETED", "FAILED", "REJECTED"]);
  let currentRunId = "";
  let pollTimer = 0;
  let accessState = null;

  function ready() {
    return Boolean(capability(accessState, "x_search").ready);
  }

  function splitHandles(value) {
    return String(value || "").split(/[,，]/).map((item) => item.trim()).filter(Boolean);
  }

  function paintCapability() {
    const node = document.getElementById("x-search-capability");
    if (!node) return;
    const item = capability(accessState, "x_search");
    const [label, tone] = capabilityLabel(item);
    node.className = `assistant-badge ${tone}`;
    node.textContent = label;
    const submit = document.getElementById("x-search-submit");
    if (submit) submit.disabled = !item.ready;
  }

  function empty(title, copy) {
    return `<div class="assistant-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div>`;
  }

  function renderHistory(rows) {
    const node = document.getElementById("x-search-history");
    if (!node) return;
    if (!rows.length) {
      node.innerHTML = empty("尚无搜索记录", "此处只显示当前用户按权限可见的 X Search Run。");
      return;
    }
    node.innerHTML = rows.map((row) => `
      <button class="assistant-history-item" type="button" data-run-id="${escapeHtml(row.run_id)}" ${row.run_id === currentRunId ? 'aria-current="true"' : ""}>
        <strong>${escapeHtml((row.request && row.request.input) || "X Search")}</strong>
        <small>${escapeHtml(row.status)} · ${escapeHtml(row.created_at || "")}</small>
      </button>`).join("");
    node.querySelectorAll("[data-run-id]").forEach((button) => {
      button.addEventListener("click", () => openRun(button.dataset.runId));
    });
  }

  function renderEvents(run, events) {
    const node = document.getElementById("x-search-result");
    const stages = (events || []).map((item) => `<li><strong>${escapeHtml(item.stage)}</strong><span>${escapeHtml(item.message || "")}</span></li>`).join("");
    const answer = run.result && run.result.answer ? `<p>${escapeHtml(run.result.answer)}</p>` : "";
    node.innerHTML = `
      <h3>${escapeHtml(run.status)}</h3>
      <p>${escapeHtml(run.error_message || run.model_display_name || "正在执行 X Search")}</p>
      ${answer}
      <ol class="assistant-flow">${stages || "<li><strong>ACCEPTED</strong><span>已接受请求</span></li>"}</ol>
      ${badge(run.status, TERMINAL.has(run.status) ? (run.status === "COMPLETED" ? "good" : "bad") : "warn")}`;
  }

  function renderSources(rows) {
    const node = document.getElementById("x-search-sources");
    if (!rows.length) {
      node.innerHTML = empty("尚无 [X] 引用", "每张来源卡将显示账号、可访问链接、发布时间（若提供）与检索时间，并标注为外部实时线索。");
      return;
    }
    node.innerHTML = rows.map((row) => `
      <article class="assistant-source-card">
        <strong>${escapeHtml(row.citation_label || "[X]")} ${escapeHtml(row.author_handle || "")}</strong>
        <p>${escapeHtml(row.title || row.excerpt || "")}</p>
        <small>${escapeHtml(row.published_at || "发布时间未提供")} · 检索于 ${escapeHtml(row.retrieved_at || "")}</small>
        <div><a href="${escapeHtml(row.canonical_url)}" target="_blank" rel="noreferrer">${escapeHtml(row.canonical_url)}</a></div>
        <span class="assistant-badge">外部实时线索</span>
      </article>`).join("");
  }

  function stopPoll() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = 0;
    }
  }

  async function refreshHistory() {
    const rows = await KBotAssistantApi.json("/x-search/runs", "GET");
    renderHistory(Array.isArray(rows) ? rows : []);
  }

  async function loadRun(runId) {
    const [run, events, sources] = await Promise.all([
      KBotAssistantApi.json(`/x-search/runs/${runId}`, "GET"),
      KBotAssistantApi.json(`/x-search/runs/${runId}/events`, "GET"),
      KBotAssistantApi.json(`/x-search/runs/${runId}/sources`, "GET"),
    ]);
    renderEvents(run, Array.isArray(events) ? events : []);
    renderSources(Array.isArray(sources) ? sources : []);
    return run;
  }

  function schedulePoll(runId) {
    stopPoll();
    pollTimer = setTimeout(async () => {
      try {
        const run = await loadRun(runId);
        if (!TERMINAL.has(run.status) && currentRunId === runId) schedulePoll(runId);
        else await refreshHistory();
      } catch (error) {
        toast(error.message || "无法刷新搜索进度", "error");
      }
    }, 1500);
  }

  async function openRun(runId) {
    currentRunId = runId;
    try {
      const run = await loadRun(runId);
      await refreshHistory();
      if (!TERMINAL.has(run.status)) schedulePoll(runId);
    } catch (error) {
      toast(error.message || "无法打开搜索 Run", "error");
    }
  }

  function resetComposer() {
    stopPoll();
    currentRunId = "";
    document.getElementById("x-search-form")?.reset();
    document.getElementById("x-search-result").innerHTML = `<h3>尚未创建搜索 Run</h3><p>真实运行时，此区域将按结构化事件显示“请求已接受、正在执行 X Search、整理来源、生成结论、已完成”。不会显示隐藏推理过程。</p>`;
    document.getElementById("x-search-sources").innerHTML = empty("尚无 [X] 引用", "每张来源卡将显示账号、可访问链接、发布时间（若提供）与检索时间，并标注为外部实时线索。");
    refreshHistory().catch((error) => toast(error.message || "无法刷新搜索历史", "error"));
  }

  KBotAssistantShell.ready.then(async (access) => {
    if (!access) return;
    accessState = access;
    paintCapability();
    document.getElementById("x-search-new")?.addEventListener("click", resetComposer);
    document.getElementById("x-search-refresh")?.addEventListener("click", () => {
      refreshHistory().catch((error) => toast(error.message || "无法刷新搜索历史", "error"));
    });
    document.getElementById("x-search-form")?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const from = document.getElementById("x-search-from").value;
      const to = document.getElementById("x-search-to").value;
      if (from && to && from > to) {
        toast("起始日期不能晚于结束日期。", "error");
        return;
      }
      if (!ready()) {
        toast("模型能力尚未验收，未创建 X Search Run。", "error");
        return;
      }
      const mode = document.getElementById("x-search-handle-mode").value;
      const handleList = splitHandles(document.getElementById("x-search-handles").value);
      const payload = { input: document.getElementById("x-search-query").value.trim() };
      if (from) payload.from_date = from;
      if (to) payload.to_date = to;
      if (mode === "allowed" && handleList.length) payload.allowed_x_handles = handleList;
      if (mode === "excluded" && handleList.length) payload.excluded_x_handles = handleList;
      if (document.getElementById("x-search-image").checked) payload.enable_image_understanding = true;
      if (document.getElementById("x-search-video").checked) payload.enable_video_understanding = true;
      try {
        const run = await KBotAssistantApi.json("/x-search/runs", "POST", payload, {
          headers: { "Idempotency-Key": KBotAssistantApi.requestId() },
        });
        currentRunId = run.run_id;
        await openRun(run.run_id);
      } catch (error) {
        toast(error.message || "创建 X Search Run 失败", "error");
      }
    });
    try {
      await refreshHistory();
    } catch (error) {
      toast(error.message || "无法加载搜索历史", "error");
    }
  });
})();
