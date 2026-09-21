/* X 实时搜索：未就绪时不创建 Run，就绪后把结论整理成综述、帖子与来源卡。 */
(function () {
  "use strict";

  const { badge, escapeHtml, toast } = KBotKnowledgeShell;
  const TERMINAL = new Set(["COMPLETED", "FAILED", "REJECTED"]);
  const STAGE_LABEL = {
    ACCEPTED: "已接受请求",
    SEARCHING: "正在执行 X Search",
    ORGANIZING_SOURCES: "正在整理来源",
    COMPOSING: "正在生成结论",
    COMPLETED: "已完成",
    FAILED: "失败",
    REJECTED: "已拒绝",
  };
  let currentRunId = "";
  let pollTimer = 0;
  let agents = [];
  let currentSources = [];

  function ready() {
    return Boolean(document.getElementById("x-search-agent")?.value);
  }

  function splitHandles(value) {
    return String(value || "").split(/[,，]/).map((item) => item.trim()).filter(Boolean);
  }

  function paintCapability() {
    const node = document.getElementById("x-search-capability");
    if (!node) return;
    const available = agents.length > 0;
    node.className = `knowledge-badge ${available ? "good" : "warn"}`;
    node.textContent = available ? "Grok Agent 已就绪" : "没有绑定 Grok 的 ACTIVE Agent";
    const submit = document.getElementById("x-search-submit");
    if (submit) submit.disabled = !ready();
  }

  async function loadAgents() {
    const rows = await KBotKnowledgeApi.json("/agents", "GET");
    agents = (Array.isArray(rows) ? rows : []).filter(
      (row) => row.status === "ACTIVE" && row.models?.x_search_llm,
    );
    const select = document.getElementById("x-search-agent");
    select.innerHTML = '<option value="">请选择绑定 Grok 的 Agent</option>' + agents.map(
      (row) => `<option value="${escapeHtml(row.agent_id)}">${escapeHtml(row.display_name || row.agent_id)}</option>`,
    ).join("");
    select.addEventListener("change", paintCapability);
    paintCapability();
  }

  function empty(title, copy) {
    return `<div class="knowledge-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div>`;
  }

  function stageLabel(value) {
    const key = String(value || "").trim();
    return STAGE_LABEL[key] || key || "进行中";
  }

  function displayHandle(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    return raw.startsWith("@") ? raw : `@${raw}`;
  }

  function displayTime(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    const iso = raw.match(/^(\d{4}-\d{2}-\d{2})/);
    return iso ? iso[1] : raw;
  }

  function normalizeUrl(value) {
    return String(value || "").trim().replace(/\/+$/, "");
  }

  function sourceByUrl(sources) {
    const map = new Map();
    (sources || []).forEach((row) => {
      const url = normalizeUrl(row.canonical_url);
      if (url) map.set(url, row);
    });
    return map;
  }

  function citationChip(label, url, asLink) {
    const text = escapeHtml(label);
    const citation = escapeHtml(label);
    if (asLink && url) {
      return `<a class="x-cite" data-citation="${citation}" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${text}</a>`;
    }
    return `<button type="button" class="x-cite" data-citation="${citation}">${text}</button>`;
  }

  function citationNode(label, url, asLink) {
    const template = document.createElement("template");
    template.innerHTML = citationChip(label, url, asLink);
    return template.content.firstElementChild;
  }

  function normalizeCitationMarkdown(text) {
    return String(text || "").replace(
      /\[\[(\d+)\]\]\((https?:[^)\s]+)\)/g,
      (_match, index, url) => `[X${index}](${url})`,
    );
  }

  function enhanceCitationLinks(root, urlMap) {
    root.querySelectorAll("a[href]").forEach((anchor) => {
      const url = normalizeUrl(anchor.getAttribute("href"));
      const row = urlMap.get(url);
      if (!row) {
        anchor.classList.add("x-inline-link");
        return;
      }
      anchor.replaceWith(citationNode(row.citation_label || "[X]", "", false));
    });
  }

  function enhanceCitationText(root) {
    const pattern = /\[\[(\d+)\]\]|\[(X\d+)\]/g;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (node.parentElement?.closest("code, pre, a, button")) return;
      const text = node.textContent || "";
      pattern.lastIndex = 0;
      if (!pattern.test(text)) return;
      pattern.lastIndex = 0;
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      for (const match of text.matchAll(pattern)) {
        fragment.append(document.createTextNode(text.slice(cursor, match.index)));
        const label = match[1] ? `[X${match[1]}]` : `[${match[2]}]`;
        fragment.append(citationNode(label, "", false));
        cursor = match.index + match[0].length;
      }
      fragment.append(document.createTextNode(text.slice(cursor)));
      node.replaceWith(fragment);
    });
  }

  function renderAnswer(text, sources) {
    const urlMap = sourceByUrl(sources);
    const template = document.createElement("template");
    template.innerHTML = KBotMarkdown.render(normalizeCitationMarkdown(text));
    enhanceCitationLinks(template.content, urlMap);
    enhanceCitationText(template.content);
    return `<section class="x-answer">${template.innerHTML}</section>`;
  }

  function bindCitations(root) {
    root.querySelectorAll("button.x-cite").forEach((button) => {
      button.addEventListener("click", () => highlightCitation(button.dataset.citation));
    });
  }

  function highlightCitation(label) {
    const selected = String(label || "");
    document.querySelectorAll("[data-citation]").forEach((node) => {
      node.classList.toggle("is-active", node.getAttribute("data-citation") === selected);
    });
    const card = document.querySelector(`.knowledge-source-card[data-citation="${CSS.escape(selected)}"]`);
    if (card) card.scrollIntoView({ block: "nearest" });
  }

  function renderHistory(rows) {
    const node = document.getElementById("x-search-history");
    if (!node) return;
    if (!rows.length) {
      node.innerHTML = empty("尚无搜索记录", "此处只显示当前用户按权限可见的 X Search Run。");
      return;
    }
    node.innerHTML = rows.map((row) => `
      <article class="knowledge-history-item" data-run-id="${escapeHtml(row.run_id)}" ${row.run_id === currentRunId ? 'aria-current="true"' : ""}>
        <button class="knowledge-history-open" type="button" data-run-id="${escapeHtml(row.run_id)}">
          <strong>${escapeHtml((row.request && row.request.input) || "X Search")}</strong>
          <small>${escapeHtml(stageLabel(row.status))} · ${escapeHtml(displayTime(row.created_at) || "")}</small>
        </button>
        <button class="small danger knowledge-history-delete" type="button" data-run-id="${escapeHtml(row.run_id)}">删除</button>
      </article>`).join("");
    node.querySelectorAll(".knowledge-history-open").forEach((button) => {
      button.addEventListener("click", () => openRun(button.dataset.runId));
    });
    node.querySelectorAll(".knowledge-history-delete").forEach((button) => {
      button.addEventListener("click", () => deleteRun(button.dataset.runId));
    });
  }

  function renderEvents(run, events, sources) {
    const node = document.getElementById("x-search-result");
    const terminal = TERMINAL.has(run.status);
    const answer = run.result && run.result.answer;
    const sourceCount = (sources || []).length;
    const tone = run.status === "COMPLETED" ? "good" : (run.status === "FAILED" || run.status === "REJECTED" ? "bad" : "warn");
    const query = run.request && run.request.input ? String(run.request.input) : "";
    let body = "";
    if (run.status === "FAILED" || run.status === "REJECTED") {
      body = `<div class="knowledge-notice warn">${escapeHtml(run.error_message || "搜索失败")}</div>`;
    } else if (answer) {
      body = renderAnswer(answer, sources);
    } else {
      const stages = (events || []).map((item) => `<li><strong>${escapeHtml(stageLabel(item.stage))}</strong><span>${escapeHtml(item.message || "")}</span></li>`).join("");
      body = `<ol class="knowledge-flow">${stages || "<li><strong>已接受请求</strong><span>正在等待公开阶段</span></li>"}</ol>`;
    }
    node.className = answer ? "x-search-result" : "knowledge-run-placeholder";
    node.innerHTML = `
      <header class="x-result-head">
        <div>
          <h3>${escapeHtml(stageLabel(run.status))}</h3>
          <p>${escapeHtml(run.error_message && terminal && run.status !== "COMPLETED" ? run.error_message : (run.model_display_name || "X Search"))}${sourceCount ? ` · ${sourceCount} 条来源` : ""}</p>
        </div>
        ${badge("外部实时线索", tone === "good" ? "" : tone)}
      </header>
      ${query ? `<p class="x-query-recap">${escapeHtml(query)}</p>` : ""}
      ${body}`;
    node.querySelectorAll("[data-copy-code]").forEach((button) => {
      button.addEventListener("click", () => KBotMarkdown.copyCode(button));
    });
    bindCitations(node);
  }

  function renderSourceCard(row) {
    const handle = displayHandle(row.author_handle);
    const published = displayTime(row.published_at);
    const retrieved = displayTime(row.retrieved_at);
    const summary = row.excerpt || row.title || "";
    return `<article class="knowledge-source-card" data-citation="${escapeHtml(row.citation_label || "")}">
      <header>
        <span class="x-cite is-label">${escapeHtml(row.citation_label || "[X]")}</span>
        <strong>${escapeHtml(handle || "账号未提供")}</strong>
      </header>
      ${summary ? `<p>${escapeHtml(summary)}</p>` : ""}
      <footer>
        <small>${escapeHtml(published || "发布时间未提供")}${retrieved ? ` · 检索于 ${escapeHtml(retrieved)}` : ""}</small>
        <a href="${escapeHtml(row.canonical_url)}" target="_blank" rel="noreferrer">打开原文</a>
      </footer>
    </article>`;
  }

  function renderSources(rows) {
    const node = document.getElementById("x-search-sources");
    if (!rows.length) {
      node.innerHTML = empty("尚无 [X] 引用", "每张来源卡将显示账号、可访问链接、发布时间（若提供）与检索时间，并标注为外部实时线索。");
      return;
    }
    const grouped = new Map();
    const order = [];
    rows.forEach((row) => {
      const key = displayTime(row.published_at) || "时间未提供";
      if (!grouped.has(key)) {
        grouped.set(key, []);
        order.push(key);
      }
      grouped.get(key).push(row);
    });
    order.sort((left, right) => {
      if (left === "时间未提供") return 1;
      if (right === "时间未提供") return -1;
      return right.localeCompare(left);
    });
    node.innerHTML = order.map((key) => `
      <section class="x-source-day">
        <h3>${escapeHtml(key)}</h3>
        ${grouped.get(key).map(renderSourceCard).join("")}
      </section>`).join("");
    node.querySelectorAll(".knowledge-source-card").forEach((card) => {
      card.addEventListener("click", (event) => {
        if (event.target.closest("a")) return;
        highlightCitation(card.getAttribute("data-citation"));
      });
    });
  }

  function stopPoll() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = 0;
    }
  }

  async function refreshHistory() {
    const rows = await KBotKnowledgeApi.json("/x-search/runs", "GET");
    renderHistory(Array.isArray(rows) ? rows : []);
  }

  async function loadRun(runId) {
    const [run, events, sources] = await Promise.all([
      KBotKnowledgeApi.json(`/x-search/runs/${runId}`, "GET"),
      KBotKnowledgeApi.json(`/x-search/runs/${runId}/events`, "GET"),
      KBotKnowledgeApi.json(`/x-search/runs/${runId}/sources`, "GET"),
    ]);
    currentSources = Array.isArray(sources) ? sources : [];
    renderEvents(run, Array.isArray(events) ? events : [], currentSources);
    renderSources(currentSources);
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

  async function deleteRun(runId) {
    const confirmed = globalThis.confirm("删除后无法恢复。确认删除这条搜索记录？");
    if (!confirmed) return;
    try {
      await KBotKnowledgeApi.json(`/x-search/runs/${runId}`, "DELETE");
      if (currentRunId === runId) resetComposer();
      else await refreshHistory();
      toast("已删除搜索记录");
    } catch (error) {
      toast(error.message || "无法删除搜索记录", "error");
    }
  }

  function resetComposer() {
    stopPoll();
    currentRunId = "";
    currentSources = [];
    document.getElementById("x-search-form")?.reset();
    const result = document.getElementById("x-search-result");
    result.className = "knowledge-run-placeholder";
    result.innerHTML = `<h3>尚未创建搜索 Run</h3><p>真实运行时，此区域将按结构化事件显示“请求已接受、正在执行 X Search、整理来源、生成结论、已完成”。不会显示隐藏推理过程。</p>`;
    document.getElementById("x-search-sources").innerHTML = empty("尚无 [X] 引用", "每张来源卡将显示账号、可访问链接、发布时间（若提供）与检索时间，并标注为外部实时线索。");
    refreshHistory().catch((error) => toast(error.message || "无法刷新搜索历史", "error"));
  }

  KBotKnowledgeShell.ready.then(async (access) => {
    if (!access) return;
    await loadAgents();
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
        toast("请选择已绑定 Grok 模型的 ACTIVE Agent。", "error");
        return;
      }
      const mode = document.getElementById("x-search-handle-mode").value;
      const handleList = splitHandles(document.getElementById("x-search-handles").value);
      const payload = {
        agent_id: document.getElementById("x-search-agent").value,
        input: document.getElementById("x-search-query").value.trim(),
      };
      if (from) payload.from_date = from;
      if (to) payload.to_date = to;
      if (mode === "allowed" && handleList.length) payload.allowed_x_handles = handleList;
      if (mode === "excluded" && handleList.length) payload.excluded_x_handles = handleList;
      if (document.getElementById("x-search-image").checked) payload.enable_image_understanding = true;
      if (document.getElementById("x-search-video").checked) payload.enable_video_understanding = true;
      try {
        const run = await KBotKnowledgeApi.json("/x-search/runs", "POST", payload, {
          headers: { "Idempotency-Key": KBotKnowledgeApi.requestId() },
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
