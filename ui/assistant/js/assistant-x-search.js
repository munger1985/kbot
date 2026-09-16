/* X 实时搜索：未就绪时不创建 Run，就绪后把结论整理成综述、帖子与来源卡。 */
(function () {
  "use strict";

  const { badge, capability, capabilityLabel, escapeHtml, toast } = KBotAssistantShell;
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
  let accessState = null;
  let currentSources = [];

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

  function inlineFormat(text, urlMap) {
    const pattern = /\[\[(\d+)\]\]\((https?:[^)\s]+)\)|\[(X\d+)\](?:\((https?:[^)\s]+)\))?|\[([^\]]{1,80})\]\((https?:[^)\s]+)\)/g;
    let last = 0;
    let out = "";
    let match;
    const source = String(text || "");
    while ((match = pattern.exec(source))) {
      out += escapeHtml(source.slice(last, match.index));
      if (match[1]) {
        const url = match[2];
        const row = urlMap.get(normalizeUrl(url));
        const label = (row && row.citation_label) || `[X${match[1]}]`;
        out += citationChip(label, url, !row);
      } else if (match[3]) {
        const label = `[${match[3]}]`;
        const url = match[4] || "";
        const row = url ? urlMap.get(normalizeUrl(url)) : null;
        out += citationChip(label, url, Boolean(url) && !row);
      } else {
        const url = match[6];
        const row = urlMap.get(normalizeUrl(url));
        if (row) out += citationChip(row.citation_label, url, false);
        else {
          out += `<a class="x-inline-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(match[5])}</a>`;
        }
      }
      last = pattern.lastIndex;
    }
    out += escapeHtml(source.slice(last));
    return out;
  }

  function parseAnswerBlocks(text) {
    const lines = String(text || "").replaceAll("\r\n", "\n").trim().split("\n");
    const blocks = [];
    let para = [];
    let list = [];
    const flushPara = () => {
      const value = para.join("\n").trim();
      if (value) blocks.push({ type: "p", text: value });
      para = [];
    };
    const flushList = () => {
      if (list.length) blocks.push({ type: "list", items: list.slice() });
      list = [];
    };
    lines.forEach((line) => {
      const item = line.match(/^\s*(?:[-*]|\d+\.)\s+(.+)$/);
      if (item) {
        flushPara();
        list.push(item[1]);
        return;
      }
      if (!line.trim()) {
        flushPara();
        flushList();
        return;
      }
      flushList();
      para.push(line);
    });
    flushPara();
    flushList();
    return blocks;
  }

  function renderPost(item, urlMap) {
    const match = String(item || "").match(/^(.{2,36}?)[：:]([\s\S]+)$/);
    const kicker = match ? match[1].trim() : "";
    const body = match ? match[2].trim() : String(item || "");
    return `<article class="x-post">
      ${kicker ? `<p class="x-post-kicker">${escapeHtml(kicker)}</p>` : ""}
      <div class="x-post-body">${inlineFormat(body, urlMap)}</div>
    </article>`;
  }

  function maybePromoteInlineList(blocks) {
    if (blocks.length !== 1 || blocks[0].type !== "p") return blocks;
    const parts = blocks[0].text.split(/\s+-\s+/);
    if (parts.length < 3) return blocks;
    const intro = parts[0].trim();
    const items = parts.slice(1).map((item) => item.trim()).filter(Boolean);
    const last = items[items.length - 1] || "";
    const peeled = last.match(/^(.*?\]\(https?:[^)]+\))\s+(.+)$/);
    const result = [];
    if (intro) result.push({ type: "p", text: intro });
    if (peeled) {
      result.push({ type: "list", items: items.slice(0, -1).concat(peeled[1].trim()) });
      if (peeled[2].trim()) result.push({ type: "p", text: peeled[2].trim() });
    } else {
      result.push({ type: "list", items });
    }
    return result;
  }

  function renderAnswer(text, sources) {
    const urlMap = sourceByUrl(sources);
    const blocks = maybePromoteInlineList(parseAnswerBlocks(text));
    if (!blocks.length) return "";
    const firstList = blocks.findIndex((block) => block.type === "list");
    let briefing = blocks;
    let lists = [];
    let closing = [];
    if (firstList >= 0) {
      briefing = blocks.slice(0, firstList);
      let index = firstList;
      while (index < blocks.length && blocks[index].type === "list") {
        lists.push(blocks[index]);
        index += 1;
      }
      closing = blocks.slice(index);
    }
    const parts = [];
    if (briefing.length) {
      parts.push(`<section class="x-brief"><h4>检索说明</h4>${briefing.map((block) => `<p>${inlineFormat(block.text, urlMap)}</p>`).join("")}</section>`);
    }
    lists.forEach((block) => {
      parts.push(`<section class="x-posts"><h4>按时间整理</h4><div class="x-post-list">${block.items.map((item) => renderPost(item, urlMap)).join("")}</div></section>`);
    });
    if (closing.length) {
      parts.push(`<section class="x-brief"><h4>主题归纳</h4>${closing.map((block) => `<p>${inlineFormat(block.text, urlMap)}</p>`).join("")}</section>`);
    }
    return parts.join("");
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
    const card = document.querySelector(`.assistant-source-card[data-citation="${CSS.escape(selected)}"]`);
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
      <article class="assistant-history-item" data-run-id="${escapeHtml(row.run_id)}" ${row.run_id === currentRunId ? 'aria-current="true"' : ""}>
        <button class="assistant-history-open" type="button" data-run-id="${escapeHtml(row.run_id)}">
          <strong>${escapeHtml((row.request && row.request.input) || "X Search")}</strong>
          <small>${escapeHtml(stageLabel(row.status))} · ${escapeHtml(displayTime(row.created_at) || "")}</small>
        </button>
        <button class="small danger assistant-history-delete" type="button" data-run-id="${escapeHtml(row.run_id)}">删除</button>
      </article>`).join("");
    node.querySelectorAll(".assistant-history-open").forEach((button) => {
      button.addEventListener("click", () => openRun(button.dataset.runId));
    });
    node.querySelectorAll(".assistant-history-delete").forEach((button) => {
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
      body = `<div class="assistant-notice warn">${escapeHtml(run.error_message || "搜索失败")}</div>`;
    } else if (answer) {
      body = renderAnswer(answer, sources);
    } else {
      const stages = (events || []).map((item) => `<li><strong>${escapeHtml(stageLabel(item.stage))}</strong><span>${escapeHtml(item.message || "")}</span></li>`).join("");
      body = `<ol class="assistant-flow">${stages || "<li><strong>已接受请求</strong><span>正在等待公开阶段</span></li>"}</ol>`;
    }
    node.className = answer ? "x-search-result" : "assistant-run-placeholder";
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
    bindCitations(node);
  }

  function renderSourceCard(row) {
    const handle = displayHandle(row.author_handle);
    const published = displayTime(row.published_at);
    const retrieved = displayTime(row.retrieved_at);
    const summary = row.excerpt || row.title || "";
    return `<article class="assistant-source-card" data-citation="${escapeHtml(row.citation_label || "")}">
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
    node.querySelectorAll(".assistant-source-card").forEach((card) => {
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
    const rows = await KBotAssistantApi.json("/x-search/runs", "GET");
    renderHistory(Array.isArray(rows) ? rows : []);
  }

  async function loadRun(runId) {
    const [run, events, sources] = await Promise.all([
      KBotAssistantApi.json(`/x-search/runs/${runId}`, "GET"),
      KBotAssistantApi.json(`/x-search/runs/${runId}/events`, "GET"),
      KBotAssistantApi.json(`/x-search/runs/${runId}/sources`, "GET"),
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
      await KBotAssistantApi.json(`/x-search/runs/${runId}`, "DELETE");
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
    result.className = "assistant-run-placeholder";
    result.innerHTML = `<h3>尚未创建搜索 Run</h3><p>真实运行时，此区域将按结构化事件显示“请求已接受、正在执行 X Search、整理来源、生成结论、已完成”。不会显示隐藏推理过程。</p>`;
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
