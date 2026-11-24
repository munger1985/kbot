(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/operations-knowledge";
  const shell = KBotAIOpsShell;
  const find = id => document.getElementById("aiops-search-" + id);
  const form = document.querySelector(".aiops-search-form");
  const submit = form.querySelector('button[type="submit"]');
  const scope = find("scope");
  const library = find("library");
  const query = find("query");
  let activeSearch;
  let result;
  let selected = 0;
  let error = "";
  let librariesUnavailable = false;

  function showResult() {
    const hit = result?.results[selected];
    find("hit-title").textContent = hit?.title || "";
    find("passages").innerHTML = (hit?.passages || []).map(passage =>
      `<blockquote class="aiops-search-quote">${shell.escape(passage.quote)}${passage.truncated ? "<small>此处仅展示片段开头。</small>" : ""}</blockquote>`).join("");
    find("hits").querySelectorAll("[data-result]").forEach((button, index) => {
      button.setAttribute("aria-pressed", index === selected ? "true" : "false");
    });
  }

  function render() {
    const manual = scope.value === "MANUAL";
    find("library-field").hidden = !manual;
    find("library-error").hidden = !manual || !librariesUnavailable;
    find("case-hint").hidden = manual;
    submit.disabled = !query.value.trim() || Boolean(activeSearch);
    submit.textContent = activeSearch ? "检索中…" : "检索";
    find("error").hidden = !error;
    find("error").textContent = error;
    find("warnings").hidden = !result?.warnings.length;
    find("warnings").innerHTML = (result?.warnings || []).map(message => `<li>${shell.escape(message)}</li>`).join("");
    const rows = result?.results || [];
    find("split").hidden = !rows.length;
    find("empty").hidden = Boolean(rows.length || error);
    find("empty").textContent = activeSearch ? "正在检索已发布内容…" : ({
      NO_ELIGIBLE_RESOURCES: "当前范围没有符合检索条件的已发布资料。",
      NO_MATCH: "没有找到匹配的正文片段。",
      NOT_VERIFIED: "本次检索未能充分核验，不表示没有匹配资料。",
    })[result?.retrieval_conclusion] || "输入要查的内容，查看匹配的正文片段。";
    find("count").textContent = `${rows.length} 项`;
    find("hits").innerHTML = rows.map((row, index) =>
      `<li><button type="button" class="secondary-button aiops-search-hit" data-result="${index}"><span>${shell.escape(row.title)}</span><small>v${shell.escape(row.version)}</small></button></li>`).join("");
    showResult();
  }

  function clearResult() {
    activeSearch?.abort();
    activeSearch = undefined;
    result = null;
    selected = 0;
    error = "";
  }

  async function search(event) {
    event.preventDefault();
    const text = query.value.trim();
    if (!text || activeSearch) return;
    clearResult();
    const search = activeSearch = new AbortController();
    render();
    try {
      const response = await KBotAIOpsAuth.request(`${api}/search`, { method: "POST", signal: search.signal,
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({
          query: text, source_kind: scope.value, knowledge_base_id: scope.value === "MANUAL" ? library.value || null : null,
        }) });
      if (activeSearch === search && !search.signal.aborted) result = response;
    } catch (cause) {
      if (activeSearch === search && !search.signal.aborted) error = cause.message || "检索失败，请稍后重试。";
    } finally {
      if (activeSearch === search) {
        activeSearch = undefined;
        render();
      }
    }
  }

  shell.ready.then(async access => {
    if (!access?.permissions.includes("aiops:knowledge_manage")) return;
    form.addEventListener("submit", search);
    query.addEventListener("input", () => { submit.disabled = !query.value.trim() || Boolean(activeSearch); });
    form.addEventListener("change", event => {
      if (event.target !== scope && event.target !== library) return;
      clearResult();
      render();
    });
    find("hits").addEventListener("click", event => {
      const button = event.target.closest("[data-result]");
      if (!button) return;
      selected = Number(button.dataset.result);
      showResult();
    });
    const loadLibraries = new AbortController();
    window.addEventListener("pagehide", () => {
      loadLibraries.abort();
      clearResult();
      render();
    });
    try {
      const data = await KBotAIOpsAuth.request(`${api}/knowledge-bases`, { signal: loadLibraries.signal });
      for (const item of data.items.filter(item => item.status === "ACTIVE")) {
        library.add(new Option(item.display_name, item.knowledge_base_id));
      }
    } catch {
      if (!loadLibraries.signal.aborted) librariesUnavailable = true;
    }
    if (!loadLibraries.signal.aborted) find("library-error").hidden = scope.value !== "MANUAL" || !librariesUnavailable;
  }).catch(error => shell.toast(error.message));
})();
