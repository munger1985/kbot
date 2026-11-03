(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/output-jobs";
  const stages = { ACCEPTED: "已排队", VALIDATING: "核验来源", ASSEMBLING: "装配报告", REDACTING: "提炼与脱敏",
    INGESTING: "知识入库", INDEXING: "等待索引", FINALIZING: "确认产物", COMPLETED: "已完成", FAILED: "失败" };
  function mount(element, options = {}) {
    if (!element) return null;
    const shell = globalThis.KBotAIOpsShell, escape = shell.escape;
    let rows = [], before = "", next = null, loading = false, retrying = "", error = "", sequence = 0;
    let timer, controller, stopped = false, failures = 0, conversationId = options.conversationId || "";
    const focusedId = new URLSearchParams(location.search).get("job_id") || "";
    const active = row => ["QUEUED", "RUNNING"].includes(row.status);
    function resultHref(row) {
      if (!row.result) return "";
      if (row.result.report_id) return "./report-detail.html?" + new URLSearchParams({
        id: row.result.report_id, report_version: row.result.report_version, job_id: row.job_id });
      return "./cases.html?" + new URLSearchParams({ asset_id: row.result.asset_id,
        asset_version_id: row.result.asset_version_id, job_id: row.job_id });
    }
    function label(row) {
      if (row.status === "FAILED") return "生成失败";
      if (row.status === "SUCCEEDED") return row.job_type === "CASE_EXTRACTION" ? "草稿已就绪" : "报告已生成";
      return stages[row.stage] || "处理中";
    }
    function render() {
      const expanded = element.querySelector("details")?.open || false;
      element.hidden = Boolean(options.requireConversation && !conversationId);
      if (element.hidden) return;
      element.innerHTML = `<header class="resource-list-header"><div class="resource-list-heading"><h2>我的生成记录</h2><p>后台处理，可离开页面。案例成功代表草稿就绪，人工发布后才供 Agent 检索。</p></div><div class="resource-list-tools"><button type="button" class="secondary-button" data-output-refresh ${loading ? "disabled" : ""}>刷新生成记录</button></div></header>${error ? `<p class="empty-cell" role="alert">${escape(error)}；可手动刷新重试。</p>` : ""}<div class="table-scroll"><table><thead><tr><th>输出</th><th>提交时间</th><th>阶段</th><th>结果 / 原因</th><th>操作</th></tr></thead><tbody>${rows.map(row => `<tr data-job-id="${escape(row.job_id)}"><td>${row.job_type === "CASE_EXTRACTION" ? "沉淀案例" : "生成报告"}<small class="table-secondary technical-value">${escape(row.job_id)}</small></td><td>${escape(shell.fmt(row.submitted_at))}</td><td><span class="status-pill ${row.status === "FAILED" ? "status-danger" : row.status === "SUCCEEDED" ? "status-success" : "status-warning"}">${label(row)}</span></td><td>${escape(row.status === "FAILED" ? row.error_message || row.error_code || "生成失败" : row.status === "SUCCEEDED" ? "已保存至对应页面" : "后台处理，无需停留")}</td><td>${row.result ? `<a class="row-action" href="${escape(resultHref(row))}">${row.job_type === "CASE_EXTRACTION" ? "查看案例版本" : "查看报告"}</a>` : ""}${row.status === "FAILED" && row.retryable ? `<button class="row-action" type="button" data-output-retry="${escape(row.job_id)}" ${retrying ? "disabled" : ""}>重试生成</button>` : ""}</td></tr>`).join("") || `<tr><td class="empty-cell" colspan="5">${loading ? "正在读取生成记录…" : "暂无生成任务。"}</td></tr>`}</tbody></table></div><footer class="button-row">${before ? '<button type="button" class="secondary-button" data-output-refresh>最新记录</button>' : ""}${next ? '<button type="button" class="secondary-button" data-output-earlier>更早记录</button>' : ""}</footer>`;
      if (options.requireConversation) {
        const summary = `<summary class="resource-list-header">我的生成记录 · ${rows.length} 项 · ${rows.filter(active).length} 项处理中<span>查看阶段、结果与重试</span></summary>`;
        element.innerHTML = `<details ${expanded ? "open" : ""}>${summary}${element.innerHTML}</details>`;
      }
    }
    function schedule() {
      clearTimeout(timer);
      if (!stopped && !document.hidden && failures < 3 && rows.some(active)) timer = setTimeout(load, 5000);
    }
    async function load() {
      clearTimeout(timer);
      controller?.abort();
      controller = new AbortController();
      const current = ++sequence;
      if (stopped || (options.requireConversation && !conversationId)) { render(); return; }
      loading = true;
      render();
      const query = new URLSearchParams({ limit: "10" });
      if (conversationId) query.set("conversation_id", conversationId);
      if (options.jobType) query.set("job_type", options.jobType);
      if (before) query.set("before_id", before);
      try {
        const page = await KBotAIOpsAuth.request(api + "?" + query, { signal: controller.signal });
        if (current !== sequence || stopped) return;
        let items = page.items || [];
        if (focusedId && !items.some(row => row.job_id === focusedId)) {
          const focused = await KBotAIOpsAuth.request(api + "/" + encodeURIComponent(focusedId), { signal: controller.signal });
          if (current !== sequence || stopped) return;
          if ((!options.jobType || focused.job_type === options.jobType)
            && (!conversationId || (focused.source.kind === "CONVERSATION" && focused.source.source_id === conversationId))) items = [focused, ...items];
        }
        const previous = new Map(rows.map(row => [row.job_id, row.status]));
        const completed = items.some(row => row.status === "SUCCEEDED" && ["QUEUED", "RUNNING"].includes(previous.get(row.job_id)));
        rows = items; next = page.next_cursor; error = ""; failures = 0;
        if (completed) options.onCompleted?.();
      } catch (cause) {
        if (current !== sequence || stopped) return;
        error = cause.message || "无法读取生成记录"; failures += 1;
      } finally {
        if (current === sequence && !stopped) { loading = false; render(); schedule(); }
      }
    }
    async function refresh() { before = ""; failures = 0; await load(); }
    async function retry(id) {
      const row = rows.find(item => item.job_id === id);
      if (!row || retrying) return;
      retrying = id; render();
      try {
        await KBotAIOpsAuth.request(api + "/" + encodeURIComponent(id) + ":retry", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ expected_row_version: row.row_version }),
        });
        await refresh();
      } catch (cause) { shell.toast(cause.message || "无法重试，请刷新核对任务状态"); }
      finally { retrying = ""; render(); }
    }
    function click(event) {
      if (event.target.closest("[data-output-refresh]")) void refresh();
      if (event.target.closest("[data-output-earlier]") && !loading && next) { before = next; void load(); }
      const button = event.target.closest("[data-output-retry]");
      if (button) void retry(button.dataset.outputRetry);
    }
    function visibility() { if (!document.hidden) void load(); else clearTimeout(timer); }
    function pagehide(event) {
      if (!event.persisted) { destroy(); return; }
      sequence += 1; controller?.abort(); clearTimeout(timer);
    }
    function pageshow(event) {
      if (event.persisted) { failures = 0; void load(); }
    }
    function destroy() {
      stopped = true; sequence += 1; controller?.abort(); clearTimeout(timer);
      element.removeEventListener("click", click);
      document.removeEventListener("visibilitychange", visibility);
      window.removeEventListener("pagehide", pagehide);
      window.removeEventListener("pageshow", pageshow);
    }
    function setConversation(value) {
      if (conversationId === value) return;
      conversationId = value || ""; rows = []; before = ""; next = null; failures = 0; error = ""; void load();
    }
    element.classList.add("table-card", "aiops-output-jobs");
    element.setAttribute("aria-label", "我的生成记录");
    element.addEventListener("click", click);
    document.addEventListener("visibilitychange", visibility);
    window.addEventListener("pagehide", pagehide);
    window.addEventListener("pageshow", pageshow);
    void load();
    return { refresh, setConversation, destroy };
  }
  globalThis.KBotAIOpsOutputJobs = Object.freeze({ mount });
})();
