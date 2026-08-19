/* KBot Agent Run 全链路开发调试页面。 */
(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    runs: [],
    selectedRunId: null,
    projection: null,
    autoRefreshTimer: null,
  };

  KBotUI.bindAuthForm($("#auth-form"), async () => {
    await loadRuns();
  });

  function formatTime(value) {
    if (!value) return "-";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
  }

  function formatDuration(value) {
    if (value == null) return "-";
    const milliseconds = Number(value);
    if (milliseconds < 1000) return `${milliseconds.toFixed(0)} ms`;
    return `${(milliseconds / 1000).toFixed(2)} s`;
  }

  function statusClass(status) {
    const value = String(status || "").toUpperCase();
    if (["COMPLETED", "SUCCEEDED", "READY"].includes(value)) return "ok";
    if (["FAILED", "EXPIRED"].includes(value)) return "error";
    if (["RETRY_WAIT", "WARNING", "DEGRADED"].includes(value)) return "warn";
    return "";
  }

  function filteredRuns() {
    const form = $("#run-filter-form");
    const keyword = form.elements.keyword.value.trim().toLowerCase();
    const status = form.elements.status.value;
    return state.runs.filter((run) => {
      if (status && run.status !== status) return false;
      if (!keyword) return true;
      return [
        run.run_id,
        run.agent_id,
        run.actor_id,
        run.original_input,
        run.status,
        run.error_code,
      ]
        .join(" ")
        .toLowerCase()
        .includes(keyword);
    });
  }

  function renderRunList() {
    const rows = filteredRuns();
    $("#run-list").innerHTML =
      rows
        .map(
          (run) => `
          <button class="run-list-item ${
            run.run_id === state.selectedRunId ? "active" : ""
          }" data-run-id="${KBotUI.escapeHtml(run.run_id)}" type="button">
            <span class="run-list-head">
              <strong>${KBotUI.escapeHtml(run.status)}</strong>
              <span>${KBotUI.escapeHtml(formatDuration(run.duration_ms))}</span>
            </span>
            <span class="run-question">${KBotUI.escapeHtml(
              run.original_input || "无输入"
            )}</span>
            <span class="muted">${KBotUI.escapeHtml(
              formatTime(run.created_at)
            )} · ${KBotUI.escapeHtml(run.run_id)}</span>
          </button>`
        )
        .join("") || '<p class="muted">没有符合条件的 Run。</p>';
  }

  async function loadRuns(options) {
    const limit = $("#run-filter-form").elements.limit.value;
    if (!options?.quiet) {
      KBotUI.setStatus($("#run-list-status"), "正在加载 Run…");
    }
    try {
      const payload = await KBotUI.api(
        `/api/v1/development/agent-runs?limit=${limit}`
      );
      state.runs = payload.runs || [];
      renderRunList();
      KBotUI.setStatus(
        $("#run-list-status"),
        `已读取 ${state.runs.length} 个 Run`,
        "ok"
      );
      if (
        state.selectedRunId &&
        state.runs.some((run) => run.run_id === state.selectedRunId)
      ) {
        await loadRun(state.selectedRunId, { quiet: true });
      }
    } catch (error) {
      KBotUI.setStatus($("#run-list-status"), error.message, "error");
    }
  }

  async function loadRun(runId, options) {
    state.selectedRunId = runId;
    renderRunList();
    if (!options?.quiet) {
      $("#run-overview").innerHTML = '<p class="muted">正在聚合执行链路…</p>';
    }
    try {
      state.projection = await KBotUI.api(
        `/api/v1/development/agent-runs/${runId}?log_limit=500`
      );
      renderProjection();
    } catch (error) {
      $("#run-overview").innerHTML = `<p class="status error">${KBotUI.escapeHtml(
        error.message
      )}</p>`;
    }
  }

  function renderProjection() {
    const projection = state.projection || {};
    const run = projection.run || {};
    $("#run-overview").innerHTML = `
      <div class="debug-summary-grid">
        ${summary("状态", run.status, statusClass(run.status))}
        ${summary("总耗时", formatDuration(run.duration_ms))}
        ${summary("任务", String((projection.tasks || []).length))}
        ${summary("事件", String((projection.events || []).length))}
        ${summary("Artifacts", String((projection.artifacts || []).length))}
        ${summary("关联日志", String((projection.logs || []).length))}
      </div>
      <dl class="debug-meta">
        <dt>问题</dt><dd>${KBotUI.escapeHtml(run.original_input || "-")}</dd>
        <dt>Run ID</dt><dd>${KBotUI.escapeHtml(run.run_id || "-")}</dd>
        <dt>Agent</dt><dd>${KBotUI.escapeHtml(run.agent_id || "-")}</dd>
        <dt>Trace</dt><dd>${KBotUI.escapeHtml(run.trace_id || "-")}</dd>
        <dt>时间</dt><dd>${KBotUI.escapeHtml(formatTime(run.created_at))}</dd>
      </dl>`;
    renderTimeline();
    renderRetrieval();
    renderModels();
    renderTasks();
    renderEvents();
    renderArtifacts();
    renderLogs();
    renderErrors();
  }

  function summary(label, value, kind) {
    return `<div class="debug-summary ${KBotUI.escapeHtml(kind || "")}">
      <span>${KBotUI.escapeHtml(label)}</span>
      <strong>${KBotUI.escapeHtml(value)}</strong>
    </div>`;
  }

  function renderTimeline() {
    const events = state.projection?.events || [];
    const visibleTypes = new Set([
      "RUN_CREATED",
      "memory.context_loaded",
      "RUN_STARTED",
      "query.rewritten",
      "retrieval.completed",
      "answer.completed",
      "RUN_COMPLETED",
      "RUN_FAILED",
      "RUN_CANCELLED",
    ]);
    const steps = events.filter((event) => visibleTypes.has(event.event_type));
    const labels = {
      RUN_CREATED: "接收问题",
      "memory.context_loaded": "加载会话与长期记忆",
      RUN_STARTED: "意图路由与执行计划",
      "query.rewritten": "上下文问题改写",
      "retrieval.completed": "Knowledge Core 两阶段检索",
      "answer.completed": "回答与引用生成",
      RUN_COMPLETED: "Run 完成",
      RUN_FAILED: "Run 失败",
      RUN_CANCELLED: "Run 已取消",
    };
    $("#run-timeline").innerHTML =
      steps
        .map(
          (event, index) => `
          <button class="timeline-step ${KBotUI.escapeHtml(
            statusClass(
              event.payload?.status ||
                (event.event_type === "RUN_FAILED" ? "FAILED" : "COMPLETED")
            )
          )}" data-kind="event" data-index="${events.indexOf(
            event
          )}" type="button">
            <span class="timeline-marker">${index + 1}</span>
            <span>
              <strong>${KBotUI.escapeHtml(
                labels[event.event_type] || event.event_type
              )}</strong>
              <small>${KBotUI.escapeHtml(
                event.payload?.public_summary ||
                  event.payload?.status ||
                  event.event_type
              )} · ${KBotUI.escapeHtml(formatTime(event.created_at))}</small>
            </span>
          </button>`
        )
        .join("") || '<p class="muted">Run 尚未生成阶段事件。</p>';
  }

  function citationArtifact() {
    return (state.projection?.artifacts || [])
      .filter((item) => item.artifact_type === "CITATION_PACK")
      .at(-1);
  }

  function renderRetrieval() {
    const artifact = citationArtifact();
    if (!artifact) {
      $("#tab-retrieval").innerHTML =
        '<p class="muted">该 Run 没有生成 CITATION_PACK，可能未进入文档检索或在检索前结束。</p>';
      return;
    }
    const payload = artifact.payload || {};
    const report = payload.retrieval_report || {};
    const diagnostics =
      report.diagnostics ||
      payload.citation_pack?.query_plan?.diagnostics ||
      {};
    const discovery = diagnostics.discovery || {};
    const evidence = diagnostics.evidence || {};
    $("#tab-retrieval").innerHTML = `
      <div class="debug-summary-grid retrieval-summary">
        ${summary("Discovery 全文", String(discovery.text_hits ?? "-"))}
        ${summary("Discovery 向量", String(discovery.vector_hits ?? "-"))}
        ${summary("Bundle 候选", String(discovery.bundle_candidates ?? report.discovery_candidate_count ?? "-"))}
        ${summary("Evidence 全文", String(evidence.text_hits ?? "-"))}
        ${summary("Evidence 向量", String(evidence.vector_hits ?? "-"))}
        ${summary("Anchor", String(evidence.selected_anchors ?? "-"))}
        ${summary("上下文扩展", String(evidence.expanded_contexts ?? "-"))}
        ${summary("引用组", String(evidence.citation_groups ?? report.citation_count ?? "-"))}
      </div>
      <div class="debug-columns">
        ${jsonCard("Discovery 诊断", discovery)}
        ${jsonCard("Evidence 诊断", evidence)}
      </div>
      <div class="actions">
        <button data-kind="artifact" data-id="${KBotUI.escapeHtml(
          artifact.artifact_id
        )}" type="button">查看完整 Citation Pack</button>
      </div>`;
  }

  function jsonCard(title, value) {
    return `<article class="debug-card">
      <h3>${KBotUI.escapeHtml(title)}</h3>
      <pre>${KBotUI.escapeHtml(KBotUI.json(value))}</pre>
    </article>`;
  }

  function renderModels() {
    const run = state.projection?.run || {};
    const config = run.config_snapshot || {};
    const agentModels =
      config.agent?.models || config.agent?.config?.models || {};
    const allLogs = state.projection?.logs || [];
    const modelLogs = allLogs.filter((log) => {
      const source = `${log.location || ""} ${log.message || ""}`.toLowerCase();
      return ["model", "embedding", "llm", "vlm", "嵌入", "模型"].some(
        (value) => source.includes(value)
      );
    });
    $("#tab-models").innerHTML = `
      <div class="debug-columns">
        ${jsonCard("Agent 模型快照", agentModels)}
      </div>
      <h3>关联模型日志</h3>
      ${recordTable(
        ["时间", "服务", "位置", "级别", "消息"],
        modelLogs.map((log) => ({
          cells: [
            formatTime(log.timestamp),
            log.service_name,
            log.location,
            log.level,
            log.message,
          ],
          attributes: `data-kind="log" data-index="${allLogs.indexOf(log)}"`,
        }))
      )}`;
  }

  function renderTasks() {
    const tasks = state.projection?.tasks || [];
    $("#tab-tasks").innerHTML = recordTable(
      ["Task", "Skill", "状态", "尝试", "耗时"],
      tasks.map((task, index) => ({
        cells: [
          task.task_key,
          task.skill_id || "-",
          task.status,
          `${task.attempt}/${task.max_attempts}`,
          formatDuration(task.duration_ms),
        ],
        attributes: `data-kind="task" data-index="${index}"`,
      }))
    );
  }

  function renderEvents() {
    const events = state.projection?.events || [];
    $("#tab-events").innerHTML = recordTable(
      ["序号", "时间", "事件", "Task", "摘要"],
      events.map((event, index) => ({
        cells: [
          `#${event.sequence_no}`,
          formatTime(event.created_at),
          event.event_type,
          event.task_id || "-",
          event.payload?.public_summary ||
            event.payload?.status ||
            event.payload?.task_key ||
            "-",
        ],
        attributes: `data-kind="event" data-index="${index}"`,
      }))
    );
  }

  function renderArtifacts() {
    const artifacts = state.projection?.artifacts || [];
    $("#tab-artifacts").innerHTML = recordTable(
      ["时间", "类型", "Producer", "Task", "ID"],
      artifacts.map((artifact) => ({
        cells: [
          formatTime(artifact.created_at),
          artifact.artifact_type,
          artifact.producer,
          artifact.task_id || "-",
          artifact.artifact_id,
        ],
        attributes: `data-kind="artifact" data-id="${KBotUI.escapeHtml(
          artifact.artifact_id
        )}"`,
      }))
    );
  }

  function renderLogs() {
    const logs = state.projection?.logs || [];
    $("#tab-logs").innerHTML = recordTable(
      ["时间", "服务", "进程", "级别", "消息"],
      logs.map((log, index) => ({
        cells: [
          formatTime(log.timestamp),
          log.service_name,
          log.process,
          log.level,
          log.message,
        ],
        attributes: `data-kind="log" data-index="${index}"`,
      }))
    );
  }

  function renderErrors() {
    const tasks = (state.projection?.tasks || []).filter(
      (task) =>
        task.error_code ||
        ["FAILED", "RETRY_WAIT"].includes(String(task.status))
    );
    const logs = (state.projection?.logs || []).filter((log) =>
      ["ERROR", "CRITICAL", "WARNING"].includes(log.level)
    );
    const cards = [
      ...tasks.map((item) => jsonCard(`Task ${item.task_key}`, item)),
      ...logs.map((item) => jsonCard(`${item.service_name} ${item.level}`, item)),
    ];
    $("#tab-errors").innerHTML =
      cards.join("") || '<p class="muted">该 Run 没有错误、警告或重试记录。</p>';
  }

  function recordTable(headers, rows) {
    if (!rows.length) return '<p class="muted">暂无记录。</p>';
    return `<div class="log-table-wrap"><table class="log-table debug-table">
      <thead><tr>${headers
        .map((header) => `<th>${KBotUI.escapeHtml(header)}</th>`)
        .join("")}</tr></thead>
      <tbody>${rows
        .map(
          (row) => `<tr tabindex="0" ${row.attributes || ""}>${row.cells
            .map((cell) => `<td>${KBotUI.escapeHtml(cell ?? "-")}</td>`)
            .join("")}</tr>`
        )
        .join("")}</tbody>
    </table></div>`;
  }

  function showDetail(target) {
    const kind = target.dataset.kind;
    let value = null;
    if (kind === "task") {
      value = state.projection?.tasks?.[Number(target.dataset.index)];
    } else if (kind === "event") {
      value = state.projection?.events?.[Number(target.dataset.index)];
    } else if (kind === "log") {
      value = state.projection?.logs?.[Number(target.dataset.index)];
    } else if (kind === "artifact") {
      value = (state.projection?.artifacts || []).find(
        (item) => item.artifact_id === target.dataset.id
      );
    }
    if (value) $("#debug-detail").textContent = KBotUI.json(value);
  }

  $("#run-filter-form").addEventListener("submit", (event) => {
    event.preventDefault();
    loadRuns();
  });
  $("#run-filter-form").elements.keyword.addEventListener(
    "input",
    renderRunList
  );
  $("#run-filter-form").elements.status.addEventListener(
    "change",
    renderRunList
  );
  $("#run-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-run-id]");
    if (button) loadRun(button.dataset.runId);
  });
  $(".debug-console").addEventListener("click", (event) => {
    const target = event.target.closest("[data-kind]");
    if (target) showDetail(target);
  });
  $(".debug-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("[data-tab]");
    if (!button) return;
    for (const item of document.querySelectorAll(".debug-tabs button")) {
      item.classList.toggle("active", item === button);
    }
    for (const panel of document.querySelectorAll(".debug-tab-panel")) {
      panel.classList.toggle("hidden", panel.id !== `tab-${button.dataset.tab}`);
    }
  });
  $("#auto-refresh").addEventListener("click", (event) => {
    if (state.autoRefreshTimer) {
      window.clearInterval(state.autoRefreshTimer);
      state.autoRefreshTimer = null;
      event.currentTarget.textContent = "自动刷新：关";
      return;
    }
    state.autoRefreshTimer = window.setInterval(
      () => loadRuns({ quiet: true }),
      5000
    );
    event.currentTarget.textContent = "自动刷新：5 秒";
  });

  loadRuns();
})();
