/* 文生图：未就绪时不创建 Run；预览必须经 requestBlob 生成对象 URL。 */
(function () {
  "use strict";

  const { badge, capability, capabilityLabel, escapeHtml, toast } = KBotAssistantShell;
  const TERMINAL = new Set(["COMPLETED", "FAILED", "REJECTED"]);
  const STAGE_LABEL = {
    ACCEPTED: "已接受请求",
    GENERATING: "正在生成",
    COMPLETED: "已完成",
    FAILED: "失败",
    REJECTED: "内容限制拒绝",
  };
  let currentRunId = "";
  let pollTimer = 0;
  let accessState = null;
  const previewUrls = [];

  function ready() {
    return Boolean(capability(accessState, "image_generation").ready);
  }

  function paintCapability() {
    const node = document.getElementById("image-capability");
    if (!node) return;
    const item = capability(accessState, "image_generation");
    const [label, tone] = capabilityLabel(item);
    node.className = `assistant-badge ${tone}`;
    node.textContent = label;
    const submit = document.getElementById("image-submit");
    if (submit) submit.disabled = !item.ready;
  }

  function empty(title, copy) {
    return `<div class="assistant-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div>`;
  }

  function stageLabel(value) {
    const key = String(value || "").trim();
    return STAGE_LABEL[key] || key || "进行中";
  }

  function displayTime(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    const iso = raw.match(/^(\d{4}-\d{2}-\d{2})/);
    return iso ? iso[1] : raw;
  }

  function revokePreviews() {
    while (previewUrls.length) {
      URL.revokeObjectURL(previewUrls.pop());
    }
  }

  function setMeta(rows) {
    const node = document.getElementById("image-meta");
    if (!node) return;
    node.innerHTML = rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  }

  function fillComposer(run) {
    const prompt = document.getElementById("image-prompt");
    const ratio = document.getElementById("image-ratio");
    const count = document.getElementById("image-count");
    const request = run && run.request ? run.request : {};
    if (prompt) prompt.value = request.prompt || "";
    if (ratio && request.aspect_ratio) ratio.value = request.aspect_ratio;
    if (count && request.count) count.value = String(request.count);
  }

  function renderHistory(rows) {
    const node = document.getElementById("image-history");
    if (!node) return;
    if (!rows.length) {
      node.innerHTML = empty("尚无图片任务", "草稿、完成任务和失败任务在此按权限显示。");
      return;
    }
    node.innerHTML = rows.map((row) => `
      <article class="assistant-history-item" data-run-id="${escapeHtml(row.run_id)}" ${row.run_id === currentRunId ? 'aria-current="true"' : ""}>
        <button class="assistant-history-open" type="button" data-run-id="${escapeHtml(row.run_id)}">
          <strong>${escapeHtml((row.request && row.request.prompt) || "文生图")}</strong>
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

  function renderEvents(run, events) {
    const node = document.getElementById("image-result");
    const terminal = TERMINAL.has(run.status);
    const tone = terminal ? (run.status === "COMPLETED" ? "good" : "bad") : "warn";
    const stages = (events || []).map((item) => `
      <li>
        <strong>${escapeHtml(stageLabel(item.stage))}</strong>
        <span>${escapeHtml(item.message || "")}</span>
      </li>`).join("");
    node.className = "image-stage-status";
    node.innerHTML = `
      <div class="image-status-row">
        <div>
          <h3>${escapeHtml(stageLabel(run.status))}</h3>
          <p>${escapeHtml(run.error_message || run.model_display_name || "真实图片产物将显示在上方画幅中。")}</p>
        </div>
        ${badge(run.status, tone)}
      </div>
      <ol class="image-status-rail">${stages || "<li><strong>已接受请求</strong><span>已接受文生图请求</span></li>"}</ol>`;
    setMeta([
      ["状态", stageLabel(run.status)],
      ["模型", run.model_display_name || "已绑定模型"],
      ["提示词版本", run.prompt_revision_id || "—"],
      ["资产访问", run.status === "COMPLETED" ? "当前 Domain 授权" : "待生成"],
    ]);
  }

  function emptyPreview(title, copy) {
    const node = document.getElementById("image-preview");
    node.className = "image-stage-frame is-empty";
    node.innerHTML = `<div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div>`;
  }

  async function renderPreview(run) {
    const node = document.getElementById("image-preview");
    revokePreviews();
    const assetIds = (run.result && run.result.asset_ids) || [];
    if (run.status !== "COMPLETED" || !assetIds.length) {
      emptyPreview(
        stageLabel(run.status),
        run.error_message || "图片生成成功后显示受控预览；二进制存对象存储，不保存为数据库 BLOB。",
      );
      return;
    }
    const images = [];
    for (const assetId of assetIds) {
      const blob = await KBotAssistantApi.requestBlob(`/media-assets/${assetId}/content`);
      const objectUrl = URL.createObjectURL(blob);
      previewUrls.push(objectUrl);
      images.push(`<figure class="image-print"><img alt="生成图片 ${escapeHtml(assetId)}" src="${objectUrl}"></figure>`);
    }
    node.className = assetIds.length > 1 ? "image-stage-frame is-grid" : "image-stage-frame";
    node.innerHTML = images.join("");
  }

  function stopPoll() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = 0;
    }
  }

  async function refreshHistory() {
    const rows = await KBotAssistantApi.json("/image-generations/runs", "GET");
    renderHistory(Array.isArray(rows) ? rows : []);
  }

  async function loadRun(runId) {
    const [run, events] = await Promise.all([
      KBotAssistantApi.json(`/image-generations/runs/${runId}`, "GET"),
      KBotAssistantApi.json(`/image-generations/runs/${runId}/events`, "GET"),
    ]);
    fillComposer(run);
    renderEvents(run, Array.isArray(events) ? events : []);
    await renderPreview(run);
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
        toast(error.message || "无法刷新生成进度", "error");
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
      toast(error.message || "无法打开图片生成 Run", "error");
    }
  }

  async function deleteRun(runId) {
    const confirmed = globalThis.confirm("删除后无法恢复。确认删除这条文生图记录？");
    if (!confirmed) return;
    try {
      await KBotAssistantApi.json(`/image-generations/runs/${runId}`, "DELETE");
      if (currentRunId === runId) resetComposer();
      else await refreshHistory();
      toast("已删除文生图记录");
    } catch (error) {
      toast(error.message || "无法删除文生图记录", "error");
    }
  }

  function resetComposer() {
    stopPoll();
    currentRunId = "";
    revokePreviews();
    document.getElementById("image-form")?.reset();
    const result = document.getElementById("image-result");
    result.className = "image-stage-status";
    result.innerHTML = `
      <div class="image-status-row">
        <div>
          <h3>尚未创建生成任务</h3>
          <p>真实运行时，这里显示已接受、生成中、内容限制拒绝、失败或图片已就绪；不以示例图代替真实结果。</p>
        </div>
      </div>`;
    emptyPreview("等待真实图片产物", "图片生成成功后显示受控预览；二进制存对象存储，不保存为数据库 BLOB。");
    setMeta([["状态", "未创建 Run"], ["模型", "等待管理员绑定"], ["提示词版本", "—"], ["资产访问", "待授权"]]);
    refreshHistory().catch((error) => toast(error.message || "无法刷新图片历史", "error"));
  }

  KBotAssistantShell.ready.then(async (access) => {
    if (!access) return;
    accessState = access;
    paintCapability();
    document.getElementById("image-new")?.addEventListener("click", resetComposer);
    document.getElementById("image-refresh")?.addEventListener("click", () => {
      refreshHistory().catch((error) => toast(error.message || "无法刷新图片历史", "error"));
    });
    document.getElementById("image-form")?.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!ready()) {
        toast("模型能力尚未验收，未创建图片生成任务。", "error");
        return;
      }
      const payload = {
        prompt: document.getElementById("image-prompt").value.trim(),
        aspect_ratio: document.getElementById("image-ratio").value,
        count: Number(document.getElementById("image-count").value || 1),
      };
      try {
        const run = await KBotAssistantApi.json("/image-generations/runs", "POST", payload, {
          headers: { "Idempotency-Key": KBotAssistantApi.requestId() },
        });
        currentRunId = run.run_id;
        await openRun(run.run_id);
      } catch (error) {
        toast(error.message || "创建文生图 Run 失败", "error");
      }
    });
    try {
      await refreshHistory();
    } catch (error) {
      toast(error.message || "无法加载图片历史", "error");
    }
  });
})();
