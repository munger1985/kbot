/* 文生图：未就绪时不创建 Run；预览必须经 requestBlob 生成对象 URL。 */
(function () {
  "use strict";

  const { badge, capability, capabilityLabel, escapeHtml, toast } = KBotAssistantShell;
  const TERMINAL = new Set(["COMPLETED", "FAILED", "REJECTED"]);
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

  function renderHistory(rows) {
    const node = document.getElementById("image-history");
    if (!node) return;
    if (!rows.length) {
      node.innerHTML = empty("尚无图片任务", "草稿、完成任务和失败任务在此按权限显示。");
      return;
    }
    node.innerHTML = rows.map((row) => `
      <button class="assistant-history-item" type="button" data-run-id="${escapeHtml(row.run_id)}" ${row.run_id === currentRunId ? 'aria-current="true"' : ""}>
        <strong>${escapeHtml((row.request && row.request.prompt) || "文生图")}</strong>
        <small>${escapeHtml(row.status)} · ${escapeHtml(row.created_at || "")}</small>
      </button>`).join("");
    node.querySelectorAll("[data-run-id]").forEach((button) => {
      button.addEventListener("click", () => openRun(button.dataset.runId));
    });
  }

  function renderEvents(run, events) {
    const node = document.getElementById("image-result");
    const stages = (events || []).map((item) => `<li><strong>${escapeHtml(item.stage)}</strong><span>${escapeHtml(item.message || "")}</span></li>`).join("");
    node.innerHTML = `
      <h3>${escapeHtml(run.status)}</h3>
      <p>${escapeHtml(run.error_message || run.model_display_name || "正在生成图片")}</p>
      <ol class="assistant-flow">${stages || "<li><strong>ACCEPTED</strong><span>已接受请求</span></li>"}</ol>
      ${badge(run.status, TERMINAL.has(run.status) ? (run.status === "COMPLETED" ? "good" : "bad") : "warn")}`;
    setMeta([
      ["状态", run.status],
      ["模型", run.model_display_name || "已绑定模型"],
      ["提示词版本", run.prompt_revision_id || "—"],
      ["资产访问", run.status === "COMPLETED" ? "当前 Domain 授权" : "待生成"],
    ]);
  }

  async function renderPreview(run) {
    const node = document.getElementById("image-preview");
    revokePreviews();
    const assetIds = (run.result && run.result.asset_ids) || [];
    if (run.status !== "COMPLETED" || !assetIds.length) {
      node.innerHTML = `<div><strong>等待真实图片产物</strong><p>${escapeHtml(run.error_message || "图片生成成功后显示受控预览；二进制存对象存储，不保存为数据库 BLOB。")}</p></div>`;
      return;
    }
    const images = [];
    for (const assetId of assetIds) {
      const blob = await KBotAssistantApi.requestBlob(`/media-assets/${assetId}/content`);
      const objectUrl = URL.createObjectURL(blob);
      previewUrls.push(objectUrl);
      images.push(`<img alt="生成图片 ${escapeHtml(assetId)}" src="${objectUrl}">`);
    }
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

  function resetComposer() {
    stopPoll();
    currentRunId = "";
    revokePreviews();
    document.getElementById("image-form")?.reset();
    document.getElementById("image-result").innerHTML = `<h3>尚未创建生成任务</h3><p>真实运行时，这里将显示任务已接受、生成中、内容限制拒绝、失败或图片资产已就绪；不以示例图片代替真实结果。</p>`;
    document.getElementById("image-preview").innerHTML = `<div><strong>等待真实图片产物</strong><p>图片生成成功后显示受控预览；二进制存对象存储，不保存为数据库 BLOB。</p></div>`;
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
