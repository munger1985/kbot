(function () {
  "use strict";

  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  let collection = null;
  let modelPolicy = null;
  let models = [];

  function modelId(row) {
    return String(row.model_id || row.id || row.ai_model_id || "");
  }

  function modelName(row) {
    return row.display_name || row.served_model_name || modelId(row);
  }

  function modelById(id) {
    return models.find((row) => modelId(row) === String(id || ""));
  }

  function categoryName(category) {
    return { 1: "LLM", 2: "文本 Embedding", 3: "视觉 Embedding", 5: "VLM" }[Number(category)] || String(category || "—");
  }

  function showPageError(error, fallback) {
    const element = $("knowledge-core-page-error");
    const request = error?.requestId ? `；request_id: ${error.requestId}` : "";
    element.textContent = `${error?.message || fallback}${request}`;
    element.hidden = false;
    KBotKmShell.showError(error, fallback);
  }

  function clearPageError() {
    $("knowledge-core-page-error").hidden = true;
    $("knowledge-core-page-error").textContent = "";
  }

  function populateModelSelects() {
    document.querySelectorAll("[data-model-category]").forEach((select) => {
      const category = Number(select.dataset.modelCategory);
      const optional = category === 3 || category === 5;
      const options = models
        .filter((row) => Number(row.category) === category)
        .map((row) => `<option value="${KBotKmShell.escapeHtml(modelId(row))}">${KBotKmShell.escapeHtml(modelName(row))}</option>`)
        .join("");
      select.innerHTML = `${optional ? '<option value="">不启用</option>' : '<option value="">请选择模型</option>'}${options}`;
    });
  }

  function renderSummary() {
    const summary = $("knowledge-core-summary");
    const status = $("knowledge-core-status");
    const createButton = $("create-knowledge-core");
    const editButton = $("edit-knowledge-core-models");
    if (!collection) {
      status.innerHTML = KBotKmShell.badge("MISSING");
      summary.innerHTML = '<p class="km-help">当前 Domain 尚未创建 assets Collection。创建后才能新增数据来源并同步 Asset。</p>';
      createButton.hidden = false;
      editButton.hidden = true;
      KBotKmShell.renderEmpty($("knowledge-core-model-rows"), 4, "尚未配置模型");
      return;
    }
    createButton.hidden = true;
    editButton.hidden = false;
    const nextStatus = collection.status === "ACTIVE" ? "DISABLED" : "ACTIVE";
    const actionLabel = nextStatus === "ACTIVE" ? "启用" : "停用";
    status.innerHTML = `${KBotKmShell.badge(collection.status)} <button class="small${nextStatus === "DISABLED" ? " danger" : ""}" id="change-knowledge-core-status" data-status="${nextStatus}">${actionLabel}</button>`;
    summary.innerHTML = `<div class="km-grid"><div class="span-4"><span class="km-cell-sub">Collection ID</span><div><code>${KBotKmShell.escapeHtml(collection.collection_id)}</code></div></div><div class="span-4"><span class="km-cell-sub">默认安全级别</span><div>${KBotKmShell.escapeHtml(collection.default_security_level)}</div></div><div class="span-4"><span class="km-cell-sub">版本 / 更新时间</span><div>v${KBotKmShell.escapeHtml(collection.row_version)} · ${KBotKmShell.escapeHtml(KBotKmShell.formatDate(collection.updated_at))}</div></div><div class="span-12"><span class="km-cell-sub">描述</span><div>${KBotKmShell.escapeHtml(collection.description || "—")}</div></div></div>`;
    $("change-knowledge-core-status").addEventListener("click", changeStatus);
    renderModels();
  }

  function renderModels() {
    const roles = [
      ["parser_llm", "Parser LLM", 1, "预留角色；当前解析流水线未调用"],
      ["parser_vlm", "Parser VLM", 5, "图片与视觉页面解析，可更新"],
      ["embedding", "文本 Embedding", 2, modelPolicy?.embedding_change_allowed ? "当前允许更换" : "已有解析活动，不可更换"],
      ["visual_embedding", "视觉 Embedding", 3, modelPolicy?.visual_embedding_change_allowed ? "当前允许配置" : "已有解析活动，不可更换或移除"],
    ];
    $("knowledge-core-model-rows").innerHTML = roles.map(([role, label, category, rule]) => {
      const id = collection.models?.[role];
      const model = modelById(id);
      return `<tr><td><span class="km-cell-main">${label}</span><div class="km-cell-sub"><code>${role}</code></div></td><td>${id ? `<span class="km-cell-main">${KBotKmShell.escapeHtml(model ? modelName(model) : "模型目录中不可用")}</span><div class="km-cell-sub">${KBotKmShell.escapeHtml(KBotKmShell.shortId(id))}</div>` : "—"}</td><td>${categoryName(category)}</td><td>${rule}</td></tr>`;
    }).join("");
  }

  async function load() {
    clearPageError();
    try {
      const payload = await KBotKmApi.request(`${base}/knowledge-core`);
      collection = payload.collection || null;
      modelPolicy = payload.model_policy || null;
      renderSummary();
    } catch (error) {
      showPageError(error, "KM Knowledge Core 加载失败");
    }
  }

  function openCreate() {
    const form = $("knowledge-core-form");
    form.reset();
    populateModelSelects();
    form.elements.embedding.disabled = false;
    form.elements.parser_vlm.disabled = false;
    form.elements.visual_embedding.disabled = false;
    $("knowledge-core-embedding-help").textContent = "尚无 Asset 进入解析流程时可更换。";
    $("knowledge-core-visual-embedding-help").textContent = "尚无 Asset 进入解析流程时可更换；解析后仍可首次启用。";
    form.elements.mode.value = "create";
    form.elements.default_security_level.value = "1";
    form.elements.description.value = "KM Portal Asset 文档固定 Collection";
    $("knowledge-core-security-field").hidden = false;
    $("knowledge-core-description-field").hidden = false;
    $("knowledge-core-dialog-title").textContent = "创建 KM Knowledge Core";
    $("save-knowledge-core").textContent = "创建";
    KBotKmShell.openDialog("knowledge-core-dialog");
  }

  function openModels() {
    if (!collection) return;
    const form = $("knowledge-core-form");
    form.reset();
    populateModelSelects();
    form.elements.mode.value = "models";
    form.elements.expected_row_version.value = collection.row_version;
    form.elements.parser_llm.value = collection.models?.parser_llm || "";
    form.elements.parser_vlm.value = collection.models?.parser_vlm || "";
    form.elements.embedding.value = collection.models?.embedding || "";
    form.elements.embedding.disabled = !modelPolicy?.embedding_change_allowed;
    $("knowledge-core-embedding-help").textContent = form.elements.embedding.disabled
      ? "已有 Asset 进入解析流程，文本 Embedding 不可更换。"
      : "尚无 Asset 进入解析流程，可以更换。";
    form.elements.visual_embedding.value = collection.models?.visual_embedding || "";
    form.elements.visual_embedding.disabled = !modelPolicy?.visual_embedding_change_allowed;
    $("knowledge-core-visual-embedding-help").textContent = form.elements.visual_embedding.disabled
      ? "已有 Asset 进入解析流程，视觉 Embedding 不可更换或移除。"
      : "当前允许配置；解析后仍可首次启用。";
    $("knowledge-core-security-field").hidden = true;
    $("knowledge-core-description-field").hidden = true;
    $("knowledge-core-dialog-title").textContent = "配置 Knowledge Core 模型";
    $("save-knowledge-core").textContent = "保存模型";
    KBotKmShell.openDialog("knowledge-core-dialog");
  }

  async function save(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const values = Object.fromEntries(new FormData(form));
    const editing = values.mode === "models";
    const payload = {
      parser_llm: values.parser_llm,
      parser_vlm: values.parser_vlm || null,
      embedding: editing && form.elements.embedding.disabled ? collection.models.embedding : values.embedding,
      visual_embedding: editing && form.elements.visual_embedding.disabled ? (collection.models.visual_embedding || null) : (values.visual_embedding || null),
    };
    if (editing) payload.expected_row_version = Number(values.expected_row_version);
    else {
      payload.description = values.description || null;
      payload.default_security_level = Number(values.default_security_level);
    }
    const endpoint = editing
      ? `${base}/knowledge-core/models`
      : `${base}/knowledge-core`;
    KBotKmShell.setBusy($("save-knowledge-core"), true, editing ? "保存中…" : "创建中…");
    try {
      await KBotKmApi.json(endpoint, editing ? "PUT" : "POST", payload);
      form.elements.embedding.disabled = false;
      form.elements.parser_vlm.disabled = false;
      form.elements.visual_embedding.disabled = false;
      KBotKmShell.closeDialog("knowledge-core-dialog");
      KBotKmShell.toast(editing ? "Knowledge Core 模型已更新" : "Knowledge Core 已创建", "success");
      await load();
    } catch (error) {
      KBotKmShell.showError(error, editing ? "模型更新失败" : "Knowledge Core 创建失败");
      if (error?.status === 409) await load();
    } finally {
      KBotKmShell.setBusy($("save-knowledge-core"), false);
    }
  }

  async function changeStatus(event) {
    const button = event.currentTarget;
    const nextStatus = button.dataset.status;
    KBotKmShell.setBusy(button, true, nextStatus === "ACTIVE" ? "启用中…" : "停用中…");
    try {
      await KBotKmApi.json(`${base}/knowledge-core/status`, "PATCH", { status: nextStatus });
      KBotKmShell.toast(nextStatus === "ACTIVE" ? "Knowledge Core 已启用" : "Knowledge Core 已停用", "success");
      await load();
    } catch (error) {
      KBotKmShell.showError(error, "Knowledge Core 状态更新失败");
    } finally {
      KBotKmShell.setBusy(button, false);
    }
  }

  async function initialize() {
    try {
      models = KBotKmApi.items(await KBotKmApi.request(`${base}/model-catalog`));
      populateModelSelects();
      await load();
    } catch (error) {
      showPageError(error, "Knowledge Core 模型目录加载失败");
    }
  }

  window.addEventListener("DOMContentLoaded", () => {
    $("refresh-knowledge-core").addEventListener("click", load);
    $("create-knowledge-core").addEventListener("click", openCreate);
    $("edit-knowledge-core-models").addEventListener("click", openModels);
    $("knowledge-core-form").addEventListener("submit", save);
  });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
