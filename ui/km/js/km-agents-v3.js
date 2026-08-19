/* 独立物理文件名用于绕过忽略查询参数的旧代理缓存。 */
(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  let agents = [], sources = [], models = [];

  function showPageError(error, fallback) {
    const element = $("agent-page-error");
    const path = error?.path ? `；接口: ${error.path}` : "";
    const code = error?.code ? `；错误码: ${error.code}` : "";
    const request = error?.requestId ? `；request_id: ${error.requestId}` : "";
    element.textContent = `${error?.message || fallback}${path}${code}${request}`;
    element.hidden = false;
    KBotKmShell.showError(error, fallback);
  }

  function clearPageError() {
    $("agent-page-error").hidden = true;
    $("agent-page-error").textContent = "";
  }

  async function initialize() {
    clearPageError();
    try {
      const [sourcePayload, modelPayload] = await Promise.all([
        KBotKmApi.request(`${base}/sources`),
        KBotKmApi.request(`${base}/model-catalog`),
      ]);
      sources = KBotKmApi.items(sourcePayload); models = KBotKmApi.items(modelPayload);
      $("agent-source").innerHTML = sources.map((row) => `<option value="${KBotKmShell.escapeHtml(row.source_id)}">${KBotKmShell.escapeHtml(row.display_name)}</option>`).join("");
      populateModels(); await load();
    } catch (error) { showPageError(error, "Agent 配置依赖加载失败"); }
  }
  function modelCategory(row) { return String(row.category ?? row.model_type ?? row.model_category ?? "").toUpperCase(); }
  function isEmbedding(row) { return ["2", "TXT_EMBEDDING", "EMBEDDING"].includes(modelCategory(row)); }
  function isLlm(row) { return ["1", "LLM"].includes(modelCategory(row)); }
  function modelId(row) { return row.model_id || row.id || row.ai_model_id; }
  function modelName(row) { return row.display_name || row.model_name || row.name || modelId(row); }
  function populateModels() {
    document.querySelectorAll("[data-model-kind]").forEach((select) => {
      const optional = select.name === "data_planner_llm";
      const filtered = models.filter((row) => select.dataset.modelKind === "embedding" ? isEmbedding(row) : isLlm(row));
      select.innerHTML = `${optional ? '<option value="">跟随 Composer</option>' : '<option value="">请选择模型</option>'}${filtered.map((row) => `<option value="${KBotKmShell.escapeHtml(modelId(row))}">${KBotKmShell.escapeHtml(modelName(row))}</option>`).join("")}`;
    });
  }
  function sourceName(id) { return sources.find((row) => String(row.source_id) === String(id))?.display_name || KBotKmShell.shortId(id); }
  async function load() {
    try { agents = KBotKmApi.items(await KBotKmApi.request(`${base}/agents`)); render(); }
    catch (error) { showPageError(error, "KM Agent 加载失败"); }
  }
  function render() {
    const body = $("agent-rows");
    if (!agents.length) return KBotKmShell.renderEmpty(body, 6, "尚未创建 KM Agent");
    body.innerHTML = agents.map((row, index) => `<tr><td><span class="km-cell-main">${KBotKmShell.escapeHtml(row.display_name)}</span><div class="km-cell-sub">${KBotKmShell.escapeHtml(KBotKmShell.shortId(row.agent_id))}</div></td><td>${KBotKmShell.escapeHtml(sourceName(row.source_id))}</td><td>${KBotKmShell.badge(row.status)}</td><td>${KBotKmShell.escapeHtml(`${Object.keys(row.models || {}).length} 项`)}</td><td>${row.row_version}</td><td><div class="km-actions"><button class="small" data-edit="${index}">编辑</button>${row.status === "ACTIVE" ? '<a class="km-button small" href="./chat.html">开始问答</a>' : `<button class="small" data-activate="${index}">激活</button>`}</div></td></tr>`).join("");
  }
  function openCreate() {
    const form = $("agent-form");
    form.reset();
    form.elements.agent_id.value = "";
    form.elements.expected_row_version.value = "";
    populateModels();
    $("agent-dialog-title").textContent = "创建 KM Agent";
    $("agent-status-field").hidden = false;
    $("save-agent").textContent = "创建";
    KBotKmShell.openDialog("agent-dialog");
  }
  function openEdit(index) {
    const row = agents[index];
    const form = $("agent-form");
    form.reset();
    populateModels();
    form.elements.agent_id.value = row.agent_id;
    form.elements.expected_row_version.value = row.row_version;
    form.elements.source_id.value = row.source_id;
    form.elements.display_name.value = row.display_name || "";
    form.elements.description.value = row.description || "";
    form.elements.instruction.value = row.instruction || "";
    Object.entries(row.models || {}).forEach(([role, modelIdValue]) => {
      if (form.elements[role]) form.elements[role].value = modelIdValue;
    });
    $("agent-dialog-title").textContent = "编辑 KM Agent";
    $("agent-status-field").hidden = true;
    $("save-agent").textContent = "保存修改";
    KBotKmShell.openDialog("agent-dialog");
  }
  async function save(event) {
    event.preventDefault(); const form = event.currentTarget; const values = Object.fromEntries(new FormData(form)); const modelRoles = ["context_llm", "composer_llm", "memory_llm", "memory_embedding", "router_llm", "data_planner_llm"]; const selected = {};
    modelRoles.forEach((role) => { if (values[role]) selected[role] = values[role]; });
    const editing = Boolean(values.agent_id);
    const payload = { source_id: values.source_id, display_name: values.display_name, description: values.description || null, instruction: values.instruction || null, models: selected };
    if (editing) payload.expected_row_version = Number(values.expected_row_version);
    else payload.status = values.status;
    KBotKmShell.setBusy($("save-agent"), true, editing ? "保存中…" : "创建中…");
    try { await KBotKmApi.json(editing ? `${base}/agents/${values.agent_id}` : `${base}/agents`, editing ? "PATCH" : "POST", payload); form.reset(); populateModels(); KBotKmShell.closeDialog("agent-dialog"); KBotKmShell.toast(editing ? "KM Agent 已更新" : "KM Agent 已创建", "success"); await load(); }
    catch (error) { KBotKmShell.showError(error, editing ? "KM Agent 更新失败" : "KM Agent 创建失败"); }
    finally { KBotKmShell.setBusy($("save-agent"), false); }
  }
  async function activate(index, button) { const row = agents[index]; KBotKmShell.setBusy(button, true); try { await KBotKmApi.json(`${base}/agents/${row.agent_id}/activate`, "POST", { expected_row_version: row.row_version }); KBotKmShell.toast("KM Agent 已激活", "success"); await load(); } catch (error) { KBotKmShell.showError(error, "Agent 激活失败"); } finally { KBotKmShell.setBusy(button, false); } }
  window.addEventListener("DOMContentLoaded", () => { $("new-agent").addEventListener("click", openCreate); $("refresh-agents").addEventListener("click", load); $("agent-form").addEventListener("submit", save); $("agent-rows").addEventListener("click", (event) => { const editButton = event.target.closest("[data-edit]"); if (editButton) { openEdit(Number(editButton.dataset.edit)); return; } const activateButton = event.target.closest("[data-activate]"); if (activateButton) activate(Number(activateButton.dataset.activate), activateButton); }); });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
