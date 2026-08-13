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
    if (!agents.length) return KBotKmShell.renderEmpty(body, 7, "尚未创建 KM Agent");
    body.innerHTML = agents.map((row, index) => `<tr><td><span class="km-cell-main">${KBotKmShell.escapeHtml(row.display_name)}</span><div class="km-cell-sub">${KBotKmShell.escapeHtml(KBotKmShell.shortId(row.agent_id))}</div></td><td>${KBotKmShell.escapeHtml(sourceName(row.source_id))}</td><td>${KBotKmShell.badge(row.status)}</td><td>${KBotKmShell.escapeHtml(`${Object.keys(row.models || {}).length} 项`)}</td><td>${row.do_rerank ? "启用" : "关闭"}</td><td>${row.row_version}</td><td>${row.status === "ACTIVE" ? '<a class="km-button small" href="./chat.html">开始问答</a>' : `<button class="small" data-activate="${index}">激活</button>`}</td></tr>`).join("");
  }
  async function save(event) {
    event.preventDefault(); const form = event.currentTarget; const values = Object.fromEntries(new FormData(form)); const modelRoles = ["context_llm", "composer_llm", "memory_llm", "memory_embedding", "router_llm", "data_planner_llm"]; const selected = {};
    modelRoles.forEach((role) => { if (values[role]) selected[role] = values[role]; });
    const payload = { source_id: values.source_id, display_name: values.display_name, description: values.description || null, instruction: values.instruction || null, models: selected, do_rerank: form.elements.do_rerank.checked, status: values.status };
    KBotKmShell.setBusy($("save-agent"), true, "创建中…");
    try { await KBotKmApi.json(`${base}/agents`, "POST", payload); form.reset(); populateModels(); KBotKmShell.closeDialog("agent-dialog"); KBotKmShell.toast("KM Agent 已创建", "success"); await load(); }
    catch (error) { KBotKmShell.showError(error, "KM Agent 创建失败"); }
    finally { KBotKmShell.setBusy($("save-agent"), false); }
  }
  async function activate(index, button) { const row = agents[index]; KBotKmShell.setBusy(button, true); try { await KBotKmApi.json(`${base}/agents/${row.agent_id}/activate`, "POST", { expected_row_version: row.row_version }); KBotKmShell.toast("KM Agent 已激活", "success"); await load(); } catch (error) { KBotKmShell.showError(error, "Agent 激活失败"); } finally { KBotKmShell.setBusy(button, false); } }
  window.addEventListener("DOMContentLoaded", () => { $("new-agent").addEventListener("click", () => KBotKmShell.openDialog("agent-dialog")); $("refresh-agents").addEventListener("click", load); $("agent-form").addEventListener("submit", save); $("agent-rows").addEventListener("click", (event) => { const button = event.target.closest("[data-activate]"); if (button) activate(Number(button.dataset.activate), button); }); });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
