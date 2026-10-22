(function () {
  "use strict";

  const bindingApi = "/api/v1/apps/aiops/operations-knowledge/knowledge-answering/binding";
  const knowledgeBasesApi = "/api/v1/apps/aiops/operations-knowledge/knowledge-bases";
  const shell = globalThis.KBotAIOpsShell;
  let binding = null;
  let knowledgeBases = [];
  let models = [];

  function renderModelOptions() {
    const select = document.querySelector('[name="composer_model_id"]');
    const llms = models.filter((item) => Number(item.category) === 1 && item.status === "ACTIVE");
    const selected = binding?.composer_model_id || "";
    const selectedAvailable = !selected || llms.some((item) => String(item.model_id) === String(selected));
    const missing = selectedAvailable ? "" : '<option value="" selected>当前绑定模型已不可用，请重新选择</option>';
    const options = llms.map((item) => `<option value="${shell.escape(item.model_id)}" ${String(item.model_id) === String(selected) ? "selected" : ""}>${shell.escape(item.display_name)} · ${shell.escape(item.provider || "未标注 Provider")}</option>`).join("");
    select.innerHTML = `${missing}<option value="">请选择回答模型</option>${options}`;
  }

  function renderKnowledgeBases() {
    const selected = new Set(binding?.knowledge_base_ids || []);
    const host = document.getElementById("knowledge-assistant-bases");
    host.innerHTML = knowledgeBases.length
      ? knowledgeBases.map((item) => `<label class="assistant-base-card"><input type="checkbox" name="knowledge_base_ids" value="${shell.escape(item.knowledge_base_id)}" ${selected.has(item.knowledge_base_id) ? "checked" : ""}><span><strong>${shell.escape(item.display_name)}</strong><small>${shell.escape(item.description || "独立运维知识索引空间")}</small></span></label>`).join("")
      : '<p class="ops-empty">当前 Domain 还没有运维知识库，请先创建知识库。</p>';
  }

  function renderStatus() {
    const status = document.getElementById("assistant-config-status");
    status.className = `ops-badge ${binding?.configured ? "good" : "warn"}`;
    status.textContent = binding?.configured ? "已配置" : "待配置";
    document.getElementById("assistant-config-meta").textContent = binding?.updated_at
      ? `更新于 ${shell.fmt(binding.updated_at)} · 配置版本 ${binding.row_version}`
      : "尚未保存 Domain 配置";
  }

  async function load() {
    const [bindingResult, knowledgeBasePage, modelCatalog] = await Promise.all([
      KBotAIOpsAuth.request(bindingApi),
      KBotAIOpsAuth.request(knowledgeBasesApi),
      KBotAIOpsAuth.request("/api/v1/model-catalog"),
    ]);
    binding = bindingResult;
    knowledgeBases = knowledgeBasePage.items || [];
    models = Array.isArray(modelCatalog) ? modelCatalog : [];
    renderModelOptions();
    renderKnowledgeBases();
    renderStatus();
  }

  async function save(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const knowledgeBaseIds = [...form.querySelectorAll('[name="knowledge_base_ids"]:checked')].map((item) => item.value);
    const result = document.getElementById("knowledge-assistant-result");
    if (!knowledgeBaseIds.length) { result.textContent = "请至少选择一个运维知识库"; return; }
    const button = document.getElementById("save-knowledge-assistant");
    button.disabled = true;
    result.textContent = "正在保存…";
    const payload = {
      composer_model_id: form.elements.composer_model_id.value,
      knowledge_base_ids: knowledgeBaseIds,
    };
    if (binding?.row_version != null) payload.expected_row_version = binding.row_version;
    try {
      binding = await KBotAIOpsAuth.request(bindingApi, { method: "PUT", body: JSON.stringify(payload) });
      result.textContent = "";
      renderStatus();
      shell.toast("知识问答 Agent 配置已保存");
    } catch (error) {
      result.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  shell.ready.then(async () => {
    document.getElementById("knowledge-assistant-form").onsubmit = save;
    try { await load(); } catch (error) { shell.toast(error.message); }
  });
})();
