(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/operations-knowledge/knowledge-bases";
  const shell = globalThis.KBotAIOpsShell;
  let rows = [];
  let editing = null;
  let modelCatalog = [];
  let saving = false;
  const modelRoles = {
    parser_ocr: [6, "请选择 OCR 模型"],
    parser_vlm: [5, "不启用视觉解析"],
    parser_llm: [1, "请选择 LLM 解析模型"],
    retrieval_llm: [1, "请选择检索规划模型"],
    embedding: [2, "请选择文本向量模型"],
    visual_embedding: [3, "不启用视觉向量"],
  };

  async function loadModelCatalog() {
    if (!modelCatalog.length) {
      const catalog = await KBotAIOpsAuth.request("/api/v1/model-catalog");
      modelCatalog = Array.isArray(catalog) ? catalog : [];
    }
  }
  function fillModelSelect(form, role, selected = "") {
    const [category, emptyLabel] = modelRoles[role];
    const options = modelCatalog
      .filter((item) => Number(item.category) === category)
      .map((item) => `<option value="${shell.escape(item.model_id)}" ${String(item.model_id) === String(selected) ? "selected" : ""}>${shell.escape(item.display_name)} · ${shell.escape(item.provider)}</option>`)
      .join("");
    const selectionAvailable = !selected || modelCatalog.some((item) => String(item.model_id) === String(selected) && Number(item.category) === category);
    const missing = selectionAvailable ? "" : `<option value="${shell.escape(selected)}" selected disabled>当前绑定模型已不可用</option>`;
    form.elements[role].innerHTML = `${missing}<option value="">${shell.escape(emptyLabel)}</option>${options}`;
  }
  function updateSaveState() {
    const form = document.getElementById("knowledge-base-form");
    const validModels = Object.entries(modelRoles).every(([role, [category]]) => {
      const selected = form.elements[role].value;
      return !form.elements[role].required && !selected || modelCatalog.some(item => String(item.model_id) === selected && Number(item.category) === category);
    });
    document.getElementById("save-knowledge-base").disabled = saving || !form.elements.display_name.value.trim() || !validModels;
    document.querySelectorAll('#knowledge-base-dialog [data-close]').forEach(button => { button.disabled = saving; });
  }

  function render() {
    const body = document.getElementById("knowledge-base-rows");
    const keyword = document.getElementById("knowledge-base-search").value.trim().toLocaleLowerCase();
    const visible = rows.filter((item) => !keyword || [item.display_name, item.description].some((value) => String(value || "").toLocaleLowerCase().includes(keyword)));
    document.getElementById("knowledge-base-count").textContent = `${visible.length} 项`;
    body.innerHTML = visible.length ? visible.map((item) => {
      const usage = Number(item.usage?.assets || 0) + Number(item.usage?.agent_versions || 0);
      return `<tr><td><strong>${shell.escape(item.display_name)}</strong><small class="table-secondary">${shell.escape(item.description || "暂无说明")}</small><small class="table-secondary technical-value">${shell.escape(item.knowledge_base_id)}</small></td><td>${Number(item.usage?.assets || 0)} 份资料 · ${Number(item.usage?.agent_versions || 0)} 个当前 Agent 版本</td><td>${shell.escape(shell.fmt(item.updated_at))}</td><td><span class="status-pill status-success">启用</span></td><td><button class="row-action" type="button" data-edit="${shell.escape(item.knowledge_base_id)}">编辑</button><button class="danger-row-action" type="button" data-delete="${shell.escape(item.knowledge_base_id)}" ${usage ? 'disabled title="仍有资料或 Agent 引用"' : 'title="删除知识库"'}>删除</button></td></tr>`;
    }).join("") : '<tr><td colspan="5" class="empty-cell">当前 Domain 还没有运维知识库。</td></tr>';
  }
  async function load() { const page = await KBotAIOpsAuth.request(api); rows = page.items || []; render(); }
  async function open(row = null) {
    editing = row;
    const form = document.getElementById("knowledge-base-form"); form.reset();
    const configured = row
      ? await Promise.all([
        loadModelCatalog(),
        KBotAIOpsAuth.request(`${api}/${encodeURIComponent(row.knowledge_base_id)}/models`),
      ]).then(([, result]) => result)
      : await loadModelCatalog().then(() => ({ models: {} }));
    form.elements.display_name.value = row?.display_name || ""; form.elements.description.value = row?.description || "";
    Object.keys(modelRoles).forEach((role) => fillModelSelect(form, role, configured.models?.[role] || ""));
    document.getElementById("knowledge-base-dialog-title").textContent = row ? "编辑运维知识库" : "新建运维知识库";
    document.getElementById("save-knowledge-base").textContent = "保存知识库";
    document.getElementById("knowledge-base-result").textContent = "";
    document.getElementById("knowledge-base-result").hidden = true;
    updateSaveState();
    document.getElementById("knowledge-base-dialog").showModal();
  }
  async function save(event) {
    event.preventDefault(); const form = event.currentTarget;
    updateSaveState();
    if (saving || document.getElementById("save-knowledge-base").disabled) return;
    saving = true;
    updateSaveState();
    const models = Object.fromEntries(Object.keys(modelRoles)
      .map((role) => [role, form.elements[role].value])
      .filter(([, value]) => Boolean(value)));
    const payload = { display_name: form.elements.display_name.value.trim(), description: form.elements.description.value.trim() || null, models };
    if (editing) payload.expected_row_version = editing.row_version;
    try { await KBotAIOpsAuth.request(editing ? `${api}/${encodeURIComponent(editing.knowledge_base_id)}` : api, { method: editing ? "PUT" : "POST", body: JSON.stringify(payload) }); document.getElementById("knowledge-base-dialog").close(); shell.toast(editing ? "知识库已更新" : "知识库已创建"); await load(); }
    catch (error) { document.getElementById("knowledge-base-result").textContent = error.message; document.getElementById("knowledge-base-result").hidden = false; }
    finally { saving = false; updateSaveState(); }
  }
  async function remove(row) {
    if (!confirm(`确认删除知识库“${row.display_name}”？只有无资料且未被统一 Agent 引用的知识库可以删除。`)) return;
    try { await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(row.knowledge_base_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${row.row_version}"` } }); shell.toast("知识库已删除"); await load(); } catch (error) { shell.toast(error.message); }
  }
  shell.ready.then(async () => {
    document.getElementById("create-knowledge-base").onclick = () => open().catch((error) => shell.toast(error.message));
    document.getElementById("knowledge-base-search").oninput = render;
    document.getElementById("refresh-knowledge-bases").onclick = () => load().catch((error) => shell.toast(error.message));
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    document.getElementById("knowledge-base-form").onsubmit = save;
    document.getElementById("knowledge-base-form").oninput = updateSaveState;
    document.getElementById("knowledge-base-dialog").addEventListener("cancel", event => { if (saving) event.preventDefault(); });
    document.getElementById("knowledge-base-rows").onclick = (event) => { const edit = event.target.closest("[data-edit]"); const deletion = event.target.closest("[data-delete]"); const id = edit?.dataset.edit || deletion?.dataset.delete; const row = rows.find((item) => item.knowledge_base_id === id); if (edit && row) void open(row).catch((error) => shell.toast(error.message)); if (deletion && row) void remove(row); };
    try { await load(); } catch (error) { shell.toast(error.message); }
  });
})();
