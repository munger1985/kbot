(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/operations-knowledge/knowledge-bases";
  const shell = globalThis.KBotAIOpsShell;
  let rows = [];
  let editing = null;
  let modelCatalog = [];
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
    const missing = selectionAvailable ? "" : '<option value="" selected>当前绑定模型已不可用，请重新选择</option>';
    form.elements[role].innerHTML = `${missing}<option value="">${shell.escape(emptyLabel)}</option>${options}`;
  }

  function render() {
    const body = document.getElementById("knowledge-base-rows");
    body.innerHTML = rows.length ? rows.map((item) => `<tr>
      <td><strong>${shell.escape(item.display_name)}</strong><small>${shell.escape(item.description || "未填写说明")}</small></td>
      <td>${shell.badge(item.status)}</td><td>${Number(item.usage?.assets || 0)}</td><td>诊断 ${Number(item.usage?.agent_versions || 0)} · 问答 ${Number(item.usage?.knowledge_assistants || 0)}</td><td>${shell.escape(shell.fmt(item.updated_at))}</td>
      <td><div class="ops-actions"><a href="./operations-knowledge.html?knowledge_base_id=${encodeURIComponent(item.knowledge_base_id)}">管理资料</a><button type="button" data-edit="${shell.escape(item.knowledge_base_id)}">编辑</button><button class="danger" type="button" data-delete="${shell.escape(item.knowledge_base_id)}">删除</button></div></td>
    </tr>`).join("") : '<tr><td colspan="6" class="ops-empty">当前 Domain 还没有运维知识库。</td></tr>';
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
    document.getElementById("save-knowledge-base").textContent = row ? "保存修改" : "创建知识库";
    document.getElementById("knowledge-base-result").textContent = "";
    document.getElementById("knowledge-base-dialog").showModal();
  }
  async function save(event) {
    event.preventDefault(); const form = event.currentTarget; const button = document.getElementById("save-knowledge-base"); button.disabled = true;
    const models = Object.fromEntries(Object.keys(modelRoles)
      .map((role) => [role, form.elements[role].value])
      .filter(([, value]) => Boolean(value)));
    const payload = { display_name: form.elements.display_name.value.trim(), description: form.elements.description.value.trim() || null, models };
    if (editing) payload.expected_row_version = editing.row_version;
    try { await KBotAIOpsAuth.request(editing ? `${api}/${encodeURIComponent(editing.knowledge_base_id)}` : api, { method: editing ? "PUT" : "POST", body: JSON.stringify(payload) }); document.getElementById("knowledge-base-dialog").close(); shell.toast(editing ? "知识库已更新" : "知识库已创建"); await load(); }
    catch (error) { document.getElementById("knowledge-base-result").textContent = error.message; }
    finally { button.disabled = false; }
  }
  async function remove(row) {
    if (!confirm(`确认删除知识库“${row.display_name}”？只有无资料、无诊断 Agent 和无知识问答 Agent 引用的知识库可以删除。`)) return;
    try { await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(row.knowledge_base_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${row.row_version}"` } }); shell.toast("知识库已删除"); await load(); } catch (error) { shell.toast(error.message); }
  }
  shell.ready.then(async () => {
    document.getElementById("create-knowledge-base").onclick = () => open().catch((error) => shell.toast(error.message));
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    document.getElementById("knowledge-base-form").onsubmit = save;
    document.getElementById("knowledge-base-rows").onclick = (event) => { const edit = event.target.closest("[data-edit]"); const deletion = event.target.closest("[data-delete]"); const id = edit?.dataset.edit || deletion?.dataset.delete; const row = rows.find((item) => item.knowledge_base_id === id); if (edit && row) void open(row).catch((error) => shell.toast(error.message)); if (deletion && row) void remove(row); };
    try { await load(); } catch (error) { shell.toast(error.message); }
  });
})();
