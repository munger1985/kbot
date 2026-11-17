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
    const body = document.getElementById("knowledge-base-list");
    const keyword = document.getElementById("knowledge-base-search").value.trim().toLocaleLowerCase();
    const visible = rows.filter((item) => !keyword || [item.display_name, item.description, item.knowledge_base_id].some((value) => String(value || "").toLocaleLowerCase().includes(keyword)));
    document.getElementById("knowledge-base-count").textContent = `${visible.length} 项`;
    const cards = visible.map((item) => {
      const usage = Number(item.usage?.assets || 0) + Number(item.usage?.agent_versions || 0);
      return `<article class="management-catalog-card knowledge-base-card${item.status === 'ACTIVE' ? '' : ' is-disabled'}">
        <header><div><p class="eyebrow">运维知识库</p><h3 title="${shell.escape(item.display_name)}">${shell.escape(item.display_name)}</h3></div><span class="status-pill ${item.status === 'ACTIVE' ? 'status-success' : 'status-muted'}">${item.status === 'ACTIVE' ? '启用' : '停用'}</span></header>
        <div class="aiops-catalog-card-body">
          <p class="knowledge-base-description" title="${shell.escape(item.description || "暂无说明")}">${shell.escape(item.description || "暂无说明")}</p>
          <small class="technical-meta knowledge-base-id" title="${shell.escape(item.knowledge_base_id)}">${shell.escape(item.knowledge_base_id)}</small>
          <dl><div><dt>运维资料</dt><dd>${Number(item.usage?.assets || 0)} 份</dd></div><div><dt>Agent 版本引用</dt><dd>${Number(item.usage?.agent_versions || 0)} 个</dd></div></dl>
          <p class="knowledge-base-updated">更新于 <time datetime="${shell.escape(item.updated_at)}">${shell.escape(shell.fmt(item.updated_at))}</time></p>
        </div>
        <footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions"><button class="row-action" type="button" data-edit="${shell.escape(item.knowledge_base_id)}" ${saving ? 'disabled' : ''}>编辑</button><button class="row-action" type="button" data-toggle-status="${shell.escape(item.knowledge_base_id)}" ${saving ? 'disabled' : ''}>${item.status === 'ACTIVE' ? '停用' : '启用'}</button></div><div class="management-catalog-overflow"><button class="row-action management-catalog-more" type="button" aria-haspopup="menu" aria-expanded="false"><span>更多</span><span aria-hidden="true">⌄</span></button><div class="management-catalog-menu" role="menu" hidden><button class="danger-row-action" type="button" data-delete="${shell.escape(item.knowledge_base_id)}" ${saving || usage ? 'disabled' : ''} title="${usage ? '仍有资料或 Agent 引用' : '删除知识库'}">删除</button></div></div></footer>
      </article>`;
    }).join("");
    body.innerHTML = `<div class="management-catalog-grid"><button id="create-knowledge-base" class="management-catalog-add-card" type="button"><span class="management-catalog-add-mark" aria-hidden="true">+</span><strong>新建知识库</strong><small>为运维资料创建独立的知识索引空间</small></button>${cards}</div>${visible.length ? '' : `<p class="empty-cell" role="status">${keyword ? '没有匹配的知识库，请调整搜索条件。' : '当前 Domain 还没有运维知识库。'}</p>`}`;
  }
  async function load() {
    const body = document.getElementById("knowledge-base-list");
    const refresh = document.getElementById("refresh-knowledge-bases");
    body.setAttribute("aria-busy", "true");
    refresh.disabled = true;
    refresh.textContent = "刷新中…";
    try { const page = await KBotAIOpsAuth.request(api); rows = page.items || []; }
    finally { render(); body.setAttribute("aria-busy", "false"); refresh.disabled = false; refresh.textContent = "刷新"; }
  }
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
    if (!await shell.confirmAction({ title: "删除知识库？", message: `确认删除知识库“${row.display_name}”？只有无资料且未被统一 Agent 引用的知识库可以删除。`, confirmLabel: "删除知识库", tone: "danger" })) return;
    try { await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(row.knowledge_base_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${row.row_version}"` } }); shell.toast("知识库已删除"); await load(); } catch (error) { shell.toast(error.message); }
  }
  async function toggleStatus(row) {
    if (saving) return;
    const active = row.status === 'ACTIVE';
    if (active && !await shell.confirmAction({ title: '停用运维知识库？', message: `停用“${row.display_name}”后，Agent 将不能检索该库，也不能向其上传新资料；已有资料、索引和绑定会保留。`, confirmLabel: '停用知识库', tone: 'warning' })) return;
    saving = true;
    render();
    try {
      await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(row.knowledge_base_id)}:${active ? 'disable' : 'enable'}`, {
        method: 'POST', body: JSON.stringify({ expected_row_version: row.row_version }),
      });
      shell.toast(active ? '运维知识库已停用' : '运维知识库已启用');
      await load();
    } catch (error) { shell.toast(error.message); }
    finally { saving = false; render(); }
  }
  function closeMenus() {
    document.querySelectorAll('#knowledge-base-list .management-catalog-menu').forEach(menu => { menu.hidden = true; });
    document.querySelectorAll('#knowledge-base-list .management-catalog-more').forEach(button => button.setAttribute('aria-expanded', 'false'));
  }
  shell.ready.then(async () => {
    document.getElementById("knowledge-base-search").oninput = render;
    document.getElementById("refresh-knowledge-bases").onclick = () => load().catch((error) => shell.toast(error.message));
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    document.getElementById("knowledge-base-form").onsubmit = save;
    document.getElementById("knowledge-base-form").oninput = updateSaveState;
    document.getElementById("knowledge-base-dialog").addEventListener("cancel", event => { if (saving) event.preventDefault(); });
    document.addEventListener('pointerdown', event => { if (!event.target.closest('#knowledge-base-list .management-catalog-overflow')) closeMenus(); });
    document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenus(); });
    document.getElementById("knowledge-base-list").onclick = (event) => {
      const more = event.target.closest('.management-catalog-more');
      if (more) {
        const menu = more.nextElementSibling;
        const open = menu.hidden;
        closeMenus();
        menu.hidden = !open;
        more.setAttribute('aria-expanded', String(open));
        return;
      }
      if (event.target.closest('.management-catalog-menu')) closeMenus();
      if (event.target.closest("#create-knowledge-base")) { void open().catch((error) => shell.toast(error.message)); return; }
      const toggle = event.target.closest('[data-toggle-status]');
      if (toggle && !toggle.disabled) { const row = rows.find(item => item.knowledge_base_id === toggle.dataset.toggleStatus); if (row) void toggleStatus(row); return; }
      const edit = event.target.closest("[data-edit]"); const deletion = event.target.closest("[data-delete]"); const id = edit?.dataset.edit || deletion?.dataset.delete; const row = rows.find((item) => item.knowledge_base_id === id); if (edit && row) void open(row).catch((error) => shell.toast(error.message)); if (deletion && row && !deletion.disabled) void remove(row);
    };
    try { await load(); } catch (error) { shell.toast(error.message); }
  });
})();
