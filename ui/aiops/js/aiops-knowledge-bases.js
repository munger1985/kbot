(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/operations-knowledge/knowledge-bases";
  const shell = globalThis.KBotAIOpsShell;
  let rows = [];
  let editing = null;
  let reviewingCapabilities = false;
  let modelCatalog = [];
  let editorGeneration = 0;
  const indexFields = [["embedding", 2, "不配置文本 Embedding"], ["visual_embedding", 3, "不配置视觉 Embedding"]];
  const modelFields = [["tagging_model_id", 1, "不配置标签模型"], ["ocr", 6, "不配置 OCR"], ["vlm", 5, "不配置 VLM"], ...indexFields];
  let saving = false;
  const capabilityLabels = { NOT_ENABLED: "未启用", PENDING_VERIFICATION: "已配置待验证", AVAILABLE: "可用", DEGRADED: "异常" };
  const indexLabels = { EMPTY: "暂无资料", NOT_READY: "索引待就绪", PARTIAL: "部分资料已就绪", READY: "索引就绪", ERROR: "索引异常" };
  const directionLabels = { TEXT_TO_TEXT: "文本搜文本", TEXT_TO_IMAGE: "文本搜图", IMAGE_TO_TEXT: "图搜文本", IMAGE_TO_IMAGE: "图搜图" };
  const taggingLabel = (state) => capabilityLabels[state] || "未配置";

  async function loadModelCatalog() {
    const catalog = await KBotAIOpsAuth.request("/api/v1/model-catalog");
    return Array.isArray(catalog) ? catalog : [];
  }
  function fillModelSelect(form, [name, category, emptyLabel], selected = "", catalogReady = false) {
    const select = form.elements[name];
    const models = modelCatalog.filter(item => Number(item.category) === category);
    select.replaceChildren(new Option(emptyLabel, ""));
    if (selected && !models.some((item) => String(item.model_id) === selected)) {
      select.add(new Option(catalogReady ? "当前绑定不在可选目录" : "保留当前绑定", selected));
    }
    for (const item of models) {
      const direction = name === "visual_embedding" ? globalThis.KBotKnowledgeCapabilitiesReview.textToImageLabel(item) : "";
      select.add(new Option([item.display_name, item.provider, direction].filter(Boolean).join(" · "), item.model_id));
    }
    select.value = selected;
    select.disabled = Boolean(editing?.index_models?.[name] && indexFields.some(([role]) => role === name));
  }
  function renderCapabilities(row) {
    const element = document.getElementById("knowledge-base-capabilities");
    const empty = document.getElementById("knowledge-base-capabilities-empty");
    const caps = row?.capabilities;
    element.hidden = !caps;
    empty.hidden = Boolean(caps);
    if (!caps) {
      element.replaceChildren();
      return;
    }
    const items = [["原生文档解析", capabilityLabels[caps.native_parse]], ["OCR 补强", capabilityLabels[caps.ocr]],
      ["视觉补强", capabilityLabels[caps.visual_enrichment]], ["词法检索", capabilityLabels[caps.lexical]],
      ["索引覆盖", indexLabels[row.index?.state]]];
    for (const { direction, state } of caps.semantic_directions || []) {
      items.push([directionLabels[direction] || "语义检索", capabilityLabels[state]]);
    }
    element.replaceChildren(...items.map(([name, value]) => {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = name;
      dd.textContent = value || "状态未知";
      const item = document.createElement("div");
      item.append(dt, dd);
      return item;
    }));
  }
  function updateSaveState() {
    const form = document.getElementById("knowledge-base-form");
    document.getElementById("save-knowledge-base").disabled = saving || !form.elements.display_name.value.trim();
    document.querySelectorAll('#knowledge-base-dialog [data-close]').forEach(button => { button.disabled = saving; });
    document.getElementById("back-knowledge-base").disabled = saving;
  }

  function renderReview(form, preview) {
    const restoreFocus = preview.contains(document.activeElement);
    preview.innerHTML = globalThis.KBotKnowledgeCapabilitiesReview.render(
      form.elements.display_name.value.trim(),
      { embedding: form.elements.embedding.value, visual_embedding: form.elements.visual_embedding.value },
      { ocr: form.elements.ocr.value, vlm: form.elements.vlm.value }, modelCatalog, shell.escape);
    if (restoreFocus) preview.querySelector("[data-knowledge-capability-title]").focus({ preventScroll: true });
  }

  function showStep(reviewing) {
    reviewingCapabilities = reviewing;
    const form = document.getElementById("knowledge-base-form");
    form.querySelectorAll(":scope > section.drawer-form-section").forEach(section => { section.hidden = reviewing; });
    const preview = document.getElementById("knowledge-base-review");
    const liveCapabilities = document.getElementById("knowledge-base-live-capabilities");
    preview.hidden = !reviewing;
    liveCapabilities.hidden = !reviewing || !editing;
    liveCapabilities.open = false;
    if (reviewing) renderReview(form, preview);
    else preview.replaceChildren();
    document.getElementById("knowledge-base-editor-step").textContent = reviewing
      ? "第 2 步，共 2 步 · 确认检索能力" : "第 1 步，共 2 步 · 填写信息与选择模型";
    document.getElementById("cancel-knowledge-base").hidden = reviewing;
    document.getElementById("back-knowledge-base").hidden = !reviewing;
    const submit = document.getElementById("save-knowledge-base");
    submit.textContent = !reviewing ? "下一步" : editing ? "保存知识库" : "确认创建";
    document.getElementById("knowledge-base-result").hidden = true;
    form.scrollTop = 0;
    if (reviewing) preview.querySelector("[data-knowledge-capability-title]").focus({ preventScroll: true });
    else if (form.closest("dialog").open) submit.focus({ preventScroll: true });
  }

  function render() {
    const body = document.getElementById("knowledge-base-list");
    const keyword = document.getElementById("knowledge-base-search").value.trim().toLocaleLowerCase();
    const visible = rows.filter((item) => !keyword || [item.display_name, item.description, item.knowledge_base_id].some((value) => String(value || "").toLocaleLowerCase().includes(keyword)));
    document.getElementById("knowledge-base-count").textContent = `${visible.length} 项`;
    const cards = visible.map((item) => {
      return `<article class="management-catalog-card knowledge-base-card${item.status === 'ACTIVE' ? '' : ' is-disabled'}">
        <header><div><p class="eyebrow">运维知识库</p><h3 title="${shell.escape(item.display_name)}">${shell.escape(item.display_name)}</h3></div><span class="status-pill ${item.status === 'ACTIVE' ? 'status-success' : 'status-muted'}">${item.status === 'ACTIVE' ? '启用' : '停用'}</span></header>
        <div class="aiops-catalog-card-body">
          <p class="knowledge-base-description" title="${shell.escape(item.description || "暂无说明")}">${shell.escape(item.description || "暂无说明")}</p>
          <small class="technical-meta knowledge-base-id" title="${shell.escape(item.knowledge_base_id)}">${shell.escape(item.knowledge_base_id)}</small>
          <dl><div><dt>运维资料</dt><dd>${Number(item.usage?.assets || 0)} 份</dd></div><div><dt>Agent 版本引用</dt><dd>当前 ${Number(item.usage?.current_agent_versions || 0)} · 历史 ${Number(item.usage?.historical_agent_versions || 0)}</dd></div></dl>
          <p class="knowledge-base-updated">更新于 <time datetime="${shell.escape(item.updated_at)}">${shell.escape(shell.fmt(item.updated_at))}</time></p>
        </div>
        <footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions"><button class="row-action" type="button" data-edit="${shell.escape(item.knowledge_base_id)}" ${saving ? 'disabled' : ''}>编辑</button><button class="row-action" type="button" data-toggle-status="${shell.escape(item.knowledge_base_id)}" ${saving ? 'disabled' : ''}>${item.status === 'ACTIVE' ? '停用' : '启用'}</button></div><div class="management-catalog-overflow"><button class="row-action management-catalog-more" type="button" aria-haspopup="menu" aria-expanded="false"><span>更多</span><span aria-hidden="true">⌄</span></button><div class="management-catalog-menu" role="menu" hidden><button class="danger-row-action" type="button" data-archive="${shell.escape(item.knowledge_base_id)}" ${saving ? 'disabled' : ''}>归档</button></div></div></footer>
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
    const generation = ++editorGeneration;
    editing = row;
    const dialog = document.getElementById("knowledge-base-dialog");
    const form = document.getElementById("knowledge-base-form"); form.reset();
    form.elements.display_name.value = row?.display_name || "";
    form.elements.description.value = row?.description || "";
    modelCatalog = [];
    for (const field of modelFields) fillModelSelect(form, field, (field[0] === "tagging_model_id" ? row?.tagging?.model_id : (row?.parse_models?.[field[0]] || row?.index_models?.[field[0]])) || "");
    renderCapabilities(row);
    const hint = document.getElementById("knowledge-base-tagging-hint");
    const status = row ? "当前状态：" + taggingLabel(row.tagging?.state) + "。" : "";
    hint.textContent = status + "可先上传资料；审批开始标签处理前需要有效模型和标签词表。";
    document.getElementById("knowledge-base-dialog-title").textContent = row ? "编辑运维知识库" : "新建运维知识库";
    showStep(false);
    document.getElementById("knowledge-base-result").textContent = "";
    document.getElementById("knowledge-base-result").hidden = true;
    updateSaveState();
    dialog.showModal();
    form.scrollTop = 0;
    try {
      const models = await loadModelCatalog();
      if (dialog.open && editorGeneration === generation) {
        modelCatalog = models;
        for (const field of modelFields) fillModelSelect(form, field, form.elements[field[0]].value, true);
        if (reviewingCapabilities) renderReview(form, document.getElementById("knowledge-base-review"));
      }
    } catch (error) {
      if (dialog.open && editorGeneration === generation) hint.textContent += " 模型目录暂不可用，仍可保存基本信息。";
    }
  }

  async function save(event) {
    event.preventDefault(); const form = event.currentTarget;
    updateSaveState();
    if (saving || document.getElementById("save-knowledge-base").disabled) return;
    if (!reviewingCapabilities) return showStep(true);
    saving = true;
    updateSaveState();
    const selectedModel = form.elements.tagging_model_id.value || null;
    const payload = { display_name: form.elements.display_name.value.trim(), description: form.elements.description.value.trim() || null };
    if (selectedModel !== (editing?.tagging?.model_id || null)) payload.tagging_model_id = selectedModel;
    const parseModels = { ocr: form.elements.ocr.value || null, vlm: form.elements.vlm.value || null };
    if (Object.entries(parseModels).some(([role, value]) => value !== (editing?.parse_models?.[role] || null))) payload.parse_models = parseModels;
    const indexModels = Object.fromEntries(indexFields.map(([role]) => [role, form.elements[role].value || null]));
    if (Object.entries(indexModels).some(([role, value]) => value !== (editing?.index_models?.[role] || null))) payload.index_models = indexModels;
    if (editing) payload.expected_row_version = editing.row_version;
    try { await KBotAIOpsAuth.request(editing ? `${api}/${encodeURIComponent(editing.knowledge_base_id)}` : api, { method: editing ? "PUT" : "POST", body: JSON.stringify(payload) }); document.getElementById("knowledge-base-dialog").close(); shell.toast(editing ? "知识库已更新" : "知识库已创建"); await load(); }
    catch (error) { document.getElementById("knowledge-base-result").textContent = error.message; document.getElementById("knowledge-base-result").hidden = false; }
    finally { saving = false; updateSaveState(); render(); }
  }
  async function archive(row) {
    if (saving) return;
    if (!await shell.confirmAction({ title: "归档运维知识库？", message: `归档“${row.display_name}”后，它将从知识库目录移除，不能再上传新资料或供 Agent 检索。已有资料、索引和 Agent 版本绑定会保留。`, confirmLabel: "归档知识库", tone: "danger" })) return;
    if (saving) return;
    saving = true;
    render();
    try { await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(row.knowledge_base_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${row.row_version}"` } }); shell.toast("知识库已归档"); await load(); }
    catch (error) {
      if (error.status === 404 || error.status === 412) {
        await load().catch(refreshError => shell.toast(refreshError.message));
        shell.toast(error.status === 404 ? "知识库已不可用，目录已刷新。" : "知识库版本已变化，未归档，请重新确认。");
      } else shell.toast(error.message);
    } finally { saving = false; render(); }
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
    document.getElementById("back-knowledge-base").onclick = () => showStep(false);
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
      const edit = event.target.closest("[data-edit]"); const archiveButton = event.target.closest("[data-archive]"); const id = edit?.dataset.edit || archiveButton?.dataset.archive; const row = rows.find((item) => item.knowledge_base_id === id); if (edit && row) void open(row).catch((error) => shell.toast(error.message)); if (archiveButton && row && !archiveButton.disabled) void archive(row);
    };
    try { await load(); } catch (error) { shell.toast(error.message); }
  });
})();
