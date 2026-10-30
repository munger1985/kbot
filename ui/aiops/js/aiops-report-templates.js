(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const escape = shell.escape;
  const sectionLabels = { EXECUTIVE_SUMMARY: "执行摘要", SCOPE: "诊断范围", RISK_OVERVIEW: "风险概览", TREND: "趋势分析", FINDINGS: "诊断发现", ROOT_CAUSE: "根因分析", RECOMMENDATIONS: "建议与行动", ACTIONS: "处置记录", EVIDENCE_BOUNDARY: "证据边界", EVIDENCE_APPENDIX: "证据附录" };
  const requiredSections = new Set(["EXECUTIVE_SUMMARY", "EVIDENCE_BOUNDARY"]);
  const defaultSections = ["EXECUTIVE_SUMMARY", "SCOPE", "FINDINGS", "ROOT_CAUSE", "RECOMMENDATIONS", "EVIDENCE_BOUNDARY"];
  let selectedSections = [...defaultSections];
  let checkCatalog = null;
  let editingInspectionTemplate = null;
  let editingSessionTemplate = null;
  let sessionTemplates = [];
  let inspectionTemplates = [];
  let actionBusy = false;
  const checkedValues = (form, name) => [...form.querySelectorAll(`[name="${name}"]:checked`)].map(item => item.value);
  const sectionLabel = kind => sectionLabels[kind] || kind;
  function matchesSearch(item, session = false) {
    const keyword = document.getElementById("template-search").value.trim().toLocaleLowerCase();
    return !keyword || [item.display_name, ...(session ? (item.sections || []).map(sectionLabel) : [])].some(value => String(value || "").toLocaleLowerCase().includes(keyword));
  }
  function orderedSections() {
    const movable = [...new Set(selectedSections)].filter(kind => !requiredSections.has(kind));
    return ["EXECUTIVE_SUMMARY", ...movable, "EVIDENCE_BOUNDARY"];
  }
  function renderOutline() {
    document.getElementById("session-template-outline").innerHTML = orderedSections().map((kind, index) => `<li><span>${String(index + 1).padStart(2, "0")}</span><strong>${escape(sectionLabel(kind))}</strong>${requiredSections.has(kind) ? "" : `<div class="button-row"><button class="text-button" type="button" aria-label="上移章节" data-move-section="${escape(kind)}" data-offset="-1">↑</button><button class="text-button" type="button" aria-label="下移章节" data-move-section="${escape(kind)}" data-offset="1">↓</button></div>`}</li>`).join("");
  }
  function renderSessionSections() {
    document.getElementById("session-template-sections").innerHTML = '<legend>报告章节</legend>' + Object.entries(sectionLabels).map(([kind, label]) => `<label class="check-field"><input type="checkbox" name="section" value="${kind}" ${selectedSections.includes(kind) ? "checked" : ""} ${requiredSections.has(kind) ? "disabled" : ""}><span class="check-field-copy"><strong>${label}</strong><small>${kind}</small></span></label>`).join("");
    renderOutline();
  }
  function renderInspectionChecks(selectedIds = []) {
    document.getElementById("inspection-template-checks").innerHTML = (checkCatalog?.groups || []).map(group => `<fieldset class="aiops-template-check-group"><legend>${escape(group.display_name)}</legend>${(group.checks || []).map(check => `<label class="check-field"><input type="checkbox" name="check_id" value="${escape(check.check_id)}" ${selectedIds.includes(check.check_id) ? "checked" : ""} ${check.availability !== "READY" ? "disabled" : ""}><span class="check-field-copy"><strong>${escape(check.display_name)}</strong><small>${escape(check.check_id)}${check.trend_required ? " · 需要历史趋势" : ""}${check.availability !== "READY" ? " · 尚未可用" : ""}</small></span></label>`).join("")}</fieldset>`).join("");
  }
  function renderInspectionTemplates() {
    document.getElementById("ops-inspection-template-body").innerHTML = inspectionTemplates.filter(item => matchesSearch(item)).map(item => `<article class="management-catalog-card"><header><div><p class="eyebrow">INSPECTION TEMPLATE</p><h3>${escape(item.display_name)}</h3></div><span class="status-pill ${item.status === "ACTIVE" ? "status-success" : "status-muted"}">${item.status === "ACTIVE" ? "已启用" : "已停用"}</span></header><dl><div><dt>当前版本</dt><dd>v${escape(item.version_no)}</dd></div><div><dt>检查项目</dt><dd>${(item.selected_check_ids || []).length} 项</dd></div></dl><footer class="management-catalog-actions"><span class="management-catalog-note">更新于 ${escape(shell.fmt(item.updated_at))}</span><button class="row-action" type="button" data-edit-inspection-template="${escape(item.inspection_template_id)}">创建新版本</button></footer></article>`).join("");
  }
  function renderSessionTemplates() {
    document.getElementById("ops-session-report-template-body").innerHTML = sessionTemplates.filter(item => matchesSearch(item, true)).map(item => {
      const actions = `${!item.is_default && item.status === "ACTIVE" ? `<button class="row-action" type="button" data-template-mutation data-default-session-template="${escape(item.template_ref)}">设为默认</button>` : ""}${item.system_defined ? `<button class="row-action" type="button" data-copy-session-template="${escape(item.template_ref)}">复制</button>` : `<button class="row-action" type="button" data-edit-session-template="${escape(item.template_id)}">新版本</button><button class="row-action" type="button" data-history-session-template="${escape(item.template_id)}">版本</button>`}`;
      const overflow = item.system_defined ? "" : `<div class="management-catalog-overflow"><button class="row-action management-catalog-more" type="button" aria-haspopup="menu" aria-expanded="false"><span>更多</span><span aria-hidden="true">⌄</span></button><div class="management-catalog-menu" role="menu" hidden><button class="row-action" type="button" data-template-mutation data-toggle-session-template="${escape(item.template_id)}" data-row-version="${escape(item.row_version)}" data-next-status="${item.status === "ACTIVE" ? "INACTIVE" : "ACTIVE"}">${item.status === "ACTIVE" ? "停用模板" : "启用模板"}</button></div></div>`;
      return `<article class="management-catalog-card${item.system_defined ? "" : " is-custom"}${item.status === "ACTIVE" ? "" : " is-disabled"}"><header><div><p class="eyebrow">${item.system_defined ? "SYSTEM TEMPLATE" : "CUSTOM TEMPLATE"}</p><h3>${escape(item.display_name)}</h3></div><span class="status-pill ${item.is_default || item.status === "ACTIVE" ? "status-success" : "status-muted"}">${item.is_default ? "默认" : item.status === "ACTIVE" ? "已启用" : "已停用"}</span></header><p>${escape((item.sections || []).map(sectionLabel).join("、") || "尚未配置报告章节。")}</p><dl><div><dt>当前版本</dt><dd>v${escape(item.version_no ?? item.version ?? "—")}</dd></div><div><dt>使用情况</dt><dd>${Number(item.usage_count || 0)} 份报告</dd></div></dl><footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions">${actions}</div>${overflow}</footer></article>`;
    }).join("");
    setActionBusy(actionBusy);
  }
  async function loadTemplates() {
    const refresh = document.getElementById("refresh-templates");
    refresh.disabled = true; refresh.textContent = "刷新中…";
    try {
      [checkCatalog, inspectionTemplates, sessionTemplates] = await Promise.all([
        KBotAIOpsAuth.request(`${api}/inspection-check-catalog`),
        KBotAIOpsAuth.request(`${api}/inspection-templates`),
        KBotAIOpsAuth.request(`${api}/session-report-templates`),
      ]);
      renderInspectionTemplates(); renderSessionTemplates();
    } catch (error) { shell.toast(error.message); }
    finally { refresh.disabled = false; refresh.textContent = "刷新目录"; }
  }
  function setActionBusy(value) {
    actionBusy = value;
    document.querySelectorAll("[data-template-mutation]").forEach(button => { button.disabled = value; });
  }
  async function mutateTemplate(path, method, body) {
    if (actionBusy) return;
    setActionBusy(true);
    try {
      await KBotAIOpsAuth.request(`${api}${path}`, { method, body: JSON.stringify(body) });
      await loadTemplates();
    } catch (error) { shell.toast(error.message); }
    finally { setActionBusy(false); }
  }
  function closeMenus(except = null) {
    document.querySelectorAll(".management-catalog-overflow").forEach(root => {
      if (root === except) return;
      root.querySelector(".management-catalog-menu").hidden = true;
      root.querySelector(".management-catalog-more").setAttribute("aria-expanded", "false");
    });
  }
  function canSave(form, kind) {
    return Boolean(form.elements.display_name.value.trim()) && (kind === "inspection" ? checkedValues(form, "check_id").length > 0 : selectedSections.length > 0);
  }
  function updateSaveButton(form, kind) {
    const busy = form.closest("dialog").dataset.busy === "true";
    const button = form.querySelector('[type="submit"]');
    button.disabled = busy || !canSave(form, kind); button.textContent = busy ? "保存中…" : "保存";
  }
  function openEditor(kind, editing = null, clone = null) {
    const isInspection = kind === "inspection";
    const dialog = document.getElementById(isInspection ? "inspection-template-dialog" : "session-report-template-dialog");
    const form = dialog.querySelector("form");
    if (isInspection) editingInspectionTemplate = editing;
    else editingSessionTemplate = editing;
    form.reset();
    form.elements.display_name.disabled = Boolean(editing);
    form.elements.display_name.value = editing?.display_name || (clone ? `${clone.display_name} 副本` : "");
    const result = form.querySelector('[role="status"]'); result.hidden = true; result.textContent = "";
    if (isInspection) renderInspectionChecks(editing?.selected_check_ids || []);
    else {
      selectedSections = editing ? (editing.definition?.sections || []).map(section => section.kind) : [...(clone?.sections || defaultSections)];
      renderSessionSections();
    }
    dialog.querySelector("h2").textContent = editing ? "创建新版本" : "新建模板";
    dialog.setAttribute("aria-label", editing ? "创建模板新版本" : "新建模板");
    updateSaveButton(form, kind);
    dialog.showModal();
    dialog.scrollTop = 0;
  }
  async function openVersion(kind, id) {
    try {
      const path = kind === "inspection" ? "inspection-templates" : "session-report-templates";
      const detail = await KBotAIOpsAuth.request(`${api}/${path}/${encodeURIComponent(id)}`);
      openEditor(kind, detail);
    } catch (error) { shell.toast(error.message); }
  }
  async function showSessionTemplateHistory(templateId) {
    try {
      const versions = await KBotAIOpsAuth.request(`${api}/session-report-templates/${encodeURIComponent(templateId)}/versions`);
      document.getElementById("template-history-title").textContent = sessionTemplates.find(item => item.template_id === templateId)?.display_name || "模板版本历史";
      document.getElementById("template-history-list").innerHTML = versions.map(item => `<article><header><strong>v${escape(item.version_no)}</strong>${item.current ? '<span class="status-pill status-success">当前版本</span>' : ""}<time>${escape(shell.fmt(item.created_at))}</time></header><p>${escape((item.definition?.sections || []).map(section => sectionLabel(section.kind)).join(" → "))}</p><small class="technical-meta">${escape(item.content_hash)}</small></article>`).join("");
      document.getElementById("template-history-drawer").showModal();
    } catch (error) { shell.toast(error.message); }
  }
  function bindEditor(kind) {
    const isInspection = kind === "inspection";
    const dialog = document.getElementById(isInspection ? "inspection-template-dialog" : "session-report-template-dialog");
    const form = dialog.querySelector("form");
    dialog.querySelectorAll(isInspection ? "[data-close-inspection-template]" : "[data-close-session-report-template]").forEach(button => { button.onclick = () => { if (!dialog.dataset.busy) dialog.close(); }; });
    form.addEventListener("input", () => updateSaveButton(form, kind));
    form.addEventListener("change", event => {
      if (event.target.name === "section") {
        if (event.target.checked) selectedSections.push(event.target.value);
        else selectedSections = selectedSections.filter(kind => kind !== event.target.value);
        renderOutline();
      }
      updateSaveButton(form, kind);
    });
    form.addEventListener("click", event => {
      const button = event.target.closest("[data-move-section]");
      if (!button || dialog.dataset.busy) return;
      const movable = selectedSections.filter(kind => !requiredSections.has(kind));
      const index = movable.indexOf(button.dataset.moveSection), target = index + Number(button.dataset.offset);
      if (index < 0 || target < 0 || target >= movable.length) return;
      [movable[index], movable[target]] = [movable[target], movable[index]];
      selectedSections = ["EXECUTIVE_SUMMARY", ...movable, "EVIDENCE_BOUNDARY"];
      renderOutline();
      // 重绘后恢复键盘焦点，连续上移/下移仍可操作同一章节。
      form.querySelector(`[data-move-section="${button.dataset.moveSection}"][data-offset="${button.dataset.offset}"]`)?.focus();
    });
    form.onsubmit = async event => {
      event.preventDefault();
      if (dialog.dataset.busy || !canSave(form, kind)) return;
      const editing = isInspection ? editingInspectionTemplate : editingSessionTemplate;
      const selected = isInspection ? { selected_check_ids: checkedValues(form, "check_id") } : { definition: { schema_version: "SESSION_REPORT_TEMPLATE.v1", sections: orderedSections().map(kind => ({ kind })) } };
      const id = isInspection ? editing?.inspection_template_id : editing?.template_id;
      const path = `${api}/${isInspection ? "inspection-templates" : "session-report-templates"}${id ? `/${encodeURIComponent(id)}/versions` : ""}`;
      const body = editing ? { expected_row_version: editing.row_version, ...selected } : { display_name: form.elements.display_name.value.trim(), ...selected };
      const result = form.querySelector('[role="status"]'); result.hidden = true;
      dialog.dataset.busy = "true";
      const controls = [...dialog.querySelectorAll("button,input")].map(node => [node, node.disabled]);
      controls.forEach(([node]) => { node.disabled = true; });
      updateSaveButton(form, kind);
      try {
        await KBotAIOpsAuth.request(path, { method: "POST", headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() }, body: JSON.stringify(body) });
        dialog.close(); shell.toast("模板已保存并冻结新版本"); await loadTemplates();
      } catch (error) { result.textContent = error.message; result.hidden = false; }
      finally {
        delete dialog.dataset.busy;
        controls.forEach(([node, disabled]) => { node.disabled = disabled; });
        updateSaveButton(form, kind);
      }
    };
  }
  shell.ready.then(async () => {
    document.getElementById("template-search").oninput = () => { renderInspectionTemplates(); renderSessionTemplates(); };
    document.getElementById("refresh-templates").onclick = loadTemplates;
    document.getElementById("create-inspection-template").onclick = () => openEditor("inspection");
    document.getElementById("create-session-report-template").onclick = () => openEditor("session");
    document.getElementById("close-template-history").onclick = () => document.getElementById("template-history-drawer").close();
    document.querySelectorAll("[data-template-tab]").forEach(button => { button.onclick = () => {
      document.querySelectorAll("[data-template-tab]").forEach(tab => { tab.classList.toggle("active", tab === button); tab.setAttribute("aria-selected", String(tab === button)); });
      document.getElementById("inspection-template-catalog").hidden = button.dataset.templateTab !== "inspection";
      document.getElementById("session-template-catalog").hidden = button.dataset.templateTab !== "session";
      closeMenus();
    }; });
    document.getElementById("ops-inspection-template-body").onclick = event => {
      const button = event.target.closest("[data-edit-inspection-template]");
      if (button) openVersion("inspection", button.dataset.editInspectionTemplate);
    };
    document.getElementById("ops-session-report-template-body").onclick = event => {
      const button = event.target.closest("button"); if (!button) return;
      if (button.matches(".management-catalog-more")) {
        const root = button.closest(".management-catalog-overflow");
        const open = button.getAttribute("aria-expanded") !== "true";
        closeMenus(root); button.setAttribute("aria-expanded", String(open)); root.querySelector(".management-catalog-menu").hidden = !open;
        return;
      }
      closeMenus();
      if (button.dataset.editSessionTemplate) openVersion("session", button.dataset.editSessionTemplate);
      if (button.dataset.historySessionTemplate) showSessionTemplateHistory(button.dataset.historySessionTemplate);
      if (button.dataset.copySessionTemplate) openEditor("session", null, sessionTemplates.find(item => item.template_ref === button.dataset.copySessionTemplate));
      if (button.dataset.defaultSessionTemplate) mutateTemplate("/session-report-templates/default", "PUT", { template_ref: button.dataset.defaultSessionTemplate });
      if (button.dataset.toggleSessionTemplate) mutateTemplate(`/session-report-templates/${encodeURIComponent(button.dataset.toggleSessionTemplate)}`, "PATCH", { expected_row_version: Number(button.dataset.rowVersion), status: button.dataset.nextStatus });
    };
    document.addEventListener("pointerdown", event => { if (!event.target.closest(".management-catalog-overflow")) closeMenus(); });
    document.addEventListener("keydown", event => { if (event.key === "Escape") closeMenus(); });
    bindEditor("inspection"); bindEditor("session");
    await loadTemplates();
  });
})();
