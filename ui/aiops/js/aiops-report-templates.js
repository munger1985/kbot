(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let checkCatalog = null;
  let editingInspectionTemplate = null;
  let editingSessionTemplate = null;

  const checkedValues = (form, name) => Array.from(form.querySelectorAll(`[name="${name}"]:checked`)).map((item) => item.value);

  function renderInspectionChecks(selectedIds = []) {
    const root = document.getElementById("inspection-template-checks");
    const selected = new Set(selectedIds);
    root.innerHTML = (checkCatalog?.groups || []).map((group) => `<fieldset class="inspection-check-group"><legend>${shell.escape(group.display_name)}</legend>${(group.checks || []).map((check) => `<label class="inspection-check-item${check.availability !== "READY" ? " is-disabled" : ""}"><input type="checkbox" name="check_id" value="${shell.escape(check.check_id)}" ${selected.has(check.check_id) ? "checked" : ""} ${check.availability !== "READY" ? "disabled" : ""}><span><strong>${shell.escape(check.display_name)}</strong><small>${check.availability === "READY" ? (check.trend_required ? "支持趋势分析" : "当前状态检查") : "规划中"}</small></span></label>`).join("")}</fieldset>`).join("");
    if (!(checkCatalog?.groups || []).length) root.innerHTML = '<p class="ops-empty">检查目录不可用。</p>';
  }

  async function loadInspectionTemplates() {
    const body = document.getElementById("ops-inspection-template-body");
    try {
      const rows = await KBotAIOpsAuth.request(`${api}/inspection-templates`);
      body.innerHTML = rows.map((item) => `<tr><td>${shell.escape(item.display_name)}</td><td>v${shell.escape(item.version_no)}</td><td>${shell.escape((item.selected_check_ids || []).length)} 项</td><td><code>${shell.escape(shell.short(item.content_hash))}</code></td><td>${shell.badge(item.status || "ACTIVE")}</td><td><button type="button" data-edit-inspection-template="${shell.escape(item.inspection_template_id)}">发布新版本</button></td></tr>`).join("") || '<tr><td class="ops-empty" colspan="6">暂无巡检模板</td></tr>';
      body.querySelectorAll("[data-edit-inspection-template]").forEach((button) => {
        button.onclick = () => openInspectionTemplateVersion(button.dataset.editInspectionTemplate);
      });
    } catch (error) {
      body.innerHTML = `<tr><td class="ops-empty" colspan="6">${shell.escape(error.message)}</td></tr>`;
    }
  }

  async function loadSessionTemplates() {
    const body = document.getElementById("ops-session-report-template-body");
    try {
      const rows = await KBotAIOpsAuth.request(`${api}/session-report-templates`);
      body.innerHTML = rows.map((item) => `<tr><td>${shell.escape(item.display_name)}</td><td>${shell.escape((item.sections || []).join("、") || "受控章节")}</td><td>${shell.escape(item.version || item.version_no || "—")}</td><td><code>${shell.escape(shell.short(item.content_hash))}</code></td><td>${item.system_defined ? shell.badge("系统预设") : shell.badge(item.status || "ACTIVE")}</td><td>${item.system_defined ? "—" : `<button type="button" data-edit-session-template="${shell.escape(item.template_id)}">发布新版本</button>`}</td></tr>`).join("") || '<tr><td class="ops-empty" colspan="6">暂无会话报告模板</td></tr>';
      body.querySelectorAll("[data-edit-session-template]").forEach((button) => {
        button.onclick = () => openSessionTemplateVersion(button.dataset.editSessionTemplate);
      });
    } catch (error) {
      body.innerHTML = `<tr><td class="ops-empty" colspan="6">${shell.escape(error.message)}</td></tr>`;
    }
  }

  async function openInspectionTemplateVersion(templateId) {
    try {
      editingInspectionTemplate = await KBotAIOpsAuth.request(`${api}/inspection-templates/${encodeURIComponent(templateId)}`);
      const dialog = document.getElementById("inspection-template-dialog");
      const form = document.getElementById("inspection-template-form");
      form.reset();
      form.elements.display_name.value = editingInspectionTemplate.display_name;
      form.elements.display_name.disabled = true;
      renderInspectionChecks(editingInspectionTemplate.selected_check_ids || []);
      dialog.querySelector("h2").textContent = "发布巡检模板新版本";
      dialog.showModal();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function bindInspectionTemplate() {
    const dialog = document.getElementById("inspection-template-dialog");
    const form = document.getElementById("inspection-template-form");
    document.getElementById("create-inspection-template").onclick = () => {
      editingInspectionTemplate = null;
      form.reset();
      form.elements.display_name.disabled = false;
      renderInspectionChecks();
      dialog.querySelector("h2").textContent = "新建巡检模板";
      dialog.showModal();
    };
    dialog.querySelector("[data-close-inspection-template]").onclick = () => dialog.close();
    form.onsubmit = async (event) => {
      event.preventDefault();
      const selected = checkedValues(form, "check_id");
      const result = document.getElementById("inspection-template-result");
      if (!selected.length) {
        result.textContent = "请至少勾选一个已开放检查项。";
        return;
      }
      try {
        const path = editingInspectionTemplate
          ? `${api}/inspection-templates/${encodeURIComponent(editingInspectionTemplate.inspection_template_id)}/versions`
          : `${api}/inspection-templates`;
        const body = editingInspectionTemplate
          ? { expected_row_version: editingInspectionTemplate.row_version, selected_check_ids: selected }
          : { display_name: form.elements.display_name.value.trim(), selected_check_ids: selected };
        await KBotAIOpsAuth.request(path, {
          method: "POST",
          headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify(body),
        });
        editingInspectionTemplate = null;
        dialog.close();
        shell.toast("巡检模板版本已保存");
        await loadInspectionTemplates();
      } catch (error) {
        result.textContent = error.message;
      }
    };
  }

  async function openSessionTemplateVersion(templateId) {
    try {
      editingSessionTemplate = await KBotAIOpsAuth.request(`${api}/session-report-templates/${encodeURIComponent(templateId)}`);
      const dialog = document.getElementById("session-report-template-dialog");
      const form = document.getElementById("session-report-template-form");
      form.reset();
      form.elements.display_name.value = editingSessionTemplate.display_name;
      form.elements.display_name.disabled = true;
      const selected = new Set((editingSessionTemplate.definition?.sections || []).map((item) => item.kind));
      form.querySelectorAll('[name="section"]').forEach((input) => { input.checked = selected.has(input.value); });
      dialog.querySelector("h2").textContent = "发布会话报告模板新版本";
      dialog.showModal();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  function bindSessionTemplate() {
    const dialog = document.getElementById("session-report-template-dialog");
    const form = document.getElementById("session-report-template-form");
    document.getElementById("create-session-report-template").onclick = () => {
      editingSessionTemplate = null;
      form.reset();
      form.elements.display_name.disabled = false;
      dialog.querySelector("h2").textContent = "新建会话报告模板";
      dialog.showModal();
    };
    dialog.querySelector("[data-close-session-report-template]").onclick = () => dialog.close();
    form.onsubmit = async (event) => {
      event.preventDefault();
      const sections = checkedValues(form, "section");
      const result = document.getElementById("session-report-template-result");
      if (!sections.length) {
        result.textContent = "请至少选择一个展示章节。";
        return;
      }
      try {
        const path = editingSessionTemplate
          ? `${api}/session-report-templates/${encodeURIComponent(editingSessionTemplate.template_id)}/versions`
          : `${api}/session-report-templates`;
        const definition = { schema_version: "SESSION_REPORT_TEMPLATE.v1", sections: sections.map((kind) => ({ kind })) };
        const body = editingSessionTemplate
          ? { expected_row_version: editingSessionTemplate.row_version, definition }
          : { display_name: form.elements.display_name.value.trim(), definition };
        await KBotAIOpsAuth.request(path, {
          method: "POST",
          headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify(body),
        });
        editingSessionTemplate = null;
        dialog.close();
        shell.toast("会话报告模板版本已保存");
        await loadSessionTemplates();
      } catch (error) {
        result.textContent = error.message;
      }
    };
  }

  shell.ready.then(async () => {
    try {
      checkCatalog = await KBotAIOpsAuth.request(`${api}/inspection-check-catalog`);
    } catch (error) {
      checkCatalog = null;
    }
    await Promise.all([loadInspectionTemplates(), loadSessionTemplates()]);
    bindInspectionTemplate();
    bindSessionTemplate();
  });
})();
