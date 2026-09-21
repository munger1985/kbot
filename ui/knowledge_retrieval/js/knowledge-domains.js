/* Domain 生命周期页：创建、修改、停用与手动切域，不在浏览器填写隔离标识。 */
(function () {
  "use strict";

  const { badge, escapeHtml, toast } = KBotKnowledgeShell;
  const PORTAL_NAME = "knowledge_retrieval_portal";
  let rows = [];
  let editingId = null;

  function openDialog(id) { document.getElementById(id)?.showModal(); }
  function closeDialog(id) { document.getElementById(id)?.close(); }
  function session() { return globalThis.KBotKnowledgeAuth?.load?.() || {}; }

  function emptyRow(columns, title, copy) {
    return `<tr><td colspan="${columns}"><div class="knowledge-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div></td></tr>`;
  }

  function statusTone(status) {
    return status === "ACTIVE" ? "good" : "warn";
  }

  function isPortal(row) {
    return String(row?.name || "") === PORTAL_NAME;
  }

  function isCurrent(row) {
    return String(session().domain_id || "") === String(row?.domain_id || "");
  }

  function resetForm() {
    editingId = null;
    document.getElementById("domain-dialog-title").textContent = "创建 Domain";
    document.getElementById("domain-submit").textContent = "创建";
    document.getElementById("domain-form")?.reset();
    document.getElementById("domain-id").value = "";
    document.getElementById("domain-row-version").value = "";
    document.getElementById("domain-name").disabled = false;
  }

  function fillForm(row) {
    editingId = row.domain_id;
    document.getElementById("domain-dialog-title").textContent = "编辑 Domain";
    document.getElementById("domain-submit").textContent = "保存";
    document.getElementById("domain-id").value = row.domain_id;
    document.getElementById("domain-row-version").value = row.row_version;
    document.getElementById("domain-name").value = row.name || "";
    document.getElementById("domain-name").disabled = isPortal(row);
    document.getElementById("domain-description").value = row.description || "";
  }

  function actionButtons(row) {
    const id = escapeHtml(row.domain_id);
    const buttons = [];
    if (isCurrent(row)) {
      buttons.push('<span class="knowledge-badge good">当前</span>');
    } else if (row.status === "ACTIVE") {
      buttons.push(`<button class="small" type="button" data-action="switch" data-domain-id="${id}">切换</button>`);
    }
    buttons.push(`<button class="small" type="button" data-action="edit" data-domain-id="${id}">编辑</button>`);
    if (!isPortal(row) && row.status === "DISABLED") {
      buttons.push(`<button class="small" type="button" data-action="enable" data-domain-id="${id}">启用</button>`);
    }
    if (!isPortal(row) && row.status !== "DISABLED") {
      buttons.push(`<button class="small danger" type="button" data-action="delete" data-domain-id="${id}">删除</button>`);
    }
    return `<div class="knowledge-row-actions">${buttons.join("")}</div>`;
  }

  function renderRows() {
    const node = document.getElementById("domain-rows");
    if (!node) return;
    if (!rows.length) {
      node.innerHTML = emptyRow(6, "尚无可显示的 Domain", "接入平台 Domain 生命周期后，列表将按当前 App 授权范围加载；不会使用浏览器填写的 ID 作为隔离边界。");
      return;
    }
    node.innerHTML = rows.map((row) => `<tr>
      <td>${escapeHtml(row.name || "未命名")}</td>
      <td>${escapeHtml(row.domain_id)}</td>
      <td>${badge(row.status || "UNKNOWN", statusTone(row.status))}</td>
      <td>${escapeHtml(row.description || "—")}</td>
      <td>${escapeHtml(row.updated_at || row.created_at || "—")}</td>
      <td>${actionButtons(row)}</td>
    </tr>`).join("");
    node.querySelectorAll("[data-action]").forEach((button) => {
      button.addEventListener("click", () => {
        handleAction(button.dataset.action, button.dataset.domainId).catch((error) => {
          toast(error.message || "Domain 操作失败", "error");
        });
      });
    });
  }

  function findRow(domainId) {
    return rows.find((row) => String(row.domain_id) === String(domainId));
  }

  async function loadRows() {
    const payload = await KBotKnowledgeApi.json("/domains", "GET");
    rows = Array.isArray(payload?.items) ? payload.items : [];
    renderRows();
  }

  async function switchDomain(row) {
    const current = session();
    const next = await KBotKnowledgeApi.json("/api/v1/auth/switch-domain", "POST", {
      domain_id: Number(row.domain_id),
    });
    KBotKnowledgeAuth.save({ ...current, ...next });
    location.reload();
  }

  async function handleAction(action, domainId) {
    const row = findRow(domainId);
    if (!row) return;
    if (action === "switch") {
      await switchDomain(row);
      return;
    }
    if (action === "edit") {
      fillForm(row);
      openDialog("domain-dialog");
      return;
    }
    if (action === "enable") {
      await KBotKnowledgeApi.json(`/domains/${row.domain_id}`, "PATCH", {
        expected_row_version: row.row_version,
        status: "ACTIVE",
      });
      await loadRows();
      toast("Domain 已启用。");
      return;
    }
    if (action === "delete") {
      const confirmed = globalThis.confirm(`删除会将 Domain 停用，不会物理删除。确认停用「${row.name}」？`);
      if (!confirmed) return;
      await KBotKnowledgeApi.json(
        KBotKnowledgeApi.withQuery(`/domains/${row.domain_id}`, {
          expected_row_version: row.row_version,
        }),
        "DELETE",
      );
      await loadRows();
      toast("Domain 已停用。当前会话不会自动切换。");
    }
  }

  async function submitForm(event) {
    event.preventDefault();
    const name = String(document.getElementById("domain-name").value || "").trim();
    const description = String(document.getElementById("domain-description").value || "").trim();
    if (!name) {
      toast("请填写 Domain 名称。", "error");
      return;
    }
    try {
      if (editingId == null) {
        await KBotKnowledgeApi.json("/domains", "POST", {
          name,
          description: description || null,
        });
        closeDialog("domain-dialog");
        resetForm();
        await loadRows();
        toast("Domain 已创建。当前会话不会自动切换，需要时请手动切换。");
        return;
      }
      const payload = {
        expected_row_version: Number(document.getElementById("domain-row-version").value),
        description: description || null,
      };
      if (!document.getElementById("domain-name").disabled) payload.name = name;
      await KBotKnowledgeApi.json(`/domains/${editingId}`, "PATCH", payload);
      closeDialog("domain-dialog");
      resetForm();
      await loadRows();
      toast("Domain 已保存。");
    } catch (error) {
      toast(error.message || "无法保存 Domain", "error");
    }
  }

  KBotKnowledgeShell.ready.then(async (access) => {
    if (!access) return;
    document.querySelectorAll("[data-open-dialog]").forEach((button) => {
      button.addEventListener("click", () => {
        resetForm();
        openDialog(button.dataset.openDialog);
      });
    });
    document.querySelectorAll("[data-close-dialog]").forEach((button) => {
      button.addEventListener("click", () => closeDialog(button.dataset.closeDialog));
    });
    document.getElementById("domain-form")?.addEventListener("submit", submitForm);
    try {
      await loadRows();
    } catch (error) {
      toast(error.message || "无法加载 Domain", "error");
    }
  });
})();
