/* Knowledge Core 生命周期页：创建、修改、启停与删除，隔离边界只来自当前 Token。 */
(function () {
  "use strict";

  const { badge, escapeHtml, toast } = KBotAssistantShell;
  let rows = [];
  let catalog = [];
  let editingId = null;

  function openDialog(id) { document.getElementById(id)?.showModal(); }
  function closeDialog(id) { document.getElementById(id)?.close(); }
  function session() { return globalThis.KBotAssistantAuth?.load?.() || {}; }

  function emptyRow(columns, title, copy) {
    return `<tr><td colspan="${columns}"><div class="assistant-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div></td></tr>`;
  }

  function statusTone(status) {
    if (status === "ACTIVE") return "good";
    if (status === "DISABLED" || status === "DELETING") return "warn";
    return "warn";
  }

  function collectionItems(payload) {
    if (Array.isArray(payload)) return payload;
    if (Array.isArray(payload?.collections)) return payload.collections;
    if (Array.isArray(payload?.items)) return payload.items;
    return [];
  }

  function modelsOf(row) {
    return row?.models || row?.models_json || {};
  }

  function modelLabel(modelId) {
    if (!modelId) return "—";
    const row = catalog.find((item) => String(item.model_id) === String(modelId));
    return row?.display_name || row?.served_model_name || modelId;
  }

  function fillSelect(select, items, currentId, emptyLabel) {
    const options = [`<option value="">${escapeHtml(emptyLabel)}</option>`].concat(
      items.map((row) => {
        const selected = String(row.model_id) === String(currentId || "") ? " selected" : "";
        const label = row.display_name || row.served_model_name || row.model_id;
        return `<option value="${escapeHtml(row.model_id)}"${selected}>${escapeHtml(label)}</option>`;
      }),
    );
    select.innerHTML = options.join("");
    select.disabled = false;
  }

  function currentDomainLabel() {
    const current = session();
    return current.domain_name || current.domain_id || "当前 Domain";
  }

  function resetForm() {
    editingId = null;
    document.getElementById("kc-dialog-title").textContent = "创建 Knowledge Core";
    document.getElementById("kc-submit").textContent = "创建";
    document.getElementById("kc-form")?.reset();
    document.getElementById("kc-id").value = "";
    document.getElementById("kc-row-version").value = "";
    document.getElementById("kc-domain").value = currentDomainLabel();
    fillSelect(
      document.getElementById("kc-embedding"),
      catalog.filter((row) => Number(row.category) === 2),
      "",
      catalog.some((row) => Number(row.category) === 2) ? "选择文本 Embedding" : "当前没有可用的文本 Embedding",
    );
    fillSelect(
      document.getElementById("kc-visual-embedding"),
      catalog.filter((row) => Number(row.category) === 3),
      "",
      "不绑定",
    );
  }

  function fillForm(row) {
    editingId = row.collection_id;
    const models = modelsOf(row);
    document.getElementById("kc-dialog-title").textContent = "编辑 Knowledge Core";
    document.getElementById("kc-submit").textContent = "保存";
    document.getElementById("kc-id").value = row.collection_id;
    document.getElementById("kc-row-version").value = row.row_version;
    document.getElementById("kc-domain").value = currentDomainLabel();
    document.getElementById("kc-name").value = row.display_name || "";
    document.getElementById("kc-description").value = row.description || "";
    document.getElementById("kc-security").value = String(row.default_security_level ?? 1);
    fillSelect(
      document.getElementById("kc-embedding"),
      catalog.filter((item) => Number(item.category) === 2),
      models.embedding,
      "选择文本 Embedding",
    );
    fillSelect(
      document.getElementById("kc-visual-embedding"),
      catalog.filter((item) => Number(item.category) === 3),
      models.visual_embedding,
      "不绑定",
    );
  }

  function actionButtons(row) {
    const id = escapeHtml(row.collection_id);
    if (row.status === "DELETING") {
      return '<span class="assistant-badge warn">清理中</span>';
    }
    const buttons = [
      `<button class="small" type="button" data-action="edit" data-collection-id="${id}">编辑</button>`,
    ];
    if (row.status === "ACTIVE") {
      buttons.push(`<button class="small" type="button" data-action="disable" data-collection-id="${id}">停用</button>`);
    } else if (row.status === "DISABLED") {
      buttons.push(`<button class="small" type="button" data-action="enable" data-collection-id="${id}">启用</button>`);
    }
    buttons.push(`<button class="small danger" type="button" data-action="delete" data-collection-id="${id}">删除</button>`);
    return `<div class="assistant-row-actions">${buttons.join("")}</div>`;
  }

  function renderRows() {
    const node = document.getElementById("kc-rows");
    if (!node) return;
    if (!rows.length) {
      node.innerHTML = emptyRow(7, "当前 Domain 尚无 Knowledge Core", "需先创建或切换到有权限的 Domain。没有 ACTIVE Knowledge Core 的 Agent 只能保存草稿，不能启用。");
      return;
    }
    node.innerHTML = rows.map((row) => {
      const models = modelsOf(row);
      return `<tr>
        <td>${escapeHtml(row.display_name || "未命名")}</td>
        <td>${badge(row.status || "UNKNOWN", statusTone(row.status))}</td>
        <td>${escapeHtml(row.collection_id || "—")}</td>
        <td>${escapeHtml(modelLabel(models.embedding))}</td>
        <td>${escapeHtml(row.default_security_level ?? "—")}</td>
        <td>${escapeHtml(row.updated_at || "—")}</td>
        <td>${actionButtons(row)}</td>
      </tr>`;
    }).join("");
    node.querySelectorAll("[data-action]").forEach((button) => {
      button.addEventListener("click", () => {
        handleAction(button.dataset.action, button.dataset.collectionId).catch((error) => {
          toast(error.message || "Knowledge Core 操作失败", "error");
        });
      });
    });
  }

  function findRow(collectionId) {
    return rows.find((row) => String(row.collection_id) === String(collectionId));
  }

  async function loadPage() {
    const [listed, catalogRows] = await Promise.all([
      KBotAssistantApi.json("/knowledge-cores", "GET"),
      KBotAssistantApi.json("/model-catalog", "GET"),
    ]);
    rows = collectionItems(listed);
    catalog = (Array.isArray(catalogRows) ? catalogRows : []).filter(
      (row) => String(row.status || "").toUpperCase() === "ACTIVE",
    );
    renderRows();
  }

  async function changeStatus(row, status) {
    await KBotAssistantApi.json(`/knowledge-cores/${row.collection_id}`, "PATCH", {
      expected_row_version: row.row_version,
      status,
    });
    await loadPage();
    toast(status === "ACTIVE" ? "Knowledge Core 已启用。" : "Knowledge Core 已停用。");
  }

  async function handleAction(action, collectionId) {
    const row = findRow(collectionId);
    if (!row) return;
    if (action === "edit") {
      fillForm(row);
      openDialog("kc-dialog");
      return;
    }
    if (action === "enable") {
      await changeStatus(row, "ACTIVE");
      return;
    }
    if (action === "disable") {
      await changeStatus(row, "DISABLED");
      return;
    }
    if (action === "delete") {
      const confirmed = globalThis.confirm(`删除会进入清理流程，不能撤销。确认删除「${row.display_name}」？`);
      if (!confirmed) return;
      await KBotAssistantApi.json(`/knowledge-cores/${row.collection_id}`, "DELETE");
      await loadPage();
      toast("已提交删除，Knowledge Core 正在清理。");
    }
  }

  function createPayload() {
    const embedding = document.getElementById("kc-embedding").value;
    const visual = document.getElementById("kc-visual-embedding").value;
    const payload = {
      display_name: String(document.getElementById("kc-name").value || "").trim(),
      description: String(document.getElementById("kc-description").value || "").trim() || null,
      default_security_level: Number(document.getElementById("kc-security").value || 1),
      embedding,
    };
    if (visual) payload.visual_embedding = visual;
    return payload;
  }

  async function submitForm(event) {
    event.preventDefault();
    const payload = createPayload();
    if (!payload.display_name) {
      toast("请填写 Knowledge Core 名称。", "error");
      return;
    }
    if (!payload.embedding) {
      toast("请选择已启用的文本 Embedding 模型。", "error");
      return;
    }
    try {
      if (editingId == null) {
        await KBotAssistantApi.json("/knowledge-cores", "POST", payload);
        closeDialog("kc-dialog");
        resetForm();
        await loadPage();
        toast("Knowledge Core 已创建。");
        return;
      }
      const current = findRow(editingId) || {};
      const models = modelsOf(current);
      const update = {
        expected_row_version: Number(document.getElementById("kc-row-version").value),
        display_name: payload.display_name,
        description: payload.description,
        default_security_level: payload.default_security_level,
      };
      if (payload.embedding !== String(models.embedding || "")) update.embedding = payload.embedding;
      const currentVisual = models.visual_embedding ? String(models.visual_embedding) : "";
      const nextVisual = payload.visual_embedding ? String(payload.visual_embedding) : "";
      if (nextVisual !== currentVisual) update.visual_embedding = payload.visual_embedding || null;
      await KBotAssistantApi.json(`/knowledge-cores/${editingId}`, "PATCH", update);
      closeDialog("kc-dialog");
      resetForm();
      await loadPage();
      toast("Knowledge Core 已保存。");
    } catch (error) {
      toast(error.message || "无法保存 Knowledge Core", "error");
    }
  }

  KBotAssistantShell.ready.then(async (access) => {
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
    document.getElementById("kc-form")?.addEventListener("submit", submitForm);
    try {
      await loadPage();
    } catch (error) {
      toast(error.message || "无法加载 Knowledge Core", "error");
    }
  });
})();
