/* Agent 生命周期页：管理草稿、版本资源、启停与归档，不在浏览器推断服务端绑定有效性。 */
(function () {
  "use strict";

  const { badge, escapeHtml, toast } = KBotKnowledgeShell;
  const MODEL_ROLES = Object.freeze({
    context_llm: "agent-context-llm",
    composer_llm: "agent-composer-llm",
    memory_llm: "agent-memory-llm",
    memory_embedding: "agent-memory-embedding",
    router_llm: "agent-router-llm",
    x_search_llm: "agent-x-search-llm",
    data_planner_llm: "agent-data-planner-llm",
  });
  const state = {
    rows: [],
    options: { knowledge_cores: [], data_models: [], models: [] },
    editing: null,
  };

  function element(id) { return document.getElementById(id); }
  function openDialog() { element("agent-dialog")?.showModal(); }
  function closeDialog() { element("agent-dialog")?.close(); }

  function emptyRow(title, copy) {
    return `<tr><td colspan="6"><div class="knowledge-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div></td></tr>`;
  }

  function statusLabel(status) {
    return ({ DRAFT: "草稿", ACTIVE: "已启用", DISABLED: "已停用", ARCHIVED: "已归档" })[status] || status || "未知";
  }

  function statusTone(status) {
    if (status === "ACTIVE") return "good";
    if (status === "ARCHIVED") return "bad";
    return "warn";
  }

  function knowledgeCore(id) {
    return state.options.knowledge_cores.find((row) => String(row.collection_id) === String(id));
  }

  function dataModel(id) {
    return state.options.data_models.find((row) => String(row.semantic_model_id) === String(id));
  }

  function catalogModel(id) {
    return state.options.models.find((row) => String(row.model_id) === String(id));
  }

  function resourceMarkup(row) {
    const core = knowledgeCore(row.knowledge_core_id);
    const coreLabel = core?.display_name || row.knowledge_core_id || "未绑定 Knowledge Core";
    const models = (row.data_model_ids || []).map((id) => dataModel(id)?.display_name || id);
    const dataMarkup = models.length
      ? `<span class="agent-resource-list"><strong>问数</strong>${models.map((name) => `<span>${escapeHtml(name)}</span>`).join("")}</span>`
      : '<span class="agent-resource-list muted"><strong>问数</strong><span>未绑定</span></span>';
    return `<div class="agent-resource-stack">
      <span class="agent-resource-list"><strong>知识</strong><span>${escapeHtml(coreLabel)}</span></span>
      ${dataMarkup}
    </div>`;
  }

  function modelMarkup(row) {
    const labels = Object.entries(row.models || {}).map(([role, id]) => {
      const model = catalogModel(id);
      const name = model?.display_name || model?.served_model_name || id;
      return `<span title="${escapeHtml(role)}">${escapeHtml(name)}</span>`;
    });
    return labels.length ? `<div class="agent-model-list">${labels.join("")}</div>` : "—";
  }

  function actionButtons(row) {
    const id = escapeHtml(row.agent_id);
    if (row.status === "ARCHIVED") return '<span class="knowledge-badge">只读</span>';
    const lifecycle = row.status === "ACTIVE"
      ? `<button class="small" type="button" data-agent-action="disable" data-agent-id="${id}">停用</button>`
      : `<button class="small primary" type="button" data-agent-action="activate" data-agent-id="${id}">启用</button>`;
    return `<div class="knowledge-row-actions">
      <button class="small" type="button" data-agent-action="edit" data-agent-id="${id}">编辑</button>
      ${lifecycle}
      <button class="small danger" type="button" data-agent-action="archive" data-agent-id="${id}">删除</button>
    </div>`;
  }

  function filteredRows() {
    const keyword = String(element("agent-filter")?.value || "").trim().toLocaleLowerCase();
    const status = element("agent-status-filter")?.value || "";
    return state.rows.filter((row) => {
      if (status && row.status !== status) return false;
      if (!keyword) return true;
      return [row.display_name, row.description].some((value) => String(value || "").toLocaleLowerCase().includes(keyword));
    });
  }

  function renderRows() {
    const node = element("agent-rows");
    if (!node) return;
    element("agent-count").textContent = String(state.rows.length);
    const rows = filteredRows();
    if (!rows.length) {
      node.innerHTML = emptyRow(
        state.rows.length ? "没有匹配的 Agent" : "尚未创建 Agent",
        state.rows.length ? "请调整搜索条件或状态筛选。" : "创建草稿并配置 Knowledge Core、运行模型和可选的问数模型。",
      );
      return;
    }
    node.innerHTML = rows.map((row) => `<tr>
      <td><strong>${escapeHtml(row.display_name || "未命名")}</strong><small class="agent-cell-copy">${escapeHtml(row.description || "无说明")}</small></td>
      <td>${badge(statusLabel(row.status), statusTone(row.status))}</td>
      <td>${resourceMarkup(row)}</td>
      <td>${modelMarkup(row)}</td>
      <td><strong>v${escapeHtml(row.version_no || 1)}</strong><small class="agent-cell-copy">配置版本 ${escapeHtml(row.row_version || "—")}</small></td>
      <td>${actionButtons(row)}</td>
    </tr>`).join("");
    node.querySelectorAll("[data-agent-action]").forEach((button) => {
      button.addEventListener("click", () => handleAction(button.dataset.agentAction, button.dataset.agentId).catch(showError));
    });
  }

  function modelLabel(row) {
    return row.display_name || row.served_model_name || row.model_id;
  }

  function setSelectOptions(select, rows, currentId, emptyLabel) {
    const known = rows.some((row) => String(row.model_id) === String(currentId || ""));
    const options = [`<option value="">${escapeHtml(emptyLabel)}</option>`];
    options.push(...rows.map((row) => `<option value="${escapeHtml(row.model_id)}"${String(row.model_id) === String(currentId || "") ? " selected" : ""}>${escapeHtml(modelLabel(row))}</option>`));
    if (currentId && !known) options.push(`<option value="${escapeHtml(currentId)}" selected>${escapeHtml(currentId)}（当前绑定，目录不可用）</option>`);
    select.innerHTML = options.join("");
  }

  function fillResourceOptions(current) {
    const currentCore = current?.knowledge_core_id || "";
    element("agent-knowledge-core").innerHTML = ['<option value="">暂不绑定</option>'].concat(
      state.options.knowledge_cores.map((row) => {
        const selected = String(row.collection_id) === String(currentCore) ? " selected" : "";
        const unavailable = !row.selectable && !selected ? " disabled" : "";
        const suffix = row.selectable ? "" : ` · ${row.status || "不可用"}`;
        return `<option value="${escapeHtml(row.collection_id)}"${selected}${unavailable}>${escapeHtml((row.display_name || row.collection_id) + suffix)}</option>`;
      }),
    ).join("");

    const selectedIds = new Set((current?.data_model_ids || []).map(String));
    const dataModels = state.options.data_models.map((row) => {
      const selected = selectedIds.has(String(row.semantic_model_id));
      const disabled = !row.selectable && !selected;
      const reason = row.selectable ? "" : " · 未发布";
      return `<option value="${escapeHtml(row.semantic_model_id)}"${selected ? " selected" : ""}${disabled ? " disabled" : ""}>${escapeHtml((row.display_name || row.semantic_model_id) + reason)}</option>`;
    });
    for (const id of selectedIds) {
      if (!state.options.data_models.some((row) => String(row.semantic_model_id) === id)) {
        dataModels.push(`<option value="${escapeHtml(id)}" selected>${escapeHtml(id)}（当前绑定，目录不可用）</option>`);
      }
    }
    element("agent-data-models").innerHTML = dataModels.length ? dataModels.join("") : '<option value="" disabled>当前没有可用的问数模型</option>';

    const llms = state.options.models.filter((row) => KBotKnowledgeApi.modelCategory(row) === KBotKnowledgeApi.ModelCategory.LLM);
    const grokModels = llms.filter(KBotKnowledgeApi.isGrokModel);
    const embeddings = state.options.models.filter((row) => KBotKnowledgeApi.modelCategory(row) === KBotKnowledgeApi.ModelCategory.TXT_EMBEDDING);
    Object.entries(MODEL_ROLES).forEach(([role, id]) => {
      const placeholder = role === "data_planner_llm" ? "跟随回答生成 LLM" : "不绑定";
      const rows = role === "memory_embedding" ? embeddings : role === "x_search_llm" ? grokModels : llms;
      setSelectOptions(element(id), rows, current?.models?.[role], placeholder);
    });
  }

  function resetForm() {
    state.editing = null;
    element("agent-form").reset();
    element("agent-id").value = "";
    element("agent-row-version").value = "";
    element("agent-dialog-title").textContent = "创建 Agent";
    element("agent-dialog-subtitle").textContent = "新 Agent 将以草稿状态保存。";
    element("agent-submit").textContent = "保存草稿";
    element("agent-status").value = "DRAFT";
    element("agent-status").disabled = true;
    element("agent-config").value = "{}";
    fillResourceOptions(null);
  }

  function fillForm(row) {
    state.editing = structuredClone(row);
    element("agent-id").value = row.agent_id;
    element("agent-row-version").value = row.row_version;
    element("agent-dialog-title").textContent = "编辑 Agent";
    element("agent-dialog-subtitle").textContent = `当前运行版本 v${row.version_no || 1}；保存运行配置后将生成新版本。`;
    element("agent-submit").textContent = "保存修改";
    element("agent-name").value = row.display_name || "";
    element("agent-description").value = row.description || "";
    element("agent-instruction").value = row.instruction || "";
    element("agent-status").disabled = false;
    element("agent-status").value = row.status === "ACTIVE" ? "ACTIVE" : row.status === "DISABLED" ? "DISABLED" : "DRAFT";
    element("agent-config").value = JSON.stringify(row.config || {}, null, 2);
    fillResourceOptions(row);
  }

  function selectedDataModels() {
    return [...element("agent-data-models").selectedOptions].map((option) => option.value).filter(Boolean);
  }

  function selectedModels() {
    return Object.fromEntries(Object.entries(MODEL_ROLES).flatMap(([role, id]) => {
      const value = element(id).value;
      return value ? [[role, value]] : [];
    }));
  }

  function parseConfig() {
    const source = String(element("agent-config").value || "").trim();
    if (!source) return {};
    const value = JSON.parse(source);
    if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("高级配置必须是 JSON 对象。");
    return value;
  }

  function stable(value) {
    if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
    if (value && typeof value === "object") return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
    return JSON.stringify(value);
  }

  function versionValues() {
    return {
      knowledge_core_id: element("agent-knowledge-core").value || null,
      data_model_ids: selectedDataModels(),
      models: selectedModels(),
      instruction: String(element("agent-instruction").value || "").trim() || null,
      config: parseConfig(),
    };
  }

  function versionChanges(next, current) {
    const changes = {};
    for (const key of Object.keys(next)) {
      const before = key === "data_model_ids" ? [...(current?.[key] || [])].map(String).sort() : current?.[key] ?? (key === "models" || key === "config" ? {} : null);
      const after = key === "data_model_ids" ? [...next[key]].map(String).sort() : next[key];
      if (stable(before) !== stable(after)) changes[key] = next[key];
    }
    return changes;
  }

  async function submitForm(event) {
    event.preventDefault();
    const submit = element("agent-submit");
    const displayName = String(element("agent-name").value || "").trim();
    if (!displayName) {
      toast("请填写 Agent 名称。", "error");
      return;
    }
    let version;
    try {
      version = versionValues();
    } catch (error) {
      toast(error.message || "高级配置不是有效 JSON。", "error");
      return;
    }
    submit.disabled = true;
    try {
      const description = String(element("agent-description").value || "").trim() || null;
      if (!state.editing) {
        await KBotKnowledgeApi.json("/agents", "POST", {
          display_name: displayName,
          description,
          ...version,
          status: "DRAFT",
        });
        closeDialog();
        await loadAll();
        toast("Agent 草稿已创建，问数模型已自动同步。" );
        return;
      }

      const changes = versionChanges(version, state.editing);
      const targetStatus = element("agent-status").value;
      const payload = {
        expected_row_version: Number(state.editing.row_version),
        ...changes,
      };
      if (displayName !== state.editing.display_name) payload.display_name = displayName;
      if (description !== (state.editing.description || null)) payload.description = description;
      if (targetStatus !== state.editing.status) payload.status = targetStatus;
      if (Object.keys(payload).length === 1) {
        closeDialog();
        toast("Agent 配置没有变化。");
        return;
      }
      await KBotKnowledgeApi.json(`/agents/${encodeURIComponent(state.editing.agent_id)}`, "PATCH", payload);
      closeDialog();
      await loadAll();
      toast("Agent 已保存。");
    } finally {
      submit.disabled = false;
    }
  }

  function findRow(agentId) {
    return state.rows.find((row) => String(row.agent_id) === String(agentId));
  }

  async function handleAction(action, agentId) {
    const row = findRow(agentId);
    if (!row) return;
    if (action === "edit") {
      const detail = await KBotKnowledgeApi.json(`/agents/${encodeURIComponent(agentId)}`, "GET");
      fillForm(detail);
      openDialog();
      return;
    }
    if (action === "activate") {
      await KBotKnowledgeApi.json(`/agents/${encodeURIComponent(agentId)}`, "PATCH", {
        expected_row_version: row.row_version,
        status: "ACTIVE",
      });
      await loadAll();
      toast("Agent 已启用。");
      return;
    }
    if (action === "disable") {
      await KBotKnowledgeApi.json(`/agents/${encodeURIComponent(agentId)}`, "PATCH", {
        expected_row_version: row.row_version,
        status: "DISABLED",
      });
      await loadAll();
      toast("Agent 已停用。");
      return;
    }
    if (action === "archive") {
      if (!globalThis.confirm(`删除将归档 Agent，不能在页面恢复，但会保留历史版本和运行记录。确认删除「${row.display_name}」？`)) return;
      await KBotKnowledgeApi.json(
        KBotKnowledgeApi.withQuery(`/agents/${encodeURIComponent(agentId)}`, { expected_row_version: row.row_version }),
        "DELETE",
      );
      await loadAll();
      toast("Agent 已删除并归档。");
    }
  }

  async function loadAll() {
    const refresh = element("agent-refresh");
    refresh.disabled = true;
    try {
      const [rows, options] = await Promise.all([
        KBotKnowledgeApi.json("/agents", "GET"),
        KBotKnowledgeApi.json("/agents/options", "GET"),
      ]);
      state.rows = Array.isArray(rows) ? rows : KBotKnowledgeApi.items(rows);
      state.options = {
        knowledge_cores: Array.isArray(options?.knowledge_cores) ? options.knowledge_cores : [],
        data_models: Array.isArray(options?.data_models) ? options.data_models : [],
        models: Array.isArray(options?.models) ? options.models : [],
      };
      renderRows();
    } finally {
      refresh.disabled = false;
    }
  }

  function showError(error) {
    toast(error?.message || "Agent 操作失败", "error");
  }

  KBotKnowledgeShell.ready.then(async (access) => {
    if (!access) return;
    element("agent-create")?.addEventListener("click", () => {
      resetForm();
      openDialog();
    });
    element("agent-refresh")?.addEventListener("click", () => loadAll().catch(showError));
    element("agent-filter")?.addEventListener("input", renderRows);
    element("agent-status-filter")?.addEventListener("change", renderRows);
    element("agent-form")?.addEventListener("submit", (event) => submitForm(event).catch(showError));
    document.querySelectorAll("[data-close-dialog='agent-dialog']").forEach((button) => button.addEventListener("click", closeDialog));
    try {
      await loadAll();
    } catch (error) {
      showError(error);
      element("agent-rows").innerHTML = emptyRow("无法加载 Agent", error.message || "请稍后重试。");
    }
  });
})();
