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
    query_ocr: "agent-query-ocr",
    query_vlm: "agent-query-vlm",
  });
  const ROUTER_THRESHOLDS = [
    [0.4, "较宽松（40%）"], [0.6, "标准（60%，默认）"], [0.8, "较严格（80%）"],
  ];
  const state = {
    rows: [],
    options: { libraries: [], data_models: [], models: [] },
    editing: null,
  };

  function element(id) { return document.getElementById(id); }
  function openDialog() { element("agent-dialog")?.showModal(); }
  function closeDialog() { element("agent-dialog")?.close(); }

  function emptyRow(title, copy) {
    return `<tr><td colspan="5" class="empty-cell">${escapeHtml(title)}</td></tr>`;
  }

  function statusLabel(status) {
    return ({ DRAFT: "草稿", ACTIVE: "已启用", DISABLED: "已停用", ARCHIVED: "已归档" })[status] || status || "未知";
  }

  function statusTone(status) {
    if (status === "ACTIVE") return "good";
    if (status === "ARCHIVED") return "bad";
    return "warn";
  }

  function actionButtons(row) {
    const id = escapeHtml(row.agent_id);
    if (row.status === "ARCHIVED") return '<span class="knowledge-badge">只读</span>';
    const button = (action, label, danger = false) =>
      `<button class="${danger ? "danger-row-action" : "row-action"}" type="button" data-agent-action="${action}" data-agent-id="${id}">${label}</button>`;
    const lifecycle = row.status === "ACTIVE"
      ? button("disable", "停用", true)
      : button("activate", "启用");
    return `<div class="knowledge-row-actions">
      ${button("edit", "编辑")}
      ${lifecycle}
      ${row.status === "ACTIVE" ? "" : button("archive", "删除", true)}
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
        state.rows.length ? "请调整搜索条件或状态筛选。" : "创建草稿并配置 资料库、运行模型和可选的问数模型。",
      );
      return;
    }
    node.innerHTML = rows.map((row) => `<tr>
      <td><strong class="table-primary">${escapeHtml(row.display_name || "未命名")}</strong></td>
      <td>${escapeHtml(row.description || "—")}</td>
      <td><div class="tag-list">${(row.enabled_capabilities || []).map(key => `<span class="tag">${escapeHtml(({KNOWLEDGE_CHAT:"知识检索",DOCUMENT:"知识检索",DATA_QUERY:"数据查询",CONVERSATION:"通用对话",X_SEARCH:"X 实时搜索"})[key] || key)}</span>`).join("") || '<span class="tag">待配置</span>'}</div></td>
      <td>${badge(statusLabel(row.status), statusTone(row.status))}</td>
      <td>${actionButtons(row)}</td>
    </tr>`).join("");
    node.querySelectorAll("[data-agent-action]").forEach((button) => {
      button.addEventListener("click", () => handleAction(button.dataset.agentAction, button.dataset.agentId).catch(showError));
    });
  }

  function modelLabel(row) {
    return row.display_name || row.provider_model_name || row.model_id;
  }

  function setSelectOptions(select, rows, currentId, emptyLabel) {
    const known = rows.some((row) => String(row.model_id) === String(currentId || ""));
    const options = [`<option value="">${escapeHtml(emptyLabel)}</option>`];
    options.push(...rows.map((row) => `<option value="${escapeHtml(row.model_id)}"${String(row.model_id) === String(currentId || "") ? " selected" : ""}>${escapeHtml(modelLabel(row))}</option>`));
    if (currentId && !known) options.push(`<option value="${escapeHtml(currentId)}" selected>${escapeHtml(currentId)}（当前绑定，目录不可用）</option>`);
    select.innerHTML = options.join("");
  }

  function fillResourceOptions(current) {
    const currentCore = current?.library_id || "";
    element("agent-knowledge-core").innerHTML = ['<option value="">暂不绑定</option>'].concat(
      state.options.libraries.map((row) => {
        const selected = String(row.library_id) === String(currentCore) ? " selected" : "";
        const unavailable = !row.selectable && !selected ? " disabled" : "";
        const suffix = row.selectable ? "" : ` · ${row.status || "不可用"}`;
        return `<option value="${escapeHtml(row.library_id)}"${selected}${unavailable}>${escapeHtml((row.display_name || row.library_id) + suffix)}</option>`;
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
    const categories = { query_ocr: KBotKnowledgeApi.ModelCategory.OCR, query_vlm: KBotKnowledgeApi.ModelCategory.VLM };
    Object.entries(MODEL_ROLES).forEach(([role, id]) => {
      const imageModel = Object.hasOwn(categories, role);
      const rows = imageModel ? state.options.models.filter(row => KBotKnowledgeApi.modelCategory(row) === categories[role])
        : role === "memory_embedding" ? embeddings : role === "x_search_llm" ? grokModels : llms;
      const placeholder = role === "data_planner_llm" ? "跟随回答生成 LLM"
        : imageModel && !rows.length ? "没有可选的已启用模型" : "不绑定";
      setSelectOptions(element(id), rows, current?.models?.[role], placeholder);
    });
  }

  function objectValue(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function configuredThreshold(config) {
    const raw = config.router_confidence_threshold;
    const number = typeof raw === "number" || (typeof raw === "string" && raw.trim()) ? Number(raw) : NaN;
    return Number.isFinite(number) ? Math.min(1, Math.max(0, number)) : 0.6;
  }

  function configuredMemory(config) {
    const value = objectValue(config.memory).episodic_enabled;
    return typeof value === "boolean" ? value : true;
  }

  function fillRuntimeOptions(config) {
    const base = objectValue(config);
    const threshold = configuredThreshold(base), select = element("agent-router-threshold");
    select.replaceChildren(...ROUTER_THRESHOLDS.map(([value, name]) => new Option(name, String(value))));
    if (!ROUTER_THRESHOLDS.some(([value]) => value === threshold)) {
      select.add(new Option(`当前自定义（${new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 2 }).format(threshold)}）`, String(threshold)));
    }
    select.value = String(threshold);
    element("agent-episodic-memory").value = String(configuredMemory(base));
  }

  function selectedConfig() {
    const base = objectValue(state.editing?.config);
    const selected = { ...base };
    const threshold = Number(element("agent-router-threshold").value);
    const episodic = element("agent-episodic-memory").value === "true";
    if (threshold !== configuredThreshold(base)) selected.router_confidence_threshold = threshold;
    if (episodic !== configuredMemory(base)) selected.memory = { ...objectValue(base.memory), episodic_enabled: episodic };
    return selected;
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
    fillRuntimeOptions(null);
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
    fillRuntimeOptions(row.config);
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

  function stable(value) {
    if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
    if (value && typeof value === "object") return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
    return JSON.stringify(value);
  }

  function versionValues() {
    return {
      library_id: element("agent-knowledge-core").value || null,
      data_model_ids: selectedDataModels(),
      models: selectedModels(),
      instruction: String(element("agent-instruction").value || "").trim() || null,
      config: selectedConfig(),
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
    const version = versionValues();
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
      if (!await KBotKnowledgeShell.confirmAction({ title: "删除 Agent？", message: `删除将归档 Agent，不能在页面恢复，但会保留历史版本和运行记录。确认删除「${row.display_name}」？`, confirmLabel: "删除 Agent", tone: "danger" })) return;
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
        libraries: Array.isArray(options?.libraries) ? options.libraries : [],
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
    const help = element("agent-x-search-help"), tooltip = element("agent-x-search-tooltip");
    const helpLabel = help.parentElement;
    const resetHelp = () => { delete helpLabel.dataset.helpDismissed; };
    helpLabel.addEventListener("pointerleave", resetHelp);
    help.addEventListener("blur", resetHelp);
    element("agent-dialog").addEventListener("keydown", event => {
      if (event.key !== "Escape" || getComputedStyle(tooltip).visibility !== "visible") return;
      event.preventDefault();
      helpLabel.dataset.helpDismissed = "";
    });
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
