/* 问数模型管理：数据库连接、结构发现、语义建模、验证、策略与 Agent 绑定。 */
(function () {
  "use strict";

  const { badge, escapeHtml, toast } = KBotAssistantShell;
  const terminalValidationStatuses = new Set([
    "COMPLETED", "COMPLETED_EMPTY", "FAILED", "REJECTED",
    "TIMED_OUT", "CANCELLED",
  ]);
  const activeSnapshotStatuses = new Set([
    "REQUESTED", "DISCOVERING", "CAPTURING",
  ]);
  const state = {
    connectors: [],
    sources: [],
    sourceId: "",
    snapshot: null,
    selectedObjectIds: new Set(),
    models: [],
    modelDetails: new Map(),
    currentModel: null,
    policies: [],
    agents: [],
    catalog: [],
    manualObject: null,
    connectionRevision: 0,
    testedConnectionRevision: -1,
    snapshotPollToken: 0,
  };

  const element = (id) => document.getElementById(id);
  const items = (payload) => KBotAssistantApi.items(payload);
  const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

  function openDialog(id) { element(id)?.showModal(); }
  function closeDialog(id) { element(id)?.close(); }

  function emptyRow(columns, title, copy) {
    return `<tr><td colspan="${columns}"><div class="assistant-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div></td></tr>`;
  }

  function statusView(value) {
    const status = String(value || "UNKNOWN").toUpperCase();
    const labels = {
      ACTIVE: "已发布",
      DRAFT: "草稿",
      REVIEW: "待审核",
      RETIRED: "已停用",
      DISABLED: "已停用",
      FAILED: "失败",
      REQUESTED: "等待发现",
      DISCOVERING: "正在发现对象",
      WAITING_SELECTION: "等待选择表",
      CAPTURING: "正在采集结构",
      PARTIAL_READY: "部分可用",
      READY: "可用",
      SUPERSEDED: "历史快照",
      QUEUED: "排队中",
      RUNNING: "处理中",
      SUCCEEDED: "已完成",
      MANUAL: "人工结构",
      EXCLUDED: "未选择",
    };
    const good = new Set(["ACTIVE", "READY", "SUCCEEDED", "MANUAL"]);
    const bad = new Set(["FAILED", "REJECTED"]);
    return [labels[status] || status, good.has(status) ? "good" : bad.has(status) ? "bad" : "warn"];
  }

  function statusBadge(value) {
    const [label, tone] = statusView(value);
    return badge(label, tone);
  }

  function setWorkflowStep(step) {
    const order = ["source", "discover", "capture", "model", "publish"];
    const current = Math.max(0, order.indexOf(step));
    document.querySelectorAll("[data-workflow-step]").forEach((node) => {
      const index = order.indexOf(node.dataset.workflowStep);
      node.classList.toggle("is-active", index === current);
      node.classList.toggle("is-complete", index < current);
    });
  }

  function connectorOptions() {
    const rows = state.connectors.length ? state.connectors : [
      { source_type: "ORACLE", display_name: "Oracle" },
      { source_type: "POSTGRESQL", display_name: "PostgreSQL" },
      { source_type: "MYSQL", display_name: "MySQL" },
    ];
    element("data-source-type").innerHTML = rows.map((row) => (
      `<option value="${escapeHtml(row.source_type)}">${escapeHtml(row.display_name || row.source_type)}</option>`
    )).join("");
    setDefaultPort();
  }

  function setDefaultPort() {
    const ports = { ORACLE: 1521, POSTGRESQL: 5432, MYSQL: 3306 };
    const type = element("data-source-type")?.value;
    if (ports[type]) element("data-source-port").value = String(ports[type]);
  }

  function connectionPayload() {
    const schemas = String(element("data-source-schemas").value || "")
      .split(/[,，\s]+/).map((item) => item.trim()).filter(Boolean);
    return {
      source_type: element("data-source-type").value,
      endpoint: {
        host: element("data-source-host").value.trim(),
        port: Number(element("data-source-port").value),
        database: element("data-source-database").value.trim(),
        allowed_schemas: [...new Set(schemas)],
        tls_enabled: element("data-source-tls").checked,
      },
      credentials: {
        username: element("data-source-username").value.trim(),
        password: element("data-source-password").value,
      },
    };
  }

  function invalidateConnectionTest() {
    state.connectionRevision += 1;
    state.testedConnectionRevision = -1;
    const node = element("connection-test-status");
    node.className = "assistant-notice warn";
    node.textContent = "连接参数已变更，保存前请重新测试。";
  }

  async function testConnection() {
    const form = element("data-source-form");
    if (!form.reportValidity()) return;
    const payload = connectionPayload();
    const revision = state.connectionRevision;
    const button = element("test-data-source");
    button.disabled = true;
    element("connection-test-status").textContent = "正在测试只读连接…";
    try {
      const result = await KBotAssistantApi.json(
        "/data-models/data-sources/test-connection", "POST", payload,
      );
      if (!result?.ok) throw new Error("数据库拒绝连接或只读能力检查未通过");
      if (state.connectionRevision !== revision) {
        element("connection-test-status").textContent = "连接参数已变更，请重新测试。";
        return;
      }
      state.testedConnectionRevision = revision;
      const version = result.database_version ? `；${result.database_version}` : "";
      const node = element("connection-test-status");
      node.className = "assistant-notice";
      node.textContent = `连接测试通过${version}。凭据仅会在保存时加密传输。`;
    } catch (error) {
      state.testedConnectionRevision = -1;
      const node = element("connection-test-status");
      node.className = "assistant-notice warn";
      node.textContent = error.message || "连接测试失败";
    } finally {
      button.disabled = false;
    }
  }

  async function createSource(event) {
    event.preventDefault();
    const payload = connectionPayload();
    if (state.testedConnectionRevision !== state.connectionRevision) {
      toast("请先使用当前参数完成连接测试。", "error");
      return;
    }
    const created = await KBotAssistantApi.json(
      "/data-models/data-sources",
      "POST",
      {
        ...payload,
        display_name: element("data-source-name").value.trim(),
        auto_discover_schema: element("data-source-auto-discover").checked,
      },
    );
    element("data-source-password").value = "";
    state.testedConnectionRevision = -1;
    const testStatus = element("connection-test-status");
    testStatus.className = "assistant-notice warn";
    testStatus.textContent = "保存前必须先完成一次连接测试。";
    closeDialog("data-source-dialog");
    toast("数据库连接已保存，正在发现允许 Schema 中的表和视图。");
    await loadSources();
    await selectSource(created.data_source_id);
  }

  function renderSources() {
    const node = element("data-source-list");
    if (!state.sources.length) {
      node.innerHTML = '<div class="assistant-empty"><div><strong>尚无数据库连接</strong><p>配置只读连接并完成测试后，才能发现表和建立问数模型。</p></div></div>';
      setWorkflowStep("source");
      return;
    }
    node.innerHTML = state.sources.map((row) => {
      const current = String(row.data_source_id) === String(state.sourceId);
      return `<button type="button" class="data-source-option" data-source-id="${escapeHtml(row.data_source_id)}" aria-current="${current ? "true" : "false"}">
        <span><strong>${escapeHtml(row.display_name)}</strong><small>${escapeHtml(row.source_type)} · v${escapeHtml(row.current_version)}</small></span>
        ${statusBadge(row.status)}
      </button>`;
    }).join("");
    node.querySelectorAll("[data-source-id]").forEach((button) => {
      button.addEventListener("click", () => {
        selectSource(button.dataset.sourceId).catch(showError);
      });
    });
  }

  async function loadSources() {
    const payload = await KBotAssistantApi.json(
      "/data-models/data-sources?limit=200", "GET",
    );
    state.sources = items(payload);
    renderSources();
  }

  function snapshotSummary(snapshot) {
    if (!snapshot) return "尚未创建结构快照";
    return [
      `发现 ${snapshot.discovered_count || 0}`,
      `已选 ${snapshot.selected_count || 0}`,
      `成功 ${snapshot.succeeded_count || 0}`,
      `失败 ${snapshot.failed_count || 0}`,
    ].join(" · ");
  }

  function filteredObjects() {
    const keyword = String(element("schema-object-filter")?.value || "").trim().toLowerCase();
    const rows = state.snapshot?.objects || [];
    if (!keyword) return rows;
    return rows.filter((row) => (
      `${row.schema_name} ${row.object_name} ${row.object_type}`.toLowerCase().includes(keyword)
    ));
  }

  function renderSchemaObjects() {
    const body = element("schema-object-rows");
    const snapshot = state.snapshot;
    if (!snapshot) {
      body.innerHTML = emptyRow(7, "等待结构发现", "创建或重新发现结构快照后，将在这里选择表和视图。");
      return;
    }
    const rows = filteredObjects();
    if (!rows.length) {
      body.innerHTML = emptyRow(7, "没有匹配对象", "请调整筛选条件，或等待 Schema 发现完成。");
      return;
    }
    const selectable = snapshot.status === "WAITING_SELECTION";
    body.innerHTML = rows.map((row) => {
      const objectId = String(row.schema_snapshot_object_id);
      const checked = selectable
        ? state.selectedObjectIds.has(objectId)
        : Boolean(row.selected);
      const failed = row.status === "FAILED";
      return `<tr>
        <td><input type="checkbox" data-schema-object="${escapeHtml(objectId)}" ${checked ? "checked" : ""} ${selectable ? "" : "disabled"} aria-label="选择 ${escapeHtml(row.object_name)}"></td>
        <td>${escapeHtml(row.schema_name)}</td>
        <td><strong>${escapeHtml(row.object_name)}</strong>${row.error_code ? `<small class="data-object-error">${escapeHtml(row.error_code)}</small>` : ""}</td>
        <td>${escapeHtml(row.object_type)}</td>
        <td>${escapeHtml(row.column_count || "—")}</td>
        <td>${statusBadge(row.status)}</td>
        <td>${failed ? `<button class="small" type="button" data-retry-object="${escapeHtml(objectId)}">重试</button> <button class="small" type="button" data-manual-object="${escapeHtml(objectId)}">补录结构</button>` : "—"}</td>
      </tr>`;
    }).join("");
    body.querySelectorAll("[data-schema-object]").forEach((input) => {
      input.addEventListener("change", () => {
        if (input.checked) state.selectedObjectIds.add(input.dataset.schemaObject);
        else state.selectedObjectIds.delete(input.dataset.schemaObject);
        updateSelectionControls();
      });
    });
    body.querySelectorAll("[data-retry-object]").forEach((button) => {
      button.addEventListener("click", () => retryObject(button.dataset.retryObject).catch(showError));
    });
    body.querySelectorAll("[data-manual-object]").forEach((button) => {
      button.addEventListener("click", () => openManualDdl(button.dataset.manualObject));
    });
    updateSelectionControls();
  }

  function updateSelectionControls() {
    const selectable = state.snapshot?.status === "WAITING_SELECTION";
    const visible = selectable ? filteredObjects() : [];
    const selectedCount = visible.filter((row) => (
      state.selectedObjectIds.has(String(row.schema_snapshot_object_id))
    )).length;
    const selectAll = element("schema-select-all");
    element("confirm-schema-selection").disabled = !selectable || !state.selectedObjectIds.size;
    selectAll.disabled = !selectable || !visible.length;
    selectAll.checked = Boolean(visible.length && selectedCount === visible.length);
    selectAll.indeterminate = selectedCount > 0 && selectedCount < visible.length;
  }

  function renderSnapshot() {
    const snapshot = state.snapshot;
    const source = state.sources.find((row) => String(row.data_source_id) === String(state.sourceId));
    element("schema-panel-title").textContent = source ? `${source.display_name} · Schema` : "Schema 与表对象";
    element("schema-panel-copy").textContent = snapshot
      ? `快照 ${snapshot.schema_snapshot_id} · ${snapshotSummary(snapshot)}`
      : "当前连接尚无结构快照，可立即发起发现。";
    element("schema-summary").innerHTML = snapshot
      ? `<div>${statusBadge(snapshot.status)}<span>${escapeHtml(snapshotSummary(snapshot))}</span></div><small>${escapeHtml(snapshot.completed_at || snapshot.created_at || "等待处理")}</small>`
      : "";
    element("request-snapshot").disabled = !state.sourceId || activeSnapshotStatuses.has(snapshot?.status);
    element("open-model-draft").disabled = !["READY", "PARTIAL_READY"].includes(snapshot?.status);
    if (snapshot?.status === "WAITING_SELECTION") setWorkflowStep("capture");
    else if (["READY", "PARTIAL_READY"].includes(snapshot?.status)) setWorkflowStep("model");
    else if (snapshot) setWorkflowStep("discover");
    renderSchemaObjects();
  }

  async function loadSnapshot(snapshotId, pollToken) {
    const snapshot = await KBotAssistantApi.json(
      `/data-models/snapshots/${encodeURIComponent(snapshotId)}`, "GET",
    );
    if (pollToken !== undefined && pollToken !== state.snapshotPollToken) return;
    const changed = String(state.snapshot?.schema_snapshot_id || "") !== String(snapshot.schema_snapshot_id);
    state.snapshot = snapshot;
    if (changed) {
      state.selectedObjectIds = new Set(
        (snapshot.objects || []).filter((row) => row.selected).map((row) => String(row.schema_snapshot_object_id)),
      );
    }
    renderSnapshot();
    if (activeSnapshotStatuses.has(snapshot.status)) scheduleSnapshotPoll(snapshot.schema_snapshot_id);
  }

  function scheduleSnapshotPoll(snapshotId) {
    const token = ++state.snapshotPollToken;
    setTimeout(() => {
      loadSnapshot(snapshotId, token).catch((error) => toast(error.message || "结构进度刷新失败", "error"));
    }, 2500);
  }

  async function selectSource(sourceId) {
    state.sourceId = String(sourceId || "");
    state.snapshot = null;
    state.selectedObjectIds.clear();
    state.snapshotPollToken += 1;
    renderSources();
    renderSnapshot();
    if (!state.sourceId) return;
    const payload = await KBotAssistantApi.json(
      `/data-models/data-sources/${encodeURIComponent(state.sourceId)}/snapshots`, "GET",
    );
    const snapshots = items(payload);
    if (snapshots.length) await loadSnapshot(snapshots[0].schema_snapshot_id);
  }

  async function requestSnapshot() {
    if (!state.sourceId) return;
    const result = await KBotAssistantApi.json(
      `/data-models/data-sources/${encodeURIComponent(state.sourceId)}/snapshots`, "POST", {},
    );
    toast("已发起 Schema 发现。");
    await loadSnapshot(result.schema_snapshot_id);
  }

  async function confirmSelection() {
    if (!state.snapshot || !state.selectedObjectIds.size) return;
    await KBotAssistantApi.json(
      `/data-models/snapshots/${encodeURIComponent(state.snapshot.schema_snapshot_id)}/selection`,
      "POST",
      { object_ids: [...state.selectedObjectIds] },
    );
    toast(`已选择 ${state.selectedObjectIds.size} 个对象，正在采集字段结构。`);
    await loadSnapshot(state.snapshot.schema_snapshot_id);
  }

  async function retryObject(objectId) {
    await KBotAssistantApi.json(
      `/data-models/snapshots/${encodeURIComponent(state.snapshot.schema_snapshot_id)}/objects/${encodeURIComponent(objectId)}/retry`,
      "POST", {},
    );
    toast("失败对象已重新排队。");
    await loadSnapshot(state.snapshot.schema_snapshot_id);
  }

  function openManualDdl(objectId) {
    state.manualObject = (state.snapshot?.objects || []).find(
      (row) => String(row.schema_snapshot_object_id) === String(objectId),
    );
    if (!state.manualObject) return;
    element("manual-ddl-object").textContent = `仅补录 ${state.manualObject.schema_name}.${state.manualObject.object_name} 的结构。`;
    element("manual-ddl").value = "";
    openDialog("manual-ddl-dialog");
  }

  async function saveManualDdl(event) {
    event.preventDefault();
    if (!state.manualObject || !state.snapshot) return;
    await KBotAssistantApi.json(
      `/data-models/snapshots/${encodeURIComponent(state.snapshot.schema_snapshot_id)}/objects/${encodeURIComponent(state.manualObject.schema_snapshot_object_id)}/manual-ddl`,
      "POST",
      { ddl: element("manual-ddl").value.trim() },
    );
    closeDialog("manual-ddl-dialog");
    toast("人工结构已保存，系统不会执行该 DDL。");
    await loadSnapshot(state.snapshot.schema_snapshot_id);
  }

  function llmModels() {
    return state.catalog.filter((row) => (
      KBotAssistantApi.isActiveModel(row)
      && KBotAssistantApi.modelCategory(row) === KBotAssistantApi.ModelCategory.LLM
    ));
  }

  function fillLlmSelect(id, emptyLabel) {
    const node = element(id);
    const rows = llmModels();
    node.innerHTML = [
      `<option value="">${escapeHtml(emptyLabel)}</option>`,
      ...rows.map((row) => `<option value="${escapeHtml(row.model_id)}">${escapeHtml(row.display_name || row.served_model_name || row.model_id)}</option>`),
    ].join("");
  }

  function openModelDraft() {
    if (!state.snapshot) return;
    const usable = (state.snapshot.objects || []).filter((row) => (
      row.selected && ["READY", "MANUAL"].includes(row.status)
    ));
    element("data-model-selection-copy").textContent = `将使用 ${usable.length} 个已采集对象生成草稿；金额口径、状态过滤和敏感级别仍需人工复核。`;
    fillLlmSelect("data-model-ai", "不使用 AI，仅按字段结构生成");
    openDialog("data-model-dialog");
  }

  async function generateModel(event) {
    event.preventDefault();
    const aiModelId = element("data-model-ai").value;
    if (aiModelId && !element("data-model-ai-consent").checked) {
      toast("使用 AI 增强前必须明确同意发送结构元数据。", "error");
      return;
    }
    const objectIds = (state.snapshot.objects || [])
      .filter((row) => row.selected && ["READY", "MANUAL"].includes(row.status))
      .map((row) => row.schema_snapshot_object_id);
    const receipt = await KBotAssistantApi.json(
      `/data-models/snapshots/${encodeURIComponent(state.snapshot.schema_snapshot_id)}/semantic-model-draft`,
      "POST",
      {
        display_name: element("data-model-name").value.trim(),
        description: element("data-model-description").value.trim() || null,
        business_context: element("data-model-context").value.trim() || null,
        object_ids: objectIds.length <= 64 ? objectIds : [],
        ai_model_id: aiModelId || null,
        allow_ai_metadata: Boolean(aiModelId),
      },
    );
    closeDialog("data-model-dialog");
    toast("语义模型草稿已进入生成队列。");
    await waitForGeneration(receipt.generation_job_id);
  }

  async function waitForGeneration(jobId) {
    for (let attempt = 0; attempt < 80; attempt += 1) {
      const job = await KBotAssistantApi.json(
        `/data-models/generation-jobs/${encodeURIComponent(jobId)}`, "GET",
      );
      if (job.status === "SUCCEEDED") {
        toast("语义模型草稿已生成，请复核定义并运行测试问题。");
        await loadModels();
        if (job.semantic_model_id) await openSemanticModel(job.semantic_model_id);
        return;
      }
      if (job.status === "FAILED") throw new Error(job.error_code || "语义模型生成失败");
      await delay(1500);
    }
    throw new Error("语义模型生成仍在处理中，请稍后刷新页面。 ");
  }

  function latestVersion(detail) {
    return [...(detail?.versions || [])].sort(
      (left, right) => Number(right.version_no || 0) - Number(left.version_no || 0),
    )[0] || null;
  }

  function sourceName(sourceId) {
    return state.sources.find((row) => String(row.data_source_id) === String(sourceId))?.display_name || "—";
  }

  function policyCount(modelId) {
    return state.policies.filter((row) => (
      row.status === "ACTIVE"
      && (row.semantic_model_ids || []).map(String).includes(String(modelId))
    )).length;
  }

  function renderModels() {
    const body = element("data-model-rows");
    element("data-model-count").textContent = `${state.models.length} 个模型`;
    if (!state.models.length) {
      body.innerHTML = emptyRow(7, "尚无语义数据模型", "完成数据库连接、对象选择和结构采集后，点击“建立数据模型”。");
      return;
    }
    body.innerHTML = state.models.map((row) => {
      const detail = state.modelDetails.get(String(row.semantic_model_id)) || row;
      const version = latestVersion(detail);
      const datasets = version?.definition?.datasets?.length || 0;
      return `<tr>
        <td><strong>${escapeHtml(row.display_name)}</strong><small class="data-model-description">${escapeHtml(row.description || "未填写说明")}</small></td>
        <td>${escapeHtml(sourceName(version?.data_source_id))}</td>
        <td>${version ? `v${escapeHtml(version.version_no)}` : "—"}</td>
        <td>${statusBadge(version?.status || "DRAFT")}</td>
        <td>${escapeHtml(datasets)}</td>
        <td>${escapeHtml(policyCount(row.semantic_model_id))}</td>
        <td><button type="button" class="small" data-open-model="${escapeHtml(row.semantic_model_id)}">配置</button></td>
      </tr>`;
    }).join("");
    body.querySelectorAll("[data-open-model]").forEach((button) => {
      button.addEventListener("click", () => openSemanticModel(button.dataset.openModel).catch(showError));
    });
    if (state.models.some((row) => latestVersion(state.modelDetails.get(String(row.semantic_model_id)))?.status === "ACTIVE")) {
      setWorkflowStep("publish");
    }
  }

  async function loadModels() {
    const payload = await KBotAssistantApi.json("/data-models?limit=200", "GET");
    state.models = items(payload);
    const details = await Promise.all(state.models.map(async (row) => {
      try {
        return await KBotAssistantApi.json(`/data-models/${encodeURIComponent(row.semantic_model_id)}`, "GET");
      } catch (_) {
        return row;
      }
    }));
    state.modelDetails = new Map(details.map((detail) => [String(detail.semantic_model_id), detail]));
    renderModels();
  }

  function updateModelActions(version) {
    const status = version?.status || "";
    element("save-model-definition").disabled = status !== "DRAFT";
    element("semantic-model-definition").readOnly = status !== "DRAFT";
    element("submit-model-review").hidden = status !== "DRAFT";
    element("publish-model").hidden = status !== "REVIEW";
    element("configure-model-policy").disabled = status !== "ACTIVE";
    element("bind-model-agent").disabled = status !== "ACTIVE" || policyCount(version?.semantic_model_id) === 0;
    element("run-model-validation").disabled = !["DRAFT", "REVIEW"].includes(status) || !llmModels().length;
  }

  async function openSemanticModel(modelId) {
    const detail = await KBotAssistantApi.json(`/data-models/${encodeURIComponent(modelId)}`, "GET");
    state.modelDetails.set(String(modelId), detail);
    state.currentModel = { detail, version: latestVersion(detail) };
    const version = state.currentModel.version;
    element("semantic-model-title").textContent = detail.display_name;
    element("semantic-model-meta").textContent = version
      ? `v${version.version_no} · ${statusView(version.status)[0]} · ${sourceName(version.data_source_id)}`
      : "尚无模型版本";
    element("semantic-model-definition").value = JSON.stringify(version?.definition || {}, null, 2);
    element("validation-result").innerHTML = "";
    element("validation-status").className = "assistant-badge";
    element("validation-status").textContent = "尚未验证";
    fillLlmSelect("validation-model", "选择一个已启用的 LLM");
    updateModelActions(version);
    openDialog("semantic-model-dialog");
  }

  async function refreshCurrentModel() {
    const modelId = state.currentModel?.detail?.semantic_model_id;
    if (!modelId) return;
    await loadModels();
    await openSemanticModel(modelId);
  }

  async function saveDefinition() {
    const { detail, version } = state.currentModel || {};
    if (!version || version.status !== "DRAFT") return;
    let definition;
    try {
      definition = JSON.parse(element("semantic-model-definition").value);
    } catch (_) {
      toast("模型定义不是合法 JSON。", "error");
      return;
    }
    await KBotAssistantApi.json(
      `/data-models/${encodeURIComponent(detail.semantic_model_id)}/versions/${encodeURIComponent(version.semantic_model_version_id)}`,
      "PATCH",
      { definition, expected_row_version: version.row_version },
    );
    toast("模型定义草稿已保存。");
    await refreshCurrentModel();
  }

  function renderValidation(result) {
    const node = element("validation-result");
    const columns = (result.columns || []).map((column) => column.name || column.label || column.column_name).filter(Boolean);
    const rows = result.preview_rows || [];
    if (!rows.length) {
      node.innerHTML = `<div class="assistant-notice ${result.error_code ? "warn" : ""}">${escapeHtml(result.error_code || "验证完成，但结果为空。")}</div>`;
      return;
    }
    const names = columns.length ? columns : Object.keys(rows[0] || {});
    node.innerHTML = `<div class="assistant-table-wrap"><table class="assistant-table validation-table"><thead><tr>${names.map((name) => `<th>${escapeHtml(name)}</th>`).join("")}</tr></thead><tbody>${rows.slice(0, 20).map((row) => `<tr>${names.map((name) => `<td>${escapeHtml(row[name] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div><small>${escapeHtml(result.row_count ?? rows.length)} 行${result.truncated ? " · 已截断" : ""}</small>`;
  }

  async function runValidation() {
    const { detail, version } = state.currentModel || {};
    const question = element("validation-question").value.trim();
    const aiModelId = element("validation-model").value;
    if (question.length < 2 || !aiModelId) {
      toast("请填写测试问题并选择规划模型。", "error");
      return;
    }
    if (!element("validation-ai-consent").checked) {
      toast("运行验证前必须同意发送语义目录与测试问题。", "error");
      return;
    }
    element("validation-status").textContent = "正在验证";
    element("validation-status").className = "assistant-badge warn";
    const receipt = await KBotAssistantApi.json(
      `/data-models/${encodeURIComponent(detail.semantic_model_id)}/versions/${encodeURIComponent(version.semantic_model_version_id)}/validations`,
      "POST",
      { question, ai_model_id: aiModelId, allow_ai_metadata: true },
      { headers: { "Idempotency-Key": KBotAssistantApi.requestId() } },
    );
    for (let attempt = 0; attempt < 80; attempt += 1) {
      const result = await KBotAssistantApi.json(
        `/data-models/${encodeURIComponent(detail.semantic_model_id)}/versions/${encodeURIComponent(version.semantic_model_version_id)}/validations/${encodeURIComponent(receipt.data_query_run_id)}`,
        "GET",
      );
      if (terminalValidationStatuses.has(result.status)) {
        const successful = ["COMPLETED", "COMPLETED_EMPTY"].includes(result.status);
        element("validation-status").textContent = successful ? "验证通过" : "验证失败";
        element("validation-status").className = `assistant-badge ${successful ? "good" : "bad"}`;
        renderValidation(result);
        return;
      }
      await delay(1500);
    }
    throw new Error("验证仍在处理中，请稍后重新打开模型查看。 ");
  }

  async function submitReview() {
    const { detail, version } = state.currentModel || {};
    await KBotAssistantApi.json(
      `/data-models/${encodeURIComponent(detail.semantic_model_id)}/versions/${encodeURIComponent(version.semantic_model_version_id)}/submit-review`,
      "POST", { expected_row_version: version.row_version },
    );
    toast("模型已提交审核。发布仍需单独确认。");
    await refreshCurrentModel();
  }

  async function publishModel() {
    const { detail, version } = state.currentModel || {};
    await KBotAssistantApi.json(
      `/data-models/${encodeURIComponent(detail.semantic_model_id)}/versions/${encodeURIComponent(version.semantic_model_version_id)}/publish`,
      "POST",
      {
        schema_snapshot_id: version.schema_snapshot_id,
        expected_row_version: version.row_version,
      },
    );
    toast("语义数据模型已发布，可以配置策略和 Agent Binding。");
    await refreshCurrentModel();
  }

  function renderPolicySubjects(subjects) {
    const members = subjects.members || [];
    const roles = subjects.roles || [];
    element("policy-members").innerHTML = members.length
      ? members.map((row) => `<label><input type="checkbox" name="policy-member" value="${escapeHtml(row.id)}"> <span>${escapeHtml(row.display_name || row.username || row.id)}</span></label>`).join("")
      : "<p>当前 Domain 没有可选用户。</p>";
    element("policy-roles").innerHTML = roles.length
      ? roles.map((row) => `<label><input type="checkbox" name="policy-role" value="${escapeHtml(row.code)}"> <span>${escapeHtml(row.display_name || row.code)}</span></label>`).join("")
      : "<p>当前 App 没有可选角色。</p>";
  }

  async function openPolicy() {
    const version = state.currentModel?.version;
    if (version?.status !== "ACTIVE") return;
    const subjects = await KBotAssistantApi.json("/data-models/policy-subjects", "GET");
    renderPolicySubjects(subjects);
    openDialog("policy-dialog");
  }

  function checkedValues(name) {
    return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((node) => node.value);
  }

  async function createPolicy(event) {
    event.preventDefault();
    const modelId = state.currentModel?.detail?.semantic_model_id;
    const actorIds = checkedValues("policy-member");
    const roles = checkedValues("policy-role");
    if (!actorIds.length && !roles.length) {
      toast("策略至少需要选择一个用户或角色。", "error");
      return;
    }
    await KBotAssistantApi.json(
      "/data-models/policy-bindings", "POST",
      {
        semantic_model_ids: [modelId],
        actor_ids: actorIds,
        roles,
        budget: {
          max_rows: Number(element("policy-max-rows").value),
          max_result_bytes: Number(element("policy-max-bytes").value),
          statement_timeout_seconds: Number(element("policy-timeout").value),
          max_concurrent_runs: Number(element("policy-concurrency").value),
        },
      },
    );
    closeDialog("policy-dialog");
    toast("问数策略已创建。");
    await loadPolicies();
    await loadModels();
    updateModelActions(state.currentModel?.version);
  }

  async function loadPolicies() {
    const payload = await KBotAssistantApi.json("/data-models/policy-bindings?limit=200", "GET");
    state.policies = items(payload);
  }

  async function loadAgents() {
    const payload = await KBotAssistantApi.json("/data-models/agents", "GET");
    state.agents = Array.isArray(payload) ? payload : items(payload);
  }

  async function openAgentBinding() {
    const modelId = state.currentModel?.detail?.semantic_model_id;
    const draftAgents = state.agents.filter((row) => row.status === "DRAFT");
    const policies = state.policies.filter((row) => (
      row.status === "ACTIVE"
      && (row.semantic_model_ids || []).map(String).includes(String(modelId))
    ));
    element("query-binding-agent").innerHTML = draftAgents.length
      ? draftAgents.map((row) => `<option value="${escapeHtml(row.agent_id)}">${escapeHtml(row.display_name || row.agent_id)} · 草稿</option>`).join("")
      : '<option value="">没有可绑定的草稿 Agent</option>';
    element("query-binding-policy").innerHTML = policies.length
      ? policies.map((row) => `<option value="${escapeHtml(row.policy_binding_id)}">策略 ${escapeHtml(row.policy_binding_id)} · ${escapeHtml(row.status)}</option>`).join("")
      : '<option value="">请先创建有效策略</option>';
    openDialog("agent-query-binding-dialog");
  }

  async function createAgentBinding(event) {
    event.preventDefault();
    const agentId = element("query-binding-agent").value;
    const policyId = element("query-binding-policy").value;
    if (!agentId || !policyId) {
      toast("请选择 Agent 和有效策略。", "error");
      return;
    }
    await KBotAssistantApi.json(
      "/data-models/agent-bindings", "POST",
      {
        agent_id: agentId,
        semantic_model_id: state.currentModel.detail.semantic_model_id,
        policy_binding_id: policyId,
      },
    );
    closeDialog("agent-query-binding-dialog");
    toast("Agent 问数绑定已创建，请在 Agent 页面确认模型范围并启用。");
  }

  async function loadReferenceData() {
    const [connectors, catalog, policies, agents] = await Promise.all([
      KBotAssistantApi.json("/data-models/connector-capabilities", "GET"),
      KBotAssistantApi.json("/model-catalog", "GET"),
      KBotAssistantApi.json("/data-models/policy-bindings?limit=200", "GET"),
      KBotAssistantApi.json("/data-models/agents", "GET"),
    ]);
    state.connectors = items(connectors);
    state.catalog = items(catalog);
    state.policies = items(policies);
    state.agents = Array.isArray(agents) ? agents : items(agents);
    connectorOptions();
  }

  async function loadPage() {
    await loadReferenceData();
    await loadSources();
    await loadModels();
    if (state.sourceId && state.sources.some((row) => String(row.data_source_id) === state.sourceId)) {
      await selectSource(state.sourceId);
    } else if (state.sources.length) {
      await selectSource(state.sources[0].data_source_id);
    }
  }

  function showError(error) {
    toast(error?.message || "问数模型操作失败", "error");
  }

  function bindEvents() {
    document.querySelectorAll("[data-open-dialog]").forEach((button) => {
      button.addEventListener("click", () => openDialog(button.dataset.openDialog));
    });
    document.querySelectorAll("[data-close-dialog]").forEach((button) => {
      button.addEventListener("click", () => closeDialog(button.dataset.closeDialog));
    });
    element("data-source-type").addEventListener("change", () => {
      setDefaultPort();
      invalidateConnectionTest();
    });
    [
      "data-source-type",
      "data-source-host",
      "data-source-port",
      "data-source-database",
      "data-source-schemas",
      "data-source-username",
      "data-source-password",
      "data-source-tls",
    ].forEach((id) => {
      element(id).addEventListener("input", invalidateConnectionTest);
    });
    element("data-source-form").addEventListener("submit", (event) => createSource(event).catch(showError));
    element("test-data-source").addEventListener("click", () => testConnection().catch(showError));
    element("request-snapshot").addEventListener("click", () => requestSnapshot().catch(showError));
    element("schema-object-filter").addEventListener("input", renderSchemaObjects);
    element("schema-select-all").addEventListener("change", (event) => {
      filteredObjects().forEach((row) => {
        const id = String(row.schema_snapshot_object_id);
        if (event.target.checked) state.selectedObjectIds.add(id);
        else state.selectedObjectIds.delete(id);
      });
      renderSchemaObjects();
    });
    element("confirm-schema-selection").addEventListener("click", () => confirmSelection().catch(showError));
    element("open-model-draft").addEventListener("click", openModelDraft);
    element("data-model-form").addEventListener("submit", (event) => generateModel(event).catch(showError));
    element("save-model-definition").addEventListener("click", () => saveDefinition().catch(showError));
    element("run-model-validation").addEventListener("click", () => runValidation().catch(showError));
    element("submit-model-review").addEventListener("click", () => submitReview().catch(showError));
    element("publish-model").addEventListener("click", () => publishModel().catch(showError));
    element("configure-model-policy").addEventListener("click", () => openPolicy().catch(showError));
    element("policy-form").addEventListener("submit", (event) => createPolicy(event).catch(showError));
    element("bind-model-agent").addEventListener("click", () => openAgentBinding().catch(showError));
    element("agent-query-binding-form").addEventListener("submit", (event) => createAgentBinding(event).catch(showError));
    element("manual-ddl-form").addEventListener("submit", (event) => saveManualDdl(event).catch(showError));
    element("data-model-refresh").addEventListener("click", () => loadPage().catch(showError));
  }

  KBotAssistantShell.ready.then(async (access) => {
    if (!access) return;
    bindEvents();
    try {
      await loadPage();
    } catch (error) {
      showError(error);
      element("data-source-list").innerHTML = '<div class="assistant-empty"><div><strong>无法读取问数配置</strong><p>请确认 Data Query API 与 Worker 均已启动。</p></div></div>';
      element("data-model-rows").innerHTML = emptyRow(7, "无法读取语义模型", "请检查 Data Query 服务状态和当前 Domain 权限。");
    }
  });
})();
