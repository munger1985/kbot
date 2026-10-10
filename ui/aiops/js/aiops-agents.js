(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let agents = [];
  let targets = [];
  let models = [];
  const actionDraftsByTarget = new Map();
  const actionCatalogsByTarget = new Map();
  let editing = null;

  const escape = (value) => shell.escape(value ?? "—");
  const targetName = (id) => targets.find((item) => item.target_id === id)?.display_name || shell.short(id);

  const scopeLabels = {
    schemas: "Schema",
    dynamic_parameters: "动态参数及允许值",
    resource_manager_plans: "Resource Manager Plan",
    privilege_grantees: "允许授权的本地用户",
    system_privileges: "系统权限",
    object_privileges: "对象权限",
  };

  function uniqueValues(...groups) {
    return [...new Set(groups.flat().map((value) => String(value || "").trim()).filter(Boolean))]
      .sort((left, right) => left.localeCompare(right, "zh"));
  }

  function groupedActions(catalog) {
    const grouped = new Map();
    (catalog.actions || [])
      .filter((item) => item.execution_mode === "EXECUTABLE_AFTER_APPROVAL")
      .forEach((action) => {
        const current = grouped.get(action.action_id);
        if (!current) {
          grouped.set(action.action_id, { ...action, scope_requirements: [...(action.scope_requirements || [])] });
          return;
        }
        current.currently_executable = current.currently_executable || action.currently_executable;
        current.scope_requirements = uniqueValues(current.scope_requirements, action.scope_requirements || []);
      });
    return [...grouped.values()];
  }

  function optionPicker(kind, label, options, selected, help) {
    const normalizedOptions = (options || []).map((value) => String(value).toUpperCase());
    const normalizedSelected = (selected || []).map((value) => String(value).toUpperCase());
    const values = uniqueValues(normalizedOptions, normalizedSelected);
    const selectedValues = new Set(normalizedSelected);
    const choices = values.length
      ? values.map((value) => `<label class="agent-scope-option" data-option-label="${escape(value.toLocaleLowerCase())}">
          <input type="checkbox" data-action-scope="${escape(kind)}" value="${escape(value)}" ${selectedValues.has(value) ? "checked" : ""}>
          <span>${escape(value)}</span>
        </label>`).join("")
      : '<p class="agent-scope-empty">当前 Target 没有可选项。请先在 Target 页面执行连接测试，刷新数据库目录。</p>';
    const search = values.length > 8
      ? `<input class="agent-scope-search" type="search" data-scope-search="${escape(kind)}" placeholder="搜索${escape(label)}" aria-label="搜索${escape(label)}">`
      : "";
    return `<section class="agent-scope-panel" data-scope-kind="${escape(kind)}" hidden>
      <div class="agent-scope-title"><div><strong>${escape(label)}</strong><small>${escape(help)}</small></div><span data-scope-count>已选 ${selectedValues.size} 项</span></div>
      ${search}<div class="agent-scope-options">${choices}</div>
    </section>`;
  }

  function dynamicParameterPicker(options, configured) {
    const configuredByName = new Map((configured || []).map((item) => [String(item.name).toLowerCase(), (item.allowed_values || []).map((value) => String(value).toUpperCase())]));
    const catalogByName = new Map((options || []).map((item) => [String(item.name).toLowerCase(), (item.allowed_values || []).map((value) => String(value).toUpperCase())]));
    configuredByName.forEach((values, name) => catalogByName.set(name, uniqueValues(catalogByName.get(name) || [], values)));
    const rows = [...catalogByName.entries()].map(([name, values]) => {
      const selected = new Set(configuredByName.get(name) || []);
      return `<div class="agent-parameter-row" data-dynamic-parameter="${escape(name)}">
        <strong>${escape(name)}</strong>
        <div class="agent-parameter-values">${values.map((value) => `<label><input type="checkbox" data-dynamic-value value="${escape(value)}" ${selected.has(value) ? "checked" : ""}><span>${escape(value)}</span></label>`).join("")}</div>
      </div>`;
    }).join("");
    return `<section class="agent-scope-panel" data-scope-kind="dynamic_parameters" hidden>
      <div class="agent-scope-title"><div><strong>动态参数及允许值</strong><small>逐个选择参数和值，未选择的值不会获得授权。</small></div><span data-scope-count>已选 0 项</span></div>
      <div class="agent-parameter-list">${rows || '<p class="agent-scope-empty">动作目录没有可授权的动态参数。</p>'}</div>
    </section>`;
  }
  function showResult(message = "", tone = "") {
    const result = document.getElementById("agent-result");
    result.textContent = message;
    result.dataset.tone = tone;
  }

  function renderSummary() {
    const active = agents.filter((item) => item.status === "ACTIVE").length;
    document.getElementById("agent-summary").innerHTML = [
      `<span><strong>${agents.length}</strong> 全部</span>`,
      `<span><strong>${active}</strong> 已启用</span>`,
      `<span><strong>${targets.length}</strong> 可用 Target</span>`,
    ].join("");
  }

  function renderRows() {
    const body = document.getElementById("agent-rows");
    renderSummary();
    if (!agents.length) {
      body.innerHTML = '<tr><td class="ops-empty" colspan="6">当前范围内暂无 Agent</td></tr>';
      return;
    }
    body.innerHTML = agents.map((agent) => {
      const targetNames = (agent.target_ids || []).map((id) => escape(targetName(id)));
      const actionCount = (agent.controlled_action_execution || []).reduce((count, item) => count + (item.allowed_action_ids || []).length, 0);
      const access = actionCount ? `诊断 + ${actionCount} 个受控动作` : "仅诊断";
      return `<tr>
        <td><strong>${escape(agent.display_name)}</strong><small class="agent-row-description">${escape(agent.description || "未填写说明")}</small></td>
        <td>${shell.badge(agent.status)}</td>
        <td><strong>${targetNames.length} 个 Target</strong><small class="agent-row-description">${targetNames.join("、") || "—"}</small></td>
        <td><strong>${access}</strong><small class="agent-row-description">${targetNames.join("、") || "—"}</small></td>
        <td>${agent.auto_alert_enabled ? `<strong>${escape(agent.auto_observe_min_severity)} 起 · Target L${escape(agent.auto_observe_min_target_level)}+</strong><small class="agent-row-description">冷却 ${escape(agent.alert_cooldown_minutes)} 分钟</small>` : "已关闭"}</td>
        <td><div class="ops-actions"><button type="button" data-agent-action="edit" data-agent-id="${escape(agent.agent_id)}">编辑</button><button type="button" class="danger" data-agent-action="delete" data-agent-id="${escape(agent.agent_id)}">删除</button></div></td>
      </tr>`;
    }).join("");
    body.querySelectorAll("[data-agent-action]").forEach((button) => {
      button.addEventListener("click", () => {
        if (button.dataset.agentAction === "edit") {
          void openEdit(button.dataset.agentId);
          return;
        }
        const agent = agents.find((item) => item.agent_id === button.dataset.agentId);
        if (agent) void deleteAgent(agent, button);
      });
    });
  }

  async function deleteAgent(agent, button) {
    if (!globalThis.confirm(`确认删除 Agent“${agent.display_name}”？删除后不再接收新任务，历史运行和报告仍会保留。`)) return;
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${api}/agents/${encodeURIComponent(agent.agent_id)}`, {
        method: "DELETE",
        headers: { "If-Match": `"rv-${agent.row_version}"` },
      });
      shell.toast("Agent 已删除");
      await load();
    } catch (error) {
      shell.toast(error.message);
      button.disabled = false;
    }
  }


  const LLM_PROVIDER_LABELS = {
    local_deepseek: "本地部署 · DeepSeek",
    api_deepseek: "API 调用 · DeepSeek",
    api_qwen: "API 调用 · Qwen",
    chatgpt: "API 调用 · OpenAI",
    oci: "API 调用 · OCI",
  };

  function isLocalLlmProvider(provider) {
    return String(provider || "").trim().toLowerCase() === "local_deepseek";
  }

  function llmProviderLabel(provider) {
    const key = String(provider || "").trim().toLowerCase();
    return LLM_PROVIDER_LABELS[key] || provider || "未知提供方";
  }

  function llmOptionLabel(model) {
    return `${model.display_name} · ${model.served_model_name} · ${llmProviderLabel(model.provider)}`;
  }

  function sortedDiagnosisModels() {
    return models
      .filter((model) => Number(model.category) === 1)
      .slice()
      .sort((left, right) => {
        const leftLocal = isLocalLlmProvider(left.provider) ? 0 : 1;
        const rightLocal = isLocalLlmProvider(right.provider) ? 0 : 1;
        if (leftLocal !== rightLocal) return leftLocal - rightLocal;
        return String(left.display_name || "").localeCompare(String(right.display_name || ""), "zh");
      });
  }

  function firstLocalDeepseekId() {
    const local = sortedDiagnosisModels().find((model) => isLocalLlmProvider(model.provider));
    return local ? local.model_id : "";
  }

  function renderResources() {
    document.getElementById("agent-targets").innerHTML = targets.length
      ? targets.map((target) => `<label class="agent-switch-row"><input type="checkbox" name="target_ids" value="${escape(target.target_id)}"><span><strong>${escape(target.display_name)}</strong><small>${escape(target.db_type)} · ${target.readonly_connection_enabled ? "只读直连" : "仅监控"}${target.controlled_change_enabled ? " · 允许受控变更" : ""}</small></span></label>`).join("")
      : '<div class="ops-error">暂无已启用的 Target。请先创建 Target，并完成监控 Label 映射后启用。</div>';
    const diagnosisModels = sortedDiagnosisModels();
    document.getElementById("agent-planner-model").innerHTML = diagnosisModels.length
      ? '<option value="">请选择规划模型</option>' + diagnosisModels.map((model) => `<option value="${escape(model.model_id)}">${escape(llmOptionLabel(model))}</option>`).join("")
      : '<option value="">没有已启用的 LLM，请先配置模型服务</option>';
    document.getElementById("agent-model").innerHTML = diagnosisModels.length
      ? '<option value="">请选择诊断模型</option>' + diagnosisModels.map((model) => `<option value="${escape(model.model_id)}">${escape(llmOptionLabel(model))}</option>`).join("")
      : '<option value="">没有已启用的 LLM，请先配置模型服务</option>';
    renderImageModelOptions("agent-ocr-model", 6, "不启用 OCR", "OCR");
    renderImageModelOptions("agent-vlm-model", 5, "不启用 VLM", "VLM");
  }

  function renderImageModelOptions(elementId, category, disabledLabel, capabilityName) {
    const imageModels = models.filter((model) => Number(model.category) === category);
    document.getElementById(elementId).innerHTML = imageModels.length
      ? `<option value="">${disabledLabel}</option>` + imageModels.map((model) => `<option value="${escape(model.model_id)}">${escape(model.display_name)} · ${escape(model.served_model_name)}</option>`).join("")
      : `<option value="">没有已启用的 ${capabilityName} 模型</option>`;
  }

  function toggleTargetFields() {
    const selected = selectedTargetIds();
    const changeTargets = targets.filter((item) => selected.includes(item.target_id) && item.controlled_change_enabled);
    document.getElementById("agent-change-help").textContent = !selected.length
      ? "请选择至少一个 Target。"
      : changeTargets.length
        ? `当前有 ${changeTargets.length} 个 Target 可配置；未选择动作时保持只读诊断。`
        : "所选 Target 均未启用受控变更，Agent 保持只读诊断。";
    renderControlledActions();
    loadActionCatalogs(changeTargets.map((item) => item.target_id))
      .then(renderControlledActions)
      .catch((error) => showResult(`读取动作目录失败：${error.message}`, "bad"));
  }

  async function loadActionCatalogs(targetIds) {
    await Promise.all(targetIds.map(async (targetId) => {
      if (actionCatalogsByTarget.has(targetId)) return;
      const catalog = await KBotAIOpsAuth.request(`${api}/action-catalog/${encodeURIComponent(targetId)}`);
      actionCatalogsByTarget.set(targetId, catalog);
    }));
  }

  function renderControlledActions() {
    captureActionDrafts();
    const selected = selectedTargetIds();
    const container = document.getElementById("agent-controlled-actions");
    if (!selected.length) {
      container.innerHTML = '<div class="ops-empty">未选择数据库 Target，无数据库直连或变更权限。</div>';
      return;
    }
    container.innerHTML = selected.map((targetId) => {
      const target = targets.find((item) => item.target_id === targetId);
      if (!target?.controlled_change_enabled) {
        return `<article class="agent-source-card"><strong>${escape(target?.display_name)}</strong><small>仅只读诊断；该 Target 未启用受控变更。</small></article>`;
      }
      const catalog = actionCatalogsByTarget.get(targetId);
      if (!catalog) return `<article class="agent-source-card"><strong>${escape(target.display_name)}</strong><small>正在读取动作目录…</small></article>`;
      const configured = actionDraftsByTarget.get(targetId)
        || (editing?.controlled_action_execution || []).find((item) => item.target_id === targetId)
        || {};
      const selectedActions = new Set(configured.allowed_action_ids || []);
      const actions = groupedActions(catalog);
      const choices = actions.length
        ? actions.map((action) => `<label class="agent-switch-row agent-action-choice">
          <input type="checkbox" data-action-target="${escape(targetId)}" data-scope-requirements="${escape((action.scope_requirements || []).join(","))}" value="${escape(action.action_id)}" ${selectedActions.has(action.action_id) ? "checked" : ""} ${action.currently_executable ? "" : "disabled"}>
          <span><strong>${escape(action.action_id)}</strong><small>${escape(action.action_family)} · ${escape(action.risk_level)} · ${escape(action.lock_impact)}</small></span>
        </label>`).join("")
        : '<small>当前 Target 没有可授权的受控动作，保持只读诊断。</small>';
      const scopes = configured.object_scopes || {};
      const options = catalog.scope_options || {};
      return `<article class="agent-source-card agent-action-policy" data-action-policy-target="${escape(targetId)}">
        <div class="agent-policy-head"><div><strong>${escape(target.display_name)}</strong><small>默认只读；勾选的动作仍须逐条人工审批，范围按所选动作展开。</small></div><span>只读为默认</span></div>
        <div class="agent-action-list">${choices}</div>
        <div class="agent-policy-settings" data-action-policy-settings hidden>
          <div class="agent-execution-limit"><div><strong>执行频率</strong><small>限制此 Agent 在当前 Target 上每天最多执行的受控动作次数。</small></div><label>每日上限 <input data-action-limit type="number" min="1" max="10000" value="${escape(configured.max_daily_executions || 10)}"></label></div>
          <div class="agent-scope-grid">
            ${optionPicker("schemas", "允许操作的 Schema", options.schemas || [], scopes.schemas || [], "来自 Target 最近一次只读目录探测，可多选。")}
            ${dynamicParameterPicker(options.dynamic_parameters || [], scopes.dynamic_parameters || [])}
            ${optionPicker("resource_manager_plans", "Resource Manager Plan", options.resource_manager_plans || [], scopes.resource_manager_plans || [], "只显示数据库中可发现的有效 Plan。")}
            ${optionPicker("privilege_grantees", "允许授权的本地用户", options.privilege_grantees || [], scopes.privilege_grantees || [], "排除 Oracle 维护用户与公共用户。")}
            ${optionPicker("system_privileges", "系统权限", options.system_privileges || [], scopes.system_privileges || [], "只提供受控动作 Catalog 允许的权限。")}
            ${optionPicker("object_privileges", "对象权限", options.object_privileges || [], scopes.object_privileges || [], "只提供受控动作 Catalog 允许的权限。")}
          </div>
        </div>
      </article>`;
    }).join("");
    container.querySelectorAll("[data-action-policy-target]").forEach(syncActionScopeVisibility);
  }

  function captureActionDrafts() {
    document.querySelectorAll("[data-action-policy-target]").forEach((card) => {
      actionDraftsByTarget.set(card.dataset.actionPolicyTarget, {
        allowed_action_ids: [...card.querySelectorAll("[data-action-target]:checked")].map((input) => input.value),
        max_daily_executions: Number(card.querySelector("[data-action-limit]")?.value || 10),
        object_scopes: {
          schemas: selectedScopeValues(card, "schemas"),
          dynamic_parameters: selectedDynamicParameters(card),
          resource_manager_plans: selectedScopeValues(card, "resource_manager_plans"),
          privilege_grantees: selectedScopeValues(card, "privilege_grantees"),
          system_privileges: selectedScopeValues(card, "system_privileges"),
          object_privileges: selectedScopeValues(card, "object_privileges"),
        },
      });
    });
  }

  function requiredScopeKinds(card) {
    return new Set([...card.querySelectorAll("[data-action-target]:checked")]
      .flatMap((input) => String(input.dataset.scopeRequirements || "").split(","))
      .map((value) => value.trim()).filter(Boolean));
  }

  function updateScopeCounts(card) {
    card.querySelectorAll("[data-scope-kind]").forEach((panel) => {
      const count = panel.querySelectorAll('input[type="checkbox"]:checked').length;
      const indicator = panel.querySelector("[data-scope-count]");
      if (indicator) indicator.textContent = `已选 ${count} 项`;
    });
  }

  function syncActionScopeVisibility(card) {
    const selectedActions = card.querySelectorAll("[data-action-target]:checked");
    const settings = card.querySelector("[data-action-policy-settings]");
    if (!settings) return;
    settings.hidden = !selectedActions.length;
    const requirements = requiredScopeKinds(card);
    card.querySelectorAll("[data-scope-kind]").forEach((panel) => {
      panel.hidden = !requirements.has(panel.dataset.scopeKind);
    });
    card.classList.toggle("selected", Boolean(selectedActions.length));
    updateScopeCounts(card);
  }

  function selectedTargetIds() {
    return [...document.querySelectorAll('[name="target_ids"]:checked')].map((input) => input.value);
  }

  function toggleAlertSettings() {
    const enabled = document.querySelector('[name="auto_alert_enabled"]').checked;
    document.getElementById("agent-alert-settings").classList.toggle("agent-settings-disabled", !enabled);
    document.getElementById("agent-min-severity").disabled = !enabled;
    document.getElementById("agent-min-target-level").disabled = !enabled;
    document.getElementById("agent-cooldown").disabled = !enabled;
  }

  function openCreate() {
    editing = null;
    actionDraftsByTarget.clear();
    actionCatalogsByTarget.clear();
    document.getElementById("agent-controlled-actions").innerHTML = "";
    const form = document.getElementById("agent-form");
    form.reset();
    form.elements.status.value = "DRAFT";
    form.elements.alert_cooldown_minutes.value = 15;
    form.elements.auto_observe_min_target_level.value = 1;
    form.elements.auto_alert_enabled.checked = true;
    form.elements.status.disabled = true;
    const localModelId = firstLocalDeepseekId();
    form.elements.planner_model_id.value = localModelId;
    form.elements.diagnosis_model_id.value = localModelId;
    document.getElementById("agent-status-help").textContent = "新增 Agent 固定保存为草稿；创建成功后可在编辑时启用。";
    toggleTargetFields();
    toggleAlertSettings();
    showResult();
    document.getElementById("agent-dialog-title").textContent = "新增 Agent";
    document.getElementById("save-agent").textContent = "创建 Agent";
    document.getElementById("agent-dialog").showModal();
  }

  async function openEdit(agentId) {
    editing = agents.find((item) => item.agent_id === agentId);
    if (!editing) return;
    actionDraftsByTarget.clear();
    actionCatalogsByTarget.clear();
    document.getElementById("agent-controlled-actions").innerHTML = "";
    const form = document.getElementById("agent-form");
    form.reset();
    form.elements.status.disabled = false;
    form.elements.display_name.value = editing.display_name;
    form.elements.description.value = editing.description || "";
    form.elements.status.value = editing.status;
    form.elements.auto_alert_enabled.checked = Boolean(editing.auto_alert_enabled);
    form.elements.auto_observe_min_severity.value = editing.auto_observe_min_severity || "CRITICAL";
    form.elements.auto_observe_min_target_level.value = editing.auto_observe_min_target_level || 1;
    form.elements.alert_cooldown_minutes.value = editing.alert_cooldown_minutes ?? 15;
    form.elements.planner_model_id.value = editing.models?.planner_llm || "";
    form.elements.diagnosis_model_id.value = editing.models?.diagnosis_llm || "";
    form.elements.ocr_model_id.value = editing.image_capabilities?.ocr?.default_model_id || "";
    form.elements.vlm_model_id.value = editing.image_capabilities?.vlm?.default_model_id || "";
    form.elements.instruction.value = editing.instruction || "";
    form.querySelectorAll('[name="target_ids"]').forEach((input) => {
      input.checked = (editing.target_ids || []).includes(input.value);
    });
    document.getElementById("agent-status-help").textContent = "启用前会检查每个 Target 均已启用，并至少存在一条引用已启用监控源的有效 Label 映射。";
    toggleTargetFields();
    toggleAlertSettings();
    showResult();
    document.getElementById("agent-dialog-title").textContent = "修改 Agent";
    document.getElementById("save-agent").textContent = "保存修改";
    document.getElementById("agent-dialog").showModal();
  }

  function selectedScopeValues(card, kind) {
    return [...card.querySelectorAll(`[data-action-scope="${kind}"]:checked`)]
      .map((input) => input.value);
  }

  function selectedDynamicParameters(card) {
    return [...card.querySelectorAll("[data-dynamic-parameter]")].map((row) => ({
      name: row.dataset.dynamicParameter,
      allowed_values: [...row.querySelectorAll("[data-dynamic-value]:checked")].map((input) => input.value),
    })).filter((item) => item.allowed_values.length);
  }

  function assertRequiredScopes(requirements, values) {
    requirements.forEach((kind) => {
      if (!values[kind]?.length) {
        throw new Error(`请为已选受控动作选择${scopeLabels[kind] || kind}。`);
      }
    });
  }

  function payload(form) {
    const targetIds = selectedTargetIds();
    if (!targetIds.length) throw new Error("至少选择一个 Target。");
    const plannerModelId = form.elements.planner_model_id.value.trim();
    const diagnosisModelId = form.elements.diagnosis_model_id.value.trim();
    const ocrModelId = form.elements.ocr_model_id.value.trim();
    const vlmModelId = form.elements.vlm_model_id.value.trim();
    if (!plannerModelId) throw new Error("请选择规划模型。");
    if (!diagnosisModelId) throw new Error("请选择诊断模型。");
    const imageCapabilities = {};
    if (ocrModelId) {
      imageCapabilities.ocr = {
        allowed_model_ids: [ocrModelId],
        default_model_id: ocrModelId,
      };
    }
    if (vlmModelId) {
      imageCapabilities.vlm = {
        allowed_model_ids: [vlmModelId],
        default_model_id: vlmModelId,
      };
    }
    const autoAlertEnabled = form.elements.auto_alert_enabled.checked;
    const controlledActionExecution = targetIds.map((targetId) => {
      const card = document.querySelector(`[data-action-policy-target="${CSS.escape(targetId)}"]`);
      if (!card) return null;
      const actionIds = [...card.querySelectorAll("[data-action-target]:checked")].map((input) => input.value);
      if (!actionIds.length) return null;
      const requirements = requiredScopeKinds(card);
      const scopeValues = {
        schemas: requirements.has("schemas") ? selectedScopeValues(card, "schemas") : [],
        dynamic_parameters: requirements.has("dynamic_parameters") ? selectedDynamicParameters(card) : [],
        resource_manager_plans: requirements.has("resource_manager_plans") ? selectedScopeValues(card, "resource_manager_plans") : [],
        privilege_grantees: requirements.has("privilege_grantees") ? selectedScopeValues(card, "privilege_grantees") : [],
        system_privileges: requirements.has("system_privileges") ? selectedScopeValues(card, "system_privileges") : [],
        object_privileges: requirements.has("object_privileges") ? selectedScopeValues(card, "object_privileges") : [],
      };
      assertRequiredScopes(requirements, scopeValues);
      return {
        target_id: targetId,
        enabled: true,
        allowed_action_ids: actionIds,
        object_scopes: {
          schemas: scopeValues.schemas,
          exclude_system_objects: true,
          dynamic_parameters: scopeValues.dynamic_parameters,
          resource_manager_plans: scopeValues.resource_manager_plans,
          privilege_grantees: scopeValues.privilege_grantees,
          system_privileges: scopeValues.system_privileges,
          object_privileges: scopeValues.object_privileges,
        },
        max_daily_executions: Number(card.querySelector("[data-action-limit]").value),
      };
    }).filter(Boolean);
    return {
      display_name: form.elements.display_name.value.trim(),
      description: form.elements.description.value.trim() || null,
      status: editing ? form.elements.status.value : "DRAFT",
      target_ids: targetIds,
      controlled_action_execution: controlledActionExecution,
      auto_alert_enabled: autoAlertEnabled,
      auto_observe_min_severity: autoAlertEnabled ? form.elements.auto_observe_min_severity.value : (editing?.auto_observe_min_severity || "CRITICAL"),
      auto_observe_min_target_level: autoAlertEnabled ? Number(form.elements.auto_observe_min_target_level.value) : (editing?.auto_observe_min_target_level || 1),
      alert_cooldown_minutes: autoAlertEnabled ? Number(form.elements.alert_cooldown_minutes.value) : (editing?.alert_cooldown_minutes ?? 15),
      models: {
        planner_llm: plannerModelId,
        diagnosis_llm: diagnosisModelId,
      },
      instruction: form.elements.instruction.value.trim() || null,
      image_capabilities: imageCapabilities,
      config: {},
    };
  }

  async function save(event) {
    event.preventDefault();
    const button = document.getElementById("save-agent");
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = editing ? "保存中…" : "创建中…";
    showResult(editing ? "正在校验 Agent 配置…" : "正在创建 Agent…");
    try {
      const body = payload(event.currentTarget);
      if (editing) body.expected_row_version = editing.row_version;
      await KBotAIOpsAuth.request(editing ? `${api}/agents/${encodeURIComponent(editing.agent_id)}` : `${api}/agents`, {
        method: editing ? "PATCH" : "POST",
        body: JSON.stringify(body),
      });
      document.getElementById("agent-dialog").close();
      shell.toast(editing ? "Agent 已更新" : "Agent 已创建");
      await load();
    } catch (error) {
      showResult(error.message, "bad");
      shell.toast(error.message);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
  }

  async function load() {
    const [agentRows, targetPage, modelRows] = await Promise.all([
      KBotAIOpsAuth.request(`${api}/agents`),
      KBotAIOpsAuth.request(`${api}/targets?status=ENABLED&limit=200`),
      KBotAIOpsAuth.request("/api/v1/model-catalog"),
    ]);
    agents = Array.isArray(agentRows) ? agentRows : [];
    targets = targetPage.items || [];
    models = Array.isArray(modelRows) ? modelRows : [];
    renderResources();
    renderRows();
  }

  shell.ready.then(async () => {
    const dialog = document.getElementById("agent-dialog");
    dialog.querySelectorAll("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => dialog.close()));
    document.getElementById("create-agent").addEventListener("click", openCreate);
    document.getElementById("agent-targets").addEventListener("change", () => toggleTargetFields());
    document.getElementById("agent-controlled-actions").addEventListener("change", (event) => {
      const card = event.target.closest("[data-action-policy-target]");
      if (!card) return;
      if (event.target.matches("[data-action-target]")) syncActionScopeVisibility(card);
      else if (event.target.matches('[type="checkbox"]')) updateScopeCounts(card);
    });
    document.getElementById("agent-controlled-actions").addEventListener("input", (event) => {
      if (!event.target.matches("[data-scope-search]")) return;
      const query = event.target.value.trim().toLocaleLowerCase();
      const panel = event.target.closest("[data-scope-kind]");
      panel.querySelectorAll("[data-option-label]").forEach((option) => {
        option.hidden = Boolean(query && !option.dataset.optionLabel.includes(query));
      });
    });
    document.querySelector('[name="auto_alert_enabled"]').addEventListener("change", toggleAlertSettings);
    document.getElementById("agent-form").addEventListener("submit", save);
    try {
      await load();
    } catch (error) {
      document.getElementById("agent-rows").innerHTML = `<tr><td class="ops-empty" colspan="6">${escape(error.message)}</td></tr>`;
    }
  });
})();
