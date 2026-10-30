(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const escape = (value) => shell.escape(value ?? "—");
  const values = (items) => Array.isArray(items) ? items : [];
  const capabilityCatalog = [
    ["RESOURCE_CATALOG_QUERY", "资源信息", "业务系统拓扑、主机、IP 与资源属性"],
    ["KNOWLEDGE_QA", "运维知识", "已审批手册与历史诊断案例"],
    ["MONITORING_QUERY", "监控现状", "实时指标、趋势与资源状态"],
    ["ALERT_QUERY", "告警查询", "当前告警与历史告警"],
    ["INSPECTION_QUERY", "巡检查询", "巡检执行结果与报告"],
    ["REPORT_CASE_QUERY", "报告与案例", "历史诊断报告与已审核案例"],
    ["ACTIVE_DIAGNOSIS", "主动诊断", "按自然语言范围执行有界诊断"],
    ["CONTROLLED_CHANGE", "受控变更", "仅保留已有资源级授权；新增授权需在资源治理中完成"],
  ];
  const capabilityLabels = Object.fromEntries(capabilityCatalog.map(([key, label]) => [key, label]));
  const severityLabels = { INFO: "信息及以上", WARNING: "警告及以上", HIGH: "高危及以上", CRITICAL: "仅严重告警" };
  const providerLabels = {
    local_deepseek: "本地部署 · DeepSeek",
    api_deepseek: "API 调用 · DeepSeek",
    api_qwen: "API 调用 · Qwen",
    chatgpt: "API 调用 · OpenAI",
    oci: "API 调用 · OCI",
  };

  const boundaryStates = new Map();
  const boundaryKey = (ids) => [...new Set(ids)].sort().join(",");
  const boundaryLabels = (agent) => ["监控分析", ...(agent.readonly_database_enabled !== false ? ["只读数据库"] : []), ...(agent.allow_agent_execution === true ? ["允许变更"] : [])];
  function configurationStatus(ids, field) {
    if (!ids.length) return "请先选择业务系统";
    const state = boundaryStates.get(boundaryKey(ids));
    if (!state || state.status === "loading") return "配置核验中";
    if (state.status === "error") return "配置状态未知";
    if (!state.items.length) return "尚无数据库 Target";
    return `${state.items.filter(item => item[field] === true).length}/${state.items.length} 个 Target 配置就绪`;
  }
  async function loadBoundary(ids, force = false) {
    const key = boundaryKey(ids);
    if (!key || (!force && boundaryStates.has(key))) return;
    const requestState = { status: "loading", items: [] };
    boundaryStates.set(key, requestState);
    try {
      const query = new URLSearchParams();
      key.split(",").forEach(id => query.append("business_system_ids", id));
      const result = await KBotAIOpsAuth.request(`${api}/agents/boundary-resources?${query}`);
      if (!Array.isArray(result.items)) throw new Error("配置摘要无效");
      if (boundaryStates.get(key) === requestState) boundaryStates.set(key, { status: "ready", items: result.items });
    } catch {
      if (boundaryStates.get(key) === requestState) boundaryStates.set(key, { status: "error", items: [] });
    }
    renderRows();
    renderBoundaryResources();
    if (currentStep === 6) renderReview();
  }
  function targetGaps(item) {
    const form = document.getElementById("agent-form");
    const readonly = form.elements.readonly_database_enabled.checked;
    const change = form.elements.allow_agent_execution.checked;
    return [
      !item.target_enabled && "Target 未启用，不能执行数据库诊断或变更",
      !item.scope_enabled && "业务系统未上线，不能执行诊断或变更",
      !item.monitoring_configured && "尚未绑定监控源，无法读取监控数据",
      readonly && !item.readonly_connection_enabled && "Agent 已允许只读，但 Target 未开放只读连接",
      (readonly || change) && !item.connection_configured && "Target 缺少数据库连接配置，不能执行数据库诊断或变更",
      readonly && item.readonly_connection_enabled && !item.diagnostic_credential_configured && "Target 缺少只读凭据",
      change && !item.controlled_change_enabled && "Agent 已允许受控变更，但 Target 未允许变更，因此该目标不能执行变更",
      change && item.controlled_change_enabled && !item.execution_credential_configured && "Target 缺少变更凭据，不能执行变更",
    ].filter(Boolean).join("；");
  }
  function targetIssues() {
    if (!selectionSets.systems.size) return ["尚未选择业务系统"];
    const state = boundaryStates.get(boundaryKey([...selectionSets.systems]));
    if (!state || state.status === "loading") return ["Target 配置核验中"];
    if (state.status === "error") return ["Target 配置状态未知，请重新核验"];
    if (!state.items.length) return ["所选业务系统尚无数据库 Target"];
    return state.items.flatMap(item => targetGaps(item) ? [item.display_name + "：" + targetGaps(item)] : []);
  }
  function renderBoundaryResources() {
    if (!document.getElementById("agent-dialog").open) return;
    const ids = [...selectionSets.systems];
    const form = document.getElementById("agent-form");
    const state = boundaryStates.get(boundaryKey(ids));
    const readonly = form.elements.readonly_database_enabled.checked;
    const change = form.elements.allow_agent_execution.checked;
    document.getElementById("agent-boundary-resources").innerHTML = `<h4>关联 Target 配置与限制</h4><p>账号和连接由业务系统资源（Target）维护；配置就绪不代表连接健康或动作可执行。</p><p>监控：${configurationStatus(ids, "monitoring_configured")}</p>${readonly ? `<p>只读：${configurationStatus(ids, "readonly_configured")}</p>` : ""}${change ? `<p>变更：${configurationStatus(ids, "change_configured")}</p>` : ""}<ul>${values(state?.items).map(item => `<li><strong>${escape(item.display_name)} · ${escape(item.db_type)}</strong><small>所属系统：${escape(values(item.business_system_ids).map(systemName).join("、"))}</small><small>监控：${item.monitoring_configured ? "已配置" : "未配置"} · 只读：${item.readonly_configured ? "已配置" : "未配置"} · 变更：${item.change_configured ? "已配置" : "未配置"}</small><small>${escape(targetGaps(item) || "所选边界的 Target 配置已就绪；实际执行仍需权限与审批校验")}</small><a href="business-system-workspace.html?view=readonly&id=${encodeURIComponent(item.business_system_ids[0])}" target="_blank" rel="noopener">查看业务系统资源 ↗</a></li>`).join("")}</ul><button id="retry-boundary" class="secondary-button" type="button" ${!ids.length || state?.status === "loading" ? "disabled" : ""}>重新核验配置</button>`;
    document.getElementById("retry-boundary")?.addEventListener("click", () => { loadBoundary(ids, true); });
  }

  let agents = [];
  let systems = [];
  let models = [];
  let knowledgeTags = [];
  let knowledgeBases = [];
  let editing = null;
  let currentStep = 1;
  let pickerKind = null;
  let pickerPage = 1;
  let saving = false;
  let loading = true;
  let catalogReady = false;
  const pickerPageSize = 12;
  const queryCapabilities = ["RESOURCE_CATALOG_QUERY", "ALERT_QUERY", "INSPECTION_QUERY", "REPORT_CASE_QUERY"];
  const selectionSets = {
    systems: new Set(),
    knowledgeBases: new Set(),
    tags: new Set(),
  };

  const systemName = (id) => systems.find((item) => item.business_system_id === id)?.display_name || "业务系统已不可用";
  const selected = (name) => [...document.querySelectorAll(`[name="${name}"]:checked`)].map((input) => input.value);

  function showResult(message = "", tone = "") {
    const result = document.getElementById("agent-result");
    result.textContent = message;
    result.dataset.tone = tone;
    result.hidden = !message;
  }

  function modelLabel(model) {
    const provider = providerLabels[String(model.provider || "").toLowerCase()] || model.provider || "未知提供方";
    return `${model.display_name} · ${provider}`;
  }

  function llmModels() {
    return models.filter((item) => Number(item.category) === 1).slice().sort((left, right) => {
      const leftLocal = String(left.provider || "").toLowerCase() === "local_deepseek" ? 0 : 1;
      const rightLocal = String(right.provider || "").toLowerCase() === "local_deepseek" ? 0 : 1;
      return leftLocal - rightLocal || String(left.display_name || "").localeCompare(String(right.display_name || ""), "zh");
    });
  }

  function defaultLlmId() {
    return llmModels().find((item) => String(item.provider || "").toLowerCase() === "local_deepseek")?.model_id || llmModels()[0]?.model_id || "";
  }

  function renderModelOptions(elementId, category, emptyLabel) {
    const rows = category === 1 ? llmModels() : models.filter((item) => Number(item.category) === category);
    document.getElementById(elementId).innerHTML = `<option value="">${emptyLabel}</option>${rows.map((item) => `<option value="${escape(item.model_id)}">${escape(category === 1 ? modelLabel(item) : item.display_name)}</option>`).join("")}`;
  }

  function renderResources() {
    document.getElementById("agent-capabilities").innerHTML = capabilityCatalog
      .filter(([key]) => queryCapabilities.includes(key))
      .map(([key, label, help]) => `<label><input type="checkbox" name="capabilities" value="${key}"><span><strong>${label}</strong><small>${help}</small></span></label>`).join("");
    ["agent-planner-model", "agent-model", "agent-composer-model", "agent-memory-model"].forEach((id) => renderModelOptions(id, 1, id === "agent-memory-model" ? "复用回答组织模型" : id === "agent-model" ? "未配置诊断模型" : "请选择"));
    renderModelOptions("agent-ocr-model", 6, "不启用 OCR");
    renderModelOptions("agent-vlm-model", 5, "不启用 VLM");
    renderSelectionSummaries();
  }

  function selectionRows(kind) {
    if (kind === "systems") return systems.filter((item) => selectionSets.systems.has(String(item.business_system_id))).map((item) => ({ id: String(item.business_system_id), title: item.display_name }));
    if (kind === "knowledgeBases") return knowledgeBases.filter((item) => selectionSets.knowledgeBases.has(String(item.knowledge_base_id))).map((item) => ({ id: String(item.knowledge_base_id), title: item.display_name }));
    return knowledgeTags.filter((item) => selectionSets.tags.has(String(item.tag_id))).map((item) => ({ id: String(item.tag_id), title: item.display_name }));
  }

  function renderSelectionSummary(kind, elementId) {
    const rows = selectionRows(kind);
    const element = document.getElementById(elementId);
    element.innerHTML = rows.map((item) => `<span class="agent-selection-chip">${escape(item.title)}<button type="button" data-remove-selection="${kind}" data-selection-id="${escape(item.id)}" aria-label="移除 ${escape(item.title)}">×</button></span>`).join("");
    if (!rows.length && kind !== "tags") element.innerHTML = `<p class="agent-empty-selection">${kind === "systems" ? "尚未选择业务系统。" : "尚未绑定知识库。"}</p>`;
    element.querySelectorAll("[data-remove-selection]").forEach((button) => {
      button.onclick = () => {
        selectionSets[button.dataset.removeSelection].delete(button.dataset.selectionId);
        renderSelectionSummaries();
      };
    });
  }

  function renderSelectionSummaries() {
    renderSelectionSummary("systems", "agent-business-system-summary");
    if (document.getElementById("agent-dialog").open) { renderBoundaryResources(); void loadBoundary([...selectionSets.systems]); }
    renderSelectionSummary("knowledgeBases", "agent-knowledge-base-summary");
    renderSelectionSummary("tags", "agent-knowledge-tag-summary");
    const count = selectionSets.tags.size;
    document.getElementById("agent-knowledge-tag-help").textContent = count
      ? `仅检索 ${count} 个标签`
      : "不限制标签，使用所选知识库中的全部已审批知识";
    document.getElementById("open-agent-tag-picker").textContent = count ? "调整标签" : "限制标签";
  }

  function setBusy() {
    document.getElementById("create-agent").disabled = saving || loading || !catalogReady;
    document.getElementById("refresh-agents").disabled = saving || loading;
    document.getElementById("refresh-agents").textContent = loading ? "刷新中…" : "刷新目录";
    document.querySelectorAll('#agent-rows button, #agent-dialog button').forEach((button) => { button.disabled = saving; });
  }

  function renderRows() {
    const body = document.getElementById("agent-rows");
    const keyword = document.getElementById("agent-search").value.trim().toLocaleLowerCase();
    const visibleAgents = agents.filter((agent) => !keyword || [agent.display_name, agent.description || "", ...values(agent.allowed_business_system_ids).map(systemName), ...values(agent.capabilities).map((item) => capabilityLabels[item] || item)].join(" ").toLocaleLowerCase().includes(keyword));
    if (!visibleAgents.length) {
      body.innerHTML = "";
      return;
    }
    body.innerHTML = visibleAgents.map((agent) => {
      const names = values(agent.allowed_business_system_ids).map(systemName);
      const capabilities = values(agent.capabilities).map((item) => capabilityLabels[item] || item);
      const status = { ACTIVE: ["运行中", "status-success"], DRAFT: ["草稿", "status-warning"], DISABLED: ["已停用", "status-muted"] }[agent.status] || [agent.status, "status-muted"];
      const button = (action, label, danger = false) => `<button type="button" class="${danger ? "danger-row-action" : "row-action"}" data-agent-action="${action}" data-agent-id="${escape(agent.agent_id)}" ${saving ? "disabled" : ""}>${label}</button>`;
      return `<article class="management-catalog-card"><header><div><p class="eyebrow">AIOPS AGENT</p><h3>${escape(agent.display_name)}</h3></div><span class="status-pill ${status[1]}">${escape(status[0])}</span></header><p>${escape(agent.description || "尚未填写 Agent 说明。")}</p><dl><div><dt>业务系统</dt><dd>${names.length} 个</dd></div><div><dt>能力范围</dt><dd class="agent-scope" title="${escape(capabilities.join("、"))}"><span>${escape(capabilities.slice(0, 3).join("、") || "未配置")}</span>${capabilities.length > 3 ? `<small>+${capabilities.length - 3}</small>` : ""}</dd></div></dl><div class="agent-boundary-badges" aria-label="已选诊断边界">${boundaryLabels(agent).map(label => `<span class="status-pill status-muted">${escape(label)}</span>`).join("")}</div><p class="agent-boundary-status">监控：${configurationStatus(values(agent.allowed_business_system_ids), "monitoring_configured")}${agent.readonly_database_enabled !== false ? `<br>只读：${configurationStatus(values(agent.allowed_business_system_ids), "readonly_configured")}` : ""}${agent.allow_agent_execution === true ? `<br>变更：${configurationStatus(values(agent.allowed_business_system_ids), "change_configured")} · 仍需动作授权与审批` : ""}</p><footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions">${button("edit", "编辑")}${agent.status === "ACTIVE" ? button("disable", "停用", true) : ["DRAFT", "DISABLED"].includes(agent.status) ? button("enable", "启用") : ""}</div><div class="management-catalog-overflow"><button type="button" class="row-action management-catalog-more" aria-haspopup="menu" aria-expanded="false"><span>更多</span><span aria-hidden="true">⌄</span></button><div class="management-catalog-menu" role="menu" hidden>${button("delete", "删除 Agent", true)}</div></div></footer></article>`;
    }).join("");
    body.querySelectorAll("[data-agent-action]").forEach((button) => {
      button.onclick = () => {
        const agent = agents.find((item) => item.agent_id === button.dataset.agentId);
        closeMenus();
        if (button.dataset.agentAction === "edit") openEdit(button.dataset.agentId).catch((error) => shell.toast(error.message));
        else if (button.dataset.agentAction === "delete") deleteAgent(agent);
        else commandAgent(agent, button.dataset.agentAction);
      };
    });
    body.querySelectorAll(".management-catalog-more").forEach((button) => {
      button.onclick = () => {
        const open = button.getAttribute("aria-expanded") !== "true";
        closeMenus();
        button.setAttribute("aria-expanded", String(open));
        button.nextElementSibling.hidden = !open;
      };
    });
  }

  function closeMenus() {
    document.querySelectorAll(".management-catalog-more").forEach((button) => {
      button.setAttribute("aria-expanded", "false");
      button.nextElementSibling.hidden = true;
    });
  }

  function renderCapabilityGroups() {
    document.querySelector('[name="capabilities"][value="MONITORING_QUERY"]').checked = true;
    const capabilities = selected("capabilities");
    const count = queryCapabilities.filter((key) => capabilities.includes(key)).length;
    document.getElementById("agent-query-capability-count").textContent = `基础查询已启用 ${count}/${queryCapabilities.length} 项。`;
    document.getElementById("agent-diagnosis-group").classList.toggle("selected", capabilities.includes("ACTIVE_DIAGNOSIS"));
    document.getElementById("agent-diagnosis-label").textContent = `诊断模型${capabilities.includes("ACTIVE_DIAGNOSIS") ? "（智能诊断必选）" : "（可选）"}`;
    const changeNote = document.getElementById("agent-controlled-change-note");
    changeNote.hidden = !document.getElementById("agent-form").elements.allow_agent_execution.checked && !values(editing?.capabilities).includes("CONTROLLED_CHANGE");
    const actionPolicies = editing?.controlled_action_execution;
    changeNote.textContent = !Array.isArray(actionPolicies)
      ? "本页不修改资源动作授权；保存时沿用服务端现有配置，实际变更仍需授权与审批。"
      : actionPolicies.some((policy) => policy.enabled && values(policy.allowed_action_ids).length)
        ? "受控变更沿用已有资源动作授权配置；实际执行仍需通过目标权限、执行开关和审批校验。"
        : "当前未配置已启用的资源动作授权；可以保存诊断配置，但不能执行变更。";
    document.querySelectorAll('[name="knowledge_source_kinds"]').forEach((input) => input.closest("label").classList.toggle("selected", input.checked));
  }

  function resetChecks() {
    document.querySelectorAll('#agent-form input[type="checkbox"]').forEach((input) => { input.checked = input.disabled; });
    Object.values(selectionSets).forEach((selection) => selection.clear());
    renderSelectionSummaries();
  }

  function setChecked(name, selectedValues) {
    const chosen = new Set(values(selectedValues).map(String));
    if (name === "allowed_business_system_ids") {
      selectionSets.systems = chosen;
      renderSelectionSummaries();
      return;
    }
    if (name === "knowledge_base_ids") {
      selectionSets.knowledgeBases = chosen;
      renderSelectionSummaries();
      return;
    }
    if (name === "knowledge_tag_ids") {
      selectionSets.tags = chosen;
      renderSelectionSummaries();
      return;
    }
    document.querySelectorAll(`[name="${name}"]`).forEach((input) => { input.checked = chosen.has(String(input.value)); });
  }

  function toggleKnowledgePolicy() {
    const enabled = document.querySelector('[name="knowledge_enabled"]').checked;
    document.getElementById("agent-knowledge-policy-fields").hidden = !enabled;
    const knowledgeCapability = document.querySelector('[name="capabilities"][value="KNOWLEDGE_QA"]');
    if (knowledgeCapability) knowledgeCapability.checked = enabled;
    document.querySelector('[name="knowledge_enabled"] + span').textContent = enabled ? "已启用" : "未启用";
  }

  function missingRuntimeModels() {
    const form = document.getElementById("agent-form");
    return [
      !form.elements.planner_model_id.value && "规划模型",
      !form.elements.composer_model_id.value && "回答组织模型",
      selected("capabilities").includes("ACTIVE_DIAGNOSIS") && !form.elements.diagnosis_model_id.value && "诊断模型",
    ].filter(Boolean);
  }
  function stepError(step) {
    const form = document.getElementById("agent-form");
    if (step === 1 && !form.elements.display_name.value.trim()) return "请填写 Agent 名称。";
    if (step === 2 && form.elements.knowledge_enabled.checked) {
      if (!selected("knowledge_source_kinds").length) return "请至少选择一种知识来源。";
      if (!selectionSets.knowledgeBases.size) return "请至少绑定一个运维知识库。";
      if (!form.elements.maximum_references.value || !form.elements.maximum_references.checkValidity()) return "最多引用须为 1 到 20 的整数。";
    }
    if (step === 3 && !selectionSets.systems.size) return "请至少选择一个业务系统。";
    if (step === 4 && form.elements.auto_alert_enabled.checked) {
      if (!["INFO", "WARNING", "HIGH", "CRITICAL"].includes(form.elements.auto_observe_min_severity.value)) return "请选择有效的最低告警级别。";
      const level = Number(form.elements.auto_observe_min_target_level.value);
      if (!Number.isInteger(level) || level < 1 || level > 5) return "请选择有效的 Target 重要级。";
      if (!form.elements.alert_cooldown_seconds.value || !form.elements.alert_cooldown_seconds.checkValidity()) return "冷却时间须为 0 到 86400 的整数。";
    }
    if (step === 5 && editing?.status === "ACTIVE" && missingRuntimeModels().length) return "运行中的 Agent 必须配置" + missingRuntimeModels().join("、") + "。";
    return "";
  }
  function validateStep(step) {
    const message = stepError(step);
    showResult(message, message ? "bad" : "");
    return !message;
  }
  function toggleAutoAlert() {
    const form = document.getElementById("agent-form");
    document.getElementById("agent-trigger-fields").hidden = !form.elements.auto_alert_enabled.checked;
  }
  function renderReview() {
    const form = document.getElementById("agent-form");
    const capabilities = selected("capabilities").map((item) => capabilityLabels[item] || item);
    const knowledgeEnabled = form.elements.knowledge_enabled.checked;
    const card = (step, title, value, detail) => `<article><span>${escape(title)}</span><strong>${escape(value)}</strong><small>${escape(detail)}</small><button type="button" class="row-action" data-review-step="${step}">返回修改${escape(title)}</button></article>`;
    const modelName = (name) => models.find(item => item.model_id === form.elements[name].value)?.display_name || (form.elements[name].value ? "原配置（当前目录不可用）" : "未配置");
    document.getElementById("agent-review").innerHTML = [
      card(1, "基本信息", form.elements.display_name.value || "未填写名称", form.elements.description.value || "未填写职责说明"),
      card(2, "能力与知识", capabilities.join("、"), knowledgeEnabled ? `知识问答：${selectionSets.knowledgeBases.size} 个知识库 · ${selected("knowledge_source_kinds").map(source => source === "MANUAL" ? "运维手册" : "诊断案例").join("、")} · 最多引用 ${form.elements.maximum_references.value} 项 · 最高 L${form.elements.max_security_level.value} · ${selectionSets.tags.size ? selectionSets.tags.size + " 个标签" : "全部标签"}` : "知识问答：未启用"),
      card(3, "诊断边界", boundaryLabels({ readonly_database_enabled: form.elements.readonly_database_enabled.checked, allow_agent_execution: form.elements.allow_agent_execution.checked }).join("、"), (selectionRows("systems").map(item => item.title).join("、") || "未选择业务系统") + "；实际权限取 Agent、Target、动作范围、平台和审批的交集。"),
      card(4, "自动触发", form.elements.auto_alert_enabled.checked ? "告警自动触发已开启" : "告警自动触发已关闭", `已保存条件：${severityLabels[form.elements.auto_observe_min_severity.value] || form.elements.auto_observe_min_severity.value} · Target L${form.elements.auto_observe_min_target_level.value} 及以上 · 冷却 ${form.elements.alert_cooldown_seconds.value} 秒；手工提问不受影响。`),
      card(5, "模型与指令", `规划：${modelName("planner_model_id")} · 回答：${modelName("composer_model_id")} · 诊断：${modelName("diagnosis_model_id")}`, `摘要：${form.elements.memory_model_id.value ? modelName("memory_model_id") : "复用回答组织模型"} · OCR：${modelName("ocr_model_id")} · VLM：${modelName("vlm_model_id")}；专业指令：${form.elements.instruction.value || "未填写"}`),
    ].join("");
    const required = [1, 2, 3, 4, 5].flatMap(step => stepError(step) ? [{ step, message: stepError(step) }] : []);
    document.getElementById("agent-review-required").innerHTML = '<h4>保存前必填检查</h4>' + (required.length ? "<ul>" + required.map(issue => `<li>${escape(issue.message)} <button type="button" class="row-action" data-review-step="${issue.step}">返回修改</button></li>`).join("") + "</ul>" : "<p>必填配置检查通过。</p>") + (missingRuntimeModels().length && editing?.status !== "ACTIVE" ? `<p>启用前还需配置：${escape(missingRuntimeModels().join("、"))}；当前可保存草稿。</p>` : "");
    const issues = targetIssues();
    document.getElementById("agent-review-targets").innerHTML = '<h4>Target 配置缺口</h4>' + (issues.length ? "<ul>" + issues.map(issue => `<li>${escape(issue)}</li>`).join("") + "</ul>" : "<p>所选边界的 Target 配置已就绪；不代表连接健康或动作可执行。</p>") + "<p>缺少 Target 配置仍可保存；不会自动补权限、凭据或授权。</p>";
    document.querySelectorAll("[data-review-step]").forEach(button => { button.onclick = () => { showResult(); setStep(button.dataset.reviewStep); }; });
  }

  function setStep(step) {
    currentStep = Math.max(1, Math.min(6, Number(step) || 1));
    document.querySelectorAll("[data-agent-step]").forEach((panel) => { panel.hidden = Number(panel.dataset.agentStep) !== currentStep; });
    document.querySelectorAll("[data-agent-step-button]").forEach((button) => {
      const buttonStep = Number(button.dataset.agentStepButton);
      button.classList.toggle("active", buttonStep === currentStep);
      button.classList.toggle("complete", buttonStep < currentStep && !stepError(buttonStep));
      button.disabled = saving;
      button.querySelector("b").textContent = buttonStep < currentStep && !stepError(buttonStep) ? "✓" : `0${buttonStep}`;
    });
    document.getElementById("agent-previous-step").hidden = currentStep === 1;
    document.getElementById("agent-next-step").hidden = currentStep === 6;
    document.getElementById("save-agent").hidden = currentStep !== 6;
    if (currentStep === 6) renderReview();
    if (currentStep === 3) { renderBoundaryResources(); void loadBoundary([...selectionSets.systems]); }
  }

  function nextStep() {
    if (!validateStep(currentStep)) return;
    setStep(currentStep + 1);
  }

  function pickerState() {
    if (pickerKind === "systems") return { selection: selectionSets.systems, limit: 32 };
    if (pickerKind === "knowledgeBases") return { selection: selectionSets.knowledgeBases, limit: 32 };
    return { selection: selectionSets.tags, limit: null };
  }

  function pickerCandidates() {
    if (pickerKind === "systems") return systems.filter((item) => item.status === "ENABLED").map((item) => ({ id: String(item.business_system_id), title: item.display_name, meta: `${item.system_code} · ${item.environment} · ${Number(item.resource_count || 0)} 个资源`, environment: item.environment }));
    if (pickerKind === "knowledgeBases") return knowledgeBases.filter((item) => item.status === "ACTIVE").map((item) => ({ id: String(item.knowledge_base_id), title: item.display_name, meta: item.description || "独立运维知识索引空间" }));
    return knowledgeTags.filter((item) => item.status === "ACTIVE").map((item) => ({ id: String(item.tag_id), title: item.display_name, meta: "已启用知识标签" }));
  }

  function renderPicker() {
    const query = document.getElementById("agent-picker-search").value.trim().toLocaleLowerCase();
    const environment = document.getElementById("agent-picker-environment").value;
    const { selection, limit } = pickerState();
    const filtered = pickerCandidates().filter((item) => (!environment || item.environment === environment) && (!query || `${item.title} ${item.meta}`.toLocaleLowerCase().includes(query)));
    const pageCount = Math.max(1, Math.ceil(filtered.length / pickerPageSize));
    pickerPage = Math.min(pickerPage, pageCount);
    const pageRows = filtered.slice((pickerPage - 1) * pickerPageSize, pickerPage * pickerPageSize);
    const list = document.getElementById("agent-picker-list");
    list.innerHTML = pageRows.length ? pageRows.map((item) => {
      const checked = selection.has(item.id);
      const disabled = !checked && limit !== null && selection.size >= limit;
      return `<button class="agent-picker-row ${checked ? "selected" : ""}" type="button" data-picker-id="${escape(item.id)}" ${disabled ? "disabled" : ""}><span class="agent-picker-check">${checked ? "✓" : ""}</span><span><strong>${escape(item.title)}</strong><small>${escape(item.meta)}</small></span></button>`;
    }).join("") : '<p class="agent-picker-empty">没有匹配的可选项。</p>';
    list.querySelectorAll("[data-picker-id]").forEach((button) => {
      button.onclick = () => {
        if (selection.has(button.dataset.pickerId)) selection.delete(button.dataset.pickerId);
        else if (limit === null || selection.size < limit) selection.add(button.dataset.pickerId);
        renderPicker();
        renderSelectionSummaries();
      };
    });
    document.getElementById("agent-picker-count").textContent = `已选择 ${selection.size}${limit === null ? "" : ` / ${limit}`}`;
    document.getElementById("agent-picker-page-summary").textContent = `共 ${filtered.length} 项 · 第 ${pickerPage}/${pageCount} 页`;
    document.getElementById("agent-picker-previous").disabled = pickerPage <= 1;
    document.getElementById("agent-picker-next").disabled = pickerPage >= pageCount;
  }

  function openPicker(kind) {
    pickerKind = kind;
    pickerPage = 1;
    document.getElementById("agent-picker-search").value = "";
    document.getElementById("agent-picker-environment").value = "";
    document.getElementById("agent-picker-environment").hidden = kind !== "systems";
    document.getElementById("agent-picker-title").textContent = kind === "systems" ? "选择业务系统" : kind === "knowledgeBases" ? "选择运维知识库" : "限制知识标签范围";
    renderPicker();
    document.getElementById("agent-picker-dialog").showModal();
  }

  function openCreate() {
    editing = null;
    const form = document.getElementById("agent-form");
    form.reset();
    resetChecks();
    document.getElementById("agent-dialog-title").textContent = "新建 Agent";
    document.getElementById("save-agent").textContent = "创建 Agent";
    const defaultModel = defaultLlmId();
    form.elements.planner_model_id.value = defaultModel;
    form.elements.diagnosis_model_id.value = defaultModel;
    form.elements.composer_model_id.value = defaultModel;
    form.elements.memory_model_id.value = "";
    setChecked("capabilities", ["RESOURCE_CATALOG_QUERY", "KNOWLEDGE_QA", "MONITORING_QUERY", "ALERT_QUERY", "INSPECTION_QUERY", "REPORT_CASE_QUERY", "ACTIVE_DIAGNOSIS"]);
    setChecked("knowledge_source_kinds", ["MANUAL", "DIAGNOSIS_CASE"]);
    form.elements.auto_alert_enabled.checked = true;
    toggleAutoAlert();
    form.elements.knowledge_enabled.checked = true;
    form.elements.readonly_database_enabled.checked = false;
    form.elements.auto_observe_min_severity.value = "CRITICAL";
    form.elements.auto_observe_min_target_level.value = "1";
    form.elements.allow_agent_execution.checked = false;
    form.elements.alert_cooldown_seconds.value = "900";
    form.elements.max_security_level.value = "3";
    form.elements.maximum_references.value = "8";
    toggleKnowledgePolicy();
    renderCapabilityGroups();
    setStep(1);
    showResult();
    document.getElementById("agent-dialog").showModal();
    void loadBoundary([...selectionSets.systems], true);
  }

  async function openEdit(agentId) {
    editing = await KBotAIOpsAuth.request(`${api}/agents/${encodeURIComponent(agentId)}`);
    const form = document.getElementById("agent-form");
    form.reset();
    resetChecks();
    document.getElementById("agent-dialog-title").textContent = "编辑 Agent";
    document.getElementById("save-agent").textContent = "保存新版本";
    form.elements.display_name.value = editing.display_name || "";
    form.elements.description.value = editing.description || "";
    for (const [name, modelId] of Object.entries({
      planner_model_id: editing.models?.planner_llm, diagnosis_model_id: editing.models?.diagnosis_llm,
      composer_model_id: editing.models?.composer_llm, memory_model_id: editing.models?.memory_llm,
      ocr_model_id: editing.image_capabilities?.ocr?.default_model_id, vlm_model_id: editing.image_capabilities?.vlm?.default_model_id,
    })) {
      const control = form.elements[name];
      if (modelId && ![...control.options].some(option => option.value === modelId)) control.add(new Option("原配置（当前目录不可用）", modelId));
    }
    form.elements.planner_model_id.value = editing.models?.planner_llm || "";
    form.elements.diagnosis_model_id.value = editing.models?.diagnosis_llm || "";
    form.elements.composer_model_id.value = editing.models?.composer_llm || "";
    form.elements.memory_model_id.value = editing.models?.memory_llm || "";
    form.elements.ocr_model_id.value = editing.image_capabilities?.ocr?.default_model_id || "";
    form.elements.vlm_model_id.value = editing.image_capabilities?.vlm?.default_model_id || "";
    form.elements.instruction.value = editing.instruction || "";
    form.elements.auto_alert_enabled.checked = editing.auto_alert_enabled ?? true;
    toggleAutoAlert();
    form.elements.readonly_database_enabled.checked = editing.readonly_database_enabled !== false;
    form.elements.allow_agent_execution.checked = editing.allow_agent_execution === true;
    form.elements.auto_observe_min_severity.value = editing.auto_observe_min_severity || "CRITICAL";
    form.elements.auto_observe_min_target_level.value = String(editing.auto_observe_min_target_level || 1);
    form.elements.alert_cooldown_seconds.value = String(editing.alert_cooldown_seconds ?? 900);
    const policy = editing.knowledge_policy || {};
    form.elements.knowledge_enabled.checked = policy.enabled !== false;
    form.elements.max_security_level.value = String(policy.max_security_level || 3);
    form.elements.maximum_references.value = String(policy.maximum_references || 8);
    setChecked("allowed_business_system_ids", editing.allowed_business_system_ids);
    setChecked("capabilities", editing.capabilities);
    setChecked("knowledge_source_kinds", policy.source_kinds || ["MANUAL", "DIAGNOSIS_CASE"]);
    setChecked("knowledge_base_ids", policy.knowledge_base_ids);
    setChecked("knowledge_tag_ids", policy.tag_ids);
    const controlled = document.querySelector('[name="capabilities"][value="CONTROLLED_CHANGE"]');
    if (controlled && values(editing.capabilities).includes("CONTROLLED_CHANGE")) controlled.checked = true;
    toggleKnowledgePolicy();
    renderCapabilityGroups();
    setStep(1);
    showResult();
    document.getElementById("agent-dialog").showModal();
    void loadBoundary([...selectionSets.systems], true);
  }

  function imageCapability(form, name) {
    const modelId = form.elements[`${name}_model_id`].value;
    return modelId ? { default_model_id: modelId, allowed_model_ids: [modelId] } : null;
  }

  function buildPayload(form) {
    const capabilityValues = [...new Set([...selected("capabilities"), "MONITORING_QUERY"])];
    if (editing && values(editing.capabilities).includes("CONTROLLED_CHANGE")) capabilityValues.push("CONTROLLED_CHANGE");
    const knowledgeEnabled = form.elements.knowledge_enabled.checked;
    const modelsPayload = {
      ...editing?.models,
      planner_llm: form.elements.planner_model_id.value || null,
      diagnosis_llm: form.elements.diagnosis_model_id.value || null,
      composer_llm: form.elements.composer_model_id.value || null,
      memory_llm: form.elements.memory_model_id.value || form.elements.composer_model_id.value || null,
    };
    return {
      display_name: form.elements.display_name.value.trim(),
      description: form.elements.description.value.trim() || null,
      allowed_business_system_ids: [...selectionSets.systems],
      capabilities: [...new Set(capabilityValues)],
      knowledge_policy: {
        enabled: knowledgeEnabled,
        source_kinds: selected("knowledge_source_kinds"),
        knowledge_base_ids: [...selectionSets.knowledgeBases],
        tag_ids: [...selectionSets.tags],
        max_security_level: Number(form.elements.max_security_level.value || 3),
        maximum_references: Number(form.elements.maximum_references.value || 8),
      },
      // 本页不维护资源动作授权，PATCH 省略以保留服务端配置，不将未加载字段写成空授权。
      auto_alert_enabled: form.elements.auto_alert_enabled.checked,
      readonly_database_enabled: form.elements.readonly_database_enabled.checked,
      allow_agent_execution: form.elements.allow_agent_execution.checked,
      auto_observe_min_severity: form.elements.auto_observe_min_severity.value,
      auto_observe_min_target_level: Number(form.elements.auto_observe_min_target_level.value || 1),
      alert_cooldown_seconds: Number(form.elements.alert_cooldown_seconds.value),
      models: Object.fromEntries(Object.entries(modelsPayload).filter(([, value]) => value)),
      image_capabilities: { ocr: imageCapability(form, "ocr"), vlm: imageCapability(form, "vlm") },
      instruction: form.elements.instruction.value.trim() || null,
      config: editing?.config || {},
      ...(editing ? { expected_row_version: editing.row_version } : {}),
    };
  }

  async function save(event) {
    event.preventDefault();
    if (saving) return;
    if (currentStep < 6) {
      nextStep();
      return;
    }
    const form = event.currentTarget;
    for (let step = 1; step <= 5; step += 1) {
      if (!validateStep(step)) { setStep(step); return; }
    }
    const body = buildPayload(form);
    const button = document.getElementById("save-agent");
    const buttonLabel = button.textContent;
    const wasEditing = Boolean(editing);
    saving = true;
    setBusy();
    button.textContent = wasEditing ? "保存中…" : "创建中…";
    try {
      const path = editing ? `${api}/agents/${encodeURIComponent(editing.agent_id)}` : `${api}/agents`;
      const options = { method: editing ? "PATCH" : "POST", body: JSON.stringify(body) };
      if (editing) options.headers = { "If-Match": `"rv-${editing.row_version}"` };
      await KBotAIOpsAuth.request(path, options);
      document.getElementById("agent-dialog").close();
      shell.toast(editing ? "Agent 新版本已保存" : "Agent 已创建");
      await load();
    } catch (error) {
      showResult(error.message, "bad");
    } finally {
      saving = false;
      setBusy();
      button.textContent = buttonLabel;
    }
  }

  function confirmDelete(agent) {
    const dialog = document.getElementById("agent-confirm-dialog");
    document.getElementById("agent-confirm-message").textContent = `“${agent.display_name}”将停止接收新任务，历史运行与报告仍会保留。`;
    dialog.returnValue = "cancel";
    return new Promise((resolve) => {
      dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), { once: true });
      dialog.showModal();
      dialog.querySelector('[value="cancel"]').focus();
    });
  }

  async function commandAgent(agent, command) {
    if (!agent || saving) return;
    saving = true;
    setBusy();
    try {
      await KBotAIOpsAuth.request(`${api}/agents/${encodeURIComponent(agent.agent_id)}`, {
        method: "PATCH", body: JSON.stringify({ expected_row_version: agent.row_version, status: command === "enable" ? "ACTIVE" : "DISABLED" }),
      });
      await load();
    } catch (error) {
      shell.toast(error.message);
    } finally {
      saving = false;
      setBusy();
    }
  }

  async function deleteAgent(agent) {
    if (!agent || saving || !(await confirmDelete(agent))) return;
    saving = true;
    setBusy();
    try {
      await KBotAIOpsAuth.request(`${api}/agents/${encodeURIComponent(agent.agent_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${agent.row_version}"` } });
      shell.toast("Agent 已删除");
      await load();
    } catch (error) {
      shell.toast(error.message);
    } finally {
      saving = false;
      setBusy();
    }
  }

  async function load() {
    loading = true;
    setBusy();
    const pageError = document.getElementById("agent-page-error");
    pageError.hidden = true;
    try {
      const [agentRows, systemPage, modelRows, tagPage, knowledgeBasePage] = await Promise.all([
        KBotAIOpsAuth.request(`${api}/agents`),
        KBotAIOpsAuth.request(`${api}/business-systems?limit=200`),
        KBotAIOpsAuth.request("/api/v1/model-catalog"),
        KBotAIOpsAuth.request(`${api}/operations-knowledge/tags?limit=200`),
        KBotAIOpsAuth.request(`${api}/operations-knowledge/knowledge-bases`),
      ]);
      agents = values(agentRows);
      boundaryStates.clear();
      systems = values(systemPage?.items);
      models = values(modelRows);
      knowledgeTags = values(tagPage?.items || tagPage);
      knowledgeBases = values(knowledgeBasePage?.items || knowledgeBasePage);
      renderResources();
      const environments = [...new Set(systems.filter((item) => item.status === "ENABLED").map((item) => item.environment).filter(Boolean))].sort((left, right) => String(left).localeCompare(String(right), "zh"));
      document.getElementById("agent-picker-environment").innerHTML = '<option value="">全部环境</option>' + environments.map((item) => `<option value="${escape(item)}">${escape(item)}</option>`).join("");
      renderRows();
      for (const agent of agents) void loadBoundary(values(agent.allowed_business_system_ids));
      catalogReady = true;
    } catch (error) {
      pageError.textContent = `Agent 加载失败：${error.message}`;
      pageError.hidden = false;
      if (!catalogReady) document.getElementById("agent-rows").innerHTML = '<p class="resource-list-empty">无法读取 Agent 目录，请点击“刷新目录”重试。</p>';
    } finally {
      loading = false;
      setBusy();
    }
  }

  shell.ready.then(async (access) => {
    if (!access) return;
    const dialog = document.getElementById("agent-dialog");
    dialog.querySelectorAll("[data-close-dialog]").forEach((button) => { button.onclick = () => { if (!saving) dialog.close(); }; });
    [dialog, document.getElementById("agent-picker-dialog"), document.getElementById("agent-confirm-dialog")].forEach((panel) => {
      panel.addEventListener("cancel", (event) => { if (saving) event.preventDefault(); });
      panel.addEventListener("click", (event) => {
        if (event.target !== panel || saving) return;
        const rect = panel.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) panel.close();
      });
    });
    document.getElementById("create-agent").onclick = openCreate;
    document.getElementById("refresh-agents").onclick = load;
    document.getElementById("agent-search").oninput = renderRows;
    document.querySelector('[name="knowledge_enabled"]').onchange = () => { toggleKnowledgePolicy(); renderCapabilityGroups(); };
    document.querySelector('[name="auto_alert_enabled"]').onchange = toggleAutoAlert;
    document.getElementById("agent-form").addEventListener("change", () => { renderCapabilityGroups(); renderBoundaryResources(); if (currentStep === 6) renderReview(); });
    document.addEventListener("click", (event) => { if (!event.target.closest(".management-catalog-overflow")) closeMenus(); });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeMenus(); });
    document.getElementById("agent-form").onsubmit = save;
    document.querySelectorAll("[data-agent-step-button]").forEach((button) => { button.onclick = () => setStep(button.dataset.agentStepButton); });
    document.getElementById("agent-previous-step").onclick = () => setStep(currentStep - 1);
    document.getElementById("agent-next-step").onclick = nextStep;
    document.getElementById("open-agent-system-picker").onclick = () => openPicker("systems");
    document.getElementById("open-agent-knowledge-base-picker").onclick = () => openPicker("knowledgeBases");
    document.getElementById("open-agent-tag-picker").onclick = () => openPicker("tags");
    document.getElementById("close-agent-picker").onclick = () => document.getElementById("agent-picker-dialog").close();
    document.getElementById("apply-agent-picker").onclick = () => document.getElementById("agent-picker-dialog").close();
    document.getElementById("agent-picker-search").oninput = () => { pickerPage = 1; renderPicker(); };
    document.getElementById("agent-picker-environment").onchange = () => { pickerPage = 1; renderPicker(); };
    document.getElementById("agent-picker-previous").onclick = () => { pickerPage -= 1; renderPicker(); };
    document.getElementById("agent-picker-next").onclick = () => { pickerPage += 1; renderPicker(); };
    await load();
  }).catch((error) => {
    document.getElementById("create-agent").disabled = true;
    const pageError = document.getElementById("agent-page-error");
    pageError.textContent = `Agent 加载失败：${error.message}`;
    pageError.hidden = false;
    document.getElementById("agent-rows").innerHTML = '<p class="resource-list-empty">无法读取 Agent 目录。</p>';
  });
})();
