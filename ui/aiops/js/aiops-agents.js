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
    ["REPORT_CASE_QUERY", "历史案例", "已审核发布的诊断案例"],
    ["ACTIVE_DIAGNOSIS", "主动诊断", "按自然语言范围执行有界诊断"],
    ["CONTROLLED_CHANGE", "受控变更", "仅保留已有资源级授权；新增授权需在资源治理中完成"],
  ];
  const capabilityLabels = Object.fromEntries(capabilityCatalog.map(([key, label]) => [key, label]));
  const providerLabels = {
    local_deepseek: "本地部署 · DeepSeek",
    api_deepseek: "API 调用 · DeepSeek",
    api_qwen: "API 调用 · Qwen",
    chatgpt: "API 调用 · OpenAI",
    oci: "API 调用 · OCI",
  };

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
  const queryCapabilities = ["RESOURCE_CATALOG_QUERY", "MONITORING_QUERY", "ALERT_QUERY", "INSPECTION_QUERY", "REPORT_CASE_QUERY"];
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
    ["agent-planner-model", "agent-model", "agent-composer-model", "agent-memory-model"].forEach((id) => renderModelOptions(id, 1, id === "agent-memory-model" ? "复用回答组织模型" : id === "agent-model" ? "不启用主动诊断" : "请选择"));
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
      body.innerHTML = `<p class="resource-list-empty">${keyword ? "没有匹配的 Agent" : "当前范围内暂无 Agent"}</p>`;
      return;
    }
    body.innerHTML = visibleAgents.map((agent) => {
      const names = values(agent.allowed_business_system_ids).map(systemName);
      const capabilities = values(agent.capabilities).map((item) => capabilityLabels[item] || item);
      const status = { ACTIVE: ["运行中", "status-success"], DRAFT: ["草稿", "status-warning"], DISABLED: ["已停用", "status-muted"] }[agent.status] || [agent.status, "status-muted"];
      const button = (action, label, danger = false) => `<button type="button" class="${danger ? "danger-row-action" : "row-action"}" data-agent-action="${action}" data-agent-id="${escape(agent.agent_id)}" ${saving ? "disabled" : ""}>${label}</button>`;
      return `<article class="management-catalog-card"><header><div><p class="eyebrow">AIOPS AGENT</p><h3>${escape(agent.display_name)}</h3></div><span class="status-pill ${status[1]}">${escape(status[0])}</span></header><p>${escape(agent.description || "尚未填写 Agent 说明。")}</p><dl><div><dt>业务系统</dt><dd>${names.length} 个</dd></div><div><dt>能力范围</dt><dd class="agent-scope" title="${escape(capabilities.join("、"))}"><span>${escape(capabilities.slice(0, 3).join("、") || "未配置")}</span>${capabilities.length > 3 ? `<small>+${capabilities.length - 3}</small>` : ""}</dd></div></dl><footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions">${button("edit", "编辑")}${agent.status === "ACTIVE" ? button("disable", "停用", true) : ["DRAFT", "DISABLED"].includes(agent.status) ? button("enable", "启用") : ""}</div><div class="management-catalog-overflow"><button type="button" class="row-action management-catalog-more" aria-haspopup="menu" aria-expanded="false"><span>更多</span><span aria-hidden="true">⌄</span></button><div class="management-catalog-menu" role="menu" hidden>${button("delete", "删除 Agent", true)}</div></div></footer></article>`;
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
    const capabilities = selected("capabilities");
    const count = queryCapabilities.filter((key) => capabilities.includes(key)).length;
    document.getElementById("agent-query-capabilities").checked = count === queryCapabilities.length;
    document.getElementById("agent-query-capabilities").indeterminate = count > 0 && count < queryCapabilities.length;
    document.getElementById("agent-query-capability-count").textContent = `资源、监控、告警、巡检与报告 · 已启用 ${count}/${queryCapabilities.length} 项`;
    document.getElementById("agent-query-group").classList.toggle("selected", count > 0);
    document.getElementById("agent-diagnosis-group").classList.toggle("selected", capabilities.includes("ACTIVE_DIAGNOSIS"));
    document.getElementById("agent-diagnosis-label").textContent = `诊断模型${capabilities.includes("ACTIVE_DIAGNOSIS") ? "（智能诊断必选）" : "（可选）"}`;
    document.getElementById("agent-controlled-change-note").hidden = !values(editing?.capabilities).includes("CONTROLLED_CHANGE");
    document.querySelectorAll('[name="knowledge_source_kinds"]').forEach((input) => input.closest("label").classList.toggle("selected", input.checked));
  }

  function resetChecks() {
    document.querySelectorAll('#agent-form input[type="checkbox"]').forEach((input) => { input.checked = false; });
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

  function validateStep(step) {
    const form = document.getElementById("agent-form");
    showResult();
    if (step === 1 && !form.elements.display_name.value.trim()) {
      showResult("请填写 Agent 名称。", "bad");
      return false;
    }
    if (step === 2 && !selectionSets.systems.size) {
      showResult("请至少选择一个业务系统。", "bad");
      return false;
    }
    if (step === 2 && !selected("capabilities").length && !values(editing?.capabilities).includes("CONTROLLED_CHANGE")) {
      showResult("请至少启用一项能力。", "bad");
      return false;
    }
    if (step === 2 && form.elements.knowledge_enabled.checked && !selected("knowledge_source_kinds").length) {
      showResult("请至少选择一种知识来源。", "bad");
      return false;
    }
    if (step === 2 && form.elements.knowledge_enabled.checked && !selectionSets.knowledgeBases.size) {
      showResult("请至少绑定一个运维知识库。", "bad");
      return false;
    }
    if (step === 2 && form.elements.knowledge_enabled.checked && !form.elements.maximum_references.checkValidity()) {
      showResult("最多引用须为 1 到 20 的整数。", "bad");
      return false;
    }
    if (step === 3 && form.elements.capture_enabled.checked && !form.elements.capture_objects.value.trim()) {
      showResult("开启计划采集前请明确允许对象范围。", "bad");
      return false;
    }
    if (step === 3 && (!form.elements.alert_cooldown_seconds.checkValidity() || !form.elements.capture_timeout.checkValidity())) {
      showResult("请填写有效的冷却时间和单语句时限。", "bad");
      return false;
    }
    if (step === 4 && editing?.status === "ACTIVE") {
      const missing = [];
      if (!form.elements.planner_model_id.value) missing.push("规划模型");
      if (!form.elements.composer_model_id.value) missing.push("回答组织模型");
      if (selected("capabilities").includes("ACTIVE_DIAGNOSIS") && !form.elements.diagnosis_model_id.value) missing.push("诊断模型");
      if (missing.length) {
        showResult(`运行中的 Agent 必须配置${missing.join("、")}。`, "bad");
        return false;
      }
    }
    return true;
  }

  function renderReview() {
    const capabilities = selected("capabilities").map((item) => capabilityLabels[item] || item);
    const knowledgeEnabled = document.getElementById("agent-form").elements.knowledge_enabled.checked;
    document.getElementById("agent-review").innerHTML = [
      `<article><span>业务系统</span><strong>${selectionSets.systems.size} 个</strong><small>${escape(selectionRows("systems").map((item) => item.title).join("、") || "未选择")}</small></article>`,
      `<article><span>能力范围</span><strong>${capabilities.length} 项</strong><small>${escape(capabilities.join("、") || "未启用")}</small></article>`,
      `<article><span>运维知识</span><strong>${knowledgeEnabled ? `${selectionSets.knowledgeBases.size} 个知识库` : "未启用"}</strong><small>${knowledgeEnabled ? `最高 L${escape(document.getElementById("agent-form").elements.max_security_level.value)} · ${selectionSets.tags.size ? `${selectionSets.tags.size} 个标签` : "全部标签"}` : ""}</small></article>`,
      `<article><span>模型角色</span><strong>${["planner_model_id", "composer_model_id", "diagnosis_model_id"].filter((name) => document.getElementById("agent-form").elements[name].value).length} 个已配置</strong><small>规划、回答组织与诊断模型</small></article>`,
    ].join("");
  }

  function setStep(step) {
    currentStep = Math.max(1, Math.min(5, Number(step) || 1));
    document.querySelectorAll("[data-agent-step]").forEach((panel) => { panel.hidden = Number(panel.dataset.agentStep) !== currentStep; });
    document.querySelectorAll("[data-agent-step-button]").forEach((button) => {
      const buttonStep = Number(button.dataset.agentStepButton);
      button.classList.toggle("active", buttonStep === currentStep);
      button.classList.toggle("complete", buttonStep < currentStep);
      button.disabled = saving;
      button.querySelector("b").textContent = buttonStep < currentStep ? "✓" : `0${buttonStep}`;
    });
    document.getElementById("agent-previous-step").hidden = currentStep === 1;
    document.getElementById("agent-next-step").hidden = currentStep === 5;
    document.getElementById("save-agent").hidden = currentStep !== 5;
    if (currentStep === 5) renderReview();
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
    form.elements.knowledge_enabled.checked = true;
    form.elements.readonly_database_enabled.checked = true;
    form.elements.allow_agent_execution.checked = false;
    form.elements.alert_cooldown_seconds.value = "900";
    form.elements.capture_enabled.checked = false;
    form.elements.capture_distinct.checked = true;
    form.elements.capture_native_binds.checked = false;
    form.elements.capture_objects.value = "";
    form.elements.capture_timeout.value = "5";
    form.elements.max_security_level.value = "3";
    form.elements.maximum_references.value = "8";
    toggleKnowledgePolicy();
    renderCapabilityGroups();
    setStep(1);
    showResult();
    document.getElementById("agent-dialog").showModal();
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
    form.elements.planner_model_id.value = editing.models?.planner_llm || "";
    form.elements.diagnosis_model_id.value = editing.models?.diagnosis_llm || "";
    form.elements.composer_model_id.value = editing.models?.composer_llm || "";
    form.elements.memory_model_id.value = editing.models?.memory_llm || "";
    form.elements.ocr_model_id.value = editing.image_capabilities?.ocr?.default_model_id || "";
    form.elements.vlm_model_id.value = editing.image_capabilities?.vlm?.default_model_id || "";
    form.elements.instruction.value = editing.instruction || "";
    form.elements.readonly_database_enabled.checked = editing.readonly_database_enabled !== false;
    form.elements.allow_agent_execution.checked = editing.allow_agent_execution === true;
    form.elements.auto_observe_min_severity.value = editing.auto_observe_min_severity || "CRITICAL";
    form.elements.auto_observe_min_target_level.value = String(editing.auto_observe_min_target_level || 1);
    form.elements.alert_cooldown_seconds.value = String(editing.alert_cooldown_seconds ?? 900);
    const capture = editing.sql_plan_capture || {};
    form.elements.capture_enabled.checked = capture.enabled === true;
    form.elements.capture_objects.value = values(capture.allowed_objects).join("\n");
    form.elements.capture_distinct.checked = capture.require_distinct_approver !== false;
    form.elements.capture_native_binds.checked = capture.native_bind_comparison_enabled === true;
    form.elements.capture_timeout.value = String(capture.max_statement_timeout_seconds ?? 5);
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
  }

  function imageCapability(form, name) {
    const modelId = form.elements[`${name}_model_id`].value;
    return modelId ? { default_model_id: modelId, allowed_model_ids: [modelId] } : null;
  }

  function buildPayload(form) {
    const capabilityValues = selected("capabilities");
    if (editing && values(editing.capabilities).includes("CONTROLLED_CHANGE")) capabilityValues.push("CONTROLLED_CHANGE");
    const knowledgeEnabled = form.elements.knowledge_enabled.checked;
    const modelsPayload = {
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
        source_kinds: knowledgeEnabled ? selected("knowledge_source_kinds") : ["MANUAL", "DIAGNOSIS_CASE"],
        knowledge_base_ids: knowledgeEnabled ? [...selectionSets.knowledgeBases] : [],
        tag_ids: knowledgeEnabled ? [...selectionSets.tags] : [],
        max_security_level: Number(form.elements.max_security_level.value || 3),
        maximum_references: Number(form.elements.maximum_references.value || 8),
      },
      controlled_action_execution: values(editing?.controlled_action_execution),
      sql_plan_capture: {
        enabled: form.elements.capture_enabled.checked,
        allowed_objects: form.elements.capture_objects.value.split("\n").map((name) => name.trim()).filter(Boolean),
        require_distinct_approver: form.elements.capture_distinct.checked,
        native_bind_comparison_enabled: form.elements.capture_native_binds.checked,
        max_statement_timeout_seconds: Number(form.elements.capture_timeout.value),
      },
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
    if (currentStep < 5) {
      nextStep();
      return;
    }
    const form = event.currentTarget;
    for (let step = 1; step <= 4; step += 1) {
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
      systems = values(systemPage?.items);
      models = values(modelRows);
      knowledgeTags = values(tagPage?.items || tagPage);
      knowledgeBases = values(knowledgeBasePage?.items || knowledgeBasePage);
      renderResources();
      const environments = [...new Set(systems.filter((item) => item.status === "ENABLED").map((item) => item.environment).filter(Boolean))].sort((left, right) => String(left).localeCompare(String(right), "zh"));
      document.getElementById("agent-picker-environment").innerHTML = '<option value="">全部环境</option>' + environments.map((item) => `<option value="${escape(item)}">${escape(item)}</option>`).join("");
      renderRows();
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
    document.getElementById("agent-query-capabilities").onchange = (event) => {
      document.querySelectorAll('[name="capabilities"]').forEach((input) => { if (queryCapabilities.includes(input.value)) input.checked = event.target.checked; });
      renderCapabilityGroups();
    };
    document.getElementById("agent-form").addEventListener("change", renderCapabilityGroups);
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
