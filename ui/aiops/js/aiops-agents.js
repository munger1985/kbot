(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const escape = (value) => shell.escape(value ?? "—");
  const values = (items) => Array.isArray(items) ? items : [];
  const capabilityCatalog = [
    ["RESOURCE_CATALOG_QUERY", "资源信息", "查询业务系统拓扑、主机、IP 与资源属性"],
    ["KNOWLEDGE_QA", "运维知识", "检索已审批手册与历史诊断案例"],
    ["MONITORING_QUERY", "监控现状", "查询业务系统当前指标与趋势"],
    ["ALERT_QUERY", "告警查询", "查询当前与历史告警"],
    ["INSPECTION_QUERY", "巡检查询", "读取巡检结果与报告"],
    ["REPORT_CASE_QUERY", "报告与案例", "查询诊断、告警和巡检沉淀"],
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
  let editing = null;

  const systemName = (id) => systems.find((item) => item.business_system_id === id)?.display_name || shell.short(id);
  const selected = (name) => [...document.querySelectorAll(`[name="${name}"]:checked`)].map((input) => input.value);

  function showResult(message = "", tone = "") {
    const result = document.getElementById("agent-result");
    result.textContent = message;
    result.dataset.tone = tone;
  }

  function modelLabel(model) {
    const provider = providerLabels[String(model.provider || "").toLowerCase()] || model.provider || "未知提供方";
    return `${model.display_name} · ${model.served_model_name} · ${provider}`;
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
    const rows = models.filter((item) => Number(item.category) === category);
    document.getElementById(elementId).innerHTML = `<option value="">${emptyLabel}</option>${rows.map((item) => `<option value="${escape(item.model_id)}">${escape(category === 1 ? modelLabel(item) : `${item.display_name} · ${item.served_model_name}`)}</option>`).join("")}`;
  }

  function renderResources() {
    document.getElementById("agent-business-systems").innerHTML = systems.length
      ? systems.map((item) => `<label class="agent-switch-row"><input type="checkbox" name="allowed_business_system_ids" value="${escape(item.business_system_id)}"><span><strong>${escape(item.display_name)}</strong><small>${escape(item.system_code)} · ${escape(item.environment)} · ${Number(item.resource_count || 0)} 个资源 · ${item.status === "ENABLED" ? "已上线" : "未上线"}</small></span></label>`).join("")
      : '<div class="ops-error">暂无业务系统。请先建立业务系统与资源拓扑。</div>';
    document.getElementById("agent-capabilities").innerHTML = capabilityCatalog.map(([key, label, help]) => `<label class="agent-switch-row"><input type="checkbox" name="capabilities" value="${key}" ${key === "CONTROLLED_CHANGE" ? "disabled" : ""}><span><strong>${label}</strong><small>${help}</small></span></label>`).join("");
    document.getElementById("agent-knowledge-tags").innerHTML = knowledgeTags.length
      ? knowledgeTags.filter((item) => item.status === "ACTIVE").map((item) => `<label class="agent-scope-option"><input type="checkbox" name="knowledge_tag_ids" value="${escape(item.tag_id)}"><span>${escape(item.display_name)}</span></label>`).join("")
      : '<p class="agent-scope-empty">未配置知识标签时，将在安全级别内使用全部已审批运维知识。</p>';
    ["agent-planner-model", "agent-model", "agent-composer-model", "agent-memory-model"].forEach((id) => renderModelOptions(id, 1, id === "agent-memory-model" ? "复用回答模型" : "请选择 LLM"));
    renderModelOptions("agent-ocr-model", 6, "不启用 OCR");
    renderModelOptions("agent-vlm-model", 5, "不启用 VLM");
  }

  function renderSummary() {
    const active = agents.filter((item) => item.status === "ACTIVE").length;
    document.getElementById("agent-summary").innerHTML = [
      `<span><strong>${agents.length}</strong> 全部</span>`,
      `<span><strong>${active}</strong> 已启用</span>`,
      `<span><strong>${systems.length}</strong> 业务系统</span>`,
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
      const names = values(agent.allowed_business_system_ids).map(systemName);
      const capabilities = values(agent.capabilities).map((item) => capabilityLabels[item] || item);
      const knowledge = agent.knowledge_policy?.enabled
        ? `${values(agent.knowledge_policy.source_kinds).length} 类来源 · 最多 ${agent.knowledge_policy.maximum_references || 8} 条引用`
        : "未启用";
      return `<tr>
        <td><strong>${escape(agent.display_name)}</strong><small class="agent-row-description">${escape(agent.description || "未填写说明")}</small></td>
        <td>${shell.badge(agent.status)}</td>
        <td><strong>${names.length} 个业务系统</strong><small class="agent-row-description">${escape(names.join("、") || "—")}</small></td>
        <td><strong>${capabilities.length} 项能力</strong><small class="agent-row-description">${escape(capabilities.join("、") || "—")}</small></td>
        <td><strong>${escape(knowledge)}</strong><small class="agent-row-description">${agent.auto_alert_enabled ? `自动观察 ${escape(agent.auto_observe_min_severity || "CRITICAL")} 起` : "自动观察已关闭"}</small></td>
        <td><div class="ops-actions"><button type="button" data-agent-action="edit" data-agent-id="${escape(agent.agent_id)}">编辑</button><button type="button" class="danger" data-agent-action="delete" data-agent-id="${escape(agent.agent_id)}">删除</button></div></td>
      </tr>`;
    }).join("");
    body.querySelectorAll("[data-agent-action]").forEach((button) => {
      button.onclick = () => button.dataset.agentAction === "edit"
        ? openEdit(button.dataset.agentId)
        : deleteAgent(agents.find((item) => item.agent_id === button.dataset.agentId), button);
    });
  }

  function resetChecks() {
    document.querySelectorAll('#agent-form input[type="checkbox"]').forEach((input) => { input.checked = false; });
  }

  function setChecked(name, selectedValues) {
    const chosen = new Set(values(selectedValues).map(String));
    document.querySelectorAll(`[name="${name}"]`).forEach((input) => { input.checked = chosen.has(String(input.value)); });
  }

  function toggleKnowledgePolicy() {
    const enabled = document.querySelector('[name="knowledge_enabled"]').checked;
    document.getElementById("agent-knowledge-policy-fields").hidden = !enabled;
    const knowledgeCapability = document.querySelector('[name="capabilities"][value="KNOWLEDGE_QA"]');
    if (knowledgeCapability) knowledgeCapability.checked = enabled;
  }

  function toggleAlertSettings() {
    const enabled = document.querySelector('[name="auto_alert_enabled"]').checked;
    document.getElementById("agent-alert-settings").hidden = !enabled;
    document.getElementById("agent-alert-settings").querySelectorAll("select,input").forEach((input) => { input.disabled = !enabled; });
  }

  function openCreate() {
    editing = null;
    const form = document.getElementById("agent-form");
    form.reset();
    resetChecks();
    document.getElementById("agent-dialog-title").textContent = "新增 Agent";
    document.getElementById("save-agent").textContent = "创建 Agent";
    form.elements.status.value = "DRAFT";
    const defaultModel = defaultLlmId();
    form.elements.planner_model_id.value = defaultModel;
    form.elements.diagnosis_model_id.value = defaultModel;
    form.elements.composer_model_id.value = defaultModel;
    form.elements.memory_model_id.value = "";
    setChecked("capabilities", ["RESOURCE_CATALOG_QUERY", "KNOWLEDGE_QA", "MONITORING_QUERY", "ALERT_QUERY", "INSPECTION_QUERY", "REPORT_CASE_QUERY", "ACTIVE_DIAGNOSIS"]);
    setChecked("knowledge_source_kinds", ["MANUAL", "DIAGNOSIS_CASE"]);
    form.elements.knowledge_enabled.checked = true;
    form.elements.auto_alert_enabled.checked = true;
    form.elements.max_security_level.value = "3";
    form.elements.maximum_references.value = "8";
    toggleKnowledgePolicy();
    toggleAlertSettings();
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
    form.elements.status.value = editing.status || "DRAFT";
    form.elements.planner_model_id.value = editing.models?.planner_llm || "";
    form.elements.diagnosis_model_id.value = editing.models?.diagnosis_llm || "";
    form.elements.composer_model_id.value = editing.models?.composer_llm || "";
    form.elements.memory_model_id.value = editing.models?.memory_llm || "";
    form.elements.ocr_model_id.value = editing.image_capabilities?.ocr?.default_model_id || "";
    form.elements.vlm_model_id.value = editing.image_capabilities?.vlm?.default_model_id || "";
    form.elements.instruction.value = editing.instruction || "";
    form.elements.auto_alert_enabled.checked = editing.auto_alert_enabled !== false;
    form.elements.auto_observe_min_severity.value = editing.auto_observe_min_severity || "CRITICAL";
    form.elements.auto_observe_min_target_level.value = String(editing.auto_observe_min_target_level || 1);
    form.elements.alert_cooldown_minutes.value = String(editing.alert_cooldown_minutes ?? 15);
    const policy = editing.knowledge_policy || {};
    form.elements.knowledge_enabled.checked = policy.enabled !== false;
    form.elements.max_security_level.value = String(policy.max_security_level || 3);
    form.elements.maximum_references.value = String(policy.maximum_references || 8);
    setChecked("allowed_business_system_ids", editing.allowed_business_system_ids);
    setChecked("capabilities", editing.capabilities);
    setChecked("knowledge_source_kinds", policy.source_kinds || ["MANUAL", "DIAGNOSIS_CASE"]);
    setChecked("knowledge_tag_ids", policy.tag_ids);
    const controlled = document.querySelector('[name="capabilities"][value="CONTROLLED_CHANGE"]');
    if (controlled && values(editing.capabilities).includes("CONTROLLED_CHANGE")) controlled.checked = true;
    toggleKnowledgePolicy();
    toggleAlertSettings();
    showResult();
    document.getElementById("agent-dialog").showModal();
  }

  function imageCapability(form, name) {
    const modelId = form.elements[`${name}_model_id`].value || null;
    return { enabled: Boolean(modelId), default_model_id: modelId, allowed_model_ids: modelId ? [modelId] : [] };
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
      allowed_business_system_ids: selected("allowed_business_system_ids"),
      capabilities: [...new Set(capabilityValues)],
      knowledge_policy: {
        enabled: knowledgeEnabled,
        source_kinds: knowledgeEnabled ? selected("knowledge_source_kinds") : ["MANUAL", "DIAGNOSIS_CASE"],
        tag_ids: knowledgeEnabled ? selected("knowledge_tag_ids") : [],
        max_security_level: Number(form.elements.max_security_level.value || 3),
        maximum_references: Number(form.elements.maximum_references.value || 8),
      },
      controlled_action_execution: values(editing?.controlled_action_execution),
      auto_alert_enabled: form.elements.auto_alert_enabled.checked,
      auto_observe_min_severity: form.elements.auto_observe_min_severity.value,
      auto_observe_min_target_level: Number(form.elements.auto_observe_min_target_level.value || 1),
      alert_cooldown_minutes: Number(form.elements.alert_cooldown_minutes.value || 15),
      models: Object.fromEntries(Object.entries(modelsPayload).filter(([, value]) => value)),
      image_capabilities: { ocr: imageCapability(form, "ocr"), vlm: imageCapability(form, "vlm") },
      instruction: form.elements.instruction.value.trim() || null,
      config: editing?.config || {},
      status: form.elements.status.value,
    };
  }

  async function save(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const body = buildPayload(form);
    if (!body.allowed_business_system_ids.length) return showResult("请至少选择一个业务系统。", "bad");
    if (!body.capabilities.length) return showResult("请至少启用一项能力。", "bad");
    if (body.knowledge_policy.enabled && !body.knowledge_policy.source_kinds.length) return showResult("请至少选择一种知识来源。", "bad");
    const button = document.getElementById("save-agent");
    const buttonLabel = button.textContent;
    const wasEditing = Boolean(editing);
    button.disabled = true;
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
      button.disabled = false;
      button.textContent = buttonLabel;
    }
  }

  async function deleteAgent(agent, button) {
    if (!agent || !globalThis.confirm(`确认删除 Agent“${agent.display_name}”？历史运行和报告仍会保留。`)) return;
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${api}/agents/${encodeURIComponent(agent.agent_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${agent.row_version}"` } });
      shell.toast("Agent 已删除");
      await load();
    } catch (error) {
      shell.toast(error.message);
      button.disabled = false;
    }
  }

  async function load() {
    const [agentRows, systemPage, modelRows, tagPage] = await Promise.all([
      KBotAIOpsAuth.request(`${api}/agents`),
      KBotAIOpsAuth.request(`${api}/business-systems?limit=200`),
      KBotAIOpsAuth.request(`${api}/agent-model-catalog`),
      KBotAIOpsAuth.request(`${api}/operations-knowledge/tags?limit=200`),
    ]);
    agents = values(agentRows);
    systems = values(systemPage?.items);
    models = values(modelRows);
    knowledgeTags = values(tagPage?.items || tagPage);
    renderResources();
    renderRows();
  }

  addEventListener("DOMContentLoaded", async () => {
    await shell.initialize();
    const dialog = document.getElementById("agent-dialog");
    dialog.querySelectorAll("[data-close-dialog]").forEach((button) => { button.onclick = () => dialog.close(); });
    document.getElementById("create-agent").onclick = openCreate;
    document.querySelector('[name="knowledge_enabled"]').onchange = toggleKnowledgePolicy;
    document.querySelector('[name="auto_alert_enabled"]').onchange = toggleAlertSettings;
    document.getElementById("agent-form").onsubmit = save;
    await load();
  }, { once: true });
})();
