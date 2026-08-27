(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let agents = [];
  let sources = [];
  let targets = [];
  let models = [];
  let sourceBindings = [];
  let bindingTargetId = "";
  let editing = null;

  const escape = (value) => shell.escape(value ?? "—");
  const sourceName = (id) => sources.find((item) => item.source_id === id)?.display_name || shell.short(id);
  const targetName = (id) => targets.find((item) => item.target_id === id)?.display_name || shell.short(id);

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
      `<span><strong>${sources.length}</strong> 可用监控源</span>`,
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
      const sourceNames = (agent.diagnostic_source_ids || []).map((id) => escape(sourceName(id)));
      const target = agent.target_id ? escape(targetName(agent.target_id)) : "不直连数据库";
      const access = agent.allow_change_execution ? "只读诊断 + 审批后变更" : (agent.target_id ? "只读诊断" : "仅监控证据");
      return `<tr>
        <td><strong>${escape(agent.display_name)}</strong><small class="agent-row-description">${escape(agent.description || "未填写说明")}</small></td>
        <td>${shell.badge(agent.status)}</td>
        <td><strong>${sourceNames.length} 个监控源</strong><small class="agent-row-description">${sourceNames.join("、") || "—"}</small></td>
        <td><strong>${access}</strong><small class="agent-row-description">${target}</small></td>
        <td>${agent.auto_alert_enabled ? `<strong>${escape(agent.auto_observe_min_severity)} 起</strong><small class="agent-row-description">冷却 ${escape(agent.alert_cooldown_minutes)} 分钟</small>` : "已关闭"}</td>
        <td><button type="button" data-agent-id="${escape(agent.agent_id)}">编辑</button></td>
      </tr>`;
    }).join("");
    body.querySelectorAll("[data-agent-id]").forEach((button) => {
      button.addEventListener("click", () => void openEdit(button.dataset.agentId));
    });
  }

  function locatorLabel(sourceType) {
    if (sourceType === "PROMETHEUS") return "Prometheus instance 标签值";
    if (sourceType === "ALERTMANAGER") return "告警中的目标标签值";
    if (sourceType === "ZABBIX") return "Zabbix Host 名称";
    if (sourceType === "OEM") return "OEM Target 标识";
    if (sourceType === "LOKI") return "关联标识";
    return "监控系统中的目标标识";
  }

  function locatorHelp(sourceType) {
    if (sourceType === "PROMETHEUS") return "填写 Prometheus 指标 instance 标签的值，例如 oracle-dev-190。";
    if (sourceType === "ALERTMANAGER") return "填写告警 target_label 对应的值；Oracle 一键部署通常与 Target Key 相同。";
    if (sourceType === "LOKI") return "关联标识用于唯一映射该 Target；下面的精确标签用于查询日志。";
    return "填写该数据库在此监控源中的唯一标识。";
  }

  function sourceCard(source) {
    const sourceId = escape(source.source_id);
    const lokiFields = source.source_type === "LOKI" ? `<div class="agent-loki-fields">
      <div class="ops-field">
        <label>日志任务标签 job</label>
        <input data-loki-job maxlength="256" value="oracle_alert" placeholder="oracle_alert">
      </div>
      <div class="ops-field">
        <label>目标标签名称</label>
        <input data-loki-target-label maxlength="64" value="target_key" placeholder="target_key">
      </div>
      <div class="ops-field">
        <label>目标标签值</label>
        <input data-loki-target-value maxlength="256" placeholder="例如 oracle-dev-190">
      </div>
    </div>` : "";
    const prometheusFields = source.source_type === "PROMETHEUS" ? `<div class="agent-prometheus-fields">
      <div class="ops-field">
        <label>数据库主机的 Node Exporter target_key</label>
        <input data-prometheus-host-target maxlength="256" placeholder="例如 dev-db-host-190">
        <small>填写 Prometheus 中该数据库所在主机的 Node Exporter target_key；它可以与数据库 instance 不同。</small>
      </div>
    </div>` : "";
    return `<article class="agent-source-card" data-source-id="${sourceId}" data-source-type="${escape(source.source_type)}">
      <label class="agent-source-choice">
        <input type="checkbox" name="diagnostic_source_ids" value="${sourceId}">
        <span class="agent-source-identity"><strong>${escape(source.display_name)}</strong><small>${escape(source.source_type)}</small></span>
        <span class="agent-source-health">${escape(source.connectivity_status)}</span>
      </label>
      <div class="agent-source-mapping" hidden>
        <div class="agent-mapping-head"><strong>Target 映射</strong><span data-binding-state>尚未配置</span></div>
        <div class="ops-field">
          <label>${locatorLabel(source.source_type)}</label>
          <input data-locator-key maxlength="512" placeholder="例如 oracle-dev-190">
          <small>${locatorHelp(source.source_type)}</small>
        </div>
        ${prometheusFields}
        ${lokiFields}
      </div>
    </article>`;
  }

  function renderResources() {
    document.getElementById("agent-sources").innerHTML = sources.length
      ? sources.map(sourceCard).join("")
      : '<div class="ops-error">没有已启用且可连接的监控源，请先完成监控源配置。</div>';
    document.getElementById("agent-target").innerHTML = '<option value="">不允许数据库直连诊断</option>' + targets.map((target) => `<option value="${escape(target.target_id)}">${escape(target.display_name)} · ${escape(target.db_type)}${target.execution_credential_configured ? " · 执行凭据就绪" : " · 执行凭据待配置"}</option>`).join("");
    const diagnosisModels = models.filter((model) => Number(model.category) === 1);
    document.getElementById("agent-model").innerHTML = diagnosisModels.length
      ? '<option value="">请选择诊断模型</option>' + diagnosisModels.map((model) => `<option value="${escape(model.model_id)}">${escape(model.display_name)} · ${escape(model.served_model_name)}</option>`).join("")
      : '<option value="">没有已启用的 LLM，请先配置模型服务</option>';
  }

  function bindingFor(sourceId) {
    const matches = sourceBindings.filter((item) => item.source_id === sourceId);
    return matches.find((item) => item.status === "ACTIVE") || matches[0] || null;
  }

  function resetMappingInputs() {
    document.querySelectorAll(".agent-source-card").forEach((card) => {
      card.querySelector("[data-locator-key]").value = "";
      const job = card.querySelector("[data-loki-job]");
      const label = card.querySelector("[data-loki-target-label]");
      const value = card.querySelector("[data-loki-target-value]");
      const hostTarget = card.querySelector("[data-prometheus-host-target]");
      if (job) job.value = "oracle_alert";
      if (label) label.value = "target_key";
      if (value) {
        value.value = "";
        delete value.dataset.userEdited;
      }
      if (hostTarget) hostTarget.value = "";
      card.querySelector("[data-binding-state]").textContent = "尚未配置";
      card.dataset.bindingId = "";
    });
  }

  function applyBindings() {
    document.querySelectorAll(".agent-source-card").forEach((card) => {
      const binding = bindingFor(card.dataset.sourceId);
      if (!binding) return;
      card.dataset.bindingId = binding.binding_id;
      card.querySelector("[data-locator-key]").value = binding.source_locator_key || "";
      const labels = binding.source_locator?.labels || {};
      const job = card.querySelector("[data-loki-job]");
      const label = card.querySelector("[data-loki-target-label]");
      const value = card.querySelector("[data-loki-target-value]");
      const hostTarget = card.querySelector("[data-prometheus-host-target]");
      if (job) job.value = labels.job || "oracle_alert";
      if (label && value) {
        const targetLabel = Object.keys(labels).find((name) => name !== "job") || "target_key";
        label.value = targetLabel;
        value.value = labels[targetLabel] || "";
        value.dataset.userEdited = "true";
      }
      if (hostTarget) {
        hostTarget.value = binding.source_locator?.host_target_key || "";
      }
      const health = binding.health_status && binding.health_status !== "UNKNOWN" ? ` · ${binding.health_status}` : "";
      card.querySelector("[data-binding-state]").textContent = `${binding.status === "ACTIVE" ? "已建立" : "已停用"}${health}`;
    });
  }

  async function loadBindings(targetId) {
    resetMappingInputs();
    sourceBindings = [];
    bindingTargetId = "";
    if (!targetId) {
      syncSourceMappingVisibility();
      return;
    }
    const expectedTarget = targetId;
    document.getElementById("agent-binding-summary").textContent = "正在读取该 Target 的监控源映射…";
    const bindings = await KBotAIOpsAuth.request(`${api}/targets/${encodeURIComponent(targetId)}/source-bindings`);
    if (document.getElementById("agent-target").value !== expectedTarget) return;
    sourceBindings = Array.isArray(bindings) ? bindings : [];
    bindingTargetId = targetId;
    applyBindings();
    syncSourceMappingVisibility();
  }

  function syncSourceMappingVisibility() {
    const targetId = document.getElementById("agent-target").value;
    let selectedCount = 0;
    let mappedCount = 0;
    document.querySelectorAll(".agent-source-card").forEach((card) => {
      const checked = card.querySelector('[name="diagnostic_source_ids"]').checked;
      const show = Boolean(targetId && checked);
      card.classList.toggle("selected", checked);
      card.querySelector(".agent-source-mapping").hidden = !show;
      if (checked) selectedCount += 1;
      if (show && bindingFor(card.dataset.sourceId)?.status === "ACTIVE") mappedCount += 1;
    });
    const summary = document.getElementById("agent-binding-summary");
    if (!targetId) summary.textContent = "当前只使用监控证据；选择 Target 后可同时配置数据库只读诊断。";
    else if (!selectedCount) summary.textContent = "请选择监控源，随后填写该 Target 在监控系统中的标识。";
    else summary.textContent = `${selectedCount} 个监控源已选择，${mappedCount} 个已有有效 Target 映射；保存时会补齐或更新。`;
  }

  function toggleTargetFields() {
    const targetId = document.getElementById("agent-target").value;
    const target = targets.find((item) => item.target_id === targetId);
    const executionConfigured = Boolean(target?.execution_credential_configured);
    const executionToggle = document.querySelector('[name="allow_change_execution"]');
    document.getElementById("agent-change-field").hidden = !targetId;
    executionToggle.disabled = !targetId;
    if (!targetId) executionToggle.checked = false;
    document.getElementById("agent-change-help").textContent = executionConfigured
      ? "只开放系统支持的受控动作，仍必须进入人工审批链，不代表无人审批自动执行。"
      : "可以先保存允许变更；实际执行前仍必须在运维目标中配置独立的执行凭据，并通过人工审批。";
    syncSourceMappingVisibility();
  }

  function toggleAlertSettings() {
    const enabled = document.querySelector('[name="auto_alert_enabled"]').checked;
    document.getElementById("agent-alert-settings").classList.toggle("agent-settings-disabled", !enabled);
    document.getElementById("agent-min-severity").disabled = !enabled;
    document.getElementById("agent-cooldown").disabled = !enabled;
  }

  function openCreate() {
    editing = null;
    sourceBindings = [];
    bindingTargetId = "";
    const form = document.getElementById("agent-form");
    form.reset();
    resetMappingInputs();
    form.elements.status.value = "DRAFT";
    form.elements.alert_cooldown_minutes.value = 15;
    form.elements.auto_alert_enabled.checked = true;
    form.elements.status.disabled = true;
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
    const form = document.getElementById("agent-form");
    form.reset();
    resetMappingInputs();
    form.elements.status.disabled = false;
    form.elements.display_name.value = editing.display_name;
    form.elements.description.value = editing.description || "";
    form.elements.status.value = editing.status;
    form.elements.target_id.value = editing.target_id || "";
    form.elements.allow_change_execution.checked = Boolean(editing.allow_change_execution);
    form.elements.auto_alert_enabled.checked = Boolean(editing.auto_alert_enabled);
    form.elements.auto_observe_min_severity.value = editing.auto_observe_min_severity || "CRITICAL";
    form.elements.alert_cooldown_minutes.value = editing.alert_cooldown_minutes ?? 15;
    form.elements.diagnosis_model_id.value = editing.models?.diagnosis_llm || "";
    form.elements.instruction.value = editing.instruction || "";
    form.querySelectorAll('[name="diagnostic_source_ids"]').forEach((input) => {
      input.checked = (editing.diagnostic_source_ids || []).includes(input.value);
    });
    document.getElementById("agent-status-help").textContent = "启用前会检查监控源、Target 连通性以及两者之间的有效映射。";
    toggleTargetFields();
    toggleAlertSettings();
    showResult();
    document.getElementById("agent-dialog-title").textContent = "修改 Agent";
    document.getElementById("save-agent").textContent = "保存修改";
    document.getElementById("agent-dialog").showModal();
    if (editing.target_id) {
      try {
        await loadBindings(editing.target_id);
      } catch (error) {
        showResult(`读取 Target 映射失败：${error.message}`, "bad");
      }
    }
  }

  function payload(form) {
    const selectedSources = [...form.querySelectorAll('[name="diagnostic_source_ids"]:checked')].map((input) => input.value);
    if (!selectedSources.length) throw new Error("至少选择一个监控源。");
    const modelId = form.elements.diagnosis_model_id.value.trim();
    if (!modelId) throw new Error("请选择诊断模型。");
    const autoAlertEnabled = form.elements.auto_alert_enabled.checked;
    return {
      display_name: form.elements.display_name.value.trim(),
      description: form.elements.description.value.trim() || null,
      status: editing ? form.elements.status.value : "DRAFT",
      diagnostic_source_ids: selectedSources,
      target_id: form.elements.target_id.value || null,
      allow_change_execution: form.elements.allow_change_execution.checked,
      auto_alert_enabled: autoAlertEnabled,
      auto_observe_min_severity: autoAlertEnabled ? form.elements.auto_observe_min_severity.value : (editing?.auto_observe_min_severity || "CRITICAL"),
      alert_cooldown_minutes: autoAlertEnabled ? Number(form.elements.alert_cooldown_minutes.value) : (editing?.alert_cooldown_minutes ?? 15),
      models: { diagnosis_llm: modelId },
      instruction: form.elements.instruction.value.trim() || null,
      image_capabilities: {},
      config: {},
    };
  }

  function normalized(value) {
    if (!value || typeof value !== "object") return value;
    if (Array.isArray(value)) return value.map(normalized);
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, normalized(value[key])]));
  }

  function sameJson(left, right) {
    return JSON.stringify(normalized(left || {})) === JSON.stringify(normalized(right || {}));
  }

  function collectBindingPlans(targetId, selectedSourceIds) {
    if (!targetId) return [];
    return selectedSourceIds.map((sourceId) => {
      const card = [...document.querySelectorAll(".agent-source-card")].find((item) => item.dataset.sourceId === sourceId);
      const source = sources.find((item) => item.source_id === sourceId);
      const locatorKey = card.querySelector("[data-locator-key]").value.trim();
      if (!locatorKey) throw new Error(`${source.display_name}：请填写 Target 在监控系统中的标识。`);
      let sourceLocator = {};
      if (source.source_type === "LOKI") {
        const job = card.querySelector("[data-loki-job]").value.trim();
        const targetLabel = card.querySelector("[data-loki-target-label]").value.trim();
        const targetValue = card.querySelector("[data-loki-target-value]").value.trim();
        if (!job || !targetLabel || !targetValue) throw new Error(`${source.display_name}：请完整填写 Loki 日志标签。`);
        if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(targetLabel) || targetLabel === "job") throw new Error(`${source.display_name}：目标标签名称格式无效，且不能与 job 重复。`);
        sourceLocator = { labels: { job, [targetLabel]: targetValue } };
      }
      if (source.source_type === "PROMETHEUS") {
        const hostTargetKey = card.querySelector("[data-prometheus-host-target]").value.trim();
        if (!hostTargetKey) throw new Error(`${source.display_name}：请填写数据库主机的 Node Exporter target_key。`);
        sourceLocator = { host_target_key: hostTargetKey };
      }
      return { source, locatorKey, sourceLocator, existing: bindingFor(sourceId) };
    });
  }

  async function ensureSourceBindings(targetId, plans) {
    if (!targetId || !plans.length) return;
    if (bindingTargetId !== targetId) await loadBindings(targetId);
    for (let index = 0; index < plans.length; index += 1) {
      const plan = plans[index];
      plan.existing = bindingFor(plan.source.source_id);
      showResult(`正在配置监控源映射（${index + 1}/${plans.length}）：${plan.source.display_name}…`);
      let current = plan.existing;
      if (!current) {
        current = await KBotAIOpsAuth.request(`${api}/targets/${encodeURIComponent(targetId)}/source-bindings`, {
          method: "POST",
          headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify({
            source_id: plan.source.source_id,
            source_locator_key: plan.locatorKey,
            source_locator: plan.sourceLocator,
            role: "PRIMARY",
            priority: 100,
          }),
        });
      } else if (current.source_locator_key !== plan.locatorKey || !sameJson(current.source_locator, plan.sourceLocator)) {
        current = await KBotAIOpsAuth.request(`${api}/targets/${encodeURIComponent(targetId)}/source-bindings/${encodeURIComponent(current.binding_id)}`, {
          method: "PATCH",
          headers: { "If-Match": `"rv-${current.row_version}"` },
          body: JSON.stringify({ source_locator_key: plan.locatorKey, source_locator: plan.sourceLocator }),
        });
      }
      if (current.status !== "ACTIVE") {
        current = await KBotAIOpsAuth.request(`${api}/targets/${encodeURIComponent(targetId)}/source-bindings/${encodeURIComponent(current.binding_id)}/enable`, {
          method: "POST",
          headers: { "If-Match": `"rv-${current.row_version}"`, "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify({}),
        });
      }
      sourceBindings = sourceBindings.filter((item) => item.binding_id !== current.binding_id);
      sourceBindings.push(current);
    }
    applyBindings();
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
      const plans = collectBindingPlans(body.target_id, body.diagnostic_source_ids);
      await ensureSourceBindings(body.target_id, plans);
      if (editing) body.expected_row_version = editing.row_version;
      await KBotAIOpsAuth.request(editing ? `${api}/agents/${encodeURIComponent(editing.agent_id)}` : `${api}/agents`, {
        method: editing ? "PATCH" : "POST",
        body: JSON.stringify(body),
      });
      document.getElementById("agent-dialog").close();
      shell.toast(editing ? "Agent 已更新，监控源映射已同步" : "Agent 已创建");
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
    const [agentRows, sourcePage, targetPage, modelRows] = await Promise.all([
      KBotAIOpsAuth.request(`${api}/agents`),
      KBotAIOpsAuth.request(`${api}/diagnostic-sources?status=ENABLED&limit=200`),
      KBotAIOpsAuth.request(`${api}/targets?status=ENABLED&limit=200`),
      KBotAIOpsAuth.request(`${api}/model-catalog`),
    ]);
    agents = Array.isArray(agentRows) ? agentRows : [];
    sources = (sourcePage.items || []).filter((item) => ["CONNECTED", "DEGRADED"].includes(item.connectivity_status));
    targets = (targetPage.items || []).filter((item) => ["CONNECTED", "DEGRADED"].includes(item.connectivity_status));
    models = Array.isArray(modelRows) ? modelRows : [];
    renderResources();
    renderRows();
  }

  shell.ready.then(async () => {
    const dialog = document.getElementById("agent-dialog");
    dialog.querySelectorAll("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => dialog.close()));
    document.getElementById("create-agent").addEventListener("click", openCreate);
    document.getElementById("agent-target").addEventListener("change", async (event) => {
      toggleTargetFields();
      try {
        await loadBindings(event.currentTarget.value);
      } catch (error) {
        showResult(`读取 Target 映射失败：${error.message}`, "bad");
      }
    });
    document.getElementById("agent-sources").addEventListener("change", (event) => {
      if (event.target.matches('[name="diagnostic_source_ids"]')) syncSourceMappingVisibility();
    });
    document.getElementById("agent-sources").addEventListener("input", (event) => {
      if (!event.target.matches("[data-locator-key]")) return;
      const card = event.target.closest(".agent-source-card");
      const lokiValue = card.querySelector("[data-loki-target-value]");
      if (lokiValue && !lokiValue.dataset.userEdited) lokiValue.value = event.target.value;
    });
    document.getElementById("agent-sources").addEventListener("change", (event) => {
      if (event.target.matches("[data-loki-target-value]")) event.target.dataset.userEdited = "true";
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
