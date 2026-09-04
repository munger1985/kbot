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
  const bindingsByTarget = new Map();
  const draftsByTarget = new Map();
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
      const targetNames = (agent.target_ids || []).map((id) => escape(targetName(id)));
      const access = "诊断；受控动作逐次审批";
      return `<tr>
        <td><strong>${escape(agent.display_name)}</strong><small class="agent-row-description">${escape(agent.description || "未填写说明")}</small></td>
        <td>${shell.badge(agent.status)}</td>
        <td><strong>${sourceNames.length} 个监控源</strong><small class="agent-row-description">${sourceNames.join("、") || "—"}</small></td>
        <td><strong>${access}</strong><small class="agent-row-description">${targetNames.join("、") || "—"}</small></td>
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
    const prometheusHint = source.source_type === "PROMETHEUS"
      ? '<small class="agent-source-requirement">需要配置 Oracle Exporter 与 Node Exporter 两个 target_key</small>'
      : "";
    const prometheusFields = source.source_type === "PROMETHEUS" ? `<div class="agent-prometheus-fields">
      <p><strong>主机指标映射</strong>CPU、内存、磁盘、文件系统和网络指标使用 Node Exporter 标签，不能沿用 Oracle Exporter 标签。</p>
      <div class="ops-field">
        <label>数据库主机的 Node Exporter target_key</label>
        <input data-prometheus-host-target maxlength="256" placeholder="例如 dev-db-host-190">
        <small>可在 Prometheus 查询 <code>count by (target_key) (node_uname_info{job=&quot;node&quot;})</code> 确认，然后填写该数据库所在主机对应的值。</small>
      </div>
    </div>` : "";
    return `<article class="agent-source-card" data-source-id="${sourceId}" data-source-type="${escape(source.source_type)}">
      <label class="agent-source-choice">
        <input type="checkbox" name="diagnostic_source_ids" value="${sourceId}">
        <span class="agent-source-identity"><strong>${escape(source.display_name)}</strong><small>${escape(source.source_type)}</small>${prometheusHint}</span>
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
      : '<div class="ops-error">没有已启用的监控源，请先完成监控源配置。</div>';
    document.getElementById("agent-targets").innerHTML = targets.length
      ? targets.map((target) => `<label class="agent-switch-row"><input type="checkbox" name="target_ids" value="${escape(target.target_id)}"><span><strong>${escape(target.display_name)}</strong><small>${escape(target.db_type)} · ${target.readonly_connection_enabled ? "只读直连" : "仅监控"}${target.controlled_change_enabled ? " · 允许受控变更" : ""}</small></span></label>`).join("")
      : '<div class="ops-error">没有已启用的逻辑 Target，请先创建并启用运维目标。</div>';
    const diagnosisModels = models.filter((model) => Number(model.category) === 1);
    document.getElementById("agent-planner-model").innerHTML = diagnosisModels.length
      ? '<option value="">请选择规划模型</option>' + diagnosisModels.map((model) => `<option value="${escape(model.model_id)}">${escape(model.display_name)} · ${escape(model.served_model_name)}</option>`).join("")
      : '<option value="">没有已启用的 LLM，请先配置模型服务</option>';
    document.getElementById("agent-model").innerHTML = diagnosisModels.length
      ? '<option value="">请选择诊断模型</option>' + diagnosisModels.map((model) => `<option value="${escape(model.model_id)}">${escape(model.display_name)} · ${escape(model.served_model_name)}</option>`).join("")
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

  function bindingFor(sourceId) {
    const matches = sourceBindings.filter((item) => item.source_id === sourceId);
    return matches.find((item) => item.status === "ACTIVE") || matches[0] || null;
  }

  function sourceCards() {
    return [...document.querySelectorAll("#agent-sources .agent-source-card[data-source-id]")];
  }

  function sourceCardFor(sourceId) {
    return sourceCards().find((card) => card.dataset.sourceId === sourceId) || null;
  }

  function resetMappingInputs() {
    sourceCards().forEach((card) => {
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
    sourceCards().forEach((card) => {
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
    captureMappingDraft();
    resetMappingInputs();
    sourceBindings = [];
    bindingTargetId = "";
    if (!targetId) {
      syncSourceMappingVisibility();
      return;
    }
    const expectedTarget = targetId;
    document.getElementById("agent-binding-summary").textContent = "正在读取该 Target 的监控源映射…";
    const bindings = bindingsByTarget.has(targetId)
      ? bindingsByTarget.get(targetId)
      : await KBotAIOpsAuth.request(`${api}/targets/${encodeURIComponent(targetId)}/source-bindings`);
    if (document.getElementById("agent-mapping-target").value !== expectedTarget) return;
    sourceBindings = Array.isArray(bindings) ? bindings : [];
    bindingsByTarget.set(targetId, sourceBindings);
    bindingTargetId = targetId;
    applyBindings();
    applyMappingDraft(targetId);
    syncSourceMappingVisibility();
  }

  function syncSourceMappingVisibility() {
    const targetId = document.getElementById("agent-mapping-target").value;
    let selectedCount = 0;
    let mappedCount = 0;
    sourceCards().forEach((card) => {
      const choice = card.querySelector('[name="diagnostic_source_ids"]');
      const mapping = card.querySelector(".agent-source-mapping");
      if (!choice || !mapping) return;
      const checked = choice.checked;
      const show = Boolean(targetId && checked);
      card.classList.toggle("selected", checked);
      mapping.hidden = !show;
      if (checked) selectedCount += 1;
      if (show && bindingFor(card.dataset.sourceId)?.status === "ACTIVE") mappedCount += 1;
    });
    const summary = document.getElementById("agent-binding-summary");
    if (!targetId) summary.textContent = "请先选择至少一个逻辑 Target。";
    else if (!selectedCount) summary.textContent = "请选择监控源，随后填写该 Target 在监控系统中的标识。";
    else summary.textContent = `${selectedCount} 个监控源已选择，${mappedCount} 个已有有效 Target 映射；保存时会补齐或更新。`;
  }

  function toggleTargetFields() {
    syncMappingTargetOptions();
    syncSourceMappingVisibility();
  }

  function selectedTargetIds() {
    return [...document.querySelectorAll('[name="target_ids"]:checked')].map((input) => input.value);
  }

  function captureMappingDraft() {
    if (!bindingTargetId) return;
    const draft = {};
    sourceCards().forEach((card) => {
      const labels = {};
      const job = card.querySelector("[data-loki-job]");
      const label = card.querySelector("[data-loki-target-label]");
      const value = card.querySelector("[data-loki-target-value]");
      if (job && label && value && job.value.trim() && label.value.trim()) {
        labels.job = job.value.trim();
        labels[label.value.trim()] = value.value.trim();
      }
      draft[card.dataset.sourceId] = {
        locatorKey: card.querySelector("[data-locator-key]").value.trim(),
        sourceLocator: labels.job ? { labels } : (card.querySelector("[data-prometheus-host-target]") ? { host_target_key: card.querySelector("[data-prometheus-host-target]").value.trim() } : {}),
      };
    });
    draftsByTarget.set(bindingTargetId, draft);
  }

  function applyMappingDraft(targetId) {
    const draft = draftsByTarget.get(targetId);
    if (!draft) return;
    sourceCards().forEach((card) => {
      const item = draft[card.dataset.sourceId];
      if (!item) return;
      card.querySelector("[data-locator-key]").value = item.locatorKey || "";
      const host = card.querySelector("[data-prometheus-host-target]");
      if (host) host.value = item.sourceLocator?.host_target_key || "";
      const labels = item.sourceLocator?.labels || {};
      const job = card.querySelector("[data-loki-job]");
      const label = card.querySelector("[data-loki-target-label]");
      const value = card.querySelector("[data-loki-target-value]");
      if (job && label && value) {
        const targetLabel = Object.keys(labels).find((name) => name !== "job") || "target_key";
        job.value = labels.job || "oracle_alert";
        label.value = targetLabel;
        value.value = labels[targetLabel] || "";
      }
    });
  }

  function syncMappingTargetOptions() {
    const select = document.getElementById("agent-mapping-target");
    const selected = selectedTargetIds();
    const previous = selected.includes(select.value) ? select.value : selected[0] || "";
    select.innerHTML = '<option value="">请选择</option>' + selected.map((id) => `<option value="${escape(id)}">${escape(targetName(id))}</option>`).join("");
    select.value = previous;
    if (previous && previous !== bindingTargetId) loadBindings(previous).catch((error) => showResult(error.message, "bad"));
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
    bindingsByTarget.clear();
    draftsByTarget.clear();
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
    sourceBindings = [];
    bindingTargetId = "";
    bindingsByTarget.clear();
    draftsByTarget.clear();
    const form = document.getElementById("agent-form");
    form.reset();
    resetMappingInputs();
    form.elements.status.disabled = false;
    form.elements.display_name.value = editing.display_name;
    form.elements.description.value = editing.description || "";
    form.elements.status.value = editing.status;
    form.elements.auto_alert_enabled.checked = Boolean(editing.auto_alert_enabled);
    form.elements.auto_observe_min_severity.value = editing.auto_observe_min_severity || "CRITICAL";
    form.elements.alert_cooldown_minutes.value = editing.alert_cooldown_minutes ?? 15;
    form.elements.planner_model_id.value = editing.models?.planner_llm || "";
    form.elements.diagnosis_model_id.value = editing.models?.diagnosis_llm || "";
    form.elements.ocr_model_id.value = editing.image_capabilities?.ocr?.default_model_id || "";
    form.elements.vlm_model_id.value = editing.image_capabilities?.vlm?.default_model_id || "";
    form.elements.instruction.value = editing.instruction || "";
    form.querySelectorAll('[name="diagnostic_source_ids"]').forEach((input) => {
      input.checked = (editing.diagnostic_source_ids || []).includes(input.value);
    });
    form.querySelectorAll('[name="target_ids"]').forEach((input) => {
      input.checked = (editing.target_ids || []).includes(input.value);
    });
    document.getElementById("agent-status-help").textContent = "启用前会检查逻辑 Target、监控源及至少一条有效映射；运行时连接异常不会阻止 Agent 诊断。";
    toggleTargetFields();
    toggleAlertSettings();
    showResult();
    document.getElementById("agent-dialog-title").textContent = "修改 Agent";
    document.getElementById("save-agent").textContent = "保存修改";
    document.getElementById("agent-dialog").showModal();
    const firstTargetId = editing.target_ids?.[0];
    if (firstTargetId) {
      try {
        document.getElementById("agent-mapping-target").value = firstTargetId;
        await loadBindings(firstTargetId);
      } catch (error) {
        showResult(`读取 Target 映射失败：${error.message}`, "bad");
      }
    }
  }

  function payload(form) {
    const selectedSources = [...form.querySelectorAll('[name="diagnostic_source_ids"]:checked')].map((input) => input.value);
    const targetIds = selectedTargetIds();
    if (!selectedSources.length) throw new Error("至少选择一个监控源。");
    if (!targetIds.length) throw new Error("至少选择一个逻辑 Target。");
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
    return {
      display_name: form.elements.display_name.value.trim(),
      description: form.elements.description.value.trim() || null,
      status: editing ? form.elements.status.value : "DRAFT",
      diagnostic_source_ids: selectedSources,
      target_ids: targetIds,
      auto_alert_enabled: autoAlertEnabled,
      auto_observe_min_severity: autoAlertEnabled ? form.elements.auto_observe_min_severity.value : (editing?.auto_observe_min_severity || "CRITICAL"),
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
      const card = sourceCardFor(sourceId);
      const source = sources.find((item) => item.source_id === sourceId);
      if (!card || !source) throw new Error("监控源配置已经变化，请刷新页面后重试。");
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
    bindingsByTarget.set(targetId, sourceBindings);
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
      captureMappingDraft();
      for (const targetId of body.target_ids) {
        document.getElementById("agent-mapping-target").value = targetId;
        await loadBindings(targetId);
        const plans = collectBindingPlans(targetId, body.diagnostic_source_ids);
        await ensureSourceBindings(targetId, plans);
      }
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
    sources = sourcePage.items || [];
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
    document.getElementById("agent-mapping-target").addEventListener("change", async (event) => {
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
