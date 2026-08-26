(function () {
  "use strict";
  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let agents = [], sources = [], targets = [], editing = null;
  const escape = (value) => shell.escape(value ?? "—");
  const sourceName = (id) => sources.find((item) => item.source_id === id)?.display_name || shell.short(id);
  const targetName = (id) => targets.find((item) => item.target_id === id)?.display_name || shell.short(id);

  function showResult(message = "") { document.getElementById("agent-result").textContent = message; }
  function renderRows() {
    const body = document.getElementById("agent-rows");
    if (!agents.length) { body.innerHTML = '<tr><td class="ops-empty" colspan="7">当前范围内暂无 Agent</td></tr>'; return; }
    body.innerHTML = agents.map((agent) => `<tr><td><strong>${escape(agent.display_name)}</strong><br><small>${escape(agent.description || "")}</small></td><td>${shell.badge(agent.status)}</td><td>${(agent.diagnostic_source_ids || []).map((id) => escape(sourceName(id))).join("、") || "—"}</td><td>${agent.target_id ? escape(targetName(agent.target_id)) : "未开放"}</td><td>${agent.allow_change_execution ? "人工审批后可执行" : "仅诊断"}</td><td>${agent.auto_alert_enabled ? `${escape(agent.auto_observe_min_severity)} / ${escape(agent.alert_cooldown_minutes)} 分钟` : "关闭"}</td><td><button type="button" data-agent-id="${escape(agent.agent_id)}">编辑</button></td></tr>`).join("");
    body.querySelectorAll("[data-agent-id]").forEach((button) => button.addEventListener("click", () => openEdit(button.dataset.agentId)));
  }
  function renderResources() {
    document.getElementById("agent-sources").innerHTML = sources.length ? sources.map((source) => `<label><input type="checkbox" name="diagnostic_source_ids" value="${escape(source.source_id)}">${escape(source.display_name)} <small>${escape(source.source_type)} · ${escape(source.connectivity_status)}</small></label>`).join("") : '<span class="ops-error">没有已启用且可连接的监控源，请先完成监控源配置。</span>';
    document.getElementById("agent-target").innerHTML = '<option value="">不允许数据库直连诊断</option>' + targets.map((target) => `<option value="${escape(target.target_id)}">${escape(target.display_name)} · ${escape(target.db_type)}</option>`).join("");
  }
  function toggleTargetFields() {
    const selected = Boolean(document.getElementById("agent-target").value);
    document.getElementById("agent-change-field").hidden = !selected;
    if (!selected) document.querySelector('[name="allow_change_execution"]').checked = false;
  }
  function openCreate() {
    editing = null;
    const form = document.getElementById("agent-form"); form.reset();
    form.elements.status.value = "DRAFT"; form.elements.alert_cooldown_minutes.value = 15; form.elements.auto_alert_enabled.checked = true;
    toggleTargetFields(); showResult();
    document.getElementById("agent-dialog-title").textContent = "新增 Agent";
    document.getElementById("save-agent").textContent = "创建 Agent";
    document.getElementById("agent-dialog").showModal();
  }
  function openEdit(agentId) {
    editing = agents.find((item) => item.agent_id === agentId); if (!editing) return;
    const form = document.getElementById("agent-form"); form.reset();
    form.elements.display_name.value = editing.display_name; form.elements.description.value = editing.description || ""; form.elements.status.value = editing.status;
    form.elements.target_id.value = editing.target_id || ""; form.elements.allow_change_execution.checked = Boolean(editing.allow_change_execution);
    form.elements.auto_alert_enabled.checked = Boolean(editing.auto_alert_enabled); form.elements.auto_observe_min_severity.value = editing.auto_observe_min_severity || "CRITICAL";
    form.elements.alert_cooldown_minutes.value = editing.alert_cooldown_minutes ?? 15; form.elements.diagnosis_model_id.value = editing.models?.diagnosis || ""; form.elements.instruction.value = editing.instruction || "";
    form.querySelectorAll('[name="diagnostic_source_ids"]').forEach((input) => { input.checked = (editing.diagnostic_source_ids || []).includes(input.value); });
    toggleTargetFields(); showResult();
    document.getElementById("agent-dialog-title").textContent = "修改 Agent"; document.getElementById("save-agent").textContent = "保存修改"; document.getElementById("agent-dialog").showModal();
  }
  function payload(form) {
    const selectedSources = [...form.querySelectorAll('[name="diagnostic_source_ids"]:checked')].map((input) => input.value);
    if (!selectedSources.length) throw new Error("至少选择一个监控源。");
    const modelId = form.elements.diagnosis_model_id.value.trim();
    return { display_name: form.elements.display_name.value.trim(), description: form.elements.description.value.trim() || null, status: form.elements.status.value, diagnostic_source_ids: selectedSources, target_id: form.elements.target_id.value || null, allow_change_execution: form.elements.allow_change_execution.checked, auto_alert_enabled: form.elements.auto_alert_enabled.checked, auto_observe_min_severity: form.elements.auto_observe_min_severity.value, alert_cooldown_minutes: Number(form.elements.alert_cooldown_minutes.value), models: modelId ? { diagnosis: modelId } : {}, instruction: form.elements.instruction.value.trim() || null, image_capabilities: {}, config: {} };
  }
  async function save(event) {
    event.preventDefault(); const button = document.getElementById("save-agent"); button.disabled = true;
    try {
      const body = payload(event.currentTarget); if (editing) body.expected_row_version = editing.row_version;
      await KBotAIOpsAuth.request(editing ? `${api}/agents/${encodeURIComponent(editing.agent_id)}` : `${api}/agents`, { method: editing ? "PATCH" : "POST", body: JSON.stringify(body) });
      document.getElementById("agent-dialog").close(); shell.toast(editing ? "Agent 已更新并生成新版本" : "Agent 已创建"); await load();
    } catch (error) { showResult(error.message); } finally { button.disabled = false; }
  }
  async function load() {
    const [agentRows, sourcePage, targetPage] = await Promise.all([KBotAIOpsAuth.request(`${api}/agents`), KBotAIOpsAuth.request(`${api}/diagnostic-sources?status=ENABLED&limit=200`), KBotAIOpsAuth.request(`${api}/targets?status=ENABLED&limit=200`)]);
    agents = Array.isArray(agentRows) ? agentRows : []; sources = (sourcePage.items || []).filter((item) => ["CONNECTED", "DEGRADED"].includes(item.connectivity_status)); targets = (targetPage.items || []).filter((item) => ["CONNECTED", "DEGRADED"].includes(item.connectivity_status)); renderResources(); renderRows();
  }
  shell.ready.then(async () => {
    const dialog = document.getElementById("agent-dialog"); dialog.querySelectorAll("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => dialog.close()));
    document.getElementById("create-agent").addEventListener("click", openCreate); document.getElementById("agent-target").addEventListener("change", toggleTargetFields); document.getElementById("agent-form").addEventListener("submit", save);
    try { await load(); } catch (error) { document.getElementById("agent-rows").innerHTML = `<tr><td class="ops-empty" colspan="7">${escape(error.message)}</td></tr>`; }
  });
})();
