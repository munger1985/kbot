(function () {
  "use strict";
  if (document.body.dataset.page !== "recovery-drills") return;
  const appApi = "/api/v1/apps/aiops", shell = globalThis.KBotAIOpsShell, esc = shell.escape;
  const request = (path, init) => KBotAIOpsAuth.request(appApi + path, init);
  const query = new URLSearchParams(location.search);
  const sourceWorkItemId = query.get("workItemId") || query.get("work_item_id") || "";
  const sourceOptions = {
    ORACLE: [["ORACLE_RMAN", "Oracle RMAN"], ["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]],
    POSTGRESQL: [["POSTGRESQL_BASEBACKUP", "pg_basebackup"], ["POSTGRESQL_PGBACKREST", "pgBackRest"], ["POSTGRESQL_BARMAN", "Barman"], ["POSTGRESQL_WALG", "WAL-G"], ["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]],
    MYSQL: [["MYSQL_XTRABACKUP", "Percona XtraBackup"], ["MYSQL_ENTERPRISE_BACKUP", "MySQL Enterprise Backup"], ["MYSQL_LOGICAL_DUMP", "逻辑导出"], ["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]],
  };
  const form = document.getElementById("recovery-drill-form");
  const select = document.getElementById("recovery-target-select");
  const confirmDialog = document.getElementById("recovery-confirm-dialog");
  const rows = value => Array.isArray(value) ? value : [];
  const seconds = value => value == null ? "—" : `${value} 秒`;
  const assuranceLabels = {BACKUP_METADATA:"备份记录核对",RESTORE_VALIDATE:"介质恢复校验",DATABASE_OPEN:"数据库启动验证",APPLICATION_VALIDATED:"业务可用性验证"};
  const resultLabels = {PASS:"通过",FAIL:"失败",INCONCLUSIVE:"无法确认"};
  const statusLabels = {SUBMITTED:"待审核",VERIFIED:"已审核",REJECTED:"已驳回"};
  const state = {resources:[],resourceId:"",profile:null,drills:[],canManage:false,loading:false,submitting:false,reviewingId:"",pendingReview:null,sequence:0};
  const resource = () => state.resources.find(item => item.resource_id === state.resourceId);
  function controls() {
    select.disabled = Boolean(sourceWorkItemId) || state.loading || state.submitting || Boolean(state.reviewingId);
    const submit = document.getElementById("submit-recovery-drill");submit.disabled = state.submitting || !state.resourceId;submit.textContent = state.submitting ? "正在提交…" : "提交待审核记录";
    document.getElementById("refresh-recovery").disabled = state.loading || state.submitting || Boolean(state.reviewingId);
    document.querySelectorAll("[data-review-drill]").forEach(button => {button.disabled = Boolean(state.reviewingId);});
    form.elements.evidence_reference.required = form.elements.result.value === "PASS";form.elements.evidence_hash.required = form.elements.result.value === "PASS";
  }
  function updateDate(input) {
    input.previousElementSibling.textContent = input.value ? shell.fmt(input.value) : "请选择日期和时间";
    input.parentElement.classList.toggle("is-empty",!input.value);
  }
  function resetDraft() {
    form.reset();form.elements.assurance_level.value = state.profile?.required_assurance_level || "DATABASE_OPEN";
    document.getElementById("recovery-backup-source").innerHTML = (sourceOptions[resource()?.db_type] || []).map(([value,label]) => `<option value="${value}">${label}</option>`).join("");
    form.elements.backup_source_type.value = state.profile?.required_backup_source_types?.[0] || sourceOptions[resource()?.db_type]?.[0]?.[0] || "";
    form.querySelectorAll('.date-time-picker-native').forEach(updateDate);controls();
  }
  function renderProfile() {
    const profile = state.profile, badge = document.getElementById("recovery-workspace-state");
    badge.className = `status-pill ${profile?.drill_required ? "status-success" : "status-muted"}`;
    badge.textContent = profile ? profile.drill_required ? `策略 v${profile.version_no}` : "无需定期演练" : "未配置策略";
    const facts = [["RPO",profile ? seconds(profile.rpo_seconds) : "未配置"],["RTO",profile ? seconds(profile.rto_seconds) : "未配置"],["演练要求",profile?.drill_required ? `每 ${profile.required_drill_interval_days} 天` : "不要求定期演练"],["最低等级",profile?.drill_required ? assuranceLabels[profile.required_assurance_level] || profile.required_assurance_level : "不适用"]];
    document.getElementById("recovery-profile-summary").innerHTML = facts.map(([label,value]) => `<div><dt>${label}</dt><dd>${esc(value)}</dd></div>`).join("");
  }
  function renderHistory() {
    document.getElementById("recovery-history-count").textContent = `${state.drills.length} 次`;
    document.getElementById("recovery-drill-history").innerHTML = state.drills.length ? `<div class="management-catalog-grid">${state.drills.map(item => `<article class="management-catalog-card"><header><div><p class="eyebrow">RECOVERY DRILL</p><h3>${esc(shell.fmt(item.simulated_failure_at))}</h3></div><span class="status-pill ${item.status === "VERIFIED" ? "status-success" : item.status === "REJECTED" ? "status-muted" : "status-warning"}">${esc(statusLabels[item.status] || item.status)}</span></header><p>${esc(assuranceLabels[item.assurance_level] || item.assurance_level)} · ${esc(item.backup_source_type)} · ${esc(item.source_trust_level)}</p><dl><div><dt>演练结果</dt><dd>${esc(resultLabels[item.result] || item.result)}</dd></div><div><dt>实测 RPO / RTO</dt><dd>${esc(seconds(item.achieved_rpo_seconds))} / ${esc(seconds(item.achieved_rto_seconds))}</dd></div></dl>${state.canManage && item.status === "SUBMITTED" ? `<footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions"><button class="row-action" type="button" data-review-drill="${esc(item.drill_id)}" data-decision="VERIFY">通过</button><button class="danger-row-action" type="button" data-review-drill="${esc(item.drill_id)}" data-decision="REJECT">驳回</button></div></footer>` : '<footer class="management-catalog-actions"><span class="management-catalog-note">审核已完成</span></footer>'}</article>`).join("")}</div>` : '<p class="resource-list-empty">当前数据库资源尚无恢复演练记录。</p>';
    controls();
  }
  async function loadWorkspace() {
    const id = state.resourceId, sequence = ++state.sequence;
    if (!id) {state.profile = null;state.drills = [];renderProfile();renderHistory();resetDraft();return;}
    state.loading = true;controls();
    try {
      const [profile,page] = await Promise.all([request(`/managed-resources/${encodeURIComponent(id)}/protection-profile`),request(`/managed-resources/${encodeURIComponent(id)}/recovery-drills`)]);
      if (sequence !== state.sequence) return;state.profile = profile;state.drills = rows(page.items);renderProfile();renderHistory();resetDraft();
    } catch (error) {if (sequence === state.sequence) shell.toast(error.message || "无法读取恢复演练工作区");}
    finally {if (sequence === state.sequence) {state.loading = false;controls();}}
  }
  async function submitDrill(event) {
    event.preventDefault();if (state.submitting || !resource()) return;
    const fields = Object.fromEntries(new FormData(form)), iso = name => fields[name] ? new Date(fields[name]).toISOString() : null;
    const reference = fields.evidence_reference.trim(), hash = fields.evidence_hash.trim();
    if (fields.result === "PASS" && (!reference || !/^[a-f0-9]{64}$/.test(hash))) return shell.toast("通过的演练必须填写证据引用和 64 位小写 SHA-256。");
    if ((reference || hash) && (!reference || !/^[a-f0-9]{64}$/.test(hash))) return shell.toast("证据引用和 64 位小写 SHA-256 必须同时填写。");
    if (fields.recovered_through_at && new Date(fields.recovered_through_at) > new Date(fields.simulated_failure_at)) return shell.toast("实际恢复到的数据时间不能晚于模拟故障时间。");
    if (fields.service_validated_at && new Date(fields.service_validated_at) < new Date(fields.simulated_failure_at)) return shell.toast("验证完成时间不能早于模拟故障时间。");
    if (fields.result === "PASS" && ["DATABASE_OPEN","APPLICATION_VALIDATED"].includes(fields.assurance_level) && !fields.service_validated_at) return shell.toast("数据库启动或业务可用性验证通过时必须填写验证完成时间。");
    state.submitting = true;controls();
    try {
      await request(`/managed-resources/${encodeURIComponent(state.resourceId)}/recovery-drills`,{method:"POST",headers:{"Idempotency-Key":KBotAIOpsAuth.uuid()},body:JSON.stringify({source_work_item_id:sourceWorkItemId || null,scenario:fields.scenario,assurance_level:fields.assurance_level,backup_source_type:fields.backup_source_type,environment:fields.environment,result:fields.result,simulated_failure_at:iso("simulated_failure_at"),recovered_through_at:iso("recovered_through_at"),service_validated_at:iso("service_validated_at"),recovery_marker:{kind:resource().db_type},evidence:reference ? [{evidence_kind:fields.evidence_kind,reference,content_hash:hash}] : [],notes:fields.notes.trim() || null})});
      shell.toast(sourceWorkItemId ? "演练结果已提交，工作项进入待审核" : "计划外演练记录已提交审核");await loadWorkspace();
    } catch (error) {shell.toast(error.message || "无法提交恢复演练记录");}
    finally {state.submitting = false;controls();}
  }
  function openReview(button) {
    if (state.reviewingId || state.pendingReview) return;
    const item = state.drills.find(item => item.drill_id === button.dataset.reviewDrill);if (!item) return;
    const decision = button.dataset.decision;state.pendingReview = {item,decision};
    document.getElementById("recovery-confirm-title").textContent = decision === "VERIFY" ? "确认审核通过？" : "确认驳回？";
    document.getElementById("recovery-confirm-message").textContent = decision === "VERIFY" ? "满足当前恢复策略的记录会关闭本周期工单并生成下一周期工单。" : "来源工单会退回处理中，由原处理人补充或重新演练。";
    confirmDialog.querySelector("[data-confirm-review]").textContent = decision === "VERIFY" ? "审核通过" : "驳回";
    confirmDialog.showModal();confirmDialog.querySelector("[data-cancel-review]").focus();
  }
  async function reviewDrill() {
    const pending = state.pendingReview;if (!pending || state.reviewingId) return;
    const {item,decision} = pending;state.pendingReview = null;state.reviewingId = item.drill_id;confirmDialog.close();controls();
    try {
      await request(`/managed-resources/${encodeURIComponent(state.resourceId)}/recovery-drills/${encodeURIComponent(item.drill_id)}:review`,{method:"POST",headers:{"If-Match":`"rv-${item.row_version}"`,"Idempotency-Key":KBotAIOpsAuth.uuid()},body:JSON.stringify({decision,review_note:null})});
      shell.toast(decision === "VERIFY" ? "演练记录已审核" : "演练记录已驳回");await loadWorkspace();
    } catch (error) {shell.toast(error.message || "无法审核恢复演练");}
    finally {state.reviewingId = "";controls();}
  }
  async function initialize() {
    renderProfile();renderHistory();
    const page = await request("/business-systems?limit=200");
    const systems = await Promise.all(rows(page.items).filter(item => item.status === "ENABLED").map(item => request(`/business-systems/${encodeURIComponent(item.business_system_id)}`)));
    state.resources = systems.flatMap(system => rows(system.resources).filter(item => item.status === "ACTIVE" && item.resource?.resource_type_key === "DATABASE" && item.resource.database).map(item => ({resource_id:item.resource.resource_id,display_name:item.resource.display_name,system_name:system.display_name,db_type:item.resource.database.db_type,version_code:item.resource.database.version_code})));
    select.innerHTML = state.resources.map(item => `<option value="${esc(item.resource_id)}">${esc(item.system_name)} / ${esc(item.display_name)} · ${esc(item.db_type)} ${esc(item.version_code)}</option>`).join("");
    const requested = query.get("resourceId") || query.get("resource_id");state.resourceId = state.resources.find(item => item.resource_id === requested)?.resource_id || state.resources[0]?.resource_id || "";select.value = state.resourceId;
    await loadWorkspace();
  }
  shell.ready.then(access => {
    state.canManage = rows(access.permissions).includes("aiops:work_item_manage");
    const banner = document.getElementById("recovery-work-item-context");banner.hidden = !sourceWorkItemId;banner.querySelector("code").textContent = sourceWorkItemId;
    document.getElementById("recovery-entry-title").textContent = sourceWorkItemId ? "填写演练结果" : "补录计划外演练";
    form.onsubmit = submitDrill;form.oninput = event => {if (event.target.classList.contains("date-time-picker-native")) updateDate(event.target);};
    form.onchange = event => {if (event.target.classList.contains("date-time-picker-native")) {updateDate(event.target);event.target.blur();}controls();};
    select.onchange = () => {if (state.submitting || state.reviewingId) return;state.resourceId = select.value;void loadWorkspace();};
    document.getElementById("refresh-recovery").onclick = loadWorkspace;
    document.getElementById("recovery-drill-history").onclick = event => {const button = event.target.closest("[data-review-drill]");if (button) openReview(button);};
    confirmDialog.onclick = event => {if (event.target.closest("[data-cancel-review]")) {state.pendingReview = null;confirmDialog.close();}else if (event.target.closest("[data-confirm-review]")) void reviewDrill();};
    confirmDialog.addEventListener("close",() => {state.pendingReview = null;});
    return initialize();
  }).catch(error => shell.toast(error.message));
})();
