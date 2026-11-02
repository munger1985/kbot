(function () {
  "use strict";
  if (document.body.dataset.page !== "inspection-detail") return;
  const shell = globalThis.KBotAIOpsShell, esc = shell.escape;
  const request = (path,init) => KBotAIOpsAuth.request(`/api/v1/apps/aiops${path}`,init);
  const rows = value => Array.isArray(value) ? value : [];
  const params = new URLSearchParams(location.search), fireId = params.get("id") || "";
  const root = document.getElementById("inspection-detail-content");
  const state = {fire:null,targets:[],targetId:params.get("targetId") || "",runId:params.get("runId") || "",agents:[],conversation:null,canManage:false,selection:0,loadSequence:0,lookupLoading:false,lookupFailed:false,continuation:false,agentsLoading:false,submitting:false,agentId:"",message:""};
  function batchTargets() {
    const items = new Map();
    for (const resource of rows(state.fire?.system_scope?.resources)) {
      if (resource.target_id && !items.has(resource.target_id)) items.set(resource.target_id,{...resource,run:null});
    }
    for (const run of rows(state.fire?.runs)) {
      const target = items.get(run.target_id) || state.targets.find(item => item.target_id === run.target_id);
      items.set(run.target_id,{target_id:run.target_id,display_name:target?.display_name || run.target_id,db_type:target?.db_type,run});
    }
    return [...items.values()];
  }
  const selectedTarget = () => batchTargets().find(target => target.target_id === state.targetId);
  const selectedRun = () => selectedTarget()?.run;
  const statusLabel = value => ({CREATED:"已创建",QUEUED:"等待执行",RUNNING:"运行中",COMPLETED:"已完成",PARTIAL:"部分完成",FAILED:"失败",CANCELLED:"已取消",READY:"结论就绪",GENERATING:"生成中",EXPIRED:"已超时"})[value] || value;
  const statusClass = value => ["FAILED","CANCELLED","EXPIRED"].includes(value) ? "status-danger" : ["PARTIAL","CREATED","QUEUED","RUNNING","GENERATING"].includes(value) ? "status-warning" : ["COMPLETED","READY"].includes(value) ? "status-success" : "status-muted";
  const badge = (label,status) => `<span class="status-pill ${statusClass(status)}">${esc(label)}</span>`;
  const observation = items => `<dl class="inspection-observation">${items.map(([label,value]) => `<div><dt>${esc(label)}</dt><dd>${value}</dd></div>`).join("")}</dl>`;
  const canContinue = () => Boolean(selectedRun() && !state.lookupLoading && !state.lookupFailed && (state.conversation || !["CREATED","QUEUED","RUNNING"].includes(selectedRun().status)));
  function noRunText() {
    return state.fire.error_message || state.fire.error_code || (["CREATED","QUEUED","RUNNING"].includes(state.fire.status)
      ? "巡检尚未生成目标运行，正在等待调度或启动取证。"
      : "本批次已结束，但未生成目标取证运行，未形成巡检报告。请核对执行日志和来源计划后重新执行。");
  }
  function batchSummary() {
    if (state.fire.status === "COMPLETED") return "本批次执行已完成。数据库健康与证据边界请以各库正式报告为准。";
    if (["CREATED","QUEUED","RUNNING"].includes(state.fire.status)) return "批次正在调度或执行，刷新可查看最新进展。";
    return "本批次未全部完成，请在数据库清单中逐库核对执行结果与失败原因。";
  }
  function missingRunText() {
    return ["CREATED","QUEUED","RUNNING"].includes(state.fire.status)
      ? "该数据库尚未生成运行记录，正在等待调度或启动取证。"
      : "该数据库在本批次冻结范围内，但未生成运行记录。不能视为已完成，也没有可展示的报告。";
  }
  function resultHtml(target) {
    const run = target.run;
    const report = run?.report_id ? `<a class="primary-button" href="./report-detail.html?id=${encodeURIComponent(run.report_id)}&amp;inspectionFireId=${encodeURIComponent(fireId)}&amp;runId=${encodeURIComponent(run.run_id)}">查看报告</a>` : "";
    let content = `<p class="notice ${["CREATED","QUEUED","RUNNING"].includes(state.fire.status) ? "notice-warning" : "notice-error"}" role="status">${esc(missingRunText())}</p>`;
    if (run) {
      const facts = [["业务检查语义满足率",run.evidence_coverage_percent],["数据库工具执行覆盖率",run.tool_execution_coverage_percent],["监控采样覆盖率",run.monitoring_sample_coverage_percent]].map(([label,value]) => ({label,value:typeof value === "number" && Number.isFinite(value) ? Math.min(100,Math.max(0,value)) : null}));
      const quality = facts.map(fact => `<div class="inspection-quality-item"><span>${esc(fact.label)}</span><strong>${fact.value === null ? "未记录" : `${fact.value}%`}</strong><div class="inspection-coverage-track" aria-hidden="true">${fact.value === null ? "" : `<span style="width:${fact.value}%"></span>`}</div></div>`).join("");
      const failure = run.error_message || run.error_code ? `<div class="notice notice-error" role="alert"><strong>执行异常</strong><p>${esc(run.error_message || "目标运行失败，请核对执行日志。")}</p>${run.error_code ? `<code>${esc(run.error_code)}</code>` : ""}</div>` : "";
      content = `<div class="inspection-run-statuses">${badge(`执行${statusLabel(run.status)}`,run.status)}${badge(run.report_status ? `报告${statusLabel(run.report_status)}` : "暂无报告状态",run.report_status)}</div>${failure}${!run.report_id ? `<p class="inspection-empty">${["CREATED","QUEUED","RUNNING"].includes(run.status) ? "报告尚在生成。" : "本目标未生成正式报告，不能展示为空白报告。"}</p>` : ""}<section class="inspection-quality" aria-label="取证与报告质量"><h3>取证与报告质量</h3><p>执行状态不代表数据库健康；证据缺口不改变执行结果。</p><div class="inspection-quality-grid">${quality}</div>${observation([["报告证据缺口",typeof run.evidence_gap_count === "number" ? `${esc(run.evidence_gap_count)} 项` : "未记录"],["执行完成时间",esc(shell.fmt(run.completed_at))]])}</section><p class="inspection-report-note">查看报告与运维报告使用同一份正式报告，包含完整结论、检查结果与证据边界。</p>`;
    }
    return `<section id="inspection-selected-result" class="table-card inspection-selected-target" aria-label="当前数据库结果"><header class="inspection-result-heading"><div><p class="eyebrow">数据库执行结果</p><h2>${esc(target.display_name)}</h2><p>${esc(target.db_type || "数据库类型未记录")}</p></div>${report}</header>${content}</section>`;
  }
  function render() {
    const fire = state.fire;if (!fire) return;
    document.getElementById("inspection-refresh").disabled = state.submitting;
    const plan = document.getElementById("inspection-source-plan");plan.hidden = !state.canManage;plan.href = `./inspection-plans.html?planId=${encodeURIComponent(fire.plan_id)}`;
    const targets = batchTargets(), target = selectedTarget();
    const reportCount = new Set(rows(fire.runs).filter(run => run.report_id).map(run => run.report_id)).size;
    const cards = targets.map(item => `<button class="inspection-target-card${state.targetId === item.target_id ? " active" : ""}" type="button" aria-pressed="${state.targetId === item.target_id}" aria-controls="inspection-selected-result" data-target-id="${esc(item.target_id)}"${state.submitting ? " disabled" : ""}><span class="inspection-target-identity"><strong>${esc(item.display_name)}</strong><small>${esc(item.db_type || "数据库类型未记录")}</small></span>${badge(item.run ? statusLabel(item.run.status) : "未生成运行",item.run?.status)}<span class="inspection-target-report">${item.run?.report_id ? "已关联正式报告" : "暂无正式报告"}<span aria-hidden="true">→</span></span></button>`).join("");
    const scope = fire.system_scope;
    const focusedTargetId = document.activeElement?.closest("[data-target-id]")?.dataset.targetId;
    const recordOpen = root.querySelector(".inspection-batch-record")?.open;
    const emptyNotice = !rows(fire.runs).length ? `<p class="notice ${["CREATED","QUEUED","RUNNING"].includes(fire.status) ? "notice-warning" : "notice-error"}" role="status">${esc(noRunText())}</p>` : "";
    root.innerHTML = `<section class="table-card inspection-batch-summary" aria-label="巡检事实"><div class="inspection-batch-heading"><div><p class="eyebrow">日常巡检 / 批次结果</p><h1>${esc(fire.plan_display_name)}</h1><p class="inspection-system-name">${esc(fire.business_system_name || "本批次未记录业务系统名称")}</p></div>${badge(statusLabel(fire.status),fire.status)}</div><p class="inspection-summary-note">${esc(batchSummary())}</p><dl class="inspection-batch-metrics" aria-label="批次统计"><div><dt>数据库目标</dt><dd>${esc(fire.target_count)}<small>个</small></dd></div><div><dt>执行完成</dt><dd>${esc(fire.completed_count)}<small>个</small></dd></div><div${fire.failed_count > 0 ? ' class="has-failures"' : ""}><dt>执行失败</dt><dd>${esc(fire.failed_count)}<small>个</small></dd></div><div><dt>可查看报告</dt><dd>${reportCount}<small>份</small></dd></div></dl><div class="inspection-batch-times"><span>计划触发 <time>${esc(shell.fmt(fire.scheduled_at))}</time></span><span>完成时间 <time>${esc(shell.fmt(fire.completed_at))}</time></span></div>${fire.error_message || fire.error_code ? `<p class="notice notice-error" role="alert">${esc(fire.error_message || fire.error_code)}</p>` : ""}</section>
      <div class="inspection-results-layout"><section class="table-card inspection-targets" aria-label="目标结果"><header class="inspection-section-heading"><h2>数据库执行清单</h2><p>选择数据库，查看本次执行与报告。</p></header><div class="inspection-target-grid">${cards}</div>${!targets.length ? '<p class="inspection-empty">没有可展示的数据库清单。</p>' : ""}${fire.target_count > targets.length ? `<p class="inspection-list-note">另有 ${esc(fire.target_count - targets.length)} 个预期目标未返回身份信息，不能视为完成。</p>` : ""}</section><div class="inspection-result-workspace">${target ? resultHtml(target) : ""}${selectedRun() ? '<section class="table-card inspection-follow-up" aria-label="继续深入诊断"><header class="inspection-section-heading"><h2>对结果有疑问？</h2><p>继承当前数据库的巡检上下文继续核验，正式报告保持不变。</p></header><div id="inspection-continuation" class="inspection-continuation"></div></section>' : ""}${emptyNotice}</div></div>
      <details class="table-card inspection-batch-record"${recordOpen ? " open" : ""}><summary>批次记录与冻结范围</summary><div class="inspection-record-body">${observation([["批次编号",esc(fire.fire_id)],["来源计划编号",esc(fire.plan_id)]])}${scope ? `<h3>本批次冻结的数据库范围</h3><ul>${rows(scope.resources).map(resource => `<li>${esc(resource.display_name)} · ${esc(resource.db_type)}</li>`).join("")}</ul>${rows(scope.excluded_resources).length ? `<h3>未纳入本次执行的资源</h3><ul>${rows(scope.excluded_resources).map(resource => `<li>${esc(resource.display_name)} · ${esc(resource.reason)}</li>`).join("")}</ul>` : ""}` : '<p class="inspection-empty">本批次未记录冻结范围，不以当前系统资源补造历史清单。</p>'}</div></details>`;
    renderContinuation();
    if (focusedTargetId) [...root.querySelectorAll("[data-target-id]")].find(button => button.dataset.targetId === focusedTargetId)?.focus({preventScroll:true});
  }
  function updateControls() {
    const form = root.querySelector("form");if (!form) return;
    form.elements.agent_id.disabled = state.agentsLoading || state.submitting;form.elements.message.disabled = state.submitting;
    form.querySelector("[data-return-results]").disabled = state.submitting;
    const submit = form.querySelector('[type="submit"]');submit.disabled = state.submitting || state.agentsLoading || !state.agentId || !state.message.trim();submit.textContent = state.submitting ? "正在创建会话…" : "进入诊断对话";
  }
  function renderContinuation() {
    const pane = document.getElementById("inspection-continuation");if (!pane) return;
    if (!state.continuation) {
      pane.innerHTML = `${state.lookupFailed ? '<p class="notice notice-error">关联会话读取失败，请刷新后重试。</p>' : ""}<button class="secondary-button" type="button" data-continue${canContinue() ? "" : " disabled"}>${state.lookupLoading ? "正在检查会话…" : state.conversation ? "进入深入诊断会话" : "继续深入诊断"}</button>`;return;
    }
    const agents = state.agents.filter(agent => agent.status === "ACTIVE");
    pane.innerHTML = `<form class="quick-create-form inspection-continuation-form"><label>诊断 Agent<select name="agent_id" required><option value="" disabled>${state.agentsLoading ? "正在读取可用 Agent…" : "请选择 Agent"}</option>${agents.map(agent => `<option value="${esc(agent.agent_id)}">${esc(agent.display_name)} · ${rows(agent.allowed_business_system_ids).length} 个业务系统</option>`).join("")}</select></label><label>对现有结果的疑问<textarea name="message" rows="6" maxlength="32000" required></textarea></label><footer class="button-row"><button class="secondary-button" type="button" data-return-results>返回结果</button><button class="primary-button" type="submit">进入诊断对话</button></footer></form>`;
    const form = pane.querySelector("form");form.elements.agent_id.value = state.agentId;form.elements.message.value = state.message;
    form.oninput = () => {state.agentId = form.elements.agent_id.value;state.message = form.elements.message.value;updateControls();};form.onchange = form.oninput;form.onsubmit = startContinuation;updateControls();
  }
  async function selectTarget(target) {
    if (state.submitting) return;
    const selection = ++state.selection;state.targetId = target.target_id;state.runId = target.run?.run_id || "";state.continuation = false;state.conversation = null;state.lookupLoading = Boolean(target.run);state.lookupFailed = false;state.agentsLoading = false;state.agentId = "";state.message = "";
    const url = new URL(location.href);url.searchParams.set("targetId",target.target_id);if (target.run) url.searchParams.set("runId",target.run.run_id);else url.searchParams.delete("runId");history.replaceState(null,"",url);render();
    if (!target.run) return;
    try {const conversations = await request(`/conversations?run_id=${encodeURIComponent(target.run.run_id)}&limit=1`);if (selection === state.selection) state.conversation = rows(conversations)[0] || null;}
    catch(error) {if (selection === state.selection) {state.lookupFailed = true;shell.toast(error.message || "无法检查关联的深入诊断会话");}}
    finally {if (selection === state.selection) {state.lookupLoading = false;renderContinuation();}}
  }
  async function openContinuation() {
    if (!canContinue()) return;
    if (state.conversation) {location.href = `./chat.html?conversation=${encodeURIComponent(state.conversation.conversation_id)}`;return;}
    state.continuation = true;state.message ||= "我对本次巡检结论或证据仍有疑问，请继续核验，并说明需要重点确认的状态差异。";
    const selection = state.selection;state.agentsLoading = !state.agents.length;renderContinuation();if (!state.agentsLoading) return;
    try {const agents = await request("/agents");if (selection === state.selection) state.agents = rows(agents);}
    catch(error) {shell.toast(error.message || "无法读取可用 Agent");}
    finally {if (selection === state.selection) {state.agentsLoading = false;if (state.continuation) renderContinuation();}}
  }
  async function startContinuation(event) {
    event.preventDefault();if (!canContinue() || state.submitting || !state.agentId || !state.message.trim()) return;
    state.submitting = true;document.getElementById("inspection-refresh").disabled = true;root.querySelectorAll("[data-target-id]").forEach(button => {button.disabled = true;});updateControls();
    try {
      const receipt = await request("/conversations",{method:"POST",headers:{"Idempotency-Key":KBotAIOpsAuth.uuid()},body:JSON.stringify({agent_id:state.agentId,source_run_id:state.runId,content:[{content_type:"TEXT",text:state.message.trim()}]})});
      location.href = `./chat.html?conversation=${encodeURIComponent(receipt.conversation_id)}`;
    } catch(error) {shell.toast(error.message || "无法继续深入诊断");}
    finally {state.submitting = false;render();}
  }
  async function load() {
    if (state.submitting) return;
    const sequence = ++state.loadSequence;++state.selection;
    const loading = document.getElementById("inspection-loading"), errorNode = document.getElementById("inspection-load-error");
    loading.hidden = false;errorNode.hidden = true;root.innerHTML = "";state.fire = null;
    try {
      if (!fireId) throw new Error("URL 缺少巡检批次 id");
      const [fire,targets] = await Promise.all([request(`/inspection-fires/${encodeURIComponent(fireId)}`),request("/targets?limit=200")]);if (sequence !== state.loadSequence) return;
      state.fire = fire;state.targets = rows(targets.items);render();
      const targetsInBatch = batchTargets(), target = targetsInBatch.find(item => item.run?.run_id === state.runId || item.target_id === state.targetId) || targetsInBatch[0];if (target) await selectTarget(target);
    } catch(error) {if (sequence === state.loadSequence) {errorNode.hidden = false;errorNode.textContent = "无法读取巡检批次，请重试或返回日常巡检。";shell.toast(error.message);}}
    finally {if (sequence === state.loadSequence) loading.hidden = true;}
  }
  shell.ready.then(access => {
    state.canManage = rows(access.permissions).includes("aiops:plan_manage");
    const planId = params.get("planId");if (planId) document.getElementById("inspection-back").href = `./inspections.html?planId=${encodeURIComponent(planId)}`;
    document.getElementById("inspection-refresh").onclick = load;
    root.onclick = event => {
      if (state.submitting) return;
      const button = event.target.closest("[data-target-id]");
      if (button) {const target = batchTargets().find(item => item.target_id === button.dataset.targetId);if (target) void selectTarget(target);}
      else if (event.target.closest("[data-continue]")) void openContinuation();
      else if (event.target.closest("[data-return-results]")) {state.continuation = false;renderContinuation();}
    };
    return load();
  }).catch(error => shell.toast(error.message));
})();
