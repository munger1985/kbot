(function () {
  "use strict";
  if (document.body.dataset.page !== "inspections") return;
  const shell = globalThis.KBotAIOpsShell, esc = shell.escape;
  const request = (path, init) => KBotAIOpsAuth.request(`/api/v1/apps/aiops${path}`, init);
  const rows = value => Array.isArray(value) ? value : [];
  const text = (id, value) => { document.getElementById(id).textContent = value ?? ""; };
  const badge = (label, tone) => `<span class="status-pill ${tone}">${esc(label)}</span>`;
  const statusLabel = value => ({ ACTIVE:"进行中", QUEUED:"等待执行", COMPLETED:"已完成", FAILED:"失败", CANCELLED:"已取消", SKIPPED:"已跳过", PENDING:"等待处理", READY:"结论就绪", PARTIAL:"部分完成", GENERATING:"生成中", RUNNING:"运行中" })[value] || value;
  const statusClass = value => ["FAILED","CANCELLED"].includes(value) ? "status-danger" : ["PENDING","RUNNING","ACTIVE","PARTIAL","GENERATING"].includes(value) ? "status-warning" : "status-success";
  const drawer = document.getElementById("inspection-drawer"), body = document.getElementById("inspection-action-body");
  const params = new URLSearchParams(location.search);
  const state = {items:[],targets:[],agents:[],selected:null,run:null,result:null,selection:0,conversation:null,lookupLoading:false,lookupFailed:false,continuation:false,loadingAgents:false,submitting:false,agentId:"",message:"",planId:params.get("planId") || params.get("plan_id") || "",canManage:false,listRequest:0};
  const targetName = id => state.targets.find(target => target.target_id === id)?.display_name || id;
  const canContinue = () => Boolean(state.run && !state.lookupLoading && !state.lookupFailed && (state.conversation || !["CREATED","QUEUED","RUNNING"].includes(state.run.status)));
  function quality() {
    const result = state.result, payload = result?.payload || {}, scope = payload.scope || {};
    const summary = rows(state.selected?.runs).find(run => run.run_id === state.run?.ops_run_id);
    let status, percent = null, covered = 0, total = 0;
    if (!result?.final_artifact) status = ["CREATED","QUEUED","RUNNING","WAITING_INPUT","WAITING_APPROVAL"].includes(state.run?.status) ? "GENERATING" : ["FAILED","CANCELLED","EXPIRED"].includes(state.run?.status) ? "FAILED" : "PARTIAL";
    else if (result.final_artifact.schema_version === "REPORT_CONTENT.v1") {
      status = payload.status === "READY" ? "READY" : payload.status === "FAILED" ? "FAILED" : "PARTIAL";
      if (typeof scope.evidence_coverage_percent === "number" && Number.isFinite(scope.evidence_coverage_percent)) percent = scope.evidence_coverage_percent;
      covered = scope.evidence_covered_count || 0; total = scope.evidence_total_count || 0;
    } else {
      const valid = items => rows(items).filter(item => item && typeof item === "object" && !Array.isArray(item));
      const evidence = new Set(valid(payload.evidence).map((item,index) => String(item.step_id || item.tool_id || item.evidence_ref || `evidence-${index}`)));
      const all = new Set(evidence), gaps = valid(payload.evidence_gaps);
      gaps.forEach((item,index) => all.add(String(item.step_id || item.code || `gap-${index}`)));
      covered = evidence.size; total = all.size; percent = total ? Math.round(covered * 10000 / total) / 100 : null;
      status = payload.sufficiency_status === "ANSWERABLE" && !gaps.length ? "READY" : "PARTIAL";
    }
    if (percent === null && typeof summary?.evidence_coverage_percent === "number") {
      percent = summary.evidence_coverage_percent; covered = summary.evidence_covered_count ?? 0; total = summary.evidence_total_count ?? 0;
      status = summary.report_status === "READY" ? "READY" : summary.report_status === "FAILED" ? "FAILED" : "PARTIAL";
    }
    return {label:({READY:"证据充分",PARTIAL:"证据有缺口",GENERATING:"结论生成中",FAILED:"诊断未完成"})[status],tone:status === "READY" ? "status-success" : status === "FAILED" ? "status-danger" : "status-warning",coverage:percent === null ? ["CREATED","QUEUED","RUNNING"].includes(state.run?.status) ? "覆盖率待生成" : "历史报告未记录覆盖率" : `证据覆盖 ${percent}%（${covered}/${total}）`};
  }
  function runQuality(run) {
    if (typeof run.evidence_coverage_percent === "number") {
      const parts = [`语义满足 ${run.evidence_coverage_percent}%`];
      if (typeof run.tool_execution_coverage_percent === "number") parts.push(`数据库检查 ${run.tool_execution_coverage_percent}%`);
      if (typeof run.monitoring_sample_coverage_percent === "number") parts.push(`监控采样 ${run.monitoring_sample_coverage_percent}%`);
      if (typeof run.evidence_gap_count === "number") parts.push(`缺口 ${run.evidence_gap_count} 项`);
      return parts.join(" · ");
    }
    return ({READY:"证据完整",PARTIAL:"证据有缺口",FAILED:"报告生成失败",GENERATING:"报告生成中"})[run.report_status] || "";
  }
  function renderFacts() {
    const detail = state.selected, evidence = quality();
    if (!detail) return;
    const group = (label,count,content) => `<section class="inspection-fact-group"><header><span>${label}</span><span>${count}</span></header>${content}</section>`;
    const observation = items => `<dl class="inspection-observation">${items.map(([label,value]) => `<div><dt>${label}</dt><dd>${esc(value)}</dd></div>`).join("")}</dl>`;
    text("inspection-status",`执行${statusLabel(detail.status)}`);document.getElementById("inspection-status").className = `status-pill ${statusClass(detail.status)}`;
    text("inspection-observed",shell.fmt(detail.scheduled_at));text("inspection-title",detail.plan_display_name);
    text("inspection-counts",`${detail.target_count} 个目标 · ${detail.completed_count} 个完成 · ${detail.failed_count} 个失败`);
    const error = document.getElementById("inspection-error");error.hidden = !(detail.error_message || detail.error_code);error.querySelector("span").textContent = detail.error_message || detail.error_code || "";
    document.getElementById("inspection-facts").innerHTML = group("批次概况",`${detail.target_count} 个目标`,observation([["执行状态",statusLabel(detail.status)],["计划触发",shell.fmt(detail.scheduled_at)],["完成时间",shell.fmt(detail.completed_at)]]))
      + group("目标结果",rows(detail.runs).length,`<div class="inspection-run-tabs">${rows(detail.runs).map(run => `<button type="button" class="${run.run_id === state.run?.ops_run_id ? "active" : ""}" data-inspection-run="${esc(run.run_id)}"><strong>${esc(targetName(run.target_id))}</strong><span class="inspection-run-statuses">${badge(`执行${statusLabel(run.status)}`,statusClass(run.status))}${runQuality(run) ? badge(runQuality(run),statusClass(run.report_status)) : ""}</span></button>`).join("")}</div>${rows(detail.runs).length ? "" : '<p class="inspection-empty">本次巡检没有生成目标运行记录。</p>'}`)
      + (state.run ? group("当前目标","质量",observation([["数据库目标",targetName(state.run.target_id)],["Agent 执行",statusLabel(state.run.status)],["报告质量",evidence.label],["证据覆盖",evidence.coverage]])) : "")
      + (state.canManage ? '<section class="inspection-fact-group"><button class="secondary-button inspection-plan-link" type="button" data-inspection-source-plan>查看来源计划</button></section>' : "");
  }
  function renderMode() {
    text("inspection-mode-title",state.continuation ? "继续深入诊断" : "Agent 巡检结论");
    text("inspection-mode-description",state.continuation ? "选择 Agent，并说明对本次巡检结果的具体疑问。" : "阅读 Agent 结论，并与左侧执行和证据事实持续对照。");
    const result = drawer.querySelector('[role="tab"][data-inspection-results]'), continuation = drawer.querySelector('[role="tab"][data-inspection-continue]');
    result.classList.toggle("active",!state.continuation);result.setAttribute("aria-selected",String(!state.continuation));
    continuation.classList.toggle("active",state.continuation);continuation.setAttribute("aria-selected",String(state.continuation));continuation.disabled = !canContinue();continuation.textContent = state.conversation ? "进入深入诊断" : "继续调查";
  }
  function renderResult() {
    const evidence = quality(), ready = Boolean(state.result?.final_artifact);
    const content = state.result?.final_artifact?.schema_version === "REPORT_CONTENT.v1" ? KBotAIOpsTurnResult.inspection(state.result.payload) : `<div class="aiops-turn-result">${KBotAIOpsTurnResult.html(state.result?.payload,ready ? "本次诊断尚未形成可展示的四段结论。" : "本次巡检尚未形成可展示结果。")}</div>`;
    body.innerHTML = `<section class="inspection-result-view"><header class="inspection-result-summary"><div><h4>Agent 巡检诊断结果</h4><p>执行完成度与报告质量分别展示，证据缺口不会改变执行结果。</p></div><div class="inspection-result-statuses">${state.run ? badge(`执行${statusLabel(state.run.status)}`,statusClass(state.run.status)) : ""}${badge(evidence.label,evidence.tone)}${badge(evidence.coverage,evidence.tone)}</div></header><div class="inspection-diagnosis">${content}</div><div class="inspection-handoff${canContinue() ? "" : " waiting"}"><div><h4>${state.conversation ? "已有深入诊断会话" : ready ? "对结论或证据存疑？" : "巡检尚未形成可用结果"}</h4><p>${state.conversation ? "直接返回此前从本次巡检发起的会话，继续查看或追问。" : ready ? "带上具体问题继续调查；完整巡检结果会随会话继承。" : "运行结束后可继承巡检上下文继续调查。"}</p></div><button class="primary-button" type="button" data-inspection-continue${canContinue() ? "" : " disabled"}>${state.lookupLoading ? "正在检查会话…" : state.conversation ? "进入深入诊断会话" : "继续深入诊断"}</button></div></section>`;
    KBotAIOpsTurnResult.mount(body);
  }
  function updateControls() {
    const form = body.querySelector("form");if (!form) return;
    form.elements.agent_id.disabled = state.loadingAgents || state.submitting;form.elements.message.disabled = state.submitting;
    form.querySelector('[data-inspection-results]').disabled = state.submitting;
    const submit = form.querySelector('[type="submit"]');submit.disabled = state.loadingAgents || state.submitting || !state.agentId || !state.message.trim();submit.textContent = state.submitting ? "正在创建会话…" : "进入诊断对话";
  }
  function renderContinuation() {
    const agents = state.agents.filter(agent => agent.status === "ACTIVE");
    body.innerHTML = `<form class="quick-create-form inspection-continuation-form"><section class="inspection-continuation-context"><span>继承</span><div><strong>完整巡检结论、执行上下文与证据</strong><p>当前目标、本次 Agent 结论、检查项证据和数据缺口会一起交给新会话。</p></div></section><label>诊断 Agent<select name="agent_id" required><option value="" disabled>${state.loadingAgents ? "正在读取可用 Agent…" : agents.length ? "请选择 Agent" : "暂无已启用 Agent"}</option>${agents.map(agent => `<option value="${esc(agent.agent_id)}">${esc(agent.display_name)} · ${rows(agent.allowed_business_system_ids).length} 个业务系统</option>`).join("")}</select><small>Agent 会根据本次巡检来源和其业务系统授权范围解析诊断资源。</small></label><label>对现有结果的疑问<textarea name="message" rows="8" maxlength="32000" required></textarea><small>请指出存疑的结论、证据或状态差异，问题会与完整巡检结果一起进入会话。</small></label><footer class="inspection-continuation-actions"><span>巡检事实会继续保留在左侧供核对。</span><div><button class="secondary-button" type="button" data-inspection-results>返回结果</button><button class="primary-button" type="submit">进入诊断对话</button></div></footer></form>`;
    const form = body.querySelector("form");form.elements.agent_id.value = state.agentId;form.elements.message.value = state.message;
    form.oninput = () => {state.agentId = form.elements.agent_id.value;state.message = form.elements.message.value;updateControls();};form.onchange = form.oninput;
    form.onsubmit = startContinuation;updateControls();
  }
  async function openContinuation() {
    if (!canContinue() || state.continuation) return;
    if (state.conversation) {location.href = `./chat.html?conversation=${encodeURIComponent(state.conversation.conversation_id)}`;return;}
    state.continuation = true;state.message ||= "我对本次巡检结论或证据仍有疑问，请继续核验，并说明需要重点确认的状态差异。";
    const selection = state.selection;state.loadingAgents = !state.agents.length;renderMode();renderContinuation();
    if (!state.loadingAgents) return;
    try {const payload = await request("/agents");if (selection === state.selection) state.agents = rows(payload);}
    catch (error) {if (selection === state.selection) shell.toast(error.message || "无法读取可用 Agent");}
    finally {if (selection === state.selection) {state.loadingAgents = false;if (state.continuation) renderContinuation();}}
  }
  async function startContinuation(event) {
    event.preventDefault();if (state.submitting || !state.run || !state.agentId || !state.message.trim()) return;
    state.submitting = true;updateControls();
    try {
      const receipt = await request("/conversations",{method:"POST",headers:{"Idempotency-Key":KBotAIOpsAuth.uuid()},body:JSON.stringify({agent_id:state.agentId,source_run_id:state.run.ops_run_id,content:[{content_type:"TEXT",text:state.message.trim()}]})});
      location.href = `./chat.html?conversation=${encodeURIComponent(receipt.conversation_id)}`;
    } catch (error) {shell.toast(error.message || "无法继续深入诊断");}
    finally {state.submitting = false;updateControls();}
  }
  function resetContinuation() {state.conversation = null;state.lookupLoading = false;state.lookupFailed = false;state.continuation = false;state.loadingAgents = false;state.agentId = "";state.message = "";}
  async function selectRun(summary) {
    if (state.submitting) return;
    const selection = ++state.selection;resetContinuation();state.run = null;state.result = null;
    try {
      const run = await request(`/runs/${encodeURIComponent(summary.run_id)}`);if (selection !== state.selection) return;
      state.run = run;state.lookupLoading = true;renderFacts();renderMode();renderResult();
      try {const result = await request(`/runs/${encodeURIComponent(run.ops_run_id)}/result`);if (selection !== state.selection) return;state.result = result;}
      catch (error) {if (![404,409].includes(error.status)) throw error;}
      if (selection !== state.selection) return;
      try {const conversations = await request(`/conversations?run_id=${encodeURIComponent(run.ops_run_id)}&limit=1`);if (selection !== state.selection) return;state.conversation = rows(conversations)[0] || null;}
      catch (error) {if (selection !== state.selection) return;state.lookupFailed = true;shell.toast(error.message || "无法检查关联的深入诊断会话");}
      finally {if (selection === state.selection) state.lookupLoading = false;}
    } catch (error) {if (selection === state.selection) {state.lookupLoading = false;state.lookupFailed = true;shell.toast(error.message || "无法读取目标巡检结果");}}
    if (selection === state.selection && state.selected) {renderFacts();renderMode();renderResult();}
  }
  function close() {
    if (state.submitting) return;
    ++state.selection;state.selected = null;state.run = null;state.result = null;resetContinuation();
    if (drawer.open) drawer.close();body.innerHTML = "";KBotAIOpsTurnResult.mount(body);
  }
  async function open(item) {
    if (state.submitting) return;
    close();const selection = state.selection;
    try {
      const detail = await request(`/inspection-fires/${encodeURIComponent(item.fire_id)}`);if (selection !== state.selection) return;
      state.selected = detail;drawer.showModal();renderFacts();renderMode();renderResult();
      if (detail.runs?.[0]) await selectRun(detail.runs[0]);
    } catch (error) {if (selection === state.selection) shell.toast(error.message || "无法读取巡检报告");}
  }
  function renderList() {
    const items = state.planId ? state.items.filter(item => item.plan_id === state.planId) : state.items;
    const attention = items.filter(item => ["PARTIAL","FAILED"].includes(item.status)).length;
    text("inspection-total",items.length);text("inspection-running",items.filter(item => ["QUEUED","RUNNING"].includes(item.status)).length);text("inspection-attention",attention);text("inspection-completed",items.filter(item => item.status === "COMPLETED").length);
    document.getElementById("inspection-attention-card").classList.toggle("is-danger",Boolean(attention));
    text("inspection-list-title",state.planId ? "计划巡检报告" : "巡检报告");text("inspection-count",`${items.length} 批`);document.getElementById("inspection-clear-plan").hidden = !state.planId;
    document.getElementById("inspection-body").innerHTML = items.length ? items.map(item => `<tr><td><strong class="table-primary">${esc(item.plan_display_name)}</strong><small class="table-secondary technical-value">${esc(item.fire_id)}</small></td><td>${esc(shell.fmt(item.scheduled_at))}${item.completed_at ? `<small class="table-secondary">完成于 ${esc(shell.fmt(item.completed_at))}</small>` : ""}</td><td>${badge(statusLabel(item.status),statusClass(item.status))}</td><td>${item.target_count} 个目标</td><td>${item.completed_count} / ${item.failed_count}</td><td><button class="row-action" type="button" data-inspection-id="${esc(item.fire_id)}">查看报告</button></td></tr>`).join("") : `<tr><td colspan="6" class="empty-cell">${state.planId ? "该计划尚未产生巡检报告。" : "当前业务空间还没有巡检报告。"}</td></tr>`;
  }
  async function load() {
    const sequence = ++state.listRequest;
    document.getElementById("inspection-loading").hidden = false;
    document.querySelectorAll("main > .inspection-tabs,main > .inspection-summary-grid,main > .table-card").forEach(node => {node.hidden = true;});
    try {const [fires,targets] = await Promise.all([request("/inspection-fires?limit=100"),request("/targets?limit=200")]);if (sequence !== state.listRequest) return;state.items = rows(fires.items);state.targets = rows(targets.items);}
    catch (error) {if (sequence === state.listRequest) shell.toast(error.message || "无法读取日常巡检");}
    finally {if (sequence === state.listRequest) {renderList();document.getElementById("inspection-loading").hidden = true;document.querySelectorAll("main > .inspection-tabs,main > .inspection-summary-grid,main > .table-card").forEach(node => {node.hidden = false;});}}
  }
  shell.ready.then(access => {
    state.canManage = rows(access.permissions).includes("aiops:plan_manage");
    document.querySelectorAll("[data-plan-manage]").forEach(node => {node.hidden = !state.canManage;});
    document.querySelectorAll("[data-inspection-plans]").forEach(node => {node.onclick = () => {location.href = "./inspection-plans.html";};});
    document.getElementById("refresh-workspace").onclick = load;
    document.getElementById("inspection-clear-plan").onclick = () => {state.planId = "";const url = new URL(location.href);url.searchParams.delete("plan_id");url.searchParams.delete("planId");history.replaceState(null,"",url);renderList();};
    document.getElementById("inspection-body").onclick = event => {const button = event.target.closest("[data-inspection-id]");if (button) {const item = state.items.find(item => item.fire_id === button.dataset.inspectionId);if (item) void open(item);}};
    drawer.onclick = event => {
      if (state.submitting) return;
      if (event.target.closest("[data-close-inspection]")) close();
      else if (event.target.closest("[data-inspection-source-plan]")) location.href = `./inspection-plans.html?planId=${encodeURIComponent(state.selected.plan_id)}`;
      else if (event.target.closest("[data-inspection-results]")) {state.continuation = false;renderMode();renderResult();}
      else if (event.target.closest("[data-inspection-continue]")) void openContinuation();
      else {const button = event.target.closest("[data-inspection-run]");if (button) {const summary = state.selected.runs.find(run => run.run_id === button.dataset.inspectionRun);if (summary) void selectRun(summary);}}
    };
    drawer.addEventListener("cancel",event => {if (state.submitting) event.preventDefault();});
    drawer.addEventListener("close",() => {if (state.selected && !drawer.open) close();});
    return load();
  }).catch(error => shell.toast(error.message));
})();
