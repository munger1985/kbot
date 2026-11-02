(function () {
  "use strict";
  if (document.body.dataset.page !== "inspection-detail") return;
  const shell = globalThis.KBotAIOpsShell, esc = shell.escape;
  const request = (path,init) => KBotAIOpsAuth.request(`/api/v1/apps/aiops${path}`,init);
  const rows = value => Array.isArray(value) ? value : [];
  const params = new URLSearchParams(location.search), fireId = params.get("id") || "";
  const root = document.getElementById("inspection-detail-content");
  const state = {fire:null,targets:[],runId:params.get("runId") || "",agents:[],conversation:null,canManage:false,selection:0,loadSequence:0,lookupLoading:false,lookupFailed:false,continuation:false,agentsLoading:false,submitting:false,agentId:"",message:""};
  const selectedRun = () => rows(state.fire?.runs).find(run => run.run_id === state.runId);
  const targetName = id => state.targets.find(target => target.target_id === id)?.display_name || id;
  const statusLabel = value => ({CREATED:"已创建",QUEUED:"等待执行",RUNNING:"运行中",COMPLETED:"已完成",PARTIAL:"部分完成",FAILED:"失败",CANCELLED:"已取消",READY:"结论就绪",GENERATING:"生成中",EXPIRED:"已超时"})[value] || value;
  const statusClass = value => ["FAILED","CANCELLED","EXPIRED"].includes(value) ? "status-danger" : ["PARTIAL","CREATED","QUEUED","RUNNING","GENERATING"].includes(value) ? "status-warning" : "status-success";
  const badge = (label,status) => `<span class="status-pill ${statusClass(status)}">${esc(label)}</span>`;
  const observation = items => `<dl class="inspection-observation">${items.map(([label,value]) => `<div><dt>${esc(label)}</dt><dd>${value}</dd></div>`).join("")}</dl>`;
  const canContinue = () => Boolean(selectedRun() && !state.lookupLoading && !state.lookupFailed && (state.conversation || !["CREATED","QUEUED","RUNNING"].includes(selectedRun().status)));
  function noRunText() {
    return state.fire.error_message || state.fire.error_code || (["QUEUED","RUNNING"].includes(state.fire.status)
      ? "巡检尚未生成目标运行，正在等待调度或启动取证。"
      : "本批次已结束，但未生成目标取证运行，未形成巡检报告。请核对执行日志和来源计划后重新执行。");
  }
  function render() {
    const fire = state.fire;if (!fire) return;
    document.getElementById("inspection-refresh").disabled = state.submitting;
    const plan = document.getElementById("inspection-source-plan");plan.hidden = !state.canManage;plan.href = `./inspection-plans.html?planId=${encodeURIComponent(fire.plan_id)}`;
    const cards = rows(fire.runs).map(run => {
      const facts = [["业务检查语义满足率",run.evidence_coverage_percent],["数据库工具执行覆盖率",run.tool_execution_coverage_percent],["监控采样覆盖率",run.monitoring_sample_coverage_percent]].filter(([,value]) => typeof value === "number").map(([label,value]) => [label,`${value}%`]);
      if (typeof run.evidence_gap_count === "number") facts.push(["报告证据缺口",`${run.evidence_gap_count} 项`]);
      const failure = run.error_message || run.error_code ? `<div class="notice notice-error" role="alert"><strong>执行未完成</strong><p>${esc(run.error_message || "目标运行失败，请核对执行日志。")}</p>${run.error_code ? `<code>${esc(run.error_code)}</code>` : ""}</div>` : "";
      const report = run.report_id ? `<a class="primary-button" href="./report-detail.html?id=${encodeURIComponent(run.report_id)}&amp;inspectionFireId=${encodeURIComponent(fireId)}&amp;runId=${encodeURIComponent(run.run_id)}">查看报告</a>` : "";
      return `<article class="table-card inspection-target-card${state.runId === run.run_id ? " active" : ""}"><h3>${esc(targetName(run.target_id))}</h3><div class="inspection-run-statuses">${badge(`执行${statusLabel(run.status)}`,run.status)}${run.report_status ? badge(`报告${statusLabel(run.report_status)}`,run.report_status) : ""}</div>${observation(facts)}${failure}${!run.report_id ? `<p class="inspection-empty">${["CREATED","QUEUED","RUNNING"].includes(run.status) ? "报告尚在生成。" : "本目标未生成正式报告，不能展示为空白报告。"}</p>` : ""}<footer class="button-row">${report}<button class="secondary-button" type="button" data-select-run="${esc(run.run_id)}"${state.submitting ? " disabled" : ""}>${state.runId === run.run_id ? "当前调查目标" : "选择调查目标"}</button></footer></article>`;
    }).join("");
    root.innerHTML = `<section class="table-card inspection-batch-summary" aria-label="巡检事实"><p class="eyebrow">巡检批次</p><h1>${esc(fire.plan_display_name)}</h1>${fire.business_system_name ? `<p>业务系统：${esc(fire.business_system_name)}</p>` : ""}${fire.system_scope ? `<details><summary>本批次冻结的数据库范围</summary><ul>${rows(fire.system_scope.resources).map(resource => `<li>${esc(resource.display_name)} · ${esc(resource.db_type)}</li>`).join("")}${rows(fire.system_scope.excluded_resources).map(resource => `<li>${esc(resource.display_name)} · ${esc(resource.reason)}</li>`).join("")}</ul></details>` : ""}<p>${fire.target_count} 个目标 · ${fire.completed_count} 个完成 · ${fire.failed_count} 个失败</p>${observation([["执行状态",badge(statusLabel(fire.status),fire.status)],["计划触发",esc(shell.fmt(fire.scheduled_at))],["完成时间",esc(shell.fmt(fire.completed_at))]])}${fire.error_message || fire.error_code ? `<p class="notice notice-error">${esc(fire.error_message || fire.error_code)}</p>` : ""}</section><section class="inspection-targets" aria-label="目标结果"><h2>目标执行与报告</h2><p>执行完成度与报告质量分别展示；查看报告与运维报告使用同一份正式报告。</p><div class="inspection-target-grid">${cards}</div>${rows(fire.runs).length ? "" : `<p class="notice notice-error">${esc(noRunText())}</p>`}</section>${selectedRun() ? `<section class="table-card inspection-follow-up" aria-label="继续深入诊断"><header><h2>${esc(targetName(selectedRun().target_id))} · 继续深入诊断</h2><p>继承完整巡检结论、执行上下文与证据；正式报告不会因继续调查而改变。</p></header><div id="inspection-continuation"></div></section>` : ""}`;
    renderContinuation();
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
      pane.innerHTML = `${state.lookupFailed ? '<p class="notice notice-error">关联会话读取失败，请刷新后重试。</p>' : ""}<button class="primary-button" type="button" data-continue${canContinue() ? "" : " disabled"}>${state.lookupLoading ? "正在检查会话…" : state.conversation ? "进入深入诊断会话" : "继续深入诊断"}</button>`;return;
    }
    const agents = state.agents.filter(agent => agent.status === "ACTIVE");
    pane.innerHTML = `<form class="quick-create-form inspection-continuation-form"><label>诊断 Agent<select name="agent_id" required><option value="" disabled>${state.agentsLoading ? "正在读取可用 Agent…" : "请选择 Agent"}</option>${agents.map(agent => `<option value="${esc(agent.agent_id)}">${esc(agent.display_name)} · ${rows(agent.allowed_business_system_ids).length} 个业务系统</option>`).join("")}</select></label><label>对现有结果的疑问<textarea name="message" rows="6" maxlength="32000" required></textarea></label><footer class="button-row"><button class="secondary-button" type="button" data-return-results>返回结果</button><button class="primary-button" type="submit">进入诊断对话</button></footer></form>`;
    const form = pane.querySelector("form");form.elements.agent_id.value = state.agentId;form.elements.message.value = state.message;
    form.oninput = () => {state.agentId = form.elements.agent_id.value;state.message = form.elements.message.value;updateControls();};form.onchange = form.oninput;form.onsubmit = startContinuation;updateControls();
  }
  async function selectRun(run) {
    if (state.submitting) return;
    const selection = ++state.selection;state.runId = run.run_id;state.continuation = false;state.conversation = null;state.lookupLoading = true;state.lookupFailed = false;state.agentId = "";state.message = "";
    const url = new URL(location.href);url.searchParams.set("runId",run.run_id);history.replaceState(null,"",url);render();
    try {const conversations = await request(`/conversations?run_id=${encodeURIComponent(run.run_id)}&limit=1`);if (selection === state.selection) state.conversation = rows(conversations)[0] || null;}
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
    finally {state.agentsLoading = false;if (selection === state.selection && state.continuation) renderContinuation();}
  }
  async function startContinuation(event) {
    event.preventDefault();if (!canContinue() || state.submitting || !state.agentId || !state.message.trim()) return;
    state.submitting = true;document.getElementById("inspection-refresh").disabled = true;root.querySelectorAll("[data-select-run]").forEach(button => {button.disabled = true;});updateControls();
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
      const run = rows(fire.runs).find(run => run.run_id === state.runId) || fire.runs?.[0];if (run) await selectRun(run);
    } catch(error) {if (sequence === state.loadSequence) {errorNode.hidden = false;errorNode.textContent = "无法读取巡检批次，请重试或返回日常巡检。";shell.toast(error.message);}}
    finally {if (sequence === state.loadSequence) loading.hidden = true;}
  }
  shell.ready.then(access => {
    state.canManage = rows(access.permissions).includes("aiops:plan_manage");
    const planId = params.get("planId");if (planId) document.getElementById("inspection-back").href = `./inspections.html?planId=${encodeURIComponent(planId)}`;
    document.getElementById("inspection-refresh").onclick = load;
    root.onclick = event => {
      if (state.submitting) return;
      const button = event.target.closest("[data-select-run]");
      if (button) {const run = state.fire.runs.find(run => run.run_id === button.dataset.selectRun);if (run) void selectRun(run);}
      else if (event.target.closest("[data-continue]")) void openContinuation();
      else if (event.target.closest("[data-return-results]")) {state.continuation = false;renderContinuation();}
    };
    return load();
  }).catch(error => shell.toast(error.message));
})();
