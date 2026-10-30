(function () {
  "use strict";
  if (document.body.dataset.page === "run-detail") {const params = new URLSearchParams(location.search);if (params.has("id")) {params.set("runId",params.get("id"));params.delete("id");}location.replace(`./runs.html?${params}`);return;}
  if (document.body.dataset.page !== "runs") return;
  const shell = globalThis.KBotAIOpsShell, esc = shell.escape, api = "/api/v1/apps/aiops";
  const request = (path,init) => KBotAIOpsAuth.request(api + path,init);
  const rows = value => Array.isArray(value) ? value : [];
  const terminal = new Set(["COMPLETED","PARTIAL","FAILED","CANCELLED","EXPIRED"]);
  const state = {items:[],selected:null,result:null,activeId:"",timer:null,terminalRefreshes:0,sequence:0,selectionSequence:0,busy:false,confirm:null};
  const drawer = document.getElementById("run-drawer");
  const text = (id,value) => {document.getElementById(id).textContent = value ?? "";};
  const workflowLabel = value => ({CHAT_TURN:"智能诊断",ALERT_DIAGNOSIS:"告警诊断",INSPECTION:"日常巡检",CHANGE:"受控变更",VERIFICATION:"结果验证",EVIDENCE_FOLLOWUP:"后台未来观察"})[value] || value;
  const triggerLabel = value => ({CHAT:"对话发起",API:"接口触发",ROOT:"根任务",ALERT:"监控告警",SCHEDULE:"巡检计划"})[value] || value;
  const statusLabel = value => ({CREATED:"已创建",QUEUED:"等待执行",RUNNING:"运行中",WAITING_USER:"等待补充信息",COMPLETED:"已完成",PARTIAL:"部分完成",FAILED:"失败",CANCELLED:"已取消",EXPIRED:"已过期"})[value] || value;
  const statusClass = value => value === "FAILED" ? "status-danger" : ["RUNNING","CREATED"].includes(value) ? "status-warning" : "status-success";
  async function load() {
    const sequence = ++state.sequence;document.getElementById("runs-loading").hidden = false;document.getElementById("runs-table").hidden = true;
    try {const page = await request("/runs?limit=100");if (sequence !== state.sequence) return;state.items = rows(page.items);}
    catch (error) {if (sequence === state.sequence) shell.toast(error.message || "无法读取运行记录");}
    finally {if (sequence === state.sequence) {
      text("runs-count",`${state.items.length} 条`);
      document.getElementById("runs-body").innerHTML = state.items.length ? state.items.map(run => `<tr><td><strong class="table-primary">${esc(workflowLabel(run.workflow_kind))}</strong><small class="table-secondary technical-value">${esc(run.ops_run_id)}</small></td><td>${esc(triggerLabel(run.trigger_type))}</td><td><span class="status-pill ${statusClass(run.status)}">${esc(statusLabel(run.status))}</span></td><td>${esc(shell.fmt(run.created_at))}</td><td>${esc(shell.fmt(run.completed_at))}</td><td><button class="row-action" type="button" data-open-run="${esc(run.ops_run_id)}">查看详情</button></td></tr>`).join("") : '<tr><td colspan="6" class="empty-cell">当前业务空间还没有运行记录。</td></tr>';
      document.getElementById("runs-loading").hidden = true;document.getElementById("runs-table").hidden = false;
    }}
  }
  function observationHtml(payload) {
    const summary = payload?.observation_series;if (!summary) return "";
    const cell = value => value === null || value === undefined ? "—" : typeof value === "object" ? JSON.stringify(value) : String(value);
    return `<section class="aiops-observation-series"><header><h3>后台未来观察</h3><p>已采集 ${esc(summary.completed_count)}/${esc(summary.requested.count)} 次，间隔至少 ${esc(summary.requested.interval_seconds)} 秒。仅代表以下实际时点，不覆盖过去请求窗口，未自动计算差分。</p></header>${rows(payload.tool_outcomes).map(point => {
      const observation = point.observation;
      return `<section class="key-value-card">${observation ? `<h4>${esc(shell.fmt(observation.captured_at))} · ${esc(observation.row_count)} 行${observation.truncated ? " · 已截断" : ""}</h4><div class="table-scroll" tabindex="0" aria-label="未来观察原始结果，可横向滚动"><table><thead><tr>${rows(observation.columns).map(column => `<th>${esc(column.name)}</th>`).join("")}</tr></thead><tbody>${rows(observation.rows).map(row => `<tr>${row.map(value => `<td><span title="${esc(cell(value))}">${esc(cell(value))}</span></td>`).join("")}</tr>`).join("")}</tbody></table></div>${observation.row_count ? "" : '<p class="table-secondary">此采集时点没有匹配记录；不是未执行查询。</p>'}` : ""}${point.gap ? `<p class="table-secondary">${esc(point.gap.code === "FOLLOWUP_PENDING" ? "等待后续采样时点，已完成的观察保留如下。" : `${point.gap.detail}（${point.gap.code}）`)}</p>` : ""}</section>`;
    }).join("")}</section>`;
  }
  function renderDetail() {
    const run = state.selected;if (!run) return;
    text("run-detail-eyebrow",`${workflowLabel(run.workflow_kind)} · ${statusLabel(run.status)}`);text("run-detail-id",run.ops_run_id);
    document.getElementById("run-detail-facts").innerHTML = `<table class="key-value-table"><tbody><tr><th>触发方式</th><td>${esc(triggerLabel(run.trigger_type))}</td></tr><tr><th>目标</th><td class="technical-value">${esc(run.target_id)}</td></tr><tr><th>创建时间</th><td>${esc(shell.fmt(run.created_at))}</td></tr><tr><th>完成时间</th><td>${esc(shell.fmt(run.completed_at))}</td></tr></tbody></table>`;
    const empty = state.result?.final_artifact ? "本次诊断尚未形成可展示的四段结论。" : !terminal.has(run.status) ? `Agent 正在诊断，当前状态：${run.status}。页面会自动更新结果。` : `本次运行已结束但未形成可展示结果，最终状态：${run.status}。`;
    const output = document.getElementById("run-result");
    if (run.workflow_kind === "EVIDENCE_FOLLOWUP" && state.result?.payload) output.innerHTML = observationHtml(state.result.payload);
    else output.innerHTML = `<div class="aiops-turn-result">${KBotAIOpsTurnResult.html(state.result?.payload,empty)}</div>`;
    KBotAIOpsTurnResult.mount(output);
    const lifecycle = document.getElementById("run-lifecycle"), items = rows(run.readonly_lifecycle?.items);
    lifecycle.hidden = !items.length;
    const labels = {IN_PROGRESS:"执行租约有效，尚无终止回执",RELEASED:"连接清理完成，额度已归还",CANCEL_REQUESTED:"已请求取消，尚未确认终止",TERMINATION_CONFIRMED:"自有查询清理已获协议确认",TERMINATION_UNKNOWN:"终止状态未知，不能确认数据库已停止"};
    lifecycle.innerHTML = items.length ? `<h3>只读查询清理回执</h3><p>仅描述本次执行器自有查询的清理状态，不是 SQL 根因证据。运行已取消不等于数据库已停止；回执可能稍后到达，可刷新查看。</p><div class="table-scroll"><table><thead><tr><th>数据库</th><th>清理状态</th><th>租约</th></tr></thead><tbody>${items.map(item => `<tr><td>${esc(item.observation?.db_type ?? "—")}</td><td>${esc(labels[item.status] || "终止状态未知")}</td><td class="technical-value">${esc(item.lease_id)}</td></tr>`).join("")}</tbody></table></div>${run.readonly_lifecycle.truncated ? "<p>仅展示最近 50 条自有查询租约，不代表本次运行的全部查询。</p>" : ""}` : "";
    document.getElementById("continue-run-diagnosis").hidden = !state.result?.final_artifact || run.workflow_kind === "EVIDENCE_FOLLOWUP";
    document.getElementById("cancel-future-observation").hidden = run.workflow_kind !== "EVIDENCE_FOLLOWUP" || terminal.has(run.status);
    document.getElementById("cancel-future-observation").disabled = state.busy;
  }
  function clearRefresh() {clearTimeout(state.timer);state.timer = null;}
  async function open(runId,{background=false}={}) {
    if (!background) {state.activeId = runId;state.terminalRefreshes = 0;clearRefresh();}
    if (state.activeId !== runId) return;
    const sequence = ++state.selectionSequence;
    const current = () => state.activeId === runId && sequence === state.selectionSequence;
    try {
      const detail = await request(`/runs/${encodeURIComponent(runId)}`);if (!current()) return;
      state.selected = detail;
      if (!background) {state.result = null;renderDetail();if (!drawer.open) drawer.showModal();drawer.scrollTop = 0;}
      try {const output = await request(`/runs/${encodeURIComponent(runId)}/result`);if (!current()) return;state.result = output;}
      catch (error) {if (![404,409].includes(error.status)) throw error;if (!current()) return;state.result = null;}
      if (!current()) return;renderDetail();
      const pending = rows(detail.readonly_lifecycle?.items).some(item => ["IN_PROGRESS","CANCEL_REQUESTED","TERMINATION_UNKNOWN"].includes(item.status));
      if (!terminal.has(detail.status) || (pending && state.terminalRefreshes++ < 3)) {clearRefresh();state.timer = setTimeout(() => open(runId,{background:true}),3000);}
    } catch (error) {
      if (!current()) return;
      if (background) {if (!state.selected || !terminal.has(state.selected.status) || state.terminalRefreshes++ < 3) state.timer = setTimeout(() => open(runId,{background:true}),5000);}
      else shell.toast(error.message || "无法读取运行详情");
    }
  }
  function close() {state.activeId = "";++state.selectionSequence;clearRefresh();state.selected = null;state.result = null;if (drawer.open) drawer.close();}
  function settleConfirmation(value) {const resolve = state.confirm;state.confirm = null;document.getElementById("run-confirm-dialog").close();resolve?.(value);}
  async function cancelObservation() {
    const run = state.selected;if (!run || state.busy || state.confirm || run.workflow_kind !== "EVIDENCE_FOLLOWUP" || terminal.has(run.status)) return;
    const confirm = document.getElementById("run-confirm-dialog");confirm.showModal();confirm.querySelector("[data-cancel-run-confirm]").focus();
    if (!await new Promise(resolve => {state.confirm = resolve;})) return;
    state.busy = true;renderDetail();
    try {await request(`/runs/${encodeURIComponent(run.ops_run_id)}/commands`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({command_id:KBotAIOpsAuth.uuid(),idempotency_key:KBotAIOpsAuth.uuid(),ops_run_id:run.ops_run_id,command:{command_type:"CANCEL_RUN",expected_row_version:run.row_version}})});shell.toast("已请求取消后续采样，已采集结果仍保留");if (state.activeId === run.ops_run_id) await open(run.ops_run_id);}
    catch (error) {shell.toast(error.message || "无法取消未来观察");}
    finally {state.busy = false;renderDetail();}
  }
  shell.ready.then(async () => {
    document.getElementById("refresh-runs").onclick = load;document.getElementById("runs-body").onclick = event => {const button = event.target.closest("[data-open-run]");if (button) open(button.dataset.openRun);};
    drawer.onclick = event => {if (event.target.closest("[data-close-run]")) close();};drawer.addEventListener("close",close);
    document.getElementById("refresh-run-detail").onclick = () => {if (state.activeId) open(state.activeId);};
    document.getElementById("continue-run-diagnosis").onclick = () => {const run = state.selected;if (run && state.result?.final_artifact) location.href = `./chat.html?${new URLSearchParams({targetId:run.target_id,sourceType:"RUN",sourceId:run.ops_run_id})}`;};
    document.getElementById("cancel-future-observation").onclick = cancelObservation;
    const confirm = document.getElementById("run-confirm-dialog");confirm.querySelector("[data-cancel-run-confirm]").onclick = () => settleConfirmation(false);confirm.querySelector("[data-confirm-run]").onclick = () => settleConfirmation(true);confirm.addEventListener("cancel",event => {event.preventDefault();settleConfirmation(false);});
    addEventListener("pagehide",close,{once:true});await load();const params = new URLSearchParams(location.search), runId = params.get("runId") || params.get("run_id");if (runId) await open(runId);
  }).catch(error => shell.toast(error.message));
})();
