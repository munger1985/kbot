/* 告警列表、事实与诊断抽屉对齐 Ammolite；写入仍使用 KBot 的公开会话合同。 */
(() => {
  "use strict";
  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell, esc = shell.escape;
  const request = (path,init) => KBotAIOpsAuth.request(`${api}${path}`,init);
  const rows = value => Array.isArray(value) ? value : [];
  const terminal = new Set(["COMPLETED","PARTIAL","FAILED","CANCELLED","EXPIRED"]);
  const failed = new Set(["FAILED","CANCELLED","EXPIRED"]);
  const drawer = document.getElementById("situation-drawer");
  const body = document.getElementById("situation-action-body");
  const state = {items:[],selected:null,run:null,result:null,activeId:"",timer:null,agents:[],conversation:null,lookupLoading:false,lookupFailed:false,continuation:false,agentId:"",message:"",loadingAgents:false,submitting:false};
  const statusLabel = value => ({OPEN:"活动告警",RESOLVED:"当前无活动告警",CLOSED:"已关闭"})[value] || value;
  const statusClass = value => ["RESOLVED","CLOSED"].includes(value)?"status-success":value==="OPEN"?"status-danger":"status-warning";
  const badge = (label,tone) => `<span class="status-pill ${tone}">${esc(label)}</span>`;
  const ready = () => Boolean(state.result?.final_artifact);
  const inProgress = () => state.run?.status === "RUNNING";
  const canContinue = () => Boolean(state.selected && !state.lookupLoading && !state.lookupFailed && (state.conversation || !inProgress()));
  const text = (id,value) => {document.getElementById(id).textContent=value;};
  function executionStatus() {
    if(ready()) return {label:"执行已完成",tone:"status-success"};
    if(!state.run) return {label:"等待诊断",tone:"status-muted"};
    const labels={CREATED:"等待执行",QUEUED:"排队中",RUNNING:"自动诊断中",WAITING_INPUT:"等待输入",WAITING_APPROVAL:"等待审批",COMPLETED:"未形成结果",PARTIAL:"部分完成",FAILED:"诊断失败",CANCELLED:"已取消",EXPIRED:"已过期"};
    return {label:labels[state.run.status] || state.run.status,tone:failed.has(state.run.status)?"status-danger":"status-warning"};
  }
  function quality() {
    const payload=state.result?.payload || {}, artifact=state.result?.final_artifact;
    let status,percent=null,covered=0,total=0;
    if(!artifact) status=["CREATED","QUEUED","RUNNING","WAITING_INPUT","WAITING_APPROVAL"].includes(state.run?.status)?"GENERATING":failed.has(state.run?.status)?"FAILED":"PARTIAL";
    else if(artifact.schema_version==="REPORT_CONTENT.v1") {
      status=payload.status==="READY"?"READY":payload.status==="FAILED"?"FAILED":"PARTIAL";
      const scope=payload.scope || {};
      if(typeof scope.evidence_coverage_percent==="number" && Number.isFinite(scope.evidence_coverage_percent)) percent=scope.evidence_coverage_percent;
      covered=scope.evidence_covered_count || 0;total=scope.evidence_total_count || 0;
    } else {
      const valid=value=>rows(value).filter(item=>item && typeof item==="object" && !Array.isArray(item));
      const evidence=new Set(valid(payload.evidence).map((item,index)=>String(item.step_id || item.tool_id || item.evidence_ref || `evidence-${index}`)));
      const all=new Set(evidence),gaps=valid(payload.evidence_gaps);
      gaps.forEach((item,index)=>all.add(String(item.step_id || item.code || `gap-${index}`)));
      covered=evidence.size;total=all.size;percent=total?Math.round(covered*10000/total)/100:null;
      status=payload.sufficiency_status==="ANSWERABLE" && !gaps.length?"READY":"PARTIAL";
    }
    return {label:({READY:"证据充分",PARTIAL:"证据有缺口",GENERATING:"结论生成中",FAILED:"诊断未完成"})[status],tone:status==="READY"?"status-success":status==="FAILED"?"status-danger":"status-warning",coverage:percent===null?inProgress()?"覆盖率待生成":"历史报告未记录覆盖率":`证据覆盖 ${percent}%（${covered}/${total}）`};
  }
  function emptyText() {
    if(ready()) return "本次诊断尚未形成可展示的四段结论。";
    if(state.run && !terminal.has(state.run.status)) return `Agent 正在诊断，当前状态：${state.run.status}。页面会自动更新结果。`;
    if(state.run) return `本次自动诊断已结束但未形成可展示结果，最终状态：${state.run.status}。`;
    return "告警已接收，正在等待 Agent 自动诊断任务启动。";
  }
  function renderFacts() {
    const detail=state.selected,execution=executionStatus(),evidence=quality();
    const sources=rows(detail.monitoring_sources),signals=rows(detail.signal_events);
    const group=(title,count,content)=>`<section class="aiops-situation-fact-group"><header><span>${title}</span><span>${count}</span></header>${content}</section>`;
    const observation=items=>`<dl class="aiops-situation-observation">${items.map(([label,value])=>`<div><dt>${label}</dt><dd>${esc(value)}</dd></div>`).join("")}</dl>`;
    const status=document.getElementById("situation-status");status.textContent=`${detail.severity} · ${statusLabel(detail.status)}`;status.className=`status-pill ${statusClass(detail.status)}`;
    text("situation-observed",shell.fmt(detail.last_observed_at));text("situation-title",detail.title);document.getElementById("situation-title").title=detail.title;
    text("situation-counts",`${sources.length} 个监控来源 · ${detail.event_count} 个信号`);
    document.getElementById("situation-facts").innerHTML=group("监控来源",sources.length,`<div class="aiops-situation-source-list">${sources.length?sources.map(item=>`<article class="aiops-situation-source"><strong title="${esc(item.display_name)}">${esc(item.display_name)}</strong><code>${esc(item.source_type)}</code><span>${item.event_count} 个信号 · ${esc(item.latest_severity)}${item.latest_status?` · ${esc(statusLabel(item.latest_status))}`:""}</span></article>`).join(""):'<p class="aiops-situation-empty">未提供关联监控来源。</p>'}</div>`)
      +group("观测窗口","时间",observation([["首次观测",shell.fmt(detail.first_observed_at)],["最近观测",shell.fmt(detail.last_observed_at)]]))
      +group("执行与证据","质量",observation([["Agent 执行",execution.label],["诊断质量",evidence.label],["证据覆盖",evidence.coverage]]))
      +group("代表信号",signals.length,`<div class="aiops-situation-signal-list">${signals.length?signals.map(item=>`<article class="aiops-situation-signal"><strong>${esc(item.event_class)} · ${esc(item.severity)}</strong><p>${esc(item.summary || statusLabel(item.normalized_status))}</p><time>${esc(shell.fmt(item.occurred_at))}</time></article>`).join(""):'<p class="aiops-situation-empty">未提供关联信号。</p>'}</div>`);
  }
  function renderMode() {
    text("situation-mode-title",state.continuation?"继续深入诊断":"自动诊断结果");
    text("situation-mode-description",state.continuation?ready()?"选择 Agent，并说明对现有结果的具体疑问。":"选择 Agent，从现有告警情境继续诊断。":"阅读 Agent 结论，并与左侧原始事实持续对照。");
    const results=drawer.querySelector("[data-situation-results]"),continuation=drawer.querySelector('[role="tab"][data-situation-continue]');
    results.classList.toggle("active",!state.continuation);results.setAttribute("aria-selected",String(!state.continuation));
    continuation.classList.toggle("active",state.continuation);continuation.setAttribute("aria-selected",String(state.continuation));continuation.disabled=!canContinue();continuation.textContent=state.conversation?"进入深入诊断":"继续调查";
  }
  function renderResult() {
    const execution=executionStatus(),evidence=quality();
    const title=state.conversation?"已有深入诊断会话":ready()?"对结论或证据存疑？":inProgress()?"Agent 正在形成自动诊断":"自动诊断未形成可用结果";
    const description=state.conversation?"直接返回此前从本告警发起的会话，继续查看或追问。":ready()?"带上具体问题继续调查；完整自动诊断会随会话继承。":inProgress()?"运行结束后即可继续深入调查。":"可从当前告警情境和监控证据继续诊断。";
    body.innerHTML=`<section class="aiops-situation-result-view"><header class="aiops-situation-result-summary"><div><h4>Agent 自动诊断结果</h4><p>执行完成度与证据质量分别展示，证据缺口不会改变执行结果。</p></div><div class="aiops-diagnosis-statuses">${badge(execution.label,execution.tone)}${badge(evidence.label,evidence.tone)}${badge(evidence.coverage,evidence.tone)}</div></header><div class="aiops-situation-diagnosis"><div class="aiops-turn-result">${KBotAIOpsTurnResult.html(state.result?.payload,emptyText())}</div></div><div class="aiops-situation-handoff${inProgress()&&!state.conversation?" waiting":""}"><div><h4>${title}</h4><p>${description}</p></div><button class="primary-button" type="button" data-situation-continue${canContinue()?"":" disabled"}>${state.lookupLoading?"正在检查会话…":state.conversation?"进入深入诊断会话":"继续深入诊断"}</button></div></section>`;
    KBotAIOpsTurnResult.mount(body);
  }
  function updateContinuationControls() {
    const form=body.querySelector("form");if(!form)return;
    form.elements.agent_id.disabled=state.loadingAgents || state.submitting;
    form.elements.message.disabled=state.submitting;
    form.querySelector('[type="submit"]').disabled=state.loadingAgents || state.submitting || !state.agentId || !state.message.trim();
    form.querySelector('[type="submit"]').textContent=state.submitting?"正在创建会话…":"进入诊断对话";
    form.querySelector('[data-situation-results]').disabled=state.submitting;
  }
  function renderContinuation() {
    const agents=state.agents.filter(agent=>agent.status==="ACTIVE");
    body.innerHTML=`<form class="quick-create-form aiops-continuation-form"><section class="aiops-continuation-context"><span>继承</span><div><strong>${ready()?"完整自动诊断结果与告警证据":"告警情境与监控证据"}</strong><p>${ready()?"当前告警、监控来源、代表信号和本次 Agent 结论会一起交给新会话。":"自动诊断未形成最终结果，新会话将从当前告警、监控来源和代表信号继续。"}</p></div></section><label>诊断 Agent<select name="agent_id" required><option value="" disabled>${state.loadingAgents?"正在读取可用 Agent…":agents.length?"请选择 Agent":"暂无已启用 Agent"}</option>${agents.map(agent=>`<option value="${esc(agent.agent_id)}">${esc(agent.display_name)} · ${rows(agent.allowed_business_system_ids).length} 个业务系统</option>`).join("")}</select><small>Agent 会根据告警或诊断来源和其业务系统授权范围解析诊断资源。</small></label><label>${ready()?"对现有结果的疑问":"补充诊断问题"}<textarea name="message" rows="8" maxlength="32000" required></textarea><small>${ready()?"请指出存疑的结论、证据或状态差异，问题会与完整自动诊断一起进入会话。":"请补充希望 Agent 重点核验的问题，现有告警情境和监控证据会一起进入会话。"}</small></label><footer class="aiops-continuation-actions"><span>告警事实会继续保留在左侧供核对。</span><div><button class="secondary-button" type="button" data-situation-results>返回结果</button><button class="primary-button" type="submit">进入诊断对话</button></div></footer></form>`;
    const form=body.querySelector("form");form.elements.agent_id.value=state.agentId;form.elements.message.value=state.message;
    form.oninput=()=>{state.agentId=form.elements.agent_id.value;state.message=form.elements.message.value;updateContinuationControls();};
    form.onchange=form.oninput;
    form.onsubmit=startContinuation;updateContinuationControls();
  }
  async function openContinuation() {
    if(!canContinue() || state.continuation)return;
    if(state.conversation){location.href=`./chat.html?conversation=${encodeURIComponent(state.conversation.conversation_id)}`;return;}
    state.continuation=true;
    if(!state.message)state.message=ready()?"我对自动诊断结果仍有疑问，请继续核验相关证据，并说明需要重点确认的状态差异。":"自动诊断未形成可用结果，请基于告警情境和监控证据继续诊断，并说明自动诊断未完成的可能原因。";
    const id=state.activeId;
    state.loadingAgents=!state.agents.length;renderMode();renderContinuation();
    if(!state.loadingAgents)return;
    try{const agents=await request("/agents");if(id===state.activeId)state.agents=rows(agents);}
    catch(error){if(id===state.activeId)shell.toast(error.message || "无法读取可用 Agent");}
    finally{if(id===state.activeId){state.loadingAgents=false;if(state.continuation)renderContinuation();}}
  }
  async function startContinuation(event) {
    event.preventDefault();
    if(state.submitting || !state.selected || !state.agentId || !state.message.trim())return;
    const source=ready()&&state.run?{source_run_id:state.run.ops_run_id}:{source_situation_id:state.selected.situation_id};
    state.submitting=true;updateContinuationControls();
    try{
      const receipt=await request("/conversations",{method:"POST",headers:{"Idempotency-Key":KBotAIOpsAuth.uuid()},body:JSON.stringify({agent_id:state.agentId,...source,content:[{content_type:"TEXT",text:state.message.trim()}]})});
      location.href=`./chat.html?conversation=${encodeURIComponent(receipt.conversation_id)}`;
    }catch(error){shell.toast(error.message || "无法继续深入诊断");}
    finally{state.submitting=false;updateContinuationControls();}
  }
  function close() {
    clearTimeout(state.timer);state.activeId="";state.selected=null;state.run=null;state.result=null;state.continuation=false;state.agentId="";state.message="";state.conversation=null;state.lookupLoading=false;state.lookupFailed=false;state.loadingAgents=false;state.submitting=false;
    if(drawer.open)drawer.close();body.innerHTML="";KBotAIOpsTurnResult.mount(body);
  }
  function scheduleRefresh(item,delay) {clearTimeout(state.timer);state.timer=setTimeout(()=>open(item,{background:true}),delay);}
  async function open(item,{background=false}={}) {
    const id=String(item.situation_id);
    if(!background){close();state.activeId=id;}
    if(state.activeId!==id)return;
    try{
      const detail=await request(`/situations/${encodeURIComponent(id)}`);if(state.activeId!==id)return;
      state.selected=detail;
      if(!drawer.open)drawer.showModal();
      if(!background){state.lookupLoading=true;renderFacts();renderMode();renderResult();}
      let run=null,result=null;
      if(detail.run_ids?.[0]){
        run=await request(`/runs/${encodeURIComponent(detail.run_ids[0])}`);
        try{result=await request(`/runs/${encodeURIComponent(run.ops_run_id)}/result`);}
        catch(error){if(![404,409].includes(error.status))throw error;}
      }
      if(state.activeId!==id)return;state.run=run;state.result=result;
      if(!background){
        try{const conversations=await request(`/conversations?situation_id=${encodeURIComponent(id)}&limit=1`);if(state.activeId!==id)return;state.conversation=rows(conversations)[0] || null;}
        catch(error){if(state.activeId!==id)return;state.lookupFailed=true;shell.toast(error.message || "无法检查关联的深入诊断会话");}
        finally{if(state.activeId===id)state.lookupLoading=false;}
      }
      renderFacts();renderMode();if(!state.continuation)renderResult();
      if(!run || !terminal.has(run.status))scheduleRefresh(item,3000);
      else if(!["RESOLVED","CLOSED"].includes(detail.status))scheduleRefresh(item,15000);
    }catch(error){if(background){if(state.activeId===id)scheduleRefresh(item,5000);}else shell.toast(error.message || "无法读取告警诊断详情");}
  }
  async function load() {
    const loading=document.getElementById("situations-loading"),table=document.getElementById("situations-table");loading.hidden=false;table.hidden=true;
    try{state.items=rows((await request("/situations?limit=100")).items);}
    catch(error){shell.toast(error.message || "无法读取告警诊断");}
    finally{
      text("situations-count",`${state.items.length} 条`);
      document.getElementById("situations-body").innerHTML=state.items.length?state.items.map(item=>`<tr><td><strong class="table-primary">${esc(item.title)}</strong><small class="table-secondary">${esc(item.summary || item.situation_type)}</small></td><td>${esc(item.severity)}</td><td>${badge(statusLabel(item.status),statusClass(item.status))}</td><td>${item.event_count}</td><td>${esc(shell.fmt(item.last_observed_at))}</td><td><button class="row-action" type="button" data-situation-id="${esc(item.situation_id)}">查看诊断</button></td></tr>`).join(""):'<tr><td colspan="6" class="empty-cell">当前业务空间还没有告警诊断记录。</td></tr>';
      loading.hidden=true;table.hidden=false;
    }
  }
  shell.ready.then(()=>{
    document.getElementById("refresh-workspace").onclick=load;
    document.getElementById("situations-body").onclick=event=>{const trigger=event.target.closest("[data-situation-id]");if(trigger){const item=state.items.find(item=>item.situation_id===trigger.dataset.situationId);if(item)void open(item);}};
    drawer.onclick=event=>{if(event.target.closest("[data-close-situation]"))close();else if(event.target.closest("[data-situation-results]")){state.continuation=false;renderMode();renderResult();}else if(event.target.closest("[data-situation-continue]"))void openContinuation();};
    drawer.addEventListener("close",()=>{if(state.selected && !drawer.open)close();});window.addEventListener("pagehide",close);
    return load();
  }).catch(error=>shell.toast(error.message));
})();
