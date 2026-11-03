import { mountTimePicker, normalizeTime } from "../../shared/time-picker.js?v=20261102_1";

(function () {
  "use strict";
  if (document.body.dataset.page === "inspection-plan-detail") {location.replace(`./inspection-plans.html${location.search}`);return;}
  if (document.body.dataset.page !== "inspection-plans") return;
  const shell = globalThis.KBotAIOpsShell, esc = shell.escape;
  const api = "/api/v1/apps/aiops";
  const request = (path, init) => KBotAIOpsAuth.request(api + path, init);
  const rows = value => Array.isArray(value) ? value : [];
  const drawer = document.getElementById("inspection-plan-dialog"), form = document.getElementById("inspection-plan-form");
  const timePicker = mountTimePicker(form.querySelector(".time-picker"));
  const systemFilter = new URLSearchParams(location.search).get("businessSystemId") || "";
  const state = {systems:[],scopeRows:[],scopeLoading:false,scopeError:"",scopeSequence:0,plans:[],agents:[],templates:[],fires:[],editing:null,busy:false,loading:false,sequence:0,editSequence:0,confirm:null};
  const weekdays = [{value:"1",label:"一"},{value:"2",label:"二"},{value:"3",label:"三"},{value:"4",label:"四"},{value:"5",label:"五"},{value:"6",label:"六"},{value:"0",label:"日"}];
  const activeAgents = () => state.agents.filter(item => item.status === "ACTIVE");
  const eligibleAgents = () => activeAgents().filter(agent => rows(agent.allowed_business_system_ids).includes(form.elements.business_system_id.value) && rows(agent.capabilities).includes("ACTIVE_DIAGNOSIS"));
  const visiblePlans = () => state.plans.filter(plan => !systemFilter || plan.business_system_id === systemFilter);
  function populateAgents() {
    const selected = form.elements.agent_id.value;
    form.elements.agent_id.innerHTML = '<option value="">请选择已启用 Agent</option>' + eligibleAgents().map(agent => `<option value="${esc(agent.agent_id)}">${esc(agent.display_name)}</option>`).join("");
    form.elements.agent_id.value = eligibleAgents().some(agent => agent.agent_id === selected) ? selected : "";
  }
  async function loadSystemScope() {
    const sequence = ++state.scopeSequence;
    state.scopeRows = [];state.scopeError = "";state.scopeLoading = Boolean(form.elements.business_system_id.value);
    const loading = document.getElementById("inspection-system-loading"), error = document.getElementById("inspection-system-error"), list = document.getElementById("inspection-system-resources");
    loading.hidden = !state.scopeLoading;error.hidden = true;list.innerHTML = "";populateAgents();updateForm();
    if (!state.scopeLoading) return;
    try {
      const detail = await request(`/business-systems/${encodeURIComponent(form.elements.business_system_id.value)}`);
      if (sequence !== state.scopeSequence) return;
      state.scopeRows = rows(detail.resources).filter(row => row.resource.resource_type_key === "DATABASE").map(row => ({
        resource_id:row.resource.resource_id,display_name:row.resource.display_name,target_id:row.resource.database?.target_id,
        state:row.status !== "ACTIVE" || row.resource.lifecycle_status !== "ACTIVE" ? "已排除：资源或关联停用" : !row.resource.database?.target_id ? "配置不完整，不能遗漏" : "纳入巡检"
      }));
      list.innerHTML = state.scopeRows.length ? state.scopeRows.map(row => `<li>${esc(row.display_name)} · ${esc(row.state)}</li>`).join("") : "<li>该系统没有数据库资源，不能创建巡检计划。</li>";
    } catch(cause) {if (sequence === state.scopeSequence) {state.scopeError = "无法读取系统数据库范围，请重试。";error.hidden = false;error.textContent = state.scopeError;shell.toast(cause.message);}}
    finally {if (sequence === state.scopeSequence) {state.scopeLoading = false;loading.hidden = true;updateForm();}}
  }
  const activeTemplates = () => state.templates.filter(item => item.status === "ACTIVE");
  function cronExpression() {
    if (form.elements.schedule_type.value === "CRON") return form.elements.cron_expression.value.trim();
    if (normalizeTime(form.elements.execution_time.value) !== form.elements.execution_time.value || !form.elements.execution_time.value) return "";
    const [hour,minute] = form.elements.execution_time.value.split(":");
    if (!hour || !minute) return "";
    const days = weekdays.filter(item => [...form.querySelectorAll('[name="weekdays"]:checked')].some(input => input.value === item.value)).map(item => item.value).join(",");
    return form.elements.schedule_type.value === "DAILY" ? `${Number(minute)} ${Number(hour)} * * *` : days ? `${Number(minute)} ${Number(hour)} * * ${days}` : "";
  }
  function ready() {
    const timeout = Number(form.elements.timeout_seconds.value);
    return Boolean(!state.scopeLoading && !state.scopeError && state.scopeRows.some(row => row.state === "纳入巡检") && !state.scopeRows.some(row => row.state === "配置不完整，不能遗漏") && form.elements.business_system_id.value && eligibleAgents().some(agent => agent.agent_id === form.elements.agent_id.value) && form.elements.display_name.value.trim() && form.elements.agent_id.value && form.elements.timezone.value && form.elements.inspection_template_id.value && Number.isInteger(timeout) && timeout >= 1 && timeout <= 86400 && cronExpression());
  }
  function updateForm() {
    const type = form.elements.schedule_type.value;
    document.getElementById("inspection-plan-weekdays").hidden = type !== "WEEKLY";
    document.getElementById("inspection-plan-time").hidden = type === "CRON";
    document.getElementById("inspection-plan-cron").hidden = type !== "CRON";
    form.elements.execution_time.required = type !== "CRON";form.elements.cron_expression.required = type === "CRON";
    document.getElementById("inspection-plan-type-help").hidden = !state.editing;
    const time = form.elements.execution_time.value, zone = form.elements.timezone.value;
    timePicker.sync();if (type === "CRON") timePicker.close();
    const selectedDays = weekdays.filter(item => form.querySelector(`[name="weekdays"][value="${item.value}"]`).checked).map(item => `周${item.label}`).join("、");
    document.getElementById("inspection-schedule-summary").textContent = type === "DAILY" ? `每天 ${time}（${zone}）` : type === "CRON" ? `Cron ${form.elements.cron_expression.value || "尚未填写"}（${zone}）` : `${selectedDays || "尚未选择执行日"} ${time}（${zone}）`;
    const template = state.templates.find(item => item.inspection_template_id === form.elements.inspection_template_id.value), summary = document.getElementById("inspection-template-summary");
    summary.hidden = !template;document.getElementById("inspection-template-empty").hidden = Boolean(template);
    document.getElementById("inspection-upgrade-label").hidden = !state.editing;
    if (template) summary.innerHTML = `<span>${state.editing ? "模板当前可用版本" : "当前模板"}</span><strong>${esc(template.display_name)} · v${esc(template.version_no)}</strong><small>${state.editing ? "默认保留原绑定版本；勾选升级选项才使用所选模板当前版本。" : `${rows(template.selected_check_ids).length} 个检查项；计划创建时会冻结当前模板版本。`}</small>`;
    drawer.querySelectorAll("[data-close-plan]").forEach(button => {button.disabled = state.busy;});
    form.querySelectorAll('[name="schedule_type"]').forEach(input => {input.disabled = Boolean(state.editing);});
    document.getElementById("save-inspection-plan").disabled = state.busy || !ready();
    document.getElementById("save-inspection-plan").textContent = state.busy ? "保存中…" : state.editing ? "保存修改" : "创建并启用";
  }
  function statusPresentation(status) {
    return ({ACTIVE:{label:"已启用",tone:"status-success"},PAUSED:{label:"已暂停",tone:"status-warning"},DISABLED:{label:"已停用",tone:"status-muted"}})[status] || {label:status,tone:"status-muted"};
  }
  function render() {
    const notice = document.getElementById("inspection-plan-notice");
    notice.hidden = state.loading || Boolean(activeAgents().length && activeTemplates().length);
    notice.innerHTML = !activeAgents().length ? "<strong>暂无可用 Agent</strong><span>请先在 Agent 管理中启用至少一个已授权业务系统的 Agent，再创建巡检计划。</span>" : "<strong>暂无巡检内容模板</strong><span>请先在模板中心创建并冻结巡检内容模板，再创建巡检计划。</span>";
    document.getElementById("inspection-plan-count").textContent = `${visiblePlans().length} 个`;
    document.getElementById("inspection-plan-catalog").innerHTML = `<button id="create-inspection-plan" class="management-catalog-add-card" type="button"${state.loading || !activeAgents().length || !activeTemplates().length ? " disabled" : ""}><span class="management-catalog-add-mark" aria-hidden="true">+</span><strong>新建巡检计划</strong><small>选择业务系统、内容模板与执行 Agent</small></button>` + visiblePlans().map(plan => {
      const status = statusPresentation(plan.status), fire = state.fires.find(item => item.plan_id === plan.plan_id);
      const action = (name,label,kind="row-action") => `<button class="${kind}" type="button" data-plan-action="${name}"${state.busy ? " disabled" : ""}>${label}</button>`;
      return `<article class="management-catalog-card" data-plan-id="${esc(plan.plan_id)}"><header><div><p class="eyebrow">INSPECTION PLAN</p><h3>${esc(plan.display_name)}</h3></div><span class="status-pill ${status.tone}">${status.label}</span></header><p>最近执行：${fire ? esc(shell.fmt(fire.scheduled_at)) : "尚未执行"}<br />下次执行：${esc(shell.fmt(plan.next_run_at))}</p><dl><div><dt>业务系统</dt><dd>${esc(state.systems.find(system => system.business_system_id === plan.business_system_id)?.display_name || plan.business_system_id)}</dd></div><div><dt>执行 Agent</dt><dd>${esc(state.agents.find(item => item.agent_id === plan.agent_id)?.display_name || "Agent 已不可用")}</dd></div><div><dt>执行周期</dt><dd>${esc(({DAILY:"每天",WEEKLY:"每周",CRON:"高级规则"})[plan.schedule_type] || plan.schedule_type)} · ${esc(plan.timezone)}</dd></div></dl><footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions">${fire ? action("reports","报告") : ""}${action("edit","编辑")}${plan.status === "ACTIVE" ? action("pause","暂停") : action("activate","启用")}</div><div class="management-catalog-overflow"><button class="row-action management-catalog-more" type="button" aria-haspopup="menu" aria-expanded="false"><span>更多</span><span aria-hidden="true">⌄</span></button><div class="management-catalog-menu" role="menu" hidden>${plan.status === "DISABLED" ? action("delete","删除计划","danger-row-action") : action("disable","停用计划","danger-row-action")}</div></div></footer></article>`;
    }).join("");
  }
  async function load() {
    const sequence = ++state.sequence;state.loading = true;
    document.getElementById("inspection-plans-loading").hidden = false;document.getElementById("inspection-plan-list").hidden = true;
    try {
      const [plans,agents,templates,fires,systems] = await Promise.all([request("/inspection-plans?limit=100"),request("/agents"),request("/inspection-templates"),request("/inspection-fires?limit=100"),request("/business-systems?limit=200")]);
      if (sequence !== state.sequence) return;
      state.plans = rows(plans.items);state.agents = rows(agents);state.templates = rows(templates);state.fires = rows(fires.items);state.systems = rows(systems.items);
    } catch (error) {if (sequence === state.sequence) shell.toast(error.message || "无法读取巡检计划");}
    finally {if (sequence === state.sequence) {state.loading = false;render();document.getElementById("inspection-plans-loading").hidden = true;document.getElementById("inspection-plan-list").hidden = false;}}
  }
  function populate(plan) {
    form.reset();state.editing = plan || null;
    form.elements.business_system_id.innerHTML = '<option value="">请选择业务系统</option>' + state.systems.map(system => `<option value="${esc(system.business_system_id)}"${system.status !== "ENABLED" ? " disabled" : ""}>${esc(system.display_name)}</option>`).join("");
    form.elements.business_system_id.disabled = Boolean(plan);
    form.elements.business_system_id.value = plan?.business_system_id || systemFilter;
    populateAgents();
    form.elements.inspection_template_id.innerHTML = '<option value="">请选择巡检内容模板</option>' + state.templates.map(item => `<option value="${esc(item.inspection_template_id)}"${item.status !== "ACTIVE" ? " disabled" : ""}>${esc(item.display_name)} · v${esc(item.version_no)} · ${rows(item.selected_check_ids).length} 项</option>`).join("");
    if (plan) {
      for (const key of ["display_name","business_system_id","agent_id","schedule_type","cron_expression","timezone","inspection_template_id","timeout_seconds","overlap_policy","misfire_policy"]) form.elements[key].value = plan[key] ?? "";
      const parts = String(plan.cron_expression).split(/\s+/);
      form.elements.execution_time.value = parts.length === 5 ? `${String(Number(parts[1])).padStart(2,"0")}:${String(Number(parts[0])).padStart(2,"0")}` : "02:00";
      if (plan.schedule_type === "WEEKLY") form.querySelectorAll('[name="weekdays"]').forEach(input => {input.checked = parts[4]?.split(",").includes(input.value);});
    } else if (activeTemplates().length === 1) form.elements.inspection_template_id.value = activeTemplates()[0].inspection_template_id;
    document.getElementById("inspection-plan-title").textContent = plan ? "编辑巡检计划" : "新建巡检计划";
    void loadSystemScope();updateForm();if (!drawer.open) drawer.showModal();drawer.scrollTop = 0;
  }
  function openCreate() {
    if (state.busy) return;
    ++state.editSequence;
    if (!activeAgents().length) {shell.toast("请先在 Agent 管理中启用至少一个已授权业务系统的 Agent。");return;}
    if (!activeTemplates().length) {shell.toast("请先在模板中心创建巡检内容模板。");return;}
    populate(null);
  }
  async function openEdit(planId) {
    if (state.busy) return;
    const sequence = ++state.editSequence;
    try {const plan = await request(`/inspection-plans/${encodeURIComponent(planId)}`);if (sequence === state.editSequence) populate(plan);}
    catch (error) {if (sequence === state.editSequence) shell.toast(error.message || "无法读取巡检计划详情");}
  }
  function close() {if (state.busy) return;++state.editSequence;if (drawer.open) drawer.close();state.editing = null;}
  async function save(event) {
    event.preventDefault();if (state.busy || !ready()) return;
    const current = state.editing, payload = {display_name:form.elements.display_name.value.trim(),agent_id:form.elements.agent_id.value,cron_expression:cronExpression(),timezone:form.elements.timezone.value,timeout_seconds:Number(form.elements.timeout_seconds.value),overlap_policy:form.elements.overlap_policy.value,misfire_policy:form.elements.misfire_policy.value};
    if (!current) {payload.business_system_id = form.elements.business_system_id.value;payload.schedule_type = form.elements.schedule_type.value;payload.inspection_template_id = form.elements.inspection_template_id.value;}
    else if (form.elements.inspection_template_id.value !== current.inspection_template_id || form.elements.upgrade_template_version.checked) payload.inspection_template_id = form.elements.inspection_template_id.value;
    state.busy = true;updateForm();
    try {
      await request(current ? `/inspection-plans/${encodeURIComponent(current.plan_id)}` : "/inspection-plans",{method:current ? "PATCH" : "POST",headers:{"Content-Type":"application/json",...(current ? {"If-Match":`"rv-${current.row_version}"`} : {"Idempotency-Key":KBotAIOpsAuth.uuid()})},body:JSON.stringify(payload)});
      state.busy = false;close();shell.toast(current ? "巡检计划已更新" : "巡检计划已创建并启用");await load();
    } catch (error) {shell.toast(error.message || (current ? "无法更新巡检计划" : "无法创建巡检计划"));}
    finally {state.busy = false;updateForm();}
  }
  function settleConfirmation(value) {const resolve = state.confirm;state.confirm = null;document.getElementById("inspection-plan-confirm").close();resolve?.(value);}
  async function command(plan,action) {
    if (state.busy || state.confirm) return;
    const copy = ({activate:{title:"启用巡检计划？",message:"系统将重新计算下一次执行时间并恢复调度。",label:"启用",tone:"warning"},pause:{title:"暂停巡检计划？",message:"暂停后不会创建新的巡检批次，既有运行记录和报告不受影响。",label:"暂停",tone:"warning"},disable:{title:"停用巡检计划？",message:"停用后计划不会继续调度，既有运行记录和报告不受影响。",label:"停用",tone:"danger"},delete:{title:"删除巡检计划？",message:`将永久删除“${plan.display_name}”。只有已停用且从未产生执行记录的计划可以删除。`,label:"删除计划",tone:"danger"}})[action];
    if (!copy) return;
    const dialog = document.getElementById("inspection-plan-confirm");dialog.className = `confirm-dialog confirm-dialog--${copy.tone}`;
    document.getElementById("inspection-plan-confirm-title").textContent = copy.title;document.getElementById("inspection-plan-confirm-message").textContent = copy.message;
    const button = dialog.querySelector("[data-confirm-plan]");button.className = copy.tone === "danger" ? "danger-button" : "primary-button";button.textContent = copy.label;
    dialog.showModal();dialog.querySelector("[data-cancel-plan-confirm]").focus();
    if (!await new Promise(resolve => {state.confirm = resolve;})) return;
    state.busy = true;render();
    try {
      await request(`/inspection-plans/${encodeURIComponent(plan.plan_id)}${action === "delete" ? "" : `/${action}`}`,{method:action === "delete" ? "DELETE" : "POST",headers:{"Idempotency-Key":KBotAIOpsAuth.uuid(),"If-Match":`"rv-${plan.row_version}"`}});
      shell.toast(action === "delete" ? "巡检计划已删除" : `巡检计划已${copy.label}`);await load();
    } catch (error) {shell.toast(error.message || `无法${copy.label}巡检计划`);}
    finally {state.busy = false;render();}
  }
  function closeMenus() {document.querySelectorAll(".management-catalog-menu").forEach(menu => {menu.hidden = true;});document.querySelectorAll(".management-catalog-more").forEach(button => button.setAttribute("aria-expanded","false"));}
  shell.ready.then(async access => {
    if (!rows(access.permissions).includes("aiops:plan_manage")) {location.replace("./inspections.html");return;}
    document.querySelector("[data-inspection-reports]").onclick = () => {location.href = "./inspections.html";};
    document.getElementById("refresh-inspection-plans").onclick = load;
    form.onsubmit = save;form.oninput = updateForm;form.onchange = event => {if (event.target.name === "business_system_id") void loadSystemScope();updateForm();};
    drawer.onclick = event => {if (event.target.closest("[data-close-plan]")) close();};
    drawer.addEventListener("cancel",event => {if (state.busy) event.preventDefault();});drawer.addEventListener("close",close);
    document.getElementById("inspection-plan-catalog").onclick = event => {
      if (event.target.closest("#create-inspection-plan")) {openCreate();return;}
      const more = event.target.closest(".management-catalog-more");
      if (more) {const menu = more.nextElementSibling, open = menu.hidden;closeMenus();menu.hidden = !open;more.setAttribute("aria-expanded",String(open));return;}
      const button = event.target.closest("[data-plan-action]"), plan = state.plans.find(item => item.plan_id === button?.closest("[data-plan-id]")?.dataset.planId);
      if (!plan || state.busy) return;closeMenus();
      if (button.dataset.planAction === "edit") openEdit(plan.plan_id);
      else if (button.dataset.planAction === "reports") location.href = `./inspections.html?planId=${encodeURIComponent(plan.plan_id)}`;
      else command(plan,button.dataset.planAction);
    };
    document.addEventListener("pointerdown",event => {if (!event.target.closest(".management-catalog-overflow")) closeMenus();});
    document.addEventListener("keydown",event => {if (event.key === "Escape") closeMenus();});
    const confirm = document.getElementById("inspection-plan-confirm");confirm.querySelector("[data-cancel-plan-confirm]").onclick = () => settleConfirmation(false);confirm.querySelector("[data-confirm-plan]").onclick = () => settleConfirmation(true);confirm.addEventListener("cancel",event => {event.preventDefault();settleConfirmation(false);});
    await load();const planId = new URLSearchParams(location.search).get("id");if (planId) await openEdit(planId);else if (new URLSearchParams(location.search).get("create") === "1") openCreate();
  }).catch(error => shell.toast(error.message));
})();
