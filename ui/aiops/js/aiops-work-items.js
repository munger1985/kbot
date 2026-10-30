(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/work-items";
  const rootApi = "/api/v1/apps/aiops";
  const groupApi = "/api/v1/apps/aiops/responsibility-groups";
  const shell = globalThis.KBotAIOpsShell;
  const state = {
    access: null, cursor: null, filters: { visibility: "SCOPE" }, item: null,
    items: new Map(), processingItem: null, processingContext: null, assignmentItem: null,
    scopedTargets: [], people: [], handling: false, assigning: false, suppressReset: false, listRequest: 0,
  };
  const labels = {
    PENDING_TRIAGE: "待研判", OPEN: "待处理", IN_PROGRESS: "处理中",
    WAITING: "等待中", PENDING_VERIFICATION: "处理完成", RESOLVED: "已解决",
    CLOSED: "已关闭", CANCELLED: "已取消", TRIAGE: "研判",
    DIAGNOSIS: "诊断", REMEDIATION: "处置", APPROVAL: "审批",
    EXECUTION: "执行", VERIFICATION: "验证", P1: "P1 紧急",
    P2: "P2 高", P3: "P3 中", P4: "P4 低",
    FIXED: "已修复", MITIGATED: "已缓解", OBSERVED_STABLE: "观察稳定",
    ACCEPTED_RISK: "风险已接受", FALSE_POSITIVE: "误报", DUPLICATE: "重复事项",
    RECOVERY_DRILL: "恢复演练", SCHEDULE: "策略计划",
    CREATED: "创建工作项", ASSIGNED: "更新责任分派", STATUS_CHANGED: "更新处理状态",
    CLAIMED: "开始处理", RETURNED_TO_GROUP: "退回责任组", DBA_COMPLETED: "标记处理完成",
    OCCURRENCE_MERGED: "归并新发生记录", VERIFICATION_COMPLETED: "完成验证运行",
  };
  const text = (value) => labels[value] || value || "—";
  const values = (items) => Array.isArray(items) ? items : [];
  const jsonBody = (value) => ({ method: "POST", body: JSON.stringify(value) });
  const query = (values) => {
    const params = new URLSearchParams();
    Object.entries(values).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== "" && value !== false) params.set(key, String(value));
    });
    return params.toString() ? `?${params}` : "";
  };
  function processingAlertHtml(item, situation) {
    const occurrence = values(item.occurrences)[0];
    const finding = occurrence?.finding || {};
    const object = finding.object_ref || {};
    const objectLabel = object.object_name || object.sql_id || object.object_kind || "";
    const facts = occurrence ? [
      ["发现类型", occurrence.finding_type],
      ["确认度", occurrence.confirmation],
      ["关联对象", objectLabel],
      ["观测时间", shell.fmt(occurrence.observed_at)],
    ].filter((row) => row[0] !== "关联对象" || row[1]).map((row) => `<div><dt>${shell.escape(row[0])}</dt><dd>${shell.escape(row[1] || "")}</dd></div>`).join("") : "";
    const signals = values(situation?.signal_events).map((signal) => `<li><strong>${shell.escape(signal.event_class)} · ${shell.escape(signal.severity)}</strong><span>${shell.escape(signal.summary || signal.normalized_status)}</span><time>${shell.escape(shell.fmt(signal.occurred_at))}</time></li>`).join("");
    const evidence = values(occurrence?.evidence_refs).map((ref) => `<code>${shell.escape(ref)}</code>`).join("");
    return `<p class="work-item-context-summary">${shell.escape(finding.impact || item.summary)}</p>${facts ? `<dl class="work-item-context-facts">${facts}</dl>` : ""}${signals ? `<ul class="work-item-signal-list">${signals}</ul>` : ""}${evidence ? `<div class="work-item-context-evidence">${evidence}</div>` : ""}`;
  }
  async function loadProcessingContext(item) {
    const diagnosis = document.getElementById("work-item-agent-diagnosis");
    const renderer = globalThis.KBotAIOpsTurnResult;
    document.getElementById("work-item-alert-context").innerHTML = processingAlertHtml(item,null);
    diagnosis.innerHTML = renderer.html(null,"正在读取 Agent 初步诊断…");
    document.getElementById("work-item-continue-diagnosis").hidden = true;
    const severity = document.getElementById("work-item-processing-severity");
    const occurrence = values(item.occurrences)[0];
    severity.textContent = occurrence?.severity || item.severity;
    severity.className = `status-pill ${occurrence?.severity === "CRITICAL" ? "status-info" : "status-warning"}`;
    const runId = occurrence?.ops_run_id || values(item.links).find((link) => ["RUN", "VERIFICATION"].includes(link.resource_kind))?.resource_id;
    const situationId = occurrence?.situation_id || values(item.links).find((link) => link.resource_kind === "SITUATION")?.resource_id;
    const [situationResult, runResult, diagnosisResult] = await Promise.allSettled([
      situationId ? KBotAIOpsAuth.request(`${rootApi}/situations/${encodeURIComponent(situationId)}`) : Promise.resolve(null),
      runId ? KBotAIOpsAuth.request(`${rootApi}/runs/${encodeURIComponent(runId)}`) : Promise.resolve(null),
      runId ? KBotAIOpsAuth.request(`${rootApi}/runs/${encodeURIComponent(runId)}/result`) : Promise.resolve(null),
    ]);
    if (state.processingItem !== item) return;
    const situation = situationResult.status === "fulfilled" ? situationResult.value : null;
    const run = runResult.status === "fulfilled" ? runResult.value : null;
    const result = diagnosisResult.status === "fulfilled" ? diagnosisResult.value : null;
    const effectiveRunId = run?.ops_run_id || occurrence?.ops_run_id;
    const source = result?.final_artifact && effectiveRunId
      ? { sourceType: "RUN", sourceId: effectiveRunId }
      : occurrence?.situation_id
        ? { sourceType: "SITUATION", sourceId: occurrence.situation_id }
        : effectiveRunId ? { sourceType: "RUN", sourceId: effectiveRunId } : null;
    state.processingContext = { situation, run, result, source };
    document.getElementById("work-item-alert-context").innerHTML = processingAlertHtml(item, situation);
    const emptyText = runId && runResult.status === "rejected" && diagnosisResult.status === "rejected"
      ? "关联的 Agent 初步诊断暂时无法读取。"
      : run && !result ? `自动诊断运行状态：${run.status}，尚未形成可展示结果。` : "该工作项没有关联可展示的 Agent 初步诊断。";
    diagnosis.innerHTML = renderer.html(result?.payload,emptyText);
    renderer.mount(diagnosis);
    document.getElementById("work-item-continue-diagnosis").hidden = !source;
  }
  function continueProcessingDiagnosis() {
    const item = state.processingItem;
    const source = state.processingContext?.source;
    if (!item || !source) return shell.toast("该工作项没有可继承的诊断上下文");
    location.href = `./chat.html?${new URLSearchParams({targetId:item.target_id,...source})}`;
  }
  function renderQueues(summary) {
    const definitions = [
      ["my_open", "我的待办", "mine", "当前分派给我的活动事项", "accent"],
      ["unassigned", "未分派", "unassigned", "等待责任组内 DBA 领取", ""],
      ["urgent", "P1 / P2", "urgent", "需要优先研判与处置", "warning"],
      ["overdue", "已超 SLA", "overdue", "解决时限已经超期", "danger"],
      ["pending_verification", "待验证", "verification", "等待复核处理结果", "info"],
    ];
    document.getElementById("work-item-queues").innerHTML = definitions.map(([key, label, filter, description, tone]) =>
      `<button class="operational-summary-card ${tone ? `is-${tone} ` : ""}${queueIsActive(filter) ? "is-active" : ""}" type="button" data-queue="${filter}" aria-pressed="${queueIsActive(filter)}"><header><span>${label}</span><strong>${Number(summary?.[key] || 0)}</strong></header><p>${description}</p></button>`
    ).join("");
    document.getElementById("work-item-overdue").setAttribute("aria-pressed",String(Boolean(state.filters.overdue)));
  }
  function queueIsActive(value) {
    const filters = state.filters;
    if (value === "mine") return Boolean(filters.assignee_user_id) && !filters.status && !filters.priority && !filters.unassigned && !filters.overdue;
    if (value === "unassigned") return Boolean(filters.unassigned) && !filters.status && !filters.priority && !filters.overdue;
    if (value === "urgent") return filters.priority === "P1,P2" && !filters.status && !filters.assignee_user_id && !filters.unassigned && !filters.overdue;
    if (value === "overdue") return Boolean(filters.overdue) && !filters.status && !filters.priority && !filters.assignee_user_id && !filters.unassigned;
    return filters.status === "PENDING_VERIFICATION" && !filters.priority && !filters.assignee_user_id && !filters.unassigned && !filters.overdue;
  }
  async function loadScopedTargets() {
    const [targetPage, groupPage] = await Promise.all([
      KBotAIOpsAuth.request(`${rootApi}/targets?limit=200`),
      KBotAIOpsAuth.request(groupApi),
    ]);
    const groups = values(groupPage?.items).filter((group) => group.status === "ACTIVE");
    const details = await Promise.all(groups.map((group) =>
      KBotAIOpsAuth.request(`${groupApi}/${encodeURIComponent(group.responsibility_group_id)}`)
    ));
    state.people = groups.length ? values((await KBotAIOpsAuth.request(`${groupApi}/${encodeURIComponent(groups[0].responsibility_group_id)}/member-candidates`)).items) : [];
    const userGroupIds = new Set(details.filter((group) =>
      values(group.members).some((member) => member.status === "ACTIVE"
        && String(member.user_id) === String(state.access.user_id))
    ).map((group) => String(group.responsibility_group_id)));
    state.scopedTargets = values(targetPage?.items).filter((target) =>
      target.status === "ENABLED"
      && target.responsibility_group_id
      && userGroupIds.has(String(target.responsibility_group_id)));
    const select = document.getElementById("work-item-target-filter");
    select.innerHTML = '<option value="">责任范围内全部实例</option>'
      + state.scopedTargets.map((target) => `<option value="${shell.escape(target.target_id)}">${shell.escape(target.display_name || shell.short(target.target_id))} · ${shell.escape(target.db_type || "")}</option>`).join("");
  }
  function listRow(item) {
    const person = state.people.find(person=>String(person.user_id)===String(item.assignee_user_id));
    const assignee = person?.display_name || person?.username || item.responsibility_group_name || (item.assignee_user_id ? "责任人已不可用" : "未分派");
    const detail = `./work-item-detail.html?id=${encodeURIComponent(item.work_item_id)}&visibility=${encodeURIComponent(state.filters.visibility || "SCOPE")}`;
    const canHandle = new Set(state.access.permissions || []).has("aiops:work_item_handle");
    const canManage = new Set(state.access.permissions || []).has("aiops:work_item_manage");
    const isMine = item.assignee_user_id === state.access.user_id;
    const canProcess = canHandle && item.responsibility_group_id && (
      (item.status === "IN_PROGRESS" && (isMine || canManage))
      || (["PENDING_TRIAGE", "OPEN", "WAITING"].includes(item.status) && (!item.assignee_user_id || isMine || canManage))
    );
    const canAssign = canManage && item.responsibility_group_id
      && ["PENDING_TRIAGE", "OPEN", "WAITING", "IN_PROGRESS"].includes(item.status);
    const processAction = canProcess
      ? `<button class="row-action" type="button" data-handle-id="${shell.escape(item.work_item_id)}">${isMine && item.status === "IN_PROGRESS" ? "继续处理" : "处理"}</button>` : "";
    const assignmentAction = canAssign
      ? `<button class="row-action" type="button" data-assignment-id="${shell.escape(item.work_item_id)}">分派</button>` : "";
    const typeLabels = {INCIDENT_RESPONSE:"故障响应",RISK_REMEDIATION:"风险整改",PROBLEM_INVESTIGATION:"问题调查",OPTIMIZATION:"优化事项",OBSERVABILITY_GAP:"监控缺口",RECOVERY_OBSERVATION:"恢复观察",RECOVERY_DRILL:"恢复演练"};
    const tone = ["CLOSED","RESOLVED"].includes(item.status) ? "status-success" : ["WAITING","PENDING_TRIAGE","PENDING_VERIFICATION"].includes(item.status) ? "status-warning" : item.status === "CANCELLED" ? "status-muted" : "status-info";
    return `<tr><td><strong>${shell.escape(item.priority)}</strong><small class="table-secondary">${shell.escape(item.severity)}</small></td><td><button class="work-item-title-link" type="button" data-open-work-item="${shell.escape(detail)}"><strong class="table-primary">${shell.escape(item.title)}</strong></button><small class="table-secondary">${shell.escape(item.item_key)} · ${shell.escape(typeLabels[item.work_type] || item.work_type)}</small></td><td><strong class="table-primary">${shell.escape(item.target_name)}</strong></td><td><span class="status-pill ${tone}">${shell.escape(text(item.status))}</span></td><td>${shell.escape(assignee)}</td><td${item.resolution_due_at && new Date(item.resolution_due_at).getTime()<Date.now() ? ' class="status-danger-text"' : ""}>${shell.escape(shell.fmt(item.resolution_due_at))}</td><td>${Number(item.occurrence_count || 0)}</td><td><div class="work-item-row-actions">${processAction}${assignmentAction}</div></td></tr>`;
  }
  async function handleItem(button) {
    if (state.handling) return;
    const source = state.items.get(button.dataset.handleId);
    if (!source) return;
    state.handling = true;
    button.disabled = true;
    try {
      let item = source;
      if (item.assignee_user_id && item.assignee_user_id !== state.access.user_id) {
        item = await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(item.work_item_id)}/assignment`, {
          method: "PATCH",
          body: JSON.stringify({ expected_row_version: item.row_version, responsibility_group_id: item.responsibility_group_id, assignee_user_id: state.access.user_id }),
        });
      }
      if (item.status !== "IN_PROGRESS") {
        item = await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(item.work_item_id)}:claim`, jsonBody({
          expected_row_version: Number(item.row_version),
        }));
        shell.toast("工作项已由你领取并进入处理中");
      }
      if (item.work_type === "RECOVERY_DRILL") {
        location.href = `./recovery-drills.html?target_id=${encodeURIComponent(item.target_id)}&work_item_id=${encodeURIComponent(item.work_item_id)}`;
        return;
      }
      const detail = await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(item.work_item_id)}?visibility=SCOPE`);
      state.processingItem = detail;
      document.getElementById("work-item-processing-title").textContent = detail.title;
      document.getElementById("work-item-processing-target").textContent = detail.target_name || "—";
      document.getElementById("work-item-processing-status").textContent = text(detail.status);
      document.getElementById("work-item-processing-started").textContent = shell.fmt(detail.assigned_at);
      document.getElementById("work-item-processing-form").reset();
      document.getElementById("work-item-processing-dialog").showModal();
      await loadProcessingContext(detail);
      await loadList();
    } catch (error) {
      shell.toast(error.message);
      await loadList();
    } finally {
      state.handling = false;
    }
  }
  async function openAssignment(button) {
    if (state.assigning) return;
    const item = state.items.get(button.dataset.assignmentId);
    if (!item) return;
    state.assigning = true;
    button.disabled = true;
    try {
      const groupId = encodeURIComponent(item.responsibility_group_id);
      const [group, directory] = await Promise.all([
        KBotAIOpsAuth.request(`${groupApi}/${groupId}`),
        KBotAIOpsAuth.request(`${groupApi}/${groupId}/member-candidates`),
      ]);
      const eligiblePeople = new Map(values(directory.items).map((person) => [String(person.user_id), person]));
      const members = values(group.members).flatMap((member) => {
        const person = eligiblePeople.get(String(member.user_id));
        return member.status === "ACTIVE" && String(member.user_id) !== String(state.access.user_id) && person
          ? [{ ...member, ...person }]
          : [];
      });
      const select = document.getElementById("work-item-assignment-user");
      select.innerHTML = `<option value="" disabled>${members.length ? "请选择组内其他成员" : "责任组内没有其他有效成员"}</option>`
        + members.map((member) => `<option value="${shell.escape(member.user_id)}">${shell.escape(member.display_name || member.username || member.user_id)} · ${member.member_role === "LEAD" ? "组长" : "成员"}</option>`).join("");
      select.value = members.some((member) => String(member.user_id) === String(item.assignee_user_id)) ? item.assignee_user_id : "";
      select.disabled = !members.length;
      document.getElementById("work-item-assignment-title").textContent = item.title;
      document.getElementById("work-item-assignment-group").value = item.responsibility_group_name || "未设置责任组";
      document.querySelector('#work-item-assignment-form [type="submit"]').disabled = !select.value;
      state.assignmentItem = item;
      document.getElementById("work-item-assignment-dialog").showModal();
    } catch (error) {
      shell.toast(error.message);
    } finally {
      state.assigning = false;
      button.disabled = false;
    }
  }
  async function loadList({ append = false } = {}) {
    const body = document.getElementById("work-item-body");
    const sequence = ++state.listRequest;
    const panel = body.closest(".work-item-list-panel");
    const loading = document.getElementById("work-item-list-loading");
    const moreButton = document.getElementById("work-item-more");
    const values = { ...state.filters, limit: 50, cursor: append ? state.cursor : null };
    loading.hidden = append;
    panel.hidden = !append;
    if (append) {moreButton.disabled = true;moreButton.textContent = "正在加载…";}
    try {
      const payload = await KBotAIOpsAuth.request(api + query(values));
      if (sequence !== state.listRequest) return;
      if (!append) state.items.clear();
      (payload.items || []).forEach((item) => state.items.set(item.work_item_id, item));
      const rows = (payload.items || []).map(listRow).join("");
      body.innerHTML = append ? body.innerHTML + rows : rows || '<tr><td colspan="8" class="empty-cell">当前筛选范围内没有工作项。</td></tr>';
      renderQueues(payload.queue_summary);
      state.cursor = payload.next_cursor || null;
      const more = document.getElementById("work-item-more");
      more.hidden = !payload.has_more;
      more.closest("footer").hidden = !payload.has_more;
      document.getElementById("work-item-count").textContent = `当前显示 ${state.items.size} 项 · 按解决 SLA 排序`;
    } catch (error) {
      if (sequence !== state.listRequest) return;
      shell.toast(error.message || "无法读取工作项");
      body.innerHTML = [...state.items.values()].map(listRow).join("") || '<tr><td colspan="8" class="empty-cell">当前筛选范围内没有工作项。</td></tr>';
    } finally {
      if (sequence === state.listRequest) {loading.hidden = true;panel.hidden = false;moreButton.disabled = false;moreButton.textContent = "加载更多";}
    }
  }
  function applyQueue(value) {
    const form = document.getElementById("work-item-filters");
    const targetId = state.filters.target_id || "";
    const visibility = state.filters.visibility || "SCOPE";
    state.suppressReset = true;
    form.reset();
    state.suppressReset = false;
    form.elements.target_id.value = targetId;
    form.elements.visibility.value = visibility;
    state.filters = {
      visibility,
      target_id: targetId,
    };
    if (value === "mine") state.filters.assignee_user_id = state.access.user_id;
    if (value === "unassigned") state.filters.unassigned = true;
    if (value === "urgent") state.filters.priority = "P1,P2";
    if (value === "verification") state.filters.status = "PENDING_VERIFICATION";
    if (value === "overdue") state.filters.overdue = true;
    form.elements.ownership.value = value === "mine" || value === "unassigned" ? value : "";
    form.elements.status.value = value === "verification" ? "PENDING_VERIFICATION" : "";
    form.elements.priority.value = value === "urgent" ? "P1,P2" : "";
    loadList();
  }
  function bindCreateWorkItem() {
    const dialog = document.getElementById("work-item-create-dialog");
    const form = document.getElementById("work-item-create-form");
    const trigger = document.getElementById("work-item-create");
    const submit = form.querySelector('[type="submit"]');
    const canManage = new Set(state.access.permissions || []).has("aiops:work_item_manage");
    trigger.hidden = !canManage;
    let targets = [], groups = [], memberRequest = 0;
    const selectedGroup = () => targets.find(target=>target.target_id===form.elements.target_id.value)?.responsibility_group_id || "";
    const loadMembers = async () => {
      const sequence = ++memberRequest;
      const id = selectedGroup();
      form.querySelector("[data-create-group]").value = groups.find(group=>group.responsibility_group_id===id)?.name || "未配置有效责任组";
      submit.disabled = Boolean(dialog.dataset.busy) || !id;
      const select = form.elements.assignee_user_id;
      select.disabled = true;
      select.innerHTML = '<option value="">正在读取成员…</option>';
      try {
        const group = id ? await KBotAIOpsAuth.request(`${groupApi}/${encodeURIComponent(id)}`) : {members:[]};
        if(sequence!==memberRequest) return;
        select.innerHTML = '<option value="">由组内 DBA 领取</option>'+values(group.members).filter(member=>member.status==="ACTIVE").map(member=>{
          const person = state.people.find(person=>person.user_id===member.user_id);
          return `<option value="${shell.escape(member.user_id)}">${shell.escape(person?.display_name || person?.username || "用户已不可用")} · ${member.member_role==="LEAD"?"组长":"成员"}</option>`;
        }).join("");
      } catch(error) {if(sequence===memberRequest){select.innerHTML='<option value="">由组内 DBA 领取</option>';shell.toast(error.message);}}
      finally {if(sequence===memberRequest)select.disabled=!id;}
    };
    trigger.onclick = async () => {
      if (!canManage || dialog.open) return;
      dialog.showModal();
      if(targets.length && groups.length) return;
      try {
        const [targetPage,groupPage] = await Promise.all([KBotAIOpsAuth.request(`${rootApi}/targets?limit=200`),KBotAIOpsAuth.request(groupApi)]);
        targets=values(targetPage.items);
        groups=values(groupPage.items).filter(group=>group.status==="ACTIVE");
        const groupIds=new Set(groups.map(group=>group.responsibility_group_id));
        const eligible=targets.filter(target=>target.status==="ENABLED" && target.responsibility_group_id && groupIds.has(target.responsibility_group_id));
        form.elements.target_id.innerHTML='<option value="">请选择要处理的数据库实例</option>'+eligible.map(target=>`<option value="${shell.escape(target.target_id)}">${shell.escape(target.display_name)} · ${shell.escape(target.db_type)}</option>`).join("");
        form.elements.target_id.value=eligible[0]?.target_id || "";
        await loadMembers();
      } catch(error) {shell.toast(error.message || "无法读取数据库实例");}
    };
    form.elements.target_id.onchange=loadMembers;
    dialog.querySelectorAll("[data-close-create]").forEach(button=>{button.onclick=()=>{if(!dialog.dataset.busy)dialog.close();};});
    dialog.addEventListener("cancel",event=>{if(dialog.dataset.busy)event.preventDefault();});
    form.onsubmit=async event=>{
      event.preventDefault();
      if(dialog.dataset.busy || !selectedGroup()) return;
      const data=Object.fromEntries(new FormData(form));
      if(!data.target_id || !data.title.trim() || !data.summary.trim()) return shell.toast("请选择数据库实例，并填写标题与处理说明。");
      dialog.dataset.busy="true";
      submit.disabled=true;
      submit.textContent="正在创建…";
      try {
        const item=await KBotAIOpsAuth.request(api,{method:"POST",headers:{"Idempotency-Key":KBotAIOpsAuth.uuid()},body:JSON.stringify({...data,title:data.title.trim(),summary:data.summary.trim(),assignee_user_id:data.assignee_user_id?.trim() || null})});
        shell.toast("工作项已创建");
        location.href=`./work-item-detail.html?id=${encodeURIComponent(item.work_item_id)}&visibility=${encodeURIComponent(state.filters.visibility)}`;
      } catch(error) {shell.toast(error.message || "无法创建工作项");}
      finally {delete dialog.dataset.busy;submit.disabled=!selectedGroup();submit.textContent="创建工作项";}
    };
  }

  function bindList() {
    const form = document.getElementById("work-item-filters");
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const values = Object.fromEntries(new FormData(form));
      state.filters = {
        target_id: values.target_id, status: values.status, priority: values.priority,
        assignee_user_id: values.ownership === "mine" ? state.access.user_id : "",
        unassigned: values.ownership === "unassigned",
        visibility: values.visibility || "SCOPE",
        overdue: state.filters.overdue,
      };
      loadList();
    });
    form.addEventListener("change", () => form.requestSubmit());
    form.addEventListener("reset", () => {
      if (state.suppressReset) return;
      setTimeout(() => { state.filters = { visibility: "SCOPE" };loadList(); }, 0);
    });
    document.getElementById("work-item-overdue").onclick = () => {state.filters.overdue=!state.filters.overdue;loadList();};
    document.getElementById("work-items-refresh").onclick = () => loadList();
    document.getElementById("work-item-more").onclick = () => loadList({ append: true });
    document.getElementById("work-item-queues").onclick = (event) => {
      const button = event.target.closest("[data-queue]");
      if (button) applyQueue(button.dataset.queue);
    };
    document.getElementById("work-item-body").onclick = (event) => {
      const open = event.target.closest("[data-open-work-item]");
      if (open) location.href = open.dataset.openWorkItem;
      const handleButton = event.target.closest("[data-handle-id]");
      if (handleButton) return void handleItem(handleButton);
      const assignmentButton = event.target.closest("[data-assignment-id]");
      if (assignmentButton) void openAssignment(assignmentButton);
    };
    document.querySelectorAll("[data-close-work-item-dialog]").forEach((button) => {
      button.onclick = () => {const dialog=button.closest("dialog");if(!dialog.dataset.busy)dialog.close();};
    });
    for (const dialog of document.querySelectorAll("dialog.work-item-modal")) {
      dialog.addEventListener("cancel",event=>{if(dialog.dataset.busy)event.preventDefault();});
      dialog.addEventListener("click",event=>{
        const box=dialog.getBoundingClientRect();
        if(event.target===dialog && !dialog.dataset.busy && (event.clientX<box.left || event.clientX>box.right || event.clientY<box.top || event.clientY>box.bottom)) dialog.close();
      });
    }
    document.getElementById("work-item-processing-dialog").addEventListener("close", () => {
      state.processingItem = null;
      state.processingContext = null;
    });
    document.getElementById("work-item-assignment-dialog").addEventListener("close", () => { state.assignmentItem = null; });
    document.getElementById("work-item-assignment-user").onchange = event => {document.querySelector('#work-item-assignment-form [type="submit"]').disabled=!event.target.value;};
    document.getElementById("work-item-continue-diagnosis").onclick = (event) => {
      void continueProcessingDiagnosis(event.currentTarget);
    };
    document.getElementById("work-item-processing-form").onsubmit = async (event) => {
      event.preventDefault();
      const dialog = document.getElementById("work-item-processing-dialog");
      if (!state.processingItem || dialog.dataset.busy) return;
      const submit = event.currentTarget.querySelector("button[type=submit]");
      const values = Object.fromEntries(new FormData(event.currentTarget));
      if (!String(values.completion_note || "").trim()) return shell.toast("请填写处理结果。");
      dialog.dataset.busy = "true";
      dialog.querySelectorAll('[data-close-work-item-dialog],[type="submit"]').forEach(button=>{button.disabled=true;});
      submit.textContent = "正在提交…";
      try {
        await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(state.processingItem.work_item_id)}:complete`, jsonBody({
          expected_row_version: state.processingItem.row_version,
          resolution_code: values.resolution_code,
          completion_note: String(values.completion_note || "").trim(),
          evidence_refs: [],
        }));
        document.getElementById("work-item-processing-dialog").close();
        state.processingItem = null;
        shell.toast("处理结果已记录，工作项已标记完成");
        await loadList();
      } catch (error) { shell.toast(error.message); }
      finally { delete dialog.dataset.busy;dialog.querySelectorAll('[data-close-work-item-dialog],[type="submit"]').forEach(button=>{button.disabled=false;});submit.textContent="标记工作完成"; }
    };
    document.getElementById("work-item-assignment-form").onsubmit = async (event) => {
      event.preventDefault();
      const dialog = document.getElementById("work-item-assignment-dialog");
      if (!state.assignmentItem || dialog.dataset.busy) return;
      const submit = event.currentTarget.querySelector("button[type=submit]");
      const assignee = document.getElementById("work-item-assignment-user").value;
      if (!assignee) return;
      dialog.dataset.busy = "true";
      dialog.querySelectorAll('[data-close-work-item-dialog],[type="submit"]').forEach(button=>{button.disabled=true;});
      submit.textContent = "正在分派…";
      try {
        await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(state.assignmentItem.work_item_id)}/assignment`, {
          method: "PATCH",
          body: JSON.stringify({ expected_row_version: state.assignmentItem.row_version, responsibility_group_id: state.assignmentItem.responsibility_group_id, assignee_user_id: assignee }),
        });
        document.getElementById("work-item-assignment-dialog").close();
        state.assignmentItem = null;
        shell.toast("工作项已分派；被分派人点击“处理”后开始处理");
        await loadList();
      } catch (error) { shell.toast(error.message); }
      finally { delete dialog.dataset.busy;dialog.querySelectorAll('[data-close-work-item-dialog],[type="submit"]').forEach(button=>{button.disabled=false;});submit.disabled=!document.getElementById("work-item-assignment-user").value;submit.textContent="确认分派"; }
    };
  }
  function overview(item) {
    const esc = shell.escape;
    const small = (value,technical=false) => `<small class="table-secondary${technical?" technical-value":""}">${esc(value)}</small>`;
    const rows = [
      ["来源 Agent",esc(item.agent_name || "待归属")+(item.agent_id?small(item.agent_id,true):"")],
      ["数据库实例",esc(item.target_name)+small(item.target_id,true)],
      ["责任人",esc(item.assignee_user_id || "未领取")+small(item.responsibility_group_name || item.responsibility_group_id || "未设置责任组")],
      ["最近分派 / 领取",esc(shell.fmt(item.assigned_at))],
      ["当前阶段",esc(item.phase || "")],
      ["首次发现",esc(shell.fmt(item.first_observed_at))],
      ["最近发现",`${esc(shell.fmt(item.last_observed_at))} · 累计 ${Number(item.occurrence_count || 0)} 次`],
      ["确认时限",esc(shell.fmt(item.acknowledgement_due_at))],
      ["解决时限",esc(shell.fmt(item.resolution_due_at))],
      ["验证时限",esc(shell.fmt(item.verification_due_at))],
    ];
    if(item.completed_at) rows.push(["DBA 完成",`${esc(text(item.resolution_code))} · ${esc(item.completion_note || "")}${small(`${item.completed_by || ""} · ${shell.fmt(item.completed_at)}`)}`]);
    return rows.map(([label,value])=>`<tr><th>${label}</th><td>${value}</td></tr>`).join("");
  }
  function occurrenceRow(row) {
    const finding = row.finding || {};
    const object = finding.object_ref || {};
    const evidence = (row.evidence_refs || []).length
      ? `<div class="work-item-evidence">${row.evidence_refs.map((ref) => `<code>${shell.escape(ref)}</code>`).join("")}</div>` : "";
    return `<article><header><div><span class="status-pill ${row.severity==="CRITICAL"?"status-info":"status-warning"}">${shell.escape(row.severity)}</span><strong>${shell.escape(row.finding_type)} · ${shell.escape(object.object_name || object.sql_id || object.object_kind || "数据采集链路")}</strong></div><time>${shell.escape(shell.fmt(row.observed_at))}</time></header><p>${shell.escape(finding.impact || state.item.summary)}</p><small>确认度 ${shell.escape(row.confirmation)} · Run ${shell.escape(row.ops_run_id)}</small>${evidence}</article>`;
  }
  function resourceLink(row) {
    const paths = { RUN: "runs.html", VERIFICATION: "runs.html", REPORT: `report-detail.html?id=${encodeURIComponent(row.resource_id)}`, SITUATION:"situations.html" };
    const href = paths[row.resource_kind] ? `./${paths[row.resource_kind]}` : "";
    const content = `<strong>${shell.escape(row.label || row.resource_id)}</strong>`;
    return `<tr><th>${shell.escape(row.resource_kind)}</th><td>${href?`<a class="text-button" href="${href}">${content}</a>`:content}<small class="table-secondary">${shell.escape(row.role)}${row.status?` · ${shell.escape(row.status)}`:""} · ${shell.escape(row.resource_id)}</small></td></tr>`;
  }
  function activityRow(row) {
    const transition = row.from_status || row.to_status
      ? `<p>${shell.escape(text(row.from_status))} → ${shell.escape(text(row.to_status))}</p>` : "";
    return `<article><time>${shell.escape(shell.fmt(row.created_at))}</time><div><strong>${shell.escape(text(row.activity_type))}</strong>${transition}<small class="technical-value">操作人 ${shell.escape(row.actor_id)}</small></div></article>`;
  }
  function renderDetail(item) {
    state.item = item;
    document.getElementById("work-item-title").textContent = item.title;
    document.getElementById("work-item-heading-title").textContent = item.title;
    document.getElementById("work-item-summary").textContent = item.summary;
    document.getElementById("work-item-key").textContent = `${item.item_key} · ${item.priority}`;
    const status = document.getElementById("work-item-heading-status");
    status.textContent = text(item.status);
    status.className = `status-pill ${["CLOSED","RESOLVED"].includes(item.status)?"status-success":["WAITING","PENDING_TRIAGE","PENDING_VERIFICATION"].includes(item.status)?"status-warning":item.status==="CANCELLED"?"status-muted":"status-info"}`;
    document.getElementById("work-item-overview").innerHTML = overview(item);
    document.getElementById("work-item-occurrences").innerHTML = (item.occurrences || []).length
      ? `<div class="work-item-record-list">${item.occurrences.map(occurrenceRow).join("")}</div>` : '<p class="empty-cell">人工创建的工作项暂无自动诊断发生记录。</p>';
    document.getElementById("work-item-links").innerHTML = (item.links || []).length
      ? item.links.map(resourceLink).join("") : '<tr><td colspan="2">暂无关联 Run、Situation、Report、Proposal 或 Verification。</td></tr>';
    document.getElementById("work-item-activities").innerHTML = (item.activities || []).length
      ? `<div class="work-item-timeline">${item.activities.map(activityRow).join("")}</div>` : '<p class="empty-cell">暂无活动记录。</p>';
  }
  async function loadDetail() {
    const params = new URLSearchParams(location.search);
    const id = params.get("id");
    const canManage = new Set(state.access.permissions || []).has("aiops:work_item_manage");
    const requestedVisibility = params.get("visibility") || "SCOPE";
    const visibility = canManage && ["ALL", "EXCEPTIONS"].includes(requestedVisibility) ? requestedVisibility : "SCOPE";
    if (!id) {
      location.replace("./dashboard.html");return;
    }
    const loading=document.getElementById("work-item-detail-loading");
    const content=document.getElementById("work-item-detail-content");
    loading.hidden=false;content.hidden=true;
    try {
      const item = await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(id)}?visibility=${encodeURIComponent(visibility)}`);
      document.querySelector(".ops-back-link").href = `./dashboard.html?visibility=${encodeURIComponent(visibility)}`;
      renderDetail(item);
    } catch (error) {
      shell.toast(error.message || "无法读取工作项详情");
    } finally {loading.hidden=true;content.hidden=!state.item;}
  }
  function bindDetail() {
    document.getElementById("work-item-detail-refresh").onclick = loadDetail;
  }
  shell.ready.then((access) => {
    state.access = access;
    if (document.body.dataset.page === "dashboard") {
      globalThis.KBotAIOpsWorkItems = { refresh: () => loadList() };
      const canManage = new Set(access.permissions || []).has("aiops:work_item_manage");
      const requested = new URLSearchParams(location.search).get("visibility") || "SCOPE";
      state.filters.visibility = canManage && ["ALL", "EXCEPTIONS"].includes(requested) ? requested : "SCOPE";
      const visibility = document.querySelector("[name=visibility]");
      if (visibility) {
        visibility.closest("label").hidden = !canManage;
        visibility.value = state.filters.visibility;
      }
      bindList();
      bindCreateWorkItem();
      void loadScopedTargets()
        .catch((error) => shell.toast(`无法读取责任范围内数据库实例：${error.message}`))
        .finally(() => loadList());
    }
    if (document.body.dataset.page === "work-item-detail") { bindDetail(); loadDetail(); }
  });
})();
