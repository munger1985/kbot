(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/work-items";
  const rootApi = "/api/v1/apps/aiops";
  const groupApi = "/api/v1/apps/aiops/responsibility-groups";
  const shell = globalThis.KBotAIOpsShell;
  const markdown = globalThis.KBotMarkdown;
  const state = {
    access: null, cursor: null, filters: { visibility: "SCOPE" }, item: null,
    items: new Map(), processingItem: null, processingContext: null, assignmentItem: null,
    scopedTargets: [], handling: false, assigning: false, suppressReset: false,
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
  };
  const text = (value) => labels[value] || value || "—";
  const values = (items) => Array.isArray(items) ? items : [];
  const displayText = (value, dictionary = {}) => String(value ?? "")
    .split(/(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g)
    .map((part, index) => index % 2 ? part : part.replace(
      /(^|[^A-Za-z0-9_])([A-Z][A-Z0-9_]{2,})(?=$|[^A-Za-z0-9_])/g,
      (match, prefix, code) => `${prefix}${dictionary[code] || code}`,
    ))
    .join("");
  const jsonBody = (value) => ({ method: "POST", body: JSON.stringify(value) });
  const query = (values) => {
    const params = new URLSearchParams();
    Object.entries(values).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== "" && value !== false) params.set(key, String(value));
    });
    return params.toString() ? `?${params}` : "";
  };
  function priority(value) {
    return `<span class="ops-work-item-priority priority-${shell.escape(String(value).toLowerCase())}">${shell.escape(text(value))}</span>`;
  }
  function due(value) {
    if (!value) return "—";
    const overdue = new Date(value).getTime() < Date.now();
    return `<time class="${overdue ? "is-overdue" : ""}">${shell.escape(shell.fmt(value))}</time>`;
  }
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
    ].filter((row) => row[1]).map((row) => `<div><dt>${shell.escape(row[0])}</dt><dd>${shell.escape(row[1])}</dd></div>`).join("") : "";
    const signals = values(situation?.signal_events).map((signal) => `<li><strong>${shell.escape(signal.event_class)} · ${shell.escape(signal.severity)}</strong><span>${shell.escape(signal.summary || signal.normalized_status)}</span><time>${shell.escape(shell.fmt(signal.occurred_at))}</time></li>`).join("");
    const sourceAlerts = signals || values(situation?.monitoring_sources).map((source) => `<li><strong>${shell.escape(source.latest_event_class || source.source_type || "ALERT")} · ${shell.escape(source.latest_severity || item.severity)}</strong><span>${shell.escape(source.latest_summary || item.summary)}</span><time>${shell.escape(shell.fmt(source.last_observed_at))}</time></li>`).join("");
    const evidence = values(occurrence?.evidence_refs).map((ref) => `<code>${shell.escape(ref)}</code>`).join("");
    return `<p class="ops-work-item-context-summary">${shell.escape(finding.impact || item.summary)}</p>${facts ? `<dl class="ops-work-item-context-facts">${facts}</dl>` : ""}${sourceAlerts ? `<ul class="ops-work-item-signal-list">${sourceAlerts}</ul>` : ""}${evidence ? `<div class="ops-work-item-context-evidence">${evidence}</div>` : ""}`;
  }
  function findingSummaryHtml(payload) {
    const rows = values(payload?.findings).map((finding) => `<article><header><strong>${shell.escape(finding.finding_type || "发现")}</strong>${shell.badge(finding.severity || "INFO")}</header><p>${shell.escape(finding.impact || "")}</p></article>`).join("");
    return rows ? `<section class="ops-work-item-diagnosis-findings">${rows}</section>` : "";
  }
  function processingDiagnosisHtml(result, run) {
    if (!result) {
      if (run) return `<div class="ops-empty">自动诊断运行状态：${shell.escape(run.status)}，尚未形成可展示结果。</div>`;
      return '<div class="ops-empty">该工作项没有关联可展示的 Agent 初步诊断。</div>';
    }
    const blocks = values(result.payload?.blocks);
    const dictionary = result.payload?.display_dictionary || {};
    if (result.final_artifact?.schema_version === "AIOPS_TURN_RESULT.v1" && blocks.length) {
      const html = blocks.map((block) => {
        const payload = block.payload || {};
        if (block.block_type === "FINDING_CARDS") return findingSummaryHtml(payload);
        if (["ANALYSIS_MARKDOWN", "SOLUTION_MARKDOWN", "MARKDOWN"].includes(block.block_type)) {
          const heading = block.block_type === "ANALYSIS_MARKDOWN" ? "分析" : block.block_type === "SOLUTION_MARKDOWN" ? "解决方案" : "";
          const body = markdown.render(displayText(payload.markdown || payload.text || "", dictionary));
          return heading ? `<section class="ops-analysis"><header><strong>${heading}</strong></header>${body}</section>` : body;
        }
        return "";
      }).join("");
      return html || '<div class="ops-empty">Agent 已完成诊断，但没有可展示的文字结论。</div>';
    }
    const payload = result.payload || {};
    const answer = payload.direct_answer?.answer_text || payload.diagnosis_rationale || payload.summary || "Agent 已完成初步诊断，但没有可展示的文字结论。";
    return `<div class="ops-result-markdown">${markdown.render(displayText(answer, dictionary))}</div>`;
  }
  async function loadProcessingContext(item) {
    document.getElementById("work-item-alert-context").innerHTML = '<div class="ops-empty">正在读取错误与告警上下文…</div>';
    document.getElementById("work-item-agent-diagnosis").innerHTML = '<div class="ops-empty">正在读取 Agent 初步诊断…</div>';
    document.getElementById("work-item-diagnosis-agent").innerHTML = '<option value="">正在读取可用 Agent…</option>';
    document.getElementById("work-item-diagnosis-agent").disabled = true;
    document.getElementById("work-item-continue-diagnosis").disabled = true;
    const occurrence = values(item.occurrences)[0];
    const runId = occurrence?.ops_run_id || values(item.links).find((link) => ["RUN", "VERIFICATION"].includes(link.resource_kind))?.resource_id;
    const situationId = occurrence?.situation_id || values(item.links).find((link) => link.resource_kind === "SITUATION")?.resource_id;
    const [situationResult, runResult, diagnosisResult, agentResult] = await Promise.allSettled([
      situationId ? KBotAIOpsAuth.request(`${rootApi}/situations/${encodeURIComponent(situationId)}`) : Promise.resolve(null),
      runId ? KBotAIOpsAuth.request(`${rootApi}/runs/${encodeURIComponent(runId)}`) : Promise.resolve(null),
      runId ? KBotAIOpsAuth.request(`${rootApi}/runs/${encodeURIComponent(runId)}/result`) : Promise.resolve(null),
      KBotAIOpsAuth.request(`${rootApi}/agents`),
    ]);
    const situation = situationResult.status === "fulfilled" ? situationResult.value : null;
    const run = runResult.status === "fulfilled" ? runResult.value : null;
    const result = diagnosisResult.status === "fulfilled" ? diagnosisResult.value : null;
    const agents = agentResult.status === "fulfilled"
      ? values(agentResult.value).filter((agent) => agent.status === "ACTIVE")
      : [];
    const source = result?.final_artifact && runId
      ? { source_run_id: runId }
      : situationId
        ? { source_situation_id: situationId }
        : runId ? { source_run_id: runId } : null;
    state.processingContext = { situation, run, result, source };
    document.getElementById("work-item-alert-context").innerHTML = processingAlertHtml(item, situation);
    document.getElementById("work-item-agent-diagnosis").innerHTML = processingDiagnosisHtml(result, run);
    const agentSelect = document.getElementById("work-item-diagnosis-agent");
    agentSelect.innerHTML = '<option value="">请选择 Agent</option>' + agents.map((agent) => `<option value="${shell.escape(agent.agent_id)}">${shell.escape(agent.display_name || agent.agent_key || shell.short(agent.agent_id))}</option>`).join("");
    if (agents.some((agent) => String(agent.agent_id) === String(item.agent_id))) agentSelect.value = item.agent_id;
    agentSelect.disabled = !source || !agents.length;
    const prompt = document.getElementById("work-item-diagnosis-prompt");
    prompt.value = source ? `请基于工作项“${item.title}”已有的告警、诊断结论和证据继续分析：` : "该工作项没有可继承的诊断上下文。";
    prompt.disabled = !source;
    document.getElementById("work-item-diagnosis-note").textContent = source
      ? `系统将在服务端继承本次${source.source_run_id ? "自动诊断结果与证据" : "告警情境及其监控信号"}。`
      : "人工创建且未关联 Situation 或 Run 的工作项不能继续诊断。";
    const continueButton = document.getElementById("work-item-continue-diagnosis");
    continueButton.disabled = !source || !agents.length;
  }
  async function continueProcessingDiagnosis(button) {
    const item = state.processingItem;
    const source = state.processingContext?.source;
    const agentId = document.getElementById("work-item-diagnosis-agent").value;
    const message = document.getElementById("work-item-diagnosis-prompt").value.trim();
    if (!item || !source) return shell.toast("该工作项没有可继承的诊断上下文");
    if (!agentId) return shell.toast("请选择继续诊断使用的 Agent");
    if (!message) return shell.toast("请填写继续追问的内容");
    button.disabled = true;
    try {
      const body = {
        agent_id: agentId,
        target_id: item.target_id,
        content: [{ content_type: "TEXT", text: message }],
        ...source,
      };
      const result = await KBotAIOpsAuth.request(`${rootApi}/conversations`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify(body),
      });
      location.href = `./chat.html?conversation=${encodeURIComponent(result.conversation_id)}`;
    } catch (error) {
      shell.toast(error.message);
      button.disabled = false;
    }
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
      `<button class="${tone ? `is-${tone} ` : ""}${queueIsActive(filter) ? "is-active" : ""}" type="button" data-queue="${filter}" aria-pressed="${queueIsActive(filter)}"><header><span>${label}</span><strong>${Number(summary?.[key] || 0)}</strong></header><p>${description}</p></button>`
    ).join("");
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
    const assignee = item.assignee_user_id || item.responsibility_group_name || item.responsibility_group_id || "未分派";
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
      ? `<button class="primary" type="button" data-handle-id="${shell.escape(item.work_item_id)}">${isMine && item.status === "IN_PROGRESS" ? "继续处理" : "处理"}</button>` : "";
    const assignmentAction = canAssign
      ? `<button type="button" data-assignment-id="${shell.escape(item.work_item_id)}">分派</button>` : "";
    return `<tr><td>${priority(item.priority)}</td><td><a href="${detail}"><strong>${shell.escape(item.title)}</strong></a><small>${shell.escape(item.item_key)} · ${shell.escape(text(item.source_kind))}</small></td><td><strong>${shell.escape(item.target_name)}</strong><small>${shell.escape(shell.short(item.target_id))}</small></td><td>${shell.badge(text(item.status))}<small>${shell.escape(text(item.phase))}</small></td><td>${shell.escape(assignee)}</td><td>${shell.escape(shell.fmt(item.last_observed_at))}</td><td>${due(item.resolution_due_at)}</td><td>${Number(item.occurrence_count || 0)}</td><td><div class="ops-actions">${processAction}${assignmentAction}</div></td></tr>`;
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
      document.getElementById("work-item-assignment-group").value = item.responsibility_group_name || item.responsibility_group_id;
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
    const values = { ...state.filters, limit: 50, cursor: append ? state.cursor : null };
    if (!append) body.innerHTML = '<tr><td colspan="9" class="ops-empty">正在加载工作项…</td></tr>';
    try {
      const payload = await KBotAIOpsAuth.request(api + query(values));
      if (!append) state.items.clear();
      (payload.items || []).forEach((item) => state.items.set(item.work_item_id, item));
      const rows = (payload.items || []).map(listRow).join("");
      body.innerHTML = append ? body.innerHTML + rows : rows || '<tr><td colspan="9" class="ops-empty">当前筛选范围内暂无工作项</td></tr>';
      renderQueues(payload.queue_summary);
      state.cursor = payload.next_cursor || null;
      const more = document.getElementById("work-item-more");
      more.hidden = !payload.has_more;
      document.getElementById("work-item-count").textContent = `本页 ${payload.items?.length || 0} 项 · 按解决 SLA 排序`;
    } catch (error) {
      body.innerHTML = `<tr><td colspan="9" class="ops-empty">${shell.escape(error.message)}</td></tr>`;
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
    loadList();
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
      };
      loadList();
    });
    form.addEventListener("reset", () => {
      if (state.suppressReset) return;
      setTimeout(() => { state.filters = { visibility: "SCOPE" }; loadList(); }, 0);
    });
    document.getElementById("work-items-refresh").onclick = () => loadList();
    document.getElementById("work-item-more").onclick = () => loadList({ append: true });
    document.getElementById("work-item-queues").onclick = (event) => {
      const button = event.target.closest("[data-queue]");
      if (button) applyQueue(button.dataset.queue);
    };
    document.getElementById("work-item-body").onclick = (event) => {
      const handleButton = event.target.closest("[data-handle-id]");
      if (handleButton) return void handleItem(handleButton);
      const assignmentButton = event.target.closest("[data-assignment-id]");
      if (assignmentButton) void openAssignment(assignmentButton);
    };
    document.querySelectorAll("[data-close-work-item-dialog]").forEach((button) => {
      button.onclick = () => button.closest("dialog").close();
    });
    document.getElementById("work-item-processing-dialog").addEventListener("close", () => {
      state.processingItem = null;
      state.processingContext = null;
    });
    document.getElementById("work-item-assignment-dialog").addEventListener("close", () => { state.assignmentItem = null; });
    document.getElementById("work-item-continue-diagnosis").onclick = (event) => {
      void continueProcessingDiagnosis(event.currentTarget);
    };
    document.getElementById("work-item-processing-form").onsubmit = async (event) => {
      event.preventDefault();
      if (!state.processingItem) return;
      const submit = event.currentTarget.querySelector("button[type=submit]");
      const values = Object.fromEntries(new FormData(event.currentTarget));
      submit.disabled = true;
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
      finally { submit.disabled = false; }
    };
    document.getElementById("work-item-assignment-form").onsubmit = async (event) => {
      event.preventDefault();
      if (!state.assignmentItem) return;
      const submit = event.currentTarget.querySelector("button[type=submit]");
      const assignee = document.getElementById("work-item-assignment-user").value;
      if (!assignee) return;
      submit.disabled = true;
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
      finally { submit.disabled = false; }
    };
  }
  function overview(item) {
    return `<article><span>优先级</span>${priority(item.priority)}</article><article><span>状态</span><strong>${shell.escape(text(item.status))}</strong><small>${shell.escape(text(item.phase))}</small></article><article><span>来源 Agent</span><strong>${shell.escape(item.agent_name || "待归属")}</strong><small>${shell.escape(item.agent_id ? shell.short(item.agent_id) : "历史工作项")}</small></article><article><span>数据库实例</span><strong>${shell.escape(item.target_name)}</strong><small>${shell.escape(shell.short(item.target_id))}</small></article><article><span>责任人</span><strong>${shell.escape(item.assignee_user_id || "未领取")}</strong><small>${shell.escape(item.responsibility_group_name || item.responsibility_group_id || "未设置责任组")}</small></article><article><span>最近分派 / 领取</span><strong>${shell.escape(shell.fmt(item.assigned_at))}</strong><small>${shell.escape(item.assignee_user_id || "尚未领取")}</small></article><article><span>最后发现</span><strong>${shell.escape(shell.fmt(item.last_observed_at))}</strong><small>累计 ${Number(item.occurrence_count)} 次</small></article><article><span>解决 SLA</span><strong>${due(item.resolution_due_at)}</strong><small>验证期限 ${shell.escape(shell.fmt(item.verification_due_at))}</small></article>`;
  }
  function occurrenceRow(row) {
    const finding = row.finding || {};
    const object = finding.object_ref || {};
    const evidence = (row.evidence_refs || []).length
      ? `<div class="ops-work-item-evidence">${row.evidence_refs.map((ref) => `<code>${shell.escape(ref)}</code>`).join("")}</div>` : "";
    return `<article class="ops-work-item-occurrence"><header><div>${shell.badge(row.severity)}<strong>${shell.escape(row.finding_type)}</strong></div><time>${shell.escape(shell.fmt(row.observed_at))}</time></header><p>${shell.escape(finding.impact || state.item.summary)}</p><dl><div><dt>对象</dt><dd>${shell.escape(object.object_name || object.sql_id || object.object_kind || "数据采集链路")}</dd></div><div><dt>确认度</dt><dd>${shell.escape(row.confirmation)}</dd></div><div><dt>来源 Run</dt><dd><a href="./run-detail.html?id=${encodeURIComponent(row.ops_run_id)}">${shell.escape(shell.short(row.ops_run_id))}</a></dd></div></dl>${evidence}</article>`;
  }
  function resourceLink(row) {
    const paths = { RUN: "run-detail.html?id=", VERIFICATION: "run-detail.html?id=", REPORT: "report-detail.html?id=" };
    const href = paths[row.resource_kind] ? `./${paths[row.resource_kind]}${encodeURIComponent(row.resource_id)}` : "";
    const content = `<strong>${shell.escape(row.label || text(row.resource_kind))}</strong><small>${shell.escape(row.role)} · ${shell.escape(shell.short(row.resource_id))}${row.status ? ` · ${shell.escape(row.status)}` : ""}</small>`;
    return href ? `<a href="${href}">${content}</a>` : `<article>${content}</article>`;
  }
  function activityRow(row) {
    const transition = row.from_status || row.to_status
      ? `<small>${shell.escape(text(row.from_status))} → ${shell.escape(text(row.to_status))}</small>` : "";
    return `<article><time>${shell.escape(shell.fmt(row.created_at))}</time><div><strong>${shell.escape(text(row.activity_type))}</strong>${transition}<small>${shell.escape(row.actor_id)}</small></div></article>`;
  }
  function renderDetail(item) {
    state.item = item;
    document.getElementById("work-item-title").textContent = item.title;
    document.getElementById("work-item-summary").textContent = item.summary;
    document.getElementById("work-item-key").textContent = item.item_key;
    document.getElementById("work-item-overview").innerHTML = overview(item);
    document.getElementById("work-item-occurrences").innerHTML = (item.occurrences || []).length
      ? item.occurrences.map(occurrenceRow).join("") : '<div class="ops-empty">人工创建的工作项暂无自动诊断发生记录</div>';
    document.getElementById("work-item-links").innerHTML = (item.links || []).length
      ? item.links.map(resourceLink).join("") : '<div class="ops-empty">暂无关联资源</div>';
    document.getElementById("work-item-activities").innerHTML = (item.activities || []).length
      ? item.activities.map(activityRow).join("") : '<div class="ops-empty">暂无活动记录</div>';
    const result = document.getElementById("work-item-result");
    result.hidden = !item.completed_at;
    if (item.completed_at) {
      document.getElementById("work-item-result-body").innerHTML = `<dl class="ops-detail"><dt>处理结论</dt><dd>${shell.escape(text(item.resolution_code))}</dd><dt>处理结果</dt><dd>${shell.escape(item.completion_note || "—")}</dd><dt>完成人</dt><dd>${shell.escape(item.completed_by || "—")}</dd><dt>完成时间</dt><dd>${shell.escape(shell.fmt(item.completed_at))}</dd></dl>`;
    }
  }
  async function loadDetail() {
    const params = new URLSearchParams(location.search);
    const id = params.get("id");
    const canManage = new Set(state.access.permissions || []).has("aiops:work_item_manage");
    const requestedVisibility = params.get("visibility") || "SCOPE";
    const visibility = canManage && ["ALL", "EXCEPTIONS"].includes(requestedVisibility) ? requestedVisibility : "SCOPE";
    if (!id) {
      document.getElementById("work-item-overview").innerHTML = '<div class="ops-empty">缺少工作项 ID</div>';
      return;
    }
    try {
      const item = await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(id)}?visibility=${encodeURIComponent(visibility)}`);
      document.querySelector(".ops-back-link").href = `./dashboard.html?visibility=${encodeURIComponent(visibility)}`;
      renderDetail(item);
    } catch (error) {
      document.getElementById("work-item-overview").innerHTML = `<div class="ops-empty">${shell.escape(error.message)}</div>`;
    }
  }
  function bindDetail() {
    document.getElementById("work-item-detail-refresh").onclick = loadDetail;
  }
  shell.ready.then((access) => {
    state.access = access;
    if (document.body.dataset.page === "dashboard") {
      const canManage = new Set(access.permissions || []).has("aiops:work_item_manage");
      const requested = new URLSearchParams(location.search).get("visibility") || "SCOPE";
      state.filters.visibility = canManage && ["ALL", "EXCEPTIONS"].includes(requested) ? requested : "SCOPE";
      const visibility = document.querySelector("[name=visibility]");
      if (visibility) {
        visibility.closest("label").hidden = !canManage;
        visibility.value = state.filters.visibility;
      }
      bindList();
      void loadScopedTargets()
        .catch((error) => shell.toast(`无法读取责任范围内数据库实例：${error.message}`))
        .finally(() => loadList());
    }
    if (document.body.dataset.page === "work-item-detail") { bindDetail(); loadDetail(); }
  });
})();
