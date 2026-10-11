(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/work-items";
  const groupApi = "/api/v1/apps/aiops/responsibility-groups";
  const shell = globalThis.KBotAIOpsShell;
  const state = { access: null, cursor: null, filters: {}, item: null, groups: [] };
  const transitions = {
    PENDING_TRIAGE: ["OPEN", "CANCELLED"],
    OPEN: ["IN_PROGRESS", "WAITING", "CANCELLED"],
    IN_PROGRESS: ["WAITING", "CANCELLED"],
    WAITING: ["IN_PROGRESS", "CANCELLED"],
    PENDING_VERIFICATION: [],
    RESOLVED: ["IN_PROGRESS", "CLOSED"],
    CLOSED: [], CANCELLED: [],
  };
  const labels = {
    PENDING_TRIAGE: "待研判", OPEN: "待处理", IN_PROGRESS: "处理中",
    WAITING: "等待中", PENDING_VERIFICATION: "待验证", RESOLVED: "已解决",
    CLOSED: "已关闭", CANCELLED: "已取消", TRIAGE: "研判",
    DIAGNOSIS: "诊断", REMEDIATION: "处置", APPROVAL: "审批",
    EXECUTION: "执行", VERIFICATION: "验证", P1: "P1 紧急",
    P2: "P2 高", P3: "P3 中", P4: "P4 低",
  };
  const text = (value) => labels[value] || value || "—";
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
  function renderQueues(summary) {
    const definitions = [
      ["my_open", "我的待办", "mine"], ["unassigned", "未分派", "unassigned"],
      ["urgent", "P1 / P2", "urgent"], ["overdue", "已超 SLA", "overdue"],
      ["pending_verification", "待验证", "verification"],
    ];
    document.getElementById("work-item-queues").innerHTML = definitions.map(([key, label, filter]) =>
      `<button type="button" data-queue="${filter}"><span>${label}</span><strong>${Number(summary?.[key] || 0)}</strong></button>`
    ).join("");
  }
  function listRow(item) {
    const assignee = item.assignee_user_id || item.responsibility_group_name || item.responsibility_group_id || "未分派";
    return `<tr><td>${priority(item.priority)}</td><td><a href="./work-item-detail.html?id=${encodeURIComponent(item.work_item_id)}"><strong>${shell.escape(item.title)}</strong></a><small>${shell.escape(item.item_key)} · ${shell.escape(text(item.source_kind))}</small></td><td><strong>${shell.escape(item.target_name)}</strong><small>${shell.escape(shell.short(item.target_id))}</small></td><td>${shell.badge(text(item.status))}<small>${shell.escape(text(item.phase))}</small></td><td>${shell.escape(assignee)}</td><td>${shell.escape(shell.fmt(item.last_observed_at))}</td><td>${due(item.resolution_due_at)}</td><td>${Number(item.occurrence_count || 0)}</td><td><a class="ops-button" href="./work-item-detail.html?id=${encodeURIComponent(item.work_item_id)}">处理</a></td></tr>`;
  }
  async function loadList({ append = false } = {}) {
    const body = document.getElementById("work-item-body");
    const values = { ...state.filters, limit: 50, cursor: append ? state.cursor : null };
    if (!append) body.innerHTML = '<tr><td colspan="9" class="ops-empty">正在加载工作项…</td></tr>';
    try {
      const payload = await KBotAIOpsAuth.request(api + query(values));
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
    form.reset();
    state.filters = {};
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
        status: values.status, priority: values.priority,
        assignee_user_id: values.ownership === "mine" ? state.access.user_id : "",
        unassigned: values.ownership === "unassigned",
      };
      loadList();
    });
    form.addEventListener("reset", () => setTimeout(() => { state.filters = {}; loadList(); }, 0));
    document.getElementById("work-items-refresh").onclick = () => loadList();
    document.getElementById("work-item-more").onclick = () => loadList({ append: true });
    document.getElementById("work-item-queues").onclick = (event) => {
      const button = event.target.closest("[data-queue]");
      if (button) applyQueue(button.dataset.queue);
    };
  }
  function overview(item) {
    return `<article><span>优先级</span>${priority(item.priority)}</article><article><span>状态</span><strong>${shell.escape(text(item.status))}</strong><small>${shell.escape(text(item.phase))}</small></article><article><span>数据库实例</span><strong>${shell.escape(item.target_name)}</strong><small>${shell.escape(shell.short(item.target_id))}</small></article><article><span>责任人</span><strong>${shell.escape(item.assignee_user_id || "未领取")}</strong><small>${shell.escape(item.responsibility_group_name || item.responsibility_group_id || "未设置责任组")}</small></article><article><span>最后发现</span><strong>${shell.escape(shell.fmt(item.last_observed_at))}</strong><small>累计 ${Number(item.occurrence_count)} 次</small></article><article><span>解决 SLA</span><strong>${due(item.resolution_due_at)}</strong><small>验证期限 ${shell.escape(shell.fmt(item.verification_due_at))}</small></article>`;
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
  async function loadAssignmentMembers(groupId, selectedUserId = "") {
    const select = document.getElementById("work-item-assignment").elements.assignee_user_id;
    select.disabled = !groupId;
    select.innerHTML = '<option value="">由组内 DBA 领取</option>';
    if (!groupId) return;
    try {
      const group = await KBotAIOpsAuth.request(`${groupApi}/${encodeURIComponent(groupId)}`);
      const members = (group.members || []).filter((member) => member.status === "ACTIVE");
      const currentIsEligible = members.some((member) => String(member.user_id) === String(selectedUserId));
      select.innerHTML = '<option value="">由组内 DBA 领取</option>'
        + (!selectedUserId || currentIsEligible ? "" : `<option value="${shell.escape(selectedUserId)}" disabled>${shell.escape(selectedUserId)} · 已失效</option>`)
        + members.map((member) => `<option value="${shell.escape(member.user_id)}">${shell.escape(member.user_id)} · ${member.member_role === "LEAD" ? "组长" : "成员"}</option>`).join("");
      select.value = selectedUserId || "";
    } catch (error) {
      select.innerHTML = `<option value="">${shell.escape(error.message)}</option>`;
      select.disabled = true;
    }
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
    const assignment = document.getElementById("work-item-assignment");
    assignment.elements.responsibility_group_id.innerHTML = '<option value="">未分派</option>' + state.groups.map((group) => `<option value="${shell.escape(group.responsibility_group_id)}" ${group.status === "ACTIVE" ? "" : "disabled"}>${shell.escape(group.name)}${group.status === "ACTIVE" ? "" : " · 已停用"}</option>`).join("");
    assignment.elements.responsibility_group_id.value = item.responsibility_group_id || "";
    void loadAssignmentMembers(item.responsibility_group_id || "", item.assignee_user_id || "");
    document.getElementById("work-item-assignment-panel").hidden = !new Set(state.access.permissions || []).has("aiops:work_item_manage");
    const canHandle = new Set(state.access.permissions || []).has("aiops:work_item_handle");
    document.getElementById("work-item-handle-panel").hidden = !canHandle;
    document.getElementById("work-item-claim").hidden = Boolean(item.assignee_user_id || !item.responsibility_group_id);
    document.getElementById("work-item-return").hidden = item.assignee_user_id !== state.access.user_id;
    document.getElementById("work-item-completion").hidden = !["OPEN", "IN_PROGRESS", "WAITING"].includes(item.status);
    const transition = document.getElementById("work-item-transition");
    transition.elements.status.innerHTML = (transitions[item.status] || []).map((value) => `<option value="${value}">${shell.escape(text(value))}</option>`).join("");
    transition.elements.phase.value = item.phase;
    transition.querySelector("button").disabled = !(transitions[item.status] || []).length;
  }
  async function loadDetail() {
    const id = new URLSearchParams(location.search).get("id");
    if (!id) {
      document.getElementById("work-item-overview").innerHTML = '<div class="ops-empty">缺少工作项 ID</div>';
      return;
    }
    try {
      const [item, groupPage] = await Promise.all([
        KBotAIOpsAuth.request(`${api}/${encodeURIComponent(id)}`),
        KBotAIOpsAuth.request(groupApi),
      ]);
      state.groups = groupPage.items || [];
      renderDetail(item);
    } catch (error) {
      document.getElementById("work-item-overview").innerHTML = `<div class="ops-empty">${shell.escape(error.message)}</div>`;
    }
  }
  function bindDetail() {
    document.getElementById("work-item-detail-refresh").onclick = loadDetail;
    document.getElementById("work-item-assignment").onsubmit = async (event) => {
      event.preventDefault();
      const values = Object.fromEntries(new FormData(event.currentTarget));
      try {
        const item = await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(state.item.work_item_id)}/assignment`, {
          method: "PATCH", body: JSON.stringify({ expected_row_version: state.item.row_version, responsibility_group_id: values.responsibility_group_id || null, assignee_user_id: values.assignee_user_id || null }),
        });
        shell.toast("责任分派已更新"); renderDetail(item);
      } catch (error) { shell.toast(error.message); }
    };
    document.getElementById("work-item-assignment").elements.responsibility_group_id.onchange = (event) => {
      void loadAssignmentMembers(event.target.value);
    };
    document.getElementById("work-item-transition").onsubmit = async (event) => {
      event.preventDefault();
      const values = Object.fromEntries(new FormData(event.currentTarget));
      try {
        const item = await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(state.item.work_item_id)}/transitions`, jsonBody({
          expected_row_version: state.item.row_version, status: values.status,
          phase: values.phase, wait_reason: values.wait_reason || null,
          resolution_code: values.resolution_code || null,
          resolution_note: values.resolution_note || null,
        }));
        shell.toast("工作项状态已更新"); renderDetail(item);
      } catch (error) { shell.toast(error.message); }
    };
    document.getElementById("work-item-claim").onclick = async () => {
      try { renderDetail(await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(state.item.work_item_id)}:claim`, jsonBody({ expected_row_version: state.item.row_version }))); shell.toast("工作项已领取"); }
      catch (error) { shell.toast(error.message); }
    };
    document.getElementById("work-item-return").onclick = async () => {
      try { renderDetail(await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(state.item.work_item_id)}:return-to-group`, jsonBody({ expected_row_version: state.item.row_version }))); shell.toast("工作项已退回责任组队列"); }
      catch (error) { shell.toast(error.message); }
    };
    document.getElementById("work-item-completion").onsubmit = async (event) => {
      event.preventDefault(); const values = Object.fromEntries(new FormData(event.currentTarget));
      try { renderDetail(await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(state.item.work_item_id)}:complete`, jsonBody({ expected_row_version: state.item.row_version, resolution_code: values.resolution_code, completion_note: values.completion_note, evidence_refs: [] }))); shell.toast("已标记 DBA 工作完成，等待验证"); }
      catch (error) { shell.toast(error.message); }
    };
  }
  shell.ready.then((access) => {
    state.access = access;
    if (document.body.dataset.page === "work-items") { bindList(); loadList(); }
    if (document.body.dataset.page === "work-item-detail") { bindDetail(); loadDetail(); }
  });
})();
