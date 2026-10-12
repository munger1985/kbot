(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/responsibility-groups";
  const shell = globalThis.KBotAIOpsShell;
  let groups = [];
  let editing = null;
  let selected = null;
  let candidates = [];
  let confirmResolver = null;

  const values = (value) => Array.isArray(value) ? value : [];
  const candidateName = (userId) => {
    const candidate = candidates.find((item) => String(item.user_id) === String(userId));
    return candidate?.display_name || candidate?.username || userId;
  };

  function renderGroups() {
    const body = document.getElementById("groups-body");
    body.innerHTML = groups.map((group) => `<tr>
      <td><strong>${shell.escape(group.name)}</strong><small>${shell.escape(group.description || "—")}</small></td>
      <td>${shell.badge(group.status)}</td>
      <td>${shell.escape(group.lead_user_id || "未设置")}</td>
      <td>${Number(group.active_member_count || 0)}</td>
      <td><span class="ops-actions"><button type="button" data-edit-group="${shell.escape(group.responsibility_group_id)}">修改</button><button type="button" data-members-group="${shell.escape(group.responsibility_group_id)}">人员</button><button type="button" class="danger" data-delete-group="${shell.escape(group.responsibility_group_id)}" ${group.status === "INACTIVE" ? "disabled title=\"责任组已停用\"" : ""}>删除</button></span></td>
    </tr>`).join("") || '<tr><td colspan="5" class="ops-empty">尚未创建责任组</td></tr>';
    body.querySelectorAll("[data-edit-group]").forEach((button) => {
      button.addEventListener("click", () => openEditor(groups.find((item) => item.responsibility_group_id === button.dataset.editGroup)));
    });
    body.querySelectorAll("[data-members-group]").forEach((button) => {
      button.addEventListener("click", () => void openMembers(button.dataset.membersGroup));
    });
    body.querySelectorAll("[data-delete-group]").forEach((button) => {
      button.addEventListener("click", () => void deleteGroup(groups.find((item) => item.responsibility_group_id === button.dataset.deleteGroup)));
    });
  }

  function renderMembers() {
    if (!selected) return;
    document.getElementById("group-members-title").textContent = `${selected.name} · 人员`;
    document.getElementById("group-members-status").textContent = selected.status === "ACTIVE" ? "当前可承接新工作项" : "已停用，不再承接新工作项";
    const memberContainer = document.getElementById("group-members");
    memberContainer.innerHTML = values(selected.members).map((member) => `<div class="ops-responsibility-group-member">
      <span><strong>${shell.escape(candidateName(member.user_id))}</strong><small>${member.member_role === "LEAD" ? "组长" : "成员"} · ${Number(member.active_work_item_count || 0)} 个活动工作项${member.status === "ACTIVE" ? "" : " · 已停用"}</small></span>
      <span class="ops-actions">${member.member_role === "LEAD" || member.status !== "ACTIVE" ? "" : `<button type="button" data-lead-user="${shell.escape(member.user_id)}">设为组长</button>`}<button type="button" class="danger" data-remove-user="${shell.escape(member.user_id)}" ${Number(member.active_work_item_count || 0) > 0 || member.status !== "ACTIVE" ? "disabled" : ""}>移出</button></span>
    </div>`).join("") || '<p class="ops-empty">尚未添加人员</p>';
    memberContainer.querySelectorAll("[data-lead-user]").forEach((button) => {
      button.addEventListener("click", () => void setLead(button.dataset.leadUser));
    });
    memberContainer.querySelectorAll("[data-remove-user]").forEach((button) => {
      button.addEventListener("click", () => void removeMember(button.dataset.removeUser));
    });
    const memberSelect = document.getElementById("group-member-form").elements.user_id;
    const activeUserIds = new Set(values(selected.members).filter((member) => member.status === "ACTIVE").map((member) => String(member.user_id)));
    memberSelect.innerHTML = '<option value="">请选择人员</option>' + candidates.filter((candidate) => !activeUserIds.has(String(candidate.user_id))).map((candidate) => `<option value="${shell.escape(candidate.user_id)}">${shell.escape(candidate.display_name || candidate.username || candidate.user_id)}</option>`).join("");
  }

  async function load() {
    const body = document.getElementById("groups-body");
    try {
      const page = await KBotAIOpsAuth.request(api);
      groups = values(page.items).filter((item) => item.status === "ACTIVE");
      renderGroups();
    } catch (error) {
      body.innerHTML = `<tr><td colspan="5" class="ops-empty">${shell.escape(error.message)}</td></tr>`;
    }
  }

  function openEditor(group = null) {
    editing = group || null;
    const form = document.getElementById("group-editor-form");
    form.reset();
    form.elements.name.value = editing?.name || "";
    form.elements.description.value = editing?.description || "";
    document.getElementById("group-editor-title").textContent = editing ? "修改责任组" : "新增责任组";
    document.getElementById("group-editor-hint").textContent = editing ? "修改责任组名称或说明；既有工作项历史不会改变。" : "建立 DBA 工作路由边界，创建后再添加具体人员。";
    document.getElementById("group-editor-submit").textContent = editing ? "保存修改" : "创建责任组";
    document.getElementById("group-editor-dialog").showModal();
  }

  async function saveEditor(form) {
    const fields = Object.fromEntries(new FormData(form));
    const payload = { name: fields.name.trim(), description: fields.description.trim() || null };
    if (editing) {
      payload.expected_row_version = editing.row_version;
    }
    try {
      await KBotAIOpsAuth.request(editing ? `${api}/${encodeURIComponent(editing.responsibility_group_id)}` : api, {
        method: editing ? "PATCH" : "POST",
        body: JSON.stringify(payload),
      });
      document.getElementById("group-editor-dialog").close();
      shell.toast(editing ? "责任组已更新" : "责任组已创建");
      editing = null;
      await load();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function openMembers(groupId) {
    try {
      [selected, candidates] = await Promise.all([
        KBotAIOpsAuth.request(`${api}/${encodeURIComponent(groupId)}`),
        KBotAIOpsAuth.request(`${api}/${encodeURIComponent(groupId)}/member-candidates`).then((page) => values(page.items)),
      ]);
      renderMembers();
      const dialog = document.getElementById("group-members-dialog");
      if (!dialog.open) dialog.showModal();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function refreshMembers() {
    if (!selected) return;
    const groupId = selected.responsibility_group_id;
    await load();
    await openMembers(groupId);
  }

  async function addMember(form) {
    if (!selected) return;
    const fields = Object.fromEntries(new FormData(form));
    try {
      await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(selected.responsibility_group_id)}/members/${encodeURIComponent(fields.user_id)}`, {
        method: "PUT",
        body: JSON.stringify({ member_role: "MEMBER" }),
      });
      form.reset();
      shell.toast("人员已加入责任组；其 AIOps 权限未发生变化");
      await refreshMembers();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function setLead(userId) {
    if (!selected) return;
    try {
      await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(selected.responsibility_group_id)}/members/${encodeURIComponent(userId)}`, {
        method: "PUT",
        body: JSON.stringify({ member_role: "LEAD" }),
      });
      shell.toast("责任组组长已更新");
      await refreshMembers();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  function askConfirmation({ title, message, confirmLabel }) {
    if (confirmResolver) settleConfirmation(false);
    document.getElementById("group-confirm-title").textContent = title;
    document.getElementById("group-confirm-message").textContent = message;
    document.getElementById("group-confirm-submit").textContent = confirmLabel;
    document.getElementById("group-confirm-dialog").showModal();
    return new Promise((resolve) => { confirmResolver = resolve; });
  }

  function settleConfirmation(confirmed) {
    const resolve = confirmResolver;
    confirmResolver = null;
    const dialog = document.getElementById("group-confirm-dialog");
    if (dialog.open) dialog.close();
    resolve?.(confirmed);
  }

  async function removeMember(userId) {
    if (!selected) return;
    const name = candidateName(userId);
    if (!await askConfirmation({
      title: "移出责任组人员？",
      message: `确认将“${name}”移出“${selected.name}”？该操作不修改用户的 AIOps 权限。`,
      confirmLabel: "确认移出",
    })) return;
    try {
      await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(selected.responsibility_group_id)}/members/${encodeURIComponent(userId)}`, { method: "DELETE" });
      shell.toast("人员已移出责任组");
      await refreshMembers();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function deleteGroup(group) {
    if (!group || group.status === "INACTIVE") return;
    try {
      const detail = await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(group.responsibility_group_id)}`);
      const assignedCount = values(detail.members).reduce((total, member) => total + Number(member.active_work_item_count || 0), 0);
      if (Number(group.target_binding_count || 0) || assignedCount) {
        shell.toast("请先在运维目标中改派关联 Target，并处理或改派组内活动工作项");
        return;
      }
      if (!await askConfirmation({
        title: "删除责任组？",
        message: `“${group.name}”将被停用，不能再承接新工作项；为保留审计和历史引用，系统不会物理删除既有记录。`,
        confirmLabel: "确认删除",
      })) return;
      await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(group.responsibility_group_id)}`, {
        method: "PATCH",
        body: JSON.stringify({ expected_row_version: group.row_version, status: "INACTIVE" }),
      });
      shell.toast("责任组已删除并保留历史记录");
      await load();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  shell.ready.then(() => {
    document.getElementById("groups-refresh").addEventListener("click", () => void load());
    document.getElementById("group-create-open").addEventListener("click", () => openEditor());
    document.querySelectorAll("[data-close-group-editor]").forEach((button) => button.addEventListener("click", () => document.getElementById("group-editor-dialog").close()));
    document.querySelectorAll("[data-close-group-members]").forEach((button) => button.addEventListener("click", () => document.getElementById("group-members-dialog").close()));
    document.getElementById("group-editor-form").addEventListener("submit", (event) => {
      event.preventDefault();
      void saveEditor(event.currentTarget);
    });
    document.getElementById("group-member-form").addEventListener("submit", (event) => {
      event.preventDefault();
      void addMember(event.currentTarget);
    });
    document.getElementById("group-confirm-form").addEventListener("submit", (event) => {
      event.preventDefault();
      settleConfirmation(true);
    });
    document.querySelector("[data-cancel-group-confirm]").addEventListener("click", () => settleConfirmation(false));
    document.getElementById("group-confirm-dialog").addEventListener("cancel", (event) => {
      event.preventDefault();
      settleConfirmation(false);
    });
    void load();
  });
})();
