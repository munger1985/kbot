(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/responsibility-groups";
  const shell = globalThis.KBotAIOpsShell;
  let groups = [];
  let selected = null;
  let candidates = [];

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
      <td><strong>${Number(group.target_binding_count || 0)} 个 Target</strong><small>${Number(group.agent_binding_count || 0)} 个 Agent</small></td>
      <td>${Number(group.unassigned_work_item_count || 0)}</td>
      <td><button type="button" data-group-id="${shell.escape(group.responsibility_group_id)}">维护</button></td>
    </tr>`).join("") || '<tr><td colspan="7" class="ops-empty">尚未创建责任组</td></tr>';
    body.querySelectorAll("[data-group-id]").forEach((button) => {
      button.addEventListener("click", () => void openGroup(button.dataset.groupId));
    });
  }

  function renderMaintenance() {
    const panel = document.getElementById("group-maintenance");
    if (!selected) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    document.getElementById("group-maintenance-title").textContent = `${selected.name} · 成员`;
    document.getElementById("group-maintenance-status").textContent = selected.status === "ACTIVE" ? "当前可承接新工作项" : "已停用，不再承接新工作项";
    const memberContainer = document.getElementById("group-members");
    memberContainer.innerHTML = values(selected.members).map((member) => `<div class="ops-responsibility-group-member">
      <span><strong>${shell.escape(candidateName(member.user_id))}</strong><small>${member.member_role === "LEAD" ? "组长" : "成员"} · ${Number(member.active_work_item_count || 0)} 个活动工作项${member.status === "ACTIVE" ? "" : " · 已停用"}</small></span>
      <button type="button" class="danger" data-remove-user="${shell.escape(member.user_id)}" ${Number(member.active_work_item_count || 0) > 0 || member.status !== "ACTIVE" ? "disabled" : ""}>移出</button>
    </div>`).join("") || '<p class="ops-empty">尚未添加成员</p>';
    memberContainer.querySelectorAll("[data-remove-user]").forEach((button) => {
      button.addEventListener("click", () => void removeMember(button.dataset.removeUser));
    });
    const memberSelect = document.getElementById("group-member-form").elements.user_id;
    memberSelect.innerHTML = '<option value="">请选择</option>' + candidates.map((candidate) => `<option value="${shell.escape(candidate.user_id)}">${shell.escape(candidate.display_name || candidate.username || candidate.user_id)}</option>`).join("");
    const toggle = document.getElementById("group-status-toggle");
    toggle.textContent = selected.status === "ACTIVE" ? "停用责任组" : "启用责任组";
  }

  async function load() {
    const body = document.getElementById("groups-body");
    try {
      const page = await KBotAIOpsAuth.request(api);
      groups = values(page.items);
      renderGroups();
      if (selected) {
        const current = groups.find((item) => item.responsibility_group_id === selected.responsibility_group_id);
        if (current) await openGroup(current.responsibility_group_id);
        else { selected = null; renderMaintenance(); }
      }
    } catch (error) {
      body.innerHTML = `<tr><td colspan="7" class="ops-empty">${shell.escape(error.message)}</td></tr>`;
    }
  }

  async function openGroup(groupId) {
    try {
      [selected, candidates] = await Promise.all([
        KBotAIOpsAuth.request(`${api}/${encodeURIComponent(groupId)}`),
        KBotAIOpsAuth.request(`${api}/${encodeURIComponent(groupId)}/member-candidates`).then((page) => values(page.items)),
      ]);
      renderMaintenance();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function addMember(event) {
    event.preventDefault();
    if (!selected) return;
    const fields = Object.fromEntries(new FormData(event.currentTarget));
    try {
      await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(selected.responsibility_group_id)}/members/${encodeURIComponent(fields.user_id)}`, {
        method: "PUT",
        body: JSON.stringify({ member_role: fields.member_role }),
      });
      event.currentTarget.reset();
      shell.toast(fields.member_role === "LEAD" ? "组长已更新" : "成员已加入责任组");
      await load();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function removeMember(userId) {
    if (!selected) return;
    try {
      await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(selected.responsibility_group_id)}/members/${encodeURIComponent(userId)}`, { method: "DELETE" });
      shell.toast("成员已移出责任组");
      await load();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function toggleStatus() {
    if (!selected) return;
    const disabling = selected.status === "ACTIVE";
    try {
      await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(selected.responsibility_group_id)}`, {
        method: "PATCH",
        body: JSON.stringify({ expected_row_version: selected.row_version, status: disabling ? "INACTIVE" : "ACTIVE" }),
      });
      shell.toast(disabling ? "责任组已停用" : "责任组已启用");
      await load();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  shell.ready.then(() => {
    document.getElementById("groups-refresh").addEventListener("click", () => void load());
    document.getElementById("group-member-form").addEventListener("submit", addMember);
    document.getElementById("group-status-toggle").addEventListener("click", () => void toggleStatus());
    document.getElementById("group-create").addEventListener("submit", async (event) => {
      event.preventDefault();
      const fields = Object.fromEntries(new FormData(event.currentTarget));
      try {
        await KBotAIOpsAuth.request(api, {
          method: "POST",
          body: JSON.stringify({ name: fields.name.trim(), description: fields.description.trim() || null }),
        });
        event.currentTarget.reset();
        shell.toast("责任组已创建");
        await load();
      } catch (error) {
        shell.toast(error.message);
      }
    });
    void load();
  });
})();
