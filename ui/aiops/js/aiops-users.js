(function () {
  "use strict";
  KBotAIOpsShell.ready.then(async access => {
    if (!access || document.body.dataset.page !== "users") return;
    const byId = id => document.getElementById(id);
    const form = byId("user-form"), status = byId("user-status"), formStatus = byId("user-form-status");
    const api = "/api/v1/apps/aiops/members";
    const userStates = new Map([
      ["ACTIVE", ["已启用", "status-active"]], ["DISABLED", ["已停用", "status-disabled"]],
    ]);
    let busy = false, loading = false, roleLoading = false, rolesReady = false, roleCount = 0, confirming = false;
    const protectedUser = row => row.protected === true || row.membership?.initial_admin === true;
    const currentUser = row => String(row.user_id).toLowerCase() === String(access.user_id).toLowerCase();
    function controls() {
      byId("user-new").disabled = busy || loading || roleLoading || !rolesReady || !roleCount;
      byId("user-refresh").disabled = busy || loading || roleLoading;
      form.querySelectorAll("input, select").forEach(node => { node.disabled = busy; });
      document.querySelectorAll("[data-user-close]").forEach(node => { node.disabled = busy; });
      byId("user-create-submit").disabled = busy || !rolesReady || !roleCount;
      byId("user-create-submit").textContent = busy ? "处理中…" : "创建用户";
      byId("user-list").querySelectorAll("[data-user-delete]").forEach(node => { node.disabled = busy || loading || node.dataset.blocked === "true"; });
    }
    function reset() {
      form.reset(); formStatus.textContent = "";
      form.querySelectorAll("details").forEach(node => { node.open = false; });
    }
    function close() {
      if (!busy) KBotDrawer.close("user-create-drawer");
    }
    byId("user-create-drawer").addEventListener("drawer:closed", reset);
    document.querySelectorAll("[data-user-close]").forEach(node => { node.onclick = close; });
    async function load() {
      loading = true; controls();
      const container = byId("user-list"), count = byId("user-count");
      let rows;
      try { rows = await KBotAIOpsAuth.request(api); }
      catch (error) {
        const message = document.createElement("li"); message.className = "resource-list-empty";
        message.textContent = "用户列表读取失败，请刷新后重试。";
        container.replaceChildren(message); count.textContent = "暂不可用";
        status.textContent = error.message || "用户列表读取失败，请刷新后重试。";
        return false;
      }
      finally { loading = false; controls(); }
      const users = (rows.items || rows).filter(row => row.account_origin === "APP" && row.owner_app_id === "aiops");
      container.replaceChildren();
      count.textContent = users.length + " 位用户";
      if (!users.length) {
        const message = document.createElement("li"); message.className = "resource-list-empty";
        message.textContent = "暂无 AIOps 自建用户。"; container.append(message);
      }
      for (const row of users) {
        const line = document.createElement("li"), identity = document.createElement("div"), name = document.createElement("strong");
        const userId = document.createElement("small"), state = document.createElement("span"), button = document.createElement("a");
        line.className = "user-list-row"; identity.className = "user-list-identity";
        name.className = "user-list-name"; name.textContent = row.display_name || row.user_id;
        userId.className = "user-list-id"; userId.textContent = row.user_id;
        const presentation = userStates.get(row.status) || ["状态未知", "status-muted"];
        state.className = "status-pill " + presentation[1]; state.textContent = presentation[0];
        button.className = "row-action"; button.textContent = "编辑／飞书绑定";
        button.href = "./user-detail.html?" + new URLSearchParams({userId: row.user_id});
        const actions = document.createElement("div"), remove = document.createElement("button");
        actions.className = "user-list-actions";
        remove.type = "button"; remove.className = "danger-row-action"; remove.textContent = "删除";
        remove.dataset.userDelete = row.user_id;
        remove.dataset.blocked = String(protectedUser(row) || currentUser(row));
        if (protectedUser(row)) remove.title = "受保护账号不能删除";
        else if (currentUser(row)) remove.title = "不能删除当前登录账号";
        remove.onclick = () => removeUser(row);
        button.onclick = event => { if (busy || confirming) event.preventDefault(); };
        actions.append(button, remove);
        identity.append(name, userId); line.append(identity, state, actions); container.append(line);
      }
      controls(); return true;
    }
    async function removeUser(row) {
      if (busy || loading || confirming || protectedUser(row) || currentUser(row)) return;
      const trigger = document.activeElement;
      confirming = true;
      try {
        if (!await KBotAIOpsShell.confirmAction({title:"删除用户“" + (row.display_name || row.user_id) + "”？",
          message:"将永久删除账号“" + row.user_id + "”及其登录凭据、角色和业务空间授权，该账号将无法继续登录。此操作不可恢复。",
          confirmLabel:"删除用户", tone:"danger"})) return;
        busy = true; controls();
        await KBotAIOpsAuth.request(api + "/" + encodeURIComponent(row.user_id), {method:"DELETE"});
        status.textContent = "用户已删除。";
        if (!await load()) status.textContent = "用户已删除，但列表刷新失败，请刷新后重试。";
      } catch (error) { status.textContent = error.message || "删除用户失败，请刷新后重试。"; }
      finally {
        confirming = false; busy = false; controls();
        if (trigger?.isConnected) trigger.focus({preventScroll:true});
        else byId("user-refresh").focus({preventScroll:true});
      }
    }
    form.onsubmit = async event => {
      event.preventDefault(); if (busy || confirming || !rolesReady || !roleCount) return;
      const userId = form.elements.user_id.value, openId = form.elements.feishu_open_id.value.trim() || null;
      const body = {display_name: form.elements.display_name.value || null, feishu_open_id: openId,
        user_id: userId, password: form.elements.password.value, must_change_password: true,
        role_bindings: [{role_code: form.elements.role_code.value, scope_mode: "SELECTED_DOMAINS", domain_ids: [Number(access.domain_id)]}]};
      busy = true; controls(); formStatus.textContent = "正在创建用户…";
      let created = false;
      try {
        await KBotAIOpsAuth.request(api, {method: "POST", body: JSON.stringify(body)});
        created = true;
      } catch (error) { formStatus.textContent = error.message || "创建用户失败，请核对填写内容。"; }
      finally { busy = false; controls(); }
      if (created) {
        close();
        status.textContent = openId ? "用户已创建；候选飞书身份需本人在“我的通知设置”中完成验证。" : "用户已创建。";
        if (!await load()) status.textContent = "用户已创建，但列表刷新失败，请刷新后重试。";
        byId("user-new").focus({preventScroll:true});
      }
    };
    document.getElementById("user-new").onclick = () => {
      if (busy || loading || confirming || !rolesReady || !roleCount) return;
      reset(); controls(); KBotDrawer.open("user-create-drawer");
    };
    async function loadRoles() {
      roleLoading = true; rolesReady = false; roleCount = 0; controls();
      form.elements.role_code.length = 1;
      try {
        const roles = await KBotAIOpsAuth.request(api + "/assignable-roles");
        for (const role of roles.items || roles) {
          if (role.status !== "ACTIVE") continue;
          const option = document.createElement("option"); option.value = role.role_code; option.textContent = role.display_name || role.role_code; form.elements.role_code.append(option); ++roleCount;
        }
        rolesReady = true;
        if (!roleCount) status.textContent = "当前账号没有可在本空间分配的角色，暂不能新建用户；仍可管理已有用户。";
      } catch (error) { status.textContent = "初始角色读取失败，暂不能新建用户；仍可管理已有用户，请刷新后重试。"; }
      finally { roleLoading = false; controls(); }
    }
    byId("user-refresh").onclick = async () => {
      if (busy || loading || roleLoading || confirming) return;
      status.textContent = "";
      await Promise.all([loadRoles(), load()]);
    };
    await byId("user-refresh").onclick();
  });
})();
