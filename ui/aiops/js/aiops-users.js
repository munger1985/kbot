(function () {
  "use strict";
  KBotAIOpsShell.ready.then(async access => {
    if (!access || document.body.dataset.page !== "users") return;
    const form = document.getElementById("user-form"), status = document.getElementById("user-status");
    const api = "/api/v1/apps/aiops/members";
    const userStates = new Map([
      ["ACTIVE", ["已启用", "status-active"]], ["DISABLED", ["已停用", "status-disabled"]],
    ]);
    let selected = null, busy = false;
    function showBindingStatus(binding) {
      const state = ({VERIFIED: ["已验证", "status-success"], UNVERIFIED: ["待本人验证", "status-warning"],
        DISABLED: ["已解绑", "status-muted"]})[binding?.status] || [binding ? "状态待确认" : "未绑定", "status-muted"];
      const element = document.getElementById("user-binding-status");
      element.textContent = state[0]; element.className = "status-pill " + state[1];
    }
    function reset() {
      selected = null; form.reset(); form.elements.user_id.disabled = false;
      for (const name of ["password", "role_code"]) { form.elements[name].disabled = false; form.elements[name].required = true; }
      document.getElementById("user-form-title").textContent = "新建用户";
      showBindingStatus(null);
    }
    async function edit(userId) {
      try {
        selected = await KBotAIOpsAuth.request(api + "/" + encodeURIComponent(userId));
        form.elements.user_id.value = selected.user_id; form.elements.user_id.disabled = true;
        form.elements.display_name.value = selected.display_name || "";
        form.elements.feishu_open_id.value = selected.feishu_binding?.external_id || "";
        for (const name of ["password", "role_code"]) { form.elements[name].disabled = true; form.elements[name].required = false; }
        form.elements.password.value = "";
        document.getElementById("user-form-title").textContent = "编辑用户";
        showBindingStatus(selected.feishu_binding);
      } catch (error) { status.textContent = error.message; }
    }
    async function load() {
      const container = document.getElementById("user-list"), count = document.getElementById("user-count");
      let rows;
      try { rows = await KBotAIOpsAuth.request(api); }
      catch (error) {
        const message = document.createElement("li"); message.className = "resource-list-empty";
        message.textContent = "用户列表读取失败，请刷新后重试。";
        container.replaceChildren(message); count.textContent = "暂不可用";
        throw error;
      }
      const users = (rows.items || rows).filter(row => row.account_origin === "APP" && row.owner_app_id === "aiops");
      container.replaceChildren();
      count.textContent = users.length + " 位用户";
      if (!users.length) {
        const message = document.createElement("li"); message.className = "resource-list-empty";
        message.textContent = "暂无 AIOps 自建用户。"; container.append(message);
      }
      for (const row of users) {
        const line = document.createElement("li"), identity = document.createElement("div"), name = document.createElement("strong");
        const userId = document.createElement("small"), state = document.createElement("span"), button = document.createElement("button");
        line.className = "user-list-row"; identity.className = "user-list-identity";
        name.className = "user-list-name"; name.textContent = row.display_name || row.user_id;
        userId.className = "user-list-id"; userId.textContent = row.user_id;
        const presentation = userStates.get(row.status) || ["状态未知", "status-muted"];
        state.className = "status-pill " + presentation[1]; state.textContent = presentation[0];
        button.className = "row-action"; button.textContent = "编辑／飞书绑定";
        button.type = "button"; button.onclick = () => edit(row.user_id);
        identity.append(name, userId); line.append(identity, state, button); container.append(line);
      }
    }
    form.onsubmit = async event => {
      event.preventDefault(); if (busy) return; busy = true;
      const userId = form.elements.user_id.value, openId = form.elements.feishu_open_id.value.trim() || null;
      const body = {display_name: form.elements.display_name.value || null, feishu_open_id: openId};
      if (selected) {
        body.expected_binding_id = selected.feishu_binding?.id || null;
        body.expected_binding_version = selected.feishu_binding?.row_version || null;
      } else {
        Object.assign(body, {user_id: userId, password: form.elements.password.value, must_change_password: true,
          role_bindings: [{role_code: form.elements.role_code.value, scope_mode: "SELECTED_DOMAINS", domain_ids: [Number(access.domain_id)]}]});
      }
      try {
        await KBotAIOpsAuth.request(api + (selected ? "/" + encodeURIComponent(userId) : ""), {method: selected ? "PATCH" : "POST", body: JSON.stringify(body)});
        status.textContent = "用户已保存；候选飞书身份需本人在“我的通知设置”中发送验证码并确认。";
        reset(); await load();
      } catch (error) { status.textContent = error.message; }
      finally { busy = false; }
    };
    document.getElementById("user-new").onclick = reset;
    try {
      const roles = await KBotAIOpsAuth.request(api + "/assignable-roles");
      for (const role of roles.items || roles) {
        if (role.status !== "ACTIVE") continue;
        const option = document.createElement("option"); option.value = role.role_code; option.textContent = role.display_name || role.role_code; form.elements.role_code.append(option);
      }
      if (!form.elements.role_code.options.length) status.textContent = "当前账号没有可在本空间分配的角色；仍可编辑已有用户的飞书绑定。";
      await load();
    } catch (error) { status.textContent = error.message; }
  });
})();
