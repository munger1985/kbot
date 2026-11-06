(function () {
  "use strict";
  KBotAIOpsShell.ready.then(async access => {
    if (!access || document.body.dataset.page !== "users") return;
    const form = document.getElementById("user-form"), status = document.getElementById("user-status");
    const api = "/api/v1/apps/aiops/members";
    let selected = null, busy = false;
    function reset() {
      selected = null; form.reset(); form.elements.user_id.disabled = false;
      for (const name of ["password", "role_code"]) { form.elements[name].disabled = false; form.elements[name].required = true; }
      document.getElementById("user-form-title").textContent = "新建用户";
      document.getElementById("user-binding-status").textContent = "未绑定";
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
        document.getElementById("user-binding-status").textContent = selected.feishu_binding?.status || "未绑定";
      } catch (error) { status.textContent = error.message; }
    }
    async function load() {
      const rows = await KBotAIOpsAuth.request(api), container = document.getElementById("user-list");
      container.replaceChildren();
      for (const row of rows.items || rows) {
        if (row.account_origin !== "APP" || row.owner_app_id !== "aiops") continue;
        const line = document.createElement("div"), label = document.createElement("span"), button = document.createElement("button");
        label.textContent = (row.display_name || row.user_id) + " · " + row.status;
        button.textContent = "编辑／飞书绑定"; button.type = "button"; button.onclick = () => edit(row.user_id);
        line.append(label, button); container.append(line);
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
        status.textContent = "用户已保存；候选飞书身份需本人在通知渠道页面发送验证码并确认。";
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
