(function () {
  "use strict";
  KBotAIOpsShell.ready.then(async access => {
    if (!access || document.body.dataset.page !== "users") return;
    const form = document.getElementById("user-form"), status = document.getElementById("user-status");
    const api = "/api/v1/apps/aiops/members";
    const userStates = new Map([
      ["ACTIVE", ["已启用", "status-active"]], ["DISABLED", ["已停用", "status-disabled"]],
    ]);
    let busy = false;
    function reset() {
      form.reset(); form.elements.user_id.disabled = false;
      for (const name of ["password", "role_code"]) { form.elements[name].disabled = false; form.elements[name].required = true; }
      document.getElementById("user-form-title").textContent = "新建用户";
      const bindingStatus = document.getElementById("user-binding-status");
      bindingStatus.textContent = "未绑定"; bindingStatus.className = "status-pill status-muted";
    }
    function focusForm(field) {
      form.closest("section").scrollIntoView({block: "start"});
      field.focus({preventScroll: true});
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
        const userId = document.createElement("small"), state = document.createElement("span"), button = document.createElement("a");
        line.className = "user-list-row"; identity.className = "user-list-identity";
        name.className = "user-list-name"; name.textContent = row.display_name || row.user_id;
        userId.className = "user-list-id"; userId.textContent = row.user_id;
        const presentation = userStates.get(row.status) || ["状态未知", "status-muted"];
        state.className = "status-pill " + presentation[1]; state.textContent = presentation[0];
        button.className = "row-action"; button.textContent = "编辑／飞书绑定";
        button.href = "./user-detail.html?" + new URLSearchParams({userId: row.user_id});
        identity.append(name, userId); line.append(identity, state, button); container.append(line);
      }
    }
    form.onsubmit = async event => {
      event.preventDefault(); if (busy) return; busy = true;
      const userId = form.elements.user_id.value, openId = form.elements.feishu_open_id.value.trim() || null;
      const body = {display_name: form.elements.display_name.value || null, feishu_open_id: openId,
        user_id: userId, password: form.elements.password.value, must_change_password: true,
        role_bindings: [{role_code: form.elements.role_code.value, scope_mode: "SELECTED_DOMAINS", domain_ids: [Number(access.domain_id)]}]};
      try {
        await KBotAIOpsAuth.request(api, {method: "POST", body: JSON.stringify(body)});
        status.textContent = "用户已保存；候选飞书身份需本人在“我的通知设置”中发送验证码并确认。";
        reset(); await load();
      } catch (error) { status.textContent = error.message; }
      finally { busy = false; }
    };
    document.getElementById("user-new").onclick = () => {
      reset(); status.textContent = ""; focusForm(form.elements.user_id);
    };
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
