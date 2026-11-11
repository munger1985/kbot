(function () {
  "use strict";
  KBotAIOpsShell.ready.then(async access => {
    if (!access || document.body.dataset.page !== "user-detail") return;
    const byId = id => document.getElementById(id);
    const userId = new URLSearchParams(location.search).get("userId");
    const status = byId("user-detail-status"), content = byId("user-detail-content");
    const bindingForm = byId("user-binding-form"), profileForm = byId("user-profile-form");
    const challenge = byId("user-binding-challenge");
    const api = "/api/v1/apps/aiops/members/" + encodeURIComponent(userId || "");
    let selected = null, caps = [], busy = false;
    function controls() {
      content.querySelectorAll("input, button").forEach(node => { node.disabled = busy || !selected; });
      byId("user-detail-refresh").disabled = busy;
      challenge.hidden = selected?.feishu_binding?.status !== "UNVERIFIED" || !caps.includes("binding");
      challenge.disabled = busy || bindingForm.elements.feishu_open_id.value.trim() !== selected?.feishu_binding?.external_id;
    }
    function render() {
      byId("user-detail-name").textContent = selected.display_name || selected.user_id;
      byId("user-detail-id").textContent = selected.user_id;
      byId("user-detail-state").textContent = ({ACTIVE:"已启用", DISABLED:"已停用"})[selected.status] || "状态未知";
      profileForm.elements.display_name.value = selected.display_name || "";
      bindingForm.elements.feishu_open_id.value = selected.feishu_binding?.external_id || "";
      const state = ({VERIFIED:["已验证", "status-success"], UNVERIFIED:["待本人验证", "status-warning"], DISABLED:["已解绑", "status-muted"]})[selected.feishu_binding?.status]
        || [selected.feishu_binding ? "状态待确认" : "未绑定", "status-muted"];
      byId("user-binding-status").textContent = state[0];
      byId("user-binding-status").className = "status-pill " + state[1];
      content.hidden = false;
    }
    async function load() {
      selected = null; controls();
      try {
        const [user, capabilities] = await Promise.all([
          KBotAIOpsAuth.request(api),
          KBotAIOpsAuth.request("/api/v1/notification-delivery/capabilities").catch(() => ({capabilities:[]})),
        ]);
        selected = user; caps = capabilities.capabilities || []; render();
        return true;
      } catch (error) {
        content.hidden = true;
        status.textContent = error.message || "用户读取失败，请刷新后重试。";
        return false;
      } finally { controls(); }
    }
    async function action(task, message) {
      if (busy || !selected) return;
      busy = true; controls(); status.textContent = "处理中…";
      try {
        await task();
        // 保存后重新读取绑定版本，后续保存仍携带最新的并发校验字段。
        if (await load()) status.textContent = message;
      } catch (error) { status.textContent = error.message || "操作未完成，请核对当前状态。"; }
      finally { busy = false; controls(); }
    }
    bindingForm.onsubmit = event => {
      event.preventDefault();
      const body = {feishu_open_id: bindingForm.elements.feishu_open_id.value.trim() || null,
        expected_binding_id:selected?.feishu_binding?.id || null, expected_binding_version:selected?.feishu_binding?.row_version || null};
      void action(() => KBotAIOpsAuth.request(api, {method:"PATCH", body:JSON.stringify(body)}), "绑定已保存；新增或更换身份需本人验证。");
    };
    profileForm.onsubmit = event => {
      event.preventDefault();
      const body = {display_name:profileForm.elements.display_name.value.trim() || null};
      void action(() => KBotAIOpsAuth.request(api, {method:"PATCH", body:JSON.stringify(body)}), "用户资料已保存。");
    };
    bindingForm.elements.feishu_open_id.oninput = controls;
    challenge.onclick = () => {
      if (challenge.hidden || challenge.disabled || !selected?.feishu_binding) return;
      const id = selected.feishu_binding.id;
      void action(() => KBotAIOpsAuth.request("/api/v1/notification-delivery/bindings/" + encodeURIComponent(id) + "/challenge", {method:"POST", body:"{}"}), "验证码已入队，请本人登录“我的通知设置”确认。");
    };
    byId("user-detail-refresh").onclick = async () => {
      if (busy) return;
      busy = true; controls(); status.textContent = "正在读取用户…";
      try { if (await load()) status.textContent = ""; }
      finally { busy = false; controls(); }
    };
    if (!userId || userId.length > 256) {
      status.textContent = "缺少有效的用户 ID，请返回用户列表重新选择。";
      byId("user-detail-refresh").disabled = true;
      return;
    }
    await byId("user-detail-refresh").onclick();
  });
})();
