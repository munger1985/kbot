import { confirmAction } from "./kbot-confirm.js";

const root = "/notification-delivery";

export function mountFeishuSettings(container, request) {
  container.innerHTML = `
    <p class="feishu-settings-help">绑定并验证你的飞书身份后可接收飞书通知。未绑定不影响站内通知。</p>
    <p class="feishu-settings-state"><strong>绑定状态：</strong><span data-state>未读取</span></p>
    <form data-binding-form>
      <label>飞书 open_id<input name="open_id" required pattern="ou_.+" maxlength="256" spellcheck="false" autocomplete="off" placeholder="ou_…"></label>
      <small>使用平台通知渠道所用的同一个飞书应用的 open_id；如不确定，请联系渠道管理员确认应用。</small>
      <div class="feishu-settings-actions"><button type="submit">保存绑定</button><button type="button" data-unbind hidden>解绑</button></div>
    </form>
    <form data-verify-form hidden>
      <label>飞书验证码<input name="code" required inputmode="numeric" pattern="[0-9]{6}" maxlength="6" autocomplete="one-time-code"></label>
      <div class="feishu-settings-actions"><button type="button" data-send>发送验证码</button><button type="submit">确认是本人</button></div>
      <small>验证码仅由本人确认，限时限次，请勿转发。</small>
    </form>
    <p class="feishu-settings-result" data-result role="status"></p>`;
  const find = (selector) => container.querySelector(selector);
  const bindingForm = find("[data-binding-form]"), verifyForm = find("[data-verify-form]");
  const openId = bindingForm.elements.open_id, code = verifyForm.elements.code;
  const state = find("[data-state]"), result = find("[data-result]"), unbind = find("[data-unbind]");
  let binding = null, busy = false, loaded = false;

  function show(message, error = false) {
    result.textContent = message;
    result.classList.toggle("is-error", error);
  }
  function sync() {
    for (const control of container.querySelectorAll("button,input")) control.disabled = busy || !loaded;
  }
  async function load() {
    const data = await request(root + "/me/bindings");
    const active = data.items.filter((item) => item.target_type === "open_id" && item.status !== "DISABLED");
    if (active.length > 1) throw new Error("当前账号的绑定状态异常，请联系渠道管理员。");
    binding = active[0] || null;
    state.textContent = binding?.status === "VERIFIED" ? "已验证" : binding ? "待本人验证" : "未绑定";
    openId.value = binding?.external_id || "";
    code.value = "";
    unbind.hidden = !binding;
    verifyForm.hidden = binding?.status !== "UNVERIFIED";
    loaded = true;
  }
  async function run(task, success = "") {
    if (busy) return;
    busy = true;
    sync();
    show("正在处理…");
    try { await task(); show(success); }
    catch (error) { show(error.message || "操作未完成，请重试。", true); }
    finally { busy = false; sync(); }
  }
  function refresh() {
    if (busy) return;
    loaded = false;
    binding = null;
    openId.value = code.value = "";
    unbind.hidden = verifyForm.hidden = true;
    state.textContent = "正在读取…";
    return run(async () => {
      try { await load(); }
      catch (error) { state.textContent = "读取失败"; throw error; }
    });
  }
  function change(value) {
    return request(root + "/me/bindings/feishu", "PUT", {
      feishu_open_id: value,
      expected_binding_id: binding?.id ?? null,
      expected_row_version: binding?.row_version ?? null,
    });
  }
  bindingForm.onsubmit = (event) => {
    event.preventDefault();
    if (!loaded || openId.value === binding?.external_id) return;
    void run(async () => {
      await change(openId.value);
      await load();
    }, "已保存，请发送验证码并完成验证。");
  };
  unbind.onclick = async () => {
    if (!binding || busy || !await confirmAction({ title: "解绑飞书身份？", message: "解绑后不再向该飞书账号投递个人通知，站内通知不受影响。", confirmLabel: "确认解绑" })) return;
    void run(async () => { await change(null); await load(); }, "已解绑飞书身份。");
  };
  find("[data-send]").onclick = () => {
    if (!binding || busy) return;
    void run(async () => {
      await request(root + "/me/bindings/" + encodeURIComponent(binding.id) + "/challenge", "POST", {});
      await load();
    }, "验证码已入队，请在飞书中查看。");
  };
  verifyForm.onsubmit = (event) => {
    event.preventDefault();
    if (!binding || busy) return;
    void run(async () => {
      await request(root + "/bindings/" + encodeURIComponent(binding.id) + "/confirm", "POST", {
        code: code.value, expected_row_version: binding.row_version,
      });
      await load();
    }, "飞书身份已验证。");
  };
  sync();
  return { refresh, isBusy: () => busy };
}
