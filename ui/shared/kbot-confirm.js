/* 静态 App 共用的确认组件：挂到 body，避免页面布局覆盖模态框尺寸。 */
let active = null;
let stylesheet = null;

function loadStyles() {
  if (!stylesheet) stylesheet = new Promise((resolve, reject) => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("./kbot-confirm.css?v=20261105_1", import.meta.url).href;
    link.onload = resolve;
    link.onerror = () => { link.remove(); stylesheet = null; reject(new Error("无法加载确认框，请刷新后重试。")); };
    document.head.append(link);
  });
  return stylesheet;
}

function finish(confirmed) {
  const current = active;
  if (!current) return;
  const { dialog, reason, option, trigger, resolve } = current;
  if (confirmed && reason) {
    reason.setCustomValidity(reason.value.trim() ? "" : "请填写原因，不能只输入空格。");
    if (!reason.reportValidity()) return;
  }
  active = null;
  const result = { confirmed, optionSelected: confirmed && Boolean(option?.checked), reason: confirmed && reason ? reason.value.trim() : null };
  if (dialog?.open) dialog.close();
  dialog?.remove();
  if (trigger?.isConnected) trigger.focus({ preventScroll: true });
  resolve(result);
}

function mount(options) {
  const dialog = document.createElement("dialog");
  const danger = options.tone !== "warning";
  dialog.id = "kbot-confirm-dialog";
  dialog.className = `confirm-dialog kbot-confirm-dialog confirm-dialog--${danger ? "danger" : "warning"}`;
  dialog.setAttribute("role", "alertdialog");
  dialog.setAttribute("aria-modal", "true");
  dialog.setAttribute("aria-labelledby", "kbot-confirm-title");
  dialog.setAttribute("aria-describedby", "kbot-confirm-message");
  // 只使用固定结构；标题、名称和提示均以 textContent 写入。
  dialog.innerHTML = `<form class="kbot-confirm-form"><header class="confirm-dialog-header"><span class="confirm-dialog-mark" aria-hidden="true"></span><div><p class="confirm-dialog-kicker"></p><h2 id="kbot-confirm-title"></h2></div></header><p id="kbot-confirm-message" class="confirm-dialog-message"></p><label class="confirm-dialog-option" hidden><input type="checkbox"><span><strong></strong><small></small></span></label><label class="kbot-confirm-reason" hidden><span></span><textarea name="reason" rows="4"></textarea></label><footer class="confirm-dialog-actions"><button type="button" class="secondary-button" value="cancel" autofocus>取消</button><button type="submit" value="confirm"></button></footer></form>`;
  dialog.querySelector(".confirm-dialog-mark").textContent = danger ? "!" : "?";
  dialog.querySelector(".confirm-dialog-kicker").textContent = danger ? "请确认此操作" : "需要你的确认";
  dialog.querySelector("h2").textContent = options.title;
  dialog.querySelector(".confirm-dialog-message").textContent = options.message;
  const approve = dialog.querySelector('[value="confirm"]');
  approve.textContent = options.confirmLabel || "确认";
  approve.className = danger ? "danger-button" : "primary-button";
  const checkbox = dialog.querySelector('.confirm-dialog-option input');
  if (options.option) {
    const label = checkbox.closest("label");
    label.hidden = false;
    label.querySelector("strong").textContent = options.option.label;
    label.querySelector("small").textContent = options.option.description || "";
    checkbox.checked = Boolean(options.option.defaultSelected);
  }
  const reason = dialog.querySelector("textarea");
  if (options.reason) {
    reason.closest("label").hidden = false;
    reason.closest("label").querySelector("span").textContent = options.reason.label || "原因";
    reason.placeholder = options.reason.placeholder || "请填写原因";
    reason.maxLength = options.reason.maxLength || 1000;
    reason.required = true;
    reason.oninput = () => reason.setCustomValidity("");
  }
  Object.assign(active, { dialog, option: checkbox, reason: options.reason ? reason : null });
  dialog.querySelector("form").onsubmit = event => { event.preventDefault(); finish(true); };
  dialog.querySelector('[value="cancel"]').onclick = () => finish(false);
  dialog.addEventListener("cancel", event => { event.preventDefault(); finish(false); });
  dialog.addEventListener("close", () => { if (active?.dialog === dialog) finish(false); });
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) finish(false);
  });
  document.body.append(dialog);
  dialog.showModal();
  dialog.querySelector('[value="cancel"]').focus();
}

export function confirmActionWithOption(options) {
  // 包括首次加载样式的时间窗口；重复点击不能打开两次或确认上一项操作。
  if (active) return Promise.resolve({ confirmed: false, optionSelected: false, reason: null });
  return new Promise(resolve => {
    const current = { resolve, trigger: document.activeElement };
    active = current;
    loadStyles().then(() => { if (active === current) mount(options); }).catch(error => {
      if (active !== current) return;
      finish(false);
      console.error(error.message);
    });
  });
}

export async function confirmAction(options) {
  return (await confirmActionWithOption(options)).confirmed;
}

export async function confirmActionWithReason(options) {
  return confirmActionWithOption(options);
}

addEventListener("pagehide", () => finish(false));
