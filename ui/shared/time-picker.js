// 通用时分控件；与 KBot ui/shared/time-picker.js 同步，不包含业务或时区逻辑。
let sequence = 0;
const pad = value => String(value).padStart(2, "0");

export function normalizeTime(value) {
  const source = String(value).trim();
  const match = /^(\d{1,2}):(\d{2})$/.exec(source) || /^(\d{2})(\d{2})$/.exec(source);
  return match && Number(match[1]) < 24 && Number(match[2]) < 60
    ? `${pad(Number(match[1]))}:${match[2]}` : "";
}

export function isTimeInRange(value, min = "", max = "") {
  const time = normalizeTime(value), lower = normalizeTime(min), upper = normalizeTime(max);
  if (!time) return false;
  if (lower && upper && lower > upper) return time >= lower || time <= upper;
  return (!lower || time >= lower) && (!upper || time <= upper);
}

export function mountTimePicker(root) {
  const input = root.querySelector(".time-picker-input"), trigger = root.querySelector(".time-picker-trigger");
  if (!input || !trigger) throw new Error("时间控件缺少输入框或展开按钮");
  const doc = root.ownerDocument, win = doc.defaultView, id = `time-picker-${++sequence}`;
  const panel = doc.createElement("div");
  panel.id = id;
  panel.className = "time-picker-panel";
  panel.setAttribute("popover", "manual");
  const nativePopover = typeof panel.showPopover === "function";
  if (!nativePopover) panel.removeAttribute("popover");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "选择时间");
  panel.hidden = true;
  panel.innerHTML = `<div class="time-picker-heading"><span>24 小时制</span><strong class="time-picker-preview">--:--</strong></div>
    <div class="time-picker-columns">${[24, 60].map((count, part) => `<div class="time-picker-column"><span id="${id}-label-${part}" class="time-picker-column-label">${part ? "分钟" : "小时"}</span><div class="time-picker-list" role="listbox" tabindex="0" aria-labelledby="${id}-label-${part}" data-part="${part}">${Array.from({length: count}, (_, value) => `<div class="time-picker-option" role="option" id="${id}-${part}-${value}" data-value="${value}" aria-selected="false">${pad(value)}</div>`).join("")}</div></div>`).join("")}</div>
    <div class="time-picker-footer"><button class="secondary-button" type="button" data-time-cancel>取消</button><button class="primary-button" type="button" data-time-confirm>确定</button></div>`;
  root.append(panel);
  const lists = [...panel.querySelectorAll(".time-picker-list")], confirm = panel.querySelector("[data-time-confirm]");
  let opened = false, draft = [null, null], currentValue = input.value;
  const listeners = [];
  function listen(target, type, handler, options) {
    target.addEventListener(type, handler, options);
    listeners.push(() => target.removeEventListener(type, handler, options));
  }
  const contains = target => root.contains(target) || panel.contains(target);
  const draftValue = () => draft.every(value => value !== null) ? draft.map(pad).join(":") : "";
  // text 输入没有原生 min/max 属性，约束始终从 Attribute 读取。
  function valid(value) { return isTimeInRange(value, input.getAttribute("min"), input.getAttribute("max")); }
  function available(part, value) {
    if (part === 0) return Array.from({length:60}, (_, minute) => valid(`${pad(value)}:${pad(minute)}`)).some(Boolean);
    return draft[0] === null || valid(`${pad(draft[0])}:${pad(value)}`);
  }
  function validate() {
    input.setCustomValidity(!input.value || valid(input.value) ? "" : "请输入有效的 24 小时时间（HH:mm），并满足可选范围。");
  }
  function render() {
    panel.querySelector(".time-picker-preview").textContent = draft.map(value => value === null ? "--" : pad(value)).join(":");
    lists.forEach((list, part) => {
      [...list.children].forEach(option => {
        option.setAttribute("aria-selected", String(Number(option.dataset.value) === draft[part]));
        option.setAttribute("aria-disabled", String(!available(part, Number(option.dataset.value))));
      });
      if (draft[part] === null) list.removeAttribute("aria-activedescendant");
      else list.setAttribute("aria-activedescendant", `${id}-${part}-${draft[part]}`);
    });
    confirm.disabled = !valid(draftValue());
  }
  function reveal(part) {
    const list = lists[part], option = list.querySelector('[aria-selected="true"]');
    if (option) list.scrollTop = option.offsetTop - (list.clientHeight - option.clientHeight) / 2;
  }
  function position() {
    if (!opened) return;
    const rect = root.getBoundingClientRect(), viewport = win.visualViewport;
    const width = viewport?.width ?? win.innerWidth, height = viewport?.height ?? win.innerHeight;
    const x = viewport?.offsetLeft ?? 0, y = viewport?.offsetTop ?? 0, gap = 8, edge = 12;
    if (!root.getClientRects().length || rect.bottom < y || rect.top > y + height) { close(); return; }
    panel.style.width = `${Math.min(280, width - edge * 2)}px`;
    panel.style.maxHeight = `${height - edge * 2}px`;
    const box = panel.getBoundingClientRect();
    const top = rect.bottom + gap + box.height <= y + height - edge ? rect.bottom + gap : rect.top - gap - box.height;
    panel.style.left = `${Math.max(x + edge, Math.min(rect.left, x + width - box.width - edge))}px`;
    panel.style.top = `${Math.max(y + edge, Math.min(top, y + height - box.height - edge))}px`;
  }
  function close(restoreFocus = false) {
    if (!opened) return;
    opened = false;
    if (nativePopover && panel.matches(":popover-open")) panel.hidePopover();
    panel.hidden = true;
    input.setAttribute("aria-expanded", "false");trigger.setAttribute("aria-expanded", "false");
    if (restoreFocus && !input.disabled) input.focus({preventScroll:true});
  }
  function open(focusList = false) {
    if (input.disabled || input.readOnly) return;
    if (!opened) {
      const time = normalizeTime(input.value);
      draft = time ? time.split(":").map(Number) : [null, null];
      opened = true;render();panel.hidden = false;
      if (nativePopover) panel.showPopover();
      else (root.closest("dialog") || root.closest(".business-app-shell,.admin-shell") || doc.body).append(panel);
      input.setAttribute("aria-expanded", "true");trigger.setAttribute("aria-expanded", "true");
      position();lists.forEach((_, part) => reveal(part));
    }
    if (focusList) lists[0].focus({preventScroll:true});
  }
  function publish(value) {
    input.value = value;validate();
    input.dispatchEvent(new win.Event("input", {bubbles:true}));
    input.dispatchEvent(new win.Event("change", {bubbles:true}));
  }
  function commit() {
    if (!valid(draftValue())) return;
    publish(draftValue());close(true);
  }
  function sync() {
    if (input.value !== currentValue) {
      currentValue = input.value;
      if (opened) {const time = normalizeTime(input.value);draft = time ? time.split(":").map(Number) : [null, null];}
    }
    trigger.disabled = input.disabled || input.readOnly;
    root.classList.toggle("is-disabled", trigger.disabled);
    validate();
    if (trigger.disabled) close();
    else if (opened) {render();position();}
  }
  input.setAttribute("aria-haspopup", "dialog");input.setAttribute("aria-controls", id);input.setAttribute("aria-expanded", "false");
  trigger.setAttribute("aria-haspopup", "dialog");trigger.setAttribute("aria-controls", id);trigger.setAttribute("aria-expanded", "false");
  listen(input, "click", () => open());
  listen(trigger, "click", () => opened ? close(true) : open(true));
  listen(input, "input", () => {
    validate();
    if (opened) {const time = normalizeTime(input.value);draft = time ? time.split(":").map(Number) : [null, null];render();}
  });
  listen(input, "blur", () => {
    const time = normalizeTime(input.value);
    if (time && time !== input.value) publish(time);
    validate();
  });
  listen(input, "keydown", event => {
    if (event.key === "ArrowDown") {event.preventDefault();open(true);}
    if (event.key === "Enter") {const time = normalizeTime(input.value);if (time && input.value !== time) publish(time);close();}
  });
  lists.forEach((list, part) => {
    function select(value) {if (available(part, value)) {draft[part] = value;render();reveal(part);}}
    listen(list, "click", event => {
      const option = event.target.closest("[data-value]");
      if (option && option.getAttribute("aria-disabled") !== "true") {select(Number(option.dataset.value));list.focus({preventScroll:true});}
    });
    listen(list, "keydown", event => {
      const count = part ? 60 : 24;
      if (["ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(event.key)) {
        event.preventDefault();
        const direction = ["ArrowUp", "PageUp", "End"].includes(event.key) ? -1 : 1;
        let value = event.key === "Home" ? 0 : event.key === "End" ? count - 1 : ((draft[part] ?? (direction > 0 ? -1 : 0)) + (event.key.startsWith("Page") ? direction * 5 : direction) + count) % count;
        for (let i = 0; i < count; i++, value = (value + direction + count) % count) {
          if (available(part, value)) {select(value);break;}
        }
      } else if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
        event.preventDefault();lists[event.key === "ArrowLeft" ? 0 : 1].focus({preventScroll:true});
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();if (!part) lists[1].focus({preventScroll:true});else commit();
      }
    });
  });
  listen(confirm, "click", commit);
  listen(panel.querySelector("[data-time-cancel]"), "click", () => close(true));
  listen(panel, "keydown", event => {
    if (event.key !== "Tab") return;
    const first = lists[0], last = confirm;
    if ((!event.shiftKey && doc.activeElement === last) || (event.shiftKey && doc.activeElement === first)) {
      event.preventDefault();close();
      if (event.shiftKey) {trigger.focus({preventScroll:true});return;}
      const scope = input.form || root.closest("dialog") || doc;
      const controls = [...scope.querySelectorAll('input,select,textarea,button,a[href],[tabindex]')]
        .filter(node => !panel.contains(node) && !node.disabled && node.tabIndex >= 0 && node.getClientRects().length);
      (controls[(controls.indexOf(trigger) + 1) % controls.length] || input).focus({preventScroll:true});
    }
  });
  listen(doc, "pointerdown", event => {if (opened && !contains(event.target)) close();}, true);
  listen(doc, "focusin", event => {if (opened && !contains(event.target)) close();});
  listen(doc, "keydown", event => {
    if (opened && event.key === "Escape") {event.preventDefault();event.stopPropagation();close(true);}
  }, true);
  listen(doc, "scroll", event => {if (opened && !panel.contains(event.target)) position();}, true);
  listen(win, "resize", position);
  if (win.visualViewport) {listen(win.visualViewport, "resize", position);listen(win.visualViewport, "scroll", position);}
  if (input.form) listen(input.form, "reset", () => {close();win.queueMicrotask(sync);});
  const dialog = root.closest("dialog");
  if (dialog) listen(dialog, "close", () => close());
  const observer = new win.MutationObserver(sync);
  observer.observe(input, {attributes:true, attributeFilter:["disabled", "readonly", "required", "min", "max"]});
  sync();
  return {open, close, sync, destroy() {close();observer.disconnect();listeners.forEach(remove => remove());panel.remove();}};
}
