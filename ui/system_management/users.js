import { mountPlatformMenu } from "./platform-menu.js?v=20261125_platform_menu";
import { confirmAction } from "../shared/kbot-confirm.js";

const session = globalThis.KBotPlatformSession;
const portal = "/ui/index.html";
const root = "/platform/users";
const userPath = (id) => root + "/" + encodeURIComponent(id);
const $ = (id) => document.getElementById(id);
const userStatus = (state) => ({ ACTIVE: "已启用", DISABLED: "已停用" })[state] || "状态未知";
const badge = (text, active) => node("span", text, "media-badge" + (active ? " good" : ""));
const input = (form, name) => form.elements.namedItem(name);
const secretLength = (value) => new TextEncoder().encode(value).length;
const isSelf = (actor, user) => user.user_id.toLowerCase() === actor.user_id.toLowerCase();
const canEditUser = (actor, user) => user?.account_origin === "PLATFORM" && !user.protected && !isSelf(actor, user);
let busy = false;

function node(tag, value = "", className = "") {
  const element = document.createElement(tag);
  element.textContent = value;
  if (className) element.className = className;
  return element;
}

function button(text, onClick, className = "small") {
  const element = node("button", text, className);
  element.type = "button";
  element.onclick = onClick;
  return element;
}

function message(text, error = false, target = $("user-result")) {
  target.textContent = text;
  target.classList.toggle("user-error", error);
}

function value(form, name) {
  return input(form, name).value;
}

function open(dialog) {
  dialog.querySelector("form").reset();
  message("", false, dialog.querySelector("[data-dialog-error]"));
  dialog.showModal();
}

async function act(task, { dialog, success = "", onSuccess } = {}) {
  if (busy) return;
  busy = true;
  const drawer = $("user-detail-drawer");
  const surface = drawer.open ? drawer : $("system-content");
  const result = drawer.open ? $("user-detail-result") : $("user-result");
  const controls = [...(dialog || surface).querySelectorAll(dialog ? ":is(button,input,select):not(:disabled)" : "button:not(:disabled)")];
  controls.forEach((element) => { element.disabled = true; });
  const status = dialog?.querySelector("[data-dialog-error]") || result;
  message("正在处理…", false, status);
  try {
    await task();
    dialog?.close();
    message(success, false, result);
    onSuccess?.();
  } catch (error) {
    const passwordRejected = dialog === $("user-password-dialog")
      && ["INVALID_CREDENTIALS", "PASSWORD_REUSED"].includes(error.code);
    if (error.status === 401 && !passwordRejected) location.replace(portal);
    else message(error.message || "操作未完成，请重试。", true, status);
  } finally {
    busy = false;
    controls.forEach((element) => { element.disabled = false; });
  }
}

function setupDirectory(actor) {
  const search = $("user-search"), rows = $("user-rows"), limit = 50;
  const headings = rows.closest("table").tHead.rows[0].cells;
  const drawer = $("user-detail-drawer"), createTrigger = $("user-create");
  const passwordDialog = $("user-password-dialog"), passwordForm = $("user-password-form");
  let offset = 0, generation = 0;
  let activeDetail = null, passwordTarget = null;
  let resetAccessRequest, resettableApps = new Map();

  function showDetail(user, trigger, changed = false) {
    if (busy || drawer.open) return;
    const current = { id: user.user_id, trigger, changed };
    activeDetail = current;
    void setupDetail(actor, user, {
      isCurrent: () => activeDetail === current && drawer.open,
      onChange: () => { current.changed = true; },
    });
  }

  drawer.addEventListener("close", async () => {
    if (drawer.open || !activeDetail) return;
    const { id, trigger, changed } = activeDetail;
    activeDetail = null;
    if (changed) await load(offset, { keepRows: true });
    if (document.querySelector("dialog[open]")) return;
    const replacement = trigger.isConnected ? trigger : [...rows.querySelectorAll("[data-user-control]")]
      .find((item) => item.dataset.userId === id && item.dataset.userControl === trigger.dataset.userControl);
    (replacement || input(search, "search")).focus({ preventScroll: true });
  });

  function detailButton(user, label, className, control) {
    const trigger = button(label, () => showDetail(user, trigger), className);
    trigger.dataset.userId = user.user_id;
    trigger.dataset.userControl = control;
    trigger.setAttribute("aria-haspopup", "dialog");
    trigger.setAttribute("aria-controls", drawer.id);
    return trigger;
  }

  function passwordButton(user, { path, appName, own = false }) {
    const trigger = button(own ? "修改密码" : "重置密码", () => {
      if (busy) return;
      passwordTarget = { path, own };
      $("user-password-title").textContent = trigger.textContent;
      $("user-password-target").textContent = [...new Set([user.display_name, user.user_id, appName].filter(Boolean))].join(" · ");
      $("user-password-help").textContent = own ? "修改后，当前页面保留登录，其他旧会话将失效。"
        : "重置后原有会话失效，用户下次登录必须修改密码。";
      $("user-current-password-field").hidden = !own;
      input(passwordForm, "current_password").disabled = !own;
      $("user-password-label").textContent = own ? "新密码" : "临时密码";
      $("user-password-save").textContent = own ? "确认修改" : "确认重置";
      open(passwordDialog);
      input(passwordForm, own ? "current_password" : "password").focus();
    }, "platform-row-action");
    trigger.setAttribute("aria-haspopup", "dialog");
    trigger.setAttribute("aria-controls", passwordDialog.id);
    return trigger;
  }

  function rowActions(user) {
    const actions = node("div", "", "platform-action-group");
    actions.append(detailButton(user, user.protected || user.account_origin === "APP" ? "查看" : "查看与授权", "platform-row-action", "action"));
    if (user.protected && user.account_origin === "APP" && !isSelf(actor, user) && resettableApps.has(user.owner_app_id)) {
      actions.append(passwordButton(user, {
        path: "/platform/apps/" + encodeURIComponent(user.owner_app_id) + "/initial-admin/password",
        appName: resettableApps.get(user.owner_app_id),
      }));
    } else if (user.protected && user.account_origin === "PLATFORM" && isSelf(actor, user)) {
      actions.append(passwordButton(user, { own: true }));
    }
    if (!canEditUser(actor, user)) return actions;
    const name = user.display_name || user.user_id;
    const active = user.status === "ACTIVE";
    const updateStatus = async () => {
      if (busy) return;
      const label = active ? "停用" : "启用";
      if (!await confirmAction({ title: `${label}账号“${name}”？`, confirmLabel: `${label}账号`, tone: "warning",
        message: active ? "停用后无法登录，所有 App 访问暂停，已有授权保留。" : "启用后可登录并使用已有的 App 授权。" })) return;
      await act(async () => {
        await session.request(userPath(user.user_id), "PATCH", { status: active ? "DISABLED" : "ACTIVE" });
        await load(offset, { keepRows: true });
      }, { success: `账号已${label}。` });
    };
    const removeUser = async () => {
      if (busy || !await confirmAction({ title: `删除用户“${name}”？`, confirmLabel: "删除用户", tone: "danger",
        message: `账号“${user.user_id}”的全部 App 授权将被撤销，此操作不可恢复。` })) return;
      await act(async () => {
        await session.request(userPath(user.user_id), "DELETE");
        await load(offset, { keepRows: true });
      }, { success: "用户已删除。" });
    };
    actions.append(button(active ? "停用" : "启用", updateStatus, "platform-row-action"),
      passwordButton(user, { path: userPath(user.user_id) + "/password" }),
      button("删除", removeUser, "platform-row-action is-danger"));
    return actions;
  }
  function tableCell(column, ...children) {
    const cell = node("td");
    cell.setAttribute("role", "cell");
    if (column > 0) {
      const label = node("span", headings[column].textContent, "platform-cell-label");
      label.setAttribute("aria-hidden", "true");
      cell.append(label);
    }
    cell.append(...children);
    return cell;
  }
  function empty(text) {
    const row = node("tr", "", "platform-table-empty"), cell = tableCell(0, text);
    row.setAttribute("role", "row");
    cell.colSpan = 4;
    row.append(cell);
    rows.replaceChildren(row);
  }
  async function load(nextOffset, { keepRows = false, verifyResetAccess = false } = {}) {
    const current = ++generation;
    $("user-prev").disabled = $("user-next").disabled = true;
    const query = new URLSearchParams({ offset: nextOffset, limit });
    const keyword = value(search, "search").trim(), state = value(search, "status");
    if (keyword) query.set("search", keyword);
    if (state) query.set("status", state);
    if (!keepRows) {
      empty("正在读取用户…");
      $("user-count").textContent = "";
      $("user-page").textContent = "";
    }
    message("");
    try {
      // App 列表与初始管理员重置使用同一项平台权限；服务端仍会逐次校验重置请求。
      if (verifyResetAccess || !resetAccessRequest) {
        resetAccessRequest = session.request("/platform/apps").then((items) => ({ items }))
          .catch((error) => { if (error.status === 401) throw error; return { error }; });
      }
      const [{ items, total }, apps] = await Promise.all([session.request(root + "?" + query), resetAccessRequest]);
      if (current !== generation) return;
      resettableApps = new Map((apps.items || []).map((app) => [app.app_id, app.display_name]));
      if (nextOffset > 0 && !items.length) return load(Math.max(0, nextOffset - limit), { keepRows });
      offset = nextOffset;
      rows.replaceChildren(...items.map((user) => {
        const row = node("tr");
        row.setAttribute("role", "row");
        const title = detailButton(user, user.display_name || user.user_id, "platform-cell-title platform-cell-link", "name");
        const identity = tableCell(0, title);
        if (user.protected) identity.append(node("span", "受保护账号", "platform-protected-label"));
        if (user.display_name && user.display_name !== user.user_id) {
          identity.append(node("small", user.user_id, "platform-cell-meta"));
        }
        const names = user.assigned_app_names || [];
        const appSummary = node("div");
        if (names.length) {
          appSummary.append(node("strong", `${names.length} 个 App`, "platform-cell-title"),
            node("small", names.join("、"), "platform-cell-meta"));
        } else appSummary.append(node("span", "未分配", "platform-cell-meta"));
        const status = node("span", userStatus(user.status), `platform-status${user.status === "ACTIVE" ? " is-success" : ""}`);
        const action = tableCell(3, rowActions(user));
        action.className = "platform-table-actions";
        row.append(identity, tableCell(1, appSummary), tableCell(2, status), action);
        return row;
      }));
      if (!items.length) empty("没有符合条件的用户。");
      $("user-count").textContent = `${total} 人`;
      $("user-page").textContent = total ? `${offset + 1}–${offset + items.length} / ${total}` : "0 人";
      $("user-prev").disabled = offset === 0;
      $("user-next").disabled = offset + limit >= total;
      if (apps.error && apps.error.status !== 403) message("暂无法核对 App 初始管理员的密码重置权限，请刷新重试。", true);
    } catch (error) {
      if (current !== generation) return;
      if (error.status === 401) return location.replace(portal);
      empty("用户读取失败。");
      message(error.message, true);
    }
  }
  search.onsubmit = (event) => { event.preventDefault(); void load(0); };
  input(search, "status").onchange = () => load(0);
  $("user-prev").onclick = () => load(offset - limit);
  $("user-next").onclick = () => load(offset + limit);
  $("user-refresh").onclick = () => load(offset, { verifyResetAccess: true });

  const dialog = $("user-create-dialog"), form = $("user-create-form");
  passwordDialog.addEventListener("close", () => { if (!passwordDialog.open) passwordTarget = null; });
  passwordForm.onsubmit = (event) => {
    event.preventDefault();
    if (!passwordDialog.open || !passwordTarget) return;
    const { path, own } = passwordTarget, password = value(passwordForm, "password");
    void act(async () => {
      if (secretLength(password) > 72) throw new Error("密码不能超过 72 字节。");
      if (own) {
        await session.changePassword({ current_password: value(passwordForm, "current_password"), new_password: password });
      } else {
        await session.request(path, "POST", { password, must_change_password: true });
      }
    }, { dialog: passwordDialog, success: own ? "密码已修改。" : "密码已重置，用户下次登录须修改。" });
  };
  createTrigger.onclick = () => open(dialog);
  form.onsubmit = async (event) => {
    event.preventDefault();
    let user;
    await act(async () => {
      const password = value(form, "password");
      if (secretLength(password) > 72) throw new Error("密码不能超过 72 字节。");
      user = await session.request(root, "POST", {
        user_id: value(form, "user_id").trim(), display_name: value(form, "display_name").trim() || null,
        password, max_security_level: Number(value(form, "max_security_level")), must_change_password: true,
      });
    }, { dialog });
    if (user) showDetail(user, createTrigger, true);
  };
  return load(0, { verifyResetAccess: true });
}

function scopeLabel(binding, app) {
  if (binding.scope_mode === "ALL_APP_DOMAINS") return "全部业务空间";
  return (binding.domain_ids || []).map((id) =>
    app.domains.find((domain) => String(domain.domain_id) === String(id))?.name || String(id),
  ).join("、") || "未指定业务空间";
}

function setupDetail(actor, selectedUser, { isCurrent, onChange }) {
  const id = selectedUser.user_id;
  const path = userPath(id);
  const drawer = $("user-detail-drawer"), detail = $("user-detail"), body = $("user-detail-body");
  const result = $("user-detail-result"), refresh = $("user-detail-refresh");
  const profile = $("user-profile-form"), security = $("user-security-form");
  const grantDialog = $("user-grant-dialog"), grantForm = $("user-grant-form");
  const addApp = $("user-app-add");
  let user, catalog = [], saveGrant;
  const self = () => id?.toLowerCase() === actor.user_id.toLowerCase();
  const editable = () => canEditUser(actor, user);
  const grantPath = (app) => path + "/app-grants/" + encodeURIComponent(app.app_id);
  const focusAccess = () => $("user-app-access-title").focus({ preventScroll: true });
  const rolesOf = (membership) => membership?.role_bindings || [];
  const activeDomains = (app) => app.domains.filter((domain) => domain.active);
  const assignableRoles = (app) => app.status === "ACTIVE" && app.member_assignable && app.manageable
    ? app.roles.filter((role) => role.manageable && role.status === "ACTIVE"
      && (role.scope_policy === "ALL_APP_DOMAINS" || activeDomains(app).length)) : [];
  const scopeMode = (role) => role.scope_policy === "ALL_APP_DOMAINS" ? "ALL_APP_DOMAINS" : "SELECTED_DOMAINS";

  async function changeRoles(app, membership, roleCode, replacement) {
    // 接口按 App 整组保存：单角色操作必须保留其他绑定，不能顺带清理停用角色或空间。
    const roles = assignableRoles(app), domains = new Set(activeDomains(app).map((domain) => String(domain.domain_id)));
    const bindings = rolesOf(membership).filter((binding) => binding.role_code !== roleCode).map((binding) => {
      const role = roles.find((item) => item.role_code === binding.role_code);
      const domainIds = (binding.domain_ids || []).map(String);
      if (!role || binding.status !== "ACTIVE" || binding.scope_mode !== scopeMode(role)
        || (binding.scope_mode === "SELECTED_DOMAINS" && (!domainIds.length || domainIds.some((id) => !domains.has(id))))) {
        const name = app.roles.find((item) => item.role_code === binding.role_code)?.display_name || binding.role_code;
        throw new Error(`现有角色“${name}”或其空间范围已不可用。请先调整该角色的范围或撤销该角色。`);
      }
      return { role_code: binding.role_code, scope_mode: binding.scope_mode, domain_ids: domainIds };
    });
    if (replacement) bindings.push(replacement);
    if (!bindings.length) throw new Error("App 至少需要一个角色。请通过「撤销整个 App 访问」移除最后一个角色。");
    await session.request(grantPath(app), "PUT", { role_bindings: bindings });
    onChange();
    await load();
  }

  function showGrantEditor({ app, choices = [], membership, binding }) {
    if (busy || grantDialog.open || !isCurrent() || !editable()) return;
    const newApp = !app, restoring = membership && membership.status !== "ACTIVE";
    const restoringRole = binding && binding.status !== "ACTIVE";
    const title = newApp ? "授予 App 访问" : restoring ? "恢复 App 授权" : restoringRole ? "恢复业务角色"
      : binding ? "编辑角色范围" : "授予业务角色";
    const appSelect = input(grantForm, "app_id"), roleSelect = input(grantForm, "role_code");
    const scope = $("user-grant-scope"), save = $("user-grant-save");
    const error = grantDialog.querySelector("[data-dialog-error]");
    let activeApp = app, roles = [], selectedRole, spaces = [];
    message("", false, error);
    $("user-grant-title").textContent = title;
    $("user-grant-target").textContent = `用户：${user.display_name || user.user_id}`;
    $("user-grant-role-help").textContent = binding ? "编辑范围不会改变角色。" : newApp
      ? "首次授予 App 访问时，必须同时分配一个业务角色。" : "选择单个角色及其业务空间范围。";
    save.textContent = newApp ? "确认授权" : restoring || restoringRole ? "确认恢复" : binding ? "保存范围" : "授予角色";
    $("user-grant-app-field").hidden = !newApp;
    appSelect.disabled = !newApp;
    appSelect.replaceChildren(new Option("选择 App", ""), ...choices.map((item) => new Option(item.display_name, item.app_id)));

    function sync() {
      save.disabled = !selectedRole
        || (scopeMode(selectedRole) === "SELECTED_DOMAINS" && !spaces.some((field) => field.checked));
    }

    function renderScope() {
      selectedRole = roles.find((role) => role.role_code === roleSelect.value);
      scope.replaceChildren();
      scope.hidden = !selectedRole;
      spaces = [];
      if (selectedRole && scopeMode(selectedRole) === "ALL_APP_DOMAINS") {
        scope.append(node("strong", "全部业务空间"), node("p", "该角色固定覆盖当前及未来新增的全部业务空间。"));
      } else if (selectedRole) {
        const saved = (binding || (restoring && rolesOf(membership).find((item) => item.role_code === selectedRole.role_code)));
        const selectedDomains = (saved?.domain_ids || []).map(String);
        const domains = activeDomains(activeApp);
        const fieldset = node("fieldset", "", "user-spaces user-scope-choices");
        fieldset.append(node("legend", "业务空间范围"), node("p", "至少选择一个业务空间。"));
        for (const domain of domains) {
          const label = node("label", "", "user-choice"), field = node("input");
          field.type = "checkbox";
          field.value = String(domain.domain_id);
          field.checked = selectedDomains.includes(field.value);
          field.onchange = sync;
          label.append(field, node("span", domain.name));
          fieldset.append(label);
          spaces.push(field);
        }
        scope.append(fieldset);
        if (selectedDomains.some((id) => !domains.some((domain) => String(domain.domain_id) === id))) {
          scope.append(node("p", "此角色中已停用的业务空间不会包含在保存后的范围中。", "media-notice warn"));
        }
      }
      sync();
    }

    function renderRoles() {
      $("user-grant-context").textContent = activeApp ? `${activeApp.display_name} · 业务角色` : "App 业务访问";
      roles = activeApp ? assignableRoles(activeApp).filter((role) => binding ? role.role_code === binding.role_code
        : restoring || !rolesOf(membership).some((item) => item.role_code === role.role_code)) : [];
      roleSelect.disabled = !activeApp || Boolean(binding);
      roleSelect.replaceChildren(new Option(activeApp ? "选择角色" : "请先选择 App", ""),
        ...roles.map((role) => new Option(role.display_name, role.role_code)));
      roleSelect.value = binding?.role_code || "";
      renderScope();
    }
    appSelect.onchange = () => {
      activeApp = choices.find((item) => item.app_id === appSelect.value);
      message("", false, error);
      renderRoles();
    };
    roleSelect.onchange = () => { message("", false, error); renderScope(); };
    saveGrant = () => {
      if (save.disabled || !grantDialog.open || !isCurrent()) return;
      const replacement = { role_code: selectedRole.role_code, scope_mode: scopeMode(selectedRole),
        domain_ids: spaces.filter((field) => field.checked).map((field) => field.value) };
      void act(() => changeRoles(activeApp, membership, replacement.role_code, replacement), {
        dialog: grantDialog, onSuccess: focusAccess, success: newApp ? `${activeApp.display_name}访问已授予。`
          : restoring ? `${activeApp.display_name}访问已恢复。`
          : restoringRole ? "业务角色已恢复。" : binding ? "角色范围已更新。" : "业务角色已授予。",
      });
    };
    renderRoles();
    grantDialog.showModal();
    grantDialog.querySelector(".media-dialog-body").scrollTop = 0;
    (newApp ? appSelect : binding ? spaces[0] || save : roleSelect).focus();
  }

  function renderApp(app, membership) {
    const active = membership.status === "ACTIVE", bound = rolesOf(membership);
    const canManage = editable() && app.manageable && membership.member_source === "PLATFORM_GRANT"
      && bound.every((binding) => app.roles.some((role) => role.role_code === binding.role_code && role.manageable));
    const assignable = canManage ? assignableRoles(app) : [];
    const card = node("article", "", "user-app");
    const heading = node("header");
    heading.append(node("h5", app.display_name), badge(
      app.status === "DISABLED" ? "已分配 · App 已停用" : active ? "已授权" : "授权已停用",
      active && app.status === "ACTIVE",
    ));
    card.append(heading);
    if (bound.length || (active && assignable.length)) {
      const rolesTitle = node("div", "", "user-app-role-heading");
      rolesTitle.append(node("strong", "已分配角色"), node("span", `${bound.length} 个`));
      const group = node("div", "", "user-role-grid");
      for (const binding of bound) {
        const role = app.roles.find((item) => item.role_code === binding.role_code);
        const name = role?.display_name || binding.role_code;
        const row = node("article", "", "user-role");
        row.append(node("span", "业务角色", "user-role-kind"), node("h6", name), node("p", scopeLabel(binding, app)));
        if (binding.status !== "ACTIVE" || role?.status === "DISABLED") row.append(badge("角色已停用", false));
        if (active && assignable.length) {
          const actions = node("footer", "", "platform-action-group");
          if (assignable.includes(role) && (scopeMode(role) !== binding.scope_mode
            || role.scope_policy === "SELECTABLE" || binding.status !== "ACTIVE")) {
            const edit = button(binding.status === "ACTIVE" ? "编辑范围" : "恢复角色",
              () => showGrantEditor({ app, membership, binding }), "platform-row-action");
            edit.setAttribute("aria-haspopup", "dialog");
            edit.setAttribute("aria-controls", grantDialog.id);
            actions.append(edit);
          }
          if (bound.length > 1) actions.append(button("撤销角色", async () => {
            if (busy || !await confirmAction({ title: `撤销“${name}”？`, confirmLabel: "撤销角色", tone: "danger",
              message: "该角色的权限和空间范围将失效，其他角色及 App 访问不变。" }) || !isCurrent()) return;
            await act(() => changeRoles(app, membership, binding.role_code), { success: "业务角色已撤销。", onSuccess: focusAccess });
          }, "platform-row-action is-danger"));
          if (actions.childElementCount) row.append(actions);
        }
        group.append(row);
      }
      if (active && assignable.some((role) => !bound.some((binding) => binding.role_code === role.role_code))) {
        const add = button("", () => showGrantEditor({ app, membership }), "user-role-add");
        const plus = node("span", "＋");
        plus.setAttribute("aria-hidden", "true");
        add.append(plus, node("strong", "授予其他角色"), node("small", "选择业务角色和空间范围"));
        add.setAttribute("aria-haspopup", "dialog");
        add.setAttribute("aria-controls", grantDialog.id);
        group.append(add);
      }
      card.append(rolesTitle, group);
    }
    if (canManage) {
      if (!active && assignable.length) card.append(button("恢复 App 授权", () => showGrantEditor({ app, membership })));
      const actions = node("footer", "", "user-app-revoke");
      actions.append(node("p", "撤销后，该 App 的全部角色将失效，其他 App 不受影响。"));
      actions.append(button("撤销整个 App 访问", async () => {
        if (busy || !await confirmAction({ title: `撤销${app.display_name}访问？`, confirmLabel: "撤销 App 访问", tone: "danger",
          message: "用户将无法进入此 App，该 App 的角色和空间范围也会失效。其他 App 授权不受影响。" }) || !isCurrent()) return;
        await act(async () => {
          await session.request(grantPath(app), "DELETE");
          onChange();
          await load();
        }, { success: `${app.display_name}访问已撤销。`, onSuccess: focusAccess });
      }, "small user-danger"));
      card.append(actions);
    }
    return card;
  }

  function render() {
    const memberships = new Map(user.app_memberships.map((item) => [item.app_id, item]));
    const apps = [...memberships.keys()].map((appId) => catalog.find((app) => app.app_id === appId)
      || { app_id: appId, display_name: appId, roles: [], domains: [] });
    const choices = catalog.filter((app) => !memberships.has(app.app_id) && assignableRoles(app).length);
    $("user-name").textContent = user.display_name || user.user_id;
    $("user-id").textContent = user.user_id;
    $("user-state").textContent = userStatus(user.status);
    $("user-security").textContent = user.max_security_level;
    $("user-assigned").textContent = user.app_memberships.filter((item) => item.status === "ACTIVE").length;
    $("user-locked").hidden = editable();
    $("user-locked").textContent = user.protected ? "这是受保护的初始化账号，不能修改资料、状态或授权。"
      : self() ? "不能通过管理入口修改自己的账号或授权。"
      : "该账号尚未纳入平台管理，请联系平台管理员。";
    addApp.hidden = !editable() || !choices.length;
    addApp.onclick = () => showGrantEditor({ choices });
    $("user-apps").replaceChildren(...(apps.length
      ? apps.map((app) => renderApp(app, memberships.get(app.app_id)))
      : [node("p", "暂无已授权的 App。", "user-app-empty")]));
    if (editable() && catalog.length && !catalog.some((app) => app.manageable)) {
      $("user-apps").prepend(node("p", "当前账号可以维护用户资料，但没有 App 授权权限。", "media-notice"));
    }
    $("user-profile").hidden = security.hidden = !editable();
    input(profile, "display_name").value = user.display_name || "";
    input(security, "max_security_level").value = user.max_security_level;
    detail.hidden = false;
  }
  async function load() {
    const [record, options] = await Promise.all([
      session.request(path),
      session.request("/platform/app-grant-options").catch((error) => {
        if (error.status === 401) throw error;
        return { error };
      }),
    ]);
    if (!isCurrent()) return;
    user = record;
    catalog = options.items || [];
    render();
    if (options.error) $("user-apps").prepend(node("p", "授权目录暂不可用，当前仅可查看已有授权。请刷新后重试。", "media-notice warn"));
  }
  async function reload() {
    if (busy || !isCurrent()) return;
    detail.hidden = true;
    refresh.disabled = true;
    message("正在读取用户与授权…", false, result);
    body.setAttribute("aria-busy", "true");
    try {
      await load();
      if (isCurrent()) message("", false, result);
    } catch (error) {
      if (!isCurrent()) return;
      if (error.status === 401) location.replace(portal);
      else message(error.message || "无法读取用户详情，请刷新后重试。", true, result);
    } finally {
      if (isCurrent()) {
        body.removeAttribute("aria-busy");
        refresh.disabled = false;
      }
    }
  }
  function patch(data, success) {
    return act(async () => { await session.request(path, "PATCH", data); onChange(); await load(); }, { success });
  }
  profile.onsubmit = (event) => {
    event.preventDefault();
    if (editable()) void patch({
      display_name: value(profile, "display_name").trim() || null,
    }, "账号资料已保存。");
  };
  security.onsubmit = (event) => {
    event.preventDefault();
    if (editable()) void patch({
      max_security_level: Number(value(security, "max_security_level")),
    }, "安全等级已保存。");
  };
  grantForm.onsubmit = (event) => { event.preventDefault(); saveGrant?.(); };
  refresh.onclick = reload;
  detail.hidden = true;
  $("user-name").textContent = selectedUser.display_name || id;
  $("user-id").textContent = id;
  drawer.returnValue = "";
  drawer.showModal();
  body.scrollTop = 0;
  if (!id || id.length > 256 || /[\/\\]/.test(id) || id === "." || id === "..") {
    body.removeAttribute("aria-busy");
    refresh.disabled = true;
    return message("缺少有效的用户账号，请从用户目录打开详情。", true, result);
  }
  return reload();
}

async function initialize() {
  const actor = session.load();
  if (!actor || actor.must_change_password) return location.replace(portal);
  $("system-user").textContent = actor.display_name || actor.user_id;
  try {
    const entries = await session.request("/auth/entries");
    if (entries.user_management !== true) {
      if (entries.system_management === true) return location.replace("./models.html");
      return message("当前账号没有用户管理权限。", true, $("system-status"));
    }
    mountPlatformMenu(entries, "user_management");
    document.querySelectorAll("dialog").forEach((dialog) => {
      dialog.addEventListener("cancel", (event) => { if (busy) event.preventDefault(); });
      dialog.addEventListener("close", () => { dialog.querySelector('input[type="password"]')?.form.reset(); });
      dialog.querySelectorAll("[data-dialog-close]").forEach((button) => { button.onclick = () => { if (!busy) dialog.close(); }; });
    });
    $("system-status").hidden = true;
    $("system-content").hidden = false;
    await setupDirectory(actor);
  } catch (error) {
    if (error.status === 401) return location.replace(portal);
    $("system-status").hidden = false;
    message(error.message || "无法读取平台管理权限。", true, $("system-status"));
  }
}

addEventListener("pageshow", (event) => { if (event.persisted) location.reload(); });
void initialize();
