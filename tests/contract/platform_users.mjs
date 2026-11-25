// 平台用户目录、账号管理和多 App 授权只访问隔离的 Mock API。
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(process.env.KBOT_BROWSER_REQUIRE_FROM
  || fileURLToPath(new URL("../../../Ammolite/apps/portal-web/package.json", import.meta.url)));
const { chromium, expect } = require("@playwright/test");
const origin = "http://kbot-platform-users.test", root = new URL("../../", import.meta.url);
const usersURL = "/ui/system_management/users.html";
const userId = "平台+用户&1", admin = "ADMIN";
const base = "/api/v1/platform/users";
const profilePath = base + "/" + encodeURIComponent(userId);
const domainId = "9007199254741993";
const testDomainId = "13";
const browser = await chromium.launch({ headless: true });

async function setup({ userManage = true, grant = true, appManage = true, models = false, width = 1280, actor = admin } = {}) {
  const page = await browser.newPage({ viewport: { width, height: 850 } });
  const errors = [], writes = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const account = (user_id, display_name, protectedAccount = false, owner_app_id = null) => ({
    user_id, display_name, account_origin: owner_app_id ? "APP" : "PLATFORM", owner_app_id,
    status: "ACTIVE", max_security_level: 1, protected: protectedAccount,
    app_memberships: [],
  });
  const accounts = new Map([
    ["initial-operator", account("initial-operator", "初始管理员", true, "aiops")],
    [admin, account(admin, "全局管理员", true)],
  ]);
  if (!grant) accounts.set("viewer", account("viewer", "只读授权用户"));
  const catalog = [
    { app_id: "aiops", display_name: "AIOps", status: "ACTIVE", member_assignable: true, manageable: grant,
      roles: [
        { role_code: "app_user", display_name: "运维用户", scope_policy: "SELECTABLE", status: "ACTIVE", manageable: grant },
        { role_code: "app_admin", display_name: "运维管理员", scope_policy: "ALL_APP_DOMAINS", status: "ACTIVE", manageable: grant },
      ], domains: [{ domain_id: domainId, name: "生产空间", active: true },
        { domain_id: testDomainId, name: "测试空间", active: true }, { domain_id: "4", name: "停用空间", active: false }] },
    { app_id: "knowledge_retrieval", display_name: "知识检索", status: "ACTIVE", member_assignable: true, manageable: grant,
      roles: [{ role_code: "app_admin", display_name: "知识管理员", scope_policy: "ALL_APP_DOMAINS", status: "ACTIVE", manageable: grant }],
      domains: [{ domain_id: "22", name: "知识空间", active: true }] },
    { app_id: "disabled_app", display_name: "已停用的 App", status: "DISABLED", member_assignable: true, manageable: grant,
      roles: [{ role_code: "app_admin", display_name: "管理员", scope_policy: "ALL_APP_DOMAINS", status: "ACTIVE", manageable: grant }],
      domains: [] },
  ];
  await page.addInitScript((user) => {
    sessionStorage.setItem("kbot.platform.session.v1", JSON.stringify({
      user_id: user, entry_kind: "PLATFORM", access_token: "mock-platform", expires_at: "2099-01-01T00:00:00Z",
    }));
  }, actor);
  await page.route(origin + "/**", async (route) => {
    const request = route.request(), url = new URL(request.url()), path = url.pathname;
    if (path === "/ui/runtime-config.js") {
      return route.fulfill({ contentType: "text/javascript", body: "window.KBOT_UI_CONFIG={mainApiBaseUrl: location.origin, enabledApps: []};" });
    }
    if (path.startsWith("/ui/")) return route.fulfill({ path: fileURLToPath(new URL(path.slice(1), root)) });
    assert.match(request.headers().authorization, /^Bearer mock-platform(-updated)?$/);
    const json = (body, status = 200) => route.fulfill({ json: body, status });
    const method = request.method(), body = request.postDataJSON();
    if (path === "/api/v1/auth/entries") return json({ apps: [], system_management: models, user_management: userManage });
    if (path === "/api/v1/platform/apps") return appManage ? json(catalog) : json({}, 403);
    if (path === "/api/v1/platform/app-grant-options") return json({ items: catalog });
    if (path.startsWith("/api/v1/platform/models")) return json([]);
    if (method !== "GET") writes.push({ path, method, body });
    if (path === "/api/v1/platform/apps/aiops/initial-admin/password") {
      return appManage ? json({ user_id: "initial-operator", must_change_password: true }) : json({}, 403);
    }
    if (path === "/api/v1/auth/password") {
      if (body.current_password !== "MockCurrent@2026") {
        return json({ detail: { code: "INVALID_CREDENTIALS", message: "当前密码错误" } }, 401);
      }
      return json({ user_id: actor, entry_kind: "PLATFORM", access_token: "mock-platform-updated", expires_at: "2099-01-01T00:00:00Z" });
    }
    if (path === base && method === "GET") {
      const state = url.searchParams.get("status"), search = url.searchParams.get("search");
      const items = [...accounts.values()]
        .filter((user) => (!state || user.status === state) && (!search || user.display_name.includes(search)))
        .map((user) => ({ ...user, assigned_app_names: user.app_memberships.map((item) => catalog.find((app) => app.app_id === item.app_id).display_name) }));
      return json({ items, total: items.length, offset: 0, limit: 50 });
    }
    if (path === base && method === "POST") {
      const user = account(body.user_id, body.display_name);
      accounts.set(user.user_id, user);
      return json(user, 201);
    }
    const match = path.slice(base.length).match(/^\/([^/]+)(.*)$/);
    const user = match && accounts.get(decodeURIComponent(match[1])), suffix = match?.[2];
    if (!user) return json({ detail: { message: "用户不存在" } }, 404);
    if (!suffix && method === "GET") return json(user);
    if (!suffix && method === "PATCH") {
      assert.ok(!("feishu_open_id" in body));
      Object.assign(user, body);
      return json(user);
    }
    if (suffix === "/password") return json({});
    if (suffix.startsWith("/app-grants/")) {
      const app_id = suffix.split("/").at(-1);
      user.app_memberships = user.app_memberships.filter((item) => item.app_id !== app_id);
      if (method === "PUT") user.app_memberships.push({ app_id, status: "ACTIVE", member_source: "PLATFORM_GRANT",
        role_bindings: body.role_bindings.map((item) => ({ ...item, status: "ACTIVE" })) });
      return json({});
    }
    if (!suffix && method === "DELETE") {
      accounts.delete(user.user_id);
      return json({});
    }
    return json({}, 404);
  });
  return { page, errors, writes, accounts };
}

try {
  const { page, errors, writes, accounts } = await setup({ models: true });
  const nativeDialogs = [];
  page.on("dialog", async (dialog) => { nativeDialogs.push(dialog.message()); await dialog.dismiss(); });
  await page.goto(origin + "/ui/index.html");
  await page.getByRole("link", { name: /平台管理/ }).click();
  await page.waitForURL("**/system_management/users.html");
  const nav = page.getByRole("navigation", { name: "平台管理" });
  const usersLink = nav.getByRole("link", { name: "用户管理" });
  const modelsLink = nav.getByRole("link", { name: "AI 模型管理" });
  await expect(usersLink).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("button", { name: "平台管理菜单", exact: true })).toBeHidden();
  const sidebar = await nav.boundingBox();
  const heading = await page.getByRole("heading", { name: "用户管理", exact: true }).boundingBox();
  assert.deepEqual({ x: sidebar.x, y: sidebar.y, width: sidebar.width }, { x: 0, y: 64, width: 240 });
  assert.ok(heading.x > sidebar.x + sidebar.width);
  await modelsLink.click();
  await page.waitForURL("**/system_management/models.html");
  await expect(modelsLink).toHaveAttribute("aria-current", "page");
  await expect(usersLink).not.toHaveAttribute("aria-current");
  await usersLink.click();
  await page.waitForURL("**/system_management/users.html");
  const directory = page.getByRole("table", { name: "用户目录" });
  await expect(directory.getByRole("columnheader")).toHaveCount(4);
  const protectedRow = directory.getByRole("row").filter({ hasText: "初始管理员" });
  await expect(protectedRow.getByRole("button", { name: "初始管理员" })).toBeVisible();
  await expect(protectedRow.getByText("受保护账号")).toBeVisible();
  await expect(protectedRow.getByRole("button", { name: /停用|启用|删除/ })).toHaveCount(0);
  await expect(protectedRow.getByRole("button", { name: "重置密码" })).toBeVisible();
  await expect(protectedRow.getByText("已启用", { exact: true })).toHaveCSS("border-radius", "999px");
  await expect(protectedRow).toHaveCSS("display", "table-row");
  await page.getByRole("button", { name: "新建用户" }).click();
  const create = page.getByRole("dialog", { name: "新建平台用户" });
  await create.getByLabel("账号", { exact: true }).fill(userId);
  await create.getByLabel("显示名称").fill("跨 App 测试用户");
  await create.getByLabel("初始密码").fill("MockPassword@Test26");
  await create.getByRole("button", { name: "创建并前往授权" }).click();
  const details = page.getByRole("dialog", { name: "用户详情与授权" });
  const closeDetails = details.getByRole("button", { name: "关闭用户详情" });
  await expect(details.getByRole("heading", { name: "跨 App 测试用户" })).toBeVisible();
  await expect(page).toHaveURL(origin + usersURL);
  await expect.poll(async () => {
    const { x, y, width, height } = await details.boundingBox();
    return [Math.round(x), y, width, height];
  }).toEqual([520, 0, 760, 850]);
  assert.equal(writes[0].body.user_id, userId);
  assert.equal(writes[0].body.must_change_password, true);
  assert.equal(writes[0].body.role_bindings, undefined);
  const aiops = page.locator(".user-app").filter({ has: page.getByRole("heading", { name: "AIOps", exact: true }) });
  const knowledge = page.locator(".user-app").filter({ has: page.getByRole("heading", { name: "知识检索", exact: true }) });
  const confirm = page.getByRole("alertdialog");
  const addApp = details.getByRole("button", { name: /授予 App 访问/ });
  const drawer = page.locator("#user-grant-dialog");
  const selectApp = drawer.getByLabel("App", { exact: true });
  const role = drawer.getByLabel("业务角色", { exact: true });
  const save = drawer.getByRole("button", { name: "确认授权" });
  await expect(details.getByText("暂无已授权的 App。")).toBeVisible();
  await expect(details.locator(".user-app")).toHaveCount(0);
  await addApp.click();
  await expect(drawer).toHaveAccessibleName("授予 App 访问");
  await expect(selectApp).toBeFocused();
  await expect(role).toBeDisabled();
  await expect(selectApp.locator("option")).toHaveText(["选择 App", "AIOps", "知识检索"]);
  await selectApp.selectOption("aiops");
  await role.selectOption("app_user");
  await expect(save).toBeDisabled();
  await expect(drawer.getByText("停用空间")).toHaveCount(0);
  await expect(drawer.getByText("知识空间")).toHaveCount(0);
  await drawer.getByRole("checkbox", { name: "生产空间" }).check();
  await selectApp.selectOption("knowledge_retrieval");
  await expect(role).toHaveValue("");
  await expect(drawer.getByRole("checkbox")).toHaveCount(0);
  await role.selectOption("app_admin");
  await expect(drawer.getByText("全部业务空间", { exact: true })).toBeVisible();
  await drawer.getByRole("button", { name: "取消" }).click();
  await expect(addApp).toBeFocused();
  assert.equal(writes.length, 1);
  await addApp.click();
  await selectApp.selectOption("aiops");
  await role.selectOption("app_user");
  await drawer.getByRole("checkbox", { name: "生产空间" }).check();
  await save.click();
  await expect(aiops.getByText("生产空间")).toBeVisible();
  await expect(details.getByRole("heading", { name: "App 授权", exact: true })).toBeFocused();
  await expect(knowledge).toHaveCount(0);
  assert.deepEqual(writes.at(-1), { path: profilePath + "/app-grants/aiops", method: "PUT",
    body: { role_bindings: [{ role_code: "app_user", scope_mode: "SELECTED_DOMAINS", domain_ids: [domainId] }] } });

  await aiops.getByRole("button", { name: /授予其他角色/ }).click();
  await expect(drawer).toHaveAccessibleName("授予业务角色");
  await expect(selectApp).toBeHidden();
  await expect(role.locator("option")).toHaveText(["选择角色", "运维管理员"]);
  await role.selectOption("app_admin");
  await drawer.getByRole("button", { name: "授予角色", exact: true }).click();
  await expect(aiops.getByText(/全部业务空间/)).toBeVisible();
  assert.deepEqual(writes.at(-1).body.role_bindings, [
    { role_code: "app_user", scope_mode: "SELECTED_DOMAINS", domain_ids: [domainId] },
    { role_code: "app_admin", scope_mode: "ALL_APP_DOMAINS", domain_ids: [] },
  ]);
  // 模拟空间停用：修改其他角色不应顺带清理旧授权，须显式编辑受影响的角色。
  accounts.get(userId).app_memberships[0].role_bindings[0].domain_ids.push("4");
  await details.getByRole("button", { name: "刷新" }).click();
  const adminRole = aiops.locator(".user-role").filter({ has: page.getByRole("heading", { name: "运维管理员", exact: true }) });
  const before = writes.length;
  await adminRole.getByRole("button", { name: "撤销角色" }).click();
  await confirm.getByRole("button", { name: "撤销角色" }).click();
  await expect(details.getByText(/现有角色“运维用户”或其空间范围已不可用/)).toBeVisible();
  await expect(adminRole).toBeVisible();
  assert.equal(writes.length, before);
  const operationalRole = aiops.locator(".user-role").filter({ has: page.getByRole("heading", { name: "运维用户", exact: true }) });
  await operationalRole.getByRole("button", { name: "编辑范围" }).click();
  await expect(drawer).toHaveAccessibleName("编辑角色范围");
  await expect(role).toHaveValue("app_user");
  await expect(role).toBeDisabled();
  await expect(drawer.getByRole("checkbox", { name: "生产空间" })).toBeChecked();
  await expect(drawer.getByText(/已停用的业务空间不会包含/)).toBeVisible();
  await drawer.getByRole("checkbox", { name: "生产空间" }).uncheck();
  await expect(drawer.getByRole("button", { name: "保存范围" })).toBeDisabled();
  await drawer.getByRole("checkbox", { name: "测试空间" }).check();
  await drawer.getByRole("button", { name: "保存范围" }).click();
  await expect(operationalRole.getByText("测试空间")).toBeVisible();
  assert.deepEqual(writes.at(-1).body.role_bindings, [
    { role_code: "app_admin", scope_mode: "ALL_APP_DOMAINS", domain_ids: [] },
    { role_code: "app_user", scope_mode: "SELECTED_DOMAINS", domain_ids: [testDomainId] },
  ]);

  await addApp.click();
  await expect(selectApp.locator("option")).toHaveText(["选择 App", "知识检索"]);
  await selectApp.selectOption("knowledge_retrieval");
  await role.selectOption("app_admin");
  await save.click();
  await expect(knowledge.getByText("已授权")).toBeVisible();
  await expect(addApp).toBeHidden();
  await closeDetails.click();
  await expect(details).toBeHidden();
  await expect(page.getByRole("button", { name: "新建用户" })).toBeFocused();
  const assignedRow = directory.getByRole("row").filter({ hasText: "跨 App 测试用户" });
  await expect(assignedRow.getByText("2 个 App")).toBeVisible();
  await expect(assignedRow.getByText("AIOps、知识检索")).toBeVisible();
  await assignedRow.getByRole("button", { name: "跨 App 测试用户" }).click();
  await expect(aiops.getByText("已授权")).toBeVisible();
  await expect(details.getByRole("region", { name: "账号资料" }).getByLabel("安全等级")).toHaveCount(0);
  await expect(details.getByRole("button", { name: /删除|停用|启用|重置密码/ })).toHaveCount(0);
  await expect(details.getByText(/飞书/)).toHaveCount(0);
  const authorization = details.getByRole("region", { name: "用户授权" });
  await authorization.getByLabel("安全等级").selectOption("2");
  await authorization.getByRole("button", { name: "保存安全等级" }).click();
  await expect(page.locator("#user-security")).toHaveText("2");
  assert.deepEqual(writes.at(-1), { path: profilePath, method: "PATCH", body: { max_security_level: 2 } });
  await operationalRole.getByRole("button", { name: "撤销角色" }).click();
  await confirm.getByRole("button", { name: "撤销角色" }).click();
  await expect(operationalRole).toHaveCount(0);
  await expect(aiops.getByText("运维管理员")).toBeVisible();
  await expect(aiops.getByRole("button", { name: "撤销角色" })).toHaveCount(0);
  assert.deepEqual(writes.at(-1), { path: profilePath + "/app-grants/aiops", method: "PUT",
    body: { role_bindings: [{ role_code: "app_admin", scope_mode: "ALL_APP_DOMAINS", domain_ids: [] }] } });
  const revokeApp = aiops.getByRole("button", { name: "撤销整个 App 访问" });
  await revokeApp.click();
  await confirm.getByRole("button", { name: "取消" }).click();
  await expect(revokeApp).toBeFocused();
  await revokeApp.click();
  await confirm.getByRole("button", { name: "撤销 App 访问" }).click();
  await expect(aiops).toHaveCount(0);
  await expect(knowledge.getByText("已授权")).toBeVisible();
  await expect(addApp).toBeVisible();
  assert.deepEqual(writes.at(-1), { path: profilePath + "/app-grants/aiops", method: "DELETE", body: null });
  await closeDetails.click();
  await expect(details).toBeHidden();
  await assignedRow.getByRole("button", { name: "停用", exact: true }).click();
  await confirm.getByRole("button", { name: "停用账号" }).click();
  await expect(assignedRow.getByText("已停用", { exact: true })).toBeVisible();
  await expect(assignedRow.getByText("知识检索", { exact: true })).toBeVisible();
  await assignedRow.getByRole("button", { name: "启用", exact: true }).click();
  await confirm.getByRole("button", { name: "启用账号" }).click();
  await expect(assignedRow.getByText("已启用", { exact: true })).toBeVisible();
  await assignedRow.getByRole("button", { name: "重置密码" }).click();
  await page.getByLabel("临时密码").fill("MockPassword@Reset2026");
  await page.getByRole("button", { name: "确认重置" }).click();
  await expect(page.getByText(/密码已重置/)).toBeVisible();
  await assignedRow.getByRole("button", { name: "删除" }).click();
  await confirm.getByRole("button", { name: "删除用户" }).click();
  await expect(details).toBeHidden();
  await expect(page).toHaveURL(origin + usersURL);
  await expect(assignedRow).toHaveCount(0);

  // 受保护 App 初始管理员通过 App 专用入口重置，不开放账号或授权变更。
  await protectedRow.getByRole("button", { name: "重置密码" }).click();
  const reset = page.getByRole("dialog", { name: "重置密码" });
  await expect(reset.getByText("初始管理员 · initial-operator · AIOps")).toBeVisible();
  await expect(reset.getByLabel("当前密码")).toBeHidden();
  await reset.getByLabel("临时密码").fill("MockAppAdmin@2026");
  await reset.getByRole("button", { name: "确认重置" }).click();
  await expect(reset).toBeHidden();
  await expect(page.getByText(/密码已重置/)).toBeVisible();
  assert.deepEqual(writes.at(-1), { path: "/api/v1/platform/apps/aiops/initial-admin/password", method: "POST",
    body: { password: "MockAppAdmin@2026", must_change_password: true } });
  const globalRow = directory.getByRole("row").filter({ hasText: "全局管理员" });
  await expect(globalRow.getByRole("button", { name: /重置密码|启用|停用|删除/ })).toHaveCount(0);
  await globalRow.getByRole("button", { name: "修改密码" }).click();
  const modify = page.getByRole("dialog", { name: "修改密码" });
  await expect(modify.getByLabel("当前密码")).toBeFocused();
  await expect(modify.getByLabel("新密码")).toHaveValue("");
  await modify.getByLabel("当前密码").fill("WrongMockCurrent");
  await modify.getByLabel("新密码").fill("MockNewAdmin@2026");
  await modify.getByRole("button", { name: "确认修改" }).click();
  await expect(modify.getByText("当前密码错误")).toBeVisible();
  await modify.getByLabel("当前密码").fill("MockCurrent@2026");
  await modify.getByRole("button", { name: "确认修改" }).click();
  await expect(modify).toBeHidden();
  await expect(page.getByText("密码已修改。")).toBeVisible();
  assert.deepEqual(writes.at(-1), { path: "/api/v1/auth/password", method: "POST",
    body: { current_password: "MockCurrent@2026", new_password: "MockNewAdmin@2026" } });
  assert.equal(await page.evaluate(() => JSON.parse(sessionStorage.getItem("kbot.platform.session.v1")).access_token), "mock-platform-updated");

  const viewOnly = protectedRow.getByRole("button", { name: "查看", exact: true });
  await viewOnly.click();
  await expect(details.getByText("这是受保护的初始化账号，不能修改资料、状态或授权。")).toBeVisible();
  await expect(details.getByRole("button", { name: "删除用户" })).toBeHidden();
  await page.keyboard.press("Escape");
  await expect(details).toBeHidden();
  await expect(viewOnly).toBeFocused();
  assert.deepEqual(errors, []);
  assert.deepEqual(nativeDialogs, []);
  await page.close();

  const scoped = await setup({ grant: false, appManage: false, width: 390, actor: "user-manager" });
  await scoped.page.goto(origin + usersURL);
  await expect(scoped.page.getByRole("heading", { name: "用户目录" })).toBeVisible();
  assert.ok(await scoped.page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  const mobileTable = scoped.page.getByRole("table", { name: "用户目录" });
  for (const name of ["初始管理员", "全局管理员"]) {
    const row = mobileTable.getByRole("row").filter({ hasText: name });
    await expect(row).toBeVisible();
    await expect(row.getByRole("button", { name: /修改密码|重置密码/ })).toHaveCount(0);
  }
  const mobileRow = mobileTable.getByRole("row").filter({ hasText: "只读授权用户" });
  await expect(mobileTable.getByRole("columnheader")).toHaveCount(4);
  await expect(mobileRow).toHaveCSS("display", "grid");
  await expect(mobileRow.getByText("账号状态", { exact: true })).toBeVisible();
  const mobileAction = mobileRow.getByRole("button", { name: "查看与授权" });
  await expect(mobileAction).toBeVisible();
  const actionBox = await mobileAction.boundingBox();
  assert.ok(actionBox.x >= 0 && actionBox.x + actionBox.width <= 390);
  assert.ok(await mobileTable.evaluate((table) => table.scrollWidth <= table.clientWidth));
  await scoped.page.getByRole("searchbox", { name: "搜索用户" }).fill("不存在的账号");
  await scoped.page.getByRole("button", { name: "查询" }).click();
  await expect(mobileTable.getByText("没有符合条件的用户。")).toBeVisible();
  await scoped.page.getByRole("searchbox", { name: "搜索用户" }).fill("只读");
  await scoped.page.getByRole("button", { name: "查询" }).click();
  const trigger = scoped.page.getByRole("button", { name: "平台管理菜单", exact: true });
  const mobileNav = scoped.page.getByRole("navigation", { name: "平台管理" });
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(mobileNav).toBeHidden();
  await trigger.click();
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  const mobileUsers = mobileNav.getByRole("link", { name: "用户管理" });
  await expect(mobileUsers).toBeFocused();
  await expect(mobileNav.locator("a")).toHaveCount(1);
  await scoped.page.keyboard.press("Escape");
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(trigger).toBeFocused();
  await expect(mobileNav).toBeHidden();
  await trigger.click();
  await scoped.page.getByRole("button", { name: "关闭平台管理菜单" }).click({ position: { x: 345, y: 60 } });
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(trigger).toBeFocused();
  await mobileAction.click();
  const mobileDetails = scoped.page.getByRole("dialog", { name: "用户详情与授权" });
  await expect(mobileDetails.getByText(/没有 App 授权权限/)).toBeVisible();
  await expect(mobileDetails.getByRole("button", { name: /授予 App 访问/ })).toHaveCount(0);
  await expect(mobileDetails).toHaveCSS("width", "390px");
  await expect(mobileDetails.locator("#user-detail-body")).toHaveCSS("overflow-y", "auto");
  await mobileDetails.getByRole("button", { name: "关闭", exact: true }).click();
  await expect(mobileDetails).toBeHidden();
  await expect(mobileAction).toBeFocused();
  await expect(scoped.page.getByRole("searchbox", { name: "搜索用户" })).toHaveValue("只读");
  await expect(mobileTable.getByText("初始管理员")).toHaveCount(0);
  assert.deepEqual(scoped.errors, []);
  await scoped.page.close();

  const denied = await setup({ userManage: false });
  await denied.page.goto(origin + usersURL);
  await expect(denied.page.getByText("当前账号没有用户管理权限。")).toBeVisible();
  await expect(denied.page.getByRole("button", { name: "新建用户" })).toBeHidden();
  await expect(denied.page.locator("#platform-navigation")).toHaveCount(0);
  await denied.page.close();
  console.log("平台用户浏览器回归通过：目录表格、响应式详情抽屉、保留筛选、跨 App 授权和账号维护。");
} finally {
  await browser.close();
}
