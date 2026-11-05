// 知识检索视觉与交互回归：真实静态页面、隔离 API，不读取或修改真实业务数据。
// node tests/contract/knowledge_retrieval_ui_style.mjs
// 默认复用 Portal 的浏览器测试依赖；可用 KBOT_BROWSER_REQUIRE_FROM 指定其他 package.json。
import assert from "node:assert/strict";
import { readFile, mkdtemp } from "node:fs/promises";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const require = createRequire(process.env.KBOT_BROWSER_REQUIRE_FROM || resolve(root, "../ammolite_cube/apps/portal-web/package.json"));
const { chromium } = require("@playwright/test");
const artifacts = await mkdtemp(resolve(tmpdir(), "knowledge-retrieval-ui-"));
const permissions = ["use", "knowledge_chat", "x_search", "domain_manage", "knowledge_core_manage", "data_model_manage", "agent_manage"].map(key => `knowledge_retrieval:${key}`);
const access = { permissions, domain_id: 1, domain_name: "企业知识工作域", display_name: "知识管理员", user_id: "fixture-user" };
const agent = { agent_id: "agent-1", display_name: "企业知识助手", status: "ACTIVE", version_no: 1, knowledge_core_id: "core-1", data_model_ids: [], models: { x_search_llm: "grok-fixture" } };
const domain = { domain_id: 1, domain_code: "fixture", display_name: "企业知识工作域", status: "ACTIVE", row_version: 1 };
const server = createServer(async (req, res) => {
  const pathname = new URL(req.url, "http://localhost").pathname;
  if (pathname === "/ui/runtime-config.js") {
    res.setHeader("Content-Type", "application/javascript");
    res.end("globalThis.KBOT_UI_CONFIG={mainApiBaseUrl:location.origin};");
    return;
  }
  const target = resolve(root, `.${decodeURIComponent(pathname)}`);
  if (!target.startsWith(resolve(root, "ui") + sep)) { res.writeHead(403).end(); return; }
  try {
    res.setHeader("Content-Type", ({ ".html": "text/html", ".css": "text/css", ".js": "application/javascript" })[extname(target)] || "application/octet-stream");
    res.end(await readFile(target));
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const errors = [], unexpected = [], writes = [], domainSwitches = [];
let browser;

async function newPage(width, granted = permissions, session = true, switchMode = "") {
  const page = await browser.newPage({ viewport: { width, height: 1000 } });
  page.setDefaultTimeout(10000);
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => {
    if (message.type() !== "error") return;
    if (switchMode === "reject" && message.text().includes("403 (Forbidden)")) return;
    errors.push(message.text());
  });
  page.on("response", response => {
    if (response.status() >= 400 && !(switchMode === "reject" && response.status() === 403 && response.url().endsWith("/auth/switch-domain"))) errors.push(`${response.status()} ${response.url()}`);
  });
  if (session) await page.addInitScript(value => {
    if (!sessionStorage.getItem("kbot.knowledge.session.v1")) sessionStorage.setItem("kbot.knowledge.session.v1", JSON.stringify({ ...value, access_token: "fixture-session" }));
  }, access);
  page.domainReads = [];
  await page.route("**/api/**", async route => {
    const { pathname } = new URL(route.request().url());
    const path = pathname.replace("/api/v1/apps/knowledge-retrieval", "");
    const method = route.request().method();
    if (method !== "GET" && !(switchMode && path === "/api/v1/auth/switch-domain")) writes.push(`${method} ${path}`);
    const switched = route.request().headers().authorization === "Bearer fixture-domain-2";
    const current = switched ? { domain_id: 2, domain_name: "客户资料" } : { domain_id: 1, domain_name: access.domain_name };
    if (path === "/knowledge-cores") page.domainReads.push(current.domain_id);
    let result;
    if (path === "/api/v1/auth/domains") result = { app_id: "knowledge_retrieval", domain_id: current.domain_id, domains: [{ domain_id: 1, name: access.domain_name, status: "ACTIVE" }, { domain_id: 2, name: "客户资料", status: "ACTIVE" }, { domain_id: 3, name: "已停用", status: "DISABLED" }] };
    else if (path === "/api/v1/auth/switch-domain" && switchMode) {
      domainSwitches.push(route.request().postDataJSON());
      if (switchMode === "reject") {
        await route.fulfill({ status: 403, contentType: "application/json", body: JSON.stringify({ detail: { code: "DOMAIN_ACCESS_DENIED", message: "所选工作域访问权限已撤销" } }) });
        return;
      }
      result = { ...access, domain_id: 2, domain_name: "客户资料", app_id: "knowledge_retrieval", access_token: "fixture-domain-2", entry_kind: "BUSINESS", must_change_password: false };
    }
    else if (path === "/access") result = { ...access, ...current, permissions: switched ? granted.filter(value => value !== "knowledge_retrieval:agent_manage") : granted };
    else if (path === "/agents") result = [agent];
    else if (path === "/agents/options") result = { knowledge_cores: [], data_models: [], models: [] };
    else if (path === "/domains") result = { items: [domain] };
    else if (["/model-catalog", "/knowledge-cores", "/data-models", "/data-models/connector-capabilities", "/data-models/data-sources", "/conversations", "/x-search/runs"].includes(path)) result = { items: [] };
    else { unexpected.push(`${method} ${pathname}`); result = {}; }
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(result) });
  });
  return page;
}

try {
  browser = await chromium.launch({ headless: true });
  const pages = Object.keys(JSON.parse(await readFile(resolve(root, "ui/knowledge_retrieval/reference-pages.json"), "utf8")).pages);
  if (!process.env.KNOWLEDGE_UI_SWITCH_ONLY) for (const width of [1440, 1024, 390, 320]) {
    for (const name of pages) {
      console.log(`检查：${name}/${width}`);
      const page = await newPage(width, permissions, name !== "login");
      await page.goto(`${origin}/ui/knowledge_retrieval/${name}.html`);
      if (name !== "login") await page.waitForFunction(() => document.body.dataset.access === "ready");
      await page.waitForLoadState("networkidle");
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${name}/${width} 页面横向溢出`);
      assert.equal(await page.evaluate(() => getComputedStyle(document.body).getPropertyValue("--accent").trim()), "#087f74");
      if (name !== "login") {
        assert.equal(await page.locator("#knowledge-shell-content > main").count(), 1);
        assert.equal(await page.locator('.business-nav-item[aria-current="page"]').count(), 1);
        const sidebar = page.locator("#knowledge-app-navigation");
        if (width > 900) {
          assert.equal(Math.round((await sidebar.boundingBox()).width), 276);
          assert.equal(await page.locator("#knowledge-nav-trigger").isVisible(), false);
        } else {
          assert.equal(await sidebar.isVisible(), false);
          await page.locator("#knowledge-nav-trigger").click();
          await sidebar.waitFor({ state: "visible" });
          assert.equal(await sidebar.isVisible(), true);
          await page.keyboard.press("Escape");
          await sidebar.waitFor({ state: "hidden" });
          assert.equal(await sidebar.isVisible(), false);
          assert.equal(await page.locator("#knowledge-nav-trigger").getAttribute("aria-expanded"), "false");
        }
        assert.equal(await page.locator("#knowledge-account-menu").isVisible(), false);
        await page.locator("#knowledge-account-toggle").click();
        assert.equal(await page.locator("#knowledge-account-menu").isVisible(), true);
        await page.keyboard.press("Escape");
        assert.equal(await page.locator("#knowledge-account-menu").isVisible(), false);
      } else {
        const bounds = await page.locator(".knowledge-login-card").boundingBox();
        assert.ok(Math.abs(bounds.x + bounds.width / 2 - width / 2) < 2, "登录卡片应居中");
        assert.equal(await page.locator("#password-section").isVisible(), false);
      }
      if (name === "knowledge") assert.equal(await page.locator("#knowledge-agent").inputValue(), "agent-1");
      if (name === "x-search") assert.equal(await page.locator("#x-search-agent option").count(), 2);
      if (name === "agents") {
        await page.locator("#agent-create").click();
        assert.equal(await page.locator("#agent-dialog").isVisible(), true);
        assert.equal(await page.locator("#agent-dialog").evaluate(node => node.scrollWidth > node.clientWidth + 1), false);
        await page.screenshot({ path: resolve(artifacts, `agent-dialog-${width}.png`), fullPage: true });
        await page.locator('#agent-dialog .knowledge-close').click();
      }
      await page.locator("main h1:visible").first().click();
      await page.screenshot({ path: resolve(artifacts, `${name}-${width}.png`), fullPage: true });
      await page.close();
    }
  }
  // 无管理权限时不渲染空资源分组，也不展示未授权的业务入口。
  const limited = await newPage(1440, ["knowledge_retrieval:use"]);
  await limited.goto(`${origin}/ui/knowledge_retrieval/dashboard.html`);
  await limited.waitForFunction(() => document.body.dataset.access === "ready");
  assert.equal(await limited.locator(".business-nav-group").count(), 1);
  assert.equal(await limited.locator(".business-nav-item").count(), 1);
  assert.equal(await limited.locator(".business-nav-group-count").innerText(), "1");
  assert.equal(await limited.locator(".knowledge-entry a").count(), 0);
  await limited.goto(`${origin}/ui/knowledge_retrieval/agents.html`);
  await limited.waitForURL("**/dashboard.html");
  await limited.waitForFunction(() => document.body.dataset.access === "ready");
  // 窄屏分组展开、遮罩关闭及跨页导航均使用正式 Shell。
  const navigation = await newPage(390);
  await navigation.goto(`${origin}/ui/knowledge_retrieval/dashboard.html`);
  await navigation.waitForFunction(() => document.body.dataset.access === "ready");
  await navigation.locator("#knowledge-nav-trigger").click();
  await navigation.locator("#knowledge-app-navigation").waitFor({ state: "visible" });
  await navigation.locator('[aria-controls="knowledge-nav-resources"]').click();
  assert.equal(await navigation.locator("#knowledge-nav-resources").isVisible(), true);
  assert.equal(await navigation.locator("#knowledge-nav-workspace").isVisible(), false);
  await navigation.locator("#knowledge-nav-scrim").click({ position: { x: 380, y: 100 } });
  await navigation.locator("#knowledge-app-navigation").waitFor({ state: "hidden" });
  await navigation.locator("#knowledge-nav-trigger").click();
  await navigation.locator('a[href="./agents.html"]').click();
  await navigation.waitForURL("**/agents.html");
  await navigation.waitForFunction(() => document.body.dataset.access === "ready");
  assert.equal((await navigation.locator('#knowledge-nav-resources a[aria-current="page"]').textContent()).trim(), "Agent 管理");
  assert.equal(await navigation.locator("#knowledge-app-navigation").isVisible(), false);
  // 未登录仍跳登录；首次改密仍在原有流程内完成。
  const anonymous = await newPage(390, permissions, false);
  await anonymous.goto(`${origin}/ui/knowledge_retrieval/agents.html`);
  await anonymous.waitForURL("**/login.html");
  const firstLogin = await newPage(390);
  await firstLogin.addInitScript(() => {
    const value = JSON.parse(sessionStorage.getItem("kbot.knowledge.session.v1"));
    sessionStorage.setItem("kbot.knowledge.session.v1", JSON.stringify({ ...value, must_change_password: true }));
  });
  await firstLogin.goto(`${origin}/ui/knowledge_retrieval/login.html`);
  assert.equal(await firstLogin.locator("#password-section").isVisible(), true);
  assert.equal(await firstLogin.locator("#login-section").isVisible(), false);
  // 知识库页无需 Domain 管理权限，即可发现、切换已授权工作域。
  const readerPermissions = permissions.filter(value => value !== "knowledge_retrieval:domain_manage");
  for (const width of [1440, 390, 320]) {
    const page = await newPage(width, readerPermissions, true, "switch");
    await page.goto(`${origin}/ui/knowledge_retrieval/knowledge-cores.html`);
    await page.waitForFunction(() => document.body.dataset.access === "ready");
    await page.waitForLoadState("networkidle");
    await page.evaluate(() => { const marker = document.createElement("p"); marker.id = "old-domain-resource"; marker.textContent = "旧工作域资源"; document.querySelector("main").append(marker); });
    assert.equal(await page.locator('a[href="./domains.html"]').count(), 0);
    await page.locator("#knowledge-domain-toggle").click();
    await page.locator('[data-switch-domain="2"]').waitFor({ state: "visible" });
    assert.equal(await page.locator('[data-switch-domain="1"]').isDisabled(), true);
    assert.equal(await page.locator('[data-switch-domain="3"]').count(), 0);
    const bounds = await page.locator("#knowledge-domain-menu").boundingBox();
    assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width + 1, "工作域菜单不得超出屏幕");
    await page.screenshot({ path: resolve(artifacts, `domain-switcher-${width}.png`), fullPage: true });
    await Promise.all([page.waitForNavigation({ waitUntil: "networkidle" }), page.locator('[data-switch-domain="2"]').click()]);
    await page.waitForFunction(() => document.body.dataset.access === "ready");
    assert.ok(page.url().endsWith("/knowledge-cores.html"));
    assert.equal(await page.locator("#knowledge-domain-toggle .business-context-copy strong").innerText(), "客户资料");
    assert.equal(await page.locator("#old-domain-resource").count(), 0);
    assert.equal(await page.locator('a[href="./agents.html"]').count(), 0, "新工作域必须重新获取权限");
    assert.deepEqual(page.domainReads, [1, 2], "知识库请求必须从旧凭据切换到新凭据");
    const saved = await page.evaluate(() => JSON.parse(sessionStorage.getItem("kbot.knowledge.session.v1")));
    assert.equal(saved.domain_id, 2);
    assert.equal(saved.access_token, "fixture-domain-2");
    await page.close();
  }
  const rejected = await newPage(390, readerPermissions, true, "reject");
  await rejected.goto(`${origin}/ui/knowledge_retrieval/knowledge-cores.html`);
  await rejected.waitForFunction(() => document.body.dataset.access === "ready");
  await rejected.locator("#knowledge-domain-toggle").click();
  await rejected.locator('[data-switch-domain="2"]').click();
  await rejected.locator(".knowledge-toast.error").waitFor({ state: "visible" });
  assert.equal(await rejected.locator("#knowledge-domain-toggle .business-context-copy strong").innerText(), access.domain_name);
  assert.equal(await rejected.locator('[data-switch-domain="2"]').isEnabled(), true, "失败后应允许重试");
  assert.equal(await rejected.evaluate(() => JSON.parse(sessionStorage.getItem("kbot.knowledge.session.v1")).domain_id), 1);
  assert.deepEqual(domainSwitches, [{ domain_id: 2 }, { domain_id: 2 }, { domain_id: 2 }, { domain_id: 2 }]);
  assert.deepEqual(unexpected, [], "出现未声明 API 请求");
  assert.deepEqual(writes, [], "样式验收不得写入业务数据");
  assert.deepEqual(errors, [], "页面出现浏览器错误");
  console.log(`通过：${process.env.KNOWLEDGE_UI_SWITCH_ONLY ? "工作域定向回归" : "13 个页面 × 4 个宽度"}、导航、权限、登录与知识库工作域切换（含失败保留原会话）。截图：${artifacts}`);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
