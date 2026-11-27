// 使用正式页面和真实前端鉴权，只在原标签页建立会话，API 全部隔离。
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(process.env.KBOT_BROWSER_REQUIRE_FROM
  || fileURLToPath(new URL("../../../ammolite_cube/apps/portal-web/package.json", import.meta.url)));
const { chromium, expect } = require("@playwright/test");
const origin = "http://kbot-agent-resource.test";
const root = new URL("../../ui/", import.meta.url);
const agentPage = "/ui/aiops/agents.html";
const workspace = "/ui/aiops/business-system-workspace.html";
const api = "/api/v1/apps/aiops";
const token = "fixture-aiops-session";
const systemId = "019ffff0-0000-7000-8000-000000000033";
const system = { business_system_id: systemId, system_code: "LAB", display_name: "隔离业务系统", status: "ENABLED",
  environment: "TEST", importance_level: 3, resources: [], relations: [] };
const agent = { agent_id: "agent-1", display_name: "隔离 Agent", status: "ACTIVE", version_no: 1,
  allowed_business_system_ids: [systemId], capabilities: ["MONITORING_QUERY"], knowledge_policy: { enabled: false } };
const responses = new Map([
  [`${api}/access`, { permissions: ["aiops:use", "aiops:agent_manage", "aiops:target_manage"], domain_id: 7, display_name: "测试用户" }],
  [`${api}/agents`, [agent]], [`${api}/agents/agent-1`, agent], [`${api}/business-systems`, { items: [system] }],
  [`${api}/business-systems/${systemId}`, system],
  [`${api}/agents/boundary-resources`, { items: [{ resource_id: "resource-1", display_name: "测试资源",
    business_system_ids: [systemId], resource_type_key: "VIRTUAL_MACHINE", monitoring_configured: true,
    target_enabled: true, scope_enabled: true }] }],
  ["/api/v1/model-catalog", []], [`${api}/operations-knowledge/tags`, []], [`${api}/operations-knowledge/knowledge-bases`, []],
  [`${api}/responsibility-groups`, { items: [] }], [`${api}/diagnostic-sources`, { items: [] }],
  ["/api/v1/notifications/summary", { unread: 0 }],
]);

const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext();
  const unexpected = [], systemReads = [];
  const errors = [];
  context.on("page", page => page.on("pageerror", error => errors.push(error.message)));
  await context.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    const path = url.pathname;
    if (path.startsWith("/api/")) {
      if (route.request().method() !== "GET" || route.request().headers().authorization !== `Bearer ${token}` || !responses.has(path)) {
        unexpected.push(`${route.request().method()} ${path}`);
        return route.fulfill({ status: 401, json: { detail: "未通过隔离会话或路由验证" } });
      }
      if (path === `${api}/business-systems/${systemId}`) systemReads.push(path);
      return route.fulfill({ json: responses.get(path) });
    }
    if (path === "/ui/runtime-config.js") return route.fulfill({ contentType: "application/javascript", body: `globalThis.KBOT_UI_CONFIG = {mainApiBaseUrl: ${JSON.stringify(origin)}};` });
    const type = { html: "text/html", css: "text/css", js: "application/javascript" }[path.split(".").pop()];
    if (path.startsWith("/ui/") && type) {
      try {
        return await route.fulfill({ contentType: `${type}; charset=utf-8`, body: await readFile(new URL(path.slice(4), root)) });
      } catch { return route.fulfill({ status: 404 }); }
    }
    return route.fulfill({ status: 404 });
  });

  const page = await context.newPage();
  // 先加载同源空白页再写测试会话，绝不能向新标签页预注入会话。
  await context.route(`${origin}/fixture-start`, route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto(`${origin}/fixture-start`);
  await page.evaluate(value => sessionStorage.setItem("kbot.aiops.session.v1", JSON.stringify({ access_token: value })), token);
  await page.goto(origin + agentPage);
  await page.getByRole("button", { name: "编辑", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "编辑 Agent" });
  await dialog.getByLabel("Agent 名称").fill("未保存的 Agent 名称");
  await dialog.getByRole("button", { name: /诊断边界/ }).click();
  const openResource = dialog.getByText("查看业务系统资源 ↗", { exact: true });
  const [popup] = await Promise.all([page.waitForEvent("popup"), openResource.click()]);
  await expect(popup.locator("#workspace-title")).toHaveText(system.display_name, { timeout: 5000 });
  const target = new URL(popup.url());
  assert.equal(target.origin, origin);
  assert.equal(target.pathname, workspace);
  assert.deepEqual(Object.fromEntries(target.searchParams), { id: systemId, view: "readonly" });
  assert.equal(await popup.evaluate(() => window.opener), null, "业务系统页不能保留对原标签页的引用");
  await expect(popup.locator("#save-draft")).toBeHidden();
  assert.equal(systemReads.length, 1);
  await popup.close();

  await page.evaluate(() => { window.open = () => null; });
  await openResource.click();
  await expect(dialog.locator(".ops-toast")).toContainText("浏览器阻止了新标签页");
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: /基本信息/ }).click();
  await expect(dialog.getByLabel("Agent 名称")).toHaveValue("未保存的 Agent 名称");
  assert.equal(new URL(page.url()).pathname, agentPage);

  const anonymous = await context.newPage();
  await anonymous.goto(`${origin}${workspace}?id=${systemId}&view=readonly`);
  await expect(anonymous).toHaveURL(`${origin}/ui/index.html`);
  assert.equal(systemReads.length, 1, "无会话的标签页不能读取资源");
  assert.deepEqual(unexpected, []);
  assert.deepEqual(errors, []);
  console.log("通过：业务系统新标签页复用当前会话、保持只读且不保留 opener；拦截弹窗时保留 Agent 草稿。");
} finally {
  await browser.close();
}
