// 统一入口开通知识检索首域的回归：真实页面与会话脚本，只使用 Mock API。
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(process.env.KBOT_BROWSER_REQUIRE_FROM
  || fileURLToPath(new URL("../../../ammolite_cube/apps/portal-web/package.json", import.meta.url)));
const { chromium, expect } = require("@playwright/test");
const origin = "http://knowledge-entry.test";
const root = new URL("../../", import.meta.url);
const workspace = { access_token: "mock-initial", entry_kind: "WORKSPACE", user_id: "example" };
const browser = await chromium.launch({ headless: true });

try {
  const page = await browser.newPage();
  const calls = [];
  await page.route(`${origin}/**`, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === "/ui/runtime-config.js") {
      return route.fulfill({ contentType: "text/javascript", body: "globalThis.KBOT_UI_CONFIG={mainApiBaseUrl:location.origin};" });
    }
    if (path === "/ui/knowledge_retrieval/knowledge.html") return route.fulfill({ contentType: "text/html", body: "" });
    if (path.startsWith("/ui/")) return route.fulfill({ path: fileURLToPath(new URL(path.slice(1), root)) });
    calls.push({ path, bearer: request.headers().authorization, body: request.postDataJSON() });
    switch (path) {
      case "/api/v1/auth/login":
        return route.fulfill({ json: { ...workspace, must_change_password: true } });
      case "/api/v1/auth/password":
        return route.fulfill({ json: { ...workspace, access_token: "mock-updated", must_change_password: false } });
      case "/api/v1/auth/entries":
        return route.fulfill({ json: { system_management: false, apps: [{
          app_id: "knowledge_retrieval", display_name: "知识检索", domains: [], can_create_first_domain: true,
        }] } });
      case "/api/v1/auth/knowledge-first-domain":
        return route.fulfill({ json: {
          access_token: "mock-business", entry_kind: "BUSINESS", user_id: "example",
          app_id: "knowledge_retrieval", domain_id: 7,
        } });
      default:
        return route.fulfill({ status: 404 });
    }
  });
  for (const app of ["aiops", "media_studio", "knowledge_retrieval"]) {
    await page.goto(`${origin}/ui/${app}/login.html`);
    await page.waitForURL("**/ui/index.html");
    await expect(page.locator("#portal-sign-in")).toBeVisible();
  }
  await page.locator("#portal-login [name=user_id]").fill("example");
  await page.locator("#portal-login [name=password]").fill("MockPassword@Test26");
  await page.locator("#portal-login button").click();
  await expect(page.locator("#portal-password")).toBeVisible();
  assert.deepEqual(calls.map(({ path }) => path), ["/api/v1/auth/login"]);
  await page.locator("#portal-password [name=current_password]").fill("MockPassword@Test26");
  await page.locator("#portal-password [name=new_password]").fill("NewMockPassword@Test27");
  await page.locator("#portal-password button").click();
  await page.locator('[data-app="knowledge_retrieval"]').click();
  await expect(page.locator("#portal-first-domain-dialog")).toBeVisible();
  assert.equal(await page.evaluate(() => sessionStorage.getItem("kbot.knowledge.session.v1")), null);
  await page.getByLabel("工作域名称").fill("研发知识");
  await page.getByRole("button", { name: "创建并进入" }).click();
  await page.waitForURL("**/knowledge.html");
  assert.deepEqual(calls.map(({ path, bearer }) => [path, bearer]), [
    ["/api/v1/auth/login", undefined],
    ["/api/v1/auth/password", "Bearer mock-initial"],
    ["/api/v1/auth/entries", "Bearer mock-updated"],
    ["/api/v1/auth/knowledge-first-domain", "Bearer mock-updated"],
  ]);
  assert.deepEqual(calls.at(-1).body, { name: "研发知识", description: null });
  const saved = await page.evaluate(() => JSON.parse(sessionStorage.getItem("kbot.knowledge.session.v1")));
  assert.equal(saved.access_token, "mock-business");
  assert.equal(saved.domain_id, 7);
  console.log("知识检索浏览器回归通过：旧入口统一跳转、首次改密、免密开通首域。");
} finally {
  await browser.close();
}
