// 知识检索登录浏览器回归：真实页面与会话脚本，只使用 Mock API。
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(process.env.KBOT_BROWSER_REQUIRE_FROM
  || fileURLToPath(new URL("../../../ammolite_cube/apps/portal-web/package.json", import.meta.url)));
const { chromium } = require("@playwright/test");
const origin = "http://knowledge-entry.test";
const root = "/ui/knowledge_retrieval/";
const auth = "/api/v1/apps/knowledge-retrieval/auth";
const html = await readFile(new URL("../../ui/knowledge_retrieval/login.html", import.meta.url), "utf8");
const script = await readFile(new URL("../../ui/knowledge_retrieval/js/knowledge-auth.js", import.meta.url), "utf8");

const browser = await chromium.launch({ headless: true });

async function exercise(challenge, entries, interact) {
  const page = await browser.newPage();
  const observed = [];
  const session = { action: "ENTERED", access_token: "mock-business-token", domain_id: 7 };
  await page.route(`${origin}/**`, async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (pathname === root + "login.html") {
      return route.fulfill({ contentType: "text/html; charset=utf-8", body: html });
    }
    if (pathname === "/ui/runtime-config.js") {
      return route.fulfill({ contentType: "text/javascript", body: "globalThis.KBOT_UI_CONFIG={mainApiBaseUrl:location.origin};" });
    }
    if (pathname === root + "js/knowledge-auth.js") {
      return route.fulfill({ contentType: "text/javascript", body: script });
    }
    if (pathname === auth + "/login") return route.fulfill({ json: challenge || session });
    if (pathname === auth + "/entry") {
      const { action, ...payload } = request.postDataJSON();
      const result = entries[action];
      observed.push({ action, payload, bearer: request.headers().authorization });
      return route.fulfill({ json: result || session });
    }
    return route.fulfill({ contentType: "text/html", body: "" });
  });
  try {
    await page.goto(origin + root + "login.html");
    await page.locator("#login-password").fill("MockPassword@Test26");
    await page.locator("#login-form button[type=submit]").click();
    await interact(page);
    await page.waitForURL("**/knowledge.html");
    const saved = await page.evaluate(() => JSON.parse(sessionStorage.getItem("kbot.knowledge.session.v1")));
    assert.equal(saved.access_token, "mock-business-token");
    assert.equal(saved.domain_id, 7);
    assert.equal(saved.entry_ticket, undefined);
    return observed;
  } finally {
    await page.close();
  }
}

async function assertNoSession(page) {
  assert.equal(await page.evaluate(() => sessionStorage.getItem("kbot.knowledge.session.v1")), null);
}

try {
  assert.deepEqual(await exercise(null, {}, async () => {}), []);

  const selected = await exercise({
    action: "SELECT_DOMAIN", entry_ticket: "select-ticket",
    domains: [{ domain_id: 2, name: "第一个域" }, { domain_id: 7, name: "第二个域" }],
  }, {}, async (page) => {
    await page.locator("#select-domain-form").waitFor();
    await assertNoSession(page);
    await page.locator("#entry-domain").selectOption("7");
    await page.locator("#select-domain-form button").click();
  });
  assert.deepEqual(selected, [{ action: "SELECT_DOMAIN", payload: { domain_id: 7 }, bearer: "Bearer select-ticket" }]);

  const firstDomain = await exercise({
    action: "CHANGE_PASSWORD", entry_ticket: "password-ticket", domains: [],
  }, {
    CHANGE_PASSWORD: { action: "CREATE_FIRST_DOMAIN", entry_ticket: "create-ticket", domains: [] },
  }, async (page) => {
    await page.locator("#password-section").waitFor();
    await assertNoSession(page);
    await page.locator("#current-password").fill("MockPassword@Test26");
    await page.locator("#new-password").fill("NewMockPassword@Test27");
    await page.locator("#confirm-password").fill("NewMockPassword@Test27");
    await page.locator("#password-form button").click();
    await page.locator("#first-domain-form").waitFor();
    await assertNoSession(page);
    await page.locator("#first-domain-name").fill("研发知识");
    await page.locator("#first-domain-form button").click();
  });
  assert.deepEqual(firstDomain.map(({ action, bearer }) => [action, bearer]), [
    ["CHANGE_PASSWORD", "Bearer password-ticket"], ["CREATE_FIRST_DOMAIN", "Bearer create-ticket"],
  ]);
  assert.deepEqual(firstDomain[1].payload, { name: "研发知识", description: null });
  console.log("知识检索浏览器回归通过：单域、多域选择、先改密再创建首域");
} finally {
  await browser.close();
}
