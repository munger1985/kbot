// 工作台及系统管理浏览器回归，只使用隔离的 Mock API。
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(process.env.KBOT_BROWSER_REQUIRE_FROM
  || fileURLToPath(new URL("../../../ammolite_cube/apps/portal-web/package.json", import.meta.url)));
const { chromium, expect } = require("@playwright/test");
const origin = "http://kbot-workspace.test";
const root = new URL("../../", import.meta.url);
const modelPage = "/ui/system_management/models.html";
const browser = await chromium.launch({ headless: true });

const platform = {
  access_token: "mock-platform",
  entry_kind: "PLATFORM",
  user_id: "example",
  display_name: "测试用户",
  expires_at: "2099-01-01T00:00:00Z",
};
const domain = (domain_id, name) => ({ domain_id, name, status: "ACTIVE" });
const catalogModel = {
  model_id: "test-model", display_name: "测试模型",
  provider_model_name: "Vendor/Original", category: 1, provider: "api_deepseek",
  api_endpoint: "https://models.example.test/v1", status: "ACTIVE", row_version: 1,
  model_params: { temperature: 0.6, max_tokens: 2048 },
};
const providerOptions = [
  { category: 1, provider: "api_deepseek", required_fields: ["api_endpoint", "api_key"],
    secret_fields: ["api_key"], allowed_model_params: ["temperature", "max_tokens", "config_file", "oci_auth", "compartment_id"] },
  { category: 1, provider: "oci", required_fields: ["api_endpoint", "model_params.compartment_id"],
    secret_fields: ["api_key", "model_params.config_file", "model_params.oci_auth"],
    allowed_model_params: ["temperature", "compartment_id", "config_file", "oci_auth"] },
  { category: 2, provider: "local_bge", required_fields: ["model_params.model_path", "model_params.embedding_dimension"],
    secret_fields: ["api_key"], allowed_model_params: ["model_path", "embedding_dimension", "normalize"] },
  { category: 6, provider: "local_rapidocr", required_fields: ["model_params.model_path"], secret_fields: [],
    allowed_model_params: ["model_path", "languages", "rapidocr_params"] },
  { category: 8, provider: "api_oci_image", required_fields: ["api_endpoint", "model_params.project"],
    secret_fields: ["api_key", "model_params.config_file", "model_params.oci_auth"],
    allowed_model_params: ["project", "config_file", "oci_auth"] },
];

async function setup({ allowed, firstLogin = false, notification = null }) {
  const page = await browser.newPage();
  const rootSession = allowed ? platform : { ...platform, entry_kind: "WORKSPACE", access_token: "mock-workspace" };
  if (notification) await page.addInitScript((id) => {
    sessionStorage.setItem("kbot.aiops.notification_entry.v1", id);
  }, notification);
  const calls = [];
  let model = { ...catalogModel };
  const imageModel = { ...catalogModel, model_id: "image-model", display_name: "测试图像模型", category: 8 };
  const draftModel = { ...catalogModel, model_id: "draft-model", display_name: "测试草稿模型", status: "DRAFT" };
  await page.route(`${origin}/**`, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === "/ui/runtime-config.js") {
      const enabled = allowed ? ["knowledge_retrieval", "aiops", "media_studio"] : ["knowledge_retrieval", "aiops"];
      return route.fulfill({ contentType: "text/javascript", body:
        `window.KBOT_UI_CONFIG={mainApiBaseUrl:location.origin,enabledApps:${JSON.stringify(enabled)}};` });
    }
    if (["/ui/knowledge_retrieval/knowledge.html", "/ui/aiops/notification-open.html"].includes(path)) {
      return route.fulfill({ contentType: "text/html", body: "" });
    }
    if (path.startsWith("/ui/")) {
      return route.fulfill({ path: fileURLToPath(new URL(path.slice(1), root)) });
    }
    calls.push({ path, bearer: request.headers().authorization, body: request.postDataJSON() });
    if (path === "/api/v1/auth/login") {
      assert.deepEqual(Object.keys(request.postDataJSON()).sort(), ["password", "user_id"]);
      return route.fulfill({ json: { ...rootSession, must_change_password: firstLogin } });
    }
    if (path === "/api/v1/auth/password") {
      return route.fulfill({ json: { ...rootSession, access_token: "mock-updated", must_change_password: false } });
    }
    if (path === "/api/v1/auth/entries") {
      return route.fulfill({ json: { system_management: allowed, apps: [
        { app_id: "knowledge_retrieval", display_name: "知识检索", domains: [domain(3, "研发"), domain(7, "财务")] },
        { app_id: "media_studio", display_name: "多媒体创作", domains: [domain(3, "研发")] },
        ...(allowed ? [{ app_id: "aiops", display_name: "AIOps", domains: [domain(5, "运维")] }] : []),
      ] } });
    }
    if (path === "/api/v1/auth/exchange") {
      const { app_id, domain_id } = request.postDataJSON();
      return route.fulfill({ json: { entry_kind: "BUSINESS", access_token: "mock-business",
        user_id: platform.user_id, app_id, domain_id } });
    }
    if (path === "/api/v1/platform/models/provider-options") return route.fulfill({ json: providerOptions });
    if (path === "/api/v1/platform/models/1/test-model:test-connection") {
      return route.fulfill({ json: { latency_ms: 42 } });
    }
    if (path === "/api/v1/platform/models/8/image-model:test-connection") {
      return route.fulfill({ status: 502, json: { detail: { message: "模型未能完成测试推理" } } });
    }
    if (path === "/api/v1/platform/models/1/test-model" && request.method() === "PATCH") {
      model = { ...model, ...request.postDataJSON(), row_version: 2 };
      return route.fulfill({ json: model });
    }
    if (path === "/api/v1/platform/models" && request.method() === "POST") {
      return route.fulfill({ json: { ...request.postDataJSON(), row_version: 1 } });
    }
    if (path === "/api/v1/platform/models") return route.fulfill({ json: [model, imageModel, draftModel] });
    return route.fulfill({ status: 404 });
  });
  await page.goto(origin + (allowed ? "/ui/index.html" : "/ui/knowledge_retrieval/login.html"));
  await page.waitForURL("**/ui/index.html");
  await expect(page.locator("#portal-sign-in")).toBeVisible();
  await expect(page.locator("form:visible")).toHaveCount(1);
  await expect(page.getByText("使用 App 专属账号")).toHaveCount(0);
  await expect(page.locator(".app-card:visible")).toHaveCount(0);
  for (const [width, height] of [[1440, 900], [390, 844]]) {
    await page.setViewportSize({ width, height });
    const cover = await page.locator(".portal-cover").boundingBox();
    const pane = await page.locator(".portal-pane").boundingBox();
    const button = await page.locator("#portal-login button").boundingBox();
    assert.ok(cover && pane && button);
    if (width > 720) assert.ok(cover.x + cover.width <= pane.x + 1);
    else assert.ok(cover.y + cover.height <= pane.y + 1);
    assert.ok(button.y + button.height < height);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  }
  await page.locator("#portal-login [name=user_id]").fill("example");
  await page.locator("#portal-login [name=password]").fill("MockPassword@Test26");
  await page.locator("#portal-login button").click();
  return { page, calls };
}

async function assertCardLayout(page, width, columns) {
  await page.setViewportSize({ width, height: 900 });
  const cards = await page.locator(".app-card:visible").evaluateAll((nodes) => nodes.map((node) => {
    const box = node.getBoundingClientRect();
    return { x: node.offsetLeft, width: node.offsetWidth, height: node.offsetHeight,
      linkFits: node.querySelector(".app-link").getBoundingClientRect().bottom <= box.bottom };
  }));
  assert.equal(new Set(cards.map(({ x }) => x)).size, columns);
  for (const card of cards) {
    assert.deepEqual([card.width, card.height], [256, 208]);
    assert.equal(card.linkFits, true);
  }
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
}

async function assertModelEditor({ page, calls }) {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.locator('[data-model-action="edit"][data-model-id="test-model"]').click();
  const display = page.getByLabel("列表显示名称");
  await expect(page.getByLabel("KBot 调用名")).toHaveCount(0);
  const vendor = page.getByLabel("厂商模型名称 / ID");
  await expect(display).toBeEditable();
  await expect(vendor).toBeEditable();
  assert.ok(Math.abs((await display.boundingBox()).y - (await vendor.boundingBox()).y) < 1);
  await expect(page.locator("#model-oci-secret-field")).toBeHidden();
  await expect(page.locator("#model-oci-endpoint-help")).toBeHidden();
  await expect(page.locator("#model-api-key-field")).toBeVisible();
  await expect(page.locator("#model-param-compartment_id")).toHaveCount(0);
  await expect(page.locator("#model-param-temperature")).toHaveAttribute("type", "number");
  await expect(page.locator("#model-param-temperature")).toHaveValue("0.6");
  await page.setViewportSize({ width: 390, height: 844 });
  const [servedMobile, vendorMobile] = await Promise.all([display.boundingBox(), vendor.boundingBox()]);
  assert.equal(servedMobile.x, vendorMobile.x);
  assert.equal(servedMobile.width, vendorMobile.width);
  assert.ok(vendorMobile.y + vendorMobile.height <= servedMobile.y);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.locator("#model-param-temperature").fill("0");
  await vendor.fill("Vendor/New-Model");
  await page.getByRole("button", { name: "保存模型" }).click();
  await expect(page.locator("#model-rows")).toContainText("Vendor/New-Model");
  const patch = calls.find(({ path }) => path.endsWith("/1/test-model"));
  assert.deepEqual(patch.body, {
    expected_row_version: 1, provider_model_name: "Vendor/New-Model",
    model_params: { temperature: 0, max_tokens: 2048 },
  });

  await page.getByRole("button", { name: "新增模型" }).click();
  await page.locator("#model-provider").selectOption("oci");
  const configUpload = page.locator("#model-oci-config-upload");
  const keyUpload = page.locator("#model-oci-key-upload");
  const keyContent = page.locator("#model-oci-key-content");
  await expect(page.locator("#model-api-key-label")).toContainText("OCI");
  await expect(configUpload).toBeVisible();
  await expect(keyUpload).toBeVisible();
  await expect(page.locator("#model-oci-endpoint-help")).toBeVisible();
  await configUpload.setInputFiles({ name: "test.config", mimeType: "text/plain",
    buffer: Buffer.from("[DEFAULT]\nregion=us-ashburn-1\nkey_file=/fake/key.pem\n") });
  await keyUpload.setInputFiles({ name: "test.pem", mimeType: "text/plain",
    buffer: Buffer.from("-----BEGIN PRIVATE KEY-----\nSYNTHETIC\n-----END PRIVATE KEY-----") });
  await expect(page.locator("#model-oci-region")).toHaveValue("us-ashburn-1");
  await expect(keyContent).toHaveValue(/SYNTHETIC/);
  await page.locator("#model-provider").selectOption("api_deepseek");
  await expect(configUpload).toBeHidden();
  await expect(keyUpload).toBeHidden();
  await expect(configUpload).toHaveValue("");
  await expect(keyUpload).toHaveValue("");
  await expect(keyContent).toHaveValue("");
  await expect(page.locator("#model-oci-region")).toHaveValue("");
  await page.locator("#model-category").selectOption("8");
  await expect(page.locator("#model-oci-config-upload")).toBeVisible();

  await page.locator("#model-category").selectOption("2");
  await expect(page.locator("#model-oci-config-upload")).toBeHidden();
  await expect(page.locator("#model-api-key-field")).toBeHidden();
  await expect(page.locator("#model-endpoint-field")).toBeHidden();
  await vendor.fill("BAAI/bge-example");
  await page.getByLabel("列表显示名称").fill("示例向量模型");
  await page.locator("#model-param-model_path").fill("/models/example");
  await page.locator("#model-param-embedding_dimension").fill("1024");
  await page.locator("#model-param-normalize").selectOption("false");
  await page.getByRole("button", { name: "保存模型" }).click();
  await expect(page.locator("#model-dialog")).toBeHidden();
  const created = calls.find(({ path, body }) => path.endsWith("/platform/models") && body);
  assert.equal(created.body.provider_model_name, "BAAI/bge-example");
  assert.ok(!("served_model_name" in created.body));
  assert.deepEqual(created.body.model_params, { model_path: "/models/example", embedding_dimension: 1024, normalize: false });
  assert.equal(created.body.api_key, null);

  await page.getByRole("button", { name: "新增模型" }).click();
  await page.locator("#model-category").selectOption("6");
  await vendor.fill("RapidOCR");
  await page.getByLabel("列表显示名称").fill("示例 OCR");
  await page.locator("#model-param-model_path").fill("/models/ocr");
  await page.locator("#model-param-languages").fill("ch_sim， en");
  await page.getByRole("button", { name: "添加参数" }).click();
  await page.getByLabel("RapidOCR 参数名").fill("Global.text_score");
  await page.getByLabel("RapidOCR 参数类型").selectOption("number");
  await page.getByLabel("RapidOCR 参数值").fill("0.6");
  await page.getByRole("button", { name: "保存模型" }).click();
  await expect(page.locator("#model-dialog")).toBeHidden();
  const ocr = calls.find(({ path, body }) => path.endsWith("/platform/models") && body?.category === 6);
  assert.deepEqual(ocr.body.model_params, {
    model_path: "/models/ocr", languages: ["ch_sim", "en"], rapidocr_params: { "Global.text_score": 0.6 },
  });
}

async function assertModelConnectionTest({ page, calls }) {
  const button = (id) => page.locator(`[data-model-action="test"][data-model-id="${id}"]`);
  const findTest = (id) => calls.find(({ path }) => path.endsWith(`/${id}:test-connection`));
  await expect(button("draft-model")).toBeDisabled();
  await button("test-model").click();
  await expect(page.locator("#system-toast")).toContainText("测试通过，耗时 42 ms");
  assert.equal(findTest("test-model").body, null);
  assert.equal(findTest("test-model").bearer, "Bearer mock-updated");
  await button("image-model").click();
  const confirm = page.getByRole("alertdialog", { name: "测试图像生成模型？" });
  await expect(confirm).toContainText("产生费用");
  await confirm.getByRole("button", { name: "取消" }).click();
  assert.equal(findTest("image-model"), undefined);
  await button("image-model").click();
  await confirm.getByRole("button", { name: "开始测试" }).click();
  await expect(page.locator("#system-toast")).toContainText("模型未能完成测试推理");
  assert.equal(findTest("image-model").body, null);
}

try {
  const ordinary = await setup({ allowed: false });
  await expect(ordinary.page.locator(".app-card:visible")).toHaveCount(1);
  await expect(ordinary.page.locator('[data-app="knowledge_retrieval"]')).toBeVisible();
  await expect(ordinary.page.locator("#system-entry")).toBeHidden();
  await assertCardLayout(ordinary.page, 1440, 1);
  await ordinary.page.locator('[data-app="knowledge_retrieval"]').click();
  await ordinary.page.locator("#portal-domain-form select").selectOption("7");
  await ordinary.page.locator('#portal-domain-form [type="submit"]').click();
  await ordinary.page.waitForURL("**/knowledge.html");
  assert.deepEqual(ordinary.calls.find((call) => call.path.endsWith("/exchange")), {
    path: "/api/v1/auth/exchange", bearer: "Bearer mock-workspace",
    body: { app_id: "knowledge_retrieval", domain_id: 7 },
  });
  const stored = await ordinary.page.evaluate(() => JSON.stringify(sessionStorage));
  assert.match(stored, /mock-workspace/);
  assert.match(stored, /mock-business/);
  assert.doesNotMatch(stored, /MockPassword/);
  await ordinary.page.goto(origin + modelPage);
  await expect(ordinary.page.locator("#system-status")).toContainText("没有管理");
  assert.ok(!ordinary.calls.some((call) => call.path.startsWith("/api/v1/platform/models")));
  await ordinary.page.goto(origin + "/ui/index.html");
  await ordinary.page.locator("#portal-logout").click();
  await expect(ordinary.page.locator("#portal-sign-in")).toBeVisible();
  assert.equal(await ordinary.page.evaluate(() => sessionStorage.length), 0);
  assert.deepEqual(ordinary.calls.filter(({ body }) => body?.password).map(({ path }) => path), ["/api/v1/auth/login"]);

  const admin = await setup({ allowed: true, firstLogin: true });
  await expect(admin.page.locator("#portal-password")).toBeVisible();
  assert.ok(!admin.calls.some((call) => call.path.endsWith("/entries")));
  await admin.page.locator('[name="current_password"]').fill("MockPassword@Test26");
  await admin.page.locator('[name="new_password"]').fill("UpdatedPassword@Test27");
  await admin.page.locator("#portal-password button").click();
  await expect(admin.page.locator(".app-card:visible")).toHaveCount(4);
  for (const [width, columns] of [[1440, 4], [900, 3], [390, 1]]) {
    await assertCardLayout(admin.page, width, columns);
  }
  await admin.page.locator("#system-entry").click();
  await expect(admin.page.getByRole("heading", { name: "AI 模型管理" })).toBeVisible();
  await expect(admin.page.locator("#model-rows")).toContainText("测试模型");
  assert.ok(admin.calls.some((call) => call.path.endsWith("/platform/models") && call.bearer === "Bearer mock-updated"));
  assert.ok(!admin.calls.some((call) => call.path.endsWith("/exchange")));
  await assertModelConnectionTest(admin);
  await assertModelEditor(admin);
  const notification = "019fea8d-74a0-758a-8d1e-e768419ebae0";
  const notified = await setup({ allowed: true, notification });
  await expect(notified.page.locator("#portal-notification")).toBeVisible();
  await notified.page.locator('[data-app="aiops"]').click();
  await notified.page.waitForURL(`**/notification-open.html?entry=${notification}`);
  assert.deepEqual(notified.calls.find((call) => call.path.endsWith("/exchange")).body, { app_id: "aiops", domain_id: 5 });
  console.log("工作台浏览器回归通过：SSO、权限卡片、模型编辑和连通性测试。");
} finally {
  await browser.close();
}
