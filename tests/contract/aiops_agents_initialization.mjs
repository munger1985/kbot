import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const root = new URL("../../ui/aiops/", import.meta.url);
const html = readFileSync(new URL("agents.html", root), "utf8");
const shellSource = readFileSync(new URL("js/aiops-shell.js", root), "utf8");
const agentSource = readFileSync(new URL("js/aiops-agents.js", root), "utf8");
const publicPaths = new Set(Object.keys(JSON.parse(readFileSync(
  new URL("../../docs/openapi/main_api_public_v1.json", import.meta.url), "utf8",
)).paths));
const api = "/api/v1/apps/aiops";
const access = { permissions: ["aiops:agent_manage"], domain_id: "domain", user_id: "user" };
const paths = [
  `${api}/agents`, `${api}/business-systems?limit=200`, "/api/v1/model-catalog",
  `${api}/operations-knowledge/tags?limit=200`, `${api}/operations-knowledge/knowledge-bases`,
];
const flush = () => new Promise((resolve) => setImmediate(resolve));

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function setup({ session = true, accessResult = access, responses = {} } = {}) {
  const listeners = [];
  const errors = [];
  const requests = [];
  const redirects = [];
  const elements = new Map();
  function element() {
    return { innerHTML: "", textContent: "", value: "", disabled: false, dataset: {}, hidden: false,
      querySelectorAll: () => [], addEventListener() {}, remove() {} };
  }
  for (const match of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
    const node = element();
    node.disabled = /\sdisabled(?:\s|>)/.test(match[0]);
    elements.set(match[1], node);
  }
  elements.get("agent-rows").innerHTML = "正在加载 Agent…";
  const controls = new Map([
    ['[name="knowledge_enabled"]', element()],
    ['[name="auto_alert_enabled"]', element()],
  ]);
  const document = {
    body: {
      dataset: { page: "agents" }, classList: { contains: () => false, add() {} }, append() {},
      insertAdjacentHTML() {
        for (const id of ["ops-logout", "ops-domain", "ops-user", "ops-nav-trigger", "ops-nav-scrim", "business-app-navigation", "ops-account-toggle", "ops-account-menu"]) elements.set(id, element());
      },
    },
    getElementById(id) { return elements.get(id) || null; },
    querySelector(selector) { return controls.get(selector) || null; },
    querySelectorAll: () => [], addEventListener() {}, createElement: element,
  };
  const context = {
    document, location: { replace: (path) => redirects.push(path) }, setTimeout() {},
    addEventListener(name, callback) { if (name === "DOMContentLoaded") listeners.push(callback); },
    KBotAIOpsAuth: {
      load: () => session ? { access_token: "test-token" } : null,
      async request(path) {
        requests.push(path);
        assert.ok(publicPaths.has(path.split("?")[0]), `公开合同不存在此接口：${path}`);
        if (path === `${api}/access`) {
          if (accessResult instanceof Error) throw accessResult;
          return accessResult;
        }
        assert.ok(paths.includes(path), `不应请求未声明的接口：${path}`);
        const result = responses[path] ?? (path.includes("business-systems") ? { items: [] } : []);
        if (result instanceof Error) throw result;
        return result;
      },
    },
  };
  runInNewContext(shellSource, context, { filename: "aiops-shell.js" });
  runInNewContext(agentSource, context, { filename: "aiops-agents.js" });
  return {
    requests, redirects, errors, elements,
    async start() {
      for (const callback of listeners) Promise.resolve(callback()).catch((error) => errors.push(error));
      await flush();
    },
  };
}

test("公共 Shell 就绪前不加载目录、不开放新增", async () => {
  const pending = deferred();
  const app = setup({ accessResult: pending.promise });
  await app.start();
  assert.deepEqual(app.errors, []);
  assert.deepEqual(app.requests, [`${api}/access`]);
  assert.equal(app.elements.get("create-agent").disabled, true);
  assert.match(app.elements.get("agent-rows").innerHTML, /正在加载/);
  pending.resolve(access);
  await flush();
  assert.equal(app.requests.length, 6);
  assert.equal(app.elements.get("agent-rows").innerHTML, "", "空目录只保留页面中的新增卡片");
  assert.equal(app.elements.get("create-agent").disabled, false);
});

test("实际 Shell 与页面脚本完成列表渲染和事件绑定", async () => {
  const app = setup({ responses: {
    [paths[0]]: [{ agent_id: "agent", display_name: "诊断 <Agent>", status: "ACTIVE",
      allowed_business_system_ids: ["system"], capabilities: ["RESOURCE_CATALOG_QUERY"] }],
    [paths[1]]: { items: [{ business_system_id: "system", display_name: "生产系统", status: "ENABLED" }] },
    [paths[2]]: [
      { model_id: "model", display_name: "测试模型", category: 1, served_model_name: "test" },
      { model_id: "ocr", display_name: "测试 OCR", category: 6, served_model_name: "ocr" },
      { model_id: "vlm", display_name: "测试 VLM", category: 5, served_model_name: "vlm" },
    ],
  } });
  await app.start();
  assert.deepEqual(app.errors, []);
  assert.deepEqual(app.requests.slice(1), paths);
  assert.match(app.elements.get("agent-rows").innerHTML, /诊断 &lt;Agent&gt;/);
  assert.match(app.elements.get("agent-rows").innerHTML, /业务系统<\/dt><dd>1 个/);
  app.elements.get("agent-search").value = "生产系统";
  app.elements.get("agent-search").oninput();
  assert.match(app.elements.get("agent-rows").innerHTML, /诊断 &lt;Agent&gt;/);
  assert.match(app.elements.get("agent-planner-model").innerHTML, /测试模型/);
  assert.match(app.elements.get("agent-ocr-model").innerHTML, /测试 OCR/);
  assert.match(app.elements.get("agent-vlm-model").innerHTML, /测试 VLM/);
  assert.doesNotMatch(app.elements.get("agent-planner-model").innerHTML, /测试 OCR|测试 VLM/);
  assert.equal(app.elements.get("create-agent").disabled, false);
  assert.equal(typeof app.elements.get("create-agent").onclick, "function");
  assert.equal(typeof app.elements.get("agent-form").onsubmit, "function");
});

test("目录未全部返回前保留加载态并禁用新增", async () => {
  const pending = deferred();
  const app = setup({ responses: { [paths[2]]: pending.promise } });
  await app.start();
  assert.deepEqual(app.errors, []);
  assert.match(app.elements.get("agent-rows").innerHTML, /正在加载/);
  assert.equal(app.elements.get("create-agent").disabled, true);
  pending.resolve([]);
  await flush();
  assert.equal(app.elements.get("agent-rows").innerHTML, "", "空目录只保留页面中的新增卡片");
  assert.equal(app.elements.get("create-agent").disabled, false);
});

test("未登录时只跳转、不请求目录", async () => {
  const app = setup({ session: false });
  await app.start();
  assert.deepEqual(app.errors, []);
  assert.deepEqual(app.redirects, ["./login.html"]);
  assert.deepEqual(app.requests, []);
  assert.equal(app.elements.get("create-agent").disabled, true);
});

test("无管理权限时只跳转、不请求目录", async () => {
  const app = setup({ accessResult: { ...access, permissions: [] } });
  await app.start();
  assert.deepEqual(app.errors, []);
  assert.deepEqual(app.redirects, ["./dashboard.html"]);
  assert.deepEqual(app.requests, [`${api}/access`]);
  assert.equal(app.elements.get("create-agent").disabled, true);
});

test("公共 Shell 认证失败时结束加载并显示原因", async () => {
  const app = setup({ accessResult: new Error("会话已过期") });
  await app.start();
  assert.deepEqual(app.errors, []);
  assert.deepEqual(app.requests, [`${api}/access`]);
  assert.match(app.elements.get("agent-page-error").textContent, /Agent 加载失败.*会话已过期/);
  assert.doesNotMatch(app.elements.get("agent-rows").innerHTML, /正在加载/);
  assert.equal(app.elements.get("create-agent").disabled, true);
});

for (const path of paths) {
  test(`依赖接口失败时结束加载并转义错误：${path}`, async () => {
    const app = setup({ responses: { [path]: new Error('接口失败 <img src=x onerror="alert(1)">') } });
    await app.start();
    assert.deepEqual(app.errors, []);
    const content = app.elements.get("agent-rows").innerHTML;
    assert.match(app.elements.get("agent-page-error").textContent, /Agent 加载失败.*接口失败/);
    assert.equal(app.elements.get("agent-page-error").hidden, false);
    assert.doesNotMatch(content, /接口失败/);
    assert.doesNotMatch(content, /正在加载|<img/);
    assert.equal(app.elements.get("create-agent").disabled, true);
    assert.equal(app.elements.get("refresh-agents").disabled, false);
  });
}

test("目录加载失败后可原地刷新恢复", async () => {
  const responses = { [paths[2]]: new Error("目录暂不可用") };
  const app = setup({ responses });
  await app.start();
  assert.equal(app.elements.get("create-agent").disabled, true);
  responses[paths[2]] = [];
  await app.elements.get("refresh-agents").onclick();
  assert.equal(app.elements.get("agent-page-error").hidden, true);
  assert.equal(app.elements.get("agent-rows").innerHTML, "", "空目录只保留页面中的新增卡片");
  assert.equal(app.elements.get("create-agent").disabled, false);
});
