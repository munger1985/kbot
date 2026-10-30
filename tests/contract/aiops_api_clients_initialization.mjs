// 旧 Target 管理拆除后，验证平台 API 客户端页仍只在 Shell 授权就绪后读取目录。
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const read = name => readFileSync(new URL(`../../ui/aiops/js/${name}`, import.meta.url), "utf8");
const shellSource = read("aiops-shell.js");
const source = read("aiops-api-clients.js");
const flush = () => new Promise(resolve => setImmediate(resolve));
function setup({ access = {}, page = "api-clients", panel = true, response = [] } = {}) {
  const requests = [], messages = [], button = { disabled: false }, output = { innerHTML: "正在加载…" };
  const context = {
    addEventListener() {},
    document: {
      body: { dataset: { page } },
      getElementById: id => id === "ops-simple" && panel ? output : null,
      querySelectorAll: selector => { assert.equal(selector, "header.ops-head button"); return [button]; },
    },
    KBotAIOpsAuth: { async request(path, options) {
      requests.push(path);
      assert.equal(path, "/api/v1/apps/aiops/api-clients");
      assert.equal(options, undefined, "客户端目录不得触发写操作");
      if (response instanceof Error) throw response;
      return response;
    } },
  };
  runInNewContext(shellSource, context);
  context.KBotAIOpsShell.ready = access instanceof Error ? Promise.reject(access) : Promise.resolve(access);
  context.KBotAIOpsShell.toast = message => messages.push(message);
  runInNewContext(source, context);
  return { requests, messages, button, output };
}

test("公共 Shell 未就绪前不加载平台客户端目录", async () => {
  let release;
  const app = setup({ access: new Promise(resolve => { release = resolve; }) });
  await flush();
  assert.deepEqual(app.requests, []);
  release({});
  await flush();
  assert.equal(app.requests.length, 1);
  assert.equal(app.button.disabled, true);
});

test("授权后读取目录，并转义返回内容且保持未接入的写操作禁用", async () => {
  const app = setup({ response: [{ display_name: '<img src="x" onerror="alert(1)">' }] });
  await flush();
  assert.equal(app.requests.length, 1);
  assert.match(app.output.innerHTML, /&lt;img/);
  assert.doesNotMatch(app.output.innerHTML, /<img/);
  assert.equal(app.button.disabled, true);
  assert.match(app.button.title, /对应配置表单接入后开放/);
  assert.deepEqual(app.messages, []);
});

test("目录失败显示安全错误，Shell 失败只显示认证错误且不请求目录", async () => {
  const failure = setup({ response: new Error("<失败>") });
  const authFailure = setup({ access: new Error("会话失效") });
  await flush();
  assert.match(failure.output.innerHTML, /ops-error.*&lt;失败&gt;/);
  assert.deepEqual(authFailure.requests, []);
  assert.deepEqual(authFailure.messages, ["会话失效"]);
});

for (const options of [{ access: null }, { page: "business-system-workspace" }, { panel: false }]) {
  test(`不满足页面就绪条件时不请求目录：${JSON.stringify(options)}`, async () => {
    const app = setup(options);
    await flush();
    assert.deepEqual(app.requests, []);
  });
}
