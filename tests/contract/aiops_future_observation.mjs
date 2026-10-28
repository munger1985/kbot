import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const file = fileURLToPath(new URL("../../ui/aiops/js/aiops-pages.js", import.meta.url));
const source = readFileSync(file, "utf8");
const instrumented = source.replace(/\}\)\(\);\s*$/, "globalThis.futureTest = { renderFutureObservation };})();");
assert.notEqual(source, instrumented);
const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const calls = [], timers = new Map();
let timerId = 0;
let cancel, refresh;
const panel = {
  html: "",
  get innerHTML() { return this.html; },
  set innerHTML(value) { this.html = value; cancel = { disabled: false }; refresh = {}; },
  querySelector(selector) {
    return selector === "[data-refresh-future]" ? refresh
      : selector === "[data-cancel-future]" && this.html.includes("data-cancel-future") ? cancel : null;
  },
};
const run = { ops_run_id: "future-run", workflow_kind: "EVIDENCE_FOLLOWUP", status: "RUNNING", row_version: 7 };
const sql = 'SELECT "<unsafe>" FROM orders';
const result = { payload: {
  observation_series: { completed_count: 1, requested: { count: 3, interval_seconds: 30 } },
  tool_outcomes: [{ observation: { captured_at: "2026-10-28T06:00:00Z", row_count: 1, truncated: false,
    columns: [{ name: "sql_text" }], rows: [[sql]] } }],
} };
const context = {
  KBotAIOpsShell: { escape, fmt: (value) => value, ready: new Promise(() => {}),
    badge: (value) => `<span>${escape(value)}</span>`, toast() {} },
  KBotAIOpsAuth: {
    uuid: () => "fresh-command",
    async request(path, options = {}) {
      calls.push({ path, options });
      if (path.endsWith("/commands")) { run.status = "CANCELLED"; return { status: "CANCELLED" }; }
      return path.endsWith("/result") ? result : run;
    },
  },
  document: { getElementById: () => panel, contains: () => true },
  location: { search: "?id=future-run" }, URLSearchParams,
  confirm: () => true,
  setTimeout: (callback) => { const id = ++timerId; timers.set(id, callback); return id; },
  clearTimeout: (id) => timers.delete(id),
};
context.window = context;
runInNewContext(instrumented, context, { filename: file });
await context.futureTest.renderFutureObservation(run, panel);
assert.ok(panel.html.includes("后台未来观察"));
assert.ok(panel.html.includes("已采集 1/3 次"));
assert.ok(panel.html.includes("不覆盖过去请求窗口"));
assert.ok(panel.html.includes(escape(sql)) && !panel.html.includes(sql));
assert.ok(panel.html.includes('class="ops-future-cell"'));
assert.ok(cancel && refresh.onclick);
assert.equal(timers.size, 1);
await context.futureTest.renderFutureObservation(run, panel);
assert.equal(timers.size, 1, "手动刷新不能叠加轮询定时器");
await cancel.onclick();
const command = calls.find((item) => item.path.endsWith("/commands"));
assert.ok(command);
const body = JSON.parse(command.options.body);
assert.equal(body.ops_run_id, "future-run");
assert.equal(body.command.command_type, "CANCEL_RUN");
assert.equal(body.command.expected_row_version, 7);
assert.equal(timers.size, 0);
assert.ok(!panel.html.includes("data-cancel-future"));
assert.ok(panel.html.includes(escape(sql)), "取消不能抹去已采集的原始时点");
console.log("PASS 后台未来观察结果、转义、刷新及取消合同");
