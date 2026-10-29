import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const file = fileURLToPath(new URL("../../ui/aiops/js/aiops-pages.js", import.meta.url));
const source = readFileSync(file, "utf8");
const instrumented = source.replace(/\}\)\(\);\s*$/, "globalThis.lifecycleTest = { readonlyLifecycleHtml, scheduleRunRefresh, renderDetail };})();");
assert.notEqual(source, instrumented);
const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const timers = new Map(), calls = [];
let counter = 0, refresh;
const panel = { html: "", get innerHTML() { return this.html; }, set innerHTML(value) { this.html = value; refresh = {}; }, querySelector: () => refresh };
const run = { ops_run_id: "run1", status: "CANCELLED", workflow_kind: "CHAT_TURN", readonly_lifecycle: {
  items: [
    { lease_id: "<script>unsafe</script>", status: "TERMINATION_UNKNOWN" },
    { lease_id: "lease2", status: "CANCEL_REQUESTED", observation: { db_type: "ORACLE" } },
    { lease_id: "lease3", status: "TERMINATION_CONFIRMED", observation: { db_type: "MYSQL" } },
  ], truncated: true,
} };
const context = { KBotAIOpsShell: { escape, fmt: String, ready: new Promise(() => {}) },
  KBotAIOpsAuth: { async request(path) { calls.push(path); return run; } },
  location: { search: "?id=run1" }, URLSearchParams,
  document: { getElementById: () => panel, contains: () => true },
  setTimeout(fn) { const id = ++counter; timers.set(id, fn); return id; },
  clearTimeout(id) { timers.delete(id); },
};
context.window = context;
runInNewContext(instrumented, context, { filename: file });
await context.lifecycleTest.renderDetail("run-detail");
assert.ok(panel.html.includes("只读查询清理回执"));
assert.ok(panel.html.includes("终止状态未知，不能确认数据库已停止"));
assert.ok(panel.html.includes("已请求取消，尚未确认终止"));
assert.ok(panel.html.includes("自有查询清理已获协议确认"));
assert.ok(panel.html.includes("最近 50 条"));
assert.ok(panel.html.includes(escape("<script>unsafe</script>")) && !panel.html.includes("<script>unsafe</script>"));
assert.equal(timers.size, 1, "终态仍允许回执迟到后的有界刷新");
for (let index = 0; index < 3; index++) await context.lifecycleTest.renderDetail("run-detail");
assert.equal(timers.size, 0, "未确认不能无限后台轮询");
await refresh.onclick();
assert.equal(timers.size, 1, "手动刷新可重新等待迟到回执");
run.readonly_lifecycle.items = [{ lease_id: "lease3", status: "TERMINATION_CONFIRMED" }];
await context.lifecycleTest.renderDetail("run-detail");
assert.equal(timers.size, 0, "确认后停止终态轮询");
assert.ok(calls.every(path => path.includes("/runs/run1")));
console.log("PASS 只读清理回执、未知边界、转义、终态有界刷新及手动恢复");
