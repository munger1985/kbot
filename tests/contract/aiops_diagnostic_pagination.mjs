import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const file = fileURLToPath(new URL("../../ui/aiops/js/aiops-workspaces.js", import.meta.url));
const source = readFileSync(file, "utf8");
const instrumented = source.replace(/\}\)\(\);\s*$/, "globalThis.paginationTest = { investigationPlanHtml };})();");
assert.notEqual(source, instrumented);
const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const context = { KBotAIOpsShell: { escape, ready: new Promise(() => {}) }, addEventListener() {} };
context.window = context;
runInNewContext(instrumented, context, { filename: file });
const html = context.paginationTest.investigationPlanHtml({ actions: [{
  action_id: "a1", question: "核对目录", tool_id: "db.postgresql.readonly_query",
  pagination: { page_size: 200, max_pages: 4 },
}] });
assert.ok(html.includes("同次查询分批，每页最多 200 行、最多 4 页"));
assert.ok(!html.includes("按需采样"));
const sourceHtml = context.paginationTest.investigationPlanHtml({ actions: [{
  action_id: "a1", question: "核对历史", tool_id: "monitor.query_native_metric",
  pagination: { page_size: 100, max_pages: 3 },
}] });
assert.ok(sourceHtml.includes("来源原生续取（非一致快照），每页最多 100 行、最多 3 页"));
assert.ok(!sourceHtml.includes("同次查询分批"));
const unsafe = context.paginationTest.investigationPlanHtml({ actions: [{
  action_id: "a1", question: "<img src=x onerror=alert(1)>",
  pagination: { page_size: "<script>alert(1)</script>", max_pages: 4 },
}] });
assert.ok(!unsafe.includes("<script>") && !unsafe.includes("<img"));
assert.ok(unsafe.includes("&lt;script&gt;"));
console.log("PASS 有界游标分页计划展示与转义合同");
