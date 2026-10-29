import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const sourcePath = fileURLToPath(new URL("../../ui/aiops/js/aiops-workspaces.js", import.meta.url));
const source = readFileSync(sourcePath, "utf8");
const instrumented = source.replace(
  /\}\)\(\);\s*$/,
  "globalThis.testTable = { answerBlockHtml, topSqlColumnClass, sqlPlanColumnClass, normalizePrimaryTopSqlBlocks, displayText };})();",
);
assert.notEqual(instrumented, source);
const escape = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const context = {
  KBotAIOpsShell: { escape, fmt: (value) => value, ready: new Promise(() => {}) },
  KBotMarkdown: {},
  addEventListener: () => {},
};
runInNewContext(readFileSync(fileURLToPath(new URL("../../ui/aiops/js/aiops-sql-plan-viewer.js", import.meta.url)), "utf8"), context);
runInNewContext(instrumented, context, { filename: sourcePath });

assert.equal(context.testTable.displayText("FIXED TABLE FULL → SYS.TABLE；FIXED；TABLE；`TABLE`", {FIXED: "已修复", TABLE: "数据表格"}),
  "FIXED TABLE FULL → SYS.TABLE；已修复；数据表格；`TABLE`");
assert.equal(context.testTable.displayText("```sql\nSELECT FIXED FROM SYS.TABLE\n```", {FIXED: "已修复", TABLE: "数据表格"}),
  "```sql\nSELECT FIXED FROM SYS.TABLE\n```");

const digest = "a061f3c35618dfc240a0b107a2139a89b7290b5ef4958ef258789ee714926538";
const sql = 'SELECT "<unsafe>" FROM orders WHERE id = 1';
const columns = [
  { name: "session_id", label: "会话", logical_type: "INTEGER" },
  { name: "digest", label: "Digest", logical_type: "STRING" },
  { name: "sql_text", label: "SQL 文本", logical_type: "STRING" },
  { name: "elapsed_seconds", label: "总耗时（秒）", logical_type: "DECIMAL" },
];
const block = {
  block_type: "TABLE",
  payload: {
    presentation_role: "PRIMARY", result_kind: "TOP_SQL_LIST", title: "Top SQL",
    columns, rows: [[331794, digest, sql, "0.000242"]], row_count: 1,
  },
};
const html = context.testTable.answerBlockHtml(block, {});
assert.match(html, /ops-primary-result is-top-sql/);
assert.ok(html.indexOf('class="is-sql">SQL 文本') < html.indexOf('class="is-number">会话'));
assert.match(html, /<th class="is-number">会话<\/th>/);
assert.ok(!html.includes('>Digest</th>'));
assert.ok(!html.includes(`title="${digest}"`));
assert.ok(html.includes(`title="${escape(sql)}"`));
assert.ok(html.includes(escape(sql)));
assert.ok(!html.includes(sql));
assert.ok(html.includes(`继续深入分析 SQL ${digest}`));
assert.ok(html.includes('aria-label="查询结果表格，可横向滚动"'));

const windowTable = context.testTable.answerBlockHtml({
  block_type: "TABLE", payload: {
    presentation_role: "PRIMARY", result_kind: "TOP_SQL_LIST",
    measurement_note: "两次采样间已完成语句的增量",
    columns: [
      { name: "schema_name", label: "数据库" },
      { name: "digest", label: "Digest" },
      { name: "sql_text", label: "SQL 文本" },
      { name: "execution_count", label: "执行次数", logical_type: "INTEGER" },
      { name: "sample_window_seconds", label: "采样时长（秒）", logical_type: "DECIMAL" },
    ],
    rows: [["aiops_demo", digest, sql, 5, "5.1"]],
  },
}, {});
assert.ok(windowTable.includes("两次采样间已完成语句的增量"));
assert.ok(windowTable.includes('class="is-number">采样时长（秒）'));
assert.ok(!windowTable.includes('>Digest</th>'));
assert.ok(windowTable.includes(`继续深入分析 SQL ${digest}`));

const currentSql = context.testTable.answerBlockHtml({
  block_type: "TABLE", payload: {
    presentation_role: "PRIMARY", result_kind: "CURRENT_SQL_LIST", title: "当前活跃 SQL",
    columns: [
      { name: "query_id", label: "Query ID" },
      { name: "query_text", label: "SQL 文本" },
      { name: "session_count", label: "会话数", logical_type: "INTEGER" },
    ],
    rows: [["123456", sql, 2]], row_count: 1,
  },
}, {});
assert.match(currentSql, /ops-primary-result is-top-sql/);
assert.ok(currentSql.includes(`title="${escape(sql)}"`));
assert.ok(currentSql.includes("继续深入分析 SQL 123456"));
assert.ok(currentSql.includes('class="is-number">会话数'));

const comparisonTable = context.testTable.answerBlockHtml({
  block_type: "TABLE", payload: {
    presentation_role: "PRIMARY", result_kind: "SQL_SUBJECT_COMPARISON", title: "SQL 统计主体对比",
    follow_up_label: "仅诊断此主体", row_selections: ["opaque-original-row-ref"],
    columns: [{ name: "digest" }, { name: "schema_name", label: "数据库" }, { name: "sql_text", label: "SQL 文本" }],
    rows: [[digest, null, sql]], row_count: 1,
  },
}, {});
assert.match(comparisonTable, /ops-primary-result is-top-sql/);
assert.ok(comparisonTable.includes('data-sql-selection="opaque-original-row-ref"'));
assert.ok(comparisonTable.includes("仅诊断此主体"));
assert.ok(comparisonTable.includes(`title="${digest}"`));
assert.ok(comparisonTable.includes(`title="${escape(sql)}"`));
assert.ok(!comparisonTable.includes(sql));

const noIdentifier = context.testTable.answerBlockHtml({
  block_type: "TABLE", payload: {
    presentation_role: "PRIMARY", result_kind: "CURRENT_SQL_LIST",
    columns: [{ name: "query_text", label: "SQL 文本" }], rows: [[sql]],
  },
}, {});
assert.match(noIdentifier, /ops-primary-result is-top-sql/);
assert.ok(!noIdentifier.includes("data-primary-follow-up"));

const emptyTopSql = context.testTable.answerBlockHtml({
  block_type: "TABLE", payload: {
    presentation_role: "PRIMARY", result_kind: "TOP_SQL_LIST", columns, rows: [], row_count: 0,
  },
}, {});
assert.match(emptyTopSql, /ops-primary-result is-top-sql/);
assert.ok(emptyTopSql.includes("空结果不代表未执行取证"));
assert.ok(!emptyTopSql.includes("data-primary-follow-up"));

const nonPrimary = context.testTable.answerBlockHtml({
  block_type: "TABLE", payload: {
    result_kind: "TOP_SQL_LIST", columns, rows: [[331794, digest, sql, "0.000242"]],
  },
}, {});
assert.ok(!nonPrimary.includes("ops-primary-result"));
assert.ok(!nonPrimary.includes("data-primary-follow-up"));

for (const name of ["sql_id", "query_id"]) {
  const idTable = context.testTable.answerBlockHtml({
    block_type: "TABLE", payload: {
      presentation_role: "PRIMARY", result_kind: "TOP_SQL_LIST",
      columns: [{ name, label: name }, columns[2]], rows: [["abc123", "SELECT 1"]],
    },
  }, {});
  assert.ok(idTable.includes(`>${name}</th>`));
  assert.ok(idTable.includes('title="abc123"'));
}

const plain = context.testTable.answerBlockHtml({
  block_type: "TABLE", payload: { columns: [columns[0]], rows: [[null]] },
}, {});
assert.ok(plain.includes("<td>-</td>"));
assert.ok(!plain.includes("is-top-sql"));

const alert = context.testTable.answerBlockHtml({
  block_type: "TABLE", payload: {
    presentation_role: "PRIMARY", result_kind: "ALERT_LOG_LIST",
    columns: [{ name: "observed_at", label: "发生时间" }, { name: "message_text", label: "日志正文" }],
    rows: [["2026-10-27", "测试告警"]],
  },
}, {});
assert.ok(alert.indexOf("发生时间") < alert.indexOf("日志正文"));
assert.ok(!alert.includes("is-top-sql"));

const historicalCurrent = {
  block_type: "TABLE",
  payload: { presentation_role: "PRIMARY", result_kind: "TOP_SQL_LIST", title: "Top SQL",
    measurement_semantics: "CURRENT_ACTIVITY", columns, rows: [] },
};
const historicalCumulative = {
  block_type: "TABLE",
  payload: { ...block.payload, measurement_semantics: "CUMULATIVE_SINCE_LOAD" },
};
const normalized = context.testTable.normalizePrimaryTopSqlBlocks([historicalCurrent, historicalCumulative]);
assert.equal(normalized[0].payload.presentation_role, "SUPPORTING");
assert.equal(normalized[0].payload.title, "当前活跃 SQL 快照");
assert.equal(normalized[1].payload.presentation_role, "PRIMARY");
assert.equal(historicalCurrent.payload.presentation_role, "PRIMARY");

const executionPlan = context.testTable.answerBlockHtml({
  block_type: "TABLE", payload: {
    presentation_role: "PRIMARY", result_kind: "SQL_EXECUTION_PLAN", title: "执行计划 · 缓存计划",
    measurement_note: "优化器成本不是毫秒", has_row_actions: false,
    columns: [{ name: "node_id", label: "节点" }, { name: "operation", label: "执行操作" },
      { name: "object", label: "对象" }],
    rows: [[0, "SELECT STATEMENT", null], [1, "└─ TABLE ACCESS FULL", 'MES."<unsafe>"']], row_count: 2,
  },
}, {});
assert.ok(executionPlan.includes("执行计划 · 缓存计划"));
assert.ok(executionPlan.includes("└─ TABLE ACCESS FULL"));
assert.ok(executionPlan.includes("优化器成本不是毫秒"));
assert.ok(executionPlan.includes(escape('MES."<unsafe>"')));
assert.ok(!executionPlan.includes("data-primary-follow-up"));
assert.ok(!executionPlan.includes(">操作</th>"));
assert.ok(executionPlan.includes('class="ops-primary-result is-sql-plan"'));
assert.ok(executionPlan.includes('>节点</th>'));
assert.ok(executionPlan.includes('>执行操作</th>'));
assert.ok(executionPlan.includes('>对象</th>'));
assert.ok(executionPlan.includes('role="tab"'));
assert.ok(executionPlan.includes('原始计划'));
assert.ok(executionPlan.includes('aria-label="原始执行计划，可横向滚动"'));
for (const name of ["cost", "estimated_rows", "actual_rows", "loops", "actual_elapsed_time_us", "actual_total_time_per_loop_ms"]) {
  assert.equal(context.testTable.sqlPlanColumnClass({ name }), "is-number");
}
assert.equal(context.testTable.sqlPlanColumnClass({ name: "index" }), "is-plan-object");
assert.ok(!html.includes("is-sql-plan"));
assert.ok(!alert.includes("is-sql-plan"));
