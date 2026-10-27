import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const sourcePath = fileURLToPath(new URL("../../ui/aiops/js/aiops-workspaces.js", import.meta.url));
const source = readFileSync(sourcePath, "utf8");
const instrumented = source.replace(
  /\}\)\(\);\s*$/,
  "globalThis.testTable = { answerBlockHtml, topSqlColumnClass };})();",
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
runInNewContext(instrumented, context, { filename: sourcePath });

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
