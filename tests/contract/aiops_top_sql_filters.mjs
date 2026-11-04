// 使用真实两端表单与过滤逻辑验证指标切换和服务端查询参数，不连接后端。
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../..", import.meta.url));
const portal = resolve(root, "../ammolite_cube/apps/portal-web");
const require = createRequire(resolve(portal, "package.json"));
const { chromium } = require("@playwright/test");
const ts = require("typescript");
const schema = [
  { name: "window", label: "时间范围", type: "select", required: true, default: "15m",
    options: [{ value: "15m", label: "最近 15 分钟" }, { value: "custom", label: "自定义时间" }] },
  { name: "begin_time", label: "开始时间", type: "datetime" },
  { name: "end_time", label: "结束时间", type: "datetime" },
  { name: "limit", label: "返回数量", type: "integer", required: true, default: 10, min: 1, max: 50 },
  { name: "rank_metric", label: "排序指标", type: "select", required: true, default: "TOTAL_DURATION", options: [
    { value: "TOTAL_DURATION", label: "总耗时" },
    { value: "CPU_TIME", label: "CPU 消耗（秒）", supported_db_types: ["ORACLE"] },
    { value: "SHARED_BLOCK_ACCESSES", label: "共享块访问", supported_db_types: ["POSTGRESQL"] },
    { value: "ROWS_EXAMINED", label: "扫描行数", supported_db_types: ["MYSQL"] },
  ] },
  { name: "sort_direction", label: "排序方向", type: "select", required: true, default: "DESC",
    options: [{ value: "DESC", label: "降序" }, { value: "ASC", label: "升序" }] },
  { name: "min_executions", label: "最低执行次数", type: "integer", min: 0 },
  { name: "min_metric_value", label: "所选指标最低值", type: "number", min: 0 },
  { name: "min_cpu_seconds", label: "最低 CPU 消耗（秒）", type: "number", min: 0, supported_db_types: ["ORACLE"] },
  { name: "min_logical_reads", label: "最低逻辑读", type: "integer", min: 0, supported_db_types: ["ORACLE"] },
];
const starter = { starter_id: "database.sql.top-current", title: "Top SQL", description: "资源排行",
  supported_db_types: ["ORACLE", "POSTGRESQL", "MYSQL"], execution_mode: "DIAGNOSTIC", category: "QUERY",
  input_schema: schema, status: "AVAILABLE" };
const systems = [{ business_system_id: "sys", display_name: "测试系统", environment: "TEST",
  targets: ["ORACLE", "POSTGRESQL", "MYSQL"].map(db_type => ({ db_type, target_id: db_type,
    resource_id: db_type, target_name: db_type, resource_name: db_type,
    status: "ENABLED", readonly_connection_enabled: true, connectivity_status: "CONNECTED" })) }];

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.setContent('<select id="agent-select"><option value="agent">Agent</option></select><dialog id="starter-dialog"><h2 id="starter-dialog-title"></h2><p id="starter-dialog-description"></p><form><div id="starter-dialog-content"></div></form></dialog>');
  await page.evaluate(() => {
    globalThis.requestRecords = [];
    globalThis.KBotAIOpsShell = { escape: value => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll('"', "&quot;"),
      ready: new Promise(() => {}), toast: () => {} };
    globalThis.KBotMarkdown = {};
    globalThis.KBotAIOpsAuth = { uuid: () => "test-only",
      request: async (path, options) => {
        requestRecords.push({ path, options });
        throw new Error("测试仅检查请求，不启动真实诊断");
      } };
  });
  const source = readFileSync(resolve(root, "ui/aiops/js/aiops-workspaces.js"), "utf8");
  const instrumented = source.replace(/\}\)\(\);\s*$/, "globalThis.topSqlFilterTest = { selectStarter, seed(starter, systems) { state.starters=[starter]; state.starterBusinessSystems=systems; state.starterCatalogVersion='2.2.0'; } };})();");
  assert.notEqual(source, instrumented);
  await page.addScriptTag({ content: instrumented });
  await page.evaluate(({ starter, systems }) => {
    topSqlFilterTest.seed(starter, systems);
    topSqlFilterTest.selectStarter(starter.starter_id);
  }, { starter, systems });
  await page.locator('[name="starter_business_system_id"]').selectOption("sys");
  const rank = page.locator('[name="rank_metric"]');
  await page.locator('[name="starter_resource_id"][value="ORACLE"]').check();
  await rank.selectOption("CPU_TIME");
  await page.locator('[name="min_cpu_seconds"]').fill("1.25");
  await page.locator('[name="min_logical_reads"]').fill("100");
  await page.locator('[name="starter_resource_id"][value="POSTGRESQL"]').check();
  assert.equal(await rank.inputValue(), "TOTAL_DURATION");
  assert.equal(await page.locator('[name="min_cpu_seconds"]').isVisible(), false);
  assert.equal(await page.locator('[name="min_cpu_seconds"]').isDisabled(), true);
  assert.equal(await page.locator('[name="min_cpu_seconds"]').inputValue(), "");
  assert.equal(await rank.locator('option[value="CPU_TIME"]').evaluate(option => option.disabled), true);
  await rank.selectOption("SHARED_BLOCK_ACCESSES");
  await page.locator('[name="starter_resource_id"][value="MYSQL"]').check();
  assert.equal(await rank.inputValue(), "TOTAL_DURATION");
  await rank.selectOption("ROWS_EXAMINED");
  await page.locator('[name="starter_resource_id"][value="ORACLE"]').check();
  await rank.selectOption("CPU_TIME");
  await page.locator('[name="sort_direction"]').selectOption("ASC");
  await page.locator('[name="min_executions"]').fill("3");
  await page.locator('[name="min_cpu_seconds"]').fill("1.25");
  await page.locator('[name="min_logical_reads"]').fill("100");
  await page.locator('[name="min_metric_value"]').fill("0.5");
  await page.locator('[data-starter-submit]').click();
  const request = await page.evaluate(() => requestRecords[0]);
  const body = JSON.parse(request.options.body);
  assert.deepEqual(body.starter.parameters, { window: "15m", limit: 10, rank_metric: "CPU_TIME",
    sort_direction: "ASC", min_executions: 3, min_metric_value: "0.5", min_cpu_seconds: "1.25", min_logical_reads: 100 });
  assert.deepEqual(body.context_hint.resource_ids, ["ORACLE"]);
  assert.equal(body.starter.catalog_version, "2.2.0");
  await page.waitForFunction(() => document.querySelector('#starter-dialog').dataset.busy !== 'true');
  await page.locator('[name="min_logical_reads"]').fill("9007199254740993");
  await page.locator('[data-starter-submit]').click();
  const preciseRequest = await page.evaluate(() => requestRecords[1]);
  assert.equal(JSON.parse(preciseRequest.options.body).starter.parameters.min_logical_reads, "9007199254740993");

  // Portal 使用真实模板以及 computed/watch 片段，不重写其能力过滤算法。
  const vue = readFileSync(resolve(portal, "src/features/aiops/business/AIOpsDiagnosisView.vue"), "utf8");
  const templateStart = vue.indexOf('<template v-for="field in selectedStarterFields"');
  const templateEnd = vue.indexOf("</template>", templateStart) + "</template>".length;
  const template = vue.slice(templateStart, templateEnd);
  const logic = ts.transpileModule(vue.slice(vue.indexOf("const selectedStarterFields = computed("),
    vue.indexOf("const starterCategoryLabels:")), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  await page.setContent('<main id="app"></main>');
  await page.addScriptTag({ path: resolve(portal, "node_modules/vue/dist/vue.global.js") });
  await page.evaluate(({ starter, template, logic }) => {
    const { computed, watch, ref, nextTick, createApp } = Vue;
    const selectedStarter = ref(starter), selectedStarterTarget = ref({ db_type: "ORACLE" });
    const starterParameters = ref({ window: "15m", rank_metric: "CPU_TIME", sort_direction: "DESC", min_cpu_seconds: "1.25" });
    const { selectedStarterFields, updateStarterParameter } = new Function("computed", "watch", "selectedStarter", "selectedStarterTarget", "starterParameters",
      logic + "; return { selectedStarterFields, updateStarterParameter };")(computed, watch, selectedStarter, selectedStarterTarget, starterParameters);
    globalThis.portalFilters = { selectedStarterTarget, starterParameters };
    createApp({ template, setup: () => ({ selectedStarter, selectedStarterFields, starterParameters, updateStarterParameter,
      submitting: false, diagnosticInProgress: false }) }).mount("#app");
  }, { starter, template, logic });
  assert.equal(await page.getByText("最低 CPU 消耗（秒）", { exact: true }).isVisible(), true);
  await page.locator('label').filter({ hasText: "最低逻辑读" }).locator('input').fill("9007199254740993");
  assert.equal(await page.evaluate(() => portalFilters.starterParameters.value.min_logical_reads), "9007199254740993");
  await page.locator('label').filter({ hasText: "最低 CPU 消耗（秒）" }).locator('input').fill("1.234567890123456789");
  assert.equal(await page.evaluate(() => portalFilters.starterParameters.value.min_cpu_seconds), "1.234567890123456789");
  await page.evaluate(() => { portalFilters.selectedStarterTarget.value = { db_type: "POSTGRESQL" }; });
  await page.waitForFunction(() => portalFilters.starterParameters.value.rank_metric === "TOTAL_DURATION");
  assert.equal(await page.getByText("最低 CPU 消耗（秒）", { exact: true }).count(), 0);
  assert.equal(await page.locator('option[value="CPU_TIME"]').count(), 0);
  assert.equal(await page.locator('option[value="SHARED_BLOCK_ACCESSES"]').count(), 1);
  assert.equal(await page.evaluate(() => "min_cpu_seconds" in portalFilters.starterParameters.value), false);
  await page.evaluate(() => { portalFilters.selectedStarterTarget.value = { db_type: "MYSQL" }; });
  await page.waitForFunction(() => document.querySelector('option[value="ROWS_EXAMINED"]'));
  assert.equal(await page.locator('option[value="SHARED_BLOCK_ACCESSES"]').count(), 0);
  assert.equal(await page.locator('input[type="number"][step="any"]').count(), 1);
  assert.deepEqual(errors, []);
  console.log("两端 Top SQL 资源指标切换、阈值清理与服务端参数回归通过");
} finally {
  await browser.close();
}
