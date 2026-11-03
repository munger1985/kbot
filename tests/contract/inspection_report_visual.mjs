// 隔离巡检报告视觉验收：真实两端渲染器 + 冻结业务样例，不连接线上 API。
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const ammolite = path.basename(root) === "ammolite_cube" ? root : path.resolve(root, "../ammolite_cube");
const kbot = path.basename(root) === "kbot4" ? root : path.resolve(root, "../kbot4");
const portal = path.join(ammolite, "apps/portal-web");
const require = createRequire(path.join(portal, "package.json"));
const { createServer } = await import(pathToFileURL(require.resolve("vite")));
const { default: vue } = await import(pathToFileURL(require.resolve("@vitejs/plugin-vue")));
const { chromium } = require("@playwright/test");
const sampleDir = process.argv[2] || "/tmp/inspection-report-ammolite-20261102";
const samples = JSON.parse(await fs.readFile(path.join(sampleDir, "samples.json"), "utf8"));
const resultDir = path.join(sampleDir, "browser");
await fs.mkdir(resultDir, { recursive: true });

const harness = {
  name: "inspection-report-harness",
  resolveId(id) {
    if (id === "/__inspection_harness__.js") return path.join(portal, "__inspection_harness__.js");
  },
  load(id) {
    if (id !== path.join(portal, "__inspection_harness__.js")) return;
    return 'import {createApp,h} from "vue"; import {createRouter,createWebHistory,RouterView} from "vue-router";'
      + 'import Report from "/src/features/aiops/business/AIOpsReportDetailView.vue"; import "/src/shared/ui/styles.css";'
      + 'const router=createRouter({history:createWebHistory(),routes:[{path:"/sample/:domainCode/:reportId",component:Report,meta:{title:"报告详情"}}]});'
      + 'createApp({render:()=>h(RouterView)}).use(router).mount("#app");';
  },
  configureServer(server) {
    server.middlewares.use(async (req, res, next) => {
      const url = new URL(req.url, "http://localhost");
      if (url.searchParams.has("html-proxy")) return next();
      if (url.pathname.startsWith("/kbot/")) {
        const file = path.resolve(kbot, "ui", url.pathname.slice(6));
        if (!file.startsWith(path.join(kbot, "ui") + path.sep)) { res.statusCode = 403; res.end(); return; }
        try {
          const body = await fs.readFile(file);
          res.setHeader("Content-Type", ({".html":"text/html",".css":"text/css",".js":"application/javascript"})[path.extname(file)] || "application/octet-stream");
          res.end(body);
        } catch { res.statusCode = 404; res.end(); }
        return;
      }
      if (!url.pathname.startsWith("/sample/")) return next();
      const html = '<!doctype html><html lang="zh-CN"><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><div id="app"></div><script type="module" src="/__inspection_harness__.js"></script></body></html>';
      res.setHeader("Content-Type", "text/html");
      res.end(await server.transformIndexHtml(req.url, html));
    });
  },
};
const server = await createServer({
  root: portal, configFile: false, plugins: [harness, vue()],
  resolve: {alias: {"@": path.join(portal, "src")}},
  server: {host: "127.0.0.1", port: 0, fs: {allow: [ammolite, kbot]}},
});
await server.listen();
const port = server.httpServer.address().port;
let browser;
try {
  browser = await chromium.launch({headless: true});
  for (const product of ["ammolite", "kbot4"]) {
    for (const [name, original] of Object.entries(samples)) {
      for (const width of [1440, 900, 390]) {
        const presentation = structuredClone(original);
        const page = await browser.newPage({viewport: {width, height: 1000}});
        const errors = [];
        page.on("pageerror", error => errors.push(error.message));
        page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
        const monthly = name === "monthly", data = presentation[monthly ? "monthly_inspection" : "daily_inspection"];
        if (name === "normal") data.checks.push({
          title: "实例累计统计", required: false, status_display: "仅观测",
          actual_display: ["sample_app", "sample_test", "sample_admin"].map(database =>
            `数据库：${database}，后端进程数：4，事务提交：1735503，事务回滚：78，数据块读取：116943，数据块命中：13658818976`
          ).join("；"),
          basis: "按冻结检查规则判断；无适用规则时仅展示观测。",
        });
        const report = {report_id: name, report_type: monthly ? "INSPECTION_MONTHLY" : "INSPECTION_DAILY", status: "READY",
          target_id: "wms", title: data.header.title, report_version: 1, template_ref: "system:inspection.daily",
          template_version: "1", summary: data.conclusion.text, period_start: "2026-11-01T00:00:00Z", period_end: "2026-11-02T00:00:00Z"};
        await page.route("https://**", route => route.abort());
        await page.route("**/api/**", route => {
          const url = new URL(route.request().url());
          if (!url.pathname.startsWith("/api/")) return route.continue();
          const result = url.pathname.endsWith("/presentation") ? presentation
            : url.pathname.includes("/targets/") ? {target_id:"wms",display_name:"WMS 生产库"} : report;
          return route.fulfill({json: result});
        });
        if (product === "kbot4") {
          await page.addInitScript(() => {
            const escape = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
            globalThis.KBotAIOpsShell = {ready:Promise.resolve(),escape,fmt:value=>value || "未记录",toast:()=>{}};
            globalThis.KBotAIOpsAuth = {request:async url => (await fetch(url)).json()};
          });
          await page.route(/\/(runtime-config|aiops-auth|aiops-shell)\.js(\?.*)?$/, route => route.fulfill({contentType:"application/javascript",body:""}));
        }
        const url = product === "ammolite" ? `/sample/test/${name}?domainId=test-domain` : `/kbot/aiops/report-detail.html?id=${name}`;
        await page.goto(`http://127.0.0.1:${port}${url}`);
        try { await page.locator(".daily-conclusion").waitFor({timeout:15000}); }
        catch (error) { console.error(product,name,width,errors,await page.locator("body").innerText()); throw error; }
        assert.equal(await page.locator(".daily-conclusion-copy").textContent(), data.conclusion.text);
        const appendix = page.locator(".inspection-appendix");
        assert.equal(await appendix.getAttribute("open"), null);
        assert.equal(await appendix.locator(".inspection-appendix-body").isVisible(), false);
        assert.equal(await page.locator(".daily-report-table").count() > 0, true);
        if (!monthly) {
          const statements = (data.sql_details || []).flatMap(group => group.statements);
          const cards = page.locator(".inspection-sql-card");
          assert.equal(await cards.count(), statements.length);
          for (let index = 0; index < statements.length; index++) {
            const statement = statements[index], card = cards.nth(index);
            assert.equal(await card.locator(":scope > pre > code").textContent(), statement.statement_preview);
            for (const value of [statement.execution_count_display, statement.total_latency_display,
              statement.average_latency_display, statement.maximum_latency_display])
              assert.ok((await card.innerText()).includes(value));
            assert.equal(await card.locator("script").count(), 0);
            const columns = await card.locator(".inspection-sql-stats").evaluate(element =>
              getComputedStyle(element).gridTemplateColumns.split(" ").length
            );
            assert.equal(columns, width <= 760 ? 2 : 4);
            if (statement.statement_truncated) {
              const full = card.locator(".inspection-sql-full-text");
              assert.equal(await full.getAttribute("open"), null);
              await full.locator("summary").click();
              assert.equal(await full.locator("code").textContent(), statement.statement_text);
              await full.locator("summary").click();
            }
          }
          assert.ok(!(await page.locator(".daily-checks-table").innerText()).includes("皮秒"));
          if (name === "attention" && statements.length)
            await cards.first().locator("..").screenshot({path:path.join(resultDir,`${product}-sql-${width}.png`)});
          if (name === "normal") {
            assert.equal(statements.length, 0);
            assert.ok((await page.locator(".inspection-sql-summary").innerText()).includes("当前没有问题 SQL"));
          }
          if (["attention", "abnormal"].includes(name)) {
            assert.equal(statements.length, 8);
            assert.ok((await cards.first().innerText()).includes("Agent 初步分析"));
            assert.ok((await cards.first().innerText()).includes("初步优化建议"));
          }
          const comparisons = data.metrics.filter(metric => metric.baseline_applicable);
          const table = page.locator(".daily-baseline-table");
          assert.equal(await table.count(), comparisons.length ? 1 : 0);
          const metricRows = table.locator("tbody tr");
          assert.equal(await metricRows.count(), comparisons.length);
          for (let index = 0; index < comparisons.length; index++) {
            const metric = comparisons[index], row = metricRows.nth(index);
            for (const value of [metric.comparison_display, metric.comparison_label, metric.baseline_display, metric.change_display])
              assert.ok((await row.innerText()).includes(value), `${product}/${name}/${width} 基线字段未保真：${value}`);
          }
          assert.ok((await page.locator(".aiops-formal-report").innerText()).includes("不直接决定健康结论"));
          if (name === "normal") assert.ok(await table.locator("svg").count() > 0);
          if (name === "incomplete") assert.equal(await table.locator("svg").count(), 0);
        }
        if (!monthly && width > 760) {
          const table = page.locator(".daily-checks-table");
          assert.equal(await table.count(), 1);
          const columns = await table.locator("thead th").evaluateAll(cells =>
            cells.map(cell => cell.getBoundingClientRect().width)
          );
          const total = columns.reduce((sum, value) => sum + value, 0);
          assert.ok(columns[1] < total * 0.15, `${product}/${width} 结果列过宽`);
          assert.ok(columns[2] > total * 0.40, `${product}/${width} 实测列过窄`);
          assert.ok(columns[2] >= columns[1] * 3, `${product}/${width} 列宽未按内容分配`);
          assert.equal(await table.locator("tbody td").evaluateAll(cells =>
            cells.every(cell => getComputedStyle(cell).verticalAlign === "top")
          ), true, `${product}/${name}/${width} 明细未顶端对齐`);
        }
        if (product === "kbot4") {
          assert.equal(await page.locator("#formal-report-loading").isVisible(), false);
          assert.equal(await page.locator("#formal-report-error").isVisible(), false);
        }
        if (monthly) {
          assert.equal(await page.getByText("ERP 生产库", {exact: true}).count() > 0, true);
          assert.equal(await page.getByText("WMS 生产库", {exact: true}).count() > 0, true);
          assert.equal(await page.getByText("CRM 生产库", {exact: true}).count() > 0, true);
        }
        await appendix.locator(":scope > summary").click();
        assert.equal(await appendix.locator(".inspection-appendix-body").isVisible(), true);
        const titleLayout = await appendix.locator(":scope > summary").evaluate(element => ({
          height: element.getBoundingClientRect().height,
          lineHeight: parseFloat(getComputedStyle(element).lineHeight),
        }));
        assert.ok(titleLayout.height <= titleLayout.lineHeight + 2, `${product}/${name}/${width} 附录标题意外换行`);
        const sources = data.evidence_boundary.sources || [];
        const trace = appendix.locator(".inspection-trace");
        assert.equal(await trace.count(), sources.length ? 1 : 0);
        if (sources.length) {
          assert.equal(await trace.getAttribute("open"), null);
          assert.equal(await trace.locator(".inspection-source-list").isVisible(), false);
          await page.screenshot({path:path.join(resultDir,`${product}-${name}-${width}-appendix.png`),fullPage:true});
          await trace.locator(":scope > summary").click();
          assert.ok((await trace.innerText()).includes("不属于巡检问题"));
          const items = trace.locator(".inspection-source-list > li");
          assert.equal(await items.count(), sources.length);
          for (let index = 0; index < sources.length; index++) {
            assert.equal(await items.nth(index).textContent(), sources[index]);
            assert.ok(sources[index].includes("证据编号："));
            assert.ok(sources[index].includes("内容指纹（Hash）："));
            assert.equal(await items.nth(index).evaluate(element => getComputedStyle(element, "::before").content), "none");
          }
        }
        const overflowing = await page.evaluate(() => [...document.querySelectorAll(".daily-table-wrap,.aiops-formal-report,.inspection-appendix,.inspection-appendix-body,.inspection-trace,.inspection-sql-card,.inspection-sql-card pre,.inspection-sql-stats")].map(el => ({cls:el.className,width:el.clientWidth,scroll:el.scrollWidth})).filter(item=>item.scroll > item.width+2));
        assert.deepEqual(overflowing, [], `${product}/${name}/${width} 横向溢出`);
        assert.ok(await page.locator(".aiops-formal-report").evaluate(element =>
          element.getBoundingClientRect().right <= innerWidth + 2
        ), `${product}/${name}/${width} 报告超出视口`);
        assert.deepEqual(errors, [], `${product}/${name}/${width} 浏览器错误`);
        await page.screenshot({path:path.join(resultDir,`${product}-${name}-${width}.png`),fullPage:true});
        await page.close();
      }
    }
  }
  console.log("PASS 两端各 5 类报告 × 3 个宽度：SQL 毫秒统计、独立排版、全文展开及安全转义；基线、冻结结论与附录，无横向溢出或浏览器异常");
} finally {
  await browser?.close();
  await server.close();
}
