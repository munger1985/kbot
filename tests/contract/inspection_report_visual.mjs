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
    for (const [name, presentation] of Object.entries(samples)) {
      for (const width of [1440, 390]) {
        const page = await browser.newPage({viewport: {width, height: 1000}});
        const errors = [];
        page.on("pageerror", error => errors.push(error.message));
        page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
        const monthly = name === "monthly", data = presentation[monthly ? "monthly_inspection" : "daily_inspection"];
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
        assert.equal(await page.locator(".daily-evidence").getAttribute("open"), null);
        assert.equal(await page.locator(".daily-report-table").count() > 0, true);
        if (product === "kbot4") {
          assert.equal(await page.locator("#formal-report-loading").isVisible(), false);
          assert.equal(await page.locator("#formal-report-error").isVisible(), false);
        }
        if (monthly) {
          assert.equal(await page.getByText("ERP 生产库", {exact: true}).count() > 0, true);
          assert.equal(await page.getByText("WMS 生产库", {exact: true}).count() > 0, true);
          assert.equal(await page.getByText("CRM 生产库", {exact: true}).count() > 0, true);
        }
        const overflowing = await page.evaluate(() => [...document.querySelectorAll(".daily-table-wrap,.aiops-formal-report")].map(el => ({cls:el.className,width:el.clientWidth,scroll:el.scrollWidth})).filter(item=>item.scroll > item.width+2));
        assert.deepEqual(overflowing, [], `${product}/${name}/${width} 横向溢出`);
        assert.deepEqual(errors, [], `${product}/${name}/${width} 浏览器错误`);
        await page.screenshot({path:path.join(resultDir,`${product}-${name}-${width}.png`),fullPage:true});
        await page.close();
      }
    }
  }
  console.log("PASS 两端各 5 类报告 × 2 个宽度：冻结结论、逐库展示、折叠附录、无表格横向溢出及浏览器异常");
} finally {
  await browser?.close();
  await server.close();
}
