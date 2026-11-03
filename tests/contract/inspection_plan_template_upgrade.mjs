// 巡检计划模板升级的真实两端页面回归；全部 API 使用隔离替身，不连接线上。
// node tests/contract/inspection_plan_template_upgrade.mjs
import assert from "node:assert/strict";
import { readFile, mkdtemp } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const ammolite = path.basename(root) === "ammolite_cube" ? root : path.resolve(root, "../ammolite_cube");
const kbot = path.basename(root) === "kbot4" ? root : path.resolve(root, "../kbot4");
const portal = path.join(ammolite, "apps/portal-web");
const require = createRequire(path.join(portal, "package.json"));
const { chromium } = require("@playwright/test");
const { createServer } = await import(pathToFileURL(require.resolve("vite")));
const { default: vue } = await import(pathToFileURL(require.resolve("@vitejs/plugin-vue")));
const artifacts = await mkdtemp(path.join(tmpdir(), "inspection-plan-upgrade-"));
const entry = `
import {createApp,h} from "vue";
import {createRouter,createWebHistory,RouterView} from "vue-router";
import Plans from "/src/features/aiops/management/AIOpsInspectionPlansView.vue";
import "/src/shared/ui/styles.css";
const router=createRouter({history:createWebHistory(),routes:[{path:"/plans/:domainCode",component:Plans}]});
createApp({render:()=>h(RouterView)}).use(router).mount("#app");
`;
const server = await createServer({
  root: portal, configFile: false, cacheDir: path.join(artifacts, "vite-cache"),
  resolve: {alias: {"@": path.join(portal, "src")}},
  plugins: [vue(), {
    name: "inspection-plan-upgrade-isolated",
    resolveId(id) { if (id === "/__plans_fixture__.js") return path.join(portal, "__plans_fixture__.js"); },
    load(id) { if (id === path.join(portal, "__plans_fixture__.js")) return entry; },
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url, "http://localhost");
        if (url.pathname.startsWith("/kbot/")) {
          const file = path.resolve(kbot, "ui", url.pathname.slice(6));
          if (!file.startsWith(path.join(kbot, "ui") + path.sep)) {res.statusCode = 403;res.end();return;}
          try {
            res.setHeader("Content-Type", ({".html":"text/html; charset=utf-8",".js":"application/javascript",".css":"text/css"})[path.extname(file)] || "application/octet-stream");
            res.end(await readFile(file));
          } catch { res.statusCode = 404;res.end(); }
          return;
        }
        if (!url.pathname.startsWith("/plans/")) return next();
        res.setHeader("Content-Type", "text/html; charset=utf-8");
        res.end(await server.transformIndexHtml(req.url, '<!doctype html><html lang="zh-CN"><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body class="business-app-aiops"><div id="app"></div><script type="module" src="/__plans_fixture__.js"></script></body></html>'));
      });
    },
  }],
  server: {host: "127.0.0.1", port: 0, fs: {allow: [ammolite, kbot]}},
  logLevel: "error",
});
let browser, cases = 0;
try {
  await server.listen();
  browser = await chromium.launch({headless: true});
  const origin = "http://127.0.0.1:" + server.httpServer.address().port;
  for (const product of ["ammolite", "kbot4"]) for (const width of [1440, 390]) {
    const page = await browser.newPage({viewport: {width, height: 1000}}), errors = [], writes = [];
    page.setDefaultTimeout(10000);
    page.on("pageerror", error => errors.push(error.message));
    const templates = [
      {inspection_template_id:"daily", display_name:"标准日报", version_no:2, status:"ACTIVE", selected_check_ids:["availability"], is_default:true, template_kind:"DAILY", row_version:1},
      {inspection_template_id:"custom", display_name:"业务日报", version_no:4, status:"ACTIVE", selected_check_ids:["availability"], row_version:1},
    ];
    const plan = {
      plan_id:"plan-1", display_name:"生产库巡检", business_system_id:"system-1", agent_id:"agent-1",
      schedule_type:"DAILY", cron_expression:"0 2 * * *", timezone:"Asia/Shanghai", status:"ACTIVE",
      inspection_template_id:"daily", inspection_template_name:"标准日报", inspection_template_version:1,
      inspection_template_version_id:"daily-v1", selected_check_ids:["availability"], timeout_seconds:1800,
      overlap_policy:"SKIP", misfire_policy:"LATEST_ONLY", system_target_count:1, row_version:1,
    };
    let failSave = false;
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url()), method = route.request().method();
      if (!url.pathname.startsWith("/api/")) return route.continue();
      let result;
      if (method === "PATCH" && url.pathname.endsWith("/inspection-plans/plan-1")) {
        const payload = route.request().postDataJSON();
        writes.push(payload);
        assert.equal(route.request().headers()["if-match"], '"rv-' + plan.row_version + '"');
        assert.equal(Object.hasOwn(payload, "upgrade_template_version"), false, "升级是命令，不是持久化自动跟随开关");
        if (failSave) {
          failSave = false;
          await route.fulfill({status:503, json:{error:{message:"隔离保存失败"}, message:"隔离保存失败"}});
          return;
        }
        if (payload.inspection_template_id) {
          const selected = templates.find(row => row.inspection_template_id === payload.inspection_template_id);
          plan.inspection_template_version = selected.version_no;
          plan.inspection_template_version_id = selected.inspection_template_id + "-v" + selected.version_no;
          plan.inspection_template_name = selected.display_name;
        }
        Object.assign(plan, payload);plan.row_version++;result = plan;
      } else if (url.pathname.endsWith("/inspection-plans/plan-1")) result = plan;
      else if (url.pathname.endsWith("/inspection-plans")) result = {items:[plan]};
      else if (url.pathname.endsWith("/inspection-templates")) result = templates;
      else if (url.pathname.endsWith("/agents")) result = [{agent_id:"agent-1", display_name:"巡检 Agent", status:"ACTIVE", capabilities:["ACTIVE_DIAGNOSIS"], allowed_business_system_ids:["system-1"]}];
      else if (url.pathname.endsWith("/business-systems/system-1")) result = {resources:[{status:"ACTIVE", resource:{resource_id:"resource-1", display_name:"ERP", resource_type_key:"DATABASE", lifecycle_status:"ACTIVE", database:{target_id:"target-1"}}}]};
      else if (url.pathname.endsWith("/business-systems")) result = {items:[{business_system_id:"system-1", display_name:"生产系统", status:"ENABLED"}]};
      else if (url.pathname.endsWith("/inspection-fires")) result = {items:[]};
      else throw new Error("未预期的 API：" + method + " " + url.pathname);
      await route.fulfill({json:result});
    });
    if (product === "kbot4") {
      await page.addInitScript(() => {
        const escape = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
        const ready = new Promise(resolve => document.addEventListener("DOMContentLoaded", () => {
          document.body.classList.add("aiops-static-page", "business-app-aiops");
          document.querySelectorAll("body > dialog").forEach(dialog => document.querySelector("main").append(dialog));
          resolve({permissions:["aiops:plan_manage"]});
        }, {once:true}));
        globalThis.KBotAIOpsShell = {ready, escape, fmt:value => value || "尚未执行", toast:() => {}};
        globalThis.KBotAIOpsAuth = {uuid:() => crypto.randomUUID(), request:async (url, init) => {
          const response = await fetch(url, init), body = await response.json();
          if (!response.ok) throw new Error(body.error?.message || body.message);
          return body;
        }};
      });
      await page.route(/\/(runtime-config|aiops-auth|aiops-shell)\.js(\?.*)?$/, route => route.fulfill({contentType:"application/javascript", body:""}));
    }
    const summary = page.locator(".inspection-template-summary");
    const upgrade = page.getByRole("checkbox", {name:/本次保存升级到/});
    const save = page.getByRole("button", {name:"保存修改", exact:true});
    const open = async () => {
      await page.getByRole("button", {name:"编辑", exact:true}).click();
      await summary.waitFor();
      await page.waitForFunction(() => !document.querySelector('button[type="submit"]').disabled);
    };
    const saveClose = async () => {await save.click();await summary.waitFor({state:"hidden"});};
    const reload = async () => {await page.reload();await page.getByRole("button", {name:"编辑", exact:true}).waitFor();};
    await page.goto(origin + (product === "ammolite" ? "/plans/test?domainId=test" : "/kbot/aiops/inspection-plans.html"));
    await open();
    assert.match(await summary.innerText(), /标准日报 · v1/);
    assert.match(await summary.innerText(), /最新发布版本：v2/);
    assert.equal(await upgrade.isChecked(), false);
    // 普通修改不改变绑定版本，重开仍显示真实的旧版本。
    await saveClose();assert.equal(Object.hasOwn(writes.at(-1), "inspection_template_id"), false);
    await open();assert.match(await summary.innerText(), /标准日报 · v1/);
    await upgrade.check();assert.match(await summary.innerText(), /保存后升级到 v2/);
    // 保存失败保留待升级选择和当前版本，不伪报升级完成。
    failSave = true;await save.click();
    await page.waitForFunction(() => !document.querySelector('button[type="submit"]').disabled);
    assert.equal(await upgrade.isChecked(), true);
    assert.match(await summary.innerText(), /标准日报 · v1/);
    await saveClose();assert.equal(writes.at(-1).inspection_template_id, "daily");
    await open();assert.match(await summary.innerText(), /标准日报 · v2/);
    assert.match(await summary.innerText(), /已使用当前版本/);
    assert.equal(await upgrade.isVisible(), false);
    await saveClose();assert.equal(Object.hasOwn(writes.at(-1), "inspection_template_id"), false);
    // 新版发布不会自动升级；切换模板不能残留之前的升级请求。
    templates[0].version_no = 3;await reload();await open();
    assert.match(await summary.innerText(), /标准日报 · v2/);
    await upgrade.check();
    const select = page.getByLabel(/巡检内容模板/);
    await select.selectOption("custom");assert.equal(await upgrade.isVisible(), false);
    assert.match(await summary.innerText(), /保存后改用 业务日报 · v4/);
    await select.selectOption("daily");assert.equal(await upgrade.isChecked(), false);
    await summary.scrollIntoViewIfNeeded();
    assert.equal(await summary.evaluate(node => node.scrollWidth > node.clientWidth + 1), false);
    await page.screenshot({path:path.join(artifacts, product + "-" + width + "-pending.png"), fullPage:true, animations:"disabled"});
    await select.selectOption("custom");await saveClose();
    assert.equal(writes.at(-1).inspection_template_id, "custom");
    await open();assert.match(await summary.innerText(), /业务日报 · v4/);
    assert.match(await summary.innerText(), /已使用当前版本/);
    await summary.scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(artifacts, product + "-" + width + "-saved.png"), fullPage:true, animations:"disabled"});
    await page.getByRole("button", {name:"取消", exact:true}).click();
    templates[1].version_no = 5;templates[1].status = "INACTIVE";await reload();await open();
    assert.equal(await upgrade.isVisible(), false);
    assert.match(await summary.innerText(), /模板已停用，保留计划当前版本/);
    await page.getByRole("button", {name:"取消", exact:true}).click();
    await page.getByRole("button", {name:/新建巡检计划/}).click();
    await summary.waitFor();assert.match(await summary.innerText(), /标准日报 · v3/);
    assert.equal(await upgrade.isVisible(), false);
    assert.deepEqual(errors, []);
    await page.close();cases++;console.log(product + " / " + width + "：保存、失败重试、重开回显、后续发布、切换与停用场景通过");
  }
  console.log(cases + " 组真实页面回归通过；截图：" + artifacts);
} finally {await browser?.close();await server.close();}
