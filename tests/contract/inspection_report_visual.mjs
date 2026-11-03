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
const catalog = JSON.parse(await fs.readFile(path.join(sampleDir, "catalog.json"), "utf8"));
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
      + 'import Report from "/src/features/aiops/business/AIOpsReportDetailView.vue"; import Templates from "/src/features/aiops/management/AIOpsReportTemplatesView.vue"; import "/src/shared/ui/styles.css";'
      + 'const router=createRouter({history:createWebHistory(),routes:[{path:"/sample/:domainCode/:reportId",component:Report,meta:{title:"报告详情"}},{path:"/template/:domainCode",component:Templates,meta:{title:"模板中心"}}]});'
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
      if (!url.pathname.startsWith("/sample/") && !url.pathname.startsWith("/template/")) return next();
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
async function verifyTemplateDetail(page, product, width, entry) {
  const detail = page.locator(".inspection-template-detail:visible");
  await detail.locator(".template-content-head h2").waitFor();
  assert.equal(await detail.evaluate(node => node.scrollWidth > node.clientWidth + 2), false);
  const picker = detail.locator(".template-version-control");
  const pickerLimit = width <= 480 ? (await detail.locator(".template-content-head").boundingBox()).width : 260;
  assert.ok((await picker.boundingBox()).width <= pickerLimit + 1, "桌面版本选择限宽，窄屏不能超出内容区");
  assert.ok(await detail.locator(".template-directory li").count());
  assert.ok(await detail.locator(".template-check-topic").count());
  assert.equal(await detail.locator(".template-check-topics").evaluate(node => getComputedStyle(node).gridTemplateColumns.split(" ").length), 1, "主题必须单列自然高度，不能使用等高大卡片");
  for (const topic of await detail.locator(".template-check-topic").all()) {
    assert.equal(await topic.locator("ul").evaluate(node => getComputedStyle(node).display), "flex");
    assert.ok(await topic.locator("li").count());
    if (width >= 1024) assert.ok((await topic.boundingBox()).height < 240, "主题行不能因卡片等高而产生大片空白");
  }
  for (const theme of ["light", "dark"]) {
    await page.evaluate(value => document.documentElement.dataset.theme = value, theme);
    await page.screenshot({path: path.join(resultDir, `${product}-template-detail-${entry}-${theme}-${width}.png`), fullPage: true, animations: "disabled"});
    assert.equal(await detail.evaluate(node => node.scrollWidth > node.clientWidth + 2), false);
  }
  await page.evaluate(() => document.documentElement.dataset.theme = "light");
}
let browser;
try {
  browser = await chromium.launch({headless: true});
  // 编辑器通过真实页面操作验收，全部 API 被隔离替身拦截。
  for (const product of ["ammolite", "kbot4"]) {
    for (const width of [1440,1024,390]) {
      const page=await browser.newPage({viewport:{width,height:1000}}), errors=[], writes=[];
      console.log('日报分步编辑器验收：'+product+' / '+width);
      page.setDefaultTimeout(10000);
      page.on("pageerror",error=>errors.push(error.message));
      page.on("console",message=>{if(message.type()==="error")errors.push(message.text());});
      const ready=catalog.groups.flatMap(group=>group.checks).filter(check=>check.availability==="READY").map(check=>check.check_id);
      const template={inspection_template_id:"sample-template",display_name:"共享日报模板",status:"ACTIVE",source_kind:"CUSTOM",template_kind:"DAILY",is_default:false,has_draft:false,version_no:1,row_version:1,selected_check_ids:ready,daily_report_layout:catalog.daily_report_layout,usage_count:3};
      template.definition={display_name:template.display_name,selected_check_ids:ready,daily_report_layout:structuredClone(catalog.daily_report_layout)};
      const drafts=new Map(), draftSaves=[];
      const builtin={...template,inspection_template_id:"builtin-template",display_name:"系统标准日报",source_kind:"BUILTIN",is_default:true};
      let newDraft=null;
      let failPreview=false, failPublish=false, publishRequests=0, previewGate=null;
      const previewInputs=[];
      await page.route("**/api/**",async route=>{
        const url=new URL(route.request().url()), method=route.request().method();
        if (!url.pathname.startsWith("/api/")) return route.continue();
        let result;
        if(url.pathname.endsWith("/inspection-check-catalog")) result=catalog;
        else if(url.pathname.endsWith("/inspection-templates:preview")) {
          const input=route.request().postDataJSON(), preview=structuredClone(samples[{NORMAL:"normal",ATTENTION:"attention",INCOMPLETE:"incomplete"}[input.scenario]]);
          previewInputs.push(input);
          if(previewGate) { const gate=previewGate;previewGate=null;gate.started();await gate.promise; }
          if(failPreview) {failPreview=false;await route.fulfill({json:{error:{message:"模拟预览失败"}}});return;}
          preview.daily_inspection.preview=true;
          preview.daily_inspection.header.title=input.daily_report_layout.title;
          preview.daily_inspection.header.database_type=input.database_type;
          const byId=Object.fromEntries(preview.daily_inspection.sections.map(region=>[region.section_id,region]));
          preview.daily_inspection.sections=input.daily_report_layout.sections.map((settings,index)=>({...byId[settings.section_id],...settings,ordinal:index+1}));
          result=preview;
        } else if(method==="PUT" && url.pathname.endsWith("/draft")) {
          const input=route.request().postDataJSON(),id=url.pathname.split("/").at(-2);
          const row=id==="new-template" ? newDraft:template;
          assert.equal(input.expected_row_version,row.row_version);
          row.row_version++;row.draft={display_name:input.display_name,selected_check_ids:input.selected_check_ids,daily_report_layout:input.daily_report_layout};row.has_draft=true;
          draftSaves.push(input);drafts.set(id,structuredClone(row.draft));result=structuredClone(row);
        } else if(method==="POST" && url.pathname.endsWith("/inspection-templates")) {
          const input=route.request().postDataJSON();
          newDraft={...template,...input,inspection_template_id:"new-template",row_version:1,version_no:null,definition:null,draft:input,has_draft:true};
          drafts.set("new-template",input);result=newDraft;
        } else if(method==="POST" && url.pathname.endsWith(":publish")) {
          publishRequests++;
          const id=url.pathname.split("/").at(-1).split(":")[0],row=id==="new-template"?newDraft:template;
          assert.equal(route.request().postDataJSON().expected_row_version,row.row_version);
          if(failPublish) {failPublish=false;await route.fulfill({status:409,json:{message:"版本已变化，请刷新后重试。",error:{code:"ROW_VERSION_CONFLICT",message:"版本已变化，请刷新后重试。"}}});return;}
          writes.push(structuredClone(drafts.get(id)));
          row.definition=structuredClone(row.draft);row.display_name=row.definition.display_name;row.selected_check_ids=row.definition.selected_check_ids;row.daily_report_layout=row.definition.daily_report_layout;row.draft=null;row.has_draft=false;row.version_no=(row.version_no||0)+1;row.row_version++;
          result=row;
        } else if(url.pathname.endsWith("/inspection-templates")) result=[template,builtin];
        else if(url.pathname.endsWith("/sample-template")) result=template;
        else if(url.pathname.endsWith("/builtin-template")) result=builtin;
        else if(url.pathname.endsWith("/versions")) result=[{template_version_id:"version-1",version_no:template.version_no,created_by:"template-admin",created_at:"2026-11-03T00:00:00Z",current:true,definition:template.definition}];
        else if(url.pathname.endsWith("/plans")) result=[{inspection_plan_id:"plan-1",display_name:"生产库巡检",template_version_no:1,is_latest:false}];
        else result=[];
        await route.fulfill({json:result});
      });
      if(product==="kbot4") {
        await page.addInitScript(()=>{
          const escape=value=>String(value??"").replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
          globalThis.KBotAIOpsShell={ready:Promise.resolve(),escape,fmt:value=>value||"未记录",toast:()=>{}};
          globalThis.KBotAIOpsAuth={request:async(url,init)=>{const response=await fetch(url,init),body=await response.json();if(!response.ok)throw new Error(body.error?.message || body.message);return body;}};
        });
        await page.route(/\/(runtime-config|aiops-auth|aiops-shell)\.js(\?.*)?$/,route=>route.fulfill({contentType:"application/javascript",body:""}));
      }
      await page.goto("http://127.0.0.1:"+port+(product==="ammolite" ? "/template/test?domainId=test" : "/kbot/aiops/report-templates.html"));
      await page.locator(".management-catalog-card").first().waitFor();
      const cards=page.locator(".management-catalog-card");
      assert.equal(await cards.count(),2);
      for(const card of await cards.all())assert.equal(await card.locator(".management-catalog-primary-actions > button").count(),1);
      // 首次进入详情也必须完整排版，不能依赖先打开编辑器带入样式。
      await page.getByRole("button",{name:"查看模板",exact:true}).click();
      await verifyTemplateDetail(page, product, width, "direct");
      await page.getByRole("button",{name:"返回模板中心",exact:true}).click();
      await page.screenshot({path:path.join(resultDir,product+"-template-cards-"+width+".png"),fullPage:false});
      const builtinCard=cards.filter({hasText:"系统标准日报"});
      await builtinCard.getByRole("button",{name:/更多/}).click();
      assert.ok((await builtinCard.getByRole("menu").innerText()).includes("基于此创建"));
      assert.equal((await builtinCard.getByRole("menu").innerText()).includes("停用"),false);
      await page.keyboard.press("Escape");
      try { await page.getByRole("button",{name:"编辑模板",exact:true}).click({timeout:10000}); }
      catch(error) { console.error(product,width,errors,await page.locator("body").innerText()); throw error; }
      const editor=page.locator(product==="kbot4" ? "#daily-template-editor" : "main.daily-template-editor");
      await editor.waitFor();
      assert.equal(await editor.evaluate(node=>node.tagName==="DIALOG" || Boolean(node.closest("dialog"))),false);
      assert.equal(await editor.locator("[data-topic]").count(),9);
      assert.equal(await editor.locator(".daily-adaptation").getAttribute("open"),null);
      assert.equal(await editor.getByRole("button",{name:"发布新版本",exact:true}).isVisible(),false);
      await page.screenshot({path:path.join(resultDir,product+"-template-checks-"+width+".png"),fullPage:false});
      await editor.locator('[data-topic="PERFORMANCE"]').click();
      await editor.locator('[data-check-group="实例吞吐与运行线程"]').uncheck();
      const expected=ready.filter(id=>id!=="mysql.instance.throughput");
      await editor.locator(".daily-adaptation > summary").click();
      await editor.getByLabel("查看适配类型").selectOption("MYSQL");
      await editor.getByLabel("查看适配类型").selectOption("POSTGRESQL");
      await editor.locator('[data-editor-tab="layout"]').click();
      await editor.getByRole("button",{name:/03 · 系统资源使用/}).click();
      const title=editor.getByLabel("日报标题");
      await title.fill("共享日报 · 版式验收");
      assert.equal(await editor.getByLabel("日报标题").count(),1);
      await editor.getByLabel("统一正常明细偏好").selectOption("SUMMARY");
      await editor.getByLabel("本区域正常明细").selectOption("FULL");
      assert.equal(await editor.getByLabel("统一正常明细偏好").inputValue(),"CUSTOM");
      await editor.getByLabel("本区域正常明细").selectOption("SUMMARY");
      await editor.locator(".daily-layout-outline > div").nth(3).getByRole("button",{name:"上移区域"}).click();
      await page.evaluate(()=>window.scrollTo(0,0));
      await page.screenshot({path:path.join(resultDir,product+"-template-layout-"+width+".png"),fullPage:false});
      await editor.locator('[data-editor-tab="checks"]').click();
      assert.equal(await editor.locator('[data-check-group="实例吞吐与运行线程"]').isChecked(),false);
      await editor.locator('[data-editor-tab="layout"]').click();
      assert.equal(await title.inputValue(),"共享日报 · 版式验收");
      const savedDraft=page.waitForResponse(response=>response.url().endsWith("/draft") && response.request().method()==="PUT");
      await editor.getByRole("button",{name:"保存草稿",exact:true}).click();await savedDraft;
      assert.equal(publishRequests,0);
      await editor.getByRole("button",{name:"返回模板中心",exact:true}).click();
      await page.getByRole("button",{name:"编辑模板",exact:true}).click();
      await editor.locator('[data-editor-tab="layout"]').click();
      assert.equal(await title.inputValue(),"共享日报 · 版式验收");
      await editor.locator('[data-editor-tab="preview"]').click();
      await editor.getByRole("heading",{name:"共享日报 · 版式验收",exact:true}).waitFor();
      await editor.getByText("模拟预览已更新，可核对后发布。",{exact:true}).waitFor();
      assert.equal(await editor.locator("[data-report-section]").count(),12);
      assert.ok((await editor.innerText()).includes("模拟版式预览"));
      const publish=editor.getByRole("button",{name:"发布新版本",exact:true});
      assert.equal(await publish.isEnabled(),false);
      for(const [type,scene] of [["MYSQL","NORMAL"],["POSTGRESQL","INCOMPLETE"],["ORACLE","ATTENTION"]]) {
        await editor.getByLabel("模拟数据库类型").selectOption(type);
        await editor.getByLabel("模拟场景").selectOption(scene);
        assert.equal(await publish.isEnabled(),false);
        await editor.getByRole("button",{name:"刷新模拟预览",exact:true}).click();
        await editor.getByText("模拟预览已更新，可核对后发布。",{exact:true}).waitFor();
        assert.deepEqual(previewInputs.at(-1).selected_check_ids,expected);
        assert.equal(previewInputs.at(-1).database_type,type);
        assert.equal(previewInputs.at(-1).scenario,scene);
      }
      // 失败和运行中修改都不能把旧样例标成最新，也不能据此发布。
      failPreview=true;
      await editor.getByRole("button",{name:"刷新模拟预览",exact:true}).click();
      await editor.getByRole("alert").filter({hasText:"旧预览不代表当前配置"}).waitFor();
      assert.equal(await publish.isEnabled(),false);
      let release,started;
      const began=new Promise(resolve=>{started=resolve;});
      previewGate={promise:new Promise(resolve=>{release=resolve;}),started};
      await editor.getByRole("button",{name:"刷新模拟预览",exact:true}).click();
      await began;
      await editor.getByLabel("模拟数据库类型").selectOption("MYSQL");
      release();
      await editor.getByRole("button",{name:"刷新模拟预览",exact:true}).waitFor();
      assert.equal(await publish.isEnabled(),false);
      await editor.getByRole("button",{name:"刷新模拟预览",exact:true}).click();
      await editor.getByText("模拟预览已更新，可核对后发布。",{exact:true}).waitFor();
      await page.evaluate(()=>window.scrollTo(0,0));
      await page.screenshot({path:path.join(resultDir,product+"-template-preview-"+width+".png"),fullPage:false});
      const layoutIssues=await editor.evaluate(node=>({
        overflow:node.scrollWidth>node.clientWidth+2,
        scrolls:[...node.querySelectorAll("*")].filter(child=>child.clientHeight>0 && child.scrollHeight>child.clientHeight+2 && ["auto","scroll"].includes(getComputedStyle(child).overflowY)).map(child=>child.className),
      }));
      assert.deepEqual(layoutIssues,{overflow:false,scrolls:[]});
      await editor.getByLabel("我已核对预览与变更，确认发布新版本；已有计划不自动升级。").check();
      failPublish=true;
      await publish.click();
      await editor.getByRole("alert").filter({hasText:"版本已变化"}).waitFor();
      assert.equal(await publish.isEnabled(),false);
      await editor.getByLabel("我已核对预览与变更，确认发布新版本；已有计划不自动升级。").check();
      await publish.evaluate(button=>{button.click();button.click();});
      await editor.waitFor({state:"hidden"});
      assert.equal(writes.length,1);
      assert.equal(publishRequests,2);
      assert.equal(writes[0].daily_report_layout.title,"共享日报 · 版式验收");
      assert.equal(writes[0].daily_report_layout.sections.find(item=>item.section_id==="RESOURCES").normal_detail,"SUMMARY");
      assert.equal(writes[0].daily_report_layout.sections[2].section_id,"CAPACITY");
      assert.deepEqual(writes[0].selected_check_ids,expected);
      // 关闭并重开后，前一会话迟到的预览不能污染新草稿。
      await page.getByRole("button",{name:"编辑模板",exact:true}).click();
      const delayedStarted=new Promise(resolve=>{started=resolve;});
      previewGate={promise:new Promise(resolve=>{release=resolve;}),started};
      await editor.locator('[data-editor-tab="preview"]').click();
      await delayedStarted;
      await editor.getByRole("button",{name:"返回模板中心",exact:true}).click();
      await page.getByRole("button",{name:"编辑模板",exact:true}).click();
      release();
      await editor.locator('[data-editor-tab="layout"]').click();
      assert.equal(await editor.getByLabel("日报标题").inputValue(),"共享日报 · 版式验收");
      assert.equal(await editor.locator(".daily-report-head").count(),0);
      await editor.getByRole("button",{name:"返回模板中心",exact:true}).click();
      await page.locator(".management-catalog-add-card").filter({hasText:"新建巡检模板"}).click();
      await editor.locator('[data-editor-tab="layout"]').click();
      assert.equal(await editor.getByLabel("模板名称").isEnabled(),true);
      await editor.getByLabel("模板名称").fill("通用日报新模板");
      await editor.locator('[data-editor-tab="preview"]').click();
      await editor.getByText("模拟预览已更新，可核对后发布。",{exact:true}).waitFor();
      await editor.getByLabel("我已核对预览与变更，确认发布新版本；已有计划不自动升级。").check();
      await publish.click();
      await editor.waitFor({state:"hidden"});
      assert.equal(writes.length,2);
      assert.equal(writes[1].display_name,"通用日报新模板");
      assert.equal(draftSaves.length >= 3,true);
      const autoIds=new Set([...Object.values(catalog.daily_check_policy.baseline),...Object.values(catalog.daily_check_policy.supplemental)].flat());
      assert.ok(writes[1].selected_check_ids.length);
      assert.ok(writes[1].selected_check_ids.every(id=>autoIds.has(id)));
      // 内置模板只读详情、历史及引用计划使用完整页面，不弹抽屉。
      await page.getByRole("button",{name:"查看模板",exact:true}).click();
      await page.locator(".inspection-template-detail h1").filter({hasText:"系统标准日报"}).waitFor();
      await page.screenshot({path:path.join(resultDir,product+"-template-detail-"+width+".png"),fullPage:false});
      const detailPage=product==="kbot4" ? page.locator("section.daily-template-editor").last() : page.locator("main.daily-template-editor");
      assert.equal(await detailPage.evaluate(node=>node.scrollWidth>node.clientWidth+2),false);
      await verifyTemplateDetail(page, product, width, "after-edit");
      await page.getByRole("button",{name:"版本历史",exact:true}).click();
      await page.getByText(/发布人：.*template-admin/).waitFor();
      assert.equal(await page.getByRole("button",{name:"基于此版本编辑",exact:true}).count(),0);
      await page.getByRole("button",{name:"引用计划",exact:true}).click();
      await page.getByText(/生产库巡检.*v1/).waitFor();
      await page.getByRole("button",{name:"返回模板中心",exact:true}).click();
      // 409 是专门注入的版本竞争，不把预期 HTTP 失败算作控制台异常。
      assert.deepEqual(errors.filter(message=>!message.includes("409 (Conflict)")),[]);
      await page.close();
    }
  }
  if (!process.argv.includes("--templates-only")) for (const product of ["ammolite", "kbot4"]) {
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
          assert.deepEqual(await page.locator("[data-report-section]").evaluateAll(nodes=>nodes.map(node=>node.dataset.reportSection)),data.sections.map(region=>region.section_id));
          assert.equal(await page.locator(".daily-document-index [data-section-link]").count(),data.sections.length);
          assert.equal(await page.locator(".daily-document-content #daily-sql").count(),1);
          for (const issue of data.finding_index || []) assert.equal(await page.locator('[id="'+issue.anchor+'"]').count(),1);
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
              const full = card.locator(".inspection-sql-full-text").filter({has:page.locator("code")});
              assert.equal(await full.getAttribute("open"), null);
              await full.locator("summary").click();
              assert.equal(await full.locator("code").textContent(), statement.statement_text);
              await full.locator("summary").click();
            }
          }
          assert.ok(!(await page.locator(".daily-checks-table").allTextContents()).join(" ").includes("皮秒"));
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
          const table = page.locator(".daily-report-section .daily-checks-table").first();
          assert.ok(await table.count() > 0);
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
          assert.ok((await trace.innerText()).includes("仅供技术核查"));
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
  console.log(process.argv.includes("--templates-only") ? "PASS 两端分步编辑器 × 3 个宽度：选择保留、版式、模拟预览、失败/竞争/会话隔离、确认发布与无横向溢出" : "PASS 两端各 "+Object.keys(samples).length+" 类报告 × 3 个宽度，以及完整模板编辑/模拟预览/发布：共享区域、SQL 分析、基线、附录、安全转义与无横向溢出");
} finally {
  await browser?.close();
  await server.close();
}
