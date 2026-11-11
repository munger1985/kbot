// 使用正式页面验证新增后的首屏位置；所有 API 隔离，禁止向真实环境写入。
// node tests/contract/aiops_management_navigation.mjs
import assert from 'node:assert/strict';
import {readFileSync, mkdtempSync} from 'node:fs';
import {createServer} from 'node:http';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {dirname, extname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {templateAlignmentFixture} from './aiops_template_alignment.mjs';

const root=resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const portal=resolve(process.env.AMMOLITE_ROOT || resolve(root, '../ammolite_cube'), 'apps/portal-web');
const require=createRequire(resolve(portal, 'package.json'));
const {chromium}=require('@playwright/test');
const {createServer:createViteServer}=await import(require.resolve('vite'));
const vue=(await import(require.resolve('@vitejs/plugin-vue'))).default;
const artifacts=mkdtempSync(resolve(tmpdir(), 'aiops-management-navigation-'));
const {catalog, layout}=templateAlignmentFixture(root);
const permissions=['aiops:use','aiops:agent_manage','aiops:plan_manage'];
const template={inspection_template_id:'template-1',display_name:'隔离巡检模板',source_kind:'CUSTOM',template_kind:'DAILY',status:'ACTIVE',version_no:1,row_version:1,selected_check_ids:['availability'],daily_report_layout:layout,current_version_id:'version-1',has_draft:false,definition:{display_name:'隔离巡检模板',selected_check_ids:['availability'],daily_report_layout:layout}};
const agent={agent_id:'agent-1',display_name:'隔离 Agent',status:'ACTIVE',version_no:1,allowed_business_system_ids:[],capabilities:['MONITORING_QUERY'],knowledge_policy:{enabled:false},execution_policy:{monitoring_enabled:true,readonly_database_enabled:false,change_database_enabled:false},auto_trigger_policy:{enabled:false},models:{}};

function installFixture(data) {
  globalThis.fixtureFailures=[];
  globalThis.fixtureWrites=[];
  globalThis.fixtureRequest=async(path, init={})=>{
    if(init.method && init.method!=='GET') {
      fixtureWrites.push(path);
      throw new Error('首屏导航回归禁止写操作');
    }
    const suffix=new URL(path,'http://fixture').pathname;
    if(suffix.endsWith('/access')) {
      await new Promise(done=>setTimeout(done, Number(new URLSearchParams(location.search).get('accessDelay') || 0)));
      return {permissions:data.permissions,domain_id:'domain-1',user_id:'user-1',display_name:'隔离用户'};
    }
    if(suffix.endsWith('/inspection-check-catalog'))return data.catalog;
    if(suffix.endsWith('/inspection-templates'))return [data.template];
    if(suffix.endsWith('/inspection-templates/template-1'))return data.template;
    if(suffix.endsWith('/inspection-templates/template-1/versions'))return [{template_version_id:'version-1',version_no:1,current:true,definition:data.template.definition}];
    if(suffix.endsWith('/inspection-templates/template-1/plans'))return [];
    if(suffix.endsWith('/session-report-templates'))return [];
    if(suffix.endsWith('/agents'))return [data.agent];
    if(suffix.endsWith('/agents/agent-1'))return data.agent;
    if(suffix.endsWith('/agents/boundary-resources'))return {items:[]};
    if(suffix.endsWith('/business-systems') || suffix.endsWith('/operations-knowledge/tags') || suffix.endsWith('/operations-knowledge/knowledge-bases'))return {items:[]};
    if(suffix.endsWith('/model-catalog') || suffix.endsWith('/agent-model-catalog'))return [];
    if(suffix.endsWith('/notifications/summary'))return {unread:0};
    if(suffix.endsWith('/output-jobs'))return {items:[],has_more:false,next_cursor:null};
    fixtureFailures.push(path);
    throw new Error('未定义的隔离请求：'+path);
  };
  globalThis.KBotAIOpsAuth={load:()=>({access_token:'isolated'}),request:fixtureRequest,uuid:()=>crypto.randomUUID()};
}

const entry=`
import {createApp,computed,h} from 'vue';
import {createRouter,createWebHistory,RouterView} from 'vue-router';
import {businessAppPermissionsKey} from '/src/app/businessAppContext.ts';
import '/src/shared/ui/styles.css';
const page=new URLSearchParams(location.search).get('page');
const component=(page==='agents' ? await import('/src/features/aiops/management/AIOpsAgentsView.vue') : await import('/src/features/aiops/management/AIOpsReportTemplatesView.vue')).default;
const router=createRouter({history:createWebHistory(),routes:[{path:'/apps/aiops/:domainCode/:pathMatch(.*)*',component,meta:{title:page==='agents'?'Agent 管理':'模板中心'}}]});
const app=createApp({render:()=>h(RouterView)});
app.use(router);
app.provide(businessAppPermissionsKey,computed(()=>Object.fromEntries(${JSON.stringify(permissions)}.map(key=>[key,true]))));
await router.replace('/apps/aiops/fixture/'+page+'?domainId=domain-1');
app.mount('#reference');
`;
const reference=await createViteServer({root:portal,configFile:false,cacheDir:resolve(artifacts,'vite-cache'),logLevel:'error',
  resolve:{alias:[{find:'@',replacement:resolve(portal,'src')}]},
  server:{host:'127.0.0.1',port:0,fs:{allow:[portal]}},
  plugins:[vue(),{name:'isolated-management-navigation',enforce:'pre',
    resolveId(id){if(['@/shared/api/http',resolve(portal,'src/shared/api/http'),resolve(portal,'src/shared/api/http.ts')].includes(id))return '\0fixture-http';if(id==='/fixture-entry.js')return '\0fixture-entry';},
    load(id){if(id==='\0fixture-http')return 'export const businessFetch=(path,options={})=>globalThis.fixtureRequest(path,options.init || {});';if(id==='\0fixture-entry')return entry;},
    configureServer(server){server.middlewares.use((req,res,next)=>{
      if(!req.url.startsWith('/reference.html'))return next();
      res.setHeader('Content-Type','text/html');
      res.end('<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><div class="business-app-shell business-app-aiops"><header class="business-app-topbar"></header><div class="business-app-layout"><aside class="business-app-sidebar"></aside><div id="reference" class="business-app-content"></div></div></div><script type="module" src="/fixture-entry.js"></script></body></html>');
    });},
  }],
});
const staticServer=createServer((req,res)=>{
  try {
    const path=resolve(root,'.'+new URL(req.url,'http://fixture').pathname);
    if(!path.startsWith(root+'/ui/'))return res.writeHead(404).end();
    res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css'}[extname(path)] || 'text/plain')+'; charset=utf-8');
    res.end(readFileSync(path));
  } catch {res.writeHead(404).end();}
});
await reference.listen();
await new Promise(done=>staticServer.listen(0,'127.0.0.1',done));
const browser=await chromium.launch({headless:true});
const failures=[];
let checked=0;
async function inFirstViewport(page,locator,label) {
  // 只读取位置，不使用点击、scrollIntoView 或截图前滚动掩盖首屏空白。
  const box=await locator.boundingBox();
  assert.ok(box && box.y>=0 && box.y+box.height<=page.viewportSize().height,label+'必须直接显示在首屏');
}
try {
  for(const platform of ['KBot','Ammolite'])for(const width of [1440,1024,420,320])for(const name of (process.env.AIOPS_PAGES || 'agents,report-templates').split(',')) {
    const context=await browser.newContext({viewport:{width,height:1000}});
    context.setDefaultTimeout(8000);
    await context.addInitScript(installFixture,{catalog,template,agent,permissions});
    await context.route(/fonts\.(googleapis|gstatic)\.com/,route=>route.fulfill({contentType:'text/css',body:''}));
    await context.route('**/aiops-auth.js*',route=>route.fulfill({contentType:'text/javascript',body:''}));
    const page=await context.newPage();
    const errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    try {
      const base=platform==='KBot'?`http://127.0.0.1:${staticServer.address().port}/ui/aiops/${name}.html`:`http://127.0.0.1:${reference.httpServer.address().port}/reference.html?page=${name}`;
      for(const delay of platform==='KBot'?[0,150]:[0]) {
        await page.goto(base+(base.includes('?')?'&':'?')+'accessDelay='+delay);
        if(name==='agents') {
          for(const mode of ['create','edit']) {
            await page.getByRole('button',{name:mode==='create'?/^新建 Agent/:'编辑',exact:mode!=='create'}).click();
            const dialog=page.getByRole('dialog',{name:mode==='create'?'新建 Agent':'编辑 Agent',exact:true});
            await dialog.waitFor();
            await dialog.boundingBox();
            await page.evaluate(()=>Promise.all(document.getAnimations().map(animation=>animation.finished.catch(()=>{}))));
            const box=await dialog.boundingBox();
            assert.ok(Math.abs(box.x+box.width-width)<=1,`${platform} ${width} Agent 抽屉必须贴右边，实际右边界 ${box.x+box.width}`);
            assert.ok(Math.abs(box.y)<=1 && Math.abs(box.height-1000)<=1,'Agent 抽屉必须占满视口高度');
            await inFirstViewport(page,dialog.locator('.admin-drawer-header'),'Agent 抽屉标题');
            if(mode==='create' && delay===0)await page.screenshot({path:resolve(artifacts,`${platform}-${name}-${width}-create.png`)});
            await dialog.getByRole('button',{name:'关闭',exact:true}).click();
            await dialog.waitFor({state:'hidden'});
          }
        } else {
          for(const mode of ['create','edit']) {
            await page.getByRole('button',{name:'编辑模板',exact:true}).waitFor();
            await page.getByRole('button',{name:mode==='create'?/^新建巡检模板/:'编辑模板',exact:mode!=='create'}).click();
            const editor=page.locator('.daily-template-editor:visible').filter({has:page.locator('.daily-editor-steps')});
            await editor.waitFor();
            await inFirstViewport(page,editor.locator('h1'),'模板编辑标题');
            await inFirstViewport(page,editor.locator('.daily-editor-steps'),'模板编辑步骤');
            if(platform==='KBot')assert.equal(await editor.evaluate(node=>node.parentElement.id),'ops-shell-content','编辑器应与目录共用内容壳层');
            if(mode==='create' && delay===0)await page.screenshot({path:resolve(artifacts,`${platform}-${name}-${width}-create.png`)});
            await editor.getByRole('button',{name:'返回模板中心',exact:true}).click();
            await editor.waitFor({state:'hidden'});
            await inFirstViewport(page,page.getByRole('heading',{name:'模板中心',exact:true}),'模板目录标题');
          }
          await page.getByRole('button',{name:'隔离巡检模板',exact:true}).click();
          const detail=page.locator('.inspection-template-detail:visible');
          await detail.getByRole('heading',{name:'隔离巡检模板',exact:true}).waitFor();
          await inFirstViewport(page,detail.locator('h1'),'模板详情标题');
          if(platform==='KBot')assert.equal(await detail.evaluate(node=>node.parentElement.id),'ops-shell-content','详情应与目录共用内容壳层');
          await detail.getByRole('button',{name:'返回模板中心',exact:true}).click();
        }
        assert.deepEqual(await page.evaluate(()=>fixtureWrites),[],'导航不得发起业务写操作');
        assert.deepEqual(await page.evaluate(()=>fixtureFailures),[],'不允许静默忽略未定义的请求');
        assert.deepEqual(errors,[],'不允许浏览器脚本异常');
        checked++;
      }
    } catch(error) {
      await page.screenshot({path:resolve(artifacts,`${platform}-${name}-${width}.png`)});
      failures.push(`${platform} ${name} ${width}: ${error.message}`);
      console.error(failures.at(-1));
      console.error(await page.evaluate(()=>({errors:fixtureFailures,writes:fixtureWrites,editorHidden:document.getElementById('daily-template-editor')?.hidden,editorParent:document.getElementById('daily-template-editor')?.parentElement?.id,createBound:Boolean(document.getElementById('create-inspection-template')?.onclick)})),errors);
    } finally {await context.close();}
  }
  console.log(`首屏导航回归：${checked} 个场景通过；截图目录 ${artifacts}`);
  assert.deepEqual(failures,[]);
} finally {
  await browser.close();await reference.close();await new Promise(done=>staticServer.close(done));
}
