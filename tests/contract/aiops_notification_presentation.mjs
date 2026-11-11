// 使用两仓正式页面对照通知渠道间距与抽屉文字；隔离全部 API，禁止发送通知。
// node tests/contract/aiops_notification_presentation.mjs
import assert from 'node:assert/strict';
import {readFileSync, mkdtempSync} from 'node:fs';
import {createServer} from 'node:http';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {dirname, extname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const portal=resolve(process.env.AMMOLITE_ROOT || resolve(root, '../ammolite_cube'), 'apps/portal-web');
const require=createRequire(resolve(portal, 'package.json'));
const {chromium}=require('@playwright/test');
const {createServer:createViteServer}=await import(require.resolve('vite'));
const vue=(await import(require.resolve('@vitejs/plugin-vue'))).default;
const artifacts=mkdtempSync(resolve(tmpdir(), 'aiops-notification-presentation-'));

function installFixture({configured}) {
  globalThis.fixtureFailures=[];
  globalThis.fixtureWrites=[];
  globalThis.fixtureRequest=async(path, options={})=>{
    if(options.method && options.method!=='GET') {
      fixtureWrites.push(path);
      throw new Error('样式回归禁止业务写入');
    }
    const suffix=new URL(path, 'http://fixture').pathname;
    if(suffix.endsWith('/access'))return {app_id:'aiops', roles:['app_admin'], permissions:['aiops:use'], domain_id:12, domain_name:'隔离空间', user_id:'fixture'};
    if(suffix.endsWith('/capabilities'))return {capabilities:['channel', 'binding', 'read', 'retry']};
    if(suffix.endsWith('/settings'))return {public_origin:configured?'https://portal.example.com':'', row_version:configured?1:0};
    if(suffix.endsWith('/targets'))return {connections:configured?[{id:'fixture', provider:'FEISHU', name:'隔离飞书', app_id:'cli_fixture', enabled:true, health:'TOKEN_VALID', row_version:1}]:[], bindings:[]};
    if(suffix.endsWith('/catalog'))return {events:{}, providers:[]};
    if(suffix.endsWith('/policies'))return {items:[]};
    if(suffix.endsWith('/notifications/summary'))return {unread:0};
    if(suffix.endsWith('/output-jobs'))return {items:[], has_more:false, next_cursor:null};
    fixtureFailures.push(path);
    throw new Error('未定义的隔离请求：'+path);
  };
  globalThis.KBotAIOpsAuth={load:()=>({access_token:'isolated'}), request:fixtureRequest};
}

const entry=`
import {createApp} from 'vue';
import {createRouter,createWebHistory} from 'vue-router';
import NotificationChannelsView from '/src/features/governance/NotificationChannelsView.vue';
import '/src/shared/ui/styles.css';
const router=createRouter({history:createWebHistory(),routes:[{path:'/:pathMatch(.*)*',component:NotificationChannelsView}]});
const app=createApp(NotificationChannelsView);
app.use(router);
await router.isReady();
app.mount('#reference');
`;
const reference=await createViteServer({root:portal, configFile:false, cacheDir:resolve(artifacts, 'vite-cache'), logLevel:'error',
  resolve:{alias:[{find:'@', replacement:resolve(portal, 'src')}]},
  server:{host:'127.0.0.1', port:0, fs:{allow:[portal]}},
  plugins:[vue(), {name:'isolated-notification-presentation', enforce:'pre',
    resolveId(id, importer) {
      if(['@/shared/api/http', resolve(portal, 'src/shared/api/http'), resolve(portal, 'src/shared/api/http.ts')].includes(id))return '\0fixture-http';
      if(id==='./http' && importer?.endsWith('/src/shared/api/notificationDelivery.ts'))return '\0fixture-http';
      if(id==='/fixture-entry.js')return '\0fixture-entry';
    },
    load(id) {
      if(id==='\0fixture-http')return 'export const apiFetch=(path,options={})=>globalThis.fixtureRequest(path,options);';
      if(id==='\0fixture-entry')return entry;
    },
    configureServer(server) {
      server.middlewares.use((req, res, next)=>{
        if(!req.url.startsWith('/reference.html'))return next();
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        res.end('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><div id="reference"></div><script type="module" src="/fixture-entry.js"></script></body></html>');
      });
    },
  }],
});
const staticServer=createServer((req, res)=>{
  try {
    const path=resolve(root, '.'+new URL(req.url, 'http://fixture').pathname);
    if(!path.startsWith(root+'/ui/'))return res.writeHead(404).end();
    res.setHeader('Content-Type', ({'.html':'text/html', '.js':'text/javascript', '.css':'text/css'}[extname(path)] || 'text/plain')+'; charset=utf-8');
    res.end(readFileSync(path));
  } catch {res.writeHead(404).end();}
});

async function snapshot(page, kind) {
  return page.evaluate(kind=>{
    const prefix=kind==='kbot'?'#delivery-channels-content':'.delivery-page';
    const blocks=[...document.querySelectorAll(prefix+' > .delivery-card, '+prefix+' > .delivery-channel-grid')].map(node=>node.getBoundingClientRect());
    const cards=[...document.querySelectorAll('.delivery-channel-grid > .delivery-card')].map(node=>node.getBoundingClientRect());
    return {
      sectionGaps:blocks.slice(1).map((rect,index)=>Math.round((rect.top-blocks[index].bottom)*1000)/1000),
      channelGap:Math.round((cards[1].top>cards[0].top?cards[1].top-cards[0].bottom:cards[1].left-cards[0].right)*1000)/1000,
      overflow:document.documentElement.scrollWidth>innerWidth,
    };
  }, kind);
}
async function styles(locator, properties=['fontFamily', 'fontSize', 'fontWeight', 'lineHeight', 'color']) {
  return locator.evaluate((node, properties)=>{
    const css=getComputedStyle(node);
    return Object.fromEntries(properties.map(key=>[key, css[key]]));
  }, properties);
}

await reference.listen();
await new Promise(done=>staticServer.listen(0, '127.0.0.1', done));
const browser=await chromium.launch({headless:true});
let checked=0;
try {
  for(const width of [1440, 420])for(const configured of [false, true]) {
    const context=await browser.newContext({viewport:{width, height:1000}});
    context.setDefaultTimeout(10000);
    await context.addInitScript(installFixture, {configured});
    await context.route('**/aiops-auth.js*', route=>route.fulfill({contentType:'text/javascript', body:''}));
    await context.route(/fonts\.(googleapis|gstatic)\.com/, route=>route.fulfill({contentType:'text/css', body:''}));
    await context.route(/\/api\/v1\//, route=>route.abort('blockedbyclient'));
    const kbot=await context.newPage(), ammolite=await context.newPage();
    const errors=[];
    for(const page of [kbot, ammolite])page.on('pageerror', error=>errors.push(error.message));
    try {
      await Promise.all([
        kbot.goto(`http://127.0.0.1:${staticServer.address().port}/ui/aiops/notification-channels.html`),
        ammolite.goto(reference.resolvedUrls.local[0]+'reference.html'),
      ]);
      for(const page of [kbot, ammolite])await page.getByRole('button', {name:configured?'编辑配置':'配置飞书', exact:true}).waitFor();
      const actual=await snapshot(kbot, 'kbot'), expected=await snapshot(ammolite, 'ammolite');
      assert.equal(await kbot.locator('#delivery-status').isVisible(), false, '空状态行不应留下额外留白');
      for(const [name, page] of [['kbot', kbot], ['ammolite', ammolite]])await page.screenshot({path:resolve(artifacts, `${name}-${width}-${configured}-page.png`), fullPage:true});
      console.log(JSON.stringify({width, configured, actual, expected}));
      assert.deepEqual(actual, expected, '卡片间距和窄屏溢出必须与 Ammolite 一致');
      assert.deepEqual(expected.sectionGaps, [20, 20]);
      assert.equal(expected.channelGap, 20);
      for(const page of [kbot, ammolite])await page.getByRole('button', {name:'编辑公共设置', exact:true}).click();
      const kbotDrawer=kbot.getByRole('dialog', {name:'通知公共设置', exact:true});
      const referenceDrawer=ammolite.getByRole('dialog', {name:'通知公共设置', exact:true});
      for(const drawer of [kbotDrawer, referenceDrawer])await drawer.waitFor();
      for(const selector of ['.field-label', '.control-field > small', '.delivery-form > .field-help', 'input[name="public_origin"], input[type="url"]', 'h2']) {
        assert.deepEqual(await styles(kbotDrawer.locator(selector)), await styles(referenceDrawer.locator(selector)), `抽屉文字样式不一致：${selector}`);
      }
      for(const selector of ['.admin-drawer-header', '.admin-drawer-content', '.admin-drawer-footer']) {
        const padding=['paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft'];
        assert.deepEqual(await styles(kbotDrawer.locator(selector), padding), await styles(referenceDrawer.locator(selector), padding), `抽屉内边距不一致：${selector}`);
      }
      for(const [name, page, drawer] of [['kbot', kbot, kbotDrawer], ['ammolite', ammolite, referenceDrawer]]) {
        const box=await drawer.boundingBox();
        assert.ok(Math.abs(box.x+box.width-width)<1, '抽屉必须靠右');
        assert.equal(await drawer.locator('input[type="url"]').inputValue(), configured?'https://portal.example.com':'');
        await page.screenshot({path:resolve(artifacts, `${name}-${width}-${configured}-drawer.png`)});
        await drawer.getByRole('button', {name:'取消', exact:true}).click();
        await drawer.waitFor({state:'hidden'});
        assert.deepEqual(await page.evaluate(()=>fixtureFailures), []);
        assert.deepEqual(await page.evaluate(()=>fixtureWrites), []);
      }
      // 同页的飞书配置抽屉也复用相同字段语义，不改变表单值或提交方式。
      for(const page of [kbot, ammolite])await page.getByRole('button', {name:configured?'编辑配置':'配置飞书', exact:true}).click();
      const kbotEditor=kbot.getByRole('dialog', {name:configured?'编辑飞书配置':'配置飞书', exact:true});
      const referenceEditor=ammolite.getByRole('dialog', {name:configured?'编辑飞书配置':'配置飞书', exact:true});
      for(const drawer of [kbotEditor, referenceEditor])await drawer.waitFor();
      for(const selector of ['.field-label', '.delivery-form > .field-help']) {
        const fields=kbotEditor.locator(selector), referenceFields=referenceEditor.locator(selector);
        assert.equal(await fields.count(), await referenceFields.count());
        for(let index=0;index<await fields.count();index++)assert.deepEqual(await styles(fields.nth(index)), await styles(referenceFields.nth(index)));
      }
      for(const [page, drawer] of [[kbot, kbotEditor], [ammolite, referenceEditor]]) {
        await drawer.getByRole('button', {name:'取消', exact:true}).click();
        await drawer.waitFor({state:'hidden'});
        assert.deepEqual(await page.evaluate(()=>fixtureWrites), []);
      }
      assert.deepEqual(errors, []);
      checked++;
    } catch (error) {
      for(const [name, page] of [['kbot', kbot], ['ammolite', ammolite]])await page.screenshot({path:resolve(artifacts, `${name}-${width}-${configured}-failure.png`), fullPage:true});
      console.error({artifacts, errors});
      throw error;
    } finally {await context.close();}
  }
  console.log(`通知渠道样式对照：${checked} 个场景通过；截图目录 ${artifacts}`);
} finally {
  await browser.close();
  await reference.close();
  await new Promise(done=>staticServer.close(done));
}
