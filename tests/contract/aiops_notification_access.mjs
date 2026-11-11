// 验证通知治理的菜单、直达页面和鉴权等待态；不连接真实后端、不发送通知。
// node tests/contract/aiops_notification_access.mjs
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createServer} from 'node:http';
import {createRequire} from 'node:module';
import {dirname,extname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const require=createRequire(resolve(root,'../ammolite_cube/apps/portal-web/package.json'));
const {chromium}=require('@playwright/test');
const server=createServer((req,res)=>{
  try {
    const path=resolve(root,'.'+new URL(req.url,'http://fixture').pathname);
    if(!path.startsWith(root+'/ui/'))return res.writeHead(404).end();
    res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css'}[extname(path)] || 'text/plain')+'; charset=utf-8');
    res.end(readFileSync(path));
  } catch {res.writeHead(404).end();}
});
function installFixture({roles}) {
  globalThis.fixtureRequests=[];
  globalThis.fixtureFailures=[];
  globalThis.fixtureRequest=async(path,init={})=>{
    const pathname=new URL(path,'http://fixture').pathname;
    fixtureRequests.push(pathname);
    await globalThis.fixtureReportRequest(pathname);
    if(init.method && init.method!=='GET')throw new Error('权限回归禁止业务写入');
    if(pathname==='/api/v1/apps/aiops/access') {
      await new Promise(done=>{globalThis.fixtureReleaseAccess=done;});
      return {app_id:'aiops',user_id:'fixture',domain_id:12,domain_name:'隔离测试空间',roles,permissions:['aiops:use']};
    }
    if(pathname==='/api/v1/notifications/summary')return {unread:0};
    if(pathname.endsWith('/output-jobs'))return {items:[],has_more:false,next_cursor:null};
    if(pathname.endsWith('/capabilities'))return {capabilities:roles.includes('app_admin')?['channel','binding','read','retry']:[]};
    if(pathname.endsWith('/catalog'))return {events:{},providers:['FEISHU']};
    if(pathname.endsWith('/me/bindings'))return {items:[]};
    if(pathname.endsWith('/targets'))return {connections:[],bindings:[]};
    if(pathname.endsWith('/policies'))return {items:[]};
    if(pathname.endsWith('/settings'))return {public_origin:'',row_version:0};
    if(pathname.endsWith('/jobs'))return {items:[],has_more:false,limit:20,offset:0};
    if(pathname.endsWith('/health'))return {items:[],metrics:{acceptance_rate:null}};
    fixtureFailures.push(pathname);
    throw new Error('未定义的隔离请求：'+pathname);
  };
  globalThis.KBotAIOpsAuth={load:()=>({access_token:'isolated'}),request:fixtureRequest};
}
await new Promise(done=>server.listen(0,'127.0.0.1',done));
const browser=await chromium.launch({headless:true});
let checked=0;
try {
  for(const width of [1440,420])for(const role of ['app_user','app_manager','app_admin','platform_admin']) {
    const context=await browser.newContext({viewport:{width,height:1000}});
    context.setDefaultTimeout(8000);
    const requests=[];
    await context.exposeFunction('fixtureReportRequest',path=>{requests.push(path);});
    await context.addInitScript(installFixture,{roles:[role]});
    await context.route('**/aiops-auth.js*',route=>route.fulfill({contentType:'text/javascript',body:''}));
    await context.route(/fonts\.(googleapis|gstatic)\.com/,route=>route.fulfill({contentType:'text/css',body:''}));
    await context.route('**/ui/aiops/dashboard.html',route=>route.fulfill({contentType:'text/html; charset=utf-8',body:'<!doctype html><meta charset="utf-8"><h1>运维总览</h1>'}));
    const page=await context.newPage();
    const errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    try {
      const base=`http://127.0.0.1:${server.address().port}/ui/aiops/`;
      // 每个角色仍能进入自己的通知设置，但治理菜单只向应用管理员展示。
      await page.goto(base+'notification-settings.html');
      await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseAccess));
      await page.evaluate(()=>fixtureReleaseAccess());
      await page.waitForFunction(()=>document.getElementById('delivery-mine').textContent.includes('尚未绑定飞书'));
      for(const name of ['notification-channels','notification-delivery']) {
        assert.equal(await page.locator(`.business-app-sidebar a[href="./${name}.html"]`).count(),role==='app_admin'?1:0);
      }
      for(const name of ['notification-channels','notification-delivery']) {
        requests.length=0;
        await page.goto(base+name+'.html');
        await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseAccess));
        assert.equal(await page.locator('main.ops-main').isVisible(),false,'权限未返回前不能闪现治理页面');
        assert.deepEqual(await page.evaluate(()=>fixtureRequests),['/api/v1/apps/aiops/access']);
        await page.evaluate(()=>fixtureReleaseAccess());
        if(role==='app_admin') {
          await page.locator('main.ops-main').waitFor();
          if(name==='notification-channels')await page.getByRole('button',{name:'配置飞书',exact:true}).waitFor();
          else await page.getByText('没有符合筛选条件的投递记录。',{exact:true}).waitFor();
          assert.equal(await page.locator('#delivery-status').innerText(),'');
          assert.ok((await page.evaluate(()=>fixtureRequests)).some(path=>path.endsWith(name==='notification-channels'?'/targets':'/jobs')));
        } else {
          await page.waitForURL('**/dashboard.html');
          await page.getByRole('heading',{name:'运维总览',exact:true}).waitFor();
          assert.deepEqual(requests,['/api/v1/apps/aiops/access'],'无权直达只做权限检查，不加载通知治理接口');
        }
        assert.deepEqual(await page.evaluate(()=>fixtureFailures),[]);
        checked++;
      }
      assert.deepEqual(errors,[]);
    } finally {await context.close();}
  }
  console.log(`通知权限浏览器回归：${checked} 个直达场景通过；四类角色、双宽度菜单过滤及个人设置通过。`);
} finally {
  await browser.close();await new Promise(done=>server.close(done));
}
