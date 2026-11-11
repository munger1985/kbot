// 真实 AIOps 顶栏通知面板的样式与交互回归；全部 API 隔离，不读取或修改线上通知。
// node tests/contract/aiops_notification_center.mjs
import assert from 'node:assert/strict';
import {readFileSync, mkdtempSync} from 'node:fs';
import {createServer} from 'node:http';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {dirname, extname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require=createRequire(resolve(process.env.AMMOLITE_ROOT || resolve(root, '../ammolite_cube'), 'apps/portal-web/package.json'));
const {chromium}=require('@playwright/test');
const artifacts=mkdtempSync(resolve(tmpdir(), 'aiops-notification-center-'));
const situationId='019ffff0-0000-7000-8000-000000000013';
const items=[
  {inbox_id:'fixture-1', title:'数据库异常已确认：Oracle Test DB', summary:'Oracle Test DB：Oracle 数据库不可用', source_severity:'CRITICAL', read_at:null, row_version:3,
    domain_id:12, route_key:'aiops.situation.detail', route_params:{domain_id:'12', situation_id:situationId}},
  {inbox_id:'fixture-2', title:'数据库异常已恢复：Oracle Test DB · 长名称 <img src=x onerror="throw 1">', summary:'Oracle Test DB 的相关故障信号已全部恢复。长内容与技术标识也应自动换行：'+('database_resource_'.repeat(10)),
    source_severity:'CRITICAL', read_at:null},
];
const server=createServer((req, res)=>{
  try {
    const path=resolve(root, '.'+new URL(req.url, 'http://fixture').pathname);
    if(!path.startsWith(root+'/ui/'))return res.writeHead(404).end();
    res.setHeader('Content-Type', ({'.html':'text/html', '.js':'text/javascript', '.css':'text/css'}[extname(path)] || 'text/plain')+'; charset=utf-8');
    res.end(readFileSync(path));
  } catch {res.writeHead(404).end();}
});
function installFixture({mode, items}) {
  globalThis.fixtureFailures=[];
  globalThis.fixtureRequest=async(path, options={})=>{
    const pathname=new URL(path, 'http://fixture').pathname;
    await recordRequest({path, method:options.method || 'GET', body:options.body?JSON.parse(options.body):null});
    if(options.method && options.method!=='GET') {
      if(options.method==='PATCH' && pathname==='/api/v1/notifications/fixture-1')return {};
      fixtureFailures.push(path);
      throw new Error('禁止非预期写入');
    }
    if(pathname.endsWith('/access'))return {app_id:'aiops', user_id:'fixture', domain_id:12, domain_name:'隔离空间', roles:['app_admin'], permissions:['aiops:use', 'aiops:member_manage']};
    if(pathname.endsWith('/assignable-roles') || pathname==='/api/v1/apps/aiops/members')return {items:[]};
    if(pathname==='/api/v1/notifications/summary')return {unread:mode==='empty'?0:mode==='many'?120:2};
    if(pathname==='/api/v1/notifications') {
      if(mode==='failure')throw new Error('隔离通知读取失败');
      return {items:mode==='empty'?[]:mode==='many'?Array.from({length:20}, (_, index)=>({...items[0], title:'数据库通知 '+(index+1)})):items};
    }
    if(pathname==='/api/v1/notifications/fixture-1')return items[0];
    if(pathname.endsWith('/output-jobs'))return {items:[], has_more:false, next_cursor:null};
    fixtureFailures.push(path);
    throw new Error('未定义的隔离请求：'+path);
  };
  globalThis.KBotAIOpsAuth={load:()=>({access_token:'isolated'}), request:fixtureRequest};
}

await new Promise(done=>server.listen(0, '127.0.0.1', done));
const browser=await chromium.launch({headless:true});
let checked=0;
try {
  for(const width of [1440, 420])for(const theme of ['light', 'dark'])for(const mode of ['normal', 'empty', 'failure', 'many']) {
    const context=await browser.newContext({viewport:{width, height:800}});
    context.setDefaultTimeout(8000);
    const requests=[];
    await context.exposeFunction('recordRequest', value=>requests.push(value));
    await context.addInitScript(installFixture, {mode, items});
    await context.route('**/aiops-auth.js*', route=>route.fulfill({contentType:'text/javascript', body:''}));
    await context.route(/fonts\.(googleapis|gstatic)\.com/, route=>route.fulfill({contentType:'text/css', body:''}));
    await context.route(/\/api\/v1\//, route=>route.abort('blockedbyclient'));
    await context.route('**/ui/aiops/situations.html*', route=>route.fulfill({contentType:'text/html; charset=utf-8', body:'<!doctype html><meta charset="utf-8"><h1>告警诊断</h1>'}));
    const page=await context.newPage(), errors=[];
    page.on('pageerror', error=>errors.push(error.message));
    try {
      await page.goto(`http://127.0.0.1:${server.address().port}/ui/aiops/users.html`);
      await page.evaluate(theme=>{document.documentElement.dataset.theme=theme;}, theme);
      const toggle=page.getByRole('button', {name:'打开通知中心', exact:true});
      await toggle.waitFor();
      const panel=page.locator('.kbot-notification-panel');
      assert.equal(await panel.isVisible(), false);
      assert.equal(await toggle.locator('svg path').getAttribute('d'), 'M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4');
      const badge=toggle.locator('[data-notification-count]');
      await page.waitForFunction(()=>document.querySelector('[data-notification-count]').textContent!=='');
      assert.equal(await badge.isVisible(), mode!=='empty');
      if(mode!=='empty')assert.equal(await badge.innerText(), mode==='many'?'99+':'2');
      if(mode==='normal')await page.screenshot({path:resolve(artifacts, `bell-${width}-${theme}.png`)});
      await toggle.click();
      await page.waitForFunction(()=>Boolean(document.querySelector('[data-notification-status]').textContent));
      const bounds=await panel.boundingBox();
      assert.ok(bounds.x>=0 && bounds.x+bounds.width<=width, '通知面板不能超出屏幕');
      assert.equal(bounds.x+bounds.width, width, '通知抽屉必须贴右侧');
      assert.equal(bounds.y, 0);
      assert.equal(bounds.height, 800, '通知抽屉必须占满可视高度');
      assert.equal(await panel.locator('header').evaluate(node=>getComputedStyle(node).paddingLeft), width>600?'28px':'16px');
      assert.equal(await panel.locator('[role="status"]').evaluate(node=>getComputedStyle(node).paddingLeft), width>600?'28px':'16px');
      assert.equal(await panel.locator('[data-notification-close]').evaluate(node=>document.activeElement===node), true, '打开抽屉聚焦关闭按钮');
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth), false);
      if(mode==='normal' || mode==='many') {
        const articles=panel.locator('article');
        assert.equal(await articles.count(), mode==='normal'?2:20);
        const first=articles.first();
        assert.equal(await first.evaluate(node=>getComputedStyle(node).display), 'grid');
        assert.equal(await first.evaluate(node=>getComputedStyle(node).paddingLeft), '16px');
        assert.equal(await first.locator('p').evaluate(node=>getComputedStyle(node).marginTop), '0px');
        const severity=await first.locator('small').boundingBox(), action=await first.getByRole('button', {name:'查看结果'}).boundingBox();
        if(width>600)assert.ok(action.x>=severity.x+severity.width+13, '原始告警级别与操作按钮不能粘在一起');
        else assert.ok(action.y>=severity.y+severity.height+7, '窄屏查看按钮必须独立换行');
        assert.equal(await articles.locator('img').count(), 0);
        const list=panel.locator('[data-notification-list]');
        if(mode==='many') {
          assert.ok(await list.evaluate(node=>node.scrollHeight>node.clientHeight), '长列表应在面板内部滚动');
          await list.evaluate(node=>{node.scrollTop=node.scrollHeight;});
          assert.equal((await panel.locator('header').boundingBox()).y, 0, '列表滚动不能带走面板标题');
        } else await page.screenshot({path:resolve(artifacts, `center-${width}-${theme}.png`)});
      } else {
        assert.equal(await panel.locator('article').count(), 0);
        assert.equal(await panel.locator('[role="status"]').innerText(), mode==='empty'?'暂无通知。':'通知暂不可用，请刷新重试。');
      }
      await panel.getByRole('button', {name:'刷新', exact:true}).click();
      await page.keyboard.press('Escape');
      assert.equal(await panel.isVisible(), true, '与 Ammolite 一致，Esc 不关闭抽屉');
      if(width>600) {
        await page.mouse.click(20, 400);
        assert.equal(await panel.isVisible(), true, '与 Ammolite 一致，背景误触不关闭抽屉');
      }
      await panel.locator('footer a').focus();
      await page.keyboard.press('Tab');
      assert.equal(await panel.locator('[data-notification-close]').evaluate(node=>document.activeElement===node), true, '抽屉内必须循环焦点');
      await panel.getByRole('button', {name:'关闭通知中心', exact:true}).click();
      assert.equal(await panel.isVisible(), false);
      assert.equal(await toggle.evaluate(node=>document.activeElement===node), true);
      if(mode==='normal') {
        await toggle.click();
        await panel.getByRole('button', {name:'查看结果', exact:true}).first().click();
        await page.waitForURL('**/situations.html?situationId='+situationId);
        assert.deepEqual(requests.filter(value=>value.method!=='GET'), [{path:'/api/v1/notifications/fixture-1', method:'PATCH', body:{read:true, expected_row_version:3}}]);
      } else assert.deepEqual(requests.filter(value=>value.method!=='GET'), []);
      assert.deepEqual(await page.evaluate(()=>fixtureFailures), []);
      assert.deepEqual(errors, []);
      checked++;
    } catch (error) {
      await page.screenshot({path:resolve(artifacts, `failure-${width}-${theme}-${mode}.png`), fullPage:true});
      console.error({artifacts, width, theme, mode, errors});
      throw error;
    } finally {await context.close();}
  }
  console.log(`通知中心回归：${checked} 个场景通过；桌面／窄屏、明暗主题、空列表、失败、滚动与结果跳转通过。截图目录 ${artifacts}`);
} finally {
  await browser.close();
  await new Promise(done=>server.close(done));
}
