// 对齐 Ammolite 成员详情的飞书身份交互；只使用隔离接口，不发送真实验证码。
// node tests/contract/aiops_user_detail.mjs
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
const artifacts=mkdtempSync(resolve(tmpdir(), 'aiops-user-detail-'));
const base='/api/v1/apps/aiops/members/aiopsadmin';
const server=createServer((req, res)=>{
  try {
    const path=resolve(root, '.'+new URL(req.url, 'http://fixture').pathname);
    if(!path.startsWith(root+'/ui/'))return res.writeHead(404).end();
    res.setHeader('Content-Type', ({'.html':'text/html', '.js':'text/javascript', '.css':'text/css'}[extname(path)] || 'text/plain')+'; charset=utf-8');
    res.end(readFileSync(path));
  } catch {res.writeHead(404).end();}
});
function installFixture(mode) {
  const base='/api/v1/apps/aiops/members/aiopsadmin';
  globalThis.fixtureFailures=[];
  globalThis.fixtureReadFail=mode==='failure';
  globalThis.fixtureUser={user_id:'aiopsadmin', display_name:'AIOps 管理员', status:'ACTIVE',
    feishu_binding:mode==='none'?null:{id:'fixture-binding', row_version:3, external_id:'ou_fixture', status:['unverified', 'caps-failure', 'no-binding-cap'].includes(mode)?'UNVERIFIED':mode==='disabled'?'DISABLED':'VERIFIED'}};
  globalThis.fixtureRequest=async(path, options={})=>{
    const pathname=new URL(path, 'http://fixture').pathname;
    const body=options.body?JSON.parse(options.body):null;
    await recordRequest({path:pathname, method:options.method || 'GET', body});
    if(options.method && options.method!=='GET') {
      if(options.method==='PATCH' && pathname===base) {
        if(mode==='conflict')throw new Error('绑定版本冲突，请刷新后重试。');
        if('feishu_open_id' in body) {
          assertBindingVersion(body);
          fixtureUser.feishu_binding={id:'fixture-binding', row_version:(fixtureUser.feishu_binding?.row_version || 0)+1,
            external_id:body.feishu_open_id, status:body.feishu_open_id?'UNVERIFIED':'DISABLED'};
        } else if('display_name' in body)fixtureUser.display_name=body.display_name;
        else throw new Error('未定义的修改');
        return {};
      }
      if(options.method==='POST' && pathname==='/api/v1/notification-delivery/bindings/fixture-binding/challenge')return {};
      fixtureFailures.push(pathname); throw new Error('禁止非预期写入');
    }
    if(pathname.endsWith('/access'))return {app_id:'aiops', user_id:'fixture', domain_id:12, domain_name:'隔离空间',
      roles:mode==='no-binding-cap'?['app_user']:['app_admin'], permissions:mode==='denied'?['aiops:use']:['aiops:use', 'aiops:member_manage']};
    if(pathname===base) {
      if(fixtureReadFail)throw new Error('用户读取失败，请刷新后重试。');
      return structuredClone(fixtureUser);
    }
    if(pathname==='/api/v1/notification-delivery/capabilities') {
      if(mode==='caps-failure')throw new Error('能力读取失败');
      return {capabilities:mode==='no-binding-cap'?[]:['binding']};
    }
    if(pathname==='/api/v1/notifications/summary')return {unread:0};
    if(pathname.endsWith('/output-jobs'))return {items:[], has_more:false, next_cursor:null};
    fixtureFailures.push(pathname); throw new Error('未定义的隔离请求：'+pathname);
  };
  function assertBindingVersion(body) {
    if(body.expected_binding_id!==(fixtureUser.feishu_binding?.id || null) || body.expected_binding_version!==(fixtureUser.feishu_binding?.row_version || null))throw new Error('未携带最新绑定版本');
  }
  globalThis.KBotAIOpsAuth={load:()=>({access_token:'isolated'}), request:fixtureRequest};
}

await new Promise(done=>server.listen(0, '127.0.0.1', done));
const browser=await chromium.launch({headless:true});
let checked=0;
try {
  for(const width of [1440, 420])for(const theme of ['light', 'dark'])for(const mode of ['verified', 'unverified', 'none', 'disabled', 'denied', 'failure', 'conflict', 'caps-failure', 'no-binding-cap']) {
    const context=await browser.newContext({viewport:{width, height:1000}});
    context.setDefaultTimeout(8000);
    const requests=[], errors=[];
    await context.exposeFunction('recordRequest', value=>requests.push(value));
    await context.addInitScript(installFixture, mode);
    await context.route('**/aiops-auth.js*', route=>route.fulfill({contentType:'text/javascript', body:''}));
    await context.route(/fonts\.(googleapis|gstatic)\.com/, route=>route.fulfill({contentType:'text/css', body:''}));
    await context.route(/\/api\/v1\//, route=>route.abort('blockedbyclient'));
    await context.route('**/ui/aiops/dashboard.html', route=>route.fulfill({contentType:'text/html; charset=utf-8', body:'<!doctype html><meta charset="utf-8"><h1>运维总览</h1>'}));
    const page=await context.newPage();
    page.on('pageerror', error=>errors.push(error.message));
    try {
      await page.goto(`http://127.0.0.1:${server.address().port}/ui/aiops/user-detail.html?userId=aiopsadmin`);
      await page.evaluate(theme=>{document.documentElement.dataset.theme=theme;}, theme);
      if(mode==='denied') {
        await page.getByRole('heading', {name:'运维总览', exact:true}).waitFor();
        assert.deepEqual(requests.map(value=>value.path), ['/api/v1/apps/aiops/access']);
      } else {
        if(mode==='failure') {
          await page.getByText('用户读取失败，请刷新后重试。', {exact:true}).waitFor();
          assert.equal(await page.locator('#user-detail-content').isVisible(), false);
          await page.evaluate(()=>{fixtureReadFail=false;});
          await page.getByRole('button', {name:'刷新', exact:true}).click();
        }
        const openId=page.locator('#member-feishu-open-id');
        await openId.waitFor();
        await page.waitForFunction(()=>!document.getElementById('user-binding-save').disabled);
        assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth), false, '详情不能横向溢出');
        assert.equal(await page.locator('#user-detail-name').innerText(), 'AIOps 管理员');
        const back=page.getByRole('link', {name:'← 返回用户', exact:true});
        assert.ok((await back.boundingBox()).width<200, '返回入口不能铺满后居中');
        const active=page.locator('a[href="./users.html"][aria-current="location"]');
        assert.equal(await active.count(), 1, '详情必须保持用户管理导航高亮');
        assert.equal(await openId.inputValue(), mode==='none'?'':'ou_fixture');
        const challenge=page.getByRole('button', {name:'发送验证码', exact:true});
        assert.equal(await challenge.isVisible(), mode==='unverified');
        const help=page.locator('.member-feishu-help');
        assert.equal(await help.evaluate(node=>node.open), false);
        await help.locator('summary').click();
        assert.equal(await help.locator('ol').isVisible(), true);
        await help.locator('summary').click();
        assert.deepEqual(requests.filter(value=>value.method!=='GET'), [], '查看详情与展开帮助不能自动写入');
        if(mode==='verified')await page.screenshot({path:resolve(artifacts, `user-detail-${width}-${theme}.png`), fullPage:true});
        if(mode==='unverified') {
          await challenge.click();
          await page.getByText('验证码已入队，请本人登录“我的通知设置”确认。', {exact:true}).waitFor();
          assert.deepEqual(requests.filter(value=>value.method==='POST'), [{path:'/api/v1/notification-delivery/bindings/fixture-binding/challenge', method:'POST', body:{}}]);
        }
        await openId.fill('ou_replacement');
        if(mode==='unverified')assert.equal(await challenge.isDisabled(), true, '未保存的新身份不能触发旧身份验证');
        await page.getByRole('button', {name:'保存绑定', exact:true}).click();
        if(mode==='conflict') {
          await page.getByText('绑定版本冲突，请刷新后重试。', {exact:true}).waitFor();
          assert.equal(await openId.inputValue(), 'ou_replacement', '失败须保留用户输入');
          assert.equal(await page.getByRole('button', {name:'保存绑定', exact:true}).isEnabled(), true);
        } else {
          await page.getByText('绑定已保存；新增或更换身份需本人验证。', {exact:true}).waitFor();
          const writes=requests.filter(value=>value.method==='PATCH');
          assert.deepEqual(writes, [{path:base, method:'PATCH', body:{feishu_open_id:'ou_replacement', expected_binding_id:mode==='none'?null:'fixture-binding', expected_binding_version:mode==='none'?null:3}}]);
          assert.equal(await page.locator('#user-binding-status').innerText(), '待本人验证');
          await openId.fill('');
          await page.getByRole('button', {name:'保存绑定', exact:true}).click();
          await page.waitForFunction(()=>document.getElementById('user-binding-status').textContent==='已解绑');
          assert.deepEqual(requests.filter(value=>value.method==='PATCH').at(-1).body,
            {feishu_open_id:null, expected_binding_id:'fixture-binding', expected_binding_version:mode==='none'?1:4});
          await page.locator('#user-display-name').fill('修改后的名称');
          await page.getByRole('button', {name:'保存资料', exact:true}).click();
          await page.getByText('用户资料已保存。', {exact:true}).waitFor();
          assert.deepEqual(requests.filter(value=>value.method==='PATCH').at(-1).body, {display_name:'修改后的名称'}, '资料保存不能连带修改绑定');
          assert.equal(await page.locator('#user-detail-name').innerText(), '修改后的名称');
        }
        assert.deepEqual(await page.evaluate(()=>fixtureFailures), []);
        if(mode==='verified')for(const query of ['', '?userId='+('x'.repeat(257))]) {
          const reads=requests.filter(value=>value.path===base).length;
          await page.goto(`http://127.0.0.1:${server.address().port}/ui/aiops/user-detail.html${query}`);
          await page.getByText('缺少有效的用户 ID，请返回用户列表重新选择。', {exact:true}).waitFor();
          assert.equal(await page.locator('#user-detail-content').isVisible(), false);
          assert.equal(requests.filter(value=>value.path===base).length, reads, '无效定位不能读取用户详情');
        }
      }
      assert.deepEqual(errors, []);
      checked++;
    } catch(error) {
      await page.screenshot({path:resolve(artifacts, `failure-${width}-${theme}-${mode}.png`), fullPage:true});
      console.error({artifacts, width, theme, mode, errors}); throw error;
    } finally {await context.close();}
  }
  console.log(`用户详情回归：${checked} 个场景通过；身份更换、解绑、版本校验、独立资料保存、验证码权限和失败恢复通过。截图目录 ${artifacts}`);
} finally {await browser.close(); await new Promise(done=>server.close(done));}
