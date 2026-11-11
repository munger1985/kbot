// 正式用户页的列表排版与编辑合同回归；接口全部隔离，不操作真实账号或飞书。
// node tests/contract/aiops_users_presentation.mjs
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
const artifacts=mkdtempSync(resolve(tmpdir(), 'aiops-users-presentation-'));
const users=[
  {user_id:'aiopsadmin', display_name:'AIOps 管理员', status:'ACTIVE', account_origin:'APP', owner_app_id:'aiops'},
  {user_id:'disabled-user', display_name:'已停用的用户', status:'DISABLED', account_origin:'APP', owner_app_id:'aiops'},
  {user_id:'ops_'+('long_user_'.repeat(10)), display_name:'长名称与 <img src=x onerror="throw 1"> 按普通文字显示', status:'UNRECOGNIZED', account_origin:'APP', owner_app_id:'aiops'},
  {user_id:'platform-admin', display_name:'不显示的平台账号', status:'ACTIVE', account_origin:'PLATFORM', owner_app_id:null},
  {user_id:'other-app-admin', display_name:'不显示的其他应用账号', status:'ACTIVE', account_origin:'APP', owner_app_id:'media_studio'},
];
const server=createServer((req, res)=>{
  try {
    const path=resolve(root, '.'+new URL(req.url, 'http://fixture').pathname);
    if(!path.startsWith(root+'/ui/'))return res.writeHead(404).end();
    res.setHeader('Content-Type', ({'.html':'text/html', '.js':'text/javascript', '.css':'text/css'}[extname(path)] || 'text/plain')+'; charset=utf-8');
    res.end(readFileSync(path));
  } catch {res.writeHead(404).end();}
});
function installFixture({users, mode}) {
  const base='/api/v1/apps/aiops/members';
  globalThis.fixtureRequests=[];
  globalThis.fixtureWrites=[];
  globalThis.fixtureFailures=[];
  globalThis.fixtureRequest=async(path, options={})=>{
    const pathname=new URL(path, 'http://fixture').pathname;
    fixtureRequests.push(pathname);
    await globalThis.recordRequest(pathname);
    if(options.method && options.method!=='GET') {
      if(mode!=='normal' || options.method!=='POST' || pathname!==base) {
        fixtureFailures.push(pathname);
        throw new Error('禁止非预期写入');
      }
      fixtureWrites.push({path:pathname, method:options.method, body:JSON.parse(options.body)});
      return {};
    }
    if(pathname.endsWith('/access'))return {app_id:'aiops', user_id:'fixture', domain_id:12, domain_name:'隔离空间',
      roles:mode==='denied'?['app_user']:['app_admin'], permissions:mode==='denied'?['aiops:use']:['aiops:use', 'aiops:member_manage']};
    if(pathname===base+'/assignable-roles')return {items:[{role_code:'app_user', display_name:'应用用户', status:'ACTIVE'}]};
    if(pathname===base) {
      if(mode==='failure')throw new Error('隔离用户列表读取失败');
      return {items:mode==='empty'?users.filter(row=>row.account_origin!=='APP' || row.owner_app_id!=='aiops'):users};
    }
    if(pathname===base+'/aiopsadmin')return {...users[0], feishu_binding:{id:'fixture-binding', row_version:3, external_id:'ou_fixture', status:'VERIFIED'}};
    if(pathname==='/api/v1/notification-delivery/capabilities')return {capabilities:['binding']};
    if(pathname.endsWith('/notifications/summary'))return {unread:0};
    if(pathname.endsWith('/output-jobs'))return {items:[], has_more:false, next_cursor:null};
    fixtureFailures.push(pathname);
    throw new Error('未定义的隔离请求：'+pathname);
  };
  globalThis.KBotAIOpsAuth={load:()=>({access_token:'isolated'}), request:fixtureRequest};
}

await new Promise(done=>server.listen(0, '127.0.0.1', done));
const browser=await chromium.launch({headless:true});
let checked=0;
try {
  for(const width of [1440, 420])for(const mode of ['normal', 'empty', 'failure', 'denied']) {
    const context=await browser.newContext({viewport:{width, height:1000}});
    context.setDefaultTimeout(8000);
    await context.addInitScript(installFixture, {users, mode});
    await context.route('**/aiops-auth.js*', route=>route.fulfill({contentType:'text/javascript', body:''}));
    await context.route(/fonts\.(googleapis|gstatic)\.com/, route=>route.fulfill({contentType:'text/css', body:''}));
    await context.route(/\/api\/v1\//, route=>route.abort('blockedbyclient'));
    await context.route('**/ui/aiops/dashboard.html', route=>route.fulfill({contentType:'text/html; charset=utf-8', body:'<!doctype html><meta charset="utf-8"><h1>运维总览</h1>'}));
    const page=await context.newPage();
    const errors=[], requests=[];
    await context.exposeFunction('recordRequest', path=>requests.push(path));
    page.on('pageerror', error=>errors.push(error.message));
    try {
      await page.goto(`http://127.0.0.1:${server.address().port}/ui/aiops/users.html`);
      if(mode==='denied') {
        await page.getByRole('heading', {name:'运维总览', exact:true}).waitFor();
        assert.deepEqual(requests, ['/api/v1/apps/aiops/access']);
      } else {
        await page.waitForFunction(()=>document.getElementById('user-count').textContent!=='读取中');
        assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth), false, '页面不能横向溢出');
        assert.equal(await page.locator('#user-list').evaluate(node=>getComputedStyle(node).listStyleType), 'none');
        if(mode==='normal') {
          const rows=page.locator('.user-list-row');
          assert.equal(await rows.count(), 3);
          assert.equal(await page.locator('#user-count').innerText(), '3 位用户');
          assert.deepEqual(await rows.locator('.status-pill').allTextContents(), ['已启用', '已停用', '状态未知']);
          assert.deepEqual(await rows.locator('.user-list-id').allTextContents(), users.slice(0, 3).map(row=>row.user_id));
          assert.equal(await rows.nth(2).locator('.user-list-name').innerText(), users[2].display_name);
          assert.equal(await page.locator('#user-list img').count(), 0);
          for(const row of await rows.all()) {
            const identity=await row.locator('.user-list-identity').boundingBox();
            const state=await row.locator('.status-pill').boundingBox();
            const action=await row.getByRole('link', {name:'编辑／飞书绑定', exact:true}).boundingBox();
            assert.ok(state.x>=identity.x+identity.width+11, '用户信息与状态必须分开');
            if(width>640)assert.ok(action.x>=state.x+state.width+15, '状态与按钮必须分开');
            else assert.ok(action.y>=Math.max(identity.y+identity.height, state.y+state.height)+11, '窄屏按钮必须换行');
          }
          await page.screenshot({path:resolve(artifacts, `users-${width}.png`), fullPage:true});
          await page.locator('#user-id').fill('fixture-new-user');
          await page.locator('#user-display-name').fill('隔离新用户');
          await page.locator('#user-password').fill('Isolated-Password-123!');
          await page.locator('#user-role').selectOption('app_user');
          await page.locator('#user-feishu-open-id').fill('ou_new_fixture');
          await page.getByRole('button', {name:'保存用户', exact:true}).click();
          await page.waitForFunction(()=>document.getElementById('user-status').textContent.includes('用户已保存'));
          assert.deepEqual(await page.evaluate(()=>fixtureWrites), [{path:'/api/v1/apps/aiops/members', method:'POST', body:{
            display_name:'隔离新用户', feishu_open_id:'ou_new_fixture', user_id:'fixture-new-user', password:'Isolated-Password-123!',
            must_change_password:true, role_bindings:[{role_code:'app_user', scope_mode:'SELECTED_DOMAINS', domain_ids:[12]}],
          }}], '新建用户仍使用原有账号、初始角色与候选身份合同');
          assert.equal(await rows.first().getByRole('link', {name:'编辑／飞书绑定', exact:true}).getAttribute('href'), './user-detail.html?userId=aiopsadmin');
          await rows.first().getByRole('link', {name:'编辑／飞书绑定', exact:true}).click();
          await page.locator('#member-feishu-open-id').waitFor();
          assert.ok(page.url().endsWith('/user-detail.html?userId=aiopsadmin'));
          assert.equal(await page.locator('#member-feishu-open-id').inputValue(), 'ou_fixture');
          assert.equal(await page.locator('#user-binding-status').innerText(), '已验证');
          assert.deepEqual(await page.evaluate(()=>fixtureWrites), [], '进入详情不能自动写入或发送验证码');
          await page.getByRole('link', {name:'← 返回用户', exact:true}).click();
          await page.locator('.user-list-row').first().waitFor();
          await page.getByRole('button', {name:'新建用户', exact:true}).click();
          assert.equal(await page.locator('#user-id').evaluate(node=>document.activeElement===node), true, '新建用户必须聚焦用户 ID');
          assert.equal(await page.locator('#user-id').inputValue(), '');
          assert.equal(await page.locator('#user-feishu-open-id').inputValue(), '');
          assert.equal(await page.locator('#user-status').innerText(), '');
        } else {
          assert.equal(await page.locator('.user-list-row').count(), 0);
          assert.equal(await page.locator('#user-count').innerText(), mode==='empty'?'0 位用户':'暂不可用');
          await page.getByText(mode==='empty'?'暂无 AIOps 自建用户。':'用户列表读取失败，请刷新后重试。', {exact:true}).waitFor();
          assert.deepEqual(await page.evaluate(()=>fixtureWrites), []);
        }
        assert.deepEqual(await page.evaluate(()=>fixtureFailures), []);
      }
      assert.deepEqual(errors, []);
      checked++;
    } catch (error) {
      await page.screenshot({path:resolve(artifacts, `failure-${width}-${mode}.png`), fullPage:true});
      console.error({artifacts, mode, width, errors});
      throw error;
    } finally {await context.close();}
  }
  console.log(`用户列表回归：${checked} 个场景通过；桌面／窄屏、状态、空列表、失败、权限与隔离编辑通过。截图目录 ${artifacts}`);
} finally {
  await browser.close();
  await new Promise(done=>server.close(done));
}
