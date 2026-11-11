// 用户管理列表、新建抽屉与删除确认回归；所有写操作仅作用于浏览器夹具。
// node tests/contract/aiops_users_crud.mjs
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
const artifacts=mkdtempSync(resolve(tmpdir(), 'aiops-users-crud-'));
const base='/api/v1/apps/aiops/members', managedId='managed+user&1';
const server=createServer((req, res)=>{
  try {
    const path=resolve(root, '.'+new URL(req.url, 'http://fixture').pathname);
    if(!path.startsWith(root+'/ui/'))return res.writeHead(404).end();
    res.setHeader('Content-Type', ({'.html':'text/html', '.js':'text/javascript', '.css':'text/css'}[extname(path)] || 'text/plain')+'; charset=utf-8');
    res.end(readFileSync(path));
  } catch {res.writeHead(404).end();}
});
function installFixture(mode) {
  const base='/api/v1/apps/aiops/members', managedId='managed+user&1';
  globalThis.fixtureFailures=[];
  globalThis.fixtureHoldWrites=false;
  globalThis.fixtureWrites=[];
  globalThis.fixtureUsers=[
    {user_id:managedId, display_name:'隔离用户 <img src=x onerror="throw 1">', status:'ACTIVE', protected:false},
    {user_id:'initial-admin', display_name:'受保护的初始管理员', status:'ACTIVE', protected:true, membership:{initial_admin:true}},
    {user_id:'signed-in-admin', display_name:'当前账号', status:'ACTIVE', protected:false},
  ].map(row=>({...row, account_origin:'APP', owner_app_id:'aiops'}));
  globalThis.fixtureRequest=async(path, options={})=>{
    const pathname=new URL(path, 'http://fixture').pathname;
    if(options.method && options.method!=='GET') {
      const write={path:pathname, method:options.method, body:options.body?JSON.parse(options.body):null};
      fixtureWrites.push(write);
      if(fixtureHoldWrites)await new Promise(done=>{globalThis.fixtureReleaseWrite=done;});
      if(options.method==='POST' && pathname===base) {
        if(mode==='create-failure')throw new Error('隔离创建失败，请检查用户 ID。');
        fixtureUsers.push({...write.body, status:'ACTIVE', protected:false, account_origin:'APP', owner_app_id:'aiops'});
        return {user_id:write.body.user_id};
      }
      if(options.method==='DELETE' && pathname===base+'/'+encodeURIComponent(managedId)) {
        if(mode==='delete-failure')throw new Error('隔离删除失败，请刷新后重试。');
        fixtureUsers=fixtureUsers.filter(row=>row.user_id!==managedId);
        return {user_id:managedId, deleted:true};
      }
      fixtureFailures.push(pathname); throw new Error('禁止非预期写入');
    }
    if(pathname.endsWith('/access'))return {app_id:'aiops', user_id:'signed-in-admin', domain_id:12, domain_name:'隔离空间', roles:['app_admin'], permissions:['aiops:use', 'aiops:member_manage']};
    if(pathname===base+'/assignable-roles') {
      if(mode==='roles-failure')throw new Error('角色读取失败');
      return {items:mode==='no-roles'?[]:[{role_code:'app_user', display_name:'应用用户', status:'ACTIVE'}]};
    }
    if(pathname===base)return {items:structuredClone(fixtureUsers)};
    if(pathname==='/api/v1/notifications/summary')return {unread:0};
    if(pathname.endsWith('/output-jobs'))return {items:[], has_more:false, next_cursor:null};
    fixtureFailures.push(pathname); throw new Error('未定义的隔离请求：'+pathname);
  };
  globalThis.KBotAIOpsAuth={load:()=>({access_token:'isolated'}), request:fixtureRequest};
}

await new Promise(done=>server.listen(0, '127.0.0.1', done));
const browser=await chromium.launch({headless:true});
let checked=0;
try {
  for(const width of [1440, 420])for(const theme of ['light', 'dark'])for(const mode of ['normal', 'create-failure', 'delete-failure', 'roles-failure', 'no-roles', 'protected']) {
    const context=await browser.newContext({viewport:{width, height:1000}});
    context.setDefaultTimeout(8000);
    await context.addInitScript(installFixture, mode);
    await context.route('**/aiops-auth.js*', route=>route.fulfill({contentType:'text/javascript', body:''}));
    await context.route(/fonts\.(googleapis|gstatic)\.com/, route=>route.fulfill({contentType:'text/css', body:''}));
    await context.route(/\/api\/v1\//, route=>route.abort('blockedbyclient'));
    const page=await context.newPage(), errors=[];
    page.on('pageerror', error=>errors.push(error.message));
    try {
      await page.goto(`http://127.0.0.1:${server.address().port}/ui/aiops/users.html`);
      await page.evaluate(theme=>{document.documentElement.dataset.theme=theme;}, theme);
      await page.waitForFunction(()=>document.getElementById('user-count').textContent==='3 位用户' && !document.getElementById('user-refresh').disabled);
      const drawer=page.getByRole('dialog', {name:'新建用户', exact:true});
      const rows=page.locator('.user-list-row'), create=page.getByRole('button', {name:'新建用户', exact:true});
      const managed=rows.filter({has:page.locator('.user-list-id', {hasText:managedId})});
      assert.equal(await rows.count(), 3);
      assert.equal(await drawer.isVisible(), false);
      assert.equal(await page.locator('#user-form').isVisible(), false, '默认只显示列表');
      assert.equal(await page.locator('#user-list img').count(), 0);
      assert.ok((await rows.first().boundingBox()).y<450, '首屏必须能看到已有用户');
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth), false);
      for(const id of ['initial-admin', 'signed-in-admin']) {
        const remove=rows.filter({has:page.locator('.user-list-id', {hasText:id})}).getByRole('button', {name:'删除', exact:true});
        assert.equal(await remove.isDisabled(), true, '受保护账号和当前账号不能删除');
        await remove.dispatchEvent('click');
      }
      assert.deepEqual(await page.evaluate(()=>fixtureWrites), [], '进入页面及受保护操作不能发写请求');
      if(mode==='roles-failure' || mode==='no-roles') {
        assert.equal(await create.isDisabled(), true);
        assert.match(await page.locator('#user-status').innerText(), mode==='roles-failure'?/初始角色读取失败/:/没有可在本空间分配的角色/);
        assert.equal(await managed.getByRole('link', {name:'编辑／飞书绑定', exact:true}).isVisible(), true, '角色故障不能阻断已有用户管理');
      } else if(mode!=='delete-failure' && mode!=='protected') {
        await create.click();
        await drawer.waitFor();
        await page.waitForFunction(()=>document.activeElement===document.getElementById('user-id'));
        await drawer.evaluate(node=>Promise.all(node.getAnimations().map(animation=>animation.finished)));
        const bounds=await drawer.boundingBox();
        assert.equal(bounds.x+bounds.width, width, '新建抽屉必须靠右');
        assert.equal(bounds.y, 0);
        assert.equal(bounds.height, 1000);
        assert.ok(bounds.x>=0 && bounds.width<=width);
        assert.equal(await page.evaluate(()=>Boolean(document.elementFromPoint(innerWidth-40, 30)?.closest('#user-create-drawer'))), true, '抽屉右上角不能被页面顶栏遮挡');
        await page.keyboard.press('Escape');
        assert.equal(await drawer.isVisible(), true, 'Esc 不能关闭业务抽屉');
        if(width>700)await page.mouse.click(20, 500);
        assert.equal(await drawer.isVisible(), true, '背景误触不能关闭抽屉');
        await page.locator('#user-create-submit').focus();
        await page.keyboard.press('Tab');
        assert.equal(await drawer.getByRole('button', {name:'关闭新建用户'}).evaluate(node=>document.activeElement===node), true, '抽屉必须循环焦点');
        await page.locator('#user-password').fill('Cancelled-Password-123!');
        await drawer.getByRole('button', {name:'取消', exact:true}).click();
        assert.equal(await drawer.isVisible(), false);
        assert.equal(await create.evaluate(node=>document.activeElement===node), true);
        assert.equal(await page.locator('#user-password').inputValue(), '', '取消必须清除密码');
        assert.deepEqual(await page.evaluate(()=>fixtureWrites), []);
        await create.click();
        await page.locator('#user-id').fill('fixture-new-user');
        await page.locator('#user-display-name').fill('隔离新用户');
        await page.locator('#user-password').fill('Isolated-Password-123!');
        await page.locator('#user-role').selectOption('app_user');
        await page.locator('#user-feishu-open-id').fill('ou_fixture_new');
        if(mode==='normal') {
          await drawer.evaluate(node=>Promise.all(node.parentElement.getAnimations({subtree:true}).map(animation=>animation.finished)));
          await page.screenshot({path:resolve(artifacts, `create-${width}-${theme}.png`)});
        }
        await page.evaluate(()=>{fixtureHoldWrites=true;});
        await drawer.getByRole('button', {name:'创建用户', exact:true}).click();
        await page.waitForFunction(()=>fixtureWrites.length===1);
        assert.equal(await drawer.getByRole('button', {name:'关闭新建用户'}).isDisabled(), true);
        assert.equal(await drawer.getByRole('button', {name:'取消', exact:true}).isDisabled(), true);
        assert.equal(await page.locator('#user-id').isDisabled(), true);
        await page.evaluate(()=>document.getElementById('user-form').requestSubmit());
        assert.equal(await page.evaluate(()=>fixtureWrites.length), 1, '处理中不能重复提交');
        await page.evaluate(()=>{fixtureHoldWrites=false; fixtureReleaseWrite();});
        if(mode==='create-failure') {
          await drawer.getByText('隔离创建失败，请检查用户 ID。', {exact:true}).waitFor();
          assert.equal(await page.locator('#user-id').inputValue(), 'fixture-new-user');
          assert.equal(await page.locator('#user-password').inputValue(), 'Isolated-Password-123!');
          assert.equal(await drawer.isVisible(), true, '创建失败保留抽屉和用户输入');
          await drawer.getByRole('button', {name:'取消', exact:true}).click();
          assert.equal(await rows.count(), 3);
        } else {
          await drawer.waitFor({state:'hidden'});
          await page.waitForFunction(()=>document.getElementById('user-count').textContent==='4 位用户');
          assert.equal(await create.evaluate(node=>document.activeElement===node), true);
          assert.equal(await page.locator('#user-password').inputValue(), '');
          assert.equal(await rows.filter({has:page.locator('.user-list-id', {hasText:'fixture-new-user'})}).count(), 1);
        }
        assert.deepEqual((await page.evaluate(()=>fixtureWrites))[0], {path:base, method:'POST', body:{
          display_name:'隔离新用户', feishu_open_id:'ou_fixture_new', user_id:'fixture-new-user', password:'Isolated-Password-123!',
          must_change_password:true, role_bindings:[{role_code:'app_user', scope_mode:'SELECTED_DOMAINS', domain_ids:[12]}],
        }});
      }
      if(!['create-failure', 'protected'].includes(mode)) {
        const remove=managed.getByRole('button', {name:'删除', exact:true});
        await remove.click();
        const confirmation=page.getByRole('alertdialog');
        await confirmation.waitFor();
        assert.match(await confirmation.innerText(), /不可恢复/);
        assert.equal(await confirmation.locator('img').count(), 0, '确认内容须按普通文本渲染');
        assert.equal((await page.evaluate(()=>fixtureWrites)).some(row=>row.method==='DELETE'), false, '确认前不能删除');
        await confirmation.getByRole('button', {name:'取消', exact:true}).click();
        assert.equal(await managed.count(), 1);
        assert.equal((await page.evaluate(()=>fixtureWrites)).some(row=>row.method==='DELETE'), false);
        await remove.click();
        await confirmation.getByRole('button', {name:'删除用户', exact:true}).click();
        if(mode==='delete-failure') {
          await page.getByText('隔离删除失败，请刷新后重试。', {exact:true}).waitFor();
          assert.equal(await managed.count(), 1, '删除失败不能移除列表行');
          assert.equal(await remove.isEnabled(), true);
        } else {
          await managed.waitFor({state:'detached'});
          assert.equal(await page.locator('#user-status').innerText(), '用户已删除。');
        }
        assert.deepEqual((await page.evaluate(()=>fixtureWrites)).filter(row=>row.method==='DELETE'), [{path:base+'/'+encodeURIComponent(managedId), method:'DELETE', body:null}]);
      }
      if(mode==='normal')await page.screenshot({path:resolve(artifacts, `list-${width}-${theme}.png`), fullPage:true});
      assert.deepEqual(await page.evaluate(()=>fixtureFailures), []);
      assert.deepEqual(errors, []);
      checked++;
    } catch(error) {
      await page.screenshot({path:resolve(artifacts, `failure-${width}-${theme}-${mode}.png`), fullPage:true});
      console.error({artifacts, width, theme, mode, errors}); throw error;
    } finally {await context.close();}
  }
  console.log(`用户 CRUD 回归：${checked} 个场景通过；列表首屏、右侧抽屉、取消与焦点、重复提交、角色故障、删除确认及账号保护通过。截图目录 ${artifacts}`);
} finally {await browser.close(); await new Promise(done=>server.close(done));}
