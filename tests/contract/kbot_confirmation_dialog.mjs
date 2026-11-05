// 使用真实静态页面和 Chromium 验证共享确认框；所有业务请求仅使用本地桩。
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { resolve, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(resolve(root, '../ammolite_cube/apps/portal-web/package.json'));
const { chromium } = require('@playwright/test');
const artifacts = mkdtempSync(resolve(tmpdir(), 'kbot-confirmation-'));
// 不把其他工作尚未发布的页面或第三方脚本纳入本次产品入口清单。
const scripts = execFileSync('git', ['ls-files', 'ui/**/*.js'], { cwd: root, encoding: 'utf8' }).trim().split('\n');
for (const path of scripts.filter(path => !path.startsWith('ui/vendor/'))) {
  assert.doesNotMatch(readFileSync(resolve(root, path), 'utf8'), /(?<![\w.])(?:globalThis\.|window\.)?confirm\s*\(/, path + ' 必须使用共享确认 API');
}

const server = createServer((request, response) => {
  const url = new URL(request.url, 'http://localhost');
  const path = resolve(root, '.' + url.pathname);
  if (!path.startsWith(root + '/ui/')) { response.writeHead(404).end(); return; }
  try {
    response.setHeader('Content-Type', ({ '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript' })[extname(path)] + '; charset=utf-8');
    response.end(readFileSync(path));
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ headless: true });
const writes = [], errors = [];
const system = { business_system_id: 'system-1', system_code: 'ERP', display_name: 'ERP (demo)', status: 'DISABLED', environment: 'PROD', importance_level: 4, resource_count: 1, responsibility_group_id: 'group-1', row_version: 7 };
const page = await browser.newPage({ viewport: { width: 2048, height: 1000 } });
page.on('pageerror', error => errors.push(error.message));
page.on('dialog', async dialog => { errors.push('不应出现浏览器原生确认框'); await dialog.dismiss(); });
await page.route('**/ui/aiops/js/aiops-auth.js', route => route.fulfill({ contentType: 'text/javascript', body: `globalThis.KBotAIOpsAuth={load:()=>({access_token:'isolated-fixture'}),uuid:()=>crypto.randomUUID(),request:async(path,init={})=>{const response=await fetch(path,init);if(!response.ok)throw Error('隔离请求失败');return response.json();}};` }));
await page.route('**/api/**', async route => {
  const request = route.request(), url = new URL(request.url());
  let body = { items: [] };
  if (request.method() !== 'GET') {
    writes.push({ path: url.pathname, method: request.method(), headers: request.headers(), body: request.postDataJSON() });
    system.status = url.pathname.endsWith('/disable') ? 'DISABLED' : 'ENABLED';
    body = system;
  } else if (url.pathname.endsWith('/access')) body = { domain_id: 'domain-1', domain_name: '实验空间', display_name: '测试用户', permissions: ['aiops:use', 'aiops:target_manage', 'aiops:diagnostic_source_manage'] };
  else if (url.pathname.endsWith('/business-systems')) body = { items: [{ ...system }] };
  else if (url.pathname.endsWith('/responsibility-groups')) body = { items: [{ responsibility_group_id: 'group-1', name: '数据库组' }] };
  await route.fulfill({ json: body });
});
const base = `http://127.0.0.1:${server.address().port}`;
const dialog = page.locator('#kbot-confirm-dialog');
async function geometry(width) {
  await page.setViewportSize({ width, height: 900 });
  const rect = await dialog.boundingBox();
  assert.ok(rect.width <= 468.1 && rect.width <= width - 47, JSON.stringify(rect));
  assert.ok(Math.abs(rect.x + rect.width / 2 - width / 2) < 1, '确认框必须水平居中');
  assert.ok(await dialog.evaluate(node => node.parentElement === document.body));
  assert.equal(await dialog.evaluate(node => node.scrollWidth > node.clientWidth + 1), false);
  assert.equal(await page.locator('.confirm-dialog-actions button:focus').getAttribute('value'), 'cancel');
}
try {
  await page.goto(base + '/ui/aiops/business-systems.html');
  await page.getByRole('button', { name: '重新上线', exact: true }).waitFor();
  for (const width of [2048, 1440, 420, 320]) {
    await page.getByRole('button', { name: '重新上线', exact: true }).click();
    await dialog.waitFor();
    await geometry(width);
    await page.screenshot({ path: resolve(artifacts, `business-enable-${width}.png`) });
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'detached' });
    assert.equal(writes.length, 0, 'Esc 不得提交上线');
  }
  await page.getByRole('button', { name: '重新上线', exact: true }).click();
  await dialog.waitFor();
  await dialog.locator('[value="confirm"]').evaluate(button => { button.click(); button.click(); });
  await page.waitForFunction(() => document.querySelector('[data-offline-system]'));
  assert.equal(writes.length, 1);
  assert.ok(writes[0].path.endsWith('/enable'));
  assert.equal(writes[0].headers['if-match'], '"rv-7"');
  await page.locator('.business-system-more summary').click();
  await page.getByRole('button', { name: '下线业务系统', exact: true }).click();
  await dialog.waitFor();
  await geometry(320);
  await dialog.locator('[value="confirm"]').click();
  assert.equal(writes.length, 1, '下线原因必填');
  await dialog.locator('textarea').fill('   ');
  await dialog.locator('[value="confirm"]').click();
  assert.equal(writes.length, 1, '纯空白不是有效下线原因');
  await dialog.locator('textarea').fill('  联调测试结束  ');
  await dialog.locator('[value="confirm"]').click();
  await page.waitForFunction(() => document.querySelector('[data-enable-system]'));
  assert.equal(writes.length, 2);
  assert.equal(writes[1].body.reason, '联调测试结束');

  // 验证同一组件的长文本、复选项重置、重复打开及业务抽屉上的顶层模态行为。
  const open = options => page.evaluate(options => { globalThis.confirmationResult = null; KBotAIOpsShell.confirmActionWithOption(options).then(result => { globalThis.confirmationResult = result; }); }, options);
  await open({ title: '删除运维资料？', message: '<script>不得执行</script> ' + '长名称_'.repeat(100), tone: 'danger', option: { label: '同时删除物理文件', description: '共享文件会保留。' } });
  await dialog.waitFor();
  await geometry(320);
  assert.equal(await dialog.locator('script').count(), 0);
  await dialog.locator('input[type="checkbox"]').check();
  const duplicate = await page.evaluate(() => KBotAIOpsShell.confirmAction({ title: '重复操作', message: '不得替换原确认框' }));
  assert.equal(duplicate, false);
  assert.equal(await dialog.count(), 1);
  await dialog.locator('[value="cancel"]').click();
  assert.deepEqual(await page.evaluate(() => confirmationResult), { confirmed: false, optionSelected: false, reason: null });
  await open({ title: '删除资料？', message: '再次确认', option: { label: '同时删除物理文件' } });
  await dialog.waitFor();
  assert.equal(await dialog.locator('input').isChecked(), false);
  await dialog.locator('input').check();
  await dialog.locator('[value="confirm"]').click();
  assert.equal((await page.evaluate(() => confirmationResult)).optionSelected, true);
  await page.evaluate(() => { const drawer = document.createElement('dialog'); drawer.id = 'fixture-drawer'; drawer.innerHTML = '<button id="drawer-trigger">抽屉内操作</button>'; document.body.append(drawer); drawer.showModal(); document.getElementById('drawer-trigger').focus(); });
  await open({ title: '抽屉内确认？', message: '同一共享组件', tone: 'warning' });
  await dialog.waitFor();
  await page.mouse.click(2, 2);
  await dialog.waitFor({ state: 'detached' });
  assert.equal(await page.locator('#fixture-drawer').evaluate(node => node.open), true);
  assert.equal(await page.evaluate(() => document.activeElement.id), 'drawer-trigger');
  assert.deepEqual(errors, []);
  console.log('通过：全仓原生确认审计、真实业务系统上线/下线、2048/1440/420/320 宽度、原因校验、取消/重复确认、源文件选项、文本安全与焦点恢复。');
  console.log('截图：' + artifacts);
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
