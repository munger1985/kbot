// 使用两端真实硬件模块验证 Exadata 短期引用刷新；不连接 OEM 或写入业务数据。
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';

const root = fileURLToPath(new URL('../../', import.meta.url));
const kbot = resolve(root, 'ui/aiops/js/aiops-hardware-resources.js');
const context = { exports: {} };
if (existsSync(kbot)) {
  runInNewContext(readFileSync(kbot, 'utf8'), context);
} else {
  const require = createRequire(resolve(root, 'apps/portal-web/package.json'));
  const ts = require('typescript');
  const source = readFileSync(resolve(root, 'apps/portal-web/src/features/aiops/management/hardwareResources.ts'), 'utf8');
  runInNewContext(ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText, context);
}
const hardware = context.KBotAIOpsHardware || context.exports.default;
const copy = value => JSON.parse(JSON.stringify(value));
const guid = 'a'.repeat(32);
const cell = { target_id: 'b'.repeat(32), target_type: 'oracle_exadata', display_name: 'Cell 01',
  parent_target_id: guid, membership_origin: 'OEM_ASSOCIATION', evidence_ref: null };
const host = { ...cell, target_id: 'c'.repeat(32), target_type: 'host', display_name: 'DB Server 01' };
const candidate = (changes = {}) => ({ locator_key: guid, candidate_ref: 'new-ref',
  display_name: '测试 Exadata', members: [copy(cell), copy(host)], ...changes });
function selected() {
  const item = hardware.create('EXADATA', 'INFRASTRUCTURE', 'ops', () => 'draft-1');
  item.display_name = '测试 Exadata';
  Object.assign(item.hardware, { source_id: 'oem', candidate_ref: 'old-ref', locator_key: guid,
    candidates: [candidate({ candidate_ref: 'old-ref' })] });
  return item;
}
const options = { requireSelection: true };
let passed = 0;
async function test(name, run) {
  await run();
  passed++;
  console.log('通过：' + name);
}

await test('同一整机、同一成员刷新后可取得新引用，且仅查询原 GUID', async () => {
  const item = selected();
  await hardware.discover(item, async (path, init) => {
    assert.ok(path.endsWith('/exadata-discoveries'));
    assert.equal(JSON.parse(init.body).root_id, guid);
    return { items: [candidate()] };
  }, options);
  assert.equal(item.hardware.candidate_ref, 'new-ref');
  assert.equal(hardware.error(item), '');
  assert.equal(hardware.mapping(item).candidate_ref, 'new-ref');
});
await test('成员及字段顺序变化不算成员归属变化', async () => {
  const item = selected();
  const reordered = [host, cell].map(row => Object.fromEntries(Object.entries(row).reverse()));
  await hardware.discover(item, async () => ({ items: [candidate({ members: reordered })] }), options);
  assert.equal(item.hardware.candidate_ref, 'new-ref');
});
for (const [name, mutate] of [
  ['成员新增', rows => [...rows, { ...host, target_id: 'd'.repeat(32) }]],
  ['成员删除', rows => rows.slice(1)],
  ['成员 GUID', rows => [{ ...rows[0], target_id: 'd'.repeat(32) }, rows[1]]],
  ['成员类型', rows => [{ ...rows[0], target_type: 'host' }, rows[1]]],
  ['成员名称', rows => [{ ...rows[0], display_name: '另一设备' }, rows[1]]],
  ['父节点归属', rows => [{ ...rows[0], parent_target_id: 'd'.repeat(32) }, rows[1]]],
  ['成员归属来源', rows => [{ ...rows[0], membership_origin: 'IMPORT_CONFIRMED' }, rows[1]]],
  ['确认凭据', rows => [{ ...rows[0], evidence_ref: 'changed' }, rows[1]]],
]) {
  await test(name + '变化拒绝继续提交并清除旧引用', async () => {
    const item = selected();
    await assert.rejects(hardware.discover(item, async () => ({ items: [candidate({ members: mutate([copy(cell), copy(host)]) })] }), options), /成员.*变化/);
    assert.equal(item.hardware.candidate_ref, '');
    assert.equal(item.hardware.locator_key, guid);
  });
}
for (const [name, items] of [
  ['整机消失', []],
  ['同名但不同整机', [candidate({ locator_key: 'e'.repeat(32) })]],
  ['整机 GUID 重复', [candidate(), candidate({ candidate_ref: 'another-ref' })]],
]) {
  await test(name + '不得自动换绑', async () => {
    const item = selected();
    await assert.rejects(hardware.discover(item, async () => ({ items }), options), /整机候选.*不存在|整机候选.*不唯一/);
    assert.equal(item.hardware.candidate_ref, '');
  });
}
await test('候选缺少成员证据时不得续签', async () => {
  const item = selected();
  delete item.hardware.candidates[0].members;
  await assert.rejects(hardware.discover(item, async () => ({ items: [candidate()] }), options), /重新发现.*确认/);
  assert.equal(item.hardware.candidate_ref, '');
});
await test('未明确选择整机不得自动选择第一项', async () => {
  const item = selected();
  item.hardware.candidate_ref = '';
  await assert.rejects(hardware.discover(item, async () => ({ items: [candidate()] }), options), /重新发现.*确认/);
  assert.equal(item.hardware.candidate_ref, '');
});
await test('显式导入清单原样用于复核及发布', async () => {
  const item = selected();
  const manifest = [{ ...host, evidence_ref: 'operator-confirmed' }];
  delete manifest[0].membership_origin;
  item.hardware.component_manifest_text = JSON.stringify(manifest);
  item.hardware.confirmed_manifest = copy(manifest);
  await hardware.discover(item, async (_path, init) => {
    assert.deepEqual(JSON.parse(init.body), { root_id: guid, component_manifest: manifest });
    return { items: [candidate()] };
  }, options);
  assert.deepEqual(copy(hardware.mapping(item).component_manifest), manifest);
});
await test('导入清单编辑后必须由用户重新发现确认', async () => {
  const item = selected();
  item.hardware.component_manifest_text = JSON.stringify([{ ...host, evidence_ref: 'new-proof' }]);
  await assert.rejects(hardware.discover(item, async () => ({ items: [candidate()] }), options), /组件清单.*变化/);
  assert.equal(item.hardware.candidate_ref, '');
});
await test('发现失败清除旧引用，不吞掉上游错误', async () => {
  const item = selected();
  await assert.rejects(hardware.discover(item, async () => { throw new Error('模拟 OEM 不可达'); }, options), /OEM 不可达/);
  assert.equal(item.hardware.candidate_ref, '');
  assert.equal(item.hardware.discovering, false);
});
for (const [name, mutate] of [
  ['监控源', state => { state.source_id = 'another-oem'; state.revision++; }],
  ['选定整机', state => { state.locator_key = 'e'.repeat(32); }],
  ['导入清单', state => { state.component_manifest_text = '[{}]'; }],
  ['并发发现', state => { state.revision++; }],
  ['用户重新选择候选', state => { state.candidate_ref = 'user-selection'; }],
]) {
  await test('请求期间' + name + '变化不得沿用过期响应提交', async () => {
    const item = selected();
    let release;
    const pending = hardware.discover(item, () => new Promise(resolve => { release = resolve; }), options);
    mutate(item.hardware);
    release({ items: [candidate()] });
    await assert.rejects(pending, /配置已变化/);
    assert.notEqual(item.hardware.candidate_ref, 'new-ref');
  });
}
await test('手动重新发现保留候选展示但不静默确认新成员', async () => {
  const item = selected();
  await hardware.discover(item, async () => ({ items: [candidate({ members: [copy(cell)] })] }));
  assert.equal(item.hardware.candidate_ref, '');
  assert.equal(item.hardware.candidates.length, 1);
});
console.log('Exadata 候选刷新回归通过：' + passed + ' 项。');
