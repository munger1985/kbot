// 验证两种前端共用的硬件绑定与显式关系语义，不连接真实监控源。
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
let id = 0;
const uuid = () => 'id-' + (++id);
const host = hardware.create('PHYSICAL_SERVER', 'INFRASTRUCTURE', 'ops', uuid);
host.display_name = '主机 01';
assert.equal(hardware.error(host), '');
host.hardware.source_id = 'prometheus';
assert.match(hardware.error(host), /候选/);
host.hardware.locator_key = 'host-01';
const pages = [];
await hardware.discover(host, async (_path, options) => {
  const body = JSON.parse(options.body);
  pages.push(body);
  return body.cursor ? { host_items: [{ candidate_ref: 'fresh-host', locator_key: 'host-01' }], has_more: false }
    : { host_items: [], has_more: true, next_cursor: 'page-2' };
});
assert.deepEqual(pages.map(page => page.resource_scope), ['HOST', 'HOST']);
assert.equal(host.hardware.candidate_ref, 'fresh-host');
assert.deepEqual(copy(hardware.mapping(host)), { source_id: 'prometheus', candidate_ref: 'fresh-host' });
assert.equal(hardware.error(host), '');
const db1 = { draft_resource_id: 'db1', display_name: '主库' };
const db2 = { draft_resource_id: 'db2', display_name: '从库' };
const resources = [db1, db2, host];
let relations = hardware.setLinked(host, ['db1', 'db2'], resources, [], uuid);
assert.equal(relations.length, 2);
assert.ok(relations.every(row => row.relation_type_key === 'RUNS_ON'));
const retainedId = relations[0].draft_relation_id;
relations = hardware.setLinked(host, ['db1'], resources, relations, uuid);
assert.equal(relations.length, 1);
assert.equal(relations[0].draft_relation_id, retainedId);
assert.deepEqual(copy(hardware.linked(host, relations)), ['db1']);
const saved = { binding_id: 'binding', row_version: 7, source_id: 'prometheus', source_locator_key: 'host-01', status: 'ACTIVE' };
host.hardware = hardware.draft([saved]);
assert.equal(hardware.changed(host), false);
assert.equal(hardware.mapping(host, true), undefined);
host.hardware.locator_key = 'host-02';
assert.match(hardware.error(host), /重新发现/);
host.hardware.candidate_ref = 'new-host';
assert.equal(hardware.mapping(host, true).binding_row_version, 7);
assert.equal(hardware.mapping(host, true).binding_id, 'binding');
host.hardware.source_id = '';
assert.match(hardware.error(host), /不能隐式移除/);
assert.deepEqual(copy(hardware.relationSpec({ resource_type_key: 'STORAGE' })), ['USES_STORAGE', 'CONSUMER', 'PROVIDER']);
assert.deepEqual(copy(hardware.relationSpec({ resource_type_key: 'NETWORK_DEVICE' })), ['DEPENDS_ON', 'SOURCE', 'TARGET']);
host.hardware = hardware.draft();
host.hardware.source_id = 'first';
let release;
const pending = hardware.discover(host, () => new Promise(resolve => { release = resolve; }));
host.hardware.source_id = 'second';
host.hardware.revision++;
release({ host_items: [{ candidate_ref: 'stale', locator_key: 'first' }], has_more: false });
await pending;
assert.equal(host.hardware.candidates.length, 0);
console.log('通过：独立主机发现分页、草稿引用刷新、映射版本、显式多数据库关系、存储与网络关系及过期响应隔离。');
