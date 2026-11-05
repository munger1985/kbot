// 使用真实布局代码验证单实例、多实例、共享主机与 RAC + DG；不访问线上服务。
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { runInNewContext } from 'node:vm';

const root = resolve(import.meta.dirname, '../..');
const context = {};
if (existsSync(resolve(root, 'ui/aiops/js/aiops-topology-layout.js'))) {
  for (const name of ['layout', 'view']) runInNewContext(readFileSync(resolve(root, `ui/aiops/js/aiops-topology-${name}.js`), 'utf8'), context);
} else {
  const require = createRequire(resolve(root, 'apps/portal-web/package.json'));
  const ts = require('typescript');
  for (const [name, global] of [['topologyLayout', 'KBotAIOpsTopologyLayout'], ['topologyView', 'KBotAIOpsTopologyView']]) {
    const exports = {}, source = readFileSync(resolve(root, `apps/portal-web/src/features/aiops/management/${name}.ts`), 'utf8');
    runInNewContext(ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText, { exports, require: () => ({ default: context.KBotAIOpsTopologyLayout }) });
    context[global] = exports.default;
  }
}
const { graphModel } = context.KBotAIOpsTopologyLayout;
const { renderTopology, createState } = context.KBotAIOpsTopologyView;
const resource = (id, type = 'DATABASE', location = '') => ({ draft_resource_id: id, resource_type_key: type, display_name: id, location_name: location, saved: true, ...(type === 'DATABASE' ? { database: { db_type: 'ORACLE', version_code: '19c', deployment_type: 'RAC', db_role: id.startsWith('dr') ? 'STANDBY' : 'PRIMARY' } } : {}) });
const relation = (id, type, pairs) => ({ draft_relation_id: id, relation_type_key: type, display_name: id, participants: pairs.map(([id, role], index) => ({ draft_resource_id: id, participant_role: role, display_order: index })) });
const runs = (db, host) => relation(`on-${db}-${host}`, 'RUNS_ON', [[db, 'WORKLOAD'], [host, 'HOST']]);
function uniqueNodes(model, ids) {
  assert.equal(model.resourceNodes.length, ids.length);
  assert.equal(new Set(model.resourceNodes.map(node => node.resource.draft_resource_id)).size, ids.length);
  assert.ok(Number.isFinite(model.width) && Number.isFinite(model.height));
}
let model = graphModel([resource('db'), resource('host', 'VIRTUAL_MACHINE')], [runs('db', 'host')]);
uniqueNodes(model, ['db', 'host']);
assert.equal(model.boxes.filter(box => box.kind === 'host').length, 1);
assert.equal(model.edges[0].contained, true);
assert.equal(model.boxes.filter(box => box.kind === 'site').length, 0);
assert.ok(model.resourceNodes.find(node => node.resource.draft_resource_id === 'db').y > model.resourceNodes.find(node => node.resource.draft_resource_id === 'host').y);
model = graphModel([resource('db1'), resource('db2'), resource('host', 'PHYSICAL_SERVER')], [runs('db1', 'host'), runs('db2', 'host')]);
uniqueNodes(model, ['db1', 'db2', 'host']);
assert.equal(model.edges.filter(edge => edge.contained).length, 2);
const ids = ['db1', 'db2', 'dr1', 'dr2'];
const resources = ids.map(id => resource(id, 'DATABASE', id.startsWith('dr') ? '机房 B' : '机房 A'));
resources.push(...ids.map(id => resource(`host-${id}`, 'PHYSICAL_SERVER', id.startsWith('dr') ? '机房 B' : '机房 A')));
const relations = [relation('rac-a', 'ORACLE_RAC', [['db1', 'MEMBER'], ['db2', 'MEMBER']]), relation('rac-b', 'ORACLE_RAC', [['dr1', 'MEMBER'], ['dr2', 'MEMBER']]), relation('dg', 'DATA_GUARD', [['db1', 'PRIMARY'], ['db2', 'PRIMARY'], ['dr1', 'STANDBY'], ['dr2', 'STANDBY']]), ...ids.map(id => runs(id, `host-${id}`))];
model = graphModel(resources, relations);
uniqueNodes(model, resources);
assert.equal(model.boxes.filter(box => box.kind === 'cluster').length, 2);
assert.equal(model.boxes.filter(box => box.kind === 'site').length, 2);
const dg = model.edges.find(edge => edge.relation.draft_relation_id === 'dg');
assert.equal(dg.paths.length, 1, 'DG 端点按数据库集群去重，不能生成四条实例间复制线');
assert.equal(dg.kind, 'replication');
assert.equal(model.edges.filter(edge => edge.contained).length, 4);
const sharedResources = [...resources.filter(resource => !resource.draft_resource_id.startsWith('host-')), resource('shared', 'PHYSICAL_SERVER')];
const sharedRelations = [...relations.filter(relation => relation.relation_type_key !== 'RUNS_ON'), ...ids.map(id => runs(id, 'shared'))];
model = graphModel(sharedResources, sharedRelations);
uniqueNodes(model, sharedResources);
assert.equal(model.edges.filter(edge => edge.contained).length, 0, '跨集群共享主机不能被强制归入任一 RAC');
model = graphModel(sharedResources, sharedRelations, 'host');
uniqueNodes(model, sharedResources);
assert.equal(model.edges.filter(edge => edge.contained).length, 4);
assert.equal(model.boxes.filter(box => box.kind === 'host').length, 1);
assert.equal(model.boxes.filter(box => box.kind === 'cluster').length, 2, '主机视角保留 RAC 逻辑摘要供 DG 连接');
const cycleResources = [resource('h1', 'VIRTUAL_MACHINE'), resource('h2', 'CONTAINER')];
model = graphModel(cycleResources, [runs('h1', 'h2'), runs('h2', 'h1')]);
uniqueNodes(model, cycleResources);
model = graphModel([resource('db'), resource('h1', 'VIRTUAL_MACHINE'), resource('h2', 'VIRTUAL_MACHINE')], [runs('db', 'h1'), runs('db', 'h2')]);
uniqueNodes(model, ['db', 'h1', 'h2']);
assert.equal(model.edges.filter(edge => edge.contained).length, 0);
const malicious = resource('unsafe'); malicious.display_name = '<img src=x onerror=alert(1)>';
const html = renderTopology({ systemName: '<script>alert(1)</script>', resources: [malicious], relations: [] }, createState());
assert.ok(!html.includes('<script>') && !html.includes('<img'));
assert.ok(html.includes('&lt;img'));
const projection = context.KBotAIOpsTopologyLayout.hostProjection({ resources: [{ resource_id: 'shared', resource_type_key: 'PHYSICAL_SERVER', display_name: '共享主机', business_systems: [{ display_name: '系统 A' }, { display_name: '系统 B' }] }], relations: [] });
assert.equal(projection.resources[0].business_system_names.join(' / '), '系统 A / 系统 B');
console.log('通过：单实例、多数据库、RAC + DG、真实位置、共享主机双视角、循环/多承载防重复与 HTML 转义。');
