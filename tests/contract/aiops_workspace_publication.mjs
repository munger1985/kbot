// 验证真实工作区脚本的草稿、监控发现与发布编排；请求全部由本地桩处理。
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';

const file = fileURLToPath(new URL('../../ui/aiops/js/aiops-business-system-workspace.js', import.meta.url));
const source = readFileSync(file, 'utf8');
const instrumented = source.replace(/\}\)\(\);\s*$/, `globalThis.workspaceTest = {
  saveDraft, commit, validateResource, refreshDiscoveryReferences, draftPayload,
  seed(value) {
    system = value.system; resources = value.resources; relations = value.relations || []; sources = value.sources || [];
    detail = value.detail || null; draftId = value.draftId || ''; draftRevision = value.draftRevision || 0;
    loading = false; step = 2; furthest = 2; dirty = true; workspaceReadOnly = false;
  },
  state() { return { system, resources, relations, draftId, draftRevision, saving, dirty, workspaceReadOnly }; },
};})();`);
assert.notEqual(source, instrumented);
const copy = value => JSON.parse(JSON.stringify(value));
const flush = () => new Promise(resolve => setImmediate(resolve));

function fixture(id = 'new') {
  const nodes = new Map(), calls = [], notifications = [], navigations = [];
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, { hidden: false, disabled: false, dataset: {}, style: {}, classList: { toggle() {} }, addEventListener() {}, querySelectorAll() { return []; }, querySelector() { return null; }, close() {}, append() {} });
    return nodes.get(id);
  };
  const location = { href: `http://fixture/ui/aiops/business-system-workspace.html?id=${id}&domainId=domain-42&view=edit&draft=old-draft` };
  Object.defineProperty(location, 'search', { get() { return new URL(location.href).search; } });
  location.replace = value => { const url = new URL(value, location.href); navigations.push(url); location.href = url.href; };
  let handler = async (path) => { throw new Error(`未定义的隔离请求：${path}`); };
  const context = {
    location, URL, URLSearchParams,
    history: { replaceState(_state, _unused, url) { location.href = new URL(url, location.href).href; } },
    document: { getElementById: node, querySelector: node, querySelectorAll() { return []; }, addEventListener() {} },
    KBotAIOpsShell: { escape: value => String(value ?? ''), toast: value => notifications.push(value), ready: new Promise(() => {}) },
    KBotAIOpsAuth: { request(path, options = {}) { const call = { path: path.replace('/api/v1/apps/aiops', ''), ...options }; if (call.body) call.payload = JSON.parse(call.body); calls.push(call); return handler(call); } },
    KBotAIOpsTopology: { dispose() {}, mount() {}, render() { return ''; } },
    crypto: { getRandomValues(bytes) { return bytes.fill(1); } }, addEventListener() {}, setTimeout() {},
  };
  runInNewContext(instrumented, context, { filename: file });
  return { api: context.workspaceTest, node, calls, notifications, navigations, location, handle(value) { handler = value; } };
}

const system = { display_name: 'MES 业务系统', environment: 'PROD', importance_level: 4, responsibility_group_id: 'group-1', description: '生产系统' };
const resource = (db_type = 'ORACLE', overrides = {}) => ({
  draft_resource_id: 'draft-resource-1', mode: 'NEW', resource_type_key: 'DATABASE', display_name: 'MES 主库', architecture_layer_key: 'DATA', owner_group_id: 'group-2', location_name: '生产机房', display_order: 0, saved: false,
  database: { db_type, version_code: { ORACLE: '19c', MYSQL: '8.4', POSTGRESQL: '16' }[db_type], environment: 'PROD', deployment_type: 'SINGLE_INSTANCE', rac_node_number: null, db_role: 'PRIMARY', host: 'db.example.invalid', port: db_type === 'ORACLE' ? 1521 : db_type === 'MYSQL' ? 3306 : 5432, database: 'MES', oracle_scope: 'PDB', oracle_pdb_name: 'MESPDB', tls_enabled: db_type !== 'ORACLE', username: 'fixture_readonly', password: 'fixture-only-not-a-real-secret', source_id: 'source-1', locator_key: 'mes-production', candidate_ref: '', host_candidate_ref: '', candidates: [], host_candidates: [], ...overrides },
});
const monitor = { source_id: 'source-1', source_type: 'OEM', status: 'ENABLED' };

// 草稿失败不能丢失页面内容；成功后下一次保存必须带服务端返回的 revision。
{
  const f = fixture();
  f.api.seed({ system: copy(system), resources: [resource()], sources: [monitor] });
  f.handle(async () => { throw new Error('隔离测试：草稿暂时不可用'); });
  await f.api.saveDraft();
  assert.equal(f.api.state().dirty, true);
  assert.equal(f.api.state().saving, false);
  assert.equal(f.api.state().resources[0].database.password, 'fixture-only-not-a-real-secret');
  f.handle(async call => ({ draft_id: 'draft-1', revision: call.method === 'PUT' ? 8 : 7 }));
  await f.api.saveDraft();
  assert.equal(f.calls.at(-1).method, 'POST');
  assert.equal(f.calls.at(-1).payload.mode, 'CREATE');
  assert.equal(f.calls.at(-1).payload.business_system_id, null);
  assert.equal(f.api.state().dirty, false);
  assert.equal(new URL(f.location.href).searchParams.get('domainId'), 'domain-42');
  assert.equal(new URL(f.location.href).searchParams.get('draft'), 'draft-1');
  await f.api.saveDraft();
  assert.equal(f.calls.at(-1).method, 'PUT');
  assert.equal(f.calls.at(-1).payload.revision, 7);
  assert.equal(f.api.state().draftRevision, 8);
}

// 发布只提交配置；不要求先测试连接，也不能在创建资源前丢失业务空间或留下旧草稿链接。
for (const engine of ['ORACLE', 'MYSQL', 'POSTGRESQL']) {
  const f = fixture();
  f.api.seed({ system: copy(system), resources: [resource(engine)], sources: [monitor] });
  let finish;
  f.handle(async call => {
    if (call.path === '/business-system-drafts') return { draft_id: 'draft-1', revision: 7 };
    if (call.path === '/business-system-drafts/draft-1:commit') return new Promise(resolve => { finish = resolve; });
    throw new Error(`不能发生额外写操作：${call.path}`);
  });
  const pending = f.api.commit();
  await flush();
  assert.equal(f.api.state().saving, true);
  assert.equal(f.node('commit').disabled, true);
  await f.api.commit();
  assert.equal(f.calls.filter(call => call.path.endsWith(':commit')).length, 1, '重复点击发布不能重复提交');
  const draft = f.calls[0].payload;
  assert.equal(draft.payload.publish.resources[0].database.target.endpoint[engine === 'ORACLE' ? 'service' : 'database'], 'MES');
  assert.equal(draft.payload.publish.resources[0].database.target.oracle_container_scope, engine === 'ORACLE' ? 'PDB' : null);
  assert.equal(draft.payload.publish.resources[0].database.target.controlled_change_enabled, false);
  assert.equal(draft.payload.publish.resources[0].database.target.diagnostic_credential.username, 'fixture_readonly');
  assert.equal(f.calls.at(-1).headers['If-Match'], '"rv-7"');
  assert.match(f.calls.at(-1).headers['Idempotency-Key'], /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  finish({ business_system_id: 'system-published', ...copy(system), status: 'ENABLED', resources: [], relations: [] });
  await pending;
  assert.equal(f.api.state().dirty, false);
  assert.equal(f.navigations.length, 1);
  const url = f.navigations[0];
  assert.equal(url.searchParams.get('id'), 'system-published');
  assert.equal(url.searchParams.get('domainId'), 'domain-42');
  assert.equal(url.searchParams.has('draft'), false);
}

// 创建发布失败时保留已保存草稿，重试更新同一草稿，不重复创建业务系统配置。
{
  const f = fixture();
  f.api.seed({ system: copy(system), resources: [resource()], sources: [monitor] });
  f.handle(async call => {
    if (call.path === '/business-system-drafts') return { draft_id: 'draft-1', revision: 7 };
    if (call.path === '/business-system-drafts/draft-1:commit') throw new Error('隔离测试：发布暂时失败');
    throw new Error(`未定义的隔离请求：${call.path}`);
  });
  await f.api.commit();
  assert.equal(f.api.state().draftId, 'draft-1');
  assert.equal(f.api.state().saving, false);
  assert.equal(f.navigations.length, 0);
  assert.equal(new URL(f.location.href).searchParams.get('draft'), 'draft-1');
  assert.match(f.notifications.at(-1), /发布暂时失败/);
  f.handle(async call => {
    if (call.method === 'PUT' && call.path === '/business-system-drafts/draft-1') return { draft_id: 'draft-1', revision: 8 };
    if (call.path === '/business-system-drafts/draft-1:commit') return { business_system_id: 'system-published' };
    throw new Error(`未定义的隔离请求：${call.path}`);
  });
  await f.api.commit();
  assert.equal(f.calls.filter(call => call.path === '/business-system-drafts' && call.method === 'POST').length, 1);
  assert.equal(f.calls.at(-1).headers['If-Match'], '"rv-8"');
  assert.equal(f.navigations.length, 1);
}

// 编辑发布按各自 ETag 写入资源和采集策略，成功后清除服务端草稿及页面草稿参数。
{
  const f = fixture('system-1'), item = resource();
  Object.assign(item, { mode: 'EXISTING', resource_id: 'resource-1', system_resource_id: 'system-resource-1', resource_row_version: 4, system_row_version: 5, saved: true });
  item.workload_snapshot = { target_id: 'target-1', target_row_version: 8, policy: { enabled: true, collection_mode: 'VISIBLE_SET', visible_statement_limit: 250, interval_minutes: 5, retention_days: 30, top_statement_limit: 100 } };
  const detail = { business_system_id: 'system-1', system_code: 'MES', ...copy(system), status: 'DISABLED', row_version: 3, resources: [{ system_resource_id: 'system-resource-1', architecture_layer_key: 'DATA', status: 'ACTIVE', row_version: 5, resource: { resource_id: 'resource-1', row_version: 4, display_name: item.display_name, resource_type_key: 'DATABASE', owner_group_id: item.owner_group_id, database: { ...copy(item.database), target_id: 'target-1', endpoint: { host: 'db.example.invalid', port: 1521, service: 'MES' } } } }], relations: [] };
  f.api.seed({ system: copy(system), resources: [item], sources: [monitor], detail, draftId: 'edit-draft', draftRevision: 11 });
  f.handle(async call => call.path.includes('/business-systems/system-1') ? copy(detail) : {});
  await f.api.commit();
  const patch = f.calls.find(call => call.method === 'PATCH' && call.path === '/managed-resources/resource-1');
  assert.equal(patch.headers['If-Match'], '"rv-4"');
  assert.deepEqual(patch.payload.database.workload_snapshot_policy, item.workload_snapshot.policy);
  assert.equal(patch.payload.database.target_row_version, 8);
  assert.equal(patch.payload.database.business_system_id, 'system-1');
  assert.equal(patch.payload.database.system_resource_id, 'system-resource-1');
  assert.equal(patch.payload.database.diagnostic_credential, undefined);
  const deletion = f.calls.find(call => call.method === 'DELETE');
  assert.equal(deletion.path, '/business-system-drafts/edit-draft');
  assert.equal(deletion.headers['If-Match'], '"rv-11"');
  assert.equal(f.api.state().draftId, '');
  assert.equal(f.api.state().dirty, false);
  assert.equal(new URL(f.location.href).searchParams.has('draft'), false);
  assert.equal(new URL(f.location.href).searchParams.get('domainId'), 'domain-42');
}

// 发布前按稳定定位键刷新监控引用，不能提交失效的候选引用，也不能猜测替代候选。
{
  const f = fixture(), item = resource('MYSQL', {
    candidate_ref: 'old-database-reference', host_candidate_ref: 'old-host-reference',
    candidates: [{ candidate_ref: 'old-database-reference', locator_key: 'mysql-main' }],
    host_candidates: [{ candidate_ref: 'old-host-reference', locator_key: 'host-main' }],
  });
  f.api.seed({ system: copy(system), resources: [item], sources: [{ ...monitor, source_type: 'PROMETHEUS' }] });
  f.handle(async () => ({ items: [{ candidate_ref: 'new-database-reference', locator_key: 'mysql-main', display_name: '生产数据库' }], host_items: [{ candidate_ref: 'new-host-reference', locator_key: 'host-main' }] }));
  await f.api.refreshDiscoveryReferences();
  assert.equal(item.database.candidate_ref, 'new-database-reference');
  assert.equal(item.database.host_candidate_ref, 'new-host-reference');
  f.handle(async () => ({ items: [], host_items: [] }));
  await assert.rejects(f.api.refreshDiscoveryReferences(), /原数据库监控 Target 已不存在/);
}

console.log('通过：草稿失败恢复与 revision、三引擎发布载荷、发布互斥、上下文与草稿 URL、监控候选稳定定位合同。');
