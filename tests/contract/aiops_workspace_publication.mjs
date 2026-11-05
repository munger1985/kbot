// 验证真实工作区脚本的草稿、监控发现与发布编排；请求全部由本地桩处理。
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';

const file = fileURLToPath(new URL('../../ui/aiops/js/aiops-business-system-workspace.js', import.meta.url));
const source = readFileSync(file, 'utf8');
const instrumented = source.replace(/\}\)\(\);\s*$/, `globalThis.workspaceTest = {
  saveDraft, commit, validateResource, refreshDiscoveryReferences, draftPayload, loadDiscoveryCandidates, monitoringPayload, usesDiscovery, monitoringDraft, monitoringChanged, existingDatabasePatch, selectMonitoringBinding, resourceEditorHtml, updateResourceField,
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
  runInNewContext(readFileSync(new URL("../../ui/aiops/js/aiops-hardware-resources.js", import.meta.url), "utf8"), context);
  runInNewContext(instrumented, context, { filename: file });
  return { api: context.workspaceTest, node, calls, notifications, navigations, location, handle(value) { handler = value; } };
}

const system = { display_name: 'MES 业务系统', environment: 'PROD', importance_level: 4, responsibility_group_id: 'group-1', description: '生产系统' };
const resource = (db_type = 'ORACLE', overrides = {}) => ({
  draft_resource_id: 'draft-resource-1', mode: 'NEW', resource_type_key: 'DATABASE', display_name: 'MES 主库', architecture_layer_key: 'DATA', owner_group_id: 'group-2', location_name: '生产机房', display_order: 0, saved: false,
  database: { db_type, version_code: { ORACLE: '19c', MYSQL: '8.4', POSTGRESQL: '16' }[db_type], environment: 'PROD', deployment_type: 'SINGLE_INSTANCE', rac_node_number: null, db_role: 'PRIMARY', host: 'db.example.invalid', port: db_type === 'ORACLE' ? 1521 : db_type === 'MYSQL' ? 3306 : 5432, database: 'MES', oracle_scope: 'PDB', oracle_pdb_name: 'MESPDB', tls_enabled: db_type !== 'ORACLE', username: 'fixture_readonly', password: 'fixture-only-not-a-real-secret', source_id: 'source-1', locator_key: 'mes-production', candidate_ref: '', host_candidate_ref: '', candidates: [], host_candidates: [], ...overrides },
});
const monitor = { source_id: 'source-1', source_type: 'ALERTMANAGER', status: 'ENABLED' };

// OEM 候选只显示可读名称；新增、编辑及 RAC 仍保留原候选引用和稳定定位键。
for (const sourceType of ['OEM', 'PROMETHEUS', 'ZABBIX']) {
  for (const mode of ['NEW', 'EXISTING']) {
    for (const deployment of ['SINGLE_INSTANCE', 'RAC']) {
      const f = fixture(mode === 'NEW' ? 'new' : 'system-1');
      const candidate = { candidate_ref: 'signed-database-reference', locator_key: '44bc8bf5d569de338cce77db1db5cb1c', display_name: 'ERP · CDB_ROOT · ORCLCDB · 19.3.0.0.0 · oracle19c.lab' };
      const item = resource('ORACLE', { deployment_type: deployment, candidate_ref: candidate.candidate_ref, candidate_locator_key: candidate.locator_key, candidates: [candidate] });
      item.mode = mode;
      if (mode === 'EXISTING') item.system_resource_id = 'system-resource-1';
      const members = [item];
      if (deployment === 'RAC') {
        item.rac_group_id = 'rac-1';
        item.database.rac_node_number = 1;
        const other = copy(item);
        other.draft_resource_id = 'draft-resource-2';
        other.database.rac_node_number = 2;
        members.push(other);
      }
      f.api.seed({ system: copy(system), resources: members, sources: [{ ...monitor, source_type: sourceType }] });
      const before = copy(item.database);
      const html = f.api.resourceEditorHtml(item);
      const selects = [...html.matchAll(/<select\b[^>]*(?:name="candidate_ref"|data-rac-field="candidate_ref")[^>]*>([\s\S]*?)<\/select>/g)];
      assert.equal(selects.length, mode === 'NEW' && deployment === 'RAC' ? 2 : 1);
      for (const [, options] of selects) {
        assert.ok(options.includes(`value="${candidate.candidate_ref}" selected`));
        assert.ok(options.includes(candidate.display_name));
        assert.equal(options.includes(candidate.locator_key), sourceType !== 'OEM');
      }
      assert.deepEqual(copy(item.database), before);
      assert.equal(f.api.monitoringPayload(item).candidate_ref, candidate.candidate_ref);
    }
  }
}

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
  assert.equal(f.navigations.length, 0, '仅保存草稿必须保留编辑页');
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
  assert.equal(f.navigations.length, 0, '发布请求尚未完成不能返回列表');
  await f.api.commit();
  assert.equal(f.calls.filter(call => call.path.endsWith(':commit')).length, 1, '重复点击发布不能重复提交');
  const draft = f.calls[0].payload;
  assert.equal(draft.payload.publish.resources[0].database.target.endpoint[engine === 'ORACLE' ? 'service' : 'database'], 'MES');
  assert.equal(draft.payload.publish.resources[0].database.target.oracle_container_scope, engine === 'ORACLE' ? 'PDB' : null);
  assert.equal(draft.payload.publish.resources[0].database.target.controlled_change_enabled, false);
  assert.deepEqual(draft.payload.publish.resources[0].database.target.workload_snapshot_policy, {
    enabled: false, interval_minutes: 1,
    retention_days: 30, top_statement_limit: 100, round_timeout_seconds: 30,
  });
  assert.equal(draft.payload.publish.resources[0].database.target.diagnostic_credential.username, 'fixture_readonly');
  assert.equal(f.calls.at(-1).headers['If-Match'], '"rv-7"');
  assert.match(f.calls.at(-1).headers['Idempotency-Key'], /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  finish({ business_system_id: 'system-published', ...copy(system), status: 'ENABLED', resources: [], relations: [] });
  await pending;
  assert.equal(f.api.state().dirty, false);
  assert.equal(f.navigations.length, 1);
  const url = f.navigations[0];
  assert.equal(url.pathname, '/ui/aiops/business-systems.html');
  assert.equal(url.searchParams.has('id'), false);
  assert.equal(url.searchParams.get('domainId'), 'domain-42');
  assert.equal(url.searchParams.has('draft'), false);
  assert.equal(url.searchParams.has('view'), false);
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
  item.workload_snapshot = { target_id: 'target-1', target_row_version: 8, policy: { enabled: true, interval_minutes: 1, retention_days: 30, top_statement_limit: 100, round_timeout_seconds: 30, collection_mode: 'VISIBLE_SET', visible_statement_limit: 250 } };
  const detail = { business_system_id: 'system-1', system_code: 'MES', ...copy(system), status: 'DISABLED', row_version: 3, resources: [{ system_resource_id: 'system-resource-1', architecture_layer_key: 'DATA', status: 'ACTIVE', row_version: 5, resource: { resource_id: 'resource-1', row_version: 4, display_name: item.display_name, resource_type_key: 'DATABASE', owner_group_id: item.owner_group_id, database: { ...copy(item.database), target_id: 'target-1', endpoint: { host: 'db.example.invalid', port: 1521, service: 'MES' } } } }], relations: [] };
  f.api.seed({ system: copy(system), resources: [item], sources: [monitor], detail, draftId: 'edit-draft', draftRevision: 11 });
  let finishCleanup;
  f.handle(async call => {
    if (call.method === 'DELETE') return new Promise(resolve => { finishCleanup = resolve; });
    return call.path.includes('/business-systems/system-1') ? copy(detail) : {};
  });
  const pending = f.api.commit();
  await flush();
  assert.equal(typeof finishCleanup, 'function', f.notifications.join('；'));
  assert.equal(f.navigations.length, 0, '编辑保存及草稿清理全部成功后才能返回列表');
  finishCleanup({});
  await pending;
  const patch = f.calls.find(call => call.method === 'PATCH' && call.path === '/managed-resources/resource-1');
  assert.equal(patch.headers['If-Match'], '"rv-4"');
  assert.equal(patch.payload.database.workload_snapshot_policy.collection_mode, undefined);
  assert.equal(patch.payload.database.workload_snapshot_policy.visible_statement_limit, undefined);
  assert.deepEqual(patch.payload.database.workload_snapshot_policy, { enabled: true, interval_minutes: 1, retention_days: 30, top_statement_limit: 100, round_timeout_seconds: 30 });
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
  assert.equal(f.navigations.length, 1);
  assert.equal(f.navigations[0].pathname, '/ui/aiops/business-systems.html');
  assert.equal(f.navigations[0].searchParams.has('id'), false);
  assert.equal(f.navigations[0].searchParams.has('view'), false);
}

// 编辑保存或草稿清理失败时不得离开工作区，也不能丢失重试所需的编辑内容与草稿版本。
for (const failure of ['PATCH', 'DELETE']) {
  const f = fixture('system-1');
  const detail = { business_system_id: 'system-1', ...copy(system), status: 'DISABLED', row_version: 3, resources: [], relations: [] };
  const item = resource();
  f.api.seed({ system: copy(system), resources: [item], sources: [monitor], detail, draftId: 'edit-draft', draftRevision: 11 });
  f.handle(async call => {
    if (call.method === failure) throw new Error(`隔离测试：${failure} 失败`);
    if (call.path.endsWith('/resources')) return { ...copy(detail), resources: [{ system_resource_id: 'sr-1', row_version: 1, resource: { resource_id: 'resource-1', row_version: 1 } }] };
    return copy(detail);
  });
  await f.api.commit();
  assert.equal(f.navigations.length, 0);
  assert.equal(f.api.state().draftId, 'edit-draft');
  assert.equal(f.api.state().draftRevision, 11);
  assert.equal(f.api.state().dirty, true);
  assert.equal(f.api.state().saving, false);
  assert.equal(f.api.state().system.display_name, system.display_name);
  assert.match(f.notifications.at(-1), new RegExp(`${failure} 失败`));
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
  assert.equal(Object.hasOwn(f.api.monitoringPayload(item), 'host_candidate_ref'), false);
  f.handle(async () => ({ items: [], host_items: [] }));
  await assert.rejects(f.api.refreshDiscoveryReferences(), /原数据库监控 Target 已不存在/);
}

// 未发布资源的采集设置同时保留在可编辑草稿和正式创建载荷中。
for (const engine of ['ORACLE', 'MYSQL', 'POSTGRESQL']) {
  const f = fixture();
  const selected = { enabled: true, interval_minutes: 2,
    retention_days: 60, top_statement_limit: 80, round_timeout_seconds: 47 };
  f.api.seed({ system: copy(system), resources: [resource(engine, { workload_snapshot_policy: selected })], sources: [monitor] });
  f.handle(async call => call.path.endsWith(':commit') ? { business_system_id: 'published' } : { draft_id: 'draft-1', revision: 7 });
  await f.api.commit();
  assert.equal(f.calls.filter(call => call.path.endsWith(':commit')).length, 1);
  const body = f.calls[0].payload.payload;
  assert.deepEqual(body.resources[0].database.workload_snapshot_policy, selected);
  assert.deepEqual(body.publish.resources[0].database.target.workload_snapshot_policy, selected);
  assert.equal(f.calls.some(call => call.path.startsWith('/targets/')), false);
}
for (const policy of [{ enabled: true, interval_minutes: 0 }, { top_statement_limit: 101 }, { enabled: 'true' }]) {
  const f = fixture();
  f.api.seed({ system: copy(system), resources: [resource('ORACLE', { workload_snapshot_policy: policy })], sources: [monitor] });
  await f.api.commit();
  assert.equal(f.calls.length, 0, '无效采集策略不能进入发布写操作');
  assert.ok(f.notifications.some(value => value.includes('历史 SQL 采集策略无效')));
}
// OEM 使用服务级 Source，数据库在业务系统中选取；发布前刷新引用，不要求主机 Label。
{
  const f = fixture(), guid = '1'.repeat(32);
  const item = resource('ORACLE', {
    candidate_ref: 'expired-ref', candidate_locator_key: guid,
    candidates: [{ candidate_ref: 'expired-ref', locator_key: guid }],
  });
  f.api.seed({ system: copy(system), resources: [item], sources: [{ ...monitor, source_type: 'OEM' }] });
  let discoveries = 0;
  f.handle(async call => {
    if (call.path.endsWith('/instance-discoveries')) {
      discoveries++;
      assert.deepEqual(call.payload.db_types, ['ORACLE']);
      assert.equal(call.payload.source_locator_key, undefined);
      if (!call.payload.cursor) return { items: [], host_items: [], has_more: true, next_cursor: 'page-2' };
      return { items: [{ candidate_ref: 'fresh-ref', locator_key: guid, display_name: 'CDB / PDB' }], host_items: [], has_more: false };
    }
    if (call.path === '/business-system-drafts') return { draft_id: 'draft-oem', revision: 1 };
    if (call.path.endsWith(':commit')) return { business_system_id: 'published-oem' };
    throw new Error('不允许额外请求：' + call.path);
  });
  assert.equal(f.api.usesDiscovery(item), true);
  await f.api.commit();
  assert.equal(discoveries, 2);
  const saved = f.calls.find(call => call.path === '/business-system-drafts');
  assert.ok(saved);
  const binding = saved.payload.payload.publish.resources[0].database.monitoring;
  assert.equal(binding.candidate_ref, 'fresh-ref');
  assert.equal(Object.hasOwn(binding, "host_candidate_ref"), false);
  assert.equal(binding.source_binding, undefined);
  assert.equal(binding.source_locator_key, undefined);
}
{
  const f = fixture(), item = resource('ORACLE', {
    candidate_ref: 'old-ref', candidate_locator_key: '1'.repeat(32),
    candidates: [{ candidate_ref: 'old-ref', locator_key: '1'.repeat(32), display_name: 'same-name' }],
  });
  f.api.seed({ system: copy(system), resources: [item], sources: [{ ...monitor, source_type: 'OEM' }] });
  f.handle(async () => ({ items: [{ candidate_ref: 'replacement', locator_key: '2'.repeat(32), display_name: 'same-name' }], host_items: [], has_more: false }));
  await f.api.commit();
  assert.equal(f.calls.some(call => !call.path.endsWith('/instance-discoveries')), false);
  assert.match(f.notifications.at(-1), /原数据库监控 Target 已不存在/);
}
{
  const f = fixture();
  f.handle(async () => ({ items: [], has_more: true, next_cursor: 'repeat' }));
  await assert.rejects(f.api.loadDiscoveryCandidates('source-1', 'ORACLE'), /分页无效/);
  assert.equal(f.calls.length, 2);
}
console.log('通过：草稿恢复、三引擎发布、OEM 分页候选与短期引用刷新、GUID 重建拒绝、无手填定位键及主机 Label 要求。');

// 已有资源与新建资源显示同一组可编辑连接字段，改绑走原资源 PATCH，不重建 Target。
{
  const f = fixture('system-1');
  const oldBinding = { binding_id: 'binding-1', source_id: 'source-1', source_locator_key: 'old-instance', source_locator: { host_target_key: 'old-host' }, row_version: 6, role: 'PRIMARY', priority: 100 };
  const database = { ...resource().database, target_id: 'target-1', target_row_version: 7, diagnostic_username: 'fixture_readonly', oracle_container_scope: 'PDB', oracle_pdb_name: 'MESPDB', endpoint: { host: 'db.example.invalid', port: 1521, service: 'MES', database: null, tls_enabled: false, tls_profile_ref: null }, monitoring_bindings: [oldBinding] };
  const item = resource();
  Object.assign(item, { mode: 'EXISTING', resource_id: 'resource-1', system_resource_id: 'sr-1', resource_row_version: 4, system_row_version: 5 });
  Object.assign(item.database, { ...f.api.monitoringDraft(database), target_row_version: 7, password: '' });
  const detail = { ...copy(system), business_system_id: 'system-1', status: 'DISABLED', row_version: 3, resources: [{ system_resource_id: 'sr-1', row_version: 5, resource: { resource_id: 'resource-1', database: copy(database) } }], relations: [] };
  f.api.seed({ system: copy(system), resources: [item], detail, sources: [{ ...monitor, source_type: 'PROMETHEUS' }, { source_id: 'oem-1', source_type: 'OEM', display_name: 'OEM' }] });
  const html = f.api.resourceEditorHtml(item);
  for (const name of ['host', 'port', 'database', 'oracle_scope', 'oracle_pdb_name', 'username', 'password', 'source_id', 'candidate_ref', 'version_code', 'db_role', 'owner_group_id', 'location_name']) {
    assert.match(html, new RegExp('name="' + name + '"'), name + ' 在已有资源编辑中不能消失');
  }
  assert.match(html, /name="db_type" disabled/);
  assert.match(html, /name="deployment_type" disabled/);
  assert.match(html, /留空保留原密码/);
  assert.equal(f.api.monitoringChanged(item), false);
  let patch = f.api.existingDatabasePatch(item);
  assert.equal(patch.monitoring, undefined);
  assert.equal(patch.diagnostic_password, undefined);
  assert.equal(patch.endpoint, undefined);
  f.handle(async () => { throw new Error('未修改映射不应请求 OEM'); });
  await f.api.refreshDiscoveryReferences();
  assert.equal(f.calls.length, 0);

  item.database.source_id = 'oem-1';
  item.database.candidate_ref = 'old-ref';
  item.database.candidate_locator_key = '1'.repeat(32);
  item.database.candidates = [{ candidate_ref: 'old-ref', locator_key: '1'.repeat(32) }];
  item.database.host_candidate_ref = '';
  item.database.host_candidate_locator_key = '';
  item.database.host_candidates = [];
  item.database.host = 'new.example.invalid';
  item.database.port = 1522;
  item.database.username = 'new_reader';
  assert.equal(f.api.monitoringChanged(item), true);
  f.handle(async call => {
    if (call.path.includes('instance-discoveries')) return { items: [{ candidate_ref: 'fresh-ref', locator_key: '1'.repeat(32) }], host_items: [] };
    if (call.path === '/targets/test-connection') return { ok: true };
    if (call.path === '/managed-resources/resource-1') return { row_version: 5, database: { ...database, target_row_version: 8, monitoring_bindings: [{ ...oldBinding, source_id: 'oem-1', source_locator_key: '1'.repeat(32), source_locator: {}, row_version: 7 }], endpoint: { ...database.endpoint, host: 'new.example.invalid', port: 1522 } } };
    if (call.path.includes('/business-systems/system-1')) return copy(detail);
    throw new Error('不应重建资源：' + call.path);
  });
  await f.api.validateResource(item);
  const test = f.calls.find(call => call.path === '/targets/test-connection');
  assert.equal(test.payload.saved_target_id, 'target-1');
  assert.equal(test.payload.diagnostic_username, 'new_reader');
  assert.equal(test.payload.diagnostic_credential, undefined);
  await f.api.commit();
  assert.equal(f.navigations.length, 1, f.notifications.join('；'));
  patch = f.calls.find(call => call.path === '/managed-resources/resource-1').payload.database;
  assert.equal(patch.monitoring.source_id, 'oem-1');
  assert.equal(patch.monitoring.candidate_ref, 'fresh-ref');
  assert.equal(patch.monitoring.binding_id, 'binding-1');
  assert.equal(patch.monitoring.binding_row_version, 6);
  assert.equal(patch.target_row_version, 7);
  assert.equal(patch.business_system_id, 'system-1');
  assert.equal(patch.system_resource_id, 'sr-1');
  assert.equal(patch.endpoint.host, 'new.example.invalid');
  assert.equal(patch.endpoint.port, 1522);
  assert.equal(patch.diagnostic_username, 'new_reader');
  assert.equal(patch.diagnostic_password, undefined);
  assert.equal(f.calls.some(call => call.path.endsWith('/resources') && call.method === 'POST'), false);
  assert.equal(item.database.original_binding.source_id, 'oem-1');

  // 其他映射可单独选择，也可以新增；不会被隐式删除。
  item.database.monitoring_bindings.push({ ...oldBinding, binding_id: 'binding-2', role: 'SUPPLEMENTARY' });
  f.api.selectMonitoringBinding(item, 'binding-2');
  assert.equal(item.database.original_binding.binding_id, 'binding-2');
  f.api.selectMonitoringBinding(item, '');
  assert.equal(item.database.original_binding, null);
  assert.equal(item.database.monitoring_bindings.length, 2);
}

// 已有 RAC 节点独立显示与编辑连接和监控，不能广播到同组其他节点。
{
  const f = fixture('system-1'), item = resource();
  Object.assign(item, { mode: 'EXISTING', resource_id: 'r-1', system_resource_id: 'sr-1', rac_group_id: 'rac-1' });
  Object.assign(item.database, { deployment_type: 'RAC', rac_node_number: 1, password: '' });
  const other = copy(item); other.draft_resource_id = 'node-2'; other.database.rac_node_number = 2;
  f.api.seed({ system: copy(system), resources: [item, other], sources: [monitor] });
  const html = f.api.resourceEditorHtml(item);
  assert.match(html, /RAC 节点编号/);
  assert.match(html, /name="rac_cluster_name"/);
  assert.match(html, /name="host"/);
  assert.match(html, /name="source_id"/);
  f.api.updateResourceField(item, { name: 'source_id', value: '' });
  assert.equal(other.database.source_id, 'source-1');
  const previousName = other.display_name;
  f.api.updateResourceField(item, { name: 'rac_cluster_name', value: 'ERP RAC 新名称' });
  assert.equal(other.rac_cluster_name, 'ERP RAC 新名称');
  assert.equal(other.display_name, previousName, '改集群名称不能覆盖已有节点资源名称');
}
console.log('通过：已有数据库字段完整性、原绑定回显、OEM 改绑发布、密码留空和连接测试、多映射保留、RAC 节点隔离。');
