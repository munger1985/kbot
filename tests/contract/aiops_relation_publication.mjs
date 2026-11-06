// 执行真实发布函数，使用严格合同及版本校验桩，不访问远端服务。
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const root = fileURLToPath(new URL('../../', import.meta.url));
const portalPath = root + 'apps/portal-web/src/features/aiops/management/BusinessSystemWorkspace.vue';
const portal = existsSync(portalPath);
const source = readFileSync(portal ? portalPath : root + 'ui/aiops/js/aiops-business-system-workspace.js', 'utf8');
const extract = (start, end) => {
  const from = source.indexOf(start), to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from, '必须抽取到真实生产函数');
  return source.slice(from, to);
};
let functions = extract('function relationPatchPayload(', 'function draftPayload(')
  + extract('async function updateSystem(', 'async function commit(');
if (portal) {
  const require = createRequire(root + 'apps/portal-web/package.json');
  const ts = require('typescript');
  functions = ts.transpileModule(functions, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
}
const copy = value => structuredClone(value);
const mutableKeys = ['attributes', 'confirmation_status', 'display_name', 'participants', 'responsibility_group_id'].sort();
const relation = (id = 'relation-1', extra = {}) => ({
  draft_relation_id: id || 'new-relation', ...(id ? { relation_id: id, row_version: 2 } : {}),
  relation_type_key: 'DEPENDS_ON', origin: 'MANUAL', display_name: ' 测试关系 ',
  responsibility_group_id: '', attributes: {},
  participants: ['a', 'b'].map((id, index) => ({ draft_resource_id: id, participant_role: index ? 'TARGET' : 'SOURCE', display_order: index })),
  ...extra,
});
function fixture(original = [relation()], edited = copy(original), { newResource = false } = {}) {
  const resources = ['a', 'b'].map((id, index) => ({
    draft_resource_id: id, ...(!(newResource && index) ? {
      system_resource_id: 'sys-' + id, resource_id: 'managed-' + id, resource_row_version: 3, system_row_version: 4,
    } : {}),
    display_name: id, owner_group_id: 'group', location_name: '', attributes: {}, architecture_layer_key: 'DATA', display_order: index,
  }));
  const state = {
    row_version: 5, resources: resources.filter(item => item.system_resource_id).map(item => ({
      system_resource_id: item.system_resource_id, row_version: item.system_row_version,
      resource: { resource_id: item.resource_id, row_version: item.resource_row_version },
    })),
    relations: copy(original).map(item => ({ ...item, participants: item.participants.map(p => ({
      system_resource_id: 'sys-' + p.draft_resource_id, participant_role: p.participant_role, display_order: p.display_order,
    })) })),
  };
  const env = { detail: copy(state), resources, relations: edited, state, calls: [], failCreateNumber: 0, creates: 0, deleted: [], rejectDelete: false };
  const checkVersion = (headers, version) => assert.equal(headers['If-Match'], '"rv-' + version + '"');
  env.request = async (url, options = {}) => {
    const path = url.replace(/^.*\/ops/, ''), init = options.init || options, method = init.method || 'GET';
    const body = init.body ? JSON.parse(init.body) : undefined;
    env.calls.push({ path, method, body, headers: init.headers || {} });
    if (path === '/business-systems/system') {
      if (method === 'PATCH') { checkVersion(init.headers, state.row_version); state.row_version += 1; }
      return copy(state);
    }
    if (path.startsWith('/managed-resources/')) {
      const item = state.resources.find(item => path.endsWith('/' + item.resource.resource_id));
      checkVersion(init.headers, item.resource.row_version);
      item.resource.row_version += 1;
      return copy(item.resource);
    }
    const resourceMatch = path.match(/\/resources(?:\/([^/]+))?$/);
    if (resourceMatch) {
      if (method === 'POST') {
        const item = { system_resource_id: 'sys-' + body.draft_resource_id, row_version: 1,
          resource: { resource_id: 'managed-' + body.draft_resource_id, row_version: 1 } };
        state.resources.push(item);
      } else {
        const item = state.resources.find(item => item.system_resource_id === resourceMatch[1]);
        checkVersion(init.headers, item.row_version);
        item.row_version += 1;
      }
      return copy(state);
    }
    const match = path.match(/\/relations(?:\/([^/]+))?$/);
    assert.ok(match, '未定义的请求：' + method + ' ' + path);
    if (method === 'DELETE') {
      const index = state.relations.findIndex(item => item.relation_id === match[1]);
      assert.ok(index >= 0, '不能重复删除同一关系');
      checkVersion(init.headers, state.relations[index].row_version);
      if (env.rejectDelete) throw new Error('隔离测试：删除失败');
      state.relations.splice(index, 1); env.deleted.push(match[1]);
    } else {
      const allowed = method === 'PATCH' ? mutableKeys : [...mutableKeys, 'origin', 'relation_type_key'].sort();
      assert.deepEqual(Object.keys(body).sort(), allowed, '必须严格区分创建与更新合同');
      assert.ok(body.participants.every(item => item.system_resource_id && !('draft_resource_id' in item)));
      if (method === 'PATCH') {
        const item = state.relations.find(item => item.relation_id === match[1]);
        assert.ok(item, '不能更新已删除的关系');
        checkVersion(init.headers, item.row_version);
        Object.assign(item, body, { row_version: item.row_version + 1 });
      } else {
        env.creates += 1;
        if (env.creates === env.failCreateNumber) throw new Error('隔离测试：发布中断');
        state.relations.push({ ...body, relation_id: 'created-' + env.creates, row_version: 1 });
      }
    }
    return copy(state);
  };
  const bindings = portal ? `
    const detail = { value: env.detail }, resources = { value: env.resources }, relations = { value: env.relations };
    const base = { value: '/ops' }, options = { value: {} }, systemId = { value: 'system' }, isPublishedView = { value: false };
    const businessFetch = env.request;
  ` : `
    const detail = env.detail, resources = env.resources, relations = env.relations, systemId = 'system';
    const request = env.request, isPublishedView = () => false;
  `;
  const api = new Function('env', `
    ${bindings}
    const hardware = { mapping: () => ({}), draft: () => ({}) }, uuid7 = () => 'test-idempotency-key';
    const systemPayload = () => ({}), resourcePayload = item => ({ draft_resource_id: item.draft_resource_id });
    const isInferredRelation = item => item.attributes?.inferred === true;
    ${functions}
    return { updateSystem, relationPayload, relationPatchPayload };
  `)(env);
  return { env, ...api };
}

test('旧关系仅 PATCH 可变字段，并将新资源参与者映射为服务端标识', async () => {
  const f = fixture(undefined, undefined, { newResource: true });
  await f.updateSystem();
  const call = f.env.calls.find(call => call.path.includes('/relations/'));
  assert.equal(call.method, 'PATCH');
  assert.deepEqual(Object.keys(call.body).sort(), mutableKeys);
  assert.deepEqual(call.body.participants.map(item => item.system_resource_id), ['sys-a', 'sys-b']);
  assert.equal(f.env.relations[0].row_version, 3);
  assert.equal(f.env.state.relations[0].display_name, '测试关系');
});

test('新关系 POST 保留类型与来源，草稿参与者仍使用 draft ID', async () => {
  const item = relation(null, { origin: 'IMPORT' }), f = fixture([], [item]);
  assert.equal(f.relationPayload(item).participants[0].draft_resource_id, 'a');
  await f.updateSystem();
  assert.equal(f.env.state.relations[0].origin, 'IMPORT');
  assert.equal(f.env.state.relations[0].relation_type_key, item.relation_type_key);
  assert.equal(item.relation_id, 'created-1');
});

for (const changes of [{ relation_type_key: 'RUNS_ON' }, { origin: 'DISCOVERY' }]) {
  test('类型或来源变更应 DELETE + POST，不能悄悄丢弃修改：' + JSON.stringify(changes), async () => {
    const f = fixture([relation()], [relation('relation-1', changes)]);
    await f.updateSystem();
    assert.deepEqual(f.env.calls.filter(call => call.path.includes('/relations')).map(call => call.method), ['DELETE', 'POST']);
    for (const [key, value] of Object.entries(changes)) assert.equal(f.env.state.relations[0][key], value);
    assert.equal(f.env.relations[0].relation_id, 'created-1');
  });
}

test('重建中断后重试，不重复删除旧关系', async () => {
  const f = fixture([relation()], [relation('relation-1', { relation_type_key: 'RUNS_ON' })]);
  f.env.failCreateNumber = 1;
  await assert.rejects(f.updateSystem(), /发布中断/);
  assert.equal(f.env.relations[0].relation_id, undefined);
  assert.deepEqual(f.env.detail.relations, []);
  await f.updateSystem();
  assert.deepEqual(f.env.deleted, ['relation-1']);
  assert.equal(f.env.state.relations.length, 1);
});

test('后续关系失败后重试，已成功 PATCH 使用新版本，已创建关系不重复 POST', async () => {
  const f = fixture([relation()], [relation(), relation(null), relation(null, { draft_relation_id: 'another' })]);
  f.env.failCreateNumber = 2;
  await assert.rejects(f.updateSystem(), /发布中断/);
  assert.equal(f.env.relations[0].row_version, 3);
  assert.equal(f.env.relations[1].relation_id, 'created-1');
  await f.updateSystem();
  assert.equal(f.env.creates, 3);
  assert.equal(f.env.state.relations.length, 3);
  assert.equal(f.env.relations[0].row_version, 4);
  assert.equal(f.env.relations[1].row_version, 2);
});

test('删除失败不能清除待重建关系标识；已删除的普通关系重试不重复删除', async () => {
  const f = fixture([relation()], [relation('relation-1', { origin: 'DISCOVERY' })]);
  f.env.rejectDelete = true;
  await assert.rejects(f.updateSystem(), /删除失败/);
  assert.equal(f.env.relations[0].relation_id, 'relation-1');
  assert.equal(f.env.detail.relations.length, 1);
  const g = fixture([relation()], [relation(null)]);
  g.env.failCreateNumber = 1;
  await assert.rejects(g.updateSystem(), /发布中断/);
  await g.updateSystem();
  assert.deepEqual(g.env.deleted, ['relation-1']);
});

test('缺省来源按手工关系处理，不额外重建；推导关系创建保留 DISCOVERY', async () => {
  const original = relation(), edited = relation();
  delete original.origin; delete edited.origin;
  const f = fixture([original], [edited]);
  await f.updateSystem();
  assert.deepEqual(f.env.deleted, []);
  assert.equal(f.relationPayload(relation(null, { origin: undefined, attributes: { inferred: true } })).origin, 'DISCOVERY');
});
