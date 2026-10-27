import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';

const file = fileURLToPath(new URL('../../ui/aiops/js/aiops-business-system-workspace.js', import.meta.url));
const source = readFileSync(file, 'utf8');
const instrumented = source.replace(/\}\)\(\);\s*$/, `globalThis.snapshotTest = {
  snapshotHtml, snapshotTargetId, snapshotDirty, loadSnapshotPolicy, snapshotOperation,
  seed(value) { detail = value.detail; resources = value.resources; activeResourceId = value.active; },
  setActive(value) { activeResourceId = value; },
  state() { return snapshotState; },
};})();`);
assert.notEqual(source, instrumented);
const node = { addEventListener() {}, querySelectorAll() { return []; }, classList: { toggle() {} } };
const doc = { getElementById() { return node; }, querySelectorAll() { return []; }, addEventListener() {} };
const calls = [];
const pending = [];
const policy = { enabled: false, interval_minutes: 5, retention_days: 30, top_statement_limit: 100, round_timeout_seconds: 47 };
const target = (version, override = {}) => ({ row_version: version, workload_snapshot_policy: { ...policy, ...override } });
let handler = async () => target(7);
const context = {
  document: doc, location: { search: '?id=sys', href: 'http://test/' }, URLSearchParams, URL,
  KBotAIOpsShell: { escape: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;'), toast() {}, ready: new Promise(() => {}) },
  KBotAIOpsAuth: { request(path, options = {}) { calls.push({ path, options }); return handler(path, options); } },
  crypto: { getRandomValues(bytes) { return bytes.fill(1); } }, addEventListener() {}, confirm() { return true; },
};
runInNewContext(instrumented, context, { filename: file });
const api = context.snapshotTest;
const resources = [{ draft_resource_id: 'a', system_resource_id: 'node1', database: { db_type: 'ORACLE' } }, { draft_resource_id: 'b', system_resource_id: 'node2', database: { db_type: 'ORACLE' } }];
api.seed({ resources, active: 'a', detail: { resources: [{ system_resource_id: 'node1', resource: { database: { target_id: 'target1' } } }, { system_resource_id: 'node2', resource: { database: { target_id: 'target2' } } }] } });
assert.equal(api.snapshotTargetId(resources[1]), 'target2');
for (const [db_type, expected] of [['ORACLE', 'V$SQLSTATS（非 AWR'], ['MYSQL', 'performance_schema SQL digest'], ['POSTGRESQL', 'pg_stat_statements']]) assert.ok(api.snapshotHtml({ database: { db_type } }).includes(expected));
await api.loadSnapshotPolicy();
assert.equal(calls.at(-1).path, '/api/v1/apps/aiops/targets/target1');
api.state().draft.enabled = true;
api.state().draft.interval_minutes = 2;
assert.equal(api.snapshotDirty(), true);
handler = async (path, options) => target(8, JSON.parse(options.body).workload_snapshot_policy);
await api.snapshotOperation('save');
const save = calls.at(-1);
assert.equal(save.options.headers['If-Match'], '"rv-7"');
assert.deepEqual(JSON.parse(save.options.body), { workload_snapshot_policy: { ...policy, enabled: true, interval_minutes: 2 } });
assert.equal(api.snapshotDirty(), false);
handler = async () => ({ ...target(9, { enabled: true, interval_minutes: 2 }), connectivity_status: 'CONNECTED' });
await api.snapshotOperation('refresh');
assert.equal(calls.at(-1).options.headers['If-Match'], '"rv-8"');
assert.ok(calls.at(-1).options.headers['Idempotency-Key']);
api.state().draft.interval_minutes = '';
const count = calls.length;
await api.snapshotOperation('save');
assert.equal(calls.length, count);
assert.match(api.state().error, /有效整数/);
api.state().draft.interval_minutes = 3;
handler = async () => { throw Object.assign(new Error('conflict'), { status: 412 }); };
await api.snapshotOperation('save');
assert.match(api.state().error, /重新加载/);
assert.equal(api.snapshotDirty(), true);
handler = (path) => new Promise(resolve => pending.push({ path, resolve }));
const first = api.loadSnapshotPolicy();
api.setActive('b');
const second = api.loadSnapshotPolicy();
pending[1].resolve(target(22));
await second;
pending[0].resolve(target(11));
await first;
assert.equal(api.state().targetId, 'target2');
assert.equal(api.state().target.row_version, 22);
