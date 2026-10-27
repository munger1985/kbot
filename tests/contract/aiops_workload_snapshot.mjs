import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';

const file = fileURLToPath(new URL('../../ui/aiops/js/aiops-business-system-workspace.js', import.meta.url));
const source = readFileSync(file, 'utf8');
const instrumented = source.replace(/\}\)\(\);\s*$/, `globalThis.snapshotTest = {
  snapshotHtml, snapshotTargetId, stageSnapshotPolicy, loadSnapshotPolicy, snapshotOperation, draftPayload,
  seed(value) { detail = value.detail; resources = value.resources; activeResourceId = value.active; workspaceReadOnly = value.published; },
  setActive(value) { activeResourceId = value; }, setPublished(value) { workspaceReadOnly = value; },
  state() { return snapshotState; },
};})();`);
assert.notEqual(source, instrumented);
const notifications = [];
let popoversSupported = true;
const node = { addEventListener() {}, querySelectorAll() { return []; }, classList: { toggle() {} }, append(value) { notifications.push(value); } };
const doc = { getElementById() { return node; }, querySelector() { return node; }, querySelectorAll() { return []; }, addEventListener() {}, body: { append(value) { notifications.push(value); } }, createElement() { const value = { style: {}, setAttribute(name, data) { this[name] = data; }, remove() {} }; if (popoversSupported) value.showPopover = function () { this.open = true; }; return value; } };
const calls = [], pending = [];
const policy = { enabled: false, interval_minutes: 5, retention_days: 30, top_statement_limit: 100, round_timeout_seconds: 47 };
const target = (version, override = {}) => ({ row_version: version, workload_snapshot_policy: { ...policy, ...override } });
let handler = async () => target(7);
const context = {
  document: doc, location: { search: '?id=sys', href: 'http://test/' }, URLSearchParams, URL,
  KBotAIOpsShell: { escape: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;'), toast() {}, ready: new Promise(() => {}) },
  KBotAIOpsAuth: { request(path, options = {}) { calls.push({ path, options }); return handler(path, options); } },
  crypto: { getRandomValues(bytes) { return bytes.fill(1); } }, addEventListener() {}, confirm() { return true; }, setTimeout() {},
};
runInNewContext(instrumented, context, { filename: file });
const api = context.snapshotTest;
const resources = [{ draft_resource_id: 'a', system_resource_id: 'node1', database: { db_type: 'ORACLE' } }, { draft_resource_id: 'b', system_resource_id: 'node2', database: { db_type: 'ORACLE' } }];
api.seed({ resources, active: 'a', published: false, detail: { resources: [{ system_resource_id: 'node1', resource: { database: { target_id: 'target1' } } }, { system_resource_id: 'node2', resource: { database: { target_id: 'target2' } } }] } });
assert.equal(api.snapshotTargetId(resources[1]), 'target2');
for (const [db_type, expected] of [['ORACLE', 'V$SQLSTATS（非 AWR'], ['MYSQL', 'performance_schema SQL digest'], ['POSTGRESQL', 'pg_stat_statements']]) assert.ok(api.snapshotHtml({ database: { db_type } }).includes(expected));
await api.loadSnapshotPolicy();
assert.equal(calls.at(-1).path, '/api/v1/apps/aiops/targets/target1');
api.state().draft.enabled = true;
api.state().draft.interval_minutes = 2;
api.stageSnapshotPolicy();
assert.equal(resources[0].workload_snapshot.target_row_version, 7);
assert.deepEqual(JSON.parse(JSON.stringify(resources[0].workload_snapshot.policy)), { ...policy, enabled: true, interval_minutes: 2 });
assert.equal(calls.length, 1);
api.setPublished(true);
assert.ok(!api.snapshotHtml(resources[0]).includes('data-snapshot-field'));
assert.ok(!api.snapshotHtml(resources[0]).includes('id="snapshot-refresh"'));
const count = calls.length;
await api.snapshotOperation();
assert.equal(calls.length, count);
api.setPublished(false);
handler = async () => ({ ...target(8), connectivity_status: 'CONNECTED' });
await api.snapshotOperation();
assert.equal(calls.at(-1).options.headers['If-Match'], '"rv-7"');
assert.ok(calls.at(-1).options.headers['Idempotency-Key']);
assert.equal(resources[0].workload_snapshot.target_row_version, 8);
assert.equal(notifications.length, 1);
assert.equal(notifications[0].className, 'ops-toast');
assert.equal(notifications[0].role, 'status');
assert.equal(notifications[0].popover, 'manual');
assert.equal(notifications[0].open, true);
assert.match(notifications[0].textContent, /连接检查已完成：CONNECTED/);
assert.ok(!api.snapshotHtml(resources[0]).includes('连接检查已完成'));
popoversSupported = false;
handler = async () => ({ ...target(9), connectivity_status: 'FAILED' });
await api.snapshotOperation();
assert.equal(notifications.length, 2);
assert.match(notifications[1].textContent, /连接检查已完成：FAILED/);
assert.equal(notifications[1].popover, undefined);
handler = async () => { throw Object.assign(new Error('conflict'), { status: 412 }); };
await api.snapshotOperation();
assert.match(api.state().error, /其他操作修改/);
assert.equal(notifications.length, 2);
assert.equal(resources[0].workload_snapshot.policy.interval_minutes, 2);
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
