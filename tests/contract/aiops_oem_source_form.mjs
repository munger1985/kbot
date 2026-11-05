// 运行真实 OEM 监控源表单脚本，验证只提交服务级信息且编辑不清空既有指标配置。
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const script = readFileSync(new URL('../../ui/aiops/js/aiops-configurations.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../ui/aiops/diagnostic-sources.html', import.meta.url), 'utf8');
assert.ok(!html.includes('oem_standard_mappings'));
assert.ok(!html.includes('标准指标原生映射'));
const nodes = new Map();
const node = id => {
  if (!nodes.has(id)) nodes.set(id, { hidden: false, textContent: '' });
  return nodes.get(id);
};
const form = node('diagnostic-source-form');
form.elements = Object.fromEntries(Object.entries({
  source_type: 'OEM', display_name: '实验 OEM', endpoint: 'https://oem.example/em/api',
  source_username: 'reader', source_password: 'fixture-only', access_token: '', loki_tenant_id: '',
}).map(([name, value]) => [name, { value }]));
const context = {
  document: { body: { dataset: { page: 'diagnostic-sources' } }, getElementById: node },
  KBotAIOpsShell: { ready: new Promise(() => {}), escape: value => String(value ?? '') },
  KBotAIOpsAuth: {}, URLSearchParams, location: { search: '' },
};
const instrumented = script.replace(/\}\)\(\);\s*$/, 'globalThis.formTest = { payload, renderType, edit(value) { state.editing = value; } };})();');
runInNewContext(instrumented, context);
context.formTest.renderType();
assert.match(node('source-endpoint-help').textContent, /具体数据库在业务系统中选择/);
assert.ok(!node('source-endpoint-help').textContent.includes('GUID'));
const created = JSON.parse(JSON.stringify(context.formTest.payload()));
assert.deepEqual(created, {
  display_name: '实验 OEM', source_type: 'OEM', endpoint: 'https://oem.example/em/api',
  credentials: { username: 'reader', password: 'fixture-only' }, config: {},
});
context.formTest.edit({ source_id: 'existing', config: { standard_metric_mappings: { preserved: true } } });
form.elements.source_username.value = '';
form.elements.source_password.value = '';
const edited = context.formTest.payload();
assert.equal(edited.config, undefined);
assert.equal(edited.credentials, undefined);
assert.equal(edited.source_type, undefined);
form.elements.source_username.value = 'reader';
assert.throws(() => context.formTest.payload(), /同时填写/);
console.log('通过：OEM 新建仅服务连接、无 GUID/指标 JSON，编辑保留受管凭据及既有映射。');
