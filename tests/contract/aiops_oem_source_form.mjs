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
  exadata_metric_transport: 'OEM_REST', exadata_emcli_timezone: 'UTC', exadata_emcli_certificate_sha256: '',
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
  credentials: { username: 'reader', password: 'fixture-only' }, config: { exadata_metric_transport: 'OEM_REST' },
});
context.formTest.edit({ source_id: 'existing', config: { standard_metric_mappings: { preserved: true } } });
form.elements.source_username.value = '';
form.elements.source_password.value = '';
const edited = context.formTest.payload();
assert.equal(edited.config.exadata_metric_transport, 'OEM_REST');
assert.equal(edited.config.standard_metric_mappings, undefined, '既有映射由后端保留，表单不能覆盖');
assert.equal(edited.credentials, undefined);
assert.equal(edited.source_type, undefined);
form.elements.source_username.value = 'reader';
assert.throws(() => context.formTest.payload(), /同时填写/);
form.elements.source_username.value = '';
form.elements.exadata_metric_transport.value = 'EMCLI';
form.elements.exadata_emcli_certificate_sha256.value = 'a'.repeat(64);
context.formTest.renderType();
const native = context.formTest.payload();
assert.equal(native.config.exadata_metric_transport, 'EMCLI');
assert.equal(native.config.exadata_emcli_timezone, 'UTC');
assert.equal(native.config.exadata_emcli_certificate_sha256, 'a'.repeat(64));
assert.equal(form.elements.exadata_emcli_certificate_sha256.required, true);
assert.equal(form.elements.exadata_emcli_certificate_sha256.disabled, false);
form.elements.source_type.value = 'PROMETHEUS';
form.elements.exadata_emcli_certificate_sha256.value = 'unfinished-pin';
context.formTest.renderType();
assert.equal(form.elements.exadata_emcli_certificate_sha256.required, false);
assert.equal(form.elements.exadata_emcli_certificate_sha256.disabled, true, '隐藏的未完成指纹不阻止其他来源表单');
assert.deepEqual(JSON.parse(JSON.stringify(context.formTest.payload().config)), {});
console.log('通过：OEM 默认 REST、显式 Exadata EMCLI、受管凭据/映射保留，非 OEM 表单不受隐藏字段影响。');
