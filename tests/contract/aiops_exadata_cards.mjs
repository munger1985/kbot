// 两端独立执行：验证纯展示语义、真实生产组件、ECharts 和响应式布局；不连接现场 OEM。
import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { createServer as createNetServer } from 'node:net'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import vm from 'node:vm'

const root = fileURLToPath(new URL('../../', import.meta.url))
const isKbot = existsSync(resolve(root, 'ui/aiops/monitoring.html'))
const kbot = isKbot ? root : process.env.KBOT_ROOT || resolve(root, '../kbot4')
const ammolite = isKbot ? process.env.AMMOLITE_ROOT || resolve(root, '../ammolite_cube') : root
const portal = resolve(ammolite, 'apps/portal-web'), require = createRequire(resolve(portal, 'package.json'))
const { chromium, expect } = require('@playwright/test'), ts = require('typescript')
const { createServer } = await import(pathToFileURL(resolve(portal, 'node_modules/vite/dist/node/index.js')))
const read = path => readFileSync(path, 'utf8')
const helper = read(resolve(portal, 'src/features/aiops/business/exadataPresentation.ts'))
const compiled = ts.transpileModule(helper, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const generated = '// 由 Ammolite exadataPresentation.ts 同步，合同测试校验两端展示语义一致。\n(function () {\n  const exports = {};\n' + compiled + '\n  globalThis.KBotAIOpsExadataPresentation = exports;\n})();\n'
assert.equal(read(resolve(kbot, 'ui/aiops/js/aiops-exadata-presentation.js')), generated, 'KBot 展示脚本必须与完整 TypeScript 源一致，不只比较示例输出')
const context = { exports: {} }; vm.runInNewContext(compiled, context)
const other = {}; vm.runInNewContext(read(resolve(kbot, 'ui/aiops/js/aiops-exadata-presentation.js')), other)
const api = isKbot ? other.KBotAIOpsExadataPresentation : context.exports
const clone = value => JSON.parse(JSON.stringify(value))
assert.equal(read(resolve(kbot, 'ui/aiops/css/aiops-exadata-metrics.css')), read(resolve(portal, 'src/features/aiops/business/exadataMetrics.css')), '两端 Exadata CSS 必须同步')

const member = (id, name) => ({ target_id: id, target_type: 'oracle_exadata', display_name: name })
const members = [member('cell-a', 'SIM CELL A'), member('cell-b', 'SIM CELL B')]
const statusCatalog = JSON.parse(read(resolve(ammolite, 'services/aiops_agent/src/aiops_agent/resources/exadata_oem/status.v1.json')))
assert.deepEqual(JSON.parse(read(resolve(kbot, 'services/aiops_agent/src/aiops_agent/resources/exadata_oem/status.v1.json'))), statusCatalog, '两端状态审核目录一致')
const statusesByCode = new Map(statusCatalog.groups.flatMap(group => group.metric_codes.map(code => [code, group.status])))
const metric = (selector, type = 'NUMBER', target = 'oracle_exadata') => {
  const code = `${target}.${selector.replace('/', '.')}`
  return { code, selector, display_name: selector, native_value_type: type, status: statusesByCode.get(code) || null }
}
const cpu = metric('Cell_Metric/cpu_utilization'), memory = metric('Cell_Metric/memory_utilization')
const status = ['Status', 'msStatus', 'rsStatus'].map(name => metric(`CellSrv_Status_Metric/${name}`, 'STRING'))
const sample = (definition, value, overrides = {}) => ({ component_id: 'cell-a', metric_code: definition.code, selector: definition.selector, observed_at: '2026-11-11T12:00:00Z', value, dimensions: {}, unit: '%', native_value_type: definition.native_value_type, quality: 'OBSERVED', ...overrides })
const overview = { resource_id: 'resource', binding_id: 'binding', binding_version: 1, source_version: 2, catalog_sha256: 'c'.repeat(64), query_mode: 'LATEST_SNAPSHOT', window_start: null, window_end: null, members, definitions: [...status, cpu, memory], samples: [...status.map(item => sample(item, '1', { unit: 'n/a' })), sample(cpu, '27.210000000000001'), sample(memory, '0')], gaps: [{ code: 'THRESHOLDS_UNVERIFIED', detail: '阈值与状态字典未验证，不判定健康。' }], total_queries: 5, next_offset: null, catalog_version: 'exadata-oem-v1', captured_at: '2026-11-11T12:01:00Z' }

const dbCPU = metric('CPUUsageSummary/OverAllUsagePct', 'NUMBER', 'host'), dbMemory = metric('Load/memUsedPct', 'NUMBER', 'host')
const switchAP = metric('AccessPointResponse/statusMsg', 'STRING', 'oracle_si_netswitch')
const deviceView = { ...clone(overview), members: [...members, {target_id:'db-a',target_type:'host',display_name:'SIM DB SERVER'}, {target_id:'switch-a',target_type:'oracle_si_netswitch',display_name:'SIM SWITCH'}, {target_id:'pdu-a',target_type:'oracle_si_pdu',display_name:'SIM PDU'}, {target_id:'rack-a',target_type:'oracle_si_rack',display_name:'SIM RACK'}],
  definitions: [...overview.definitions, dbCPU, dbMemory, switchAP],
  samples: [...overview.samples, sample(dbCPU, '0', {component_id:'db-a'}), sample(dbMemory, '37.125', {component_id:'db-a'}), sample(switchAP, 'UP', {component_id:'switch-a',unit:'n/a',dimensions:{ap_guid:'AP-1'}})],
  device_coverage: [
    {component_id:'cell-a',metric_codes:overview.definitions.map(item=>item.code),queried_metric_codes:overview.definitions.map(item=>item.code),note:''},
    {component_id:'cell-b',metric_codes:[cpu.code],queried_metric_codes:[],note:''},
    {component_id:'db-a',metric_codes:[dbCPU.code,dbMemory.code],queried_metric_codes:[dbCPU.code,dbMemory.code],note:''},
    {component_id:'switch-a',metric_codes:[switchAP.code],queried_metric_codes:[switchAP.code],note:''},
    {component_id:'pdu-a',metric_codes:['oracle_si_pdu.PduHttpResponse.Status'],queried_metric_codes:[],note:''},
    {component_id:'rack-a',metric_codes:[],queried_metric_codes:[],note:'此类目标尚无审核通过的独立状态读取指标'},
  ] }
const deviceRows = api.buildExadataDevices(deviceView)
assert.equal(deviceRows.length, 6, '指标分页不能隐藏已确认设备')
assert.equal(deviceRows.find(item=>item.member.target_id==='db-a').typeLabel, 'DB Server')
assert.ok(deviceRows.find(item=>item.member.target_id==='db-a').resources.some(item=>item.value.startsWith('0 %')), 'DB Server CPU 零值保留')
assert.equal(deviceRows.find(item=>item.member.target_id==='switch-a').states[0].label, '可达')
assert.equal(deviceRows.find(item=>item.member.target_id==='pdu-a').states.length, 0, '未查询不伪造健康或无采样')
assert.ok(deviceRows.find(item=>item.member.target_id==='pdu-a').note.includes('本页未查询'))
assert.ok(deviceRows.find(item=>item.member.target_id==='rack-a').note.includes('尚无审核'))
assert.equal(api.buildExadataDevices(overview).length, 0, '冻结旧报告不补造覆盖范围')
assert.equal(api.exadataOverviewNotes(deviceView).length, 0)
assert.ok(api.exadataOverviewNotes(overview).some(note => note.includes('DB Server') && note.includes('CPU/内存')))
assert.ok(api.exadataOverviewNotes(overview).some(note => note.includes('交换机')))

const cards = api.buildExadataCards(overview)
assert.equal(cards.length, 5)
assert.equal(cards[3].value, '27.210000000000001', '原始精度不得被图表数值转换覆盖')
assert.equal(cards[4].value, '0'); assert.equal(cards[4].percent, 0, '真零值必须可展示')
assert.equal(cards[0].quality, '已采样'); assert.equal(cards[0].value, '1', '翻译状态仍保留原始采样')
assert.equal(api.exadataStatusItems(cards[0])[0].label, '可用')
const palette = { accent: 'blue', info: 'cyan', muted: 'gray', line: 'gray', surface: 'white', ink: 'black' }
for (const value of [null, '', ' ', 'NaN', 'Infinity', '0x10', 'false']) assert.equal(api.exadataNumeric(value), null)
assert.equal(api.exadataNumeric('1.2e-3'), 0.0012)
assert.equal(api.exadataChartKind(cards[3], 'LATEST_SNAPSHOT'), 'line', 'CPU 默认显示真实历史趋势')
assert.equal(api.exadataChartKind(cards[4], 'LATEST_SNAPSHOT'), 'line', '内存真实零值显示为趋势采样点')
const ring = api.exadataChartOption({ ...cards[4], selector: 'Filesystem_Utilization_Metric/utilization' }, palette, 'donut')
assert.equal(ring.series[0].type, 'pie')
assert.deepEqual(clone(ring.series[0].data.map(item => item.value)), [0, 100])
assert.equal(ring.series[0].data[1].derived, true, '未使用部分标为计算值，不伪造OEM采样')
const preciseBaseline = { unit: '%', baseline: { unavailable_reason: null }, comparisons: [{ dimensions: {},
  average: '37.95429398496240601503759398496240601504', deviation_percent: '3.914460945120556658462827400372799012244',
  current: '39.44', difference: '1.48570601503759398496240601503759398496', sample_count: 7, observed_at: overview.captured_at }] }
const originalBaseline = JSON.stringify(preciseBaseline)
const historyView = { ...overview, query_mode: 'HISTORY', samples: [sample(cpu, '8'), sample(cpu, null, { observed_at: '2026-11-11T12:00:30Z', quality: 'EMPTY' }), sample(cpu, '13', { observed_at: '2026-11-11T12:01:00Z' })] }
const trendCard = api.exadataTrendCard(cards[3], historyView)
assert.equal(trendCard.value, cards[3].value, '历史趋势不覆盖当前快照摘要')
assert.deepEqual(clone(api.exadataChartOption(trendCard, palette).series[0].data.map(item => item.value[1])), [8, null, 13], '只使用OEM真实历史，不复制最新快照')
assert.equal(api.exadataTrendCard(cards[3], undefined).samples.length, 0)
assert.equal(api.exadataTrendCard(cards[3], '失败').samples.length, 0)
assert.equal(api.exadataTrendCard({ ...cards[3], unit: 'ms' }, historyView).samples.length, 0, '趋势单位不匹配不得合并')
assert.ok(api.exadataViewMatches(overview, historyView))
assert.equal(api.exadataViewMatches(overview, { ...historyView, binding_version: 2 }), false)
const trendBaseline = { ...preciseBaseline, component_id: cards[3].componentId, metric_code: cards[3].metricCode, baseline: { unavailable_reason: null, period: '1w' } }
const trendOption = api.exadataChartOption(trendCard, palette, 'line', trendBaseline, '1w')
assert.equal(trendOption.series[0].markLine.data[0].yAxis, Number(preciseBaseline.comparisons[0].average))
assert.equal(trendOption.series[0].markLine.label.formatter, '7 天均值 37.95')
assert.equal(trendOption.series[0].markLine.data[0].lineStyle.type, 'dashed')
for (const other of [{ ...trendBaseline, unit: 'ms' }, { ...trendBaseline, component_id: 'cell-b' }, { ...trendBaseline, comparisons: [{ ...trendBaseline.comparisons[0], dimensions: { other: 'a' } }] }]) {
  assert.equal(api.exadataChartOption(trendCard, palette, 'line', other, '1w').series[0].markLine, undefined, '基线严格匹配组件、指标、单位及完整维度')
}
assert.equal(api.exadataChartOption(trendCard, palette, 'line', trendBaseline, '1m').series[0].markLine, undefined, '切换周期不保留旧基准线')
assert.equal(api.exadataBaselineSummary(preciseBaseline, '1w'), '7 天均值 37.95 % · 偏离 3.91%')
assert.equal(api.exadataBaselineSummary(preciseBaseline, '1m'), '30 天均值 37.95 % · 偏离 3.91%')
assert.ok(api.exadataBaselineLines(preciseBaseline, '1w')[0].includes('留存均值 37.95 %'))
assert.ok(api.exadataBaselineLines(preciseBaseline, '1w')[0].includes('差值 1.49 个百分点；偏离 3.91%'))
assert.equal(JSON.stringify(preciseBaseline), originalBaseline, '只格式化展示，不修改底层精度或原始证据')
for (const [average, deviation, expected] of [['0', null, '均值 0.00 % · 零均值，偏离不可定义'],
  ['20', '-3.915', '均值 20.00 % · 偏离 -3.92%'], ['1.2e-3', '1.005', '均值 0.00 % · 偏离 1.01%'],
  ['NaN', 'Infinity', '均值 无有效数值 % · 偏离 无有效数值%']]) {
  const result = { ...preciseBaseline, comparisons: [{ ...preciseBaseline.comparisons[0], average, deviation_percent: deviation }] }
  assert.ok(api.exadataBaselineSummary(result, '1w').endsWith(expected))
}
const separated = api.buildExadataCards({ ...overview, samples: [sample(cpu, '1', { dimensions: { disk: 'same', rack: 'a' } }), sample(cpu, '2', { dimensions: { rack: 'a', disk: 'same' }, observed_at: '2026-11-11T12:01:00Z' }), sample(cpu, '3', { component_id: 'cell-b' }), sample(cpu, '4', { unit: 'ms' })], gaps: [] })
assert.equal(separated.length, 3, '不同 Cell 和原生单位绝不合并')
assert.equal(separated[0].series.length, 1, '维度键顺序不改变序列身份')
const dimensional = api.buildExadataCards({ ...overview, samples: [sample(cpu, '2', { dimensions: { object_name: 'disk-a' } }), sample(cpu, '9', { dimensions: { object_name: 'disk-b' } })], gaps: [] })[0]
assert.equal(dimensional.series.length, 2)
assert.equal(dimensional.value, '2 个维度', '多维度不求和、不以最大值冒充整机值')
assert.equal(dimensional.percent, null)
assert.equal(api.exadataChartOption(dimensional, palette).series.length, 2)
assert.equal(api.exadataChartKind(dimensional, 'LATEST_SNAPSHOT'), 'line', 'CPU 多维各自成线，不求和绘成饼图')
const bars = api.exadataChartOption(dimensional, palette, 'bar')
assert.equal(bars.series[0].type, 'bar')
assert.deepEqual(clone(bars.series[0].data.map(item => item.value)), [2, 9])
assert.equal(bars.xAxis.max, undefined, '不将负载等无界数值截断到100')
assert.ok(!bars.tooltip.formatter({ data: { raw: '<script>', dimension: '<img>', at: '<svg>' } }).includes('<img'))
const manyDimensions = { ...dimensional, series: Array.from({length:12}, (_, index) => ({ ...dimensional.series[0], label:'disk='+index })) }
const scrolling = api.exadataChartOption(manyDimensions, palette, 'bar')
assert.equal(scrolling.series[0].data.length, 12, '柱图不静默截掉第七个以后的维度')
assert.equal(scrolling.dataZoom.length, 2)
const missingBar = api.buildExadataCards({ ...overview, samples: [sample(cpu, '0', { dimensions: { disk:'a' } }), sample(cpu, null, { dimensions:{disk:'b'}, quality:'EMPTY' })], gaps:[] })[0]
assert.deepEqual(clone(api.exadataChartOption(missingBar,palette,'bar').series[0].data.map(item=>item.value)),[0,null], '空维度不补零，真零不丢弃')
assert.equal(api.exadataChartKind(cards[0], 'LATEST_SNAPSHOT'), null)
const nullLatest = api.buildExadataCards({ ...overview, query_mode: 'HISTORY', samples: [sample(cpu, '8'), sample(cpu, null, { observed_at: '2026-11-11T12:01:00Z', quality: 'EMPTY' })] })[0]
assert.equal(nullLatest.value, '无数据', '末次空值不回填旧的非空值')
assert.equal(nullLatest.percent, null, '历史模式不生成当前百分比仪表')
const option = api.exadataChartOption(nullLatest, palette)
assert.deepEqual(clone(option.series[0].data.map(item => item.value[1])), [8, null])
assert.equal(option.series[0].connectNulls, false)
assert.equal(option.series[0].markLine, undefined, '未验证阈值不画健康分界线')
assert.ok(!option.tooltip.formatter([{ seriesName: '<img onerror=alert(1)>', data: { raw: '<script>', at: 'now' } }]).includes('<img'))
assert.equal(api.exadataChartOption(cards[0], palette).series.length, 0, '字符串状态不能转换成数值时序')
assert.equal(statusesByCode.size, 35, '全部状态类指标纳入审核，而不只是三个服务')
const hostState = { ...metric('OEMTarget/status', 'STRING', 'host'), semantics: 'OEM_TARGET_AVAILABILITY' }
const hostCard = api.buildExadataCards({...overview, definitions:[hostState], samples:[sample(hostState, 'UP')], gaps:[]})[0]
assert.equal(hostCard.currentStatus, true)
assert.equal(api.exadataStatusItems(hostCard)[0].quality, '状态已读取', '当前状态取证不能冒充设备采样')
const pduState = metric('PduHttpResponse/Status', 'NUMBER', 'oracle_si_pdu')
for (const raw of ['1', '1.0', '01.00', '+1.0', '1e0', '10e-1', '.1e1']) {
  const card = api.buildExadataCards({...overview, definitions:[pduState], samples:[sample(pduState, raw)], gaps:[]})[0]
  assert.equal(api.exadataStatusItems(card)[0].label, '可达', `NUMBER ${raw} 与审核的 1 精确等值`)
  assert.equal(card.value, raw, '不改写原始证据值')
}
for (const raw of ['1.0000000000000000001', '0.9999999999999999999', 'NaN', 'Infinity', '1_0', '0x1', '１', '1e9999999']) {
  const card = api.buildExadataCards({...overview, definitions:[pduState], samples:[sample(pduState, raw)], gaps:[]})[0]
  assert.equal(api.exadataStatusItems(card)[0].tone, 'unknown', `${raw} 不能被浮点舍入或非法解析为健康状态`)
}
assert.notEqual(api.exadataStatusKey('1.0', 'STRING'), api.exadataStatusKey('1', 'STRING'))
const reviewed = ['metrics.v1.json', 'device_metrics.v1.json'].flatMap(file => JSON.parse(read(resolve(ammolite, 'services/aiops_agent/src/aiops_agent/resources/exadata_oem', file))).metrics)
const profiles = JSON.parse(read(resolve(ammolite, 'services/aiops_agent/src/aiops_agent/resources/exadata_oem/monitoring_profiles.v1.json'))).profiles
const chartAudit = { performance:0, status:0, configuration:0 }
for (const definition of reviewed.filter(item=>profiles.some(profile=>profile.metric_codes.includes(item.code)))) {
  const typed = {...definition,status:statusesByCode.get(definition.code)||null}
  const card = api.buildExadataCards({...overview, definitions:[typed],samples:[sample(typed,'25')],gaps:[]})[0]
  const section = api.exadataCardSection(card); chartAudit[section]++
  assert.equal(!!api.exadataChartKind(card,'LATEST_SNAPSHOT'), section==='performance', `${definition.code} 必须有明确图表或非图表决策`)
  const historical = api.buildExadataCards({...overview, query_mode:'HISTORY',definitions:[typed],samples:[sample(typed,'25'),sample(typed,'30',{observed_at:'2026-11-11T12:01:00Z'})],gaps:[]})[0]
  assert.equal(api.exadataChartKind(historical,'HISTORY'), section==='performance'&&typed.native_value_type==='NUMBER'?'line':null)
}
assert.deepEqual(chartAudit, {performance:233,status:28,configuration:29}, '覆盖8场景290个独立指标，不仅4个磁盘IO示例')
const monitoringPalette = {...palette,success:'green',warning:'yellow',danger:'red'}
assert.deepEqual(clone(api.exadataChartOption(dimensional,monitoringPalette,'bar').color),['blue','cyan','green','yellow','red','gray'], '使用Prometheus同序六色序列色板，非健康阈值')
for (const [code, statusDefinition] of statusesByCode) {
  const definition = { ...reviewed.find(item => item.code === code), status: statusDefinition }
  assert.ok(definition.selector, `${code} 必须属于真实指标目录`)
  for (const [raw, expected] of Object.entries(statusDefinition.values)) {
    const card = api.buildExadataCards({ ...overview, definitions: [definition], samples: [sample(definition, raw)], gaps: [] })[0]
    assert.equal(card.numeric, false, `${code} 即使原厂为 NUMBER 也不能当数值图表`)
    assert.equal(api.exadataStatusItems(card)[0].label, expected.label)
    assert.equal(api.exadataStatusItems(card)[0].tone, expected.tone)
    assert.equal(api.exadataChartOption(card, palette).series.length, 0)
    for (const quality of ['STALE', 'TRUNCATED', 'EMPTY']) {
      const stale = api.buildExadataCards({ ...overview, definitions: [definition], samples: [sample(definition, raw, { quality })], gaps: [] })[0]
      assert.equal(api.exadataStatusItems(stale)[0].tone, 'unknown', `${code}/${quality} 不得显示为可用`)
    }
  }
  for (const raw of ['unreviewed', '', null]) {
    const card = api.buildExadataCards({ ...overview, definitions: [definition], samples: [sample(definition, raw)], gaps: [] })[0]
    assert.equal(api.exadataStatusItems(card)[0].tone, 'unknown')
  }
}
for (const definition of reviewed.filter(item => !statusesByCode.has(item.code))) {
  const card = api.buildExadataCards({ ...overview, definitions: [definition], samples: [sample(definition, '1')], gaps: [] })[0]
  assert.equal(card.status, null, `${definition.code} 不因名称包含状态字样或类型是 STRING 而变成可用性`)
}
const mixedStates = api.buildExadataCards({ ...overview, samples: [sample(status[0], '1', { dimensions: { disk: 'a' } }), sample(status[0], '0', { dimensions: { disk: 'b' } }), sample(status[0], '2', { component_id: 'cell-b' })], gaps: [] })
assert.deepEqual(clone(api.exadataStatusItems(mixedStates[0]).map(item => item.label)), ['可用', '不可用'])
assert.equal(api.exadataStatusItems(mixedStates[1])[0].label, '未知')
const stateReadError = api.buildExadataCards({ ...overview, samples: [], gaps: [{ component_id: 'cell-a', metric_code: status[0].code, code: 'SOURCE_PROVIDER_ERROR', detail: '错误' }] })[0]
assert.equal(api.exadataStatusItems(stateReadError)[0].label, '读取失败')
for (const [quality, expected] of [['STALE', '旧快照'], ['TRUNCATED', '采样截断'], ['EMPTY', '无有效数据']]) assert.equal(api.buildExadataCards({ ...overview, samples: [sample(cpu, quality === 'EMPTY' ? null : '1', { quality })] })[0].quality, expected)
const empty = api.buildExadataCards({ ...overview, samples: [], gaps: [{ component_id: 'cell-a', metric_code: cpu.code, code: 'NO_DATA', detail: '无采样' }] })
assert.equal(empty.length, 1, '只为明确失败查询建空卡，不为未查询组件猜测卡片')
assert.equal(empty[0].percent, null)
const providerFailure = api.buildExadataCards({ ...overview, samples: [], gaps: [{ component_id: 'cell-a', metric_code: cpu.code, code: 'SOURCE_PROVIDER_ERROR', detail: 'OEM 服务端 HTTP 5xx' }] })[0]
assert.equal(providerFailure.value, '读取失败', '上游服务错误不能冒充无采样')
assert.equal(providerFailure.quality, 'OEM 服务错误')
assert.equal(providerFailure.percent, null)
assert.equal(providerFailure.tone, 'warning')
for (const code of ['SOURCE_RESPONSE_INVALID', 'SOURCE_AUTH_FAILED', 'SOURCE_EMCLI_UNAVAILABLE', 'SOURCE_EMCLI_TIMEOUT', 'SOURCE_BUSY']) {
  const failed = api.buildExadataCards({ ...overview, samples: [], gaps: [{ component_id: 'cell-a', metric_code: cpu.code, code, detail: '测试来源读取错误' }] })[0]
  assert.equal(failed.value, '读取失败', `${code} 不能冒充 OEM 空采样`)
  assert.equal(failed.quality, '读取错误')
  assert.equal(failed.percent, null)
}
assert.deepEqual(clone(other.KBotAIOpsExadataPresentation.buildExadataCards(overview)), clone(context.exports.buildExadataCards(overview)), '两端纯展示输出一致')

for (const definition of [cpu, status[0]]) {
  const unsupported = api.buildExadataCards({...overview, samples:[], gaps:[{component_id:'cell-a', metric_code:definition.code, code:'SOURCE_QUERY_UNSUPPORTED', detail:'原厂查询不支持'}]})[0]
  assert.equal(unsupported.value, '不支持读取')
  assert.equal(unsupported.quality, '不支持读取')
  assert.equal(unsupported.tone, 'empty')
  if (unsupported.status) assert.equal(api.exadataStatusItems(unsupported)[0].label, '不支持读取')
}
assert.equal(api.exadataGapLabel('SOURCE_QUERY_UNSUPPORTED'), '不支持读取')
assert.equal(api.exadataGapLabel('SOURCE_PROVIDER_ERROR'), '查询失败')
assert.equal(api.exadataCardSection(cards[3]), 'performance')
assert.equal(api.exadataCardSection(cards[0]), 'status')
for (const selector of ['Cell_Metric/temperature_lower','Cell_Metric/temperature_upper','Cell_Metric/cl_iorm_boost','Cell_Metric/iorm_mode']) {
  assert.equal(api.exadataCardSection(api.buildExadataCards({...overview, definitions:[metric(selector)], samples:[sample(metric(selector), '32')], gaps:[]})[0]), 'configuration')
}
assert.equal(api.exadataDeviceGroup('oracle_si_rack'), 'assets')
assert.equal(api.exadataDeviceGroup('host'), 'nodes')
assert.equal(api.exadataDeviceGroup('oracle_si_pdu'), 'equipment')
assert.deepEqual(clone(api.exadataSceneMembers(deviceView.members,'dbserver').map(item=>item.target_id)),['db-a'])
assert.equal(api.exadataDeviceScene('host'), 'dbserver')
assert.equal(api.exadataDeviceScene('oracle_exadata'), 'cell')
assert.equal(api.exadataHasTrend(cards[3]), false, '单点不得生成趋势')
assert.equal(api.exadataHasTrend(nullLatest), false, '一个有效点加空值不构成趋势')
for (const selector of ['Aggregated_Exadata_Diskgroup_Capacity_Metric/count', 'Aggregated_Exadata_Diskgroup_Capacity_Metric/size']) {
  const definition = metric(selector, 'STRING')
  const card = api.buildExadataCards({...overview, definitions:[definition], samples:[sample(definition, '9007199254740993')], gaps:[]})[0]
  assert.equal(api.exadataCardSection(card), 'performance', '真实 ASM 字符串容量保留容量卡')
  assert.equal(card.numeric, false, '不因布局强改原厂类型')
  assert.equal(api.exadataChartKind(card, 'LATEST_SNAPSHOT'), 'bar', '已审核的字符串容量可作近似绘图，原始文本仍保留')
  assert.equal(card.value, '9007199254740993')
}
const sceneFixtures = {
  dbserver: [[dbCPU, '%', ['27']], [dbMemory, '%', ['57']]],
  disk: ['avg_cd_io_load', 'avg_cd_io_util', 'avg_cd_large_read_iops', 'avg_cd_large_read_throughput'].map((name,index) => [metric('Aggregated_HardNFlashDisk_Metric/'+name), ['%','%','I/O per second','MB per second'][index], [['2','2'],['24','24'],['360','180'],['96','48']][index], 'celldisk_type', ['FlashDisk','HardDisk']]),
  flash: [[metric('FlashCache_Metric/default_hits_rate'), '%', ['92']], [metric('FlashCache_Metric/fc_read_throughput'), 'MB per second', ['100','20'], 'object_name', ['flash-a','flash-b']]],
  iorm: [[metric('IORMDB_Metric/db_iops'), 'I/O per second', ['120','30'], 'db_name', ['ORCL','TEST']], [metric('IORMDB_Metric/db_io_wt_sm_rq'), 'ms', ['0','3'], 'db_name', ['ORCL','TEST']]],
  network: [[metric('NetworkPortPerformance/InOctetsRate','NUMBER','oracle_si_network_port_switch'), 'bytes per second', ['1024','2048'], 'PortId', ['port-1','port-2']]],
  capacity: [[metric('Aggregated_Exadata_Capacity_Metric/Usage'), '%', ['42']], [metric('Aggregated_Exadata_Capacity_Metric/size'), 'GB', ['900','300'], 'celldisk_type', ['HardDisk','FlashDisk']], [metric('Aggregated_Exadata_Diskgroup_Capacity_Metric/size','STRING'), 'GB', ['9007199254740993']], [metric('Aggregated_Exadata_Diskgroup_Capacity_Metric/count','STRING'), 'n/a', ['12']]],
}
for (const [scene, rows] of Object.entries(sceneFixtures)) for (const [definition,unit,values,dimension,labels] of rows) {
  const card = api.buildExadataCards({...overview, definitions:[definition],samples:values.map((value,index)=>sample(definition,value,{unit,dimensions:dimension?{[dimension]:labels[index]}:{}})),gaps:[]})[0]
  assert.ok(api.exadataChartKind(card,'LATEST_SNAPSHOT'), `${scene}/${definition.selector} 应有数值图表`)
  if (scene === 'disk' || scene === 'iorm') assert.equal(api.exadataChartKind(card,'LATEST_SNAPSHOT'),'bar')
}
const incomplete = api.buildExadataCards({...overview,samples:[],gaps:[{component_id:'cell-a',metric_code:cpu.code,code:'COLLECTION_TIMEOUT',detail:'未完成查询'}]})[0]
assert.equal(incomplete.value, '未完成查询', '预算超时不是 OEM 空采样')

const artifacts = mkdtempSync(resolve(tmpdir(), 'aiops-exadata-cards-'))
const staticFiles = Object.fromEntries(['aiops-auth.js', 'aiops-exadata-presentation.js', 'aiops-exadata-metrics.js'].map(name => [`/__kbot/${name}`, resolve(kbot, 'ui/aiops/js', name)]))
for (const name of ['aiops.css', 'ammolite-shared.css', 'reference-monitoring.css', 'alignment.css', 'aiops-exadata-metrics.css']) staticFiles[`/__kbot/${name}`] = resolve(kbot, 'ui/aiops/css', name)
staticFiles['/__kbot/echarts.js'] = resolve(kbot, 'ui/vendor/echarts.min.js')
const portProbe = createNetServer()
await new Promise(resolve => portProbe.listen(0, '127.0.0.1', resolve))
const freePort = portProbe.address().port
await new Promise(resolve => portProbe.close(resolve))
const server = await createServer({ root: portal, configFile: resolve(portal, 'vite.config.ts'), server: { host: '127.0.0.1', port: freePort }, plugins: [{ name: 'exadata-contract-fixture', configureServer(server) {
  server.middlewares.use(async (req, res, next) => {
    const path = req.url?.split('?')[0]
    if (staticFiles[path]) { res.setHeader('Content-Type', path.endsWith('.css') ? 'text/css' : 'text/javascript'); res.end(read(staticFiles[path])); return }
    if (path !== '/__exadata-fixture') return next()
    const html = isKbot ? `<html><head>${['aiops.css', 'ammolite-shared.css', 'reference-monitoring.css', 'alignment.css', 'aiops-exadata-metrics.css'].map(name => `<link rel="stylesheet" href="/__kbot/${name}">`).join('')}</head><body data-page="monitoring"><main id="app" style="padding:20px"><section id="fixture"></section></main><script>window.KBOT_UI_CONFIG={mainApiBaseUrl:location.origin}; sessionStorage.setItem('kbot.aiops.session.v1',JSON.stringify({access_token:'fixture-not-a-real-token'})); window.KBotAIOpsShell={escape:value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt:value=>value};</script><script src="/__kbot/echarts.js"></script><script src="/__kbot/aiops-auth.js"></script><script src="/__kbot/aiops-exadata-presentation.js"></script><script src="/__kbot/aiops-exadata-metrics.js"></script><script>window.fixtureComponent=new KBotAIOpsExadataMetrics(document.querySelector('#fixture'),{systemId:'system',sourceId:'source',instanceId:'resource'});</script></body></html>`
      : `<html><head></head><body><main id="app" style="padding:20px"></main><script type="module">import{createApp}from'vue';import Component from '/src/features/aiops/business/AIOpsExadataMetrics.vue';import '/src/shared/ui/styles.css';window.fixtureApp=createApp(Component,{domainCode:'test',domainId:'domain',systemId:'system',sourceId:'source',resourceId:'resource',refresh:0});window.fixtureApp.mount('#app');</script></body></html>`
    res.setHeader('Content-Type', 'text/html; charset=utf-8'); res.end(await server.transformIndexHtml(req.url, html.replace('<head>', '<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">')))
  })
} }] })
await server.listen()
const address = server.httpServer.address(), origin = `http://127.0.0.1:${address.port}`
const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH })
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } }), errors = [], calls = []
  async function assertCardSpacing() {
    const spacing = await page.locator('.exadata-metric-card').evaluateAll(nodes => nodes.map(card => {
      const rect = card.getBoundingClientRect(), chart = card.querySelector('.exadata-chart')?.getBoundingClientRect()
      const padding = selector => {
        const css = getComputedStyle(card.querySelector(selector))
        return [parseFloat(css.paddingLeft), parseFloat(css.paddingRight)]
      }
      const evidence = card.querySelector(':scope > .exadata-evidence'), css = getComputedStyle(evidence)
      return { header: padding(':scope > header'), value: padding('.exadata-card-value'), time: padding('.exadata-card-time'),
        caption: card.querySelector('.exadata-chart-caption') ? padding('.exadata-chart-caption') : null,
        evidence: padding(':scope > .exadata-evidence'), evidenceBlock: [parseFloat(css.paddingTop), parseFloat(css.paddingBottom)],
        marker: getComputedStyle(evidence.querySelector('summary')).listStylePosition,
        chart: chart ? [chart.left - rect.left, rect.right - chart.right] : null }
    }))
    assert.ok(spacing.length)
    for (const card of spacing) {
      assert.ok([card.header, card.value, card.time, card.caption, card.evidence, card.chart].filter(Boolean).flat().every(value => value >= 16), `卡片标题、内容、图表和入口距离边框至少16px：${JSON.stringify(card)}`)
      assert.ok(card.evidenceBlock.every(value => value >= 12), '查看数据入口上下留白至少12px')
      assert.equal(card.marker, 'inside', '展开标记不得突出到卡片边框')
    }
  }
  page.on('pageerror', error => errors.push(error.message))
  await page.route('https://fonts.**', route => route.abort())
  let fail = false, unavailable = false, providerError = false, stateFixture = false, devicesFixture = false, baselineFixture = false, baselineFailure = false, slowWeek = false
  let historyFailure = false, historyIdentityMismatch = false
  await page.route('**/api/v1/apps/aiops/**', async route => {
    const request = route.request(), body = request.postDataJSON(); calls.push({ body, headers: request.headers(), url: request.url() })
    if (!request.headers()['content-type']?.includes('application/json')) return route.fulfill({ status: 422, json: { detail: '请求体必须为 JSON' } })
    if (fail) return route.fulfill({ status: 502, json: { detail: '模拟来源读取失败' } })
    if (body.mode === 'HISTORY' && historyFailure) return route.fulfill({ status: 502, json: { detail: '模拟历史读取失败' } })
    if (request.url().endsWith('/baselines')) {
      if (baselineFailure) return route.fulfill({ status: 502, json: { detail: '模拟历史均值失败，当前采样保留' } })
      if (slowWeek && body.period === '1w') await new Promise(resolve => setTimeout(resolve, 400))
      const memory = body.metric_code.endsWith('memory_utilization')
      return route.fulfill({ json: { component_id: body.component_id, metric_code: body.metric_code, binding_id: 'binding', binding_version: 1, source_version: 2, catalog_sha256: 'c'.repeat(64), unit: '%', captured_at: overview.captured_at,
        baseline: { period: body.period, start: '2026-10-12T12:00:00Z', end: overview.captured_at, unavailable_reason: null, note: '留存均值' },
        comparisons: [{ dimensions: {}, observed_at: overview.samples[3].observed_at, current: memory ? '0' : '27.210000000000001', average: memory ? '0' : '20', sample_count: 7, difference: memory ? '0' : '7.210000000000001', deviation_percent: memory ? null : '36.050000000000005' }] } })
    }
    let result = clone(overview)
    if (sceneFixtures[body.scene] && !baselineFixture) {
      const rows = sceneFixtures[body.scene]
      result.definitions = rows.map(row => row[0])
      result.samples = rows.flatMap(([definition,unit,values,dimension,labels]) => values.map((value,index)=>sample(definition,value,{unit,dimensions:dimension?{[dimension]:labels[index]}:{}})))
      result.gaps = []; result.total_queries = rows.length; result.next_offset = null
    }
    if (body.scene === 'cell') {
      result.definitions = [...overview.definitions, ...Array.from({ length: 7 }, (_, index) => metric(`Cell_Metric/fixture_${index}`))]
      result.samples = [...overview.samples, ...result.definitions.slice(5).map((item, index) => sample(item, `${index + 1}`, { unit: null }))]
      result.total_queries = 24; result.next_offset = body.offset ? null : 12
    }
    if (body.component_id) result.samples = result.samples.filter(item => item.component_id === body.component_id)
    if (body.mode === 'HISTORY') {
      result.samples = result.samples.filter(item => item.native_value_type === 'NUMBER').flatMap(item => [{...item, value:'8.25'}, {...item, value:null, observed_at:'2026-11-11T12:00:30Z', quality:'EMPTY'}, {...item, value:'13.75', observed_at:'2026-11-11T12:01:00Z'}])
      result.gaps.push(...status.map(item => ({ component_id: 'cell-a', metric_code: item.code, code: 'SOURCE_METRIC_HISTORY_UNSUPPORTED', detail: '字符串历史不支持' })))
    }
    if (stateFixture) {
      result.definitions = reviewed.filter(item => statusesByCode.has(item.code)).map(item => ({ ...item, status: statusesByCode.get(item.code) }))
      result.samples = result.definitions.map(item => sample(item, Object.keys(item.status.values)[0] || 'unreviewed', { unit: 'n/a', dimensions: Object.fromEntries(item.keys.map(key => [key, `${key}-A`])) }))
      result.gaps = []; result.total_queries = statusesByCode.size; result.next_offset = null
    }
    if (devicesFixture) {
      result = clone(deviceView)
      if (body.component_id) {
        result.samples = result.samples.filter(item=>item.component_id===body.component_id)
        result.device_coverage = result.device_coverage.filter(item=>item.component_id===body.component_id)
      }
    }
    if (unavailable) { result.samples = []; result.gaps.push({ component_id: 'cell-a', metric_code: cpu.code, code: 'NO_DATA', detail: '无采样，不按零值显示' }) }
    if (providerError) { result.definitions = overview.definitions; result.samples = []; result.gaps = [{ component_id: 'cell-a', metric_code: cpu.code, code: 'SOURCE_PROVIDER_ERROR', detail: 'OEM 服务端 HTTP 5xx' }] }
    result.query_mode = body.mode
    if (baselineFixture) {
      Object.assign(result, { binding_id: 'binding', binding_version: 1, source_version: 2, catalog_sha256: 'c'.repeat(64) })
      result.definitions = result.definitions.map(item => ({ ...item, baseline_supported: [cpu.code, memory.code].includes(item.code) }))
    }
    if (request.url().endsWith('/diagnoses')) result = { report_id: 'fixture-report', status: 'INCONCLUSIVE', evidence: result, findings: [{ code: 'THRESHOLDS_UNVERIFIED', detail: '不能判定健康' }] }
    if (body.mode === 'HISTORY' && historyIdentityMismatch) result.binding_version = 99
    return route.fulfill({ json: result })
  })
  await page.goto(`${origin}/__exadata-fixture`, { waitUntil: 'networkidle' })
  await expect(page.locator('.exadata-metric-card')).toHaveCount(2)
  await expect(page.locator('.exadata-status-row')).toHaveCount(3)
  await expect(page.locator('.exadata-status-value')).toHaveText(['可用', '可用', '可用'])
  await expect(page.locator('.exadata-status-panel .exadata-card-value')).toHaveCount(0)
  assert.ok(await page.locator('.exadata-status-row').first().evaluate(node => node.getBoundingClientRect().height < 100), '状态采用紧凑行，不使用大卡片')
  await expect(page.locator('.exadata-percent')).toHaveCount(0)
  await expect(page.locator('.exadata-chart[data-chart-kind="line"] canvas')).toHaveCount(2)
  await expect(page.locator('.exadata-boundary')).toHaveCount(0)
  await expect(page.locator('.exadata-oem')).not.toContainText('不把采集成功视为健康')
  await expect(page.locator('.exadata-evidence[open]')).toHaveCount(0)
  await expect(page.locator('.exadata-card-value strong').nth(0)).toHaveText('27.210000000000001')
  await expect(page.locator('.exadata-card-value strong').nth(1)).toHaveText('0')
  await expect(page.locator('.exadata-status-value.state-success')).toHaveCount(3)
  await expect(page.locator('.status-success')).toHaveCount(0)
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-latest.png`), fullPage: true })
  const cpuCard = page.locator(`[data-metric-code="${cpu.code}"]`)
  await cpuCard.locator('summary').click(); await expect(cpuCard.locator('tbody tr').first()).toBeVisible()
  await expect(cpuCard.locator('tbody')).toContainText('27.210000000000001')
  await cpuCard.locator('summary').click()
  for (const [width, columns] of [[1920, 4], [1440, 3], [1000, 2], [780, 1], [390, 1]]) {
    await page.setViewportSize({ width, height: 1000 })
    const layout = await page.locator('.exadata-metric-grid').evaluate(node => ({ columns: getComputedStyle(node).gridTemplateColumns.split(' ').length, overflow: document.documentElement.scrollWidth - innerWidth, cards: [...node.children].map(card => ({ width: card.getBoundingClientRect().width, border: getComputedStyle(card).borderTopWidth, background: getComputedStyle(card).backgroundColor })) }))
    assert.equal(layout.columns, columns, `宽度 ${width} 下卡片列数`)
    if (layout.overflow > 1) {
      console.log(await page.evaluate(() => [...document.querySelectorAll('body,main,section,header,div,label,select,p,small')].filter(node => node.getBoundingClientRect().width > innerWidth).map(node => ({ tag: node.tagName, class: node.className, width: node.getBoundingClientRect().width, min: getComputedStyle(node).minWidth, text: node.textContent.slice(0, 45) })).slice(0, 20)))
    }
    assert.ok(layout.overflow <= 1, `宽度 ${width} 不得产生横向溢出：${JSON.stringify(layout)}`)
    assert.ok(layout.cards.every(card => card.width > 200 && card.border === '2px' && card.background !== 'rgba(0, 0, 0, 0)'), '卡片具有真实样式，不是裸表格')
    await assertCardSpacing()
  }
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-mobile.png`), fullPage: true })
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.locator('[data-mode]').selectOption('HISTORY')
  await expect(page.locator('.exadata-chart canvas')).toHaveCount(2)
  await expect(page.locator('.exadata-boundary')).toHaveCount(0)
  await expect(page.locator('.exadata-oem')).not.toContainText('OEM 留存分辨率；不推断精确增量或跨组件均值。')
  await expect(page.locator('.exadata-oem')).not.toContainText('不把采集成功视为健康')
  await expect(page.locator('.exadata-percent')).toHaveCount(0)
  await expect(page.locator('.exadata-metric-card')).toHaveCount(2)
  await expect(page.locator('.exadata-status-row')).toHaveCount(3)
  await expect(page.locator('.exadata-status-value')).toHaveText(['历史不支持', '历史不支持', '历史不支持'])
  await expect(page.locator('.exadata-status-value.state-success')).toHaveCount(0)
  const chart = await cpuCard.locator('.exadata-chart').evaluate(node => ({ width: node.clientWidth, height: node.clientHeight }))
  assert.ok(chart.width > 200 && chart.height >= 128)
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-history.png`), fullPage: true })
  await page.locator('[data-mode]').selectOption('LATEST_SNAPSHOT')
  for (const [scene,rows] of Object.entries(sceneFixtures)) {
    await page.locator(`[data-scene-id="${scene}"]`).click()
    await expect(page.locator('.exadata-metric-card')).toHaveCount(rows.length)
    await expect(page.locator('.exadata-chart canvas')).toHaveCount(rows.length)
    await expect(page.locator('.exadata-metric-card > .exadata-baseline')).toHaveCount(0)
    await expect(page.locator('.exadata-evidence[open]')).toHaveCount(0)
    await expect(page.locator('.exadata-metric-card > footer')).toHaveCount(0)
    if (scene === 'disk') {
      await expect(page.locator('.exadata-chart[data-chart-kind="bar"]')).toHaveCount(4)
      await page.locator('.exadata-metric-card summary').nth(2).click()
      await expect(page.locator('.exadata-metric-card').nth(2).locator('tbody')).toContainText('360')
      await expect(page.locator('.exadata-metric-card').nth(2).locator('tbody')).toContainText('celldisk_type=FlashDisk')
      await page.locator('.exadata-metric-card summary').nth(2).click()
    }
    for (const width of [390,1440]) {
      await page.setViewportSize({width,height:1000})
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth<=1), `${scene} 图表不横向溢出`)
    }
    await page.screenshot({path:resolve(artifacts,`${isKbot?'kbot':'ammolite'}-${scene}-charts.png`),fullPage:true})
    if (scene === 'disk') {
      await page.evaluate(()=>document.documentElement.dataset.theme='dark')
      await page.locator('[data-refresh]').click()
      await expect(page.locator('.exadata-chart canvas')).toHaveCount(4)
      await page.screenshot({path:resolve(artifacts,`${isKbot?'kbot':'ammolite'}-disk-dark.png`),fullPage:true})
      await page.evaluate(()=>document.documentElement.dataset.theme='light')
    }
    await page.locator('[data-mode]').selectOption('HISTORY')
    const numericCount = rows.filter(row=>row[0].native_value_type==='NUMBER').length
    await expect(page.locator('.exadata-chart[data-chart-kind="line"] canvas')).toHaveCount(numericCount)
    await page.locator('[data-mode]').selectOption('LATEST_SNAPSHOT')
  }
  await page.locator('[data-mode]').selectOption('HISTORY')
  await page.locator('[data-scene-id="cell"]').click()
  await expect(page.locator('.exadata-metric-card')).toHaveCount(9)
  await expect(page.locator('.exadata-chart canvas')).toHaveCount(9)
  await page.locator('[data-next]').click(); await expect(page.locator('[data-next]')).toBeDisabled()
  assert.equal(calls.at(-1).body.offset, 12)
  await page.locator('[data-first]').click(); await expect(page.locator('[data-next]')).toBeEnabled()
  await page.locator('[data-component]').selectOption('cell-a'); await expect(page.locator('.exadata-oem')).toHaveAttribute('aria-busy', 'false')
  assert.equal(calls.at(-1).body.component_id, 'cell-a'); assert.equal(calls.at(-1).body.offset, 0)
  await page.locator('[data-diagnose]').click(); await expect(page.locator('.exadata-report')).toContainText('不能判定健康')
  assert.ok(calls.at(-1).headers['idempotency-key'])
  unavailable = true; await page.locator('[data-refresh]').click()
  await expect(page.locator('.exadata-card-value strong').filter({ hasText: '无数据' })).toHaveCount(1)
  await expect(page.locator('.exadata-percent')).toHaveCount(0)
  unavailable = false; stateFixture = true
  await page.locator('[data-mode]').selectOption('LATEST_SNAPSHOT')
  await expect(page.locator('.exadata-status-item')).toHaveCount(statusesByCode.size)
  await expect(page.locator('.exadata-status-row')).toHaveCount(statusesByCode.size)
  await expect(page.locator('.exadata-metric-card')).toHaveCount(0)
  await expect(page.locator('.exadata-status-panel')).toContainText('端口掉线事件')
  await expect(page.locator('.exadata-status-panel')).toContainText('未触发')
  await expect(page.locator('.exadata-oem')).toContainText('未点亮')
  await expect(page.locator('.exadata-status-panel')).toContainText('PortId=PortId-A')
  await expect(page.locator('.exadata-status-panel')).toContainText('ComponentId=ComponentId-A')
  for (const width of [390, 780, 1440]) {
    await page.setViewportSize({ width, height: 1000 })
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth <= 1), `全部状态在宽度 ${width} 不得横向溢出`)
  }
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-all-states.png`), fullPage: true })
  stateFixture = false; devicesFixture = true
  await page.locator('[data-component]').selectOption('')
  await page.locator('[data-scene-id="overview"]').click()
  await expect(page.locator('[data-device-id]')).toHaveCount(6)
  await expect(page.locator('[data-device-id="db-a"]')).toContainText('DB Server')
  await expect(page.locator('[data-device-id="db-a"]')).toContainText('37.125')
  await expect(page.locator('[data-device-id="switch-a"]')).toContainText('可达')
  await expect(page.locator('[data-device-id="pdu-a"]')).toContainText('本页未查询')
  await expect(page.locator('[data-device-id="pdu-a"] .state-success')).toHaveCount(0)
  await expect(page.locator('[data-device-id="rack-a"]')).toContainText('机架')
  for (const width of [390, 780, 1440]) {
    await page.setViewportSize({ width, height: 1000 })
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth <= 1), '整机设备清单不横向溢出')
    await assertCardSpacing()
    const insets = await page.locator('.exadata-resource-row').evaluateAll(rows => rows.map(row => {
      const css = getComputedStyle(row)
      return [parseFloat(css.paddingLeft), parseFloat(css.paddingRight), parseFloat(css.paddingBottom)]
    }))
    assert.ok(insets.flat().every(value => value >= 16), '整机嵌套卡片与设备边框留白至少16px')
  }
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-devices.png`), fullPage: true })
  await page.evaluate(() => document.documentElement.dataset.theme = 'dark')
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-devices-dark.png`), fullPage: true })
  await page.locator('.exadata-assets > summary').click()
  await expect(page.locator('[data-device-id="rack-a"]')).toBeVisible()
  await page.locator('.exadata-assets > summary').click()
  await page.evaluate(() => document.documentElement.dataset.theme = 'light')
  await expect(page.locator('.exadata-metric-card')).toHaveCount(4)
  await expect(page.locator('.exadata-chart[data-chart-kind="line"] canvas')).toHaveCount(4)
  await expect(page.locator('[data-device-group="nodes"]')).toContainText('SIM DB SERVER')
  await expect(page.locator('[data-device-group="equipment"]')).toContainText('SIM SWITCH')
  await expect(page.locator('.exadata-assets')).not.toHaveAttribute('open', '')
  assert.equal(await page.locator('[data-resource-code="' + dbMemory.code + '"]').count(), 1, '概览只呈现一次节点内存')
  await page.locator('[data-device="db-a"]').click()
  await expect(page.locator('.exadata-detail')).toContainText('SIM DB SERVER')
  assert.equal(calls.at(-1).body.scene, 'dbserver')
  assert.equal(calls.at(-1).body.component_id, 'db-a')
  await page.locator('[data-scene-id="overview"]').click()
  await page.locator('[data-device="switch-a"]').click()
  await expect(page.locator('.exadata-detail')).toContainText('SIM SWITCH')
  await expect(page.locator('[data-device-id]')).toHaveCount(0)
  assert.equal(calls.at(-1).body.component_id, 'switch-a')
  assert.equal(calls.at(-1).body.offset, 0)
  devicesFixture = false
  providerError = true; unavailable = false; await page.locator('[data-refresh]').click()
  await expect(page.locator('.exadata-metric-card')).toHaveCount(1)
  await expect(page.locator('.exadata-metric-card')).toContainText('OEM 服务错误')
  await expect(page.locator('.exadata-card-value strong')).toHaveText('读取失败')
  await expect(page.locator('.exadata-chart')).toHaveCount(0)
  fail = true; await page.locator('[data-refresh]').click()
  await expect(page.locator('[role="alert"]')).toContainText('模拟来源读取失败')
  await expect(page.locator('.exadata-metric-card')).toHaveCount(0)
  await expect(page.locator('.exadata-status-row')).toHaveCount(0)
  fail = false; providerError = false; baselineFixture = true
  await page.locator('[data-component]').selectOption('')
  await expect(page.locator('.exadata-metric-card')).toHaveCount(2)
  await expect(page.locator('.exadata-baseline')).toHaveCount(2)
  await expect(cpuCard.locator('.exadata-baseline')).toContainText('7 天留存均值 20.00')
  await expect(cpuCard.locator('.exadata-baseline')).toContainText('偏离 36.05%')
  await expect(cpuCard.locator('.exadata-baseline')).not.toContainText('36.050000000000005')
  await expect(page.locator(`[data-metric-code="${memory.code}"] .exadata-baseline`)).toContainText('均值为零，相对偏离不可定义')
  assert.ok(calls.filter(call => call.url.endsWith('/baselines')).every(call => [cpu.code, memory.code].includes(call.body.metric_code)), '状态不读取历史均值')
  await page.locator('[data-baseline]').selectOption('1m')
  await expect(cpuCard.locator('.exadata-baseline')).toContainText('30 天留存均值')
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-baseline.png`), fullPage: true })
  slowWeek = true
  await page.locator('[data-baseline]').selectOption('1w')
  await page.locator('[data-baseline]').selectOption('1m')
  await expect(cpuCard.locator('.exadata-baseline')).toContainText('30 天留存均值')
  await page.waitForTimeout(500)
  await expect(cpuCard.locator('.exadata-baseline')).not.toContainText('7 天留存均值')
  baselineFailure = true
  await page.locator('[data-baseline]').selectOption('1w')
  await expect(cpuCard.locator('.exadata-baseline')).toContainText('模拟历史均值失败')
  await expect(cpuCard.locator('.exadata-card-value strong')).toHaveText('27.210000000000001')
  await page.locator('[data-baseline]').selectOption('')
  await expect(page.locator('.exadata-baseline')).toHaveCount(0)
  historyFailure = true
  await page.locator('[data-refresh]').click()
  await expect(cpuCard.locator('.exadata-chart-caption')).toContainText('历史趋势读取失败')
  await expect(cpuCard.locator('.exadata-card-value strong')).toHaveText('27.210000000000001')
  await expect(cpuCard.locator('.exadata-evidence > summary')).toContainText('1 条')
  historyFailure = false; historyIdentityMismatch = true
  await page.locator('[data-refresh]').click()
  await expect(cpuCard.locator('.exadata-chart-caption')).toContainText('资源或目录已变化')
  await expect(cpuCard.locator('.exadata-evidence > summary')).toContainText('1 条')
  historyIdentityMismatch = false
  await page.locator('[data-refresh]').click()
  await expect(cpuCard.locator('.exadata-evidence > summary')).toContainText('4 条')
  await expect(cpuCard.locator('tbody tr')).toHaveCount(4)
  await expect(cpuCard.locator('tbody tr').first().locator('td')).toHaveCount(5)
  await expect(cpuCard.locator('tbody tr').first().locator('td').first()).toHaveText('当前快照')
  await expect(cpuCard.locator('tbody tr').nth(1).locator('td').first()).toHaveText('历史采样')
  assert.ok(calls.every(call => call.headers['content-type']?.includes('application/json')), `使用原有请求封装的 JSON 合同：${JSON.stringify(calls.map(call => ({url:call.url,type:call.headers['content-type']})))}`)
  assert.deepEqual(errors, [])
  await page.close()
  console.log(JSON.stringify({ ok: true, repository: isKbot ? 'kbot' : 'ammolite', semantics: 'passed', browser: 'passed', widths: [1920, 1440, 1000, 780, 390], artifacts }))
} finally { await browser.close(); await server.close() }
