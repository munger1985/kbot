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
const overview = { query_mode: 'LATEST_SNAPSHOT', window_start: null, window_end: null, members, definitions: [...status, cpu, memory], samples: [...status.map(item => sample(item, '1', { unit: 'n/a' })), sample(cpu, '27.210000000000001'), sample(memory, '0')], gaps: [{ code: 'THRESHOLDS_UNVERIFIED', detail: '阈值与状态字典未验证，不判定健康。' }], total_queries: 5, next_offset: null, catalog_version: 'exadata-oem-v1', captured_at: '2026-11-11T12:01:00Z' }

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
const separated = api.buildExadataCards({ ...overview, samples: [sample(cpu, '1', { dimensions: { disk: 'same', rack: 'a' } }), sample(cpu, '2', { dimensions: { rack: 'a', disk: 'same' }, observed_at: '2026-11-11T12:01:00Z' }), sample(cpu, '3', { component_id: 'cell-b' }), sample(cpu, '4', { unit: 'ms' })], gaps: [] })
assert.equal(separated.length, 3, '不同 Cell 和原生单位绝不合并')
assert.equal(separated[0].series.length, 1, '维度键顺序不改变序列身份')
const dimensional = api.buildExadataCards({ ...overview, samples: [sample(cpu, '2', { dimensions: { object_name: 'disk-a' } }), sample(cpu, '9', { dimensions: { object_name: 'disk-b' } })], gaps: [] })[0]
assert.equal(dimensional.series.length, 2)
assert.equal(dimensional.value, '2 个维度', '多维度不求和、不以最大值冒充整机值')
assert.equal(dimensional.percent, null)
assert.equal(api.exadataChartOption(dimensional, palette).series.length, 2)
const nullLatest = api.buildExadataCards({ ...overview, query_mode: 'HISTORY', samples: [sample(cpu, '8'), sample(cpu, null, { observed_at: '2026-11-11T12:01:00Z', quality: 'EMPTY' })] })[0]
assert.equal(nullLatest.value, '无数据', '末次空值不回填旧的非空值')
assert.equal(nullLatest.percent, null, '历史模式不生成当前百分比仪表')
const option = api.exadataChartOption(nullLatest, palette)
assert.deepEqual(clone(option.series[0].data.map(item => item.value[1])), [8, null])
assert.equal(option.series[0].connectNulls, false)
assert.equal(option.series[0].markLine, undefined, '未验证阈值不画健康分界线')
assert.ok(!option.tooltip.formatter([{ seriesName: '<img onerror=alert(1)>', data: { raw: '<script>', at: 'now' } }]).includes('<img'))
assert.equal(api.exadataChartOption(cards[0], palette).series.length, 0, '字符串状态不能转换成数值时序')
assert.equal(statusesByCode.size, 26, '全部状态类指标纳入审核，而不只是三个服务')
const reviewed = ['metrics.v1.json', 'device_metrics.v1.json'].flatMap(file => JSON.parse(read(resolve(ammolite, 'services/aiops_agent/src/aiops_agent/resources/exadata_oem', file))).metrics)
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
  page.on('pageerror', error => errors.push(error.message))
  await page.route('https://fonts.**', route => route.abort())
  let fail = false, unavailable = false, providerError = false, stateFixture = false, devicesFixture = false, baselineFixture = false, baselineFailure = false, slowWeek = false
  await page.route('**/api/v1/apps/aiops/**', async route => {
    const request = route.request(), body = request.postDataJSON(); calls.push({ body, headers: request.headers(), url: request.url() })
    if (!request.headers()['content-type']?.includes('application/json')) return route.fulfill({ status: 422, json: { detail: '请求体必须为 JSON' } })
    if (fail) return route.fulfill({ status: 502, json: { detail: '模拟来源读取失败' } })
    if (request.url().endsWith('/baselines')) {
      if (baselineFailure) return route.fulfill({ status: 502, json: { detail: '模拟历史均值失败，当前采样保留' } })
      if (slowWeek && body.period === '1w') await new Promise(resolve => setTimeout(resolve, 400))
      const memory = body.metric_code.endsWith('memory_utilization')
      return route.fulfill({ json: { component_id: body.component_id, metric_code: body.metric_code, binding_id: 'binding', binding_version: 1, source_version: 2, catalog_sha256: 'c'.repeat(64), unit: '%', captured_at: overview.captured_at,
        baseline: { period: body.period, start: '2026-10-12T12:00:00Z', end: overview.captured_at, unavailable_reason: null, note: '留存均值' },
        comparisons: [{ dimensions: {}, observed_at: overview.samples[3].observed_at, current: memory ? '0' : '27.210000000000001', average: memory ? '0' : '20', sample_count: 7, difference: memory ? '0' : '7.210000000000001', deviation_percent: memory ? null : '36.050000000000005' }] } })
    }
    let result = clone(overview)
    if (body.scene === 'cell') {
      result.definitions = [...overview.definitions, ...Array.from({ length: 7 }, (_, index) => metric(`Cell_Metric/fixture_${index}`))]
      result.samples = [...overview.samples, ...result.definitions.slice(5).map((item, index) => sample(item, `${index + 1}`, { unit: null }))]
      result.total_queries = 24; result.next_offset = body.offset ? null : 12
    }
    if (body.component_id) result.samples = result.samples.filter(item => item.component_id === body.component_id)
    if (body.mode === 'HISTORY') {
      const definitions = result.definitions.filter(item => item.native_value_type === 'NUMBER')
      result.samples = definitions.flatMap(item => [sample(item, '8.25'), sample(item, null, { observed_at: '2026-11-11T12:00:30Z', quality: 'EMPTY' }), sample(item, '13.75', { observed_at: '2026-11-11T12:01:00Z' })])
      result.gaps.push(...status.map(item => ({ component_id: 'cell-a', metric_code: item.code, code: 'SOURCE_METRIC_HISTORY_UNSUPPORTED', detail: '字符串历史不支持' })))
    }
    if (stateFixture) {
      result.definitions = reviewed.filter(item => statusesByCode.has(item.code)).map(item => ({ ...item, status: statusesByCode.get(item.code) }))
      result.samples = result.definitions.map(item => sample(item, Object.keys(item.status.values)[0] || 'unreviewed', { unit: 'n/a', dimensions: Object.fromEntries(item.keys.map(key => [key, `${key}-A`])) }))
      result.gaps = []; result.total_queries = 26; result.next_offset = null
    }
    if (devicesFixture) {
      result = clone(deviceView)
      if (body.component_id) {
        result.samples = result.samples.filter(item=>item.component_id===body.component_id)
        result.device_coverage = result.device_coverage.filter(item=>item.component_id===body.component_id)
      }
    }
    if (unavailable) { result.samples = []; result.gaps.push({ component_id: 'cell-a', metric_code: cpu.code, code: 'NO_DATA', detail: '无采样，不按零值显示' }) }
    if (providerError) { result.samples = []; result.gaps = [{ component_id: 'cell-a', metric_code: cpu.code, code: 'SOURCE_PROVIDER_ERROR', detail: 'OEM 服务端 HTTP 5xx' }] }
    result.query_mode = body.mode
    if (baselineFixture) {
      Object.assign(result, { binding_id: 'binding', binding_version: 1, source_version: 2, catalog_sha256: 'c'.repeat(64) })
      result.definitions = result.definitions.map(item => ({ ...item, baseline_supported: [cpu.code, memory.code].includes(item.code) }))
    }
    if (request.url().endsWith('/diagnoses')) result = { report_id: 'fixture-report', status: 'INCONCLUSIVE', evidence: result, findings: [{ code: 'THRESHOLDS_UNVERIFIED', detail: '不能判定健康' }] }
    return route.fulfill({ json: result })
  })
  await page.goto(`${origin}/__exadata-fixture`, { waitUntil: 'networkidle' })
  await expect(page.locator('.exadata-metric-card')).toHaveCount(2)
  await expect(page.locator('.exadata-status-row')).toHaveCount(3)
  await expect(page.locator('.exadata-status-value')).toHaveText(['可用', '可用', '可用'])
  await expect(page.locator('.exadata-status-panel .exadata-card-value')).toHaveCount(0)
  assert.ok(await page.locator('.exadata-status-row').first().evaluate(node => node.getBoundingClientRect().height < 100), '状态采用紧凑行，不使用大卡片')
  await expect(page.locator('.exadata-percent')).toHaveCount(2)
  await expect(page.locator('.exadata-chart')).toHaveCount(0)
  await expect(page.locator('.exadata-boundary')).toHaveCount(0)
  await expect(page.locator('.exadata-oem')).not.toContainText('不把采集成功视为健康')
  await expect(page.locator('.exadata-evidence[open]')).toHaveCount(0)
  await expect(page.locator('.exadata-card-value strong').nth(0)).toHaveText('27.210000000000001')
  await expect(page.locator('.exadata-card-value strong').nth(1)).toHaveText('0')
  await expect(page.locator('.exadata-status-value.state-success')).toHaveCount(3)
  await expect(page.locator('.status-success')).toHaveCount(0)
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-latest.png`), fullPage: true })
  const cpuCard = page.locator(`[data-metric-code="${cpu.code}"]`)
  await cpuCard.locator('summary').click(); await expect(cpuCard.locator('tbody tr')).toBeVisible()
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
  await page.locator('[data-scene]').selectOption('cell')
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
  await expect(page.locator('.exadata-status-item')).toHaveCount(26)
  await expect(page.locator('.exadata-status-row')).toHaveCount(26)
  await expect(page.locator('.exadata-metric-card')).toHaveCount(0)
  await expect(page.locator('.exadata-status-panel')).toContainText('端口掉线事件')
  await expect(page.locator('.exadata-status-panel')).toContainText('未触发')
  await expect(page.locator('.exadata-status-panel')).toContainText('未点亮')
  await expect(page.locator('.exadata-status-panel')).toContainText('PortId=PortId-A')
  await expect(page.locator('.exadata-status-panel')).toContainText('ComponentId=ComponentId-A')
  for (const width of [390, 780, 1440]) {
    await page.setViewportSize({ width, height: 1000 })
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth <= 1), `全部状态在宽度 ${width} 不得横向溢出`)
  }
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-all-states.png`), fullPage: true })
  stateFixture = false; devicesFixture = true
  await page.locator('[data-component]').selectOption('')
  await page.locator('[data-scene]').selectOption('overview')
  await expect(page.locator('[data-device-id]')).toHaveCount(6)
  await expect(page.locator('[data-device-id="db-a"]')).toContainText('DB Server')
  await expect(page.locator('[data-device-id="db-a"]')).toContainText('37.125')
  await expect(page.locator('[data-device-id="switch-a"]')).toContainText('可达')
  await expect(page.locator('[data-device-id="pdu-a"]')).toContainText('本页未查询')
  await expect(page.locator('[data-device-id="pdu-a"] .state-success')).toHaveCount(0)
  await expect(page.locator('[data-device-id="rack-a"]')).toContainText('尚无审核')
  for (const width of [390, 780, 1440]) {
    await page.setViewportSize({ width, height: 1000 })
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth <= 1), '整机设备清单不横向溢出')
  }
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-devices.png`), fullPage: true })
  await page.locator('[data-device="switch-a"]').click()
  await expect(page.locator('[data-device-id]')).toHaveCount(1)
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
  await expect(cpuCard.locator('.exadata-baseline')).toContainText('7 天留存均值 20')
  await expect(cpuCard.locator('.exadata-baseline')).toContainText('36.050000000000005%')
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
  assert.ok(calls.every(call => call.headers['content-type']?.includes('application/json')), `使用原有请求封装的 JSON 合同：${JSON.stringify(calls.map(call => ({url:call.url,type:call.headers['content-type']})))}`)
  assert.deepEqual(errors, [])
  await page.close()
  console.log(JSON.stringify({ ok: true, repository: isKbot ? 'kbot' : 'ammolite', semantics: 'passed', browser: 'passed', widths: [1920, 1440, 1000, 780, 390], artifacts }))
} finally { await browser.close(); await server.close() }
