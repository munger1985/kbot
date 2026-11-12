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
const metric = (selector, type = 'NUMBER') => ({ code: `oracle_exadata.${selector.replace('/', '.')}`, selector, display_name: selector, native_value_type: type })
const cpu = metric('Cell_Metric/cpu_utilization'), memory = metric('Cell_Metric/memory_utilization')
const status = ['Status', 'msStatus', 'rsStatus'].map(name => metric(`CellSrv_Status_Metric/${name}`, 'STRING'))
const sample = (definition, value, overrides = {}) => ({ component_id: 'cell-a', metric_code: definition.code, selector: definition.selector, observed_at: '2026-11-11T12:00:00Z', value, dimensions: {}, unit: '%', native_value_type: definition.native_value_type, quality: 'OBSERVED', ...overrides })
const overview = { query_mode: 'LATEST_SNAPSHOT', window_start: null, window_end: null, members, definitions: [...status, cpu, memory], samples: [...status.map(item => sample(item, '1', { unit: 'n/a' })), sample(cpu, '27.210000000000001'), sample(memory, '0')], gaps: [{ code: 'THRESHOLDS_UNVERIFIED', detail: '阈值与状态字典未验证，不判定健康。' }], total_queries: 5, next_offset: null, catalog_version: 'exadata-oem-v1', captured_at: '2026-11-11T12:01:00Z' }
const cards = api.buildExadataCards(overview)
assert.equal(cards.length, 5)
assert.equal(cards[3].value, '27.210000000000001', '原始精度不得被图表数值转换覆盖')
assert.equal(cards[4].value, '0'); assert.equal(cards[4].percent, 0, '真零值必须可展示')
assert.equal(cards[0].quality, '原始状态'); assert.equal(cards[0].value, '1', '未知状态字典不得把 1 翻译为正常')
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
for (const [quality, expected] of [['STALE', '旧快照'], ['TRUNCATED', '采样截断'], ['EMPTY', '无有效数据']]) assert.equal(api.buildExadataCards({ ...overview, samples: [sample(cpu, quality === 'EMPTY' ? null : '1', { quality })] })[0].quality, expected)
const empty = api.buildExadataCards({ ...overview, samples: [], gaps: [{ component_id: 'cell-a', metric_code: cpu.code, code: 'NO_DATA', detail: '无采样' }] })
assert.equal(empty.length, 1, '只为明确失败查询建空卡，不为未查询组件猜测卡片')
assert.equal(empty[0].percent, null)
const providerFailure = api.buildExadataCards({ ...overview, samples: [], gaps: [{ component_id: 'cell-a', metric_code: cpu.code, code: 'SOURCE_PROVIDER_ERROR', detail: 'OEM 服务端 HTTP 5xx' }] })[0]
assert.equal(providerFailure.value, '读取失败', '上游服务错误不能冒充无采样')
assert.equal(providerFailure.quality, 'OEM 服务错误')
assert.equal(providerFailure.percent, null)
assert.equal(providerFailure.tone, 'warning')
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
  let fail = false, unavailable = false, providerError = false
  await page.route('**/api/v1/apps/aiops/**', async route => {
    const request = route.request(), body = request.postDataJSON(); calls.push({ body, headers: request.headers(), url: request.url() })
    if (!request.headers()['content-type']?.includes('application/json')) return route.fulfill({ status: 422, json: { detail: '请求体必须为 JSON' } })
    if (fail) return route.fulfill({ status: 502, json: { detail: '模拟来源读取失败' } })
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
    if (unavailable) { result.samples = []; result.gaps.push({ component_id: 'cell-a', metric_code: cpu.code, code: 'NO_DATA', detail: '无采样，不按零值显示' }) }
    if (providerError) { result.samples = []; result.gaps = [{ component_id: 'cell-a', metric_code: cpu.code, code: 'SOURCE_PROVIDER_ERROR', detail: 'OEM 服务端 HTTP 5xx' }] }
    result.query_mode = body.mode
    if (request.url().endsWith('/diagnoses')) result = { report_id: 'fixture-report', status: 'INCONCLUSIVE', evidence: result, findings: [{ code: 'THRESHOLDS_UNVERIFIED', detail: '不能判定健康' }] }
    return route.fulfill({ json: result })
  })
  await page.goto(`${origin}/__exadata-fixture`, { waitUntil: 'networkidle' })
  await expect(page.locator('.exadata-metric-card')).toHaveCount(5)
  await expect(page.locator('.exadata-percent')).toHaveCount(2)
  await expect(page.locator('.exadata-chart')).toHaveCount(0)
  await expect(page.locator('.exadata-boundary')).toHaveCount(0)
  await expect(page.locator('.exadata-oem')).not.toContainText('不把采集成功视为健康')
  await expect(page.locator('.exadata-evidence[open]')).toHaveCount(0)
  await expect(page.locator('.exadata-card-value strong').nth(3)).toHaveText('27.210000000000001')
  await expect(page.locator('.exadata-card-value strong').nth(4)).toHaveText('0')
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
  await expect(page.locator('.exadata-metric-card')).toHaveCount(5)
  await expect(page.locator('.exadata-metric-card').filter({ hasText: '历史不支持' })).toHaveCount(3)
  const chart = await cpuCard.locator('.exadata-chart').evaluate(node => ({ width: node.clientWidth, height: node.clientHeight }))
  assert.ok(chart.width > 200 && chart.height >= 128)
  await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-history.png`), fullPage: true })
  await page.locator('[data-scene]').selectOption('cell')
  await expect(page.locator('.exadata-metric-card')).toHaveCount(12)
  await expect(page.locator('.exadata-chart canvas')).toHaveCount(9)
  await page.locator('[data-next]').click(); await expect(page.locator('[data-next]')).toBeDisabled()
  assert.equal(calls.at(-1).body.offset, 12)
  await page.locator('[data-first]').click(); await expect(page.locator('[data-next]')).toBeEnabled()
  await page.locator('[data-component]').selectOption('cell-a'); await expect(page.locator('.exadata-oem')).toHaveAttribute('aria-busy', 'false')
  assert.equal(calls.at(-1).body.component_id, 'cell-a'); assert.equal(calls.at(-1).body.offset, 0)
  await page.locator('[data-diagnose]').click(); await expect(page.locator('.exadata-report')).toContainText('不能判定健康')
  assert.ok(calls.at(-1).headers['idempotency-key'])
  unavailable = true; await page.locator('[data-refresh]').click()
  await expect(page.locator('.exadata-card-value strong').filter({ hasText: '无数据' })).toHaveCount(4)
  await expect(page.locator('.exadata-percent')).toHaveCount(0)
  providerError = true; unavailable = false; await page.locator('[data-refresh]').click()
  await expect(page.locator('.exadata-metric-card')).toHaveCount(1)
  await expect(page.locator('.exadata-metric-card')).toContainText('OEM 服务错误')
  await expect(page.locator('.exadata-card-value strong')).toHaveText('读取失败')
  await expect(page.locator('.exadata-chart')).toHaveCount(0)
  fail = true; await page.locator('[data-refresh]').click()
  await expect(page.locator('[role="alert"]')).toContainText('模拟来源读取失败')
  await expect(page.locator('.exadata-metric-card')).toHaveCount(0)
  assert.ok(calls.every(call => call.headers['content-type']?.includes('application/json')), `使用原有请求封装的 JSON 合同：${JSON.stringify(calls.map(call => ({url:call.url,type:call.headers['content-type']})))}`)
  assert.deepEqual(errors, [])
  await page.close()
  console.log(JSON.stringify({ ok: true, repository: isKbot ? 'kbot' : 'ammolite', semantics: 'passed', browser: 'passed', widths: [1920, 1440, 1000, 780, 390], artifacts }))
} finally { await browser.close(); await server.close() }
