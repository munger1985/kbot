// OEM 两端展示验收：使用生产组件与ECharts，采样为固定替身，不访问现场OEM。
import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { createServer as createNetServer } from 'node:net'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import vm from 'node:vm'
const root = fileURLToPath(new URL('../../', import.meta.url)), isKbot = existsSync(resolve(root, 'ui/aiops'))
const kbot = isKbot ? root : process.env.KBOT_ROOT || resolve(root, '../kbot4')
const ammolite = isKbot ? process.env.AMMOLITE_ROOT || resolve(root, '../ammolite_cube') : root
const portal = resolve(ammolite, 'apps/portal-web'), require = createRequire(resolve(portal, 'package.json'))
const { chromium, expect } = require('@playwright/test'), ts = require('typescript')
const read = path => readFileSync(path, 'utf8')
const context = { exports: {} }
const compiled = ts.transpileModule(read(resolve(portal, 'src/features/aiops/business/oemMonitoringPresentation.ts')), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
vm.runInNewContext(compiled, context)
const other = {}; vm.runInNewContext(read(resolve(kbot, 'ui/aiops/js/aiops-oem-presentation.js')), other)
const api = isKbot ? other.KBotAIOpsOEMPresentation : context.exports
const clone = value => JSON.parse(JSON.stringify(value))
const definition = 'services/aiops_agent/src/aiops_agent/resources/monitoring/oem_overview.v1.json'
const catalog = JSON.parse(read(resolve(ammolite, definition)))
assert.deepEqual(JSON.parse(read(resolve(kbot, definition))), catalog)
const specs = Object.values(catalog.profiles).flat().map(spec => ({ ...spec, mode: spec.mode || 'HISTORY', ranking: spec.ranking || 'DESC', baseline_supported: spec.metric?.native_value_type === 'NUMBER' && spec.baseline !== false }))
assert.equal(specs.length, 16)
const palette = { accent: 'blue', info: 'cyan', success: 'green', warning: 'orange', danger: 'red', muted: 'gray', surface: 'white', ink: 'black', line: 'gray' }
const points = [{ observed_at: '2026-11-12T00:00:00Z', value: 21, raw_value: '21', quality: 'GOOD' }, { observed_at: '2026-11-12T00:15:00Z', value: null, raw_value: null, quality: 'GOOD' }, { observed_at: '2026-11-12T00:30:00Z', value: 42.125, raw_value: '42.125000000000000001', quality: 'GOOD' }]
const series = { series_key: 'a', dimensions: {}, points, baseline: { average: 37.954293984962406, sample_count: 7, period: '1w' } }
for (const metric of ['CPUUsageSummary/OverAllUsagePct', 'Load/memUsedPct']) {
  const panel = { metric_code: metric, title: metric, unit: 'percent', native: { mode: 'HISTORY' }, series: [series] }
  assert.equal(api.oemChartKind(panel), 'line')
  const option = api.oemChartOption(panel, palette, '7 天')
  assert.equal(option.series[0].type, 'line'); assert.equal(option.series[0].connectNulls, false)
  assert.deepEqual(clone(option.series[0].data.map(point => point.value[1])), [21, null, 42.125])
  assert.equal(option.series[0].data.at(-1).raw, '42.125000000000000001', 'tooltip保留OEM原值')
  assert.equal(option.series[0].markLine.data[0].yAxis, series.baseline.average)
  assert.equal(option.series[0].markLine.label.formatter, '7 天均值 37.95')
  assert.equal(option.series[0].markLine.data[0].lineStyle.type, 'dashed')
  assert.equal(api.oemChartOption({ ...panel, series: [{ ...series, baseline: null }] }, palette).series[0].markLine, undefined)
  const invalid = { ...panel, series: [{ ...series, points: [points[0], { ...points[2], value: null, raw_value: null }] }] }
  assert.equal(api.oemChartOption(invalid, palette).series[0].data.at(-1).value[1], null, '末次缺失不回填旧值')
}
for (const spec of specs.filter(spec => spec.metric)) {
  const metric = spec.metric
  const panel = { metric_code: metric.name, title: spec.title, unit: metric.unit === '%' ? 'percent' : metric.unit || '1', native: { mode: spec.mode }, series: [{ ...series, dimensions: Object.fromEntries(metric.dimensions.map(key => [key, 'a'])) }] }
  assert.ok(Object.hasOwn(api.oemMetricCharts, metric.name), `${metric.name} 必须逐项审核`)
  assert.ok(api.oemChartKind(panel))
  if (['wait_sess_cls/dbtime_waitclass_pct', 'UserBlock/count', 'Filesystems/pctAvailable'].includes(metric.name)) assert.equal(api.oemChartKind(panel), 'bar')
  if (spec.panel_id.endsWith('.status')) assert.equal(api.oemChartOption(panel, palette).series.length, 0, '状态不画数值曲线')
  if (api.oemMetricCharts[metric.name] === 'donut') {
    assert.equal(api.oemChartKind({ ...panel, series: [series, { ...series, series_key: 'b', dimensions: { other: 'b' } }] }), 'bar', '多维百分比不饼图求和')
    assert.equal(api.oemChartKind({ ...panel, series: [{ ...series, points: [{ ...points[0], value: 101 }] }] }), 'bar', '越界值不裁剪为占用环图')
  }
}
const artifacts = mkdtempSync(resolve(tmpdir(), 'aiops-oem-charts-'))
const { createServer } = await import(pathToFileURL(resolve(portal, 'node_modules/vite/dist/node/index.js')))
const assets = Object.fromEntries(['aiops-oem-presentation.js', 'aiops-oem-metrics.js'].map(name => [`/__kbot/${name}`, resolve(kbot, 'ui/aiops/js', name)]))
assets['/__kbot/echarts.js'] = resolve(kbot, 'ui/vendor/echarts.min.js')
for (const name of ['aiops.css', 'ammolite-shared.css', 'reference-monitoring.css', 'alignment.css', 'aiops-oem-metrics.css']) assets[`/__kbot/${name}`] = resolve(kbot, 'ui/aiops/css', name)
const probe = createNetServer()
await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve))
const freePort = probe.address().port
await new Promise(resolve => probe.close(resolve))
const server = await createServer({ root: portal, configFile: resolve(portal, 'vite.config.ts'), server: { host: '127.0.0.1', port: freePort }, plugins: [{ name: 'oem-charts-fixture', configureServer(server) {
  server.middlewares.use(async (req, res, next) => {
    const path = req.url?.split('?')[0]
    if (assets[path]) { res.setHeader('Content-Type', path.endsWith('.css') ? 'text/css' : 'text/javascript'); res.end(read(assets[path])); return }
    if (path !== '/__oem-fixture') return next()
    const html = isKbot ? `<html><head>${['aiops.css', 'ammolite-shared.css', 'reference-monitoring.css', 'alignment.css', 'aiops-oem-metrics.css'].map(name => `<link rel="stylesheet" href="/__kbot/${name}">`).join('')}</head><body data-page="monitoring"><main id="app" style="padding:20px"><section id="fixture" class="oem-overview"></section></main><script>window.KBotAIOpsShell={escape:value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt:value=>value};window.KBotAIOpsAuth={request:async(url,init={})=>{const response=await fetch(url,{...init,headers:{'Content-Type':'application/json',...init.headers}});if(!response.ok)throw Error('读取失败');return response.json();}};</script><script src="/__kbot/echarts.js"></script><script src="/__kbot/aiops-oem-presentation.js"></script><script src="/__kbot/aiops-oem-metrics.js"></script><script>window.fixture=new KBotAIOpsOEMMetrics(document.querySelector('#fixture'),{sourceId:'source',instanceId:'resource',sourceName:'OEM',instanceName:'SIM HOST',window:'1h',baseline:'1w'});window.fixtureSetBaseline=value=>fixture.refresh('1h',value);window.fixtureOption=node=>echarts.getInstanceByDom(node)?.getOption();</script></body></html>`
      : `<html><head></head><body><main id="app" style="padding:20px"></main><script type="module">import{createApp,ref,h}from'vue';import*as engine from'echarts/core';import Component from '/src/features/aiops/business/AIOpsOEMMetrics.vue';import '/src/shared/ui/styles.css';const baseline=ref('1w');window.fixtureSetBaseline=value=>baseline.value=value;window.fixtureOption=node=>engine.getInstanceByDom(node)?.getOption();createApp({setup:()=>()=>h(Component,{domainCode:'test',domainId:'domain',sourceId:'source',instanceId:'resource',sourceName:'OEM',instanceName:'SIM HOST',window:'1h',baseline:baseline.value,refresh:0})}).mount('#app');</script></body></html>`
    res.setHeader('Content-Type', 'text/html; charset=utf-8'); res.end(await server.transformIndexHtml(req.url, html.replace('<head>', '<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">')))
  })
} }] })
await server.listen()
const origin = `http://127.0.0.1:${server.httpServer.address().port}`
const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH })
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } }), errors = [], calls = []
  page.on('pageerror', error => errors.push(error.message)); await page.route('https://fonts.**', route => route.abort())
  let missing = false
  await page.route('**/api/v1/apps/aiops/**', async route => {
    if (route.request().method() === 'GET') return route.fulfill({ json: { overview_panels: specs, discovered_count: 15, truncated: false } })
    const body = route.request().postDataJSON(); calls.push(body)
    const spec = specs.find(spec => spec.metric?.name === body.metric_name), metric = spec.metric
    const count = metric.dimensions.length && metric.name !== 'Recovery_Area/Free_Space' ? 2 : 1
    const rows = Array.from({ length: count }, (_, index) => ({ dimensions: Object.fromEntries(metric.dimensions.map(key => [key, `${key}-${index}`])), average: body.baseline === '1m' ? '30.5' : '37.954293984962406015', sample_count: 7 }))
    const points = rows.flatMap((row, index) => (body.mode === 'LATEST' ? [2] : [0, 1, 2]).map(step => ({ observed_at: `2026-11-12T00:${['00', '15', '30'][step]}:00Z`, dimensions: row.dimensions, value: missing && body.metric_name === 'Load/memUsedPct' && step === 2 ? null : metric.native_value_type === 'STRING' ? 'OPEN' : step === 1 ? null : `${index + 42.125}` })))
    return route.fulfill({ json: { points, query_mode: body.mode, truncated: false, warnings: [], baseline: body.baseline ? { period: body.baseline, start: '2026-10-12T00:00:00Z', end: '2026-11-12T00:30:00Z', note: '留存采样均值', unavailable_reason: null, series: rows } : null } })
  })
  await page.goto(origin + '/__oem-fixture', { waitUntil: 'networkidle' })
  await expect(page.locator('.monitoring-panel')).toHaveCount(16)
  const card = title => page.locator('.monitoring-panel').filter({ has: page.locator('header h2', { hasText: title }) })
  const cpu = card('CPU 使用率'), memory = card('内存使用率')
  for (const target of [cpu, memory]) {
    await expect(target.locator('[data-chart-kind="line"] canvas')).toHaveCount(1)
    await expect(target.locator('.monitoring-baseline')).toContainText('37.95')
    const option = await target.locator('.monitoring-chart-canvas').evaluate(node => window.fixtureOption(node))
    assert.equal(option.series[0].type, 'line')
    assert.equal(option.series[0].markLine.data[0].yAxis, 37.954293984962406)
    assert.ok(option.series[0].markLine.label.formatter.includes('37.95'))
    assert.equal(option.series[0].data[1].value[1], null)
  }
  await expect(card('文件系统剩余空间比例').locator('[data-chart-kind="bar"] canvas')).toHaveCount(1)
  await expect(card('数据库等待类别').locator('[data-chart-kind="bar"] canvas')).toHaveCount(1)
  await expect(card('恢复区使用率').locator('[data-chart-kind="donut"] canvas')).toHaveCount(1)
  await expect(card('数据库运行状态').locator('canvas')).toHaveCount(0)
  await cpu.getByRole('button', { name: '查看数据', exact: true }).click()
  await expect(page.getByRole('dialog')).toBeVisible(); await expect(page.getByRole('dialog')).toContainText('42.125')
  await page.getByRole('dialog').getByRole('button', { name: /关闭/ }).first().click()
  await page.evaluate(() => window.fixtureSetBaseline('1m'))
  await expect(cpu.locator('.monitoring-baseline')).toContainText('30.5')
  await expect.poll(() => cpu.locator('.monitoring-chart-canvas').evaluate(node => window.fixtureOption(node)?.series[0].markLine.data[0].yAxis)).toBe(30.5)
  missing = true; await page.evaluate(() => window.fixtureSetBaseline('1w'))
  await expect(memory.locator('.monitoring-current-value strong')).toHaveText(/^—(?:\s*%)?$/)
  for (const width of [1920, 1440, 780, 390]) {
    await page.setViewportSize({ width, height: 1000 })
    const overflow = await page.evaluate(() => ({ width: document.documentElement.scrollWidth - innerWidth, nodes: [...document.querySelectorAll('section,article,p,header,div')].filter(node => node.getBoundingClientRect().right > innerWidth + 1).slice(0, 10).map(node => ({ class: node.className, right: node.getBoundingClientRect().right, text: node.textContent.slice(0, 50) })) }))
    assert.ok(overflow.width <= 1, `OEM ${width} 不横向溢出：${JSON.stringify(overflow)}`)
  }
  await page.setViewportSize({ width: 1440, height: 1000 }); await page.screenshot({ path: resolve(artifacts, `${isKbot ? 'kbot' : 'ammolite'}-oem.png`), fullPage: true })
  assert.ok(calls.every(call => ['HISTORY', 'LATEST'].includes(call.mode)))
  assert.deepEqual(errors, [])
  console.log(JSON.stringify({ ok: true, repository: isKbot ? 'kbot' : 'ammolite', reviewedPanels: specs.length, artifacts }))
} finally { await browser.close(); await server.close() }
