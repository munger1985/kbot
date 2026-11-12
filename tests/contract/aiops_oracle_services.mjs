// 两端服务生产组件验收；仅使用固定采样替身，不伪称现场 ASM/RAC 验收。
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { createServer as createNetServer } from 'node:net'
import { resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
const root = fileURLToPath(new URL('../../', import.meta.url)), kbot = existsSync(resolve(root, 'ui/aiops'))
const portal = resolve(kbot ? resolve(root, '../ammolite_cube') : root, 'apps/portal-web')
const other = kbot ? root : resolve(root, '../kbot4'), require = createRequire(resolve(portal, 'package.json'))
const { chromium, expect } = require('@playwright/test')
const { createServer } = await import(pathToFileURL(resolve(portal, 'node_modules/vite/dist/node/index.js')))
const state = { name: 'listener.availability', native_value_type: 'NUMBER', unit: 'state', semantics: 'STATE', state_values: { '1': { label: '可响应', tone: 'success' }, '0': { label: '不可响应', tone: 'danger' } } }
const specs = [
  { panel_id: state.name, title: 'Listener 响应', semantics: 'STATE', metric: state, mode: 'LATEST', baseline_supported: false },
  { panel_id: 'listener.tns_ping_ms', title: 'TNS 响应时间', semantics: 'GAUGE', metric: { name: 'listener.tns_ping_ms', native_value_type: 'NUMBER', unit: 'ms', semantics: 'GAUGE' }, mode: 'HISTORY', baseline_supported: true },
  { panel_id: 'listener.registered_services', title: '已配置服务', semantics: 'PROPERTY', metric: { name: 'listener.registered_services', native_value_type: 'STRING', unit: 'n/a', semantics: 'PROPERTY' }, mode: 'LATEST', baseline_supported: false },
  { panel_id: 'listener.tcp', title: 'TCP 可达性', semantics: 'STATE', metric: null, mode: 'LATEST', baseline_supported: false, unavailable_reason: '当前来源没有映射独立 TCP 状态' },
]
const assets = {}
for (const name of ['aiops-oem-presentation.js', 'aiops-oem-metrics.js', 'aiops-exadata-presentation.js']) assets['/__assets/'+name] = resolve(other, 'ui/aiops/js', name)
assets['/__assets/echarts.js'] = resolve(other, 'ui/vendor/echarts.min.js')
for (const name of ['aiops.css', 'ammolite-shared.css', 'reference-monitoring.css', 'alignment.css', 'aiops-oem-metrics.css']) assets['/__assets/'+name] = resolve(other, 'ui/aiops/css', name)
const probe = createNetServer()
await new Promise(resolve=>probe.listen(0,'127.0.0.1',resolve))
const freePort=probe.address().port
await new Promise(resolve=>probe.close(resolve))
const server = await createServer({ root: portal, configFile: resolve(portal, 'vite.config.ts'), server: { host: '127.0.0.1', port: freePort }, plugins: [{ name: 'oracle-service-fixture', configureServer(server) {
  server.middlewares.use(async (req, res, next) => {
    const path = req.url.split('?')[0]
    if (assets[path]) { res.setHeader('Content-Type', path.endsWith('.css') ? 'text/css' : 'text/javascript'); res.end(readFileSync(assets[path])); return }
    if (path !== '/__service') return next()
    const html = kbot ? `<html><head><meta charset="utf-8">${Object.keys(assets).filter(path=>path.endsWith('.css')).map(path=>'<link rel="stylesheet" href="'+path+'">').join('')}</head><body><main id="app"></main><script>window.KBotAIOpsShell={escape:v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt:v=>v};window.KBotAIOpsAuth={request:async(url,init={})=>(await fetch(url,{...init,headers:{'Content-Type':'application/json'}})).json()};</script>${Object.keys(assets).filter(path=>path.endsWith('.js')).map(path=>'<script src="'+path+'"></script>').join('')}<script>window.fixture=new KBotAIOpsOEMMetrics(document.querySelector('#app'),{service:true,sourceType:'PROMETHEUS',sourceName:'服务测试来源',sourceId:'source',instanceId:'resource',window:'1h',baseline:'1w'});window.refreshFixture=()=>fixture.refresh('1h','1w');</script></body></html>`
      : `<html><head><meta charset="utf-8"></head><body><main id="app"></main><script type="module">import{createApp,h,ref}from'vue';import Component from '/src/features/aiops/business/AIOpsOracleServiceMetrics.vue';import '/src/shared/ui/styles.css';const refresh=ref(0);window.refreshFixture=()=>refresh.value++;createApp({setup:()=>()=>h(Component,{domainCode:'test',domainId:'domain',sourceId:'source',instanceId:'resource',sourceName:'服务测试来源',sourceType:'PROMETHEUS',instanceName:'Listener',window:'1h',baseline:'1w',refresh:refresh.value})}).mount('#app');</script></body></html>`
    res.setHeader('Content-Type', 'text/html; charset=utf-8'); res.end(await server.transformIndexHtml(req.url, html))
  })
} }] })
await server.listen()
const browser = await chromium.launch({headless:true})
try {
  const page = await browser.newPage(), errors = [], calls = []
  let stale = false
  page.on('pageerror', error=>errors.push(error.message))
  await page.route('**/api/v1/apps/aiops/**', async route=>{
    if (route.request().method() === 'GET') return route.fulfill({json:{overview_panels:specs,truncated:false}})
    const body=route.request().postDataJSON(); calls.push(body)
    const points=(body.mode==='LATEST'?[0]:[0,1]).map(index=>({observed_at:'2026-11-12T12:0'+index+':00Z',value:body.metric_name===state.name?'1.00':body.metric_name==='listener.registered_services'?'SIM-SERVICE':String(10+index),dimensions:{},quality:stale?'STALE':'OBSERVED',current_eligible:!stale}))
    return route.fulfill({json:{points,truncated:false,baseline:body.baseline?{period:'1w',start:'2026-11-05T12:00:00Z',end:'2026-11-12T12:00:00Z',series:[{dimensions:{},average:'10',sample_count:10080}],note:'实际采样均值'}:null}})
  })
  await page.goto('http://127.0.0.1:'+server.httpServer.address().port+'/__service',{waitUntil:'networkidle'})
  await expect(page.locator('#app')).toContainText('可响应')
  await expect(page.locator('#app')).toContainText('SIM-SERVICE')
  await expect(page.locator('#app')).toContainText('当前来源没有映射独立 TCP 状态')
  assert.equal(calls.length,3,'未映射状态不发请求')
  assert.ok(calls.filter(call=>call.metric_name!== 'listener.tns_ping_ms').every(call=>call.baseline===null&&call.mode==='LATEST'))
  await expect(page.locator('.monitoring-panel')).toHaveCount(1,'状态和属性不用性能卡片')
  await expect(page.locator('.monitoring-panel canvas')).toHaveCount(1)
  await expect(page.locator('.monitoring-baseline')).toContainText('10')
  await page.locator('.monitoring-panel').getByRole('button',{name:'查看数据',exact:true}).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await expect(page.getByRole('dialog')).toContainText('11')
  await page.getByRole('dialog').getByRole('button',{name:/关闭/}).first().click()
  stale=true; await page.evaluate(()=>window.refreshFixture())
  await expect(page.locator('#app')).toContainText('当前不可判定')
  assert.deepEqual(errors,[])
  console.log(JSON.stringify({ok:true,repository:kbot?'kbot':'ammolite',checks:['状态字典','属性行','未映射不查询','趋势与基线','数据入口','过期不健康']}))
} finally { await browser.close(); await server.close() }
