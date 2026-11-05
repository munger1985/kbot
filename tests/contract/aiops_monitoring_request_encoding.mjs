// 使用真实监控页面、请求封装与浏览器验证编码和资源树，不连接真实监控源。
// 两个仓库分别执行：node tests/contract/aiops_monitoring_request_encoding.mjs
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const isKbot = existsSync(resolve(root, 'ui/aiops/monitoring.html'))
const portal = resolve(isKbot ? process.env.AMMOLITE_ROOT || resolve(root, '../ammolite_cube') : root, 'apps/portal-web')
const require = createRequire(resolve(portal, 'package.json'))
const { chromium, expect } = require('@playwright/test')
const ts = require('typescript')
const { parse, compileScript } = require('@vue/compiler-sfc')
const read = path => readFileSync(path, 'utf8')
const transpile = (source, module) => ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module },
}).outputText
const sourceId = '019ffff0-0000-7000-8000-000000000001'
const instanceId = '019ffff0-0000-7000-8000-000000000002'
const domainId = '019ffff0-0000-7000-8000-000000000003'
let compiledView, compiledHttp
if (!isKbot) {
  const injection = `
const { computed, onBeforeUnmount, onMounted, ref, watch } = Vue
const useRoute = () => window.fixture.route
const useRouter = () => ({ replace: async () => {} })
const businessFetch = window.fixture.http.businessFetch
const formatPortalDateTime = value => value || ''
const toast = { error: error => window.fixture.errors.push(error.message) }
const NavIcon = { template: '<span></span>' }
const AIOpsMonitoringPanelChart = { template: '<div></div>' }
`
  const source = read(resolve(portal, 'src/features/aiops/business/AIOpsMonitoringView.vue'))
    .replace(/^import .+$/gm, '').replace('<script setup lang="ts">', '<script setup lang="ts">' + injection)
  const { descriptor } = parse(source)
  compiledView = transpile(compileScript(descriptor, { id: 'monitoring-encoding', inlineTemplate: true }).content, ts.ModuleKind.ES2022)
    .replace(/import\s*\{([^}]+)\}\s*from\s*['"]vue['"];?/gs, (_, names) => `const { ${names.replace(/\s+as\s+/g, ': ')} } = Vue;`)
    .replace('export default', 'return')
  compiledHttp = transpile(read(resolve(portal, 'src/shared/api/http.ts')).replaceAll('import.meta.env', '({})'), ts.ModuleKind.CommonJS)
}

const browser = await chromium.launch({ headless: true })
try {
  for (const resourceType of ['PHYSICAL_SERVER', 'VIRTUAL_MACHINE', 'CONTAINER', 'STORAGE', 'NETWORK_DEVICE', 'DATABASE']) {
    const page = await browser.newPage()
    const errors = [], calls = []
    page.on('pageerror', error => errors.push(error.message))
    const hardware = resourceType !== 'DATABASE'
    const source = { source_id: sourceId, display_name: '测试监控源', source_type: 'PROMETHEUS',
      monitoring_readiness: 'READY', diagnostic_readiness: 'READY', capability_gaps: [], diagnostic_gaps: [] }
    const instance = { instance_id: instanceId, display_name: '测试监控资源', db_type: hardware ? null : 'ORACLE',
      resource_type_key: resourceType, status: 'AVAILABLE', business_system_id: 'system-b', business_system_name: '测试业务系统乙',
      monitoring_readiness: 'READY', diagnostic_readiness: 'READY', capability_gaps: [], diagnostic_gaps: [] }
    const instances = [instance,
      { ...instance, instance_id: '019ffff0-0000-7000-8000-000000000004', business_system_id: 'system-a', business_system_name: '测试业务系统甲' },
      { ...instance, instance_id: '019ffff0-0000-7000-8000-000000000005', business_system_id: null, business_system_name: null }]
    const profile = { profile_id: hardware ? 'host-overview' : 'database-overview', version: '1.0.0',
      display_name: '测试指标范围', description: '', purpose: hardware ? 'HOST' : 'OVERVIEW', is_default: true,
      metric_codes: [], supported_db_types: ['ORACLE'], supported_database_versions: {}, resolved_variant_ids: [],
      capacity_warning_percent: 80, capacity_critical_percent: 90 }
    const html = isKbot ? read(resolve(root, 'ui/aiops/monitoring.html'))
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, '').replace(/<link\b[^>]*>/g, '') : '<div id="app"></div>'
    await page.route('https://fixture.test/**', async route => {
      const request = route.request(), url = new URL(request.url()), path = url.pathname
      if (path === '/monitoring') return route.fulfill({ contentType: 'text/html', body: html })
      let body
      if (path.endsWith('/sources')) body = [source]
      else if (path.endsWith('/instances')) body = instances
      else if (path.endsWith('/profiles')) body = [profile]
      else if (path.endsWith('/views') || path.endsWith('/monitoring/view')) {
        const call = { path, method: request.method(), headers: request.headers(),
          body: request.postData() ? request.postDataJSON() : null, query: Object.fromEntries(url.searchParams) }
        calls.push(call)
        // 模拟 FastAPI 的请求体解析边界，不能让测试后端默默接受 text/plain。
        if (call.method === 'POST' && call.headers['content-type'] !== 'application/json') {
          return route.fulfill({ status: 422, json: { detail: [{ loc: ['body'], msg: '请求体必须为 JSON 对象' }] } })
        }
        body = { schema_version: 'aiops.public.v1', generated_at: '2026-11-05T12:44:00Z', refresh_after_seconds: 30,
          source, profile, window: { start: '2026-11-05T11:44:00Z', end: '2026-11-05T12:44:00Z',
            name: call.body?.window || call.query.window }, instances: [instance], panels: [], gaps: [], partial: false }
      } else {
        errors.push('意外请求：' + path)
        return route.fulfill({ status: 404, json: {} })
      }
      return route.fulfill({ json: body })
    })
    await page.goto(`https://fixture.test/monitoring?targetId=${instanceId}`)
    await page.evaluate(({ domainId, instanceId }) => {
      window.fixture = { errors: [], route: { params: { domainCode: 'alpha' },
        query: { domainId, targetId: instanceId }, meta: { title: '实时监控' } } }
      localStorage.setItem('amm_admin_tenant_id', 'test-tenant')
      document.cookie = 'amm_business_csrf=test-csrf; path=/'
      sessionStorage.setItem('kbot.aiops.session.v1', JSON.stringify({ access_token: 'test-token' }))
      window.KBOT_UI_CONFIG = { mainApiBaseUrl: 'https://fixture.test' }
      window.KBotAIOpsShell = { ready: Promise.resolve(), fmt: value => value || '',
        escape: value => String(value ?? ''), toast: message => fixture.errors.push(message) }
    }, { domainId, instanceId })
    if (isKbot) {
      for (const name of ['aiops-auth.js', 'aiops-monitoring.js']) {
        await page.addScriptTag({ path: resolve(root, 'ui/aiops/js', name) })
      }
    } else {
      await page.addScriptTag({ path: resolve(portal, 'node_modules/vue/dist/vue.global.js') })
      await page.evaluate(({ compiledHttp, compiledView }) => {
        fixture.http = {}
        new Function('exports', compiledHttp)(fixture.http)
        const app = Vue.createApp(new Function('Vue', compiledView)(Vue))
        app.config.globalProperties.$route = fixture.route
        app.component('RouterLink', { props: ['to'], template: '<a><slot /></a>' }).mount('#app')
      }, { compiledHttp, compiledView })
    }
    await expect.poll(() => calls.length).toBe(1)
    const verify = (call, window) => {
      assert.equal(call.headers.accept, 'application/json')
      if (hardware) {
        assert.equal(call.path.endsWith(`/monitoring/sources/${sourceId}/views`), true)
        assert.equal(call.method, 'POST')
        assert.equal(call.headers['content-type'], 'application/json')
        assert.deepEqual(call.body, { profile_id: profile.profile_id, instance_ids: [instanceId], window })
        assert.deepEqual(call.query, {})
      } else {
        assert.equal(call.path.endsWith(`/targets/${instanceId}/monitoring/view`), true)
        assert.equal(call.method, 'GET')
        assert.equal(call.body, null)
        assert.deepEqual(call.query, { source_id: sourceId, profile_id: profile.profile_id, window, baseline: '1w' })
      }
      if (isKbot) assert.equal(call.headers.authorization, 'Bearer test-token')
      else {
        assert.equal(call.headers['x-ammolite-entry'], 'BUSINESS')
        assert.equal(call.headers['x-ammolite-app-id'], 'aiops')
        assert.equal(call.headers['x-ammolite-domain-id'], domainId)
        if (hardware) assert.equal(call.headers['x-csrf-token'], 'test-csrf')
      }
    }
    verify(calls[0], '1h')
    await expect(page.locator('.monitoring-no-panels')).toBeVisible()
    const toggles = page.locator('.monitoring-system-toggle')
    const expanded = page.locator('.monitoring-system-toggle[aria-expanded="true"]')
    const system = name => page.locator('.monitoring-system-group').filter({ has: page.locator('.monitoring-system-toggle strong', { hasText: name }) })
    const current = system('测试业务系统乙'), other = system('测试业务系统甲'), unassigned = system('未归属业务系统')
    const search = page.getByRole('searchbox', { name: '搜索业务系统或实例' })
    await expect(toggles).toHaveCount(3)
    await expect(expanded).toHaveCount(1)
    await expect(current.getByRole('button')).toHaveAttribute('aria-expanded', 'true')
    await other.getByRole('button').click()
    await expect(expanded).toHaveCount(1)
    await expect(current.locator('.monitoring-system-instances')).toBeHidden()
    await expect(other.locator('.monitoring-system-instances')).toBeVisible()
    // 刷新数据不能重新展开当前观测资源所属分组或破坏用户刚选定的展开项。
    await page.getByRole('button', { name: '刷新', exact: true }).click()
    await expect.poll(() => calls.length).toBe(2)
    verify(calls[1], '1h')
    await expect(page.getByRole('button', { name: '刷新', exact: true })).toBeEnabled()
    await expect(other.getByRole('button')).toHaveAttribute('aria-expanded', 'true')
    await other.getByRole('button').click()
    await expect(expanded).toHaveCount(0)
    await current.getByRole('button').click()
    await unassigned.getByRole('button').click()
    await expect(expanded).toHaveCount(1)
    await expect(current.locator('.monitoring-system-instances')).toBeHidden()
    await expect(unassigned.locator('.monitoring-system-instances')).toBeVisible()
    await search.fill('测试监控')
    await expect(toggles).toHaveCount(3)
    await expect(expanded).toHaveCount(1)
    await other.getByRole('button').click()
    await expect(expanded).toHaveCount(1)
    await expect(unassigned.locator('.monitoring-system-instances')).toBeHidden()
    await other.getByRole('button').click()
    await expect(expanded).toHaveCount(0)
    await search.fill('测试业务系统乙')
    await expect(toggles).toHaveCount(1)
    await expect(current.locator('.monitoring-system-instances')).toBeVisible()
    await search.fill('不存在的资源')
    await expect(expanded).toHaveCount(0)
    await expect(page.getByText('没有匹配的业务系统或实例。', { exact: true })).toBeVisible()
    await search.fill('')
    await expect(expanded).toHaveCount(1)
    await expect(unassigned.locator('.monitoring-system-instances')).toBeVisible()
    assert.equal(calls.length, 2, '搜索与折叠不得重新查询或改变观测对象')
    await page.getByRole('combobox', { name: '时间窗口', exact: true }).selectOption('6h')
    await expect.poll(() => calls.length).toBe(3)
    verify(calls[2], '6h')
    await expect(page.getByRole('button', { name: '刷新', exact: true })).toBeEnabled()
    await page.getByRole('button', { name: '刷新', exact: true }).click()
    await expect.poll(() => calls.length).toBe(4)
    verify(calls[3], '6h')
    await expect(page.getByRole('button', { name: '刷新', exact: true })).toBeEnabled()
    assert.deepEqual(await page.evaluate(() => fixture.errors), [])
    assert.deepEqual(errors, [])
    await page.close()
    console.log(`通过：${isKbot ? 'KBot' : 'Portal'} ${resourceType} 请求编码、初始定位、互斥展开、搜索、空结果和刷新保留展开项。`)
  }
} finally {
  await browser.close()
}
