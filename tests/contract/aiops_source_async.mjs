// 使用真实保存/轮询函数和表单模板；接口固定替身不接触运行中的配置库。
import assert from 'node:assert/strict'
import { readFileSync, existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const kbot = existsSync(resolve(root, 'ui/aiops'))
const portal = resolve(kbot ? resolve(root, '../Ammolite') : root, 'apps/portal-web')
const require = createRequire(resolve(portal, 'package.json'))
const { chromium, expect } = require('@playwright/test')
const ts = require('typescript')
const source = readFileSync(resolve(portal, 'src/features/aiops/management/AIOpsConfigurationView.vue'), 'utf8')
const script = source.match(/<script setup lang="ts">([^]*?)<\/script>/)[1]
const names = new Set(['createDraft', 'sourceCredentials', 'sourceConfig', 'parseStandardMappings',
  'sourcePayload', 'sourcePatchPayload', 'save', 'load', 'scheduleConnectivityReload',
  'connectivityPresentation', 'sourceConnectivityResultMessage', 'testSourceConnection'])
const ast = ts.createSourceFile('source.ts', script, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
const selected = ast.statements.filter(node => ts.isFunctionDeclaration(node) && names.has(node.name?.text))
assert.equal(selected.length, names.size)
const functions = ts.transpileModule(selected.map(node => node.getText(ast)).join('\n'),
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
const template = source.match(/<form v-else class="resource-form[^]*?<\/form>/)[0].replace('<form v-else ', '<form ')

const browser = await chromium.launch({ headless: true })
try {
  for (const connected of [true, false]) {
    const page = await browser.newPage()
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    await page.clock.install()
    const html = kbot ? readFileSync(resolve(root, 'ui/aiops/diagnostic-sources.html'), 'utf8')
      .replace(/<script\b[^>]*>[^]*?<\/script>/g, '').replace(/<link\b[^>]*>/g, '')
      : '<div id="app"></div>'
    await page.setContent(html)
    await page.evaluate(() => {
      window.messages = []
      window.catalogReads = 0
      window.writes = []
      window.sourceItem = { source_id: 'source-1', display_name: '本地 Loki', source_type: 'LOKI',
        endpoint: 'http://old.invalid:3100/', status: 'DISABLED', connectivity_status: 'UNKNOWN',
        connectivity_check_pending: false, row_version: 1, config: {}, updated_at: '2026-11-16T11:00:00Z' }
      window.mockRequest = async (url, init = {}) => {
        if (init.method === 'PATCH') {
          window.writes.push(JSON.parse(init.body))
          window.sourceItem = { ...window.sourceItem, ...JSON.parse(init.body), row_version: 2,
            connectivity_status: 'CHECKING', connectivity_check_pending: true }
          window.holdRefresh = true
          return { ...window.sourceItem }
        }
        if (String(url).includes('?limit=')) {
          window.catalogReads++
          if (window.holdRefresh) await new Promise(resolve => { window.releaseRefresh = () => { window.holdRefresh = false; resolve() } })
          return { items: [{ ...window.sourceItem }] }
        }
        return { ...window.sourceItem }
      }
    })
    if (kbot) {
      await page.evaluate(() => {
        window.KBotAIOpsShell = { ready: Promise.resolve({ permissions: ['aiops:diagnostic_source_manage'] }),
          escape: value => String(value ?? ''), fmt: String, toast: value => window.messages.push(value) }
        window.KBotAIOpsAuth = { uuid: () => 'test-request', request: window.mockRequest }
      })
      await page.addScriptTag({ path: resolve(root, 'ui/aiops/js/aiops-configurations.js') })
      await page.getByRole('button', { name: '编辑', exact: true }).click()
    } else {
      await page.addScriptTag({ path: resolve(portal, 'node_modules/vue/dist/vue.global.js') })
      await page.evaluate(({ functions, template }) => {
        const setup = new Function('Vue', functions + `
          const props = {section: 'diagnostic-sources'}, items = Vue.ref([{...window.sourceItem}]);
          const domainCode = Vue.ref('test'), domainId = Vue.ref('test');
          const editingSource = Vue.ref({...window.sourceItem}), saving = Vue.ref(false);
          const draft = Vue.ref({...createDraft(), display_name: '本地 Loki', source_type: 'LOKI',
            endpoint: 'http://old.invalid:3100/'}), formOpen = Vue.ref(true);
          const loading = Vue.ref(false), error = Vue.ref(''), responsibilityGroups = Vue.ref([]);
          const toast = Object.fromEntries(['success','warning','error','info'].map(key=>[key, value=>window.messages.push(value)]));
          const businessFetch = (url, options) => window.mockRequest(url, options.init), uuid7 = ()=> 'test-request';
          let connectivityReloadAttempts = 0, connectivityReloadTimer;
          const closeForm = async () => {formOpen.value=false}, openRequestedTarget=async()=>{};
          window.fixture = {load,items,saving,formOpen,testSourceConnection};
          return {draft,editingSource,saving,formOpen,items,connectivityPresentation,
            formSubmitLabel:'保存监控源',closeForm,save};
        `)
        Vue.createApp({ setup: () => setup(Vue), template:
          '<section><div v-if="formOpen">' + template +
          '</div><p v-for="item in items" class="fixture-state">{{connectivityPresentation(item.connectivity_status).label}}</p></section>'
        }).mount('#app')
      }, { functions, template })
    }
    const form = kbot ? page.locator('#diagnostic-source-form') : page.locator('form')
    await form.locator('input[type="url"]').fill('http://127.0.0.1:3100/')
    await form.getByRole('button', { name: '保存监控源', exact: true }).click()
    await expect.poll(() => page.evaluate(() => window.writes.length)).toBe(1)
    // 后续目录读取故意悬置，不能继续锁住提交按钮或编辑抽屉。
    if (kbot) await expect(page.locator('#diagnostic-source-dialog')).not.toBeVisible()
    else await expect(page.locator('form')).toHaveCount(0)
    assert.ok((await page.evaluate(() => window.messages)).some(value => value.includes('正在后台测试连接')))
    await expect.poll(() => page.evaluate(() => typeof window.releaseRefresh)).toBe('function')
    await page.evaluate(() => window.releaseRefresh())
    // 超过旧版九秒轮询上限，仍须继续取得后台结果。
    for (let index = 0; index < 8; index++) {
      const before = await page.evaluate(() => window.catalogReads)
      await page.clock.fastForward(1500)
      await expect.poll(() => page.evaluate(() => window.catalogReads)).toBeGreaterThan(before)
    }
    await page.evaluate(healthy => {
      window.sourceItem = {...window.sourceItem, connectivity_status:healthy?'CONNECTED':'UNREACHABLE',
        connectivity_check_pending:false, last_error_code:healthy?null:'SOURCE_UNREACHABLE'}
    }, connected)
    await page.clock.fastForward(1500)
    const state = kbot ? page.locator('#source-catalog .target-status-badge') : page.locator('.fixture-state')
    await expect(state).toHaveText(connected ? '已连通' : '不可达')
    assert.equal(await page.evaluate(() => window.sourceItem.status), 'DISABLED')
    assert.deepEqual(errors, [])
    await page.close()
  }
  console.log(JSON.stringify({ok:true,repository:kbot?'kbot4':'Ammolite',
    checks:['保存不等待网络检查或目录刷新','后台检查提示','超过旧轮询上限','成功与失败收敛','不自动启用']}))
} finally { await browser.close() }
