// 只同步 OEM 纯展示策略，不覆盖任何非 OEM 页面或采集逻辑。
import assert from 'node:assert/strict'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
const root = fileURLToPath(new URL('../../', import.meta.url))
const kbot = existsSync(resolve(root, 'ui/aiops')) ? root : process.env.KBOT_ROOT || resolve(root, '../kbot4')
const ammolite = kbot === root ? process.env.AMMOLITE_ROOT || resolve(root, '../ammolite_cube') : root
const portal = resolve(ammolite, 'apps/portal-web')
const ts = createRequire(resolve(portal, 'package.json'))('typescript')
for (const [source, target, global] of [
  ['exadataPresentation.ts', 'aiops-exadata-presentation.js', 'KBotAIOpsExadataPresentation'],
  ['oemMonitoringPresentation.ts', 'aiops-oem-presentation.js', 'KBotAIOpsOEMPresentation'],
]) {
  const compiled = ts.transpileModule(readFileSync(resolve(portal, 'src/features/aiops/business', source), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
  const content = `// 由 Ammolite ${source} 同步，合同测试校验两端展示语义一致。\n(function () {\n  const exports = {};\n${compiled}\n  globalThis.${global} = exports;\n})();\n`
  const path = resolve(kbot, 'ui/aiops/js', target)
  if (process.argv.includes('--write')) writeFileSync(path, content)
  assert.equal(readFileSync(path, 'utf8'), content, `${source} 两端展示语义必须一致`)
}
console.log('OEM 展示策略同步检查通过')
