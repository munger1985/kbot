// 同步确定性的拓扑布局与安全 HTML 投影，两端运行时不依赖另一仓库。
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const portal = resolve(process.env.AMMOLITE_ROOT || resolve(root, '../ammolite_cube'), 'apps/portal-web');
const require = createRequire(resolve(portal, 'package.json'));
const ts = require('typescript');
for (const [name, globalName] of [['topologyLayout', 'KBotAIOpsTopologyLayout'], ['topologyView', 'KBotAIOpsTopologyView']]) {
  const source = readFileSync(resolve(portal, 'src/features/aiops/management', `${name}.ts`), 'utf8');
  const js = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
  const content = `// 自动同步自 Ammolite ${name}.ts；请修改源文件后重新生成。\n(function () {\nconst exports = {};\nconst require = () => ({ default: globalThis.KBotAIOpsTopologyLayout });\n${js}\nglobalThis.${globalName} = exports.default;\n})();\n`;
  const target = resolve(root, 'ui/aiops/js', `aiops-${name === 'topologyLayout' ? 'topology-layout' : 'topology-view'}.js`);
  if (process.argv.includes('--write')) writeFileSync(target, content);
  else assert.equal(readFileSync(target, 'utf8'), content, `${name} 与 Ammolite 源码不一致`);
}
console.log('通过：两端拓扑布局和安全 HTML 投影同步。');
