// 从 Ammolite 的真实生产样式生成 KBot AIOps 基线；运行时不依赖另一个仓库。
// 检查：node tests/contract/sync_aiops_reference_styles.mjs
// 更新：node tests/contract/sync_aiops_reference_styles.mjs --write
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const portal = resolve(process.env.AMMOLITE_ROOT || resolve(root, "../ammolite_cube"), "apps/portal-web");
const require = createRequire(resolve(portal, "package.json"));
const postcss = require("postcss");
const read = (path) => readFileSync(path, "utf8");
const manifest = JSON.parse(read(resolve(root, "ui/aiops/reference-pages.json")));
const classes = new Set();
const sourceFiles = new Set([
  "src/app/BusinessAppShell.vue", "src/shared/ui/ConversationList.vue",
  "src/shared/ui/CatalogCardActions.vue", "src/shared/ui/DateTimePicker.vue",
  "src/shared/ui/AgentMarkdownContent.vue", "src/shared/ui/AgentChart.vue",
  "src/shared/ui/LifecycleStatusBadge.vue", "src/shared/ui/LifecycleToggleButton.vue",
  "src/shared/ui/ConfirmDialog.vue",
]);
for (const entry of Object.values(manifest.pages)) for (const source of entry.sources) sourceFiles.add(`src/features/aiops/${source}`);
for (const file of sourceFiles) {
  const content = read(resolve(portal, file));
  for (const match of content.matchAll(/class="([^"]+)"/g)) for (const name of match[1].split(/\s+/)) classes.add(name);
  // Vue 的动态 class 也会引用共享控件，不能只收集静态 class 属性。
  for (const match of content.matchAll(/(?::class|v-bind:class)="([^"]+)"/g)) {
    for (const token of match[1].matchAll(/[a-zA-Z_][\w-]*/g)) classes.add(token[0]);
  }
  for (const match of content.matchAll(/\.([a-zA-Z_][\w-]*)/g)) classes.add(match[1]);
}
for (const name of ["business-app-aiops", "business-app-shell", "is-viewport-contained", "is-agent-chat", "nav-item", "nav-item-icon", "nav-item-label", "business-route-active", "is-collapsible", "is-open", "is-active", "is-compact", "site-footer", "site-footer-spacer", "status-success", "status-warning", "status-danger", "status-muted", "confirm-dialog--warning", "confirm-dialog--danger"]) classes.add(name);

const sharedSource = read(resolve(portal, "src/shared/ui/styles.css"));
const shared = postcss.parse(sharedSource);
shared.walkAtRules("import", (rule) => rule.remove());
shared.walkRules((rule) => {
  if (rule.parent.type === "atrule" && /keyframes$/.test(rule.parent.name)) return;
  const selectors = rule.selectors.filter((selector) => {
    const names = [...selector.matchAll(/\.([a-zA-Z_][\w-]*)/g)].map((match) => match[1]);
    if (!names.length) return true;
    // 顶层专用页面的规则不属于 AIOps；:is() 中的公共壳层分支保持原样。
    const direct = selector.replace(/:(?:is|where|not|has)\([^)]*\)/g, "");
    const directNames = [...direct.matchAll(/\.([a-zA-Z_][\w-]*)/g)].map((match) => match[1]);
    return names.some((name) => classes.has(name)) && directNames.every((name) => classes.has(name));
  });
  if (!selectors.length) rule.remove();
  else rule.selectors = selectors;
});
shared.walkComments((comment) => comment.remove());
shared.walkAtRules((rule) => { if (rule.nodes && !rule.nodes.length) rule.remove(); });
const hash = createHash("sha256").update(sharedSource).digest("hex");
const header = `/* 自动同步自 Ammolite shared/ui/styles.css；请修改源文件后重新生成。\n * 源文件 SHA-256: ${hash}\n * 仅保留 AIOps 及其公共控件；字体使用同一系统回退栈，不从 CDN 加载。\n */\n`;
// 旧静态页的裸 button 规则不能泄漏到 Portal 控件；随后按源样式恢复字体与各类按钮。
const outputs = [["css/ammolite-shared.css", header + "button { all: revert; }\n" + shared.toString().trim() + "\n"]];
assert.ok(shared.toString().includes('.upload-mode-option {'), '动态 class 引用的上传组织方式控件必须保留完整样式');
assert.ok(shared.toString().includes('.confirm-dialog--warning .confirm-dialog-mark'), '动态生成的确认框警告主题必须与源端一致');
for (const [page, entry] of Object.entries(manifest.pages)) {
  if (!entry.sources.length) continue;
  let css = "/* 自动同步自 reference-pages.json 中声明的 Ammolite 生产组件。 */\n";
  for (const source of entry.sources) {
    const content = read(resolve(portal, `src/features/aiops/${source}`));
    const match = content.match(/<style scoped>([\s\S]*?)<\/style>/);
    const external = content.match(/<style src="([^"]+)"><\/style>/);
    if (!match && !external) continue;
    // 子组件的外置样式须在父页面之后恢复，防止静态化后的父级选择器穿透原 scoped 边界。
    const stylesheet = match ? match[1] : read(resolve(portal,"src/features/aiops",dirname(source),external[1]));
    const parsed = postcss.parse(stylesheet.replace(/:deep\(([^)]+)\)/g, "$1"));
    const componentRoot = {
      "business/AIOpsEffectVerification.vue": ".effect-verification",
      "management/WorkloadSnapshotSettings.vue": ".workload-snapshot-settings",
      "management/BusinessSystemTopology.vue": ".business-system-topology",
    }[source];
    parsed.walkRules((rule) => {
      if (rule.parent.type === "atrule" && /keyframes$/.test(rule.parent.name)) return;
      // 保留 Vue scoped 属性带来的优先级，避免 :root 公共表格规则覆盖页面规格。
      rule.selectors = rule.selectors.map((selector) => `:is([data-page="${page}"]) ${componentRoot && !selector.startsWith(".") ? `${componentRoot} ` : ""}${selector}`);
    });
    css += `\n/* ${source} */\n${parsed.toString().trim()}\n`;
  }
  outputs.push([`css/reference-${page}.css`, css]);
}
for (const [file, content] of outputs) {
  const path = resolve(root, "ui/aiops", file);
  if (process.argv.includes("--write")) writeFileSync(path, content);
  else assert.equal(read(path), content, `${file} 与 Ammolite 基线不同，请重新同步并验证布局`);
}
const actual = readdirSync(resolve(root, "ui/aiops")).filter((name) => name.endsWith(".html")).map((name) => name.slice(0, -5)).sort();
assert.deepEqual(Object.keys(manifest.pages).sort(), actual, "所有页面必须声明 Ammolite 对应页面或平台差异");
console.log(`AIOps 样式基线：${outputs.length} 份工件，${actual.length} 个页面已登记。`);
