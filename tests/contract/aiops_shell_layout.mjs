// 使用 Ammolite 的真实侧栏模板、路由、导航数据及分组函数对照正式静态 Shell。
// 运行：node tests/contract/aiops_shell_layout.mjs；仅使用隔离鉴权桩，不请求真实后端。
import assert from "node:assert/strict";
import { readFileSync, mkdtempSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const portal = resolve(process.env.AMMOLITE_ROOT || resolve(root, "../ammolite_cube"), "apps/portal-web");
const require = createRequire(resolve(portal, "package.json"));
const { chromium } = require("@playwright/test");
const { compile } = require("@vue/compiler-dom");
const typescript = require("typescript");
const read = path => readFileSync(path, "utf8");
const transpile = source => typescript.transpileModule(source, {
  compilerOptions: { module: typescript.ModuleKind.CommonJS, target: typescript.ScriptTarget.ES2022 },
}).outputText;
const shell = read(resolve(portal, "src/app/BusinessAppShell.vue"));
const aside = shell.slice(shell.indexOf('<aside id="business-app-navigation"'), shell.indexOf("</aside>") + 8);
const render = compile(aside, { mode: "function" }).code;
const helpers = transpile(shell.slice(shell.indexOf("function allowed("), shell.indexOf("function applyNavigationTitle(")));
const navigation = transpile(read(resolve(portal, "src/app/businessNavigation.ts")));
const icons = read(resolve(portal, "src/shared/ui/NavIcon.vue"));
const iconPaths = new Function(`return (${icons.split("const iconPaths: Record<string, string[]> = ")[1].split("</script>")[0].trim()})`)();
const domainIcon = shell.match(/class="business-context-icon"[^>]*><NavIcon name="([^"]+)"/)[1];
const routerSource = read(resolve(portal, "src/app/router.ts"));
const routeSection = routerSource.split("path: '/apps/aiops/:domainCode'")[1].split("path: '/apps/media-studio/:domainCode'")[0];
const childPaths = [...routeSection.matchAll(/\{ path: '([^']*)', component:/g)].map(match => match[1]);
assert.ok(childPaths.includes("manage/business-systems/:businessSystemId"));
const permissions = [
  "aiops:use", "aiops:work_item_handle", "aiops:target_manage", "aiops:diagnostic_source_manage",
  "aiops:work_item_manage", "aiops:knowledge_manage", "aiops:agent_manage", "aiops:plan_manage",
];
const manifest = JSON.parse(read(resolve(root, "ui/aiops/reference-pages.json")));
const cases = Object.entries(manifest.pages)
  // 平台登录/API 管理及租户设置不伪造业务 App 侧栏基准。
  .filter(([, entry]) => entry.route !== null && !entry.route.startsWith("/"))
  .map(([page, entry]) => [page, entry.route ? `/${entry.route.replace(/:[A-Za-z]+/g, "fixture-id")}` : ""]);
const artifacts = mkdtempSync(resolve(tmpdir(), "aiops-shell-alignment-"));
const errors = [];
const differences = [];
const server = createServer((request, response) => {
  const url = new URL(request.url, "http://localhost");
  if (url.pathname === "/shell") {
    const page = url.searchParams.get("page");
    if (!cases.some(([name]) => name === page)) { response.writeHead(404).end(); return; }
    response.setHeader("Content-Type", "text/html; charset=utf-8");
    response.end(`<!doctype html><html lang="zh-CN"><head><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/ui/aiops/css/aiops.css"><link rel="stylesheet" href="/ui/aiops/css/ammolite-shared.css"><link rel="stylesheet" href="/ui/aiops/css/alignment.css"></head><body data-page="${page}"><main class="ops-main"></main><script src="/ui/aiops/js/aiops-shell.js"></script></body></html>`);
    return;
  }
  const path = resolve(root, `.${url.pathname}`);
  if (!path.startsWith(`${root}/ui/aiops/`)) { response.writeHead(404).end(); return; }
  try {
    response.setHeader("Content-Type", `${extname(path) === ".css" ? "text/css" : "text/javascript"}; charset=utf-8`);
    response.end(read(path));
  } catch { response.writeHead(404).end(); }
});
await new Promise(done => server.listen(0, "127.0.0.1", done));
const browser = await chromium.launch({ headless: true });
const measure = page => page.evaluate(() => [...document.querySelectorAll(".business-app-sidebar, .business-app-sidebar nav, .business-nav-pinned, .business-nav-group, .business-nav-group-toggle, .business-nav-item, .business-nav-item .nav-item-icon, .business-nav-item .nav-item-label, .business-app-nav-footer")].map(node => {
  const rect = node.getBoundingClientRect(), css = getComputedStyle(node);
  return {
    label: node.matches("a, button") ? node.textContent.trim() : node.classList[0],
    active: node.classList.contains("business-route-active"), expanded: node.getAttribute("aria-expanded"),
    x: rect.x, y: rect.y, width: rect.width, height: rect.height,
    color: css.color, background: css.backgroundColor, border: css.borderColor,
    fontSize: css.fontSize, fontWeight: css.fontWeight, padding: css.padding, gap: css.gap,
    icons: [...node.querySelectorAll("svg path")].map(path => path.getAttribute("d")),
  };
}));
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript(permissions => {
    globalThis.KBotAIOpsAuth = {
      load: () => ({ access_token: "isolated-shell-fixture" }),
      request: async path => {
        if (path !== "/api/v1/apps/aiops/access") throw new Error(`禁止真实业务请求：${path}`);
        return { permissions, user_id: "fixture", domain_id: "fixture" };
      },
    };
  }, permissions);
  const kbot = await context.newPage();
  const reference = await context.newPage();
  for (const page of [kbot, reference]) page.on("pageerror", error => errors.push(error.message));
  await reference.setContent('<!doctype html><html lang="zh-CN"><body><div class="business-app-shell business-app-aiops"><header class="business-app-topbar"></header><div class="business-app-layout"><div id="reference" style="display:contents"></div><div class="business-app-content"></div></div></div></body></html>');
  await reference.addStyleTag({ content: read(resolve(portal, "src/shared/ui/styles.css")).replace(/@import url\([^)]*\);/g, "") });
  await reference.addStyleTag({ content: "*,*::before,*::after{transition:none!important;animation:none!important}" });
  await reference.addScriptTag({ path: require.resolve("vue/dist/vue.global.prod.js") });
  await reference.addScriptTag({ path: require.resolve("vue-router/dist/vue-router.global.prod.js") });
  await reference.addScriptTag({ content: `(function(){const exports={};${navigation}\nglobalThis.referenceNavigation=exports;})();` });
  await reference.evaluate(async ({ render, helpers, iconPaths, permissions, childPaths }) => {
    const router = VueRouter.createRouter({ history: VueRouter.createMemoryHistory(), routes: [{
      path: "/apps/aiops/:domainCode", component: { template: "<router-view/>" },
      children: childPaths.map(path => ({ path, component: { template: "<span/>" } })),
    }] });
    await router.push("/apps/aiops/fixture/diagnosis?domainId=fixture");
    const setup = new Function("Vue", "VueRouter", "permissions", `
      const {ref,computed,watch}=Vue;
      const props={appId:'aiops',appSlug:'aiops'};
      const route=VueRouter.useRoute();
      const appBase=computed(()=>'/apps/aiops/fixture');
      const domainId=ref('fixture');
      const bootstrap=ref({permissions:Object.fromEntries(permissions.map(key=>[key,true]))});
      const navigation=computed(()=>referenceNavigation.businessNavigation('aiops'));
      const businessNavigationRouteItem=referenceNavigation.businessNavigationRouteItem;
      const businessItems=computed(()=>navigation.value.business);
      const pinnedItems=computed(()=>navigation.value.pinned);
      const navigationGroups=computed(()=>navigation.value.groups);
      const visibleNavigationGroups=computed(()=>navigationGroups.value.filter(group=>group.items.some(allowed)));
      const allNavigationItems=computed(()=>[...pinnedItems.value,...navigationGroups.value.flatMap(group=>group.items)]);
      const openNavigationGroupKey=ref('');
      const mobileNavOpen=ref(false);
      ${helpers}
      syncOpenNavigationGroup();
      watch(()=>route.fullPath,syncOpenNavigationGroup);
      const exposed={pinnedItems,visibleNavigationGroups,allowed,target,appName:'AIOps',mobileNavOpen,openNavigationGroupKey,toggleNavigationGroup,closeMobileNav:()=>{mobileNavOpen.value=false}};
      globalThis.referenceShell=exposed;
      return exposed;
    `);
    const app = Vue.createApp({ setup: () => setup(Vue, VueRouter, permissions), render: new Function("Vue", render)(Vue) });
    app.component("NavIcon", { props: ["name"], data: () => ({ iconPaths }), template: '<svg viewBox="0 0 24 24" aria-hidden="true"><path v-for="path in iconPaths[name]" :d="path"/></svg>' });
    app.use(router);
    app.mount("#reference");
    globalThis.referenceRouter = router;
  }, { render, helpers, iconPaths, permissions, childPaths });
  const compare = async state => {
    await Promise.all([kbot, reference].map(page => page.mouse.move(0, 0)));
    await Promise.all([kbot, reference].map(page => page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))))));
    const [actual, expected] = await Promise.all([measure(kbot), measure(reference)]);
    assert.equal(actual.length, expected.length, `${state.page} 导航结构`);
    for (let index = 0; index < actual.length; index++) for (const key of Object.keys(actual[index])) {
      const a = actual[index][key], b = expected[index][key];
      if (typeof a === "number" ? Math.abs(a - b) > 1 : JSON.stringify(a) !== JSON.stringify(b)) differences.push({ ...state, index, key, actual: a, expected: b });
    }
  };
  for (const [page, suffix] of cases) {
    console.log(`侧栏对照：${page}`);
    await kbot.goto(`http://127.0.0.1:${server.address().port}/shell?page=${page}`);
    await kbot.locator(".business-app-sidebar").waitFor({ state: "attached" });
    assert.deepEqual(await kbot.locator(".business-context-icon svg path").evaluateAll(paths => paths.map(path => path.getAttribute("d"))), iconPaths[domainIcon], "业务空间图标须与源端相同");
    await kbot.addStyleTag({ content: "*,*::before,*::after{transition:none!important;animation:none!important}" });
    await reference.evaluate(async suffix => {
      const destination = `/apps/aiops/fixture${suffix}${suffix.includes('?') ? '&' : '?'}domainId=fixture`;
      if (referenceRouter.resolve(destination).matched.length !== 2) throw new Error(`基准页面没有真实路由：${destination}`);
      await referenceRouter.push(destination);
    }, suffix);
    for (const width of [1440, 900, 420]) {
      await Promise.all([kbot, reference].map(target => target.setViewportSize({ width, height: 1000 })));
      if (width <= 900 && await kbot.locator("#ops-nav-trigger").getAttribute("aria-expanded") !== "true") await kbot.locator("#ops-nav-trigger").click();
      await reference.evaluate(() => { referenceShell.mobileNavOpen.value = innerWidth <= 900; });
      await compare({ page, width, state: "initial" });
      const active = await reference.locator('.business-nav-group-toggle[aria-expanded="true"]').first();
      if (await active.count()) {
        const group = await active.locator("..").getAttribute("data-group-key");
        await Promise.all([kbot, reference].map(target => target.locator(`[data-group-key="${group}"] .business-nav-group-toggle`).click()));
        await compare({ page, width, state: "current-group-click" });
      }
    }
    await Promise.all([kbot, reference].map((target, index) => target.screenshot({ path: resolve(artifacts, `${index ? 'ammolite' : 'kbot'}-${page}.png`) })));
  }
  console.log(`侧栏截图：${artifacts}`);
  if (differences.length) console.log(JSON.stringify(differences.slice(0, 30), null, 2));
  assert.deepEqual(errors, []);
  assert.equal(differences.length, 0, `侧栏存在 ${differences.length} 项结构、交互或样式差异`);
  console.log(`通过：${cases.length} 个业务入口的真实侧栏模板、路由、图标、详情页归组和当前组交互的桌面/移动端对照。`);
} finally {
  await browser.close();
  await new Promise(done => server.close(done));
}
