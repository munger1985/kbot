// 验证正式 Shell 的权限过滤、详情页归组与手风琴交互，不连接真实后端。
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const source = readFileSync(new URL("../../ui/aiops/js/aiops-shell.js", import.meta.url), "utf8");
const permissions = [
  "aiops:use", "aiops:work_item_handle", "aiops:target_manage",
  "aiops:diagnostic_source_manage", "aiops:work_item_manage", "aiops:knowledge_manage",
  "aiops:agent_manage", "aiops:plan_manage", "aiops:api_key_manage",
];

function setup(page, allowed = permissions, domain = { domain_id: 87, domain_name: "生产运维空间" }) {
  const document = {
    body: { dataset: { page } },
    querySelectorAll: () => [],
  };
  const context = { document, addEventListener() {} };
  // 只在隔离 VM 暴露真实函数，生产脚本不增加测试入口。
  runInNewContext(source.replace("globalThis.KBotAIOpsShell = {", "globalThis.navigationContract = { shellMarkup, bindNavigationGroups }; globalThis.KBotAIOpsShell = {"), context);
  const markup = context.navigationContract.shellMarkup({ permissions: allowed, ...domain });
  const sections = [...markup.matchAll(/<section class="business-nav-group ([^"]+)" data-group-key="([^"]+)" data-current="(true|false)">([\s\S]*?)<\/section>/g)];
  const groups = sections.map(([, classes, key, current, content]) => {
    const classNames = new Set(classes.split(" "));
    const attributes = { "aria-expanded": String(classNames.has("is-open")) };
    const button = {
      getAttribute: name => attributes[name],
      setAttribute: (name, value) => { attributes[name] = value; },
      addEventListener: (name, callback) => { assert.equal(name, "click"); button.click = callback; },
    };
    const items = { hidden: !classNames.has("is-open") };
    return {
      key, button, items, dataset: { current },
      classList: { toggle(name, value) { value ? classNames.add(name) : classNames.delete(name); } },
      querySelector(selector) {
        if (selector === ".business-nav-group-toggle") return button;
        if (selector === ".nav-group-items") return items;
        if (selector === ".business-route-active") return content.includes('aria-current="page"') ? {} : null;
        throw new Error(`未声明的导航选择器：${selector}`);
      },
      isOpen: () => classNames.has("is-open"),
    };
  });
  document.querySelectorAll = selector => { assert.equal(selector, ".business-nav-group"); return groups; };
  context.navigationContract.bindNavigationGroups();
  return {
    markup, groups,
    openKeys: () => groups.filter(group => group.isOpen()).map(group => group.key),
    click: key => groups.find(group => group.key === key).button.click(),
    assertSynchronized() {
      for (const group of groups) {
        assert.equal(group.button.getAttribute("aria-expanded"), String(group.isOpen()));
        assert.equal(group.items.hidden, !group.isOpen());
      }
    },
  };
}

test("业务空间显示接口返回的真实名称，不显示数字 ID", () => {
  for (const page of ["chat", "dashboard", "monitoring", "agents", "business-system-workspace"]) {
    assert.match(setup(page).markup, /id="ops-domain">生产运维空间<\/strong>/);
    assert.doesNotMatch(setup(page).markup, /Domain 87/);
  }
});

test("名称缺失或为空时明确显示不可用，不退回 ID", () => {
  for (const domain_name of [undefined, null, "", "   "]) {
    const app = setup("chat", permissions, { domain_id: 87, domain_name });
    assert.match(app.markup, /id="ops-domain">名称暂不可用<\/strong>/);
    assert.doesNotMatch(app.markup, /Domain 87/);
  }
});

test("业务空间名称按文本转义，不能注入顶部 HTML", () => {
  const app = setup("chat", permissions, { domain_id: 87, domain_name: '  运维 <img src=x onerror="alert(1)"> & 生产  ' });
  assert.match(app.markup, /id="ops-domain">运维 &lt;img src=x onerror=&quot;alert\(1\)&quot;&gt; &amp; 生产<\/strong>/);
  assert.doesNotMatch(app.markup, /<img/);
});

test("Agent 工作台固定入口默认不展开业务分组，可独立打开并收起非当前组", () => {
  const app = setup("chat");
  assert.deepEqual(app.openKeys(), []);
  assert.match(app.markup, /business-nav-pinned[\s\S]*?aria-current="page"/);
  app.click("operations-work");
  assert.deepEqual(app.openKeys(), ["operations-work"]);
  app.click("knowledge-assets");
  assert.deepEqual(app.openKeys(), ["knowledge-assets"]);
  app.click("knowledge-assets");
  assert.deepEqual(app.openKeys(), []);
  app.assertSynchronized();
});

for (const [page, group] of [
  ["dashboard", "operations-work"], ["work-item-detail", "operations-work"],
  ["inspection-plan-detail", "operations-work"], ["cases", "knowledge-assets"],
  ["business-system-workspace", "configuration-governance"],
  ["diagnostic-source-detail", "configuration-governance"], ["run-detail", "configuration-governance"],
]) {
  test(`${page} 自动展开正确分组，点击当前组保持展开，仍允许切换其他组`, () => {
    const app = setup(page);
    assert.deepEqual(app.openKeys(), [group]);
    app.click(group);
    assert.deepEqual(app.openKeys(), [group]);
    const other = group === "knowledge-assets" ? "operations-work" : "knowledge-assets";
    app.click(other);
    assert.deepEqual(app.openKeys(), [other]);
    app.click(other);
    assert.deepEqual(app.openKeys(), []);
    app.click(group);
    app.click(group);
    assert.deepEqual(app.openKeys(), [group]);
    app.assertSynchronized();
  });
}

test("导航仅暴露授权功能，未授权分组不显示也不参与展开", () => {
  const app = setup("chat", ["aiops:use"]);
  assert.deepEqual(app.groups.map(group => group.key), [
    "operations-work", "knowledge-assets", "configuration-governance",
  ]);
  assert.doesNotMatch(app.markup, /href="\.\/(?:agents|business-systems|recovery-drills|api-clients)\.html"/);
  assert.match(app.markup, /href="\.\/runs\.html"/);
  const management = setup("agents", ["aiops:agent_manage"]);
  assert.deepEqual(management.openKeys(), ["configuration-governance"]);
  assert.doesNotMatch(management.markup, /business-nav-pinned/);
  assert.equal(management.groups.length, 1);
});

test("独立详情页仅展开归属分组，目录抽屉入口仍选中目录项", () => {
  for (const page of ["work-item-detail", "business-system-workspace"]) {
    const app = setup(page);
    assert.doesNotMatch(app.markup, /aria-current="page"/);
    assert.equal(app.groups.filter(group => group.dataset.current === "true").length, 1);
  }
  for (const [page, directory] of [["inspection-plan-detail", "inspections"], ["run-detail", "runs"], ["diagnostic-source-detail", "diagnostic-sources"]]) {
    assert.match(setup(page).markup, new RegExp(`href="\\./${directory}\\.html" aria-current="page"`));
  }
});

test("报告中心独立位于运维工作与知识资产之间，报告详情不展开任何组", () => {
  for (const page of ["reports", "report-detail"]) {
    const app = setup(page);
    assert.deepEqual(app.openKeys(), []);
    const markers = ['class="business-nav-pinned"', 'data-group-key="operations-work"', 'class="business-nav-direct"', 'data-group-key="knowledge-assets"', 'data-group-key="configuration-governance"'];
    const positions = markers.map(marker => app.markup.indexOf(marker));
    assert.ok(positions.every((value, index) => value >= 0 && (!index || value > positions[index - 1])));
    assert.equal([...app.markup.matchAll(/href="\.\/reports\.html"/g)].length, 1);
    assert.match(app.markup, /class="business-nav-direct"[^>]*>[\s\S]*?business-route-active[\s\S]*?>报告中心<\/span>/);
    assert.match(app.markup, page === "reports" ? /reports\.html" aria-current="page"/ : /reports\.html" aria-current="location"/);
  }
  assert.doesNotMatch(setup("reports", ["aiops:agent_manage"]).markup, /href="\.\/reports\.html"/);
});

test("配置与治理保留原七项并追加通知入口，知识内容入口使用独立名称", () => {
  const {markup} = setup("chat");
  const configuration = markup.split('data-group-key="configuration-governance"')[1].split('</section>')[0];
  const actual = [...configuration.matchAll(/href="\.\/([^".]+)\.html"[\s\S]*?class="nav-item-label">([^<]+)</g)].map(([,page,label]) => [page,label]);
  assert.deepEqual(actual, [["business-systems", "业务系统"], ["diagnostic-sources", "监控源"], ["responsibility-groups", "责任组"], ["knowledge-bases", "运维知识库"], ["agents", "Agent 管理"], ["report-templates", "模板中心"], ["runs", "运行记录"], ["notification-channels", "通知渠道"], ["notification-delivery", "通知投递"]]);
  assert.match(markup, /href="\.\/cases\.html"[\s\S]*?>诊断案例库<\/span>/);
  assert.match(markup, /href="\.\/operations-knowledge\.html"[\s\S]*?>运维资料库<\/span>/);
});

test("固定入口与组内链接都保留源模板的图标结构，当前主题决定可见性", () => {
  const app = setup("chat");
  const links = [...app.markup.matchAll(/<a class="nav-item business-nav-item[^>]*>([\s\S]*?)<\/a>/g)];
  assert.equal(links.length, 18);
  for (const [, content] of links) {
    assert.match(content, /<span class="nav-item-icon"><svg[^>]*>[\s\S]*?<path d="[^"]+"/);
    assert.match(content, /<span class="nav-item-label">/);
  }
});
