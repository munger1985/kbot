// 隔离浏览器回归：node tests/contract/knowledge_search_pdf.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const root = new URL("../../", import.meta.url);
const portal = new URL("../Ammolite/apps/portal-web/", root);
const require = createRequire(process.env.KBOT_BROWSER_REQUIRE_FROM || new URL("package.json", portal));
const { chromium, expect } = require("@playwright/test");
const { parse, compileScript, compileStyle } = require("@vue/compiler-sfc");
const ts = require("typescript");
const origin = "http://knowledge-search.test";

async function loadPortalComponent(page, name, path) {
  const filename = path.split("/").at(-1);
  const id = "data-v-test-" + name.toLowerCase();
  const { descriptor, errors } = parse(await readFile(new URL(path, portal), "utf8"), { filename });
  assert.deepEqual(errors, []);
  const code = ts.transpileModule(compileScript(descriptor, { id, inlineTemplate: true }).content,
    { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText
    .replace(/import\s*\{([^}]+)\}\s*from\s*['"]vue['"];?/gs, (_, names) => `const { ${names.replace(/\s+as\s+/g, ": ")} } = Vue;`)
    .replace(/^import .*?;\s*$/gm, "").replace("export default", "return")
    .replace("import.meta.env.BASE_URL", '"/"');
  const scoped = descriptor.styles.some(style => style.scoped);
  await page.addScriptTag({ content: `window.${name} = (() => { ${code} })();` +
    (scoped ? `window.${name}.__scopeId = "${id}";` : "") });
  for (const style of descriptor.styles) {
    const result = compileStyle({ filename, id, scoped: style.scoped, source: style.content });
    assert.deepEqual(result.errors, []);
    await page.addStyleTag({ content: result.code });
  }
}

function fixturePdf(chinese = false) {
  const stream = (text) => `<< /Length ${text.length} >>\nstream\n${text}endstream`;
  // 同一段落在第 2 页重复两次，另在第 1 页放一个同词副本。
  const cjk = text => Array.from(text, character => character.charCodeAt(0).toString(16).padStart(4, "0")).join("");
  const operands = chinese
    ? `<${cjk("审计人员熟悉内")}> Tj 0 -22 Td <${cjk("部控制规范。")}> Tj`
    : "(Review exact) Tj ( evidence.) Tj";
  const line = y => `BT /F1 14 Tf 50 ${y} Td ${operands} ET\n`;
  const page = contents => `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents ${contents} 0 R >>`;
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Count 2 /Kids [3 0 R 4 0 R] >>",
    page(6), page(7),
    chinese ? "<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [8 0 R] >>" :
      "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    stream(line(700)), stream(line(700) + line(130)),
  ];
  if (chinese) objects.push(
    "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light /CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 4 >> /DW 1000 >>",
  );
  let pdf = "%PDF-1.4\n";
  const offsets = [0];
  for (const [index, object] of objects.entries()) {
    offsets.push(pdf.length);
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
  }
  const xref = pdf.length;
  pdf += `xref\n0 ${offsets.length}\n0000000000 65535 f \n`;
  pdf += offsets.slice(1).map(offset => `${String(offset).padStart(10, "0")} 00000 n \n`).join("");
  return `${pdf}trailer\n<< /Size ${offsets.length} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
}

const pdf = fixturePdf();
const box = [0.08, 0.81, 0.35, 0.85];
const block = { anchor: true, locators: [{ page_no: 2, bbox: box }], segments: [
  { text: "Review ", matched: false }, { text: "exact evidence", matched: true }, { text: ".", matched: false },
] };
const context = { pdf_page: 2, has_before: false, has_after: false, blocks: [{ ...block, key: "block-2" }] };
const evidence = (key, version_id, source_title, chapter, page) => ({
  key, version_id, source_title, chapter, kind: "TEXT", passage_key: key,
  passage_display: { chapter_key: chapter, order: page, kind: "PARAGRAPH", pages: [page], title: key, text: "Review exact evidence." },
  content_ref: { retrieval_ref: "receipt", key },
});
const items = [
  evidence("段落一", "version-1", "审计指南", "审计流程", 2),
  evidence("段落二", "version-1", "审计指南", "审计原则", 1),
  evidence("段落三", "version-2", "质量手册", null, 2),
];
const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jS6sAAAAASUVORK5CYII=", "base64");
const imageItems = ["figure", "unavailable"].map(key => ({
  key, version_id: "version-1", source_title: "图像资料", chapter: null, kind: "FIGURE",
  passage_key: null, mime_type: "image/png", byte_count: png.length, locators: [{ page_no: 2 }],
  content_ref: { retrieval_ref: "receipt", key },
}));

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.setDefaultTimeout(5000);
  const errors = [];
  const downloads = [];
  let searchImages = false;
  page.on("download", download => downloads.push(download.suggestedFilename()));
  page.on("pageerror", error => errors.push(error.message));
  await page.route(`${origin}/**`, async route => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/ui/knowledge_retrieval/search.html") {
      return route.fulfill({ path: fileURLToPath(new URL(path.slice(1), root)) });
    }
    if (path === "/ui/runtime-config.js") {
      return route.fulfill({ contentType: "application/javascript", body: `
        window.KBotKnowledgeShell = { ready: Promise.resolve({ permissions: ["knowledge_retrieval:use"] }),
          escapeHtml: value => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll('"', "&quot;") };
        window.KBotKnowledgeApi = {
          async request(path) {
            const response = await fetch(path);
            if (!response.ok) throw new Error("Unexpected request: " + path);
            return response.json();
          },
          async requestBlob(path) {
            const response = await fetch(path);
            if (!response.ok) throw new Error("Unexpected request: " + path);
            return response.blob();
          }
        };` });
    }
    if (/^\/ui\/knowledge_retrieval\/js\/knowledge-(?:auth|api|shell)\.js$/.test(path)) {
      return route.fulfill({ contentType: "application/javascript", body: "" });
    }
    if (path.startsWith("/ui/")) return route.fulfill({ path: fileURLToPath(new URL(path.slice(1), root)) });
    if (path.startsWith("/vendor/")) return route.fulfill({ path: fileURLToPath(new URL("public" + path, portal)) });
    if (path === "/portal") return route.fulfill({ contentType: "text/html", body: '<div id="app"></div>' });
    if (path === "/knowledge/libraries") return route.fulfill({ json: [
      { library_id: "audit", display_name: "审计知识库", status: "ACTIVE", permissions: ["READ"], search_capabilities: [] },
    ] });
    if (path === "/knowledge/libraries/audit/search") {
      const results = searchImages ? imageItems : items;
      return route.fulfill({ json: {
        status: "COMPLETE", items: results, primary_hits: results.map(({ key }) => ({ key })), linked: [],
      } });
    }
    if (path === "/knowledge/retrievals/receipt/content/figure")
      return route.fulfill({ body: png, contentType: "image/png" });
    if (path === "/knowledge/retrievals/receipt/content/unavailable")
      return route.fulfill({ body: png, contentType: "application/octet-stream", headers: { "Content-Disposition": "attachment" } });
    if (path === "/knowledge/retrievals/receipt/" + "context/" + encodeURIComponent("段落一") + "/pdf")
      return route.fulfill({ body: pdf, contentType: "application/pdf" });
    if (path === "/knowledge/retrievals/receipt/context/" + encodeURIComponent("段落一")) return route.fulfill({ json: context });
    if (path === "/knowledge/retrievals/receipt/context/" + encodeURIComponent("段落二")) {
      return route.fulfill({ json: { ...context, pdf_page: 1, blocks: [{ ...block, key: "block-1",
        locators: [{ page_no: 1, bbox: [0.08, 0.09, 0.35, 0.14] }] }] } });
    }
    if (path === "/knowledge/retrievals/receipt/context/" + encodeURIComponent("段落二") + "/pdf")
      return route.fulfill({ body: pdf, contentType: "application/pdf" });
    return route.fulfill({ status: 404 });
  });

  async function checkImagePreview() {
    searchImages = true;
    await page.getByRole("button", { name: "检索", exact: true }).click();
    const figures = page.locator(".knowledge-search-hit").filter({ hasText: "图示" });
    await expect(figures).toHaveCount(2);
    await figures.first().click();
    const preview = page.locator(".knowledge-search-reader img.knowledge-preview");
    await expect(preview).toBeVisible();
    assert.equal(await preview.evaluate(node => node.naturalWidth), 1);
    await figures.last().click();
    await expect(page.locator('[role="alert"]:visible').filter({ hasText: "证据预览格式不匹配。" })).toBeVisible();
    await expect(preview).toHaveCount(0);
    assert.deepEqual(downloads, []);
    searchImages = false;
  }

  await page.goto(`${origin}/ui/knowledge_retrieval/search.html`);
  await page.locator("[data-library]").selectOption("audit");
  await page.getByLabel("查询文字").fill("exact evidence");
  await page.getByRole("button", { name: "检索", exact: true }).click();
  const docs = page.locator(".knowledge-search-source");
  await expect(docs).toHaveCount(2);
  await expect(docs.nth(1)).not.toHaveAttribute("open", "");
  const chapters = docs.first().locator(".knowledge-search-chapter");
  await chapters.last().locator("summary").click();
  await expect(chapters.last()).not.toHaveAttribute("open", "");
  await page.getByRole("button", { name: /段落一/ }).click();
  const reader = page.locator(".knowledge-pdf-viewport");
  const hits = reader.locator(".knowledge-pdf-hit");
  await expect(reader).toContainText("已定位并标黄命中文字。", { timeout: 10000 });
  await expect(reader.locator(".knowledge-pdf-paper")).toHaveAttribute("data-pdf-page", "2");
  await expect(hits.first()).toBeInViewport();
  assert.ok(await reader.evaluate(node => node.scrollTop > 0));
  assert.ok(await hits.evaluateAll(nodes => nodes.length && nodes.every(node => parseFloat(node.style.top) > 80)));
  await expect(chapters.last()).not.toHaveAttribute("open", "");
  await chapters.last().locator("summary").click();
  await page.getByRole("button", { name: /段落二/ }).click();
  await expect(reader.locator(".knowledge-pdf-paper")).toHaveAttribute("data-pdf-page", "1");
  await expect(reader).toContainText("已定位并标黄命中文字。");
  assert.ok(await hits.evaluateAll(nodes => nodes.length && nodes.every(node => parseFloat(node.style.top) < 20)));
  await page.getByRole("button", { name: "放大阅读区" }).click();
  await expect(page.getByRole("dialog").locator(".knowledge-pdf-hit").first()).toBeInViewport();
  assert.ok((await page.getByRole("dialog").boundingBox()).width > 1100);
  await page.getByRole("dialog").getByRole("button", { name: "下一页" }).click();
  await expect(reader.locator(".knowledge-pdf-paper")).toHaveAttribute("data-pdf-page", "2");
  await expect(hits).toHaveCount(0);
  await page.getByRole("button", { name: "回到命中页" }).click();
  await expect(hits.first()).toBeInViewport();
  await page.getByRole("button", { name: "关闭", exact: true }).click();
  await page.getByRole("button", { name: "文字上下文" }).click();
  await expect(page.locator(".knowledge-passage mark")).toHaveText("exact evidence");
  await checkImagePreview();
  await page.evaluate(() => dispatchEvent(new Event("kbot:knowledge-context-changing")));
  await expect(reader).toHaveCount(0);
  await expect(docs).toHaveCount(0);

  await page.goto(`${origin}/portal`);
  await page.addStyleTag({ content: (await readFile(new URL("src/shared/ui/styles.css", portal), "utf8")).replace(/^@import[^\n]+/gm, "") });
  await page.addScriptTag({ path: fileURLToPath(new URL("node_modules/vue/dist/vue.global.js", portal)) });
  await page.evaluate(() => {
    window.confirmState = { open: false };
    window.testRoute = Vue.reactive({ query: { domainId: "1" }, params: { domainCode: "audit" } });
    window.useRoute = () => window.testRoute;
    window.useBusinessAppPermissions = () => Vue.ref({ "knowledge_retrieval:use": true });
    window.matrix = { createMemberMatrix: () => ({}) };
    window.ApiError = class extends Error {};
    for (const name of ["CatalogCardActions", "KnowledgeParseModelFields", "KnowledgeIndexModelFields",
      "KnowledgeSearchCapabilitiesReview", "KnowledgeTaskProgress", "RouterLink"]) window[name] = { render: () => null };
    const fetchKnowledge = async (path, { init }) => {
      const response = await fetch(path.slice(path.indexOf("/knowledge/")), init);
      if (!response.ok) throw new Error("Unexpected request: " + path);
      return response;
    };
    window.businessFetch = async (...args) => (await fetchKnowledge(...args)).json();
    window.businessBlobFetch = async (...args) => {
      const response = await fetchKnowledge(...args);
      return { blob: await response.blob(), contentType: response.headers.get("content-type") };
    };
  });
  const components = "src/features/knowledge-retrieval/business/";
  await loadPortalComponent(page, "AdminDrawer", "src/shared/ui/AdminDrawer.vue");
  await loadPortalComponent(page, "KnowledgeSearchPdf", components + "KnowledgeSearchPdf.vue");
  await loadPortalComponent(page, "KnowledgeLibraryWorkspace", components + "KnowledgeLibraryWorkspace.vue");
  await page.evaluate(() => {
    Vue.createApp(window.KnowledgeLibraryWorkspace, { view: "search" }).mount("#app");
  });
  assert.deepEqual(errors, []);
  await page.locator(".knowledge-search-top select").selectOption("audit");
  await page.getByLabel("查询文字").fill("exact evidence");
  await page.getByRole("button", { name: "检索", exact: true }).click();
  await expect(docs).toHaveCount(2);
  await expect(docs.nth(1)).not.toHaveAttribute("open", "");
  const portalChapters = docs.first().locator(".knowledge-search-chapter");
  await portalChapters.last().locator("summary").click();
  await expect(portalChapters.last()).not.toHaveAttribute("open", "");
  await page.getByRole("button", { name: /段落一/ }).click();
  await expect(reader).toContainText("已定位并标黄命中文字。");
  await expect(hits.first()).toBeInViewport();
  await expect(portalChapters.last()).not.toHaveAttribute("open", "");
  await page.getByRole("button", { name: "放大阅读区" }).click();
  await expect(page.getByRole("dialog").locator(".knowledge-pdf-hit").first()).toBeInViewport();
  assert.ok((await page.getByRole("dialog").boundingBox()).width > 1100);
  await page.getByRole("button", { name: "关闭", exact: true }).click();
  await checkImagePreview();
  await page.evaluate(() => { window.testRoute.query.domainId = "2"; });
  await expect(reader).toHaveCount(0);
  await expect(docs).toHaveCount(0);

  // 验证中文字体和跨文字块匹配，以及无法消除重复文本歧义时的回退。
  await page.evaluate(async () => {
    const { mountPdfSearchReader } = await import("/vendor/pdf-search-reader.js");
    const host = Object.assign(document.createElement("div"), { className: "knowledge-pdf-viewport" });
    document.body.append(host);
    const reader = mountPdfSearchReader(host);
    window.fixtureRender = (pdf, blocks) => {
      const url = URL.createObjectURL(new Blob([pdf], { type: "application/pdf" }));
      reader.update({ url, page: 2, hitPage: 2, zoom: 0, blocks });
    };
  });
  const cjkBlock = { ...block, locators: [
    { page_no: 2, bbox: [0.08, 0.81, 0.27, 0.84] },
    { page_no: 2, bbox: [0.08, 0.84, 0.24, 0.88] },
  ], segments: [
    { text: "审计人员熟悉", matched: false }, { text: "内部控制规范", matched: true }, { text: "。", matched: false },
  ] };
  await page.evaluate(({ pdf, block }) => window.fixtureRender(pdf, [block]), { pdf: fixturePdf(true), block: cjkBlock });
  await expect(reader).toContainText("已定位并标黄命中文字。");
  const rows = await hits.evaluateAll(nodes => nodes.map(node => parseFloat(node.style.top)));
  assert.equal(rows.length, 2);
  assert.ok(rows[0] > 80 && rows[1] > rows[0]);
  await page.evaluate(({ pdf, block }) => window.fixtureRender(pdf, [block]), { pdf,
    block: { ...block, locators: [{ page_no: 2, bbox: [0, 0, 1, 1] }] } });
  await expect(reader).toContainText("仅定位到页");
  await expect(hits).toHaveCount(0);
  assert.deepEqual(errors, []);
  console.log("通过：双端原文定位、图片在线预览（不下载）、双级折叠、放大阅读和工作域清理。");
} finally {
  await browser.close();
}
