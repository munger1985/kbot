// 使用两个仓库的真实页面模板和样式，比较工作台关键区域并验证交互；不连接真实后端。
// 运行：node tests/contract/aiops_chat_layout.mjs
// 依赖：相邻 Ammolite Portal 的 Playwright/Vue 开发依赖，可用 AMMOLITE_ROOT 指定其位置。
import assert from "node:assert/strict";
import { readFileSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const ammolite = process.env.AMMOLITE_ROOT || resolve(root, "../ammolite_cube");
const portal = resolve(ammolite, "apps/portal-web");
const require = createRequire(resolve(portal, "package.json"));
const { chromium } = require("@playwright/test");
const { compile } = require("@vue/compiler-dom");
const { compileStyle } = require("@vue/compiler-sfc");
const read = (path) => readFileSync(path, "utf8");
const typescript = require("typescript");
const referenceDateTime = typescript.transpileModule(read(resolve(portal, "src/shared/lib/dateTime.ts")), {
  compilerOptions: { module: typescript.ModuleKind.CommonJS, target: typescript.ScriptTarget.ES2022 },
}).outputText;
const diagnosis = read(resolve(portal, "src/features/aiops/business/AIOpsDiagnosisView.vue"));
const conversations = read(resolve(portal, "src/shared/ui/ConversationList.vue"));
const confirmation = read(resolve(portal, "src/shared/ui/ConfirmDialog.vue"));
const navIcon = read(resolve(portal, "src/shared/ui/NavIcon.vue"));
const iconPaths = new Function(`return (${navIcon.split("const iconPaths: Record<string, string[]> = ")[1].split("</script>")[0].trim()})`)();
const scoped = (source) => source.split("<style scoped>")[1].split("</style>")[0];
const style = (source, id) => compileStyle({ source: scoped(source), filename: "reference.vue", id, scoped: true }).code;
const sharedCss = read(resolve(portal, "src/shared/ui/styles.css")).replace(/@import url\([^)]*\);/g, "");
const template = diagnosis.split("<template>\n")[1];
const scaffold = template.split('<section v-for="turn in conversationTurns"')[0]
  .replace(/<section v-if="loading"[\s\S]*?<\/section>/, "")
  .replace("<template v-else>", "");
const composer = '<form class="agent-chat-composer' + template.split('<form class="agent-chat-composer')[1].split("</form>")[0] + "</form>";
const starterDrawer = '<Transition name="drawer">' + template.split('<Transition name="drawer">')[1].split('</Transition>')[0] + '</Transition>';
const reportDrawer = '<Transition name="drawer">' + template.split('<Transition name="drawer">')[2].split('</Transition>')[0] + '</Transition>';
const closureDrawer = ('<Transition name="drawer">' + template.split('<Transition name="drawer">')[3].split('</Transition>')[0] + '</Transition>').replaceAll(' as HTMLInputElement','');
const turnMessages = '<section v-for="turn in conversationTurns"' + template.split('<section v-for="turn in conversationTurns"')[1].split('<section v-if="progressItemsFor')[0];
const markdownAnswer = '<article v-if="block.block_type === \'MARKDOWN\'"' + template.split('<article v-else-if="block.block_type === \'MARKDOWN\'"')[1].split('</article>')[0] + '</article>';
const referenceRender = compile(`${scaffold}${turnMessages}<template v-for="block in turn.answer_blocks">${markdownAnswer}</template><p v-if="turn.error_message" class="aiops-diagnosis-notice" role="status">{{turn.error_code}}：{{turn.error_message}}</p></section><span class="aiops-transcript-end" aria-hidden="true"></span></div>${composer}</section></section>${starterDrawer}${reportDrawer}${closureDrawer}</main>`, { mode: "function" }).code;
const listTemplate = conversations.split("<template>\n")[1].split("</template>\n\n<style")[0]
  .replace(/<Teleport[\s\S]*?<\/Teleport>/, "");
const listRender = compile(listTemplate, { mode: "function" }).code;
const confirmationRender = compile(confirmation.split('<template>\n')[1].split('</template>')[0], {mode:'function'}).code;
const artifactDir = mkdtempSync(resolve(tmpdir(), "aiops-chat-alignment-"));
const errors = [];
const fixtureAgents = [
  { agent_id: "agent-1", display_name: "统一运维助手", status: "ACTIVE", allowed_business_system_ids: ["system-1"], capabilities: ["RESOURCE_CATALOG_QUERY", "MONITORING_QUERY", "ALERT_QUERY"] },
  { agent_id: "agent-2", display_name: "生产数据库运维助手", status: "ACTIVE", allowed_business_system_ids: ["system-1", "system-2"], capabilities: ["RESOURCE_CATALOG_QUERY"] },
];
const fixtureSessions = [{ conversation_id: "session-1", agent_id: "agent-1", title: "MES 系统诊断", source_type: "MANUAL", updated_at: "2026-10-30T03:00:00Z" }];
const fixtureReportTemplates=[{template_ref:'system-report',display_name:'系统诊断报告',status:'ACTIVE',version_no:1,sections:['EXECUTIVE_SUMMARY','FINDINGS','EVIDENCE_BOUNDARY'],system_defined:true,is_default:true},{template_ref:'custom-report',template_version_id:'version-1',display_name:'生产诊断报告',status:'ACTIVE',version_no:2,sections:['EXECUTIVE_SUMMARY','SCOPE','RECOMMENDATIONS','EVIDENCE_BOUNDARY'],system_defined:false,is_default:false}];
const fixtureStarter = {starter_id:'database.session.inspect', title:'数据库会话调查', description:'检查当前数据库会话和等待状态。', category:'DIAGNOSTIC', execution_mode:'DIAGNOSTIC', status:'AVAILABLE', supported_db_types:['ORACLE'], input_schema:[{name:'scope',label:'调查范围',type:'select',required:true,default:'active',options:[{value:'active',label:'活跃会话'},{value:'all',label:'全部会话'}]},{name:'limit',label:'最多返回条数',type:'integer',required:true,default:20,min:1,max:100,help:'只读取指定数量的会话。'}]};
const fixtureSystems = [{business_system_id:'system-1',display_name:'MES',environment:'PROD',targets:[{target_id:'target-1',resource_id:'resource-1',target_name:'MES 数据库',resource_name:'生产主库',db_type:'ORACLE',status:'ENABLED',readonly_connection_enabled:true,connectivity_status:'CONNECTED'}]}];
const fixtureShortcuts = [
  { capability: "RESOURCE_CATALOG_QUERY", title: "查询系统资源", description: "查看授权业务系统、资源类型和拓扑关系。" },
  { capability: "MONITORING_QUERY", title: "查看监控", description: "查看当前监控状态、关键指标和异常变化。" },
  { capability: "ALERT_QUERY", title: "查看告警", description: "查询近期告警、严重程度和处理状态。" },
];

const server = createServer((request, response) => {
  try {
    const path = resolve(root, `.${new URL(request.url, "http://localhost").pathname}`);
    if (!path.startsWith(`${root}/ui/`)) { response.writeHead(404).end(); return; }
    const mime = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript" };
    response.writeHead(200, { "Content-Type": `${mime[extname(path)] || "text/plain"}; charset=utf-8` });
    response.end(read(path));
  } catch { response.writeHead(404).end(); }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext();
  const kbot = await context.newPage();
  kbot.on("pageerror", (error) => errors.push(error.message));
  await kbot.route("**/aiops-auth.js*", (route) => route.fulfill({ contentType: "text/javascript", body: "" }));
  await kbot.addInitScript(({ agents, sessions, starter, systems }) => {
    globalThis.fixtureRequests = [];
    globalThis.fixtureAgents = agents;
    globalThis.KBotAIOpsAuth = {
      load: () => ({ access_token: "browser-fixture-only" }),
      uuid: () => crypto.randomUUID(),
      request: async (path, options = {}) => {
        globalThis.fixtureRequests.push({ path, method: options.method || "GET", body: options.body });
        if (options.method === "POST") {
          if (globalThis.fixtureHoldPost) await new Promise(resolve => {globalThis.fixtureReleasePost=resolve});
          throw new Error("浏览器契约测试不执行真实诊断");
        }
        if (path.endsWith("/access")) return { permissions: ["aiops:use"], user_id: "fixture", domain_id: "fixture" };
        if (path.endsWith("/agents")) return location.search.includes("empty-agents") ? [] : globalThis.fixtureAgents;
        if (path.includes("/conversation-starters?")) return { starters: [starter], business_systems: systems, catalog_version: "fixture" };
        if (path.endsWith('/session-report-templates')) return globalThis.fixtureReportTemplates || [];
        if(path.includes('/case-candidate')) return globalThis.fixtureCaseCandidate;
        if (path.includes("/conversations?")) return sessions;
        if (path.includes("/turns?")) return globalThis.fixtureTurns || [{ turn_id: "turn-1", turn_no: 1 }];
        if (globalThis.fixtureTurns?.some(turn=>path.endsWith(`/turns/${turn.turn_id}`))) return globalThis.fixtureTurns.find(turn=>path.endsWith(`/turns/${turn.turn_id}`));
        if (path.endsWith("/turns/turn-1")) return { turn_id: "turn-1", turn_no: 1, conversation_id: "session-1", status: "COMPLETED", ops_run_id: "run-1", messages: [{ message_type: "USER_MESSAGE", payload: { text: "诊断 <script> 与业务系统" } }], answer_blocks: [{ block_type: "MARKDOWN", payload: { markdown: "已完成诊断。" } }], events: [], evidence_gaps: [] };
        if (path.endsWith("/session-1")) return sessions[0];
        throw new Error(`未定义的契约请求：${path}`);
      },
    };
  }, { agents: fixtureAgents, sessions: fixtureSessions, starter:fixtureStarter, systems:fixtureSystems });
  const url = `http://127.0.0.1:${server.address().port}/ui/aiops/chat.html`;
  await kbot.goto(url);
  await kbot.addStyleTag({ content: "*, *::before, *::after { transition:none!important; animation:none!important; }" });
  await kbot.locator(".ops-starter-card").first().waitFor();
  assert.equal(await kbot.locator(".aiops-composer-send").isDisabled(), true);
  await kbot.locator('#conversation-form textarea').fill("诊断问题");

  const reference = await context.newPage();
  reference.on("pageerror", (error) => errors.push(`Ammolite 模板：${error.message}`));
  reference.on("console", (message) => { if (message.type() === "error") console.error(message.text()); });
  await reference.setContent('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"></head><body><div class="business-app-shell business-app-aiops"><header class="business-app-topbar"></header><div class="business-app-layout"><aside class="reference-sidebar"></aside><div id="reference" class="business-app-content is-viewport-contained"></div></div></div></body></html>');
  await reference.addStyleTag({ content: `${sharedCss}\n${style(diagnosis, "data-v-diagnosis")}\n${style(conversations, "data-v-conversations")}\n@media(max-width:900px) { .reference-sidebar { display:none; } }` });
  await reference.addStyleTag({ content: "*, *::before, *::after { transition:none!important; animation:none!important; }" });
  await reference.addScriptTag({ path: require.resolve("vue/dist/vue.global.prod.js") });
  await reference.addScriptTag({ content: `(function(){const exports={};${referenceDateTime}\nglobalThis.referenceFormatPortalDateTime=exports.formatPortalDateTime;})();` });
  for (const file of ['ui/vendor/marked.umd.js','ui/vendor/purify.min.js','ui/shared/kbot-markdown.js']) await reference.addScriptTag({path:resolve(root,file)});
  await reference.evaluate(({ render, list, agents, shortcuts, sessions, icons }) => {
    const icon = { props: ["name"], data: () => ({ icons }), template: '<svg viewBox="0 0 24 24" aria-hidden="true"><path v-for="path in icons[name] || icons.default" :key="path" :d="path"/></svg>' };
    const listComponent = {
      __scopeId: "data-v-conversations", props: ["items", "selectedId", "collapsed", "label"],
      data: () => ({ openId: null, menuStyle: {} }),
      computed: { groups() { return this.items.length ? [{ label: "今天", items: this.items }] : []; } },
      methods: { compactConversationLabel: (title) => Array.from(title).slice(0, 2).join(""), formatPortalDateTime: globalThis.referenceFormatPortalDateTime, toggle() {} },
      render: new Function("Vue", list)(Vue),
    };
    const app = Vue.createApp({
      __scopeId: "data-v-diagnosis",
      computed:{selectedAgent(){return this.availableAgents.find(agent=>agent.agent_id===this.selectedAgentId)}},
      components: { NavIcon: icon, ConversationList: listComponent, AgentMarkdownContent: {props:['markdown'],template:'<div class="agent-markdown-content" v-html="rendered"></div>',computed:{rendered(){return KBotMarkdown.render(this.markdown || '')}}} },
      data: () => ({ conversationTurns:[], loading: false, sidebarOpen: false, submitting: false, currentTurn: null, continuationSource: null, conversations: [], selectedConversationId: null, sessionItems: sessions.map((item) => ({ id: item.conversation_id, title: item.title, updatedAt: item.updated_at })), selectedAgentId: "agent-1", availableAgents: agents, canCloseDiagnosis: false, canGenerateReport: false, availableCapabilityShortcuts: shortcuts, starterHomeSections: [], selectedStarterId: "", diagnosticInProgress: false, clarificationPending: false, pendingFiles: [], question: "诊断问题", canSubmit: true, starterDrawerOpen:false,starterDrawerMode:'catalog',selectedStarter:null,allStarterSections:[],starterBusinessSystems:[],selectedStarterBusinessSystem:null,selectedStarterBusinessSystemId:'',selectedStarterResourceId:'',starterParameters:{} }),
      methods: { resetConversation() {}, selectSession() {}, deleteSession() {}, changeAgent() {}, executeCapabilityShortcut() {}, selectStarter() {}, openStarterCatalog() {this.starterDrawerOpen=true}, closeStarterDrawer(){this.starterDrawerOpen=false}, executeSelectedStarter(){}, changeStarterBusinessSystem(){},starterTargetSelectable(){return true},starterTargetStatus(){return '连接正常'},starterHint(){return '需要填写参数'}, returnToStarterCatalog(){},useAutomaticStarterConfiguration(){},openAttachmentPicker() {}, selectAttachments() {}, submitOnEnter() {}, submit() {} },
      render: new Function("Vue", render)(Vue),
    });
    app.config.globalProperties.$route = { meta: { title: "Agent 工作台" } };
    Object.assign(app._component.methods,{
      conversationMessages:turn=>(turn.messages || []).filter(message=>!(turn.answer_blocks?.length && message.role==='AGENT' && message.message_type==='ASSISTANT_MESSAGE')),
      textContent:message=>message.payload?.text || '',formatTime:value=>value?new Intl.DateTimeFormat('sv-SE',{year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date(value)):'—',
      displayText:text=>text,answerMarkdown:block=>block.payload.markdown,answerCreatedAt:turn=>[...(turn.messages || [])].reverse().find(message=>message.message_type==='ASSISTANT_MESSAGE')?.created_at,
      statusPresentation:status=>({COMPLETED:{label:'已完成',tone:'status-success'},PARTIAL:{label:'部分完成',tone:'status-warning'},FAILED:{label:'失败',tone:'status-danger'}}[status] || {label:status,tone:'status-muted'}),
      turnCapabilityLabels:turn=>turn.capability_plan.requested_capabilities.map(kind=>({RESOURCE_CATALOG_QUERY:'资源信息',MONITORING_QUERY:'监控现状'}[kind] || kind)),
      turnScopeLabel:turn=>(turn.scope_resolution.business_systems || []).map(item=>item.display_name).join('、'),
      reportTemplateVersion:item=>item.version_no ?? item.version ?? '—',
      reportTemplateSections:item=>item.sections.map(kind=>({EXECUTIVE_SUMMARY:'执行摘要',SCOPE:'诊断范围',FINDINGS:'诊断发现',RECOMMENDATIONS:'建议与行动',EVIDENCE_BOUNDARY:'证据边界'}[kind] || kind)).join('、'),
      openReportTemplateDrawer(){this.reportTemplateDrawerOpen=true},generateConversationReport(){},openCaseClosureDrawer(){},
      closureLines:value=>String(value || '').split('\n').map(item=>item.trim()).filter(Boolean),toggleClosureTurn(){},closeDiagnosis(){},
    });
    const initialData=app._component.data;
    app._component.data=()=>({...initialData(),reportTemplateDrawerOpen:false,reportTemplatesLoading:false,sessionReportTemplates:[],selectedReportTemplateRef:'',generatingReport:false,caseClosureLoading:false,caseClosureSaving:false,submittingClarificationId:'',clarificationFreeText:{},caseClosureDrawerOpen:false,caseCandidate:null,caseClosureDraft:{display_name:'',symptoms:'',root_cause_summary:'',actions:'',verification_result:'NOT_VERIFIED',action_evidence_ids:[],before_evidence_ids:[],after_evidence_ids:[],applicability:'',contraindications:''}});
    Object.assign(app._component.computed,{reportableTurnCount(){return this.conversationTurns.filter(turn=>['COMPLETED','PARTIAL'].includes(turn.status)).length},incompleteTurnCount(){return this.conversationTurns.length-this.reportableTurnCount}});
    const vm = app.mount("#reference");
    globalThis.referenceVm = vm;
  }, { render: referenceRender, list: listRender, agents: fixtureAgents, shortcuts: fixtureShortcuts, sessions: fixtureSessions, icons: iconPaths });
  await reference.locator("textarea").focus();
  await reference.evaluate(render=>{
    const host=document.createElement('div');document.body.append(host);
    globalThis.referenceConfirmVm=Vue.createApp({data:()=>({confirmState:{open:false,tone:'danger',title:'删除会话？',message:'“MES 系统诊断”将从会话列表中删除。已保存的诊断审计记录不受影响。',confirmLabel:'删除会话'}}),methods:{close(){this.confirmState.open=false},approve(){this.confirmState.open=false}},render:new Function('Vue',render)(Vue)}).mount(host);
  },confirmationRender);
  assert.equal(await kbot.locator(".conversation-copy small").first().textContent(), await reference.locator(".conversation-copy time").first().textContent(), "历史会话日期须使用源端统一格式");

  const selectors = [
    [".ops-head", ".aiops-diagnosis-page-head"],
    ["#chat-workspace", ".aiops-diagnosis-workspace"],
    [".ops-conversation-head", ".aiops-diagnosis-conversation-head"],
    ["#agent-select", ".aiops-diagnosis-head-context select"],
    ["#chat-session-toggle", ".aiops-diagnosis-conversation-head .button-row > button"],
    [".ops-starter-home", ".aiops-starter-home"],
    [".ops-starter-card", ".aiops-starter-option"],
    [".ops-composer", ".aiops-diagnosis-composer"],
    [".aiops-composer-row", ".aiops-composer-row"],
    [".aiops-composer-attachment", ".aiops-composer-attachment"],
    [".aiops-composer-send", ".aiops-composer-send"],
    [".aiops-diagnosis-rail-head", ".aiops-diagnosis-rail-head"],
    ["#new-conversation", ".aiops-diagnosis-rail-head > button"],
    ["#chat-rail-toggle", ".agent-chat-session-toggle"],
    [".ops-workspace-item", ".conversation-select"],
    [".ops-starter-section > h3", ".aiops-starter-section > h3"],
    [".ops-messages", ".aiops-diagnosis-transcript"],
    [".aiops-composer-row textarea", ".aiops-composer-row textarea"],
    [".agent-chat-composer-hint", ".agent-chat-composer-hint"],
    [".conversation-copy strong", ".conversation-copy strong"],
    [".ops-conversation-group h3", ".conversation-group h3"],
    [".conversation-copy small", ".conversation-copy small"],
    [".agent-chat-session-badge", ".agent-chat-session-badge"],
  ];
  const measure = (page, side) => page.evaluate(({ selectors, side }) => {
    return selectors.map((pair) => {
      const element = document.querySelector(pair[side]);
      const box = element.getBoundingClientRect();
      const css = getComputedStyle(element);
      return { x: box.x, y: box.y, width: box.width, height: box.height, background: css.backgroundColor, color: css.color, fontSize: css.fontSize, fontWeight:css.fontWeight, borderColor:css.borderColor, radius: css.borderRadius };
    });
  }, { selectors, side });

  const differences = [];
  for (const width of [1680, 1440, 1280, 1024, 900, 640, 420, 320]) {
    await Promise.all([kbot.setViewportSize({ width, height: 1000 }), reference.setViewportSize({ width, height: 1000 })]);
    for (const expanded of [false, true]) {
      await kbot.evaluate((expanded) => {
        const button = document.getElementById("chat-session-toggle");
        if ((button.getAttribute("aria-expanded") === "true") !== expanded) button.click();
        document.querySelector("#chat-workspace").style.transition = "none";
      }, expanded);
      await reference.evaluate((expanded) => { referenceVm.sidebarOpen = expanded; document.querySelector(".aiops-diagnosis-workspace").style.transition = "none"; }, expanded);
      const actual = await measure(kbot, 0);
      const expected = await measure(reference, 1);
      for (let index = 0; index < selectors.length; index++) {
        const keys = ["x", "y", "width", "height", "background", "color", "fontSize", "fontWeight", "borderColor", "radius"];
        for (const key of keys) {
          const matches = typeof actual[index][key] === "number" ? Math.abs(actual[index][key] - expected[index][key]) <= 1 : actual[index][key] === expected[index][key];
          if (!matches) differences.push({ width, expanded, selector: selectors[index][0], key, actual: actual[index][key], expected: expected[index][key] });
        }
      }
      if (width <= 900) {
        assert.ok(await kbot.locator("#agent-select").isVisible(), `${width}px Agent 选择器应可见`);
        assert.ok(await kbot.locator(".aiops-composer-send").isVisible(), `${width}px 发送按钮应可见`);
        assert.ok(await kbot.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${width}px 不应出现页面横向溢出`);
      }
      if (width === 1440 || width === 420) {
        await kbot.screenshot({ path: resolve(artifactDir, `kbot-${width}-${expanded ? "expanded" : "collapsed"}.png`), fullPage: true });
        await reference.screenshot({ path: resolve(artifactDir, `ammolite-${width}-${expanded ? "expanded" : "collapsed"}.png`), fullPage: true });
      }
    }
  }
  for (const width of [1440, 420]) {
    await Promise.all([kbot.setViewportSize({width,height:1000}),reference.setViewportSize({width,height:1000})]);
    await kbot.locator('#open-starter-menu').click();
    await reference.evaluate(({starter,systems}) => { referenceVm.starterDrawerOpen=true; referenceVm.starterDrawerMode='catalog'; referenceVm.allStarterSections=[{category:'DIAGNOSTIC',label:'诊断检查',starters:[starter]}]; referenceVm.starterBusinessSystems=systems; },{starter:fixtureStarter,systems:fixtureSystems});
    for (const mode of ['catalog','parameters','target-selected']) {
      if (mode === 'parameters') {
        await kbot.locator('#starter-dialog [data-starter-id]').click();
        await reference.evaluate(starter=>{ referenceVm.starterDrawerMode='parameters'; referenceVm.selectedStarter=starter; referenceVm.selectedStarterId=starter.starter_id;referenceVm.starterParameters={scope:'active',limit:20};referenceVm.selectedStarterBusinessSystemId='';referenceVm.selectedStarterBusinessSystem=null;referenceVm.selectedStarterResourceId=''; },fixtureStarter);
      }
      if (mode === 'target-selected') {
        await kbot.selectOption('#starter-dialog [name="starter_business_system_id"]','system-1');
        await kbot.locator('#starter-dialog [name="starter_resource_id"]').check();
        await reference.evaluate(system=>{ referenceVm.selectedStarterBusinessSystemId=system.business_system_id;referenceVm.selectedStarterBusinessSystem=system;referenceVm.selectedStarterResourceId=system.targets[0].resource_id; },fixtureSystems[0]);
      }
      await Promise.all([kbot,reference].map(page=>page.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))))));
      const pairs = [['#starter-dialog','.aiops-starter-drawer'],['#starter-dialog .admin-drawer-header','.aiops-starter-drawer .admin-drawer-header'],['#starter-dialog-content','.aiops-starter-drawer .admin-drawer-body'],['#starter-dialog .admin-drawer-footer','.aiops-starter-drawer .admin-drawer-footer'],...(mode === 'catalog' ? [['#starter-dialog .ops-starter-card','.aiops-starter-drawer .aiops-starter-option']] : [['#starter-dialog .aiops-starter-resource-scope','.aiops-starter-drawer .aiops-starter-resource-scope'],['#starter-dialog .control-field','.aiops-starter-drawer .control-field'],...(mode === 'target-selected' ? [['#starter-dialog .aiops-starter-target-list','.aiops-starter-drawer .aiops-starter-target-list']] : [])])];
      const measureDrawer=(page,side)=>page.evaluate(({pairs,side})=>pairs.map(pair=>{const node=document.querySelector(pair[side]);const box=node.getBoundingClientRect(),css=getComputedStyle(node);return {x:box.x,y:box.y,width:box.width,height:box.height,background:css.backgroundColor,fontSize:css.fontSize,borderColor:css.borderColor};}),{pairs,side});
      const actual=await measureDrawer(kbot,0),expected=await measureDrawer(reference,1);
      for(let index=0;index<pairs.length;index++) for(const key of Object.keys(actual[index])) if(typeof actual[index][key]==='number'?Math.abs(actual[index][key]-expected[index][key])>1:actual[index][key]!==expected[index][key]) differences.push({width,state:mode,selector:pairs[index][0],key,actual:actual[index][key],expected:expected[index][key]});
      await kbot.screenshot({path:resolve(artifactDir,`kbot-starter-${mode}-${width}.png`)});
      await reference.screenshot({path:resolve(artifactDir,`ammolite-starter-${mode}-${width}.png`)});
    }
    await kbot.locator('#starter-dialog .admin-drawer-header button').click();
    await reference.evaluate(()=>{referenceVm.starterDrawerOpen=false;referenceVm.selectedStarterBusinessSystem=null;referenceVm.selectedStarterResourceId='';});
  }
  for (const width of [1440,420]) {
    await Promise.all([kbot.setViewportSize({width,height:1000}),reference.setViewportSize({width,height:1000})]);
    if(await kbot.locator('#chat-session-toggle').getAttribute('aria-expanded')!=='true') await kbot.locator('#chat-session-toggle').click();
    await kbot.locator('.ops-workspace-more').click();
    await kbot.locator('.ops-conversation-menu button').click();
    await reference.evaluate(()=>{referenceConfirmVm.confirmState.open=true});
    await Promise.all([kbot,reference].map(page=>page.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))))));
    const collect=page=>page.evaluate(()=>['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-message','.confirm-dialog-actions','.confirm-dialog-actions button'].map(selector=>{const node=document.querySelector(selector),box=node.getBoundingClientRect(),css=getComputedStyle(node);return {selector,x:box.x,y:box.y,width:box.width,height:box.height,color:css.color,background:css.backgroundColor,fontSize:css.fontSize};}));
    const actual=await collect(kbot),expected=await collect(reference);
    for(let index=0;index<actual.length;index++) for(const key of Object.keys(actual[index])) if(typeof actual[index][key]==='number'?Math.abs(actual[index][key]-expected[index][key])>1:actual[index][key]!==expected[index][key]) differences.push({width,state:'delete-confirmation',selector:actual[index].selector,key,actual:actual[index][key],expected:expected[index][key]});
    await kbot.screenshot({path:resolve(artifactDir,`kbot-delete-confirmation-${width}.png`)});
    await reference.screenshot({path:resolve(artifactDir,`ammolite-delete-confirmation-${width}.png`)});
    await kbot.locator('.confirm-dialog [value="cancel"]').click();
    await reference.evaluate(()=>{referenceConfirmVm.confirmState.open=false});
    assert.equal(await kbot.evaluate(()=>fixtureRequests.filter(request=>request.method==='DELETE').length),0);
  }
  console.log(`截图：${artifactDir}`);
  writeFileSync(resolve(artifactDir, 'differences.json'), JSON.stringify(differences, null, 2));
  if (differences.length) console.log(JSON.stringify(differences.slice(0, 80), null, 2));
  assert.equal(differences.length, 0, `工作台与 Ammolite 存在 ${differences.length} 项几何或样式差异`);

  await kbot.setViewportSize({ width: 1440, height: 1000 });
  await kbot.selectOption("#agent-select", "agent-2");
  await kbot.waitForFunction(() => document.querySelectorAll(".ops-starter-card").length === 1);
  assert.equal(await kbot.locator(".ops-starter-card").count(), 1);
  await kbot.locator('#conversation-form textarea').fill("诊断问题");
  await kbot.locator('#conversation-form textarea').press("Shift+Enter");
  assert.match(await kbot.locator('#conversation-form textarea').inputValue(), /\n/);
  await kbot.locator('#conversation-form textarea').dispatchEvent("keydown", { key: "Enter", isComposing: true });
  assert.equal(await kbot.evaluate(() => fixtureRequests.filter((item) => item.method === "POST").length), 0);
  await kbot.locator('#conversation-form textarea').press("Enter");
  await kbot.waitForFunction(() => fixtureRequests.some((item) => item.method === "POST"));
  assert.equal(await kbot.locator('#conversation-form textarea').isEnabled(), true);
  await kbot.locator("#chat-session-toggle").click();
  if (await kbot.locator("#chat-workspace").evaluate((element) => element.classList.contains("is-rail-collapsed"))) await kbot.locator("#chat-session-toggle").click();
  await kbot.locator('.ops-workspace-item[data-id="session-1"]').click();
  await kbot.waitForFunction(() => document.getElementById("conversation-title").textContent === "MES 系统诊断");
  assert.equal(await kbot.locator("#agent-select").inputValue(), "agent-1");
  assert.equal(await kbot.locator('#conversation-actions [data-close-diagnosis]').count(), 1);
  assert.equal(await kbot.locator('#conversation-actions [data-generate-report-conversation]').textContent(), "生成会话报告");
  assert.equal(await kbot.locator(".aiops-chat-message-user .ops-message-content").textContent(), "诊断 <script> 与业务系统");
  assert.equal(await kbot.locator(".ops-message-content script").count(), 0);
  await kbot.locator("#new-conversation").click();
  assert.equal(await kbot.locator("#conversation-title").textContent(), "新运维会话");
  assert.equal(await kbot.locator(".ops-starter-home").count(), 1);
  assert.equal(new URL(kbot.url()).search, "");
  await kbot.locator('#open-starter-menu').click();
  await kbot.locator('#starter-dialog [data-starter-id]').click();
  await kbot.selectOption('#starter-dialog [name="starter_business_system_id"]','system-1');
  await kbot.locator('#starter-dialog [name="starter_resource_id"]').check();
  await kbot.evaluate(()=>{globalThis.fixtureHoldPost=true});
  await kbot.locator('#starter-dialog [name="limit"]').press('Enter');
  await kbot.waitForFunction(()=>Boolean(globalThis.fixtureReleasePost));
  assert.equal(await kbot.locator('#starter-dialog .admin-drawer-header button').isDisabled(),true);
  await kbot.keyboard.press('Escape');
  assert.equal(await kbot.locator('#starter-dialog').evaluate(dialog=>dialog.open),true);
  await kbot.evaluate(()=>fixtureReleasePost());
  await kbot.waitForFunction(()=>!document.getElementById('starter-dialog').dataset.busy);
  assert.equal(await kbot.locator('#starter-dialog [name="limit"]').inputValue(),'20');
  assert.equal(await kbot.locator('#starter-dialog [data-starter-submit]').isEnabled(),true);
  await kbot.locator('#starter-dialog .admin-drawer-header button').click();
  const turns=[
    {turn_id:'text-turn-1',turn_no:1,conversation_id:'session-1',status:'COMPLETED',capability_plan:{requested_capabilities:['RESOURCE_CATALOG_QUERY','MONITORING_QUERY']},scope_resolution:{status:'RESOLVED',scope_type:'BUSINESS_SYSTEM',business_systems:[{display_name:'MES 生产系统'}],resources:[]},messages:[{message_type:'USER_MESSAGE',role:'USER',created_at:'2026-10-30T03:00:00Z',payload:{text:'诊断 <script> 与业务系统'}},{message_type:'ASSISTANT_MESSAGE',role:'AGENT',created_at:'2026-10-30T03:00:01Z',payload:{text:'数据库运行正常。'}}],answer_blocks:[],events:[],evidence_gaps:[]},
    {turn_id:'text-turn-2',turn_no:2,conversation_id:'session-1',status:'PARTIAL',messages:[{message_type:'USER_MESSAGE',role:'USER',created_at:'2026-10-30T03:00:02Z',payload:{text:'再检查一次。'}},{message_type:'ASSISTANT_MESSAGE',role:'AGENT',created_at:'2026-10-30T03:00:03Z',payload:{text:'不能重复展示这段消息。'}}],answer_blocks:[{block_no:1,block_type:'MARKDOWN',payload:{markdown:'已完成只读检查。'}}],events:[],evidence_gaps:[]},
    {turn_id:'text-turn-3',turn_no:3,conversation_id:'session-1',status:'FAILED',error_code:'FIXTURE_ERROR',error_message:'隔离环境错误状态',messages:[{message_type:'USER_MESSAGE',role:'USER',created_at:'2026-10-30T03:00:04Z',payload:{text:'检查不可达实例。'}}],answer_blocks:[],events:[],evidence_gaps:[]},
  ];
  await kbot.evaluate(turns=>{globalThis.fixtureTurns=turns},turns);
  if(await kbot.locator('#chat-session-toggle').getAttribute('aria-expanded')!=='true') await kbot.locator('#chat-session-toggle').click();
  await kbot.locator('.ops-workspace-item[data-id="session-1"]').click();
  await kbot.locator('.aiops-diagnosis-turn').first().waitFor();
  await reference.evaluate(({turns,sessions})=>{referenceVm.conversationTurns=turns;referenceVm.selectedConversationId='session-1';referenceVm.currentTurn=turns.at(-1);referenceVm.conversations=sessions;referenceVm.question='';referenceVm.canSubmit=false;referenceVm.sidebarOpen=true;referenceVm.canGenerateReport=true;referenceVm.canCloseDiagnosis=true;}, {turns,sessions:fixtureSessions});
  for(const width of [1440,1024,900,640,420,320]) {
    await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
    await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;return new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))})));
    const selectors=['.aiops-diagnosis-conversation-head','.aiops-diagnosis-turn','.aiops-turn-marker','.aiops-turn-marker .status-pill','.aiops-chat-message-user','.aiops-chat-message-user .chat-message-avatar','.aiops-chat-message-user .aiops-chat-message-body','.aiops-chat-message-user header','.aiops-chat-message-user .aiops-chat-bubble','.aiops-chat-message-agent','.aiops-chat-message-agent .chat-message-avatar','.aiops-chat-message-agent .aiops-chat-message-body','.aiops-chat-message-agent .aiops-chat-bubble','.aiops-chat-message-agent .aiops-chat-bubble p','.aiops-turn-orchestration','.aiops-turn-orchestration header','.aiops-turn-orchestration dl','.aiops-diagnosis-turn:nth-of-type(3) .aiops-diagnosis-notice'];
    const measure=page=>page.evaluate(selectors=>selectors.map(selector=>{const node=document.querySelector(selector);if(!node)return null;const box=node.getBoundingClientRect(),css=getComputedStyle(node);return {x:box.x,y:box.y,width:box.width,height:box.height,fontSize:css.fontSize,fontWeight:css.fontWeight,color:css.color,background:css.backgroundColor,border:css.border,borderRadius:css.borderRadius};}),selectors);
    const actual=await measure(kbot),expected=await measure(reference);
    for(let index=0;index<selectors.length;index++) {
      assert.equal(Boolean(actual[index]),Boolean(expected[index]),selectors[index]);
      if(!actual[index])continue;
      for(const key of Object.keys(actual[index])) if(typeof actual[index][key]==='number'?Math.abs(actual[index][key]-expected[index][key])>1:actual[index][key]!==expected[index][key]) differences.push({width,state:'messages',selector:selectors[index],key,actual:actual[index][key],expected:expected[index][key]});
    }
    for(const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifactDir,`${prefix}-messages-${width}.png`),fullPage:true});
  }
  writeFileSync(resolve(artifactDir,'differences.json'),JSON.stringify(differences,null,2));
  if(differences.length) console.log(JSON.stringify(differences.slice(0,40),null,2));
  assert.equal(differences.length,0,'消息、轮次与回答必须继续对齐真实模板');
  for(const empty of [false,true]) for(const width of [1440,420]) {
    await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
    await kbot.evaluate(templates=>{globalThis.fixtureReportTemplates=templates},empty?[]:fixtureReportTemplates);
    await kbot.locator('[data-generate-report-conversation]').click();
    await reference.evaluate(templates=>{referenceVm.sessionReportTemplates=templates;referenceVm.selectedReportTemplateRef=templates[0]?.template_ref || '';referenceVm.reportTemplateDrawerOpen=true;},empty?[]:fixtureReportTemplates);
    await Promise.all([kbot,reference].map(page=>page.locator('.aiops-report-template-drawer').waitFor()));
    await Promise.all([kbot,reference].map(page=>page.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))))));
    const selectors=['.aiops-report-template-drawer','.aiops-report-template-drawer .admin-drawer-header','.aiops-report-template-form','.aiops-report-scope','.aiops-report-template-option','.aiops-report-template-drawer .aiops-starter-empty','.aiops-report-template-drawer .admin-drawer-footer'];
    const measure=page=>page.evaluate(selectors=>selectors.map(selector=>{const node=document.querySelector(selector);if(!node)return null;const box=node.getBoundingClientRect(),css=getComputedStyle(node);return {x:box.x,y:box.y,width:box.width,height:box.height,background:css.backgroundColor,fontSize:css.fontSize,border:css.border};}),selectors);
    const actual=await measure(kbot),expected=await measure(reference);
    for(let index=0;index<selectors.length;index++) {
      assert.equal(Boolean(actual[index]),Boolean(expected[index]),selectors[index]);
      if(!actual[index])continue;
      for(const key of Object.keys(actual[index])) if(typeof actual[index][key]==='number'?Math.abs(actual[index][key]-expected[index][key])>1:actual[index][key]!==expected[index][key]) differences.push({width,state:'report-template',empty,selector:selectors[index],key,actual:actual[index][key],expected:expected[index][key]});
    }
    for(const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifactDir,`${prefix}-report-templates-${empty?'empty':'populated'}-${width}.png`)});
    assert.equal(await kbot.locator('.aiops-report-template-drawer [type="submit"]').isDisabled(),empty);
    if(!empty && width===1440) {
      await kbot.locator('.aiops-report-template-drawer input[value="custom-report"]').check();
      await kbot.evaluate(()=>{globalThis.fixtureRequests=[];globalThis.fixtureHoldPost=true;});
      await kbot.locator('.aiops-report-template-drawer [type="submit"]').click();
      await kbot.waitForFunction(()=>Boolean(globalThis.fixtureReleasePost));
      await kbot.keyboard.press('Escape');
      assert.equal(await kbot.locator('.aiops-report-template-drawer').evaluate(dialog=>dialog.open),true);
      assert.deepEqual(await kbot.evaluate(()=>JSON.parse(fixtureRequests.find(item=>item.method==='POST').body)),{conversation_id:'session-1',template_ref:'custom-report',template_version_id:'version-1',period_kind:'AD_HOC'});
      await kbot.evaluate(()=>{fixtureHoldPost=false;fixtureReleasePost();delete globalThis.fixtureReleasePost;});
      await kbot.waitForFunction(()=>!document.querySelector('.aiops-report-template-drawer').dataset.busy);
      assert.equal(await kbot.locator('.aiops-report-template-drawer input[value="custom-report"]').isChecked(),true);
    }
    await kbot.getByRole('button',{name:'关闭报告模板选择'}).click();
    await reference.evaluate(()=>{referenceVm.reportTemplateDrawerOpen=false;});
  }
  writeFileSync(resolve(artifactDir,'differences.json'),JSON.stringify(differences,null,2));
  if(differences.length) console.log(JSON.stringify(differences.slice(0,40),null,2));
  assert.equal(differences.length,0,'会话报告模板选择须与源端一致');
  const candidate={eligible:true,reasons:[],display_name:'生产慢查询调查',symptoms:['MES 响应缓慢'],root_cause_summary:'等待事件与执行计划已核对。',environment:{database_type:'ORACLE',database_version:'19c',platform:'Linux'},source_fingerprint:'fixture-fingerprint',included_turn_ids:['text-turn-1'],turn_candidates:[{turn_id:'text-turn-1',turn_no:1,eligible:true,selected:true,target_id:'target-1',summary:'只读诊断完成'},{turn_id:'text-turn-3',turn_no:3,eligible:false,selected:false,target_id:'target-1',reason:'诊断未完成'}],evidence_candidates:[{artifact_id:'action-1',role:'ACTION',label:'受控动作回执',created_at:'2026-10-30T03:00:00Z'},{artifact_id:'evidence-1',role:'DIAGNOSIS',label:'当前会话取证',created_at:'2026-10-30T03:00:00Z'}]};
  for(const eligible of [true,false]) for(const width of [1440,420]) {
    const preview={...candidate,eligible,reasons:eligible?[]:[{code:'INCOMPLETE',message:'当前诊断尚无可追溯的有效判断。'}]};
    await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
    await kbot.evaluate(preview=>{globalThis.fixtureCaseCandidate=preview},preview);
    await kbot.locator('[data-close-diagnosis]').click();
    await reference.evaluate(preview=>{referenceVm.caseCandidate=preview;referenceVm.caseClosureDraft={display_name:preview.display_name,symptoms:preview.symptoms.join('\n'),root_cause_summary:preview.root_cause_summary,actions:'',verification_result:'NOT_VERIFIED',action_evidence_ids:[],before_evidence_ids:[],after_evidence_ids:[],applicability:'',contraindications:''};referenceVm.caseClosureDrawerOpen=true;},preview);
    await Promise.all([kbot,reference].map(page=>page.locator('.aiops-case-closure-drawer').waitFor()));
    await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.aiops-case-closure-drawer').scrollTop=0;return new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))})));
    const selectors=['.aiops-case-closure-drawer','.aiops-case-closure-drawer .admin-drawer-header','.aiops-case-closure-form','.aiops-case-closure-empty','.aiops-case-evidence-section','.aiops-case-evidence-section .check-field','.aiops-case-closure-summary','.aiops-case-closure-form > .control-field','.aiops-case-evidence-grid','.aiops-case-evidence-grid fieldset','.aiops-case-evidence-grid label','.aiops-case-closure-two-column','.aiops-case-closure-drawer .admin-drawer-footer'];
    const measure=page=>page.evaluate(selectors=>selectors.map(selector=>{const node=document.querySelector(selector);if(!node)return null;const box=node.getBoundingClientRect(),css=getComputedStyle(node);return{x:box.x,y:box.y,width:box.width,height:box.height,background:css.backgroundColor,color:css.color,fontSize:css.fontSize,border:css.border};}),selectors);
    const actual=await measure(kbot),expected=await measure(reference);
    for(let index=0;index<selectors.length;index++) {
      assert.equal(Boolean(actual[index]),Boolean(expected[index]),selectors[index]);
      if(!actual[index])continue;
      for(const key of Object.keys(actual[index])) if(typeof actual[index][key]==='number'?Math.abs(actual[index][key]-expected[index][key])>1:actual[index][key]!==expected[index][key]) differences.push({width,state:'closure',eligible,selector:selectors[index],key,actual:actual[index][key],expected:expected[index][key]});
    }
    for(const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifactDir,`${prefix}-closure-${eligible?'eligible':'ineligible'}-${width}.png`)});
    if(eligible && width===1440) {
      await kbot.locator('[name="before_evidence_ids"][value="evidence-1"]').check();
      assert.equal(await kbot.locator('[name="after_evidence_ids"][value="evidence-1"]').isDisabled(),true);
      await kbot.locator('.aiops-case-closure-form [name="display_name"]').fill('   ');
      assert.equal(await kbot.locator('.aiops-case-closure-form [type="submit"]').isDisabled(),true);
      await kbot.locator('.aiops-case-closure-form [name="display_name"]').fill(candidate.display_name);
      await kbot.evaluate(()=>{globalThis.fixtureRequests=[];globalThis.fixtureHoldPost=true;});
      await kbot.locator('.aiops-case-closure-form [type="submit"]').click();
      await kbot.waitForFunction(()=>Boolean(globalThis.fixtureReleasePost));
      await kbot.keyboard.press('Escape');
      assert.equal(await kbot.locator('.aiops-case-closure-drawer').evaluate(dialog=>dialog.open),true);
      assert.equal(await kbot.getByRole('button',{name:'关闭诊断收口',exact:true}).isDisabled(),true);
      const body=await kbot.evaluate(()=>JSON.parse(fixtureRequests.find(item=>item.method==='POST').body));
      assert.deepEqual(body.included_turn_ids,['text-turn-1']);
      assert.equal(body.source_fingerprint,'fixture-fingerprint');
      assert.deepEqual(body.before_evidence_ids,['evidence-1']);
      await kbot.evaluate(()=>{fixtureHoldPost=false;fixtureReleasePost();delete globalThis.fixtureReleasePost;});
      await kbot.waitForFunction(()=>!document.querySelector('.aiops-case-closure-drawer').dataset.busy);
      assert.equal(await kbot.locator('.aiops-case-closure-form [name="display_name"]').inputValue(),candidate.display_name);
    }
    await kbot.getByRole('button',{name:'关闭诊断收口',exact:true}).click();
    await reference.evaluate(()=>{referenceVm.caseClosureDrawerOpen=false;});
  }
  writeFileSync(resolve(artifactDir,'differences.json'),JSON.stringify(differences,null,2));
  if(differences.length)console.log(JSON.stringify(differences.slice(0,40),null,2));
  assert.equal(differences.length,0,'诊断收口须与源端一致');
  await kbot.goto(`${url}?empty-agents`);
  await kbot.locator("#agent-empty-notice").waitFor();
  assert.equal(await kbot.locator('#conversation-form textarea').isDisabled(), true);
  assert.equal(await kbot.locator(".aiops-composer-send").isDisabled(), true);
  assert.equal(await kbot.locator("#add-evidence").isDisabled(), true);
  assert.deepEqual(errors, []);
  console.log(`通过：8 种宽度 × 展开/收起 × ${selectors.length} 个区域；多轮消息/范围卡片/完成及失败状态；快捷功能、会话报告模板、诊断收口抽屉的桌面及窄屏对照；键盘/输入法、保存失败恢复和隔离请求合同。`);
} finally {
  await browser.close();
  await new Promise((done) => server.close(done));
}
