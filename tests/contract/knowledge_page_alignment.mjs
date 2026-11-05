// 用真实 Ammolite Vue 页面验收 KBot；接口全部隔离，不操作真实资料。
import assert from "node:assert/strict";
import {readFileSync,mkdtempSync,writeFileSync} from "node:fs";
import {createServer} from "node:http";
import {createRequire} from "node:module";
import {tmpdir} from "node:os";
import {resolve,dirname,extname} from "node:path";
import {fileURLToPath} from "node:url";
const root=resolve(dirname(fileURLToPath(import.meta.url)),"../..");
const portal=resolve(process.env.AMMOLITE_ROOT || resolve(root,"../ammolite_cube"),"apps/portal-web");
const require=createRequire(resolve(portal,"package.json"));
const {chromium}=require("@playwright/test");
const {createServer:createViteServer}=await import(require.resolve("vite"));
const vue=(await import(require.resolve("@vitejs/plugin-vue"))).default;
const manifest=JSON.parse(readFileSync(resolve(root,"ui/knowledge_retrieval/reference-pages.json"),"utf8"));
const pages=Object.entries(manifest.pages).filter(([,entry])=>entry.sources.length).map(([key])=>key);
const artifacts=mkdtempSync(resolve(tmpdir(),"knowledge-page-alignment-"));
const permissions=["use","knowledge_chat","knowledge_core_manage","data_model_manage","agent_manage","x_search","domain_manage"].map(key=>"knowledge_retrieval:"+key);
const fixture={
  permissions,
  core:{collection_id:"core-1",display_name:"产品知识库",description:"产品手册与业务资料",bundle_count:2,status:"ACTIVE",row_version:1,models:{}},
  agent:{agent_id:"agent-1",display_name:"企业知识助手",description:"围绕产品与业务资料回答问题。",status:"ACTIVE",enabled_capabilities:["DOCUMENT"],knowledge_core_id:"core-1",data_model_ids:[],version_no:1,row_version:1,models:{}},
  source:{data_source_id:"source-1",display_name:"销售分析主库",source_type:"ORACLE",status:"ACTIVE",current_version:1},
  model:{semantic_model_id:"model-1",display_name:"销售分析模型",description:"销售分析业务口径",active_version:1,status:"ACTIVE",versions:[{version_no:1,status:"ACTIVE",definition:{datasets:[]}}]},
  review:{bundle_id:"bundle-1",bundle_revision_id:"revision-1",title:"产品手册",document_names:["产品手册.pdf"],approval_status:"PENDING"},
  processing:{bundle_id:"bundle-1",bundle_revision_id:"revision-1",title:"产品手册",revision_no:1,status:"READY",current_stage:"AVAILABLE",progress_percent:100,file_count:1,ready_count:1,failed_count:0,reviewed_at:null,completed_at:null,jobs:[],files:[{document_version_id:"document-1",name:"产品手册.pdf",status:"READY",jobs:[]}]},
  file:{bundle_id:"bundle-1",bundle_revision_id:"revision-1",bundle_title:"产品手册",revision_no:1,document_version_id:"document-1",file_name:"产品手册.pdf",status:"READY",mime_type:"application/pdf",byte_size:1024,received_at:null,completed_at:null,preview_available:true},
};
function installFixture(data,empty){
  const value=key=>empty?[]:[data[key]];
  globalThis.fixtureRequests=[];globalThis.fixtureUnexpected=[];
  globalThis.fixtureRequest=async(path,init={})=>{
    const url=new URL(path,location.origin),raw=url.pathname;
    const ref=raw.includes("/knowledge_retrieval/");
    let suffix=ref?raw.replace(/^.*\/knowledge_retrieval\/domains\/[^/]+/,""):raw.replace("/api/v1/apps/knowledge-retrieval","");
    const payload=init.body instanceof FormData?{grouping:init.body.get("grouping_mode"),files:JSON.parse(init.body.get("files") || "[]"),bundle:JSON.parse(init.body.get("bundle") || "null")}:init.body;
    fixtureRequests.push({path:raw,search:url.search,method:init.method || "GET",body:payload});
    if(init.method && init.method!=="GET"){
      if(globalThis.fixtureFailNext){globalThis.fixtureFailNext=false;throw new Error("隔离失败，请重试");}
      if(suffix.endsWith("/uploads") || suffix.endsWith("/ingestions/user-files")) {
        const files=payload.files;
        return {request_id:"fixture-upload",items:globalThis.fixtureRejectUpload?files.map((file,index)=>({status:index===0?"REJECTED":"ACCEPTED",client_file_id:file.client_file_id,message:index===0?"文件未通过受理":null})):[{status:"ACCEPTED",bundle_id:"bundle-1",bundle_revision_id:"revision-1"}]};
      }
      return {};
    }
    if(suffix==="/access")return {permissions:data.permissions,domain_id:1,domain_name:"企业知识工作域",user_id:"fixture-user",display_name:"知识管理员"};
    if(suffix==="/bootstrap")return {agents:value("agent"),collections:value("core")};
    if(suffix==="/agents" || suffix==="/manage/agents")return value("agent");
    if(suffix==="/agents/agent-1")return data.agent;
    if(suffix==="/agents/options")return {knowledge_cores:value("core"),data_models:[],models:[]};
    if(suffix==="/collections")return {collections:value("core")};
    if(suffix==="/manage/collections" || suffix==="/knowledge-cores")return value("core");
    if(suffix.endsWith("/model-catalog"))return [];
    if(suffix==="/conversations")return [];
    if(suffix.endsWith("/connector-capabilities"))return {items:[]};
    if(suffix==="/data-models/data-sources" || suffix==="/manage/data-sources")return {items:value("source")};
    if(suffix==="/data-models" || suffix==="/manage/semantic-models")return {items:value("model")};
    if(suffix==="/data-models/model-1")return data.model;
    if(suffix.endsWith("/approvals") || suffix.endsWith("/reviews"))return {items:value("review")};
    if(suffix.endsWith("/processing"))return {items:value("processing"),total:empty?0:globalThis.fixtureTotal || 1,page:Number(url.searchParams.get("page") || 1),page_size:20};
    if(suffix.endsWith("/files"))return {items:value("file"),total:empty?0:globalThis.fixtureTotal || 1,page:Number(url.searchParams.get("page") || 1),page_size:20};
    if(suffix.endsWith("/preview"))return {title:"产品手册",files:[{document_version_id:"document-1",declared_name:"产品手册.pdf",member_status:"READY",preview_available:true}]};
    if(suffix.endsWith("/snapshots"))return {items:[]};
    fixtureUnexpected.push(suffix);throw new Error("未隔离接口 "+suffix);
  };
  const fetchOriginal=globalThis.fetch;
  globalThis.fetch=async(input,init={})=>{
    const url=typeof input==="string"?input:input.url;
    if(!new URL(url,location.origin).pathname.startsWith("/api/"))return fetchOriginal(input,init);
    try{return new Response(JSON.stringify(await fixtureRequest(url,init)),{headers:{"content-type":"application/json"}});}
    catch(error){return new Response(JSON.stringify({detail:error.message}),{status:503,headers:{"content-type":"application/json"}});}
  };
  sessionStorage.setItem("kbot.knowledge.session.v1",JSON.stringify({access_token:"fixture-session",domain_id:1,domain_name:"企业知识工作域",display_name:"知识管理员",user_id:"fixture-user"}));
}
const httpModule=`export class ApiError extends Error{}; export const businessFetch=(path,options={})=>fixtureRequest(path,options.init||{});export const apiFetch=(path,options={})=>fixtureRequest(path,options);export const businessBlobFetch=async()=>new Blob([]);export const businessEventStream=async()=>new Response("");`;
const entry=`
import {createApp,computed} from 'vue';
import {createRouter,createWebHistory,RouterView} from 'vue-router';
import {businessAppPermissionsKey,businessAppPrincipalKey} from '/src/app/businessAppContext.ts';
import '/src/shared/ui/styles.css';
const name=new URLSearchParams(location.search).get('page'),empty=new URLSearchParams(location.search).get('empty')==='1';
const pages={${pages.map(key=>JSON.stringify(key)+":()=>import('/src/features/knowledge-retrieval/"+manifest.pages[key].sources[0]+"')").join(",")}};
const component=(await pages[name]()).default;
const router=createRouter({history:createWebHistory(),routes:[{path:'/apps/knowledge-retrieval/:domainCode/:pathMatch(.*)*',component,meta:{title:${JSON.stringify(Object.fromEntries(pages.map(key=>[key,manifest.pages[key].title])))}[name]}}]});
const app=createApp(RouterView);app.use(router);
app.provide(businessAppPermissionsKey,computed(()=>Object.fromEntries(${JSON.stringify(permissions)}.map(key=>[key,true]))));
app.provide(businessAppPrincipalKey,computed(()=>({id:'fixture-user',display_name:'知识管理员',username:'fixture-user'})));
await router.replace('/apps/knowledge-retrieval/fixture/'+name+'?domainId=1');
if(name==='knowledge')document.querySelector('#reference').classList.add('is-viewport-contained','is-agent-chat');
app.mount('#reference');globalThis.referenceMounted=true;`;
const vite=await createViteServer({root:portal,configFile:false,cacheDir:resolve(artifacts,"vite-cache"),plugins:[vue(),{name:"knowledge-reference",enforce:"pre",resolveId(id){if(id==="/fixture-entry.js")return "\0fixture-entry";if(["@/shared/api/http",resolve(portal,"src/shared/api/http.ts")].includes(id))return "\0fixture-http";},load(id){if(id==="\0fixture-entry")return entry;if(id==="\0fixture-http")return httpModule;},configureServer(server){server.middlewares.use((req,res,next)=>{if(!req.url.startsWith("/reference.html"))return next();res.setHeader("Content-Type","text/html");res.end('<!doctype html><html lang="zh-CN"><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><div class="business-app-shell business-app-knowledge_retrieval"><header class="business-app-topbar"></header><div class="business-app-layout"><aside class="business-app-sidebar"></aside><div class="business-app-content" id="reference"></div></div></div><style>*,*::before,*::after{transition:none!important;animation:none!important}</style><script type="module" src="/fixture-entry.js"></script></body></html>');});}}],resolve:{alias:[{find:"@",replacement:resolve(portal,"src")}]},server:{host:"127.0.0.1",port:0},logLevel:"error"});
const server=createServer((req,res)=>{try{const pathname=new URL(req.url,"http://fixture").pathname;if(pathname==="/ui/runtime-config.js"){res.setHeader("Content-Type","text/javascript");res.end("globalThis.KBOT_UI_CONFIG={mainApiBaseUrl:location.origin};");return;}const path=resolve(root,"."+pathname);if(!path.startsWith(root+"/ui/")){res.writeHead(404).end();return;}res.setHeader("Content-Type",({".html":"text/html",".css":"text/css",".js":"text/javascript"})[extname(path)]||"text/plain");res.end(readFileSync(path));}catch{res.writeHead(404).end();}});
const differences=[],errors=[];
async function compare(kbot,reference,selectors,context){
 const measure=page=>page.evaluate(selectors=>selectors.map(selector=>{
   const node=[...document.querySelectorAll(selector)].find(node=>node.getBoundingClientRect().width && node.getBoundingClientRect().height);
   if(!node)return null;
   const box=node.getBoundingClientRect(),css=getComputedStyle(node);
   return {x:Math.round(box.x),width:Math.round(box.width),fontSize:css.fontSize,fontWeight:css.fontWeight,color:css.color,background:css.backgroundColor,borderRadius:css.borderRadius,padding:css.padding};
 }),selectors);
 const [actual,expected]=await Promise.all([measure(kbot),measure(reference)]);
 for(let index=0;index<selectors.length;index++){
   if(!expected[index] || !actual[index]){if(Boolean(expected[index])!==Boolean(actual[index]))differences.push({...context,selector:selectors[index],key:"presence",actual:!!actual[index],expected:!!expected[index]});continue;}
   for(const key of Object.keys(expected[index])){
     // 文本列随平台操作和文案变化分配空间；比较表格容器几何与单元格完整视觉规格。
     if (key === "width" && /(?:th|td|h1)$/.test(selectors[index])) continue;
     const a=actual[index][key],e=expected[index][key];
     if(typeof e==="number"?Math.abs(a-e)>2:a!==e)differences.push({...context,selector:selectors[index],key,actual:a,expected:e});
   }
 }
}
let browser;
try{
 await vite.listen();await new Promise(done=>server.listen(0,"127.0.0.1",done));
 const refOrigin="http://127.0.0.1:"+vite.httpServer.address().port,kbotOrigin="http://127.0.0.1:"+server.address().port;
 browser=await chromium.launch({headless:true});
 for(const empty of [false,true])for(const name of pages){
   console.log("对照："+name+(empty?"/空数据":"/有数据"));
   const pair=await Promise.all([0,1].map(async()=>{const page=await browser.newPage({viewport:{width:1440,height:1000}});page.on("pageerror",error=>errors.push(name+":"+error.message));await page.route(/^https?:\/\//,route=>new URL(route.request().url()).hostname === "127.0.0.1" ? route.continue() : route.abort());await page.addInitScript(({data,empty,install})=>{(0,eval)("("+install+")")(data,empty);},{data:fixture,empty,install:installFixture.toString()});return page;}));
   const [kbot,reference]=pair;
   await Promise.all([kbot.goto(kbotOrigin+"/ui/knowledge_retrieval/"+name+".html"),reference.goto(refOrigin+"/reference.html?page="+name+(empty?"&empty=1":""))]);
   await kbot.waitForFunction(()=>document.body.dataset.access==="ready");
   await reference.waitForFunction(()=>globalThis.referenceMounted);
   await Promise.all(pair.map(page=>page.waitForLoadState("networkidle")));
   for(const page of pair)await page.addStyleTag({content:"*,*::before,*::after{transition:none!important;animation:none!important}"});
   for(const width of [1440,1024,390,320]){
     await Promise.all(pair.map(page=>page.setViewportSize({width,height:1000})));
     const selectors=name==="knowledge"?[".knowledge-chat-workspace",".knowledge-conversation-header",".knowledge-agent-picker",".knowledge-composer-wrap",".knowledge-chat-welcome h2"]:[".platform-home-hero",".platform-home-hero h1",".section-toolbar","main > .table-card",".table-card th",".table-card td"];
     if(!empty && ["processing","files"].includes(name))selectors.push(".knowledge-list-toolbar");
     if(!empty && name==="reviews")selectors.push(".review-queue-header",".review-queue-toolbar");
     if(!empty && name==="upload")selectors.push(".knowledge-upload-workbench",".upload-section-heading",".upload-drop-zone",".upload-submit-bar");
     // 来源的零知识库解析页没有结束 loading；KBot 使用明确空态，差异在 manifest 中登记。
     if(empty && name==="processing")selectors.splice(selectors.indexOf("main > .table-card"),1);
     if(name!=="knowledge" || !empty)await compare(kbot,reference,selectors,{name,width,empty});
     for(const page of pair)assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,name+"/"+width+" 页面溢出");
     if(width===1440 || width===390)for(const [prefix,page]of [["kbot",kbot],["ammolite",reference]])await page.screenshot({path:resolve(artifacts,prefix+"-"+name+"-"+(empty?"empty":"populated")+"-"+width+".png"),fullPage:true});
   }
   if(!empty && name==="upload"){
     for(const page of pair){await page.setViewportSize({width:1440,height:1000});await page.locator('input[type="radio"][value="SINGLE_BUNDLE"]').check();await page.locator('input[type="file"]').setInputFiles([{name:"手册.txt",mimeType:"text/plain",buffer:Buffer.from("隔离资料")},{name:"补充.txt",mimeType:"text/plain",buffer:Buffer.from("补充资料")}]);assert.equal(await page.locator(".upload-file-row").count(),2);await page.locator(".upload-file-settings select").last().selectOption("SUPPLEMENT");}
     await compare(kbot,reference,[".upload-mode-option",".upload-file-row",".upload-file-settings",".upload-submit-bar"],{name,state:"bundle-upload",width:1440});
     for(const page of pair){await page.getByRole("button",{name:"提交资料",exact:true}).click();await page.locator(".upload-receipt").waitFor();const upload=await page.evaluate(()=>fixtureRequests.find(row=>row.method==="POST"));assert.equal(upload.body.grouping,"SINGLE_BUNDLE");assert.equal(upload.body.files[1].role,"SUPPLEMENT");assert.equal(upload.body.files[0].content_sha256.length,64);assert.equal(await page.locator(".upload-file-row").count(),0);}
     await kbot.locator('input[value="EACH_FILE"]').check();
     await kbot.locator('input[type="file"]').setInputFiles([{name:"失败资料.txt",mimeType:"text/plain",buffer:Buffer.from("资料")},{name:"成功资料.txt",mimeType:"text/plain",buffer:Buffer.from("资料")}]);
     await kbot.evaluate(()=>{globalThis.fixtureRejectUpload=true;});
     await kbot.getByRole("button",{name:"提交资料",exact:true}).click();
     await kbot.waitForFunction(()=>document.querySelectorAll(".upload-file-row").length===1);
     assert.ok((await kbot.locator(".upload-file-row").innerText()).includes("失败资料.txt"));
     await kbot.evaluate(()=>{globalThis.fixtureFailNext=true;});
     await kbot.getByRole("button",{name:"提交资料",exact:true}).click();
     await kbot.locator(".knowledge-toast.error").waitFor();
     assert.equal(await kbot.locator(".upload-file-row").count(),1,"上传异常不得清空待重试文件");
     assert.equal(await kbot.locator("#upload-submit").isEnabled(),true);
   }
   if(!empty && name==="reviews"){
     for(const page of pair){await page.setViewportSize({width:1440,height:1000});await page.getByRole("button",{name:"退回",exact:true}).click();const area=page.locator('textarea');await area.fill("请补充产品版本号");await page.getByRole("button",{name:"确认退回",exact:true}).click();await page.waitForFunction(()=>fixtureRequests.some(row=>row.method==="POST"));const request=await page.evaluate(()=>fixtureRequests.find(row=>row.method==="POST"));assert.ok(String(request.body).includes("请补充产品版本号"));}
     await kbot.getByRole("button",{name:"退回",exact:true}).click();
     await kbot.locator("#library-reject-reason").fill("请保留退回原因");
     await kbot.evaluate(()=>{globalThis.fixtureFailNext=true;});
     await kbot.getByRole("button",{name:"确认退回",exact:true}).click();
     await kbot.locator(".knowledge-toast.error").waitFor();
     assert.equal(await kbot.locator("#library-reject").isVisible(),true);
     assert.equal(await kbot.locator("#library-reject-reason").inputValue(),"请保留退回原因");
     await kbot.setViewportSize({width:390,height:1000});
     assert.equal(await kbot.locator("#library-reject").evaluate(node=>node.scrollWidth>node.clientWidth+1),false);
     await kbot.screenshot({path:resolve(artifacts,"kbot-rejection-mobile.png"),fullPage:true});
   }
   if(!empty && ["processing","files"].includes(name)){
     await kbot.evaluate(()=>{globalThis.fixtureTotal=45;});
     await kbot.locator("#library-refresh").click();
     await kbot.locator("#library-next:enabled").waitFor();
     await kbot.locator("#library-next").click();
     await kbot.waitForFunction(()=>document.querySelector("#library-page").textContent==="2");
     await kbot.locator("#library-query").fill("产品 & 手册");await kbot.locator("#library-query").press("Enter");
     await kbot.waitForFunction(()=>document.querySelector("#library-page").textContent==="1");
     const latest=await kbot.evaluate(()=>fixtureRequests.at(-1));
     assert.equal(new URLSearchParams(latest.search).get("q"),"产品 & 手册");
     await kbot.getByRole("button",{name:"查看详情",exact:true}).click();
     await kbot.locator("#library-detail").waitFor();
     const bounds=await kbot.locator("#library-detail").boundingBox();assert.ok(bounds.x>=0);
     assert.ok((await kbot.locator("#library-detail").innerText()).includes("产品手册"));
     await kbot.screenshot({path:resolve(artifacts,"kbot-"+name+"-detail-mobile.png"),fullPage:true});
   }
   if(!empty && ["knowledge-cores","data-sources","data-models","agents"].includes(name)){
     const open={"knowledge-cores":'[data-action="edit"]',"data-sources":'[data-source-id]',"data-models":'[data-open-model]',agents:'[data-agent-action="edit"]'}[name];
     await kbot.locator(open).first().click();
     await kbot.locator("dialog[open]").waitFor();
     assert.equal(await kbot.locator("dialog[open]").evaluate(node=>node.scrollWidth>node.clientWidth+1),false,name+" 抽屉不可横向溢出");
     await kbot.screenshot({path:resolve(artifacts,"kbot-"+name+"-editor-mobile.png"),fullPage:true});
   }
   if(!empty && name==="knowledge"){
     await kbot.evaluate(()=>{
       const clicked=[];const original=HTMLAnchorElement.prototype.click;
       HTMLAnchorElement.prototype.click=function(){clicked.push(this.download);};
       const popup={opener:window,location:{href:"about:blank"},closed:false,close(){this.closed=true;}};
       KBotKnowledgeApi.presentBlob(new Blob(["<script>1<\/script>"],{type:"text/html"}),"资料.html",popup);
       if(popup.location.href!=="about:blank" || !popup.closed || clicked[0]!=="资料.html")throw new Error("可执行原文不能作为同源页面打开");
       HTMLAnchorElement.prototype.click=original;
     });
   }
   for(const page of pair){assert.deepEqual(await page.evaluate(()=>fixtureUnexpected),[],name+" 接口必须全部隔离");await page.close();}
 }
 assert.deepEqual(errors,[],"浏览器运行异常");
 writeFileSync(resolve(artifacts,"differences.json"),JSON.stringify(differences,null,2));
 assert.deepEqual(differences,[],"与真实 Ammolite 页面存在规格差异；详见 "+resolve(artifacts,"differences.json"));
 console.log("通过：9 个真实 Vue 页面与 KBot 的空/有数据、4 种宽度及上传/审核交互。截图："+artifacts);
}finally{await browser?.close();await vite.close();await new Promise(done=>server.close(done));}
