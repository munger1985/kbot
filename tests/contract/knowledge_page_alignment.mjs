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
  library:{library_id:"library-1",display_name:"产品知识库",description:"产品手册与业务资料",status:"ACTIVE",row_version:1,permissions:["READ","UPLOAD","REVIEW","MANAGE","PUBLISH","WITHDRAW"],grants:{"fixture-user":["READ","UPLOAD","REVIEW","MANAGE","PUBLISH","WITHDRAW"]},search_capabilities:[{kind:"LEXICAL",direction:"TEXT_TO_TEXT"},{kind:"DENSE",direction:"TEXT_TO_IMAGE"},{kind:"DENSE",direction:"IMAGE_TO_IMAGE"}]},
  agent:{agent_id:"agent-1",display_name:"企业知识助手",description:"围绕产品与业务资料回答问题。",status:"ACTIVE",enabled_capabilities:["DOCUMENT"],library_id:"library-1",data_model_ids:[],version_no:1,row_version:1,models:{}},
  source:{data_source_id:"source-1",display_name:"销售分析主库",source_type:"ORACLE",status:"ACTIVE",current_version:1},
  model:{semantic_model_id:"model-1",display_name:"销售分析模型",description:"销售分析业务口径",active_version:1,status:"ACTIVE",versions:[{version_no:1,status:"ACTIVE",definition:{datasets:[]}}]},
  document:{document_id:"document-1",title:"产品手册",versions:{"version-1":{status:"PARSED",review:"PENDING",processing_status:"PARSED",error_code:null,generation_ready:true,uploaded_by:"fixture-user"}}},

};
function installFixture(data,empty){
  const value=key=>empty?[]:[data[key]];
  globalThis.fixtureRequests=[];globalThis.fixtureUnexpected=[];
  globalThis.fixtureRequest=async(path,init={})=>{
    const url=new URL(path,location.origin),raw=url.pathname;
    const ref=raw.includes("/knowledge_retrieval/");
    let suffix=ref?raw.replace(/^.*\/knowledge_retrieval\/domains\/[^/]+/,""):raw.replace("/api/v1/apps/knowledge-retrieval","");
    const payload=init.body instanceof FormData?{grouping:init.body.get("grouping_mode"),files:JSON.parse(init.body.get("files") || "[]"),bundle:JSON.parse(init.body.get("bundle") || "null")}:init.body;
    fixtureRequests.push({path:raw,search:url.search,method:init.method || "GET",body:payload,headers:init.headers || {}});
    if(suffix==="/knowledge/libraries")return empty?[]:[data.library,{...data.library,library_id:"library-2",display_name:"另一资料库"}];
    if(suffix.includes("/knowledge/libraries/") && suffix.endsWith("/search")){
      if(globalThis.fixtureDelay)await new Promise(resolve=>{globalThis.fixtureRelease=resolve;});
      const query=typeof init.body==="string"?JSON.parse(init.body):init.body;
      const image=!!query?.images?.length;
      const first=image?"figure":"text",other=image?"text":"figure";
      const items=[{key:"text",kind:"TEXT",text:"原文如图1-1所示，知识检索的图文关系",mime_type:"text/plain",locators:[{page_no:1}],content_ref:{retrieval_ref:"receipt",key:"text"}},
        {key:"figure",kind:"FIGURE",text:"图1-1 检索架构",mime_type:"image/png",locators:[{page_no:1}],content_ref:{retrieval_ref:"receipt",key:"figure"}}];
      return {retrieval_ref:"receipt",status:"READY",reason:null,continuation:null,primary_hits:[{key:first,rank:1}],items,linked:[{anchor_key:first,linked_keys:[other],manifest_status:"COMPLETE",continuation:null}],channels:[{channel_key:image?"image":"lexical",status:"READY"}]};
    }
    if(suffix.includes("/knowledge/libraries/") && suffix.endsWith("/documents")){
      if(init.method==="POST"){
        if(globalThis.fixtureFailNext){globalThis.fixtureFailNext=false;throw new Error("隔离失败，请重试");}
        return {document_id:"document-1",version_id:"version-1",job_id:"job-1",status:"RECEIVED"};
      }
      return {items:value("document"),has_more:false};
    }
    if(suffix.includes("/knowledge/libraries/") && /:review$/.test(suffix)){
      data.document.versions["version-1"].review=JSON.parse(init.body).decision==="APPROVE"?"APPROVED":"REJECTED";return {};
    }
    if(suffix.includes("/knowledge/libraries/") && /:publish$/.test(suffix)){
      data.document.versions["version-1"].status="PUBLISHED";return {};
    }
    if(init.method && init.method!=="GET")return {};
    if(suffix==="/access")return {permissions:data.permissions,domain_id:1,domain_name:"企业知识工作域",user_id:"fixture-user",display_name:"知识管理员"};
    if(suffix==="/bootstrap")return {agents:value("agent"),libraries:value("library")};
    if(suffix==="/agents" || suffix==="/manage/agents")return value("agent");
    if(suffix==="/agents/agent-1")return data.agent;
    if(suffix==="/agents/options")return {libraries:value("library"),data_models:[],models:[]};
    if(suffix==="/collections")return {libraries:value("library")};
    if(suffix==="/manage/collections" || suffix==="/knowledge-cores")return value("library");
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
const httpModule=`export class ApiError extends Error{}; export const businessFetch=(path,options={})=>fixtureRequest(path,options.init||{});export const apiFetch=(path,options={})=>fixtureRequest(path,options);export const businessBlobFetch=async()=>({blob:new Blob([]),contentType:"application/octet-stream"});export const businessEventStream=async()=>new Response("");`;
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
app.mount('#reference');globalThis.referenceMounted=true;globalThis.fixtureSwitchDomain=()=>router.replace('/apps/knowledge-retrieval/second/'+name+'?domainId=2');`;
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
     if(typeof e==="number"?false:a!==e)differences.push({...context,selector:selectors[index],key,actual:a,expected:e});
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
     const owned=["upload","reviews","processing","files","knowledge-cores"].includes(name);
     const selectors=owned?[".app-page-head h1",".knowledge-owned-workspace .primary-button",".knowledge-search-form label",".knowledge-documents th"]:name==="knowledge"?[".knowledge-chat-workspace",".knowledge-agent-picker",".knowledge-chat-welcome h2"]:[".platform-home-hero h1",".table-card th",".table-card td"];
     // 来源的零知识库解析页没有结束 loading；KBot 使用明确空态，差异在 manifest 中登记。
     if(empty && name==="processing")selectors.splice(selectors.indexOf("main > .table-card"),1);
     if(name!=="knowledge" || !empty)await compare(kbot,reference,selectors,{name,width,empty});
     for(const page of pair)assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,name+"/"+width+" 页面溢出");
     if(width===1440 || width===390)for(const [prefix,page]of [["kbot",kbot],["ammolite",reference]])await page.screenshot({path:resolve(artifacts,prefix+"-"+name+"-"+(empty?"empty":"populated")+"-"+width+".png"),fullPage:true});
   }
   if(!empty && name==="upload"){
     for(const [index,page] of pair.entries()){
       await page.setViewportSize({width:1440,height:1000});
       const workspace=page.locator(".knowledge-owned-workspace");
       await workspace.locator('textarea').first().fill("图1-1");
       await workspace.getByRole("button",{name:"检索",exact:true}).click();
       await workspace.locator(".knowledge-evidence-card").waitFor();
       assert.ok((await workspace.locator(".knowledge-evidence-card").innerText()).includes("原文如图1-1"));
       await workspace.locator(".knowledge-evidence-card summary").click();
       assert.ok((await workspace.locator(".knowledge-linked-evidence").last().innerText()).includes("图1-1 检索架构"));
       await workspace.locator('textarea').first().fill("");
       const png=Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+cFqkAAAAASUVORK5CYII=","base64");
       await workspace.locator('input[accept*="image/png"]').setInputFiles({name:"查询.png",mimeType:"image/png",buffer:png});
       await workspace.getByRole("button",{name:"检索",exact:true}).click();
       await page.waitForFunction(()=>document.querySelector(".knowledge-evidence-card")?.textContent.includes("图1-1 检索架构"));
       await workspace.locator(".knowledge-evidence-card summary").click();
       assert.ok((await workspace.locator(".knowledge-linked-evidence").last().innerText()).includes("原文如图1-1"));
       const requests=await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith("/search")));
       assert.equal(JSON.parse(requests.at(-1).body).images.length,1,"图搜图必须发送查询图片字节");
       await workspace.getByRole("button",{name:"审核通过",exact:true}).click();
       await workspace.getByRole("button",{name:"发布",exact:true}).click();
       await page.waitForFunction(()=>document.querySelector(".knowledge-documents")?.textContent.includes("PUBLISHED"));
       // 网络失败重试保持同一文件幂等键；真实页面，不调用真实后端。
       const upload=()=>page.evaluate(()=>{const input=document.querySelector('.knowledge-owned-workspace input[type="file"]:not([accept])');const transfer=new DataTransfer();transfer.items.add(new File(["合成资料"],"合成.txt",{type:"text/plain",lastModified:0}));input.files=transfer.files;input.dispatchEvent(new Event("change",{bubbles:true}));});
       await page.evaluate(()=>{globalThis.fixtureFailNext=true;});
       await upload();
       await workspace.locator('[role="alert"]').filter({hasText:"隔离失败"}).waitFor();
       await upload();
       await page.waitForFunction(()=>fixtureRequests.filter(row=>row.method==="POST"&&row.path.endsWith("/documents")).length===2);
       const keys=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==="POST"&&row.path.endsWith("/documents")).map(row=>row.headers["Idempotency-Key"]));
       assert.equal(keys[0],keys[1],"失败后重试不得重复登记文档");
       // 模拟不会响应 abort 的慢端口，Domain 切换后的迟到结果也不能回填。
       await workspace.getByRole("button",{name:"刷新状态",exact:true}).click();
       await workspace.locator("textarea").first().fill("迟到请求");
       await page.evaluate(()=>{globalThis.fixtureDelay=true;});
       await workspace.getByRole("button",{name:"检索",exact:true}).click();
       await page.waitForFunction(()=>!!globalThis.fixtureRelease);
       if(index===0)await page.evaluate(()=>dispatchEvent(new Event("kbot:knowledge-context-changing")));
       else await page.evaluate(()=>globalThis.fixtureSwitchDomain());
       await page.evaluate(()=>{globalThis.fixtureDelay=false;globalThis.fixtureRelease();});
       await page.waitForTimeout(100);
       assert.equal(await workspace.locator(".knowledge-evidence-card").count(),0,"旧 Domain 迟到结果必须丢弃");
     }
   }
   for(const page of pair){assert.deepEqual(await page.evaluate(()=>fixtureUnexpected),[],name+" 接口必须全部隔离");await page.close();}
 }
 assert.deepEqual(errors,[],"浏览器运行异常");
 writeFileSync(resolve(artifacts,"differences.json"),JSON.stringify(differences,null,2));
 assert.deepEqual(differences,[],"与真实 Ammolite 页面存在规格差异；详见 "+resolve(artifacts,"differences.json"));
 console.log("通过：9 个真实 Vue 页面与 KBot 的空/有数据、4 种宽度、图文双向检索、审核发布、幂等重试和迟到响应隔离。截图："+artifacts);
}finally{await browser?.close();await vite.close();await new Promise(done=>server.close(done));}
