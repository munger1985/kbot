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
const permissions=["use","knowledge_chat","upload","review","knowledge_manage","knowledge_core_manage","data_model_manage","agent_manage","x_search","domain_manage"].map(key=>"knowledge_retrieval:"+key);
const fixture={
  permissions,
  library:{library_id:"library-1",display_name:"产品知识库",description:"产品手册与业务资料",status:"ACTIVE",row_version:1,updated_at:"2026-11-20T08:00:00Z",upload_max_bytes:67108864,upload_formats:{"text/plain":"txt","application/pdf":"pdf"},ocr_enabled:false,permissions:["READ","UPLOAD","REVIEW","MANAGE","PUBLISH","WITHDRAW"],grants:{"fixture-user":["READ","UPLOAD","REVIEW","MANAGE","PUBLISH","WITHDRAW"]},search_capabilities:[{kind:"LEXICAL",direction:"TEXT_TO_TEXT"},{kind:"NEAR_DUPLICATE",direction:"IMAGE_TO_IMAGE"}]},
  agent:{agent_id:"agent-1",display_name:"企业知识助手",description:"围绕产品与业务资料回答问题。",status:"ACTIVE",enabled_capabilities:["DOCUMENT"],library_id:"library-1",data_model_ids:[],version_no:1,row_version:1,models:{}},
  source:{data_source_id:"source-1",display_name:"销售分析主库",source_type:"ORACLE",status:"ACTIVE",current_version:1},
  model:{semantic_model_id:"model-1",display_name:"销售分析模型",description:"销售分析业务口径",active_version:1,status:"ACTIVE",versions:[{version_no:1,status:"ACTIVE",definition:{datasets:[]}}]},
  document:{document_id:"document-1",title:"产品手册",versions:{"version-1":{status:"RECEIVED",review:"PENDING",parse_status:"AWAITING_APPROVAL",processing_status:"AWAITING_APPROVAL",error_code:null,generation_ready:false,generation_id:null,created_at:"2026-11-20T08:00:00Z",job_id:"job-1",mime_type:"text/plain",byte_count:12,allowed_actions:["review"],parse_strategy_options:[
    {strategy:"TEXT",selectable:true,unavailable_roles:[],executor_unavailable:false},
    {strategy:"AUTO",selectable:false,unavailable_roles:["ocr"],executor_unavailable:false},
    {strategy:"VISUAL",selectable:false,unavailable_roles:["ocr","vlm"],executor_unavailable:false}
  ]}}},

};
function installFixture(data,empty,pdfBase64){
  const value=key=>empty?[]:[data[key]];
  const view=new URLSearchParams(location.search).get("page") || location.pathname.split("/").at(-1).replace(/\.html$/,"");
  if(view==="search")Object.defineProperty(navigator,"pdfViewerEnabled",{value:true,configurable:true});
  const queueView=view==="processing";
  const chatSession={conversation_id:"session-1",agent_id:data.agent.agent_id,title:"解释一下职业道德",last_active_at:"2026-11-21T08:00:00Z",row_version:4};
  let chatSessions=empty?[]:[chatSession];
  let governanceItems;
  let rebuiltStatus="FAILED";
  const fileBatches=new Map();
  const fileProgress=id=>globalThis.fixtureFileIndexBatches?.[id] ?? 0;
  const fileTitle=id=>id==="version-2"?"安装说明":data.document.title;
  const governanceVersion=id=>governanceItems?.find(item=>Object.hasOwn(item.versions,id))?.versions[id];
  const excerpt="原句：图1-1 用于说明本节检索架构。";
  const chatLocators=[{page_no:1,bbox:[0.07,0.72,0.92,0.80]}];
  const composition={status:"READY",model_delivery:"SENT",delivered_count:1,coverage_status:"COMPLETE",
    primary_hits:[{key:"a".repeat(64),rank:1}],linked:[],
    items:[{key:"a".repeat(64),kind:"TEXT",mime_type:"text/plain",locators:chatLocators}]};
  globalThis.fixtureRequests=[];globalThis.fixtureUnexpected=[];globalThis.fixturePendingSearch=[];
  globalThis.fixtureRequest=async(path,init={})=>{
    const url=new URL(path,location.origin),raw=url.pathname;
    const ref=raw.includes("/knowledge_retrieval/");
    let suffix=ref?raw.replace(/^.*\/knowledge_retrieval\/domains\/[^/]+/,""):raw.replace("/api/v1/apps/knowledge-retrieval","");
    const payload=init.body instanceof FormData?{grouping:init.body.get("grouping_mode"),files:JSON.parse(init.body.get("files") || "[]"),bundle:JSON.parse(init.body.get("bundle") || "null")}:init.body;
    fixtureRequests.push({path:raw,search:url.search,method:init.method || "GET",body:payload,headers:Object.fromEntries(new Headers(init.headers))});
    if(view==="knowledge" && suffix==="/conversations")return chatSessions;
    if(view==="knowledge" && suffix==="/conversations/session-1"){
      if(init.method==="DELETE"){chatSessions=[];return null;}
      return chatSession;
    }
    if(view==="knowledge" && suffix==="/conversations/session-1/turns"){
      if(init.method==="POST")return {turn_id:"turn-2",run_id:"run-2",event_cursor:0,turn_status:"RUNNING"};
      const answer={created_at:"2026-11-22T15:48:30Z",content:{
        text:"职业道德是从业者在工作中应当遵守的行为规范。[D1]",
        references:[{citation_label:"D1",reference_type:"APP_KNOWLEDGE",title:"审计制度.pdf"}],
        knowledge_composition:composition,
      }};
      return {turns:[{turn_id:"turn-1",run_id:"run-1",status:"COMPLETED",
        user_item:{created_at:"2026-11-22T15:48:01Z",content:{text:"什么是职业道德？"}},assistant_item:answer}]};
    }
    if(view==="knowledge" && suffix==="/runs/run-2")return {status:"COMPLETED",completed_at:"2026-11-22T16:01:27Z"};
    if(view==="knowledge" && suffix==="/runs/run-1/references/D1/preview"){
      if(globalThis.fixtureCitationServiceFailure)
        throw Object.assign(new Error("引用服务暂不可用"),{status:502,code:"AGENT_EVIDENCE_SERVICE_AUTH_FAILED"});
      const content_url=raw.replace(/preview$/,"content");
      return {citation_label:"D1",title:"审计制度.pdf",mime_type:"text/plain",preview_type:"TEXT",
        locators:globalThis.fixtureCitationWithoutBox?[{page_no:1}]:chatLocators,content_url,pdf_url:content_url+"?source_pdf=true"};
    }
    if(suffix==="/knowledge/libraries")return empty?[]:[data.library,{...data.library,library_id:"library-2",
      ...(url.searchParams.get("management")==="true" ? {display_name:"面向生产系统的资料知识库".repeat(12),description:"说明内容应截断为两行，不能撑大卡片。".repeat(16),status:"DISABLED"} : {
        display_name:queueView ? "很长的知识库名称".repeat(12) : /\/files(?:\.html)?$/.test(location.pathname) ? "另一资料库 · 一个用于验证换行的较长知识库名称" : "另一资料库",
        description:null,permissions:view==="processing" ? ["UPLOAD"] : ["READ"],
      })},...(queueView ? [{...data.library,library_id:"library-read",display_name:"仅有读取权限",permissions:["READ"]}] : [])];
    if(suffix.endsWith("/member-options"))return [{user_id:"fixture-user",display_name:"知识管理员",can_manage:true},{user_id:"new-member",display_name:"新成员",can_manage:false}];
    if(suffix==="/knowledge/libraries/library-1"){
      if(init.method==="PATCH"){
        const changes=JSON.parse(init.body);
        if(globalThis.fixtureIndexEdit){
          const {expected_revision,rebuild_indexes,...fields}=changes;
          if(expected_revision!==data.library.row_version)throw new Error("修订冲突");
          Object.assign(data.library,fields,{row_version:expected_revision+1});
          return data.library;
        }
        return {...data.library,...changes,row_version:2};
      }
      if(init.method==="DELETE")return {status:"DELETED"};
      return {...data.library,delete_block:"AGENT_REFERENCE",
        index_maintenance:{published_file_count:3,library_revision:data.library.row_version,pending_roles:[],active_file_rebuild_count:0,batch:null}};
    }
    if(suffix.endsWith("/library-1/index-rebuild")){
      const request=init.method==="POST"?JSON.parse(init.body):{};
      if(globalThis.fixtureMaintenance==="success"){
        const id=request.version_id ?? url.searchParams.get("version_id");
        if(request.action==="START"){
          if(globalThis.fixtureFailNext){globalThis.fixtureFailNext=false;throw new Error("重建任务未接收，请重试");}
          fileBatches.set(id,`batch-${id}`);
        }
        const progress=fileProgress(id);
        return {published_file_count:0,library_revision:1,pending_roles:[],
          active_file_rebuild_count:[...fileBatches.keys()].filter(key=>fileProgress(key)<10).length,
          batch:fileBatches.has(id)?{
          batch_id:fileBatches.get(id),status:progress===10?"PUBLISHED":"RUNNING",scope:"FILE",source_published:false,roles:[],
          items:[{version_id:id,title:fileTitle(id),status:progress===10?"READY":progress>0?"RUNNING":"PENDING",
            error_code:null,visual_not_applicable:false,index_progress:{committed_batches:progress,total_batches:10,validated:progress===10}}],
        }:null};
      }
      if(!globalThis.fixtureLibraryBatch)return {published_file_count:0,library_revision:1,pending_roles:[],active_file_rebuild_count:0,batch:null};
      if(init.method==="POST"){
        const {action}=request;
        if(action==="RETRY")rebuiltStatus="PUBLISHED";
      }
      return {published_file_count:1,library_revision:1,pending_roles:rebuiltStatus==="PUBLISHED"?[]:["embedding"],active_file_rebuild_count:0,
        batch:{batch_id:"batch-1",status:rebuiltStatus,scope:"LIBRARY",roles:["embedding"],
          items:[{version_id:"version-1",title:"产品手册",status:rebuiltStatus==="FAILED"?"FAILED":"READY",
            error_code:rebuiltStatus==="FAILED"?"TIMEOUT":null,visual_not_applicable:false,
            index_progress:{committed_batches:rebuiltStatus==="FAILED"?4:10,total_batches:10,validated:rebuiltStatus!=="FAILED"}}]}};
    }
    if(suffix.includes("/knowledge/libraries/") && suffix.endsWith("/search")){
      const query=typeof init.body==="string"?JSON.parse(init.body):init.body;
      // 即使浏览器已中止，也让隔离接口继续返回，以验证迟到响应隔离。
      const outcome=globalThis.fixtureDelay ? await new Promise(resolve=>{
        fixturePendingSearch.push({query,signal:init.signal,resolve});
      }) : "success";
      if(outcome==="failure")throw new Error("隔离检索失败，请重试");
      if(outcome==="empty")return {retrieval_ref:"empty",status:"COMPLETE",mode:query.mode,reason:null,
        scope_exhausted:true,continuation:null,primary_hits:[],items:[],linked:[],channels:[]};
      const image=!!query?.images?.length;
      const first=image?"figure":"text",other=image?"text":"figure";
      const text={key:"text",kind:"TEXT",text:"原文如图1-1所示，知识检索的图文关系",source_title:"产品手册",document_id:"doc-1",version_id:"version-1",mime_type:"text/plain",chapter:"第一章 检索",passage_key:"paragraph-1",locators:[{page_no:1}],content_ref:{retrieval_ref:"receipt",key:"text"},
        passage_display:{order:1,kind:"PARAGRAPH",title:"第九条",text:"第九条 内部审计人员应熟悉公司的经营业务和内部控制规范。",pages:[1],chapter_key:"chapter-1"}};
      const items=[text,{...text,key:"duplicate",text:"同一段落的重叠片段",content_ref:{retrieval_ref:"receipt",key:"duplicate"}},
        {...text,key:"another",source_title:"使用说明",document_id:"doc-2",version_id:"version-2",passage_key:"paragraph-2",chapter:null,passage_display:null,content_ref:{retrieval_ref:"receipt",key:"another"}},
        {key:"figure",kind:"FIGURE",text:"图1-1 检索架构",source_title:"产品手册",document_id:"doc-1",version_id:"version-1",mime_type:"image/png",passage_key:null,chapter:null,locators:[{page_no:1}],content_ref:{retrieval_ref:"receipt",key:"figure"}}];
      return {retrieval_ref:"receipt",status:"COMPLETE",mode:query.mode,reason:null,scope_exhausted:true,continuation:null,
        primary_hits:(image?[first]:[first,"duplicate","another"]).map((key,i)=>({key,rank:i+1})),
        items,linked:[{anchor_key:first,linked_keys:[other],manifest_status:"PARTIAL",continuation:"next-related"}],channels:[]};
    }
    if(suffix.endsWith("/retrievals/receipt/context/text")){
      const before=Number(url.searchParams.get("before")), after=Number(url.searchParams.get("after"));
      const block=(key,text)=>({key,kind:"PARAGRAPH",anchor:false,locators:[{page_no:1}],segments:[{text,matched:false}]});
      return {blocks:[...(before>2?[block("earliest","展开后显示的更早一段")]:[]),block("preceding","原文上一段"),
        {...block("hit",""),anchor:true,locators:[{page_no:1,bbox:globalThis.fixtureAmbiguousPdfHit ? [0.07,0.08,0.92,0.82] : [0.07,0.72,0.92,0.80]}],
          segments:[{text:"原句：",matched:false},{text:"图1-1",matched:true},{text:" 用于说明本节检索架构。",matched:false}]},
        block("following","原文下一段"),...(after>2?[block("later","展开后显示的更后一段")]:[])],
        has_before:before===2,has_after:after===2,pdf_page:1};
    }
    if(suffix.endsWith("/relations:continue")){
      const body=JSON.parse(init.body);
      return {retrieval_ref:"relation-receipt",linked:[{anchor_key:body.anchor_key,linked_keys:["more"],continuation:null}],
        items:[{key:"more",kind:"TEXT",text:"关联续页",source_title:"产品手册",version_id:"version-1",locators:[],content_ref:{retrieval_ref:"relation-receipt",key:"more"}}]};
    }
    if(suffix.includes("/knowledge/libraries/") && suffix.endsWith("/documents")){
      if(init.method==="POST"){
        if(globalThis.fixtureFailNext){globalThis.fixtureFailNext=false;throw new Error("隔离失败，请重试");}
        return {document_id:"document-1",version_id:"version-1",job_id:"job-1",name:"合成.txt",status:"RECEIVED"};
      }
      const state=url.searchParams.get("state"), version=data.document.versions["version-1"];
      if(queueView && globalThis.fixtureProgress && !suffix.includes("/library-2/")){
        const [stage, batches, total, validated]=globalThis.fixtureProgress;
        return {items:[{...data.document,versions:{"version-1":{
          ...version,review:"APPROVED",status:"PARSED",parse_status:"PARSED",
          processing_status:validated?"READY":"PARSED",progress_stage:stage,
          index_progress:{committed_batches:batches,total_batches:total,validated},
          allowed_actions:[],generation_ready:validated,
        }}}],has_more:false};
      }
      if(queueView && globalThis.fixtureMaintenance && !suffix.includes("/library-2/")){
        const mode=globalThis.fixtureMaintenance, parsed=mode!=="parse", ready=mode==="success";
        const exact=url.searchParams.get("version_id");
        if(ready && !exact && globalThis.fixtureDelayRebuildList){
          globalThis.fixtureDelayRebuildList=false;
          await new Promise(resolve=>{globalThis.fixtureRebuildListRelease=resolve;});
        }
        const items=(ready?["version-1","version-2"]:["version-1"]).filter(id=>!exact || exact===id).map(id=>{
          const progress=fileProgress(id), rebuilding=ready && fileBatches.has(id), finished=rebuilding && progress===10;
          const reparsing=ready && globalThis.fixtureReparsing===id;
          return {...data.document,title:fileTitle(id),document_id:id==="version-1"?data.document.document_id:"document-2",versions:{[id]:{
          ...version,review:"APPROVED",status:ready?"PUBLISHED":parsed?"PARSED":"FAILED",
          parse_status:reparsing?"RECEIVED":parsed?"PARSED":"FAILED",processing_status:reparsing?"RECEIVED":ready?"READY":parsed?"INDEX_FAILED":"FAILED",
          progress_stage:reparsing?"PARSE_QUEUED":rebuilding?(finished?"REBUILD_COMPLETE":progress?"REBUILDING":"REBUILD_QUEUED"):ready?"READY":parsed?"INDEX_FAILED":"PARSE_FAILED",
          index_progress:parsed && !reparsing?{committed_batches:rebuilding?progress:ready?10:4,total_batches:10,validated:ready && (!rebuilding || finished)}:null,
          generation_ready:ready && !reparsing,generation_id:ready && !reparsing?id.replace("version","generation"):null,error_code:ready?null:parsed?"TIMEOUT":"INVALID_CONTRACT",
          original_parse_available:!parsed || ready,allowed_actions:ready?(reparsing || (rebuilding && !finished)?[]:["rebuild-index","reprocess"]):[parsed?"retry-index":"reprocess"],
          parse_task_history:[{job_id:"prior-job",status:"FAILED",error_code:"INVALID_CONTRACT",created_at:version.created_at}],
        }}};
        });
        return {items,has_more:false};
      }
      if(globalThis.fixtureGovernanceScenario && !suffix.includes("/library-2/")){
        governanceItems??=[
          {...structuredClone(data.document),versions:{
            "version-1":structuredClone(version),
            "version-old":{...version,status:"PUBLISHED",review:"APPROVED",allowed_actions:["withdraw"]},
          }},
          ...[["second","附件二","AUTO"],["third","附件三","VISUAL"]].map(([key,title,extra])=>({
            document_id:"document-"+key,title,versions:{["version-"+key]:{
              ...structuredClone(version),parse_strategy_options:version.parse_strategy_options.map(option=>({...option,selectable:option.strategy==="TEXT" || option.strategy===extra})),
            }},
          })),
        ];
        const exact=url.searchParams.get("version_id"), doc=url.searchParams.get("document_id"), mode=url.searchParams.get("view");
        const matches=row=>!state || state==="PENDING" && row.allowed_actions.includes("review") ||
          state==="PUBLISHED" && row.status==="PUBLISHED";
        const items=governanceItems.filter(item=>!doc || item.document_id===doc).map(item=>({...item,versions:Object.fromEntries(
          Object.entries(item.versions).filter(([id,row])=>(!exact || exact===id) && matches(row))
            .map(([id,row])=>[id,mode==="reviews"?row:{...row,parse_strategy_options:undefined}]),
        )})).filter(item=>Object.keys(item.versions).length);
        return {items,has_more:false};
      }
      const matched=!state || (state==="PENDING" && version.review==="PENDING") || (state==="PUBLISHED" && version.status==="PUBLISHED");
      const versions=url.searchParams.get("view")==="files" && !url.searchParams.get("version_id") ? {
        ...data.document.versions,
        "version-office":{...version,mime_type:"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        "version-pdf":{...version,mime_type:"application/pdf"},
        "version-png":{...version,mime_type:"image/png"},
      } : data.document.versions;
      return {items:matched && !empty && !suffix.includes("/library-2/") ? [{...data.document,versions}] : [],has_more:false};
    }
    if(suffix.includes("/knowledge/libraries/") && /:review$/.test(suffix)){
      const id=suffix.split("/").at(-1).split(":")[0];
      if(id===globalThis.fixtureFailReviewId){globalThis.fixtureFailReviewId="";throw new Error("状态冲突");}
      const version=governanceVersion(id) || data.document.versions["version-1"], body=JSON.parse(init.body);
      if(body.decision==="APPROVE"){
        if(body.parse_strategy!=="TEXT")throw new Error("必须明确选择原生解析");
        Object.assign(version,{review:"APPROVED",status:"PUBLISHED",parse_status:"PARSED",processing_status:"READY",generation_ready:true,generation_id:"generation-1",allowed_actions:["withdraw"]});
      }else Object.assign(version,{review:"REJECTED",parse_status:"REJECTED",processing_status:"REJECTED",allowed_actions:[]});
      return {};
    }
    if(suffix.endsWith(":reprocess") && init.method==="POST" && globalThis.fixtureMaintenance==="success"){
      globalThis.fixtureReparsing=suffix.split("/").at(-1).split(":")[0];
      return {job_id:"replacement-job",status:"RECEIVED"};
    }
    if(init.method && init.method!=="GET")return {};
    if(suffix==="/access")return {permissions:data.permissions,domain_id:1,domain_name:"企业知识工作域",user_id:"fixture-user",display_name:"知识管理员"};
    if(suffix==="/bootstrap")return {agents:value("agent"),libraries:value("library")};
    if(suffix==="/agents" || suffix==="/manage/agents")return value("agent");
    if(suffix==="/agents/agent-1")return data.agent;
    if(suffix==="/agents/options")return {libraries:value("library"),data_models:[],models:[]};
    if(suffix==="/collections")return {libraries:value("library")};
    if(suffix==="/manage/collections" || suffix==="/knowledge-cores")return value("library");
    if(suffix.endsWith("/model-catalog"))return globalThis.fixtureIndexEdit?[
      {model_id:"dense-1",category:2,display_name:"文本模型 A"},{model_id:"dense-2",category:2,display_name:"文本模型 B"},
    ]:[];
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
  globalThis.fixtureEventStream=async(path)=>{
    if(view!=="knowledge" || !path.endsWith("/runs/run-2/events"))throw new Error("未隔离的事件流 "+path);
    const encoder=new TextEncoder();
    const frame=(type,created_at,payload)=>encoder.encode("event: "+type+"\ndata: "+JSON.stringify({event_type:type,created_at,payload})+"\n\n");
    return new Response(new ReadableStream({start(controller){
      controller.enqueue(frame("answer.delta","2026-11-22T16:01:25Z",{delta:"这是一条实时回答。[D1]"}));
      globalThis.fixtureFinishReply=()=>{
        controller.enqueue(frame("answer.completed","2026-11-22T16:01:26Z",{references:[
          {citation_label:"D1",reference_type:"APP_KNOWLEDGE",title:"实时引用.pdf"},
        ],knowledge_composition:composition}));
        controller.close();
      };
    }}),{headers:{"content-type":"text/event-stream"}});
  };
  globalThis.fixtureRequestBlob=async(url,init={})=>{
    const target=new URL(url,location.origin);
    fixtureRequests.push({path:target.pathname,search:target.search,method:init.method || "GET"});
    if(/\/versions\/(version-1|version-second)\/source$/.test(target.pathname) && target.searchParams.get("preview")==="true")
      return {blob:new Blob(["合成原文：<img src=x onerror=alert(1)>"],{type:"text/plain"}),contentType:"text/plain"};
    if(view==="knowledge" && target.pathname.includes("/runs/run-1/")){
      if(target.searchParams.get("source_pdf")!=="true")
        return {blob:new Blob([excerpt],{type:"text/plain"}),contentType:"text/plain"};
      if(globalThis.fixtureCitationPdfDenied)throw new Error("引用原文不可读取");
      const binary=atob(pdfBase64);
      return {blob:new Blob([Uint8Array.from(binary,char=>char.charCodeAt(0))],{type:"application/pdf"}),contentType:"application/pdf"};
    }
    if((target.pathname.endsWith("/versions/version-pdf/source") && target.searchParams.get("preview")==="true") ||
      target.pathname.endsWith("/retrievals/receipt/context/text/pdf")){
      const binary=globalThis.fixtureInvalidPdf?"<html>invalid</html>":atob(pdfBase64);
      return {blob:new Blob([Uint8Array.from(binary,char=>char.charCodeAt(0))],{type:"application/pdf"}),contentType:"application/pdf"};
    }
    if(target.pathname.endsWith("/versions/version-png/source") && target.searchParams.get("preview")==="true"){
      const binary=atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+cFqkAAAAASUVORK5CYII=");
      return {blob:new Blob([Uint8Array.from(binary,char=>char.charCodeAt(0))],{type:"image/png"}),contentType:"image/png"};
    }
    throw new Error("未隔离的原件读取");
  };
  globalThis.fetch=async(input,init={})=>{
    const url=typeof input==="string"?input:input.url;
    if(!new URL(url,location.origin).pathname.startsWith("/api/"))return fetchOriginal(input,init);
    try{
      if(/\/events$/.test(new URL(url,location.origin).pathname))return await fixtureEventStream(url);
      if(/\/(?:source|pdf)$/.test(new URL(url,location.origin).pathname) ||
        (view==="knowledge" && /\/runs\/run-1\/(?:references\/D1\/content|knowledge\/content\/a{64})$/.test(new URL(url,location.origin).pathname))) {
        const {blob,contentType}=await fixtureRequestBlob(url,init);
        return new Response(blob,{headers:{"content-type":contentType}});
      }
      return new Response(JSON.stringify(await fixtureRequest(url,init)),{headers:{"content-type":"application/json"}});
    }
    catch(error){return new Response(JSON.stringify({detail:error.message,code:error.code}),{status:error.status || 503,headers:{"content-type":"application/json"}});}
  };
  sessionStorage.setItem("kbot.knowledge.session.v1",JSON.stringify({access_token:"fixture-session",domain_id:1,domain_name:"企业知识工作域",display_name:"知识管理员",user_id:"fixture-user"}));
}
const httpModule=`export class ApiError extends Error{}; export const businessFetch=(path,options={})=>fixtureRequest(path,options.init||{});export const apiFetch=(path,options={})=>fixtureRequest(path,options);export const businessBlobFetch=(path,options={})=>fixtureRequestBlob(path,options.init||{});export const businessEventStream=(path)=>fixtureEventStream(path);`;
const entry=`
import {createApp,computed,h} from 'vue';
import {createRouter,createWebHistory,RouterView} from 'vue-router';
import {businessAppPermissionsKey,businessAppPrincipalKey} from '/src/app/businessAppContext.ts';
import ConfirmDialog from '/src/shared/ui/ConfirmDialog.vue';
import '/src/shared/ui/styles.css';
const name=new URLSearchParams(location.search).get('page'),empty=new URLSearchParams(location.search).get('empty')==='1';
const pages={${pages.map(key=>JSON.stringify(key)+":()=>import('/src/features/knowledge-retrieval/"+manifest.pages[key].sources[0]+"')").join(",")}};
const component=(await pages[name]()).default;
const router=createRouter({history:createWebHistory(),routes:[{path:'/apps/knowledge-retrieval/:domainCode/:pathMatch(.*)*',component,meta:{title:${JSON.stringify(Object.fromEntries(pages.map(key=>[key,manifest.pages[key].title])))}[name]}}]});
const app=createApp({render:()=>[h(RouterView),h(ConfirmDialog)]});app.use(router);
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
 const selectedPages=process.env.KNOWLEDGE_UI_PAGES?.split(",") || pages;
 let pdfBase64="";
 if(selectedPages.some(name=>["files","search","knowledge"].includes(name))){
   const pdfPage=await browser.newPage();
   await pdfPage.setContent(`<meta charset="utf-8"><style>@page{size:600px 800px;margin:0}
     body{margin:0;font:20px "WenQuanYi Micro Hei",sans-serif}section{position:relative;height:790px}
     p{position:absolute;margin:0;left:48px} section + section{break-before:page}</style>
     <section><p style="top:90px">原句：图1-1 用于说明本节检索架构。</p>
       <p style="top:595px">原句：图1-1 用于说明本节检索架构。</p></section>
     <section><svg width="300" height="120"><rect width="300" height="120" fill="#ddd"/></svg></section>`);
   pdfBase64=(await pdfPage.pdf({preferCSSPageSize:true})).toString("base64");
   await pdfPage.close();
 }
 for(const empty of [false,true])for(const name of selectedPages){
   console.log("对照："+name+(empty?"/空数据":"/有数据"));
   const pair=await Promise.all([0,1].map(async()=>{
     const page=await browser.newPage({viewport:{width:1440,height:1000},timezoneId:name==="knowledge"?"Asia/Shanghai":undefined});
     // 历史消息与当前时间同属 UTC 的一天，但在上海时区已跨过午夜。
     if(name==="knowledge")await page.clock.setFixedTime("2026-11-22T16:01:00Z");
     page.on("pageerror",error=>errors.push(name+":"+error.message));
     await page.route(/^https?:\/\//,route=>new URL(route.request().url()).hostname === "127.0.0.1" ? route.continue() : route.abort());
     await page.addInitScript(({data,empty,install,pdfBase64})=>{(0,eval)("("+install+")")(data,empty,pdfBase64);},{data:fixture,empty,install:installFixture.toString(),pdfBase64});
     return page;
   }));
   const [kbot,reference]=pair;
   await Promise.all([kbot.goto(kbotOrigin+"/ui/knowledge_retrieval/"+name+".html"),reference.goto(refOrigin+"/reference.html?page="+name+(empty?"&empty=1":""))]);
   await kbot.waitForFunction(()=>document.body.dataset.access==="ready");
   await reference.waitForFunction(()=>globalThis.referenceMounted);
   await Promise.all(pair.map(page=>page.waitForLoadState("networkidle")));
   for(const page of pair)await page.addStyleTag({content:"*,*::before,*::after{transition:none!important;animation:none!important}"});
   for(const width of name==="knowledge-cores" ? [2040,1440,1024,390,320] : [1440,1024,390,320]){
     await Promise.all(pair.map(page=>page.setViewportSize({width,height:1000})));
     const owned=["search","processing","files","knowledge-cores"].includes(name);
     const selectors=name==="knowledge"?[".knowledge-chat-workspace",".knowledge-agent-picker",".knowledge-chat-welcome h2"]:[".platform-home-hero h1",".table-card th",".table-card td"];
     if(!owned && (name!=="knowledge" || !empty))await compare(kbot,reference,selectors,{name,width,empty});
     if(owned)for(const page of pair){
       const expected={search:".knowledge-search-formbar",processing:".knowledge-directory-layout",files:".knowledge-directory-layout","knowledge-cores":".management-catalog-grid"};
       assert.equal(await page.locator(expected[name]).count(),1,`${name} 缺少定稿布局`);
       if(name!=="search")assert.equal(await page.locator(".knowledge-search-formbar").count(),0);
       assert.equal(await page.locator("main .knowledge-steps").count(),0);
       if(name!=="knowledge-cores")assert.equal(await page.locator(".knowledge-management-page").count(),0);
       if(["files","processing"].includes(name)){
         const title={files:"文件列表",processing:"解析任务列表"}[name];
         const filters=page.getByRole("search",{name:"筛选"+title});
         const disabled=name==="files" || empty;
         const picker=page.getByRole("complementary",{name:"选择知识库"});
         const list=page.locator(".knowledge-directory-table");
         assert.equal(await list.count(),1);
         assert.equal(await filters.getByRole("combobox",{name:"知识库"}).count(),0);
         assert.equal(await picker.getByRole("button").count(),empty?0:2,"仅展示当前页面有权处理的知识库");
         assert.equal(await picker.getByRole("button",{pressed:true}).count(),disabled?0:1);
         assert.equal(await page.getByRole("button",{name:"＋ 上传资料",exact:true}).count(),name==="files"?1:0);
         const [rail, current]=await Promise.all([picker.boundingBox(),page.locator(".knowledge-current-library").boundingBox()]);
         assert.ok(width>700?rail.x+rail.width<current.x:rail.y+rail.height<current.y,`${name} 选库区应在宽屏左侧、窄屏上方`);
         assert.equal(await filters.getByRole("searchbox",{name:"按文件标题查找"}).isDisabled(),disabled);
         assert.equal(await filters.getByRole("button",{name:"筛选",exact:true}).isDisabled(),disabled);
         assert.equal(await filters.getByRole("combobox",{name:name==="files"?"发布状态":"处理进度"}).isDisabled(),disabled);
         const geometry=await filters.evaluate(form=>{
           const selectors=[".search-field","select","button",".knowledge-directory-mine"];
           const [search,...actions]=selectors.flatMap(selector=>[...form.querySelectorAll(selector)].map(node=>node.getBoundingClientRect()));
           const bounds=form.getBoundingClientRect();
           const centerY=rect=>(rect.top+rect.bottom)/2;
           return {gap:parseFloat(getComputedStyle(form).gap), alignment:Math.max(...actions.map(rect=>Math.abs(centerY(search)-centerY(rect)))),
             contained:[search,...actions].every(rect=>rect.left>=bounds.left-1 && rect.right<=bounds.right+1),
             formWidth:bounds.width, controlWidths:[search,...actions].map(rect=>rect.width)};
         });
         assert.ok(geometry.gap>=4 && geometry.gap<=12 && geometry.contained,JSON.stringify({name,width,...geometry}));
         if(width>=1440 || width>=1024 && name==="files")assert.ok(geometry.alignment<=2,JSON.stringify({name,width,platform:page===kbot?"kbot":"ammolite",...geometry}));
       }
       if(["files","processing"].includes(name) && !empty){
         const cards=await page.locator(".knowledge-library-pick").evaluateAll(nodes=>nodes.map(node=>{
           const box=node.getBoundingClientRect();
           const title=node.querySelector("strong").getBoundingClientRect();
           const badge=node.querySelector(".status-pill")?.getBoundingClientRect();
           const description=node.querySelector(".knowledge-library-pick-description");
           const detail=description?.getBoundingClientRect();
           const centerY=rect=>(rect.top+rect.bottom)/2;
           const contentBottom=Math.max(title.bottom,badge?.bottom ?? 0);
           return {height:box.height, topInset:Math.min(title.top,badge?.top ?? title.top)-box.top,
             bottomInset:box.bottom-(detail?.bottom ?? contentBottom),
             aligned:!badge || Math.abs(centerY(title)-centerY(badge))<1,
             inside:title.left>box.left && title.right<box.right && (!badge || title.right<=badge.left && badge.right<box.right),
             detailBelow:!detail || detail.top>=contentBottom,
             hasDescription:!!detail, stretched:Math.abs(box.width-node.parentElement.clientWidth)<32 && node.scrollWidth<=node.clientWidth};
         }));
         assert.equal(cards.length,2);
         assert.deepEqual(cards.map(card=>card.hasDescription),[true,false]);
         for(const card of cards){
           assert.ok(card.height>=44 && card.topInset>=8 && card.bottomInset>=8,JSON.stringify(card));
           assert.ok(card.aligned && card.inside && card.detailBelow && card.stretched,JSON.stringify(card));
         }
       }
       if(name==="knowledge-cores"){
         const layout=await page.locator(".management-catalog-grid").evaluate(grid=>{
           const boxes=[...grid.children].map(card=>{
             const box=card.getBoundingClientRect(), footer=card.querySelector("footer")?.getBoundingClientRect();
             const body=card.querySelector(".aiops-catalog-card-body")?.getBoundingClientRect();
             return {width:box.width,height:box.height,padding:getComputedStyle(card).padding,
               contentFits:!footer || footer.bottom<=box.bottom && body.bottom<=footer.top+1};
           });
           return {width:grid.getBoundingClientRect().width,gap:getComputedStyle(grid).gap,boxes};
         });
         assert.equal(layout.boxes.length,empty?1:3);
         assert.equal(layout.gap,"14px","沿用 AIOps 卡片间距");
         for(const tile of layout.boxes){
           assert.ok(Math.abs(tile.width-Math.min(312,layout.width))<=1,`卡片宽度偏离 AIOps 规格：${JSON.stringify(tile)}`);
           assert.ok(Math.abs(tile.height-tile.width*6/5)<=1,`卡片高度偏离 5:6：${JSON.stringify(tile)}`);
           assert.equal(tile.padding,"22px");
           assert.ok(tile.contentFits,"卡片内容不得挤出底部操作栏");
         }
       }
     }
     for(const page of pair)assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,name+"/"+width+" 页面溢出");
     if(width===1440 || width===390)for(const [prefix,page]of [["kbot",kbot],["ammolite",reference]])await page.screenshot({path:resolve(artifacts,prefix+"-"+name+"-"+(empty?"empty":"populated")+"-"+width+".png"),fullPage:true});
   }
   if(!empty && ["knowledge","search","files","processing","knowledge-cores"].includes(name)){
     for(const [index,page] of pair.entries()){
       await page.setViewportSize({width:1440,height:1000});
       const button=(name,root=page)=>root.getByRole("button",{name,exact:true});
       const drawer=page.getByRole("dialog");
       if(name==="knowledge"){
         const rail=page.locator("#knowledge-retrieval-sessions");
         const create=button("新增会话",rail);
         const toggle=rail.locator("button[aria-controls='knowledge-retrieval-sessions']");
         const toolbar=page.locator(".knowledge-conversation-header");
         const agent=toolbar.locator("select");
         const more=button("会话操作：解释一下职业道德",rail);
         const badge=rail.locator(".agent-chat-session-badge");
         const menu=page.getByRole("menu",{name:"会话操作：解释一下职业道德"});
         const remove=menu.getByRole("menuitem",{name:"删除会话",exact:true});
         const confirmation=page.getByRole("alertdialog");
         const restoredAnswer=page.locator(".knowledge-message.assistant").filter({hasText:"职业道德是从业者在工作中应当遵守的行为规范。"});
         const restoredReference=button("引用 D1",restoredAnswer);
         await restoredReference.waitFor();
         if(index===0) await page.reload();
         else await page.goto(refOrigin+"/reference.html?page="+name);
         await restoredAnswer.waitFor();
         await restoredReference.waitFor();
         const restoredUserTime=page.locator(".knowledge-message.user time");
         await restoredUserTime.waitFor();
         assert.equal(await restoredUserTime.getAttribute("datetime"),"2026-11-22T15:48:01Z");
         assert.equal(await restoredUserTime.innerText(),"2026-11-22 23:48:01");
         assert.equal(await restoredAnswer.locator("time").getAttribute("datetime"),"2026-11-22T15:48:30Z");
         assert.equal(await restoredAnswer.locator("time").innerText(),"2026-11-22 23:48:30");
         await page.addStyleTag({content:"*,*::before,*::after{transition:none!important;animation:none!important}"});
         assert.equal(await restoredAnswer.locator(".knowledge-reference-card,[data-reference-label]").count(),0,"回答下不重复展示引用");
         await page.evaluate(()=>{globalThis.fixturePopups=0;window.open=()=>{globalThis.fixturePopups++;return null;};});
         const chatUrl=page.url();
         await page.evaluate(()=>{globalThis.fixtureCitationServiceFailure=true;});
         await restoredReference.click();
         await page.getByRole("dialog").getByText("引用服务暂不可用，请稍后重试。",{exact:true}).waitFor();
         assert.equal(page.url(),chatUrl,"内部引用服务故障不触发浏览器离开会话");
         await button("关闭资料预览").click();
         await restoredReference.waitFor();
         await page.evaluate(()=>{globalThis.fixtureCitationServiceFailure=false;});
         await restoredReference.click();
         const sourceDrawer=page.getByRole("dialog",{name:"审计制度.pdf"});
         await sourceDrawer.getByText(/正文引用 D1 · 第 1 页/).waitFor();
         await sourceDrawer.getByText("已定位并标黄命中文字。").waitFor();
         assert.equal(await sourceDrawer.locator(".knowledge-pdf-hit").count(),1,"重复文句只标黄冻结框内的引用");
         assert.ok(await sourceDrawer.locator(".knowledge-pdf-hit").evaluate(node=>parseFloat(node.style.top)>70),"定位到正文段落，不误标同页另一相同句子");
         await page.screenshot({path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-chat-reference.png")});
         await button("关闭资料预览").click();
         assert.ok(await restoredReference.evaluate(node=>document.activeElement===node),"关闭引用抽屉后焦点回正文引用");
         await page.evaluate(()=>{globalThis.fixtureCitationWithoutBox=true;});
         await restoredReference.click();
         await sourceDrawer.getByText("当前页无法核实命中文字的位置，仅定位到页。").waitFor();
         assert.equal(await sourceDrawer.locator(".knowledge-pdf-hit").count(),0,"缺少位置证据不能标黄");
         await button("关闭资料预览").click();
         await page.evaluate(()=>{globalThis.fixtureCitationWithoutBox=false;});
         await restoredAnswer.getByRole("button",{name:"查看检索结果"}).click();
         const resultsDrawer=page.getByRole("dialog",{name:"检索结果"});
         await resultsDrawer.getByText("已交给 Agent 1 项").waitFor();
         await resultsDrawer.getByText("预览文字证据 · 第 1 页",{exact:true}).click();
         await resultsDrawer.getByText("原句：图1-1 用于说明本节检索架构。",{exact:true}).waitFor();
         assert.equal(await page.getByRole("dialog").count(),1,"展开证据不叠加原文抽屉");
         assert.equal(await page.evaluate(()=>globalThis.fixturePopups),0,"打开引用和查看检索结果不弹新窗口");
         await page.screenshot({path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-chat-retrieval.png")});
         await button("关闭检索结果").click();
         await page.setViewportSize({width:390,height:844});
         await page.evaluate(()=>{globalThis.fixtureCitationPdfDenied=true;});
         await restoredReference.click();
         await sourceDrawer.getByText("原文暂不可预览，以下为本轮的引用证据。").waitFor();
         await sourceDrawer.getByText("原句：图1-1 用于说明本节检索架构。",{exact:true}).waitFor();
         assert.equal(await sourceDrawer.locator(".knowledge-pdf-hit").count(),0);
         assert.ok((await sourceDrawer.boundingBox()).width<=390);
         await page.screenshot({path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-chat-reference-mobile.png")});
         await page.keyboard.press("Escape");
         await sourceDrawer.waitFor({state:"hidden"});
         await page.setViewportSize({width:1440,height:1000});
         await badge.waitFor();
         assert.equal(await page.locator("button[aria-controls='knowledge-retrieval-sessions']").count(),1,"会话栏只有一个展开入口");
         assert.equal(await toolbar.locator("h2,button").count(),0,"主区工具栏只有 Agent 选择");
         const toolbarBox=await toolbar.boundingBox(), agentBox=await agent.boundingBox();
         assert.ok(agentBox.x-toolbarBox.x>=0 && agentBox.x-toolbarBox.x<=13 && agentBox.height===32,"Agent 下拉位于主区左上");
         assert.equal(await badge.innerText(),"解释");
         assert.equal(await more.isVisible(),false,"收起会话轨时不能显示删除入口");
         const plus=await create.locator("svg").boundingBox(), box=await create.boundingBox();
         assert.ok(Math.abs(plus.x+plus.width/2-box.x-box.width/2)<1 && Math.abs(plus.y+plus.height/2-box.y-box.height/2)<1,"加号水平垂直居中");
         const foldedToggle=await toggle.boundingBox();
         assert.ok(foldedToggle.y+foldedToggle.height<=box.y && Math.abs(foldedToggle.x-box.x)<1,"收起时三横在加号上方，两者对齐");
         const avatar=await page.locator(".knowledge-message.user .chat-message-avatar").boundingBox();
         const bubble=await page.locator(".knowledge-message.user .chat-message-bubble").boundingBox();
         assert.ok(avatar.x>=bubble.x+bubble.width && bubble.width<360,"用户头像靠右且短问题气泡不撑满");
         const send=await button("发送问题").boundingBox();
         const attachment=await button("添加查询图片").boundingBox();
         assert.ok(Math.abs(send.y+send.height/2-attachment.y-attachment.height/2)<1 && send.width===attachment.width,"附件与发送对齐");
         assert.equal(await page.locator(".knowledge-composer textarea").evaluate(node=>getComputedStyle(node).borderTopWidth),"0px","输入区只保留外层边框");
         await page.locator(".knowledge-composer-wrap input[type=file]").setInputFiles({name:"测试图.png",mimeType:"image/png",buffer:Buffer.from([1,2,3])});
         await page.getByText("测试图.png",{exact:true}).waitFor();
         await button("移除图片").click();
         await page.getByText("测试图.png",{exact:true}).waitFor({state:"hidden"});
         const composerInput=page.locator(".knowledge-composer textarea");
         await composerInput.fill("验证实时问答时间");
         await composerInput.press("Enter");
         const liveUser=page.locator(".knowledge-message.user").filter({hasText:"验证实时问答时间"});
         const liveAnswer=page.locator(".knowledge-message.assistant").filter({hasText:"这是一条实时回答。"});
         await liveAnswer.waitFor();
         assert.equal(await liveUser.locator("time").innerText(),"00:01:00");
         assert.equal(await liveAnswer.locator("time").getAttribute("datetime"),"2026-11-22T16:01:25Z");
         assert.equal(await liveAnswer.locator("time").innerText(),"00:01:25");
         await page.evaluate(()=>globalThis.fixtureFinishReply());
         const completedTime=liveAnswer.locator('time[datetime="2026-11-22T16:01:27Z"]');
         await completedTime.waitFor();
         assert.equal(await completedTime.innerText(),"00:01:27");
         await button("引用 D1",liveAnswer).waitFor();
         await liveAnswer.getByRole("button",{name:"查看检索结果"}).waitFor();
         await page.waitForFunction(()=>!document.querySelector(".knowledge-composer textarea").disabled);
         await toggle.click();
         await rail.getByText("解释一下职业道德",{exact:true}).waitFor();
         assert.equal(await rail.getByText("当前 Agent",{exact:true}).count(),0,"会话栏不重复显示当前 Agent");
         assert.equal(await rail.getByText(fixture.agent.display_name,{exact:true}).count(),0);
         assert.equal(await agent.inputValue(),fixture.agent.agent_id,"主区保留选择的 Agent");
         const openToggle=await toggle.boundingBox(), openCreate=await create.boundingBox();
         assert.ok(openToggle.x+openToggle.width<=openCreate.x && Math.abs(openToggle.y+openToggle.height/2-openCreate.y-openCreate.height/2)<1,"展开时三横在新增按钮左侧");
         assert.equal(await remove.isVisible(),false);
         await more.focus();
         await more.press("Enter");
         await remove.waitFor();
         await page.keyboard.press("Escape");
         await menu.waitFor({state:"hidden"});
         assert.equal(await more.evaluate(node=>document.activeElement===node),true);
         await more.click();
         await toggle.click();
         await menu.waitFor({state:"hidden"});
         assert.equal(await more.isVisible(),false);
         await page.screenshot({path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-chat-collapsed.png"),fullPage:true});
         await page.setViewportSize({width:390,height:844});
         assert.equal(await toggle.isVisible(),true,"窄屏收起后左上角开关仍可用");
         assert.equal(await rail.locator("nav").isVisible(),false);
         await toggle.click();
         await rail.getByText("解释一下职业道德",{exact:true}).waitFor();
         assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
         await more.click();
         await remove.waitFor();
         await page.screenshot({path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-chat-menu-390.png"),fullPage:true});
         await remove.click();
         await confirmation.getByText(/会话内容删除后无法恢复/).waitFor();
         assert.equal(await page.evaluate(()=>fixtureRequests.filter(row=>row.method==="DELETE").length),0,"打开菜单和确认框时不删除");
         await button(index===0?"永久删除":"删除会话",confirmation).click();
         await page.waitForFunction(()=>fixtureRequests.some(row=>row.method==="DELETE" && row.search==="?expected_row_version=4"));
         await rail.getByText("还没有会话",{exact:true}).waitFor();
         await create.click();
         await page.locator(".knowledge-chat-welcome h2").waitFor();
         assert.equal(await rail.locator("nav").isVisible(),false,"窄屏新会话切回聊天");
         await page.setViewportSize({width:1440,height:1000});
         await page.getByText("Enter 发送 · Shift + Enter 换行",{exact:true}).waitFor();
         const input=page.locator(".knowledge-composer textarea");
         await input.evaluate(node=>{
           globalThis.knowledgeSubmitCount=0;
           node.form.addEventListener("submit",event=>{
             event.preventDefault();
             event.stopImmediatePropagation();
             globalThis.knowledgeSubmitCount++;
           },{capture:true});
         });
         await input.fill("   ");
         await input.press("Enter");
         await input.fill("第一行");
         await input.press("End");
         await input.press("Shift+Enter");
         assert.equal(await input.inputValue(),"第一行\n","Shift+Enter 应在输入框内换行");
         await input.dispatchEvent("keydown",{key:"Enter",isComposing:true});
         await input.dispatchEvent("keydown",{key:"Enter",repeat:true});
         assert.equal(await page.evaluate(()=>globalThis.knowledgeSubmitCount),0,"空输入、输入法确认和长按重复不提交");
         await input.press("Enter");
         assert.equal(await page.evaluate(()=>globalThis.knowledgeSubmitCount),1,"Enter 通过同一个表单发送");
       }
       if(name==="search"){
         const reader=page.getByRole("region",{name:"原文阅读",exact:true});
         assert.equal(await page.getByLabel("结果类型").isVisible(),false);
         await button("更多条件").click();
         await page.getByLabel("结果类型").selectOption("TEXT");
         await page.getByLabel("主结果数").fill("20");
         await button("收起条件").click();
         await page.getByLabel("查询文字").fill("图1-1");
         await button("检索").click();
         await page.getByText(/2 份文档 · 2 处相关段落/).waitFor();
         await page.getByText(/仅文字与表格 · 20 条主结果/).waitFor();
         assert.equal(await page.locator(".knowledge-search-hit").count(),2);
         assert.equal(await page.getByText("同一段落的重叠片段",{exact:true}).count(),0);
         const firstSource=page.locator(".knowledge-search-source").filter({has:page.locator(".knowledge-search-document").filter({hasText:"产品手册"})});
         const documentToggle=firstSource.locator(":scope > summary");
         const chapterToggle=firstSource.locator(".knowledge-search-chapter > summary");
         assert.equal(await page.locator(".knowledge-search-source[open]").count(),1,"默认只展开第一个文档");
         await documentToggle.click();
         assert.equal(await chapterToggle.isVisible(),false,"折叠文档隐藏所属章节");
         await documentToggle.press("Enter");
         await chapterToggle.click();
         assert.equal(await firstSource.locator(".knowledge-search-hit").isVisible(),false,"章节可独立折叠");
         await chapterToggle.press("Space");
         await firstSource.getByRole("button",{name:/业务和内部控制规范/}).click();
         const pdf=reader.getByRole("region",{name:"PDF 原页"});
         await pdf.getByText("已定位并标黄命中文字。").waitFor({state:"attached"});
         const highlight=pdf.locator(".knowledge-pdf-hit");
         assert.equal(await highlight.count(),1,"重复文字只标中坐标匹配的一处");
         const box=await highlight.evaluate(node=>[parseFloat(node.style.top),parseFloat(node.style.width)]);
         assert.ok(box[0]>70 && box[0]<80,"命中落在页面下方的第二处: "+box);
         assert.ok(box[1]>2 && box[1]<12,"仅标中文字，不整段标黄: "+box);
         assert.ok(await pdf.evaluate(node=>node.scrollTop>100),"原页滚到命中文字");
         await button("下一页",reader).click();
         await pdf.locator('[data-pdf-page="2"]').waitFor();
         assert.equal(await highlight.count(),0,"其他页不套用高亮");
         await button("回到命中页",reader).click();
         await pdf.getByText("已定位并标黄命中文字。").waitFor({state:"attached"});
         await button("放大阅读区",reader).click();
         await drawer.waitFor({state:"visible"});
         assert.ok((await drawer.boundingBox()).width >= 1440 * .9);
         await button("放大",drawer).click();
         await drawer.getByText("125%",{exact:true}).waitFor();
         await drawer.getByText("已定位并标黄命中文字。").waitFor({state:"attached"});
         await page.keyboard.press("Escape");
         await drawer.waitFor({state:"hidden"});
         assert.equal(await firstSource.locator(".knowledge-search-hit").getAttribute("aria-pressed"),"true");
         await page.addStyleTag({content:".knowledge-search-context{max-height:200px!important}.knowledge-search-context .knowledge-passage-block:first-of-type{min-height:430px}"});
         await button("文字上下文",reader).click();
         const context=reader.getByRole("region",{name:"段落与上下文"});
         await context.waitFor();
         assert.deepEqual(await context.locator(".knowledge-passage-block p").allTextContents(),
           ["原文上一段","原句：图1-1 用于说明本节检索架构。","原文下一段"]);
         assert.equal(await context.locator("mark").textContent(),"图1-1");
         assert.ok(await reader.locator(".knowledge-search-context").evaluate(node=>node.scrollTop>0),"文字上下文自动滚到命中");
         await chapterToggle.click();
         await documentToggle.click();
         await button("PDF 原文",reader).click();
         await button("文字上下文",reader).click();
         assert.equal(await context.locator("mark").textContent(),"图1-1","收起来源不清除当前命中");
         await button("更多条件").click();
         await button("收起条件").click();
         assert.equal(await chapterToggle.isVisible(),false,"其他操作不重置文档折叠");
         await documentToggle.click();
         assert.equal(await firstSource.locator(".knowledge-search-hit").isVisible(),false,"重新展开文档保留章节折叠");
         await chapterToggle.click();
         await button("展开前文",context).click();
         await context.getByText("展开后显示的更早一段").waitFor();
         await button("展开后文",context).click();
         await context.getByText("展开后显示的更后一段").waitFor();
         await reader.getByText("图1-1 检索架构").waitFor();
         await button("继续读取关联",reader).click();
         await reader.getByText("关联续页").waitFor();
         await reader.getByText("图1-1 检索架构").waitFor();
         assert.equal(await page.locator(".knowledge-search-hit").count(),2,"关系续取不得清空主结果");
         await page.evaluate(()=>{globalThis.fixtureAmbiguousPdfHit=true;});
         await button("检索").click();
         await firstSource.getByRole("button",{name:/业务和内部控制规范/}).click();
         await pdf.getByText("当前页无法核实命中文字的位置，仅定位到页。").waitFor({state:"attached"});
         assert.equal(await highlight.count(),0,"来源范围内出现多处同文时不猜测");
         await page.evaluate(()=>{globalThis.fixtureAmbiguousPdfHit=false;});
         await button("更多条件").click();
         await page.getByRole("checkbox",{name:"按标题或日期查找"}).check();
         assert.equal(await page.getByLabel("查询文字").count(),0);
         await page.getByLabel("标题前缀").fill("制度");
         await page.getByRole("checkbox",{name:"逐页查全"}).check();
         await button("检索").click();
         await page.getByText("当前条件下的授权范围已查完。").waitFor();
         const structured=await page.evaluate(()=>JSON.parse(fixtureRequests.filter(row=>row.path.endsWith("/search")).at(-1).body));
         assert.deepEqual([structured.mode,structured.text,structured.requested_count,structured.predicates],
           ["EXHAUSTIVE",null,20,[{field:"title",operator:"PREFIX",value:"制度"}]]);
         await page.getByRole("checkbox",{name:"按标题或日期查找"}).uncheck();
         await page.getByLabel("结果类型").selectOption("ALL");
         await button("收起条件").click();
         const png=Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+cFqkAAAAASUVORK5CYII=","base64");
         await page.getByLabel("查找同一原图",{exact:true}).setInputFiles({name:"查询.png",mimeType:"image/png",buffer:png});
         await button("检索").click();
         await page.locator(".knowledge-search-hit").getByText("图1-1 检索架构").waitFor();
         const requests=await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith("/search")));
         assert.equal(JSON.parse(requests.at(-1).body).images.length,1);
         await button("移除图片").click();
         const query=page.getByLabel("查询文字"), wait=page.getByRole("region",{name:"检索等待",exact:true});
         const stop=button("停止等待",wait), awaiting=button("检索中…");
         const track=wait.locator(".knowledge-search-wait-track");
         const prefix=index===0?"kbot":"ammolite";
         // 恢复状态条真实动效，其他截图沿用测试的静态样式。
         await page.evaluate(()=>document.querySelectorAll("style").forEach(style=>{
           if(style.textContent==="*,*::before,*::after{transition:none!important;animation:none!important}")style.remove();
         }));
         await page.clock.install();
         await page.evaluate(()=>{globalThis.fixtureDelay=true;});
         await query.fill("第一次检索");
         await button("检索").click();
         await page.waitForFunction(()=>fixturePendingSearch.length===1);
         await wait.getByText("正在检索知识库…",{exact:true}).waitFor();
         assert.equal(await awaiting.isDisabled(),true,"请求未结束时禁止重复检索");
         assert.equal(await reader.getByText("检索结果返回后，可在此阅读原文。").isVisible(),true);
         assert.equal(await page.locator(".knowledge-search-hit").count(),0,"等待区不显示旧结果");
         assert.match(await track.evaluate(node=>getComputedStyle(node,"::before").animationName),/knowledge-search-wait-sweep/);
         await query.fill("下一次的条件");
         await query.press("Enter");
         await page.locator(".knowledge-search-formbar").dispatchEvent("submit");
         assert.equal(await page.evaluate(()=>fixturePendingSearch.length),1);
         assert.match(await wait.innerText(),/第一次检索/);
         assert.doesNotMatch(await wait.innerText(),/下一次的条件/);
         assert.equal(await wait.getByText(/这次检索用时较长/).isVisible(),false);
         await page.clock.fastForward(12000);
         await wait.getByText(/这次检索用时较长/).waitFor();
         const elapsed=await wait.locator(".knowledge-search-wait-time").innerText();
         assert.ok(Number(elapsed.match(/\d+/)?.[0])>=12,"计时使用实际经过时间");
         await page.screenshot({path:resolve(artifacts,prefix+"-search-wait-1440.png"),fullPage:true});
         await page.setViewportSize({width:390,height:844});
         await page.emulateMedia({reducedMotion:"reduce"});
         assert.equal(await track.evaluate(node=>getComputedStyle(node,"::before").animationName),"none");
         assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
         assert.equal(await stop.evaluate(node=>node.getBoundingClientRect().right<=innerWidth),true);
         await page.screenshot({path:resolve(artifacts,prefix+"-search-wait-390.png"),fullPage:true});
         await stop.click();
         await wait.waitFor({state:"detached"});
         await page.getByText("已停止等待。可修改条件或重新检索。",{exact:true}).waitFor();
         assert.equal(await query.inputValue(),"下一次的条件");
         assert.equal(await page.evaluate(()=>fixturePendingSearch[0].signal.aborted),true);
         await page.setViewportSize({width:1440,height:1000});
         await button("检索").click();
         await page.waitForFunction(()=>fixturePendingSearch.length===2);
         assert.equal(await page.evaluate(()=>fixturePendingSearch[1].query.text),"下一次的条件");
         await page.evaluate(()=>fixturePendingSearch[1].resolve("success"));
         await page.getByText(/2 份文档 · 2 处相关段落/).waitFor();
         await wait.waitFor({state:"detached"});
         await page.evaluate(async()=>{fixturePendingSearch[0].resolve("empty");await new Promise(resolve=>setTimeout(resolve,25));});
         assert.equal(await page.locator(".knowledge-search-hit").count(),2,"旧请求迟到时不得覆盖新结果");
         await button("检索").click();
         await page.waitForFunction(()=>fixturePendingSearch.length===3);
         await page.evaluate(()=>fixturePendingSearch[2].resolve("failure"));
         await page.getByRole("alert").waitFor();
         await wait.waitFor({state:"detached"});
         assert.equal(await query.inputValue(),"下一次的条件");
         await button("检索").click();
         await page.waitForFunction(()=>fixturePendingSearch.length===4);
         await page.evaluate(()=>fixturePendingSearch[3].resolve("empty"));
         await wait.waitFor({state:"detached"});
         assert.equal(await page.getByRole("alert").count(),0,"空结果不能显示为请求失败");
         assert.equal(await page.locator(".knowledge-search-hit").count(),0);
         await query.fill("切库前检索");
         await button("检索").click();
         await page.waitForFunction(()=>fixturePendingSearch.length===5);
         await page.locator(".knowledge-search-top select").selectOption("library-2");
         await wait.waitFor({state:"detached"});
         assert.equal(await page.evaluate(()=>fixturePendingSearch[4].signal.aborted),true,"切库终止旧等待");
         await page.evaluate(async()=>{fixturePendingSearch[4].resolve("success");await new Promise(resolve=>setTimeout(resolve,25));});
         assert.equal(await page.locator(".knowledge-search-hit").count(),0,"切库不展示旧库响应");
         await query.fill("切域前检索");
         await button("检索").click();
         await page.waitForFunction(()=>fixturePendingSearch.length===6);
         if(index===0)await page.evaluate(()=>dispatchEvent(new Event("kbot:knowledge-context-changing")));
         else await page.evaluate(()=>globalThis.fixtureSwitchDomain());
         await wait.waitFor({state:"detached"});
         assert.equal(await page.evaluate(()=>fixturePendingSearch[5].signal.aborted),true,"切域终止旧等待");
         await page.evaluate(async()=>{
           globalThis.fixtureDelay=false;fixturePendingSearch[5].resolve("success");await new Promise(resolve=>setTimeout(resolve,25));
         });
         await page.waitForFunction(()=>!document.querySelector(".knowledge-search-formbar [type=submit]:not(:disabled)") || !document.querySelector(".knowledge-search-formbar [name=query]")?.value);
         assert.equal(await page.locator(".knowledge-search-hit").count(),0,"旧 Domain 迟到结果必须丢弃");
       }
       if(name==="files"){
         const add=page.getByRole("button",{name:/上传资料/});
         const readable=page.getByRole("button",{name:/另一资料库/});
         const writable=page.getByRole("button",{name:/产品知识库/});
         const initialBackground=await writable.evaluate(node=>getComputedStyle(node).backgroundColor);
         assert.equal(await add.isDisabled(),true,"未选库不得添加");
         await readable.click();
         await page.getByText("当前没有可见文件。").waitFor();
         assert.equal(await readable.getAttribute("aria-pressed"),"true");
         assert.equal(await add.isDisabled(),true,"只读库不得添加");
         assert.equal(await page.getByRole("navigation",{name:"资料队列"}).count(),0,"只读库不显示任何审批队列");
         assert.equal(await button("审批").count(),0);
         await writable.click();
         await button("查看文件").waitFor();
         assert.equal(await writable.getAttribute("aria-pressed"),"true");
         assert.equal(await readable.getAttribute("aria-pressed"),"false");
         assert.notEqual(await writable.evaluate(node=>getComputedStyle(node).backgroundColor),initialBackground,"选中库有独立底色");
         const filters=page.getByRole("search",{name:"筛选文件列表"});
         const title=filters.getByRole("searchbox",{name:"按文件标题查找"});
         const status=filters.getByRole("combobox",{name:"发布状态"});
         await title.fill("产品");
         await status.selectOption("PUBLISHED");
         await title.press("Enter");
         await page.getByText("没有匹配当前筛选的文件。").waitFor();
         const filtered=await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith("/documents")).at(-1).search);
         assert.equal(new URLSearchParams(filtered).get("title"),"产品");
         assert.equal(new URLSearchParams(filtered).get("state"),"PUBLISHED");
         await title.clear();
         await status.selectOption("");
         await button("筛选",filters).click();
         await button("查看文件").waitFor();
         await page.locator(".knowledge-directory-head").screenshot({
           path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-files-filter-header.png"),
         });
         await page.locator(".knowledge-directory-libraries").screenshot({
           path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-files-selected-libraries.png"),
         });
         await page.screenshot({path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-files-selected.png")});
         await add.click();
         await drawer.getByRole("heading",{name:"上传资料",exact:true}).waitFor();
         await drawer.locator(".knowledge-upload-destination").getByText("产品知识库",{exact:true}).waitFor();
         assert.equal(await drawer.evaluate(node=>getComputedStyle(node).getPropertyValue("--accent")),
           await page.locator("main, .business-app-shell").first().evaluate(node=>getComputedStyle(node).getPropertyValue("--accent")),
           "上传抽屉沿用知识检索 App 主题");
         const uploadFooter=drawer.locator(".admin-drawer-footer");
         assert.equal(await button("提交文件",uploadFooter).isDisabled(),true);
         const chooserPromise=page.waitForEvent("filechooser");
         await button("选择文件",drawer).click();
         const chooser=await chooserPromise;
         assert.equal(chooser.isMultiple(),true);
         await chooser.setFiles([
           {name:"合成.txt",mimeType:"text/plain",buffer:Buffer.from("合成资料")},
           {name:"附加.txt",mimeType:"text/plain",buffer:Buffer.from("另一份独立资料")},
         ]);
         await uploadFooter.getByText("已选 2 个文件",{exact:true}).waitFor();
         assert.equal(await drawer.getByText("待提交",{exact:true}).count(),2);
         await drawer.screenshot({path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-upload-a.png")});
         assert.equal(await page.evaluate(()=>fixtureRequests.filter(row=>row.method==="POST").length),0,"选文件不能自动上传");
         await page.evaluate(()=>{globalThis.fixtureFailNext=true;});
         await button("提交文件").click();
         await page.getByText("已接收 1 / 2 个文件").waitFor();
         await page.getByText(/隔离失败|接收结果未确认/).waitFor();
         await uploadFooter.getByText("1 个文件需要重试",{exact:true}).waitFor();
         assert.equal(await drawer.getByText("接收失败",{exact:true}).count(),1);
         assert.equal(await page.getByRole("link",{name:"查看解析任务"}).count(),1);
         await button("提交未接收文件").click();
         await page.getByText("已接收 2 / 2 个文件").waitFor();
         assert.equal(await page.getByRole("link",{name:"查看解析任务"}).count(),2);
         const keys=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==="POST"&&row.path.endsWith("/documents")).map(row=>row.headers["idempotency-key"]));
         assert.ok(keys[0] && keys[1]);assert.notEqual(keys[0],keys[1]);
         assert.deepEqual(keys,[keys[0],keys[1],keys[0]],"只重试失败文件并沿用原键");
         assert.equal(await uploadFooter.getByRole("button",{name:/提交/}).count(),0);
         await button("完成",drawer).click();
         await drawer.waitFor({state:"hidden"});
         await page.setViewportSize({width:320,height:680});
         await add.click();
         const droppedName="合成的超长文件名".repeat(10)+".txt";
         const zone=drawer.locator(".knowledge-drop");
         assert.equal(await zone.evaluate((node,name)=>{
           const transfer=new DataTransfer();
           transfer.items.add(new File(["合成资料"],name,{type:"text/plain"}));
           node.dispatchEvent(new DragEvent("dragover",{bubbles:true,cancelable:true,dataTransfer:transfer}));
           const highlighted=node.hasAttribute("data-drag-active");
           node.dispatchEvent(new DragEvent("drop",{bubbles:true,cancelable:true,dataTransfer:transfer}));
           return highlighted;
         },droppedName),true,"拖入文件时应高亮投放区");
         await drawer.getByText(droppedName,{exact:true}).waitFor();
         const dimensions=await drawer.evaluate(node=>{
           const body=node.querySelector(".admin-drawer-body"), footer=node.querySelector("footer");
           return [body.scrollWidth<=body.clientWidth+1,footer.getBoundingClientRect().bottom<=innerHeight+1];
         });
         assert.deepEqual(dimensions,[true,true],"窄屏文件名不溢出，底栏固定在视口内");
         await button("关闭",uploadFooter).click();
         const abandon=page.getByRole("alertdialog",{name:"放弃未完成的文件提交？"});
         await button("取消",abandon).click();
         await drawer.getByText(droppedName,{exact:true}).waitFor();
         await button("关闭",uploadFooter).click();
         await button("放弃并关闭",abandon).click();
         await drawer.waitFor({state:"hidden"});
         assert.equal(await page.evaluate(()=>fixtureRequests.filter(row=>row.method==="POST").length),3,"放弃不触发上传");
         await page.setViewportSize({width:1440,height:1000});
       }
       if(name==="processing"){
         const filters=page.getByRole("search",{name:"筛选解析任务列表"});
         const picker=page.getByRole("complementary",{name:"选择知识库"});
         const first=picker.getByRole("button",{name:/^产品知识库/});
         const second=picker.getByRole("button",{name:/^很长的知识库名称/});
         const search=filters.getByRole("searchbox",{name:"按文件标题查找"});
         const current=page.locator(".knowledge-current-library");
         await search.fill("切换前的条件");
         await second.click();
         await page.getByText("当前没有可见文件。").waitFor();
         await current.getByRole("heading",{name:/^很长的知识库名称/}).waitFor();
         assert.equal(await second.getAttribute("aria-pressed"),"true");
         assert.equal(await first.getAttribute("aria-pressed"),"false");
         assert.equal(await search.inputValue(),"");
         assert.equal(await filters.getByRole("checkbox",{name:"仅本人"}).count(),0);
         const switched=await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith("/documents")).at(-1));
         assert.match(switched.path,/library-2\/documents$/);
         await first.click();
         await button("查看解析结果：未完成").waitFor();
         await filters.getByRole("combobox",{name:"处理进度"}).selectOption("PARSING");
         await filters.getByRole("checkbox",{name:"仅本人"}).check();
         await search.fill("产品");
         await search.press("Enter");
         await page.getByText("没有匹配当前筛选的文件。").waitFor();
         await page.waitForFunction(()=>new URLSearchParams(fixtureRequests.filter(row=>row.path.endsWith("/documents")).at(-1)?.search).get("title")==="产品");
         const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith("/documents")).at(-1));
         assert.match(request.path,/library-1\/documents$/);
         const params=new URLSearchParams(request.search);
         assert.equal(params.get("title"),"产品");
         assert.equal(params.get("state"),"PARSING");
         assert.equal(params.get("mine"),"true");
         await filters.getByRole("combobox",{name:"处理进度"}).selectOption("");
         await filters.getByRole("checkbox",{name:"仅本人"}).uncheck();
         await search.clear();
         await button("筛选",filters).click();
         await button("查看解析结果：未完成").waitFor();
       }
       if(name==="files" || name==="processing"){
         const checkDetailDrawer=async (eyebrow,filename,hasFooter)=>{
           await drawer.getByRole("heading",{name:filename,exact:true}).waitFor();
           assert.equal(await drawer.locator(".admin-drawer-header .eyebrow").textContent(),eyebrow);
           for(const [width,height] of [[1440,1000],[390,620],[320,520]]){
             await page.setViewportSize({width,height});
             const measured=await drawer.evaluate(panel=>{
               const head=panel.querySelector(".admin-drawer-header");
               const body=panel.querySelector(".admin-drawer-body");
               const grid=panel.querySelector(".knowledge-file-facts");
               const close=head.querySelector(".icon-button");
               const footer=panel.querySelector(".admin-drawer-footer");
               const rect=panel.getBoundingClientRect();
               body.scrollTop=body.scrollHeight;
               return {width:Math.round(rect.width),left:Math.round(rect.left),height:Math.round(rect.height),
                 headerPadding:getComputedStyle(head).padding,bodyPadding:getComputedStyle(body).padding,
                 gap:getComputedStyle(body).gap, columns:getComputedStyle(grid).gridTemplateColumns.split(" ").length,
                 noOverflow:body.scrollWidth<=body.clientWidth+1, closeWidth:Math.round(close.getBoundingClientRect().width),
                 top:Math.round(head.getBoundingClientRect().top), footerBottom:footer?Math.round(footer.getBoundingClientRect().bottom):null};
             });
             assert.deepEqual(measured,{width:Math.min(820,width),left:Math.max(width-820,0),height,
               headerPadding:width<=680?"21px 18px 18px":"26px 28px 22px",bodyPadding:width<=760?"18px":"22px 28px 32px",
               gap:"18px",columns:width<=760?1:3,noOverflow:true,closeWidth:40,top:0,footerBottom:hasFooter?height:null});
           }
           await page.setViewportSize({width:1440,height:1000});
           await drawer.locator(".admin-drawer-body").evaluate(node=>{node.scrollTop=0;});
         };
         if(name==="files"){
           const product=page.getByRole("row",{name:/产品手册/});
           const filename=product.locator('td[data-label="文件名"]');
           const status=product.locator('td[data-label="状态"]');
           assert.equal((await filename.textContent()).trim(),"产品手册","文件名列不展示版本标识");
           assert.equal((await status.textContent()).trim(),"待审核","不要把未发布与待审核叠在一起");
           assert.equal(await status.locator(".status-pill.status-warning").count(),1);
         }
         if(name==="files"){
           await button("查看文件").click();
           await drawer.getByText("version-1",{exact:true}).waitFor();
         }
         else {
           const result=button("查看解析结果：未完成");
           assert.equal(await result.getAttribute("aria-haspopup"),"dialog");
           assert.equal(await result.locator(".status-pill").count(),0);
           assert.match(await result.innerText(),/未完成\s*›/);
           assert.deepEqual(await result.evaluate(node=>{
             const style=getComputedStyle(node);
             return [style.borderTopWidth,style.backgroundColor];
           }),["0px","rgba(0, 0, 0, 0)"],"解析状态为文字入口，不再套边框和徽章");
           await result.press("Enter");
           await checkDetailDrawer("解析结果","产品手册",true);
           assert.deepEqual(await drawer.locator(".knowledge-file-facts dt").allTextContents(),["解析结果","创建时间","审核","发布"]);
           assert.equal(await drawer.getByText("version-1",{exact:true}).count(),0);
         }
         assert.equal(await button("通过审核",drawer).count(),0);
         if(name==="files"){
           await checkDetailDrawer("查看文件","产品手册",false);
           assert.deepEqual(await drawer.locator(".knowledge-file-facts dt").allTextContents(),["版本","创建时间","处理","索引","审核","发布"]);
           await drawer.locator("pre").getByText("合成原文：<img src=x onerror=alert(1)>",{exact:true}).waitFor();
           assert.equal(await drawer.locator("pre img").count(),0,"原文必须作为纯文本");
           assert.equal(await drawer.getByRole("button",{name:/下载|保存/}).count(),0);
           assert.equal(await drawer.locator("a[download]").count(),0);
           await drawer.getByLabel("可读版本").selectOption("version-office");
           await drawer.getByText(/不支持内联预览/).waitFor();
           assert.equal(await drawer.locator("pre").count(),0,"切版不得保留旧预览");
           const previews=await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith("/source")));
           assert.equal(previews.length,1,"不支持的格式不请求预览");
           assert.match(previews[0].path,/library-1\/versions\/version-1\/source$/);
           await drawer.getByLabel("可读版本").selectOption("version-png");
           const image=drawer.locator('section[aria-label="原文预览"] img');
           await image.waitFor();
           await page.waitForFunction(()=>document.querySelector('section[aria-label="原文预览"] img')?.naturalWidth===1);
           assert.equal(await image.getAttribute("draggable"),"false");
           assert.equal(await image.evaluate(node=>node.dispatchEvent(new MouseEvent("contextmenu",{bubbles:true,cancelable:true}))),false);
           // 无头 Chromium 可能不带原生 PDF 阅读器；分别模拟禁用与启用，验证 MIME/Blob 和回退。
           await page.evaluate(()=>Object.defineProperty(navigator,"pdfViewerEnabled",{configurable:true,value:false}));
           await drawer.getByLabel("可读版本").selectOption("version-pdf");
           await drawer.getByText(/不支持内联预览/).waitFor();
           assert.equal(await image.count(),0,"切版不得保留旧图片");
           assert.equal(await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith("/version-pdf/source")).length),0);
           await drawer.getByLabel("可读版本").selectOption("version-office");
           await page.evaluate(()=>{
             Object.defineProperty(navigator,"pdfViewerEnabled",{configurable:true,value:true});
             globalThis.fixtureInvalidPdf=true;
           });
           await drawer.getByLabel("可读版本").selectOption("version-pdf");
           await drawer.getByText("原文预览未能加载。").waitFor();
           const pdf=drawer.locator('iframe[title="当前版本 PDF 原文预览"]');
           assert.equal(await pdf.count(),0,"不渲染伪装成 PDF 的内容");
           await page.evaluate(()=>{globalThis.fixtureInvalidPdf=false;});
           await button("重试预览",drawer).click();
           await pdf.waitFor();
           const pdfUrl=await pdf.getAttribute("src");
           assert.match(pdfUrl,/^blob:/);
           assert.equal(new URL(pdfUrl).hash,"#toolbar=0","请求原生 PDF 阅读器隐藏工具栏");
           assert.deepEqual(await page.evaluate(async url=>{
             const blob=await (await fetch(url)).blob();
             return [blob.type,await blob.slice(0,5).text()];
           },pdfUrl),["application/pdf","%PDF-"]);
           await drawer.getByLabel("可读版本").selectOption("version-office");
           await drawer.getByText(/不支持内联预览/).waitFor();
           assert.equal(await pdf.count(),0);
           assert.equal(await page.evaluate(url=>fetch(url).then(()=>false,()=>true),pdfUrl),true,"切版释放旧 PDF Blob URL");
           assert.equal(await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith("/source")).every(row=>new URLSearchParams(row.search).get("preview")==="true")),true);
           await button("关闭",drawer).click();
           await drawer.waitFor({state:"hidden"});
           await page.evaluate(()=>{globalThis.fixtureGovernanceScenario=true;});
           await button("筛选",page.getByRole("search",{name:"筛选文件列表"})).click();
           const product=page.getByRole("row",{name:/产品手册/});
           const status=product.locator('td[data-label="状态"]');
           await status.getByText("已发布",{exact:true}).waitFor();
           assert.equal((await status.textContent()).trim(),"已发布","其他版本待审不应改变当前版本状态");
           assert.equal(await status.locator(".status-pill").count(),1);
           await button("审批其他版本",product).click();
           await drawer.getByText("version-1",{exact:true}).waitFor();
           await checkDetailDrawer("资料审批","产品手册",true);
           await drawer.getByText("审核通过后开始解析，索引校验通过后自动发布。",{exact:true}).waitFor();
           assert.equal(await drawer.getByText("version-old",{exact:true}).count(),0,"审批不能指向展示在文件行的已发布旧版本");
           assert.equal(await drawer.getByLabel("审核意见（选填）",{exact:true}).inputValue(),"");
           assert.equal(await button("拒绝",drawer).isEnabled(),true,"审核意见留空不阻止拒绝");
           assert.equal(await button("通过审核",drawer).isDisabled(),true,"自动判断不可用时不能静默降档");
           const strategy=drawer.getByLabel("批准时选择解析方案");
           assert.equal(await strategy.inputValue(),"");
           assert.equal(await strategy.locator('option[value=""][hidden][disabled]').textContent(),"","未选择时留空，不显示占位选项");
           assert.deepEqual(await strategy.locator("option").evaluateAll(options=>options
             .filter(option=>["AUTO","VISUAL"].includes(option.value)).map(option=>[option.value,option.disabled])),
           [["AUTO",true],["VISUAL",true]]);
           await strategy.selectOption("TEXT");
           await button("通过审核",drawer).click();
           await page.getByText("审核决定已保存。",{exact:true}).waitFor();
           assert.equal(await page.evaluate(()=>JSON.parse(fixtureRequests.find(row=>row.path.endsWith("version-1:review")).body).comment),null,"审核意见留空可通过");
           const queue=page.getByRole("navigation",{name:"资料队列"});
           await button("待审批",queue).click();
           const pending=page.getByRole("row",{name:/附件二/});
           await button("审批",pending).click();
           await drawer.getByRole("heading",{name:"附件二",exact:true}).waitFor();
           assert.equal(await strategy.inputValue(),"AUTO","允许自动判断时，单件审批默认选中自动");
           assert.equal(await button("通过审核",drawer.locator(".admin-drawer-footer")).isEnabled(),true);
           await button("关闭",drawer).click();
           await drawer.waitFor({state:"hidden"});
           const batch=page.getByRole("region",{name:"批量审批"});
           const second=page.getByRole("checkbox",{name:/附件二/}), third=page.getByRole("checkbox",{name:/附件三/});
           await second.check();
           assert.equal((await pending.locator('td[data-label="文件名"]').textContent()).trim(),"附件二");
           assert.equal((await pending.locator('td[data-label="审核"]').textContent()).trim(),"待审核");
           assert.equal((await pending.locator('td[data-label="发布"]').textContent()).trim(),"未发布","队列各列只表达自己的状态");
           await third.check();
           const batchStrategy=batch.getByLabel("批量审批解析方案");
           assert.equal(await batchStrategy.inputValue(),"","批量不预选方案");
           assert.equal(await batchStrategy.locator('option[value=""][hidden][disabled]').textContent(),"");
           assert.deepEqual(await batchStrategy.locator("option:not([hidden])").evaluateAll(nodes=>nodes.map(node=>node.value)),["TEXT"],"批量只提供所选版本共同可用的方案");
           assert.equal(await button("批量通过",batch).isDisabled(),true);
           await batchStrategy.selectOption("TEXT");
           for(const width of [390,320]){
             await page.setViewportSize({width,height:900});
             assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,"批量审批控件不得撑破窄屏");
           }
           await page.setViewportSize({width:1440,height:1000});
           await page.evaluate(()=>{globalThis.fixtureFailReviewId="version-third";});
           await button("批量通过",batch).click();
           const confirm=page.getByRole("alertdialog");
           await confirm.getByText(/解析方案：原生解析/).waitFor();
           await confirm.getByText(/索引校验通过后自动发布/).waitFor();
           assert.equal(await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith(":review")).length),1,"确认前不提交批量审批");
           await button("批量通过",confirm).click();
           await page.getByText("批量通过：1 个成功，1 个未确认。",{exact:true}).waitFor();
           await page.getByRole("list",{name:"批量审批失败的版本"}).getByText(/附件三.*version-third/).waitFor();
           assert.equal(await second.count(),0,"成功项必须离开待审批");
           assert.equal(await third.isChecked(),false,"刷新后不要自动重试未确认项");
           await third.check();
           await button("批量拒绝",batch).click();
           await button("批量拒绝",confirm).click();
           await page.getByText("批量拒绝：1 个成功，0 个未确认。",{exact:true}).waitFor();
           const decisions=await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith(":review")).map(row=>[
             row.path.split("/").at(-1),JSON.parse(row.body).decision,JSON.parse(row.body).parse_strategy ?? null]));
           assert.deepEqual(decisions,[["version-1:review","APPROVE","TEXT"],["version-second:review","APPROVE","TEXT"],
             ["version-third:review","APPROVE","TEXT"],["version-third:review","REJECT",null]]);
           assert.equal(await button("可发布",queue).count(),0);
           await button("已发布",queue).click();
           assert.equal(await button("发布",product).count(),0);
           assert.equal(await page.evaluate(()=>fixtureRequests.some(row=>row.path.endsWith(":publish"))),false);
           await button("撤回",product.first()).click();
           await drawer.getByText("version-1",{exact:true}).waitFor();
           await button("撤回此版本",drawer).click();
           await button("撤回版本",confirm).click();
           await page.getByText("已撤回版本。",{exact:true}).waitFor();
           assert.match(await page.evaluate(()=>fixtureRequests.find(row=>row.path.endsWith(":withdraw")).path),/version-1:withdraw$/);
         }
         else {
           const filters=page.getByRole("search",{name:"筛选解析任务列表"});
           const footer=drawer.locator(".admin-drawer-footer");
           assert.equal(await footer.getByRole("link",{name:"查看文件"}).count(),1);
           await page.keyboard.press("Escape");
           await drawer.waitFor({state:"hidden"});
           const taskTable=page.locator(".knowledge-directory-table");
           assert.deepEqual(await taskTable.getByRole("columnheader").allTextContents(),["文件名","版本创建时间","处理进度","解析结果"]);
           await button("进度说明").click();
           const help=page.getByRole("dialog",{name:"进度说明"});
           await help.getByText(/全部批次提交且最终校验通过后才显示 100%/).waitFor();
           await page.keyboard.press("Escape");
           await help.waitFor({state:"hidden"});
           for(const [stage,batches,total,validated,text,pct] of [
             ["INDEXING",2,null,false,"正在计算总批次",null],
             ["INDEXING",6,10,false,"6 / 10 批已提交","60"],
             ["INDEX_RETRY",4,10,false,"索引待重试","40"],
             ["INDEX_CONFIRMING",10,10,false,"校验索引",null],
             ["READY",10,10,true,"索引已就绪","100"],
           ]){
             await page.evaluate(fixture=>{globalThis.fixtureProgress=fixture;},[stage,batches,total,validated]);
             await button("刷新进度",filters).click();
             await taskTable.getByText(new RegExp(text)).first().waitFor();
             assert.equal(await taskTable.getByRole("progressbar").getAttribute("aria-valuenow"),pct);
           }
           await page.evaluate(()=>{globalThis.fixtureProgress=null;});
           await page.evaluate(()=>{globalThis.fixtureMaintenance="parse";});
           await button("筛选",filters).click();
           assert.equal(await taskTable.getByRole("columnheader").last().textContent(),"解析结果");
           assert.doesNotMatch(await taskTable.innerText(),/version-1|INVALID_CONTRACT/);
           await button("查看解析结果：失败").click();
           await checkDetailDrawer("解析结果","产品手册",true);
           await drawer.screenshot({path:resolve(artifacts,(index===0?"kbot":"ammolite")+"-processing-result.png")});
           await drawer.getByText("INVALID_CONTRACT",{exact:true}).waitFor();
           await drawer.getByText(/解析提取文件内容.*索引校验通过后自动发布/).waitFor();
           await drawer.getByText("内容没有提取成功，可先重新解析。",{exact:true}).waitFor();
           const retry=drawer.getByLabel("解析方式");
           await retry.waitFor();
           assert.equal(await retry.inputValue(),"ORIGINAL");
           await retry.selectOption("CURRENT");
           await drawer.getByLabel("解析方案",{exact:true}).selectOption("TEXT");
           await button("重新解析",footer).click();
           const confirm=page.getByRole("alertdialog");
           await button("重新解析",confirm).click();
           await page.getByText("重新解析任务已排队。",{exact:true}).waitFor();
           const submitted=await page.evaluate(()=>fixtureRequests.find(row=>row.path.endsWith(":reprocess")));
           assert.deepEqual(JSON.parse(submitted.body),{expected_job_id:"job-1",mode:"CURRENT",parse_strategy:"TEXT"});
           await page.evaluate(()=>{globalThis.fixtureMaintenance="index";});
           await button("筛选",filters).click();
           await taskTable.getByText("建索引失败",{exact:true}).waitFor();
           assert.doesNotMatch(await taskTable.innerText(),/version-1|TIMEOUT/);
           await button("查看解析结果：成功").click();
           await drawer.getByText("内容已经提取成功，只需重试建索引。",{exact:true}).waitFor();
           await drawer.getByText("TIMEOUT",{exact:true}).waitFor();
           await drawer.getByText("批次明细",{exact:true}).click();
           await drawer.getByText("第 1–4 批已提交",{exact:true}).waitFor();
           await drawer.getByText("第 5–10 批尚未提交",{exact:true}).waitFor();
           assert.equal(await button("重新解析",drawer).count(),0);
           await button("重试建索引",footer).click();
           await button("重试建索引",confirm).click();
           await page.getByText("重试建索引任务已排队。",{exact:true}).waitFor();
           assert.deepEqual(await page.evaluate(()=>JSON.parse(fixtureRequests.find(row=>row.path.endsWith(":retry-index")).body)),{expected_job_id:"job-1"});
           await page.evaluate(()=>{globalThis.fixtureMaintenance="success";});
           await button("筛选",filters).click();
           const firstFile=taskTable.getByRole("row").filter({hasText:"产品手册"});
           const secondFile=taskTable.getByRole("row").filter({hasText:"安装说明"});
           await firstFile.getByText("索引已就绪",{exact:true}).waitFor();
           await button("查看解析结果：成功",firstFile).click();
           await drawer.getByText(/重新解析会重新读取原文件并建索引.*重建索引复用已有解析结果/).waitFor();
           assert.equal(await button("重新解析",footer).isEnabled(),true);
           await page.evaluate(()=>{globalThis.fixtureFailNext=true;});
           await button("重建索引",footer).click();
           await drawer.getByText("重建任务未接收，请重试",{exact:true}).waitFor();
           await page.evaluate(()=>{globalThis.fixtureDelayRebuildList=true;});
           await button("重建索引",footer).click();
           await drawer.waitFor({state:"hidden"});
           await page.getByText("索引重建已排队，可在表格查看进度。",{exact:true}).waitFor();
           await firstFile.getByText("等待重建索引",{exact:true}).waitFor();
           assert.equal(await firstFile.getByRole("progressbar").getAttribute("aria-valuenow"),"0");
           assert.equal(await secondFile.getByRole("progressbar").getAttribute("aria-valuenow"),"100");
           await page.waitForFunction(()=>typeof globalThis.fixtureRebuildListRelease==="function");
           await page.evaluate(()=>globalThis.fixtureRebuildListRelease());
           await button("查看解析结果：成功",secondFile).click();
           await checkDetailDrawer("解析结果","安装说明",true);
           assert.equal(await button("刷新索引任务",footer).count(),0);
           await button("重建索引",footer).click();
           await drawer.waitFor({state:"hidden"});
           await secondFile.getByText("等待重建索引",{exact:true}).waitFor();
           assert.equal(await secondFile.getByRole("progressbar").getAttribute("aria-valuenow"),"0");
           await button("查看解析结果：成功",firstFile).click();
           await checkDetailDrawer("解析结果","产品手册",true);
           await button("刷新索引任务",footer).waitFor();
           assert.equal(await button("重建索引",footer).count(),0);
           assert.equal(await button("采用待发布索引",footer).count(),0);
           assert.equal(await drawer.getByRole("progressbar").getAttribute("aria-valuenow"),"0");
           await page.evaluate(()=>{globalThis.fixtureFileIndexBatches={"version-1":4};});
           await drawer.getByText("4 / 10 批已提交",{exact:true}).waitFor();
           await page.keyboard.press("Escape");
           await drawer.waitFor({state:"hidden"});
           await firstFile.getByText("正在重建索引",{exact:true}).waitFor();
           assert.equal(await firstFile.getByRole("progressbar").getAttribute("aria-valuenow"),"40");
           assert.equal(await secondFile.getByRole("progressbar").getAttribute("aria-valuenow"),"0");
           await page.evaluate(()=>{globalThis.fixtureFileIndexBatches={"version-1":10};});
           await firstFile.getByText("重建完成 · 已生效",{exact:true}).waitFor();
           assert.equal(await firstFile.getByRole("progressbar").getAttribute("aria-valuenow"),"100");
           assert.equal(await secondFile.getByRole("progressbar").getAttribute("aria-valuenow"),"0");
           await page.evaluate(()=>{globalThis.fixtureFileIndexBatches["version-2"]=10;});
           await secondFile.getByText("重建完成 · 已生效",{exact:true}).waitFor();
           assert.equal(await secondFile.getByRole("progressbar").getAttribute("aria-valuenow"),"100");
           assert.ok(await page.evaluate(()=>["version-1","version-2"].every(id=>fixtureRequests.some(
             row=>row.path.endsWith("/index-rebuild") && row.method==="GET" && row.search===`?version_id=${id}`))));
           await button("查看解析结果：成功",firstFile).click();
           await button("重建索引",footer).waitFor();
           await drawer.getByText(/重新解析期间旧版仍可检索/).waitFor();
           await button("重新解析",footer).click();
           await confirm.getByText(/期间旧版仍可检索/).waitFor();
           await button("重新解析",confirm).click();
           await drawer.waitFor({state:"hidden"});
           await firstFile.getByText("排队等候解析",{exact:true}).waitFor();
           assert.equal(await secondFile.getByRole("progressbar").getAttribute("aria-valuenow"),"100");
           assert.deepEqual(await page.evaluate(()=>JSON.parse(fixtureRequests.filter(row=>row.path.endsWith(":reprocess")).at(-1).body)),
             {expected_job_id:"job-1",mode:"ORIGINAL"});
           await page.evaluate(()=>{globalThis.fixtureLibraryBatch=true;globalThis.fixtureMaintenance="";});
           await button("索引重建").click();
           const batch=page.getByRole("region",{name:"索引重建任务"});
           assert.equal(await batch.getByRole("progressbar").getAttribute("aria-valuenow"),"40");
           assert.equal(await button("确认切换整批索引",batch).count(),0);
           await button("重试此文件",batch).click();
           await button("重试此文件",confirm).click();
           await batch.getByText(/已切换 · 文本 Embedding/).waitFor();
           assert.deepEqual(await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith("/index-rebuild")&&row.method==="POST").map(row=>JSON.parse(row.body))),[
             {action:"START",version_id:"version-1",expected_generation_id:"generation-1"},
             {action:"START",version_id:"version-1",expected_generation_id:"generation-1"},
             {action:"START",version_id:"version-2",expected_generation_id:"generation-2"},
             {action:"RETRY",batch_id:"batch-1",version_id:"version-1"},
           ]);
         }
       }
       if(name==="knowledge-cores"){
         if(process.env.KNOWLEDGE_UI_INDEX_EDIT_ONLY==="1"){
           await page.evaluate(()=>{globalThis.fixtureIndexEdit=true;});
           const card=page.locator(".knowledge-base-card").filter({hasText:"产品知识库"});
           await button("编辑",card).click();
           const model=drawer.getByLabel(/文本 Embedding/);
           await model.selectOption("dense-1");
           await button("保存知识库",drawer).click();
           await drawer.getByText("文本 Embedding：新增模型",{exact:true}).waitFor();
           await drawer.getByText(/当前有 3 个已发布文件/).waitFor();
           assert.equal(await page.evaluate(()=>fixtureRequests.filter(row=>row.method==="PATCH").length),0,"进入复核不保存");
           await button("仅保存",drawer).click();
           await page.getByText("知识库已保存。",{exact:true}).waitFor();
           await button("编辑",card).click();
           await model.selectOption("dense-2");
           await button("保存知识库",drawer).click();
           await drawer.getByText("文本 Embedding：更换模型",{exact:true}).waitFor();
           await button("保存并重建",drawer).click();
           await page.getByText("知识库已保存，索引重建已排队。",{exact:true}).waitFor();
           const changes=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==="PATCH").map(row=>JSON.parse(row.body)));
           assert.deepEqual(changes.map(({expected_revision,rebuild_indexes,index_models})=>[expected_revision,rebuild_indexes,index_models.embedding]),[
             [1,undefined,"dense-1"],[2,true,"dense-2"],
           ]);
           continue;
         }
         const search=page.getByRole("searchbox",{name:"搜索知识库"});
         await search.fill("不存在的知识库");
         await page.getByText("没有匹配的知识库，请调整筛选条件。").waitFor();
         assert.equal(await page.locator(".management-catalog-add-card").count(),1);
         await button("刷新").click();
         await page.getByText("没有匹配的知识库，请调整筛选条件。").waitFor();
         assert.equal(await search.inputValue(),"不存在的知识库");
         await search.fill("");
         await page.getByLabel("库状态").selectOption("DISABLED");
         assert.equal(await page.locator(".knowledge-base-card").count(),1);
         assert.equal(await page.locator(".knowledge-base-card.is-disabled").count(),1);
         await page.getByLabel("库状态").selectOption("");
         await page.locator(".management-catalog-add-card").click();
         await drawer.getByText("基本信息",{exact:true}).waitFor();
         assert.equal(Math.round((await drawer.boundingBox()).width),820);
         assert.equal(await drawer.getByLabel("OCR 模型（可选）").count(),1);
         assert.equal(await drawer.getByLabel("VLM 模型（可选）").count(),1);
         assert.equal(await drawer.getByLabel("批准时选择解析方案").count(),0,"建库不选择解析方案");
         await button("取消",drawer).click();
         const card=page.locator(".knowledge-base-card").filter({hasText:"产品知识库"});
         const more=button("更多",card);
         await more.click();
         await card.getByRole("menuitem",{name:"删除",exact:true}).click();
         await page.getByText(/历史 Agent 版本引用/).first().waitFor();
         assert.equal(await page.evaluate(()=>fixtureRequests.some(row=>row.method==="DELETE")),false);
         await button("成员",card).click();
         const matrix=drawer.getByRole("table",{name:"知识库权限矩阵"});
         await matrix.waitFor();
         assert.equal(await matrix.getByRole("columnheader",{name:"发布",exact:true}).count(),0);
         await button("可添加",drawer).click();
         await drawer.getByRole("button",{name:/加入 新成员/}).click();
         await button("复核改动",drawer).click();
         await button("确认保存",page.getByRole("alertdialog")).click();
         await page.getByText("成员授权已更新。",{exact:true}).waitFor();
         const grants=await page.evaluate(()=>JSON.parse(fixtureRequests.find(row=>row.method==="PATCH").body).grants);
         assert.deepEqual(grants["new-member"],["READ"]);
         assert.equal(grants["fixture-user"].includes("PUBLISH"),false);
       }
     }
   }
   for(const page of pair){assert.deepEqual(await page.evaluate(()=>fixtureUnexpected),[],name+" 接口必须全部隔离");await page.close();}
 }
 assert.deepEqual(errors,[],"浏览器运行异常");
 writeFileSync(resolve(artifacts,"differences.json"),JSON.stringify(differences,null,2));
 assert.deepEqual(differences,[],"与真实 Ammolite 页面存在规格差异；详见 "+resolve(artifacts,"differences.json"));
 console.log("通过：真实 Vue 页面与 KBot 空/有数据、4 种宽度及独立页面交互。截图："+artifacts);
}finally{await browser?.close();await vite.close();await new Promise(done=>server.close(done));}
