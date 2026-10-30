// 以 Ammolite 的真实 Vue 组件为参照，比较 KBot 正式页面；所有请求使用隔离数据。
// node tests/contract/aiops_page_alignment.mjs
import assert from "node:assert/strict";
import { readFileSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const portal = resolve(process.env.AMMOLITE_ROOT || resolve(root, "../ammolite_cube"), "apps/portal-web");
const require = createRequire(resolve(portal, "package.json"));
const { chromium } = require("@playwright/test");
const { createServer: createViteServer } = await import(require.resolve("vite"));
const vue = (await import(require.resolve("@vitejs/plugin-vue"))).default;
const manifest = JSON.parse(readFileSync(resolve(root, "ui/aiops/reference-pages.json"), "utf8"));
const pages = (process.env.AIOPS_PAGES || "business-systems,business-system-workspace,knowledge-bases,responsibility-groups,knowledge-tags,agents,cases,report-templates,monitoring,operations-knowledge,chat,dashboard,work-item-detail,situations,inspections,reports,recovery-drills,inspection-plans,inspection-plan-detail,runs,run-detail,report-detail,diagnostic-sources,diagnostic-source-detail").split(',');
const eligiblePages = Object.entries(manifest.pages).filter(([, entry]) => entry.sources.length).map(([page]) => page);
for (const page of pages) assert.ok(eligiblePages.includes(page), `${page} 尚无可验收的生产页面基准`);
if (!process.env.AIOPS_PAGES) assert.deepEqual([...pages].sort(), eligiblePages.sort(), "默认整套回归不得遗漏已具备基准的页面");
const artifacts = mkdtempSync(resolve(tmpdir(), "aiops-page-alignment-"));
const permissions = ["aiops:use", "aiops:target_manage", "aiops:agent_manage", "aiops:knowledge_manage", "aiops:plan_manage", "aiops:diagnostic_source_manage", "aiops:work_item_manage", "aiops:work_item_handle"];
const fixture = {
  permissions,
  system: { business_system_id: "system-1", system_code: "MES", display_name: "MES 生产系统", environment: "PROD", importance_level: 3, status: "ENABLED", protection_status: "CONFIGURED", resource_count: 2, resource_type_summary: [{ resource_type_key: "DATABASE", display_name: "数据库", count: 2 }], architecture_labels: ["RAC"], responsibility_group_id: "group-1", row_version: 1, updated_at: "2026-10-30T03:00:00Z" },
  group: { responsibility_group_id: "group-1", name: "生产数据库组", description: "核心业务数据库保障", status: "ACTIVE", lead_user_id: "user-1", active_member_count: 2, row_version: 1, members: [{user_id:'user-1',member_role:'LEAD',status:'ACTIVE',active_work_item_count:0},{user_id:'user-2',member_role:'MEMBER',status:'ACTIVE',active_work_item_count:1}] },
  knowledgeBase: { knowledge_base_id: "kb-1", display_name: "Oracle 运维手册", description: "生产数据库操作规程", status: "ACTIVE", usage: { assets: 3, agent_versions: 1 }, row_version: 1, updated_at: "2026-10-30T03:00:00Z" },
  tag: { tag_id: "tag-1", category: "TECHNOLOGY_STACK", display_name: "Oracle Database", description: "Oracle 数据库技术", sort_order: 1, status: "ACTIVE", row_version: 1 },
  agent: { agent_id: "agent-1", display_name: "统一运维助手", description: "生产数据库诊断", status: "ACTIVE", version_no: 1, allowed_business_system_ids: ["system-1"], capabilities: ["MONITORING_QUERY", "RESOURCE_CATALOG_QUERY"], knowledge_policy: { enabled: false }, execution_policy: { monitoring_enabled: true, readonly_database_enabled: false, change_database_enabled: false }, auto_trigger_policy: { enabled: false }, models: {}, updated_at: "2026-10-30T03:00:00Z" },
  diagnosisCase: { asset_id: 'case-1', display_name: '归档空间告警诊断', case_kind: 'DIAGNOSTIC_REFERENCE', problem_class: '归档空间', database_type: 'ORACLE', database_version: '19c', topology: 'RAC', source_report_id: 'report-1', status: 'PUBLISHED', version_no: 1, security_level: 1, updated_at: '2026-10-30T03:00:00Z' },
  inspectionTemplate: { inspection_template_id: 'template-1', display_name: '生产数据库巡检', status: 'ACTIVE', version_no: 1, selected_check_ids: ['availability'], row_version: 1, updated_at: '2026-10-30T03:00:00Z' },
  inspectionCatalog: {catalog_hash:'fixture',groups:[{group_id:'health',display_name:'可用性与容量',checks:[{check_id:'availability',display_name:'数据库可用性',availability:'READY',trend_required:false,finding_types:[]},{check_id:'capacity-trend',display_name:'容量增长趋势',availability:'READY',trend_required:true,finding_types:[]},{check_id:'planned-check',display_name:'规划检查项',availability:'PLANNED',trend_required:false,finding_types:[]}]}]},
  sessionTemplate: { template_id: 'session-template-1', template_ref: 'fixture-template', display_name: '生产诊断报告', status: 'ACTIVE', version_no: 1, row_version: 1, system_defined: false, is_default: true, usage_count: 2, sections: ['EXECUTIVE_SUMMARY', 'FINDINGS', 'EVIDENCE_BOUNDARY'] },
  knowledgeAsset: { asset_id: 'asset-1', knowledge_base_id: 'kb-1', display_name: 'Oracle 日常运维手册', asset_kind: 'MANUAL', security_level: 1, status: 'PUBLISHED', created_at: '2026-10-30T03:00:00Z', updated_at: '2026-10-30T03:00:00Z' },
  monitoringSource: { source_id: 'source-1', display_name: '生产 Prometheus', source_type: 'PROMETHEUS', monitoring_readiness: 'READY', diagnostic_readiness: 'READY', capability_gaps: [] },
  source:{source_id:'source-1',display_name:'生产 Prometheus',source_type:'PROMETHEUS',endpoint:'https://monitor.example.com',status:'ENABLED',connectivity_status:'CONNECTED',config:{},row_version:3,updated_at:'2026-10-30T03:00:00Z',webhook_configured:false,webhook_secret:{configured:false}},
  monitoringInstance: { instance_id: 'instance-1', display_name: '生产 Oracle', resource_name: 'MES 数据库', business_system_id: 'system-1', business_system_name: 'MES 生产系统', resource_id: 'resource-1', db_type: 'ORACLE', monitoring_readiness: 'READY', diagnostic_readiness: 'READY', capability_gaps: [] },
  monitoringPanels: [
    { panel_id:'cpu', title:'CPU 使用率', description:'当前主机 CPU 趋势', visualization:'TIME_SERIES', metric_code:'host.cpu.utilization', semantic_role:'CPU_UTILIZATION', unit:'percent', value_kind:'GAUGE', thresholds:{warning_value:70,critical_value:90,direction:'ABOVE'}, quality:'GOOD', summary:{}, series:[{source_id:'source-1',source_display_name:'生产 Prometheus',source_type:'PROMETHEUS',instance_id:'instance-1',instance_display_name:'MES 数据库',series_key:'cpu-1',dimensions:{},coverage_ratio:1,baseline:{period:'1w',start:'2026-10-23T03:00:00Z',end:'2026-10-30T03:00:00Z',average:30,sample_count:100,coverage_ratio:1},points:[{observed_at:'2026-10-30T02:58:00Z',value:30,quality:'GOOD'},{observed_at:'2026-10-30T02:59:00Z',value:null,quality:'INVALID'},{observed_at:'2026-10-30T03:00:00Z',value:35,quality:'GOOD'}]}] },
    { panel_id:'lag', title:'复制延迟', description:'复制数据延迟趋势', visualization:'TIME_SERIES', metric_code:'db.replication.lag', semantic_role:'REPLICATION_LAG', unit:'seconds', value_kind:'GAUGE', quality:'NO_DATA', summary:{}, series:[] },
  ],
  conversation: {conversation_id:'session-1',agent_id:'agent-1',title:'MES 系统诊断',source_type:'MANUAL',status:'ACTIVE',updated_at:'2026-10-30T03:00:00Z'},
  target: {target_id:'target-1',display_name:'MES 数据库',db_type:'ORACLE',status:'ENABLED',responsibility_group_id:'group-1',connectivity_status:'CONNECTED'},
  workItem: {work_item_id:'item-1',item_key:'OPS-001',target_id:'target-1',target_name:'MES 数据库',title:'归档空间风险核对',summary:'核对归档空间与业务窗口。',work_type:'PROBLEM_INVESTIGATION',status:'IN_PROGRESS',severity:'HIGH',priority:'P2',responsibility_group_id:'group-1',responsibility_group_name:'生产数据库组',assignee_user_id:'user-1',assigned_at:'2026-10-30T03:00:00Z',resolution_due_at:'2026-10-31T03:00:00Z',occurrence_count:2,row_version:1},
  dashboard: {generated_at:'2026-10-30T03:00:00Z',summary:{target_count:1,attention_target_count:1,healthy_count:0,critical_alert_count:1,high_alert_count:0,unreachable_count:0,stale_or_unknown_count:0,failed_automation_count:0,open_alert_count:1},automation:{run_total:5,run_succeeded:3,run_partial:1,run_failed:1,inspection_total:3,inspection_succeeded:3,inspection_partial:0,inspection_failed:0},targets:[{target_id:'target-1',display_name:'MES 数据库',db_type:'ORACLE',environment:'PROD',importance_level:3,health:'WARNING',attention_reason:'归档空间余量不足',open_alert_count:1,data_freshness:'CURRENT'}]},
  report: {report_id:'report-1',report_key:'REPORT-001',report_type:'PERFORMANCE',report_version:2,template_ref:'fixture-template',template_version:'1',title:'生产数据库诊断报告',status:'READY',summary:'已完成归档空间与运行状态核对。',published_at:'2026-10-30T03:00:00Z'},
  situation: {situation_id:'situation-1',target_id:'target-1',situation_type:'DATABASE_ALERT',title:'生产数据库会话等待告警',summary:'监控信号显示写入等待，需要核验事务状态。',status:'OPEN',severity:'HIGH',event_count:2,first_observed_at:'2026-10-30T02:00:00Z',last_observed_at:'2026-10-30T03:00:00Z'},
  inspectionFire: {fire_id:'fire-1',plan_id:'plan-1',plan_display_name:'生产数据库日常巡检',scheduled_at:'2026-10-30T02:00:00Z',completed_at:'2026-10-30T03:00:00Z',status:'PARTIAL',target_count:2,completed_count:1,failed_count:1},
  inspectionReport: {title:'生产数据库巡检报告',summary:'当前归档空间需关注，部分监控证据尚未覆盖。',status:'PARTIAL',scope:{health_status:'HIGH',evidence_coverage_percent:75,evidence_covered_count:3,evidence_total_count:4,monitoring_sample_coverage_percent:80,monitoring_low_coverage_metric_count:1,monitoring_metric_count:5,evidence_gap_count:1,check_status_counts:{CLEAR:1,FINDING:1,GAP:1}},facts:[{kind:'agent_health_inspection',markdown:'当前归档空间需要关注。请按维护窗口核对。'},{kind:'inspection_check',title:'数据库可用性',check_status:'CLEAR',summary:'实例正常运行。',observations:[{instance:'MES1',status:'OPEN'}]},{kind:'inspection_check',title:'归档空间',check_status:'FINDING',summary:'归档空间余量不足。',findings:[{finding_id:'archive-1',finding_type:'CAPACITY_RISK',severity:'HIGH',confirmation:'CONFIRMED',impact:'归档空间可能不足。',fields:{available_percent:12}}],observations:[{diskgroup:'FRA',available_percent:12}]},{kind:'inspection_check',title:'监控采样',check_status:'GAP',summary:'当前监控样本覆盖不完整。'}],recommendations:['由数据库负责人核对空间增长与保留策略。'],gaps:[{category:'MONITORING',code:'SAMPLE_GAP',detail:'缺少部分采样窗口。'}]},
  inspectionPlan:{plan_id:'plan-1',display_name:'生产数据库日常巡检',agent_id:'agent-1',schedule_type:'DAILY',cron_expression:'0 2 * * *',timezone:'Asia/Shanghai',status:'ACTIVE',inspection_template_id:'template-1',inspection_template_version_id:'template-version-1',inspection_template_name:'生产巡检',inspection_template_version:1,selected_check_ids:['availability'],timeout_seconds:1800,overlap_policy:'SKIP',misfire_policy:'LATEST_ONLY',row_version:3,next_run_at:'2026-10-31T02:00:00Z',updated_at:'2026-10-30T03:00:00Z'},
  run:{ops_run_id:'run-1',target_id:'target-1',workflow_kind:'CHAT_TURN',trigger_type:'CHAT',status:'COMPLETED',created_at:'2026-10-30T02:00:00Z',completed_at:'2026-10-30T03:00:00Z',row_version:4,readonly_lifecycle:{items:[{lease_id:'lease-1',status:'RELEASED',observation:{db_type:'ORACLE',method:'PROTOCOL',observed_at:'2026-10-30T03:00:00Z'}}],truncated:false}},
  recoveryProfile:{recovery_profile_id:'profile-1',version_no:2,rpo_seconds:300,rto_seconds:1800,drill_required:true,required_drill_interval_days:30,required_assurance_level:'DATABASE_OPEN',required_backup_source_types:['ORACLE_RMAN']},
  recoveryDrill:{drill_id:'drill-1',assurance_level:'DATABASE_OPEN',backup_source_type:'ORACLE_RMAN',environment:'ISOLATED',result:'PASS',status:'SUBMITTED',simulated_failure_at:'2026-10-30T02:00:00Z',achieved_rpo_seconds:300,achieved_rto_seconds:1200,source_trust_level:'USER_PROVIDED',row_version:2},
  chatTurns: [{
    turn_id:'results-turn-1',turn_no:1,conversation_id:'session-1',status:'PARTIAL',
    messages:[{message_id:'message-1',role:'USER',message_type:'USER_MESSAGE',created_at:'2026-10-30T03:00:00Z',payload:{text:'检查生产数据库运行状态。'}}],
    events:[
      {sequence_no:1,event_type:'planning.started',occurred_at:'2026-10-30T03:00:01Z',payload:{public_summary:'正在理解问题并选择只读取证路径',public_sections:[{title:'已确认范围',items:['MES 生产系统','仅查询当前运行状态']}]}},
      {sequence_no:2,event_type:'tool.completed',occurred_at:'2026-10-30T03:00:03Z',payload:{public_summary:'只读查询已完成',public_sections:[{title:'采集结果',items:['发现一条等待会话','保留完整查询边界']}]}},
    ],
    investigation_plan:{task_frame:{problem_statement:'定位会话等待原因'},hypotheses:[{hypothesis_id:'h1',statement:'存在长时间锁等待',rationale:'用户报告写入变慢'}],actions:[{action_id:'a1',question:'检查会话与锁等待',tool_id:'db.sessions',execution_mode:'READONLY',status:'COMPLETED',sampling:{count:2,interval_seconds:10},pagination:{page_size:100,max_pages:2}}]},
    answer_blocks:[
      {block_no:1,block_type:'TABLE',payload:{presentation_role:'PRIMARY',result_kind:'CURRENT_SQL_LIST',title:'当前 SQL',measurement_note:'本轮当前活动快照',row_count:2,captured_at:'2026-10-30T03:00:03Z',columns:[{name:'sql_id',label:'SQL ID'},{name:'sql_text',label:'SQL 文本'},{name:'elapsed_seconds',label:'耗时（秒）',logical_type:'NUMBER'}],rows:[['sql-001','select status from production_resource where resource_id = :id',12],['sql-002','select 1 from dual',null]],row_selections:['selection-1','selection-2']}},
      {block_no:2,block_type:'FINDING_CARDS',payload:{findings:[{finding_id:'finding-1',finding_type:'LOCK_WAIT',severity:'HIGH',confirmation:'CONFIRMED',impact:'当前写入被阻塞，需要确认事务边界。',fields:{waiting_session_id:123,blocking_session_id:456,wait_seconds:120,is_holder:false}}],empty_reasons:[],gaps:[{detail:'缺少历史等待趋势'}]}},
      {block_no:3,block_type:'ANALYSIS_MARKDOWN',payload:{markdown:'当前存在会话等待。请先核对事务状态。'},citations:[{turn_evidence_id:'evidence-1',label:'会话取证',evidence:{tool_id:'db.sessions',captured_at:'2026-10-30T03:00:03Z',row_count:1,truncated:false,columns:['SID','等待秒数','是否持有者'],rows:[[123,120,false]]}}]},
      {block_no:4,block_type:'SOLUTION_MARKDOWN',payload:{markdown:'由业务负责人确认后，按受控流程处理。'},citations:[{turn_evidence_id:'evidence-1',label:'会话取证'},{turn_evidence_id:'evidence-2',label:'只读查询边界'}]},
      {block_no:5,block_type:'TABLE',payload:{presentation_role:'SUPPORTING',result_kind:'TOP_SQL_LIST',measurement_semantics:'CURRENT_ACTIVITY',row_count:1,columns:[{name:'sql_id',label:'SQL ID'}],rows:[['sql-001']]}},
      {block_no:6,block_type:'FACT_CONFIRMATION',payload:{db_type:'ORACLE',target_id:'target-1',storage_media_options:[{value:'ASM',fact_type:'ASM_DISKGROUP',label:'ASM 磁盘组',description:'确认数据库实际使用的 ASM 磁盘组。',key_field:'diskgroup',field:{name:'diskgroup',label:'磁盘组名称',placeholder:'例如 DATA',required:true}},{value:'FILESYSTEM',fact_type:'FILESYSTEM_DIRECTORY',label:'文件系统',description:'确认实际的数据文件目录。',key_field:'directory',field:{name:'directory',label:'数据目录',placeholder:'例如 /data/oracle',required:true}}]}},
      {block_no:7,block_type:'FACT_CONFIRMATION',payload:{db_type:'POSTGRESQL',target_id:'target-1'}},
      {block_no:8,block_type:'IMPLEMENTATION_RUNBOOK',payload:{title:'生产数据库实施文档',profile:'oracle-storage',schema_version:'IMPLEMENTATION_RUNBOOK.v1',status:'READY',generation:{starter_id:'database.storage',supplied_parameters:{}},phases:[{phase_id:'p1',title:'核对存储配置',objective:'检查实施环境与权限',steps:[{step_id:'s1',title:'检查当前状态',rationale:'实施前确认目标数据库与业务范围',applicability:'REQUIRED',required_inputs:['目标数据库','维护窗口'],commands:[{command_id:'manual-1',executor:'MANUAL',title:'确认维护窗口',content:'联系业务负责人核对窗口。'},{command_id:'cmd-1',executor:'SQL',title:'查询实例',content:'select instance_name from v$instance;',run_as:'只读账号',risk_level:'LOW',expected_result:['返回目标实例名称']}],verification_commands:[{command_id:'verify-1',executor:'SQL',title:'验证连接',content:'select 1 from dual;'}],rollback:[],risks:['请勿在未审核时执行变更。']}]}],resolved_parameters:[{key:'db_type',label:'数据库类型',value:'ORACLE',status:'CONFIRMED',source:'TARGET'}],missing_facts:[{fact_key:'maintenance_window',reason:'实施前核对维护窗口'}],artifacts:[{artifact_id:'script-1'}]}},
      {block_no:9,block_type:'CHART',payload:{schema_version:'CHART_SPEC.v1',chart_type:'LINE',intent:'TREND',title:'数据库 CPU 趋势',layout:'OVERLAY',x_axis:{axis_type:'TIME',label:'采集时间'},y_axis:{axis_type:'VALUE',unit:'%'},source_ids:['evidence-1'],series:[{series_key:'cpu',name:'CPU 使用率',unit:'%',points:[{x:'2026-10-30T02:59:00Z',y:30},{x:'2026-10-30T03:00:00Z',y:40}]}],metadata:{coverage_ratio:0.8,coverage_warning:'当前监控采样覆盖不完整。'}}},
      {block_no:10,block_type:'PROPOSAL_SUMMARY',payload:{proposal_id:'proposal-1',action_template_id:'storage.review',execution_mode:'MANUAL_ONLY',status:'ADVISORY_READY',risk_level:'LOW',rationale:'请先核实运行边界。',impact:'仅核对目标数据库',lock_impact:'无锁变更',parameters:{target:'生产数据库'},command_preview:'select 1 from dual;',verification_plan:['核对目标实例','保留只读验证结果']}},
      {block_no:11,block_type:'REPORT_ARTIFACT_LINKS',payload:{reports:[{artifact_id:'report-1',title:'数据库原生报告',file_name:'native.html'},{title:'没有可下载文件的记录'}]}},
      {block_no:12,block_type:'TABLE',payload:{presentation_role:'PRIMARY',result_kind:'SQL_EXECUTION_PLAN',title:'执行计划',measurement_note:'优化器成本不是毫秒',row_count:2,columns:[{name:'node_id',label:'节点'},{name:'operation',label:'执行操作'}],rows:[[0,'SELECT STATEMENT'],[1,'TABLE ACCESS FULL']],plan_origin:'DATABASE_CURSOR',plan_graph:{schema_version:'SQL_PLAN_GRAPH.v1',nodes:[{node_id:0,parent_id:null,operation:'SELECT STATEMENT',metrics:{optimizer_cost:10}},{node_id:1,parent_id:0,operation:'TABLE ACCESS FULL',schema_name:'MES',relation_name:'ORDERS',metrics:{estimated_rows:100}}]}}},
    ],evidence_gaps:[{code:'HISTORY_NOT_AVAILABLE',detail:'当前没有历史采样证据。',retryable:false}],
  }],
};

function installFixture(data) {
  const streaming=location.search.includes('stream=1');
  const interaction = () => new URLSearchParams(location.search).get('interaction');
  globalThis.fixtureStreamRunning=streaming;
  const chatTurns=()=>data.chatTurns.map(turn=>{
    if (interaction()==='clarification') return {...turn,status:'WAITING_USER',answer_blocks:[],events:[],investigation_plan:null,evidence_gaps:[],capability_plan:{requested_capabilities:['ACTIVE_DIAGNOSIS']},scope_resolution:{status:'AMBIGUOUS',scope_type:'BUSINESS_SYSTEM',business_systems:[],resources:[]},clarification:{clarification_id:'clarification-1',question:'请选择需要诊断的业务系统',allow_free_text:true,candidates:[{candidate_id:'system-1',label:'MES 生产系统',description:'生产数据库与应用'},{candidate_id:'system-2',label:'ERP 业务系统',description:'财务与采购业务'}]}};
    if (interaction()==='evidence') return {...turn,status:'WAITING_USER',answer_blocks:[],investigation_plan:null,evidence_gaps:[],events:[{sequence_no:1,event_type:location.pathname.startsWith('/ui/')?'diagnostic.input_required':'evidence.requested',payload:{hitl_id:'evidence-1',evidence_request_id:'evidence-1'}}]};
    if (interaction()==='proposal') return {...turn,answer_blocks:[turn.answer_blocks.find(block=>block.block_type==='PROPOSAL_SUMMARY'),{block_no:20,block_type:'PROPOSAL_SUMMARY',payload:{proposal_id:'proposal-2',action_template_id:'storage.review',execution_mode:'APPROVAL_REQUIRED',status:'PENDING_APPROVAL',row_version:3,proposal_hash:'frozen-proposal-hash',rationale:'请逐条核对待审批变更。',risk_level:'HIGH'}}],events:[],investigation_plan:null,evidence_gaps:[]};
    return location.search.includes('empty=1')?{...turn,answer_blocks:[],events:[],investigation_plan:null,evidence_gaps:[]}:globalThis.fixtureStreamRunning?{...turn,status:'RUNNING',answer_blocks:[],events:[],investigation_plan:null,evidence_gaps:[]}:turn;
  });
  globalThis.fixtureOpenStream=()=>new Response(new ReadableStream({start(controller){
    globalThis.fixtureSendStream=event=>{
      if(event.event_type==='done') {globalThis.fixtureStreamRunning=false;controller.close();return;}
      controller.enqueue(new TextEncoder().encode(`id: ${event.sequence_no}\nevent: ${event.event_type}\ndata: ${JSON.stringify(event)}\n\n`));
    };
  }}),{headers:{'Content-Type':'text/event-stream'}});
  globalThis.fixtureRequests = [];
  globalThis.fixtureFailures = [];
  globalThis.fixtureRequest = async (path, init = {}) => {
    fixtureRequests.push({ path, method: init.method || "GET", body:init.body, headers:init.headers });
    if (init.method && init.method !== "GET") {
      if (globalThis.fixtureHoldWrites) await new Promise(resolve=>{globalThis.fixtureReleaseWrite=()=>{delete globalThis.fixtureReleaseWrite;resolve();};});
      if(globalThis.fixtureWebhookWrites) {
        if(init.method==='PATCH' && path.endsWith('/diagnostic-sources/source-1')) return {...data.source,source_type:fixtureSourceType,row_version:4,webhook_secret:{configured:true}};
        if(path.endsWith('/diagnostic-sources/source-1/webhook-key:rotate')) {
          if(globalThis.fixtureWebhookKeyFailure) {fixtureWebhookKeyFailure=false;throw new Error('隔离测试：暂时无法生成 Key');}
          return {source_id:'source-1',webhook_key:'fixture-webhook-key',created_at:'2026-10-30T03:00:00Z'};
        }
      }
      throw new Error("页面对照不执行写操作");
    }
    const suffix = new URL(path, "http://fixture").pathname.replace(/\/domains\/[^/]+\/(?:ops|manage)/, "");
    const rows = (item) => location.search.includes("empty=1") ? [] : [item];
    if (suffix.endsWith('/diagnostic-sources')) return {items:rows({...data.source,source_type:globalThis.fixtureSourceType || 'PROMETHEUS',status:globalThis.fixtureSourceStatus || 'ENABLED'})};
    if (suffix.endsWith('/diagnostic-sources/source-1')) return {...data.source,source_type:globalThis.fixtureSourceType || 'PROMETHEUS',status:globalThis.fixtureSourceStatus || 'ENABLED',webhook_configured:Boolean(globalThis.fixtureSourceWebhook),config:globalThis.fixtureSourceType==='LOKI'?{tenant_id:'fixture-tenant'}:{}};
    if (suffix.endsWith("/access")) return { permissions: [...data.permissions,...(interaction()==='proposal'?['aiops:proposal:approve']:[])], domain_id: "domain-1", user_id: "user-1", display_name: "测试用户" };
    if (suffix.endsWith('/hitl/evidence-1')) {
      const queries=[{query_id:'sessions',purpose:'核对当前会话与等待状态',sql_text:'select sid, status from v$session;',parameters:{},expected_columns:['SID','STATUS'],max_rows:100,timeout_seconds:30,timeout_hint_seconds:30}];
      return location.pathname.startsWith('/ui/')?{hitl_id:'evidence-1',hitl_type:'MANUAL_DIAGNOSTIC_SQL',status:'PENDING',row_version:2,expires_at:'2026-10-30T12:00:00Z',request:{instructions:['请由 DBA 提供当前会话证据。'],queries}}:{evidence_request_id:'evidence-1',status:'PENDING',expires_at:'2026-10-30T12:00:00Z',prompt_text:'请由 DBA 提供当前会话证据。',queries};
    }
    if (/\/proposals\/proposal-[12]\/effect$/.test(suffix)) return {frozen_plan:globalThis.fixtureEffectFrozen?{business_effect:{signals:[{key:'primary',role:'PRIMARY'}]}}:{},result:globalThis.fixtureEffectFrozen?{measured_effect:'IMPROVED',guardrails_passed:true,attribution_strength:'LIMITED',measurements:{primary:{baseline:100,after:80,unit:'ms',passed:true}},gap_codes:['CONTROL_NOT_AVAILABLE']}:{},evidence_candidates:[{artifact_id:'workload-1',target_id:'target-1',schema_version:'AIOPS_WORKLOAD_REPORT.v2',period_start:'2026-10-30T02:00:00Z',period_end:'2026-10-30T03:00:00Z',status:'READY',metrics:[{metric:'latency',label:'数据库响应耗时',statement_identity:{}}]}]};
    if (/\/proposals\/proposal-[12]$/.test(suffix)) return {target_id:'target-1',row_version:3};
    if (suffix.endsWith("/business-systems")) return { items: rows(data.system) };
    if (suffix.endsWith('/business-systems/system-1') && location.pathname.includes('business-system-workspace')) {
      const mode=new URLSearchParams(location.search).get('interaction'),rac=mode==='rac';
      return {...data.system,status:mode?'DISABLED':'ENABLED',description:'生产核心系统的数据库运维。',resources:(rac?[1,2,3]:[1,2]).map(index=>({system_resource_id:`system-resource-${index}`,architecture_layer_key:'DATA',status:'ACTIVE',display_order:index-1,row_version:1,resource:{resource_id:`resource-${index}`,row_version:1,display_name:rac&&index<3?`MES 主库 · 节点 ${index}`:index===1?'MES 数据库':'MES 备库',owner_group_id:'group-1',resource_type_key:'DATABASE',database:{target_id:`target-${index}`,db_type:'ORACLE',version_code:'19c',environment:'PROD',deployment_type:rac&&index<3?'RAC':'SINGLE_INSTANCE',rac_node_number:rac&&index<3?index:null,db_role:index===(rac?3:2)?'STANDBY':'PRIMARY',connectivity_status:'CONNECTED',monitoring_ready:true,oracle_container_scope:'NON_CDB',endpoint:{host:`db${index}.internal`,port:1521,service:'MES',tls_enabled:false},diagnostic_username:'readonly_user'}}})),relations:rac?[{relation_id:'rac-1',row_version:1,relation_type_key:'ORACLE_RAC',display_name:'MES 主库',origin:'DISCOVERY',attributes:{auto_inferred:true,inference_rule:'RAC_DEPLOYMENT'},participants:[1,2].map(index=>({system_resource_id:`system-resource-${index}`,participant_role:'MEMBER',display_order:index-1}))}]:[]};
    }
    if (suffix.endsWith('/business-systems/system-1')) return {...data.system,resources:[{status:'ACTIVE',resource:{resource_id:'resource-1',display_name:'MES 数据库',resource_type_key:'DATABASE',database:{db_type:'ORACLE',version_code:'19c'}}},{status:'ACTIVE',resource:{resource_id:'resource-2',display_name:'生产 PostgreSQL',resource_type_key:'DATABASE',database:{db_type:'POSTGRESQL',version_code:'16'}}}]};
    if (/\/managed-resources\/resource-[12]\/protection-profile$/.test(suffix)) return globalThis.fixtureRecoveryNoProfile?null:{...data.recoveryProfile,required_backup_source_types:[suffix.includes('resource-2')?'POSTGRESQL_BASEBACKUP':'ORACLE_RMAN']};
    if (/\/managed-resources\/resource-[12]\/recovery-drills$/.test(suffix)) return {items:rows({...data.recoveryDrill,status:globalThis.fixtureRecoveryStatus || 'SUBMITTED'})};
    if (suffix.endsWith('/targets')) return {items:location.pathname.includes('reports')&&!location.search.includes('empty=1')?[data.target,{...data.target,target_id:'postgres-1',display_name:'生产 PostgreSQL',db_type:'POSTGRESQL'},{...data.target,target_id:'mysql-1',display_name:'生产 MySQL',db_type:'MYSQL'}]:rows(data.target)};
    if (suffix.endsWith('/dashboard')) return location.search.includes('empty=1')?{generated_at:data.dashboard.generated_at,summary:{},automation:{},targets:[]}:data.dashboard;
    if (suffix.endsWith('/reports')) return {items:rows(data.report)};
    if (suffix.endsWith('/targets/target-1')) return location.pathname.includes('business-system-workspace')?{...data.target,row_version:3,workload_snapshot_policy:{enabled:false,collection_mode:'BOUNDED_CANDIDATES',visible_statement_limit:500,interval_minutes:5,retention_days:30,top_statement_limit:100,round_timeout_seconds:30}}:data.target;
    if (suffix.endsWith('/reports/report-1')) return {...data.report,target_id:'target-1',period_start:'2026-10-30T02:00:00Z',period_end:'2026-10-30T03:00:00Z',report_type:new URLSearchParams(location.search).get('reportMode')==='COMPARISON'?'COMPARISON':'PERFORMANCE'};
    if (suffix.endsWith('/operations-knowledge/reports/report-1/case-eligibility')) return {eligible:true,reasons:[],existing_asset_id:location.search.includes('empty=1')?'case-1':null};
    if (suffix.endsWith('/reports/report-1/presentation')) {
      const mode=new URLSearchParams(location.search).get('reportMode') || 'STANDARD';
      const header={title:'生产数据库巡检报告',report_time:'2026-10-30T03:00:00Z',observation_start:'2026-10-30T02:00:00Z',observation_end:'2026-10-30T03:00:00Z',target_name:'target-1',target_id:'target-1',environment:'生产',database_type:'ORACLE',database_version:'19c',report_version:2,report_number:'REPORT-001',responsibility_domain:'生产数据库组'};
      const boundary={methodology:'只代表实际观测窗口；无数据不代表异常。',sources:rows('冻结证据 evidence-1')};
      const metrics=rows({metric_key:'cpu',metric_code:'host.cpu',semantic_role:'CPU',display_name:'CPU 使用率',dimension_label:'生产主机',current_display:'35%',baseline_display:'28%',deviation_percent:25,baseline_status:'AVAILABLE',coverage_percent:95,trend_points:[{date:'2026-10-28',value:30,baseline_index:95},{date:'2026-10-29',value:35,baseline_index:125}],monthly_average_display:'30%',p95_display:'40%',minimum_display:'20%',maximum_display:'45%',change_percent:12.5,effective_days:28,expected_days:30,daily_points:[{date:'2026-10-01',value:20},{date:'2026-10-30',value:30}],first_display:'20%',last_display:'30%',net_change_display:'10%'});
      const common={title:'生产数据库诊断报告',status:'PARTIAL',status_display:'证据不完整',template:{display_name:'生产诊断报告',version:'1'},sections:location.search.includes('empty=1')?[]:[{kind:'EXECUTIVE_SUMMARY',items:['当前**归档空间**需要关注。']},{kind:'FINDINGS',human_edited:true,items:['已确认一项容量风险。']},{kind:'RECOMMENDATIONS',items:['请安排复核窗口。']},{kind:'EVIDENCE_BOUNDARY',items:['缺少部分历史采样，不能外推。']},{kind:'EVIDENCE_APPENDIX',items:['证据编号 evidence-1']},{kind:'EXTRA',display_name:'补充核验',items:[]}],leadership_briefing:location.search.includes('empty=1')?null:{risk_level:'HIGH',business_impact:['业务可能受影响。'],risks:['空间余量不足。'],recommendations:['安排维护窗口。']},report:{scope:{report_origin:'EXTERNAL_IMPORTED',database_type:'ORACLE',health_status:'HIGH',evidence_coverage_percent:75,evidence_covered_count:3,evidence_total_count:4,monitoring_sample_coverage_percent:80,monitoring_low_coverage_metric_count:1,monitoring_metric_count:5,evidence_gap_count:1,inspection_coverage:'保留证据边界。',monitoring_profile:{display_name:'生产监控',version:'1'},raw_report_artifact_id:'artifact-1',conversation:location.search.includes('empty=1')?null:{conversation_id:'session-1',title:'生产诊断会话',turn_count:2}}}};
      if(mode==='DAILY') common.daily_inspection={schema_version:'DAILY_INSPECTION_PRESENTATION.v1',header,conclusion:{text:'归档空间需要关注，采样覆盖仍有缺口。',health_status_display:'高风险',evidence_coverage_percent:75,monitoring_coverage_percent:70,attention_count:1,gap_count:1},baseline_metrics:metrics,capacity_rows:metrics,findings:rows({finding_id:'F1',title:'归档空间风险',status:'PARTIAL',severity:'HIGH',summary:'需要确认保留策略。'}),finding_notes:rows('未采样时点不外推。'),recommendations:rows('由责任人核对保留周期。'),gaps:rows('部分历史指标缺失。'),evidence_boundary:boundary};
      if(mode==='MONTHLY') common.monthly_inspection={schema_version:'MONTHLY_INSPECTION_PRESENTATION.v1',header:{...header,scope_type:'BUSINESS_SYSTEM',scope_name:'MES 生产系统',target_count:2,database_types:['ORACLE','POSTGRESQL'],period_start:'2026-09-01T00:00:00Z',period_end:'2026-10-01T00:00:00Z'},conclusion:{text:'本月部分空间风险仍需跟踪。',issue_count:1},coverage:{completed_run_count:28,metric_day_coverage_percent:95},period_comparison:{status:'AVAILABLE',status_display:'可比较',new_issues:[{}],persistent_issues:[],resolved_issues:[]},metrics,capacity_rows:metrics,database_checks:{status_counts:{CLEAR:3,GAP:1}},issues:rows({finding_id:'F1',title:'归档空间风险',summary:'仍需跟踪空间余量。',severity:'HIGH',affected_days:3,occurrence_count:5,first_seen:'2026-09-01',last_seen:'2026-09-30'}),recommendations:rows('持续观察空间趋势。'),gaps:rows('部分历史指标缺失。'),extensions:rows({extension_key:'SLA',status:'RESERVED'}),evidence_boundary:boundary};
      return common;
    }
    if (suffix.endsWith('/work-items')) return {items:rows(data.workItem),queue_summary:{my_open:location.search.includes('empty=1')?0:1,unassigned:0,urgent:location.search.includes('empty=1')?0:1,overdue:0,pending_verification:0},next_cursor:null,has_more:false};
    if (suffix.endsWith('/work-items/item-1')) {
      const detail=location.pathname.includes('work-item-detail') || location.pathname.includes('/work-items/');
      const empty=globalThis.fixtureProcessingEmpty || location.search.includes('empty=1');
      return {...data.workItem,phase:'DIAGNOSIS',first_observed_at:'2026-10-30T02:00:00Z',last_observed_at:'2026-10-30T03:00:00Z',...(detail&&!empty?{agent_id:'agent-1',agent_name:data.agent.display_name,completed_at:'2026-10-30T04:00:00Z',completed_by:'user-1',resolution_code:'OBSERVED_STABLE',completion_note:'已核对业务窗口，继续观察。'}:{}),links:detail&&!empty?[{resource_kind:'RUN',resource_id:'run-1',role:'DIAGNOSIS',label:'只读诊断',status:'COMPLETED'},{resource_kind:'REPORT',resource_id:'report-1',role:'EVIDENCE',label:'诊断报告'},{resource_kind:'SITUATION',resource_id:'situation-1',role:'SOURCE',label:'告警情境'},{resource_kind:'PROPOSAL',resource_id:'proposal-1',role:'REMEDIATION',status:'APPROVED'}]:[],activities:detail&&!empty?[{activity_id:'activity-1',activity_type:'CLAIMED',actor_id:'user-1',from_status:'OPEN',to_status:'IN_PROGRESS',created_at:'2026-10-30T03:00:00Z'},{activity_id:'activity-2',activity_type:'CREATED',actor_id:'user-1',created_at:'2026-10-30T02:00:00Z'}]:[],occurrences:empty?[]:[{occurrence_id:'occurrence-1',ops_run_id:'run-1',finding_type:'LOCK_WAIT',confirmation:'CONFIRMED',severity:'HIGH',observed_at:'2026-10-30T03:00:00Z',finding:{impact:'当前写入被阻塞，需要确认事务边界。',object_ref:{object_name:'生产事务'}},evidence_refs:['evidence-1']}]};
    }
    if (suffix.endsWith('/situations')) return {items:rows(data.situation)};
    if (suffix.endsWith('/inspection-fires')) return {items:rows(data.inspectionFire)};
    if (suffix.endsWith('/inspection-plans')) return {items:rows({...data.inspectionPlan,status:globalThis.fixturePlanStatus || 'ACTIVE'})};
    if (suffix.endsWith('/inspection-plans/plan-1')) return {...data.inspectionPlan,schedule_type:globalThis.fixturePlanType || 'DAILY',cron_expression:globalThis.fixturePlanCron || '0 2 * * *'};
    if (suffix.endsWith('/runs')) return {items:rows(data.run)};
    if (suffix.endsWith('/inspection-fires/fire-1')) return {...data.inspectionFire,runs:globalThis.fixtureInspectionMode==='NO_RUN'?[]:[{run_id:'run-1',target_id:'target-1',status:['RUNNING','FAILED'].includes(globalThis.fixtureInspectionMode)?fixtureInspectionMode:'COMPLETED',report_status:['RUNNING','FAILED'].includes(globalThis.fixtureInspectionMode)?'': 'PARTIAL',evidence_coverage_percent:['RUNNING','FAILED'].includes(globalThis.fixtureInspectionMode)?null:75,evidence_covered_count:3,evidence_total_count:4,tool_execution_coverage_percent:100,monitoring_sample_coverage_percent:80,evidence_gap_count:1},{run_id:'run-2',target_id:'target-2',status:'FAILED',error_message:'目标暂时无法连接'}]};
    if (suffix.endsWith('/runs/run-2')) return {ops_run_id:'run-2',target_id:'target-2',status:'FAILED'};
    if (suffix.endsWith('/runs/run-2/result')) {const error=new (globalThis.fixtureApiErrorClass || Error)('目标暂时无法连接');error.status=404;throw error;}
    if (suffix.endsWith('/situations/situation-1')) return {...data.situation,run_ids:globalThis.fixtureSituationState==='NO_RUN'?[]:['run-1'],monitoring_sources:[{diagnostic_source_id:'source-1',display_name:'生产 Prometheus',source_type:'PROMETHEUS',event_count:2,latest_severity:'HIGH',latest_status:'OPEN',last_observed_at:data.situation.last_observed_at}],signal_events:[{signal_event_id:'signal-1',event_class:'LOCK_WAIT',severity:'HIGH',normalized_status:'OPEN',summary:'当前写入事务等待超过阈值。',occurred_at:data.situation.last_observed_at}]};
    if (suffix.endsWith('/runs/run-1')) return {...data.run,...(globalThis.fixtureRunMode==='FOLLOWUP'?{workflow_kind:'EVIDENCE_FOLLOWUP',status:'RUNNING',completed_at:null}:{}),...(globalThis.fixtureRunMode==='RUNNING'||globalThis.fixtureRunMode==='FAILED'?{status:fixtureRunMode,completed_at:null}:{}),...(globalThis.fixtureRunMode==='TERMINATION_UNKNOWN'?{readonly_lifecycle:{items:[{lease_id:'lease-1',status:'TERMINATION_UNKNOWN',observation:null}],truncated:true}}:{}),...(globalThis.fixtureInspectionMode && ['RUNNING','FAILED'].includes(fixtureInspectionMode)?{status:fixtureInspectionMode}:globalThis.fixtureSituationState && globalThis.fixtureSituationState!=='NO_RUN'?{status:globalThis.fixtureSituationState}:{})};
    if (suffix.endsWith('/runs/run-1/result')) {
      if(['RUNNING','FAILED','NO_RESULT'].includes(globalThis.fixtureRunMode)) {const error=new (globalThis.fixtureApiErrorClass || Error)('尚未形成结果');error.status=404;throw error;}
      if(globalThis.fixtureRunMode==='FOLLOWUP') return {payload:{observation_series:{completed_count:1,requested:{count:3,interval_seconds:30}},tool_outcomes:[{step_id:'step-1',observation:{captured_at:'2026-10-30T03:00:00Z',columns:[{name:'SQL_ID'},{name:'ROWS'}],rows:[['123456',null],['long-result-value-for-overflow-check',{'elapsed':300}]],row_count:2,truncated:true}},{step_id:'step-2',observation:{captured_at:'2026-10-30T03:00:30Z',columns:[{name:'SQL_ID'}],rows:[],row_count:0,truncated:false},gap:{code:'FOLLOWUP_PENDING',detail:'待采集'}}]}};
      if (location.pathname.includes('inspections')) {
        if(['RUNNING','FAILED'].includes(globalThis.fixtureInspectionMode)) {const error=new (globalThis.fixtureApiErrorClass || Error)('尚未形成报告');error.status=404;throw error;}
        if(globalThis.fixtureInspectionMode!=='TURN') return {final_artifact:{schema_version:'REPORT_CONTENT.v1'},payload:data.inspectionReport};
      }
      if(['RUNNING','FAILED'].includes(globalThis.fixtureSituationState)) {const error=new (globalThis.fixtureApiErrorClass || Error)('尚未形成结果');error.status=404;throw error;}
      return {final_artifact:{schema_version:'AIOPS_TURN_RESULT.v1'},payload:{blocks:[...data.chatTurns[0].answer_blocks.filter(block=>['FINDING_CARDS','ANALYSIS_MARKDOWN','SOLUTION_MARKDOWN','CHART'].includes(block.block_type)),{block_type:'REPORT_ARTIFACT_LINKS',payload:{reports:[{title:'归档空间诊断报告'}]}},{block_type:'PROPOSAL_SUMMARY',payload:{status:'APPROVED'}},{block_type:'FACT_CONFIRMATION',payload:{instruction:'请核对运维记忆后再确认。'}}]}};
    }
    if (suffix.endsWith("/business-system-drafts")) return { items: [] };
    if (suffix.endsWith("/member-candidates")) return { items: [{ user_id: "user-1", display_name: "测试用户" },{user_id:'user-2',display_name:'值班 DBA'},{user_id:'user-3',display_name:'候选 DBA'}] };
    if (suffix.endsWith("/responsibility-groups")) return { items: rows(data.group) };
    if (suffix.endsWith('/responsibility-groups/group-1')) return {...data.group,members:location.search.includes('empty-members=1')?[]:data.group.members};
    if (suffix.endsWith("/knowledge-bases")) return { items: rows(data.knowledgeBase) };
    if (suffix.endsWith("/operations-knowledge/tags") || suffix.endsWith("/configuration/knowledge-tags")) return { items: rows(data.tag) };
    if (suffix.endsWith("/agents")) {
      const agent=location.pathname.startsWith('/ui/') ? { ...data.agent, ...data.agent.execution_policy } : data.agent;
      return location.pathname.includes('chat')?[agent]:rows(agent);
    }
    if (suffix.endsWith('/conversation-starters')) return {starters:[],business_systems:[],catalog_version:'fixture'};
    if (suffix.endsWith('/conversations')) {
      const query=new URL(path,'http://fixture').searchParams;
      return query.has('situation_id')?(globalThis.fixtureSituationConversation?[data.conversation]:[]):query.has('run_id')?(globalThis.fixtureInspectionConversation?[data.conversation]:[]):[data.conversation];
    }
    if (suffix.endsWith('/conversations/session-1')) return data.conversation;
    if (suffix.endsWith('/conversations/session-1/turns')) {
      const turns=chatTurns();
      return location.pathname.startsWith('/ui/')?turns:{items:turns};
    }
    if (suffix.endsWith('/turns/results-turn-1')) return chatTurns()[0];
    if (suffix.endsWith("/model-catalog") || suffix.endsWith("/agent-model-catalog")) return [];
    if (suffix.endsWith("/agents/boundary-resources")) return { items: [] };
    if (suffix.endsWith("/models")) return { models: {} };
    if (suffix.endsWith('/cases/overview')) return { draft: 0, review_required: 0, published: location.search.includes('empty=1') ? 0 : 1, processing: 0, retired: 0 };
    if (suffix.endsWith('/cases')) return { items: rows(data.diagnosisCase) };
    if (suffix.endsWith('/inspection-check-catalog')) return data.inspectionCatalog;
    if (suffix.endsWith('/inspection-templates')) return rows(data.inspectionTemplate);
    if (suffix.endsWith('/inspection-templates/template-1')) return data.inspectionTemplate;
    if (suffix.endsWith('/session-report-templates')) return rows(data.sessionTemplate);
    if (suffix.endsWith('/session-report-templates/session-template-1')) return {...data.sessionTemplate,definition:{sections:['EXECUTIVE_SUMMARY','RECOMMENDATIONS','TREND','FINDINGS','EVIDENCE_BOUNDARY'].map(kind=>({kind}))}};
    if (suffix.endsWith('/session-report-templates/session-template-1/versions')) return [{template_version_id:'version-1',version_no:1,current:true,created_at:'2026-10-30T03:00:00Z',content_hash:'fixture-template-content-hash',definition:{sections:['EXECUTIVE_SUMMARY','FINDINGS','EVIDENCE_BOUNDARY'].map(kind=>({kind}))}}];
    if (suffix.endsWith('/assets')) return { items: rows(data.knowledgeAsset) };
    if (suffix.endsWith('/monitoring/sources')) return rows(data.monitoringSource);
    if (suffix.endsWith('/instances')) return rows(data.monitoringInstance);
    if (suffix.endsWith('/monitoring/profiles')) return [{ profile_id: 'profile-1', display_name: '数据库总览', version: '1.0.0', metric_codes: [], is_default: true }];
    if (suffix.endsWith('/monitoring/view')) return { source: data.monitoringSource, profile: { profile_id: 'profile-1', display_name: '数据库总览', version: '1.0.0' }, generated_at: '2026-10-30T03:00:00Z', window: { start: '2026-10-30T02:00:00Z', end: '2026-10-30T03:00:00Z' }, instances: [{ ...data.monitoringInstance, status: 'AVAILABLE' }], panels: location.search.includes('panels=1') ? data.monitoringPanels : [], gaps: [], partial: false };
    fixtureFailures.push(path);
    throw new Error(`未定义的对照请求：${path}`);
  };
  globalThis.KBotAIOpsAuth = { load: () => ({ access_token: "isolated-fixture" }), request: fixtureRequest, uuid: () => crypto.randomUUID(), uuid7: () => crypto.randomUUID() };
  globalThis.fixtureDownload=async(path)=>{
    fixtureRequests.push({path,method:'DOWNLOAD'});
    await new Promise(resolve=>{globalThis.fixtureReleaseDownload=resolve;});
    throw new Error('页面对照不下载真实业务文件');
  };
  globalThis.KBotAIOpsAuth.download=fixtureDownload;
  globalThis.KBotAIOpsAuth.stream=async(path,onEvent)=>new Promise(resolve=>{
    globalThis.fixtureSendStream=event=>{
      onEvent({event:event.event_type,data:event,id:String(event.sequence_no)});
      if(event.event_type==='done') {globalThis.fixtureStreamRunning=false;resolve();}
    };
  });
}

const httpModule = `export class ApiError extends Error {}\nglobalThis.fixtureApiErrorClass=ApiError;\nexport const businessFetch = (path, options={}) => globalThis.fixtureRequest(path, options.init || {});\nexport const apiFetch = (path, init={}) => globalThis.fixtureRequest(path, init);\nexport const businessBlobFetch = path => globalThis.fixtureDownload(path); export const businessEventStream = () => globalThis.fixtureOpenStream();`;
const entry = `
import {createApp, computed, nextTick, h, Fragment} from 'vue';
import {createRouter, createWebHistory, RouterView} from 'vue-router';
import ConfirmDialog from '/src/shared/ui/ConfirmDialog.vue';
import {businessAppPermissionsKey, businessAppPrincipalKey} from '/src/app/businessAppContext.ts';
import '/src/shared/ui/styles.css';
const page = new URLSearchParams(location.search).get('page');
const pages = {${pages.map((name) => `${JSON.stringify(name)}: () => import('/src/features/aiops/${manifest.pages[name].sources[0]}')`).join(",")}};
const component = (await pages[page]()).default;
if(page==='chat') globalThis.fixtureECharts=await import('echarts/core');
document.querySelector('#reference').classList.toggle('is-viewport-contained',['monitoring','chat','business-system-workspace'].includes(page));
const router = createRouter({history:createWebHistory(),routes:[{path:page==='business-system-workspace'?'/apps/aiops/:domainCode/business-system-workspace/:businessSystemId':page==='work-item-detail'?'/apps/aiops/:domainCode/work-items/:workItemId':page==='report-detail'?'/apps/aiops/:domainCode/reports/:reportId':'/apps/aiops/:domainCode/:pathMatch(.*)*',component:['diagnostic-sources','diagnostic-source-detail'].includes(page)?{render:()=>h(component,{section:'diagnostic-sources'})}:['inspections','reports','inspection-plans','inspection-plan-detail'].includes(page)?{render:()=>h(component,{kind:page==='reports'?'reports':'inspections'})}:component,meta:{title:page==='knowledge-tags'?'AIOps 知识标签':${JSON.stringify({dashboard:'运维总览',"business-systems":"业务系统","knowledge-bases":"运维知识库","responsibility-groups":"责任组",agents:"Agent 管理", cases:"历史诊断案例", "report-templates":"模板中心",monitoring:"实时监控","operations-knowledge":"运维知识","work-item-detail":"工作项详情"})}[page]}}]});
if(page==='chat') router.getRoutes()[0].meta.title='Agent 工作台';
if(page==='situations') router.getRoutes()[0].meta.title='告警诊断';
if(['inspections','inspection-plans','inspection-plan-detail'].includes(page)) router.getRoutes()[0].meta.title='日常巡检';
if(page==='reports') router.getRoutes()[0].meta.title='运维报告';
if(page==='recovery-drills') router.getRoutes()[0].meta.title='恢复演练';
if(['runs','run-detail'].includes(page)) router.getRoutes()[0].meta.title='运行记录';
if(page==='report-detail') router.getRoutes()[0].meta.title='报告详情';
if(['diagnostic-sources','diagnostic-source-detail'].includes(page)) router.getRoutes()[0].meta.title='监控源';
const app=createApp({render:()=>h(Fragment,null,[h(RouterView),h(ConfirmDialog)])}); app.use(router);
app.provide(businessAppPermissionsKey,computed(()=>Object.fromEntries([...${JSON.stringify(permissions)},...(location.search.includes('interaction=proposal')?['aiops:change_manage']:[])].map(key=>[key,true]))));
app.provide(businessAppPrincipalKey,computed(()=>({id:'user-1',display_name:'测试用户',username:'fixture'})));
const initialQuery=new URLSearchParams(location.search);
const planQuery=['inspection-plans','inspection-plan-detail'].includes(page)?'&tab=plans':'';
await router.replace('/apps/aiops/fixture/'+(page==='business-system-workspace'?'business-system-workspace/'+(location.search.includes('empty=1')?'new':'system-1'):page==='work-item-detail'?'work-items/item-1':page==='report-detail'?'reports/report-1':page)+'?domainId=domain-1'+(page==='chat'&&!initialQuery.get('sourceType')?'&conversationId=session-1':'')+(location.search.includes('empty=1')?'&empty=1':'')+(location.search.includes('panels=1')?'&panels=1':'')+(location.search.includes('stream=1')?'&stream=1':'')+(initialQuery.get('interaction')?'&interaction='+initialQuery.get('interaction'):'')+(initialQuery.get('sourceType')?'&sourceType='+initialQuery.get('sourceType')+'&sourceId='+initialQuery.get('sourceId'):''));
if(planQuery) await router.replace({query:{...router.currentRoute.value.query,tab:'plans'}});
if(page==='run-detail'&&initialQuery.get('id')) await router.replace({query:{...router.currentRoute.value.query,runId:initialQuery.get('id')}});
if(initialQuery.get('reportMode')) await router.replace({query:{...router.currentRoute.value.query,reportMode:initialQuery.get('reportMode')}});
app.mount('#reference'); await nextTick(); globalThis.referenceMounted=true;`;
const referenceServer = await createViteServer({
  root: portal, configFile: false, cacheDir: resolve(artifacts, "vite-cache"),
  plugins: [vue(), { name: "isolated-aiops-reference", enforce: "pre", resolveId(id) { if (["@/shared/api/http", resolve(portal, "src/shared/api/http"), resolve(portal, "src/shared/api/http.ts")].includes(id)) return "\0fixture-http"; if (id === "/fixture-entry.js") return "\0fixture-entry"; }, load(id) { if (id === "\0fixture-http") return httpModule; if (id === "\0fixture-entry") return entry; }, configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (!req.url.startsWith("/reference.html")) return next();
      res.setHeader("Content-Type", "text/html");
      res.end('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"></head><body><div class="business-app-shell business-app-aiops"><header class="business-app-topbar"></header><div class="business-app-layout"><aside class="business-app-sidebar"></aside><div id="reference" class="business-app-content"></div></div></div><style>*,*::before,*::after{transition:none!important;animation:none!important}</style><script type="module" src="/fixture-entry.js"></script></body></html>');
    });
  } }],
  resolve: { alias: [{ find: "@", replacement: resolve(portal, "src") }] },
  server: { host: "127.0.0.1", port: 0, fs: { allow: [portal] } }, logLevel: "error",
});
const staticServer = createServer((req, res) => {
  try {
    const path = resolve(root, `.${new URL(req.url, "http://fixture").pathname}`);
    if (!path.startsWith(`${root}/ui/`)) { res.writeHead(404).end(); return; }
    const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
    res.setHeader("Content-Type", `${types[extname(path)] || "text/plain"}; charset=utf-8`); res.end(readFileSync(path));
  } catch { res.writeHead(404).end(); }
});
const differences = [];
const errors = [];
async function compareRegions(kbot, reference, selectors, context) {
  await Promise.all([kbot,reference].map(page=>page.mouse.move(0,0)));
  await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{if(document.activeElement?.closest('.management-catalog-card')) document.activeElement.blur();})));
  await Promise.all([kbot,reference].map(page=>page.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))))));
  const measure = page => page.evaluate(selectors => selectors.map(selector => {
    const node = [...document.querySelectorAll(selector)].find(item=>item.getBoundingClientRect().width && item.getBoundingClientRect().height);
    if (!node) return null;
    const box = node.getBoundingClientRect(), css = getComputedStyle(node);
    if (!box.width || !box.height) return null;
    return { x: box.x, y: box.y, width: box.width, height: box.height, fontSize: css.fontSize, fontWeight:css.fontWeight, color:css.color, background: css.backgroundColor, radius: css.borderRadius, border:css.border, padding:css.padding, lineHeight:css.lineHeight, appearance:css.appearance };
  }), selectors);
  const actual = await measure(kbot), expected = await measure(reference);
  for (let index = 0; index < selectors.length; index++) {
    if (!actual[index] || !expected[index]) {
      if (Boolean(actual[index]) !== Boolean(expected[index])) differences.push({ ...context, selector: selectors[index], key: 'presence', actual: Boolean(actual[index]), expected: Boolean(expected[index]) });
      continue;
    }
    for (const key of Object.keys(actual[index])) {
      const equal = typeof actual[index][key] === 'number' ? Math.abs(actual[index][key] - expected[index][key]) <= 1 : actual[index][key] === expected[index][key];
      if (!equal) differences.push({ ...context, selector: selectors[index], key, actual: actual[index][key], expected: expected[index][key] });
    }
  }
}
await referenceServer.listen();
await new Promise((done) => staticServer.listen(0, "127.0.0.1", done));
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ timezoneId: "Asia/Shanghai" });
  context.setDefaultTimeout(15_000);
  await context.addInitScript(installFixture, fixture);
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({contentType:'text/css',body:''}));
  const kbot = await context.newPage(), reference = await context.newPage();
  for (const [name, page] of [["KBot", kbot], ["Ammolite", reference]]) page.on("pageerror", (error) => { errors.push(`${name}: ${error.message}`); console.error(`${name}: ${error.message}`); });
  reference.on("console", (message) => { if (message.type() === "error") console.error(`Ammolite 控制台：${message.text()}`); });
  reference.on("requestfailed", (request) => { if (!/fonts\.(googleapis|gstatic)\.com/.test(request.url())) console.error(`Ammolite 加载失败：${request.url()} ${request.failure()?.errorText}`); });
  await kbot.route("**/aiops-auth.js*", (route) => route.fulfill({ contentType: "text/javascript", body: "" }));
  for (const name of pages) for (const scenario of (name === 'report-detail' ? ['STANDARD','DAILY','MONTHLY','COMPARISON'].flatMap(reportMode=>[{empty:false,reportMode},{empty:true,reportMode}]) : name === 'business-system-workspace' ? [{empty:false},{empty:true},{empty:false,interaction:'offline'},{empty:false,interaction:'rac'}] : name === 'monitoring' ? [{empty:false},{empty:true},{empty:false,panels:true}] : name==='chat'?[{empty:false},{empty:true},{empty:false,stream:true},...['clarification','evidence','proposal'].map(interaction=>({empty:false,interaction})),...['RUN','SITUATION'].map(continuation=>({empty:false,continuation}))]:[{empty:false},{empty:true}])) {
    const {empty, panels = false, stream=false, interaction='',continuation='',reportMode=''} = scenario;
    const params = new URLSearchParams({...(empty?{empty:'1'}:{}),...(panels?{panels:'1'}:{}),...(name==='chat'&&!continuation?{conversation:'session-1'}:{}),...(stream?{stream:'1'}:{}),...(interaction?{interaction}:{}),...(continuation?{targetId:'target-1',sourceType:continuation,sourceId:'source-1'}:{}),...(name==='work-item-detail'?{id:'item-1'}:{}),...(name==='business-system-workspace'?{id:empty?'new':'system-1'}:{}),...(name==='inspection-plan-detail'&&!empty?{id:'plan-1'}:{}),...(name==='run-detail'&&!empty?{id:'run-1'}:{}),...(name==='diagnostic-source-detail'&&!empty?{id:'source-1'}:{}),...(name==='report-detail'?{id:'report-1',reportMode}:{})});
    console.log(`对照：${name} / ${interaction || (stream?'流式执行':empty ? "空数据" : "有数据")}`);
    await Promise.all([
      kbot.goto(`http://127.0.0.1:${staticServer.address().port}/ui/aiops/${name}.html?${params}`),
      reference.goto(`http://127.0.0.1:${referenceServer.httpServer.address().port}/reference.html?page=${name}&${params}`),
    ]);
    await reference.waitForFunction(() => globalThis.referenceMounted);
    await kbot.evaluate(() => KBotAIOpsShell.ready);
    await kbot.addStyleTag({ content: "*,*::before,*::after{transition:none!important;animation:none!important}" });
    if(name==='dashboard') {
      await Promise.all([kbot,reference].map(page=>page.locator('.operational-summary-card').first().waitFor()));
      await kbot.waitForFunction(()=>!document.querySelector('#work-item-body').textContent.includes('正在读取'));
    }
    if(name==='work-item-detail') await Promise.all([kbot,reference].map(page=>page.locator('.work-item-heading').waitFor({state:'visible'})));
    if(name==='situations') await Promise.all([kbot,reference].map(page=>page.locator('.table-card').first().waitFor({state:'visible'})));
    if(name==='inspections') await Promise.all([kbot,reference].map(page=>page.locator('.inspection-summary-grid').waitFor({state:'visible'})));
    if(name==='reports') await Promise.all([kbot,reference].map(page=>page.locator('.table-card').first().waitFor({state:'visible'})));
    if(['runs','run-detail'].includes(name)) await Promise.all([kbot,reference].map(page=>page.locator('.table-card').first().waitFor({state:'visible'})));
    if(name==='report-detail') await Promise.all([kbot,reference].map(page=>page.locator('.aiops-formal-report').waitFor()));
    if(['diagnostic-sources','diagnostic-source-detail'].includes(name)) {
      await Promise.all([kbot,reference].map(page=>page.locator('.management-catalog-add-card').waitFor()));
      if(!empty) await Promise.all([kbot,reference].map(page=>page.locator('.monitor-source-card').waitFor()));
      if(name==='diagnostic-source-detail'&&!empty) {
        await reference.getByRole('button',{name:'编辑',exact:true}).click();
        await Promise.all([kbot,reference].map(page=>page.locator('.aiops-target-detail-drawer').waitFor()));
      }
    }
    if(name==='recovery-drills' && !empty) await Promise.all([kbot,reference].map(page=>page.locator('.management-catalog-card').first().waitFor({state:'visible'})));
    if(['inspection-plans','inspection-plan-detail'].includes(name)) {
      await Promise.all([kbot,reference].map(page=>page.locator('.management-catalog-add-card').waitFor()));
      if(name==='inspection-plan-detail'&&!empty) {await reference.getByRole('button',{name:'编辑',exact:true}).click();await Promise.all([kbot,reference].map(page=>page.locator('.aiops-inspection-plan-drawer').waitFor()));}
    }
    if(name==='inspection-plan-detail'&&!empty) {
      for(const width of [1440,900,420,320]) {
        await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
        await compareRegions(kbot,reference,['.aiops-inspection-plan-drawer','.admin-drawer-header','.admin-drawer-body','.drawer-form-section','.inspection-frequency-options','.inspection-schedule-summary','.inspection-template-summary','.form-grid-two','.admin-drawer-footer'],{page:name,width,state:'deep-link-edit'});
      }
      for(const page of [kbot,reference]) {assert.deepEqual(await page.evaluate(()=>fixtureFailures),[]);assert.equal(await page.getByRole('radio').first().isDisabled(),true);}
      continue;
    }
    if(name==='run-detail'&&!empty) {
      await Promise.all([kbot,reference].map(page=>page.locator('.aiops-turn-result').waitFor()));
      for(const width of [1440,900,420,320]) {
        await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
        await compareRegions(kbot,reference,['.aiops-detail-drawer','.admin-drawer-header','.key-value-card','.key-value-table','.aiops-turn-result','.aiops-turn-result section','.admin-drawer-footer'],{page:name,width,state:'deep-link-run'});
      }
      continue;
    }
    if (name==='chat') {
      if(continuation) {
        await Promise.all([kbot,reference].map(page=>page.locator('.agent-chat-composer-input:not(:disabled)').waitFor()));
        for(const width of [1440,900,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await compareRegions(kbot,reference,['.agent-chat-conversation-head','.aiops-diagnosis-head-context','.aiops-diagnosis-head-context select','.agent-chat-composer','.agent-chat-composer-input'],{page:name,width,state:`continue-${continuation}`});
        }
        for(const page of [kbot,reference]) {
          await page.locator('.agent-chat-composer-input').fill('继续核对已有诊断证据。');
          await page.evaluate(()=>{fixtureHoldWrites=true;});
          await page.locator('.agent-chat-composer-input').press('Enter');
          await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
          const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
          const body=JSON.parse(request.body);
          if(page===kbot) {
            assert.equal(body[continuation==='RUN'?'source_run_id':'source_situation_id'],'source-1');
            assert.equal(body.agent_id,'agent-1');
          } else assert.deepEqual(body.conversation.source,{source_type:continuation,[continuation==='RUN'?'run_id':'situation_id']:'source-1'});
          assert.deepEqual((body.first_turn||body).content,[{content_type:'TEXT',text:'继续核对已有诊断证据。'}]);
          await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
          await page.locator('.agent-chat-composer-input:not(:disabled)').waitFor();
          assert.equal(await page.locator('.agent-chat-composer-input').inputValue(),'继续核对已有诊断证据。');
        }
        continue;
      }
      await Promise.all([kbot,reference].map(page=>page.locator('.aiops-diagnosis-turn').waitFor()));
      assert.deepEqual(await kbot.evaluate(()=>fixtureFailures),[]);
      assert.deepEqual(await reference.evaluate(()=>fixtureFailures),[]);
      if(interaction) {
        const regions = interaction==='clarification' ? ['.aiops-turn-orchestration','.aiops-turn-orchestration > header','.aiops-turn-orchestration dl','.aiops-clarification','.aiops-clarification-candidates','.aiops-clarification-candidates button','.aiops-clarification-free-text','.aiops-clarification-free-text textarea','.aiops-clarification-free-text button'] : interaction==='evidence' ? ['.aiops-evidence-request','.aiops-evidence-request > header','.aiops-evidence-request .table-secondary','.aiops-query-card','.aiops-query-card pre','.aiops-query-card textarea','.aiops-evidence-request footer','.aiops-evidence-request footer button'] : ['.ops-proposal','.ops-proposal > header','.ops-manual-result','.ops-manual-result label','.ops-manual-result select','.ops-manual-result textarea','.ops-manual-result button','.ops-proposal-actions','.effect-verification'];
        await Promise.all([kbot,reference].map(page=>page.locator(regions[0]).first().waitFor()));
        for(const width of [1440,1024,900,640,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;scrollTo(0,0)})));
          await compareRegions(kbot,reference,regions,{page:name,width,state:interaction});
        }
        if(interaction==='clarification' || interaction==='evidence') {
          const input=interaction==='clarification'?'.aiops-clarification textarea':'.aiops-evidence-request textarea';
          const label=interaction==='clarification'?'提交并在当前轮继续':'提交并继续诊断';
          for(const page of [kbot,reference]) {
            assert.equal(await page.getByRole('button',{name:label,exact:true}).isDisabled(),true);
            await page.locator(input).fill(interaction==='clarification'?'MES 生产系统':'SID STATUS\n123 ACTIVE');
            await page.evaluate(()=>{globalThis.fixtureHoldWrites=true;});
            await page.getByRole('button',{name:label,exact:true}).click();
            await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
            assert.equal(await page.getByRole('button',{name:'正在提交…',exact:true}).isDisabled(),true);
          }
          await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;scrollTo(0,0)})));
          await compareRegions(kbot,reference,regions,{page:name,width:320,state:interaction+'-submitting'});
          for(const page of [kbot,reference]) {
            await page.evaluate(()=>{fixtureReleaseWrite();globalThis.fixtureHoldWrites=false;});
            await page.getByRole('button',{name:label,exact:true}).waitFor();
            assert.equal(await page.getByRole('button',{name:label,exact:true}).isEnabled(),true);
            assert.ok(await page.locator(input).inputValue(),'失败不能丢失人工输入');
          }
          const request=await kbot.evaluate(()=>fixtureRequests.filter(item=>item.method==='POST').at(-1));
          if(interaction==='clarification') assert.deepEqual(JSON.parse(request.body),{clarification_id:'clarification-1',idempotency_key:'clarification:clarification-1',free_text:'MES 生产系统'});
          else {
            assert.deepEqual(JSON.parse(request.body),{expected_row_version:2,responses:[{query_id:'sessions',status:'SUCCEEDED',raw_output:'SID STATUS\n123 ACTIVE'}]});
            await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'跳过补证',exact:true}).click()));
            await compareRegions(kbot,reference,['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-message','.confirm-dialog-actions'],{page:name,width:320,state:'skip-evidence'});
            await Promise.all([kbot,reference].map(page=>page.getByRole('alertdialog').getByRole('button',{name:'取消',exact:true}).click()));
            await kbot.getByRole('button',{name:'跳过补证',exact:true}).click();
            await kbot.getByRole('alertdialog').getByRole('button',{name:'跳过补证',exact:true}).click();
            await kbot.waitForFunction(()=>fixtureRequests.some(item=>item.path.endsWith('/skip')));
            const skip=await kbot.evaluate(()=>fixtureRequests.findLast(item=>item.path.endsWith('/skip')));
            assert.equal(skip.headers['If-Match'],'"rv-2"');
            assert.ok(skip.headers['Idempotency-Key']);
          }
        } else {
          await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'拒绝',exact:true}).click()));
          for(const width of [1440,420]) {
            await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
            await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;scrollTo(0,0)})));
            await compareRegions(kbot,reference,['.ops-reject-form','.ops-reject-form textarea','.ops-reject-form .button-row'],{page:name,width,state:'reject-proposal'});
          }
          for(const page of [kbot,reference]) {
            await page.locator('.ops-reject-form textarea').fill('当前维护窗口尚未获批');
            await page.evaluate(()=>{globalThis.fixtureHoldWrites=true;});
            await page.getByRole('button',{name:'确认拒绝',exact:true}).click();
            await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
            assert.equal(await page.getByRole('button',{name:'批准并执行',exact:true}).isDisabled(),true);
            assert.equal(await page.getByRole('button',{name:'回填结果',exact:true}).isDisabled(),true);
            await page.evaluate(()=>{fixtureReleaseWrite();globalThis.fixtureHoldWrites=false;});
            await page.waitForFunction(()=>!document.querySelector('.ops-proposal-actions button').disabled);
          }
          const payload=page=>page.evaluate(()=>JSON.parse(fixtureRequests.filter(item=>item.path.endsWith('/reject')).at(-1).body));
          assert.deepEqual(await payload(kbot),await payload(reference));
          await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'批准并执行',exact:true}).click()));
          await compareRegions(kbot,reference,['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-message','.confirm-dialog-actions'],{page:name,width:420,state:'approve-proposal'});
          await Promise.all([kbot,reference].map(page=>page.getByRole('alertdialog').getByRole('button',{name:'取消',exact:true}).click()));
          for(const page of [kbot,reference]) {
            await page.locator('.ops-manual-result textarea').first().fill('DBA 已核对执行结果');
            await page.evaluate(()=>{globalThis.fixtureHoldWrites=true;});
            await page.getByRole('button',{name:'回填结果',exact:true}).click();
            await page.getByRole('alertdialog').getByRole('button',{name:'回填结果',exact:true}).click();
            await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
            assert.equal(await page.getByRole('button',{name:'批准并执行',exact:true}).isDisabled(),true);
            await page.evaluate(()=>{fixtureReleaseWrite();globalThis.fixtureHoldWrites=false;});
            await page.waitForFunction(()=>!document.querySelector('.ops-manual-result button').disabled);
            const request=await page.evaluate(()=>fixtureRequests.filter(item=>item.path.endsWith('/manual-result')).at(-1));
            const payload=JSON.parse(request.body);
            assert.ok(Number.isFinite(Date.parse(payload.occurred_at)));
            delete payload.occurred_at;
            assert.deepEqual(payload,{expected_row_version:1,status:'EXECUTED',note:'DBA 已核对执行结果',bounded_output:null});
          }
          await Promise.all([kbot,reference].map(page=>page.locator('.effect-verification').first().locator('summary').first().click()));
          await Promise.all([kbot,reference].map(page=>page.locator('.effect-rule').first().waitFor()));
          for(const page of [kbot,reference]) {
            const effect=page.locator('.effect-verification[open]');
            assert.equal(await effect.getByRole('button',{name:'冻结执行前验证计划',exact:true}).isEnabled(),true);
            for(const row of await effect.locator('.effect-rule').all()) {
              await row.locator('select').nth(1).selectOption({label:'数据库响应耗时'});
              if(!(await row.locator('input').last().inputValue())) await row.locator('input').last().fill('50');
            }
            await page.evaluate(()=>{globalThis.fixtureHoldWrites=true;});
            await effect.getByRole('button',{name:'冻结执行前验证计划',exact:true}).click();
            await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
            for(const button of await effect.locator('button').all()) assert.equal(await button.isDisabled(),true,'冻结请求期间不能并发修改验证计划');
            await page.evaluate(()=>{fixtureReleaseWrite();globalThis.fixtureHoldWrites=false;});
            await page.waitForFunction(()=>!document.querySelector('.effect-verification[open] .effect-rule button').disabled);
          }
          const effectPayload=page=>page.evaluate(()=>JSON.parse(fixtureRequests.filter(item=>item.path.endsWith('/effect-plan')).at(-1).body));
          assert.deepEqual(await effectPayload(kbot),await effectPayload(reference),'冻结条件、信号用途和阈值必须与源端一致');
          for(const controlled of [false,true]) {
            if(controlled) await Promise.all([kbot,reference].map(page=>page.getByLabel('预登记匹配对照验证',{exact:false}).first().check()));
            for(const width of [1440,420]) {
              await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
              await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;scrollTo(0,0)})));
              await compareRegions(kbot,reference,['.effect-verification[open]','.effect-rule','.effect-rule input','.effect-rule select','.effect-rule .text-button','.effect-verification[open] fieldset','.effect-verification[open] fieldset button','.effect-verification[open] label','.effect-verification[open] form > .primary-button'],{page:name,width,state:controlled?'effect-controlled':'effect-plan'});
            }
          }
          await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{globalThis.fixtureEffectFrozen=true;})));
          await Promise.all([kbot,reference].map(page=>page.locator('.effect-verification[open]').getByRole('button',{name:'刷新受控证据',exact:true}).click()));
          await Promise.all([kbot,reference].map(page=>page.locator('.effect-verification[open]').getByText('冻结条件详情',{exact:true}).waitFor()));
          for(const width of [1440,420]) {
            await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
            await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;scrollTo(0,0)})));
            await compareRegions(kbot,reference,['.effect-verification[open]','.effect-verification[open] .table-scroll','.effect-verification[open] th','.effect-verification[open] td','.effect-verification[open] form > .primary-button'],{page:name,width,state:'effect-frozen'});
          }
        }
        continue;
      }
      if(stream) {
        await Promise.all([kbot,reference].map(page=>page.waitForFunction(()=>Boolean(globalThis.fixtureSendStream))));
        for(const page of [kbot,reference]) assert.equal(await page.locator('.chat-composer-textarea, textarea[name="message"]').isDisabled(),true);
        const events=[{sequence_no:1,event_type:'investigation.planned',occurred_at:'2026-10-30T03:00:01Z',payload:{public_summary:'已生成只读调查计划',plan:fixture.chatTurns[0].investigation_plan}},fixture.chatTurns[0].events[1],{sequence_no:3,event_type:'answer.delta',payload:{delta:'正在检查只读证据。'}}];
        for(const event of events) await Promise.all([kbot,reference].map(page=>page.evaluate(event=>fixtureSendStream(event),event)));
        await kbot.waitForFunction(()=>document.querySelector('.aiops-chat-message-agent .ops-message-content')?.textContent.trim()==='正在检查只读证据。');
        for(const width of [1440,1024,900,640,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;document.querySelectorAll('.aiops-live-progress-list').forEach(list=>{list.scrollTop=0});scrollTo(0,0)})));
          await compareRegions(kbot,reference,['.aiops-diagnosis-turn','.aiops-turn-marker','.aiops-live-progress','.aiops-live-progress > header','.aiops-live-progress-list','.aiops-inline-plan','.aiops-live-progress-sections','.aiops-chat-message-agent','.aiops-chat-message-agent header','.aiops-chat-message-agent .aiops-chat-bubble'],{page:name,width,state:'live-stream'});
        }
        await Promise.all([kbot,reference].map(page=>page.evaluate(()=>fixtureSendStream({sequence_no:4,event_type:'done',payload:{}}))));
        await Promise.all([kbot,reference].map(page=>page.locator('.ops-finding-row').waitFor()));
        for(const page of [kbot,reference]) assert.equal(await page.locator('.aiops-chat-message-user').count(),1,'恢复最终结果时不能重复问题');
        continue;
      }
      for (const expanded of [false,true]) {
        await Promise.all([kbot,reference].map(page=>page.evaluate(expanded=>document.querySelectorAll('.aiops-diagnosis-turn details').forEach(item=>{item.open=expanded}),expanded)));
        for (const width of [1440,1024,900,640,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;document.querySelectorAll('.aiops-live-progress-list').forEach(list=>{list.scrollTop=0});scrollTo(0,0)})));
          await compareRegions(kbot,reference,['.aiops-diagnosis-turn','.aiops-live-progress','.aiops-live-progress > header','.aiops-live-progress-list','.aiops-live-progress-list > li','.aiops-inline-plan','.aiops-live-progress-sections','.aiops-primary-result','.aiops-primary-result > header','.aiops-primary-result .table-scroll','.aiops-primary-result th','.aiops-primary-result td','.aiops-primary-result .row-action','.aiops-primary-result > footer','.ops-findings','.ops-findings-header','.ops-findings-overview','.ops-finding-row','.ops-finding-row > summary','.ops-finding-title','.ops-finding-detail','.ops-finding-detail .key-value-table','.ops-finding-gaps','.ops-analysis','.ops-analysis .agent-markdown-content','.ops-solution','.aiops-citation-card','.aiops-citation-list','.aiops-citation-evidence','.aiops-citation-table','.aiops-diagnosis-turn > details:last-child','.ops-fact-confirmation','.ops-fact-confirmation-form','.ops-fact-confirmation-form .control-field','.ops-fact-confirmation-form select','.ops-fact-confirmation-form footer','.ops-fact-confirmation-unavailable','.aiops-runbook-card','.aiops-runbook-header','.aiops-runbook-policy','.aiops-runbook-toc','.aiops-runbook-phase','.aiops-runbook-step','.aiops-runbook-manual','.aiops-runbook-command','.aiops-runbook-command pre','.aiops-runbook-command .technical-meta','.aiops-runbook-appendix','.aiops-runbook-card > footer'],{page:name,width,empty,state:expanded?'expanded-results':'results'});
          await compareRegions(kbot,reference,['.aiops-diagnosis-chart','.agent-chart figcaption','.agent-chart-canvas','.agent-chart-warning','.agent-chart .row-action','.ops-proposal','.ops-proposal > header','.ops-proposal-parameters','.ops-proposal > pre','.ops-proposal-status','.aiops-workload-report-actions','.aiops-workload-report-actions button'],{page:name,width,empty,state:expanded?'expanded-results':'results'});
          await compareRegions(kbot,reference,['.aiops-primary-result.is-sql-plan','.sql-plan-viewer','.sql-plan-toolbar','.sql-plan-tabs','.sql-plan-legend','.sql-plan-workspace','.sql-plan-canvas','.sql-plan-graph','.sql-plan-caption'],{page:name,width,empty,state:expanded?'expanded-results':'results'});
          for(const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) if([1440,420].includes(width)) await page.screenshot({path:resolve(artifacts,`${prefix}-chat-results-${empty?'empty':'populated'}-${expanded?'expanded':'collapsed'}-${width}.png`),fullPage:true});
        }
      }
      if(!empty) {
        for(const page of [kbot,reference]) {
          assert.equal(await page.locator('.aiops-citation-number').count(),2,'相同证据只能编号一次');
          assert.equal(await page.locator('.ops-finding-row[open]').count(),1,'发现卡片应支持展开字段');
          assert.ok(!(await page.locator('.aiops-citation-table').first().textContent()).includes('[object Object]'));
        }
        for(const mode of ['raw','node']) {
          await Promise.all([kbot,reference].map(page=>mode==='raw'?page.getByRole('tab',{name:'原始计划',exact:true}).click():page.getByRole('tab',{name:'图形计划',exact:true}).click()));
          if(mode==='node') await Promise.all([kbot,reference].map(page=>page.locator('.sql-plan-graph [data-plan-node="1"]').click()));
          for(const width of [1440,420]) {
            await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
            await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;scrollTo(0,0)})));
            await compareRegions(kbot,reference,['.aiops-primary-result.is-sql-plan','.sql-plan-workspace','.sql-plan-canvas','.sql-plan-inspector','.sql-plan-detail-head','.sql-plan-raw-scroll','.sql-plan-raw th','.sql-plan-raw td'],{page:name,width,state:`sql-plan-${mode}`});
          }
        }
        for (const label of ['ASM 磁盘组','文件系统']) {
          await Promise.all([kbot,reference].map(page=>page.locator('.ops-fact-confirmation-form select').selectOption({label})));
          for(const width of [1440,420]) {
            await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
            await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;scrollTo(0,0)})));
            await compareRegions(kbot,reference,['.ops-fact-confirmation','.ops-fact-confirmation-form','.ops-fact-confirmation-description','.ops-fact-confirmation-form input','.ops-fact-confirmation-form textarea','.ops-fact-confirmation-form footer','.aiops-runbook-card'],{page:name,width,state:`storage-${label}`});
          }
        }
        await Promise.all([kbot,reference].map(page=>page.locator('.ops-fact-confirmation-form input').fill('/data/oracle')));
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'确认补证',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.getByRole('alertdialog').waitFor()));
        await compareRegions(kbot,reference,['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-mark','.confirm-dialog-message','.confirm-dialog-actions'],{page:name,width:420,state:'fact-confirm'});
        await Promise.all([kbot,reference].map(page=>page.getByRole('alertdialog').getByRole('button',{name:'取消',exact:true}).click()));
        for(const page of [kbot,reference]) {
          await page.waitForFunction(()=>Boolean((globalThis.echarts || globalThis.fixtureECharts)?.getInstanceByDom(document.querySelector('.agent-chart-canvas'))));
          const option=await page.evaluate(()=>{const option=(globalThis.echarts || fixtureECharts).getInstanceByDom(document.querySelector('.agent-chart-canvas')).getOption();return {data:option.series[0].data,color:option.color,lineWidth:option.series[0].lineStyle.width,tooltipAppendTo:option.tooltip[0].appendTo};});
          assert.deepEqual(option.data,[['2026-10-30T02:59:00Z',30],['2026-10-30T03:00:00Z',40]]);
          assert.equal(option.lineWidth,2);
          assert.equal(option.tooltipAppendTo,'body');
        }
        await Promise.all([kbot,reference].map(page=>page.locator('.agent-chart').getByRole('button',{name:'查看数据',exact:true}).click()));
        for(const width of [1440,420]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{document.querySelector('.agent-chat-transcript').scrollTop=0;scrollTo(0,0)})));
          await compareRegions(kbot,reference,['.agent-chart','.agent-chart-data','.agent-chart-data table','.agent-chart-data th','.agent-chart-data td'],{page:name,width,state:'chart-data'});
        }
        await Promise.all([kbot,reference].map(page=>page.locator('.ops-fact-confirmation-form textarea').fill('DBA 已核对')));
        await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{globalThis.fixtureHoldWrites=true})));
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'确认补证',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.getByRole('alertdialog').getByRole('button',{name:'确认写入',exact:true}).click()));
        for(const page of [kbot,reference]) {
          await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
          assert.equal(await page.getByRole('button',{name:'确认补证',exact:true}).isDisabled(),true);
          const request=await page.evaluate(()=>fixtureRequests.findLast(item=>item.path.includes('target-facts:confirm')));
          assert.deepEqual(JSON.parse(request.body),{target_id:'target-1',fact_type:'FILESYSTEM_DIRECTORY',fact_key:'/data/oracle',fact_value:{directory:'/data/oracle'},note:'DBA 已核对'});
          await page.evaluate(()=>{fixtureReleaseWrite();globalThis.fixtureHoldWrites=false});
          await page.getByRole('button',{name:'确认补证',exact:true}).waitFor();
          await page.waitForFunction(()=>!document.querySelector('.ops-fact-confirmation-form [type="submit"]').disabled);
          assert.equal(await page.locator('.ops-fact-confirmation-form input').inputValue(),'/data/oracle');
        }
        await Promise.all([kbot,reference].map(page=>page.locator('input[type="file"]').setInputFiles({name:'只读诊断材料.txt',mimeType:'text/plain',buffer:Buffer.from('隔离材料，不上传。')})));
        for(const width of [1440,420]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await compareRegions(kbot,reference,['.attachment-list','.attachment-list span','.attachment-list button','.agent-chat-composer'],{page:name,width,state:'attachment-preview'});
        }
        await Promise.all([kbot,reference].map(page=>page.locator('.attachment-list').getByRole('button',{name:'移除',exact:true}).click()));
        for(const page of [kbot,reference]) assert.equal(await page.locator('.attachment-list button').count(),0);
        for(const label of ['下载 Markdown','数据库原生报告']) {
          for(const page of [kbot,reference]) {
            await page.getByRole('button',{name:label,exact:true}).click();
            await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseDownload));
            for(const button of await page.locator('.aiops-runbook-card footer button,.aiops-workload-report-actions button').all()) assert.equal(await button.isDisabled(),true,'同一工作台不能并发下载报告或实施文档');
            assert.equal(await page.getByRole('button',{name:'下载 Markdown',exact:true}).count(),1,'实施文档下载期间保持源端按钮文案');
            await page.evaluate(()=>{fixtureReleaseDownload();delete globalThis.fixtureReleaseDownload;});
            await page.waitForFunction(()=>![...document.querySelectorAll('.aiops-runbook-card footer button,.aiops-workload-report-actions button')].some(button=>button.disabled));
          }
        }
      }
      continue;
    }
    if(name==='business-system-workspace') {
      for(const page of [kbot,reference]) await page.locator('.business-step input[name="display_name"],.business-step input').first().waitFor();
      for(const width of [1440,1024,900,420]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.system-workspace','.workspace-head','.workspace-head h1','.wizard-steps','.wizard-steps button','.workspace-content','.step-surface','.form-section','.section-copy','.form-grid','.description-field','.workspace-actions'],{page:name,width,empty,state:'business'});}
      for(const page of [kbot,reference]) {if(empty) await page.locator('.business-step input').fill('MES 新业务系统');if(empty||interaction) await page.getByRole('button',{name:'下一步',exact:true}).click();else await page.locator('.wizard-steps button').nth(1).click();}
      for(const width of [1440,1024,900,420]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.composition-layout','.layer-nav','.layer-nav header','.layer-nav button','.resource-panel','.resource-panel > header','.resource-card-grid','.resource-card-shell','.resource-card','.resource-card dl','.resource-add-card'],{page:name,width,empty,state:'composition'});}
      for(const page of [kbot,reference]) await page.getByRole('tab',{name:'接入层',exact:false}).click();
      await compareRegions(kbot,reference,['.composition-layout','.layer-coming-soon'],{page:name,width:420,empty,state:'planned-layer'});
      for(const page of [kbot,reference]) await page.getByRole('tab',{name:'数据层',exact:false}).click();
      if(empty) {
        for(const page of [kbot,reference]) await page.getByRole('button',{name:'＋ 添加资源',exact:true}).click();
        for(const width of [1440,1024,900,420]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.resource-config-drawer','.resource-config-drawer .admin-drawer-header','.resource-drawer-body','.resource-type-field','.resource-type-summary','.resource-config-drawer .admin-drawer-footer'],{page:name,width,empty,state:'resource-catalog'});}
        for(const page of [kbot,reference]) await page.getByRole('button',{name:'配置数据库',exact:true}).click();
      } else for(const page of [kbot,reference]) {await page.locator('.resource-card').first().click();await page.locator('.snapshot-status').waitFor();}
      for(const width of [1440,1024,900,420]) {await Promise.all([kbot,reference].map(async page=>{await page.setViewportSize({width,height:1000});await page.locator('.resource-drawer-body').evaluate(node=>{node.scrollTop=0;});}));await compareRegions(kbot,reference,['.resource-config-drawer','.resource-config-drawer .admin-drawer-header','.resource-drawer-body','.resource-drawer-form','.editor-grid','.database-grid','.connection-layout','.connection-field-group','.connection-row','.readonly-detail-section','.readonly-detail-section dl','.snapshot-settings','.snapshot-body','.snapshot-status','.resource-config-drawer .admin-drawer-footer'],{page:name,width,empty,state:'resource-editor'});}
      if(empty) {
        for(const engine of ['MYSQL','POSTGRESQL','ORACLE']) {
          for(const page of [kbot,reference]) await page.locator('.resource-config-drawer').getByText('数据库引擎',{exact:true}).locator('..').locator('select').selectOption(engine);
          for(const width of [1440,420]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.database-grid','.connection-layout','.connection-endpoint-row','.connection-access-row','.connection-monitoring-row','.resource-drawer-editor-footer'],{page:name,width,empty,state:`engine-${engine}`});}
        }
        for(const page of [kbot,reference]) await page.locator('.resource-config-drawer').getByText('容器范围',{exact:true}).locator('..').locator('select').selectOption('PDB');
        for(const width of [1440,420]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.connection-access-row','.resource-drawer-editor-footer'],{page:name,width,empty,state:'oracle-pdb'});}
        for(const page of [kbot,reference]) await page.locator('.resource-config-drawer').getByText('部署形态',{exact:true}).locator('..').locator('select').selectOption('RAC');
        for(const width of [1440,420]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.editor-grid','.database-grid','.rac-editor','.rac-guidance','.rac-node-list','.rac-node-card','.rac-node-card header','.rac-node-card .editor-grid','.rac-add-node','.resource-drawer-editor-footer'],{page:name,width,empty,state:'new-rac'});}
      }
      if(interaction) {
        for(const page of [kbot,reference]) await page.locator('.snapshot-toggle input').check();
        for(const width of [1440,420]) {await Promise.all([kbot,reference].map(async page=>{await page.setViewportSize({width,height:1000});await page.locator(".resource-drawer-body").evaluate(node=>{node.scrollTop=0;});}));await compareRegions(kbot,reference,['.snapshot-settings','.snapshot-fields','.snapshot-toggle','.snapshot-actions','.snapshot-reload'],{page:name,width,empty,state:'staged-bounded-policy'});}
        for(const page of [kbot,reference]) await page.locator('.snapshot-fields select').selectOption('VISIBLE_SET');
        for(const width of [1440,420]) {await Promise.all([kbot,reference].map(async page=>{await page.setViewportSize({width,height:1000});await page.locator(".resource-drawer-body").evaluate(node=>{node.scrollTop=0;});}));await compareRegions(kbot,reference,['.snapshot-settings','.snapshot-fields','.snapshot-toggle','.snapshot-actions','.snapshot-reload'],{page:name,width,empty,state:'staged-visible-policy'});}
        for(const page of [kbot,reference]) {
          await page.evaluate(()=>{fixtureHoldWrites=true;});
          await page.locator('.snapshot-actions').getByRole('button',{name:'测试连接并刷新能力',exact:true}).click();
          await page.waitForFunction(()=>typeof fixtureReleaseWrite==='function');
          assert.ok(await page.getByRole('button',{name:'保存草稿',exact:true}).isDisabled(),'能力刷新期间不得保存草稿');
          assert.ok(await page.locator('.snapshot-actions button').isDisabled(),'能力刷新期间不得重复提交');
          await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
          await page.locator('.snapshot-error').waitFor();
          assert.equal(await page.locator('.snapshot-toggle input').isChecked(),true,'刷新失败不能丢失暂存策略');
          assert.equal(await page.locator('.snapshot-fields select').inputValue(),'VISIBLE_SET');
          assert.equal(await page.getByRole('button',{name:'保存草稿',exact:true}).isDisabled(),false,'刷新完成后恢复保存草稿');
        }
      }
      for(const page of [kbot,reference]) await page.locator('.resource-config-drawer .admin-drawer-header button').click();
      for(const page of [kbot,reference]) {if(empty||interaction) await page.getByRole('button',{name:'下一步',exact:true}).click();else await page.locator('.wizard-steps button').nth(2).click();}
      for(const width of [1440,1024,900,420]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.topology-step','.topology-layout','.topology-canvas','.topology-heading','.unified-graph','.standalone-strip','.topology-node','.topology-hint','.topology-properties','.topology-properties-body','.topology-properties-body > header','.resource-readonly-note','.resource-relations','.topology-properties-actions'],{page:name,width,empty,state:'topology'});}
      if(interaction) {
        for(const width of [1440,1024,900,420]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          for(const page of [kbot,reference]) {await page.locator('.graph-edge-label').waitFor();assert.equal(await page.locator('.topology-node').count(),interaction==='rac'?3:2,'同一数据库在统一画布中只能出现一次');}
          await compareRegions(kbot,reference,['.graph-stage','.graph-node-grid','.graph-node','.graph-node.cluster','.graph-node header','.graph-edge-label','.cluster-members'],{page:name,width,empty,state:`related-${interaction}`});
          const paths=await Promise.all([kbot,reference].map(page=>page.locator('.graph-edge > path').evaluateAll(nodes=>nodes.map(node=>node.getAttribute('d').match(/-?\d+(?:\.\d+)?/g).map(Number)))));
          assert.equal(paths[0].length,paths[1].length);
          for(let i=0;i<paths[0].length;i++) for(let j=0;j<paths[0][i].length;j++) assert.ok(Math.abs(paths[0][i][j]-paths[1][i][j])<=1,`${width}px 拓扑曲线坐标应与参照相同`);
        }
        for(const page of [kbot,reference]) {await page.locator('.graph-edge-label').dispatchEvent('click');await page.locator('.relation-sentence').waitFor();}
        await compareRegions(kbot,reference,['.topology-properties','.topology-properties-body','.inferred-relation-note','.relation-sentence','.relation-sentence label','.relation-editor-help'],{page:name,width:420,empty,state:'inferred-relation'});
        for(const page of [kbot,reference]) await page.locator('.relation-sentence select').nth(1).selectOption('OGG');
        await compareRegions(kbot,reference,['.relation-override-note','.relation-sentence','.topology-properties-actions'],{page:name,width:420,empty,state:'manual-override'});
        for(const page of [kbot,reference]) {await page.getByRole('button',{name:'恢复自动推导',exact:true}).click();await page.getByRole('button',{name:'删除关系',exact:true}).click();await page.getByRole('button',{name:'恢复自动识别',exact:true}).click();}
        for(const page of [kbot,reference]) {await page.locator('.graph-edge-label').waitFor();assert.equal(await page.locator('.graph-edge').count(),1);}
        if(interaction==='rac') {
          for(const page of [kbot,reference]) await page.locator('.graph-node.cluster header button').click();
          await compareRegions(kbot,reference,['.inferred-relation-note','.relation-readonly-summary','.rac-relation-members','.rac-relation-members article'],{page:name,width:420,empty,state:'fixed-rac'});
          for(const page of [kbot,reference]) assert.equal(await page.getByRole('button',{name:'删除关系',exact:true}).count(),0,'受管 RAC 不能在拓扑中删除');
        }
        for(const page of [kbot,reference]) {await page.evaluate(()=>{fixtureHoldWrites=true;});await page.getByRole('button',{name:'保存草稿',exact:true}).click();await page.waitForFunction(()=>typeof fixtureReleaseWrite==='function');assert.ok(await page.getByRole('button',{name:'保存草稿',exact:true}).isDisabled());}
        for(const page of [kbot,reference]) {
          const call=await page.evaluate(()=>fixtureRequests.find(item=>item.method==='POST'&&item.path.endsWith('/business-system-drafts'))),body=JSON.parse(call.body);
          assert.equal(body.mode,'EDIT');assert.equal(body.business_system_id,'system-1');assert.equal(body.payload.resources[0].workload_snapshot.policy.collection_mode,'VISIBLE_SET');assert.equal(body.payload.resources[0].workload_snapshot.policy.enabled,true);assert.equal(body.payload.relations.filter(item=>item.relation_type_key==='DATA_GUARD').length,1);assert.ok(Array.isArray(body.payload.dismissed_inference_keys));
          await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});await page.getByRole('button',{name:'保存草稿',exact:true}).waitFor();
          await page.waitForFunction(()=>!document.querySelector('.workspace-actions button').disabled);
        }
      }
      if(empty||interaction) {
        for(const page of [kbot,reference]) {await page.locator('.workspace-actions').getByText('返回业务系统',{exact:true}).click();await page.locator('.confirm-dialog').waitFor();}
        for(const width of [1440,420]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-mark','.confirm-dialog-message','.confirm-dialog-actions'],{page:name,width,empty,state:'leave-confirmation'});}
        for(const page of [kbot,reference]) await page.locator('.confirm-dialog').getByRole('button',{name:'取消',exact:true}).click();
      }
      for(const [label,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifacts,`${label}-${name}-${interaction|| (empty?'new':'readonly')}.png`)});
      for(const page of [kbot,reference]) assert.deepEqual(await page.evaluate(()=>fixtureFailures),[],`${name} ${interaction} 存在未定义的数据请求`);
      continue;
    }
    for (const width of (name==='report-detail'?[1440,1024,900,420,320]:[1440,1024,420])) {
      await Promise.all([kbot.setViewportSize({ width, height: 1000 }), reference.setViewportSize({ width, height: 1000 })]);
      await Promise.all([kbot, reference].map((page) => page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))))));
      assert.ok(await reference.evaluate(() => fixtureRequests.length > 0), `${name} 必须读取隔离数据，不能把请求失败的空页面当作参照`);
      for (const page of [kbot, reference]) assert.deepEqual(await page.evaluate(() => fixtureFailures), [], `${name} 存在未定义的数据请求`);
      const selectors = [".app-page-head", ".data-toolbar", ".resource-list-surface:not([hidden])", ".resource-list-surface:not([hidden]) .management-catalog-grid", ".resource-list-surface:not([hidden]) .management-catalog-add-card", ".resource-list-surface:not([hidden]) .management-catalog-card", ".table-card", ".resource-list-header", ".search-field", ".case-overview-grid", ".monitoring-dashboard-shell:not(.is-empty)", ".monitoring-context-rail", ".monitoring-rail-fields", ".monitoring-rail-footer", ".monitoring-panels"];
      const measure = (page) => page.evaluate((selectors) => {
        return selectors.map((selector) => {
          const node = document.querySelector(selector); if (!node) return null;
          const box = node.getBoundingClientRect(), css = getComputedStyle(node);
          if (!box.width || !box.height) return null;
          return { x: box.x, y: box.y, width: box.width, height: box.height, fontSize: css.fontSize, background: css.backgroundColor, radius: css.borderRadius };
        });
      }, selectors);
      const actual = await measure(kbot), expected = await measure(reference);
      if (panels) await compareRegions(kbot, reference, ['.monitoring-panel', '.monitoring-current-value', '.monitoring-baseline', '.monitoring-threshold-rail', '.monitoring-chart-canvas', '.monitoring-no-data'], {page:name,width,empty,state:'charts'});
      if(name==='dashboard') await compareRegions(kbot,reference,['.platform-home-header','.aiops-action-hero','.aiops-work-items-page','.work-item-queue-panel','.work-item-queue-grid','.operational-summary-card','.workbench-content-grid','.work-item-list-panel','.work-item-list-header','.work-item-toolbar','.work-item-list-panel table','.work-item-list-panel th','.work-item-list-panel td','.work-item-row-actions','.workbench-context-column','.workbench-context-card','.runtime-metrics','.attention-targets','.attention-targets li','.automation-metrics','.latest-report-context'],{page:name,width,empty,state:'overview'});
      if(name==='work-item-detail') await compareRegions(kbot,reference,['.platform-home-header','.platform-home-hero','.work-item-heading','.work-item-heading h2','.work-item-heading .status-pill','.work-item-detail-grid','.work-item-main-column','.key-value-card','.key-value-table','.key-value-table th','.key-value-table td','.work-item-record-list','.work-item-record-list article','.work-item-record-list header','.work-item-evidence','.work-item-timeline','.work-item-timeline article','.empty-cell'],{page:name,width,empty,state:'detail'});
      if(name==='situations') await compareRegions(kbot,reference,['.platform-home-header','.platform-home-hero','.table-card','.resource-list-header','.resource-list-tools','.table-scroll','table','th','td','.status-pill'],{page:name,width,empty,state:'situations-list'});
      if(name==='inspections') await compareRegions(kbot,reference,['.platform-home-header','.platform-home-hero','.inspection-tabs','.inspection-tabs button','.inspection-summary-grid','.inspection-summary-grid article','.inspection-summary-grid strong','.table-card','.resource-list-header','.resource-list-tools','.table-scroll','table','th','td','.status-pill'],{page:name,width,empty,state:'inspections-list'});
      if(name==='reports') await compareRegions(kbot,reference,['.platform-home-header','.platform-home-hero','.table-card','.resource-list-header','.resource-list-tools','.resource-list-filter','.resource-list-filter select','.resource-list-count','.table-scroll','table','th','td','.status-pill','.button-row'],{page:name,width,empty,state:'reports-list'});
      if(name==='recovery-drills') await compareRegions(kbot,reference,['.recovery-drills-page','.recovery-hero','.recovery-context-card','.table-card-header','.recovery-context-grid','.recovery-context-grid select','.recovery-policy-summary','.recovery-policy-summary div','.recovery-entry-card','.recovery-entry-form','.recovery-form-note','.recovery-entry-form .control-field','.date-time-picker-control','.recovery-entry-form textarea','.recovery-entry-form > footer','.recovery-history-card','.management-catalog-card','.management-catalog-card > header','.management-catalog-card > dl','.management-catalog-action-bar'],{page:name,width,empty,state:'recovery'});
      if(name==='inspection-plans') await compareRegions(kbot,reference,['.platform-home-header','.platform-home-hero','.inspection-tabs','.inspection-tabs button','.aiops-inspection-plans-page','.aiops-inspection-plans-page > .app-page-head','.notice-warning','.resource-list-surface','.management-catalog-grid','.management-catalog-add-card','.management-catalog-card','.management-catalog-card > header','.management-catalog-card > p','.management-catalog-card > dl','.management-catalog-action-bar'],{page:name,width,empty,state:'plan-list'});
      if(['runs','run-detail'].includes(name)) await compareRegions(kbot,reference,['.platform-home-header','.platform-home-hero','.table-card','.resource-list-header','.resource-list-tools','.table-scroll','table','th','td','.status-pill','.table-primary','.technical-value'],{page:name,width,empty,state:'run-list'});
      if(name==='report-detail') await compareRegions(kbot,reference,['.aiops-report-detail-page','.report-page-actions','.aiops-formal-report','.report-cover','.report-masthead','.report-release-state','.report-cover-heading','.report-cover h1','.report-abstract','.report-cover-folio','.report-leadership-briefing','.report-leadership-grid','.report-metadata','.report-metadata dt','.report-coverage-value','.report-document-body','.report-index','.report-index nav','.report-content','.report-chapter','.report-chapter-header','.report-section','.report-section-content','.report-document-footer','.daily-report-head','.daily-report-title-row','.daily-report-metadata','.daily-report-metadata dd','.daily-conclusion','.daily-conclusion-ledger','.daily-report-section','.daily-deviation-list','.daily-deviation-row','.daily-deviation-track','.daily-trend-grid','.daily-trend-card','.daily-trend-svg','.daily-table-wrap','.daily-report-table','.daily-report-table th','.daily-report-table td','.daily-findings','.daily-finding','.daily-finding > .status-pill','.daily-notes','.daily-evidence','.daily-report-footer','.monthly-comparison-copy'],{page:name,width,empty,state:reportMode});
      if(['diagnostic-sources','diagnostic-source-detail'].includes(name)) await compareRegions(kbot,reference,['.aiops-config-page','.app-page-head','.data-toolbar','.search-field','.resource-list-surface','.management-catalog-grid','.management-catalog-add-card','.monitor-source-card','.monitor-source-card > header','.monitor-source-card-statuses','.monitor-source-card dl','.management-catalog-action-bar','.status-pill',...(name==='diagnostic-source-detail'&&!empty?['.aiops-target-detail-drawer','.admin-drawer-header','.admin-drawer-body','.resource-form','.drawer-form-section','.control-field','.admin-drawer-footer']:[])],{page:name,width,empty,state:'sources'});
      for (let index = 0; index < selectors.length; index++) {
        if (!actual[index] || !expected[index]) {
          if (Boolean(actual[index]) !== Boolean(expected[index])) differences.push({ page: name, width, empty, selector: selectors[index], key: 'presence', actual: Boolean(actual[index]), expected: Boolean(expected[index]) });
          continue;
        }
        for (const key of ["x", "y", "width", "height", "fontSize", "background", "radius"]) {
          const equal = typeof actual[index][key] === "number" ? Math.abs(actual[index][key] - expected[index][key]) <= 1 : actual[index][key] === expected[index][key];
          if (!equal) differences.push({ page: name, width, empty, selector: selectors[index], key, actual: actual[index][key], expected: expected[index][key] });
        }
      }
      if (width === 1440) for (const [prefix, page] of [["kbot", kbot], ["ammolite", reference]]) await page.screenshot({ path: resolve(artifacts, `${prefix}-${name}-${empty ? "empty" : "populated"}.png`), fullPage: true });
    }
    assert.equal(await kbot.locator('.ops-toast').count(), 0, `${name} 初始加载不应出现错误提示`);
    if(name==='diagnostic-sources') {
      for(const type of ['PROMETHEUS','LOKI','ALERTMANAGER','ZABBIX','OEM']) {
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'新增监控源',exact:false}).click()));
        for(const page of [kbot,reference]) await page.locator('.resource-form select').selectOption(type);
        for(const width of [1440,1024,900,420]) {
          await Promise.all([kbot,reference].map(async page=>{await page.setViewportSize({width,height:1000});await page.locator('.aiops-target-detail-drawer').evaluate(node=>{node.scrollTop=0;});}));
          await compareRegions(kbot,reference,['.aiops-target-detail-drawer','.admin-drawer-header','.admin-drawer-body','.resource-form','.drawer-form-section','.control-field','.admin-drawer-footer','input','select','textarea'],{page:name,width,empty,state:`create-${type}`});
        }
        for(const page of [kbot,reference]) {
          const inputs=page.locator('.resource-form input:visible');await inputs.nth(0).fill('新增监控源测试');await inputs.nth(1).fill('https://monitor.example.com');
          if(type==='OEM') {await inputs.nth(2).fill('fixture');await inputs.nth(3).fill('not-a-real-secret');}
          else await inputs.nth(2).fill('not-a-real-token');
          if(type==='LOKI') await inputs.nth(3).fill('tenant-test');
          await page.evaluate(()=>{fixtureHoldWrites=true;});await page.getByRole('button',{name:'创建监控源',exact:true}).click();await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
          const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1)),payload=JSON.parse(request.body);
          assert.ok(request.path.endsWith('/diagnostic-sources'));assert.ok(request.headers['Idempotency-Key']);assert.equal(payload.source_type,type);assert.equal(payload.endpoint,'https://monitor.example.com');assert.equal(payload.display_name,'新增监控源测试');
          assert.deepEqual(payload.credentials,type==='OEM'?{username:'fixture',password:'not-a-real-secret'}:{token:'not-a-real-token'});assert.deepEqual(payload.config,type==='OEM'?{standard_metric_mappings:{}}:type==='LOKI'?{tenant_id:'tenant-test'}:{});
          assert.ok(!payload.webhook_credentials,'创建和 Webhook 凭据生成必须独立');assert.equal(await page.getByRole('button',{name:'正在提交…',exact:true}).isDisabled(),true);
          await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});await page.getByRole('button',{name:'创建监控源',exact:true}).waitFor();assert.equal(await inputs.nth(0).inputValue(),'新增监控源测试');
          await page.locator('.aiops-target-detail-drawer .admin-drawer-header button').click();
        }
      }
      if(!empty) {
        for(const page of [kbot,reference]) await page.getByRole('button',{name:'编辑',exact:true}).click();
        for(const width of [1440,1024,900,420]) {await Promise.all([kbot,reference].map(async page=>{await page.setViewportSize({width,height:1000});await page.locator('.aiops-target-detail-drawer').evaluate(node=>{node.scrollTop=0;});}));await compareRegions(kbot,reference,['.aiops-target-detail-drawer','.admin-drawer-header','.resource-form','.control-field','.admin-drawer-footer'],{page:name,width,empty,state:'edit'});}
        for(const page of [kbot,reference]) {
          assert.equal(await page.locator('.resource-form select').isDisabled(),true);await page.evaluate(()=>{fixtureHoldWrites=true;});await page.getByRole('button',{name:'保存监控源',exact:true}).click();await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
          const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='PATCH').at(-1));assert.equal(request.headers['If-Match'],'"rv-3"');const payload=JSON.parse(request.body);assert.ok(!payload.credentials);assert.ok(!payload.source_type);
          await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});await page.getByRole('button',{name:'保存监控源',exact:true}).waitFor();await page.locator('.aiops-target-detail-drawer .admin-drawer-header button').click();
          await page.locator('.management-catalog-more').click();await page.getByRole('button',{name:'停用',exact:true}).click();
        }
        for(const width of [1440,420]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-message','.confirm-dialog-actions'],{page:name,width,empty,state:'disable-confirm'});}
        for(const page of [kbot,reference]) {
          await page.evaluate(()=>{fixtureHoldWrites=true;});await page.locator('.confirm-dialog').getByRole('button',{name:'停用',exact:true}).click();await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));assert.ok(request.path.endsWith('/diagnostic-sources/source-1/disable'));assert.equal(request.headers['If-Match'],'"rv-3"');assert.ok(request.headers['Idempotency-Key']);await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
        }
        for(const type of ['ALERTMANAGER','ZABBIX']) {
          for(const page of [kbot,reference]) {await page.evaluate(type=>{fixtureSourceType=type;fixtureSourceWebhook=true;fixtureWebhookWrites=true;fixtureWebhookKeyFailure=true;},type);await page.getByRole('button',{name:'刷新目录',exact:true}).click();await page.getByRole('button',{name:'Webhook 接入',exact:true}).click();}
          for(const width of [1440,1024,900,420]) {await Promise.all([kbot,reference].map(async page=>{await page.setViewportSize({width,height:1000});await page.locator('.webhook-credential-drawer').evaluate(node=>{node.scrollTop=0;});}));await compareRegions(kbot,reference,['.webhook-credential-drawer','.webhook-credential-drawer .admin-drawer-header','.webhook-credential-intro','.webhook-credential-purpose','.webhook-config-map','.webhook-credential-intro .notice','.webhook-credential-intro .admin-drawer-footer'],{page:name,width,empty,state:`webhook-${type}`});}
          for(const page of [kbot,reference]) await page.getByRole('button',{name:'重新生成两项凭据',exact:true}).click();
          await compareRegions(kbot,reference,['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-message','.confirm-dialog-actions'],{page:name,width:420,empty,state:'webhook-confirm'});
          for(const page of [kbot,reference]) {
            await page.locator('.confirm-dialog').getByRole('button',{name:'重新生成',exact:true}).click();await page.getByRole('button',{name:'重试生成 Webhook Key',exact:true}).waitFor();
            const requests=await page.evaluate(()=>fixtureRequests),patch=requests.filter(row=>row.method==='PATCH'&&JSON.parse(row.body || '{}').webhook_credentials).at(-1),rotation=requests.filter(row=>row.path.endsWith('/webhook-key:rotate')).at(-1);
            assert.equal(patch.headers['If-Match'],'"rv-3"');assert.match(JSON.parse(patch.body).webhook_credentials.value,/^[a-f0-9]{64}$/);assert.equal(rotation.headers['If-Match'],'"rv-4"');assert.ok(rotation.headers['Idempotency-Key']);
          }
          for(const width of [1440,420]) {await Promise.all([kbot,reference].map(async page=>{await page.setViewportSize({width,height:1000});await page.locator('.webhook-credential-drawer').evaluate(node=>{node.scrollTop=0;});}));await compareRegions(kbot,reference,['.webhook-credential-drawer','.webhook-credential-result','.secret-reveal-panel','.secret-reveal-value','.webhook-credential-result .form-error'],{page:name,width,empty,state:'webhook-key-failure'});}
          for(const page of [kbot,reference]) {
            const before=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='PATCH'&&JSON.parse(row.body || '{}').webhook_credentials).length);
            await page.getByRole('button',{name:'重试生成 Webhook Key',exact:true}).click();await page.getByRole('button',{name:'复制 Key',exact:true}).waitFor();
            const rotations=await page.evaluate(()=>fixtureRequests.filter(row=>row.path.endsWith('/webhook-key:rotate')));assert.equal(rotations.at(-1).headers['Idempotency-Key'],rotations.at(-2).headers['Idempotency-Key']);assert.equal(await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='PATCH'&&JSON.parse(row.body || '{}').webhook_credentials).length),before,'重试 Key 不能再次替换 Secret');
          }
          // INI 键属于部署平台差异；用 KBot 的真实键名比较相同内容的换行和布局。
          if(type==='ALERTMANAGER') await reference.locator('.webhook-config-snippet').evaluate(node=>{node.textContent=node.textContent.replaceAll('ammolite_','kbot_');});
          for(const width of [1440,1024,900,420]) {await Promise.all([kbot,reference].map(async page=>{await page.setViewportSize({width,height:1000});await page.locator('.webhook-credential-drawer').evaluate(node=>{node.scrollTop=0;});}));await compareRegions(kbot,reference,['.webhook-credential-drawer','.webhook-credential-result','.secret-reveal-panel','.secret-reveal-value','.webhook-config-snippet','.webhook-next-steps','.webhook-credential-result .admin-drawer-footer'],{page:name,width,empty,state:'webhook-reveal'});}
          for(const page of [kbot,reference]) {await page.getByRole('button',{name:'关闭 Webhook 接入凭据',exact:true}).click();await page.locator('.secret-reveal-value').first().waitFor({state:'detached'});assert.equal(await page.locator('.secret-reveal-value').count(),0);}
        }
      }
    }
    if(name==='report-detail') {
      for(const label of ['下载 PDF','下载原始 Artifact']) for(const page of [kbot,reference]) {
        await page.getByRole('button',{name:label,exact:true}).click();await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseDownload));
        const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='DOWNLOAD').at(-1));
        assert.ok(request.path.endsWith(label==='下载 PDF'?'/reports/report-1/pdf':'/artifacts/artifact-1/content'));
        assert.equal(await page.getByRole('button',{name:label==='下载 PDF'?'正在生成下载…':'正在下载…',exact:true}).isDisabled(),true);
        await page.evaluate(()=>{fixtureReleaseDownload();delete globalThis.fixtureReleaseDownload;});await page.getByRole('button',{name:label,exact:true}).waitFor();
      }
      if(reportMode==='COMPARISON') for(const page of [kbot,reference]) {
        if(empty) {assert.ok((await page.getByRole('link',{name:'查看诊断案例',exact:true}).getAttribute('href')).includes('assetId=case-1'));continue;}
        await page.evaluate(()=>{fixtureHoldWrites=true;});await page.getByRole('button',{name:'生成待审核效果案例',exact:true}).click();await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
        const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));assert.ok(request.path.endsWith('/operations-knowledge/reports/report-1:extract-case'));
        assert.equal(await page.getByRole('button',{name:'正在提炼…',exact:true}).isDisabled(),true);await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});await page.getByRole('button',{name:'生成待审核效果案例',exact:true}).waitFor();
      }
    }
    const editors = {
      'dashboard': {button:'新建工作项',drawer:'.admin-drawer:has(> .work-item-drawer-form)',regions:['.work-item-drawer-form','.work-item-form-grid','.work-item-drawer-form .control-field']},
      'knowledge-bases': { button: '新建知识库', drawer: '.knowledge-base-drawer', regions: ['.knowledge-base-form', '.drawer-form-section', '.form-grid-two'] },
      'knowledge-tags': { button: '新增标签', drawer: '.knowledge-tag-drawer', regions: ['.knowledge-tag-form', '.drawer-form-section'] },
      'operations-knowledge': { button: '上传资料', drawer: '.knowledge-upload-drawer', regions: ['.knowledge-upload-form', '.drawer-form-section', '.upload-mode-picker', '.upload-drop-zone'] },
      'monitoring': { button: '切换数据库资源', drawer: '.monitoring-target-drawer', regions: ['.monitoring-target-drawer-body', '.monitoring-target-group'] },
      'responsibility-groups': { trigger:'.management-catalog-add-card', drawer:'.responsibility-dialog:not(.responsibility-members-dialog)', regions:['.responsibility-dialog-header','.responsibility-dialog-form','.responsibility-dialog-body','.responsibility-dialog-actions'] },
    };
    const editor = editors[name];
    if (editor && (name !== 'monitoring' || !empty)) {
      for (const width of [1440, 420]) {
        await Promise.all([kbot, reference].map(page => page.setViewportSize({width,height:1000})));
        await Promise.all([kbot, reference].map(page => (editor.trigger?page.locator(editor.trigger):page.getByRole('button', {name:editor.button,exact:true})).click()));
        await Promise.all([kbot, reference].map(page => page.locator(editor.drawer).waitFor({state:'visible'})));
        await Promise.all([kbot, reference].map(page => page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))))));
        await compareRegions(kbot, reference, [editor.drawer, `${editor.drawer} .admin-drawer-header`, `${editor.drawer} .admin-drawer-body`, `${editor.drawer} .admin-drawer-footer`, ...editor.regions], {page:name,width,empty,state:'editor'});
        for (const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) {
          await page.screenshot({path:resolve(artifacts,`${prefix}-${name}-editor-${empty?'empty':'populated'}-${width}.png`)});
          await page.locator(`${editor.drawer} ${name==='responsibility-groups'?'.responsibility-dialog-header':'.admin-drawer-header'} button`).click();
          await page.locator(editor.drawer).waitFor({state:'hidden'});
        }
      }
    }
    if (name==='dashboard' && !empty) {
      // 仅持有并拒绝写请求，验证载荷、提交互斥和失败后的草稿恢复。
      for (const page of [kbot,reference]) {
        await page.getByRole('button',{name:'新建工作项',exact:true}).click();
        await page.locator('.work-item-drawer-form input[maxlength="512"]').fill('  手工登记归档空间核对  ');
        await page.locator('.work-item-drawer-form textarea').fill('  核对业务窗口并保留结果。  ');
        await page.evaluate(()=>{fixtureHoldWrites=true;});
        await page.getByRole('button',{name:'创建工作项',exact:true}).click();
        await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
        const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
        assert.deepEqual(JSON.parse(request.body),{target_id:'target-1',work_type:'PROBLEM_INVESTIGATION',title:'手工登记归档空间核对',summary:'核对业务窗口并保留结果。',severity:'MEDIUM',priority:'P3',assignee_user_id:null});
        assert.ok(request.headers['Idempotency-Key']);
        assert.equal(await page.getByRole('button',{name:'正在创建…',exact:true}).isDisabled(),true);
        await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
        await page.getByRole('button',{name:'创建工作项',exact:true}).waitFor();
        assert.equal(await page.locator('.work-item-drawer-form textarea').inputValue(),'  核对业务窗口并保留结果。  ');
        await page.getByRole('button',{name:'关闭新建工作项',exact:true}).click();
      }
      for (const contextEmpty of [true,false]) {
        await Promise.all([kbot,reference].map(page=>page.evaluate(value=>{fixtureProcessingEmpty=value;},contextEmpty)));
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'继续处理',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.locator('.work-item-processing-modal').waitFor({state:'visible'})));
        await Promise.all([kbot,reference].map(page=>page.waitForFunction(()=>!document.querySelector('.aiops-turn-result').textContent.includes('正在读取'))));
        if(process.env.AIOPS_DEBUG) console.log(await kbot.evaluate(()=>{
          const node=document.querySelector('.work-item-processing-modal');
          const matches=[];
          const walk=rules=>{for(const rule of rules){if(rule.selectorText && node.matches(rule.selectorText))matches.push(rule.cssText);if(rule.cssRules)walk(rule.cssRules);}};
          for(const sheet of document.styleSheets)walk(sheet.cssRules);
          return {computed:{width:getComputedStyle(node).width,minWidth:getComputedStyle(node).minWidth},matches};
        }));
        for(const width of [1440,1024,900,640,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await Promise.all([kbot,reference].map(page=>page.locator('.work-item-processing-modal').evaluate(node=>{node.scrollTop=0;})));
          await compareRegions(kbot,reference,['.work-item-processing-modal','.work-item-processing-modal > header','.work-item-modal-summary','.work-item-modal-summary span','.work-item-processing-context','.work-item-processing-section','.work-item-processing-section > header','.work-item-context-summary','.work-item-context-facts','.work-item-context-evidence','.aiops-turn-result','.ops-turn-result-empty','.ops-findings','.ops-finding-row','.ops-analysis','.ops-analysis .agent-markdown-content','.ops-solution','.ops-html-report-readonly','.ops-proposal-readonly','.ops-fact-confirmation','.agent-chart','.agent-chart-canvas','.work-item-modal-form','.work-item-modal-form label','.work-item-modal-form textarea','.work-item-modal-form footer'],{page:name,width,state:contextEmpty?'processing-empty':'processing-results'});
          if(width===1440 || width===420) for(const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifacts,`${prefix}-processing-${contextEmpty?'empty':'results'}-${width}.png`)});
        }
        for(const page of [kbot,reference]) {
          assert.equal(await page.getByRole('button',{name:'继续深入诊断',exact:true}).count(),contextEmpty?0:1);
          await page.locator('.work-item-modal-form textarea').fill('  已核对，持续观察。  ');
          await page.evaluate(()=>{fixtureHoldWrites=true;});
          await page.getByRole('button',{name:'标记工作完成',exact:true}).click();
          await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
          const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.path.includes(':complete')).at(-1));
          assert.deepEqual(JSON.parse(request.body),{expected_row_version:1,resolution_code:'FIXED',completion_note:'已核对，持续观察。',evidence_refs:[]});
          assert.equal(await page.getByRole('button',{name:'暂不完成',exact:true}).isDisabled(),true);
          assert.equal(await page.getByRole('button',{name:'关闭处理窗口',exact:true}).isDisabled(),true);
          await page.keyboard.press('Escape');
          assert.equal(await page.locator('.work-item-processing-modal').isVisible(),true);
          await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
          await page.getByRole('button',{name:'标记工作完成',exact:true}).waitFor();
          assert.equal(await page.locator('.work-item-modal-form textarea').inputValue(),'  已核对，持续观察。  ');
          await page.getByRole('button',{name:'关闭处理窗口',exact:true}).click();
        }
      }
      await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'分派',exact:true}).click()));
      for(const width of [1440,420,320]) {
        await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
        await compareRegions(kbot,reference,['.work-item-assignment-modal','.work-item-assignment-modal > header','.work-item-assignment-modal .work-item-modal-form','.work-item-assignment-modal label','.work-item-assignment-modal input','.work-item-assignment-modal select','.work-item-assignment-modal footer'],{page:name,width,state:'assignment'});
      }
      for(const page of [kbot,reference]) {
        assert.equal(await page.getByRole('button',{name:'确认分派',exact:true}).isDisabled(),true);
        await page.locator('.work-item-assignment-modal select').selectOption('user-2');
        await page.evaluate(()=>{fixtureHoldWrites=true;});
        await page.getByRole('button',{name:'确认分派',exact:true}).click();
        await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
        const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='PATCH').at(-1));
        assert.deepEqual(JSON.parse(request.body),{expected_row_version:1,responsibility_group_id:'group-1',assignee_user_id:'user-2'});
        assert.equal(await page.getByRole('button',{name:'关闭分派窗口',exact:true}).isDisabled(),true);
        await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
        await page.getByRole('button',{name:'确认分派',exact:true}).waitFor();
        assert.equal(await page.locator('.work-item-assignment-modal select').inputValue(),'user-2');
        await page.getByRole('button',{name:'关闭分派窗口',exact:true}).click();
      }
      for(const page of [kbot,reference]) {
        await page.getByRole('button',{name:'仅看超时',exact:true}).click();
        await page.getByRole('combobox',{name:'按优先级筛选',exact:true}).selectOption('P2');
        await page.waitForFunction(()=>fixtureRequests.some(row=>row.path.includes('/work-items?') && row.path.includes('priority=P2') && row.path.includes('overdue=true')));
      }
    }
    if(name==='situations' && !empty) {
      for(const runState of ['COMPLETED','NO_RUN','RUNNING','FAILED']) {
        await Promise.all([kbot,reference].map(page=>page.evaluate(value=>{fixtureSituationState=value;fixtureRequests=[];},runState)));
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'查看诊断',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.locator('.aiops-detail-drawer').waitFor({state:'visible'})));
        await Promise.all([kbot,reference].map(page=>page.waitForFunction(()=>fixtureRequests.some(row=>row.path.includes('/conversations?situation_id=')))));
        const expectedExecution={COMPLETED:'执行已完成',NO_RUN:'等待诊断',RUNNING:'自动诊断中',FAILED:'诊断失败'}[runState];
        await Promise.all([kbot,reference].map(page=>page.waitForFunction(label=>document.querySelector('.aiops-situation-result-summary')?.textContent.includes(label),expectedExecution)));
        for(const width of [1440,1024,820,560,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await compareRegions(kbot,reference,['.aiops-detail-drawer','.aiops-situation-drawer-header','.aiops-situation-title-block','.aiops-situation-kicker','.aiops-situation-workspace','.aiops-situation-facts','.aiops-situation-pane-header','.aiops-situation-fact-group','.aiops-situation-source','.aiops-situation-observation','.aiops-situation-signal','.aiops-situation-action-pane','.aiops-situation-action-header','.aiops-situation-mode-switch','.aiops-situation-mode-switch button','.aiops-situation-action-body','.aiops-situation-result-summary','.aiops-diagnosis-statuses','.aiops-situation-diagnosis','.aiops-turn-result','.ops-turn-result-empty','.ops-findings','.ops-finding-row','.ops-analysis','.agent-chart','.aiops-situation-handoff'],{page:name,width,state:runState});
          if(width===1440 || width===420) for(const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifacts,`${prefix}-situation-${runState}-${width}.png`)});
        }
        if(runState==='RUNNING') for(const page of [kbot,reference]) assert.equal(await page.getByRole('button',{name:'继续深入诊断',exact:true}).isDisabled(),true);
        else {
          await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'继续深入诊断',exact:true}).click()));
          await Promise.all([kbot,reference].map(page=>page.locator('.aiops-continuation-form select:not(:disabled)').waitFor()));
          for(const width of [1440,820,420,320]) {
            await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
            await compareRegions(kbot,reference,['.aiops-situation-action-header','.aiops-continuation-form','.aiops-continuation-context','.aiops-continuation-form label','.aiops-continuation-form select','.aiops-continuation-form textarea','.aiops-continuation-actions','.aiops-continuation-actions button'],{page:name,width,state:`${runState}-continue`});
          }
          for(const page of [kbot,reference]) {
            assert.equal(await page.getByRole('button',{name:'进入诊断对话',exact:true}).isDisabled(),true);
            await page.locator('.aiops-continuation-form select').selectOption('agent-1');
            await page.locator('.aiops-continuation-form textarea').fill('  请继续核对告警与当前事务。  ');
            await page.evaluate(()=>{fixtureHoldWrites=true;});
            await page.getByRole('button',{name:'进入诊断对话',exact:true}).click();
            await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
            const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
            const payload=JSON.parse(request.body);
            if(page===kbot) {
              assert.equal(payload[runState==='COMPLETED'?'source_run_id':'source_situation_id'],runState==='COMPLETED'?'run-1':'situation-1');
              assert.ok(request.headers['Idempotency-Key']);
            } else assert.deepEqual(payload.conversation.source,runState==='COMPLETED'?{source_type:'RUN',run_id:'run-1'}:{source_type:'SITUATION',situation_id:'situation-1'});
            assert.deepEqual((payload.first_turn || payload).content,[{content_type:'TEXT',text:'请继续核对告警与当前事务。'}]);
            assert.equal(await page.locator('.aiops-continuation-form textarea').isDisabled(),true);
            assert.equal(await page.getByRole('button',{name:'返回结果',exact:true}).isDisabled(),true);
            await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
            await page.locator('.aiops-continuation-form textarea:not(:disabled)').waitFor();
            assert.equal(await page.locator('.aiops-continuation-form textarea').inputValue(),'  请继续核对告警与当前事务。  ');
          }
        }
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'关闭告警诊断详情',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.locator('.aiops-detail-drawer').waitFor({state:'hidden'})));
      }
      for(const page of [kbot,reference]) {
        await page.evaluate(()=>{fixtureSituationState='COMPLETED';fixtureSituationConversation=true;});
        await page.getByRole('button',{name:'查看诊断',exact:true}).click();
        await page.getByRole('button',{name:'进入深入诊断会话',exact:true}).waitFor();
        assert.equal(await page.getByRole('button',{name:'进入深入诊断会话',exact:true}).isEnabled(),true);
        await page.getByRole('button',{name:'关闭告警诊断详情',exact:true}).click();
      }
    }
    if(name==='runs' && !empty) {
      await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'查看详情',exact:true}).click()));
      for(const mode of ['COMPLETED','RUNNING','FAILED','TERMINATION_UNKNOWN','FOLLOWUP']) {
        console.log(`运行详情：${mode}`);
        await Promise.all([kbot,reference].map(page=>page.evaluate(mode=>{fixtureRunMode=mode;},mode)));
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'刷新详情',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.waitForFunction(mode=>mode==='RUNNING'||mode==='FAILED'?document.querySelector('.aiops-turn-result')?.textContent.includes(mode):mode==='FOLLOWUP'?Boolean(document.querySelector('.aiops-observation-series')):mode==='TERMINATION_UNKNOWN'?document.querySelector('[aria-label="只读查询清理回执"]')?.textContent.includes('仅展示最近 50'):Boolean(document.querySelector('.aiops-turn-result .ops-findings')),mode)));
        for(const width of [1440,900,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await Promise.all([kbot,reference].map(page=>page.locator('.aiops-detail-drawer').evaluate(node=>{node.scrollTop=0;})));
          await compareRegions(kbot,reference,['.aiops-detail-drawer','.admin-drawer-header','.key-value-card','.key-value-table','.aiops-turn-result','.aiops-turn-result section','[aria-label="只读查询清理回执"]','[aria-label="只读查询清理回执"] th','[aria-label="只读查询清理回执"] td','.aiops-observation-series','.aiops-observation-series .key-value-card','.aiops-observation-series th','.aiops-observation-series td span','.admin-drawer-footer','.admin-drawer-footer .primary-button'],{page:name,width,state:mode});
        }
      }
      await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'取消后续采样',exact:true}).click()));
      for(const width of [1440,420,320]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-message','.confirm-dialog-actions'],{page:name,width,state:'cancel-observation'});}
      for(const page of [kbot,reference]) {
        await page.evaluate(()=>{fixtureHoldWrites=true;});await page.getByRole('button',{name:'取消采样',exact:true}).click();await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
        const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1)),body=JSON.parse(request.body);
        assert.ok(request.path.endsWith('/runs/run-1/commands'));assert.equal(body.ops_run_id,'run-1');assert.ok(body.command_id);assert.ok(body.idempotency_key);assert.deepEqual(body.command,{command_type:'CANCEL_RUN',expected_row_version:4});
        await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
        await page.getByRole('button',{name:'关闭运行详情',exact:true}).click();
        await page.locator('.aiops-detail-drawer').waitFor({state:'hidden'});
      }
    }
    if(name==='inspection-plans' && !empty) {
      const drawer='.aiops-inspection-plan-drawer';
      for(const type of ['DAILY','WEEKLY','CRON']) {
        await Promise.all([kbot,reference].map(page=>page.locator('.management-catalog-add-card').click()));
        if(type!=='DAILY') await Promise.all([kbot,reference].map(page=>page.getByRole('radio',{name:type==='WEEKLY'?/^每周/:/^高级规则/}).check()));
        for(const width of [1440,900,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await Promise.all([kbot,reference].map(page=>page.locator(drawer).evaluate(node=>{node.scrollTop=0;})));
          await compareRegions(kbot,reference,[drawer,'.admin-drawer-header','.admin-drawer-body','.drawer-form-section','.inspection-frequency-options','.inspection-frequency-options label','.inspection-weekday-options','.date-time-picker-control','.inspection-schedule-summary','.inspection-template-summary','.form-grid-two','.admin-drawer-footer'],{page:name,width,state:`create-${type}`});
        }
        for(const page of [kbot,reference]) {
          const root=page.locator(drawer);
          await root.getByPlaceholder('例如：生产数据库每日健康巡检').fill('  夜间生产巡检  ');
          await root.getByRole('combobox',{name:/^执行 Agent/}).selectOption('agent-1');
          if(type==='CRON') await root.getByPlaceholder('0 2 * * *').fill('15 4 * * 6');
          else await root.locator('input[type=time]').fill('04:15');
          await page.evaluate(()=>{fixtureHoldWrites=true;});
          await root.getByRole('button',{name:'创建并启用',exact:true}).click();
          await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
          const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
          assert.ok(request.path.endsWith('/inspection-plans'));assert.ok(request.headers['Idempotency-Key']);
          assert.deepEqual(JSON.parse(request.body),{display_name:'夜间生产巡检',agent_id:'agent-1',cron_expression:type==='DAILY'?'15 4 * * *':type==='WEEKLY'?'15 4 * * 1,2,3,4,5':'15 4 * * 6',timezone:'Asia/Shanghai',timeout_seconds:1800,overlap_policy:'SKIP',misfire_policy:'LATEST_ONLY',schedule_type:type,inspection_template_id:'template-1'});
          assert.equal(await root.getByRole('button',{name:'关闭',exact:true}).isDisabled(),true);
          await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
          await root.getByRole('button',{name:'创建并启用',exact:true}).waitFor();
          assert.equal((await root.getByPlaceholder('例如：生产数据库每日健康巡检').inputValue()).trim(),'夜间生产巡检');
          await root.getByRole('button',{name:'取消',exact:true}).click();
        }
      }
      for(const page of [kbot,reference]) {
        await page.getByRole('button',{name:'编辑',exact:true}).click();
        const root=page.locator(drawer);await root.waitFor();
        assert.equal(await root.getByRole('radio').first().isDisabled(),true);
        await page.evaluate(()=>{fixtureHoldWrites=true;});await root.getByRole('button',{name:'保存修改',exact:true}).click();
        await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
        const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='PATCH').at(-1)),body=JSON.parse(request.body);
        assert.ok(request.path.endsWith('/inspection-plans/plan-1'));assert.equal(request.headers['If-Match'],'"rv-3"');
        assert.equal(Object.hasOwn(body,'schedule_type'),false);assert.equal(Object.hasOwn(body,'inspection_template_id'),false);
        await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});await root.getByRole('button',{name:'保存修改',exact:true}).waitFor();
        await root.getByRole('button',{name:'取消',exact:true}).click();
      }
      for(const action of ['暂停','停用计划']) {
        if(action==='停用计划') await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'更多',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:action,exact:true}).click()));
        for(const width of [1440,420,320]) {await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));await compareRegions(kbot,reference,['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-message','.confirm-dialog-actions'],{page:name,width,state:action});}
        for(const page of [kbot,reference]) {
          const root=page.locator('.confirm-dialog:visible');await page.evaluate(()=>{fixtureHoldWrites=true;});await root.getByRole('button',{name:action==='暂停'?'暂停':'停用',exact:true}).click();
          await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
          assert.ok(request.path.endsWith(`/inspection-plans/plan-1/${action==='暂停'?'pause':'disable'}`));assert.equal(request.headers['If-Match'],'"rv-3"');assert.ok(request.headers['Idempotency-Key']);
          await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});await page.getByRole('button',{name:'编辑',exact:true}).waitFor({state:'visible'});
        }
      }
    }
    if(name==='recovery-drills' && !empty) {
      const field=(page,label)=>page.locator('.recovery-entry-form .control-field').filter({has:page.getByText(label,{exact:true})}).locator('input');
      for(const page of [kbot,reference]) {
        await field(page,'模拟故障时间').fill('2026-10-30T08:00');
        await field(page,'实际恢复到的数据时间').fill('2026-10-30T07:55');
        await field(page,'验证完成时间').fill('2026-10-30T08:20');
        await field(page,'证据引用').fill('  evidence://isolated/recovery-1  ');
        await field(page,'证据 SHA-256').fill('a'.repeat(64));
        await page.locator('.recovery-entry-form textarea').fill('  完成隔离恢复和只读验证。  ');
        await page.evaluate(()=>{fixtureHoldWrites=true;});
        await page.getByRole('button',{name:'提交待审核记录',exact:true}).click();
        await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
        const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
        assert.ok(request.path.endsWith('/managed-resources/resource-1/recovery-drills'));
        assert.ok(request.headers['Idempotency-Key']);
        assert.deepEqual(JSON.parse(request.body),{source_work_item_id:null,scenario:'FULL_INSTANCE',assurance_level:'DATABASE_OPEN',backup_source_type:'ORACLE_RMAN',environment:'ISOLATED',result:'PASS',simulated_failure_at:'2026-10-30T00:00:00.000Z',recovered_through_at:'2026-10-29T23:55:00.000Z',service_validated_at:'2026-10-30T00:20:00.000Z',recovery_marker:{kind:'ORACLE'},evidence:[{evidence_kind:'DATABASE_CHECK',reference:'evidence://isolated/recovery-1',content_hash:'a'.repeat(64)}],notes:'完成隔离恢复和只读验证。'});
        assert.equal(await page.getByRole('button',{name:'正在提交…',exact:true}).isDisabled(),true);
        await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
        await page.getByRole('button',{name:'提交待审核记录',exact:true}).waitFor();
        assert.equal(await field(page,'证据 SHA-256').inputValue(),'a'.repeat(64));
        assert.equal((await page.locator('.recovery-entry-form textarea').inputValue()).trim(),'完成隔离恢复和只读验证。');
      }
      for(const decision of ['VERIFY','REJECT']) {
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:decision==='VERIFY'?'通过':'驳回',exact:true}).click()));
        for(const width of [1440,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await compareRegions(kbot,reference,['.confirm-dialog','.confirm-dialog-header','.confirm-dialog-message','.confirm-dialog-actions'],{page:name,width,state:decision});
        }
        for(const page of [kbot,reference]) {
          await page.evaluate(()=>{fixtureHoldWrites=true;});
          await page.getByRole('button',{name:decision==='VERIFY'?'审核通过':'驳回',exact:true}).last().click();
          await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
          const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
          assert.ok(request.path.endsWith('/managed-resources/resource-1/recovery-drills/drill-1:review'));
          assert.deepEqual(JSON.parse(request.body),{decision,review_note:null});
          assert.equal(request.headers['If-Match'],'"rv-2"');assert.ok(request.headers['Idempotency-Key']);
          assert.equal(await page.locator('.management-catalog-card .row-action').isDisabled(),true);
          await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
          await page.locator('.management-catalog-card .row-action:not(:disabled)').waitFor();
        }
      }
      for(const status of ['VERIFIED','REJECTED']) {
        await Promise.all([kbot,reference].map(page=>page.evaluate(status=>{fixtureRecoveryStatus=status;fixtureRequests=[];},status)));
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'刷新',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.locator('.management-catalog-note').waitFor()));
        for(const width of [1440,420]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await compareRegions(kbot,reference,['.management-catalog-card','.management-catalog-card .status-pill','.management-catalog-actions'],{page:name,width,state:status});
        }
      }
    }
    if(name==='reports' && !empty) {
      const dateInput=(page,label)=>page.locator('.admin-drawer:visible label.control-field').filter({has:page.getByText(label,{exact:true})}).locator('input');
      console.log('报告对照：PDF 下载');
      for(const page of [kbot,reference]) {
        await page.getByRole('button',{name:'下载 PDF',exact:true}).click();
        await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseDownload));
        assert.equal(await page.getByRole('button',{name:'下载中…',exact:true}).isDisabled(),true);
        const request=await page.evaluate(()=>fixtureRequests.find(row=>row.method==='DOWNLOAD'));
        assert.ok(request.path.endsWith('/reports/report-1/pdf'));
        await page.evaluate(()=>{fixtureReleaseDownload();delete globalThis.fixtureReleaseDownload;});
        await page.getByRole('button',{name:'下载 PDF',exact:true}).waitFor();
      }
      await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'生成数据库报告',exact:true}).click()));
      console.log('报告对照：数据库报告表单');
      for(const db of ['ORACLE','POSTGRESQL','MYSQL']) {
        const targetId={ORACLE:'target-1',POSTGRESQL:'postgres-1',MYSQL:'mysql-1'}[db];
        await Promise.all([kbot,reference].map(page=>page.getByRole('combobox',{name:/^数据库目标/}).selectOption(targetId)));
        for(const suffix of db==='ORACLE'?['WORKLOAD','WORKLOAD_DIFF']:['WORKLOAD','WORKLOAD_DIFF','ACTIVITY']) {
          const type=`${db}_${suffix}`;
          console.log(`报告对照：${type}`);
          await Promise.all([kbot,reference].map(page=>page.getByRole('combobox',{name:/^报告类型/}).selectOption(type)));
          for(const width of [1440,820,420,320]) {
            await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
            await compareRegions(kbot,reference,['.admin-drawer[aria-label="生成数据库运维报告"]','.admin-drawer[aria-label="生成数据库运维报告"] .admin-drawer-header','.admin-drawer[aria-label="生成数据库运维报告"] .resource-form','.admin-drawer[aria-label="生成数据库运维报告"] .control-field','.admin-drawer[aria-label="生成数据库运维报告"] .field-help','.admin-drawer[aria-label="生成数据库运维报告"] .date-time-picker-control','.admin-drawer[aria-label="生成数据库运维报告"] .admin-drawer-footer'],{page:name,width,state:type});
          }
          for(const page of [kbot,reference]) {
            const names=suffix==='WORKLOAD_DIFF'?['基线开始','基线结束','对比开始','对比结束']:['窗口开始','窗口结束'];
            const values=suffix==='WORKLOAD_DIFF'?['2026-10-30T07:00','2026-10-30T08:00','2026-10-30T08:00','2026-10-30T09:00']:['2026-10-30T08:00','2026-10-30T09:00'];
            for(let index=0;index<names.length;index++) await dateInput(page,names[index]).fill(values[index]);
            await page.evaluate(()=>{fixtureHoldWrites=true;});
            await page.getByRole('button',{name:'生成报告',exact:true}).click();
            await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
            const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
            assert.ok(request.path.endsWith(`/reports/${suffix==='WORKLOAD_DIFF'?'workload-diff':suffix==='ACTIVITY'?'activity':'workload'}`));
            assert.ok(request.headers['Idempotency-Key']);
            const payload=JSON.parse(request.body);
            assert.deepEqual(payload,suffix==='WORKLOAD_DIFF'?{target_id:targetId,baseline_start:'2026-10-29T23:00:00.000Z',baseline_end:'2026-10-30T00:00:00.000Z',after_start:'2026-10-30T00:00:00.000Z',after_end:'2026-10-30T01:00:00.000Z'}:{target_id:targetId,period_start:'2026-10-30T00:00:00.000Z',period_end:'2026-10-30T01:00:00.000Z'});
            assert.equal(await page.getByRole('button',{name:'生成中…',exact:true}).isDisabled(),true);
            assert.equal(await page.getByRole('button',{name:'关闭报告表单',exact:true}).isDisabled(),true);
            await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
            await page.getByRole('button',{name:'生成报告',exact:true}).waitFor();
            assert.equal(await page.getByRole('combobox',{name:/^报告类型/}).inputValue(),type);
            assert.equal(await dateInput(page,names[0]).inputValue(),values[0]);
          }
        }
      }
      await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'关闭报告表单',exact:true}).click()));
      await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'导入 pgBadger',exact:true}).click()));
      for(const width of [1440,820,420,320]) {
        await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
        await compareRegions(kbot,reference,['.admin-drawer[aria-label="导入 PostgreSQL pgBadger 报告"]','.admin-drawer[aria-label="导入 PostgreSQL pgBadger 报告"] .admin-drawer-header','.admin-drawer[aria-label="导入 PostgreSQL pgBadger 报告"] .resource-form','.admin-drawer[aria-label="导入 PostgreSQL pgBadger 报告"] .control-field','.admin-drawer[aria-label="导入 PostgreSQL pgBadger 报告"] .date-time-picker-control','.admin-drawer[aria-label="导入 PostgreSQL pgBadger 报告"] input[type="file"]','.admin-drawer[aria-label="导入 PostgreSQL pgBadger 报告"] .admin-drawer-footer'],{page:name,width,state:'pgbadger-import'});
      }
      for(const page of [kbot,reference]) {
        assert.equal(await page.getByRole('button',{name:'导入报告',exact:true}).isDisabled(),true);
        await dateInput(page,'报告开始').fill('2026-10-30T08:00');
        await dateInput(page,'报告结束').fill('2026-10-30T09:00');
        await page.locator('input[type="file"]').setInputFiles({name:'运行报告.json',mimeType:'application/json',buffer:Buffer.from('{"fixture":true}')});
        await page.evaluate(()=>{fixtureHoldWrites=true;});
        await page.getByRole('button',{name:'导入报告',exact:true}).click();
        await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
        const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
        const url=new URL(request.path,'http://fixture');assert.ok(url.pathname.endsWith('/postgresql/pgbadger-artifacts'));
        assert.equal(url.searchParams.get('target_id'),'postgres-1');assert.equal(url.searchParams.get('period_start'),'2026-10-30T00:00:00.000Z');assert.equal(url.searchParams.get('period_end'),'2026-10-30T01:00:00.000Z');
        assert.equal(request.headers['Content-Type'],'application/json');assert.equal(decodeURIComponent(request.headers['X-File-Name']),'运行报告.json');assert.ok(request.headers['Idempotency-Key']);
        assert.equal(await page.getByRole('button',{name:'导入中…',exact:true}).isDisabled(),true);
        await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
        await page.getByRole('button',{name:'导入报告',exact:true}).waitFor();
        assert.equal(await page.locator('input[type="file"]').evaluate(input=>input.files[0].name),'运行报告.json');
        await page.getByRole('button',{name:'关闭 pgBadger 导入表单',exact:true}).click();
      }
    }
    if(name==='inspections' && !empty) {
      const regions=['.inspection-detail-drawer','.inspection-drawer-header','.inspection-title-block','.inspection-kicker','.inspection-workspace','.inspection-facts','.inspection-pane-header','.inspection-fact-group','.inspection-observation','.inspection-run-tabs','.inspection-run-tabs button','.inspection-run-statuses','.inspection-action-pane','.inspection-action-header','.inspection-mode-switch','.inspection-action-body','.inspection-result-summary','.inspection-result-statuses','.inspection-diagnosis','.inspection-report-result','.inspection-report-heading','.inspection-status-stack','.inspection-report-section','.inspection-report-section h5','.inspection-check-list','.inspection-check-list article','.inspection-observations','.has-gaps','.ops-findings','.ops-finding-row','.aiops-turn-result','.ops-analysis','.agent-chart','.ops-turn-result-empty','.inspection-handoff'];
      for(const mode of ['REPORT','TURN','RUNNING','FAILED','NO_RUN']) {
        await Promise.all([kbot,reference].map(page=>page.evaluate(value=>{fixtureInspectionMode=value;fixtureRequests=[];},mode)));
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'查看报告',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.locator('.inspection-detail-drawer').waitFor({state:'visible'})));
        if(mode!=='NO_RUN') {
          await Promise.all([kbot,reference].map(page=>page.waitForFunction(()=>fixtureRequests.some(row=>row.path.includes('/conversations?run_id=run-1')))));
          await Promise.all([kbot,reference].map(page=>page.waitForFunction(()=>!document.querySelector('.inspection-handoff')?.textContent.includes('正在检查会话'))));
        }
        for(const width of [1440,1024,820,560,420,320]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await compareRegions(kbot,reference,regions,{page:name,width,state:mode});
          if(width===1440 || width===420) for(const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifacts,`${prefix}-inspection-${mode}-${width}.png`)});
        }
        if(['RUNNING','NO_RUN'].includes(mode)) {
          for(const page of [kbot,reference]) assert.equal(await page.getByRole('button',{name:'继续深入诊断',exact:true}).isDisabled(),true);
        } else {
          if(mode==='REPORT') {
            await Promise.all([kbot,reference].map(page=>page.locator('.inspection-run-tabs button').nth(1).click()));
            await Promise.all([kbot,reference].map(page=>page.waitForFunction(()=>fixtureRequests.some(row=>row.path.includes('/conversations?run_id=run-2')))));
            await Promise.all([kbot,reference].map(page=>page.waitForFunction(()=>document.querySelector('.inspection-result-summary')?.textContent.includes('执行失败'))));
            await compareRegions(kbot,reference,regions,{page:name,width:320,state:'second-target'});
            await Promise.all([kbot,reference].map(page=>page.evaluate(()=>{fixtureRequests=[];})));
            await Promise.all([kbot,reference].map(page=>page.locator('.inspection-run-tabs button').first().click()));
            await Promise.all([kbot,reference].map(page=>page.waitForFunction(()=>fixtureRequests.some(row=>row.path.includes('/conversations?run_id=run-1')))));
          }
          await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'继续深入诊断',exact:true}).click()));
          await Promise.all([kbot,reference].map(page=>page.locator('.inspection-continuation-form select:not(:disabled)').waitFor()));
          for(const width of [1440,820,420,320]) {
            await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
            await compareRegions(kbot,reference,['.inspection-action-header','.inspection-continuation-form','.inspection-continuation-context','.inspection-continuation-form label','.inspection-continuation-form select','.inspection-continuation-form textarea','.inspection-continuation-actions','.inspection-continuation-actions button'],{page:name,width,state:`${mode}-continue`});
          }
          for(const page of [kbot,reference]) {
            assert.equal(await page.getByRole('button',{name:'进入诊断对话',exact:true}).isDisabled(),true);
            await page.locator('.inspection-continuation-form select').selectOption('agent-1');
            await page.locator('.inspection-continuation-form textarea').fill('  请继续核对巡检证据。  ');
            await page.evaluate(()=>{fixtureHoldWrites=true;});
            await page.getByRole('button',{name:'进入诊断对话',exact:true}).click();
            await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
            const request=await page.evaluate(()=>fixtureRequests.filter(row=>row.method==='POST').at(-1));
            const payload=JSON.parse(request.body);
            if(page===kbot) {assert.equal(payload.source_run_id,'run-1');assert.ok(request.headers['Idempotency-Key']);}
            else assert.deepEqual(payload.conversation.source,{source_type:'RUN',run_id:'run-1'});
            assert.deepEqual((payload.first_turn || payload).content,[{content_type:'TEXT',text:'请继续核对巡检证据。'}]);
            assert.equal(await page.locator('.inspection-continuation-form textarea').isDisabled(),true);
            assert.equal(await page.getByRole('button',{name:'返回结果',exact:true}).isDisabled(),true);
            await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
            await page.locator('.inspection-continuation-form textarea:not(:disabled)').waitFor();
            assert.equal(await page.locator('.inspection-continuation-form textarea').inputValue(),'  请继续核对巡检证据。  ');
          }
        }
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'关闭巡检诊断详情',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.locator('.inspection-detail-drawer').waitFor({state:'hidden'})));
      }
      for(const page of [kbot,reference]) {
        await page.evaluate(()=>{fixtureInspectionMode='REPORT';fixtureInspectionConversation=true;});
        await page.getByRole('button',{name:'查看报告',exact:true}).click();
        await page.getByRole('button',{name:'进入深入诊断会话',exact:true}).waitFor();
        assert.equal(await page.getByRole('button',{name:'进入深入诊断会话',exact:true}).isEnabled(),true);
        await page.getByRole('button',{name:'关闭巡检诊断详情',exact:true}).click();
      }
    }
    if (name === 'report-templates') {
      for (const kind of ['inspection','session']) {
        await Promise.all([kbot,reference].map(page=>page.getByRole('tab',{name:kind==='inspection'?'巡检内容模板':'会话报告模板',exact:true}).click()));
        for (const width of [1440,420]) {
          await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
          await compareRegions(kbot,reference,['.resource-list-surface:not([hidden])','.resource-list-surface:not([hidden]) .management-catalog-grid','.resource-list-surface:not([hidden]) .management-catalog-card','.resource-list-surface:not([hidden]) .management-catalog-action-bar','.resource-list-surface:not([hidden]) .management-catalog-primary-actions','.resource-list-surface:not([hidden]) .management-catalog-more'],{page:name,width,empty,state:`${kind}-catalog`});
          for (const mode of empty?['create']:['create','version']) {
            await Promise.all([kbot,reference].map(page=>mode==='create'?page.locator('.resource-list-surface:visible .management-catalog-add-card').click():page.getByRole('button',{name:kind==='inspection'?'创建新版本':'新版本',exact:true}).click()));
            await Promise.all([kbot,reference].map(page=>page.locator('.aiops-template-editor:visible').waitFor()));
            await compareRegions(kbot,reference,['.aiops-template-editor','.aiops-template-editor .admin-drawer-header','.aiops-template-editor .admin-drawer-body','.aiops-template-editor .drawer-form-section','.aiops-template-editor .control-field','.aiops-template-editor .aiops-template-check-group','.aiops-template-editor .check-field','.aiops-template-editor .aiops-section-editor','.aiops-template-editor .aiops-template-preview','.aiops-template-editor .aiops-template-preview li','.aiops-template-editor .admin-drawer-footer','.aiops-template-editor .primary-button'],{page:name,width,empty,state:`${kind}-${mode}`});
            for (const page of [kbot,reference]) {
              const editor=page.locator('.aiops-template-editor:visible');
              if(kind==='inspection') assert.equal(await editor.locator('input[value="planned-check"]').isDisabled(),true);
              if(mode==='create') {
                assert.equal(await editor.getByRole('button',{name:'保存',exact:true}).isDisabled(),true);
                await editor.locator('input[maxlength="256"]').fill('测试模板');
                if(kind==='inspection') await editor.locator('input[value="availability"]').check();
              } else assert.equal(await editor.locator('input[maxlength="256"]').isDisabled(),true);
              assert.equal(await editor.getByRole('button',{name:'保存',exact:true}).isEnabled(),true);
              if(kind==='session') {
                await editor.locator('input[value="RISK_OVERVIEW"]').check();
                await editor.locator('.aiops-template-preview li').filter({hasText:'风险概览'}).getByRole('button',{name:'上移章节'}).click();
              }
            }
            if(kind==='session') assert.deepEqual(await kbot.locator('.aiops-template-editor:visible .aiops-template-preview li strong').allTextContents(),await reference.locator('.aiops-template-editor:visible .aiops-template-preview li strong').allTextContents());
            for (const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifacts,`${prefix}-${name}-${kind}-${mode}-${empty?'empty':'populated'}-${width}.png`)});
            if(width===1440 && !empty) {
              for(const page of [kbot,reference]) {
                await page.evaluate(()=>{globalThis.fixtureRequests=[];globalThis.fixtureHoldWrites=true;});
                const editor=page.locator('.aiops-template-editor:visible');
                await editor.getByRole('button',{name:'保存',exact:true}).click();
                await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
                assert.equal(await editor.getByRole('button',{name:'保存中…',exact:true}).isDisabled(),true);
                await page.evaluate(()=>{globalThis.fixtureHoldWrites=false;globalThis.fixtureReleaseWrite();delete globalThis.fixtureReleaseWrite;});
                await editor.getByRole('button',{name:'保存',exact:true}).waitFor();
                assert.equal(await editor.getByRole('button',{name:'保存',exact:true}).isEnabled(),true);
              }
              const payload=page=>page.evaluate(()=>fixtureRequests.filter(item=>item.method==='POST').map(item=>JSON.parse(item.body)));
              assert.deepEqual(await payload(kbot),await payload(reference),'保存数据必须与源端一致，含章节顺序和版本锁');
              if(kind==='session') {
                const body=(await payload(kbot))[0];
                assert.equal(body.definition.sections[0].kind,'EXECUTIVE_SUMMARY');
                assert.equal(body.definition.sections.at(-1).kind,'EVIDENCE_BOUNDARY');
              }
            }
            await Promise.all([kbot,reference].map(page=>page.locator('.aiops-template-editor:visible .admin-drawer-header button').click()));
          }
          if(kind==='session' && !empty) {
            await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'更多',exact:true}).click()));
            await compareRegions(kbot,reference,['.management-catalog-menu','.management-catalog-menu button'],{page:name,width,state:'session-menu'});
            await Promise.all([kbot,reference].map(page=>page.keyboard.press('Escape')));
            for(const page of [kbot,reference]) await page.locator('.management-catalog-menu:visible').waitFor({state:'hidden'});
            await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'版本',exact:true}).click()));
            await Promise.all([kbot,reference].map(page=>page.getByRole('dialog',{name:'模板版本历史'}).waitFor()));
            await compareRegions(kbot,reference,['.admin-drawer:not(.aiops-template-editor)','.aiops-version-list','.aiops-version-list article','.aiops-version-list article header','.aiops-version-list p','.aiops-version-list small'],{page:name,width,state:'session-history'});
            await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'关闭版本历史'}).click()));
          }
        }
      }
    }
    if (name === 'responsibility-groups' && !empty) {
      for (const width of [1440,420]) {
        await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'修改',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.locator('.responsibility-dialog-form input').waitFor()));
        await compareRegions(kbot,reference,['.responsibility-dialog:not(.responsibility-members-dialog)','.responsibility-dialog-header','.responsibility-dialog-body','.responsibility-dialog-actions'],{page:name,width,state:'edit'});
        for (const page of [kbot,reference]) await page.locator('.responsibility-dialog:not(.responsibility-members-dialog) .responsibility-dialog-header button').click();
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'人员',exact:true}).click()));
        await Promise.all([kbot,reference].map(page=>page.locator('.responsibility-members-dialog').waitFor({state:'visible'})));
        await compareRegions(kbot,reference,['.responsibility-members-dialog','.responsibility-members-dialog .responsibility-dialog-header','.responsibility-member-body','.responsibility-member-list','.member-row','.member-actions .row-action','.member-actions .danger-row-action','.responsibility-member-form','.responsibility-member-form select','.responsibility-member-form .primary-button','.responsibility-members-dialog .responsibility-dialog-actions'],{page:name,width,state:'members'});
        for (const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) {
          assert.equal(await page.getByRole('button',{name:'添加人员',exact:true}).isDisabled(),true);
          await page.locator('.responsibility-member-form select').selectOption('user-3');
          assert.equal(await page.getByRole('button',{name:'添加人员',exact:true}).isEnabled(),true);
          assert.equal(await page.getByRole('button',{name:'移出',exact:true}).last().isDisabled(),true,'已有活动事项的成员不能移出');
          await page.screenshot({path:resolve(artifacts,`${prefix}-${name}-members-${width}.png`)});
          await page.locator('.responsibility-members-dialog .responsibility-dialog-header button').click();
        }
      }
    }
    if (name === 'monitoring' && !empty) {
      for (const width of [1440,420]) {
        await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
        for (const [label,page] of [['Ammolite',reference],['KBot',kbot]]) {
          try { await page.getByRole('button',{name:/^数据质量/}).click({timeout:5000}); }
          catch(error) { await page.screenshot({path:resolve(artifacts,`${label}-quality-failure-${width}.png`),fullPage:true}); console.log(artifacts); throw error; }
        }
        await compareRegions(kbot,reference,['.monitoring-quality-detail','.monitoring-quality-detail > header','.monitoring-quality-grid','.monitoring-quality-grid dl','.monitoring-quality-grid ul'],{page:name,width,state:'quality'});
        await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:/^数据质量/}).click()));
      }
      await Promise.all([kbot,reference].map(page=>page.setViewportSize({width:1440,height:1000})));
      await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'收起数据库与过滤条件',exact:true}).click()));
      await compareRegions(kbot,reference,['.monitoring-dashboard-shell','.monitoring-context-rail','.monitoring-dashboard-main'],{page:name,width:1440,state:'collapsed'});
      await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'展开数据库与过滤条件',exact:true}).click()));
    }
    if (panels) for (const width of [1440,420]) {
      await Promise.all([kbot,reference].map(page=>page.setViewportSize({width,height:1000})));
      await Promise.all([kbot,reference].map(page=>page.getByRole('button',{name:'查看数据',exact:true}).first().click()));
      await Promise.all([kbot,reference].map(page=>page.locator('.monitoring-data-drawer').waitFor({state:'visible'})));
      await Promise.all([kbot,reference].map(page=>page.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))))));
      await compareRegions(kbot,reference,['.monitoring-data-drawer','.monitoring-data-drawer .admin-drawer-header','.monitoring-data-drawer-body','.monitoring-data-drawer .drawer-summary-grid','.monitoring-series-insights','.monitoring-data-table','.monitoring-data-drawer .admin-drawer-footer'],{page:name,width,state:'data-drawer'});
      for (const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) {
        await page.screenshot({path:resolve(artifacts,`${prefix}-monitoring-data-drawer-${width}.png`)});
        await page.locator('.monitoring-data-drawer .admin-drawer-header button').click();
        await page.locator('.monitoring-data-drawer').waitFor({state:'hidden'});
      }
    }
  }
  console.log(`页面对照截图：${artifacts}`);
  writeFileSync(resolve(artifacts, "differences.json"), JSON.stringify(differences, null, 2));
  if (differences.length) console.log(JSON.stringify(differences.slice(0, 16), null, 2));
  assert.deepEqual(errors, [], "真实页面不应产生浏览器脚本异常");
  assert.equal(differences.length, 0, "页面关键区域仍与 Ammolite 不同，详见 differences.json");
  console.log(`通过：${pages.filter(page=>!['chat','business-system-workspace'].includes(page)).length} 个业务页面的空/有数据与至少 3 种宽度对照${pages.includes('business-system-workspace')?'；业务系统工作区在 4 种宽度下的新建、运行中只读、已下线编辑、三引擎/PDB/RAC、采集策略、拓扑几何、人工校正、删除/恢复、草稿请求与失败恢复对照':''}${pages.includes('chat')?'；Agent 工作台在 6 种宽度下的结果、流式、范围确认、人工补证、变更审批与效果验证对照，以及图表、执行计划、下载互斥和失败恢复合同':''}。`);
} finally {
  await browser.close(); await referenceServer.close(); await new Promise((done) => staticServer.close(done));
}
