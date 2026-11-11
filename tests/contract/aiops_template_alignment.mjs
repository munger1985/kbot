// 模板中心整页对照使用现行三步编辑合同；仅隔离请求，不修改配置功能。
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

export function templateAlignmentFixture(root) {
  const sections=JSON.parse(readFileSync(resolve(root,'services/aiops_agent/src/aiops_agent/application/inspections/daily_sections.json'),'utf8')).sections;
  const layout={schema_version:'DAILY_REPORT_LAYOUT.v1',title:'数据库日常巡检报告',sections:sections.map(({section_id})=>({section_id,visibility:'ALWAYS',normal_detail:'FULL',show_statistics:true,show_trend:true,expanded:true}))};
  const check=(check_id,display_name,report_section,availability='READY')=>({check_id,display_name,report_section,availability,tool_id:availability==='READY'?'fixture.readonly':null,default_for:['DAILY'],supported_db_types:['ORACLE','POSTGRESQL','MYSQL'],finding_types:[],trend_required:false});
  const fixture={
    catalog:{catalog_hash:'isolated-current-template-contract',report_sections:sections,daily_report_layout:layout,
      daily_check_policy:{version:'isolated',baseline:Object.fromEntries(['ORACLE','POSTGRESQL','MYSQL'].map(key=>[key,['availability']])),supplemental:{ORACLE:[],POSTGRESQL:[],MYSQL:[]},required_metric_roles:{}},
      groups:[{group_id:'health',display_name:'可用性与容量',checks:[check('availability','数据库可用性','AVAILABILITY'),check('capacity-trend','容量增长趋势','CAPACITY'),check('planned-check','规划检查项','CAPACITY','PLANNED')]}]},
    layout,
    preview:{schema_version:'DAILY_INSPECTION_PRESENTATION.v1',preview:true,
      header:{title:layout.title,target_name:'隔离样例数据库',business_system_name:'隔离测试系统',database_type:'ORACLE',database_version:'19c',environment:'测试',timezone:'Asia/Shanghai',report_number:'FIXTURE-PREVIEW',report_version:1,report_time:'2026-10-30T03:00:00Z',observation_start:'2026-10-30T02:00:00Z',observation_end:'2026-10-30T03:00:00Z'},
      conclusion:{text:'隔离样例仅用于模板交互对照。',health_status:'NORMAL',health_status_display:'正常',completed_required_count:1,required_check_count:1,attention_count:0,incomplete_count:0},
      sections:[],checks:[],finding_index:[],limitations:[],gaps:[],evidence_boundary:{methodology:'模拟版式不连接数据库、不调用模型。',quality:{},sources:[]}},
  };
  const {catalog_hash,...catalogBody}=fixture.catalog;
  fixture.catalog.catalog_hash=createHash('sha256').update(JSON.stringify(catalogBody)).digest('hex');
  return fixture;
}

const editorFor=page=>page.locator('.daily-template-editor:visible').filter({has:page.locator('.daily-editor-steps')});
const writes=page=>page.evaluate(()=>fixtureRequests.filter(row=>row.method!=='GET'&&!row.path.endsWith(':preview')).map(row=>({path:new URL(row.path,'http://fixture').pathname.replace(/\/domains\/[^/]+\/ops/,''),method:row.method,body:JSON.parse(row.body)})));

export async function verifyInspectionTemplateEditor({kbot,reference,mode,width,empty,compareRegions,artifacts}) {
  const context={page:'report-templates',width,empty,state:'inspection-'+mode};
  const pages=[kbot,reference];
  const compare=async(selectors,state)=>{
    // 输入与按钮的自动滚动距离因页壳不同；比较前统一实际视口位置，不改任何样式。
    await Promise.all(pages.map(page=>page.evaluate(()=>window.scrollTo(0,0))));
    await compareRegions(kbot,reference,selectors,{...context,state});
  };
  for(const page of pages) {
    await page.evaluate(()=>{fixtureRequests=[];globalThis.fixtureTemplateWrites=false;globalThis.fixtureTemplateDraft=null;});
    const editor=editorFor(page);
    await editor.waitFor();
    for (const selector of ['h1','.daily-editor-steps']) {
      const box=await editor.locator(selector).boundingBox();
      assert.ok(box && box.y>=0 && box.y+box.height<=page.viewportSize().height,'模板编辑标题和步骤必须直接显示在首屏，不依赖控件自动滚动');
    }
    assert.equal(await editor.evaluate(node=>Boolean(node.closest('dialog,[role="dialog"]'))),false,'巡检模板必须使用正式全页编辑器');
    assert.equal(await editor.locator('[data-editor-tab]').count(),3);
    await editor.locator('[data-topic="CAPACITY"]').click();
    assert.equal(await editor.locator('[data-check-group="规划检查项"]').isDisabled(),true,'规划项不能追加');
    await editor.locator('[data-check-group="容量增长趋势"]').check();
    await editor.locator('[data-editor-tab="layout"]').click();
    if(mode==='create') {
      assert.equal(await editor.getByLabel('模板名称',{exact:true}).inputValue(),'');
      // 保留两端原有呈现差异：Portal 禁用空名称按钮；KBot 在保存处理器拦截。
      assert.equal(await editor.getByRole('button',{name:'保存草稿',exact:true}).isDisabled(),page===reference);
      await editor.getByRole('button',{name:'保存草稿',exact:true}).evaluate(button=>button.click());
      assert.deepEqual(await writes(page),[],'空名称不能发起保存，不以按钮外观替代业务边界');
    }
    await editor.getByLabel('模板名称',{exact:true}).fill(mode==='create'?'隔离新巡检模板':'隔离修改巡检模板');
    await editor.getByLabel('日报标题',{exact:true}).fill('隔离日报 · 三步编辑验收');
    await editor.getByLabel('统一正常明细偏好').selectOption('SUMMARY');
    await editor.locator('.daily-layout-outline > div').nth(2).getByRole('button',{name:'上移区域',exact:true}).click();
    assert.equal(await editor.getByRole('button',{name:'保存草稿',exact:true}).isEnabled(),true);
  }
  await compare(['.daily-template-editor:has(.daily-editor-steps)','.daily-editor-steps','.daily-editor-workspace','.daily-editor-summary','.daily-global-settings','.daily-template-layout','.daily-layout-outline','.daily-template-settings','.daily-template-footer'],context.state+'-layout');
  for(const page of pages) await editorFor(page).locator('[data-editor-tab="checks"]').click();
  await compare(['.daily-topic-layout','.daily-topic-nav','.daily-topic-body','.daily-check-group','.daily-check-row','.daily-adaptation'],context.state+'-checks');
  for(const page of pages) {
    const editor=editorFor(page);
    await editor.locator('[data-editor-tab="preview"]').click();
    await editor.getByText('模拟预览已更新，可核对后发布。',{exact:true}).waitFor();
    assert.equal(await editor.getByRole('button',{name:'发布新版本',exact:true}).isDisabled(),true,'预览完成不等于人工确认');
    await page.evaluate(()=>{globalThis.fixtureTemplatePreviewFailure=true;});
    await editor.getByRole('button',{name:'刷新模拟预览',exact:true}).click();
    await editor.getByRole('alert').filter({hasText:'隔离预览失败'}).waitFor();
    assert.equal(await editor.getByRole('checkbox',{name:/我已核对预览/}).isDisabled(),true,'失败预览不能用于发布');
    await editor.getByRole('button',{name:'刷新模拟预览',exact:true}).click();
    await editor.getByText('模拟预览已更新，可核对后发布。',{exact:true}).waitFor();
    await editor.getByRole('checkbox',{name:/我已核对预览/}).check();
    assert.equal(await editor.getByRole('button',{name:'发布新版本',exact:true}).isEnabled(),true);
    // 保存中不能重复保存或发布；失败必须保留用户输入并恢复操作。
    await page.evaluate(()=>{globalThis.fixtureHoldWrites=true;});
    await editor.getByRole('button',{name:'保存草稿',exact:true}).click();
    await page.waitForFunction(()=>Boolean(globalThis.fixtureReleaseWrite));
    assert.equal(await editor.locator('.daily-editor-fields').evaluate(node=>node.disabled),true);
    assert.equal(await editor.locator('.daily-template-footer .primary-button:visible').isDisabled(),true);
    const before=await writes(page);
    await editor.getByRole('button',{name:/^(保存草稿|保存中…)$/}).evaluate(button=>button.click());
    assert.deepEqual(await writes(page),before,'保存中不能重复提交');
    await page.evaluate(()=>{fixtureHoldWrites=false;fixtureReleaseWrite();});
    await editor.getByRole('alert').filter({hasText:'页面对照不执行写操作'}).waitFor();
    assert.equal(await editor.getByRole('button',{name:'保存草稿',exact:true}).isEnabled(),true);
  }
  assert.deepEqual(await writes(kbot),await writes(reference),'巡检草稿请求必须保留相同内容与版本锁');
  const draft=(await writes(kbot))[0];
  assert.equal(draft.method,mode==='create'?'POST':'PUT');
  assert.equal(draft.path,mode==='create'?'/api/v1/apps/aiops/inspection-templates':'/api/v1/apps/aiops/inspection-templates/template-1/draft');
  assert.equal(draft.body.expected_row_version,mode==='create'?undefined:1);
  assert.deepEqual(draft.body.selected_check_ids,['availability','capacity-trend']);
  assert.equal(draft.body.daily_report_layout.sections[0].section_id,'OVERVIEW');
  assert.equal(draft.body.daily_report_layout.sections.at(-1).section_id,'ACTIONS');
  assert.equal(draft.body.daily_report_layout.sections[1].section_id,'RESOURCES');
  await compare(['.daily-preview-controls','.daily-preview-paper','.daily-publish-review','.daily-template-footer'],context.state+'-preview-recovered');
  for(const [prefix,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifacts,`${prefix}-report-templates-inspection-${mode}-${empty?'empty':'populated'}-${width}.png`)});
  for(const page of pages) {
    const editor=editorFor(page);
    await page.evaluate(()=>{fixtureTemplateWrites=true;fixtureTemplatePublishFailure=true;});
    await editor.getByRole('checkbox',{name:/我已核对预览/}).check();
    await editor.getByRole('button',{name:'发布新版本',exact:true}).click();
    await editor.getByRole('alert').filter({hasText:'隔离发布失败'}).waitFor();
    assert.equal(await editor.getByRole('checkbox',{name:/我已核对预览/}).isChecked(),false);
    assert.equal(await editor.getByRole('button',{name:'发布新版本',exact:true}).isDisabled(),true,'发布失败后必须重新确认');
    await editor.getByRole('checkbox',{name:/我已核对预览/}).check();
    await editor.getByRole('button',{name:'发布新版本',exact:true}).click();
    await editor.waitFor({state:'hidden'});
    assert.equal(await page.locator('.resource-list-surface:visible').count(),1);
    const calls=await writes(page),publishes=calls.filter(row=>row.path.endsWith(':publish'));
    assert.equal(publishes.length,2,'失败重试复用已保存草稿，不丢失版本锁');
    assert.deepEqual(publishes.at(-1).body,{expected_row_version:mode==='create'?2:3});
    assert.equal(calls.filter(row=>row.method==='POST'&&row.path.endsWith('/inspection-templates')).length,mode==='create'?2:0,'首次失败未受理；发布失败后重试不能重复创建模板');
    assert.deepEqual(await page.evaluate(()=>fixtureFailures),[]);
    await page.evaluate(()=>{fixtureTemplateWrites=false;});
  }
  assert.deepEqual(await writes(kbot),await writes(reference),'预览确认、草稿持久化与发布失败重试请求须逐步一致');
}
