// 分步编辑器使用正式目录、预览与版本接口，不执行数据库查询。
import * as model from "./daily-template-model.js?v=20261103_5";
const root = document.getElementById("daily-template-editor");
const esc = value => globalThis.KBotAIOpsShell.escape(value);
const request = (path,body,method="POST") => KBotAIOpsAuth.request("/api/v1/apps/aiops"+path,{method,headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
let catalog,detail,draft,original,onSaved,automatic,topicId,regionId,engine,adaptationEngine,scenario;
let persisted="",step="checks",busy=false,previewBusy=false,stale=true,revision=0,session=0;
const node = id => root.querySelector("#"+id);
const title = key => catalog.report_sections.find(item=>item.section_id===key)?.title || key;
const valid = () => Boolean(draft.name.trim() && draft.layout.title.trim() && draft.selected.length && !model.unavailableIds(catalog,draft.selected).length);
function error(message) {node("daily-template-error").textContent=message;node("daily-template-error").hidden=!message;}
function changed() {const diff=model.changeSummary(draft,original);return !detail?.version_no || draft.name.trim()!==original.name || diff.added>0 || diff.removed>0 || diff.layoutChanged;}
function updatePublish() {
  node("daily-publish").disabled=busy || step!=="preview" || !changed() || !valid() || stale || previewBusy || !node("daily-publish-confirm").checked;
  node("daily-publish-confirm").disabled=busy || !changed() || !valid() || stale || previewBusy;
}
function renderSummary() {
  const changes=model.changeSummary(draft,original),invalid=model.unavailableIds(catalog,draft.selected);
  node("daily-editor-summary").innerHTML='<p class="eyebrow">模板摘要</p><h2>'+esc(draft.name || "未命名模板")+'</h2><dl>'+
    [["适用范围","通用日报 · 全部数据库类型"],["系统保障","自动纳入，不能移除"],["追加专项",draft.selected.filter(id=>!automatic.has(id)).length+" 个数据库适配检查"],["报告区域",draft.layout.sections.length+" 个共享区域"],["发布状态",detail?.version_no ? "基于 v"+detail.version_no+" 编辑" : "新模板 · 尚未发布"]].map(([label,value])=>'<div><dt>'+label+'</dt><dd>'+esc(value)+'</dd></div>').join("")+
    '</dl><hr><h3>发布影响</h3><p>发布形成不可变新版本；已有计划不自动升级。</p><p>修改计划并明确选择新版本后才生效。月报排版保持现状。</p>'+
    (invalid.length ? '<p class="notice notice-error">所选检查有 '+invalid.length+' 项已不可用，请刷新目录后重新选择。</p>' : "");
  node("daily-change-summary").textContent="专项新增 "+changes.added+" 项，移除 "+changes.removed+" 项；日报版式"+(changes.layoutChanged ? "有变化" : "未变化")+"。";
  node("daily-validation").hidden=valid() || step==="checks";
}
function markStale() {
  revision++;stale=true;node("daily-publish-confirm").checked=false;node("daily-preview-stale").textContent="配置已变化，刷新后查看最新效果。";updatePublish();renderSummary();
}
function renderAdaptation() {
  const adapted=model.checks(catalog).filter(check=>check.report_section===topicId && check.supported_db_types.includes(adaptationEngine));
  node("daily-adaptation-checks").innerHTML=adapted.map(check=>'<div class="daily-adaptation-row"><label class="check-field"><input type="checkbox" data-adapt-check="'+esc(check.check_id)+'" '+(automatic.has(check.check_id) || draft.selected.includes(check.check_id) ? "checked" : "")+' '+(automatic.has(check.check_id) || check.availability!=="READY" ? "disabled" : "")+'><span>'+esc(check.display_name)+' · '+(automatic.has(check.check_id) ? "自动纳入" : check.availability!=="READY" ? "尚未可用" : "可追加")+'</span></label><small>'+esc(check.check_id)+'</small></div>').join("") || '<p class="daily-editor-muted">本类型没有本主题的目录专项；监控和备份仍按执行时的绑定与策略核验。</p>';
}
function renderTopic() {
  const current=model.themes(catalog).find(item=>item.section_id===topicId),fixed=model.checkGroups(catalog,topicId,true),optional=model.checkGroups(catalog,topicId,false);
  const roles=(current.semantic_roles || []).filter(role=>role in catalog.daily_check_policy.required_metric_roles).map(role=>catalog.daily_check_policy.required_metric_roles[role]);
  node("daily-topic-nav").innerHTML=model.themes(catalog).map(item=>'<button type="button" data-topic="'+esc(item.section_id)+'" aria-pressed="'+(topicId===item.section_id)+'" class="'+(topicId===item.section_id ? "active" : "")+'">'+esc(item.title)+'</button>').join("");
  node("daily-topic-body").innerHTML='<h3>'+esc(current.title)+'</h3><p class="daily-editor-muted">'+esc(current.description)+'</p><section class="daily-check-group"><h4>系统保障 <small>自动纳入 · 不可移除</small></h4><ul class="daily-foundation-list">'+fixed.map(group=>'<li>'+esc(group.title)+'</li>').join("")+roles.map(role=>'<li>'+esc(role)+' · 按已绑定监控与冻结规则核验</li>').join("")+
    (topicId==="BACKUP" ? '<li>按启用的备份保障策略核验成功记录与时效，日志链正常不代替备份成功。</li>' : !fixed.length && !roles.length ? '<li>本主题没有目录级固定检查；适用的专项可在下方追加。</li>' : "")+
    '</ul></section><section class="daily-check-group"><h4>可追加专项 <small>同名入口保留各数据库独立检查</small></h4>'+
    optional.map((group,index)=>{const state=model.groupState(group,draft.selected);return '<label class="daily-check-row"><input type="checkbox" data-group-index="'+index+'" data-check-group="'+esc(group.title)+'" '+(state.checked ? "checked" : "")+' '+(!state.total ? "disabled" : "")+'><span><strong>'+esc(group.title)+'</strong><small>'+(!state.total ? "尚未可用，不能追加" : state.mixed ? "部分适配已选择；勾选可纳入全部可用适配" : "按适用数据库执行；选择情况可在适配说明中核对")+'</small></span></label>';}).join("")+
    (!optional.length ? '<p class="daily-editor-muted">无需追加目录专项，执行时按本主题的系统保障规则检查。</p>' : "")+'</section><details class="daily-adaptation"><summary>查看数据库适配说明</summary><label class="control-field"><span class="field-label">查看适配类型</span><select name="adaptation_engine">'+model.engines(catalog).map(value=>'<option '+(value===adaptationEngine ? "selected" : "")+'>'+esc(value)+'</option>').join("")+'</select></label><p class="daily-editor-muted">仅查看适配，不改变模板适用范围或其他数据库的选择。</p><div id="daily-adaptation-checks"></div></details>';
  optional.forEach((group,index)=>{root.querySelector('[data-group-index="'+index+'"]').indeterminate=model.groupState(group,draft.selected).mixed;});renderAdaptation();
}
function renderLayout() {
  node("daily-layout-outline").innerHTML=draft.layout.sections.map((item,index)=>'<div><button type="button" data-region="'+esc(item.section_id)+'" aria-pressed="'+(regionId===item.section_id)+'" class="'+(regionId===item.section_id ? "active" : "")+'">'+String(index+1).padStart(2,"0")+' · '+esc(title(item.section_id))+'</button>'+(index===0 || index===draft.layout.sections.length-1 ? '<small>固定</small>' : '<button type="button" data-region-move="'+index+'" data-offset="-1" aria-label="上移区域" '+(index===1 ? "disabled" : "")+'>↑</button><button type="button" data-region-move="'+index+'" data-offset="1" aria-label="下移区域" '+(index===draft.layout.sections.length-2 ? "disabled" : "")+'>↓</button>')+'</div>').join("");
  const item=draft.layout.sections.find(item=>item.section_id===regionId),fixed=["OVERVIEW","ACTIONS"].includes(regionId),editable=!["OVERVIEW","ACTIONS","SQL","BASELINE"].includes(regionId);
  let content='<h3>'+esc(title(regionId))+'</h3><p class="daily-editor-muted">'+esc(catalog.report_sections.find(item=>item.section_id===regionId)?.description || "")+'</p>';
  if(!fixed)content+='<label class="control-field"><span class="field-label">区域展示</span><select data-setting="visibility"><option value="ALWAYS" '+(item.visibility==="ALWAYS" ? "selected" : "")+'>始终展示</option><option value="WHEN_RELEVANT" '+(item.visibility==="WHEN_RELEVANT" ? "selected" : "")+'>有相关内容时展示</option></select></label>';
  if(editable) {
    content+='<label class="control-field"><span class="field-label">本区域正常明细</span><select data-setting="normal_detail"><option value="FULL" '+(item.normal_detail==="FULL" ? "selected" : "")+'>完整明细</option><option value="SUMMARY" '+(item.normal_detail==="SUMMARY" ? "selected" : "")+'>摘要，网页可展开核查</option></select></label><label class="check-field"><input type="checkbox" data-setting="expanded" '+(item.expanded ? "checked" : "")+'>完整明细模式下默认展开正常检查</label>';
    if(model.canConfigureStatistics(catalog,regionId))content+='<label class="check-field"><input type="checkbox" data-setting="show_statistics" '+(item.show_statistics ? "checked" : "")+'>显示监控窗口平均 / 最大值</label>';
  }
  if(regionId==="BASELINE")content+='<label class="check-field"><input type="checkbox" data-setting="show_trend" '+(item.show_trend ? "checked" : "")+'>显示连续采样日均趋势</label><p class="daily-editor-muted">基线不足会说明原因，不影响本次已完成检查的结论。</p>';
  if(model.fixedDescriptions[regionId])content+='<section class="daily-structure-example"><h4>本区域保留内容</h4><ul>'+model.fixedDescriptions[regionId].map(line=>'<li>'+esc(line)+'</li>').join("")+'</ul><p>这些内容由正式巡检结果生成，不在模板中手写结论。</p></section>';
  node("daily-layout-settings").innerHTML=content+'<button type="button" class="secondary-button" data-go="preview">查看报告预览</button>';
  node("daily-normal-preference").value=model.normalPreference(draft.layout);
}
function go(next) {
  if(busy)return;step=next;
  window.scrollTo(0,0);
  root.querySelectorAll("[data-editor-tab]").forEach(button=>{button.classList.toggle("active",button.dataset.editorTab===step);if(button.dataset.editorTab===step)button.setAttribute("aria-current","step");else button.removeAttribute("aria-current");});
  root.querySelectorAll("[data-editor-panel]").forEach(panel=>{panel.hidden=panel.dataset.editorPanel!==step;});
  node("daily-previous").hidden=step==="checks";node("daily-next").hidden=step==="preview";node("daily-publish").hidden=step!=="preview";
  node("daily-next").textContent=step==="checks" ? "下一步：报告编排" : "下一步：预览与发布";
  node("daily-step-help").textContent=step==="checks" ? "完成主题选择后，继续编排报告。" : step==="layout" ? "展示设置不修改阈值、取数或健康判断。" : "核对模拟预览后，确认创建不可变版本。";
  renderSummary();updatePublish();if(step==="preview" && stale && !previewBusy)preview();
}
async function preview() {
  if(previewBusy || busy)return;
  const requestedRevision=revision,requestedSession=session;
  previewBusy=true;node("daily-refresh-preview").disabled=true;node("daily-refresh-preview").textContent="生成预览中…";node("daily-preview-error").hidden=true;node("daily-preview-stale").textContent="正在生成模拟预览…";node("daily-publish-confirm").checked=false;updatePublish();
  try {
    const result=await request("/inspection-templates:preview",{selected_check_ids:draft.selected,daily_report_layout:draft.layout,database_type:engine,scenario});
    if(requestedSession!==session)return;
    if(!result.daily_inspection)throw new Error("预览未返回日报内容，请重试。");
    node("daily-template-preview").innerHTML=KBotDailyInspectionReport.render(result.daily_inspection);stale=requestedRevision!==revision;
    node("daily-preview-stale").textContent=stale ? "配置已变化，刷新后查看最新效果。" : "模拟预览已更新，可核对后发布。";
  } catch(cause) {
    if(requestedSession!==session)return;
    stale=true;node("daily-preview-error").textContent=cause.message+" 旧预览不代表当前配置，请重试。";node("daily-preview-error").hidden=false;node("daily-preview-stale").textContent="预览未更新，请重试。";
  } finally {
    if(requestedSession===session){previewBusy=false;node("daily-refresh-preview").disabled=false;node("daily-refresh-preview").textContent="刷新模拟预览";updatePublish();}
  }
}
async function close(force=false) {
  if(busy)return;
  if(!force && JSON.stringify(draft)!==persisted && !await globalThis.KBotInspectionTemplates.confirm("离开模板编辑","尚有未保存修改。返回将保留上次保存的草稿，放弃本地修改。"))return;
  session++;root.hidden=true;document.querySelector("main.aiops-template-page").hidden=false;window.scrollTo(0,0);if(!force)await onSaved();
}
function open(value,editing,callback) {
  catalog=value;detail=editing;onSaved=callback;globalThis.KBotInspectionTemplates.hideDetail();session++;revision=0;busy=false;previewBusy=false;stale=true;
  try{draft=model.createDraft(catalog,detail);original=model.createDraft(catalog,detail?.definition ? {...detail,draft:null} : detail);persisted=JSON.stringify(draft);automatic=model.automaticIds(catalog);}
  catch(cause){KBotAIOpsShell.toast(cause.message);return;}
  topicId=model.themes(catalog).find(item=>item.section_id==="CAPACITY")?.section_id || model.themes(catalog)[0].section_id;
  regionId="OVERVIEW";engine=model.engines(catalog)[0];adaptationEngine=engine;scenario="ATTENTION";
  const options=model.engines(catalog).map(value=>'<option>'+esc(value)+'</option>').join("");
  root.innerHTML='<header class="ops-head"><div><p class="eyebrow">通用日报 · 模板编辑</p><h1>'+(detail?.inspection_template_id ? "编辑巡检模板" : "新建巡检模板")+'</h1><p>先确定检查内容，再编排报告，最后预览与发布。一份模板适用于不同数据库。</p></div><button type="button" class="secondary-button" data-close-daily-editor>返回模板中心</button></header><form id="daily-template-form"><fieldset class="daily-editor-fields"><nav class="daily-editor-steps" aria-label="日报模板编辑步骤">'+
    [["checks","检查内容","选择业务主题与专项"],["layout","报告编排","设置标题与共享目录"],["preview","预览与发布","核对样例与版本影响"]].map(([key,label,help],index)=>'<button type="button" data-editor-tab="'+key+'"><span class="daily-step-number">'+(index+1)+'</span><span><strong>'+label+'</strong><small>'+help+'</small></span></button>').join("")+'</nav><div class="daily-editor-workspace"><div class="daily-editor-main">'+
    '<section class="daily-editor-panel" data-editor-panel="checks"><header class="daily-panel-head"><p class="eyebrow">第一步 · 检查什么</p><h2>围绕业务主题选择检查</h2><p>系统保障自动纳入且不可移除；可追加专项不会替代基础检查。</p></header><div class="daily-topic-layout"><nav id="daily-topic-nav" class="daily-topic-nav" aria-label="检查主题"></nav><div id="daily-topic-body" class="daily-topic-body"></div></div></section>'+
    '<section class="daily-editor-panel" data-editor-panel="layout" hidden><header class="daily-panel-head"><p class="eyebrow">第二步 · 报告怎么读</p><h2>共享报告编排</h2><p>十二个区域共用一份格式。已知问题、必要检查和未完成事项不能隐藏。</p></header><div class="daily-global-settings"><div class="daily-field-grid"><label class="control-field"><span class="field-label">模板名称</span><input name="display_name" maxlength="256" required '+""+' value="'+esc(draft.name)+'"></label><label class="control-field"><span class="field-label">日报标题</span><input name="report_title" maxlength="128" required value="'+esc(draft.layout.title)+'"></label></div><label class="control-field"><span class="field-label">统一正常明细偏好</span><select id="daily-normal-preference"><option value="CUSTOM" disabled>按区域设置（保留现有差异）</option><option value="FULL">完整明细</option><option value="SUMMARY">摘要，网页可展开核查</option></select></label></div><div class="daily-template-layout"><nav id="daily-layout-outline" class="daily-layout-outline" aria-label="日报区域配置"></nav><div id="daily-layout-settings" class="daily-template-settings"></div></div></section>'+
    '<section class="daily-editor-panel daily-template-preview" data-editor-panel="preview" hidden><header class="daily-panel-head"><p class="eyebrow">第三步 · 核对后发布</p><h2>预览与发布</h2><p>模拟样例使用正式日报渲染器，不连接目标数据库、不调用模型。</p></header><div class="daily-preview-controls"><label class="control-field"><span class="field-label">模拟数据库类型</span><select name="preview_engine">'+options+'</select></label><label class="control-field"><span class="field-label">模拟场景</span><select name="preview_scenario"><option value="NORMAL">检查正常</option><option value="ATTENTION" selected>发现问题</option><option value="INCOMPLETE">检查未完成</option></select></label><button id="daily-refresh-preview" type="button" class="secondary-button">刷新模拟预览</button><p>数据库类型只影响样例，不改变通用模板范围。</p></div><p id="daily-preview-error" class="notice notice-error" role="alert" hidden></p><p id="daily-preview-stale" class="notice" role="status"></p><div id="daily-template-preview" class="daily-preview-paper"><p class="notice">选择数据库类型与场景，生成模拟报告后核对。</p></div>'+
    '<section class="daily-publish-review"><h3>本次发布核对</h3><p id="daily-change-summary"></p><p>'+esc(detail?.version_no ? "当前 v"+detail.version_no+" → 新版本 v"+(Number(detail.version_no)+1) : "创建首个不可变版本")+'。已有计划仍使用原绑定版本，历史报告不重写。</p><label class="check-field"><input id="daily-publish-confirm" type="checkbox">我已核对预览与变更，确认发布新版本；已有计划不自动升级。</label></section></section></div><aside id="daily-editor-summary" class="daily-editor-summary" aria-label="模板摘要"></aside></div><p id="daily-validation" class="notice" role="status" hidden>请填写模板名称、日报标题，并至少保留一项可用目录检查。</p><p id="daily-template-error" class="notice notice-error" role="alert" hidden></p><footer class="daily-template-footer"><button id="daily-save-draft" type="button" class="secondary-button">保存草稿</button><p id="daily-step-help"></p><button id="daily-previous" type="button" class="secondary-button">上一步</button><button id="daily-next" type="button" class="primary-button">下一步：报告编排</button><button id="daily-publish" type="submit" class="primary-button" hidden>发布新版本</button></footer></fieldset></form>';
  renderTopic();renderLayout();go("checks");root.hidden=false;document.querySelector("main.aiops-template-page").hidden=true;window.scrollTo(0,0);
}
root.addEventListener("click",event=>{
  const button=event.target.closest("button");if(!button || busy)return;
  if(button.hasAttribute("data-close-daily-editor"))close();
  if(button.dataset.editorTab)go(button.dataset.editorTab);
  if(button.dataset.go)go(button.dataset.go);
  if(button.dataset.topic){topicId=button.dataset.topic;renderTopic();}
  if(button.dataset.region){regionId=button.dataset.region;renderLayout();}
  if(button.dataset.regionMove){
    const index=Number(button.dataset.regionMove),target=index+Number(button.dataset.offset);
    if(index<=0 || index>=draft.layout.sections.length-1 || target<=0 || target>=draft.layout.sections.length-1)return;
    [draft.layout.sections[index],draft.layout.sections[target]]=[draft.layout.sections[target],draft.layout.sections[index]];markStale();renderLayout();
  }
  if(button.id==="daily-previous")go(step==="preview" ? "layout" : "checks");
  if(button.id==="daily-next")go(step==="checks" ? "layout" : "preview");
  if(button.id==="daily-refresh-preview")preview();
  if(button.id==="daily-save-draft")saveDraft();
});
root.addEventListener("input",event=>{
  const target=event.target;if(busy)return;
  if(target.name==="display_name"){draft.name=target.value;markStale();}
  if(target.name==="report_title"){draft.layout.title=target.value;markStale();}
});
root.addEventListener("change",event=>{
  if(busy)return;const target=event.target,key=target.dataset.setting;
  if(target.id==="daily-publish-confirm"){updatePublish();return;}
  if(target.name==="adaptation_engine"){adaptationEngine=target.value;renderAdaptation();return;}
  if(target.dataset.groupIndex!==undefined){draft.selected=model.toggleGroup(model.checkGroups(catalog,topicId,false)[Number(target.dataset.groupIndex)],draft.selected,target.checked);renderTopic();}
  if(target.dataset.adaptCheck){
    const id=target.dataset.adaptCheck,open=root.querySelector(".daily-adaptation").open;if(automatic.has(id))return;
    draft.selected=target.checked ? [...new Set([...draft.selected,id])] : draft.selected.filter(value=>value!==id);
    renderTopic();root.querySelector(".daily-adaptation").open=open;
  }
  if(target.id==="daily-normal-preference"){model.setNormalPreference(draft.layout,target.value);renderLayout();}
  if(key){draft.layout.sections.find(item=>item.section_id===regionId)[key]=target.type==="checkbox" ? target.checked : target.value;node("daily-normal-preference").value=model.normalPreference(draft.layout);}
  if(target.name==="preview_engine")engine=target.value;
  if(target.name==="preview_scenario")scenario=target.value;markStale();
});
async function persistDraft() {
  const id=detail?.inspection_template_id;
  detail=await request("/inspection-templates"+(id ? "/"+encodeURIComponent(id)+"/draft" : ""),{
    ...(id ? {expected_row_version:detail.row_version}:{}),display_name:draft.name.trim(),
    selected_check_ids:draft.selected,daily_report_layout:draft.layout,
  },id ? "PUT":"POST");
  persisted=JSON.stringify(draft);renderSummary();
}
async function saveDraft() {
  if(busy || !valid())return;
  busy=true;root.querySelector(".daily-editor-fields").disabled=true;error("");
  try{await persistDraft();KBotAIOpsShell.toast("草稿已保存，不影响已发布版本或巡检计划。");}
  catch(cause){error(cause.message);}
  finally{busy=false;root.querySelector(".daily-editor-fields").disabled=false;updatePublish();}
}
root.addEventListener("submit",async event=>{
  event.preventDefault();if(busy || step!=="preview" || !changed() || !valid() || stale || previewBusy || !node("daily-publish-confirm").checked)return;
  busy=true;error("");root.setAttribute("aria-busy","true");root.querySelector("[data-close-daily-editor]").disabled=true;root.querySelector(".daily-editor-fields").disabled=true;node("daily-publish").textContent="发布中…";updatePublish();
  try {
    await persistDraft();
    await request("/inspection-templates/"+encodeURIComponent(detail.inspection_template_id)+":publish",{expected_row_version:detail.row_version});
    busy=false;await close(true);KBotAIOpsShell.toast("模板已发布；已有计划仍使用原绑定版本。");await onSaved();
  } catch(cause){error(cause.message);node("daily-publish-confirm").checked=false;}
  finally{busy=false;root.setAttribute("aria-busy","false");root.querySelector("[data-close-daily-editor]").disabled=false;root.querySelector(".daily-editor-fields").disabled=false;node("daily-publish").textContent="发布新版本";updatePublish();}
});
globalThis.KBotDailyTemplateEditor=Object.freeze({open});
