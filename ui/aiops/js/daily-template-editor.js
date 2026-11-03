// 巡检模板使用完整页面；正式日报渲染器同时承担模拟预览。
(function () {
  "use strict";
  const root = document.getElementById("daily-template-editor"), form = document.getElementById("daily-template-form");
  const esc = value => globalThis.KBotAIOpsShell.escape(value), api = "/api/v1/apps/aiops";
  let catalog, detail, layout, regionId, onSaved, busy = false, revision = 0;
  const request = (path,method,body) => KBotAIOpsAuth.request(api+path,{method,body:JSON.stringify(body)});
  const title = key => catalog.report_sections.find(item => item.section_id === key)?.title || key;
  const selected = () => [...form.querySelectorAll('[name="check_id"]:checked')].map(item => item.value);
  function markStale() {
    revision++;
    document.getElementById("daily-preview-stale").textContent = "配置已变化，刷新后查看最新效果。仅使用模拟数据，不连接数据库或调用模型。";
  }
  function switchTab(key) {
    root.querySelectorAll("[data-editor-tab]").forEach(item => item.classList.toggle("active",item.dataset.editorTab === key));
    root.querySelectorAll("[data-editor-panel]").forEach(item => { item.hidden = item.dataset.editorPanel !== key; });
  }
  function renderLayout() {
    document.getElementById("daily-layout-outline").innerHTML = '<h2>共享报告目录</h2>' + layout.sections.map((item,index) => '<div><button type="button" data-region="'+esc(item.section_id)+'" class="'+(regionId===item.section_id ? "active" : "")+'">'+String(index+1).padStart(2,"0")+' · '+esc(title(item.section_id))+'</button>'+(index > 0 && index < layout.sections.length-1 ? '<button type="button" data-region-move="'+index+'" data-offset="-1" aria-label="上移区域" '+(index===1 ? "disabled" : "")+'>↑</button><button type="button" data-region-move="'+index+'" data-offset="1" aria-label="下移区域" '+(index===layout.sections.length-2 ? "disabled" : "")+'>↓</button>' : "")+'</div>').join("");
    const item = layout.sections.find(item => item.section_id === regionId), fixed = ["OVERVIEW","ACTIONS"].includes(regionId);
    let content = '<h2>'+esc(title(regionId))+'</h2><label class="control-field"><span class="field-label">日报标题</span><input name="report_title" value="'+esc(layout.title)+'" maxlength="128" required></label><label class="control-field"><span class="field-label">区域展示</span><select data-setting="visibility" '+(fixed ? "disabled" : "")+'><option value="ALWAYS" '+(item.visibility==="ALWAYS" ? "selected" : "")+'>始终展示</option><option value="WHEN_RELEVANT" '+(item.visibility==="WHEN_RELEVANT" ? "selected" : "")+'>有相关内容时展示</option></select></label><p class="notice">已知问题、必要检查和未完成事项始终保留。未记录结果不等于正常或不适用。</p>';
    if (!["OVERVIEW","ACTIONS","SQL","BASELINE"].includes(regionId)) {
      content += '<label class="control-field"><span class="field-label">正常检查明细</span><select data-setting="normal_detail"><option value="FULL" '+(item.normal_detail==="FULL" ? "selected" : "")+'>完整明细</option><option value="SUMMARY" '+(item.normal_detail==="SUMMARY" ? "selected" : "")+'>摘要，网页可展开核查</option></select></label>';
      content += [["expanded","网页默认展开正常检查"],["show_statistics","显示监控窗口平均 / 最大值"]].map(([key,label]) => '<label class="check-field"><input type="checkbox" data-setting="'+key+'" '+(item[key] ? "checked" : "")+'>'+label+'</label>').join("");
    }
    if (regionId==="BASELINE") content += '<label class="check-field"><input type="checkbox" data-setting="show_trend" '+(item.show_trend ? "checked" : "")+'>显示连续采样日均趋势</label>';
    if (regionId==="SQL") content += '<p class="notice">保留全部问题 SQL，按严重性排序并逐条展示 Agent 分析。检查成功且没有问题时显示“当前没有问题 SQL”。</p>';
    document.getElementById("daily-layout-settings").innerHTML = content;
  }
  function error(value) { const node=document.getElementById("daily-template-error"); node.textContent=value; node.hidden=!value; }
  async function preview() {
    const button=document.getElementById("daily-refresh-preview"), version=revision;
    if(button.disabled) return;
    button.disabled=true; button.textContent="生成预览中…";
    try {
      const result=await request("/inspection-templates:preview","POST",{daily_report_layout:layout,selected_check_ids:selected(),database_type:form.elements.preview_engine.value,scenario:form.elements.preview_scenario.value});
      document.getElementById("daily-template-preview").innerHTML=KBotDailyInspectionReport.render(result.daily_inspection);
      document.getElementById("daily-preview-stale").textContent=version===revision ? "模拟预览 · 与正式日报共用渲染器 · 非真实数据库结果" : "配置已变化，请刷新模拟预览。";
    } catch(cause) { error(cause.message); }
    finally { button.disabled=false;button.textContent="刷新模拟预览"; }
  }
  function close() { if(busy)return;root.hidden=true;document.querySelector("main.aiops-template-page").hidden=false;window.scrollTo(0,0); }
  function open(value,editing,callback) {
    catalog=value;detail=editing;onSaved=callback;busy=false;
    form.reset();error("");form.elements.display_name.value=detail?.display_name || "";form.elements.display_name.disabled=Boolean(detail?.inspection_template_id);
    layout=structuredClone(detail?.daily_report_layout || catalog.daily_report_layout);regionId=layout.sections[0].section_id;
    const engines=[...new Set(catalog.groups.flatMap(group => group.checks.flatMap(check => check.supported_db_types)))];
    form.elements.preview_engine.innerHTML=engines.map(engine => '<option>'+esc(engine)+'</option>').join("");
    document.getElementById("daily-template-checks").innerHTML=catalog.groups.map(group => '<fieldset><legend>'+esc(group.display_name)+'</legend>'+group.checks.map(check => '<label class="check-field"><input type="checkbox" name="check_id" value="'+esc(check.check_id)+'" '+((detail?.selected_check_ids || []).includes(check.check_id) ? "checked" : "")+' '+(check.availability!=="READY" ? "disabled" : "")+'><span><strong>'+esc(check.display_name)+'</strong><small>'+esc(title(check.report_section)+" · "+check.supported_db_types.join(" / ")+(check.availability!=="READY" ? " · 尚未可用" : ""))+'</small></span></label>').join("")+'</fieldset>').join("");
    root.querySelector("h1").textContent=detail?.inspection_template_id ? "发布模板新版本" : "新建巡检模板";
    renderLayout();switchTab("checks");root.hidden=false;document.querySelector("main.aiops-template-page").hidden=true;markStale();window.scrollTo(0,0);preview();
  }
  root.querySelectorAll("[data-close-daily-editor]").forEach(button => {button.onclick=close;});
  root.querySelectorAll("[data-editor-tab]").forEach(button => {button.onclick=()=>switchTab(button.dataset.editorTab);});
  root.addEventListener("click",event => {
    const button=event.target.closest("button");
    if(!button || busy)return;
    if(button.dataset.region) {regionId=button.dataset.region;renderLayout();}
    if(button.dataset.regionMove) {
      const index=Number(button.dataset.regionMove), target=index+Number(button.dataset.offset);
      if(target<=0 || target>=layout.sections.length-1)return;
      [layout.sections[index],layout.sections[target]]=[layout.sections[target],layout.sections[index]];markStale();renderLayout();
    }
  });
  form.addEventListener("input",event => {
    if(event.target.name==="report_title")layout.title=event.target.value;
    if(event.target.name!=="display_name")markStale();
  });
  form.addEventListener("change",event => {
    const key=event.target.dataset.setting;
    if(key)layout.sections.find(item => item.section_id===regionId)[key]=event.target.type==="checkbox" ? event.target.checked : event.target.value;
    markStale();
  });
  document.getElementById("daily-refresh-preview").onclick=preview;
  form.onsubmit=async event => {
    event.preventDefault();
    if(busy || !selected().length || !layout.title.trim()) {error("请填写日报标题并至少选择一项专项检查。");return;}
    busy=true;error("");
    const controls=[...form.querySelectorAll("button,input,select")].map(node => [node,node.disabled]);
    controls.forEach(([node]) => {node.disabled=true;});
    // 禁用控件前已选内容仍可读取，不改变事实或检查规则。
    const body={...(detail?.inspection_template_id ? {expected_row_version:detail.row_version} : {display_name:form.elements.display_name.value.trim()}),selected_check_ids:selected(),daily_report_layout:layout};
    try {
      await request("/inspection-templates"+(detail?.inspection_template_id ? "/"+encodeURIComponent(detail.inspection_template_id)+"/versions" : ""),"POST",body);
      busy=false;close();KBotAIOpsShell.toast("模板已发布；已有计划仍使用原绑定版本。");await onSaved();
    } catch(cause) {error(cause.message);}
    finally {busy=false;controls.forEach(([node,disabled])=>{node.disabled=disabled;});}
  };
  globalThis.KBotDailyTemplateEditor=Object.freeze({open});
})();
