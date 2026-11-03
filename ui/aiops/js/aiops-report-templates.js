import "./daily-template-editor.js?v=20261103_5";
import {selectedTopics} from "./daily-template-model.js?v=20261103_5";

(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const escape = shell.escape;
  const sectionLabels = { EXECUTIVE_SUMMARY: "执行摘要", SCOPE: "诊断范围", RISK_OVERVIEW: "风险概览", TREND: "趋势分析", FINDINGS: "诊断发现", ROOT_CAUSE: "根因分析", RECOMMENDATIONS: "建议与行动", ACTIONS: "处置记录", EVIDENCE_BOUNDARY: "证据边界", EVIDENCE_APPENDIX: "证据附录" };
  const requiredSections = new Set(["EXECUTIVE_SUMMARY", "EVIDENCE_BOUNDARY"]);
  const defaultSections = ["EXECUTIVE_SUMMARY", "SCOPE", "FINDINGS", "ROOT_CAUSE", "RECOMMENDATIONS", "EVIDENCE_BOUNDARY"];
  let selectedSections = [...defaultSections];
  let checkCatalog = null;
  let editingSessionTemplate = null;
  let sessionTemplates = [];
  let inspectionTemplates = [];
  let actionBusy = false;
  const sectionLabel = kind => sectionLabels[kind] || kind;
  function matchesSearch(item, session = false) {
    const keyword = document.getElementById("template-search").value.trim().toLocaleLowerCase();
    return !keyword || [item.display_name, ...(session ? (item.sections || []).map(sectionLabel) : [])].some(value => String(value || "").toLocaleLowerCase().includes(keyword));
  }
  function orderedSections() {
    const movable = [...new Set(selectedSections)].filter(kind => !requiredSections.has(kind));
    return ["EXECUTIVE_SUMMARY", ...movable, "EVIDENCE_BOUNDARY"];
  }
  function renderOutline() {
    document.getElementById("session-template-outline").innerHTML = orderedSections().map((kind, index) => `<li><span>${String(index + 1).padStart(2, "0")}</span><strong>${escape(sectionLabel(kind))}</strong>${requiredSections.has(kind) ? "" : `<div class="button-row"><button class="text-button" type="button" aria-label="上移章节" data-move-section="${escape(kind)}" data-offset="-1">↑</button><button class="text-button" type="button" aria-label="下移章节" data-move-section="${escape(kind)}" data-offset="1">↓</button></div>`}</li>`).join("");
  }
  function renderSessionSections() {
    document.getElementById("session-template-sections").innerHTML = '<legend>报告章节</legend>' + Object.entries(sectionLabels).map(([kind, label]) => `<label class="check-field"><input type="checkbox" name="section" value="${kind}" ${selectedSections.includes(kind) ? "checked" : ""} ${requiredSections.has(kind) ? "disabled" : ""}><span class="check-field-copy"><strong>${label}</strong><small>${kind}</small></span></label>`).join("");
    renderOutline();
  }

  function renderInspectionTemplates() {
    document.getElementById("ops-inspection-template-body").innerHTML = inspectionTemplates.filter(item => matchesSearch(item)).map(item => {
      const id=escape(item.inspection_template_id),custom=item.source_kind==="CUSTOM",daily=item.template_kind==="DAILY";
      const edit=custom && daily && (item.status==="ACTIVE" || !item.version_no);
      const actions='<button class="row-action" data-'+(edit ? 'edit-inspection-template':'view-inspection')+'="'+id+'">'+(!item.version_no ? "继续编辑" : edit ? "编辑模板":"查看模板")+'</button>';
      const more=
        (item.version_no && item.status==="ACTIVE" && !item.is_default ? '<button class="row-action" data-template-mutation data-default-inspection="'+id+'">设为新计划默认</button>':"")+
        (custom && item.status!=="ACTIVE" && daily ? '<button class="row-action" data-edit-inspection-template="'+id+'">编辑模板</button>':"")+
        (custom && item.version_no ? '<button class="row-action" data-template-mutation data-toggle-inspection="'+id+'">'+(item.status==="ACTIVE" ? "停用":"启用")+'</button>':"")+
        (custom && item.has_draft ? '<button class="danger-row-action" data-template-mutation data-discard-inspection="'+id+'">放弃草稿</button>':"")+
        (custom && item.can_delete ? '<button class="danger-row-action" data-template-mutation data-delete-inspection="'+id+'">删除模板</button>':"");
      return '<article class="management-catalog-card inspection-template-card"><header><div><p class="eyebrow">'+(custom ? "自定义模板":"系统内置 · 只读")+' · '+(daily ? "日常巡检":"月度巡检")+'</p><h3><button class="row-action" data-view-inspection="'+id+'">'+escape(item.display_name)+'</button></h3></div><span class="status-pill">'+(!item.version_no ? "尚未发布" : item.status==="ACTIVE" ? "已启用":"已停用")+'</span></header>'+
        '<div class="inspection-template-flags">'+(item.is_default ? '<span class="status-pill status-success">新计划默认</span>':"")+(item.has_unpublished_changes ? '<span>有未发布修改</span>':"")+'</div>'+
        '<dl><div><dt>最新发布</dt><dd>'+(item.version_no ? "v"+escape(item.version_no):"尚未发布")+'</dd></div><div><dt>引用计划</dt><dd>'+Number(item.usage_count || 0)+' 个</dd></div>'+(item.history_count ? '<div><dt>历史批次引用</dt><dd>'+Number(item.history_count)+' 个</dd></div>':"")+'</dl>'+
        '<footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions">'+actions+'</div>'+(more ? '<div class="management-catalog-overflow"><button class="row-action management-catalog-more" aria-haspopup="menu" aria-expanded="false">更多 ⌄</button><div class="management-catalog-menu" role="menu" hidden>'+more+'</div></div>':"")+'</footer></article>';
    }).join("");
    setActionBusy(actionBusy);
  }
  function renderSessionTemplates() {
    document.getElementById("ops-session-report-template-body").innerHTML = sessionTemplates.filter(item => matchesSearch(item, true)).map(item => {
      const actions = `${!item.is_default && item.status === "ACTIVE" ? `<button class="row-action" type="button" data-template-mutation data-default-session-template="${escape(item.template_ref)}">设为默认</button>` : ""}${item.system_defined ? `<button class="row-action" type="button" data-copy-session-template="${escape(item.template_ref)}">复制</button>` : `<button class="row-action" type="button" data-edit-session-template="${escape(item.template_id)}">新版本</button><button class="row-action" type="button" data-history-session-template="${escape(item.template_id)}">版本</button>`}`;
      const overflow = item.system_defined ? "" : `<div class="management-catalog-overflow"><button class="row-action management-catalog-more" type="button" aria-haspopup="menu" aria-expanded="false"><span>更多</span><span aria-hidden="true">⌄</span></button><div class="management-catalog-menu" role="menu" hidden><button class="row-action" type="button" data-template-mutation data-toggle-session-template="${escape(item.template_id)}" data-row-version="${escape(item.row_version)}" data-next-status="${item.status === "ACTIVE" ? "INACTIVE" : "ACTIVE"}">${item.status === "ACTIVE" ? "停用模板" : "启用模板"}</button></div></div>`;
      return `<article class="management-catalog-card${item.system_defined ? "" : " is-custom"}${item.status === "ACTIVE" ? "" : " is-disabled"}"><header><div><p class="eyebrow">${item.system_defined ? "SYSTEM TEMPLATE" : "CUSTOM TEMPLATE"}</p><h3>${escape(item.display_name)}</h3></div><span class="status-pill ${item.is_default || item.status === "ACTIVE" ? "status-success" : "status-muted"}">${item.is_default ? "默认" : item.status === "ACTIVE" ? "已启用" : "已停用"}</span></header><p>${escape((item.sections || []).map(sectionLabel).join("、") || "尚未配置报告章节。")}</p><dl><div><dt>当前版本</dt><dd>v${escape(item.version_no ?? item.version ?? "—")}</dd></div><div><dt>使用情况</dt><dd>${Number(item.usage_count || 0)} 份报告</dd></div></dl><footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions">${actions}</div>${overflow}</footer></article>`;
    }).join("");
    setActionBusy(actionBusy);
  }
  async function loadTemplates() {
    const refresh = document.getElementById("refresh-templates");
    refresh.disabled = true; refresh.textContent = "刷新中…";
    try {
      [checkCatalog, inspectionTemplates, sessionTemplates] = await Promise.all([
        KBotAIOpsAuth.request(`${api}/inspection-check-catalog`),
        KBotAIOpsAuth.request(`${api}/inspection-templates`),
        KBotAIOpsAuth.request(`${api}/session-report-templates`),
      ]);
      renderInspectionTemplates(); renderSessionTemplates();
      if(new URLSearchParams(location.hash.slice(1)).get('template')) await showInspectionDetail();
    } catch (error) { shell.toast(error.message); }
    finally { refresh.disabled = false; refresh.textContent = "刷新目录"; }
  }
  function setActionBusy(value) {
    actionBusy = value;
    document.querySelectorAll("[data-template-mutation]").forEach(button => { button.disabled = value; });
  }
  async function mutateTemplate(path, method, body) {
    if (actionBusy) return;
    setActionBusy(true);
    try {
      await KBotAIOpsAuth.request(`${api}${path}`, { method, body: JSON.stringify(body) });
      await loadTemplates();
    } catch (error) { shell.toast(error.message); }
    finally { setActionBusy(false); }
  }
  function closeMenus(except = null) {
    document.querySelectorAll(".management-catalog-overflow").forEach(root => {
      if (root === except) return;
      root.querySelector(".management-catalog-menu").hidden = true;
      root.querySelector(".management-catalog-more").setAttribute("aria-expanded", "false");
    });
  }
  function canSave(form) {
    return Boolean(form.elements.display_name.value.trim()) && selectedSections.length > 0;
  }
  function updateSaveButton(form, kind) {
    const busy = form.closest("dialog").dataset.busy === "true";
    const button = form.querySelector('[type="submit"]');
    button.disabled = busy || !canSave(form, kind); button.textContent = busy ? "保存中…" : "保存";
  }
  function openEditor(kind, editing = null, clone = null) {
    if (kind === "inspection") { KBotDailyTemplateEditor.open(checkCatalog,editing,loadTemplates); return; }
    const dialog = document.getElementById("session-report-template-dialog");
    const form = dialog.querySelector("form");
    editingSessionTemplate = editing;
    form.reset();
    form.elements.display_name.disabled = Boolean(editing);
    form.elements.display_name.value = editing?.display_name || (clone ? `${clone.display_name} 副本` : "");
    const result = form.querySelector('[role="status"]'); result.hidden = true; result.textContent = "";
    selectedSections = editing ? (editing.definition?.sections || []).map(section => section.kind) : [...(clone?.sections || defaultSections)];
    renderSessionSections();
    dialog.querySelector("h2").textContent = editing ? "创建新版本" : "新建模板";
    dialog.setAttribute("aria-label", editing ? "创建模板新版本" : "新建模板");
    updateSaveButton(form, kind);
    dialog.showModal();
    dialog.scrollTop = 0;
  }
  async function openVersion(kind, id) {
    try {
      const path = kind === "inspection" ? "inspection-templates" : "session-report-templates";
      const detail = await KBotAIOpsAuth.request(`${api}/${path}/${encodeURIComponent(id)}`);
      openEditor(kind, detail);
    } catch (error) { shell.toast(error.message); }
  }
  async function showSessionTemplateHistory(templateId) {
    try {
      const versions = await KBotAIOpsAuth.request(`${api}/session-report-templates/${encodeURIComponent(templateId)}/versions`);
      document.getElementById("template-history-title").textContent = sessionTemplates.find(item => item.template_id === templateId)?.display_name || "模板版本历史";
      document.getElementById("template-history-list").innerHTML = versions.map(item => `<article><header><strong>v${escape(item.version_no)}</strong>${item.current ? '<span class="status-pill status-success">当前版本</span>' : ""}<time>${escape(shell.fmt(item.created_at))}</time></header><p>${escape((item.definition?.sections || []).map(section => sectionLabel(section.kind)).join(" → "))}</p><small class="technical-meta">${escape(item.content_hash)}</small></article>`).join("");
      document.getElementById("template-history-drawer").showModal();
    } catch (error) { shell.toast(error.message); }
  }
  const detailRoot=document.createElement("section");
  detailRoot.className="ops-main daily-template-editor aiops-page inspection-template-detail";detailRoot.hidden=true;document.querySelector("main.aiops-template-page").after(detailRoot);
  let detailData=null,detailVersions=[],detailPlans=[],detailSeq=0;
  async function confirmTemplate(title,message,confirmLabel="确认",destructive=false) {
    const dialog=document.createElement("dialog");dialog.className="confirm-dialog";
    dialog.innerHTML='<h2 id="template-confirm-title">'+escape(title)+'</h2><p>'+escape(message)+'</p><footer><button class="secondary-button" data-no>取消</button><button class="'+(destructive ? "danger-button":"primary-button")+'" data-yes>'+escape(confirmLabel)+'</button></footer>';
    dialog.setAttribute("aria-labelledby","template-confirm-title");
    document.body.append(dialog);dialog.showModal();dialog.querySelector("[data-no]").focus();
    return await new Promise(resolve=>{const finish=value=>{dialog.close();dialog.remove();resolve(value);};dialog.querySelector("[data-no]").onclick=()=>finish(false);dialog.querySelector("[data-yes]").onclick=()=>finish(true);dialog.oncancel=event=>{event.preventDefault();finish(false);};});
  }
  function businessDiff(version,previous) {
    if(!previous)return "初始版本";
    const a=version.definition,b=previous.definition;
    return "检查项新增 "+a.selected_check_ids.filter(id=>!b.selected_check_ids.includes(id)).length+"，移除 "+b.selected_check_ids.filter(id=>!a.selected_check_ids.includes(id)).length+"；日报版式"+(JSON.stringify(a.daily_report_layout)!==JSON.stringify(b.daily_report_layout) ? "有变化":"未变化")+(a.display_name!==b.display_name ? "；模板名称有变化":"");
  }
  function detailTab() {return new URLSearchParams(location.hash.slice(1)).get("tab") || "content";}
  function setDetailTab(tab) {window.scrollTo(0,0);const params=new URLSearchParams(location.hash.slice(1));params.set("tab",tab);history.replaceState(null,"","#"+params);renderInspectionDetail();}
  async function showInspectionDetail() {
    closeInspectionPreview(false);
    const params=new URLSearchParams(location.hash.slice(1)),id=params.get("template"),seq=++detailSeq;
    if(!id){detailRoot.hidden=true;document.querySelector("main.aiops-template-page").hidden=false;return;}
    detailRoot.hidden=false;document.querySelector("main.aiops-template-page").hidden=true;
    detailRoot.innerHTML='<button class="secondary-button" data-back-templates>返回模板中心</button><p role="status">正在读取模板详情…</p>';
    try{
      const [detail,versions,plans]=await Promise.all(["/"+id,"/"+id+"/versions","/"+id+"/plans"].map(suffix=>KBotAIOpsAuth.request(api+"/inspection-templates"+suffix)));
      if(seq!==detailSeq)return;
      detailData=detail;detailVersions=versions;detailPlans=plans;
      renderInspectionDetail();window.scrollTo(0,0);
    }catch(error){if(seq===detailSeq)detailRoot.innerHTML='<button class="secondary-button" data-back-templates>返回模板中心</button><p class="notice notice-error" role="alert">'+escape(error.message)+'</p>';}
  }
  const previewDrawer=document.getElementById("inspection-template-preview-drawer");
  const previewOutput=previewDrawer.querySelector("#inspection-preview-report");
  let previewVersion=null,previewSequence=0,previewBusy=false,previewActive=false,previewTrigger=null,previewPreviousOverflow="";
  function openInspectionPreview() {
    if(previewActive)return;
    const selected=detailRoot.querySelector("#inspection-version-select")?.value;
    const version=detailVersions.find(item=>item.template_version_id===selected);
    if(!version)return;
    previewVersion=version;
    previewTrigger=document.activeElement;
    previewPreviousOverflow=document.body.style.overflow;
    document.body.style.overflow="hidden";
    previewActive=true;
    document.getElementById("inspection-preview-title").textContent=detailData.display_name+" · v"+version.version_no;
    // 脱离已隐藏的模板目录容器，沿用站内抽屉的顶层模态和主题边界。
    document.body.append(previewDrawer);
    previewDrawer.showModal();
    previewDrawer.querySelector("[data-close-inspection-preview]").focus({preventScroll:true});
    previewDrawer.querySelector(".admin-drawer-body").scrollTop=0;
    void loadInspectionPreview();
  }
  function closeInspectionPreview(restoreFocus=true) {
    if(!previewActive)return;
    previewSequence++;previewActive=false;previewBusy=false;previewVersion=null;
    previewOutput.innerHTML="";
    previewDrawer.querySelector(".admin-drawer-body").setAttribute("aria-busy","false");
    document.body.style.overflow=previewPreviousOverflow;
    const trigger=previewTrigger;previewTrigger=null;
    if(previewDrawer.open)previewDrawer.close();
    if(restoreFocus && trigger?.isConnected)trigger.focus({preventScroll:true});
  }
  async function loadInspectionPreview() {
    if(!previewActive || !previewVersion || previewBusy)return;
    const seq=++previewSequence,definition=previewVersion.definition;
    previewBusy=true;
    const body=previewDrawer.querySelector(".admin-drawer-body");body.setAttribute("aria-busy","true");
    previewOutput.innerHTML='<p role="status">正在生成模拟报告…</p>';
    try {
      const report=await KBotAIOpsAuth.request(api+"/inspection-templates:preview",{method:"POST",body:JSON.stringify({selected_check_ids:definition.selected_check_ids,daily_report_layout:definition.daily_report_layout,database_type:"ORACLE",scenario:"ATTENTION"})});
      if(!report.daily_inspection)throw new Error(report.error?.message || "模拟报告未返回有效内容");
      if(seq===previewSequence && previewDrawer.open)previewOutput.innerHTML=KBotDailyInspectionReport.render(report.daily_inspection);
    } catch(error) {
      if(seq===previewSequence && previewDrawer.open)previewOutput.innerHTML='<p class="notice notice-error" role="alert">'+escape(error.message)+'</p><button class="secondary-button" data-retry-preview>重试预览</button>';
    } finally {
      if(seq===previewSequence){
        previewBusy=false;body.setAttribute("aria-busy","false");
        if(previewDrawer.open && !previewDrawer.contains(document.activeElement))previewDrawer.querySelector("[data-close-inspection-preview]").focus({preventScroll:true});
      }
    }
  }
  previewDrawer.querySelector("[data-close-inspection-preview]").onclick=()=>closeInspectionPreview();
  previewDrawer.addEventListener("cancel",event=>{event.preventDefault();closeInspectionPreview();});
  previewDrawer.addEventListener("close",()=>{if(!previewDrawer.open)closeInspectionPreview();});
  previewDrawer.addEventListener("keydown",event=>{
    if(event.key!=="Tab")return;
    const items=[...previewDrawer.querySelectorAll('button:not([disabled]),a[href],summary,input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])')].filter(item=>item.tabIndex>=0 && item.getClientRects().length);
    const first=items[0],last=items.at(-1);
    if(event.shiftKey && document.activeElement===first){event.preventDefault();last?.focus();}
    else if(!event.shiftKey && document.activeElement===last){event.preventDefault();first?.focus();}
  });
  previewDrawer.addEventListener("click",event=>{
    if(event.target===previewDrawer){
      const box=previewDrawer.getBoundingClientRect();
      if(event.clientX<box.left || event.clientX>box.right || event.clientY<box.top || event.clientY>box.bottom)closeInspectionPreview();
    }
    if(event.target.closest("[data-retry-preview]")){void loadInspectionPreview();return;}
    const anchor=event.target.closest('a[href^="#"]');
    if(anchor){event.preventDefault();previewDrawer.querySelector("#"+CSS.escape(anchor.getAttribute("href").slice(1)))?.scrollIntoView({block:"start"});}
  });
  function renderInspectionDetail() {
    if(!detailData)return;
    const tab=detailTab(),custom=detailData.source_kind==="CUSTOM";
    detailRoot.innerHTML='<header class="ops-head"><div><p class="eyebrow">巡检模板详情 · 只读发布内容</p><h1>'+escape(detailData.display_name)+'</h1></div><button class="secondary-button" data-back-templates>返回模板中心</button></header><p class="notice">'+(custom ? "自定义模板 · 草稿不影响计划，旧计划不会自动升级。":"系统内置 · 只读，需要定制时请新建自定义模板。")+'</p><nav class="segmented-control" aria-label="模板详情页签">'+[["content","模板内容"],["versions","版本历史"],["plans","引用计划"]].map(([id,label])=>'<button class="'+(tab===id?"active":"")+'" data-detail-tab="'+id+'">'+label+'</button>').join("")+'</nav><section class="daily-editor-panel" id="inspection-detail-content"></section>';
    const panel=detailRoot.querySelector("#inspection-detail-content");
    if(tab==="versions") {
      panel.innerHTML='<p>历史只读；基于旧版本编辑只替换工作草稿，不改变旧计划。</p>'+detailVersions.map((version,index)=>'<article class="daily-publish-review"><h2>v'+escape(version.version_no)+(version.current ? " · 最新发布":"")+'</h2><p>发布于 '+escape(shell.fmt(version.created_at))+' · 发布人：'+escape(version.created_by)+'</p><p>'+escape(businessDiff(version,detailVersions[index+1]))+'</p><button class="row-action" data-view-version="'+escape(version.template_version_id)+'">查看此版本</button>'+(custom && detailData.template_kind==="DAILY" ? '<button class="row-action" data-edit-version="'+escape(version.template_version_id)+'">基于此版本编辑</button>':"")+'<details><summary>技术定义</summary><pre>'+escape(JSON.stringify(version.definition,null,2))+'</pre></details></article>').join("")+(detailVersions.length?"":"<p>尚无发布版本。</p>");
    }else if(tab==="plans") {
      panel.innerHTML='<p>默认切换、停用与发布均不改变既有计划绑定。升级请在计划编辑页明确确认。</p><ul>'+detailPlans.map(plan=>'<li>'+escape(plan.display_name)+' · v'+escape(plan.template_version_no)+' · '+(plan.is_latest ? "已绑定最新发布版本":"保留原版本")+'</li>').join("")+'</ul>'+(detailPlans.length?"":"<p>暂无引用计划。</p>")+'<a class="row-action" href="./inspection-plans.html">前往巡检计划管理</a>';
    }else {
      if(!detailVersions.length){panel.innerHTML="<p>尚未发布，请返回模板中心继续编辑草稿。</p>";return;}
      panel.classList.add("template-content-panel");
      panel.innerHTML='<header class="template-content-head"><div id="inspection-version-heading"></div><label class="control-field template-version-control"><span class="field-label">查看发布版本</span><select id="inspection-version-select">'+detailVersions.map(version=>'<option value="'+escape(version.template_version_id)+'">v'+escape(version.version_no)+(version.current ? " · 最新发布":" · 历史只读")+'</option>').join("")+'</select></label></header><div id="inspection-readonly-content"></div>';
      panel.querySelector("select").onchange=renderReadonlyVersion;renderReadonlyVersion(new URLSearchParams(location.hash.slice(1)).get("version") || undefined);
    }
  }
  function renderReadonlyVersion(id) {
    const select=detailRoot.querySelector("#inspection-version-select");
    if(typeof id==="string")select.value=id;
    const version=detailVersions.find(item=>item.template_version_id===select.value),definition=version.definition;
    const sections=definition.daily_report_layout?.sections || [],topics=selectedTopics(checkCatalog,definition.selected_check_ids);
    detailRoot.querySelector("#inspection-version-heading").innerHTML='<p class="eyebrow">已发布模板</p><h2>'+escape(definition.daily_report_layout?.title || detailData.display_name)+'</h2><p class="daily-editor-muted">共享版式 · 按数据库类型适配检查</p>';
    detailRoot.querySelector("#inspection-readonly-content").innerHTML='<div class="template-readonly-grid">'+(sections.length ? '<aside class="template-directory"><h3>报告目录</h3><ol>'+sections.map(section=>'<li>'+escape(checkCatalog.report_sections.find(item=>item.section_id===section.section_id)?.title || section.section_id)+'</li>').join("")+'</ol></aside>' : "")+'<section class="template-configured-checks"><header class="template-checks-head"><h3>已配置检查</h3><p class="daily-editor-muted">按业务主题归并同名检查，不重复罗列数据库差异。</p></header><div class="template-check-topics">'+topics.map(topic=>'<section class="template-check-topic"><h4><span>'+escape(topic.title)+'</span><small>'+topic.checks.length+' 项</small></h4><ul>'+topic.checks.map(name=>'<li>'+escape(name)+'</li>').join("")+'</ul></section>').join("")+'</div>'+(!topics.length ? '<p class="daily-editor-muted">此版本未配置检查项。</p>' : "")+'</section></div>'+(detailData.template_kind==="DAILY" ? '<footer class="template-preview-action"><p class="daily-editor-muted">查看模拟样例，核对报告排版；不会执行巡检。</p><button class="secondary-button" data-preview-inspection>查看报告效果（模拟样例）</button></footer>':"");
  }
  detailRoot.onclick=async event=>{
    const button=event.target.closest("button");if(!button)return;
    if(button.hasAttribute("data-back-templates")){location.hash="";return;}
    if(button.dataset.detailTab){setDetailTab(button.dataset.detailTab);return;}
    if(button.dataset.viewVersion){setDetailTab("content");renderReadonlyVersion(button.dataset.viewVersion);return;}
    if(button.dataset.editVersion){
      if(actionBusy)return;
      if(detailData.has_draft && !await confirmTemplate("替换工作草稿","将用所选历史版本替换当前草稿；不修改发布历史和旧计划。"))return;
      setActionBusy(true);button.disabled=true;
      try{
        const version=detailVersions.find(item=>item.template_version_id===button.dataset.editVersion);
        const saved=await KBotAIOpsAuth.request(api+"/inspection-templates/"+detailData.inspection_template_id+"/draft",{method:"PUT",body:JSON.stringify({expected_row_version:detailData.row_version,display_name:version.definition.display_name,selected_check_ids:version.definition.selected_check_ids,daily_report_layout:version.definition.daily_report_layout})});
        detailRoot.hidden=true;KBotDailyTemplateEditor.open(checkCatalog,saved,loadTemplates);
      }catch(error){shell.toast(error.message);}finally{setActionBusy(false);button.disabled=false;}
    }
    if(button.hasAttribute("data-preview-inspection")){
      openInspectionPreview();
    }
  };
  globalThis.KBotInspectionTemplates=Object.freeze({confirm:confirmTemplate,hideDetail:()=>{closeInspectionPreview(false);detailSeq++;detailRoot.hidden=true;}});
  window.addEventListener("hashchange",showInspectionDetail);

  function bindEditor(kind) {
    const dialog = document.getElementById("session-report-template-dialog");
    const form = dialog.querySelector("form");
    dialog.querySelectorAll("[data-close-session-report-template]").forEach(button => { button.onclick = () => { if (!dialog.dataset.busy) dialog.close(); }; });
    form.addEventListener("input", () => updateSaveButton(form, kind));
    form.addEventListener("change", event => {
      if (event.target.name === "section") {
        if (event.target.checked) selectedSections.push(event.target.value);
        else selectedSections = selectedSections.filter(kind => kind !== event.target.value);
        renderOutline();
      }
      updateSaveButton(form, kind);
    });
    form.addEventListener("click", event => {
      const button = event.target.closest("[data-move-section]");
      if (!button || dialog.dataset.busy) return;
      const movable = selectedSections.filter(kind => !requiredSections.has(kind));
      const index = movable.indexOf(button.dataset.moveSection), target = index + Number(button.dataset.offset);
      if (index < 0 || target < 0 || target >= movable.length) return;
      [movable[index], movable[target]] = [movable[target], movable[index]];
      selectedSections = ["EXECUTIVE_SUMMARY", ...movable, "EVIDENCE_BOUNDARY"];
      renderOutline();
      // 重绘后恢复键盘焦点，连续上移/下移仍可操作同一章节。
      form.querySelector(`[data-move-section="${button.dataset.moveSection}"][data-offset="${button.dataset.offset}"]`)?.focus();
    });
    form.onsubmit = async event => {
      event.preventDefault();
      if (dialog.dataset.busy || !canSave(form, kind)) return;
      const editing = editingSessionTemplate;
      const selected = { definition: { schema_version: "SESSION_REPORT_TEMPLATE.v1", sections: orderedSections().map(kind => ({ kind })) } };
      const id = editing?.template_id;
      const path = api+"/session-report-templates"+(id ? "/"+encodeURIComponent(id)+"/versions":"");
      const body = editing ? { expected_row_version: editing.row_version, ...selected } : { display_name: form.elements.display_name.value.trim(), ...selected };
      const result = form.querySelector('[role="status"]'); result.hidden = true;
      dialog.dataset.busy = "true";
      const controls = [...dialog.querySelectorAll("button,input")].map(node => [node, node.disabled]);
      controls.forEach(([node]) => { node.disabled = true; });
      updateSaveButton(form, kind);
      try {
        await KBotAIOpsAuth.request(path, { method: "POST", headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() }, body: JSON.stringify(body) });
        dialog.close(); shell.toast("模板已保存并冻结新版本"); await loadTemplates();
      } catch (error) { result.textContent = error.message; result.hidden = false; }
      finally {
        delete dialog.dataset.busy;
        controls.forEach(([node, disabled]) => { node.disabled = disabled; });
        updateSaveButton(form, kind);
      }
    };
  }
  shell.ready.then(async () => {
    document.getElementById("template-search").oninput = () => { renderInspectionTemplates(); renderSessionTemplates(); };
    document.getElementById("refresh-templates").onclick = loadTemplates;
    document.getElementById("create-inspection-template").onclick = () => openEditor("inspection");
    document.getElementById("create-session-report-template").onclick = () => openEditor("session");
    document.getElementById("close-template-history").onclick = () => document.getElementById("template-history-drawer").close();
    document.querySelectorAll("[data-template-tab]").forEach(button => { button.onclick = () => {
      document.querySelectorAll("[data-template-tab]").forEach(tab => { tab.classList.toggle("active", tab === button); tab.setAttribute("aria-selected", String(tab === button)); });
      document.getElementById("inspection-template-catalog").hidden = button.dataset.templateTab !== "inspection";
      document.getElementById("session-template-catalog").hidden = button.dataset.templateTab !== "session";
      closeMenus();
    }; });
    document.getElementById("ops-inspection-template-body").onclick = async event => {
      const action=event.target.closest("button");if(!action || actionBusy)return;
      if(action.matches(".management-catalog-more")){
        const root=action.closest(".management-catalog-overflow"),open=action.getAttribute("aria-expanded")!=="true";
        closeMenus(root);action.setAttribute("aria-expanded",String(open));root.querySelector(".management-catalog-menu").hidden=!open;return;
      }
      closeMenus();
      const item=inspectionTemplates.find(item=>Object.values(action.dataset).includes(item.inspection_template_id));
      if(!item)return;
      if(action.dataset.editInspectionTemplate){await openVersion("inspection",item.inspection_template_id);return;}
      if(action.dataset.viewInspection){location.hash=new URLSearchParams({template:item.inspection_template_id,tab:"content"});return;}
      if(action.dataset.deleteInspection){
        setActionBusy(true);
        try {
          if(await confirmTemplate("永久删除模板","将永久删除“"+item.display_name+"”及其全部发布版本和草稿，无法恢复。仅允许删除已停用且无计划或历史批次引用的自定义模板。","永久删除",true)) {
            await KBotAIOpsAuth.request(api+"/inspection-templates/"+item.inspection_template_id,{method:"DELETE",body:JSON.stringify({expected_row_version:item.row_version})});
            shell.toast("模板及其全部版本已永久删除");await loadTemplates();
          }
        } catch(error) {shell.toast(error.message);
        } finally {setActionBusy(false);}
        return;
      }
      if(action.dataset.toggleInspection)await mutateTemplate("/inspection-templates/"+item.inspection_template_id,"PATCH",{expected_row_version:item.row_version,status:item.status==="ACTIVE"?"INACTIVE":"ACTIVE"});
      if(action.dataset.defaultInspection)await mutateTemplate("/inspection-templates/"+item.inspection_template_id+":default","POST",{expected_row_version:item.row_version});
      if(action.dataset.discardInspection && await confirmTemplate("放弃草稿",item.version_no ? "只丢弃未发布修改，保留全部版本和计划。":"删除尚未发布的模板草稿。"))await mutateTemplate("/inspection-templates/"+item.inspection_template_id+"/draft","DELETE",{expected_row_version:item.row_version});
    };
    document.getElementById("ops-session-report-template-body").onclick = event => {
      const button = event.target.closest("button"); if (!button) return;
      if (button.matches(".management-catalog-more")) {
        const root = button.closest(".management-catalog-overflow");
        const open = button.getAttribute("aria-expanded") !== "true";
        closeMenus(root); button.setAttribute("aria-expanded", String(open)); root.querySelector(".management-catalog-menu").hidden = !open;
        return;
      }
      closeMenus();
      if (button.dataset.editSessionTemplate) openVersion("session", button.dataset.editSessionTemplate);
      if (button.dataset.historySessionTemplate) showSessionTemplateHistory(button.dataset.historySessionTemplate);
      if (button.dataset.copySessionTemplate) openEditor("session", null, sessionTemplates.find(item => item.template_ref === button.dataset.copySessionTemplate));
      if (button.dataset.defaultSessionTemplate) mutateTemplate("/session-report-templates/default", "PUT", { template_ref: button.dataset.defaultSessionTemplate });
      if (button.dataset.toggleSessionTemplate) mutateTemplate(`/session-report-templates/${encodeURIComponent(button.dataset.toggleSessionTemplate)}`, "PATCH", { expected_row_version: Number(button.dataset.rowVersion), status: button.dataset.nextStatus });
    };
    document.addEventListener("pointerdown", event => { if (!event.target.closest(".management-catalog-overflow")) closeMenus(); });
    document.addEventListener("keydown", event => { if (event.key === "Escape") closeMenus(); });
    bindEditor("session");
    await loadTemplates();
  });
})();
