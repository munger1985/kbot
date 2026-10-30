(function () {
  "use strict";
  if (document.body.dataset.page !== "reports") return;
  const api = "/api/v1/apps/aiops", shell = globalThis.KBotAIOpsShell, esc = shell.escape;
  const request = (path, init) => KBotAIOpsAuth.request(api + path, init);
  const rows = value => Array.isArray(value) ? value : [];
  const state = {reports:[],targets:[],templates:[],filter:"",downloading:"",mode:"",busy:false,draft:null,sequence:0};
  const reportForm = document.getElementById("report-generate-form"), importForm = document.getElementById("report-import-form");
  const workloadTargets = () => state.targets.filter(item => ["ORACLE","MYSQL","POSTGRESQL"].includes(item.db_type));
  const pgTargets = () => state.targets.filter(item => item.db_type === "POSTGRESQL");
  const reportTypeLabel = value => String(value || "").startsWith("INSPECTION_") ? "日常巡检" : ({ORACLE_WORKLOAD:"Oracle 周期快照（非 AWR）",ORACLE_WORKLOAD_DIFF:"Oracle 快照对比（非 AWR）",INCIDENT:"故障诊断",PERFORMANCE:"性能诊断",COMPARISON:"前后对比",MYSQL_WORKLOAD:"MySQL Workload",MYSQL_WORKLOAD_DIFF:"MySQL Workload Diff",MYSQL_ACTIVITY:"MySQL Activity",MYSQL_SQL_HEALTHCHECK:"MySQL SQL Healthcheck",POSTGRESQL_WORKLOAD:"PostgreSQL Workload",POSTGRESQL_WORKLOAD_DIFF:"PostgreSQL Workload Diff",POSTGRESQL_ACTIVITY:"PostgreSQL Activity",POSTGRESQL_SQL_HEALTHCHECK:"PostgreSQL SQL Healthcheck",POSTGRESQL_PGBADGER:"PostgreSQL pgBadger"})[value] || value || "";
  const statusLabel = value => ({ACTIVE:"进行中",QUEUED:"等待执行",COMPLETED:"已完成",FAILED:"失败",CANCELLED:"已取消",SKIPPED:"已跳过",PENDING:"等待处理",PUBLISHED:"已发布",READY:"结论就绪",PARTIAL:"部分完成",GENERATING:"生成中",RUNNING:"运行中"})[value] || value;
  const statusClass = value => ["FAILED","CANCELLED"].includes(value) ? "status-danger" : ["PENDING","RUNNING","ACTIVE","PARTIAL","GENERATING"].includes(value) ? "status-warning" : "status-success";
  function localTime(offset) {const date = new Date(Date.now() + offset * 60000);date.setMinutes(date.getMinutes() - date.getTimezoneOffset());return date.toISOString().slice(0,16);}
  function dateField(name,label) {
    const value = state.draft[name];
    return `<label class="control-field"><span class="field-label">${label}</span><span class="date-time-picker-control${value ? "" : " is-empty"}"><span class="date-time-picker-display" aria-hidden="true">${esc(value ? shell.fmt(value) : "请选择日期和时间")}</span><input class="date-time-picker-native" name="${name}" value="${esc(value)}" type="datetime-local" required><span class="date-time-picker-icon" aria-hidden="true"></span></span></label>`;
  }
  function updateDate(input) {state.draft[input.name] = input.value;input.previousElementSibling.textContent = input.value ? shell.fmt(input.value) : "请选择日期和时间";input.parentElement.classList.toggle("is-empty",!input.value);}
  function renderList() {
    const items = state.filter ? state.reports.filter(item => item.template_ref === state.filter) : state.reports;
    document.getElementById("report-count").textContent = `${items.length} 份`;
    document.getElementById("reports-body").innerHTML = items.length ? items.map(item => {
      const name = state.templates.find(template => template.template_ref === item.template_ref)?.display_name || String(item.template_ref || "").replace(/^system:/,"");
      return `<tr><td><strong class="table-primary">${esc(item.title)}</strong><small class="table-secondary">版本 ${esc(item.report_version)} · ${esc(item.report_key)}</small></td><td>${esc(reportTypeLabel(item.report_type))}</td><td>${esc(name)}<small class="table-secondary">模板 v${esc(item.template_version)}</small></td><td><span class="status-pill ${statusClass(item.status)}">${esc(statusLabel(item.status))}</span></td><td>${esc(shell.fmt(item.published_at))}</td><td class="table-summary">${esc(item.summary)}</td><td><div class="button-row"><a class="row-action" href="./report-detail.html?id=${encodeURIComponent(item.report_id)}">查看报告</a><button class="row-action" type="button" data-download-report="${esc(item.report_id)}"${state.downloading ? " disabled" : ""}>${state.downloading === item.report_id ? "下载中…" : "下载 PDF"}</button></div></td></tr>`;
    }).join("") : `<tr><td colspan="7" class="empty-cell">${state.reports.length ? "没有符合当前模板筛选的报告。" : "当前业务空间还没有已生成的运维报告。"}</td></tr>`;
  }
  function updateControls() {
    document.getElementById("create-report").disabled = !workloadTargets().length || Boolean(state.mode);
    document.getElementById("import-pgbadger").disabled = !pgTargets().length || Boolean(state.mode);
    for (const id of ["report-generate-drawer","report-import-drawer"]) {
      const drawer = document.getElementById(id);
      drawer.querySelectorAll("[data-close-report]").forEach(button => {button.disabled = state.busy;});
      const submit = drawer.querySelector('[type="submit"]');if (!submit) continue;
      submit.disabled = state.busy || !state.draft?.target_id || (state.mode === "import" && !state.draft.file);
      submit.textContent = state.mode === "import" ? state.busy ? "导入中…" : "导入报告" : state.busy ? "生成中…" : "生成报告";
    }
  }
  function syncReportType() {
    const db = workloadTargets().find(item => item.target_id === state.draft.target_id)?.db_type || "MYSQL";
    const suffix = String(state.draft.report_type || "").replace(/^(ORACLE|MYSQL|POSTGRESQL)_/,"");
    state.draft.report_type = `${db}_${db === "ORACLE" && suffix === "ACTIVITY" ? "WORKLOAD" : suffix || "WORKLOAD"}`;
  }
  function renderForm() {
    const draft = state.draft;
    if (state.mode === "import") {
      importForm.innerHTML = `<label class="control-field form-span"><span class="field-label">PostgreSQL 目标</span><select name="target_id" required>${pgTargets().map(target => `<option value="${esc(target.target_id)}">${esc(target.display_name)}</option>`).join("")}</select></label>${dateField("period_start","报告开始")}${dateField("period_end","报告结束")}<label class="control-field form-span"><span class="field-label">报告文件</span><input name="file" required type="file" accept=".json,.html,.htm,.gz,application/json,text/html,application/gzip"><small class="field-help">压缩文件最大 20 MiB，解压后最大 100 MiB；禁止脚本、嵌入对象和外部资源。</small></label><footer class="admin-drawer-footer form-span"><button class="secondary-button" type="button" data-close-report>取消</button><button class="primary-button" type="submit">导入报告</button></footer>`;
      importForm.elements.target_id.value = draft.target_id;
    } else {
      const db = workloadTargets().find(item => item.target_id === draft.target_id)?.db_type || "MYSQL";
      reportForm.innerHTML = `<label class="control-field"><span class="field-label">数据库目标</span><select name="target_id" required>${workloadTargets().map(target => `<option value="${esc(target.target_id)}">${esc(target.display_name)} · ${esc(target.db_type)}</option>`).join("")}</select><small class="field-help">列出当前业务空间中的 Oracle、MySQL 与 PostgreSQL Target；Oracle 工作负载来自平台周期快照，不是 AWR。</small></label><label class="control-field"><span class="field-label">报告类型</span><select name="report_type"><option value="${db}_WORKLOAD">Workload Report</option><option value="${db}_WORKLOAD_DIFF">Workload Diff Report</option>${db !== "ORACLE" ? `<option value="${db}_ACTIVITY">Activity Report</option>` : ""}</select></label>${draft.report_type.endsWith("_WORKLOAD_DIFF") ? dateField("baseline_start","基线开始") + dateField("baseline_end","基线结束") + dateField("after_start","对比开始") + dateField("after_end","对比结束") : dateField("period_start","窗口开始") + dateField("period_end","窗口结束")}${draft.report_type.endsWith("_ACTIVITY") ? '<p class="field-help form-span">该报告来自 KBot 周期活动采样，不冒充数据库原生 ASH。</p>' : ""}<footer class="admin-drawer-footer form-span"><button class="secondary-button" type="button" data-close-report>取消</button><button class="primary-button" type="submit">生成报告</button></footer>`;
      reportForm.elements.target_id.value = draft.target_id;reportForm.elements.report_type.value = draft.report_type;
    }
    updateControls();
  }
  function open(mode) {
    if (state.mode || state.busy) return;
    state.mode = mode;state.draft = {target_id:(mode === "import" ? pgTargets() : workloadTargets())[0]?.target_id || "",report_type:"MYSQL_WORKLOAD",period_start:localTime(-60),period_end:localTime(0),baseline_start:localTime(-120),baseline_end:localTime(-60),after_start:localTime(-60),after_end:localTime(0),file:null};
    if (mode === "generate") syncReportType();renderForm();
    document.getElementById(mode === "import" ? "report-import-drawer" : "report-generate-drawer").showModal();
  }
  function close() {
    if (state.busy) return;
    state.mode = "";state.draft = null;
    for (const id of ["report-generate-drawer","report-import-drawer"]) {const drawer = document.getElementById(id);if (drawer.open) drawer.close();}
    updateControls();
  }
  function utc(value) {const date = new Date(value);if (!Number.isFinite(date.getTime())) throw new Error("请选择有效的报告时间");return date.toISOString();}
  async function submit(event) {
    event.preventDefault();if (state.busy || !state.draft?.target_id || (state.mode === "import" && !state.draft.file)) return;
    const mode = state.mode, draft = state.draft;state.busy = true;updateControls();
    try {
      if (mode === "import") {
        if (draft.file.size > 20 * 1024 * 1024) throw new Error("pgBadger 上传文件不能超过 20 MiB");
        const query = new URLSearchParams({target_id:draft.target_id,period_start:utc(draft.period_start),period_end:utc(draft.period_end)});
        await request(`/postgresql/pgbadger-artifacts?${query}`,{method:"POST",headers:{"Content-Type":draft.file.type || "application/octet-stream","Idempotency-Key":KBotAIOpsAuth.uuid(),"X-File-Name":encodeURIComponent(draft.file.name)},body:draft.file});
      } else {
        const diff = draft.report_type.endsWith("_WORKLOAD_DIFF"), activity = draft.report_type.endsWith("_ACTIVITY");
        const payload = {target_id:draft.target_id};
        for (const key of diff ? ["baseline_start","baseline_end","after_start","after_end"] : ["period_start","period_end"]) payload[key] = utc(draft[key]);
        await request(`/reports/${diff ? "workload-diff" : activity ? "activity" : "workload"}`,{method:"POST",headers:{"Idempotency-Key":KBotAIOpsAuth.uuid()},body:JSON.stringify(payload)});
      }
      state.busy = false;close();shell.toast(mode === "import" ? "PostgreSQL pgBadger 报告已导入" : `${reportTypeLabel(draft.report_type)} 已生成`);await load();
    } catch (error) {shell.toast(error.message || (mode === "import" ? "无法导入 pgBadger 报告" : "无法生成数据库运维报告"));}
    finally {state.busy = false;updateControls();}
  }
  async function load() {
    const sequence = ++state.sequence;document.getElementById("reports-loading").hidden = false;document.getElementById("reports-table").hidden = true;
    try {
      const [reports,targets,templates] = await Promise.all([request("/reports?limit=100"),request("/targets?limit=200"),request("/session-report-templates")]);
      if (sequence !== state.sequence) return;state.reports = rows(reports.items);state.targets = rows(targets.items);state.templates = rows(templates);
      const filter = document.getElementById("report-template-filter");filter.innerHTML = '<option value="">全部模板</option>' + state.templates.map(item => `<option value="${esc(item.template_ref)}">${esc(item.display_name)}</option>`).join("");filter.value = state.filter;
    } catch (error) {if (sequence === state.sequence) shell.toast(error.message || "无法读取运维报告");}
    finally {if (sequence === state.sequence) {renderList();updateControls();document.getElementById("reports-loading").hidden = true;document.getElementById("reports-table").hidden = false;}}
  }
  shell.ready.then(() => {
    document.getElementById("refresh-reports").onclick = load;document.getElementById("create-report").onclick = () => open("generate");document.getElementById("import-pgbadger").onclick = () => open("import");
    document.getElementById("report-template-filter").onchange = event => {state.filter = event.target.value;renderList();};
    document.getElementById("reports-body").onclick = async event => {
      const button = event.target.closest("[data-download-report]");if (!button || state.downloading) return;
      const item = state.reports.find(item => item.report_id === button.dataset.downloadReport);if (!item) return;
      state.downloading = item.report_id;renderList();
      try {await KBotAIOpsAuth.download(`${api}/reports/${encodeURIComponent(item.report_id)}/pdf`,`aiops-report-${item.report_id}.pdf`);shell.toast(`《${item.title}》PDF 已开始下载`);}
      catch (error) {shell.toast(error.message || "无法下载正式报告 PDF");}
      finally {state.downloading = "";renderList();}
    };
    for (const form of [reportForm,importForm]) {
      form.onsubmit = submit;
      form.oninput = event => {if (event.target.classList.contains("date-time-picker-native")) updateDate(event.target);};
      form.onchange = event => {
        const input = event.target;if (input.classList.contains("date-time-picker-native")) {updateDate(input);input.blur();}
        else if (input.name === "file") {state.draft.file = input.files?.[0] || null;updateControls();}
        else if (input.name === "target_id") {state.draft.target_id = input.value;if (state.mode === "generate") {syncReportType();renderForm();}}
        else if (input.name === "report_type") {state.draft.report_type = input.value;renderForm();}
      };
    }
    for (const id of ["report-generate-drawer","report-import-drawer"]) {
      const drawer = document.getElementById(id);drawer.onclick = event => {if (event.target.closest("[data-close-report]")) close();};
      drawer.addEventListener("cancel",event => {if (state.busy) event.preventDefault();});
      drawer.addEventListener("close",() => {if (state.mode && !drawer.open) close();});
    }
    return load();
  }).catch(error => shell.toast(error.message));
})();
