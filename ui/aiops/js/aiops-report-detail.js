(function () {
  "use strict";
  if (document.body.dataset.page !== "report-detail") return;
  const shell = globalThis.KBotAIOpsShell, esc = shell.escape, api = "/api/v1/apps/aiops";
  const request = (path,init) => KBotAIOpsAuth.request(api + path,init), rows = value => Array.isArray(value) ? value : [];
  const reportId = new URLSearchParams(location.search).get("id") || "";
  const origin = new URLSearchParams(location.search);
  const inspectionFireId = origin.get("inspectionFireId");
  if (inspectionFireId) {
    const back = document.querySelector('a[href="./reports.html"]');
    back.href = `./inspection-detail.html?id=${encodeURIComponent(inspectionFireId)}&runId=${encodeURIComponent(origin.get("runId") || "")}`;
    back.textContent = "← 返回巡检批次";
  }
  const root = document.getElementById("formal-report");
  const state = {report:null,presentation:null,target:null,eligibility:null,pdfBusy:false,rawBusy:false,extracting:false,observer:null};
  const statusLabel = value => ({PUBLISHED:"已发布",READY:"结论就绪",PARTIAL:"证据不完整",GENERATING:"生成中",FAILED:"生成失败"})[value] || value;
  const statusClass = value => value === "FAILED" ? "status-danger" : ["PARTIAL","GENERATING"].includes(value) ? "status-warning" : "status-success";
  const riskLabel = value => ({CRITICAL:"严重",HIGH:"高",MEDIUM:"中",LOW:"低",INFO:"提示"})[value] || value || "低";
  const riskClass = value => ["CRITICAL","HIGH"].includes(value) ? "status-danger" : value === "MEDIUM" ? "status-warning" : "status-success";
  const healthLabel = value => ({NORMAL:"正常",ATTENTION:"需关注",ABNORMAL:"异常",INCOMPLETE:"检查未完成",INFO:"提示",LOW:"低风险",MEDIUM:"中风险",HIGH:"高风险",CRITICAL:"严重风险",UNKNOWN:"无法判定"})[value] || value;
  const typeLabel = value => String(value).startsWith("INSPECTION_") ? "日常巡检" : ({INCIDENT:"故障诊断",PERFORMANCE:"性能诊断",COMPARISON:"前后对比",ORACLE_WORKLOAD:"Oracle 周期快照（非 AWR）",ORACLE_WORKLOAD_DIFF:"Oracle 快照对比（非 AWR）",MYSQL_WORKLOAD:"MySQL Workload Report",MYSQL_WORKLOAD_DIFF:"MySQL Workload Diff Report",MYSQL_ACTIVITY:"MySQL Activity Report",MYSQL_SQL_HEALTHCHECK:"MySQL SQL Healthcheck",POSTGRESQL_WORKLOAD:"PostgreSQL Workload Report",POSTGRESQL_WORKLOAD_DIFF:"PostgreSQL Workload Diff Report",POSTGRESQL_ACTIVITY:"PostgreSQL Activity Report",POSTGRESQL_SQL_HEALTHCHECK:"PostgreSQL SQL Healthcheck",POSTGRESQL_PGBADGER:"PostgreSQL pgBadger"})[value] || value;
  const sectionLabel = value => ({EXECUTIVE_SUMMARY:"执行摘要",SCOPE:"报告范围",ALERT_TIMELINE:"告警时间线",INSPECTION_COVERAGE:"巡检覆盖情况",RISK_OVERVIEW:"风险概览",TREND:"趋势分析",ROOT_CAUSE:"根因判断",FINDINGS:"已核验发现",RECOMMENDATIONS:"建议",ACTIONS:"处置与验证",EVIDENCE_BOUNDARY:"证据边界",EVIDENCE_APPENDIX:"证据索引"})[value] || value;
  const badge = (value,label,tone=statusClass(value)) => `<span class="status-pill ${tone}">${esc(label || statusLabel(value))}</span>`;
  const markdown = value => `<div class="agent-markdown-content">${KBotMarkdown.render(String(value || ""))}</div>`;
  const item = (label,value,cls="",ddClass="") => `<div${cls ? ` class="${cls}"` : ""}><dt>${esc(label)}</dt><dd${ddClass ? ` class="${ddClass}"` : ""}>${value}</dd></div>`;
  const list = (items,empty="",cls="daily-notes",tag="ol") => `<${tag} class="${cls}">${rows(items).length ? items.map(value => `<li>${esc(value)}</li>`).join("") : empty ? `<li>${esc(empty)}</li>` : ""}</${tag}>`;
  const percent = value => typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(2)}%` : "基线不足";
  const coverage = value => `${value ?? "未记录"}${value == null ? "" : "%"}`;
  const dailySection = (kicker,title,description,body) => `<section class="daily-report-section"><header><div><p class="daily-section-kicker">${kicker}</p><h2>${title}</h2></div>${description ? `<p>${description}</p>` : ""}</header>${body}</section>`;
  const dailyTable = (head,body,cls="") => `<div class="daily-table-wrap"><table class="daily-report-table${cls ? ` ${cls}` : ""}"><thead><tr>${head.map(label => `<th>${label}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`;
  const metricName = (metric,fallback=true) => `<td data-label="指标 / 范围"><strong>${esc(metric.display_name)}</strong><small>${esc(metric.dimension_label || (fallback ? "全部范围" : ""))}</small></td>`;
  const evidence = boundary => `<section class="daily-evidence"><h2>证据边界</h2><p>${esc(boundary?.methodology)}</p>${rows(boundary?.sources).length ? `<ol>${boundary.sources.map(value => `<li>${esc(value)}</li>`).join("")}</ol>` : ""}</section>`;
  const footer = (header,monthly=false) => `<footer class="daily-report-footer"><span>KBot AIOps · 正式${monthly ? "月度" : ""}运维报告</span><span>${esc(header.report_number)} · v${esc(header.report_version)}</span></footer>`;
  function trendGeometry(values,width,height,baseline=false) {
    if (values.length < 2) return null;
    const low = baseline ? Math.min(85,...values) : Math.min(...values), high = baseline ? Math.max(115,...values) : Math.max(...values), span = Math.max(baseline ? 1 : 1e-9,high-low), pad = baseline ? 5 : 4;
    const points = values.map((value,index) => ({x:index*width/(values.length-1),y:height-pad-((value-low)/span)*(height-2*pad)}));
    return {points:points.map(point => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "),last:points.at(-1),baselineY:height-pad-((100-low)/span)*(height-2*pad)};
  }
  function dailyTrendSvg(metric) {
    const geometry = trendGeometry(rows(metric.trend_points).map(point => Number(point.baseline_index)).filter(Number.isFinite),184,68,true);
    return geometry ? `<svg class="daily-trend-svg" viewBox="0 0 184 68" role="img" aria-label="${esc(metric.display_name)}七日趋势"><line x1="0" y1="${geometry.baselineY}" x2="184" y2="${geometry.baselineY}"></line><polyline points="${geometry.points}"></polyline><circle cx="${geometry.last.x}" cy="${geometry.last.y}" r="2.8"></circle></svg>` : '<span class="daily-empty">基线不足</span>';
  }
  function monthlyTrendSvg(metric) {
    const dates = rows(metric.daily_points).map(point => Date.parse(point.date));
    if (dates.some((date,index) => !Number.isFinite(date) || (index > 0 && date - dates[index-1] !== 86400000))) return '<span class="daily-no-baseline">日期有缺测，未连接趋势线</span>';
    const geometry = trendGeometry(rows(metric.daily_points).map(point => Number(point.value)).filter(Number.isFinite),150,42);
    return geometry ? `<svg class="daily-trend-svg monthly-trend-svg" viewBox="0 0 150 42"><polyline points="${geometry.points}"></polyline></svg>` : '<span class="daily-no-baseline">有效天数不足</span>';
  }
  function dailyTargetName(header) {
    const frozenName = String(header.target_name || "").trim(), targetId = String(header.target_id || state.report.target_id || "").trim();
    const frozenTargetIsIdentifier = !frozenName || frozenName === targetId || /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(frozenName);
    return frozenTargetIsIdentifier ? state.target?.display_name || "目标已删除或无权访问" : frozenName;
  }
  const healthTone = value => value === "NORMAL" ? "status-success" : value === "ABNORMAL" ? "status-danger" : value === "ATTENTION" ? "status-warning" : "status-muted";
  function inspectionTime(value,timezone="UTC") {
    if (!value) return "未记录";
    try { return new Intl.DateTimeFormat("zh-CN",{timeZone:timezone,year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}).format(new Date(value)); }
    catch { return shell.fmt(value); }
  }
  function inspectionHead(header,monthly=false,conclusion={}) {
    const time = value => inspectionTime(value,header.timezone || "UTC");
    const metadata = monthly
      ? [["统计周期",time(header.period_start)+" 至 "+time(header.period_end)],["运维范围",header.scope_name],["数据库数量",header.target_count],["数据库类型",rows(header.database_types).join("、")]]
      : [["观察区间",time(header.observation_start)+" 至 "+time(header.observation_end)],["业务系统",header.business_system_name || "未记录"],["数据库",dailyTargetName(header)],["类型 / 版本",header.database_type+" "+header.database_version],["环境",header.environment]];
    metadata.push(["报告时间",time(header.report_time)],["报告时区",header.timezone || "UTC"],["报告编号",header.report_number+" · v"+header.report_version]);
    return `<header class="daily-report-head${monthly ? " monthly-report-head" : ""}"><div class="daily-report-kicker">KBot AIOps · 正式${monthly ? "月度" : "日常"}巡检报告</div><div class="daily-report-title-row"><div><h1>${esc(header.title)}</h1><p>${esc(monthly ? header.scope_name : dailyTargetName(header))}</p></div>${badge(conclusion.health_status,conclusion.health_status_display || "检查未完成",healthTone(conclusion.health_status))}</div><dl class="daily-report-metadata">${metadata.map(([label,value]) => item(label,esc(value))).join("")}</dl></header>`;
  }
  function inspectionTable(headers,data,cls="") {
    return dailyTable(headers,data.map(values => `<tr>${values.map((value,index) => `<td data-label="${esc(headers[index])}">${esc(value)}</td>`).join("")}</tr>`).join(""),cls);
  }
  function inspectionAppendix(data,monthly=false) {
    const boundary = data.evidence_boundary || {}, comparison = data.period_comparison || {};
    const compare = comparison.status === "AVAILABLE"
      ? `新增 ${rows(comparison.new_issues).length} 项 · 持续 ${rows(comparison.persistent_issues).length} 项 · 已验证恢复 ${rows(comparison.resolved_issues).length} 项 · 待验证 ${rows(comparison.pending_verification_issues).length} 项`
      : comparison.status_display || "无上月报告";
    const sources = rows(boundary.sources);
    const trace = sources.length ? `<details class="inspection-trace"><summary>技术追溯信息（${sources.length} 条）</summary><p>证据编号用于定位原始记录，内容指纹用于核对记录内容；仅供技术核查，不属于巡检问题。</p>${list(sources,"","inspection-source-list","ul")}</details>` : "";
    return `<details class="daily-evidence inspection-appendix"><summary>附录：统计方法与技术追溯</summary><div class="inspection-appendix-body"><p>${esc(boundary.methodology)}</p>${monthly ? `<h3>上月对比</h3><p>${esc(compare)}</p>` : `<p>必要检查完成率：${esc(coverage(boundary.quality?.evidence_coverage_percent))}；监控采样完整率：${esc(coverage(boundary.quality?.monitoring_sample_coverage_percent))}</p>`}${rows(data.gaps).length ? `<h3>数据说明</h3>${list(data.gaps)}` : ""}${trace}</div></details>`;
  }
  function dailyHtml(daily) {
    const header = daily.header, conclusion = daily.conclusion, metrics = rows(daily.metrics);
    const ledger = (label,value) => `<div><span>${label}</span><strong>${esc(value)}</strong></div>`;
    let content = inspectionHead(header,false,conclusion)
      + `<section class="daily-conclusion"><h2>巡检结论</h2><p class="daily-conclusion-copy">${esc(conclusion.text)}</p><div class="daily-conclusion-ledger">${ledger("运行评价",conclusion.health_status_display)}${ledger("必要检查完成",`${conclusion.completed_required_count ?? "未记录"} / ${conclusion.required_check_count ?? "未记录"}`)}${ledger("问题与风险",`${conclusion.attention_count ?? 0} 项`)}${ledger("未完成事项",`${conclusion.incomplete_count ?? 0} 项`)}</div></section>`;
    if (rows(daily.limitations).length) content += dailySection("LIMITATIONS","检查限制与未完成事项","",list(daily.limitations));
    if (rows(daily.findings).length) content += dailySection("FINDINGS","问题与风险","",`<div class="daily-findings">${daily.findings.map((finding,index) => `<article class="daily-finding"><span>${index+1}</span><div><h3>${esc(finding.title)}</h3><p>${esc(finding.summary)}</p>${finding.object_name ? `<p>影响对象：${esc(finding.object_name)}</p>` : ""}${finding.basis ? `<p>判断依据：${esc(finding.basis)}</p>` : ""}${finding.recommendation ? `<p>建议：${esc(finding.recommendation)}</p>` : ""}</div>${badge(finding.severity,riskLabel(finding.severity),riskClass(finding.severity))}</article>`).join("")}</div>`);
    content += dailySection("CHECKLIST","基础检查明细","正常项目同样列出；当前快照不代表全天未发生异常。",rows(daily.checks).length
      ? inspectionTable(["检查内容","结果","实测情况","判断依据"],daily.checks.map(check => [check.title+(check.required ? "（必要）" : ""),check.status_display,check.actual_display || check.summary,check.basis]),"daily-checks-table")
      : '<p class="daily-empty">本报告未记录基础检查明细，不能据此补造正常结果。</p>');
    if (metrics.length) content += dailySection("METRICS","关键运行指标","按明确规则判读最新样本；吞吐量相对增长不自动视为异常。",inspectionTable(["指标 / 范围","最新样本","采集窗口平均 / 最大","判断"],metrics.map(metric => [metric.display_name+" "+metric.dimension_label,metric.current_display,metric.average_display+" / "+metric.maximum_display,metric.status_display])));
    const comparisons = metrics.filter(metric => metric.baseline_applicable);
    if (comparisons.length) content += dailySection("BASELINE","七天基线对比","优先比较本期均值；仅有七天汇总时比较最新样本。变化幅度仅供趋势参考，不直接决定健康结论。",dailyTable(["指标 / 范围","对比值","七天均值","变化量 / 变化率","采样日均趋势"],comparisons.map(metric => `<tr>${metricName(metric)}<td data-label="对比值">${esc(metric.comparison_display)}<small>${esc(metric.comparison_label)}</small></td><td data-label="七天均值">${esc(metric.baseline_display)}<small>${esc(metric.baseline_status === "AVAILABLE" ? metric.baseline_window_label : metric.baseline_reason)}</small></td><td data-label="变化量 / 变化率">${esc(metric.change_display)}</td><td data-label="采样日均趋势">${metric.baseline_status === "AVAILABLE" ? monthlyTrendSvg({daily_points:rows(metric.trend_points)}) : '<span class="daily-empty">暂无连续趋势</span>'}</td></tr>`).join(""),"daily-baseline-table"));
    content += dailySection("ACTIONS","处置事项","",list(daily.recommendations,conclusion.health_status === "NORMAL" ? "无额外处置事项。" : "请先完成本报告列明的检查和问题复核。"));
    return content + inspectionAppendix(daily) + footer(header);
  }
  function monthlyHtml(monthly) {
    const header = monthly.header, conclusion = monthly.conclusion;
    let content = inspectionHead(header,true,conclusion)
      + `<section class="daily-conclusion"><h2>月度运行结论</h2><p class="daily-conclusion-copy">${esc(conclusion.text)}</p><div class="daily-conclusion-ledger"><div><span>月度评价</span><strong>${esc(conclusion.health_status_display)}</strong></div><div><span>检查完整度</span><strong>${esc(conclusion.completion_display)}</strong></div><div><span>问题与风险</span><strong>${esc(conclusion.issue_count ?? 0)} 项</strong></div></div></section>`;
    if (rows(monthly.limitations).length) content += dailySection("LIMITATIONS","检查限制与未完成事项","",list(monthly.limitations));
    if (rows(monthly.target_results).length) content += dailySection("DATABASES","系统与数据库概况","月内评价与最近一次巡检状态分别展示，不把期末正常写成全月正常。",inspectionTable(["数据库","月度评价","最近巡检","问题与待办"],monthly.target_results.map(target => [target.target_name,target.health_status_display,target.latest_health_display,`${target.issue_count} 项问题；${rows(target.limitations).join("；")}`])));
    if (rows(monthly.issues).length) content += dailySection("ISSUES","稳定性与问题闭环台账","未再发现不等于已解决；巡检观测次数不等于事故次数。",`<div class="daily-findings">${monthly.issues.map((issue,index) => `<article class="daily-finding"><span>${index+1}</span><div><h3>${esc(issue.target_name || "")} · ${esc(issue.title)}</h3><p>${esc(issue.summary)}</p><small>首次 ${esc(issue.first_seen || "未记录")} · 最近 ${esc(issue.last_seen || "未记录")} · 观测 ${esc(issue.observation_count ?? 0)} 次 · ${esc(issue.lifecycle_display || "待验证")}</small>${issue.occurrence_count != null ? `<p>已确认事件 ${esc(issue.occurrence_count)} 次</p>` : ""}</div>${badge(issue.severity,riskLabel(issue.severity),riskClass(issue.severity))}</article>`).join("")}</div>`);
    if (rows(monthly.metrics).length) content += dailySection("PERFORMANCE","性能与运行趋势","统计基于日均值分布，状态日均值不代表精确月度可用率。",inspectionTable(["指标 / 数据库","日均值平均","日均值 P95","日均最小 / 最大","有效日期"],monthly.metrics.map(metric => [metric.display_name+" "+metric.dimension_label,metric.monthly_average_display,metric.p95_display,metric.minimum_display+" / "+metric.maximum_display,`${metric.effective_days} / ${metric.expected_days}`])));
    if (rows(monthly.metrics).length) content += dailySection("TRENDS","月度日均趋势","不对缺失日期插值；仅有一个有效日时不绘制趋势线。",`<div class="daily-trend-grid">${monthly.metrics.map(metric => `<article class="daily-trend-card"><h3>${esc(metric.display_name)}</h3><p>${esc(metric.dimension_label)}</p>${monthlyTrendSvg(metric)}</article>`).join("")}</div>`);
    if (rows(monthly.capacity_rows).length) content += dailySection("CAPACITY","容量与增长","首末有效日不一定等于自然月首末日，不对缺失日期插值。",inspectionTable(["容量 / 数据库","首个有效日","末个有效日","净变化"],monthly.capacity_rows.map(metric => [metric.display_name+" "+metric.dimension_label,metric.first_display,metric.last_display,metric.net_change_display])));
    content += dailySection("ACTIONS","处置建议与下月重点","",list(monthly.recommendations,conclusion.health_status === "NORMAL" ? "无新增整改事项。" : "请先完成本报告列明的问题复核及检查补齐。"));
    return content + inspectionAppendix(monthly,true) + footer(header,true);
  }
  function chaptersFrom(sections) {
    const definitions = [{key:"overview",number:"I",title:"报告概览",description:"结论、范围与本报告的适用边界。",kinds:["EXECUTIVE_SUMMARY","SCOPE"]},{key:"analysis",number:"II",title:"诊断与发现",description:"基于观测事实呈现风险、趋势与根因判断。",kinds:["ALERT_TIMELINE","INSPECTION_COVERAGE","RISK_OVERVIEW","TREND","ROOT_CAUSE","FINDINGS"]},{key:"actions",number:"III",title:"处置与建议",description:"汇总建议、处置动作及其验证结果。",kinds:["RECOMMENDATIONS","ACTIONS"]},{key:"evidence",number:"IV",title:"证据与附录",description:"明确数据缺口，并保留可追溯的证据索引。",kinds:["EVIDENCE_BOUNDARY","EVIDENCE_APPENDIX"]}];
    const ordered = new Map(sections.map((section,index) => [section.kind,{...section,ordinal:String(index+1).padStart(2,"0")}]));
    const chapters = definitions.map(chapter => ({...chapter,sections:chapter.kinds.flatMap(kind => ordered.has(kind) ? [ordered.get(kind)] : [])})).filter(chapter => chapter.sections.length);
    const known = new Set(definitions.flatMap(chapter => chapter.kinds)), additional = sections.flatMap((section,index) => known.has(section.kind) ? [] : [{...section,ordinal:String(index+1).padStart(2,"0")}]);
    if (additional.length) chapters.push({key:"additional",number:"V",title:"补充内容",description:"模板定义的其他报告章节。",sections:additional});
    return chapters;
  }
  const sectionId = kind => `report-section-${kind.toLowerCase().replaceAll("_","-")}`;
  function reportMetadataHtml(data,report,target) {
    const scope = data.report?.scope || {}, percent = scope.evidence_coverage_percent;
    let evidenceCoverage = "历史报告未记录覆盖率";
    if (typeof percent === "number" && Number.isFinite(percent)) {evidenceCoverage = `${new Intl.NumberFormat("zh-CN",{maximumFractionDigits:2}).format(percent)}%`;if (typeof scope.evidence_covered_count === "number" && typeof scope.evidence_total_count === "number") evidenceCoverage += ` · ${scope.evidence_covered_count}/${scope.evidence_total_count} 项业务检查`;}
    const quality = [scope.database_type ? `数据库类型 ${scope.database_type}` : "",scope.health_status ? `健康状态 ${healthLabel(scope.health_status)}` : "",typeof scope.monitoring_sample_coverage_percent === "number" ? `监控采样覆盖率 ${scope.monitoring_sample_coverage_percent}%（低覆盖 ${scope.monitoring_low_coverage_metric_count ?? 0}/${scope.monitoring_metric_count ?? 0}）` : "",typeof scope.evidence_gap_count === "number" ? `数据缺口 ${scope.evidence_gap_count} 项` : ""].filter(Boolean).join(" · ");
    const profile = scope.monitoring_profile, profileLabel = profile?.display_name ? `${profile.display_name}${profile.version ? ` · ${profile.version}` : ""}` : "";
    const origin = ({ORACLE_NATIVE:"Oracle 原生",AIOPS_GENERATED:"KBot 生成",EXTERNAL_IMPORTED:"外部导入"})[scope.report_origin] || scope.report_origin || "未标记";
    const conversation = scope.conversation, source = conversation?.conversation_id ? `<a class="row-action" href="./chat.html?conversation=${encodeURIComponent(conversation.conversation_id)}">${esc(conversation.title || "智能诊断会话")} · ${esc(conversation.turn_count ?? 0)} 轮</a>` : esc(origin);
    return `<section class="key-value-card report-metadata" aria-label="报告信息"><dl class="key-value-table key-value-table-grid">${item("报告类型",esc(typeLabel(report.report_type)))}${item("报告状态",badge(report.status))}${item("报告版本",`第 ${esc(report.report_version)} 版`)}${item("发布时间",esc(shell.fmt(report.published_at)))}${item("报告周期",`${esc(shell.fmt(report.period_start))} 至 ${esc(shell.fmt(report.period_end))}`)}${item("报告模板",`${esc(data.template?.display_name ?? "系统诊断报告")} · v${esc(data.template?.version ?? report.template_version)}`)}${item("报告来源",source)}<div><dt>证据质量</dt><dd class="report-coverage-value"><strong>业务检查语义满足率 ${esc(evidenceCoverage)}</strong>${quality ? `<small>${esc(quality)}</small>` : ""}<small>${esc(scope.inspection_coverage || "以证据边界章节为准")}</small>${profileLabel ? `<small>监控口径：${esc(profileLabel)}</small>` : ""}</dd></div>${item("关联目标",esc(target?.display_name || "目标已删除或无权访问"),"report-metadata-wide")}${item("报告编号",esc(report.report_key),"report-metadata-wide","technical-value")}</dl></section>`;
  }
  function standardHtml(data,report) {
    const sections = rows(data.sections), chapters = chaptersFrom(sections), briefing = data.leadership_briefing;
    const leadership = briefing ? `<section class="report-leadership-briefing" aria-label="领导简报"><header><div><p class="eyebrow">LEADERSHIP BRIEFING</p><h2>领导简报</h2><p>只保留影响、风险和建议，不展开 SID 或 SQL。</p></div>${badge(briefing.risk_level,briefing.risk_level_display || riskLabel(briefing.risk_level),riskClass(briefing.risk_level))}</header><div class="report-leadership-grid">${[["影响",briefing.business_impact],["风险",briefing.risks],["建议",briefing.recommendations]].map(([label,items]) => `<article><h3>${label}</h3><ul>${rows(items).map(value => `<li>${esc(value)}</li>`).join("")}</ul></article>`).join("")}</div></section>` : "";
    return `<header class="report-cover"><div class="report-masthead"><div><span class="report-masthead-mark" aria-hidden="true">A</span><div><strong>KBot AIOps</strong><small>OPERATIONAL INTELLIGENCE REPORT</small></div></div><div class="report-release-state"><span>${esc(typeLabel(report.report_type))}</span>${badge(data.status || report.status,data.status_display)}</div></div><div class="report-cover-heading"><p class="eyebrow">正式运维报告 · 第 ${esc(report.report_version)} 版</p><h1>${esc(data.title || report.title)}</h1><div class="agent-markdown-content report-abstract">${KBotMarkdown.render(report.summary || "")}</div></div><div class="report-cover-folio"><span>报告编号</span><strong class="technical-value">${esc(report.report_key)}</strong><span>报告周期</span><strong>${esc(shell.fmt(report.period_start))} — ${esc(shell.fmt(report.period_end))}</strong></div></header>${leadership}${reportMetadataHtml(data,report,state.target)}<div class="report-document-body"><aside class="report-index" aria-label="报告目录"><header><p class="eyebrow">DOCUMENT INDEX</p><h2>报告目录</h2><span>${sections.length} 个章节 · ${chapters.length} 个部分</span></header><nav>${chapters.map(chapter => `<section><h3><span>${chapter.number}</span>${chapter.title}</h3>${chapter.sections.map(section => `<a href="#${sectionId(section.kind)}" data-section-link="${esc(section.kind)}"><span>${section.ordinal}</span>${esc(section.display_name || sectionLabel(section.kind))}</a>`).join("")}</section>`).join("")}</nav></aside><div class="report-content">${chapters.map(chapter => `<section class="report-chapter"><header class="report-chapter-header"><span>PART ${chapter.number}</span><div><h2>${chapter.title}</h2><p>${chapter.description}</p></div></header>${chapter.sections.map(section => `<section id="${sectionId(section.kind)}" data-report-section="${esc(section.kind)}" class="report-section${section.kind === "EXECUTIVE_SUMMARY" ? " is-executive-summary" : ""}${section.kind === "EVIDENCE_BOUNDARY" && report.status === "PARTIAL" ? " is-evidence-boundary" : ""}${section.kind === "EVIDENCE_APPENDIX" ? " is-evidence-appendix" : ""}"><header class="report-section-header"><span>${section.ordinal}</span><div><h3>${esc(section.display_name || sectionLabel(section.kind))}</h3>${section.human_edited ? badge("INFO","人工调整展示","status-info") : ""}</div></header><div class="report-section-content">${rows(section.items).map(markdown).join("") || '<p class="empty-cell">本章节暂无内容。</p>'}</div></section>`).join("")}</section>`).join("")}</div></div><footer class="report-document-footer"><div><strong>KBot AIOps</strong><span>正式运维报告 · 系统冻结版本</span></div><div><span class="technical-value">${esc(report.report_key)}</span><strong>V${esc(report.report_version)}</strong></div></footer>`;
  }
  function updateActions() {
    const pdf = document.getElementById("download-report-pdf");pdf.disabled = !state.report || state.pdfBusy;pdf.textContent = state.pdfBusy ? "正在生成下载…" : "下载 PDF";
    const raw = document.getElementById("download-raw-report");raw.hidden = !state.presentation?.report?.scope?.raw_report_artifact_id;raw.disabled = state.rawBusy;raw.textContent = state.rawBusy ? "正在下载…" : "下载原始 Artifact";
    const existing = document.getElementById("report-existing-case"), extract = document.getElementById("extract-report-case");existing.hidden = !state.eligibility?.existing_asset_id;existing.href = `./cases.html?assetId=${encodeURIComponent(state.eligibility?.existing_asset_id || "")}`;
    extract.hidden = Boolean(state.eligibility?.existing_asset_id) || state.report?.report_type !== "COMPARISON" || !state.eligibility?.eligible;extract.disabled = state.extracting;extract.textContent = state.extracting ? "正在提炼…" : "生成待审核效果案例";
  }
  function observeSections() {
    state.observer?.disconnect();const sections = [...root.querySelectorAll("[data-report-section]")];
    const active = kind => root.querySelectorAll("[data-section-link]").forEach(link => {link.classList.toggle("active",link.dataset.sectionLink === kind);});
    active(sections[0]?.dataset.reportSection);
    if (!sections.length || typeof IntersectionObserver === "undefined") return;
    state.observer = new IntersectionObserver(entries => {const current = entries.filter(entry => entry.isIntersecting).sort((left,right) => left.boundingClientRect.top-right.boundingClientRect.top)[0];if (current) active(current.target.dataset.reportSection);},{rootMargin:"-18% 0px -68% 0px",threshold:[0,1]});sections.forEach(section => state.observer.observe(section));
  }
  async function load() {
    try {
      if (!reportId) throw new Error("URL 缺少报告 id");
      [state.report,state.presentation] = await Promise.all([request(`/reports/${encodeURIComponent(reportId)}`),request(`/reports/${encodeURIComponent(reportId)}/presentation`)]);
      if (["INCIDENT","COMPARISON"].includes(state.report.report_type) && state.report.status === "READY") {try {state.eligibility = await request(`/operations-knowledge/reports/${encodeURIComponent(reportId)}/case-eligibility`);} catch (_) {state.eligibility = null;}}
      try {state.target = await request(`/targets/${encodeURIComponent(state.report.target_id)}`);} catch (_) {state.target = null;}
      root.innerHTML = state.presentation.monthly_inspection ? monthlyHtml(state.presentation.monthly_inspection) : state.presentation.daily_inspection ? dailyHtml(state.presentation.daily_inspection) : standardHtml(state.presentation,state.report);root.hidden = false;observeSections();
    } catch (error) {const notice = document.getElementById("formal-report-error");notice.hidden = false;notice.querySelector("span").textContent = error.message || "无法读取报告内容";shell.toast(error.message || "无法读取报告内容");}
    finally {document.getElementById("formal-report-loading").hidden = true;updateActions();}
  }
  async function download(raw=false) {
    if (raw ? state.rawBusy || !state.presentation?.report?.scope?.raw_report_artifact_id : state.pdfBusy || !state.report) return;
    const key = raw ? "rawBusy" : "pdfBusy";state[key] = true;updateActions();
    try {await KBotAIOpsAuth.download(raw ? `${api}/artifacts/${encodeURIComponent(state.presentation.report.scope.raw_report_artifact_id)}/content` : `${api}/reports/${encodeURIComponent(reportId)}/pdf`,raw ? `aiops-report-artifact-${state.presentation.report.scope.raw_report_artifact_id}` : `aiops-report-${reportId}.pdf`);shell.toast(raw ? "原始报告 Artifact 已开始下载" : "正式报告 PDF 已开始下载");}
    catch (error) {shell.toast(error.message || (raw ? "无法下载原始报告 Artifact" : "无法下载正式报告 PDF"));}
    finally {state[key] = false;updateActions();}
  }
  shell.ready.then(() => {
    document.getElementById("download-report-pdf").onclick = () => download();document.getElementById("download-raw-report").onclick = () => download(true);
    document.getElementById("extract-report-case").onclick = async () => {
      if (state.extracting || !state.eligibility?.eligible || state.eligibility?.existing_asset_id) return;state.extracting = true;updateActions();
      try {const result = await request(`/operations-knowledge/reports/${encodeURIComponent(reportId)}:extract-case`,{method:"POST"});state.eligibility = {eligible:true,reasons:[],existing_asset_id:result.asset.asset_id};shell.toast("效果案例已进入审核流程，尚未发布");}
      catch (error) {shell.toast(error.message || "无法生成效果案例");}
      finally {state.extracting = false;updateActions();}
    };
    addEventListener("pagehide",() => state.observer?.disconnect(),{once:true});return load();
  }).catch(error => shell.toast(error.message));
})();
