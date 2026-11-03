// 正式日报与模板预览共用一个渲染器，不按数据库类型分支。
(function () {
  "use strict";
  const rows = value => Array.isArray(value) ? value : [];
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
  const risk = value => ({CRITICAL:"严重",HIGH:"高",MEDIUM:"中",LOW:"低",INFO:"参考"})[value] || value || "";
  const tone = value => ({NORMAL:"status-success",ABNORMAL:"status-danger",ATTENTION:"status-warning",INCOMPLETE:"status-warning",HIGH:"status-danger",CRITICAL:"status-danger",MEDIUM:"status-warning"})[value] || "status-muted";
  const badge = (status, text) => '<span class="status-pill ' + tone(status) + '">' + esc(text) + '</span>';
  const list = items => '<ul>' + rows(items).map(item => '<li>' + esc(item) + '</li>').join("") + '</ul>';
  const table = (headers, values, cls="") => '<div class="daily-table-wrap"><table class="daily-report-table ' + cls + '"><thead><tr>' + headers.map(item => '<th>' + esc(item) + '</th>').join("") + '</tr></thead><tbody>' + values.map(row => '<tr>' + row.map((value,index) => '<td data-label="' + esc(headers[index]) + '">' + value + '</td>').join("") + '</tr>').join("") + '</tbody></table></div>';
  function time(value, timezone) {
    if (!value) return "未记录";
    try { return new Intl.DateTimeFormat("zh-CN",{timeZone:timezone || "UTC",year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}).format(new Date(value)); }
    catch { return value; }
  }
  function checkTable(checks) {
    let content = table(["检查内容","结果","实测情况","判断依据"],checks.map(check => [
      '<strong>' + esc(check.title) + '</strong><small>' + (check.required ? "必要检查" : "专项或辅助检查") + '</small>',
      esc(check.status_display), esc(rows(check.detail_records).length ? "已取得 " + check.detail_records.length + " 条对象记录，明细见下方。" : check.actual_display || check.summary),
      esc(check.basis) + (check.classification_note ? '<small>' + esc(check.classification_note) + '</small>' : "")
    ]),"daily-checks-table");
    checks.forEach(check => {
      if (!rows(check.detail_records).length) return;
      content += '<details class="daily-object-details"><summary>' + esc(check.title) + ' · ' + check.detail_records.length + ' 条对象记录</summary>';
      content += check.detail_records.map((record,index) => '<article class="daily-object-record"><h4>对象 ' + (index+1) + '</h4><dl>' + record.map(field => '<div><dt>' + esc(field.label) + '</dt><dd>' + esc(field.value) + '</dd></div>').join("") + '</dl></article>').join("") + '</details>';
    });
    return content;
  }
  function sqlReview(statement) {
    const review = statement.agent_review || {};
    let content = '<div class="inspection-sql-analysis"><h4>Agent 初步分析' + (review.status === "COMPLETE" ? "" : "（未完成）") + '</h4><p>' + esc(review.analysis) + '</p>';
    if (review.status === "COMPLETE") {
      content += '<h4>可能原因与验证方向</h4><ul>' + rows(review.possible_causes).map(cause => '<li>' + esc(cause.possible_cause) + '<p>验证：' + esc(cause.verification) + '</p></li>').join("") + '</ul><h4>初步优化建议</h4><ol>';
      content += rows(review.recommendations).map(advice => '<li>' + esc(advice.action) + '<p>' + esc(advice.rationale) + '</p><p>验证：' + esc(advice.validation) + (advice.requires_approval ? "；实施前须审批。" : "") + '</p></li>').join("") + '</ol>';
      content += rows(review.limitations).map(value => '<p class="inspection-sql-note">' + esc(value) + '</p>').join("");
    } else content += '<h4>规则复核建议（非 Agent 分析）</h4>' + list(statement.rule_recommendations);
    return content + '</div>';
  }
  function sqlCards(group, header) {
    return '<p class="inspection-sql-summary">' + esc(group.summary) + '</p><p class="inspection-sql-note">' + esc(group.note) + '</p>' + rows(group.statements).map(statement =>
      '<article id="' + esc(statement.problem_id) + '" class="inspection-sql-card"><header><h3>' + esc(String(statement.ordinal).padStart(2,"0")) + ' · ' + esc(statement.schema_name || header.target_name) + '</h3>' + badge(statement.severity,risk(statement.severity)) + '</header><div class="inspection-sql-analysis"><h4>' + esc(statement.problem_summary) + '</h4><p>判断依据：' + esc(statement.basis) + '</p><p>' + esc(statement.text_label) + '</p></div>' +
      (statement.statement_text ? '<pre><code>' + esc(statement.statement_preview) + '</code></pre>' : '<p>来源未返回 SQL 文本。</p>') +
      (statement.statement_truncated ? '<details class="inspection-sql-full-text"><summary>预览已截短，展开来源返回的 SQL 全文</summary><pre><code>' + esc(statement.statement_text) + '</code></pre></details>' : "") +
      (statement.digest ? '<details class="inspection-sql-full-text"><summary>语句定位信息（Digest）</summary><p>' + esc(statement.digest) + '</p></details>' : "") +
      '<dl class="inspection-sql-stats">' + [["执行次数",statement.execution_count_display],["总耗时",statement.total_latency_display],["平均耗时",statement.average_latency_display],["最大耗时",statement.maximum_latency_display]].map(([label,value]) => '<div><dt>' + label + '</dt><dd>' + esc(value) + '</dd></div>').join("") + '</dl>' + sqlReview(statement) + '</article>'
    ).join("");
  }
  function trend(metric) {
    const points = rows(metric.trend_points), dates = points.map(point => Date.parse(point.date)), values = points.map(point => Number(point.value));
    if (metric.baseline_status !== "AVAILABLE" || values.length < 2 || values.some(value => !Number.isFinite(value)) || dates.some((date,index) => !Number.isFinite(date) || (index > 0 && date-dates[index-1] !== 86400000))) return "暂无连续趋势";
    const low = Math.min(...values), span = Math.max(...values)-low || 1;
    return '<svg class="daily-trend-svg" viewBox="0 0 150 42" role="img" aria-label="' + esc(metric.display_name) + '七天日均趋势"><polyline points="' + values.map((value,index) => (index*146/(values.length-1)+2) + ',' + (38-(value-low)*34/span)).join(" ") + '"/></svg>';
  }
  function render(data) {
    const header = data.header, conclusion = data.conclusion, sections = rows(data.sections);
    let content = '<div class="daily-regional-report">' + (data.preview ? '<p class="daily-preview-notice">模拟版式预览 · 非真实巡检结果 · 不连接数据库、不调用分析模型</p>' : "") +
      '<header class="daily-report-head"><p class="daily-report-kicker">数据库运维 · 日常巡检报告</p><div class="daily-report-title-row"><div><h1>' + esc(header.title) + '</h1><p>' + esc(header.business_system_name || "数据库运维") + ' · ' + esc(header.target_name) + '</p></div>' + badge(conclusion.health_status,conclusion.health_status_display) + '</div><dl class="daily-report-metadata">';
    content += [["巡检时间",time(header.report_time,header.timezone)],["观察区间",time(header.observation_start,header.timezone)+" 至 "+time(header.observation_end,header.timezone)],["数据库 / 版本",header.database_type+" "+header.database_version],["环境 / 时区",header.environment+" / "+header.timezone],["报告编号",header.report_number+" · v"+header.report_version]].map(([label,value]) => '<div><dt>' + label + '</dt><dd>' + esc(value) + '</dd></div>').join("");
    content += '</dl></header><div class="daily-document-layout"><aside class="daily-document-index" aria-label="日常巡检报告目录"><nav><h2>报告目录</h2>' + sections.map(region => '<a href="#' + esc(region.anchor) + '" data-section-link="' + esc(region.section_id) + '"><span>' + String(region.ordinal).padStart(2,"0") + '</span>' + esc(region.title) + '<small>' + esc(region.status_display) + '</small></a>').join("") + '<a href="#daily-appendix">附录 · 检查索引与技术追溯</a></nav></aside><div class="daily-document-content">';
    sections.forEach(region => {
      const key = region.section_id;
      content += '<section id="' + esc(region.anchor) + '" data-report-section="' + esc(key) + '" class="daily-report-section' + (key === "OVERVIEW" ? " daily-conclusion" : "") + (key === "SQL" ? " inspection-sql-section" : "") + (key === "BASELINE" ? " daily-baseline-section" : "") + '"><header><div><p class="daily-section-kicker">' + String(region.ordinal).padStart(2,"0") + ' / 日常巡检</p><h2>' + esc(region.title) + '</h2></div>' + badge(region.status,region.status_display) + '</header>';
      if (key === "OVERVIEW") {
        content += '<p class="daily-conclusion-copy">' + esc(conclusion.text) + '</p><div class="daily-conclusion-ledger">' + [["运行评价",conclusion.health_status_display],["必要检查完成",(conclusion.completed_required_count ?? "未记录")+" / "+(conclusion.required_check_count ?? "未记录")],["问题与风险",(conclusion.attention_count ?? 0)+" 项"],["未完成事项",(conclusion.incomplete_count ?? 0)+" 项"]].map(([label,value]) => '<div><span>' + label + '</span><strong>' + esc(value) + '</strong></div>').join("") + '</div>';
        content += table(["检查区域","结论","主要情况"], sections.filter(item => !["OVERVIEW","ACTIONS"].includes(item.section_id)).map(item => ['<a href="#'+esc(item.anchor)+'">'+esc(item.title)+'</a>',esc(item.status_display),esc(item.summary)]),"daily-region-summary");
        if (rows(data.finding_index).length) content += '<div class="daily-issue-index"><h3>重要问题定位</h3><ol>' + data.finding_index.map(issue => '<li><a href="#' + esc(issue.anchor) + '">' + esc(risk(issue.severity)+" · "+issue.title+" · "+(issue.object_name || issue.summary)) + '</a></li>').join("") + '</ol></div>';
        if (rows(data.limitations).length) content += '<div class="inspection-limitations"><h3>检查限制与未完成事项</h3>' + list(data.limitations) + '</div>';
      } else content += '<p class="daily-region-description">' + esc(region.description) + '</p><p class="daily-region-summary-copy">' + esc(region.summary) + '</p>';
      const linked = new Set(rows(region.sql_details).flatMap(group => rows(group.statements).map(item => item.problem_id)));
      const localFindings = rows(region.findings).filter(issue => !linked.has(issue.finding_id));
      if (localFindings.length) content += '<div class="daily-findings">' + localFindings.map(issue => '<article id="' + esc(issue.anchor) + '" class="daily-finding"><span>' + esc(risk(issue.severity)) + '</span><div><h3>' + esc(issue.title) + '</h3><p>' + esc(issue.summary) + '</p>' + [["影响对象",issue.object_name],["判断依据",issue.basis],["建议",issue.recommendation]].filter(([,value]) => value).map(([label,value]) => '<p>'+label+'：'+esc(value)+'</p>').join("") + '</div></article>').join("") + '</div>';
      if (key === "SQL") {
        if (!rows(region.sql_details).length && rows(region.checks).length) content += checkTable(region.checks);
        const statements = rows(region.sql_details).flatMap(group => rows(group.statements));
        if (statements.length) content += table(["SQL","严重性","主要问题","分析状态"],statements.map(item => ['<a href="#'+esc(item.problem_id)+'">SQL '+item.ordinal+'</a>',esc(risk(item.severity)),esc(item.problem_summary),item.agent_review?.status === "COMPLETE" ? "已分析" : "分析未完成"]));
        content += rows(region.sql_details).map(group => sqlCards(group,header)).join("");
      } else if (!["OVERVIEW","BASELINE","ACTIONS"].includes(key)) {
        const normal = rows(region.checks).filter(check => ["CLEAR","COMPLIANT","NOT_APPLICABLE"].includes(check.status)), other = rows(region.checks).filter(check => !normal.includes(check));
        if (other.length) content += checkTable(other);
        if (normal.length) content += '<details class="daily-normal-details"' + (region.normal_detail === "FULL" && region.expanded ? " open" : "") + '><summary>已完成的正常 / 不适用检查 · ' + normal.length + ' 项</summary>' + checkTable(normal) + '</details>';
        if (rows(region.metrics).length) content += table(["指标 / 范围","最新样本",...(region.show_statistics ? ["采集窗口平均 / 最大"] : []),"判断"],region.metrics.map(metric => ['<strong>'+esc(metric.display_name)+'</strong><small>'+esc(metric.dimension_label)+'</small>',esc(metric.current_display),...(region.show_statistics ? [esc(metric.average_display+" / "+metric.maximum_display)] : []),esc(metric.status_display)]),"daily-metrics-table");
      }
      if (key === "BASELINE" && rows(region.comparisons).length) content += table(["指标 / 范围","对比值","七天均值","变化量 / 变化率",...(region.show_trend ? ["采样日均趋势"] : [])], region.comparisons.map(metric => ['<strong>'+esc(metric.display_name)+'</strong><small>'+esc(metric.dimension_label)+'</small>',esc(metric.comparison_display)+'<small>'+esc(metric.comparison_label)+'</small>',esc(metric.baseline_display)+'<small>'+esc(metric.baseline_status === "AVAILABLE" ? metric.baseline_window_label : metric.baseline_reason)+'</small>',esc(metric.change_display),...(region.show_trend ? [trend(metric)] : [])]),"daily-baseline-table");
      if (key === "ACTIONS") content += rows(region.actions).length ? table(["优先级 / 对象","处置事项","复查依据"],region.actions.map(action => [esc(risk(action.priority))+'<small>'+esc(action.object_name)+'</small><a href="#daily-'+esc(action.section_id.toLowerCase())+'">查看相关区域</a>',esc(action.action),esc(action.validation)]),"daily-actions-table") : '<p>' + (conclusion.health_status === "NORMAL" ? "无额外处置事项。" : "请先完成本报告列明的检查和问题复核。") + '</p>';
      content += '</section>';
    });
    const boundary = data.evidence_boundary || {};
    content += '<details id="daily-appendix" class="daily-evidence inspection-appendix"><summary>附录：统计方法与技术追溯</summary><div class="inspection-appendix-body"><p>' + esc(boundary.methodology) + '</p><p>必要检查完成率：' + esc(boundary.quality?.evidence_coverage_percent ?? "未记录") + '；监控采样完整率：' + esc(boundary.quality?.monitoring_sample_coverage_percent ?? "未记录") + '</p><details><summary>完整检查索引</summary>' + checkTable(rows(data.checks)) + '</details>';
    if (rows(data.gaps).length) content += '<h3>数据说明</h3>' + list(data.gaps);
    if (rows(boundary.sources).length) content += '<details class="inspection-trace"><summary>技术追溯信息（' + boundary.sources.length + ' 条）</summary><p>证据编号用于定位原始记录，内容指纹用于核对内容，仅供技术核查。</p><ul class="inspection-source-list">' + boundary.sources.map(value => '<li>'+esc(value)+'</li>').join("") + '</ul></details>';
    return content + '</div></details></div></div><footer class="daily-report-footer"><span>' + (data.preview ? "模拟预览 · 非真实巡检结果" : "正式日常巡检报告 · 共享分区格式") + '</span><span>' + esc(header.report_number) + '</span></footer></div>';
  }
  globalThis.KBotDailyInspectionReport = Object.freeze({render});
})();
