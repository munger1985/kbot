(function () {
  "use strict";
  const esc = value => KBotAIOpsShell.escape(String(value ?? ""));
  const fmt = value => value ? KBotAIOpsShell.fmt(value) : "无采集时间";
  const { buildExadataCards, exadataRaw, exadataUnit, exadataQuality, exadataDimensions, exadataNumeric, exadataChartOption } = KBotAIOpsExadataPresentation;
  const scenes = [["overview", "整机/Cell 状态"], ["cell", "Cell 资源"], ["disk", "磁盘 IO"], ["flash", "Flash/Smart IO"], ["iorm", "IORM"], ["network", "互联/交换机"], ["capacity", "容量/硬件"]];
  class ExadataMetrics {
    constructor(root, options) {
      this.root = root; this.options = options; this.key = `${options.sourceId}:${options.instanceId}`;
      this.mode = "LATEST_SNAPSHOT"; this.scene = "overview"; this.component = ""; this.offset = 0;
      this.view = null; this.report = null; this.controller = null; this.disposed = false; this.loading = false;
      this.members = []; this.charts = []; this.observers = [];
      this.root.classList.add("exadata-oem");
      this.base = `/api/v1/apps/aiops/business-systems/${encodeURIComponent(options.systemId)}/exadata/${encodeURIComponent(options.instanceId)}/sources/${encodeURIComponent(options.sourceId)}`;
      this.refresh();
    }
    clearCharts() { this.observers.forEach(observer => observer.disconnect()); this.observers = []; this.charts.forEach(chart => chart.dispose()); this.charts = []; }
    dispose() { this.disposed = true; this.controller?.abort(); this.clearCharts(); }
    async refresh() { return this.load(false); }
    async load(diagnosis) {
      if (this.disposed) return;
      this.controller?.abort(); const controller = new AbortController(); this.controller = controller;
      this.loading = true; this.view = null; this.report = null;
      this.render(this.mode === "HISTORY" ? "正在读取 OEM 数值历史…" : "正在读取 OEM 最新快照…");
      try {
        const result = await KBotAIOpsAuth.request(this.base + (diagnosis ? "/diagnoses" : "/view"), {
          method: "POST", headers: diagnosis ? { "Idempotency-Key": KBotAIOpsAuth.uuid() } : {},
          body: JSON.stringify({ mode: this.mode, scene: this.scene, component_id: this.component || null, offset: this.offset, limit: 12 }),
          signal: controller.signal,
        });
        if (this.disposed || controller.signal.aborted) return;
        this.view = diagnosis ? result.evidence : result; this.members = this.view.members;
        this.report = diagnosis ? result : null; this.loading = false; this.render();
      } catch (error) {
        if (!controller.signal.aborted && !this.disposed) { this.loading = false; this.render(error.message || "Exadata 查询失败", true); }
      }
    }
    renderCard(card, index) {
      const history = this.view.query_mode === "HISTORY";
      const chart = history && card.numeric && card.samples.some(sample => sample.quality !== "EMPTY" && exadataNumeric(sample.value) !== null);
      return `<article class="monitoring-panel is-native exadata-metric-card tone-${esc(card.tone)}" data-metric-code="${esc(card.metricCode)}" data-component-id="${esc(card.componentId)}">
        <header><div><h3 title="${esc(card.title)}">${esc(card.title)}</h3><p title="${esc(card.componentName)}">${esc(card.componentName)}</p></div><span class="status-pill ${card.tone === "warning" ? "status-warning" : "status-muted"}">${esc(card.quality)}</span></header>
        <div class="exadata-card-value"><strong title="${esc(card.value)}">${esc(card.value)}</strong><span>${esc(card.valueUnit)}</span></div>
        <p class="exadata-card-time" title="${esc(card.observedAt)}">${history ? "窗口末次采样" : "采集时间"} · ${esc(fmt(card.observedAt))}</p>
        ${chart ? `<div class="exadata-chart" data-chart="${index}" role="img" aria-label="${esc(card.title)}数值历史；不同维度分列，空值不补零"></div>` : `<div class="exadata-snapshot">
          ${card.percent !== null ? `<div class="exadata-percent" role="img" aria-label="原生百分比 ${esc(card.value)}%，不代表健康阈值"><div><i style="width:${card.percent}%"></i></div><p><span>0%</span><span>100%</span></p></div>` : ""}
          ${card.series.length > 1 ? `<div class="exadata-dimensions">${card.series.slice(0, 4).map(series => `<div><span title="${esc(series.label)}">${esc(series.label)}</span><strong>${esc(exadataRaw(series.latest.value))} ${esc(card.valueUnit)}</strong></div>`).join("")}</div>${card.series.length > 4 ? `<p>另 ${card.series.length - 4} 个维度请查看数据。</p>` : ""}` : ""}
          <p>${esc(card.gaps[0]?.detail || (history ? "当前指标没有可绘制的数值历史，不回退快照。" : card.numeric ? "最新采集快照，不生成历史趋势。" : "保留 OEM 原始值；状态字典未验证，不判定健康。"))}</p></div>`}
        <footer><code title="${esc(card.selector)}">${esc(card.selector)}</code><span>单位 ${esc(exadataUnit(card.unit))}</span></footer>
        <details class="exadata-evidence"><summary>查看数据 · ${card.samples.length} 条</summary><p>图表仅为数值近似；原始值、维度、质量与实际采集时间如下。未验证阈值，不判定健康。</p><div class="exadata-table-scroll"><table><thead><tr><th>采集时间</th><th>原始值</th><th>维度</th><th>质量</th></tr></thead><tbody>${card.samples.map(sample => `<tr><td title="${esc(sample.observed_at)}">${esc(fmt(sample.observed_at))}</td><td>${esc(exadataRaw(sample.value))} · ${esc(exadataUnit(sample.unit))}</td><td>${esc(exadataDimensions(sample.dimensions))}</td><td>${esc(exadataQuality(sample.quality))}</td></tr>`).join("")}</tbody></table></div>${card.gaps.map(gap => `<p>${esc(gap.code)} · ${esc(gap.detail)}</p>`).join("")}</details>
      </article>`;
    }
    render(message = "", failed = false) {
      this.clearCharts();
      const view = this.view, cards = view ? buildExadataCards(view) : [], disabled = this.loading ? "disabled" : "";
      this.root.setAttribute("aria-busy", String(this.loading));
      this.root.innerHTML = `<header class="exadata-heading"><div><p class="eyebrow">EXADATA · OEM</p><h2>Exadata OEM 专用监控与只读诊断</h2><p>按已确认组件查看状态、性能与容量；仅展示当前查询分页。</p></div><span class="status-pill status-muted">${this.mode === "HISTORY" ? "数值历史" : "最新快照"}</span></header>
      <div class="exadata-toolbar"><label class="control-field"><span class="field-label">查询模式</span><select data-mode ${disabled}><option value="LATEST_SNAPSHOT" ${this.mode === "LATEST_SNAPSHOT" ? "selected" : ""}>最新快照</option><option value="HISTORY" ${this.mode === "HISTORY" ? "selected" : ""}>过去 1 小时历史（仅数值）</option></select></label><label class="control-field"><span class="field-label">场景</span><select data-scene ${disabled}>${scenes.map(([id, name]) => `<option value="${id}" ${this.scene === id ? "selected" : ""}>${name}</option>`).join("")}</select></label><label class="control-field"><span class="field-label">组件</span><select data-component ${disabled}><option value="">全部已确认成员</option>${this.members.map(member => `<option value="${esc(member.target_id)}" ${member.target_id === this.component ? "selected" : ""}>${esc(member.display_name)} · ${esc(member.target_type)}</option>`).join("")}</select></label><div class="exadata-actions"><button class="secondary-button" data-refresh type="button" ${disabled}>刷新指标</button><button class="secondary-button" data-diagnose type="button" ${disabled}>只读诊断当前页</button></div></div>
      ${message ? `<p class="exadata-message ${failed ? "notice notice-error" : ""}" role="${failed ? "alert" : "status"}">${esc(message)}</p>` : ""}
      ${view ? `<div class="exadata-result-bar"><p role="status">${cards.length} 张指标卡 · ${view.samples.length} 条采样 · 场景查询 ${this.offset + (view.total_queries ? 1 : 0)}–${Math.min(this.offset + 12, view.total_queries)} / ${view.total_queries}<small>获取时间 ${esc(fmt(view.captured_at))} · 目录 ${esc(view.catalog_version)}</small></p><div class="exadata-actions"><button class="secondary-button" data-first type="button" ${this.offset === 0 ? "disabled" : ""}>场景首页</button><button class="secondary-button" data-next type="button" ${view.next_offset === null ? "disabled" : ""}>下一页指标</button></div></div><div class="monitoring-panels exadata-metric-grid" aria-label="Exadata 指标卡片">${cards.map((card, index) => this.renderCard(card, index)).join("")}</div>${!cards.length ? '<p class="exadata-empty">此页没有可展示的指标。请查看数据缺口或选择其他场景，不按零值或健康处理。</p>' : ""}<details class="exadata-gaps"><summary>数据质量与采集边界 · ${view.gaps.length} 项</summary><ul>${view.gaps.map(gap => `<li><strong>${esc(gap.code)}</strong><p>${esc(gap.detail)}</p>${gap.metric_code ? `<code>${esc(gap.metric_code)}</code>` : ""}</li>`).join("")}</ul></details>` : ""}
      ${this.report ? `<section class="exadata-report"><h3>只读诊断报告</h3><p>结论：证据不足，不能判定健康。范围仅限本次分页，不代表整机健康。</p><code>${esc(this.report.report_id)}</code><ul>${this.report.findings.map(finding => `<li>${esc(finding.code)} · ${esc(finding.detail)}</li>`).join("")}</ul></section>` : ""}`;
      const query = selector => this.root.querySelector(selector);
      query("[data-mode]").onchange = event => { this.mode = event.target.value; this.offset = 0; this.load(false); };
      query("[data-scene]").onchange = event => { this.scene = event.target.value; this.offset = 0; this.load(false); };
      query("[data-component]").onchange = event => { this.component = event.target.value; this.offset = 0; this.load(false); };
      query("[data-refresh]").onclick = () => this.load(false);
      query("[data-diagnose]").onclick = () => this.load(true);
      if (!view) return;
      query("[data-first]").onclick = () => { this.offset = 0; this.load(false); };
      query("[data-next]").onclick = () => { this.offset = view.next_offset; this.load(false); };
      this.root.querySelectorAll("[data-chart]").forEach(node => {
        if (!globalThis.echarts) { node.textContent = "图表引擎不可用，请查看原始数据。"; return; }
        const css = getComputedStyle(this.root), color = name => css.getPropertyValue(`--${name}`).trim();
        const chart = echarts.init(node);
        chart.setOption(exadataChartOption(cards[Number(node.dataset.chart)], Object.fromEntries(["accent", "info", "muted", "line", "surface", "ink"].map(name => [name, color(name)]))));
        const observer = new ResizeObserver(() => chart.resize()); observer.observe(node);
        this.charts.push(chart); this.observers.push(observer);
      });
    }
  }
  globalThis.KBotAIOpsExadataMetrics = ExadataMetrics;
})();
