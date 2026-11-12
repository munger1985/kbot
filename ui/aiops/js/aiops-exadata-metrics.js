(function () {
  "use strict";
  const esc = value => KBotAIOpsShell.escape(String(value ?? ""));
  const fmt = value => value ? KBotAIOpsShell.fmt(value) : "无采集时间";
  const { buildExadataCards, buildExadataDevices, exadataOverviewNotes, exadataScenes, exadataSceneMembers, exadataDeviceScene, exadataDeviceGroup, exadataCardSection, exadataGapLabel, exadataStatusItems, exadataRaw, exadataUnit, exadataQuality, exadataDimensions, exadataNumeric, exadataHasTrend, exadataChartOption, exadataBaselineKey, exadataBaselineLines, exadataBaselineSummary } = KBotAIOpsExadataPresentation;
  const scenes = exadataScenes;
  class ExadataMetrics {
    constructor(root, options) {
      this.root = root; this.options = options; this.key = `${options.sourceId}:${options.instanceId}`;
      this.mode = "LATEST_SNAPSHOT"; this.scene = "overview"; this.component = ""; this.offset = 0;
      this.view = null; this.report = null; this.controller = null; this.disposed = false; this.loading = false;
      this.members = []; this.charts = []; this.observers = [];
      this.baselinePeriod = "1w"; this.baselines = {}; this.baselineController = null;
      this.root.classList.add("exadata-oem");
      this.base = `/api/v1/apps/aiops/business-systems/${encodeURIComponent(options.systemId)}/exadata/${encodeURIComponent(options.instanceId)}/sources/${encodeURIComponent(options.sourceId)}`;
      this.refresh();
    }
    clearCharts() { this.observers.forEach(observer => observer.disconnect()); this.observers = []; this.charts.forEach(chart => chart.dispose()); this.charts = []; }
    dispose() { this.disposed = true; this.controller?.abort(); this.baselineController?.abort(); this.clearCharts(); }
    async loadBaselines() {
      this.baselineController?.abort(); const controller = new AbortController(); this.baselineController = controller;
      this.baselines = {};
      const view = this.view, period = this.baselinePeriod;
      if (!view || !period || this.disposed) { if (!this.disposed) this.render(); return; }
      const queue = buildExadataCards(view).filter(card => card.baselineSupported);
      this.render();
      const worker = async () => {
        while (queue.length && !controller.signal.aborted && !this.disposed) {
          const card = queue.shift(), key = exadataBaselineKey(card);
          if (!card.samples.some(sample => sample.quality === "OBSERVED" && exadataNumeric(sample.value) !== null)) {
            this.baselines[key] = "当前没有有效采样，不计算均值偏离。"; this.render(); continue;
          }
          try {
            const result = await KBotAIOpsAuth.request(this.base + "/baselines", { method: "POST", signal: controller.signal,
              body: JSON.stringify({ component_id: card.componentId, metric_code: card.metricCode, period }) });
            if (controller.signal.aborted || this.disposed || this.view !== view) return;
            this.baselines[key] = result.binding_id === view.binding_id && result.binding_version === view.binding_version && result.source_version === view.source_version && result.catalog_sha256 === view.catalog_sha256 && result.component_id === card.componentId && result.metric_code === card.metricCode && result.unit === card.unit && result.baseline.period === period
              ? result : "资源、目录或单位已变化，请刷新当前指标后比较。";
          } catch (error) {
            if (controller.signal.aborted || this.disposed) return;
            this.baselines[key] = error.message || "历史均值读取失败，当前快照保留。";
          }
          this.render();
        }
      };
      await Promise.all([worker(), worker()]);
    }
    renderBaseline(card) {
      if (!this.baselinePeriod || !card.baselineSupported) return "";
      const result = this.baselines[exadataBaselineKey(card)];
      return '<section class="exadata-baseline" aria-label="历史均值偏离">' + exadataBaselineLines(result, this.baselinePeriod).map(line => '<p>' + esc(line) + '</p>').join("")
        + (result && typeof result !== "string" ? '<small>窗口 ' + esc(fmt(result.baseline.start)) + ' 至 ' + esc(fmt(result.baseline.end)) + '；留存采样均值，覆盖率未知。偏离不是故障阈值。</small>' : "") + '</section>';
    }
    async refresh() { return this.load(false); }
    async load(diagnosis) {
      if (this.disposed) return;
      this.controller?.abort(); const controller = new AbortController(); this.controller = controller;
      this.baselineController?.abort(); this.baselines = {};
      this.loading = true; this.view = null; this.report = null;
      this.render(this.mode === "HISTORY" ? "正在读取 OEM 数值历史…" : "正在读取 OEM 最新快照…");
      try {
        const result = await KBotAIOpsAuth.request(this.base + (diagnosis ? "/diagnoses" : "/view"), {
          method: "POST", headers: diagnosis ? { "Idempotency-Key": KBotAIOpsAuth.uuid() } : {},
          body: JSON.stringify({ mode: this.mode, scene: this.scene, component_id: this.component || null, offset: this.offset, limit: this.scene === "overview" ? 24 : 12 }),
          signal: controller.signal,
        });
        if (this.disposed || controller.signal.aborted) return;
        this.view = diagnosis ? result.evidence : result; this.members = this.view.members;
        this.report = diagnosis ? result : null; this.loading = false; this.render(); void this.loadBaselines();
      } catch (error) {
        if (!controller.signal.aborted && !this.disposed) { this.loading = false; this.render(error.message || "Exadata 查询失败", true); }
      }
    }
    renderCard(card, index) {
      const history = this.view.query_mode === "HISTORY";
      const chart = history && exadataHasTrend(card);
      return `<article class="monitoring-panel is-native exadata-metric-card tone-${esc(card.tone)}" data-metric-code="${esc(card.metricCode)}" data-component-id="${esc(card.componentId)}">
        <header><div><h3 title="${esc(card.title)}">${esc(card.title)}</h3><p title="${esc(card.componentName)}">${esc(card.componentName)}</p></div><span class="status-pill ${card.tone === "warning" ? "status-warning" : "status-muted"}">${esc(card.quality)}</span></header>
        <div class="exadata-card-value"><strong title="${esc(card.value)}">${esc(card.value)}</strong><span>${esc(card.valueUnit)}</span></div>
        <p class="exadata-card-time" title="${esc(card.observedAt)}">${history ? "窗口末次采样" : "采集时间"} · ${esc(fmt(card.observedAt))}</p>
        ${chart ? `<div class="exadata-chart" data-chart="${index}" role="img" aria-label="${esc(card.title)}数值历史；不同维度分列，空值不补零"></div>` : `<div class="exadata-snapshot">
          ${card.percent !== null ? `<div class="exadata-percent" role="img" aria-label="原生百分比 ${esc(card.value)}%，不代表健康阈值"><div><i style="width:${card.percent}%"></i></div><p><span>0%</span><span>100%</span></p></div>` : ""}
          ${card.series.length > 1 ? `<div class="exadata-dimensions">${card.series.slice(0, 4).map(series => `<div><span title="${esc(series.label)}">${esc(series.label)}</span><strong>${esc(exadataRaw(series.latest.value))} ${esc(card.valueUnit)}</strong></div>`).join("")}</div>${card.series.length > 4 ? `<p>另 ${card.series.length - 4} 个维度请查看数据。</p>` : ""}` : ""}
          <p>${esc(card.gaps[0]?.detail || (history ? "当前指标没有可绘制的数值历史，不回退快照。" : card.numeric ? "最新采集快照，不生成历史趋势。" : "最新采集值。"))}</p></div>`}
        <footer><code title="${esc(card.selector)}">${esc(card.selector)}</code><span>单位 ${esc(exadataUnit(card.unit))}</span></footer>
        ${this.renderBaseline(card)}
        <details class="exadata-evidence"><summary>查看数据 · ${card.samples.length} 条</summary><p>图表仅为数值近似；原始值、维度、质量与实际采集时间如下。未验证阈值，不判定健康。</p><div class="exadata-table-scroll"><table><thead><tr><th>采集时间</th><th>原始值</th><th>维度</th><th>质量</th></tr></thead><tbody>${card.samples.map(sample => `<tr><td title="${esc(sample.observed_at)}">${esc(fmt(sample.observed_at))}</td><td>${esc(exadataRaw(sample.value))} · ${esc(exadataUnit(sample.unit))}</td><td>${esc(exadataDimensions(sample.dimensions))}</td><td>${esc(exadataQuality(sample.quality))}</td></tr>`).join("")}</tbody></table></div>${card.gaps.map(gap => `<p>${esc(gap.code)} · ${esc(gap.detail)}</p>`).join("")}</details>
      </article>`;
    }
    renderStatus(card) {
      return `<div class="exadata-status-item" data-metric-code="${esc(card.metricCode)}" data-component-id="${esc(card.componentId)}">
        ${exadataStatusItems(card).map(item => `<div class="exadata-status-row"><div class="exadata-status-identity"><strong>${esc(card.title)}</strong><span>${esc(card.componentName)}</span>${item.dimension ? `<small>${esc(item.dimension)}</small>` : ""}</div><span class="exadata-status-value state-${esc(item.tone)}"><i aria-hidden="true"></i>${esc(item.label)}</span><div class="exadata-status-time"><span>${esc(item.quality)}</span><time title="${esc(item.observedAt)}">${esc(fmt(item.observedAt))}</time></div></div>`).join("")}
        <details class="exadata-evidence"><summary>查看数据 · ${card.samples.length} 条</summary><code>${esc(card.selector)}</code><div class="exadata-table-scroll"><table><thead><tr><th>采集时间</th><th>原始值</th><th>维度</th><th>质量</th></tr></thead><tbody>${card.samples.map(sample => `<tr><td>${esc(fmt(sample.observed_at))}</td><td>${esc(exadataRaw(sample.value))}</td><td>${esc(exadataDimensions(sample.dimensions))}</td><td>${esc(exadataQuality(sample.quality))}</td></tr>`).join("")}</tbody></table></div>${card.gaps.map(gap => `<p>${esc(gap.code)} · ${esc(gap.detail)}</p>`).join("")}</details>
      </div>`;
    }
    selectScene(id) {
      this.scene = id;
      if (id === "overview" || !exadataSceneMembers(this.members, id).some(member => member.target_id === this.component)) this.component = "";
      this.offset = 0; this.load(false);
    }
    openDevice(id) {
      this.component = id;
      const member = this.members.find(item => item.target_id === id);
      if (member) this.scene = exadataDeviceScene(member.target_type);
      this.offset = 0; this.load(false);
    }
    renderResource(card) {
      return `<section class="exadata-resource" data-resource-code="${esc(card.metricCode)}"><div><span>${esc(card.title)}</span><strong>${esc(card.value)} ${esc(card.valueUnit)}</strong></div>
        ${card.percent !== null ? `<div class="exadata-resource-meter" role="img" aria-label="${esc(card.title)} ${esc(card.value)}%"><i style="width:${card.percent}%"></i></div>` : ""}
        <small>${esc(card.quality)} · ${esc(fmt(card.observedAt))}</small>${card.baselineSupported && this.baselinePeriod ? `<p class="exadata-baseline-summary">${esc(exadataBaselineSummary(this.baselines[exadataBaselineKey(card)], this.baselinePeriod))}</p>` : ""}<details class="exadata-evidence"><summary>均值与证据</summary>${this.renderBaseline(card)}<code>${esc(card.selector)}</code><div class="exadata-table-scroll"><table><thead><tr><th>采集时间</th><th>原始值</th><th>维度</th><th>质量</th></tr></thead><tbody>${card.samples.map(sample => `<tr><td>${esc(fmt(sample.observed_at))}</td><td>${esc(exadataRaw(sample.value))}</td><td>${esc(exadataDimensions(sample.dimensions))}</td><td>${esc(sample.quality)}</td></tr>`).join("")}</tbody></table></div>${card.gaps.map(gap => `<p>${esc(gap.code)} · ${esc(gap.detail)}</p>`).join("")}</details></section>`;
    }
    renderDevices(view, cards) {
      const devices = buildExadataDevices(view), assets = devices.filter(device => exadataDeviceGroup(device.member.target_type) === "assets");
      return `<section class="exadata-overview" aria-label="Exadata 整机设备">${[["nodes", "节点资源与服务"], ["equipment", "互联与供电设备"]].map(([id, label]) => {
        const group = devices.filter(device => exadataDeviceGroup(device.member.target_type) === id);
        return `<section class="exadata-status-panel" data-device-group="${id}"><h3>${label} <span>${group.length}</span></h3>${!group.length ? '<p class="exadata-device-hint">暂无已确认设备</p>' : ""}${group.map(device => {
          const componentCards = cards.filter(card => card.componentId === device.member.target_id);
          return `<article class="exadata-device" data-device-id="${esc(device.member.target_id)}"><header><div><span>${esc(device.typeLabel)}</span><h4>${esc(device.member.display_name)}</h4></div><button class="secondary-button" type="button" data-device="${esc(device.member.target_id)}">设备详情 →</button></header>
          ${componentCards.some(card => !card.status) ? `<div class="exadata-resource-row">${componentCards.filter(card => !card.status).map(card => this.renderResource(card)).join("")}</div>` : ""}
          <div class="exadata-device-states">${componentCards.filter(card => card.status).map(card => this.renderStatus(card)).join("")}</div>
          ${device.note ? `<p class="exadata-device-hint">${esc(device.note)}</p>` : ""}</article>`;
        }).join("")}</section>`;
      }).join("")}
      ${assets.length ? `<details class="exadata-assets"><summary>资产结构 · ${assets.length}</summary><ul>${assets.map(device => `<li data-device-id="${esc(device.member.target_id)}"><span>${esc(device.typeLabel)}</span><strong>${esc(device.member.display_name)}</strong></li>`).join("")}</ul><p>结构对象仅作目录，不代替设备可用性。</p></details>` : ""}
      ${exadataOverviewNotes(view).length ? `<details class="exadata-gaps"><summary>设备发现缺口 · ${exadataOverviewNotes(view).length}</summary>${exadataOverviewNotes(view).map(note => `<p>${esc(note)}</p>`).join("")}</details>` : ""}</section>`;
    }
    renderConfiguration(cards) {
      if (!cards.length) return "";
      return `<details class="exadata-configuration"><summary>配置、标识与原始值 · ${cards.length}</summary>${cards.map(card => card.status ? this.renderStatus(card) : `<section class="exadata-config-item"><header><strong>${esc(card.title)}</strong><span>${esc(card.componentName)}</span></header><p>${esc(card.value)} ${esc(card.valueUnit)} · ${esc(card.quality)} · ${esc(fmt(card.observedAt))}</p><code>${esc(card.selector)}</code><details class="exadata-evidence"><summary>查看数据 · ${card.samples.length} 条</summary><pre>${esc(JSON.stringify(card.samples, null, 2))}</pre>${card.gaps.map(gap => `<p>${esc(gap.code)} · ${esc(gap.detail)}</p>`).join("")}</details></section>`).join("")}</details>`;
    }
    render(message = "", failed = false) {
      const preserve = this.renderedView && this.renderedView === this.view;
      const opened = preserve ? [...this.root.querySelectorAll("details")].map((node, index) => node.open ? index : -1).filter(index => index >= 0) : [];
      const focusedIndex = preserve ? [...this.root.querySelectorAll("button,select,summary")].indexOf(document.activeElement) : -1;
      this.clearCharts();
      const view = this.view, cards = view ? buildExadataCards(view) : [], disabled = this.loading ? "disabled" : "";
      const statuses = cards.filter(card => exadataCardSection(card) === "status"), metrics = cards.filter(card => exadataCardSection(card) === "performance"), configuration = cards.filter(card => exadataCardSection(card) === "configuration");
      const overview = this.scene === "overview" && !this.component && view?.device_coverage?.length;
      const devices = overview ? this.renderDevices(view, cards) : "";
      const directory = exadataSceneMembers(this.members, this.scene), current = this.members.find(member => member.target_id === this.component);
      const workspaceStart = `<div class="exadata-workspace"><aside class="exadata-directory" aria-label="设备目录"><h3>设备目录</h3><button class="secondary-button" data-home type="button">← 返回整机</button><button class="secondary-button" data-directory-all type="button" aria-pressed="${!this.component}">当前场景全部设备</button>${directory.map(member => `<button class="secondary-button" data-directory-device="${esc(member.target_id)}" type="button" aria-pressed="${this.component === member.target_id}">${esc(member.display_name)}</button>`).join("")}</aside><section class="exadata-detail" aria-label="设备详情"><header class="exadata-detail-heading"><p class="eyebrow">${esc(current?.target_type || "已确认成员")}</p><h3>${esc(current?.display_name || scenes.find(([id]) => id === this.scene)?.[1])}</h3>${current ? `<code>${esc(current.target_id)}</code>` : ""}</header>`;
      this.root.setAttribute("aria-busy", String(this.loading));
      this.root.innerHTML = `<header class="exadata-heading"><div><p class="eyebrow">EXADATA · OEM</p><h2>Exadata 设备监控</h2><p>整机巡检 · 设备详情 · 只读分析</p></div><span class="status-pill status-muted">${this.mode === "HISTORY" ? "数值历史" : "最新快照"}</span></header>
      <nav class="exadata-scenes" aria-label="Exadata 监控场景">${scenes.map(([id, name]) => `<button class="secondary-button" type="button" data-scene-id="${id}" ${this.scene === id ? 'aria-current="page"' : ""} ${disabled}>${name}</button>`).join("")}</nav><div class="exadata-toolbar"><label class="control-field"><span class="field-label">查询模式</span><select data-mode ${disabled}><option value="LATEST_SNAPSHOT" ${this.mode === "LATEST_SNAPSHOT" ? "selected" : ""}>最新快照</option><option value="HISTORY" ${this.mode === "HISTORY" ? "selected" : ""}>过去 1 小时历史（仅数值）</option></select></label><label class="control-field"><span class="field-label">设备范围</span><select data-component ${disabled}><option value="">当前场景全部设备</option>${directory.map(member => `<option value="${esc(member.target_id)}" ${member.target_id === this.component ? "selected" : ""}>${esc(member.display_name)} · ${esc(member.target_type)}</option>`).join("")}</select></label><label class="control-field"><span class="field-label">均值偏离</span><select data-baseline ${disabled}><option value="" ${!this.baselinePeriod ? "selected" : ""}>不比较</option><option value="1w" ${this.baselinePeriod === "1w" ? "selected" : ""}>7 天均值</option><option value="1m" ${this.baselinePeriod === "1m" ? "selected" : ""}>1 个月（30 天）均值</option></select></label><div class="exadata-actions"><button class="secondary-button" data-refresh type="button" ${disabled}>刷新指标</button><button class="secondary-button" data-diagnose type="button" ${disabled}>只读诊断当前页</button></div></div>
      ${message ? `<p class="exadata-message ${failed ? "notice notice-error" : ""}" role="${failed ? "alert" : "status"}">${esc(message)}</p>` : ""}
      ${view ? `<div class="exadata-result-bar"><p role="status">${view.samples.length} 条采样 · 场景查询 ${this.offset + (view.total_queries ? 1 : 0)}–${Math.min(this.offset + (this.scene === "overview" ? 24 : 12), view.total_queries)} / ${view.total_queries}<small>获取时间 ${esc(fmt(view.captured_at))} · 目录 ${esc(view.catalog_version)}</small></p><div class="exadata-actions"><button class="secondary-button" data-first type="button" ${this.offset === 0 ? "disabled" : ""}>场景首页</button><button class="secondary-button" data-next type="button" ${view.next_offset === null ? "disabled" : ""}>下一页指标</button></div></div>${devices || (workspaceStart + (statuses.length ? `<section class="exadata-status-panel" aria-label="Exadata 状态"><h3>可用性与运行状态 · ${statuses.length}</h3>${statuses.map(card => this.renderStatus(card)).join("")}</section>` : ""))}${!overview && metrics.length ? `<section class="exadata-performance"><h3>性能与容量 · ${metrics.length}</h3><div class="monitoring-panels exadata-metric-grid" aria-label="Exadata 指标卡片">${metrics.map((card, index) => this.renderCard(card, index)).join("")}</div></section>` : ""}${!overview ? this.renderConfiguration(configuration) : ""}${!cards.length ? '<p class="exadata-empty">此页没有可展示的指标。请查看数据缺口或选择其他场景，不按零值或健康处理。</p>' : ""}${!overview ? "</section></div>" : ""}<details class="exadata-gaps"><summary>数据缺口与采集说明 · ${view.gaps.length} 项</summary><ul>${view.gaps.map(gap => `<li><strong>${esc(exadataGapLabel(gap.code))} · ${esc(gap.code)}</strong>${gap.component_id ? `<p>${esc(this.members.find(member => member.target_id === gap.component_id)?.display_name || gap.component_id)}</p>` : ""}<p>${esc(gap.detail)}</p>${gap.metric_code ? `<code>${esc(gap.metric_code)}</code>` : ""}</li>`).join("")}</ul></details>` : ""}
      ${this.report ? `<section class="exadata-report"><h3>只读诊断报告</h3><p>${this.report.status === "FINDINGS" ? "发现组件状态异常，请结合证据核查。" : "尚不足以判定故障或健康。"}范围仅限本次分页，不代表整机健康。</p><code>${esc(this.report.report_id)}</code><ul>${this.report.findings.map(finding => `<li>${esc(finding.code)} · ${esc(finding.detail)}</li>`).join("")}</ul></section>` : ""}`;
      this.renderedView = view;
      const details = this.root.querySelectorAll("details");
      opened.forEach(index => { if (details[index]) details[index].open = true; });
      if (focusedIndex >= 0) this.root.querySelectorAll("button,select,summary")[focusedIndex]?.focus({ preventScroll: true });
      const query = selector => this.root.querySelector(selector);
      query("[data-mode]").onchange = event => { this.mode = event.target.value; this.offset = 0; this.load(false); };
      this.root.querySelectorAll("[data-scene-id]").forEach(button => { button.onclick = () => this.selectScene(button.dataset.sceneId); });
      query("[data-component]").onchange = event => { this.component = event.target.value; this.offset = 0; this.load(false); };
      query("[data-baseline]").onchange = event => { this.baselinePeriod = event.target.value; void this.loadBaselines(); };
      query("[data-refresh]").onclick = () => this.load(false);
      query("[data-diagnose]").onclick = () => this.load(true);
      if (!view) return;
      query("[data-first]").onclick = () => { this.offset = 0; this.load(false); };
      this.root.querySelectorAll("[data-device]").forEach(button => { button.onclick = () => { this.openDevice(button.dataset.device); }; });
      if (query("[data-home]")) query("[data-home]").onclick = () => this.selectScene("overview");
      if (query("[data-directory-all]")) query("[data-directory-all]").onclick = () => { this.component = ""; this.offset = 0; this.load(false); };
      this.root.querySelectorAll("[data-directory-device]").forEach(button => { button.onclick = () => { this.component = button.dataset.directoryDevice; this.offset = 0; this.load(false); }; });
      query("[data-next]").onclick = () => { this.offset = view.next_offset; this.load(false); };
      this.root.querySelectorAll("[data-chart]").forEach(node => {
        if (!globalThis.echarts) { node.textContent = "图表引擎不可用，请查看原始数据。"; return; }
        const css = getComputedStyle(this.root), color = name => css.getPropertyValue(`--${name}`).trim();
        const chart = echarts.init(node);
        chart.setOption(exadataChartOption(metrics[Number(node.dataset.chart)], Object.fromEntries(["accent", "info", "muted", "line", "surface", "ink"].map(name => [name, color(name)]))));
        const observer = new ResizeObserver(() => chart.resize()); observer.observe(node);
        this.charts.push(chart); this.observers.push(observer);
      });
    }
  }
  globalThis.KBotAIOpsExadataMetrics = ExadataMetrics;
})();
