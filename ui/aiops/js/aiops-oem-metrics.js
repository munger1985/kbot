(function () {
  "use strict";
  const esc = value => KBotAIOpsShell.escape(String(value ?? ""));
  const identity = dimensions => JSON.stringify(Object.entries(dimensions).sort(([a], [b]) => a.localeCompare(b)));
  const label = dimensions => Object.entries(dimensions).map(([key, value]) => `${key}=${value}`).join(" · ") || "资源整体";
  const number = value => value !== null && !(typeof value === "string" && value.trim() === "") && Number.isFinite(Number(value)) ? Number(value) : null;
  class OEMMetrics {
    constructor(root, options) {
      Object.assign(this, options);
      this.root = root; this.key = `${this.sourceId}:${this.instanceId}`;
      this.base = `/api/v1/apps/aiops/monitoring/sources/${encodeURIComponent(this.sourceId)}/resources/${encodeURIComponent(this.instanceId)}`;
      this.charts = new Map(); this.observers = new Map();
      root.innerHTML = `<header class="oem-overview-heading"><div><h2>OEM 关键指标总览</h2><p>围绕可用性、性能压力和容量风险展示。来源：${esc(this.sourceName)}</p></div><button class="secondary-button" type="button" data-more aria-haspopup="dialog">更多指标</button></header><p data-overview-note></p><p data-overview-status role="status"></p><div class="monitoring-panels oem-overview-panels"></div>`;
      root.onclick = event => {
        if (event.target.closest("[data-more]")) this.openMore();
        if (event.target.closest("[data-retry-overview]")) void this.load();
        const button = event.target.closest("[data-oem-data]");
        if (button) this.openData(button.dataset.oemData, button);
      };
      void this.load();
    }
    baselineLabel() { return this.baseline === "1m" ? "过去 1 个月（30 天）" : "过去 7 天"; }
    clearCharts() { this.charts.forEach(chart => chart.dispose()); this.observers.forEach(observer => observer.disconnect()); this.charts.clear(); this.observers.clear(); }
    dispose() { this.controller?.abort(); this.explorer?.dispose(); this.moreDialog?.remove(); this.dataDialog?.remove(); this.clearCharts(); this.root.onclick = null; }
    refresh(window, baseline) { this.window = window; this.baseline = baseline; this.explorer?.refresh(window); void this.load(); }
    async load() {
      this.controller?.abort(); this.clearCharts();
      const active = new AbortController(); this.controller = active; this.results = new Map();
      const status = this.root.querySelector("[data-overview-status]");
      status.textContent = "正在更新关键指标…";
      this.root.querySelector("[data-overview-note]").textContent = `基线：${this.baselineLabel()}有效采样均值；按相同维度比较，OEM 留存粒度与覆盖率限制请见数据详情。未配置阈值不代表健康。`;
      try {
        const catalog = await KBotAIOpsAuth.request(`${this.base}/metrics?limit=1`, { signal: active.signal });
        if (active.signal.aborted) return;
        this.specs = catalog.overview_panels;
        this.root.querySelector(".oem-overview-panels").innerHTML = this.specs.map(spec => `<div data-oem-panel="${esc(spec.panel_id)}"></div>`).join("");
        this.specs.forEach(spec => this.renderPanel(spec, null, spec.metric ? "正在读取 OEM 采样…" : spec.unavailable_reason));
        const queue = this.specs.filter(spec => spec.metric);
        let cursor = 0;
        const worker = async () => {
          while (!active.signal.aborted && cursor < queue.length) {
            const spec = queue[cursor++];
            try {
              const samples = await KBotAIOpsAuth.request(`${this.base}/metric-samples`, { method: "POST", signal: active.signal,
                body: JSON.stringify({ metric_name: spec.metric.name, mode: spec.mode, window: this.window,
                  baseline: spec.baseline_supported ? this.baseline : null }) });
              if (!active.signal.aborted) this.renderPanel(spec, samples);
            } catch (failure) {
              if (!active.signal.aborted) this.renderPanel(spec, null, failure.message || "OEM 采样读取失败");
            }
          }
        };
        await Promise.all(Array.from({ length: Math.min(3, queue.length) }, worker));
        if (!active.signal.aborted) status.textContent = catalog.truncated ? "来源目录读取被截断，部分总览指标可能未发现。" : "关键指标查询完成；采集成功不等于资源健康。";
      } catch (failure) {
        if (!active.signal.aborted) status.innerHTML = `${esc(failure.message || "OEM 总览读取失败")} <button class="secondary-button" data-retry-overview>重试总览</button>`;
      }
    }
    renderPanel(spec, samples, failure) {
      this.results.set(spec.panel_id, { spec, samples, failure });
      const groups = new Map();
      (samples?.points || []).forEach(point => { const key = identity(point.dimensions); groups.set(key, [...(groups.get(key) || []), point]); });
      (samples?.baseline?.series || []).forEach(series => { const key = identity(series.dimensions); if (!groups.has(key)) groups.set(key, []); });
      const numeric = spec.metric?.native_value_type === "NUMBER";
      const points = [...(samples?.points || [])].sort((a, b) => b.observed_at.localeCompare(a.observed_at));
      const latest = points[0];
      const stale = Boolean(latest && !latest.current_eligible && latest.value !== null);
      const unit = spec.metric?.unit === "%" ? "percent" : spec.metric?.unit || "1";
      const messages = [failure, samples?.truncated ? "采样已截断，不代表完整窗口。" : null, samples?.baseline?.unavailable_reason,
        spec.mode === "LATEST" ? "最近采集快照，非历史时序；请核对实际采集时间。" : null].filter(Boolean);
      const series = [...groups].map(([key, values]) => {
        const historical = samples?.baseline?.series.find(item => identity(item.dimensions) === key);
        const dimensions = values[0]?.dimensions || historical?.dimensions || {};
        return { series_key: key, source_id: this.sourceId, source_display_name: this.sourceName, source_type: "OEM",
          instance_id: this.instanceId, instance_display_name: this.instanceName, dimensions: { ...dimensions, __metric_label: label(dimensions) },
          coverage_ratio: null, points: values.sort((a, b) => a.observed_at.localeCompare(b.observed_at)).map(point => ({ observed_at: point.observed_at,
            value: numeric ? number(point.value) : null, raw_value: point.value, current_eligible: point.current_eligible,
            quality: point.quality === "OBSERVED" ? "GOOD" : point.quality === "STALE" ? "STALE" : "INVALID" })),
          baseline: samples?.baseline ? { ...samples.baseline, average: historical ? number(historical.average) : null, sample_count: historical?.sample_count || 0, coverage_ratio: null } : null };
      });
      series.sort((a, b) => (spec.ranking === "ASC" ? 1 : -1) * ((a.points.at(-1)?.value ?? 0) - (b.points.at(-1)?.value ?? 0)));
      const candidate = series.find(item => item.points.at(-1)?.quality === "GOOD" && item.points.at(-1)?.current_eligible && number(item.points.at(-1)?.value ?? null) !== null);
      const current = numeric ? candidate?.points.at(-1)?.raw_value ?? "—" : latest?.current_eligible ? latest.value ?? "—" : "—";
      const baselineCandidate = candidate || series.find(item => item.baseline?.average !== null && item.baseline?.average !== undefined);
      const average = baselineCandidate?.baseline?.average;
      const difference = !stale && !samples?.truncated && candidate && average !== null && average !== undefined ? candidate.points.at(-1).value - average : null;
      const delta = !candidate && average !== null && average !== undefined ? "无当前采样，暂不比较" : difference === null ? "无有效基线" : difference === 0 ? "当前与基线持平" : `当前${difference > 0 ? "高" : "低"} ${Math.abs(difference).toLocaleString("zh-CN", { maximumFractionDigits: 2 })}${unit === "percent" ? " 个百分点" : ` ${spec.metric.unit || ""}`}`;
      const hasData = Boolean(latest && latest.quality === "OBSERVED");
      const snapshots = series.filter(item => item.points.length);
      const chartPanel = { title: spec.title, metric_code: spec.metric?.name || spec.panel_id, unit, series, native: { mode: spec.mode, ranking: spec.ranking } };
      const chartApi = KBotAIOpsOEMPresentation, kind = chartApi.oemChartKind(chartPanel);
      const body = kind === "state" && snapshots.length ? `<div class="oem-snapshot-list">${stale ? "<p>快照超期或采集时效未核验，当前状态未知。</p>" : ""}${snapshots.slice(0, 5).map(item => ({ ...item.points.at(-1), dimensions: Object.fromEntries(Object.entries(item.dimensions).filter(([key]) => !key.startsWith("__"))) })).map(point => `<div><span>${esc(label(point.dimensions))}</span><strong>${point.raw_value === null ? "—（来源空值）" : esc(point.raw_value)} ${esc(spec.metric.unit)}</strong></div>`).join("")}${snapshots.length > 5 ? `<p>其余 ${snapshots.length - 5} 个维度请查看数据。</p>` : ""}</div>` : kind !== "state" && series.some(item => item.points.some(point => point.value !== null)) ? `<div class="monitoring-chart-canvas" data-chart-kind="${kind}" role="img"></div><p class="oem-chart-caption">${esc(chartApi.oemChartCaption(chartPanel))}</p>` : '<p class="monitoring-no-data">无有效采样。缺失点保持为空，不按 0 绘制。</p>';
      const node = [...this.root.querySelectorAll("[data-oem-panel]")].find(item => item.dataset.oemPanel === spec.panel_id);
      node.innerHTML = `<article class="monitoring-panel is-native ${hasData ? "" : "tone-none"}"><header><div><h2>${esc(spec.title)}</h2><p title="${esc(messages.join("；"))}">${esc(messages[0] || "OEM 原生采样，未配置健康阈值")}</p></div><div class="monitoring-panel-actions"><span class="status-pill">${samples?.truncated ? "结果截断" : stale ? "当前值不可用" : hasData ? "已采样" : "暂无数据"}</span><button class="row-action" data-oem-data="${esc(spec.panel_id)}" aria-haspopup="dialog">查看数据</button></div></header><div class="monitoring-current-value"><strong>${esc(current)} ${numeric ? esc(spec.metric?.unit === "n/a" ? "" : spec.metric?.unit) : ""}</strong><span>${esc(latest ? KBotAIOpsShell.fmt(latest.observed_at) : "当前值")}</span></div>${spec.baseline_supported ? `<div class="monitoring-baseline" title="${esc(samples?.baseline?.note)}"><span>${esc(this.baselineLabel())}采样均值 <strong>${average === null || average === undefined ? "无有效基线" : esc(average.toLocaleString("zh-CN", { maximumFractionDigits: 2 }))}</strong>${baselineCandidate?.baseline?.sample_count ? `<small> · ${baselineCandidate.baseline.sample_count} 条</small>` : ""}</span><b>${esc(delta)}</b></div>` : ""}${body}<footer><span>单位 ${esc(spec.metric?.unit || "未声明")}</span><span>覆盖率未知（OEM 留存采样）</span></footer></article>`;
      const canvas = node.querySelector(".monitoring-chart-canvas");
      this.charts.get(spec.panel_id)?.dispose(); this.observers.get(spec.panel_id)?.disconnect();
      if (canvas && globalThis.echarts) {
        canvas.setAttribute("aria-label", `${spec.title} · ${chartApi.oemChartCaption(chartPanel)}`);
        const chart = echarts.init(canvas);
        const css = getComputedStyle(this.root);
        chart.setOption(chartApi.oemChartOption(chartPanel, Object.fromEntries(["accent", "info", "success", "warning", "danger", "muted", "ink", "line", "surface", "surfaceRaised", "lineStrong", "lineSubtle", "accentSoft"].map(name => [name, css.getPropertyValue("--" + name.replace(/[A-Z]/g, letter => "-" + letter.toLowerCase())).trim()])), this.baselineLabel()), true);
        const observer = new ResizeObserver(() => chart.resize()); observer.observe(canvas);
        this.charts.set(spec.panel_id, chart); this.observers.set(spec.panel_id, observer);
      }
    }
    openData(panelId, trigger) {
      const { spec, samples, failure } = this.results.get(panelId);
      this.dataDialog?.remove();
      const dialog = document.createElement("dialog"); this.dataDialog = dialog; dialog.className = "oem-more-dialog";
      dialog.setAttribute("aria-label", `${spec.title}数据`);
      dialog.innerHTML = `<header><div><h2>${esc(spec.title)}</h2><p>${esc(spec.description)}</p></div><button class="icon-button" aria-label="关闭指标数据">×</button></header><p>${esc(failure || spec.unavailable_reason)}</p><p>${esc(samples?.baseline?.note)}</p><p>${esc(samples?.baseline?.unavailable_reason)}</p><p>未配置健康阈值；快照不代表历史，原始采样保留精确文本。</p>${samples?.truncated ? "<p>采样已截断，不代表完整窗口。</p>" : ""}<div class="oem-table"><table><thead><tr><th>采样时间</th><th>维度</th><th>原始值</th><th>${esc(this.baselineLabel())}采样均值</th><th>基线样本数</th></tr></thead><tbody>${(samples?.points || []).map(point => { const history = samples?.baseline?.series.find(item => identity(item.dimensions) === identity(point.dimensions)); return `<tr><td>${esc(KBotAIOpsShell.fmt(point.observed_at))}</td><td>${esc(label(point.dimensions))}</td><td>${point.value === null ? "—（来源空值）" : esc(point.value)}</td><td>${esc(history?.average || "无有效基线")}</td><td>${esc(history?.sample_count ?? "—")}</td></tr>`; }).join("")}</tbody></table></div>`;
      if (samples?.baseline) {
        const history = document.createElement("section");
        history.className = "drawer-section";
        history.innerHTML = `<p>基线范围：${esc(KBotAIOpsShell.fmt(samples.baseline.start))} 至 ${esc(KBotAIOpsShell.fmt(samples.baseline.end))}；留存采样均值，覆盖率未知。</p>${samples.baseline.series.map(item => `<p>${esc(label(item.dimensions))} · ${item.sample_count} 条有效采样 · 均值 ${esc(item.average)} ${esc(spec.metric?.unit)}</p>`).join("")}`;
        dialog.querySelector(".oem-table").before(history);
      }
      dialog.querySelector("button").onclick = () => dialog.close(); dialog.onclose = () => { dialog.remove(); trigger.isConnected && trigger.focus(); };
      document.body.append(dialog); dialog.showModal();
    }
    openMore() {
      if (this.moreDialog?.open) return;
      const trigger = this.root.querySelector("[data-more]");
      const dialog = document.createElement("dialog"); this.moreDialog = dialog; dialog.className = "oem-more-dialog"; dialog.setAttribute("aria-label", "更多 OEM 指标");
      dialog.innerHTML = '<header><div><h2>更多 OEM 指标</h2><p>用于专项查询；默认总览不会轮询全部目录。</p></div><button class="icon-button" aria-label="关闭更多指标">×</button></header><section class="oem-metrics"></section>';
      dialog.querySelector("button").onclick = () => dialog.close();
      dialog.onclose = () => { this.explorer?.dispose(); this.explorer = null; dialog.remove(); trigger.isConnected && trigger.focus(); };
      document.body.append(dialog); dialog.showModal();
      this.explorer = new KBotAIOpsOEMMetricExplorer(dialog.querySelector("section"), { sourceId: this.sourceId, instanceId: this.instanceId, window: this.window });
    }
  }
  globalThis.KBotAIOpsOEMMetrics = OEMMetrics;
})();
