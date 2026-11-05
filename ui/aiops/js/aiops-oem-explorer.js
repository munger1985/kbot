(function () {
  "use strict";
  const esc = value => KBotAIOpsShell.escape(String(value ?? ""));
  const dimensions = value => Object.entries(value).map(([key, item]) => `${key}=${item}`).join(" · ") || "资源整体";
  class OEMMetricExplorer {
    constructor(root, { sourceId, instanceId, window }) {
      this.root = root;
      this.key = `${sourceId}:${instanceId}`;
      this.base = `/api/v1/apps/aiops/monitoring/sources/${encodeURIComponent(sourceId)}/resources/${encodeURIComponent(instanceId)}`;
      this.window = window; this.offset = 0; this.search = ""; this.selected = null; this.mode = "HISTORY";
      root.innerHTML = `<header><h2>OEM 原生指标</h2><p>按资源实际插件发现，不受统一总览指标数量限制。原生单位与维度原样保留，不自动认定为 Prometheus 等价指标。</p></header><label class="control-field"><span class="field-label">搜索全部 OEM 指标</span><input type="search" placeholder="指标组 / 指标名称"></label><p data-catalog-status role="status"></p><div class="oem-layout"><div class="oem-directory"></div><section class="oem-samples" aria-label="OEM 指标采样"></section></div><footer></footer><p>目录可见不代表窗口内已有采样。非数值指标保留在目录中，数值时序接口不支持的类型不伪造为 0。</p>`;
      root.querySelector("input").oninput = event => {
        this.search = event.target.value;
        clearTimeout(this.timer);
        this.timer = setTimeout(() => { this.offset = 0; void this.loadCatalog(); }, 250);
      };
      root.onclick = event => {
        const button = event.target.closest("[data-metric-index]");
        if (button) { this.selected = this.catalog.entries[Number(button.dataset.metricIndex)]; this.mode = this.selected.query_supported ? "HISTORY" : "LATEST"; this.renderDirectory(); void this.loadSamples(); }
        const page = event.target.closest("[data-page]");
        if (page) { this.offset = Number(page.dataset.page); void this.loadCatalog(); }
        if (event.target.closest("[data-retry-catalog]")) void this.loadCatalog();
        if (event.target.closest("[data-retry-sample]")) void this.loadSamples();
      };
      root.onchange = event => { if (event.target.matches("[data-native-mode]")) { this.mode = event.target.value; void this.loadSamples(); } };
      void this.loadCatalog();
    }
    dispose() { clearTimeout(this.timer); this.catalogController?.abort(); this.sampleController?.abort(); this.root.onclick = null; this.root.onchange = null; }
    refresh(window) { this.window = window; void this.loadSamples(); }
    renderDirectory() {
      this.root.querySelector(".oem-directory").innerHTML = this.catalog.entries.map((metric, index) => `<button type="button" class="secondary-button" data-metric-index="${index}" aria-pressed="${this.selected?.name === metric.name}"><strong>${esc(metric.display_name)}</strong><small>${esc(metric.group)} · ${esc(metric.unit || "未声明单位")} · ${esc(metric.native_value_type)}</small><code>${esc(metric.name)}</code></button>`).join("") || "<p>没有匹配指标。</p>";
    }
    async loadCatalog() {
      this.catalogController?.abort();
      const controller = new AbortController(); this.catalogController = controller;
      const status = this.root.querySelector("[data-catalog-status]"); status.textContent = "正在读取 OEM 指标目录…";
      try {
        const query = new URLSearchParams({ search: this.search, offset: String(this.offset), limit: "100" });
        const catalog = await KBotAIOpsAuth.request(`${this.base}/metrics?${query}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        this.catalog = catalog;
        status.textContent = `匹配 ${catalog.discovered_count} 项${catalog.truncated ? "（来源目录读取已截断，不代表全部指标）" : "（目录读取完成）"}`;
        this.renderDirectory();
        this.root.querySelector("footer").innerHTML = `<button type="button" class="secondary-button" data-page="${Math.max(0, this.offset - 100)}" ${this.offset ? "" : "disabled"}>上一页</button><span>第 ${Math.floor(this.offset / 100) + 1} 页</span><button type="button" class="secondary-button" data-page="${catalog.next_offset || 0}" ${catalog.next_offset === null ? "disabled" : ""}>下一页</button>`;
        if (!this.selected) {
          this.selected = catalog.default_metric || catalog.entries.find(metric => metric.query_supported);
          this.renderDirectory(); void this.loadSamples();
        }
      } catch (error) {
        if (!controller.signal.aborted) status.innerHTML = `${esc(error.message || "OEM 指标目录读取失败")} <button class="secondary-button" data-retry-catalog>重试目录</button>`;
      }
    }
    async loadSamples() {
      this.sampleController?.abort();
      const metric = this.selected;
      if (!metric) return;
      const controller = new AbortController(); this.sampleController = controller;
      const target = this.root.querySelector(".oem-samples");
      const heading = `<h3>${esc(metric.display_name)}</h3><code>${esc(metric.name)}</code><p>原生单位：${esc(metric.unit || "未声明")} · 维度：${esc(metric.dimensions.join("、") || "资源整体")}</p><label class="control-field"><span class="field-label">读取方式</span><select data-native-mode><option value="HISTORY" ${this.mode === "HISTORY" ? "selected" : ""} ${metric.query_supported ? "" : "disabled"}>历史时序</option><option value="LATEST" ${this.mode === "LATEST" ? "selected" : ""} ${metric.latest_supported ? "" : "disabled"}>最近采集快照（非历史时序）</option></select></label>${!metric.query_supported ? `<p>该指标为 ${esc(metric.native_value_type)}，不生成数值历史曲线；读取最近采集快照并保留原始文本。</p>` : ""}`;
      if (this.mode === "LATEST" && !metric.latest_supported) { target.innerHTML = `${heading}<p>当前接口不支持此类型，不伪造为 0。</p>`; return; }
      target.innerHTML = `${heading}<p role="status">正在读取 OEM 采样…</p>`;
      try {
        const samples = await KBotAIOpsAuth.request(`${this.base}/metric-samples`, { method: "POST", signal: controller.signal, body: JSON.stringify({ metric_name: metric.name, window: this.window, mode: this.mode }) });
        if (controller.signal.aborted) return;
        target.innerHTML = `${heading}${samples.query_mode === "LATEST" ? "<p>最近采集快照：不受所选历史时间窗口约束，请核对下表采集时间。</p>" : ""}${!samples.points.length ? "<p>所选窗口内暂无采样，不代表资源宕机。可切换时间窗口重试。</p>" : samples.query_mode === "LATEST" ? "" : this.chart(samples)}${samples.truncated ? '<p role="status">采样达到单次读取预算，当前结果已截断，不代表完整窗口。</p>' : ""}${samples.points.length ? `<details ${samples.query_mode === "LATEST" ? "open" : ""}><summary>查看精确采样（${samples.points.length} 条）</summary><div class="oem-table"><table><thead><tr><th>采样时间</th><th>维度</th><th>原始值（${esc(metric.unit || "未声明单位")}）</th></tr></thead><tbody>${samples.points.map(point => `<tr><td>${esc(KBotAIOpsShell.fmt(point.observed_at))}</td><td>${esc(dimensions(point.dimensions))}</td><td>${point.value === null ? "—（来源空值）" : esc(point.value)}</td></tr>`).join("")}</tbody></table></div></details>` : ""}${samples.warnings.map(warning => `<p>${esc(warning)}</p>`).join("")}`;
        const latest = [...samples.points].sort((a, b) => b.observed_at.localeCompare(a.observed_at))[0];
        if (latest) target.querySelector("code").insertAdjacentHTML("afterend", `<p>最近一条原值：${latest.value === null ? "—（来源空值）" : esc(latest.value)} ${esc(metric.unit)} · ${esc(dimensions(latest.dimensions))} · 采集于 ${esc(KBotAIOpsShell.fmt(latest.observed_at))}</p>`);
      } catch (error) {
        if (!controller.signal.aborted) target.innerHTML = `${heading}<p role="alert">${esc(error.message || "OEM 指标采样读取失败")} <button class="secondary-button" data-retry-sample>重试采样</button></p>`;
      }
    }
    chart(samples) {
      const groups = new Map(), values = samples.points.filter(point => point.value !== null).map(point => Number(point.value)).filter(Number.isFinite);
      if (!values.length) return "<p>数值超出图表精度范围，请查看精确原值。</p>";
      const min = Math.min(...values), max = Math.max(...values), start = Date.parse(samples.window_start), duration = Date.parse(samples.window_end) - start;
      samples.points.forEach(point => { if (point.value !== null && Number.isFinite(Number(point.value))) { const key = dimensions(point.dimensions); groups.set(key, [...(groups.get(key) || []), point]); } });
      const paths = [...groups].slice(0, 32).map(([label, points], index) => {
        const coordinates = points.sort((a, b) => a.observed_at.localeCompare(b.observed_at)).map(point => [20 + (Date.parse(point.observed_at) - start) / duration * 760, 180 - (Number(point.value) - min) / (max - min || 1) * 150]);
        return `<g class="line-${index % 4}"><title>${esc(label)}</title><polyline points="${coordinates.map(point => point.join(",")).join(" ")}"/>${coordinates.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="3"/>`).join("")}</g>`;
      }).join("");
      return `<svg viewBox="0 0 800 210" role="img" aria-label="${esc(this.selected.display_name)} 原生数值趋势"><path d="M20 15V185H785"/>${paths}<text x="20" y="205">${esc(KBotAIOpsShell.fmt(samples.window_start))}</text><text x="780" y="205" text-anchor="end">${esc(KBotAIOpsShell.fmt(samples.window_end))}</text></svg><p>趋势图仅作数值近似展示（最多 32 条维度序列）；下表保留全部返回采样的精确原值。</p>`;
    }
  }
  globalThis.KBotAIOpsOEMMetricExplorer = OEMMetricExplorer;
})();
