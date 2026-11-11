(function () {
  "use strict";
  const esc = value => KBotAIOpsShell.escape(String(value ?? ""));
  const scenes = [["overview", "整机/Cell 状态"], ["cell", "Cell 资源"], ["disk", "磁盘 IO"], ["flash", "Flash/Smart IO"], ["iorm", "IORM"], ["network", "互联/交换机"], ["capacity", "容量/硬件"]];
  class ExadataMetrics {
    constructor(root, options) {
      this.root = root; this.options = options; this.key = `${options.sourceId}:${options.instanceId}`;
      this.mode = "LATEST_SNAPSHOT"; this.scene = "overview"; this.component = ""; this.offset = 0; this.view = null; this.controller = null; this.disposed = false;
      this.base = `/api/v1/apps/aiops/business-systems/${encodeURIComponent(options.systemId)}/exadata/${encodeURIComponent(options.instanceId)}/sources/${encodeURIComponent(options.sourceId)}`;
      this.refresh();
    }
    dispose() { this.disposed = true; this.controller?.abort(); }
    async refresh() { return this.load(false); }
    async load(diagnosis) {
      if (this.disposed) return;
      this.controller?.abort(); const controller = new AbortController(); this.controller = controller;
      this.render("正在读取 OEM 最新快照…");
      try {
        const result = await KBotAIOpsAuth.request(this.base + (diagnosis ? "/diagnoses" : "/view"), {
          method: "POST", headers: diagnosis ? { "Idempotency-Key": crypto.randomUUID() } : {},
          body: JSON.stringify({ mode: this.mode, scene: this.scene, component_id: this.component || null, offset: this.offset, limit: 12 }),
          signal: controller.signal,
        });
        if (this.disposed || controller.signal.aborted) return;
        this.view = diagnosis ? result.evidence : result;
        this.report = diagnosis ? result : null; this.render();
      } catch (error) { if (!controller.signal.aborted && !this.disposed) { this.view = null; this.report = null; this.render(error.message || "Exadata 查询失败"); } }
    }
    render(message = "") {
      const view = this.view;
      this.root.innerHTML = `<header><h2>Exadata OEM 专用监控与只读诊断</h2><p>默认最新采集快照，不是秒级实时；历史为 OEM 留存分辨率，不推断精确增量或跨组件均值。未知阈值不默认健康。</p></header>
      <label>查询模式<select data-mode><option value="LATEST_SNAPSHOT" ${this.mode === "LATEST_SNAPSHOT" ? "selected" : ""}>最新快照</option><option value="HISTORY" ${this.mode === "HISTORY" ? "selected" : ""}>过去 1 小时历史（仅数值）</option></select></label><label>场景<select data-scene>${scenes.map(([id, name]) => `<option value="${id}" ${this.scene === id ? "selected" : ""}>${name}</option>`).join("")}</select></label>
      <label>组件<select data-component><option value="">全部已确认成员</option>${(view?.members || []).map(m => `<option value="${esc(m.target_id)}" ${m.target_id === this.component ? "selected" : ""}>${esc(m.target_type)} · ${esc(m.display_name)}</option>`).join("")}</select></label>
      <button data-refresh type="button">刷新快照</button><button data-diagnose type="button">只读诊断当前页并保存报告</button>
      <p role="status">${esc(message)}${view ? `模式 ${esc(view.query_mode)} · 目录 ${esc(view.catalog_version)} · 场景查询 ${this.offset + 1}–${Math.min(this.offset + 12, view.total_queries)} / ${view.total_queries} · 获取时间 ${esc(view.captured_at)}` : ""}</p>
      ${view ? `<button data-first type="button">场景首页</button><button data-next type="button" ${view.next_offset === null ? "disabled" : ""}>下一页指标</button><div class="table-scroll"><table><thead><tr><th>组件 / 指标</th><th>维度</th><th>原始值 / 原生单位</th><th>采集时间 / 质量</th></tr></thead><tbody>${view.samples.map(s => `<tr><td>${esc(view.members.find(m => m.target_id === s.component_id)?.display_name || s.component_id)}<br>${esc(s.selector)}</td><td>${esc(JSON.stringify(s.dimensions))}</td><td>${s.value === null || s.value === "" ? "无数据" : esc(s.value)} · ${esc(s.unit || "OEM 未声明单位")} · ${esc(s.native_value_type)}</td><td>${esc(s.observed_at)} · ${esc(s.quality)}</td></tr>`).join("")}</tbody></table></div><ul>${view.gaps.map(g => `<li>${esc(g.code)} · ${esc(g.metric_code)} · ${esc(g.detail)}</li>`).join("")}</ul>` : ""}
      ${this.report ? `<section><h3>诊断报告 ${esc(this.report.report_id)}</h3><p>结论：证据不足，不能判定健康。范围仅限本次分页，不代表整机健康。</p><ul>${this.report.findings.map(f => `<li>${esc(f.code)} · ${esc(f.detail)}</li>`).join("")}</ul></section>` : ""}`;
      const query = selector => this.root.querySelector(selector);
      query("[data-mode]").onchange = event => { this.mode = event.target.value; this.offset = 0; this.load(false); };
      query("[data-scene]").onchange = event => { this.scene = event.target.value; this.offset = 0; this.load(false); };
      query("[data-component]").onchange = event => { this.component = event.target.value; this.offset = 0; this.load(false); };
      query("[data-refresh]").onclick = () => this.load(false);
      query("[data-diagnose]").onclick = () => this.load(true);
      if (view) {
        query("[data-first]").onclick = () => { this.offset = 0; this.load(false); };
        query("[data-next]").onclick = () => { this.offset = view.next_offset; this.load(false); };
      }
    }
  }
  globalThis.KBotAIOpsExadataMetrics = ExadataMetrics;
})();
