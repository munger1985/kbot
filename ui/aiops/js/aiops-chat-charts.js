/* 对齐 Ammolite AgentChart.vue；图形与展开数据使用同一份 CHART_SPEC.v1。 */
(() => {
  "use strict";
  const instances = new Map();
  const escape = value => globalThis.KBotAIOpsShell.escape(value);
  const categories = spec => [...new Set(spec.series.flatMap(series => series.points.map(point => String(point.x))))];
  const color = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const palette = () => [color("--accent", "#315b82"),color("--info", "#3979a8"),color("--success", "#438468"),color("--warning", "#b07a32"),color("--muted", "#758393")];
  const valueLabel = (value, unit) => {
    const number = Number(value);
    return Number.isFinite(number) ? `${number.toLocaleString(undefined,{maximumFractionDigits:Math.abs(number) >= 100 ? 1 : 2})}${unit ? ` ${unit}` : ""}` : "—";
  };
  const valid = spec => spec?.schema_version === "CHART_SPEC.v1" && ["LINE","BAR","STACKED_BAR"].includes(spec.chart_type) && Array.isArray(spec.series) && spec.series.length;
  const height = spec => spec.chart_type === "LINE" ? spec.layout === "SMALL_MULTIPLES" ? Math.min(440,Math.max(260,30 + spec.series.length * 150)) : 280 : Math.min(420,Math.max(260,150 + categories(spec).length * 32));

  function html(spec) {
    if (!valid(spec)) return "";
    const summary = `${spec.title}，${spec.series.length} 个数据序列。`;
    const coverage = spec.metadata?.coverage_ratio;
    return `<figure class="agent-chart aiops-diagnosis-chart" data-chat-chart="${escape(JSON.stringify(spec))}"><figcaption><div><strong>${escape(spec.title)}</strong>${coverage != null ? `<span>监控采样覆盖率 ${(Number(coverage) * 100).toFixed(1)}%</span>` : ""}</div><button class="row-action" type="button" data-chart-data-toggle>查看数据</button></figcaption><p class="sr-only">${escape(summary)}</p>${spec.metadata?.coverage_warning ? `<p class="agent-chart-warning">${escape(spec.metadata.coverage_warning)}</p>` : ""}<div class="agent-chart-canvas" style="height:${height(spec)}px" role="img" aria-label="${escape(summary)}"></div><div class="agent-chart-data table-card" hidden><table><thead><tr><th>${escape(spec.x_axis?.label || "分类")}</th>${spec.series.map(series => `<th>${escape(series.name)}</th>`).join("")}</tr></thead><tbody>${categories(spec).map(category => `<tr><td>${escape(category)}</td>${spec.series.map(series => {const point=series.points.find(point => String(point.x) === category);return `<td>${escape(point?.display_value ?? point?.y ?? "—")}</td>`;}).join("")}</tr>`).join("")}</tbody></table></div></figure>`;
  }

  function option(spec) {
    const tooltip = {
      renderMode:"html",appendTo:"body",confine:false,enterable:true,
      extraCssText:"max-width:min(420px,calc(100vw - 24px));max-height:min(360px,calc(100vh - 24px));overflow:auto;",
      trigger:"axis",backgroundColor:color("--surface-raised","#fff"),borderColor:color("--line-strong","#cbd5e1"),textStyle:{color:color("--ink","#1f2937")},
      formatter(params) {
        const items=(Array.isArray(params)?params:[params]).filter(item => item && typeof item === "object");
        if(!items.length) return "";
        return [escape(items[0].axisValueLabel ?? items[0].name ?? ""),...items.map(item => {const series=spec.series.find(series => series.name === item.seriesName);return `${item.marker ?? ""}${escape(item.seriesName)}：${escape(valueLabel(Array.isArray(item.value)?item.value[1]:item.value,series?.unit))}`;})].join("<br>");
      },
    };
    if(spec.chart_type === "LINE") {
      const multiples=spec.layout === "SMALL_MULTIPLES" && spec.series.length > 1;
      const grids=spec.series.map((_,index) => ({left:64,right:28,top:multiples?24 + index * 174:38,height:multiples?118:undefined,bottom:multiples?undefined:48,containLabel:false}));
      const axes=spec.series.map((series,index) => ({type:"time",gridIndex:multiples?index:0,axisLabel:{hideOverlap:true},axisLine:{lineStyle:{color:color("--line","#d8dee8")}},name:multiples?series.name:undefined,nameLocation:"middle",nameGap:32}));
      const valueAxes=spec.series.map((series,index) => ({type:"value",gridIndex:multiples?index:0,name:series.unit ?? spec.y_axis?.unit ?? undefined,scale:true,splitLine:{lineStyle:{color:color("--line-subtle","#e7ebf0")}}}));
      return {animationDuration:260,color:palette(),tooltip,legend:multiples?undefined:{top:0,data:spec.series.map(series => series.name)},grid:multiples?grids:[grids[0]],xAxis:multiples?axes:[axes[0]],yAxis:multiples?valueAxes:[valueAxes[0]],series:spec.series.map((series,index) => ({type:"line",name:series.name,showSymbol:false,smooth:false,xAxisIndex:multiples?index:0,yAxisIndex:multiples?index:0,data:series.points.map(point => [point.x,point.y]),lineStyle:{width:2},emphasis:{focus:"series"}}))};
    }
    const stacked=spec.chart_type === "STACKED_BAR", labels=categories(spec);
    return {animationDuration:260,color:stacked?[color("--accent","#315b82"),color("--info","#95a9bc"),color("--surface-strong","#dce3ea")]:palette(),tooltip:{...tooltip,axisPointer:{type:"shadow"}},legend:{top:0,data:spec.series.map(series => series.name)},grid:{left:96,right:28,top:42,bottom:28,containLabel:false},xAxis:{type:"value",max:stacked && spec.layout === "NORMALIZED"?100:undefined,name:spec.y_axis?.unit ?? undefined,splitLine:{lineStyle:{color:color("--line-subtle","#e7ebf0")}}},yAxis:{type:"category",data:labels,axisLabel:{hideOverlap:false}},series:spec.series.map(series => ({type:"bar",name:series.name,stack:series.stack ?? undefined,barMaxWidth:22,data:labels.map(category => series.points.find(point => String(point.x) === category)?.y ?? 0)}))};
  }

  function mount(root) {
    for (const [node,instance] of instances) if(!node.isConnected) {instance.observer.disconnect();instance.chart.dispose();instances.delete(node);}
    root.querySelectorAll("[data-chat-chart]").forEach(node => {
      if(instances.has(node)) return;
      const spec=JSON.parse(node.dataset.chatChart);
      const canvas=node.querySelector(".agent-chart-canvas");
      const chart=globalThis.echarts.init(canvas);
      chart.setOption(option(spec),true);
      const observer=new ResizeObserver(() => chart.resize());
      observer.observe(canvas);
      instances.set(node,{chart,observer});
      node.querySelector("[data-chart-data-toggle]").onclick=event => {
        const data=node.querySelector(".agent-chart-data");
        data.hidden=!data.hidden;
        event.currentTarget.textContent=data.hidden?"查看数据":"收起数据";
      };
    });
  }
  addEventListener("pagehide",() => {for(const {chart,observer} of instances.values()) {observer.disconnect();chart.dispose();}instances.clear();});
  globalThis.AIOpsChatCharts={html,mount,option};
})();
