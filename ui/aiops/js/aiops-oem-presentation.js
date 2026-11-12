// 由 Ammolite oemMonitoringPresentation.ts 同步，合同测试校验两端展示语义一致。
(function () {
  const exports = {};
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.oemMetricCharts = void 0;
exports.oemChartKind = oemChartKind;
exports.oemChartCaption = oemChartCaption;
exports.oemChartOption = oemChartOption;
exports.oemMetricCharts = {
    'Response/Status': 'state',
    'Response/DatabaseStatus': 'state',
    'CPUUsageSummary/OverAllUsagePct': 'line',
    'Load/memUsedPct': 'line',
    'Load/cpuLoad_1min': 'line',
    'Load/swapUtil': 'donut',
    'Filesystems/pctAvailable': 'bar',
    'DiskActivitySummary/maxavserv': 'line',
    'NetworkSummary/totalNetworkThroughPutRate': 'area',
    'instance_throughput/avg_active_sessions': 'line',
    'Database_Resource_Usage/session_usage': 'donut',
    'UserBlock/count': 'bar',
    'wait_sess_cls/dbtime_waitclass_pct': 'bar',
    'instance_throughput/physreads_ps': 'area',
    'Recovery_Area/Free_Space': 'donut',
};
function numeric(point) {
    return point?.quality === 'GOOD' && typeof point.value === 'number' && Number.isFinite(point.value) ? point.value : null;
}
function latest(series) { return [...series.points].sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at)).at(-1); }
function dimensionLabel(series) {
    return Object.entries(series.dimensions).filter(([key]) => !key.startsWith('__')).sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => `${key}=${value}`).join(' · ') || '资源整体';
}
function oemChartKind(panel) {
    const kind = exports.oemMetricCharts[panel.metric_code] || (panel.native?.mode === 'LATEST' ? 'bar' : 'line');
    if (kind === 'donut') {
        const value = panel.series.length === 1 ? numeric(latest(panel.series[0])) : null;
        return panel.unit === 'percent' && value !== null && value >= 0 && value <= 100 ? 'donut' : 'bar';
    }
    return kind;
}
function oemChartCaption(panel) {
    const kind = oemChartKind(panel);
    if (kind === 'state')
        return 'OEM 最近采集的原始状态；不推断未知编码';
    if (kind === 'donut')
        return '末次采样 · 使用 / 未使用；均值与偏离见卡片摘要';
    if (kind === 'bar')
        return '各维度末次采样对比 · 不求和；短线标记为历史均值';
    return '真实历史采样 · 虚线为同维度历史均值';
}
function oemChartOption(panel, colors, baselineLabel = '') {
    const kind = oemChartKind(panel);
    const palette = ['accent', 'info', 'success', 'warning', 'danger', 'muted'].map(key => colors[key]).filter(Boolean);
    const escape = (value) => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
    const number = (value) => new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2, notation: Math.abs(value) >= 10000 ? 'compact' : 'standard' }).format(value);
    const rows = panel.series.map((series, entityIndex) => ({ series, entityIndex, point: latest(series), average: series.baseline && series.baseline.sample_count > 0 && Number.isFinite(series.baseline.average) ? series.baseline.average : null }));
    const label = baselineLabel || (rows.find(row => row.series.baseline?.period === '1m') ? '30 天' : '7 天');
    const unit = panel.unit === 'percent' ? '%' : panel.unit === 'n/a' || panel.unit === '1' ? '' : panel.unit;
    const datum = (row, point = row.point) => ({ value: numeric(point), raw: point?.raw_value ?? (point?.value === null || point?.value === undefined ? null : String(point.value)), at: point?.observed_at, quality: point?.quality, dimension: dimensionLabel(row.series), entityIndex: row.entityIndex });
    const tooltip = { confine: true, backgroundColor: colors.surfaceRaised || colors.surface, borderColor: colors.lineStrong || colors.line, textStyle: { color: colors.ink },
        formatter: (input) => (Array.isArray(input) ? input : [input]).map((entry) => {
            const data = entry.data;
            return data?.derived ? '未使用 = 100% − 使用率（计算值）' : `${escape(data?.at)}<br>${escape(data?.dimension)} · ${data?.baseline ? escape(label) + '均值' : escape(panel.title)}：${escape(data?.raw ?? '无有效采样')} ${escape(unit)}${data?.quality && data.quality !== 'GOOD' ? '<br>' + escape(data.quality) : ''}`;
        }).join('<br>') };
    const common = { animation: false, color: palette, tooltip };
    if (kind === 'state')
        return { ...common, series: [] };
    if (kind === 'donut') {
        const value = numeric(rows[0].point);
        return { ...common, legend: { bottom: 0, textStyle: { color: colors.muted, fontSize: 10 } }, series: [{ type: 'pie', radius: ['54%', '76%'], center: ['50%', '44%'], label: { show: false },
                    data: [{ ...datum(rows[0]), name: '使用', value }, { name: '未使用', value: 100 - value, derived: true, itemStyle: { color: colors.accentSoft || colors.line } }] }] };
    }
    const axisStyle = { axisLabel: { color: colors.muted, fontSize: 10, hideOverlap: true }, axisLine: { lineStyle: { color: colors.line } }, axisTick: { show: false } };
    const values = rows.flatMap(row => [...row.series.points.flatMap(point => numeric(point) === null ? [] : [numeric(point)]), ...(row.average === null ? [] : [row.average])]);
    const bounds = { min: Math.min(0, ...values), ...(panel.unit === 'percent' ? { max: Math.max(100, ...values) } : values.length ? { max: Math.max(1, ...values) * 1.1 } : {}) };
    if (kind === 'bar') {
        return { ...common, grid: { left: 12, right: 46, top: 20, bottom: 28, containLabel: true },
            xAxis: { ...axisStyle, type: 'value', ...bounds, axisLabel: { ...axisStyle.axisLabel, formatter: number }, splitNumber: 3, splitLine: { lineStyle: { color: colors.lineSubtle || colors.line, opacity: .62 } } },
            yAxis: { ...axisStyle, type: 'category', inverse: true, data: rows.map(row => dimensionLabel(row.series)), axisLabel: { color: colors.muted, fontSize: 10, width: 110, overflow: 'truncate' } },
            dataZoom: rows.length > 6 ? [{ type: 'slider', yAxisIndex: 0, filterMode: 'none', startValue: 0, endValue: 5, right: 2, width: 10 }, { type: 'inside', yAxisIndex: 0, filterMode: 'none', startValue: 0, endValue: 5 }] : [],
            series: [{ type: 'bar', barMaxWidth: 22, data: rows.map((row, index) => ({ ...datum(row), itemStyle: { color: palette[index % palette.length], borderRadius: [0, 3, 3, 0] } })) },
                { type: 'scatter', name: `${label}均值`, symbol: 'rect', symbolSize: [2, 18], itemStyle: { color: colors.ink }, data: rows.flatMap((row, index) => row.average === null ? [] : [{ value: [row.average, index], raw: String(row.average), dimension: dimensionLabel(row.series), baseline: true, entityIndex: row.entityIndex }]) }] };
    }
    return { ...common, tooltip: { ...tooltip, trigger: 'axis' }, grid: { left: 48, right: 18, top: rows.length > 1 ? 36 : 16, bottom: 30 },
        legend: { show: rows.length > 1, type: 'scroll', top: 0, textStyle: { color: colors.muted, fontSize: 10 } },
        xAxis: { ...axisStyle, type: 'time' }, yAxis: { ...axisStyle, type: 'value', ...bounds, axisLabel: { ...axisStyle.axisLabel, formatter: number }, splitLine: { lineStyle: { color: colors.lineSubtle || colors.line, opacity: .62 } } },
        series: rows.map((row, index) => ({ name: dimensionLabel(row.series), type: 'line', smooth: false, connectNulls: false, showSymbol: true, symbolSize: 4, lineStyle: { width: 2 }, areaStyle: kind === 'area' ? { opacity: .06 } : undefined,
            data: [...row.series.points].sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at)).map(point => ({ ...datum(row, point), value: [point.observed_at, numeric(point)] })),
            markLine: row.average === null ? undefined : { silent: true, symbol: 'none', label: { show: true, position: 'insideEndTop', fontSize: 10, color: colors.muted, formatter: `${label}均值 ${number(row.average)}` },
                data: [{ yAxis: row.average, name: `${label}均值`, lineStyle: { type: 'dashed', color: palette[index % palette.length] } }] },
        })) };
}

  globalThis.KBotAIOpsOEMPresentation = exports;
})();
