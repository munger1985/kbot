// 由 Ammolite exadataPresentation.ts 同步，合同测试校验两端展示语义一致。
(function () {
  const exports = {};
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.exadataNumeric = exadataNumeric;
exports.exadataUnit = exadataUnit;
exports.exadataRaw = exadataRaw;
exports.exadataQuality = exadataQuality;
exports.exadataDimensions = exadataDimensions;
exports.buildExadataCards = buildExadataCards;
exports.exadataChartOption = exadataChartOption;
const labels = {
    'CellSrv_Status_Metric/Status': 'Cell 服务状态',
    'CellSrv_Status_Metric/msStatus': '管理服务状态',
    'CellSrv_Status_Metric/rsStatus': '重启服务状态',
    'Cell_Metric/cell_name': 'Cell 名称',
    'Cell_Metric/cpu_utilization': 'CPU 使用率',
    'Cell_Metric/memory_utilization': '内存使用率',
    'Cell_Metric/cl_runq': '运行队列长度',
    'Cell_Metric/cl_iorm_boost': 'IORM Boost',
    'Cell_Metric/iorm_mode': '磁盘 I/O 目标',
    'Cell_Metric/led_status': '指示灯状态',
    'Cell_Metric/network_received': '网络接收速率',
    'Cell_Metric/network_transmitted': '网络发送速率',
    'Cell_Metric/temperature': 'Cell 温度',
    'Cell_Metric/temperature_lower': '温度下限原始值',
    'Cell_Metric/temperature_upper': '温度上限原始值',
};
function exadataNumeric(value) {
    if (value === null || !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim()))
        return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}
function exadataUnit(unit) { return unit || 'OEM 未声明单位'; }
function exadataRaw(value) { return value === null || value === '' ? '无数据' : value; }
function exadataQuality(value) {
    return { OBSERVED: '已采样', EMPTY: '来源空值', STALE: '旧快照', TRUNCATED: '采样截断' }[value] || value;
}
function exadataDimensions(dimensions) {
    return Object.entries(dimensions).sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => `${key}=${value}`).join(' · ') || '无维度';
}
function buildExadataCards(view) {
    const groups = new Map();
    const definitions = new Map(view.definitions.map(item => [item.code, item]));
    function ensure(componentId, metricCode, unit, nativeType, selector) {
        const key = JSON.stringify([componentId, metricCode, unit, nativeType]);
        let card = groups.get(key);
        if (!card) {
            const definition = definitions.get(metricCode);
            const metricSelector = selector || definition?.selector || metricCode;
            card = { key, componentId, metricCode, selector: metricSelector,
                componentName: view.members.find(member => member.target_id === componentId)?.display_name || componentId,
                title: labels[metricSelector] || definition?.display_name || metricSelector,
                unit, numeric: nativeType === 'NUMBER', samples: [], series: [], gaps: [], quality: '', tone: '',
                value: '无数据', valueUnit: '', observedAt: null, percent: null };
            groups.set(key, card);
        }
        return card;
    }
    for (const sample of view.samples)
        ensure(sample.component_id, sample.metric_code, sample.unit, sample.native_value_type, sample.selector).samples.push(sample);
    // 仅为服务端明确返回的失败查询建空卡，不按目录或分页猜测未查询的组件。
    for (const gap of view.gaps) {
        if (!gap.component_id || !gap.metric_code)
            continue;
        const matching = [...groups.values()].filter(card => card.componentId === gap.component_id && card.metricCode === gap.metric_code);
        if (!matching.length)
            matching.push(ensure(gap.component_id, gap.metric_code, null, definitions.get(gap.metric_code)?.native_value_type || ''));
        matching.forEach(card => card.gaps.push(gap));
    }
    for (const card of groups.values()) {
        const series = new Map();
        for (const sample of card.samples) {
            const key = JSON.stringify(Object.entries(sample.dimensions).sort(([a], [b]) => a.localeCompare(b)));
            const points = series.get(key) || [];
            points.push(sample);
            series.set(key, points);
        }
        card.series = [...series].map(([key, points]) => {
            const samples = [...points].sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at));
            return { key, label: exadataDimensions(samples[0].dimensions), samples, latest: samples.at(-1) };
        });
        const latest = card.series.map(item => item.latest);
        const hasValue = latest.some(item => item.quality !== 'EMPTY' && (card.numeric ? exadataNumeric(item.value) !== null : item.value !== null && item.value !== ''));
        card.quality = card.samples.some(item => item.quality === 'TRUNCATED') ? '采样截断'
            : latest.some(item => item.quality === 'STALE') ? '旧快照'
                : !hasValue ? card.gaps.some(gap => gap.code === 'SOURCE_METRIC_HISTORY_UNSUPPORTED') ? '历史不支持' : '无有效数据'
                    : latest.some(item => item.quality === 'EMPTY' || item.value === null || item.value === '' || (card.numeric && exadataNumeric(item.value) === null)) ? '部分缺失'
                        : card.gaps.length ? '部分采样' : card.numeric ? '已采样' : '原始状态';
        card.tone = ['旧快照', '采样截断', '部分缺失', '部分采样'].includes(card.quality) ? 'warning' : hasValue ? 'observed' : 'empty';
        card.value = latest.length > 1 ? `${latest.length} 个维度` : latest[0]?.quality === 'EMPTY' ? '无数据' : exadataRaw(latest[0]?.value ?? null);
        card.valueUnit = latest.length === 1 && card.unit && card.unit !== 'n/a' ? card.unit : '';
        card.observedAt = [...latest].sort((a, b) => Date.parse(b.observed_at) - Date.parse(a.observed_at))[0]?.observed_at || null;
        const value = latest.length === 1 && latest[0].quality !== 'EMPTY' ? exadataNumeric(latest[0].value) : null;
        card.percent = view.query_mode === 'LATEST_SNAPSHOT' && card.numeric && card.unit === '%' && value !== null && value >= 0 && value <= 100 ? value : null;
    }
    return [...groups.values()];
}
// 图表仅作数值近似绘制，卡片及原始表格保留精确文本；空值不补零、不连线。
function exadataChartOption(card, colors) {
    const escape = (value) => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
    return {
        animation: false,
        color: [colors.accent, colors.info, colors.muted],
        grid: { left: 48, right: 18, top: card.series.length > 1 ? 36 : 16, bottom: 30 },
        legend: { show: card.series.length > 1, type: 'scroll', top: 0, textStyle: { color: colors.muted, fontSize: 10 } },
        tooltip: { trigger: 'axis', confine: true, backgroundColor: colors.surface, borderColor: colors.line, textStyle: { color: colors.ink },
            formatter: (params) => {
                const entries = (Array.isArray(params) ? params : [params]);
                return entries.map(entry => `${escape(entry.data?.at)}<br>${escape(entry.seriesName)}：${escape(exadataRaw(entry.data?.raw ?? null))} ${escape(exadataUnit(card.unit))}`).join('<br>');
            } },
        xAxis: { type: 'time', axisLabel: { color: colors.muted, fontSize: 10, hideOverlap: true }, axisLine: { lineStyle: { color: colors.line } }, splitLine: { show: false } },
        yAxis: { type: 'value', scale: true, axisLabel: { color: colors.muted, fontSize: 10 }, splitLine: { lineStyle: { color: colors.line, type: 'dashed' } } },
        series: card.numeric ? card.series.map(series => ({
            name: series.label, type: 'line', connectNulls: false, showSymbol: true, symbolSize: 4,
            lineStyle: { width: 2 }, areaStyle: card.series.length === 1 ? { opacity: 0.06 } : undefined,
            data: series.samples.map(sample => ({ value: [sample.observed_at, sample.quality === 'EMPTY' ? null : exadataNumeric(sample.value)], raw: sample.value, at: sample.observed_at })),
        })) : [],
    };
}

  globalThis.KBotAIOpsExadataPresentation = exports;
})();
