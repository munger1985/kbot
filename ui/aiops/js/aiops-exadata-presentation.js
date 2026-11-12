// 由 Ammolite exadataPresentation.ts 同步，合同测试校验两端展示语义一致。
(function () {
  const exports = {};
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.exadataScenes = void 0;
exports.exadataBaselineKey = exadataBaselineKey;
exports.exadataBaselineSummary = exadataBaselineSummary;
exports.exadataBaselineLines = exadataBaselineLines;
exports.exadataNumeric = exadataNumeric;
exports.exadataUnit = exadataUnit;
exports.exadataDeviceGroup = exadataDeviceGroup;
exports.exadataDeviceScene = exadataDeviceScene;
exports.exadataSceneMembers = exadataSceneMembers;
exports.exadataCardSection = exadataCardSection;
exports.exadataHasTrend = exadataHasTrend;
exports.exadataChartKind = exadataChartKind;
exports.exadataChartLabel = exadataChartLabel;
exports.exadataGapLabel = exadataGapLabel;
exports.exadataRaw = exadataRaw;
exports.exadataQuality = exadataQuality;
exports.exadataDimensions = exadataDimensions;
exports.buildExadataDevices = buildExadataDevices;
exports.exadataOverviewNotes = exadataOverviewNotes;
exports.buildExadataCards = buildExadataCards;
exports.exadataStatusKey = exadataStatusKey;
exports.exadataStatusItems = exadataStatusItems;
exports.exadataChartOption = exadataChartOption;
function exadataBaselineKey(card) { return JSON.stringify([card.componentId, card.metricCode]); }
const baselineNumberFormat = new Intl.NumberFormat('en-US', { useGrouping: false, minimumFractionDigits: 2, maximumFractionDigits: 2 });
function exadataBaselineNumber(value) {
    const numeric = exadataNumeric(value);
    return numeric === null ? '无有效数值' : baselineNumberFormat.format(numeric);
}
function exadataBaselineSummary(result, period) {
    if (!period)
        return '';
    if (!result)
        return '历史均值读取中…';
    if (typeof result === 'string' || result.baseline.unavailable_reason)
        return '均值暂不可用 · 查看证据';
    if (!result.comparisons.length)
        return '暂无可比维度';
    if (result.comparisons.length > 1)
        return result.comparisons.length + ' 个维度分别比较 · 查看证据';
    const item = result.comparisons[0];
    return `${period === '1w' ? '7 天' : '30 天'}均值 ${exadataBaselineNumber(item.average)} ${result.unit || ''} · ${item.deviation_percent === null ? '零均值，偏离不可定义' : '偏离 ' + exadataBaselineNumber(item.deviation_percent) + '%'}`;
}
function exadataBaselineLines(result, period) {
    if (!period)
        return [];
    if (!result)
        return ['历史均值读取中…'];
    if (typeof result === 'string')
        return [result];
    if (result.baseline.unavailable_reason)
        return [result.baseline.unavailable_reason];
    if (!result.comparisons.length)
        return ['无匹配的有效当前采样与历史维度，暂不比较。'];
    return result.comparisons.map(item => `${exadataDimensions(item.dimensions)} · ${period === '1w' ? '7 天' : '30 天'}留存均值 ${exadataBaselineNumber(item.average)} ${result.unit || ''}（${item.sample_count} 条）；比较采样 ${exadataBaselineNumber(item.current)}；差值 ${exadataBaselineNumber(item.difference)} ${result.unit === '%' ? '个百分点' : result.unit || ''}；${item.deviation_percent === null ? '均值为零，相对偏离不可定义' : `偏离 ${exadataBaselineNumber(item.deviation_percent)}%`}；比较采样时间 ${item.observed_at}`);
}
const labels = {
    'CPUUsageSummary/OverAllUsagePct': 'CPU 使用率',
    'Load/memUsedPct': '内存使用率',
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
    'Cell_Metric/network_sent': '网络发送速率',
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
exports.exadataScenes = [['overview', '整机概览'], ['dbserver', 'DB Server'], ['cell', 'Cell'], ['disk', '磁盘 IO'], ['flash', 'Flash / Smart IO'], ['iorm', 'IORM'], ['network', '互联设备'], ['capacity', '容量']];
function exadataDeviceGroup(type) {
    if (['oracle_dbmachine', 'oracle_exadata_grid', 'oracle_si_rack'].includes(type))
        return 'assets';
    return ['host', 'oracle_exadata', 'oracle_si_server_map'].includes(type) ? 'nodes' : 'equipment';
}
function exadataDeviceScene(type) {
    return type === 'host' || type === 'oracle_si_server_map' ? 'dbserver' : type === 'oracle_exadata' ? 'cell' : ['oracle_si_netswitch', 'oracle_si_network_port_switch', 'oracle_si_network_port_server'].includes(type) ? 'network' : 'overview';
}
function exadataSceneMembers(members, scene) {
    return members.filter(member => exadataDeviceGroup(member.target_type) !== 'assets' && (scene === 'overview'
        || scene === 'dbserver' && ['host', 'oracle_si_server_map'].includes(member.target_type)
        || ['cell', 'disk', 'flash', 'iorm'].includes(scene) && member.target_type === 'oracle_exadata'
        || scene === 'network' && ['oracle_exadata', 'oracle_si_netswitch', 'oracle_si_network_port_server', 'oracle_si_network_port_switch'].includes(member.target_type)
        || scene === 'capacity'));
}
// 按审核目录中的准确选择器分组，不从名称模糊推断健康或单位。
function exadataCardSection(card) {
    if (card.status && ['AVAILABILITY', 'EVENT'].includes(card.status.category))
        return 'status';
    if (['Aggregated_Exadata_Diskgroup_Capacity_Metric/count', 'Aggregated_Exadata_Diskgroup_Capacity_Metric/size'].includes(card.selector))
        return 'performance';
    if (card.status || !card.numeric || ['Cell_Metric/cl_iorm_boost', 'Cell_Metric/iorm_mode', 'Cell_Metric/temperature_lower', 'Cell_Metric/temperature_upper'].includes(card.selector))
        return 'configuration';
    return 'performance';
}
function exadataHasTrend(card) {
    return card.numeric && exadataChartable(card) && card.series.some(series => new Set(series.samples.filter(sample => sample.quality !== 'EMPTY' && exadataNumeric(sample.value) !== null).map(sample => sample.observed_at)).size > 1);
}
// 原厂将这两个容量列声明为字符串；只允许已审核的选择器作数值近似，不改原厂类型。
function exadataChartable(card) {
    return !card.status && exadataCardSection(card) === 'performance' && (card.numeric || ['Aggregated_Exadata_Diskgroup_Capacity_Metric/count', 'Aggregated_Exadata_Diskgroup_Capacity_Metric/size'].includes(card.selector));
}
function exadataChartKind(card, mode) {
    if (!exadataChartable(card))
        return null;
    if (mode === 'HISTORY')
        return exadataHasTrend(card) ? 'line' : null;
    if (mode !== 'LATEST_SNAPSHOT' || !card.series.some(series => series.latest.quality !== 'EMPTY' && exadataNumeric(series.latest.value) !== null))
        return null;
    // 仅占整体的使用率允许补出未使用部分；IO负载、效率、错误率等不套用占用语义。
    const occupancy = ['CPUUsageSummary/OverAllUsagePct', 'Load/memUsedPct', 'Cell_Metric/cpu_utilization', 'Cell_Metric/memory_utilization', 'Filesystem_Utilization_Metric/utilization', 'Aggregated_Exadata_Capacity_Metric/Usage'];
    return card.percent !== null && occupancy.includes(card.selector) ? 'donut' : 'bar';
}
function exadataChartLabel(kind) {
    return kind === 'line' ? '数值历史 · 各维度独立趋势' : kind === 'donut' ? '最新快照 · 使用 / 未使用' : '最新快照 · 按维度对比';
}
function exadataGapLabel(code) {
    if (['SOURCE_QUERY_UNSUPPORTED', 'SOURCE_METRIC_HISTORY_UNSUPPORTED'].includes(code))
        return '不支持读取';
    if (['NO_DATA', 'EMPTY'].includes(code))
        return '未采样';
    if (code === 'STALE')
        return '采样过期';
    if (['COLLECTION_TIMEOUT', 'QUERY_PAGE_PARTIAL'].includes(code))
        return '未完成查询';
    if (code.startsWith('SOURCE_'))
        return code === 'SOURCE_TRUNCATED' ? '采样截断' : '查询失败';
    return '采集说明';
}
function exadataRaw(value) { return value === null || value === '' ? '无数据' : value; }
function exadataQuality(value) {
    return { OBSERVED: '已采样', EMPTY: '来源空值', STALE: '旧快照', TRUNCATED: '采样截断' }[value] || value;
}
function exadataDimensions(dimensions) {
    return Object.entries(dimensions).sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => `${key}=${value}`).join(' · ') || '无维度';
}
// 整机清单不受指标分页隐藏；只呈现此页实际证据，不把未查询设备涂绿。
function buildExadataDevices(view) {
    const cards = buildExadataCards(view);
    const types = { host: 'DB Server', oracle_exadata: 'Cell', oracle_si_netswitch: '交换机', oracle_si_server_map: '服务器硬件 / ILOM', oracle_si_pdu: 'PDU', oracle_si_rack: '机架', oracle_dbmachine: '整机', oracle_exadata_grid: '存储网格', oracle_si_hw_component_server: '服务器硬件组件', oracle_si_hw_component_switch: '交换机硬件组件', oracle_si_network_port_server: '服务器端口', oracle_si_network_port_switch: '交换机端口' };
    return (view.device_coverage || []).map(coverage => {
        const member = view.members.find(item => item.target_id === coverage.component_id);
        const componentCards = cards.filter(card => card.componentId === coverage.component_id);
        const states = componentCards.filter(card => card.status).flatMap(card => exadataStatusItems(card).map(item => ({ key: `${card.key}:${item.key}`, title: `${card.title}${item.dimension ? ` · ${item.dimension}` : ''}`, label: item.label, tone: item.tone })));
        const resources = componentCards.filter(card => !card.status).map(card => ({ key: card.key, title: card.title, value: `${card.value}${card.valueUnit ? ` ${card.valueUnit}` : ''} · ${card.quality}` }));
        const remaining = coverage.metric_codes.length - coverage.queried_metric_codes.length;
        return { member, typeLabel: types[member.target_type] || member.target_type, states, resources,
            note: !coverage.metric_codes.length ? coverage.note : remaining ? `另 ${remaining} 项指标本页未查询，可查看设备指标` : '' };
    }).sort((a, b) => {
        const order = { host: 0, oracle_si_server_map: 1, oracle_exadata: 2, oracle_si_netswitch: 3, oracle_si_pdu: 4 };
        return (order[a.member.target_type] ?? 5) - (order[b.member.target_type] ?? 5) || a.member.display_name.localeCompare(b.member.display_name);
    });
}
function exadataOverviewNotes(view) {
    return [['host', 'DB Server'], ['oracle_si_netswitch', '交换机'], ['oracle_si_pdu', 'PDU']]
        .filter(([type]) => !view.members.some(member => member.target_type === type))
        .map(([, label]) => `当前已确认成员不含 ${label}；无法展示其状态${label === 'DB Server' ? '及 CPU/内存' : ''}，请检查 OEM 设备发现与整机成员确认。`);
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
                title: labels[metricSelector] || definition?.status?.display_name || definition?.display_name || metricSelector,
                status: definition?.status || null, currentStatus: definition?.semantics === 'OEM_TARGET_AVAILABILITY',
                unit, numeric: nativeType === 'NUMBER' && !definition?.status, samples: [], series: [], gaps: [], quality: '', tone: '',
                baselineSupported: !!definition?.baseline_supported && !definition?.status,
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
        const providerError = !hasValue && card.gaps.some(gap => gap.code === 'SOURCE_PROVIDER_ERROR');
        const unsupported = !hasValue && card.gaps.some(gap => gap.code === 'SOURCE_QUERY_UNSUPPORTED');
        const incomplete = !hasValue && card.gaps.some(gap => gap.code === 'COLLECTION_TIMEOUT');
        const readError = !hasValue && card.gaps.some(gap => (gap.code.startsWith('SOURCE_') && !['SOURCE_QUERY_UNSUPPORTED', 'SOURCE_METRIC_HISTORY_UNSUPPORTED', 'SOURCE_TRUNCATED'].includes(gap.code)));
        card.quality = card.samples.some(item => item.quality === 'TRUNCATED') ? '采样截断'
            : latest.some(item => item.quality === 'STALE') ? '旧快照'
                : providerError ? 'OEM 服务错误'
                    : readError ? '读取错误'
                        : unsupported ? '不支持读取'
                            : incomplete ? '查询超时'
                                : !hasValue ? card.gaps.some(gap => gap.code === 'SOURCE_METRIC_HISTORY_UNSUPPORTED') ? '历史不支持' : '无有效数据'
                                    : latest.some(item => item.quality === 'EMPTY' || item.value === null || item.value === '' || (card.numeric && exadataNumeric(item.value) === null)) ? '部分缺失'
                                        : card.gaps.length ? '部分采样' : '已采样';
        card.tone = ['旧快照', '采样截断', '部分缺失', '部分采样', 'OEM 服务错误', '读取错误'].includes(card.quality) ? 'warning' : hasValue ? 'observed' : 'empty';
        card.value = readError ? '读取失败' : unsupported ? '不支持读取' : incomplete ? '未完成查询' : latest.length > 1 ? `${latest.length} 个维度` : latest[0]?.quality === 'EMPTY' ? '无数据' : exadataRaw(latest[0]?.value ?? null);
        card.valueUnit = latest.length === 1 && card.unit && card.unit !== 'n/a' ? card.unit : '';
        card.observedAt = [...latest].sort((a, b) => Date.parse(b.observed_at) - Date.parse(a.observed_at))[0]?.observed_at || null;
        const value = latest.length === 1 && latest[0].quality !== 'EMPTY' ? exadataNumeric(latest[0].value) : null;
        card.percent = view.query_mode === 'LATEST_SNAPSHOT' && card.numeric && card.unit === '%' && value !== null && value >= 0 && value <= 100 ? value : null;
    }
    return [...groups.values()];
}
// 每个维度独立显示；旧值、截断、缺值及读取失败都不能显示绿色可用。
function exadataStatusKey(value, nativeType) {
    const raw = value.trim();
    if (nativeType !== 'NUMBER')
        return raw.toLowerCase();
    // 使用十进制系数与指数，不用 Number 将相近但不等值的状态舍入成已审核编码。
    const parts = /^([+-]?)([0-9]*)(?:\.([0-9]*))?(?:e([+-]?[0-9]{1,6}))?$/i.exec(raw);
    if (!parts || !(parts[2] || parts[3]))
        return null;
    let digits = (parts[2] + (parts[3] || '')).replace(/^0+/, '');
    if (!digits)
        return '0';
    let exponent = BigInt(parts[4] || '0') - BigInt((parts[3] || '').length);
    while (digits.endsWith('0')) {
        digits = digits.slice(0, -1);
        exponent += 1n;
    }
    return `${parts[1] === '-' ? '-' : ''}${digits}e${exponent}`;
}
function exadataStatusItems(card) {
    if (!card.status)
        return [];
    if (!card.series.length)
        return [{ key: card.key, dimension: '', label: ['读取失败', '不支持读取', '未完成查询'].includes(card.value) ? card.value : card.quality === '历史不支持' ? '历史不支持' : '无数据', tone: 'unknown', quality: card.quality, observedAt: null }];
    return card.series.map(series => {
        const sample = series.latest;
        const raw = sample.value?.trim() ?? '';
        const key = exadataStatusKey(raw, sample.native_value_type);
        const entry = key === null ? undefined : Object.entries(card.status.values).find(([value]) => exadataStatusKey(value, sample.native_value_type) === key)?.[1];
        const missing = sample.quality === 'EMPTY' || !raw;
        const valid = sample.quality === 'OBSERVED' && !card.gaps.length && !missing;
        return { key: series.key, dimension: series.label === '无维度' ? '' : series.label,
            label: missing ? '无数据' : valid ? entry?.label || '未知' : '未知',
            tone: valid ? entry?.tone || 'unknown' : 'unknown',
            quality: missing ? '无有效数据' : !entry ? '未识别状态' : sample.quality !== 'OBSERVED' ? exadataQuality(sample.quality) : card.gaps.length ? '部分采样' : card.currentStatus ? '状态已读取' : '已采样',
            observedAt: sample.observed_at };
    });
}
// 图表仅作数值近似绘制，卡片及原始表格保留精确文本；空值不补零、不连线。
function exadataChartOption(card, colors, kind = 'line') {
    const escape = (value) => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
    const palette = [colors.accent, colors.info, colors.success, colors.warning, colors.danger, colors.muted].filter(Boolean);
    const tooltip = { confine: true, backgroundColor: colors.surfaceRaised || colors.surface, borderColor: colors.lineStrong || colors.line, textStyle: { color: colors.ink } };
    const chartNumber = (value) => value === null ? '—' : Math.abs(value) >= 1e12 ? value.toExponential(2) : Math.abs(value) >= 1e4 ? new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(value) : baselineNumberFormat.format(value);
    const rawTooltip = (entry) => `${escape(entry.data?.at)}<br>${escape(entry.data?.dimension || entry.name)}：${escape(exadataRaw(entry.data?.raw ?? null))} ${escape(exadataUnit(card.unit))}<br>${escape(entry.data?.quality ? exadataQuality(entry.data.quality) : '')}`;
    const latest = card.series.map(series => ({ name: series.label === '无维度' ? '当前值' : series.label, value: series.latest.quality === 'EMPTY' ? null : exadataNumeric(series.latest.value), raw: series.latest.value, at: series.latest.observed_at, dimension: series.label, quality: series.latest.quality }));
    if (kind === 'donut' && card.percent !== null && exadataChartKind(card, 'LATEST_SNAPSHOT') === 'donut') {
        const point = latest[0];
        return { animation: false, color: [colors.accent, colors.accentSoft || colors.line],
            tooltip: { ...tooltip, trigger: 'item', formatter: (entry) => entry.data?.derived ? '未使用：100% − OEM 使用率（计算值）' : rawTooltip({ name: card.title, data: point }) },
            legend: { bottom: 0, textStyle: { color: colors.muted, fontSize: 10 } },
            series: [{ name: card.title, type: 'pie', radius: ['54%', '76%'], center: ['50%', '44%'], avoidLabelOverlap: true, label: { show: false },
                    data: [{ ...point, name: '使用', value: card.percent }, { name: '未使用', value: 100 - card.percent, derived: true, itemStyle: { color: colors.accentSoft || colors.line } }] }], };
    }
    if (kind === 'bar') {
        const axisColor = colors.faint || colors.muted;
        return { animation: false, color: palette,
            grid: { left: 12, right: latest.length > 6 ? 46 : 38, top: 28, bottom: 24, containLabel: true },
            tooltip: { ...tooltip, trigger: 'item', formatter: rawTooltip },
            xAxis: { type: 'value', name: exadataUnit(card.unit), nameLocation: 'middle', nameGap: 18, nameTextStyle: { color: axisColor, fontSize: 10 }, splitNumber: 3, axisLabel: { color: axisColor, fontSize: 10, hideOverlap: true, formatter: chartNumber }, splitLine: { lineStyle: { color: colors.lineSubtle || colors.line, opacity: .62 } } },
            yAxis: { type: 'category', inverse: true, data: latest.map(item => item.name), axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: axisColor, fontSize: 10, width: 100, overflow: 'truncate', formatter: (label) => label.replace(/(^| · )[^=·]+=/g, '$1') } },
            dataZoom: latest.length > 6 ? [{ type: 'slider', yAxisIndex: 0, filterMode: 'none', startValue: 0, endValue: 5, width: 12, right: 6, showDetail: false }, { type: 'inside', yAxisIndex: 0, filterMode: 'none', startValue: 0, endValue: 5 }] : [],
            series: exadataChartable(card) ? [{ type: 'bar', name: card.title, barMaxWidth: 22, label: { show: true, position: 'right', color: colors.ink, fontSize: 10, formatter: (entry) => chartNumber(entry.value) },
                    data: latest.map((item, index) => ({ ...item, itemStyle: { color: palette[index % palette.length], borderRadius: [0, 3, 3, 0] } })) }] : [],
        };
    }
    return {
        animation: false,
        color: palette,
        grid: { left: 48, right: 18, top: card.series.length > 1 ? 36 : 16, bottom: 30 },
        legend: { show: card.series.length > 1, type: 'scroll', top: 0, textStyle: { color: colors.muted, fontSize: 10 } },
        tooltip: { ...tooltip, trigger: 'axis',
            formatter: (params) => {
                const entries = (Array.isArray(params) ? params : [params]);
                return entries.map(entry => `${escape(entry.data?.at)}<br>${escape(entry.seriesName)}：${escape(exadataRaw(entry.data?.raw ?? null))} ${escape(exadataUnit(card.unit))}`).join('<br>');
            } },
        xAxis: { type: 'time', axisLabel: { color: colors.muted, fontSize: 10, hideOverlap: true }, axisLine: { lineStyle: { color: colors.line } }, splitLine: { show: false } },
        yAxis: { type: 'value', scale: true, axisLabel: { color: colors.muted, fontSize: 10, formatter: chartNumber }, splitLine: { lineStyle: { color: colors.lineSubtle || colors.line, opacity: .62 } } },
        series: exadataChartable(card) ? card.series.map(series => ({
            name: series.label, type: 'line', connectNulls: false, showSymbol: true, symbolSize: 4,
            lineStyle: { width: 2 }, areaStyle: card.series.length === 1 ? { opacity: 0.06 } : undefined,
            data: series.samples.map(sample => ({ value: [sample.observed_at, sample.quality === 'EMPTY' ? null : exadataNumeric(sample.value)], raw: sample.value, at: sample.observed_at })),
        })) : [],
    };
}

  globalThis.KBotAIOpsExadataPresentation = exports;
})();
