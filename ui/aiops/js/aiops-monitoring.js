(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/monitoring";
  const shell = globalThis.KBotAIOpsShell;
  const state = {
    sources: [], instances: [], profiles: [], view: null,
    sourceId: "", compareSourceId: "", profileId: "", window: "1h", selected: [], search: "",
    debounceTimer: null, refreshTimer: null, controller: null, contextController: null, charts: new Map(),
  };
  const readinessLabels = {
    READY: "就绪", DISABLED: "已停用", DISCONNECTED: "连接失败",
    CAPABILITY_MISSING: "能力缺失", NO_MAPPED_INSTANCE: "未映射",
    PARTIAL: "部分就绪", UNAVAILABLE: "不可用", AVAILABLE: "可用",
  };
  const qualityLabels = { GOOD: "正常", PARTIAL: "部分成功", NO_DATA: "无有效采样" };
  const capacityMetricCodes = [
    "db.storage.used_bytes", "db.storage.free_bytes", "db.storage.max_bytes",
  ];
  const esc = shell.escape;
  const query = new URLSearchParams(location.search);
  const sourceById = () => state.sources.find((item) => item.source_id === state.sourceId) || null;
  const profileById = () => state.profiles.find((item) => item.profile_id === state.profileId) || null;
  const compareSources = () => state.sources.filter((item) => item.source_id !== state.sourceId
    && item.source_type !== sourceById()?.source_type && item.monitoring_readiness === "READY");
  const seriesLabel = (series) => `${series.instance_display_name} · ${series.source_display_name}`;

  function label(value) { return readinessLabels[value] || value || "不可用"; }
  function tone(value) {
    if (["READY", "AVAILABLE", "GOOD"].includes(value)) return "good";
    if (["DISCONNECTED", "UNAVAILABLE", "NO_DATA"].includes(value)) return "bad";
    return "warn";
  }
  function badge(value, text) { return `<span class="ops-badge ${tone(value)}">${esc(text || label(value))}</span>`; }
  function unitLabel(unit = "") {
    return ({
      state: "状态", percent: "%", count: "个", milliseconds: "毫秒",
      seconds: "秒", transactions_per_second: "次/秒",
      queries_per_second: "次/秒", waits_per_second: "次/秒",
      errors_per_second: "次/秒", bytes_per_second: "字节/秒",
      bytes: "字节", "1": "",
    })[unit] ?? unit;
  }
  function formatValue(value, unit = "") {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "boolean" || unit === "state") return Number(value) > 0 ? "可用" : "不可用";
    const number = Number(value);
    if (Number.isFinite(number)) {
      const text = number.toLocaleString("zh-CN", { maximumFractionDigits: Math.abs(number) >= 100 ? 1 : 2 });
      if (unit === "percent") return `${text}%`;
      if (unit === "bytes") return formatBytes(number);
      if (unit === "bytes_per_second") return `${formatBytes(number)}/秒`;
      const translatedUnit = unitLabel(unit);
      return `${text}${translatedUnit ? ` ${translatedUnit}` : ""}`;
    }
    return String(value);
  }
  function numericValue(value) {
    if (typeof value === "boolean") return value ? 1 : 0;
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }
  function formatBytes(value) {
    const number = numericValue(value);
    if (number === null) return "—";
    const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    const index = number > 0 ? Math.min(Math.floor(Math.log(number) / Math.log(1024)), units.length - 1) : 0;
    return `${(number / (1024 ** index)).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} ${units[index]}`;
  }
  function latestPoint(series) {
    return [...(series?.points || [])].reverse().find((point) => point.value !== null && point.value !== undefined) || null;
  }
  function gapIdentity(gap) {
    return JSON.stringify([
      gap.scope || "", gap.source_id || "", gap.instance_id || "", gap.metric_code || "",
      gap.code || "", gap.detail || "", Boolean(gap.retryable),
    ]);
  }
  function allGaps() {
    const selectedInstances = state.instances.filter((item) => state.selected.includes(item.instance_id));
    const seen = new Set();
    return [
      ...(sourceById()?.capability_gaps || []),
      ...selectedInstances.flatMap((item) => item.capability_gaps || []),
      ...(state.view?.gaps || []),
    ].filter((gap) => {
      const identity = gapIdentity(gap);
      if (seen.has(identity)) return false;
      seen.add(identity);
      return true;
    });
  }
  function diagnosticGapText(item) {
    return (item?.diagnostic_gaps || []).map((gap) => `${gap.code}：${gap.detail}`).join("；");
  }
  function latestSampleAt() {
    return (state.view?.panels || []).flatMap((panel) => panel.series || [])
      .map((series) => latestPoint(series)?.observed_at).filter(Boolean).sort().at(-1) || null;
  }
  function latestSampleForInstance(instanceId) {
    return (state.view?.panels || []).flatMap((panel) => panel.series || [])
      .filter((series) => series.instance_id === instanceId)
      .map((series) => latestPoint(series)?.observed_at).filter(Boolean).sort().at(-1) || null;
  }
  function requestBody() {
    return JSON.stringify({ profile_id: state.profileId, instance_ids: state.selected, window: state.window });
  }
  function canQuery() {
    return sourceById()?.monitoring_readiness === "READY" && state.profileId && state.selected.length > 0;
  }
  function cancelViewRequest() {
    state.controller?.abort();
    state.controller = null;
    clearTimeout(state.debounceTimer);
    state.debounceTimer = null;
  }
  function disposeCharts() {
    state.charts.forEach((chart) => chart.dispose());
    state.charts.clear();
  }
  function setLoading(loading, text = "正在查询实时指标…") {
    const node = document.getElementById("monitoring-loading");
    node.hidden = !loading;
    node.querySelector(".ops-empty").textContent = text;
  }
  function updateDiagnosisLink() {
    const link = document.getElementById("monitoring-diagnosis-link");
    const enabled = state.selected.length === 1;
    const params = new URLSearchParams({
      target_id: enabled ? state.selected[0] : "",
      source_id: state.sourceId,
      window: state.window,
      profile_id: state.profileId,
      metric_codes: (profileById()?.metric_codes || []).join(","),
    });
    link.href = `./chat.html?${params}`;
    link.classList.toggle("is-disabled", !enabled);
    link.setAttribute("aria-disabled", String(!enabled));
  }
  function renderFreshness() {
    const view = state.view;
    const status = view ? (view.partial ? "部分成功" : "正常") : "未查询";
    document.getElementById("monitoring-freshness").textContent = [
      `生成于 ${shell.fmt(view?.generated_at)}`,
      `最新采样 ${shell.fmt(latestSampleAt())}`,
      `来源 ${sourceById()?.display_name || "—"}`,
      `已选 ${state.selected.length} / 12`, status,
    ].join(" · ");
    document.getElementById("refresh-monitoring").disabled = !canQuery();
    updateDiagnosisLink();
  }
  function gateContent(title, detail, gaps = []) {
    return `<div class="ops-monitoring-gate-copy"><span>接入状态</span><h2>${esc(title)}</h2><p>${esc(detail)}</p></div><a class="ops-button" href="./diagnostic-sources.html">前往监控接入</a>${gaps.length ? `<ul>${gaps.map((gap) => `<li><strong>${esc(gap.code)}</strong><span>${esc(gap.detail)}</span></li>`).join("")}</ul>` : ""}`;
  }
  function renderGate() {
    const gate = document.getElementById("monitoring-gate");
    const source = sourceById();
    if (!state.sources.length) {
      gate.hidden = false;
      gate.innerHTML = gateContent("尚未配置监控源", "请先接入并启用 Prometheus 或 Zabbix，再返回查看实时指标。");
      return true;
    }
    const mapping = {
      DISABLED: ["监控源未启用", "该来源已配置但当前处于停用状态。"],
      DISCONNECTED: ["监控源连接失败", "连接验证未通过，实时指标查询不会被发起。"],
      CAPABILITY_MISSING: ["监控能力不完整", "该来源缺少第一阶段所需的时序查询能力。"],
      NO_MAPPED_INSTANCE: ["尚未关联数据库实例", "请在监控接入中将发现的实例映射到运维目标。"],
    };
    const entry = mapping[source?.monitoring_readiness];
    if (entry) {
      gate.hidden = false;
      gate.innerHTML = gateContent(entry[0], entry[1], source.capability_gaps || []);
      return true;
    }
    if (!state.instances.length) {
      gate.hidden = false;
      gate.innerHTML = gateContent("尚未关联数据库实例", "当前来源下没有可授权查询的 ACTIVE 实例映射。");
      return true;
    }
    gate.hidden = true;
    return false;
  }
  function renderContext() {
    const context = document.getElementById("monitoring-context");
    context.hidden = !state.sources.length;
    document.getElementById("monitoring-source").innerHTML = state.sources.map((source) => `<option value="${esc(source.source_id)}" ${source.source_id === state.sourceId ? "selected" : ""}>${esc(source.display_name)} · ${esc(label(source.monitoring_readiness))}</option>`).join("");
    document.getElementById("monitoring-profile").innerHTML = state.profiles.map((profile) => `<option value="${esc(profile.profile_id)}" ${profile.profile_id === state.profileId ? "selected" : ""}>${esc(profile.display_name)}</option>`).join("");
    document.getElementById("monitoring-profile").disabled = !state.profiles.length;
    document.getElementById("monitoring-window").disabled = sourceById()?.monitoring_readiness !== "READY";
    document.getElementById("monitoring-window").value = state.window;
    const compareField = document.getElementById("monitoring-compare-field");
    compareField.hidden = state.selected.length !== 1;
    document.getElementById("monitoring-compare-source").innerHTML = '<option value="">不对比</option>' + compareSources().map((source) => `<option value="${esc(source.source_id)}" ${source.source_id === state.compareSourceId ? "selected" : ""}>${esc(source.display_name)} · ${esc(source.source_type)}</option>`).join("");
    const diagnosticDetail = diagnosticGapText(sourceById());
    document.getElementById("monitoring-readiness").innerHTML = `${badge(sourceById()?.monitoring_readiness, `监控 ${label(sourceById()?.monitoring_readiness)}`)}<span title="${esc(diagnosticDetail)}">${badge(sourceById()?.diagnostic_readiness, `自动诊断 ${label(sourceById()?.diagnostic_readiness)}`)}</span>`;
  }
  function filteredInstances() {
    const value = state.search.trim().toLowerCase();
    return state.instances.filter((item) => !value || `${item.display_name} ${item.db_type}`.toLowerCase().includes(value));
  }
  function renderInstances() {
    const picker = document.getElementById("monitoring-instance-picker");
    picker.hidden = renderGate();
    if (picker.hidden) return;
    document.getElementById("monitoring-selection-count").textContent = `${state.selected.length} / 12`;
    const rows = filteredInstances().map((instance) => {
      const selected = state.selected.includes(instance.instance_id);
      const result = state.view?.instances.find((item) => item.instance_id === instance.instance_id);
      const quality = result ? instanceQuality(result) : selected ? "待查询" : "未选择";
      const qualityState = quality === "正常" ? "GOOD" : quality === "部分成功" ? "PARTIAL" : quality === "不可用" ? "UNAVAILABLE" : quality === "无有效采样" ? "NO_DATA" : "PARTIAL";
      const current = state.selected.length === 1 && selected;
      return `<tr class="${selected ? "is-selected" : ""}"><td><input type="checkbox" data-instance-id="${esc(instance.instance_id)}" aria-label="选择 ${esc(instance.display_name)}" ${selected ? "checked" : ""} ${!selected && state.selected.length >= 12 ? "disabled" : ""}></td><td><strong>${esc(instance.display_name)}</strong><small>${esc(instance.db_type)} · ${esc(instance.instance_id.slice(0, 8))}</small></td><td>${badge(instance.monitoring_readiness, `监控 ${label(instance.monitoring_readiness)}`)} <span title="${esc(diagnosticGapText(instance))}">${badge(instance.diagnostic_readiness, `自动诊断 ${label(instance.diagnostic_readiness)}`)}</span></td><td>${badge(qualityState, quality)}</td><td>${esc(shell.fmt(result ? latestSampleForInstance(instance.instance_id) : null))}</td><td><button type="button" data-drill-instance="${esc(instance.instance_id)}" ${current ? "disabled" : ""}>${current ? "当前实例" : "仅看此实例"}</button></td></tr>`;
    }).join("");
    document.getElementById("monitoring-instance-list").innerHTML = rows || '<tr><td colspan="6" class="ops-empty">当前筛选条件下没有数据库实例。</td></tr>';
    renderFreshness();
  }
  function scheduleView() {
    cancelViewRequest();
    state.view = null;
    document.getElementById("monitoring-result").hidden = true;
    disposeCharts();
    renderInstances(); renderFreshness();
    if (!canQuery()) return;
    state.debounceTimer = setTimeout(loadView, 250);
  }
  async function loadView() {
    if (!canQuery()) return;
    cancelViewRequest();
    const controller = new AbortController();
    state.controller = controller;
    setLoading(true);
    try {
      if (state.selected.length === 1 && state.compareSourceId) {
        const params = new URLSearchParams({ source_id: state.sourceId, compare_source_id: state.compareSourceId, profile_id: state.profileId, window: state.window });
        state.view = await KBotAIOpsAuth.request(`/api/v1/apps/aiops/targets/${encodeURIComponent(state.selected[0])}/monitoring/view?${params}`, { signal: controller.signal });
      } else {
        state.view = await KBotAIOpsAuth.request(`${api}/sources/${encodeURIComponent(state.sourceId)}/views`, {
          method: "POST", signal: controller.signal, headers: { "Content-Type": "application/json" }, body: requestBody(),
        });
      }
      if (controller.signal.aborted) return;
      renderResult();
    } catch (error) {
      if (controller.signal.aborted) return;
      shell.toast(error.message || "无法读取实时监控数据");
      document.getElementById("monitoring-gate").hidden = false;
      document.getElementById("monitoring-gate").innerHTML = gateContent("实时指标查询失败", error.message || "请稍后重试。", sourceById()?.capability_gaps || []);
    } finally {
      if (state.controller === controller) state.controller = null;
      if (!controller.signal.aborted) setLoading(false);
      renderFreshness();
    }
  }
  async function loadSourceContext() {
    state.contextController?.abort();
    const contextController = new AbortController();
    state.contextController = contextController;
    cancelViewRequest();
    state.view = null; state.instances = []; state.profiles = []; state.selected = []; state.compareSourceId = ""; state.profileId = "";
    disposeCharts();
    document.getElementById("monitoring-result").hidden = true;
    renderContext(); renderInstances();
    if (sourceById()?.monitoring_readiness !== "READY") {
      if (state.contextController === contextController) state.contextController = null;
      return;
    }
    setLoading(true, "正在读取监控实例与指标视图…");
    try {
      const [instances, profiles] = await Promise.all([
        KBotAIOpsAuth.request(`${api}/sources/${encodeURIComponent(state.sourceId)}/instances`, { signal: contextController.signal }),
        KBotAIOpsAuth.request(`${api}/sources/${encodeURIComponent(state.sourceId)}/profiles`, { signal: contextController.signal }),
      ]);
      if (contextController.signal.aborted) return;
      state.instances = [...instances].sort((left, right) => left.display_name.localeCompare(right.display_name, "zh-CN"));
      state.profiles = profiles;
      const currentQuery = new URLSearchParams(location.search);
      const targetId = currentQuery.get("target_id") || currentQuery.get("targetId") || "";
      state.selected = [(state.instances.find((item) => item.instance_id === targetId) || state.instances[0])?.instance_id].filter(Boolean);
      const profileId = currentQuery.get("profile_id") || currentQuery.get("profileId") || "";
      state.profileId = state.profiles.find((item) => item.profile_id === profileId)?.profile_id || state.profiles[0]?.profile_id || "";
      const next = new URLSearchParams(location.search);
      next.set("source_id", state.sourceId);
      next.set("target_id", state.selected[0] || "");
      next.set("profile_id", state.profileId);
      next.set("window", state.window);
      history.replaceState(null, "", `${location.pathname}?${next}`);
      renderContext(); renderInstances(); scheduleView();
    } catch (error) {
      if (contextController.signal.aborted) return;
      shell.toast(error.message || "无法读取监控实例与指标视图");
    } finally {
      if (state.contextController === contextController) {
        state.contextController = null;
        setLoading(false);
      }
    }
  }
  function instanceQuality(instance) {
    if (instance.status === "UNAVAILABLE") return "不可用";
    const panels = (state.view?.panels || []).filter((panel) => panel.series.some((series) => series.instance_id === instance.instance_id));
    if (!panels.length || panels.every((panel) => panel.quality === "NO_DATA")) return "无有效采样";
    if (instance.status === "PARTIAL" || panels.some((panel) => panel.quality !== "GOOD")) return "部分成功";
    return "正常";
  }
  function deltaText(points, unit) {
    const valid = points.filter((point) => numericValue(point.value) !== null);
    if (valid.length < 2) return "暂无变化基线";
    const delta = numericValue(valid.at(-1).value) - numericValue(valid.at(-2).value);
    if (delta === 0) return "较前值持平";
    if (unit === "state") return `状态由${formatValue(valid.at(-2).value, unit)}变为${formatValue(valid.at(-1).value, unit)}`;
    return `较前值${delta > 0 ? "上升" : "下降"} ${formatValue(Math.abs(delta), unit)}`;
  }
  function capacitySeriesIdentity(series) {
    const dimensions = series.dimensions || {};
    return JSON.stringify([
      series.instance_id || "", series.source_id || "", dimensions.database || "", dimensions.tablespace || "", dimensions.type || "",
    ]);
  }
  function capacityRows(instances, panels) {
    const instanceIds = new Set(instances.map((instance) => instance.instance_id));
    const rows = new Map();
    panels.forEach((panel) => panel.series.filter((series) => instanceIds.has(series.instance_id)).forEach((series) => {
      const key = capacitySeriesIdentity(series);
      const row = rows.get(key) || { instance: series.instance_display_name, dimensions: series.dimensions || {}, source: series.source_display_name, metrics: {} };
      row.metrics[panel.metric_code] = { panel, series, point: latestPoint(series) };
      rows.set(key, row);
    }));
    return [...rows.values()].sort((left, right) => {
      const leftName = left.dimensions.tablespace || left.dimensions.database || left.dimensions.type || "";
      const rightName = right.dimensions.tablespace || right.dimensions.database || right.dimensions.type || "";
      return left.instance.localeCompare(right.instance, "zh-CN") || leftName.localeCompare(rightName, "zh-CN") || left.source.localeCompare(right.source, "zh-CN");
    });
  }
  function capacityDeltaText(points) {
    const valid = points.map((point) => numericValue(point.value)).filter((value) => value !== null);
    if (valid.length < 2) return "暂无变化基线";
    const delta = valid.at(-1) - valid.at(-2);
    if (delta === 0) return "较前值持平";
    return `较前值${delta > 0 ? "上升" : "下降"} ${formatBytes(Math.abs(delta))}`;
  }
  function capacityMetricCell(metric) {
    if (!metric?.point) return '<span class="ops-monitoring-capacity-missing">无有效采样</span>';
    return `<strong>${esc(formatBytes(metric.point.value))}</strong><small>${esc(capacityDeltaText(metric.series.points))}<br>${esc(shell.fmt(metric.point.observed_at))}</small>`;
  }
  function renderCapacityTable(node, instances, panels) {
    const rows = capacityRows(instances, panels);
    const body = rows.length ? rows.map((row) => {
      const used = row.metrics[capacityMetricCodes[0]];
      const free = row.metrics[capacityMetricCodes[1]];
      const maximum = row.metrics[capacityMetricCodes[2]];
      const usedValue = numericValue(used?.point?.value);
      const freeValue = numericValue(free?.point?.value);
      const maximumValue = numericValue(maximum?.point?.value);
      const usedPercent = maximumValue > 0 && usedValue !== null ? Math.max(0, Math.min(100, usedValue / maximumValue * 100)) : null;
      const freePercent = maximumValue > 0 && freeValue !== null ? Math.max(0, Math.min(100, freeValue / maximumValue * 100)) : null;
      const freeBarPercent = freePercent === null ? 0 : Math.min(100 - (usedPercent || 0), freePercent);
      const comparison = usedPercent === null && freePercent === null
        ? '<span class="ops-monitoring-capacity-missing">无法计算</span>'
        : `<div class="ops-monitoring-capacity-bar" role="img" aria-label="已用 ${usedPercent?.toFixed(1) || "—"}%，可用 ${freePercent?.toFixed(1) || "—"}%"><i class="is-used" style="width:${usedPercent || 0}%"></i><i class="is-free" style="width:${freeBarPercent}%"></i></div><small>已用 ${usedPercent?.toFixed(1) || "—"}% · 可用 ${freePercent?.toFixed(1) || "—"}%</small>`;
      const name = row.dimensions.tablespace || row.dimensions.database || "实例汇总";
      const dimensions = [row.dimensions.database, row.dimensions.type, row.source].filter((value) => value && value !== name);
      return `<tr><td><strong>${esc(row.instance)}</strong><small>${esc(row.source)}</small></td><td><strong>${esc(name)}</strong><small>${esc(dimensions.join(" · "))}</small></td><td>${capacityMetricCell(used)}</td><td>${capacityMetricCell(free)}</td><td>${capacityMetricCell(maximum)}</td><td>${comparison}</td></tr>`;
    }).join("") : '<tr><td colspan="6" class="ops-empty">当前查询未返回容量序列。</td></tr>';
    node.innerHTML = `<div class="ops-table-wrap ops-monitoring-capacity-wrap"><table class="ops-table ops-monitoring-capacity-table"><caption>数据库容量对比 · ${instances.length} 个实例</caption><thead><tr><th>数据库实例</th><th>表空间</th><th>已用容量</th><th>可用容量</th><th>最大容量</th><th>容量构成</th></tr></thead><tbody>${body}</tbody></table></div>`;
  }
  function renderSingleSummary() {
    const node = document.getElementById("monitoring-single-summary");
    const panels = (state.view.panels || []).slice(0, 6);
    node.hidden = state.view.profile.profile_id === "database-capacity" || state.view.instances.length !== 1 || !panels.length;
    if (node.hidden) return;
    const instance = state.view.instances[0];
    node.innerHTML = panels.map((panel) => {
      const code = panel.metric_code;
      const series = panel.series.find((item) => item.instance_id === instance.instance_id);
      const point = latestPoint(series);
      return `<article><span>${esc(panel.title)}</span><strong>${esc(point ? formatValue(point.value, panel.unit) : "无有效采样")}</strong><small>${point ? `${esc(deltaText(series.points, panel.unit))} · ${esc(shell.fmt(point.observed_at))}` : esc((allGaps().find((gap) => gap.instance_id === instance.instance_id && gap.metric_code === code) || {}).code || "NO_DATA")}</small></article>`;
    }).join("");
  }
  function chartOption(panel) {
    const colors = ["#465f9e", "#247455", "#9a6518", "#ad3c45", "#526f8a", "#7a5f8f"];
    const series = panel.series.filter((item) => item.points.some((point) => numericValue(point.value) !== null));
    const selected = Object.fromEntries(series.map((item, index) => [seriesLabel(item), index < 6]));
    const common = { animationDuration: 220, color: colors, legend: { top: 0, type: "scroll", selected }, grid: { left: 64, right: 26, top: 52, bottom: 46 } };
    const tooltip = { trigger: "axis", formatter: (entries) => {
      const values = Array.isArray(entries) ? entries : [entries];
      const at = values[0]?.axisValue || values[0]?.name || "";
      return [esc(shell.fmt(at)), ...values.map((entry) => `${entry.marker || ""}${esc(entry.seriesName)}：${esc(formatValue(Array.isArray(entry.value) ? entry.value[1] : entry.value, panel.unit))}`)].join("<br>");
    } };
    if (["STAT", "GAUGE"].includes(panel.visualization)) return {
      ...common, tooltip: { ...tooltip, axisPointer: { type: "shadow" } },
      xAxis: { type: "category", data: series.map(seriesLabel), axisLabel: { interval: 0, hideOverlap: true } },
      yAxis: { type: "value", name: unitLabel(panel.unit), scale: true },
      series: [{ type: "bar", name: panel.title, barMaxWidth: 28, data: series.map((item) => numericValue(latestPoint(item)?.value)) }],
    };
    return {
      ...common, tooltip,
      xAxis: { type: "time", axisLabel: { hideOverlap: true } },
      yAxis: { type: "value", name: unitLabel(panel.unit), scale: true },
      series: series.map((item) => ({ type: "line", name: seriesLabel(item), showSymbol: false, connectNulls: false, step: panel.visualization === "STATE_TIMELINE" ? "end" : false, data: item.points.map((point) => [point.observed_at, numericValue(point.value)]) })),
    };
  }
  function chartSeries(panel) {
    return panel.series.filter((item) => item.points.some((point) => numericValue(point.value) !== null));
  }
  function tableHtml(panel) {
    const times = [...new Set(panel.series.flatMap((series) => series.points.map((point) => point.observed_at)))].sort();
    return `<div class="ops-table-wrap"><table class="ops-table"><thead><tr><th>采样时间</th>${panel.series.map((series) => `<th>${esc(seriesLabel(series))}</th>`).join("")}</tr></thead><tbody>${times.map((time) => `<tr><td>${esc(shell.fmt(time))}</td>${panel.series.map((series) => { const point = series.points.find((item) => item.observed_at === time); return `<td>${point ? `${esc(formatValue(point.value, panel.unit))} · ${esc(point.quality)}` : "—"}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  function renderPanels() {
    const node = document.getElementById("monitoring-panels");
    disposeCharts();
    const isCapacity = state.view.profile.profile_id === "database-capacity";
    node.classList.toggle("is-capacity", isCapacity);
    if (!state.view.panels.length) {
      node.innerHTML = '<div class="ops-panel ops-empty">当前 Profile 没有可展示 Panel。无采样不会显示为 0 或健康。</div>';
      return;
    }
    if (isCapacity) {
      renderCapacityTable(node, state.view.instances, state.view.panels.filter((panel) => capacityMetricCodes.includes(panel.metric_code)));
      return;
    }
    node.innerHTML = state.view.panels.map((panel, index) => `<article class="ops-panel ops-monitoring-panel"><div class="ops-panel-head"><div><h2>${esc(panel.title)}</h2><p>${esc(panel.description)}</p></div><div>${badge(panel.quality, qualityLabels[panel.quality])}<button type="button" data-panel-table="${index}">查看数据</button></div></div><p class="ops-monitoring-chart-summary">${esc(panel.title)}，${panel.series.length} 个实例序列，数据质量${esc(qualityLabels[panel.quality])}。</p>${panel.series.some((series) => series.points.some((point) => numericValue(point.value) !== null)) ? `<div id="monitoring-chart-${index}" class="ops-monitoring-chart" role="img" aria-label="${esc(panel.title)}趋势图"></div>` : '<div class="ops-empty">无有效采样。缺失点保持为空，不按 0 绘制。</div>'}<div class="ops-monitoring-panel-meta"><span>指标 ${esc(panel.metric_code)}</span><span>单位 ${esc(unitLabel(panel.unit) || "无量纲")}</span>${panel.series.map((series) => `<span>${esc(seriesLabel(series))} 覆盖率 ${(series.coverage_ratio * 100).toFixed(1)}%</span>`).join("")}</div><div class="ops-monitoring-data-table" data-panel-data="${index}" hidden>${tableHtml(panel)}</div></article>`).join("");
    state.view.panels.forEach((panel, index) => {
      const container = document.getElementById(`monitoring-chart-${index}`);
      if (!container) return;
      const chart = globalThis.echarts.init(container);
      chart.setOption(chartOption(panel), true);
      chart.on("click", (event) => {
        const availableSeries = chartSeries(panel);
        const series = ["STAT", "GAUGE"].includes(panel.visualization)
          ? availableSeries[event.dataIndex]
          : availableSeries[event.seriesIndex];
        if (series) drillInto(series.instance_id);
      });
      state.charts.set(panel.panel_id, chart);
    });
  }
  function renderQuality() {
    const gaps = allGaps();
    document.getElementById("monitoring-result-state").innerHTML = badge(state.view.partial ? "PARTIAL" : "GOOD", state.view.partial ? "部分成功" : "查询完成");
    const sourceLabel = [state.view.source, state.view.compare_source].filter(Boolean).map((source) => `${source.display_name} · ${source.source_type}`).join(" / ");
    document.getElementById("monitoring-quality-context").innerHTML = `<dt>监控源</dt><dd>${esc(sourceLabel)}</dd><dt>采样窗口</dt><dd>${esc(shell.fmt(state.view.window.start))} 至 ${esc(shell.fmt(state.view.window.end))}</dd><dt>Profile</dt><dd>${esc(state.view.profile.display_name)} · ${esc(state.view.profile.version)}</dd><dt>返回实例</dt><dd>${state.view.instances.length}</dd>`;
    document.getElementById("monitoring-gaps").innerHTML = gaps.length ? gaps.map((gap) => `<article>${badge("PARTIAL", gap.scope)}<div><strong>${esc(gap.code)}${gap.metric_code ? ` · ${esc(gap.metric_code)}` : ""}</strong><p>${esc(gap.detail)}</p></div><small>${gap.retryable ? "可重试" : "需配置或检查"}</small></article>`).join("") : '<div class="ops-empty">当前查询未返回能力或采样缺口。</div>';
  }
  function renderResult() {
    document.getElementById("monitoring-gate").hidden = true;
    document.getElementById("monitoring-result").hidden = false;
    const counts = {
      available: state.view.instances.filter((item) => item.status === "AVAILABLE").length,
      partial: state.view.instances.filter((item) => item.status === "PARTIAL").length,
      unavailable: state.view.instances.filter((item) => item.status === "UNAVAILABLE").length,
    };
    document.getElementById("monitoring-quality-ledger").innerHTML = [["正常实例", counts.available], ["部分成功", counts.partial], ["不可用", counts.unavailable], ["缺口数量", allGaps().length], ["时间窗口", state.window]].map(([name, value]) => `<div><span>${esc(name)}</span><strong>${esc(value)}</strong></div>`).join("");
    renderSingleSummary(); renderPanels(); renderQuality(); renderInstances(); renderFreshness();
  }
  function drillInto(instanceId) {
    state.selected = [instanceId];
    const next = new URLSearchParams(location.search);
    next.set("source_id", state.sourceId); next.set("target_id", instanceId); next.set("profile_id", state.profileId); next.set("window", state.window);
    history.replaceState(null, "", `${location.pathname}?${next}`);
    renderContext(); renderInstances(); scheduleView();
  }
  function scheduleRefresh() {
    clearInterval(state.refreshTimer);
    const seconds = Number(document.getElementById("monitoring-refresh-interval").value || 0);
    state.refreshTimer = seconds > 0 ? setInterval(loadView, seconds * 1000) : null;
  }
  async function initialize() {
    state.window = ["15m", "1h", "6h", "24h"].includes(query.get("window")) ? query.get("window") : "1h";
    document.getElementById("monitoring-window").value = state.window;
    try {
      state.sources = await KBotAIOpsAuth.request(`${api}/sources`);
      const requested = query.get("source_id") || query.get("sourceId") || "";
      state.sourceId = state.sources.find((item) => item.source_id === requested)?.source_id || state.sources.find((item) => item.monitoring_readiness === "READY")?.source_id || state.sources[0]?.source_id || "";
      renderContext();
      await loadSourceContext();
      renderGate(); renderFreshness();
    } catch (error) {
      document.getElementById("monitoring-gate").innerHTML = gateContent("监控源目录加载失败", error.message || "请稍后重试。");
      shell.toast(error.message || "无法读取监控源目录");
    }
  }

  document.getElementById("monitoring-source").onchange = (event) => { state.sourceId = event.target.value; loadSourceContext(); };
  document.getElementById("monitoring-window").onchange = (event) => { state.window = event.target.value; scheduleView(); };
  document.getElementById("monitoring-profile").onchange = (event) => {
    state.profileId = event.target.value;
    const next = new URLSearchParams(location.search);
    next.set("profile_id", state.profileId);
    history.replaceState(null, "", `${location.pathname}?${next}`);
    scheduleView();
  };
  document.getElementById("monitoring-compare-source").onchange = (event) => { state.compareSourceId = event.target.value; scheduleView(); };
  document.getElementById("monitoring-instance-search").oninput = (event) => { state.search = event.target.value; renderInstances(); };
  document.getElementById("monitoring-instance-list").onchange = (event) => {
    const input = event.target.closest("[data-instance-id]");
    if (!input) return;
    const id = input.dataset.instanceId;
    if (input.checked && !state.selected.includes(id)) {
      if (state.selected.length >= 12) { input.checked = false; shell.toast("一次最多选择 12 个数据库实例"); return; }
      state.selected.push(id);
    } else if (!input.checked) state.selected = state.selected.filter((value) => value !== id);
    if (state.selected.length !== 1) state.compareSourceId = "";
    renderContext();
    renderInstances(); scheduleView();
  };
  document.getElementById("monitoring-select-filtered").onclick = () => { state.selected = [...new Set([...state.selected, ...filteredInstances().map((item) => item.instance_id)])].slice(0, 12); if (state.selected.length !== 1) state.compareSourceId = ""; renderContext(); renderInstances(); scheduleView(); };
  document.getElementById("monitoring-clear-selection").onclick = () => { state.selected = []; state.compareSourceId = ""; renderContext(); renderInstances(); scheduleView(); };
  document.getElementById("monitoring-instance-list").onclick = (event) => { const button = event.target.closest("[data-drill-instance]"); if (button && !button.disabled) drillInto(button.dataset.drillInstance); };
  document.getElementById("monitoring-panels").onclick = (event) => { const button = event.target.closest("[data-panel-table]"); if (!button) return; const table = document.querySelector(`[data-panel-data="${button.dataset.panelTable}"]`); table.hidden = !table.hidden; button.textContent = table.hidden ? "查看数据" : "收起数据"; };
  document.getElementById("refresh-monitoring").onclick = loadView;
  document.getElementById("monitoring-refresh-interval").onchange = scheduleRefresh;
  addEventListener("resize", () => state.charts.forEach((chart) => chart.resize()));
  shell.ready.then(initialize);
})();
