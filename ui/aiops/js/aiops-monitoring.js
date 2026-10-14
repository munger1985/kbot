(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/monitoring";
  const shell = globalThis.KBotAIOpsShell;
  const state = {
    sources: [], instances: [], profiles: [], view: null,
    sourceId: "", baseline: "1w", profileId: "", window: "1h", selected: [], search: "",
    refreshTimer: null, controller: null, contextController: null, charts: new Map(),
    visiblePanels: [], dataDrawerPanelId: "", dataDrawerTrigger: null, railCollapsed: false,
  };
  const readinessLabels = {
    READY: "就绪", DISABLED: "已停用", DISCONNECTED: "连接失败",
    CAPABILITY_MISSING: "能力缺失", NO_MAPPED_INSTANCE: "未映射",
    PARTIAL: "部分就绪", UNAVAILABLE: "不可用", AVAILABLE: "可用",
  };
  const qualityLabels = { GOOD: "正常", PARTIAL: "部分成功", NO_DATA: "无有效采样" };
  const capacityMetricRoles = ["STORAGE_USED", "STORAGE_FREE", "STORAGE_MAX", "DATABASE_SIZE"];
  const capacityRoleLabels = {
    STORAGE_USED: "已用容量", STORAGE_FREE: "可用容量",
    STORAGE_MAX: "最大容量", DATABASE_SIZE: "数据库容量",
  };
  const esc = shell.escape;
  const query = new URLSearchParams(location.search);
  const sourceById = () => state.sources.find((item) => item.source_id === state.sourceId) || null;
  const profileById = () => state.profiles.find((item) => item.profile_id === state.profileId) || null;
  const baselineLabel = () => ({ "1d": "过去 1 天", "1w": "过去 1 周", "1m": "过去 1 月" })[state.baseline] || "过去 1 周";
  const seriesSubject = (series) => {
    const dimensions = series.dimensions || {};
    return dimensions.tablespace || dimensions.tablespace_name
      || dimensions.schema || dimensions.schema_name
      || dimensions.database || dimensions.database_name || dimensions.datname
      || dimensions.mountpoint || dimensions.device || dimensions.application_name
      || dimensions.slot_name || dimensions.type || series.instance_display_name;
  };
  const seriesLabel = (series) => [seriesSubject(series), series.dimensions?.__metric_label, series.source_display_name].filter(Boolean).join(" · ");
  const tooltipSeriesLabel = (series) => [seriesSubject(series), series.dimensions?.__metric_label].filter(Boolean).join(" · ");
  const capacityScopeLabel = (scope) => ({ FILESYSTEM: "文件系统", LOGICAL: "逻辑占用" })[scope] || "";

  function label(value) { return readinessLabels[value] || value || "不可用"; }
  function setRailCollapsed(collapsed) {
    state.railCollapsed = collapsed;
    document.getElementById("monitoring-workspace").classList.toggle("is-rail-collapsed", collapsed);
    const toggle = document.getElementById("monitoring-rail-toggle");
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.setAttribute("aria-label", collapsed ? "展开实时监控查询上下文" : "收起实时监控查询上下文");
  }
  function pointQualityLabel(value) {
    return ({ GOOD: "正常", INVALID: "无效", STALE: "已过期", ESTIMATED: "估算" })[value] || value || "未知";
  }
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
      errors_per_second: "次/秒", bytes_per_second: "B/秒（自适应）",
      bytes: "GB", "1": "",
    })[unit] ?? unit;
  }
  function formatValue(value, unit = "") {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "boolean" || unit === "state") return Number(value) > 0 ? "可用" : "不可用";
    const number = Number(value);
    if (Number.isFinite(number)) {
      const text = number.toLocaleString("zh-CN", { maximumFractionDigits: Math.abs(number) >= 100 ? 1 : 2 });
      if (unit === "percent") return `${text} %`;
      if (unit === "bytes") return formatBytes(number);
      if (unit === "bytes_per_second") return formatByteRate(number);
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
    const gigabytes = number / (1024 ** 3);
    if (number > 0 && gigabytes < 0.01) return "<0.01 GB";
    return `${gigabytes.toLocaleString("zh-CN", { maximumFractionDigits: 2 })} GB`;
  }
  function formatByteRate(value) {
    const number = numericValue(value);
    if (number === null) return "—";
    const absolute = Math.abs(number);
    const scale = absolute >= 1024 ** 3
      ? { divisor: 1024 ** 3, unit: "GB/秒" }
      : absolute >= 1024 ** 2
        ? { divisor: 1024 ** 2, unit: "MB/秒" }
        : absolute >= 1024
          ? { divisor: 1024, unit: "KB/秒" }
          : { divisor: 1, unit: "B/秒" };
    const scaled = number / scale.divisor;
    if (scaled !== 0 && Math.abs(scaled) < 0.01) return `${scaled < 0 ? "-" : ""}<0.01 ${scale.unit}`;
    return `${scaled.toLocaleString("zh-CN", { maximumFractionDigits: Math.abs(scaled) >= 100 ? 1 : 2 })} ${scale.unit}`;
  }
  function axisValueLabel(value, unit) {
    if (unit === "state") return Number(value) > 0 ? "可用" : "不可用";
    if (unit === "bytes") {
      return (Number(value) / (1024 ** 3)).toLocaleString("zh-CN", { maximumFractionDigits: 2 });
    }
    if (unit === "bytes_per_second") return formatByteRate(Number(value));
    return Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 2 });
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
  function canQuery() {
    return sourceById()?.monitoring_readiness === "READY" && state.profileId && state.selected.length === 1;
  }
  function cancelViewRequest() {
    state.controller?.abort();
    state.controller = null;
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
      `实例 ${state.instances.find((item) => state.selected.includes(item.instance_id))?.display_name || "未选择"}`, status,
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
      NO_MAPPED_INSTANCE: ["尚未关联数据库资源", "请在监控接入中将发现的数据库映射到已登记资源。"],
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
    document.getElementById("monitoring-window").disabled = sourceById()?.monitoring_readiness !== "READY";
    document.getElementById("monitoring-window").value = state.window;
    document.getElementById("monitoring-baseline").value = state.baseline;
    const diagnosticDetail = diagnosticGapText(sourceById());
    document.getElementById("monitoring-readiness").innerHTML = `${badge(sourceById()?.monitoring_readiness, `监控 ${label(sourceById()?.monitoring_readiness)}`)}<span title="${esc(diagnosticDetail)}">${badge(sourceById()?.diagnostic_readiness, `自动诊断 ${label(sourceById()?.diagnostic_readiness)}`)}</span>`;
  }
  function filteredInstances() {
    const value = state.search.trim().toLowerCase();
    return state.instances.filter((item) => !value || `${item.business_system_name || ""} ${item.resource_name || ""} ${item.display_name} ${item.db_type}`.toLowerCase().includes(value));
  }
  function renderInstances() {
    const picker = document.getElementById("monitoring-instance-picker");
    picker.hidden = renderGate();
    if (picker.hidden) return;
    const instances = filteredInstances();
    const count = document.getElementById("monitoring-selection-count");
    count.classList.toggle("has-selection", Boolean(state.selected.length));
    count.innerHTML = `<strong>${state.selected.length}</strong> / ${instances.length} ${state.selected.length ? "已选择" : "请选择"}`;
    const groups = new Map();
    instances.forEach((instance) => {
      const key = `${instance.business_system_id || "unassigned"}:${instance.resource_id || "unassigned"}`;
      const group = groups.get(key) || { systemName: instance.business_system_name || "未归属业务系统", resourceName: instance.resource_name || "未归属数据库资源", items: [] };
      group.items.push(instance);
      groups.set(key, group);
    });
    const card = (instance) => {
      const selected = state.selected.includes(instance.instance_id);
      const result = state.view?.instances.find((item) => item.instance_id === instance.instance_id);
      const queryStatus = targetPickerStatus(instance, result);
      const current = state.selected.length === 1 && selected;
      const displayName = instance.resource_name || instance.display_name;
      return `<label class="ops-monitoring-instance-card ${selected ? "is-selected" : ""}"><input type="radio" name="monitoring_instance" data-instance-id="${esc(instance.instance_id)}" aria-label="选择 ${esc(displayName)}" ${selected ? "checked" : ""}><span class="ops-monitoring-instance-copy"><span><strong>${esc(displayName)}</strong><em>${esc(instance.db_type)}</em></span><small>${esc(instance.business_system_name || "未归属业务系统")}</small></span><span class="ops-monitoring-instance-action">${current ? "当前查看" : "选择查看"}</span><span class="ops-monitoring-instance-meta"><span><small>接入状态</small><span>${badge(instance.monitoring_readiness, `监控 ${label(instance.monitoring_readiness)}`)} <span title="${esc(diagnosticGapText(instance))}">${badge(instance.diagnostic_readiness, `自动诊断 ${label(instance.diagnostic_readiness)}`)}</span></span></span><span><small>查询状态</small>${badge(queryStatus.value, queryStatus.label)}</span><span><small>最新采样</small><time>${esc(shell.fmt(result ? latestSampleForInstance(instance.instance_id) : null))}</time></span></span></label>`;
    };
    const cards = [...groups.values()].map((group) => `<section class="ops-monitoring-instance-group"><header><strong>${esc(group.systemName)}</strong><span>${esc(group.resourceName)} · ${group.items.length} 个数据库入口</span></header><div>${group.items.map(card).join("")}</div></section>`).join("");
    document.getElementById("monitoring-instance-list").innerHTML = cards || '<div class="ops-empty">当前筛选条件下没有数据库实例。</div>';
    renderFreshness();
  }
  function scheduleView() {
    cancelViewRequest();
    state.view = null;
    document.getElementById("monitoring-result").hidden = true;
    disposeCharts();
    renderInstances(); renderFreshness();
    if (!canQuery()) return;
    void loadView();
  }
  function updateLocation() {
    const next = new URLSearchParams(location.search);
    next.set("source_id", state.sourceId);
    next.set("target_id", state.selected[0] || "");
    next.delete("profile_id");
    next.delete("profileId");
    next.set("window", state.window);
    next.set("baseline", state.baseline);
    history.replaceState(null, "", `${location.pathname}?${next}`);
  }
  async function loadSelectedTargetProfiles({ signal }) {
    const targetId = state.selected[0];
    if (!targetId) {
      state.profiles = [];
      state.profileId = "";
      return;
    }
    const profiles = await KBotAIOpsAuth.request(
      `/api/v1/apps/aiops/targets/${encodeURIComponent(targetId)}/monitoring/profiles`,
      { signal },
    );
    if (signal.aborted) return;
    state.profiles = profiles;
    state.profileId = state.profiles.find((item) => item.is_default)?.profile_id
      || state.profiles[0]?.profile_id || "";
  }
  async function loadView() {
    if (!canQuery()) return;
    cancelViewRequest();
    const controller = new AbortController();
    state.controller = controller;
    setLoading(true);
    try {
      const params = new URLSearchParams({ source_id: state.sourceId, profile_id: state.profileId, window: state.window, baseline: state.baseline });
      state.view = await KBotAIOpsAuth.request(`/api/v1/apps/aiops/targets/${encodeURIComponent(state.selected[0])}/monitoring/view?${params}`, { signal: controller.signal });
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
    state.view = null; state.instances = []; state.profiles = []; state.selected = []; state.profileId = "";
    disposeCharts();
    document.getElementById("monitoring-result").hidden = true;
    renderContext(); renderInstances();
    if (sourceById()?.monitoring_readiness !== "READY") {
      if (state.contextController === contextController) state.contextController = null;
      return;
    }
    setLoading(true, "正在读取监控实例与指标视图…");
    try {
      const instances = await KBotAIOpsAuth.request(
        `${api}/sources/${encodeURIComponent(state.sourceId)}/instances`,
        { signal: contextController.signal },
      );
      if (contextController.signal.aborted) return;
      state.instances = [...instances].sort((left, right) => left.display_name.localeCompare(right.display_name, "zh-CN"));
      const currentQuery = new URLSearchParams(location.search);
      const targetId = currentQuery.get("target_id") || currentQuery.get("targetId") || "";
      state.selected = [(state.instances.find((item) => item.instance_id === targetId) || state.instances[0])?.instance_id].filter(Boolean);
      await loadSelectedTargetProfiles({ signal: contextController.signal });
      if (contextController.signal.aborted) return;
      updateLocation();
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
  async function selectInstance(instanceId) {
    state.contextController?.abort();
    const contextController = new AbortController();
    state.contextController = contextController;
    cancelViewRequest();
    state.selected = [instanceId];
    state.view = null;
    state.profiles = [];
    state.profileId = "";
    document.getElementById("monitoring-result").hidden = true;
    disposeCharts();
    renderContext(); renderInstances();
    setLoading(true, "正在读取所选实例的监控指标…");
    try {
      await loadSelectedTargetProfiles({ signal: contextController.signal });
      if (contextController.signal.aborted) return;
      updateLocation();
      renderContext(); renderInstances(); scheduleView();
    } catch (error) {
      if (contextController.signal.aborted) return;
      shell.toast(error.message || "无法读取所选实例的 Monitoring Profile");
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
  function targetPickerStatus(instance, result) {
    if (result) {
      const quality = instanceQuality(result);
      const value = quality === "正常" ? "GOOD" : quality === "部分成功" ? "PARTIAL"
        : quality === "不可用" ? "UNAVAILABLE" : "NO_DATA";
      return { value, label: quality };
    }
    return {
      value: instance.monitoring_readiness,
      label: instance.monitoring_readiness === "READY" ? "监控就绪" : label(instance.monitoring_readiness),
    };
  }
  function deltaText(points, unit) {
    const valid = points.filter((point) => numericValue(point.value) !== null);
    if (valid.length < 2) return "暂无变化基线";
    const delta = numericValue(valid.at(-1).value) - numericValue(valid.at(-2).value);
    if (delta === 0) return "较前值持平";
    if (unit === "state") return `状态由${formatValue(valid.at(-2).value, unit)}变为${formatValue(valid.at(-1).value, unit)}`;
    return `较前值${delta > 0 ? "上升" : "下降"} ${formatValue(Math.abs(delta), unit)}`;
  }
  function buildCapacityTrendSeries(series, bounded) {
    const groups = new Map();
    series.forEach((item) => {
      const key = `${item.source_id}:${item.instance_id}:${seriesSubject(item)}`;
      const roles = groups.get(key) || new Map();
      roles.set(item.dimensions?.__metric_role || "DATABASE_SIZE", item);
      groups.set(key, roles);
    });
    return [...groups.values()].flatMap((roles) => {
      const used = roles.get("STORAGE_USED") || roles.get("DATABASE_SIZE");
      if (!used) return [];
      if (!bounded) return [{
        ...used, series_key: `${used.series_key}:logical-trend`,
        dimensions: { ...(used.dimensions || {}), __metric_label: "逻辑占用" },
      }];
      const free = roles.get("STORAGE_FREE");
      const max = roles.get("STORAGE_MAX");
      const baselineUsed = used.baseline?.average ?? null;
      const baselineFree = free?.baseline?.average ?? null;
      const baselineMax = max?.baseline?.average ?? null;
      const baselineTotal = baselineMax !== null && baselineMax > 0
        ? baselineMax
        : baselineUsed !== null && baselineFree !== null && baselineUsed + baselineFree > 0 ? baselineUsed + baselineFree : null;
      const baseline = used.baseline ? {
        ...used.baseline,
        average: baselineUsed !== null && baselineTotal !== null ? baselineUsed / baselineTotal * 100 : null,
      } : null;
      const points = used.points.flatMap((point) => {
        const usedValue = numericValue(point.value);
        const freeValue = seriesValueAt(free, point.observed_at);
        const maxValue = seriesValueAt(max, point.observed_at);
        const total = maxValue !== null && maxValue > 0
          ? maxValue
          : usedValue !== null && freeValue !== null && usedValue + freeValue > 0 ? usedValue + freeValue : null;
        return usedValue !== null && total !== null ? [{ ...point, value: usedValue / total * 100 }] : [];
      });
      return points.length ? [{
        ...used, series_key: `${used.series_key}:utilization-trend`,
        dimensions: { ...(used.dimensions || {}), __metric_label: "容量使用率" }, baseline, points,
      }] : [];
    });
  }
  function mergeCapacityPanels(panels) {
    const capacityPanels = panels.filter((panel) => capacityMetricRoles.includes(panel.semantic_role));
    const regularPanels = panels.filter((panel) => panel.semantic_role !== "AVAILABILITY" && !capacityMetricRoles.includes(panel.semantic_role));
    if (!capacityPanels.length) return regularPanels;
    const quality = capacityPanels.every((panel) => panel.quality === "NO_DATA")
      ? "NO_DATA" : capacityPanels.some((panel) => panel.quality !== "GOOD") ? "PARTIAL" : "GOOD";
    const hasDatabaseCapacityLimit = capacityPanels.some((panel) => ["db.storage.free_bytes", "db.storage.max_bytes"].includes(panel.metric_code));
    const metricCode = capacityPanels.map((panel) => panel.metric_code).join(",");
    const mergedSeries = capacityPanels.flatMap((panel) => panel.series.map((series) => ({
      ...series,
      dimensions: {
        ...(series.dimensions || {}), __metric_role: panel.semantic_role,
        __metric_label: capacityRoleLabels[panel.semantic_role], __metric_code: panel.metric_code,
        __capacity_scope: panel.metric_code.startsWith("host.filesystem.")
          ? "FILESYSTEM" : hasDatabaseCapacityLimit ? "DATABASE" : "LOGICAL",
      },
    })));
    regularPanels.push({
      panel_id: "database.storage.capacity", title: "数据库容量",
      description: "Oracle 按表空间展示容量；MySQL schema 与 PostgreSQL database 展示逻辑占用，真实余量由承载主机文件系统容量反映。",
      visualization: "CAPACITY_BAR", metric_code: metricCode,
      semantic_role: "STORAGE_CAPACITY", unit: "bytes", value_kind: "GAUGE", quality,
      capacity_thresholds: {
        warning_percent: state.view.profile.capacity_warning_percent,
        critical_percent: state.view.profile.capacity_critical_percent,
      },
      summary: { metric_count: capacityPanels.length },
      series: mergedSeries,
    });
    regularPanels.push({
      panel_id: "database.storage.capacity-trend",
      title: hasDatabaseCapacityLimit ? "数据库容量使用率趋势" : "数据库逻辑占用趋势",
      description: hasDatabaseCapacityLimit
        ? "按时间展示各表空间容量使用率，阈值仍由 Monitoring Profile 下发。"
        : "按时间展示各 schema 或 database 的逻辑占用绝对值，不换算为百分比。",
      visualization: "TIME_SERIES", metric_code: metricCode,
      semantic_role: "STORAGE_CAPACITY_TREND", unit: hasDatabaseCapacityLimit ? "percent" : "bytes",
      value_kind: "GAUGE", quality, summary: { metric_count: capacityPanels.length },
      capacity_thresholds: {
        warning_percent: state.view.profile.capacity_warning_percent,
        critical_percent: state.view.profile.capacity_critical_percent,
      },
      series: buildCapacityTrendSeries(mergedSeries, hasDatabaseCapacityLimit),
    });
    return regularPanels;
  }
  function renderSingleSummary() {
    const node = document.getElementById("monitoring-single-summary");
    const panels = (state.view.panels || []).slice(0, 6);
    node.hidden = state.view.instances.length !== 1 || !panels.length;
    if (node.hidden) return;
    const instance = state.view.instances[0];
    node.innerHTML = panels.map((panel) => {
      const code = panel.metric_code;
      const series = panel.series.find((item) => item.instance_id === instance.instance_id);
      const point = latestPoint(series);
      return `<article><span>${esc(panel.title)}</span><strong>${esc(point ? formatValue(point.value, panel.unit) : "无有效采样")}</strong><small>${point ? `${esc(deltaText(series.points, panel.unit))} · ${esc(shell.fmt(point.observed_at))}` : esc((allGaps().find((gap) => gap.instance_id === instance.instance_id && gap.metric_code === code) || {}).code || "NO_DATA")}</small></article>`;
    }).join("");
  }
  function seriesValueAt(series, observedAt) {
    if (!series) return null;
    const exact = series.points.find((point) => point.observed_at === observedAt);
    if (exact) return numericValue(exact.value);
    const previous = sortedPoints(series).filter((point) => point.observed_at <= observedAt).at(-1);
    return previous ? numericValue(previous.value) : null;
  }
  function capacityRows(panel) {
    const available = chartSeries(panel);
    const sourceCount = new Set(available.map((series) => series.source_id)).size;
    const groups = new Map();
    available.forEach((series) => {
      const role = series.dimensions?.__metric_role || "DATABASE_SIZE";
      const scope = series.dimensions?.__capacity_scope || "DATABASE";
      const category = seriesSubject(series);
      const key = `${series.source_id}:${series.instance_id}:${scope}:${category}`;
      const group = groups.get(key) || {
        key, category, scope, sourceName: series.source_display_name,
        instanceId: series.instance_id, roles: new Map(),
      };
      group.roles.set(role, series);
      groups.set(key, group);
    });
    const warning = panel.capacity_thresholds?.warning_percent;
    const critical = panel.capacity_thresholds?.critical_percent;
    const rows = [...groups.values()].map((group) => {
      const usedSeries = group.roles.get("STORAGE_USED") || group.roles.get("DATABASE_SIZE");
      const freeSeries = group.roles.get("STORAGE_FREE");
      const maxSeries = group.roles.get("STORAGE_MAX");
      const used = numericValue(latestPoint(usedSeries)?.value);
      const reportedFree = numericValue(latestPoint(freeSeries)?.value);
      const reportedMax = numericValue(latestPoint(maxSeries)?.value);
      const total = reportedMax !== null && reportedMax > 0
        ? reportedMax
        : used !== null && reportedFree !== null && used + reportedFree > 0 ? used + reportedFree : null;
      const free = total !== null && used !== null ? Math.max(total - used, 0) : null;
      const utilization = total !== null && used !== null ? used / total * 100 : null;
      const baselineUsed = usedSeries?.baseline?.average ?? null;
      const baselineFree = freeSeries?.baseline?.average ?? null;
      const baselineMax = maxSeries?.baseline?.average ?? null;
      const baselineTotal = baselineMax !== null && baselineMax > 0
        ? baselineMax
        : baselineUsed !== null && baselineFree !== null && baselineUsed + baselineFree > 0 ? baselineUsed + baselineFree : null;
      const baselineAverage = group.scope === "LOGICAL"
        ? baselineUsed
        : baselineUsed !== null && baselineTotal !== null ? baselineUsed / baselineTotal * 100 : null;
      const inconsistent = used !== null && (used < 0
        || (reportedFree !== null && reportedFree < 0)
        || (reportedMax !== null && reportedMax <= 0)
        || (total !== null && used > total * 1.01));
      let status = "未提供容量上限"; let statusTone = "unknown";
      if (group.scope === "LOGICAL") { status = "逻辑占用"; statusTone = "unknown"; }
      else if (inconsistent) { status = "容量数据不一致"; statusTone = "danger"; }
      else if (utilization !== null && critical !== undefined && utilization >= critical) { status = "空间紧张"; statusTone = "danger"; }
      else if (utilization !== null && warning !== undefined && utilization >= warning) { status = "需要关注"; statusTone = "warning"; }
      else if (utilization !== null) { status = "空间充足"; statusTone = "success"; }
      return {
        ...group, label: [group.category, capacityScopeLabel(group.scope), sourceCount > 1 ? group.sourceName : ""].filter(Boolean).join(" · "),
        used, free, total, utilization, baselineAverage,
        baselineSampleCount: usedSeries?.baseline?.sample_count || 0,
        inconsistent, status, statusTone,
        barPercent: utilization === null ? 0 : Math.min(Math.max(utilization, 0), 100),
      };
    });
    const largestLogicalValue = Math.max(0, ...rows.flatMap((row) => (
      row.scope === "LOGICAL" && row.used !== null ? [row.used] : []
    )));
    rows.forEach((row) => {
      if (row.scope === "LOGICAL") {
        row.barPercent = row.used !== null && largestLogicalValue > 0
          ? Math.min(Math.max(row.used / largestLogicalValue * 100, 0), 100) : 0;
      }
    });
    return rows.sort((left, right) => {
      if (left.utilization === null && right.utilization !== null) return 1;
      if (left.utilization !== null && right.utilization === null) return -1;
      return (right.utilization ?? -1) - (left.utilization ?? -1)
        || (right.used ?? -1) - (left.used ?? -1)
        || left.label.localeCompare(right.label, "zh-CN");
    });
  }
  function capacityPanelHtml(panel) {
    const rows = capacityRows(panel);
    if (!rows.length) return '<div class="ops-empty">无有效采样。缺失点保持为空，不按 0 绘制。</div>';
    const logicalOnly = rows.every((row) => row.scope === "LOGICAL");
    const thresholds = panel.capacity_thresholds || {};
    const list = rows.map((row) => `<button type="button" class="ops-capacity-row" data-capacity-instance-id="${esc(row.instanceId)}"><span class="ops-capacity-heading"><strong title="${esc(row.label)}">${esc(row.label)}</strong><span class="ops-capacity-status is-${esc(row.statusTone)}">${esc(row.status)}</span></span>${row.scope === "LOGICAL" ? `<span class="ops-capacity-bar-shell ops-capacity-bar-shell-absolute"><span class="ops-capacity-bar"><i style="width:${row.barPercent}%"></i></span><strong>${esc(row.used === null ? "无有效采样" : formatValue(row.used, panel.unit))}</strong></span>` : row.total !== null && row.used !== null ? `<span class="ops-capacity-bar-shell"><span class="ops-capacity-bar ops-capacity-threshold-rail ${row.inconsistent ? "is-inconsistent" : ""}" style="--warning-at:${thresholds.warning_percent || 80}%;--critical-at:${thresholds.critical_percent || 90}%"><i style="width:${row.barPercent}%"></i></span><strong>${row.utilization.toFixed(1)}%</strong></span>` : `<span class="ops-capacity-unbounded"><i></i><strong>${esc(row.used === null ? "无有效采样" : formatValue(row.used, panel.unit))}</strong></span>`}${row.scope !== "LOGICAL" ? `<span class="ops-capacity-values"><span><small>已用</small><strong>${esc(formatValue(row.used, panel.unit))}</strong></span><span><small>可用</small><strong>${esc(row.free === null ? "未提供" : formatValue(row.free, panel.unit))}</strong></span><span><small>最大容量</small><strong>${esc(row.total === null ? "未提供" : formatValue(row.total, panel.unit))}</strong></span></span>` : ""}</button>`).join("");
    const legend = logicalOnly
      ? '<span><i class="is-used"></i>逻辑占用</span><small>柱长按当前最大逻辑占用归一，仅用于库间大小对比，不表示使用率。</small>'
      : `<span><i class="is-used"></i>已用</span><span><i class="is-available"></i>可用</span><small>状态阈值由 ${esc(thresholds.warning_percent)}% / ${esc(thresholds.critical_percent)}% Monitoring Profile 下发</small>`;
    return `<div class="ops-capacity-overview"><div class="ops-capacity-legend">${legend}</div><div class="ops-capacity-row-list">${list}</div></div>`;
  }
  function chartOption(panel) {
    const colors = ["#465f9e", "#247455", "#9a6518", "#ad3c45", "#526f8a", "#7a5f8f"];
    const series = panel.series.filter((item) => item.points.some((point) => numericValue(point.value) !== null));
    const selected = Object.fromEntries(series.map((item, index) => [seriesLabel(item), index < 6]));
    const showLegend = series.length > 1;
    const common = { animationDuration: 220, color: colors, legend: { show: showLegend, top: 0, type: "scroll", selected, textStyle: { color: "#697386", fontSize: 10 } }, grid: { left: 54, right: 18, top: showLegend ? 38 : 16, bottom: 34 } };
    const sourceNames = [...new Set(series.map((item) => item.source_display_name).filter(Boolean))].join(" / ");
    const tooltip = { trigger: "axis", formatter: (entries) => {
      const values = Array.isArray(entries) ? entries : [entries];
      const at = values[0]?.axisValue || values[0]?.name || "";
      return [esc(shell.fmt(at)), ...values.map((entry) => {
        const item = ["STAT", "GAUGE"].includes(panel.visualization)
          ? series[Number(entry.dataIndex || 0)]
          : series[Number(entry.seriesIndex || 0)];
        return `${entry.marker || ""}${esc(item ? tooltipSeriesLabel(item) : entry.seriesName)}：${esc(formatValue(Array.isArray(entry.value) ? entry.value[1] : entry.value, panel.unit))}`;
      }), sourceNames ? `来源：${esc(sourceNames)}` : ""].filter(Boolean).join("<br>");
    } };
    const axisBounds = panel.unit === "percent" ? { min: 0, max: 100 }
      : panel.unit === "state" ? { min: 0, max: 1 } : { scale: true };
    if (["STAT", "GAUGE"].includes(panel.visualization)) return {
      ...common, tooltip: { ...tooltip, axisPointer: { type: "shadow" } },
      xAxis: { type: "category", data: series.map(seriesLabel), axisTick: { show: false }, axisLabel: { interval: 0, hideOverlap: true, color: "#697386", fontSize: 10 } },
      yAxis: { type: "value", name: panel.unit === "bytes_per_second" ? undefined : unitLabel(panel.unit), axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: "#697386", fontSize: 10, formatter: (value) => axisValueLabel(value, panel.unit) }, splitNumber: 3, splitLine: { lineStyle: { color: "#d8dde7", opacity: .62 } }, ...axisBounds },
      series: [{ type: "bar", name: panel.title, barMaxWidth: 22, itemStyle: { borderRadius: [3, 3, 0, 0] }, data: series.map((item) => numericValue(latestPoint(item)?.value)) }],
    };
    return {
      ...common, tooltip,
      xAxis: { type: "time", axisTick: { show: false }, axisLabel: { hideOverlap: true, color: "#697386", fontSize: 10 } },
      yAxis: { type: "value", name: panel.unit === "bytes_per_second" ? undefined : unitLabel(panel.unit), axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: "#697386", fontSize: 10, formatter: (value) => axisValueLabel(value, panel.unit) }, splitNumber: 3, splitLine: { lineStyle: { color: "#d8dde7", opacity: .62 } }, ...axisBounds },
      series: series.map((item, index) => ({ type: "line", name: seriesLabel(item), showSymbol: false, connectNulls: false, smooth: .16, areaStyle: index === 0 ? { opacity: .08 } : undefined, step: panel.visualization === "STATE_TIMELINE" ? "end" : false, data: item.points.map((point) => [point.observed_at, numericValue(point.value)]), markLine: index === 0 && (panel.thresholds || panel.capacity_thresholds) ? { silent: true, symbol: "none", label: { show: false }, data: [{ yAxis: panel.thresholds?.warning_value ?? panel.capacity_thresholds?.warning_percent, lineStyle: { color: "#9a6518", type: "dashed" } }, { yAxis: panel.thresholds?.critical_value ?? panel.capacity_thresholds?.critical_percent, lineStyle: { color: "#ad3c45", type: "dashed" } }] } : undefined })),
    };
  }

  function panelTone(panel) {
    if (panel.quality === "NO_DATA") return "none";
    if (panel.visualization === "CAPACITY_BAR") {
      const rows = capacityRows(panel);
      if (rows.some((row) => row.statusTone === "danger")) return "danger";
      if (rows.some((row) => row.statusTone === "warning")) return "warning";
    }
    const current = primarySnapshot(panel)?.current;
    if (current !== null && current !== undefined && panel.thresholds) {
      if (current >= panel.thresholds.critical_value) return "danger";
      if (current >= panel.thresholds.warning_value) return "warning";
    }
    return panel.quality === "PARTIAL" ? "warning" : "normal";
  }
  function primarySnapshot(panel) {
    if (panel.visualization === "CAPACITY_BAR") {
      const row = capacityRows(panel)[0];
      if (!row) return null;
      return { current: row.scope === "LOGICAL" ? row.used : row.utilization, average: row.baselineAverage,
        sampleCount: row.baselineSampleCount, unit: row.scope === "LOGICAL" ? panel.unit : "percent", subject: row.label };
    }
    return chartSeries(panel).flatMap((series) => {
      const current = numericValue(latestPoint(series)?.value);
      return current === null ? [] : [{ current, average: series.baseline?.average ?? null,
        sampleCount: series.baseline?.sample_count || 0, unit: panel.unit, subject: seriesSubject(series) }];
    }).sort((left, right) => right.current - left.current)[0] || null;
  }
  function baselinePresentation(panel) {
    const snapshot = primarySnapshot(panel);
    const average = snapshot?.average;
    const difference = snapshot && average !== null && average !== undefined ? snapshot.current - average : null;
    const differenceText = difference === null ? "无有效基线" : difference === 0 ? "当前与基线持平"
      : `当前${difference > 0 ? "高" : "低"} ${snapshot.unit === "percent" ? `${Math.abs(difference).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} 个百分点` : formatValue(Math.abs(difference), snapshot.unit)}`;
    return { snapshot, difference, differenceText };
  }
  function thresholdPresentation(panel) {
    const thresholds = panel.thresholds;
    const snapshot = primarySnapshot(panel);
    if (!thresholds || snapshot?.current === null || snapshot?.current === undefined) return null;
    const maximum = panel.unit === "percent"
      ? 100 : Math.max(thresholds.critical_value * 1.25, snapshot.current, snapshot.average || 0, 1);
    const position = (value) => `${Math.min(Math.max(value / maximum * 100, 0), 100)}%`;
    return {
      warningAt: position(thresholds.warning_value), criticalAt: position(thresholds.critical_value),
      currentAt: position(snapshot.current), baselineAt: snapshot.average === null || snapshot.average === undefined ? null : position(snapshot.average),
      warningLabel: formatValue(thresholds.warning_value, snapshot.unit),
      criticalLabel: formatValue(thresholds.critical_value, snapshot.unit), maximumLabel: formatValue(maximum, snapshot.unit),
    };
  }
  function conclusionTitle(panel, toneValue, presentation) {
    if (toneValue === "none") return `${panel.title}暂无有效采样`;
    if (toneValue === "normal") return panel.title;
    const { snapshot, difference } = presentation;
    if (!snapshot || difference === null || difference === 0) return `${panel.title}采样不完整`;
    const amount = snapshot.unit === "percent"
      ? `${Math.abs(difference).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} 个百分点`
      : formatValue(Math.abs(difference), snapshot.unit);
    return `${snapshot.subject}${difference > 0 ? "高于" : "低于"}${baselineLabel()}均值 ${amount}`;
  }
  function cardSubtitle(panel, toneValue) {
    if (toneValue === "none") return "当前窗口暂无有效采样";
    if (toneValue === "danger") return "当前值进入紧张区间";
    if (toneValue === "warning") return panel.quality === "PARTIAL" ? "采样不完整，结果仅供参考" : "当前值进入关注区间";
    return "当前值处于正常区间";
  }
  function chartSeries(panel) {
    return panel.series.filter((item) => item.points.some((point) => numericValue(point.value) !== null));
  }
  function tableHtml(panel) {
    const times = [...new Set(panel.series.flatMap((series) => series.points.map((point) => point.observed_at)))].sort();
    const rows = times.length
      ? times.map((time) => `<tr><td>${esc(shell.fmt(time))}</td>${panel.series.map((series) => { const point = series.points.find((item) => item.observed_at === time); return `<td>${point ? `${esc(formatValue(point.value, panel.unit))} · ${esc(pointQualityLabel(point.quality))}` : "—"}</td>`; }).join("")}</tr>`).join("")
      : `<tr><td colspan="${panel.series.length + 1}" class="ops-empty">当前指标没有返回采样数据。</td></tr>`;
    return `<div class="ops-table-wrap"><table class="ops-table"><thead><tr><th>采样时间</th>${panel.series.map((series) => `<th>${esc(seriesLabel(series))}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  function sortedPoints(series) {
    return [...(series.points || [])].sort((left, right) => left.observed_at.localeCompare(right.observed_at));
  }
  function changeLabel(values, unit) {
    if (values.length < 2) return "暂无变化基线";
    if (unit === "state") {
      const before = formatValue(values[0], unit);
      const after = formatValue(values.at(-1), unit);
      return before === after ? `状态保持${after}` : `由${before}变为${after}`;
    }
    const difference = values.at(-1) - values[0];
    if (difference === 0) return "较区间起点持平";
    return `较区间起点${difference > 0 ? "上升" : "下降"} ${formatValue(Math.abs(difference), unit)}`;
  }
  function rangeLabel(values, unit) {
    if (!values.length) return "—";
    if (unit === "state") return `${values.filter((value) => value > 0).length} / ${values.length} 次可用`;
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    return minimum === maximum ? `保持 ${formatValue(minimum, unit)}` : `${formatValue(minimum, unit)} – ${formatValue(maximum, unit)}`;
  }
  function averageLabel(values, unit) {
    if (!values.length) return "无有效采样";
    const average = values.reduce((total, value) => total + value, 0) / values.length;
    return unit === "state" ? `可用率 ${formatValue(average * 100, "percent")}` : formatValue(average, unit);
  }
  function seriesInsightsHtml(panel) {
    return panel.series.map((series) => {
      const points = sortedPoints(series);
      const valid = points.map((point) => ({ point, value: numericValue(point.value) })).filter((item) => item.value !== null);
      const values = valid.map((item) => item.value);
      const latest = valid.at(-1);
      return `<article><header><span>${esc(seriesLabel(series))}</span><strong>${esc(latest ? formatValue(latest.value, panel.unit) : "无有效采样")}</strong><small>当前值 · ${esc(latest ? shell.fmt(latest.point.observed_at) : "—")}</small></header><dl><div><dt>区间均值</dt><dd>${esc(averageLabel(values, panel.unit))}</dd></div><div><dt>波动范围</dt><dd>${esc(rangeLabel(values, panel.unit))}</dd></div><div><dt>整体变化</dt><dd>${esc(changeLabel(values, panel.unit))}</dd></div><div><dt>采样覆盖</dt><dd>${values.length} / ${points.length} 条有效 · 覆盖率 ${(series.coverage_ratio * 100).toFixed(1)} %</dd></div></dl></article>`;
    }).join("");
  }
  function analysisBuckets(points) {
    const runs = [];
    points.forEach((point) => {
      const previous = runs.at(-1);
      const previousPoint = previous?.at(-1);
      if (previousPoint && numericValue(previousPoint.value) === numericValue(point.value) && previousPoint.quality === point.quality) previous.push(point);
      else runs.push([point]);
    });
    if (runs.length <= 12) return runs;
    const bucketSize = Math.max(1, Math.ceil(points.length / 12));
    const buckets = [];
    for (let index = 0; index < points.length; index += bucketSize) buckets.push(points.slice(index, index + bucketSize));
    return buckets;
  }
  function intervalTableHtml(panel) {
    const rows = panel.series.flatMap((series) => {
      const points = sortedPoints(series);
      const buckets = [];
      for (const bucket of analysisBuckets(points)) {
        const values = bucket.map((point) => numericValue(point.value)).filter((value) => value !== null);
        const abnormal = bucket.filter((point) => point.quality !== "GOOD").length;
        const start = shell.fmt(bucket[0].observed_at);
        const end = shell.fmt(bucket.at(-1).observed_at);
        buckets.push(`<tr><td>${esc(start === end ? start : `${start} 至 ${end}`)}</td><td>${esc(seriesLabel(series))}</td><td><strong>${esc(averageLabel(values, panel.unit))}</strong></td><td>${esc(rangeLabel(values, panel.unit))}</td><td>${esc(changeLabel(values, panel.unit))}</td><td>${esc(abnormal ? `${abnormal} 条异常 · ${values.length} / ${bucket.length} 条有效` : `正常 · ${values.length} 条采样`)}</td></tr>`);
      }
      return buckets;
    });
    const body = rows.join("") || '<tr><td colspan="6" class="ops-empty">当前指标没有返回采样数据。</td></tr>';
    const averageHeading = panel.unit === "state" ? "可用情况" : "区间均值";
    const rangeHeading = panel.unit === "state" ? "状态分布" : "波动范围";
    return `<div class="ops-table-wrap"><table class="ops-table"><thead><tr><th>时间区间</th><th>实例与来源</th><th>${averageHeading}</th><th>${rangeHeading}</th><th>区间变化</th><th>采样质量</th></tr></thead><tbody>${body}</tbody></table></div>`;
  }
  function fillDataDrawer(panel) {
    document.getElementById("monitoring-data-drawer-title").textContent = panel.title;
    document.getElementById("monitoring-data-drawer-description").textContent = panel.description;
    document.getElementById("monitoring-data-drawer-quality").innerHTML = badge(panel.quality, qualityLabels[panel.quality]);
    document.getElementById("monitoring-data-drawer-metric").textContent = panel.metric_code;
    document.getElementById("monitoring-data-drawer-unit").textContent = unitLabel(panel.unit) || "无量纲";
    const sources = [...new Set(panel.series.map((series) => series.source_display_name).filter(Boolean))];
    document.getElementById("monitoring-data-drawer-source").textContent = sources.join(" / ") || sourceById()?.display_name || "—";
    document.getElementById("monitoring-data-drawer-insights").innerHTML = seriesInsightsHtml(panel);
    document.getElementById("monitoring-data-drawer-intervals").innerHTML = intervalTableHtml(panel);
    const rawSampleCount = panel.series.reduce((total, series) => total + series.points.length, 0);
    document.getElementById("monitoring-data-drawer-raw-summary").textContent = `查看 ${rawSampleCount} 条原始采样`;
    document.getElementById("monitoring-data-drawer-raw-table").innerHTML = tableHtml(panel);
  }
  function openDataDrawer(panel, trigger) {
    const drawer = document.getElementById("monitoring-data-drawer");
    state.dataDrawerPanelId = panel.panel_id;
    state.dataDrawerTrigger = trigger;
    fillDataDrawer(panel);
    if (!drawer.open) drawer.showModal();
    document.getElementById("close-monitoring-data-drawer").focus();
  }
  function closeDataDrawer() {
    const drawer = document.getElementById("monitoring-data-drawer");
    if (drawer.open) drawer.close();
  }
  function renderPanels() {
    const node = document.getElementById("monitoring-panels");
    disposeCharts();
    const panels = mergeCapacityPanels(state.view.panels || []);
    state.visiblePanels = panels;
    if (!panels.length) {
      closeDataDrawer();
      node.innerHTML = '<div class="ops-panel ops-empty">当前指标范围没有可展示图表。无采样不会显示为 0 或健康。</div>';
      return;
    }
    node.innerHTML = panels.map((panel, index) => {
      const toneValue = panelTone(panel);
      const presentation = baselinePresentation(panel);
      const threshold = thresholdPresentation(panel);
      const snapshot = presentation.snapshot;
      const statusLabel = ({ normal: "正常", warning: "关注", danger: "紧张", none: "无数据" })[toneValue];
      const statusValue = toneValue === "normal" ? "GOOD" : toneValue === "danger" || toneValue === "none" ? "NO_DATA" : "PARTIAL";
      const averageCoverage = panel.series.length
        ? panel.series.reduce((total, series) => total + series.coverage_ratio, 0) / panel.series.length
        : 0;
      const hasData = panel.series.some((series) => series.points.some((point) => numericValue(point.value) !== null));
      const thresholdHtml = threshold ? `<div class="ops-monitoring-threshold-rail" style="--warning-at:${threshold.warningAt};--critical-at:${threshold.criticalAt};--current-at:${threshold.currentAt};--baseline-at:${threshold.baselineAt || "0%"}"><div class="ops-monitoring-threshold-track"><i class="is-${toneValue}"></i>${threshold.baselineAt ? '<b title="所选周期基线"></b>' : ""}</div><div><span>0</span><span>关注 ${esc(threshold.warningLabel)}</span><span>紧张 ${esc(threshold.criticalLabel)}</span><span>${esc(threshold.maximumLabel)}</span></div></div>` : "";
      const heading = conclusionTitle(panel, toneValue, presentation);
      return `<article class="ops-panel ops-monitoring-panel tone-${toneValue} ${panel.visualization === "CAPACITY_BAR" ? "is-capacity-comparison" : ""}"><div class="ops-panel-head"><div><h2 title="${esc(heading)}">${esc(heading)}</h2><p>${esc(cardSubtitle(panel, toneValue))}</p></div><div>${badge(statusValue, statusLabel)}<button type="button" aria-haspopup="dialog" data-panel-id="${esc(panel.panel_id)}">查看数据</button></div></div><p class="ops-monitoring-chart-summary">${esc(panel.title)}，${panel.series.length} 个实例序列，数据质量${esc(qualityLabels[panel.quality])}。</p><div class="ops-monitoring-current-value"><strong>${esc(snapshot ? formatValue(snapshot.current, snapshot.unit) : "—")}</strong><span>${esc(snapshot?.subject || "当前值")}</span></div><div class="ops-monitoring-baseline"><span>${esc(baselineLabel())}均值 <strong>${esc(snapshot?.average === null || snapshot?.average === undefined ? "无有效采样" : formatValue(snapshot.average, snapshot.unit))}</strong></span><b>${esc(presentation.differenceText)}</b></div>${thresholdHtml}${panel.visualization === "CAPACITY_BAR" ? capacityPanelHtml(panel) : hasData ? `<div id="monitoring-chart-${index}" class="ops-monitoring-chart" role="img" aria-label="${esc(panel.title)}趋势图"></div>` : '<div class="ops-empty ops-monitoring-no-data">无有效采样。缺失点保持为空，不按 0 绘制。</div>'}<div class="ops-monitoring-panel-meta"><span>单位 ${esc(unitLabel(panel.unit) || "无量纲")}</span><span>覆盖率 ${(averageCoverage * 100).toFixed(1)} %</span></div></article>`;
    }).join("");
    panels.forEach((panel, index) => {
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
    const drawer = document.getElementById("monitoring-data-drawer");
    if (drawer.open) {
      const activePanel = panels.find((panel) => panel.panel_id === state.dataDrawerPanelId);
      if (activePanel) fillDataDrawer(activePanel); else closeDataDrawer();
    }
  }
  function renderQuality() {
    const gaps = allGaps();
    document.getElementById("monitoring-result-state").innerHTML = badge(state.view.partial ? "PARTIAL" : "GOOD", state.view.partial ? "部分成功" : "查询完成");
    const sourceLabel = [state.view.source, state.view.compare_source].filter(Boolean).map((source) => `${source.display_name} · ${source.source_type}`).join(" / ");
    document.getElementById("monitoring-quality-context").innerHTML = `<dt>监控源</dt><dd>${esc(sourceLabel)}</dd><dt>采样窗口</dt><dd>${esc(shell.fmt(state.view.window.start))} 至 ${esc(shell.fmt(state.view.window.end))}</dd><dt>指标范围</dt><dd>${esc(state.view.profile.display_name)} · ${esc(state.view.profile.version)}</dd><dt>返回实例</dt><dd>${state.view.instances.length}</dd>`;
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
    void selectInstance(instanceId);
  }
  function scheduleRefresh() {
    clearInterval(state.refreshTimer);
    const seconds = Number(document.getElementById("monitoring-refresh-interval").value || 0);
    state.refreshTimer = seconds > 0 ? setInterval(loadView, seconds * 1000) : null;
  }
  async function initialize() {
    state.window = ["15m", "1h", "6h", "24h"].includes(query.get("window")) ? query.get("window") : "1h";
    state.baseline = ["1d", "1w", "1m"].includes(query.get("baseline")) ? query.get("baseline") : "1w";
    document.getElementById("monitoring-window").value = state.window;
    document.getElementById("monitoring-baseline").value = state.baseline;
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
  document.getElementById("monitoring-window").onchange = (event) => { state.window = event.target.value; updateLocation(); scheduleView(); };
  document.getElementById("monitoring-baseline").onchange = (event) => { state.baseline = event.target.value; updateLocation(); scheduleView(); };
  document.getElementById("monitoring-instance-search").oninput = (event) => { state.search = event.target.value; renderInstances(); };
  document.getElementById("monitoring-instance-list").onchange = (event) => {
    const input = event.target.closest("[data-instance-id]");
    if (!input?.checked) return;
    void selectInstance(input.dataset.instanceId);
  };
  document.getElementById("monitoring-panels").onclick = (event) => {
    const capacityRow = event.target.closest("[data-capacity-instance-id]");
    if (capacityRow) {
      drillInto(capacityRow.dataset.capacityInstanceId);
      return;
    }
    const button = event.target.closest("[data-panel-id]");
    if (!button) return;
    const panel = state.visiblePanels.find((item) => item.panel_id === button.dataset.panelId);
    if (panel) openDataDrawer(panel, button);
  };
  document.getElementById("close-monitoring-data-drawer").onclick = closeDataDrawer;
  document.getElementById("finish-monitoring-data-drawer").onclick = closeDataDrawer;
  document.getElementById("monitoring-data-drawer").onclick = (event) => { if (event.target === event.currentTarget) closeDataDrawer(); };
  document.getElementById("monitoring-data-drawer").addEventListener("close", () => {
    const currentTrigger = [...document.querySelectorAll("[data-panel-id]")]
      .find((button) => button.dataset.panelId === state.dataDrawerPanelId);
    const trigger = state.dataDrawerTrigger?.isConnected ? state.dataDrawerTrigger : currentTrigger;
    state.dataDrawerPanelId = "";
    state.dataDrawerTrigger = null;
    trigger?.focus();
  });
  document.getElementById("refresh-monitoring").onclick = loadView;
  document.getElementById("monitoring-rail-toggle").onclick = () => setRailCollapsed(!state.railCollapsed);
  document.getElementById("monitoring-refresh-interval").onchange = scheduleRefresh;
  addEventListener("resize", () => state.charts.forEach((chart) => chart.resize()));
  shell.ready.then(initialize);
})();
