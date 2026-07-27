/* KBot development 环境本地日志浏览页面。 */
(function () {
  "use strict";

  const authForm = document.querySelector("#auth-form");
  const filterForm = document.querySelector("#log-filter-form");
  const serviceFilter = document.querySelector("#service-filter");
  const logTypeFilter = document.querySelector("#log-type-filter");
  const intervalSelect = document.querySelector("#refresh-interval");
  const rowsElement = document.querySelector("#log-rows");
  const statusElement = document.querySelector("#log-status");
  const detailElement = document.querySelector("#event-detail");
  const metaElement = document.querySelector("#event-meta");
  let eventsById = new Map();
  let refreshTimer = null;
  let requestRunning = false;

  function checkedValues(name) {
    return Array.from(
      filterForm.querySelectorAll(`input[name="${name}"]:checked`)
    ).map((input) => input.value);
  }

  function setLevels(values) {
    const enabled = new Set(values);
    for (const input of filterForm.querySelectorAll('input[name="level"]')) {
      input.checked = enabled.has(input.value);
    }
  }

  function queryPath() {
    const params = new URLSearchParams();
    params.set("service_name", serviceFilter.value);
    params.set("log_type", logTypeFilter.value);
    const keyword = filterForm.elements.keyword.value.trim();
    if (keyword) params.set("keyword", keyword);
    params.set("limit", filterForm.elements.limit.value);
    for (const value of checkedValues("level")) {
      params.append("level", value);
    }
    return `/api/v1/development/logs/events?${params.toString()}`;
  }

  function formatTime(value) {
    if (!value) return "-";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
  }

  function renderEvents(events) {
    eventsById = new Map(events.map((event) => [event.event_id, event]));
    if (!events.length) {
      rowsElement.innerHTML =
        '<tr><td colspan="5" class="muted">当前服务的这类日志为空。</td></tr>';
      return;
    }
    rowsElement.innerHTML = events
      .map(
        (event) => `
          <tr data-event-id="${KBotUI.escapeHtml(event.event_id)}" tabindex="0">
            <td class="log-time">${KBotUI.escapeHtml(formatTime(event.timestamp))}</td>
            <td><span class="log-source">${KBotUI.escapeHtml(event.process || "-")}</span></td>
            <td><span class="log-level level-${KBotUI.escapeHtml(event.level.toLowerCase())}">${KBotUI.escapeHtml(event.level)}</span></td>
            <td class="log-location" title="${KBotUI.escapeHtml(event.location)}">${KBotUI.escapeHtml(event.location || "-")}</td>
            <td class="log-message">${KBotUI.escapeHtml(event.message)}</td>
          </tr>`
      )
      .join("");
  }

  function showEvent(eventId) {
    const event = eventsById.get(eventId);
    if (!event) return;
    metaElement.textContent = [
      formatTime(event.timestamp),
      event.service_name,
      event.log_type,
      event.process,
      event.level,
      event.source_file,
      event.error_id ? `error_id=${event.error_id}` : "",
      event.request_id ? `request_id=${event.request_id}` : "",
      event.trace_id ? `trace_id=${event.trace_id}` : "",
    ]
      .filter(Boolean)
      .join(" · ");
    detailElement.textContent = event.raw || event.message;
  }

  async function loadServices() {
    const payload = await KBotUI.api(
      "/api/v1/development/logs/services",
      { domainOptional: true }
    );
    const services = payload.services || [];
    const previous = serviceFilter.value;
    serviceFilter.innerHTML = services
      .map((service) => {
        const runtime = service.runtime ? "运行" : "-";
        const access = service.access ? "访问" : "-";
        return `<option value="${KBotUI.escapeHtml(service.service_name)}">${KBotUI.escapeHtml(service.service_name)} · ${runtime}/${access}</option>`;
      })
      .join("");
    if (services.some((item) => item.service_name === previous)) {
      serviceFilter.value = previous;
    }
  }

  async function refreshLogs(options) {
    if (requestRunning) return;
    if (!serviceFilter.value) {
      renderEvents([]);
      KBotUI.setStatus(statusElement, "暂无可读取的服务日志。");
      return;
    }
    requestRunning = true;
    if (!options?.quiet) {
      KBotUI.setStatus(statusElement, "正在读取日志…");
    }
    try {
      const payload = await KBotUI.api(queryPath(), {
        domainOptional: true,
      });
      renderEvents(payload.events || []);
      KBotUI.setStatus(
        statusElement,
        `已显示 ${payload.count || 0} 条日志 · ${new Date().toLocaleTimeString()}`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus(statusElement, error.message, "error");
    } finally {
      requestRunning = false;
    }
  }

  function resetTimer() {
    if (refreshTimer) window.clearInterval(refreshTimer);
    refreshTimer = null;
    const interval = Number(intervalSelect.value);
    if (interval > 0) {
      refreshTimer = window.setInterval(
        () => refreshLogs({ quiet: true }),
        interval
      );
    }
  }

  KBotUI.bindAuthForm(authForm, async () => {
    try {
      await loadServices();
      await refreshLogs();
    } catch (error) {
      KBotUI.setStatus(statusElement, error.message, "error");
    }
  });

  filterForm.addEventListener("submit", (event) => {
    event.preventDefault();
    refreshLogs();
  });
  intervalSelect.addEventListener("change", resetTimer);
  serviceFilter.addEventListener("change", () => refreshLogs());
  logTypeFilter.addEventListener("change", () => refreshLogs());
  document.querySelector("#errors-only").addEventListener("click", () => {
    setLevels(["ERROR", "CRITICAL"]);
    refreshLogs();
  });
  document.querySelector("#all-levels").addEventListener("click", () => {
    setLevels(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]);
    refreshLogs();
  });
  rowsElement.addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-event-id]");
    if (row) showEvent(row.dataset.eventId);
  });
  rowsElement.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const row = event.target.closest("tr[data-event-id]");
    if (row) showEvent(row.dataset.eventId);
  });

  resetTimer();
  loadServices()
    .then(() => refreshLogs())
    .catch((error) => KBotUI.setStatus(statusElement, error.message, "error"));
})();
