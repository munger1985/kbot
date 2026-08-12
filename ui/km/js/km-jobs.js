(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  let rows = [], sources = [];
  async function initialize() {
    try { sources = KBotKmApi.items(await KBotKmApi.request(`${base}/sources`)); $("job-source").insertAdjacentHTML("beforeend", sources.map((row) => `<option value="${KBotKmShell.escapeHtml(row.source_id)}">${KBotKmShell.escapeHtml(row.display_name)}</option>`).join("")); await load(); }
    catch (error) { KBotKmShell.showError(error, "任务页面加载失败"); }
  }
  function sourceName(id) { return sources.find((row) => String(row.source_id) === String(id))?.display_name || KBotKmShell.shortId(id); }
  async function load() {
    try { rows = KBotKmApi.items(await KBotKmApi.request(`${base}/jobs${KBotKmApi.query({ source_id: $("job-source").value, limit: $("job-limit").value })}`)); render(); }
    catch (error) { KBotKmShell.showError(error, "同步任务查询失败"); }
  }
  function render() {
    const body = $("job-rows");
    if (!rows.length) return KBotKmShell.renderEmpty(body, 7, "暂无同步任务");
    body.innerHTML = rows.map((row, index) => `<tr><td><span class="km-cell-main">${KBotKmShell.escapeHtml(row.job_type)}</span><div class="km-cell-sub">${KBotKmShell.escapeHtml(KBotKmShell.shortId(row.job_id))}</div></td><td>${KBotKmShell.escapeHtml(sourceName(row.source_id))}</td><td>${KBotKmShell.badge(row.status)}</td><td>${row.km_asset_id ? KBotKmShell.escapeHtml(KBotKmShell.shortId(row.km_asset_id)) : "来源级任务"}</td><td>${KBotKmShell.escapeHtml(`${row.attempt_count || 0} / ${row.max_attempts || "—"}`)}${row.error_message ? `<div class="km-cell-sub">${KBotKmShell.escapeHtml(row.error_message)}</div>` : ""}</td><td>${KBotKmShell.escapeHtml(KBotKmShell.formatDate(row.created_at))}<div class="km-cell-sub">${KBotKmShell.escapeHtml(KBotKmShell.formatDate(row.completed_at))}</div></td><td><button class="small" data-view="${index}">详情</button></td></tr>`).join("");
  }
  window.addEventListener("DOMContentLoaded", () => {
    $("job-form").addEventListener("submit", (event) => { event.preventDefault(); load(); });
    $("refresh-jobs").addEventListener("click", load);
    $("job-rows").addEventListener("click", (event) => { const button = event.target.closest("[data-view]"); if (!button) return; $("job-detail-json").textContent = JSON.stringify(rows[Number(button.dataset.view)], null, 2); KBotKmShell.openDialog("job-detail-dialog"); });
  });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
