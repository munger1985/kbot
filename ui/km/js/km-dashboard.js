(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  const { badge, escapeHtml: e, formatDate, renderEmpty, showError } = KBotKmShell;

  async function load() {
    const button = $("refresh-dashboard");
    KBotKmShell.setBusy(button, true, "刷新中…");
    try {
      const [sourcesPayload, assetsPayload, jobsPayload] = await Promise.all([
        KBotKmApi.request(`${base}/sources`),
        KBotKmApi.request(`${base}/assets?limit=100`),
        KBotKmApi.request(`${base}/jobs?limit=100`),
      ]);
      const sources = KBotKmApi.items(sourcesPayload);
      const assets = KBotKmApi.items(assetsPayload);
      const jobs = KBotKmApi.items(jobsPayload);
      const failures = assets.filter((row) => ["FAILED", "DOWNLOAD_FAILED"].includes(row.ingestion_status) || row.source_status === "F");
      $("metric-sources").textContent = sources.length;
      $("metric-ready").textContent = assets.filter((row) => row.ingestion_status === "READY").length;
      $("metric-failed").textContent = failures.length;
      $("metric-jobs").textContent = jobs.length;
      renderSources(sources);
      renderFailures(failures.slice(0, 8), sources);
      renderJobs(jobs.slice(0, 10), sources);
    } catch (error) { showError(error, "工作台加载失败"); }
    finally { KBotKmShell.setBusy(button, false); }
  }

  function sourceName(id, sources) { return sources.find((row) => String(row.source_id) === String(id))?.display_name || KBotKmShell.shortId(id); }
  function renderSources(rows) {
    const body = $("dashboard-source-rows");
    if (!rows.length) return renderEmpty(body, 4, "尚未配置数据来源");
    body.innerHTML = rows.slice(0, 8).map((row) => `<tr><td><span class="km-cell-main">${e(row.display_name)}</span><div class="km-cell-sub">${e(KBotKmShell.shortId(row.source_id))}</div></td><td>${badge(row.status)} ${badge(row.model_status)}</td><td>${e(row.catalog_hash ? String(row.catalog_hash).slice(0, 12) : "—")}</td><td>${e(KBotKmShell.formatDate(row.last_sync_at))}</td></tr>`).join("");
  }
  function renderFailures(rows, sources) {
    const body = $("dashboard-failure-rows");
    if (!rows.length) return renderEmpty(body, 4, "当前加载范围内没有失败 Asset");
    body.innerHTML = rows.map((row) => `<tr><td><span class="km-cell-main">${e(row.asset_title || row.external_asset_id)}</span><div class="km-cell-sub">${e(KBotKmShell.shortId(row.km_asset_id))}</div></td><td>${e(sourceName(row.source_id, sources))}</td><td>${e(row.error_message || row.error_code || row.failure_stage || "未知错误")}</td><td>${e(formatDate(row.completed_at || row.last_update_time))}</td></tr>`).join("");
  }
  function renderJobs(rows, sources) {
    const body = $("dashboard-job-rows");
    if (!rows.length) return renderEmpty(body, 5, "暂无同步任务");
    body.innerHTML = rows.map((row) => `<tr><td><span class="km-cell-main">${e(row.job_type)}</span><div class="km-cell-sub">${e(KBotKmShell.shortId(row.job_id))}</div></td><td>${e(sourceName(row.source_id, sources))}</td><td>${badge(row.status)}</td><td>${e(`尝试 ${row.attempt_count || 0}/${row.max_attempts || "—"}`)}</td><td>${e(formatDate(row.created_at))}</td></tr>`).join("");
  }
  window.addEventListener("DOMContentLoaded", () => $("refresh-dashboard").addEventListener("click", load));
  KBotKmShell.ready.then(load).catch(() => {});
})();
