(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const batchLimit = 100;
  const $ = (id) => document.getElementById(id);
  let rows = [];
  let sources = [];
  let statusRefreshTimer = null;
  const selectedAssetIds = new Set();

  async function initialize() {
    try {
      sources = KBotKmApi.items(await KBotKmApi.request(`${base}/sources`));
      $("asset-source").insertAdjacentHTML("beforeend", sources.map((row) => `<option value="${KBotKmShell.escapeHtml(row.source_id)}">${KBotKmShell.escapeHtml(row.display_name)}</option>`).join(""));
      await load();
    } catch (error) { KBotKmShell.showError(error, "Asset 页面加载失败"); }
  }
  function sourceName(id) { return sources.find((row) => String(row.source_id) === String(id))?.display_name || KBotKmShell.shortId(id); }
  async function load(options = {}) {
    const offset = Number($("asset-offset").value || 0);
    const limit = Number($("asset-limit").value);
    try {
      rows = KBotKmApi.items(await KBotKmApi.request(`${base}/assets${KBotKmApi.query({ source_id: $("asset-source").value, ingestion_status: $("asset-status").value, offset, limit })}`));
      if (!options.preserveSelection) selectedAssetIds.clear();
      $("asset-status-text").textContent = `偏移 ${offset}，本页 ${rows.length} 条`;
      render();
    } catch (error) { KBotKmShell.showError(error, "Asset 查询失败"); }
  }
  function requiresIndexRecovery(row) {
    return row.ingestion_status === "KC_ACCEPTED"
      || (row.ingestion_status === "FAILED" && row.failure_stage === "KC_STATUS_SYNC");
  }
  function canReindex(row) {
    const recoverable = requiresIndexRecovery(row);
    const stateEligible = row.ingestion_status === "READY" || recoverable;
    const reindexActive = ["PENDING", "RUNNING", "RETRY_WAIT"].includes(row.reindex?.status);
    return stateEligible && Boolean(row.kc_bundle_revision_id) && !reindexActive;
  }
  function reindexLabel(row) { return row.ingestion_status === "READY" ? "重新索引" : "恢复索引"; }
  function render() {
    const body = $("asset-rows");
    if (!rows.length) { if (statusRefreshTimer) clearTimeout(statusRefreshTimer); updateSelection(); return KBotKmShell.renderEmpty(body, 8, "当前筛选条件下没有 Asset"); }
    body.innerHTML = rows.map((row, index) => {
      const retryable = !requiresIndexRecovery(row)
        && (["FAILED", "DOWNLOAD_FAILED"].includes(row.ingestion_status) || row.source_status === "F");
      const reindexActive = ["PENDING", "RUNNING", "RETRY_WAIT"].includes(row.reindex?.status);
      const reindexable = canReindex(row);
      const checked = selectedAssetIds.has(String(row.km_asset_id)) ? " checked" : "";
      const reindexText = { PENDING: "等待中", RUNNING: "处理中", RETRY_WAIT: "处理中", SUCCEEDED: "已完成", FAILED: "失败" }[row.reindex?.status] || "";
      const reindexStatus = row.reindex ? `<div class="km-cell-sub" title="${KBotKmShell.escapeHtml(row.reindex.error_message || "")}">重新索引：${KBotKmShell.escapeHtml(reindexText)}</div>` : "";
      return `<tr><td class="km-select-cell"><input type="checkbox" data-select="${index}" aria-label="选择 ${KBotKmShell.escapeHtml(row.asset_title || row.external_asset_id)}"${reindexable ? checked : " disabled"}></td><td><span class="km-cell-main">${KBotKmShell.escapeHtml(row.external_asset_id)}</span><div class="km-cell-sub">${KBotKmShell.escapeHtml(sourceName(row.source_id))}</div></td><td><span class="km-cell-main">${KBotKmShell.escapeHtml(row.asset_title || "未命名 Asset")}</span><div class="km-cell-sub">${KBotKmShell.escapeHtml(row.author_mail || "—")}</div></td><td>${KBotKmShell.escapeHtml([row.asset_solution, row.asset_product, row.content_category].filter(Boolean).join(" · ") || "—")}</td><td>${KBotKmShell.badge(row.ingestion_status)}<div class="km-cell-sub">源 ${KBotKmShell.escapeHtml(row.source_status || "—")}</div>${reindexStatus}</td><td>${row.kc_bundle_id ? `<code>${KBotKmShell.escapeHtml(KBotKmShell.shortId(row.kc_bundle_id))}</code>` : "—"}</td><td>${KBotKmShell.escapeHtml(KBotKmShell.formatDate(row.completed_at || row.last_update_time))}</td><td><button class="small" data-view="${index}">详情</button>${retryable ? ` <button class="small" data-retry="${index}">重试</button>` : ""}${reindexable ? ` <button class="small" data-reindex="${index}">${reindexLabel(row)}</button>` : ""}</td></tr>`;
    }).join("");
    updateSelection();
    scheduleStatusRefresh();
  }
  function reindexableRows() { return rows.filter(canReindex); }
  function scheduleStatusRefresh() {
    if (statusRefreshTimer) clearTimeout(statusRefreshTimer);
    if (!rows.some((row) => ["PENDING", "RUNNING", "RETRY_WAIT"].includes(row.reindex?.status))) return;
    statusRefreshTimer = setTimeout(() => load({ preserveSelection: true }), 5000);
  }
  function updateSelection() {
    const eligible = reindexableRows();
    const selectedCount = eligible.filter((row) => selectedAssetIds.has(String(row.km_asset_id))).length;
    $("asset-selection-text").textContent = `已选择 ${selectedCount} 条（最多 ${batchLimit} 条）`;
    $("asset-bulk-reindex").disabled = selectedCount === 0;
    $("asset-select-all").disabled = eligible.length === 0;
    const selectableCount = Math.min(eligible.length, batchLimit);
    $("asset-select-all").checked = selectableCount > 0 && selectedCount === selectableCount;
    $("asset-select-all").indeterminate = selectedCount > 0 && selectedCount < selectableCount;
  }
  async function detail(index) {
    try { const payload = await KBotKmApi.request(`${base}/assets/${rows[index].km_asset_id}`); $("asset-detail-json").textContent = JSON.stringify(payload, null, 2); KBotKmShell.openDialog("asset-detail-dialog"); }
    catch (error) { KBotKmShell.showError(error, "Asset 详情读取失败"); }
  }
  async function retry(index, button) {
    const row = rows[index]; KBotKmShell.setBusy(button, true);
    try { await KBotKmApi.json(`${base}/assets/${row.km_asset_id}/retry`, "POST", { expected_row_version: row.row_version }); KBotKmShell.toast("Asset 重试任务已提交", "success"); await load(); }
    catch (error) { KBotKmShell.showError(error, "Asset 重试失败"); }
    finally { KBotKmShell.setBusy(button, false); }
  }
  async function reindex(index, button) {
    const row = rows[index]; KBotKmShell.setBusy(button, true, "提交中…");
    try { const result = await KBotKmApi.json(`${base}/assets/${row.km_asset_id}/reindex`, "POST", { expected_row_version: row.row_version }); KBotKmShell.toast(result.tracking_status === "PENDING" ? "Asset 全文与向量重新索引任务已提交" : "重新索引已提交，但状态跟踪暂不可用", result.tracking_status === "PENDING" ? "success" : "warning"); await load(); }
    catch (error) { KBotKmShell.showError(error, "Asset 重新索引失败"); }
    finally { KBotKmShell.setBusy(button, false); }
  }
  async function batchReindex() {
    const selected = reindexableRows().filter((row) => selectedAssetIds.has(String(row.km_asset_id)));
    if (!selected.length || !window.confirm(`确认提交 ${selected.length} 个 Asset 的全文与向量重新索引任务？`)) return;
    const button = $("asset-bulk-reindex"); KBotKmShell.setBusy(button, true, "批量提交中…");
    try {
      const result = await KBotKmApi.json(`${base}/assets/actions/reindex`, "POST", { items: selected.map((row) => ({ km_asset_id: row.km_asset_id, expected_row_version: row.row_version })) });
      if (result.failed_count || result.untracked_count) {
        $("asset-detail-json").textContent = JSON.stringify(result, null, 2);
        KBotKmShell.openDialog("asset-detail-dialog");
        KBotKmShell.toast(`已提交 ${result.submitted_count} 条，失败 ${result.failed_count} 条，未跟踪 ${result.untracked_count || 0} 条`, "warning");
      } else {
        KBotKmShell.toast(`已提交 ${result.submitted_count} 个 Asset 的重新索引任务`, "success");
      }
      await load();
    } catch (error) { KBotKmShell.showError(error, "Asset 批量重新索引失败"); }
    finally { KBotKmShell.setBusy(button, false); updateSelection(); }
  }
  window.addEventListener("DOMContentLoaded", () => {
    $("asset-form").addEventListener("submit", (event) => { event.preventDefault(); $("asset-offset").value = "0"; load(); });
    $("asset-prev").addEventListener("click", () => { $("asset-offset").value = String(Math.max(0, Number($("asset-offset").value) - Number($("asset-limit").value))); load(); });
    $("asset-next").addEventListener("click", () => { if (rows.length < Number($("asset-limit").value)) return; $("asset-offset").value = String(Number($("asset-offset").value) + Number($("asset-limit").value)); load(); });
    $("asset-select-all").addEventListener("change", (event) => { selectedAssetIds.clear(); if (event.target.checked) reindexableRows().slice(0, batchLimit).forEach((row) => selectedAssetIds.add(String(row.km_asset_id))); render(); });
    $("asset-bulk-reindex").addEventListener("click", batchReindex);
    $("asset-rows").addEventListener("change", (event) => { const checkbox = event.target.closest("[data-select]"); if (!checkbox) return; const row = rows[Number(checkbox.dataset.select)]; const id = String(row.km_asset_id); if (checkbox.checked && selectedAssetIds.size >= batchLimit) { checkbox.checked = false; KBotKmShell.toast(`每批最多选择 ${batchLimit} 个 Asset`, "warning"); return; } if (checkbox.checked) selectedAssetIds.add(id); else selectedAssetIds.delete(id); updateSelection(); });
    $("asset-rows").addEventListener("click", (event) => { const view = event.target.closest("[data-view]"); const again = event.target.closest("[data-retry]"); const reindexButton = event.target.closest("[data-reindex]"); if (view) detail(Number(view.dataset.view)); if (again) retry(Number(again.dataset.retry), again); if (reindexButton) reindex(Number(reindexButton.dataset.reindex), reindexButton); });
  });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
