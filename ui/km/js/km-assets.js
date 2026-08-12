(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  let rows = [];
  let sources = [];

  async function initialize() {
    try {
      sources = KBotKmApi.items(await KBotKmApi.request(`${base}/sources`));
      $("asset-source").insertAdjacentHTML("beforeend", sources.map((row) => `<option value="${KBotKmShell.escapeHtml(row.source_id)}">${KBotKmShell.escapeHtml(row.display_name)}</option>`).join(""));
      await load();
    } catch (error) { KBotKmShell.showError(error, "Asset 页面加载失败"); }
  }
  function sourceName(id) { return sources.find((row) => String(row.source_id) === String(id))?.display_name || KBotKmShell.shortId(id); }
  async function load() {
    const offset = Number($("asset-offset").value || 0);
    const limit = Number($("asset-limit").value);
    try {
      rows = KBotKmApi.items(await KBotKmApi.request(`${base}/assets${KBotKmApi.query({ source_id: $("asset-source").value, ingestion_status: $("asset-status").value, offset, limit })}`));
      $("asset-status-text").textContent = `偏移 ${offset}，本页 ${rows.length} 条`;
      render();
    } catch (error) { KBotKmShell.showError(error, "Asset 查询失败"); }
  }
  function render() {
    const body = $("asset-rows");
    if (!rows.length) return KBotKmShell.renderEmpty(body, 7, "当前筛选条件下没有 Asset");
    body.innerHTML = rows.map((row, index) => {
      const retryable = ["FAILED", "DOWNLOAD_FAILED"].includes(row.ingestion_status) || row.source_status === "F";
      return `<tr><td><span class="km-cell-main">${KBotKmShell.escapeHtml(row.external_asset_id)}</span><div class="km-cell-sub">${KBotKmShell.escapeHtml(sourceName(row.source_id))}</div></td><td><span class="km-cell-main">${KBotKmShell.escapeHtml(row.asset_title || "未命名 Asset")}</span><div class="km-cell-sub">${KBotKmShell.escapeHtml(row.author_mail || "—")}</div></td><td>${KBotKmShell.escapeHtml([row.asset_solution, row.asset_product, row.content_category].filter(Boolean).join(" · ") || "—")}</td><td>${KBotKmShell.badge(row.ingestion_status)}<div class="km-cell-sub">源 ${KBotKmShell.escapeHtml(row.source_status || "—")}</div></td><td>${row.kc_bundle_id ? `<code>${KBotKmShell.escapeHtml(KBotKmShell.shortId(row.kc_bundle_id))}</code>` : "—"}</td><td>${KBotKmShell.escapeHtml(KBotKmShell.formatDate(row.completed_at || row.last_update_time))}</td><td><button class="small" data-view="${index}">详情</button>${retryable ? ` <button class="small" data-retry="${index}">重试</button>` : ""}</td></tr>`;
    }).join("");
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
  window.addEventListener("DOMContentLoaded", () => {
    $("asset-form").addEventListener("submit", (event) => { event.preventDefault(); $("asset-offset").value = "0"; load(); });
    $("asset-prev").addEventListener("click", () => { $("asset-offset").value = String(Math.max(0, Number($("asset-offset").value) - Number($("asset-limit").value))); load(); });
    $("asset-next").addEventListener("click", () => { if (rows.length < Number($("asset-limit").value)) return; $("asset-offset").value = String(Number($("asset-offset").value) + Number($("asset-limit").value)); load(); });
    $("asset-rows").addEventListener("click", (event) => { const view = event.target.closest("[data-view]"); const again = event.target.closest("[data-retry]"); if (view) detail(Number(view.dataset.view)); if (again) retry(Number(again.dataset.retry), again); });
  });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
