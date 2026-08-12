(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  let rows = [];

  async function loadSources() {
    const sources = KBotKmApi.items(await KBotKmApi.request(`${base}/sources`));
    $("metadb-source").innerHTML = sources.length
      ? sources.map((row) => `<option value="${KBotKmShell.escapeHtml(row.source_id)}">${KBotKmShell.escapeHtml(row.display_name)}</option>`).join("")
      : '<option value="">尚未配置来源</option>';
    if (sources.length) await loadRows();
  }
  async function loadRows() {
    const source = $("metadb-source").value;
    if (!source) return;
    const offset = Number($("metadb-offset").value || 0);
    const limit = Number($("metadb-limit").value);
    try {
      const payload = await KBotKmApi.request(`${base}/sources/${encodeURIComponent(source)}/metadb/assets${KBotKmApi.query({ processed: $("metadb-processed").value, offset, limit })}`);
      rows = KBotKmApi.items(payload);
      $("metadb-status").textContent = `偏移 ${offset}，本页 ${rows.length} 条`;
      render();
    } catch (error) { KBotKmShell.showError(error, "MetaDB 查询失败"); }
  }
  function field(row, ...keys) { return keys.map((key) => row[key]).find((value) => value !== undefined && value !== null && value !== "") ?? "—"; }
  function render() {
    const body = $("metadb-rows");
    if (!rows.length) return KBotKmShell.renderEmpty(body, 7, "当前筛选条件下没有原始记录");
    body.innerHTML = rows.map((row, index) => `<tr><td><code>${KBotKmShell.escapeHtml(field(row, "asset_id", "id"))}</code></td><td>${KBotKmShell.escapeHtml(field(row, "asset_title", "title"))}</td><td>${KBotKmShell.escapeHtml(field(row, "author_mail", "author", "created_by"))}</td><td>${KBotKmShell.escapeHtml([field(row, "asset_solution", "content_category"), field(row, "asset_product", "product")].filter((value) => value !== "—").join(" · ") || "—")}</td><td>${KBotKmShell.badge(field(row, "processed"))}</td><td>${KBotKmShell.escapeHtml(KBotKmShell.formatDate(field(row, "last_update_time", "updated_at")))}</td><td><button class="small" data-view="${index}">原始数据</button>${String(field(row, "processed")).toUpperCase() === "F" ? ` <button class="small" data-retry="${index}">重新抽取</button>` : ""}</td></tr>`).join("");
  }
  async function retry(index, button) {
    const source = $("metadb-source").value;
    const id = field(rows[index], "asset_id", "id");
    if (!source || id === "—") return KBotKmShell.toast("该记录缺少 asset_id", "error");
    KBotKmShell.setBusy(button, true);
    try {
      await KBotKmApi.json(`${base}/sources/${encodeURIComponent(source)}/metadb/assets/${encodeURIComponent(id)}/retry`, "POST");
      KBotKmShell.toast("已提交重新抽取任务", "success");
      await loadRows();
    } catch (error) { KBotKmShell.showError(error, "重新抽取提交失败"); }
    finally { KBotKmShell.setBusy(button, false); }
  }
  window.addEventListener("DOMContentLoaded", () => {
    $("metadb-form").addEventListener("submit", (event) => { event.preventDefault(); $("metadb-offset").value = "0"; loadRows(); });
    $("metadb-prev").addEventListener("click", () => { $("metadb-offset").value = String(Math.max(0, Number($("metadb-offset").value) - Number($("metadb-limit").value))); loadRows(); });
    $("metadb-next").addEventListener("click", () => { if (rows.length < Number($("metadb-limit").value)) return; $("metadb-offset").value = String(Number($("metadb-offset").value) + Number($("metadb-limit").value)); loadRows(); });
    $("metadb-rows").addEventListener("click", (event) => { const view = event.target.closest("[data-view]"); const again = event.target.closest("[data-retry]"); if (view) { $("metadb-json").textContent = JSON.stringify(rows[Number(view.dataset.view)], null, 2); KBotKmShell.openDialog("metadb-detail-dialog"); } if (again) retry(Number(again.dataset.retry), again); });
  });
  KBotKmShell.ready.then(loadSources).catch(() => {});
})();
