(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  let rows = [];
  let modelSource = null;

  async function load() {
    try {
      rows = KBotKmApi.items(await KBotKmApi.request(`${base}/sources`));
      render();
      return rows;
    } catch (error) {
      KBotKmShell.showError(error, "数据来源加载失败");
      return null;
    }
  }

  function isVersionConflict(error) {
    return error?.status === 409 && error?.code === "ROW_VERSION_CONFLICT";
  }

  async function refreshConflict(sourceId, form) {
    const refreshed = await load();
    if (!refreshed) return false;
    const current = refreshed.find((row) => String(row.source_id) === String(sourceId));
    if (!current) {
      KBotKmShell.toast("数据来源已不存在，请关闭编辑窗口后重新选择", "error");
      return true;
    }
    if (form) form.elements.expected_row_version.value = current.row_version;
    KBotKmShell.toast(
      form
        ? `来源配置已变化，版本已刷新为 ${current.row_version}；已保留当前输入，请核对后再次保存`
        : `来源配置已变化，列表已刷新到版本 ${current.row_version}，请重试操作`,
      "warning"
    );
    return true;
  }
  function render() {
    const body = $("source-rows");
    if (!rows.length) return KBotKmShell.renderEmpty(body, 7, "尚未配置 KM 数据来源");
    body.innerHTML = rows.map((row, index) => `<tr><td><span class="km-cell-main">${KBotKmShell.escapeHtml(row.display_name)}</span><div class="km-cell-sub">${KBotKmShell.escapeHtml(KBotKmShell.shortId(row.source_id))}</div></td><td>${KBotKmShell.escapeHtml(row.metadb_endpoint)}</td><td><code>${KBotKmShell.escapeHtml(KBotKmShell.shortId(row.collection_id))}</code></td><td>${KBotKmShell.badge(row.status)} ${KBotKmShell.badge(row.model_status)}</td><td>${KBotKmShell.badge(row.auto_sync_enabled ? "AUTO_ON" : "AUTO_OFF")}<div class="km-cell-sub">${KBotKmShell.escapeHtml(`${row.poll_interval_seconds}s / ${row.batch_size}条`)}</div></td><td>${row.row_version}</td><td><button class="small" data-edit="${index}">修改</button> <button class="small" data-model="${index}">数据模型</button>${row.status === "DRAFT" ? ` <button class="small" data-activate="${index}">激活</button>` : ` <button class="small" data-auto="${index}">${row.auto_sync_enabled ? "关闭后台同步" : "开启后台同步"}</button> <button class="small" data-sync="${index}">立即同步</button>`}</td></tr>`).join("");
  }
  async function action(index, kind, button) {
    const row = rows[index];
    KBotKmShell.setBusy(button, true);
    try {
      if (kind === "activate") await KBotKmApi.json(`${base}/sources/${row.source_id}/activate`, "POST", { expected_row_version: row.row_version });
      if (kind === "sync") await KBotKmApi.json(`${base}/sources/${row.source_id}/sync`, "POST");
      if (kind === "auto") await KBotKmApi.json(`${base}/sources/${row.source_id}`, "PATCH", { expected_row_version: row.row_version, auto_sync_enabled: !row.auto_sync_enabled });
      const message = kind === "activate" ? "数据来源已激活" : kind === "sync" ? "同步任务已提交" : row.auto_sync_enabled ? "后台自动同步已关闭" : "后台自动同步已开启";
      KBotKmShell.toast(message, "success");
      await load();
    } catch (error) {
      if (!(isVersionConflict(error) && await refreshConflict(row.source_id))) {
        KBotKmShell.showError(error);
      }
    }
    finally { KBotKmShell.setBusy(button, false); }
  }
  async function showModel(index) {
    modelSource = rows[index];
    try {
      const payload = await KBotKmApi.request(`${base}/sources/${modelSource.source_id}/data-model`);
      $("data-model-json").textContent = JSON.stringify(payload, null, 2);
      KBotKmShell.openDialog("data-model-dialog");
    } catch (error) { KBotKmShell.showError(error, "数据模型读取失败"); }
  }
  async function save(event) {
    event.preventDefault();
    const form = event.target;
    const values = Object.fromEntries(new FormData(form));
    const payload = {
      display_name: values.display_name, metadb_endpoint: values.metadb_endpoint,
      metadb_credentials: { username: values.metadb_username, password: values.metadb_password },
      sharepoint_credentials: { tenant_id: values.tenant_id, client_id: values.client_id, client_secret: values.client_secret },
      sharepoint_site_path: values.sharepoint_site_path,
      poll_interval_seconds: Number(values.poll_interval_seconds), batch_size: Number(values.batch_size),
    };
    KBotKmShell.setBusy($("save-source"), true, "创建中…");
    try { await KBotKmApi.json(`${base}/sources`, "POST", payload); form.reset(); KBotKmShell.closeDialog("source-dialog"); KBotKmShell.toast("数据来源已创建", "success"); await load(); }
    catch (error) { KBotKmShell.showError(error, "数据来源创建失败"); }
    finally { KBotKmShell.setBusy($("save-source"), false); }
  }
  function openEdit(index) {
    const row = rows[index]; const form = $("source-edit-form");
    form.reset();
    for (const name of ["source_id", "expected_row_version", "display_name", "metadb_endpoint", "sharepoint_site_path", "poll_interval_seconds", "batch_size"]) {
      form.elements[name].value = name === "expected_row_version" ? row.row_version : row[name];
    }
    KBotKmShell.openDialog("source-edit-dialog");
  }
  async function update(event) {
    event.preventDefault(); const form = event.target; const values = Object.fromEntries(new FormData(form));
    const metadbPair = [values.metadb_username, values.metadb_password];
    const sharepointGroup = [values.tenant_id, values.client_id, values.client_secret];
    if (metadbPair.some(Boolean) && !metadbPair.every(Boolean)) return KBotKmShell.toast("MetaDB 用户名和密码必须同时填写", "error");
    if (sharepointGroup.some(Boolean) && !sharepointGroup.every(Boolean)) return KBotKmShell.toast("SharePoint 三项凭据必须完整填写", "error");
    const payload = { expected_row_version: Number(values.expected_row_version), display_name: values.display_name, metadb_endpoint: values.metadb_endpoint, sharepoint_site_path: values.sharepoint_site_path, poll_interval_seconds: Number(values.poll_interval_seconds), batch_size: Number(values.batch_size) };
    if (metadbPair.every(Boolean)) payload.metadb_credentials = { username: values.metadb_username, password: values.metadb_password };
    if (sharepointGroup.every(Boolean)) payload.sharepoint_credentials = { tenant_id: values.tenant_id, client_id: values.client_id, client_secret: values.client_secret };
    KBotKmShell.setBusy($("update-source"), true, "保存中…");
    try { await KBotKmApi.json(`${base}/sources/${encodeURIComponent(values.source_id)}`, "PATCH", payload); KBotKmShell.closeDialog("source-edit-dialog"); KBotKmShell.toast("数据来源已更新", "success"); await load(); }
    catch (error) {
      if (!(isVersionConflict(error) && await refreshConflict(values.source_id, form))) {
        KBotKmShell.showError(error, "数据来源更新失败");
      }
    }
    finally { KBotKmShell.setBusy($("update-source"), false); }
  }
  window.addEventListener("DOMContentLoaded", () => {
    document.addEventListener("submit", (event) => {
      if (event.target.id === "source-form") save(event);
      if (event.target.id === "source-edit-form") update(event);
    });
    document.addEventListener("click", async (event) => {
      const newButton = event.target.closest("#new-source");
      if (newButton) { KBotKmShell.openDialog("source-dialog"); return; }
      const refreshButton = event.target.closest("#refresh-sources");
      if (refreshButton) { await load(); return; }
      const reconcileButton = event.target.closest("#reconcile-data-model");
      if (reconcileButton) {
        if (!modelSource) return;
        KBotKmShell.setBusy(reconcileButton, true);
        try {
          const result = await KBotKmApi.json(`${base}/sources/${modelSource.source_id}/data-model/reconcile`, "POST");
          $("data-model-json").textContent = JSON.stringify(result, null, 2);
          KBotKmShell.toast("固定数据模型已对账", "success");
          await load();
        } catch (error) { KBotKmShell.showError(error); }
        finally { KBotKmShell.setBusy(reconcileButton, false); }
        return;
      }
      for (const kind of ["edit", "model", "activate", "auto", "sync"]) {
        const button = event.target.closest(`[data-${kind}]`);
        if (!button) continue;
        if (kind === "edit") openEdit(Number(button.dataset.edit));
        else if (kind === "model") showModel(Number(button.dataset.model));
        else action(Number(button.dataset[kind]), kind, button);
        break;
      }
    });
  });
  KBotKmShell.ready.then(load).catch(() => {});
})();
