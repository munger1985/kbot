(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  let clients = [];

  function defaultExpiry() {
    const value = new Date(Date.now() + 90 * 24 * 60 * 60 * 1000);
    value.setMinutes(value.getMinutes() - value.getTimezoneOffset());
    return value.toISOString().slice(0, 16);
  }

  function showError(error, fallback) {
    const element = $("api-client-error");
    const request = error?.requestId ? `；request_id: ${error.requestId}` : "";
    element.textContent = `${error?.message || fallback}${request}`;
    element.hidden = false;
    KBotKmShell.showError(error, fallback);
  }

  function credentialSummary(row) {
    const active = (row.credentials || []).find((item) => item.status === "ACTIVE");
    if (!active) return "没有有效密钥";
    return `到期 ${KBotKmShell.formatDate(active.expires_at)} · 最近使用 ${KBotKmShell.formatDate(active.last_used_at)}`;
  }

  function render() {
    const body = $("api-client-rows");
    if (!clients.length) return KBotKmShell.renderEmpty(body, 7, "尚未创建 API 客户端");
    body.innerHTML = clients.map((row, index) => `<tr><td><span class="km-cell-main">${KBotKmShell.escapeHtml(row.display_name)}</span><div class="km-cell-sub">${KBotKmShell.shortId(row.client_id)}</div></td><td>${KBotKmShell.escapeHtml(row.subject_user_id)}</td><td>${(row.scopes || []).map((scope) => `<code>${KBotKmShell.escapeHtml(scope)}</code>`).join("<br>")}</td><td>${KBotKmShell.escapeHtml(`${(row.agent_ids || []).length} 个`)}</td><td>${KBotKmShell.badge(row.status)}</td><td class="km-cell-sub">${KBotKmShell.escapeHtml(credentialSummary(row))}</td><td><div class="km-actions"><button class="small" data-rotate="${index}" ${row.status !== "ACTIVE" ? "disabled" : ""}>轮换</button><button class="small danger" data-disable="${index}" ${row.status !== "ACTIVE" ? "disabled" : ""}>停用</button></div></td></tr>`).join("");
  }

  async function load() {
    try {
      clients = KBotKmApi.items(await KBotKmApi.request(`${base}/api-clients`));
      render();
    } catch (error) { showError(error, "API 客户端加载失败"); }
  }

  async function initialize() {
    if (!KBotKmShell.access.permissions.includes("km_asset:api_key_manage")) {
      showError(new Error("当前用户没有 API 客户端管理权限"), "权限不足");
      return;
    }
    try {
      const [scopePayload, memberPayload, agentPayload] = await Promise.all([
        KBotKmApi.request(`${base}/api-clients/scopes`),
        KBotKmApi.request(`${base}/members`),
        KBotKmApi.request(`${base}/agents`),
      ]);
      $("api-client-scopes").innerHTML = (scopePayload.items || []).map((item) => `<label><input type="checkbox" name="scopes" value="${KBotKmShell.escapeHtml(item.scope_code)}"> ${KBotKmShell.escapeHtml(item.scope_code)}</label>`).join("");
      $("api-client-subject").innerHTML = KBotKmApi.items(memberPayload).filter((item) => item.status === "ACTIVE").map((item) => `<option value="${KBotKmShell.escapeHtml(item.user_id)}">${KBotKmShell.escapeHtml(item.display_name || item.user_id)}</option>`).join("");
      $("api-client-agents").innerHTML = KBotKmApi.items(agentPayload).filter((item) => item.status === "ACTIVE").map((item) => `<option value="${KBotKmShell.escapeHtml(item.agent_id)}">${KBotKmShell.escapeHtml(item.display_name)}</option>`).join("");
      await load();
    } catch (error) { showError(error, "API 客户端初始化失败"); }
  }

  function revealKey(result) {
    $("api-key-value").textContent = result.api_key;
    KBotKmShell.openDialog("api-key-dialog");
  }

  async function create(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const values = new FormData(form);
    const payload = {
      display_name: String(values.get("display_name") || "").trim(),
      subject_user_id: String(values.get("subject_user_id") || ""),
      scopes: values.getAll("scopes"),
      agent_ids: Array.from(form.elements.agent_ids.selectedOptions, (item) => item.value),
      expires_at: new Date(String(values.get("expires_at"))).toISOString(),
      rate_limit_per_minute: Number(values.get("rate_limit_per_minute")),
    };
    KBotKmShell.setBusy($("save-api-client"), true, "生成中…");
    try {
      const result = await KBotKmApi.json(`${base}/api-clients`, "POST", payload);
      KBotKmShell.closeDialog("api-client-dialog"); revealKey(result); await load();
    } catch (error) { KBotKmShell.showError(error, "API 客户端创建失败"); }
    finally { KBotKmShell.setBusy($("save-api-client"), false); }
  }

  async function disable(index, button) {
    const row = clients[index]; KBotKmShell.setBusy(button, true, "停用中…");
    try { await KBotKmApi.json(`${base}/api-clients/${row.client_id}`, "PATCH", { status: "DISABLED" }); await load(); }
    catch (error) { KBotKmShell.showError(error, "API 客户端停用失败"); }
    finally { KBotKmShell.setBusy(button, false); }
  }

  function openRotate(index) {
    const form = $("rotate-api-key-form"); form.reset();
    form.elements.client_id.value = clients[index].client_id;
    form.elements.expires_at.value = defaultExpiry();
    KBotKmShell.openDialog("rotate-api-key-dialog");
  }

  async function rotate(event) {
    event.preventDefault(); const form = event.currentTarget;
    KBotKmShell.setBusy($("rotate-api-key"), true, "轮换中…");
    try {
      const result = await KBotKmApi.json(`${base}/api-clients/${form.elements.client_id.value}/rotate`, "POST", { expires_at: new Date(form.elements.expires_at.value).toISOString() });
      KBotKmShell.closeDialog("rotate-api-key-dialog"); revealKey(result); await load();
    } catch (error) { KBotKmShell.showError(error, "API Key 轮换失败"); }
    finally { KBotKmShell.setBusy($("rotate-api-key"), false); }
  }

  window.addEventListener("DOMContentLoaded", () => {
    $("new-api-client").addEventListener("click", () => { $("api-client-form").reset(); $("api-client-form").elements.expires_at.value = defaultExpiry(); KBotKmShell.openDialog("api-client-dialog"); });
    $("refresh-api-clients").addEventListener("click", load);
    $("api-client-form").addEventListener("submit", create);
    $("rotate-api-key-form").addEventListener("submit", rotate);
    $("copy-api-key").addEventListener("click", async () => { await navigator.clipboard.writeText($("api-key-value").textContent); KBotKmShell.toast("API Key 已复制", "success"); });
    $("api-client-rows").addEventListener("click", (event) => { const rotateButton = event.target.closest("[data-rotate]"); if (rotateButton) return openRotate(Number(rotateButton.dataset.rotate)); const disableButton = event.target.closest("[data-disable]"); if (disableButton) disable(Number(disableButton.dataset.disable), disableButton); });
  });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
