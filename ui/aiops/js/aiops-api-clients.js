(function () {
  "use strict";
  const shell = globalThis.KBotAIOpsShell;
  shell.ready.then(async (access) => {
    if (!access || document.body.dataset.page !== "api-clients") return;
    const panel = document.getElementById("ops-simple");
    if (!panel) return;
    document.querySelectorAll("header.ops-head button").forEach((button) => {
      button.disabled = true;
      button.title = "该写操作将在对应配置表单接入后开放";
    });
    try {
      const clients = await KBotAIOpsAuth.request("/api/v1/apps/aiops/api-clients");
      panel.innerHTML = `<pre class="ops-code">${shell.escape(JSON.stringify(clients, null, 2))}</pre>`;
    } catch (error) {
      panel.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`;
    }
  }).catch((error) => shell.toast(error.message));
})();
