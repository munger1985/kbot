(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const form = document.getElementById("target-form");
  const dialog = document.getElementById("target-dialog");
  const dbType = document.getElementById("target-db-type");
  const port = document.getElementById("target-port");
  const serviceField = document.getElementById("target-service-field");
  const databaseField = document.getElementById("target-database-field");
  const service = document.getElementById("target-service");
  const database = document.getElementById("target-database");

  function configureEndpoint() {
    const oracle = dbType.value === "ORACLE";
    serviceField.hidden = !oracle;
    databaseField.hidden = oracle;
    service.required = oracle;
    database.required = !oracle;
    if (oracle) database.value = "";
    else service.value = "";
    port.value = { ORACLE: 1521, MYSQL: 3306, POSTGRESQL: 5432 }[dbType.value];
  }

  function openDialog() {
    form.reset();
    dbType.value = "ORACLE";
    configureEndpoint();
    dialog.showModal();
    document.getElementById("target-display-name").focus();
  }

  async function createTarget(event) {
    event.preventDefault();
    const submit = form.querySelector('button[type="submit"]');
    const values = Object.fromEntries(new FormData(form));
    const oracle = values.db_type === "ORACLE";
    const payload = {
      display_name: values.display_name.trim(),
      db_type: values.db_type,
      version_code: values.version_code.trim() || null,
      environment: values.environment,
      db_role: values.db_role,
      endpoint: {
        host: values.host.trim(),
        port: Number(values.port),
        service: oracle ? values.service.trim() : null,
        database: oracle ? null : values.database.trim(),
        tls_enabled: form.elements.tls_enabled.checked,
      },
      diagnostic_credential: {
        username: values.diagnostic_username.trim(),
        password: values.diagnostic_password,
      },
      security_level: Number(values.security_level),
      capabilities: {},
    };
    submit.disabled = true;
    submit.textContent = "创建中…";
    try {
      await KBotAIOpsAuth.request(`${api}/targets`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify(payload),
      });
      dialog.close();
      shell.toast("运维目标已创建");
      await KBotAIOpsPages.reload();
    } catch (error) {
      shell.toast(error.message);
    } finally {
      submit.disabled = false;
      submit.textContent = "创建目标";
    }
  }

  shell.ready.then(() => {
    document.getElementById("create-target").addEventListener("click", openDialog);
    document.getElementById("close-target-dialog").addEventListener("click", () => dialog.close());
    document.getElementById("cancel-target-dialog").addEventListener("click", () => dialog.close());
    dbType.addEventListener("change", configureEndpoint);
    form.addEventListener("submit", createTarget);
  });
})();
