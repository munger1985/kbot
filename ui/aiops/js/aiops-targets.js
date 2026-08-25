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
  const result = document.getElementById("target-connection-result");

  function configureEndpoint() {
    const oracle = dbType.value === "ORACLE";
    serviceField.hidden = !oracle;
    databaseField.hidden = oracle;
    service.required = oracle;
    database.required = !oracle;
    if (oracle) database.value = "";
    else service.value = "";
    port.value = { ORACLE: 1521, MYSQL: 3306, POSTGRESQL: 5432 }[dbType.value];
    result.textContent = "";
    delete result.dataset.tone;
  }

  function openDialog() {
    form.reset();
    dbType.value = "ORACLE";
    configureEndpoint();
    dialog.showModal();
    document.getElementById("target-display-name").focus();
  }

  function connectionPayload() {
    const values = Object.fromEntries(new FormData(form));
    const oracle = values.db_type === "ORACLE";
    const endpoint = {
      host: values.host.trim(),
      port: Number(values.port),
      tls_enabled: form.elements.tls_enabled.checked,
    };
    endpoint[oracle ? "service" : "database"] = (
      oracle ? values.service : values.database
    ).trim();
    return {
      db_type: values.db_type,
      endpoint,
      diagnostic_credential: {
        username: values.diagnostic_username.trim(),
        password: values.diagnostic_password,
      },
    };
  }

  function connectionFieldsAreValid() {
    const fields = [
      dbType,
      document.getElementById("target-host"),
      port,
      dbType.value === "ORACLE" ? service : database,
      document.getElementById("target-diagnostic-username"),
      document.getElementById("target-diagnostic-password"),
    ];
    return fields.every((field) => field.reportValidity());
  }

  async function testConnection() {
    if (!connectionFieldsAreValid()) return;
    const button = document.getElementById("test-target-connection");
    button.disabled = true;
    button.textContent = "测试中…";
    result.textContent = "正在验证数据库网络、认证和最小只读查询…";
    delete result.dataset.tone;
    try {
      const response = await KBotAIOpsAuth.request(
        `${api}/targets/test-connection`,
        { method: "POST", body: JSON.stringify(connectionPayload()) },
      );
      if (!response.ok) {
        const messages = {
          AUTH_FAILED: "数据库身份验证失败，请检查只读用户名和密码。",
          TARGET_UNREACHABLE: "无法连接数据库，请检查主机、端口、Service Name/Database、网络和 TLS。",
          TIMEOUT: "数据库连接超时，请检查防火墙和访问控制。",
          CONNECTION_FAILED: "数据库连接失败，请检查连接参数。",
        };
        throw new Error(messages[response.error_code] || "数据库连接测试失败。");
      }
      result.dataset.tone = "good";
      result.textContent = `连接成功${response.database_version ? `，数据库版本 ${response.database_version}` : ""}`;
    } catch (error) {
      result.dataset.tone = "bad";
      result.textContent = error.message;
    } finally {
      button.disabled = false;
      button.textContent = "测试连接";
    }
  }

  async function createTarget(event) {
    event.preventDefault();
    const submit = form.querySelector('button[type="submit"]');
    const values = Object.fromEntries(new FormData(form));
    const payload = {
      ...connectionPayload(),
      display_name: values.display_name.trim(),
      version_code: values.version_code.trim() || null,
      environment: values.environment,
      db_role: values.db_role,
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
    document.getElementById("test-target-connection").addEventListener("click", testConnection);
    dbType.addEventListener("change", configureEndpoint);
    form.addEventListener("input", () => {
      result.textContent = "";
      delete result.dataset.tone;
    });
    form.addEventListener("submit", createTarget);
  });
})();
