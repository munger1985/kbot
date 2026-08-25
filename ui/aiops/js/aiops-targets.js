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
  const username = document.getElementById("target-diagnostic-username");
  const password = document.getElementById("target-diagnostic-password");
  const result = document.getElementById("target-connection-result");
  const submit = document.getElementById("save-target");
  let editingTarget = null;

  function clearResult() {
    result.textContent = "";
    delete result.dataset.tone;
  }

  function configureEndpoint(resetPort = true) {
    const oracle = dbType.value === "ORACLE";
    serviceField.hidden = !oracle;
    databaseField.hidden = oracle;
    service.required = oracle;
    database.required = !oracle;
    if (oracle) database.value = "";
    else service.value = "";
    if (resetPort) port.value = { ORACLE: 1521, MYSQL: 3306, POSTGRESQL: 5432 }[dbType.value];
    clearResult();
  }

  function setCredentialMode(required) {
    username.required = required;
    password.required = required;
    username.placeholder = required ? "" : "留空则不更换现有凭据";
    password.placeholder = required ? "" : "留空则不更换现有凭据";
    document.getElementById("target-credential-note").textContent = required
      ? "诊断凭据将写入 AIOps 加密凭据存储，列表和详情不会返回密码明文。"
      : "已保存的诊断凭据不会回显；用户名和密码都留空表示保持不变，同时填写才会轮换。";
  }

  function openCreate() {
    editingTarget = null;
    form.reset();
    dbType.disabled = false;
    dbType.value = "ORACLE";
    configureEndpoint();
    setCredentialMode(true);
    document.getElementById("target-dialog-title").textContent = "新增运维目标";
    submit.textContent = "创建目标";
    dialog.showModal();
    form.elements.display_name.focus();
  }

  async function openEdit(targetId) {
    try {
      const target = await KBotAIOpsAuth.request(`${api}/targets/${encodeURIComponent(targetId)}`);
      editingTarget = target;
      form.reset();
      dbType.disabled = false;
      dbType.value = target.db_type;
      configureEndpoint(false);
      form.elements.display_name.value = target.display_name;
      form.elements.version_code.value = target.version_code || "";
      form.elements.environment.value = target.environment;
      form.elements.db_role.value = target.db_role;
      form.elements.security_level.value = target.security_level;
      form.elements.host.value = target.endpoint?.host || "";
      form.elements.port.value = target.endpoint?.port || "";
      form.elements.tls_enabled.checked = Boolean(target.endpoint?.tls_enabled);
      if (target.db_type === "ORACLE") service.value = target.endpoint?.service || "";
      else database.value = target.endpoint?.database || "";
      dbType.disabled = true;
      setCredentialMode(false);
      document.getElementById("target-dialog-title").textContent = "编辑运维目标";
      submit.textContent = "保存修改";
      clearResult();
      dialog.showModal();
      form.elements.display_name.focus();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  function endpointPayload() {
    const oracle = dbType.value === "ORACLE";
    const endpoint = {
      host: form.elements.host.value.trim(),
      port: Number(port.value),
      tls_enabled: form.elements.tls_enabled.checked,
    };
    endpoint[oracle ? "service" : "database"] = (oracle ? service.value : database.value).trim();
    return endpoint;
  }

  function credentialPayload() {
    return { username: username.value.trim(), password: password.value };
  }

  function connectionFieldsAreValid() {
    if (editingTarget && (!username.value.trim() || !password.value)) {
      result.dataset.tone = "bad";
      result.textContent = "测试连接需要重新输入只读诊断用户名和密码。";
      return false;
    }
    return [dbType, form.elements.host, port, dbType.value === "ORACLE" ? service : database, username, password]
      .every((field) => field.reportValidity());
  }

  async function testConnection() {
    if (!connectionFieldsAreValid()) return;
    const button = document.getElementById("test-target-connection");
    button.disabled = true;
    button.textContent = "测试中…";
    result.textContent = "正在验证数据库网络、认证和最小只读查询…";
    delete result.dataset.tone;
    try {
      const response = await KBotAIOpsAuth.request(`${api}/targets/test-connection`, {
        method: "POST",
        body: JSON.stringify({ db_type: dbType.value, endpoint: endpointPayload(), diagnostic_credential: credentialPayload() }),
      });
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

  function targetFields() {
    return {
      display_name: form.elements.display_name.value.trim(),
      version_code: form.elements.version_code.value.trim() || null,
      environment: form.elements.environment.value,
      db_role: form.elements.db_role.value,
      endpoint: endpointPayload(),
      security_level: Number(form.elements.security_level.value),
    };
  }

  async function saveTarget(event) {
    event.preventDefault();
    const rotatingCredential = Boolean(username.value.trim() || password.value);
    if (editingTarget && rotatingCredential && (!username.value.trim() || !password.value)) {
      result.dataset.tone = "bad";
      result.textContent = "轮换凭据时必须同时填写用户名和密码。";
      return;
    }
    submit.disabled = true;
    submit.textContent = editingTarget ? "保存中…" : "创建中…";
    let baseSaved = false;
    try {
      if (!editingTarget) {
        await KBotAIOpsAuth.request(`${api}/targets`, {
          method: "POST",
          headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify({ ...targetFields(), db_type: dbType.value, capabilities: {}, diagnostic_credential: credentialPayload() }),
        });
      } else {
        let updated = await KBotAIOpsAuth.request(`${api}/targets/${encodeURIComponent(editingTarget.target_id)}`, {
          method: "PATCH",
          headers: { "If-Match": `"rv-${editingTarget.row_version}"` },
          body: JSON.stringify({ ...targetFields(), capabilities: editingTarget.capabilities || {} }),
        });
        baseSaved = true;
        if (rotatingCredential) {
          updated = await KBotAIOpsAuth.request(
            `${api}/targets/${encodeURIComponent(editingTarget.target_id)}/diagnostic-credential:rotate`,
            {
              method: "POST",
              headers: { "If-Match": `"rv-${updated.row_version}"`, "Idempotency-Key": KBotAIOpsAuth.uuid() },
              body: JSON.stringify(credentialPayload()),
            },
          );
        }
      }
      dialog.close();
      shell.toast(editingTarget ? "运维目标已更新" : "运维目标已创建");
      editingTarget = null;
      await KBotAIOpsPages.reload();
    } catch (error) {
      result.dataset.tone = "bad";
      result.textContent = baseSaved ? `基本信息已保存，但诊断凭据轮换失败：${error.message}` : error.message;
      if (baseSaved) await KBotAIOpsPages.reload();
    } finally {
      submit.disabled = false;
      submit.textContent = editingTarget ? "保存修改" : "创建目标";
    }
  }

  globalThis.KBotAIOpsTargets = { openEdit };
  shell.ready.then(() => {
    document.getElementById("create-target").addEventListener("click", openCreate);
    document.getElementById("close-target-dialog").addEventListener("click", () => dialog.close());
    document.getElementById("cancel-target-dialog").addEventListener("click", () => dialog.close());
    document.getElementById("test-target-connection").addEventListener("click", testConnection);
    dbType.addEventListener("change", () => configureEndpoint());
    form.addEventListener("input", clearResult);
    form.addEventListener("submit", saveTarget);
  });
})();
