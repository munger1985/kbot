(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const form = document.getElementById("target-form");
  const dialog = document.getElementById("target-dialog");
  const dbType = document.getElementById("target-db-type");
  const version = document.getElementById("target-version");
  const port = document.getElementById("target-port");
  const serviceField = document.getElementById("target-service-field");
  const databaseField = document.getElementById("target-database-field");
  const service = document.getElementById("target-service");
  const oracleScopeField = document.getElementById("target-oracle-scope-field");
  const oraclePdbField = document.getElementById("target-oracle-pdb-field");
  const oracleScope = document.getElementById("target-oracle-scope");
  const oraclePdbName = document.getElementById("target-oracle-pdb-name");
  const database = document.getElementById("target-database");
  const username = document.getElementById("target-diagnostic-username");
  const password = document.getElementById("target-diagnostic-password");
  const executionUsername = document.getElementById("target-execution-username");
  const executionPassword = document.getElementById("target-execution-password");
  const result = document.getElementById("target-connection-result");
  const submit = document.getElementById("save-target");
  const readonlyEnabled = document.getElementById("target-readonly-enabled");
  const changeEnabled = document.getElementById("target-change-enabled");
  const accessSummary = document.getElementById("target-access-summary");
  const monitorState = document.getElementById("target-monitor-editor-state");
  const monitorBindings = document.getElementById("target-monitor-editor-bindings");
  const monitorSource = document.getElementById("target-monitor-editor-source");
  const monitorLocatorField = document.getElementById("target-monitor-editor-locator-field");
  const monitorLocator = document.getElementById("target-monitor-editor-locator");
  const monitorLocatorLabel = document.getElementById("target-monitor-editor-locator-label");
  const monitorLocatorHelp = document.getElementById("target-monitor-editor-locator-help");
  const monitorJobField = document.getElementById("target-monitor-editor-job-field");
  const monitorJob = document.getElementById("target-monitor-editor-job");
  const monitorDiscovery = document.getElementById("target-monitor-editor-discovery");
  const monitorCandidates = document.getElementById("target-monitor-editor-candidates");
  const monitorHostSection = document.getElementById("target-monitor-editor-host-section");
  const monitorHostCandidates = document.getElementById("target-monitor-editor-host-candidates");
  const discoverMonitorLabels = document.getElementById("discover-target-monitor-editor-labels");
  let editingTarget = null;
  let availableMonitorSources = [];
  let currentMonitorBindings = [];
  const supportedVersions = {
    ORACLE: ["19c", "26ai"],
    MYSQL: ["8.4"],
    POSTGRESQL: ["16"],
  };

  function clearResult() {
    result.textContent = "";
    delete result.dataset.tone;
  }

  const monitorSourceById = (sourceId) => availableMonitorSources.find(
    (item) => String(item.source_id) === String(sourceId)
  );
  const monitorBindingBySourceId = (sourceId) => currentMonitorBindings.find(
    (item) => item.status === "ACTIVE" && String(item.source_id) === String(sourceId)
  );

  function renderMonitorBindings() {
    const active = currentMonitorBindings.filter((item) => item.status === "ACTIVE");
    monitorState.textContent = active.length ? `${active.length} 条有效映射` : "必须配置";
    monitorState.className = active.length ? "good" : "bad";
    monitorBindings.innerHTML = currentMonitorBindings.length
      ? `<div class="target-monitor-candidates">${currentMonitorBindings.map((binding) => {
        const source = monitorSourceById(binding.source_id);
        const action = binding.status === "ACTIVE" ? "disable" : "enable";
        const actionLabel = binding.status === "ACTIVE" ? "解除" : "恢复";
        const hostKey = binding.source_locator?.host_target_key;
        return `<div class="agent-switch-row"><span><strong>${shell.escape(source?.display_name || shell.short(binding.source_id))}</strong><small>${shell.escape(source?.source_type || "—")} · 数据库 <code>${shell.escape(binding.source_locator_key || binding.locator_hint)}</code> · 主机 <code>${shell.escape(hostKey || "未配置")}</code> · ${shell.escape(binding.status)}</small></span><span class="ops-actions"><button type="button" data-monitor-binding-action="${action}" data-binding-id="${shell.escape(binding.binding_id)}" data-row-version="${binding.row_version}">${actionLabel}</button><button type="button" class="danger" data-monitor-binding-delete data-binding-id="${shell.escape(binding.binding_id)}" data-row-version="${binding.row_version}">删除</button></span></div>`;
      }).join("")}</div>`
      : '<div class="ops-error">尚未绑定监控源；创建 Target 前必须选择监控源和 Label。</div>';
  }

  function renderMonitorSourceOptions() {
    const boundSourceIds = new Set(
      currentMonitorBindings
        .filter((binding) => binding.status === "ACTIVE")
        .map((binding) => String(binding.source_id))
    );
    const sources = availableMonitorSources.filter((source) => (
      source.status === "ENABLED"
      && (
        !boundSourceIds.has(String(source.source_id))
        || source.source_type === "PROMETHEUS"
      )
    ));
    monitorSource.innerHTML = '<option value="">请选择监控源</option>' + sources.map(
      (source) => `<option value="${shell.escape(source.source_id)}">${shell.escape(source.display_name)} · ${shell.escape(source.source_type)}${boundSourceIds.has(String(source.source_id)) ? " · 修改主机 Label" : ""}</option>`
    ).join("");
    monitorSource.disabled = !sources.length;
    configureMonitorSource();
  }

  function configureMonitorSource() {
    const source = monitorSourceById(monitorSource.value);
    const binding = monitorBindingBySourceId(monitorSource.value);
    const discoverable = ["PROMETHEUS", "ZABBIX"].includes(source?.source_type);
    monitorDiscovery.hidden = !discoverable;
    monitorLocatorField.hidden = discoverable || !source;
    monitorLocator.required = Boolean(source && !discoverable);
    monitorJobField.hidden = source?.source_type !== "LOKI";
    monitorHostSection.hidden = source?.source_type !== "PROMETHEUS";
    monitorCandidates.innerHTML = binding
      ? `<div class="ops-empty">已绑定数据库 Label：<code>${shell.escape(binding.source_locator_key || binding.locator_hint)}</code>；本次仅修改主机 Label。</div>`
      : '<div class="ops-empty">点击发现按钮读取当前数据库类型的候选。</div>';
    monitorHostCandidates.innerHTML = '<div class="ops-empty">点击发现按钮读取 Node Exporter 主机候选。</div>';
    if (!source) return;
    const presentation = {
      ALERTMANAGER: ["target_key label 值", "必须与告警中的 target_key 完全一致。"],
      LOKI: ["target_key label 值", "必须与日志流中的 target_key 完全一致。"],
      OEM: ["OEM Target Name", "填写 OEM 中唯一的 Target Name。"],
    }[source.source_type] || ["监控 Label 值", "填写监控系统中唯一标识当前数据库的值。"];
    monitorLocatorLabel.textContent = presentation[0];
    monitorLocatorHelp.textContent = presentation[1];
  }

  async function loadMonitorEditor(targetId = null) {
    monitorBindings.innerHTML = '<div class="ops-empty">正在读取监控配置…</div>';
    const sourcePage = await KBotAIOpsAuth.request(`${api}/diagnostic-sources?limit=200`);
    availableMonitorSources = Array.isArray(sourcePage) ? sourcePage : sourcePage.items || [];
    currentMonitorBindings = targetId
      ? await KBotAIOpsAuth.request(`${api}/targets/${encodeURIComponent(targetId)}/source-bindings`)
      : [];
    monitorLocator.value = "";
    monitorJob.value = "";
    renderMonitorBindings();
    renderMonitorSourceOptions();
  }

  async function discoverMonitorCandidates() {
    const source = monitorSourceById(monitorSource.value);
    if (!source || !["PROMETHEUS", "ZABBIX"].includes(source.source_type)) return;
    discoverMonitorLabels.disabled = true;
    monitorCandidates.innerHTML = '<div class="ops-empty">正在从监控源发现候选 Label…</div>';
    try {
      const page = await KBotAIOpsAuth.request(
        `${api}/diagnostic-sources/${encodeURIComponent(source.source_id)}/instance-discoveries`,
        { method: "POST", body: JSON.stringify({ db_types: [dbType.value], page_size: 100 }) }
      );
      const binding = monitorBindingBySourceId(source.source_id);
      const candidates = (page.items || []).filter((item) => item.mapping_status !== "MAPPED");
      if (!binding) {
        monitorCandidates.innerHTML = candidates.length
          ? candidates.map((candidate, index) => `<label class="agent-switch-row"><input type="radio" name="monitor_candidate_ref" value="${shell.escape(candidate.candidate_ref)}" ${index === 0 ? "checked" : ""}><span><strong>${shell.escape(candidate.display_name)}</strong><small>${shell.escape(candidate.db_type)} · <code>${shell.escape(candidate.locator_hint)}</code></small></span></label>`).join("")
          : '<div class="ops-empty">没有尚未映射且与当前数据库类型一致的候选 Label。</div>';
      } else {
        monitorCandidates.innerHTML = `<div class="ops-empty">已绑定数据库 Label：<code>${shell.escape(binding.source_locator_key || binding.locator_hint)}</code>；本次仅修改主机 Label。</div>`;
      }
      const hostCandidates = page.host_items || [];
      monitorHostCandidates.innerHTML = hostCandidates.length
        ? hostCandidates.map((candidate, index) => `<label class="agent-switch-row"><input type="radio" name="monitor_host_candidate_ref" value="${shell.escape(candidate.candidate_ref)}" ${index === 0 ? "checked" : ""}><span><strong>${shell.escape(candidate.display_name)}</strong><small>主机 · <code>${shell.escape(candidate.locator_hint)}</code></small></span></label>`).join("")
        : '<div class="ops-error">未发现 Node Exporter 主机 Label；请先接入目标主机监控。</div>';
    } catch (error) {
      monitorCandidates.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`;
    } finally {
      discoverMonitorLabels.disabled = false;
    }
  }

  function monitorSelectionIsValid() {
    const hasActiveBinding = currentMonitorBindings.some((item) => item.status === "ACTIVE");
    const source = monitorSourceById(monitorSource.value);
    if (!source) {
      if (hasActiveBinding) return true;
      result.dataset.tone = "bad";
      result.textContent = "请先选择监控源并配置当前数据库的 Label。";
      return false;
    }
    if (["PROMETHEUS", "ZABBIX"].includes(source.source_type)) {
      const binding = monitorBindingBySourceId(source.source_id);
      const databaseSelected = binding || form.elements.monitor_candidate_ref?.value;
      const hostSelected = source.source_type !== "PROMETHEUS" || form.elements.monitor_host_candidate_ref?.value;
      if (databaseSelected && hostSelected) return true;
      result.dataset.tone = "bad";
      result.textContent = !databaseSelected
        ? "请先发现并选择数据库 Label。"
        : "请先发现并选择数据库所在主机的 Label。";
      return false;
    }
    if (monitorLocator.value.trim()) return true;
    result.dataset.tone = "bad";
    result.textContent = "监控 Label 值不能为空。";
    return false;
  }

  async function saveSelectedMonitorBinding(target) {
    const source = monitorSourceById(monitorSource.value);
    if (!source) return;
    if (["PROMETHEUS", "ZABBIX"].includes(source.source_type)) {
      const binding = monitorBindingBySourceId(source.source_id);
      const hostCandidateRef = form.elements.monitor_host_candidate_ref?.value || null;
      if (binding) {
        await KBotAIOpsAuth.request(
          `${api}/targets/${encodeURIComponent(target.target_id)}/source-bindings/${encodeURIComponent(binding.binding_id)}`,
          {
            method: "PATCH",
            headers: { "If-Match": `"rv-${binding.row_version}"` },
            body: JSON.stringify({ host_candidate_ref: hostCandidateRef }),
          }
        );
        return;
      }
      await KBotAIOpsAuth.request(
        `${api}/diagnostic-sources/${encodeURIComponent(source.source_id)}/instance-mappings`,
        {
          method: "POST",
          headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify({
            mappings: [{
              candidate_ref: form.elements.monitor_candidate_ref.value,
              host_candidate_ref: hostCandidateRef,
              target_id: target.target_id,
            }],
          }),
        }
      );
      return;
    }
    const locatorKey = monitorLocator.value.trim();
    const labels = source.source_type === "LOKI"
      ? { target_key: locatorKey, ...(monitorJob.value.trim() ? { job: monitorJob.value.trim() } : {}) }
      : null;
    await KBotAIOpsAuth.request(`${api}/targets/${encodeURIComponent(target.target_id)}/source-bindings`, {
      method: "POST",
      headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
      body: JSON.stringify({
        source_id: source.source_id,
        source_locator_key: locatorKey,
        source_locator: labels ? { labels } : {},
        role: "PRIMARY",
        priority: 100,
      }),
    });
  }

  async function commandMonitorBinding(button) {
    if (!editingTarget) return;
    const action = button.dataset.monitorBindingAction;
    const activeCount = currentMonitorBindings.filter((item) => item.status === "ACTIVE").length;
    if (action === "disable" && activeCount <= 1) {
      result.dataset.tone = "bad";
      result.textContent = "Target 必须保留至少一条有效监控映射；请先绑定新的监控源。";
      return;
    }
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(
        `${api}/targets/${encodeURIComponent(editingTarget.target_id)}/source-bindings/${encodeURIComponent(button.dataset.bindingId)}/${action}`,
        {
          method: "POST",
          headers: {
            "If-Match": `"rv-${button.dataset.rowVersion}"`,
            "Idempotency-Key": KBotAIOpsAuth.uuid(),
          },
          body: JSON.stringify({}),
        }
      );
      await loadMonitorEditor(editingTarget.target_id);
    } catch (error) {
      result.dataset.tone = "bad";
      result.textContent = error.message;
      button.disabled = false;
    }
  }

  async function deleteMonitorBinding(button) {
    if (!editingTarget) return;
    if (!confirm("确认删除这条监控映射吗？历史告警会保留；删除最后一条有效映射时 Target 将自动停用。")) return;
    const leavesNoActiveBinding = !currentMonitorBindings.some(
      (item) => item.status === "ACTIVE" && item.binding_id !== button.dataset.bindingId
    );
    const targetWasEnabled = editingTarget.status === "ENABLED";
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(
        `${api}/targets/${encodeURIComponent(editingTarget.target_id)}/source-bindings/${encodeURIComponent(button.dataset.bindingId)}`,
        {
          method: "DELETE",
          headers: {
            "If-Match": `"rv-${button.dataset.rowVersion}"`,
            "Idempotency-Key": KBotAIOpsAuth.uuid(),
          },
        }
      );
      if (leavesNoActiveBinding && targetWasEnabled) editingTarget.status = "DISABLED";
      await loadMonitorEditor(editingTarget.target_id);
      result.dataset.tone = "good";
      result.textContent = leavesNoActiveBinding && targetWasEnabled
        ? "监控映射已删除；Target 因无有效监控映射已自动停用。"
        : "监控映射已删除。";
    } catch (error) {
      result.dataset.tone = "bad";
      result.textContent = error.message;
      button.disabled = false;
    }
  }

  function configureEndpoint(resetPort = true) {
    const oracle = dbType.value === "ORACLE";
    const readonly = readonlyEnabled.checked;
    serviceField.hidden = !oracle;
    databaseField.hidden = oracle;
    service.required = readonlyEnabled.checked && oracle;
    database.required = readonlyEnabled.checked && !oracle;
    oracleScopeField.hidden = !readonly || !oracle;
    oracleScope.required = readonly && oracle;
    oraclePdbField.hidden = !readonly || !oracle || oracleScope.value !== "PDB";
    oraclePdbName.required = readonly && oracle && oracleScope.value === "PDB";
    if (oracle) database.value = "";
    else {
      service.value = "";
      oracleScope.value = "";
      oraclePdbName.value = "";
    }
    if (resetPort) port.value = { ORACLE: 1521, MYSQL: 3306, POSTGRESQL: 5432 }[dbType.value];
    const previousVersion = version.value;
    version.replaceChildren(...supportedVersions[dbType.value].map((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      return option;
    }));
    if (!resetPort && supportedVersions[dbType.value].includes(previousVersion)) {
      version.value = previousVersion;
    }
    clearResult();
  }

  function toggleAccessFields() {
    const readonly = readonlyEnabled.checked;
    if (!readonly) changeEnabled.checked = false;
    changeEnabled.disabled = !readonly;
    document.querySelectorAll(".target-connection-field").forEach((field) => {
      field.hidden = !readonly;
    });
    document.querySelectorAll(".target-change-field").forEach((field) => {
      field.hidden = !changeEnabled.checked;
    });
    form.elements.host.required = readonly;
    port.required = readonly;
    const diagnosticCredentialRequired = readonly && !credentialConfigured("diagnostic");
    const executionCredentialRequired = changeEnabled.checked && !credentialConfigured("execution");
    username.required = diagnosticCredentialRequired;
    password.required = diagnosticCredentialRequired;
    executionUsername.required = executionCredentialRequired;
    executionPassword.required = executionCredentialRequired;
    configureEndpoint(false);
    document.getElementById("test-target-connection").hidden = !readonly;
    accessSummary.textContent = changeEnabled.checked
      ? "允许受控变更"
      : readonly ? "只读直连" : "仅监控";
    setCredentialMode();
  }

  function credentialConfigured(kind) {
    const status = editingTarget?.[`${kind}_credential`];
    return Boolean(
      status?.configured
      || editingTarget?.[`${kind}_credential_configured`]
    );
  }

  function setCredentialMode() {
    const diagnosticStored = credentialConfigured("diagnostic");
    const executionStored = credentialConfigured("execution");
    username.placeholder = diagnosticStored ? "留空则不更换现有凭据" : "请输入只读诊断用户名";
    password.placeholder = diagnosticStored ? "留空则不更换现有凭据" : "请输入只读诊断密码";
    executionUsername.placeholder = executionStored ? "留空则不更换现有执行凭据" : "启用受控变更时必须配置";
    executionPassword.placeholder = executionStored ? "留空则不更换现有执行凭据" : "用户名和密码必须同时填写";
    document.getElementById("target-credential-note").textContent = editingTarget
      ? "已保存的凭据不会回显；对应用户名和密码都留空表示保持不变，同时填写才会轮换。"
      : "凭据将写入 AIOps 加密凭据存储，列表和详情不会返回密码明文。";
  }

  async function openCreate() {
    editingTarget = null;
    form.reset();
    dbType.disabled = false;
    dbType.value = "ORACLE";
    oracleScope.value = "";
    configureEndpoint();
    readonlyEnabled.checked = false;
    changeEnabled.checked = false;
    setCredentialMode();
    toggleAccessFields();
    document.getElementById("target-dialog-title").textContent = "新增运维目标";
    submit.textContent = "创建目标";
    dialog.showModal();
    try {
      await loadMonitorEditor();
    } catch (error) {
      monitorBindings.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`;
      result.dataset.tone = "bad";
      result.textContent = "监控源读取失败，暂时不能创建 Target。";
    }
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
      form.elements.importance_level.value = target.importance_level;
      readonlyEnabled.checked = Boolean(target.readonly_connection_enabled);
      changeEnabled.checked = Boolean(target.controlled_change_enabled);
      form.elements.host.value = target.endpoint?.host || "";
      form.elements.port.value = target.endpoint?.port || "";
      form.elements.tls_enabled.checked = Boolean(target.endpoint?.tls_enabled);
      if (target.db_type === "ORACLE") service.value = target.endpoint?.service || "";
      else database.value = target.endpoint?.database || "";
      oracleScope.value = target.oracle_container_scope || "";
      oraclePdbName.value = target.oracle_pdb_name || "";
      dbType.disabled = true;
      setCredentialMode();
      toggleAccessFields();
      document.getElementById("target-dialog-title").textContent = "编辑运维目标";
      submit.textContent = "保存修改";
      clearResult();
      dialog.showModal();
      await loadMonitorEditor(target.target_id);
      form.elements.display_name.focus();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  function endpointPayload() {
    if (!readonlyEnabled.checked) return null;
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

  function executionCredentialPayload() {
    return {
      username: executionUsername.value.trim(),
      password: executionPassword.value,
    };
  }

  function oracleContainerPayload() {
    if (!readonlyEnabled.checked || dbType.value !== "ORACLE") return {};
    return {
      oracle_container_scope: oracleScope.value,
      oracle_pdb_name: oracleScope.value === "PDB"
        ? oraclePdbName.value.trim()
        : null,
    };
  }

  function connectionFieldsAreValid() {
    if (!readonlyEnabled.checked) return false;
    if (editingTarget && (!username.value.trim() || !password.value)) {
      result.dataset.tone = "bad";
      result.textContent = "测试连接需要重新输入只读诊断用户名和密码。";
      return false;
    }
    return [
      dbType,
      form.elements.host,
      port,
      dbType.value === "ORACLE" ? service : database,
      ...(dbType.value === "ORACLE"
        ? [oracleScope, ...(oracleScope.value === "PDB" ? [oraclePdbName] : [])]
        : []),
      username,
      password,
    ]
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
        body: JSON.stringify({
          db_type: dbType.value,
          version_code: version.value,
          endpoint: endpointPayload(),
          diagnostic_credential: credentialPayload(),
          ...oracleContainerPayload(),
        }),
      });
      if (!response.ok) {
        const messages = {
          AUTH_FAILED: "数据库身份验证失败，请检查只读用户名和密码。",
          TARGET_UNREACHABLE: "无法连接数据库，请检查主机、端口、Service Name/Database、网络和 TLS。",
          TIMEOUT: "数据库连接超时，请检查防火墙和访问控制。",
          CONNECTION_FAILED: "数据库连接失败，请检查连接参数。",
          ORACLE_CONTAINER_MISMATCH: "实际连接的 Oracle 容器与配置的 CDB/PDB 范围或 PDB Name 不一致。",
          ORACLE_CONTAINER_UNSUPPORTED: "PDB$SEED 不能作为运维目标。",
          UNSUPPORTED_DATABASE_VERSION: "数据库实际版本尚未完成支持验证，不能创建 Target。",
          DATABASE_VERSION_MISMATCH: "数据库实际版本与所选受支持版本不一致。",
        };
        throw new Error(messages[response.error_code] || "数据库连接测试失败。");
      }
      result.dataset.tone = "good";
      const container = response.oracle_container_scope
        ? `，实际容器 ${response.oracle_container_scope}${response.oracle_container_name ? ` / ${response.oracle_container_name}` : ""}`
        : "";
      result.textContent = `连接成功${response.database_version ? `，数据库版本 ${response.database_version}` : ""}${container}`;
    } catch (error) {
      result.dataset.tone = "bad";
      result.textContent = error.message;
    } finally {
      button.disabled = false;
      button.textContent = "测试连接";
    }
  }

  function targetFields() {
    const fields = {
      display_name: form.elements.display_name.value.trim(),
      version_code: version.value,
      environment: form.elements.environment.value,
      db_role: form.elements.db_role.value,
      readonly_connection_enabled: readonlyEnabled.checked,
      controlled_change_enabled: changeEnabled.checked,
      importance_level: Number(form.elements.importance_level.value),
      ...oracleContainerPayload(),
    };
    if (readonlyEnabled.checked) fields.endpoint = endpointPayload();
    return fields;
  }

  async function saveTarget(event) {
    event.preventDefault();
    if (!monitorSelectionIsValid()) return;
    const creating = !editingTarget;
    const rotatingCredential = readonlyEnabled.checked && Boolean(username.value.trim() || password.value);
    const rotatingExecutionCredential = Boolean(
      changeEnabled.checked && (executionUsername.value.trim() || executionPassword.value)
    );
    if (editingTarget && rotatingCredential && (!username.value.trim() || !password.value)) {
      result.dataset.tone = "bad";
      result.textContent = "轮换凭据时必须同时填写用户名和密码。";
      return;
    }
    if (
      rotatingExecutionCredential
      && (!executionUsername.value.trim() || !executionPassword.value)
    ) {
      result.dataset.tone = "bad";
      result.textContent = "配置或轮换变更执行凭据时，必须同时填写用户名和密码。";
      return;
    }
    submit.disabled = true;
    submit.textContent = editingTarget ? "保存中…" : "创建中…";
    let baseSaved = false;
    let versionSavedBeforeCredential = false;
    try {
      if (!editingTarget) {
        const createPayload = {
          ...targetFields(),
          db_type: dbType.value,
          capabilities: {},
        };
        if (readonlyEnabled.checked) createPayload.diagnostic_credential = credentialPayload();
        if (rotatingExecutionCredential) {
          createPayload.execution_credential = executionCredentialPayload();
        }
        const created = await KBotAIOpsAuth.request(`${api}/targets`, {
          method: "POST",
          headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify(createPayload),
        });
        editingTarget = created;
        baseSaved = true;
        await saveSelectedMonitorBinding(created);
      } else {
        let updated = editingTarget;
        const targetUrl = `${api}/targets/${encodeURIComponent(editingTarget.target_id)}`;
        if (rotatingCredential && version.value !== editingTarget.version_code) {
          updated = await KBotAIOpsAuth.request(targetUrl, {
            method: "PATCH",
            headers: { "If-Match": `"rv-${updated.row_version}"` },
            body: JSON.stringify({ version_code: version.value }),
          });
          versionSavedBeforeCredential = true;
        }
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
        if (rotatingExecutionCredential) {
          updated = await KBotAIOpsAuth.request(
            `${api}/targets/${encodeURIComponent(editingTarget.target_id)}/execution-credential:rotate`,
            {
              method: "POST",
              headers: { "If-Match": `"rv-${updated.row_version}"`, "Idempotency-Key": KBotAIOpsAuth.uuid() },
              body: JSON.stringify(executionCredentialPayload()),
            },
          );
        }
        updated = await KBotAIOpsAuth.request(targetUrl, {
          method: "PATCH",
          headers: { "If-Match": `"rv-${updated.row_version}"` },
          body: JSON.stringify({ ...targetFields(), capabilities: editingTarget.capabilities || {} }),
        });
        editingTarget = updated;
        baseSaved = true;
        await saveSelectedMonitorBinding(updated);
      }
      dialog.close();
      shell.toast(creating ? "运维目标及监控映射已创建" : "运维目标及监控映射已更新");
      editingTarget = null;
      await KBotAIOpsPages.reload();
    } catch (error) {
      result.dataset.tone = "bad";
      result.textContent = baseSaved
        ? `Target 基本信息已保存，但监控映射或后续配置失败：${error.message}`
        : versionSavedBeforeCredential
          ? `数据库版本已保存，但后续操作失败：${error.message}`
          : error.message;
      if (baseSaved || versionSavedBeforeCredential) await KBotAIOpsPages.reload();
    } finally {
      submit.disabled = false;
      submit.textContent = editingTarget ? "保存修改" : "创建目标";
    }
  }

  globalThis.KBotAIOpsTargets = { openEdit };
  shell.ready.then(() => {
    document.getElementById("create-target").addEventListener("click", () => void openCreate());
    document.getElementById("close-target-dialog").addEventListener("click", () => dialog.close());
    document.getElementById("cancel-target-dialog").addEventListener("click", () => dialog.close());
    document.getElementById("test-target-connection").addEventListener("click", testConnection);
    dbType.addEventListener("change", () => {
      configureEndpoint();
      configureMonitorSource();
    });
    monitorSource.addEventListener("change", configureMonitorSource);
    discoverMonitorLabels.addEventListener("click", () => void discoverMonitorCandidates());
    monitorBindings.addEventListener("click", (event) => {
      const deleteButton = event.target.closest("[data-monitor-binding-delete]");
      if (deleteButton) {
        void deleteMonitorBinding(deleteButton);
        return;
      }
      const button = event.target.closest("[data-monitor-binding-action]");
      if (button) void commandMonitorBinding(button);
    });
    oracleScope.addEventListener("change", () => configureEndpoint(false));
    readonlyEnabled.addEventListener("change", toggleAccessFields);
    changeEnabled.addEventListener("change", toggleAccessFields);
    form.addEventListener("input", clearResult);
    form.addEventListener("submit", saveTarget);
    const editTargetId = new URLSearchParams(location.search).get("edit");
    if (editTargetId) void openEdit(editTargetId);
  });
})();
