(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let editing = null;

  function showResult(id, message, tone = "bad") {
    const element = document.getElementById(id);
    if (!element) return;
    element.textContent = message;
    element.dataset.tone = tone;
  }

  function closeButtons(dialog) {
    dialog.querySelectorAll("[data-close-dialog]").forEach((button) => {
      button.addEventListener("click", () => dialog.close());
    });
  }

  const sourcePresentation = {
    PROMETHEUS: {
      placeholder: "http://prometheus.internal:9090",
      help: "Prometheus HTTP API 地址，必须能从 KBot 服务访问。",
      capabilities: ["指标时序查询", "活动告警查询"],
    },
    ALERTMANAGER: {
      placeholder: "http://alertmanager.internal:9093（可选）",
      help: "只接收 Webhook 时可以留空；填写后可同时检查 Alertmanager 就绪状态。",
      capabilities: ["告警事件接收"],
    },
    LOKI: {
      placeholder: "http://loki.internal:3100",
      help: "Loki HTTP API 地址，必须能从 KBot 服务访问。",
      capabilities: ["日志查询"],
    },
    ZABBIX: {
      placeholder: "https://zabbix.internal/api_jsonrpc.php",
      help: "请输入完整的 Zabbix JSON-RPC API 地址。",
      capabilities: ["告警事件接收", "活动事件查询", "指标时序查询"],
    },
    OEM: {
      placeholder: "https://oem.internal/em/websvcs/restful",
      help: "Oracle Enterprise Manager REST API 根地址。",
      capabilities: ["事件查询", "指标时序查询"],
    },
  };

  function renderSourceType(form) {
    const type = form.elements.source_type.value;
    const presentation = sourcePresentation[type];
    const endpoint = form.elements.endpoint;
    const isAlertmanager = type === "ALERTMANAGER";
    const receivesWebhook = isAlertmanager || type === "ZABBIX";
    endpoint.placeholder = presentation.placeholder;
    endpoint.required = !isAlertmanager;
    document.getElementById("source-endpoint-help").textContent = presentation.help;
    document.getElementById("source-token-field").hidden = false;
    document.getElementById("source-webhook-field").hidden = !receivesWebhook;
    document.getElementById("source-target-label-field").hidden = !isAlertmanager;
    document.getElementById("source-tenant-field").hidden = type !== "LOKI";
    const list = document.getElementById("source-capability-list");
    list.replaceChildren(...presentation.capabilities.map((label) => {
      const item = document.createElement("span");
      item.className = "ops-capability";
      item.textContent = label;
      return item;
    }));
  }

  function sourcePayload(form, includeIdentity = true, testing = false) {
    const type = form.elements.source_type.value;
    const endpoint = form.elements.endpoint.value.trim();
    const webhookSecret = form.elements.webhook_secret.value;
    if (
      type === "ALERTMANAGER"
      && !endpoint
      && !webhookSecret
      && (testing || !editing?.webhook_secret?.configured)
    ) {
      throw new Error("Alertmanager 必须填写访问地址或 Webhook Secret。");
    }
    const credentials = {};
    if (form.elements.token.value) {
      credentials.token = form.elements.token.value;
    }
    const payload = {
      display_name: form.elements.display_name.value.trim(),
      endpoint: endpoint || null,
      config: {},
    };
    if (type === "ALERTMANAGER") {
      payload.config.target_label = form.elements.target_label.value.trim() || "instance";
    }
    if (type === "LOKI" && form.elements.tenant_id.value.trim()) {
      payload.config.tenant_id = form.elements.tenant_id.value.trim();
    }
    if (Object.keys(credentials).length) payload.credentials = credentials;
    if (["ALERTMANAGER", "ZABBIX"].includes(type) && webhookSecret) {
      payload.webhook_credentials = { webhook_secret: webhookSecret };
    }
    if (includeIdentity) payload.source_type = type;
    return payload;
  }

  function openSourceCreate() {
    editing = null;
    const form = document.getElementById("diagnostic-source-form");
    form.reset();
    form.elements.source_type.disabled = false;
    form.elements.target_label.value = "instance";
    renderSourceType(form);
    document.getElementById("diagnostic-source-dialog-title").textContent = "新增诊断源";
    document.getElementById("save-diagnostic-source").textContent = "创建诊断源";
    showResult("source-test-result", "", "");
    document.getElementById("diagnostic-source-dialog").showModal();
    form.elements.display_name.focus();
  }

  async function openSourceEdit(sourceId) {
    try {
      const source = await KBotAIOpsAuth.request(`${api}/diagnostic-sources/${encodeURIComponent(sourceId)}`);
      editing = source;
      const form = document.getElementById("diagnostic-source-form");
      form.reset();
      form.elements.display_name.value = source.display_name;
      form.elements.source_type.disabled = false;
      form.elements.source_type.value = source.source_type;
      form.elements.source_type.disabled = true;
      form.elements.endpoint.value = source.endpoint || "";
      form.elements.target_label.value = source.config?.target_label || "instance";
      form.elements.tenant_id.value = source.config?.tenant_id || "";
      renderSourceType(form);
      document.getElementById("diagnostic-source-dialog-title").textContent = "编辑诊断源";
      document.getElementById("save-diagnostic-source").textContent = "保存修改";
      showResult("source-test-result", "", "");
      document.getElementById("diagnostic-source-dialog").showModal();
      form.elements.display_name.focus();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function saveSource(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = document.getElementById("save-diagnostic-source");
    button.disabled = true;
    try {
      const payload = sourcePayload(form, !editing);
      const saved = await KBotAIOpsAuth.request(
        editing ? `${api}/diagnostic-sources/${encodeURIComponent(editing.source_id)}` : `${api}/diagnostic-sources`,
        {
          method: editing ? "PATCH" : "POST",
          headers: editing
            ? { "If-Match": `"rv-${editing.row_version}"` }
            : { "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify(payload),
        },
      );
      document.getElementById("diagnostic-source-dialog").close();
      shell.toast(
        saved.health_check_pending
          ? "诊断源已保存，正在执行健康检查"
          : editing ? "诊断源已更新" : "诊断源已创建"
      );
      editing = null;
      await KBotAIOpsPages.reload();
    } catch (error) {
      showResult("source-test-result", error.message);
    } finally {
      button.disabled = false;
      button.textContent = editing ? "保存修改" : "创建诊断源";
    }
  }

  async function testSource() {
    const form = document.getElementById("diagnostic-source-form");
    if (!form.reportValidity()) return;
    const button = document.getElementById("test-diagnostic-source");
    button.disabled = true;
    button.textContent = "测试中…";
    showResult("source-test-result", "正在验证诊断源 API 和认证…", "");
    try {
      const response = await KBotAIOpsAuth.request(`${api}/diagnostic-sources/test-connection`, {
        method: "POST",
        body: JSON.stringify(sourcePayload(form, true, true)),
      });
      if (!response.ok) throw new Error(response.error_code || "诊断源连接测试失败。");
      const count = (response.discovered_capabilities || []).length;
      showResult("source-test-result", `连接成功，已发现 ${count} 项系统能力。`, "good");
    } catch (error) {
      showResult("source-test-result", error.message);
    } finally {
      button.disabled = false;
      button.textContent = "测试连接";
    }
  }

  function policyRules(form) {
    return {
      schema_version: "ops.policy.v1",
      allow_agent_execution: form.elements.allow_agent_execution.checked,
      readonly_database_enabled: form.elements.readonly_database_enabled.checked,
      max_risk_level: form.elements.max_risk_level.value,
      allowed_action_types: form.elements.allowed_action_types.value.split("\n").map((item) => item.trim()).filter(Boolean),
      auto_observe_min_severity: form.elements.auto_observe_min_severity.value,
      alert_cooldown_seconds: Number(form.elements.alert_cooldown_seconds.value),
    };
  }

  function openPolicyCreate() {
    editing = null;
    const form = document.getElementById("policy-form");
    form.reset();
    form.elements.policy_key.disabled = false;
    form.elements.readonly_database_enabled.checked = true;
    form.elements.alert_cooldown_seconds.value = 900;
    document.getElementById("policy-dialog-title").textContent = "新增执行策略";
    document.getElementById("policy-version-note").textContent = "创建后生成 DRAFT 版本，激活需在后续状态操作中明确执行。";
    document.getElementById("save-policy").textContent = "创建策略";
    showResult("policy-result", "", "");
    document.getElementById("policy-dialog").showModal();
    form.elements.policy_key.focus();
  }

  async function openPolicyVersion(policyId) {
    try {
      const policy = await KBotAIOpsAuth.request(`${api}/policies/${encodeURIComponent(policyId)}`);
      editing = policy;
      const form = document.getElementById("policy-form");
      form.reset();
      form.elements.policy_key.disabled = false;
      form.elements.policy_key.value = policy.policy_key;
      form.elements.policy_key.disabled = true;
      form.elements.display_name.value = policy.display_name;
      form.elements.readonly_database_enabled.checked = policy.rules.readonly_database_enabled !== false;
      form.elements.allow_agent_execution.checked = Boolean(policy.rules.allow_agent_execution);
      form.elements.max_risk_level.value = policy.rules.max_risk_level || "LOW";
      form.elements.auto_observe_min_severity.value = policy.rules.auto_observe_min_severity || "CRITICAL";
      form.elements.alert_cooldown_seconds.value = policy.rules.alert_cooldown_seconds ?? 900;
      form.elements.allowed_action_types.value = (policy.rules.allowed_action_types || []).join("\n");
      document.getElementById("policy-dialog-title").textContent = "基于当前策略创建新版本";
      document.getElementById("policy-version-note").textContent = `当前为版本 ${policy.version_no}。保存不会修改原版本，而会创建版本 ${policy.version_no + 1} 的 DRAFT。`;
      document.getElementById("save-policy").textContent = "创建新版本";
      showResult("policy-result", "", "");
      document.getElementById("policy-dialog").showModal();
      form.elements.display_name.focus();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  async function savePolicy(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = document.getElementById("save-policy");
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${api}/policies`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify({
          policy_key: editing ? editing.policy_key : form.elements.policy_key.value.trim(),
          display_name: form.elements.display_name.value.trim(),
          rules: policyRules(form),
        }),
      });
      document.getElementById("policy-dialog").close();
      shell.toast(editing ? "策略新版本已创建" : "执行策略已创建");
      editing = null;
      await KBotAIOpsPages.reload();
    } catch (error) {
      showResult("policy-result", error.message);
    } finally {
      button.disabled = false;
      button.textContent = editing ? "创建新版本" : "创建策略";
    }
  }

  function planPayload(form, create) {
    const payload = {
      display_name: form.elements.display_name.value.trim(),
      cron_expression: form.elements.cron_expression.value.trim(),
      timezone: form.elements.timezone.value.trim(),
      template_id: form.elements.template_id.value.trim(),
      template_version: form.elements.template_version.value.trim(),
      timeout_seconds: Number(form.elements.timeout_seconds.value),
      overlap_policy: form.elements.overlap_policy.value,
      misfire_policy: form.elements.misfire_policy.value,
      schedule_resolver_version: form.elements.schedule_resolver_version.value.trim(),
    };
    if (create) payload.schedule_type = form.elements.schedule_type.value;
    return payload;
  }

  function openPlanCreate() {
    editing = null;
    const form = document.getElementById("inspection-plan-form");
    form.reset();
    form.elements.schedule_type.disabled = false;
    form.elements.cron_expression.value = "0 2 * * *";
    form.elements.timezone.value = "Asia/Shanghai";
    form.elements.template_version.value = "1.0.0";
    form.elements.schedule_resolver_version.value = "v1";
    form.elements.timeout_seconds.value = 1800;
    document.getElementById("inspection-plan-dialog-title").textContent = "新增巡检计划";
    document.getElementById("save-inspection-plan").textContent = "创建计划";
    showResult("inspection-plan-result", "", "");
    document.getElementById("inspection-plan-dialog").showModal();
    form.elements.display_name.focus();
  }

  async function openPlanEdit(planId) {
    try {
      const plan = await KBotAIOpsAuth.request(`${api}/inspection-plans/${encodeURIComponent(planId)}`);
      editing = plan;
      const form = document.getElementById("inspection-plan-form");
      form.reset();
      Object.entries(planPayloadValues(plan)).forEach(([key, value]) => { form.elements[key].value = value; });
      form.elements.schedule_type.disabled = true;
      document.getElementById("inspection-plan-dialog-title").textContent = "编辑巡检计划";
      document.getElementById("save-inspection-plan").textContent = "保存修改";
      showResult("inspection-plan-result", "", "");
      document.getElementById("inspection-plan-dialog").showModal();
      form.elements.display_name.focus();
    } catch (error) {
      shell.toast(error.message);
    }
  }

  function planPayloadValues(plan) {
    return {
      display_name: plan.display_name,
      schedule_type: plan.schedule_type,
      cron_expression: plan.cron_expression,
      timezone: plan.timezone,
      template_id: plan.template_id,
      template_version: plan.template_version,
      timeout_seconds: plan.timeout_seconds,
      overlap_policy: plan.overlap_policy,
      misfire_policy: plan.misfire_policy,
      schedule_resolver_version: plan.schedule_resolver_version,
    };
  }

  async function savePlan(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = document.getElementById("save-inspection-plan");
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(
        editing ? `${api}/inspection-plans/${encodeURIComponent(editing.plan_id)}` : `${api}/inspection-plans`,
        {
          method: editing ? "PATCH" : "POST",
          headers: editing
            ? { "If-Match": `"rv-${editing.row_version}"` }
            : { "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify(planPayload(form, !editing)),
        },
      );
      document.getElementById("inspection-plan-dialog").close();
      shell.toast(editing ? "巡检计划已更新" : "巡检计划已创建");
      editing = null;
      await KBotAIOpsPages.reload();
    } catch (error) {
      showResult("inspection-plan-result", error.message);
    } finally {
      button.disabled = false;
      button.textContent = editing ? "保存修改" : "创建计划";
    }
  }

  function openEdit(page, resourceId) {
    if (page === "diagnostic-sources") return openSourceEdit(resourceId);
    if (page === "policies") return openPolicyVersion(resourceId);
    if (page === "inspection-plans") return openPlanEdit(resourceId);
    return Promise.resolve();
  }

  globalThis.KBotAIOpsConfigurations = { openEdit };
  shell.ready.then(() => {
    const page = document.body.dataset.page;
    if (page === "diagnostic-sources") {
      const dialog = document.getElementById("diagnostic-source-dialog");
      closeButtons(dialog);
      document.getElementById("create-diagnostic-source").addEventListener("click", openSourceCreate);
      document.getElementById("diagnostic-source-form").addEventListener("submit", saveSource);
      document.getElementById("test-diagnostic-source").addEventListener("click", testSource);
      document.getElementById("source-type").addEventListener("change", (event) => {
        renderSourceType(event.target.form);
      });
    } else if (page === "policies") {
      closeButtons(document.getElementById("policy-dialog"));
      document.getElementById("create-policy").addEventListener("click", openPolicyCreate);
      document.getElementById("policy-form").addEventListener("submit", savePolicy);
    } else if (page === "inspection-plans") {
      closeButtons(document.getElementById("inspection-plan-dialog"));
      document.getElementById("create-inspection-plan").addEventListener("click", openPlanCreate);
      document.getElementById("inspection-plan-form").addEventListener("submit", savePlan);
    }
  });
})();
