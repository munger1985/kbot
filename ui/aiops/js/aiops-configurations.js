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
    const creatingAlertmanager = isAlertmanager && !editing;
    const receivesWebhook = isAlertmanager || type === "ZABBIX";
    endpoint.placeholder = presentation.placeholder;
    endpoint.required = !isAlertmanager;
    document.getElementById("source-endpoint-help").textContent = presentation.help;
    document.getElementById("source-token-field").hidden = isAlertmanager && !endpoint.value.trim();
    document.getElementById("source-webhook-field").hidden = !receivesWebhook || creatingAlertmanager;
    document.getElementById("source-webhook-create-note").hidden = !creatingAlertmanager;
    document.getElementById("source-webhook-key-section").hidden = !(receivesWebhook && editing);
    document.getElementById("source-tenant-field").hidden = type !== "LOKI";
    document.getElementById("test-diagnostic-source").hidden = isAlertmanager && !endpoint.value.trim();
    if (!editing) {
      document.getElementById("save-diagnostic-source").textContent = creatingAlertmanager
        ? "创建并生成接入凭据"
        : "创建诊断源";
    }
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
    resetSourceDialogLayout(form);
    resetWebhookSecretControl(form);
    form.elements.source_type.disabled = false;
    resetWebhookKeyResult();
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
      resetSourceDialogLayout(form);
      resetWebhookSecretControl(form);
      form.elements.display_name.value = source.display_name;
      form.elements.source_type.disabled = false;
      form.elements.source_type.value = source.source_type;
      form.elements.source_type.disabled = true;
      form.elements.endpoint.value = source.endpoint || "";
      form.elements.tenant_id.value = source.config?.tenant_id || "";
      renderSourceType(form);
      resetWebhookKeyResult();
      document.getElementById("rotate-source-webhook-key").textContent = source.webhook_configured
        ? "轮换 Webhook Key"
        : "生成 Webhook Key";
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
    const creatingAlertmanager = !editing && form.elements.source_type.value === "ALERTMANAGER";
    let generatedSecret = "";
    try {
      if (creatingAlertmanager) {
        generatedSecret = createWebhookSecretValue();
        form.elements.webhook_secret.value = generatedSecret;
      }
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
      if (creatingAlertmanager) {
        editing = saved;
        form.elements.webhook_secret.value = "";
        let generatedKey = "";
        let generationError = "";
        try {
          const rotation = await requestWebhookKey(saved);
          generatedKey = rotation.webhook_key;
        } catch (error) {
          generationError = error.message;
        }
        showWebhookOnboarding(form, generatedSecret, generatedKey, generationError);
        await KBotAIOpsPages.reload();
        return;
      }
      document.getElementById("diagnostic-source-dialog").close();
      shell.toast(
        saved.connectivity_check_pending
          ? "诊断源已保存，正在执行连通性检查"
          : editing ? "诊断源已更新" : "诊断源已创建"
      );
      editing = null;
      await KBotAIOpsPages.reload();
    } catch (error) {
      showResult("source-test-result", error.message);
    } finally {
      button.disabled = false;
      if (document.getElementById("source-webhook-onboarding").hidden) {
        button.textContent = editing ? "保存修改" : "创建诊断源";
      }
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

  function createWebhookSecretValue() {
    if (!globalThis.crypto?.getRandomValues) {
      throw new Error("当前浏览器不支持安全随机数生成，请更换现代浏览器。");
    }
    const bytes = new Uint8Array(32);
    globalThis.crypto.getRandomValues(bytes);
    return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  }

  function generateWebhookSecret() {
    let secret;
    try {
      secret = createWebhookSecretValue();
    } catch (error) {
      showResult("source-test-result", error.message, "bad");
      return;
    }
    const input = document.getElementById("source-webhook-secret");
    input.value = secret;
    input.type = "text";
    document.getElementById("copy-source-webhook-secret").textContent = "复制";
    showResult("source-test-result", "新 Webhook Secret 已生成。复制后保存修改，并立即更新监控部署配置。", "good");
  }

  function resetSourceDialogLayout(form) {
    form.querySelector(".ops-form").hidden = false;
    form.querySelector(".ops-dialog-body > .ops-dialog-note").hidden = false;
    document.getElementById("source-webhook-onboarding").hidden = true;
    document.getElementById("source-created-webhook-secret").value = "";
    document.getElementById("source-created-webhook-key").value = "";
    document.getElementById("source-created-webhook-ini").value = "";
    [
      ["copy-created-webhook-secret", "复制"],
      ["copy-created-webhook-key", "复制"],
      ["copy-created-webhook-ini", "复制配置"],
    ].forEach(([id, label]) => {
      const button = document.getElementById(id);
      button.textContent = label;
      button.disabled = false;
    });
    document.getElementById("source-webhook-onboarding-note").textContent = "关闭弹窗后无法再次查看；遗失时需要轮换对应凭据。";
    document.getElementById("test-diagnostic-source").hidden = false;
    document.getElementById("save-diagnostic-source").hidden = false;
    document.getElementById("cancel-diagnostic-source").textContent = "取消";
  }

  function showWebhookOnboarding(form, secret, key, errorMessage = "") {
    form.querySelector(".ops-form").hidden = true;
    form.querySelector(".ops-dialog-body > .ops-dialog-note").hidden = true;
    document.getElementById("source-test-result").textContent = "";
    document.getElementById("source-created-webhook-secret").value = secret;
    document.getElementById("source-created-webhook-key").value = key;
    document.getElementById("source-created-webhook-ini").value = [
      `kbot_webhook_key = ${key || "生成失败，请编辑诊断源后重试"}`,
      `kbot_webhook_secret = ${secret}`,
    ].join("\n");
    document.getElementById("copy-created-webhook-key").disabled = !key;
    document.getElementById("copy-created-webhook-ini").disabled = !key;
    document.getElementById("source-webhook-onboarding-note").textContent = errorMessage
      ? `诊断源已创建，但 Webhook Key 生成失败：${errorMessage}。请先复制 Secret，关闭后编辑诊断源重试。`
      : "Secret和Key只显示一次；复制配置后即可关闭。";
    document.getElementById("source-webhook-onboarding").hidden = false;
    document.getElementById("test-diagnostic-source").hidden = true;
    document.getElementById("save-diagnostic-source").hidden = true;
    document.getElementById("cancel-diagnostic-source").textContent = "完成";
    document.getElementById("diagnostic-source-dialog-title").textContent = "Alertmanager 接入凭据";
  }

  function resetWebhookSecretControl(form) {
    form.elements.webhook_secret.type = "password";
    document.getElementById("copy-source-webhook-secret").textContent = "复制";
  }

  async function copyWebhookSecret() {
    const input = document.getElementById("source-webhook-secret");
    if (!input.value) {
      showResult("source-test-result", "请先生成或填写 Webhook Secret。", "bad");
      return;
    }
    try {
      await navigator.clipboard.writeText(input.value);
    } catch (_) {
      input.select();
      document.execCommand("copy");
      input.setSelectionRange(0, 0);
    }
    document.getElementById("copy-source-webhook-secret").textContent = "已复制";
    shell.toast("Webhook Secret 已复制");
  }

  function resetWebhookKeyResult() {
    const result = document.getElementById("source-webhook-key-result");
    result.hidden = true;
    document.getElementById("source-webhook-key-value").value = "";
    document.getElementById("source-webhook-key-note").textContent = "关闭弹窗后无法再次查看；遗失时需要重新轮换。";
    document.getElementById("copy-source-webhook-key").textContent = "复制";
  }

  function requestWebhookKey(source) {
    return KBotAIOpsAuth.request(`${api}/diagnostic-sources/${encodeURIComponent(source.source_id)}/webhook-key:rotate`, {
      method: "POST",
      headers: {
        "If-Match": `"rv-${source.row_version}"`,
        "Idempotency-Key": KBotAIOpsAuth.uuid(),
      },
    });
  }

  async function rotateWebhookKey() {
    if (!editing || !["ALERTMANAGER", "ZABBIX"].includes(editing.source_type)) return;
    const form = document.getElementById("diagnostic-source-form");
    if (form.elements.webhook_secret.value) {
      showResult("source-test-result", "请先保存新的 Webhook Secret，再生成 Webhook Key。", "bad");
      return;
    }
    if (editing.webhook_configured && !confirm("轮换后必须及时更新监控部署配置。确认继续吗？")) return;
    const button = document.getElementById("rotate-source-webhook-key");
    button.disabled = true;
    button.textContent = editing.webhook_configured ? "轮换中…" : "生成中…";
    try {
      const result = await requestWebhookKey(editing);
      document.getElementById("source-webhook-key-value").value = result.webhook_key;
      document.getElementById("source-webhook-key-result").hidden = false;
      document.getElementById("source-webhook-key-note").textContent = result.previous_key_expires_at
        ? `旧Key可使用到 ${shell.fmt(result.previous_key_expires_at)}；请在此前重新执行部署脚本。`
        : "关闭弹窗后无法再次查看；请立即复制到aiops-stack.ini。";
      editing = await KBotAIOpsAuth.request(`${api}/diagnostic-sources/${encodeURIComponent(editing.source_id)}`);
      button.textContent = "再次轮换 Webhook Key";
      shell.toast("Webhook Key 已生成，请立即复制");
    } catch (error) {
      showResult("source-test-result", error.message, "bad");
      button.textContent = editing.webhook_configured ? "轮换 Webhook Key" : "生成 Webhook Key";
    } finally {
      button.disabled = false;
    }
  }

  async function copyWebhookKey() {
    const input = document.getElementById("source-webhook-key-value");
    if (!input.value) return;
    try {
      await navigator.clipboard.writeText(input.value);
    } catch (_) {
      input.select();
      document.execCommand("copy");
      input.setSelectionRange(0, 0);
    }
    document.getElementById("copy-source-webhook-key").textContent = "已复制";
    shell.toast("Webhook Key 已复制");
  }

  async function copyCreatedWebhookValue(inputId, buttonId, message) {
    const input = document.getElementById(inputId);
    if (!input.value) return;
    try {
      await navigator.clipboard.writeText(input.value);
    } catch (_) {
      input.select();
      document.execCommand("copy");
      if (input.setSelectionRange) input.setSelectionRange(0, 0);
    }
    document.getElementById(buttonId).textContent = "已复制";
    shell.toast(message);
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
      document.getElementById("generate-source-webhook-secret").addEventListener("click", generateWebhookSecret);
      document.getElementById("copy-source-webhook-secret").addEventListener("click", copyWebhookSecret);
      document.getElementById("rotate-source-webhook-key").addEventListener("click", rotateWebhookKey);
      document.getElementById("copy-source-webhook-key").addEventListener("click", copyWebhookKey);
      document.getElementById("copy-created-webhook-secret").addEventListener("click", () => copyCreatedWebhookValue("source-created-webhook-secret", "copy-created-webhook-secret", "Webhook Secret 已复制"));
      document.getElementById("copy-created-webhook-key").addEventListener("click", () => copyCreatedWebhookValue("source-created-webhook-key", "copy-created-webhook-key", "Webhook Key 已复制"));
      document.getElementById("copy-created-webhook-ini").addEventListener("click", () => copyCreatedWebhookValue("source-created-webhook-ini", "copy-created-webhook-ini", "INI 凭据配置已复制"));
      document.getElementById("source-type").addEventListener("change", (event) => {
        renderSourceType(event.target.form);
      });
      document.getElementById("source-endpoint").addEventListener("input", (event) => {
        renderSourceType(event.target.form);
      });
    } else if (page === "inspection-plans") {
      closeButtons(document.getElementById("inspection-plan-dialog"));
      document.getElementById("create-inspection-plan").addEventListener("click", openPlanCreate);
      document.getElementById("inspection-plan-form").addEventListener("submit", savePlan);
    }
  });
})();
