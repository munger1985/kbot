(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let editing = null;
  let inspectionAgents = [];

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

  const weekdayLabels = {
    "0": "周日",
    "1": "周一",
    "2": "周二",
    "3": "周三",
    "4": "周四",
    "5": "周五",
    "6": "周六",
  };

  const intervalSchedules = {
    MINUTES_15: { cron: "*/15 * * * *", summary: "每 15 分钟执行" },
    MINUTES_30: { cron: "*/30 * * * *", summary: "每 30 分钟执行" },
    HOURS_1: { cron: "0 * * * *", summary: "每小时整点执行" },
    HOURS_2: { cron: "0 */2 * * *", summary: "每 2 小时整点执行" },
    HOURS_6: { cron: "0 */6 * * *", summary: "每 6 小时整点执行" },
    HOURS_12: { cron: "0 */12 * * *", summary: "每 12 小时整点执行" },
  };

  function scheduleTime(value) {
    const match = /^(\d{2}):(\d{2})$/.exec(value || "");
    if (!match) throw new Error("请选择有效的执行时间。");
    return { hour: Number(match[1]), minute: Number(match[2]), label: value };
  }

  function checkedWeekdays(form) {
    return Array.from(form.querySelectorAll('input[name="weekdays"]:checked'))
      .map((input) => input.value);
  }

  function weekdaySummary(days) {
    const ordered = ["1", "2", "3", "4", "5", "6", "0"];
    const positions = days.map((day) => ordered.indexOf(day));
    const consecutive = positions.every(
      (position, index) => index === 0 || position === positions[index - 1] + 1,
    );
    if (days.length === 7) return "每天";
    if (days.length >= 3 && consecutive) {
      return `每${weekdayLabels[days[0]]}至${weekdayLabels[days.at(-1)]}`;
    }
    return `每${days.map((day) => weekdayLabels[day]).join("、")}`;
  }

  function buildSchedule(form) {
    const mode = form.querySelector('input[name="schedule_mode"]:checked')?.value;
    if (!mode) {
      return {
        type: form.elements.schedule_type.value,
        cron: form.elements.cron_expression.value,
        summary: "保持现有高级调度",
      };
    }
    if (mode === "DAILY") {
      const time = scheduleTime(form.elements.daily_time.value);
      return { type: "DAILY", cron: `${time.minute} ${time.hour} * * *`, summary: `每天 ${time.label} 执行` };
    }
    if (mode === "WEEKLY") {
      const time = scheduleTime(form.elements.weekly_time.value);
      const days = checkedWeekdays(form);
      if (!days.length) throw new Error("每周巡检至少选择一个执行日期。");
      return { type: "WEEKLY", cron: `${time.minute} ${time.hour} * * ${days.join(",")}`, summary: `${weekdaySummary(days)} ${time.label} 执行` };
    }
    if (mode === "MONTHLY") {
      const time = scheduleTime(form.elements.monthly_time.value);
      const day = Number(form.elements.month_day.value);
      if (!Number.isInteger(day) || day < 1 || day > 28) throw new Error("请选择每月执行日期。");
      return { type: "CRON", cron: `${time.minute} ${time.hour} ${day} * *`, summary: `每月 ${day} 日 ${time.label} 执行` };
    }
    const interval = intervalSchedules[form.elements.interval_preset.value];
    if (!interval) throw new Error("请选择有效的重复间隔。");
    return { type: "CRON", cron: interval.cron, summary: interval.summary };
  }

  function renderSchedule(form) {
    const mode = form.querySelector('input[name="schedule_mode"]:checked')?.value;
    form.querySelectorAll("[data-schedule-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.schedulePanel !== mode;
    });
    document.getElementById("inspection-legacy-schedule").hidden = Boolean(mode);
    try {
      const schedule = buildSchedule(form);
      form.elements.schedule_type.value = schedule.type;
      form.elements.cron_expression.value = schedule.cron;
      document.getElementById("inspection-schedule-summary").textContent = schedule.summary;
    } catch (error) {
      document.getElementById("inspection-schedule-summary").textContent = error.message;
    }
    const timezone = form.elements.timezone;
    document.getElementById("inspection-timezone-summary").textContent =
      timezone.selectedOptions[0]?.textContent || timezone.value;
  }

  function selectScheduleMode(form, mode, locked) {
    form.querySelectorAll('input[name="schedule_mode"]').forEach((input) => {
      input.checked = input.value === mode;
      input.disabled = Boolean(locked && input.value !== mode);
    });
  }

  function parseCronValues(value, minimum, maximum) {
    const values = new Set();
    for (const part of value.split(",")) {
      const range = /^(\d+)(?:-(\d+))?$/.exec(part);
      if (!range) return null;
      const start = Number(range[1]);
      const end = Number(range[2] || range[1]);
      if (start < minimum || end > maximum || start > end) return null;
      for (let current = start; current <= end; current += 1) values.add(current);
    }
    return values;
  }

  function setTime(input, hour, minute) {
    input.value = `${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}`;
  }

  function ensureSelectOption(select, value, label = value) {
    if (!Array.from(select.options).some((option) => option.value === value)) {
      select.add(new Option(label, value));
    }
    select.value = value;
  }

  function cronInteger(value, minimum, maximum) {
    if (!/^\d+$/.test(value || "")) return null;
    const parsed = Number(value);
    return parsed >= minimum && parsed <= maximum ? parsed : null;
  }

  function hydrateScheduleBuilder(form, plan) {
    const fields = plan.cron_expression.trim().split(/\s+/);
    const minute = cronInteger(fields[0], 0, 59);
    const hour = cronInteger(fields[1], 0, 23);
    let mode = null;
    if (fields.length === 5 && minute !== null && hour !== null && plan.schedule_type === "DAILY" && fields.slice(2).every((field) => field === "*")) {
      mode = "DAILY";
      setTime(form.elements.daily_time, hour, minute);
    } else if (fields.length === 5 && minute !== null && hour !== null && plan.schedule_type === "WEEKLY" && fields[2] === "*" && fields[3] === "*" && fields[4] !== "*") {
      const days = parseCronValues(fields[4], 0, 7);
      if (days) {
        mode = "WEEKLY";
        setTime(form.elements.weekly_time, hour, minute);
        form.querySelectorAll('input[name="weekdays"]').forEach((input) => {
          input.checked = days.has(Number(input.value)) || (input.value === "0" && days.has(7));
        });
      }
    } else if (fields.length === 5 && minute !== null && hour !== null && plan.schedule_type === "CRON" && cronInteger(fields[2], 1, 28) !== null && fields[3] === "*" && fields[4] === "*") {
      const day = cronInteger(fields[2], 1, 28);
      if (day >= 1 && day <= 28) {
        mode = "MONTHLY";
        form.elements.month_day.value = String(day);
        setTime(form.elements.monthly_time, hour, minute);
      }
    } else if (plan.schedule_type === "CRON") {
      const interval = Object.entries(intervalSchedules).find(([, item]) => item.cron === plan.cron_expression);
      if (interval) {
        mode = "INTERVAL";
        form.elements.interval_preset.value = interval[0];
      }
    }
    selectScheduleMode(form, mode, true);
    ensureSelectOption(form.elements.timezone, plan.timezone);
    ensureSelectOption(
      form.elements.timeout_seconds,
      String(plan.timeout_seconds),
      `自定义 · ${plan.timeout_seconds} 秒`,
    );
    renderSchedule(form);
  }

  function resetScheduleBuilder(form) {
    selectScheduleMode(form, "DAILY", false);
    form.elements.daily_time.value = "02:00";
    form.elements.weekly_time.value = "09:00";
    form.elements.monthly_time.value = "09:00";
    form.elements.month_day.value = "1";
    form.elements.interval_preset.value = "MINUTES_15";
    form.querySelectorAll('input[name="weekdays"]').forEach((input) => {
      input.checked = ["1", "2", "3", "4", "5"].includes(input.value);
    });
    form.elements.timezone.value = "Asia/Shanghai";
    renderSchedule(form);
  }

  function planPayload(form, create) {
    const schedule = buildSchedule(form);
    form.elements.schedule_type.value = schedule.type;
    form.elements.cron_expression.value = schedule.cron;
    const payload = {
      display_name: form.elements.display_name.value.trim(),
      agent_id: form.elements.agent_id.value,
      cron_expression: schedule.cron,
      timezone: form.elements.timezone.value,
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
    resetScheduleBuilder(form);
    form.elements.template_id.value = "database_daily";
    form.elements.template_version.value = "1.0.0";
    form.elements.schedule_resolver_version.value = "1.0.0";
    form.elements.timeout_seconds.value = 1800;
    document.getElementById("inspection-plan-dialog-title").textContent = "新增巡检计划";
    document.getElementById("save-inspection-plan").textContent = "创建并启用";
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
      hydrateScheduleBuilder(form, plan);
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
      agent_id: plan.agent_id,
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
      shell.toast(editing ? "巡检计划已更新" : "巡检计划已创建并启用");
      editing = null;
      await KBotAIOpsPages.reload();
    } catch (error) {
      showResult("inspection-plan-result", error.message);
    } finally {
      button.disabled = false;
      button.textContent = editing ? "保存修改" : "创建并启用";
    }
  }

  function openEdit(page, resourceId) {
    if (page === "diagnostic-sources") return openSourceEdit(resourceId);
    if (page === "inspection-plans") return openPlanEdit(resourceId);
    return Promise.resolve();
  }

  globalThis.KBotAIOpsConfigurations = { openEdit };
  shell.ready.then(async () => {
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
      const planDialog = document.getElementById("inspection-plan-dialog");
      const planForm = document.getElementById("inspection-plan-form");
      try {
        const rows = await KBotAIOpsAuth.request(`${api}/agents`);
        inspectionAgents = (Array.isArray(rows) ? rows : []).filter((agent) => agent.status === "ACTIVE");
        planForm.elements.agent_id.replaceChildren(
          new Option("请选择已启用的 DBA Agent", ""),
          ...inspectionAgents.map((agent) => new Option(
            `${agent.display_name} · ${agent.target_ids?.length || 0} 个 Target`,
            agent.agent_id,
          )),
        );
        document.getElementById("plan-agent-help").textContent = inspectionAgents.length
          ? "计划触发时，将使用该 Agent 当前发布版本关联的全部 Target、模型和策略。"
          : "当前没有可用 Agent，请先创建并启用至少一名 DBA Agent。";
      } catch (error) {
        planForm.elements.agent_id.replaceChildren(new Option("Agent 读取失败", ""));
        document.getElementById("plan-agent-help").textContent = error.message;
      }
      closeButtons(planDialog);
      const monthDay = planForm.elements.month_day;
      monthDay.replaceChildren(...Array.from({ length: 28 }, (_, index) => {
        const day = index + 1;
        return new Option(`${day} 日`, String(day));
      }));
      document.getElementById("create-inspection-plan").addEventListener("click", openPlanCreate);
      planForm.addEventListener("submit", savePlan);
      planForm.querySelectorAll('input[name="schedule_mode"], input[name="weekdays"], input[type="time"], select[name="month_day"], select[name="interval_preset"], select[name="timezone"]').forEach((control) => {
        control.addEventListener("change", () => renderSchedule(planForm));
      });
    }
  });
})();
