(function () {
  "use strict";
  const appApi = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let sourceReloadTimer = null;
  let sourceReloadAttempts = 0;
  const configs = {
    targets: { path: "/targets", cols: [["display_name", "目标"], ["importance_level", "重要程度", "importance"], ["db_type", "数据库"], ["_access", "访问模式", "target-access"], ["status", "启用状态", "badge"], ["connectivity_status", "连通性", "badge"], ["observed_status", "观测状态", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "target-actions"]], detail: "target-detail.html?id=" },
    "diagnostic-sources": { path: "/diagnostic-sources", cols: [["display_name", "监控源"], ["source_type", "类型"], ["status", "启用状态", "badge"], ["connectivity_status", "连通性", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "source-actions"]], detail: "diagnostic-source-detail.html?id=" },
    "inspection-plans": { path: "/inspection-plans", cols: [["display_name", "计划"], ["agent_name", "DBA Agent"], ["schedule_type", "调度周期", "schedule"], ["timezone", "时区"], ["status", "状态", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "inspection-actions"]], detail: "inspection-plan-detail.html?id=" },
    reports: { path: "/reports", render: "report-list", detail: "report-detail.html?id=" },
  };
  const resourceId = (item) => item.ops_run_id || item.report_id || item.target_id || item.source_id || item.plan_id;
  function cell(item, [key, , type]) {
    const value = item?.[key];
    if (type === "source-actions") {
      const checking = item.connectivity_check_pending;
      const detailButton = '<button type="button" data-source-action="detail">详情</button>';
      const editButton = '<button type="button" data-source-action="edit">编辑</button>';
      const healthButton = `<button type="button" data-source-action="connectivity" ${checking ? "disabled" : ""}>${checking ? "检查中" : "检查连通性"}</button>`;
      const lifecycleButton = item.status === "ENABLED"
        ? '<button type="button" data-source-action="disable">停用</button>'
        : ["CONNECTED", "DEGRADED"].includes(item.connectivity_status) && !checking
          ? '<button type="button" class="primary" data-source-action="enable">启用</button>'
          : "";
      const deleteButton = `<button type="button" data-source-action="delete" title="${item.status === "DISABLED" ? "删除监控源" : "请先停用监控源"}">删除</button>`;
      return `<div class="ops-actions">${detailButton}${editButton}${healthButton}${lifecycleButton}${deleteButton}</div>`;
    }
    if (type === "target-actions") {
      const checking = item.connectivity_check_pending;
      const detailButton = '<button type="button" data-target-action="detail">详情</button>';
      const editButton = '<button type="button" data-target-action="edit">编辑</button>';
      const checkButton = item.readonly_connection_enabled
        ? `<button type="button" data-target-action="connectivity" ${checking ? "disabled" : ""}>${checking ? "检查中" : "检查连通性"}</button>`
        : "";
      const buttons = item.status === "ENABLED"
        ? ['<button type="button" data-target-action="disable">停用</button>']
        : (!item.readonly_connection_enabled || ["CONNECTED", "DEGRADED"].includes(item.connectivity_status))
          ? ['<button type="button" class="primary" data-target-action="enable">启用</button>']
          : [];
      const deleteButton = `<button type="button" data-target-action="delete" title="${item.status === "DISABLED" ? "删除运维目标" : "请先停用运维目标"}">删除</button>`;
      return `<div class="ops-actions">${detailButton}${editButton}${checkButton}${buttons.join("")}${deleteButton}</div>`;
    }
    if (type === "target-access") {
      const mode = item.controlled_change_enabled
        ? "允许受控变更"
        : item.readonly_connection_enabled ? "只读直连" : "仅监控";
      const detail = item.controlled_change_enabled
        ? "独立执行凭据 · 逐条审批"
        : item.readonly_connection_enabled ? "不允许数据库写操作" : "不连接数据库";
      return `<strong>${mode}</strong><small>${detail}</small>`;
    }
    if (type === "inspection-actions") {
      const action = item.status === "ACTIVE"
        ? "pause"
        : ["PAUSED", "DISABLED"].includes(item.status) ? "activate" : "";
      if (!action) return "—";
      const label = action === "activate" ? "启用" : "暂停";
      const primary = action === "activate" ? ' class="primary"' : "";
      return `<div class="ops-actions"><button type="button"${primary} data-inspection-action="${action}">${label}</button></div>`;
    }
    if (key === "connectivity_status" && item.connectivity_check_pending) {
      return shell.badge("检查中");
    }
    if (
      key === "connectivity_status"
      && Object.prototype.hasOwnProperty.call(item, "readonly_connection_enabled")
      && !item.readonly_connection_enabled
    ) {
      return "仅监控";
    }
    if (type === "schedule") {
      return shell.escape({ DAILY: "每天", WEEKLY: "每周", CRON: "灵活周期" }[value] || value || "—");
    }
    if (type === "importance") {
      const labels = { 1: "可按需忽略", 2: "较低", 3: "普通", 4: "重要", 5: "最重要" };
      return `<strong>L${shell.escape(value)}</strong><small>${shell.escape(labels[value] || "未知")}</small>`;
    }
    if (type === "badge") return shell.badge(value);
    if (type === "date") return shell.escape(shell.fmt(value));
    if (type === "id") return `<code>${shell.escape(shell.short(value))}</code>`;
    return shell.escape(value ?? "—");
  }
  function reportTypeLabel(value) {
    return {
      INCIDENT: "告警诊断",
      PERFORMANCE: "性能分析",
      INSPECTION_DAILY: "日常巡检",
      INSPECTION_WEEKLY: "周度巡检",
      INSPECTION_CUSTOM: "定期巡检",
      COMPARISON: "处置验证",
    }[value] || value || "正式报告";
  }
  const reportDisplayNames = {
    EXECUTIVE_SUMMARY: "执行摘要", SCOPE: "报告范围", ALERT_TIMELINE: "告警时间线",
    INSPECTION_COVERAGE: "巡检覆盖情况", RISK_OVERVIEW: "风险概览", TREND: "趋势分析",
    FINDINGS: "核验发现", ROOT_CAUSE: "根因分析", RECOMMENDATIONS: "处置建议",
    ACTIONS: "已执行动作", EVIDENCE_BOUNDARY: "证据边界", EVIDENCE_APPENDIX: "证据附录",
    CONFIRMED: "已确认", PROBABLE: "很可能", POSSIBLE: "可能",
    INCONCLUSIVE: "证据不足，无法定论", GENERATING: "生成中", READY: "已完成",
    PARTIAL: "部分完成", FAILED: "失败", CRITICAL: "严重", HIGH: "高",
    MEDIUM: "中", LOW: "低", INFO: "提示", MISSING_FINAL_RESULT: "缺少最终诊断结果",
    MISSING_PRIMARY_RUN: "缺少主诊断运行记录", MISSING_FINAL_ARTIFACT: "缺少最终报告产物",
    UNREPORTABLE_FINAL_RESULT: "最终结果暂不支持生成报告",
  };
  function reportDisplayText(value) {
    return String(value ?? "").replace(/[A-Z][A-Z0-9_]{2,}/g, (code) => reportDisplayNames[code] || code);
  }
  function reportStatusBadge(value, displayValue) {
    const status = String(value || "UNKNOWN").toUpperCase();
    const tone = status === "READY" ? "good" : status === "FAILED" ? "bad" : "warn";
    return `<span class="ops-badge ${tone}">${shell.escape(displayValue || reportDisplayText(status))}</span>`;
  }
  function reportRow(item, detail) {
    const publishedAt = item.published_at || item.created_at;
    const period = item.period_start && item.period_end
      ? `${shell.fmt(item.period_start)} 至 ${shell.fmt(item.period_end)}`
      : "未标注报告周期";
    return `<a class="ops-report-row" href="${detail}${encodeURIComponent(item.report_id)}"><div class="ops-report-primary"><div class="ops-report-kicker"><span>${shell.escape(reportTypeLabel(item.report_type))}</span><span>v${shell.escape(item.report_version)}</span></div><strong>${shell.escape(item.title || "正式报告")}</strong><p>${shell.escape(reportDisplayText(item.summary || "报告未提供摘要"))}</p></div><div class="ops-report-meta"><small>实际发布</small><time>${shell.escape(shell.fmt(publishedAt))}</time></div><div class="ops-report-meta ops-report-period"><small>覆盖周期</small><time>${shell.escape(period)}</time></div><div class="ops-report-state">${reportStatusBadge(item.status)}<span>查看报告</span></div></a>`;
  }
  async function renderReportList(cfg) {
    const list = document.getElementById("ops-report-list");
    const count = document.getElementById("ops-report-count");
    try {
      const payload = await KBotAIOpsAuth.request(appApi + cfg.path);
      const items = Array.isArray(payload) ? payload : payload?.items || [];
      count.textContent = `当前范围 ${items.length} 份正式报告`;
      list.innerHTML = items.length
        ? items.map((item) => reportRow(item, cfg.detail)).join("")
        : '<div class="ops-empty">当前范围内暂无正式报告</div>';
    } catch (error) {
      count.textContent = "";
      list.innerHTML = `<div class="ops-empty">${shell.escape(error.message)}</div>`;
    }
  }
  function scheduleSourceReload() {
    if (sourceReloadTimer || sourceReloadAttempts >= 6) return;
    sourceReloadAttempts += 1;
    sourceReloadTimer = setTimeout(() => {
      sourceReloadTimer = null;
      if (document.body.dataset.page === "diagnostic-sources") {
        renderList("diagnostic-sources");
      }
    }, 1500);
  }
  async function runSourceAction(button, item) {
    const action = button.dataset.sourceAction;
    const sourceId = encodeURIComponent(item.source_id);
    if (action === "detail") {
      location.href = `diagnostic-source-detail.html?id=${sourceId}`;
      return;
    }
    if (action === "edit") {
      globalThis.KBotAIOpsConfigurations?.openEdit("diagnostic-sources", item.source_id);
      return;
    }
    if (action === "delete") {
      if (item.status !== "DISABLED") {
        shell.toast("请先停用监控源，再执行删除");
        return;
      }
      const confirmed = confirm(
        `确认删除监控源“${item.display_name || item.source_id}”吗？\n\n`
        + "删除会撤销该监控源的访问凭据和 Webhook 凭据，且不可恢复。"
        + "如果仍有运维目标或运行历史引用它，系统会拒绝删除。",
      );
      if (!confirmed) return;
      button.disabled = true;
      try {
        await KBotAIOpsAuth.request(`${appApi}/diagnostic-sources/${sourceId}`, {
          method: "DELETE",
          headers: {
            "If-Match": `"rv-${item.row_version}"`,
            "Idempotency-Key": KBotAIOpsAuth.uuid(),
          },
        });
        shell.toast("监控源已删除");
        await renderList("diagnostic-sources");
      } catch (error) {
        shell.toast(error.message);
        button.disabled = false;
      }
      return;
    }
    const path = action === "connectivity"
      ? `/diagnostic-sources/${sourceId}/connectivity-checks`
      : `/diagnostic-sources/${sourceId}/${action}`;
    button.disabled = true;
    try {
      const result = await KBotAIOpsAuth.request(appApi + path, {
        method: "POST",
        headers: {
          "If-Match": `"rv-${item.row_version}"`,
          "Idempotency-Key": KBotAIOpsAuth.uuid(),
        },
        body: JSON.stringify({}),
      });
      const connectivityResult = action === "connectivity"
        ? `连通性检查结果：${result.connectivity_status || "UNKNOWN"}${result.last_error_code ? `（${result.last_error_code}）` : ""}`
        : "";
      shell.toast(action === "connectivity" ? connectivityResult : action === "enable" ? "监控源已启用" : "监控源已停用");
      await renderList("diagnostic-sources");
    } catch (error) {
      shell.toast(error.message);
      button.disabled = false;
    }
  }
  async function runTargetAction(button, item) {
    const action = button.dataset.targetAction;
    const targetId = encodeURIComponent(item.target_id);
    if (action === "detail") {
      location.href = `target-detail.html?id=${targetId}`;
      return;
    }
    if (action === "edit") {
      globalThis.KBotAIOpsTargets?.openEdit(item.target_id);
      return;
    }
    if (action === "delete") {
      if (item.status !== "DISABLED") {
        shell.toast("请先停用运维目标，再执行删除");
        return;
      }
      const confirmed = confirm(
        `确认删除运维目标“${item.display_name || item.target_id}”吗？\n\n`
        + "删除会一并清理该 Target 的监控映射、数据库凭据、运行、会话、报告、告警、工作项、恢复与负载历史，且不可恢复。"
        + "共享的监控源、Agent、策略、巡检定义和知识资产不会删除。仍有运行中任务时系统会拒绝删除。",
      );
      if (!confirmed) return;
      button.disabled = true;
      try {
        await KBotAIOpsAuth.request(`${appApi}/targets/${targetId}`, {
          method: "DELETE",
          headers: {
            "If-Match": `"rv-${item.row_version}"`,
            "Idempotency-Key": KBotAIOpsAuth.uuid(),
          },
        });
        shell.toast("运维目标已删除");
        await renderList("targets");
      } catch (error) {
        shell.toast(error.message);
        button.disabled = false;
      }
      return;
    }
    const path = action === "connectivity"
      ? `/targets/${targetId}/connectivity-checks`
      : `/targets/${targetId}/${action}`;
    button.disabled = true;
    try {
      const result = await KBotAIOpsAuth.request(appApi + path, {
        method: "POST",
        headers: {
          "If-Match": `"rv-${item.row_version}"`,
          "Idempotency-Key": KBotAIOpsAuth.uuid(),
        },
        body: JSON.stringify({}),
      });
      const connectivityResult = action === "connectivity"
        ? `连通性检查结果：${result.connectivity_status || "UNKNOWN"}${result.last_error_code ? `（${result.last_error_code}）` : ""}`
        : "";
      shell.toast(
        action === "connectivity"
          ? connectivityResult
          : action === "enable"
            ? "运维目标已启用"
            : "运维目标已停用"
      );
      await renderList("targets");
    } catch (error) {
      shell.toast(error.message);
      button.disabled = false;
    }
  }
  async function renderList(page) {
    const cfg = configs[page];
    if (cfg.render === "report-list") {
      await renderReportList(cfg);
      return;
    }
    const head = document.getElementById("ops-table-head");
    const body = document.getElementById("ops-table-body");
    head.innerHTML = `<tr>${cfg.cols.map((col) => `<th>${col[1]}</th>`).join("")}</tr>`;
    try {
      const [payload, agentRows] = await Promise.all([
        KBotAIOpsAuth.request(appApi + cfg.path),
        page === "inspection-plans" ? KBotAIOpsAuth.request(`${appApi}/agents`) : Promise.resolve([]),
      ]);
      const agentNames = new Map((Array.isArray(agentRows) ? agentRows : []).map((agent) => [String(agent.agent_id), agent.display_name]));
      const sourceItems = Array.isArray(payload) ? payload : payload?.items || [];
      const items = page === "inspection-plans"
        ? sourceItems.map((item) => ({ ...item, agent_name: agentNames.get(String(item.agent_id)) || shell.short(item.agent_id) }))
        : sourceItems;
      if (!items.length) {
        body.innerHTML = `<tr><td class="ops-empty" colspan="${cfg.cols.length}">当前范围内暂无数据</td></tr>`;
        return;
      }
      body.innerHTML = items.map((item) => `<tr ${cfg.detail ? `data-href="${cfg.detail}${encodeURIComponent(resourceId(item))}" data-resource-id="${shell.escape(resourceId(item))}"` : ""}>${cfg.cols.map((col) => `<td>${cell(item, col)}</td>`).join("")}</tr>`).join("");
      body.querySelectorAll("[data-href]").forEach((row) => {
        row.style.cursor = "pointer";
        row.addEventListener("click", () => {
          const editors = {
            targets: globalThis.KBotAIOpsTargets,
            "inspection-plans": globalThis.KBotAIOpsConfigurations,
          };
          if (editors[page]?.openEdit) {
            if (page === "targets") editors[page].openEdit(row.dataset.resourceId);
            else editors[page].openEdit(page, row.dataset.resourceId);
            return;
          }
          location.href = row.dataset.href;
        });
      });
      if (page === "diagnostic-sources") {
        body.querySelectorAll("[data-source-action]").forEach((button) => {
          button.addEventListener("click", (event) => {
            event.stopPropagation();
            const row = button.closest("tr");
            const item = items.find(
              (candidate) => String(candidate.source_id) === row.dataset.resourceId
            );
            if (item) runSourceAction(button, item);
          });
        });
        if (items.some((item) => item.connectivity_check_pending)) {
          scheduleSourceReload();
        } else {
          sourceReloadAttempts = 0;
        }
      }
      if (page === "targets") {
        body.querySelectorAll("[data-target-action]").forEach((button) => {
          button.addEventListener("click", (event) => {
            event.stopPropagation();
            const row = button.closest("tr");
            const item = items.find(
              (candidate) => String(candidate.target_id) === row.dataset.resourceId
            );
            if (item) runTargetAction(button, item);
          });
        });
      }
      if (page === "inspection-plans") {
        body.querySelectorAll("[data-inspection-action]").forEach((button) => {
          button.addEventListener("click", async (event) => {
            event.stopPropagation();
            const row = button.closest("tr");
            const item = items.find((candidate) => String(candidate.plan_id) === row.dataset.resourceId);
            if (!item) return;
            button.disabled = true;
            try {
              await KBotAIOpsAuth.request(`${appApi}/inspection-plans/${encodeURIComponent(item.plan_id)}/${button.dataset.inspectionAction}`, {
                method: "POST",
                headers: {
                  "If-Match": `"rv-${item.row_version}"`,
                  "Idempotency-Key": KBotAIOpsAuth.uuid(),
                },
                body: JSON.stringify({}),
              });
              shell.toast(button.dataset.inspectionAction === "activate" ? "巡检计划已启用" : "巡检计划已暂停");
              await renderList("inspection-plans");
            } catch (error) {
              shell.toast(error.message);
              button.disabled = false;
            }
          });
        });
      }
    } catch (error) {
      body.innerHTML = `<tr><td class="ops-empty" colspan="${cfg.cols.length}">${shell.escape(error.message)}</td></tr>`;
    }
  }

  let targetSubscription = null;

  function subscriptionElements() {
    const form = document.getElementById("target-subscription-form");
    if (!form) return null;
    return {
      form,
      follow: form.elements.follow_target,
      severity: form.elements.minimum_severity,
      settings: document.getElementById("target-subscription-settings"),
      state: document.getElementById("target-subscription-state"),
      result: document.getElementById("target-subscription-result"),
      save: document.getElementById("save-target-subscription"),
    };
  }

  function renderTargetSubscription(target) {
    const elements = subscriptionElements();
    if (!elements) return;
    const active = targetSubscription?.status === "ACTIVE";
    elements.follow.checked = active;
    elements.severity.value = targetSubscription?.minimum_severity || "HIGH";
    const stages = new Set(targetSubscription?.stages || [
      "SITUATION_DETECTED", "DIAGNOSIS_STARTED", "REPORT_READY",
      "SITUATION_RECOVERED",
    ]);
    elements.form.querySelectorAll('[name="stages"]').forEach((input) => {
      input.checked = stages.has(input.value);
    });
    elements.state.className = `ops-badge ${active ? "good" : ""}`;
    elements.state.textContent = active ? "已关注" : "未关注";
    const targetEnabled = target.status === "ENABLED";
    elements.follow.disabled = !targetEnabled && !active;
    elements.settings.classList.toggle("target-subscription-disabled", !active);
    elements.settings.querySelectorAll("input,select").forEach((input) => {
      input.disabled = !active;
    });
    elements.save.disabled = !targetEnabled && !active;
    if (!targetEnabled && !active) {
      elements.result.textContent = "Target 启用后才能关注。";
      elements.result.dataset.tone = "bad";
    } else {
      elements.result.textContent = active
        ? "当前用户将按以上条件接收站内通知。"
        : "关注后，符合条件的事件会进入通知中心。";
      elements.result.dataset.tone = "";
    }
  }

  async function loadTargetSubscription(targetId) {
    const payload = await KBotAIOpsAuth.request(
      `${appApi}/notification-subscriptions`,
    );
    targetSubscription = (payload?.items || []).find(
      (item) => String(item.target_id) === String(targetId),
    ) || null;
  }

  async function saveTargetSubscription(event, targetId, target) {
    event.preventDefault();
    const elements = subscriptionElements();
    const following = elements.follow.checked;
    const originalText = elements.save.textContent;
    elements.save.disabled = true;
    elements.save.textContent = "保存中…";
    elements.result.textContent = "正在保存当前用户的通知设置…";
    elements.result.dataset.tone = "";
    try {
      if (!following) {
        if (targetSubscription?.status === "ACTIVE") {
          await KBotAIOpsAuth.request(
            `${appApi}/notification-subscriptions/targets/${encodeURIComponent(targetId)}`,
            {
              method: "DELETE",
              headers: { "If-Match": `"rv-${targetSubscription.row_version}"` },
            },
          );
        }
      } else {
        if (target.status !== "ENABLED") throw new Error("Target 启用后才能关注。");
        const stages = [...elements.form.querySelectorAll('[name="stages"]:checked')].map((input) => input.value);
        if (!stages.length) throw new Error("至少选择一个通知阶段。");
        const headers = targetSubscription
          ? { "If-Match": `"rv-${targetSubscription.row_version}"` }
          : {};
        await KBotAIOpsAuth.request(
          `${appApi}/notification-subscriptions/targets/${encodeURIComponent(targetId)}`,
          {
            method: "PUT",
            headers,
            body: JSON.stringify({
              minimum_severity: elements.severity.value,
              stages,
            }),
          },
        );
      }
      await loadTargetSubscription(targetId);
      renderTargetSubscription(target);
      shell.toast(following ? "已更新该目标的通知关注" : "已取消关注该目标");
      document.getElementById("target-subscription-dialog")?.close();
    } catch (error) {
      elements.result.textContent = error.message;
      elements.result.dataset.tone = "bad";
    } finally {
      elements.save.disabled = (
        target.status !== "ENABLED"
        && targetSubscription?.status !== "ACTIVE"
      );
      elements.save.textContent = originalText;
    }
  }

  const factTypeLabels = {
    ASM_DISKGROUP: "ASM 磁盘组",
    DATAFILE_PATH: "数据文件目录",
    TABLESPACE_PLACEMENT: "表空间放置",
  };
  const factKeyFields = {
    ASM_DISKGROUP: "diskgroup_name",
    DATAFILE_PATH: "directory",
    TABLESPACE_PLACEMENT: "tablespace_name",
  };

  function factValueText(item) {
    const keyField = factKeyFields[item.fact_type];
    const value = item.fact_value || {};
    return value[keyField] || item.fact_key || "—";
  }

  function renderTargetFacts(items) {
    const list = document.getElementById("target-facts-list");
    if (!list) return;
    if (!items.length) {
      list.innerHTML = '<p class="ops-empty">当前没有已确认的运维记忆。</p>';
      return;
    }
    list.innerHTML = `<table class="ops-table"><thead><tr><th>类型</th><th>事实值</th><th>来源</th><th>确认时间</th><th></th></tr></thead><tbody>${items.map((item) => `<tr><td>${shell.escape(factTypeLabels[item.fact_type] || item.fact_type)}</td><td>${shell.escape(factValueText(item))}</td><td>${shell.escape(item.source || "—")}</td><td>${shell.escape(shell.fmt(item.confirmed_at))}</td><td><button type="button" data-retire-target-fact="${shell.escape(item.fact_id)}" data-version="${shell.escape(item.row_version)}">撤回</button></td></tr>`).join("")}</tbody></table>`;
  }

  async function loadTargetFacts(targetId) {
    const payload = await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/facts`);
    renderTargetFacts(payload.items || []);
  }

  async function initializeTargetFacts(targetId) {
    const form = document.getElementById("target-facts-form");
    const list = document.getElementById("target-facts-list");
    const result = document.getElementById("target-facts-result");
    if (!form || !list) return;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const factType = form.fact_type.value;
      const raw = String(form.fact_value.value || "").trim();
      const keyField = factKeyFields[factType];
      if (!raw || !keyField) {
        result.textContent = "请填写事实值。";
        result.dataset.tone = "bad";
        return;
      }
      form.querySelector("button[type=submit]").disabled = true;
      try {
        await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/facts`, {
          method: "POST",
          headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
          body: JSON.stringify({
            fact_type: factType,
            fact_key: raw,
            fact_value: { [keyField]: raw },
          }),
        });
        form.fact_value.value = "";
        result.textContent = "已写入运维记忆，不会执行 SQL。";
        result.dataset.tone = "good";
        await loadTargetFacts(targetId);
      } catch (error) {
        result.textContent = error.message;
        result.dataset.tone = "bad";
      } finally {
        form.querySelector("button[type=submit]").disabled = false;
      }
    });
    list.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-retire-target-fact]");
      if (!button) return;
      if (!confirm("确认撤回这条运维记忆吗？撤回后不再参与容量判断。")) return;
      button.disabled = true;
      try {
        await KBotAIOpsAuth.request(
          `${appApi}/targets/${encodeURIComponent(targetId)}/facts/${encodeURIComponent(button.dataset.retireTargetFact)}/retire`,
          {
            method: "POST",
            headers: {
              "If-Match": `"rv-${button.dataset.version}"`,
              "Idempotency-Key": KBotAIOpsAuth.uuid(),
            },
          },
        );
        result.textContent = "已撤回该运维记忆。";
        result.dataset.tone = "good";
        await loadTargetFacts(targetId);
      } catch (error) {
        result.textContent = error.message;
        result.dataset.tone = "bad";
        button.disabled = false;
      }
    });
    try {
      await loadTargetFacts(targetId);
    } catch (error) {
      list.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`;
    }
  }

  async function initializeTargetSubscription(targetId, target) {
    const elements = subscriptionElements();
    if (!elements) return;
    const dialog = document.getElementById("target-subscription-dialog");
    const openButton = document.getElementById("open-target-subscription");
    openButton?.addEventListener("click", () => {
      renderTargetSubscription(target);
      if (!dialog.open) dialog.showModal();
    });
    dialog?.querySelectorAll("[data-close-target-subscription]").forEach((button) => {
      button.addEventListener("click", () => dialog.close());
    });
    dialog?.addEventListener("close", () => renderTargetSubscription(target));
    elements.follow.addEventListener("change", () => {
      const enabled = elements.follow.checked;
      elements.settings.classList.toggle("target-subscription-disabled", !enabled);
      elements.settings.querySelectorAll("input,select").forEach((input) => {
        input.disabled = !enabled;
      });
      elements.result.textContent = enabled
        ? "设置通知条件后保存。"
        : "保存后将停止接收该目标的订阅通知。";
      elements.result.dataset.tone = "";
    });
    elements.form.addEventListener("submit", (event) => {
      void saveTargetSubscription(event, targetId, target);
    });
    try {
      await loadTargetSubscription(targetId);
      renderTargetSubscription(target);
    } catch (error) {
      elements.state.className = "ops-badge bad";
      elements.state.textContent = "读取失败";
      elements.result.textContent = error.message;
      elements.result.dataset.tone = "bad";
      elements.save.disabled = true;
    }
  }

  function renderRecoverySummary(profile, drills) {
    const state = document.getElementById("target-recovery-state");
    const summary = document.getElementById("target-recovery-summary");
    const latest = drills[0] || null;
    const rank = { BACKUP_METADATA: 1, RESTORE_VALIDATE: 2, DATABASE_OPEN: 3, APPLICATION_VALIDATED: 4 };
    const requiredRank = rank[profile?.required_assurance_level] || 0;
    const success = drills.find((item) => item.status === "VERIFIED" && item.result === "PASS" && (rank[item.assurance_level] || 0) >= requiredRank) || null;
    state.className = `ops-badge ${profile ? "good" : "warn"}`;
    state.textContent = profile ? `目标 v${profile.version_no}` : "目标未配置";
    summary.innerHTML = `<dl class="ops-detail"><dt>当前 RPO</dt><dd>${profile ? `${profile.rpo_seconds / 60} 分钟` : "未配置"}</dd><dt>当前 RTO</dt><dd>${profile ? `${profile.rto_seconds / 60} 分钟` : "未配置"}</dd><dt>最新尝试</dt><dd>${latest ? `${shell.escape(latest.result)} · ${shell.escape(shell.fmt(latest.simulated_failure_at))}` : "无记录"}</dd><dt>最新审核成功</dt><dd>${success ? `${shell.escape(success.assurance_level)} · ${shell.escape(shell.fmt(success.simulated_failure_at))}` : "无记录"}</dd></dl>${latest && latest.result !== "PASS" && success ? '<p class="ops-connection-result" data-tone="bad">最新尝试未通过；下方旧成功记录仅作历史证据，不能掩盖本次失败。</p>' : ""}`;
  }

  const recoverySourcesByDatabase = {
    ORACLE: ["ORACLE_RMAN", "FILESYSTEM_SNAPSHOT", "STORAGE_SNAPSHOT", "CLOUD_MANAGED_BACKUP", "THIRD_PARTY_BACKUP"],
    POSTGRESQL: ["POSTGRESQL_BASEBACKUP", "POSTGRESQL_PGBACKREST", "POSTGRESQL_BARMAN", "POSTGRESQL_WALG", "FILESYSTEM_SNAPSHOT", "STORAGE_SNAPSHOT", "CLOUD_MANAGED_BACKUP", "THIRD_PARTY_BACKUP"],
    MYSQL: ["MYSQL_XTRABACKUP", "MYSQL_ENTERPRISE_BACKUP", "MYSQL_LOGICAL_DUMP", "FILESYSTEM_SNAPSHOT", "STORAGE_SNAPSHOT", "CLOUD_MANAGED_BACKUP", "THIRD_PARTY_BACKUP"],
  };

  function configureRecoverySources(form, dbType, selected = []) {
    const values = recoverySourcesByDatabase[dbType] || [];
    const selectedValues = new Set(selected.length ? selected : values.slice(0, 1));
    form.required_backup_source_types.innerHTML = values.map((value) => `<option value="${shell.escape(value)}" ${selectedValues.has(value) ? "selected" : ""}>${shell.escape(value)}</option>`).join("");
  }

  async function loadTargetRecovery(targetId, dbType, populateProfile = true) {
    const [profile, drillsPayload] = await Promise.all([
      KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/recovery-profile`),
      KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/recovery-drills`),
    ]);
    const drills = drillsPayload?.items || [];
    renderRecoverySummary(profile, drills);
    if (profile && populateProfile) {
      const form = document.getElementById("target-recovery-profile-form");
      form.rpo_minutes.value = profile.rpo_seconds / 60;
      form.rto_minutes.value = profile.rto_seconds / 60;
      form.required_drill_interval_days.value = profile.required_drill_interval_days;
      form.required_assurance_level.value = profile.required_assurance_level;
      configureRecoverySources(form, dbType, profile.required_backup_source_types || []);
      form.source_note.value = profile.source_note || "";
    }
    return { profile, drills };
  }

  async function initializeTargetRecovery(targetId, target) {
    const profileForm = document.getElementById("target-recovery-profile-form");
    if (!profileForm) return;
    configureRecoverySources(profileForm, target.db_type);
    document.getElementById("target-recovery-drills-link").href = `./recovery-drills.html?target_id=${encodeURIComponent(targetId)}`;
    profileForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const result = document.getElementById("target-recovery-profile-result");
      try {
        await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/recovery-profile`, { method: "PUT", headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() }, body: JSON.stringify({ rpo_seconds: Number(profileForm.rpo_minutes.value) * 60, rto_seconds: Number(profileForm.rto_minutes.value) * 60, required_drill_interval_days: Number(profileForm.required_drill_interval_days.value), required_assurance_level: profileForm.required_assurance_level.value, required_backup_source_types: [...profileForm.required_backup_source_types.selectedOptions].map((option) => option.value), rto_clock_basis: "SERVICE_UNAVAILABLE_TO_VALIDATED", source_note: String(profileForm.source_note.value || "").trim() || null }) });
        result.textContent = "已保存为新的生效版本。"; result.dataset.tone = "good";
        await loadTargetRecovery(targetId, target.db_type, true);
      } catch (error) { result.textContent = error.message; result.dataset.tone = "bad"; }
    });
    try { await loadTargetRecovery(targetId, target.db_type, true); }
    catch (error) { document.getElementById("target-recovery-summary").innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`; }
  }

  const protectedReportSections = new Set(["EVIDENCE_BOUNDARY", "EVIDENCE_APPENDIX"]);
  let canManageOperationsKnowledge = false;

  function leadershipBriefingHtml(briefing) {
    if (!briefing || typeof briefing !== "object") return "";
    const list = (items) => `<ul>${(Array.isArray(items) ? items : []).map((item) => `<li>${shell.escape(reportDisplayText(item))}</li>`).join("")}</ul>`;
    const risk = String(briefing.risk_level || "LOW");
    const tone = { CRITICAL: "bad", HIGH: "bad", MEDIUM: "warn", LOW: "good", INFO: "good" }[risk] || "";
    return `<section class="ops-panel ops-leadership-briefing" data-leadership-briefing><div class="ops-panel-head"><div><h3>领导简报</h3><p>只保留影响、风险和建议，不展开 SID 或 SQL。</p></div><span class="ops-badge ${tone}">${shell.escape(briefing.risk_level_display || reportDisplayText(risk))}</span></div><div class="ops-panel-body ops-leadership-grid"><article><h4>影响</h4>${list(briefing.business_impact)}</article><article><h4>风险</h4>${list(briefing.risks)}</article><article><h4>建议</h4>${list(briefing.recommendations)}</article></div></section>`;
  }

  function targetOverviewHtml(target) {
    const environment = { PROD: "生产", STG: "测试 / 预发布", DEV: "开发" }[target.environment] || target.environment || "—";
    const role = { PRIMARY: "主库", STANDBY: "备库", UNKNOWN: "未知角色" }[target.db_role] || target.db_role || "—";
    const endpoint = target.endpoint
      ? `${target.endpoint.host}:${target.endpoint.port}${target.endpoint.service ? ` / ${target.endpoint.service}` : target.endpoint.database ? ` / ${target.endpoint.database}` : ""}`
      : "未配置（仅监控模式）";
    const credential = (value) => value?.configured
      ? `已配置${value.updated_at ? ` · ${shell.fmt(value.updated_at)}` : ""}`
      : "未配置";
    const importance = { 1: "可按需忽略", 2: "较低", 3: "普通", 4: "重要", 5: "最重要" }[target.importance_level] || "未知";
    const oracleScope = [
      target.observed_oracle_database_name || target.oracle_pdb_name,
      target.observed_oracle_container_scope || target.oracle_container_scope,
      target.observed_oracle_container_name,
    ].filter(Boolean).join(" · ");
    const row = (label, value) => `<dt>${shell.escape(label)}</dt><dd>${shell.escape(value ?? "—")}</dd>`;
    return `<div class="ops-panel-head target-overview-head"><div><p>${shell.escape(target.db_type)}${target.version_code ? ` · ${shell.escape(target.version_code)}` : ""}</p><h2>${shell.escape(target.display_name)}</h2><small>${shell.escape(environment)} · ${shell.escape(role)} · L${shell.escape(target.importance_level)} ${shell.escape(importance)}</small></div><div class="ops-actions">${shell.badge(target.status)}${shell.badge(target.connectivity_check_pending ? "CHECKING" : target.connectivity_status)}${shell.badge(target.observed_status)}</div></div><div class="ops-panel-body target-overview-grid"><section><h3>数据库范围</h3><dl class="ops-detail">${row("数据库类型", target.db_type)}${row("环境", environment)}${row("数据库角色", role)}${target.db_type === "ORACLE" ? row("Oracle 范围", oracleScope || "尚未确认") : ""}${row("重要程度", `L${target.importance_level} · ${importance}`)}</dl></section><section><h3>连接状态</h3><dl class="ops-detail">${row("诊断模式", target.readonly_connection_enabled ? "数据库只读直连" : "仅监控数据")}${row("Endpoint", endpoint)}${row("TLS", target.endpoint ? (target.endpoint.tls_enabled ? "已启用" : "未启用") : "不适用")}${row("最近检查", shell.fmt(target.last_connectivity_check_at))}${row("最近成功", shell.fmt(target.last_connectivity_success_at))}${target.last_error_code ? row("最近错误", target.last_error_code) : ""}</dl></section><section><h3>凭据与变更</h3><dl class="ops-detail">${row("诊断凭据", credential(target.diagnostic_credential))}${row("受控变更", target.controlled_change_enabled ? "允许人工审批后执行" : "未启用")}${row("执行凭据", target.controlled_change_enabled ? credential(target.execution_credential) : "不适用")}</dl></section><section><h3>观测与采集</h3><dl class="ops-detail">${row("观测状态", target.observed_status)}${row("最近观测", shell.fmt(target.last_observed_at))}${row("负载快照", target.workload_last_collected_at ? `最近采集 ${shell.fmt(target.workload_last_collected_at)}` : "尚无采集记录")}${row("活动采样", target.activity_sampler_status)}${row("最近采样", shell.fmt(target.activity_last_sampled_at))}</dl></section></div>`;
  }

  function reportPresentationHtml(data, report, versions) {
    const sections = Array.isArray(data.sections) ? data.sections : [];
    const canEdit = String(versions?.items?.[0]?.report_id || "") === String(report.report_id);
    const versionItems = (versions?.items || []).map((item) => `<option value="${shell.escape(item.report_id)}" ${String(item.report_id) === String(report.report_id) ? "selected" : ""}>v${shell.escape(item.report_version)} · ${shell.escape(shell.fmt(item.published_at))}</option>`).join("");
    return `<article class="ops-report-presentation"><header class="ops-head"><div><h2 data-report-title>${shell.escape(data.title || report.title || "正式报告")}</h2><p>${shell.escape(data.template?.display_name || "报告模板")} · ${shell.escape(data.status_display || reportDisplayText(data.status || "UNKNOWN"))} · v${shell.escape(report.report_version)}</p></div><div class="ops-actions">${canEdit ? '<button type="button" data-write-ready data-edit-report>编辑报告</button>' : ""}${canManageOperationsKnowledge ? '<button type="button" data-write-ready data-extract-case>提炼为诊断案例</button>' : ""}<button class="primary" type="button" data-write-ready data-download-report>下载 PDF</button></div></header>${leadershipBriefingHtml(data.leadership_briefing)}<section class="ops-panel"><div class="ops-panel-body"><label>历史版本 <select data-report-version>${versionItems}</select></label><p>历史版本可随时预览和重新下载；人工编辑会创建新版本，不会覆盖旧版。</p></div></section>${sections.map((section) => `<section class="ops-panel" data-report-section="${shell.escape(section.kind || "")}"><div class="ops-panel-head"><h3>${shell.escape(section.display_name || reportDisplayText(section.kind || "章节"))}${section.human_edited ? " · 人工编辑" : ""}</h3></div><div class="ops-panel-body" data-report-section-body><ul>${(section.items || []).map((item) => `<li>${shell.escape(reportDisplayText(item))}</li>`).join("")}</ul></div></section>`).join("")}</article>`;
  }

  function beginReportEdit(panel, data, report, versions) {
    const sections = Array.isArray(data.sections) ? data.sections : [];
    const editableSections = sections.filter((section) => !protectedReportSections.has(section.kind));
    panel.innerHTML = `<article class="ops-report-presentation"><header class="ops-head"><div><label>报告标题 <input data-report-title-input maxlength="512" value="${shell.escape(data.title || report.title || "")}"></label><p>人工编辑仅修改展示文字，已冻结的证据边界和证据索引不会变化。</p></div><div class="ops-actions"><button type="button" data-write-ready data-cancel-report-edit>取消</button><button class="primary" type="button" data-write-ready data-save-report-edit>保存为新版本</button></div></header>${editableSections.map((section) => `<section class="ops-panel"><div class="ops-panel-head"><h3>${shell.escape(section.display_name || reportDisplayText(section.kind || "章节"))}</h3></div><div class="ops-panel-body"><textarea data-report-edit-section="${shell.escape(section.kind)}" rows="6">${shell.escape((section.items || []).join("\n"))}</textarea></div></section>`).join("")}</article>`;
    panel.querySelector("[data-cancel-report-edit]").onclick = () => {
      void renderReportDetail(report.report_id);
    };
    panel.querySelector("[data-save-report-edit]").onclick = async (event) => {
      const button = event.currentTarget;
      const title = panel.querySelector("[data-report-title-input]").value.trim();
      const edited = [...panel.querySelectorAll("[data-report-edit-section]")].map((input) => ({
        kind: input.dataset.reportEditSection,
        items: input.value.split("\n").map((line) => line.trim()).filter(Boolean),
      }));
      if (!title) { shell.toast("报告标题不能为空"); return; }
      if (edited.some((section) => !section.items.length)) {
        shell.toast("报告章节不能为空");
        return;
      }
      button.disabled = true;
      try {
        const saved = await KBotAIOpsAuth.request(appApi + `/reports/${encodeURIComponent(report.report_id)}`, {
          method: "PATCH",
          headers: { "If-Match": `"rv-${report.report_version}"` },
          body: JSON.stringify({ title, sections: edited }),
        });
        shell.toast("已保存为新的报告版本");
        location.replace(`report-detail.html?id=${encodeURIComponent(saved.report_id)}`);
      } catch (error) {
        shell.toast(error.message);
        button.disabled = false;
      }
    };
  }

  async function renderReportDetail(id) {
    const panel = document.getElementById("ops-detail");
    const encodedId = encodeURIComponent(id);
    const [data, report, versions] = await Promise.all([
      KBotAIOpsAuth.request(appApi + `/reports/${encodedId}/presentation`),
      KBotAIOpsAuth.request(appApi + `/reports/${encodedId}`),
      KBotAIOpsAuth.request(appApi + `/reports/${encodedId}/versions`),
    ]);
    panel.innerHTML = reportPresentationHtml(data, report, versions);
    panel.querySelector("[data-download-report]").onclick = () => KBotAIOpsAuth.download(
      appApi + `/reports/${encodedId}/pdf`,
      `aiops-report-${id}.pdf`,
    ).catch((error) => shell.toast(error.message));
    const editButton = panel.querySelector("[data-edit-report]");
    if (editButton) editButton.onclick = () => beginReportEdit(panel, data, report, versions);
    const extractButton = panel.querySelector("[data-extract-case]");
    if (extractButton) extractButton.onclick = async () => {
      extractButton.disabled = true;
      try {
        await KBotAIOpsAuth.request(
          appApi + `/operations-knowledge/reports/${encodedId}:extract-case`,
          { method: "POST" },
        );
        shell.toast("诊断案例候选已创建，完成索引后请到运维知识库审核发布");
      } catch (error) {
        shell.toast(error.message);
        extractButton.disabled = false;
      }
    };
    panel.querySelector("[data-report-version]").onchange = (event) => {
      const selected = event.currentTarget.value;
      if (selected && selected !== String(report.report_id)) {
        location.href = `report-detail.html?id=${encodeURIComponent(selected)}`;
      }
    };
  }

  function sourceDetailHtml(source) {
    const sourceTypeLabels = {
      PROMETHEUS: "Prometheus",
      ALERTMANAGER: "Alertmanager",
      LOKI: "Loki",
      ZABBIX: "Zabbix",
      OEM: "Oracle Enterprise Manager",
    };
    const statusLabels = { ENABLED: "已启用", DISABLED: "已停用" };
    const connectivityLabels = {
      UNKNOWN: "尚未检查",
      CHECKING: "正在检查",
      CONNECTED: "连接正常",
      DEGRADED: "部分可用",
      MISCONFIGURED: "配置错误",
      UNREACHABLE: "无法连接",
    };
    const capabilityLabels = {
      "health.check": "健康检查",
      "event.receive": "接收告警事件",
      "event.query": "查询活动事件",
      "metric.query_range": "查询时序指标",
      "log.query": "查询日志",
      "database.query_live": "数据库实时查询",
      "database.query_history": "数据库历史查询",
      "host.inspect": "主机检查",
      "topology.resolve": "拓扑解析",
      "change.query": "变更查询",
      "workload.query": "工作负载查询",
      "action.execute": "执行受控动作",
      "instance.discover": "发现监控实例",
    };
    const row = (label, value) => `<dt>${shell.escape(label)}</dt><dd>${shell.escape(value ?? "—")}</dd>`;
    const credential = (value) => value?.configured ? "已配置" : "未配置";
    const capabilities = (value, emptyText) => {
      const keys = Object.keys(value || {}).sort();
      if (!keys.length) return `<span class="source-capability-empty">${shell.escape(emptyText)}</span>`;
      return keys.map((key) => `<span class="ops-capability" title="${shell.escape(key)}">${shell.escape(capabilityLabels[key] || key)}</span>`).join("");
    };
    const connectivity = source.connectivity_check_pending
      ? "正在检查"
      : connectivityLabels[source.connectivity_status] || source.connectivity_status || "未知";
    const endpoint = source.endpoint || (source.webhook_secret?.configured ? "仅接收 Webhook" : "未配置");
    const tenant = source.source_type === "LOKI"
      ? source.config?.tenant_id || "单租户模式"
      : "不适用";
    return `<div class="ops-panel-head source-overview-head"><div><p>${shell.escape(sourceTypeLabels[source.source_type] || source.source_type || "监控源")}</p><h2>${shell.escape(source.display_name)}</h2><small>${shell.escape(source.adapter_id)} Adapter · v${shell.escape(source.adapter_version)}</small></div><div class="ops-actions">${shell.badge(source.status)}${shell.badge(source.connectivity_check_pending ? "CHECKING" : source.connectivity_status)}</div></div><div class="ops-panel-body source-overview-grid"><section><h3>接入信息</h3><dl class="ops-detail">${row("监控源类型", sourceTypeLabels[source.source_type] || source.source_type)}${row("访问地址", endpoint)}${row("Adapter", `${source.adapter_id} · v${source.adapter_version}`)}${row("Loki 租户", tenant)}</dl></section><section><h3>连接与健康</h3><dl class="ops-detail">${row("启用状态", statusLabels[source.status] || source.status)}${row("连接状态", connectivity)}${row("最近检查", shell.fmt(source.last_connectivity_check_at))}${row("最近成功", shell.fmt(source.last_connectivity_success_at))}${row("最近错误", source.last_error_code || "无")}</dl></section><section><h3>凭据与 Webhook</h3><dl class="ops-detail">${row("访问凭据", credential(source.secret))}${row("Webhook 验签凭据", credential(source.webhook_secret))}${row("Webhook Key", source.webhook_configured ? "已生成" : "未生成")}${row("TLS Profile", credential(source.tls_profile))}</dl></section><section><h3>管理信息</h3><dl class="ops-detail">${row("监控源 ID", source.source_id)}${row("创建时间", shell.fmt(source.created_at))}${row("创建人", source.created_by)}${row("最后更新", shell.fmt(source.updated_at))}${row("更新人", source.updated_by)}</dl></section><section class="source-capability-section"><h3>系统声明能力</h3><p>由当前监控源类型和内置 Adapter 决定。</p><div class="ops-capability-list">${capabilities(source.declared_capabilities, "当前 Adapter 未声明能力")}</div></section><section class="source-capability-section"><h3>最近验证能力</h3><p>来自最近一次成功的连通性检查，不代表尚未验证的能力不可用。</p><div class="ops-capability-list">${capabilities(source.discovered_capabilities, "尚未通过连通性检查验证能力")}</div></section></div><div class="source-detail-note">具体监控 Label 与数据库对象的映射统一在“运维目标详情”中维护。</div>`;
  }

  async function initializeTargetMonitorMappings(targetId, target) {
    const form = document.getElementById("target-monitor-binding-form");
    if (!form) return;
    const sourceSelect = document.getElementById("target-monitor-source");
    const locatorField = document.getElementById("target-monitor-locator-field");
    const locatorInput = document.getElementById("target-monitor-locator");
    const locatorLabel = document.getElementById("target-monitor-locator-label");
    const locatorHelp = document.getElementById("target-monitor-locator-help");
    const jobField = document.getElementById("target-monitor-job-field");
    const jobInput = document.getElementById("target-monitor-job");
    const discovery = document.getElementById("target-monitor-discovery");
    const discoverButton = document.getElementById("discover-target-monitor-labels");
    const candidateList = document.getElementById("target-monitor-candidate-list");
    const hostSection = document.getElementById("target-monitor-host-section");
    const hostCandidateList = document.getElementById("target-monitor-host-candidate-list");
    const bindingList = document.getElementById("target-monitor-binding-list");
    const result = document.getElementById("target-monitor-binding-result");
    const state = document.getElementById("target-monitor-mapping-state");
    const saveButton = document.getElementById("save-target-monitor-binding");
    let sources = [];
    let bindings = [];
    let candidates = [];

    const sourceById = (sourceId) => sources.find((item) => String(item.source_id) === String(sourceId));
    const bindingBySourceId = (sourceId) => bindings.find(
      (item) => item.status === "ACTIVE" && String(item.source_id) === String(sourceId)
    );
    const renderBindings = () => {
      state.textContent = bindings.length ? `${bindings.length} 条有效映射` : "未配置";
      state.className = `ops-badge ${bindings.length ? "good" : "bad"}`;
      bindingList.innerHTML = bindings.length
        ? `<table class="ops-table"><thead><tr><th>监控源</th><th>类型</th><th>Label / 外部标识</th><th>状态</th></tr></thead><tbody>${bindings.map((binding) => {
          const source = sourceById(binding.source_id);
          const hostKey = binding.source_locator?.host_target_key;
          return `<tr><td><strong>${shell.escape(source?.display_name || shell.short(binding.source_id))}</strong></td><td>${shell.escape(source?.source_type || "—")}</td><td>数据库 <code>${shell.escape(binding.source_locator_key || binding.locator_hint)}</code><br>主机 <code>${shell.escape(hostKey || "未配置")}</code></td><td>${shell.badge(binding.status)}</td></tr>`;
        }).join("")}</tbody></table>`
        : '<div class="ops-error">尚未绑定监控 Label；完成至少一条映射后才能启用该 Target。</div>';
    };
    const renderSourceOptions = () => {
      const boundSourceIds = new Set(bindings.map((binding) => String(binding.source_id)));
      const available = sources.filter((source) => (
        source.status === "ENABLED"
        && (
          !boundSourceIds.has(String(source.source_id))
          || source.source_type === "PROMETHEUS"
        )
      ));
      sourceSelect.innerHTML = available.length
        ? '<option value="">请选择监控源</option>' + available.map((source) => `<option value="${shell.escape(source.source_id)}">${shell.escape(source.display_name)} · ${shell.escape(source.source_type)}${boundSourceIds.has(String(source.source_id)) ? " · 修改主机 Label" : ""}</option>`).join("")
        : '<option value="">没有尚未绑定的已启用监控源</option>';
      sourceSelect.disabled = !available.length;
      saveButton.disabled = !available.length;
    };
    const configureSource = () => {
      const source = sourceById(sourceSelect.value);
      const binding = bindingBySourceId(sourceSelect.value);
      const discoverable = ["PROMETHEUS", "ZABBIX"].includes(source?.source_type);
      discovery.hidden = !discoverable;
      locatorField.hidden = discoverable || !source;
      locatorInput.required = Boolean(source && !discoverable);
      jobField.hidden = source?.source_type !== "LOKI";
      hostSection.hidden = source?.source_type !== "PROMETHEUS";
      candidates = [];
      candidateList.innerHTML = binding
        ? `<div class="ops-empty">已绑定数据库 Label：<code>${shell.escape(binding.source_locator_key)}</code>；本次仅修改主机 Label。</div>`
        : '<div class="ops-empty">点击发现按钮读取当前数据库类型的候选。</div>';
      hostCandidateList.innerHTML = '<div class="ops-empty">点击发现按钮读取 Node Exporter 主机候选。</div>';
      result.textContent = "";
      if (!source) return;
      const presentation = {
        ALERTMANAGER: ["target_key label 值", "必须与告警中的 target_key 完全一致。"],
        LOKI: ["target_key label 值", "必须与日志流中的 target_key 完全一致。"],
        OEM: ["OEM Target Name", "填写 OEM 中唯一的 Target Name。"],
      }[source.source_type] || ["监控 Label 值", "填写监控系统中唯一标识当前数据库的值。"];
      locatorLabel.textContent = presentation[0];
      locatorHelp.textContent = presentation[1];
    };
    const loadCandidates = async () => {
      const source = sourceById(sourceSelect.value);
      if (!source || !["PROMETHEUS", "ZABBIX"].includes(source.source_type)) return;
      discoverButton.disabled = true;
      result.textContent = "正在从监控源发现候选 Label…";
      result.dataset.tone = "";
      try {
        const page = await KBotAIOpsAuth.request(`${appApi}/diagnostic-sources/${encodeURIComponent(source.source_id)}/instance-discoveries`, {
          method: "POST",
          body: JSON.stringify({ db_types: [target.db_type], page_size: 100 }),
        });
        const binding = bindingBySourceId(source.source_id);
        candidates = (page.items || []).filter((item) => item.mapping_status !== "MAPPED");
        if (!binding) {
          candidateList.innerHTML = candidates.length
            ? `<div class="target-monitor-candidates">${candidates.map((candidate, index) => `<label class="agent-switch-row"><input type="radio" name="monitor_candidate_ref" value="${shell.escape(candidate.candidate_ref)}" ${index === 0 ? "checked" : ""}><span><strong>${shell.escape(candidate.display_name)}</strong><small>${shell.escape(candidate.db_type)} · <code>${shell.escape(candidate.locator_hint)}</code></small></span></label>`).join("")}</div>`
            : '<div class="ops-empty">没有尚未映射且与当前数据库类型一致的候选 Label。</div>';
        }
        const hostCandidates = page.host_items || [];
        hostCandidateList.innerHTML = hostCandidates.length
          ? `<div class="target-monitor-candidates">${hostCandidates.map((candidate, index) => `<label class="agent-switch-row"><input type="radio" name="monitor_host_candidate_ref" value="${shell.escape(candidate.candidate_ref)}" ${index === 0 ? "checked" : ""}><span><strong>${shell.escape(candidate.display_name)}</strong><small>主机 · <code>${shell.escape(candidate.locator_hint)}</code></small></span></label>`).join("")}</div>`
          : '<div class="ops-error">未发现 Node Exporter 主机 Label；请先接入目标主机监控。</div>';
        result.textContent = `已发现 ${candidates.length} 个数据库候选、${hostCandidates.length} 个主机候选。`;
        result.dataset.tone = (binding || candidates.length) && (source.source_type !== "PROMETHEUS" || hostCandidates.length) ? "good" : "";
      } catch (error) {
        result.textContent = error.message;
        result.dataset.tone = "bad";
      } finally {
        discoverButton.disabled = false;
      }
    };
    const reloadBindings = async () => {
      const rows = await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/source-bindings`);
      bindings = Array.isArray(rows) ? rows : [];
      renderBindings();
      renderSourceOptions();
      configureSource();
    };

    try {
      const sourcePage = await KBotAIOpsAuth.request(`${appApi}/diagnostic-sources?limit=200`);
      sources = Array.isArray(sourcePage) ? sourcePage : sourcePage.items || [];
      await reloadBindings();
    } catch (error) {
      bindingList.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`;
      state.textContent = "读取失败";
      state.className = "ops-badge bad";
      return;
    }
    sourceSelect.addEventListener("change", configureSource);
    discoverButton.addEventListener("click", loadCandidates);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const source = sourceById(sourceSelect.value);
      if (!source) return;
      saveButton.disabled = true;
      result.textContent = "正在保存 Target 监控映射…";
      result.dataset.tone = "";
      try {
        if (["PROMETHEUS", "ZABBIX"].includes(source.source_type)) {
          const binding = bindingBySourceId(source.source_id);
          const hostCandidateRef = form.elements.monitor_host_candidate_ref?.value || null;
          if (source.source_type === "PROMETHEUS" && !hostCandidateRef) {
            throw new Error("请先发现并选择数据库所在主机的 Label。");
          }
          if (binding) {
            await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/source-bindings/${encodeURIComponent(binding.binding_id)}`, {
              method: "PATCH",
              headers: { "If-Match": `"rv-${binding.row_version}"` },
              body: JSON.stringify({ host_candidate_ref: hostCandidateRef }),
            });
          } else {
            const candidateRef = form.elements.monitor_candidate_ref?.value;
            if (!candidateRef) throw new Error("请先发现并选择数据库 Label。");
            await KBotAIOpsAuth.request(`${appApi}/diagnostic-sources/${encodeURIComponent(source.source_id)}/instance-mappings`, {
              method: "POST",
              headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
              body: JSON.stringify({ mappings: [{ candidate_ref: candidateRef, host_candidate_ref: hostCandidateRef, target_id: targetId }] }),
            });
          }
        } else {
          const locatorKey = locatorInput.value.trim();
          if (!locatorKey) throw new Error("监控 Label 值不能为空。");
          const labels = source.source_type === "LOKI"
            ? { target_key: locatorKey, ...(jobInput.value.trim() ? { job: jobInput.value.trim() } : {}) }
            : null;
          await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/source-bindings`, {
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
        locatorInput.value = "";
        jobInput.value = "";
        await reloadBindings();
        result.textContent = "监控源与当前 Target 的 Label 映射已保存。";
        result.dataset.tone = "good";
        shell.toast("Target 监控映射已保存");
      } catch (error) {
        result.textContent = error.message;
        result.dataset.tone = "bad";
      } finally {
        saveButton.disabled = sourceSelect.disabled;
      }
    });
  }

  async function renderDetail(page) {
    const id = new URLSearchParams(location.search).get("id");
    const paths = { "run-detail": "/runs/", "report-detail": "/reports/", "target-detail": "/targets/", "diagnostic-source-detail": "/diagnostic-sources/", "inspection-plan-detail": "/inspection-plans/" };
    const panel = document.getElementById("ops-detail");
    if (!id) { panel.innerHTML = '<div class="ops-error">URL 缺少资源 id</div>'; return; }
    try {
      if (page === "report-detail") {
        await renderReportDetail(id);
        return;
      }
      const data = await KBotAIOpsAuth.request(appApi + paths[page] + encodeURIComponent(id));
      panel.innerHTML = page === "target-detail"
        ? targetOverviewHtml(data)
        : page === "diagnostic-source-detail"
          ? sourceDetailHtml(data)
          : `<dl class="ops-detail">${Object.entries(data).filter(([, value]) => typeof value !== "object").map(([key, value]) => `<dt>${shell.escape(key)}</dt><dd>${shell.escape(value ?? "—")}</dd>`).join("")}</dl><pre class="ops-code">${shell.escape(JSON.stringify(data, null, 2))}</pre>`;
      if (page === "target-detail") {
        const editTargetAccess = document.getElementById("edit-target-access");
        if (editTargetAccess) {
          editTargetAccess.href = `targets.html?edit=${encodeURIComponent(id)}`;
        }
        await initializeTargetSubscription(id, data);
        await initializeTargetMonitorMappings(id, data);
        await initializeTargetRecovery(id, data);
        await initializeTargetFacts(id);
      }
    } catch (error) { panel.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`; }
  }
  async function renderSimple(page) {
    const paths = { "api-clients": `${appApi}/api-clients` };
    const panel = document.getElementById("ops-simple");
    if (!paths[page] || !panel) return;
    try { panel.innerHTML = `<pre class="ops-code">${shell.escape(JSON.stringify(await KBotAIOpsAuth.request(paths[page]), null, 2))}</pre>`; }
    catch (error) { panel.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`; }
  }
  shell.ready.then((access) => {
    canManageOperationsKnowledge = new Set(access?.permissions || []).has("aiops:knowledge_manage");
    document.querySelectorAll("header.ops-head button:not([onclick]):not([data-write-ready])").forEach((button) => {
      button.disabled = true;
      button.title = "该写操作将在对应配置表单接入后开放";
    });
    const page = document.body.dataset.page;
    if (configs[page]) renderList(page);
    else if (page.endsWith("-detail")) renderDetail(page);
    else if (page !== "agents") renderSimple(page);
  });
  globalThis.KBotAIOpsPages = {
    reload() {
      const page = document.body.dataset.page;
      return configs[page] ? renderList(page) : Promise.resolve();
    },
  };
})();
